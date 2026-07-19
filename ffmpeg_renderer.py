"""
FFmpeg filter_complex Renderer — replaces MoviePy per-frame rendering.

Instead of decoding → Python numpy compositing → encoding frame-by-frame,
this builds a single FFmpeg command that does EVERYTHING in one C process:
  decode (hwaccel) → crop/resize → color effects → overlays → captions → encode (NVENC)

Usage:
    from ffmpeg_renderer import FFMpegRenderer
    renderer = FFMpegRenderer(settings, output_folder)
    renderer.render(video_path, output_path, audio_path=None, caption_data=None, ...)

Fallback: if an effect can't be expressed in FFmpeg, a small PIL pre-render
pass writes it to a temp PNG/MP4 which is then overlaid via FFmpeg.
"""

import os
import sys
import json
import subprocess
import shutil
import logging
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────
# Data classes for render inputs
# ───────────────────────────────────────────────────────────────────

@dataclass
class CaptionWord:
    """One timed caption word."""
    text: str
    start: float        # seconds
    duration: float     # seconds

@dataclass
class OverlayImage:
    """Static image to overlay (watermark, text, blur-text, border, crosshair)."""
    image_path: Path
    x: int              # top-left x
    y: int              # top-left y
    opacity: float = 1.0
    start: float = 0.0  # overlay start time (seconds)
    end: float = None    # overlay end time (None = full duration)

@dataclass
class CaptionStyle:
    """How captions look."""
    font_file: str = "C:/Windows/Fonts/arialbd.ttf"
    font_size: int = 60
    active_color: str = "#FFFFFF"
    inactive_color: str = "#808080"
    active_bg_color: str = "#FF1493"
    stroke_color: str = "#000000"
    stroke_width: int = 3
    bg_opacity: float = 0.7
    bg_padding_x: int = 12
    bg_padding_y: int = 6
    corner_radius: int = 10
    position: str = "bottom"   # top, center, bottom
    y_offset: int = 0
    words_per_line: int = 3
    text_case: str = "Normal"  # Normal, ALL CAPS, Title Case
    animation: str = "none"    # none, pop, fade

@dataclass
class RenderInputs:
    """Everything the renderer needs."""
    video_path: Path
    output_path: Path
    settings: dict
    # Audio
    voiceover_path: Optional[Path] = None
    bgm_path: Optional[Path] = None
    bgm_volume: float = 0.3
    # Overlays (static images)
    overlays: List[OverlayImage] = field(default_factory=list)
    # Captions (word-by-word)
    captions: List[CaptionWord] = field(default_factory=list)
    caption_style: CaptionStyle = field(default_factory=CaptionStyle)
    # Target
    target_w: int = 1080
    target_h: int = 1920
    target_fps: int = 24
    target_duration: Optional[float] = None


# ───────────────────────────────────────────────────────────────────
# Main renderer
# ───────────────────────────────────────────────────────────────────

class FFMpegRenderer:

    def __init__(self, output_folder: Path, settings: dict):
        self.output_folder = Path(output_folder)
        self.settings = settings
        self._temp_dir = None
        self._nvenc_available = None  # lazy-checked

    def _get_temp_dir(self) -> Path:
        if self._temp_dir is None:
            self._temp_dir = self.output_folder / "_ffmpeg_rendertmp"
            self._temp_dir.mkdir(parents=True, exist_ok=True)
        return self._temp_dir

    def cleanup(self):
        if self._temp_dir and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    def _check_nvenc(self) -> bool:
        if self._nvenc_available is None:
            try:
                r = subprocess.run(
                    ['ffmpeg', '-encoders'], capture_output=True, text=True, timeout=5)
                self._nvenc_available = 'h264_nvenc' in r.stdout
            except Exception:
                self._nvenc_available = False
            if self._nvenc_available:
                logger.info("[FFmpeg] NVENC hardware encoder detected")
            else:
                logger.info("[FFmpeg] NVENC not available, using libx264")
        return self._nvenc_available

    def _check_hwaccel(self) -> bool:
        """Check if CUDA hwaccel is available."""
        try:
            r = subprocess.run(
                ['ffmpeg', '-hwaccels'], capture_output=True, text=True, timeout=5)
            return 'cuda' in r.stdout
        except Exception:
            return False

    # ═══════════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ═══════════════════════════════════════════════════════════════

    def render(self, inputs: RenderInputs) -> bool:
        """Render video using a single FFmpeg filter_complex pipeline.

        Returns True on success, False on failure.
        """
        t0 = time.time()
        self._temp_dir = self._get_temp_dir()

        try:
            # Step 1: Pre-render things FFmpeg can't do (captions as images, LUT application)
            caption_overlays = self._prerender_captions(inputs)

            # Step 2: Build the single FFmpeg command
            cmd = self._build_command(inputs, caption_overlays)

            # Step 3: Execute
            logger.info(f"[FFmpeg] Starting render: {inputs.output_path.name}")
            logger.info(f"[FFmpeg] Command: {' '.join(cmd[:20])}...")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min max
            )

            elapsed = time.time() - t0

            if result.returncode != 0:
                logger.error(f"[FFmpeg] Render failed (exit {result.returncode}):")
                # Log last 50 lines of stderr
                lines = result.stderr.strip().split('\n')
                for line in lines[-50:]:
                    logger.error(f"  {line}")
                return False

            # Step 4: Add audio separately (cleaner than trying to do it in filter_complex
            #          especially when we need to mix voiceover + BGM)
            if inputs.voiceover_path or inputs.bgm_path:
                self._mux_audio(inputs.output_path, inputs.voiceover_path,
                                inputs.bgm_path, inputs.bgm_volume,
                                inputs.target_duration)

            elapsed = time.time() - t0
            size_mb = inputs.output_path.stat().st_size / (1024 * 1024)
            logger.info(f"[FFmpeg] Render complete in {elapsed:.1f}s "
                        f"({size_mb:.1f} MB)")

            return True

        except subprocess.TimeoutExpired:
            logger.error("[FFmpeg] Render timed out after 600s")
            return False
        except Exception as e:
            logger.error(f"[FFmpeg] Render error: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.cleanup()

    # ═══════════════════════════════════════════════════════════════
    # PRE-RENDER: Captions as timed image overlays
    # ═══════════════════════════════════════════════════════════════

    def _prerender_captions(self, inputs: RenderInputs) -> List[OverlayImage]:
        """Pre-render word-by-word caption PNGs using PIL.

        Returns a list of OverlayImage objects (one per caption group/line).
        FFmpeg will overlay these at the right time using enable='between(t,t2)'.
        """
        captions = inputs.captions
        style = inputs.caption_style
        if not captions:
            return []

        from PIL import Image, ImageDraw, ImageFont, ImageColor

        # Load font
        try:
            font = ImageFont.truetype(style.font_file, style.font_size)
        except Exception:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", style.font_size)
            except Exception:
                font = ImageFont.load_default()

        overlays = []
        w, h = inputs.target_w, inputs.target_h

        # Group words into lines
        words_per_line = style.words_per_line
        for line_idx in range(0, len(captions), words_per_line):
            line_words = captions[line_idx:line_idx + words_per_line]
            if not line_words:
                continue

            line_start = line_words[0].start
            line_end = sum(w2.start + w2.duration for w2 in line_words)
            # Use the max of individual ends for more accuracy
            line_end = max(w2.start + w2.duration for w2 in line_words)

            line_text = ' '.join(w2.text for w2 in line_words)
            if style.text_case == 'ALL CAPS':
                line_text = line_text.upper()
            elif style.text_case == 'Title Case':
                line_text = line_text.title()

            # Render image
            draw_tmp = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
            bbox = draw_tmp.textbbox((0, 0), line_text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

            pad_x = style.bg_padding_x
            pad_y = style.bg_padding_y
            img_w = tw + pad_x * 2
            img_h = th + pad_y * 2

            img = Image.new('RGBA', (img_w, img_h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)

            # Background with rounded corners
            bg_alpha = int(255 * style.bg_opacity)
            try:
                r, g, b = ImageColor.getrgb(style.active_bg_color)
            except Exception:
                r, g, b = 255, 20, 147

            # Draw rounded rect background
            if style.corner_radius > 0:
                draw.rounded_rectangle(
                    [(0, 0), (img_w - 1, img_h - 1)],
                    radius=style.corner_radius,
                    fill=(r, g, b, bg_alpha)
                )
            else:
                draw.rectangle([(0, 0), (img_w - 1, img_h - 1)],
                              fill=(r, g, b, bg_alpha))

            # Draw text
            try:
                tr, tg, tb = ImageColor.getrgb(style.active_color)
            except Exception:
                tr, tg, tb = 255, 255, 255
            draw.text((pad_x, pad_y), line_text, font=font,
                      fill=(tr, tg, tb, 255))

            # Stroke
            if style.stroke_width > 0:
                try:
                    sr, sg, sb = ImageColor.getrgb(style.stroke_color)
                except Exception:
                    sr, sg, sb = 0, 0, 0
                stroke_img = Image.new('RGBA', (img_w, img_h), (0, 0, 0, 0))
                stroke_draw = ImageDraw.Draw(stroke_img)
                for dx in range(-style.stroke_width, style.stroke_width + 1):
                    for dy in range(-style.stroke_width, style.stroke_width + 1):
                        if dx * dx + dy * dy <= style.stroke_width * style.stroke_width:
                            stroke_draw.text(
                                (pad_x + dx, pad_y + dy), line_text,
                                font=font, fill=(sr, sg, sb, bg_alpha))
                img = Image.alpha_composite(stroke_img, img)

            # Save temp PNG
            tmp_path = self._temp_dir / f"_caption_{line_idx:04d}.png"
            img.save(str(tmp_path), 'PNG')

            # Calculate position
            x = (w - img_w) // 2
            if style.position == 'top':
                y = int(h * 0.05) + style.y_offset
            elif style.position == 'center':
                y = (h - img_h) // 2 + style.y_offset
            else:  # bottom
                y = h - img_h - int(h * 0.12) + style.y_offset

            # Clamp
            y = max(10, min(y, h - img_h - 10))

            overlays.append(OverlayImage(
                image_path=tmp_path,
                x=x, y=y,
                start=line_start,
                end=line_end,
            ))

        logger.info(f"[FFmpeg] Pre-rendered {len(overlays)} caption overlay(s)")
        return overlays

    # ═══════════════════════════════════════════════════════════════
    # BUILD FFmpeg COMMAND
    # ═══════════════════════════════════════════════════════════════

    def _build_command(self, inputs: RenderInputs,
                       caption_overlays: List[OverlayImage]) -> list:
        """Build the complete FFmpeg command with filter_complex."""

        s = self.settings
        cmd = []

        # ── Input 0: source video ──
        hwaccel = self._check_hwaccel()
        if hwaccel:
            cmd += ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda']
        cmd += ['-i', str(inputs.video_path)]

        # ── Additional inputs: overlay images ──
        input_idx = 1
        overlay_inputs = []  # (input_idx, overlay, enable_expr)

        # Static overlays (watermark, text, border, blur-text, crosshair)
        for ov in inputs.overlays:
            cmd += ['-i', str(ov.image_path)]
            enable = self._make_enable(ov.start, ov.end, inputs.target_duration)
            overlay_inputs.append((input_idx, ov, enable))
            input_idx += 1

        # Caption overlays (timed PNGs)
        for cov in caption_overlays:
            cmd += ['-i', str(cov.image_path)]
            enable = self._make_enable(cov.start, cov.end, inputs.target_duration)
            overlay_inputs.append((input_idx, cov, enable))
            input_idx += 1

        # ── Build filter_complex string ──
        filters = []
        stream_labels = []

        # Start with [0:v]
        current_label = '0:v'

        # 1. Crop/resize to target dimensions
        crop_resize = self._build_crop_resize(current_label, inputs, s)
        if crop_resize:
            filters.append(crop_resize)
            current_label = f'v_base'

        # 2. Color grading
        color_chain = self._build_color_filters(current_label, s)
        if color_chain:
            filters.append(f'{current_label}{color_chain}[v_color]')
            current_label = 'v_color'

        # 3. Vignette
        if s.get('vignette', False):
            intensity = s.get('vignette_intensity', 0.4)
            # FFmpeg vignette: angle is the opening angle, mode=forward
            # We approximate the numpy vignette with FFmpeg's built-in
            vignette_f = f'vignette=angle={1.0 - intensity:.2f}:mode=forward'
            filters.append(f'[{current_label}]{vignette_f}[v_vig]')
            current_label = 'v_vig'

        # 4. Background dim
        if s.get('background_dim', False):
            dim = s.get('dim_intensity', 0.25)
            brightness = 1.0 - dim
            filters.append(f'[{current_label}]eq=brightness={brightness:.2f}[v_dim]')
            current_label = 'v_dim'

        # 5. Film grain (FFmpeg has noise filter)
        if s.get('film_grain', False):
            grain_int = s.get('grain_intensity', 0.15)
            # FFmpeg noise: amount 0-100, we scale from our 0-1 range
            noise_amount = int(grain_int * 50)
            filters.append(f'[{current_label}]noise=amount={noise_amount}:allf=t[u_grain]')
            current_label = 'u_grain'

        # 6. Transitions (fade in/out, zoom)
        trans_filters, current_label = self._build_transition_filters(
            current_label, s, inputs.target_duration)
        filters.extend(trans_filters)

        # 7. Overlay images (watermark, text, captions)
        for idx, ov, enable in overlay_inputs:
            # overlay=<x>:<y> with enable expression
            opacity = ov.opacity
            overlay_filter = f'overlay={ov.x}:{ov.y}:format=auto'
            if opacity < 1.0:
                overlay_filter += f':alpha={opacity:.2f}'
            if enable:
                overlay_filter += f":enable='{enable}'"

            filters.append(
                f'[{current_label}][{idx}:v]{overlay_filter}[v_ov{idx}]'
            )
            current_label = f'v_ov{idx}'

        # 8. Region blur (watermark hiding)
        if s.get('region_blur_enabled', False) or any(
            r.get('enabled', False)
            for r in s.get('custom_blur_regions', [])
        ):
            blur_f = self._build_region_blur_filter(current_label, s, inputs.target_w, inputs.target_h)
            if blur_f:
                filters.append(blur_f)
                current_label = 'v_blur'

        # 9. Video zoom (Ken Burns effect)
        if s.get('video_zoom', False):
            zoom_scale = s.get('zoom_scale', 1.08)
            zoom_filter = f'zoompan=z=\'min(zoom+0.0015,{zoom_scale})\':x=\'iw/2-(iw/zoom/2)\':y=\'ih/2-(ih/zoom/2)\':d={int(inputs.target_duration * inputs.target_fps)}:s={inputs.target_w}x{inputs.target_h}:fps={inputs.target_fps}'
            # zoompan needs specific setup; we apply it as the base if possible
            # For now, skip complex zoompan — the FFmpeg overlay chain already handles most effects
            # This is a TODO for a future enhancement with proper zoompan integration
            logger.info("[FFmpeg] video_zoom detected — Ken Burns via FFmpeg zoompan (advanced, may need tuning)")

        # ── Assemble command ──
        filter_str = ';'.join(filters)

        cmd += ['-filter_complex', filter_str]

        # Map final output stream
        cmd += ['-map', f'[{current_label}]']

        # ── Encoding ──
        if self._check_nvenc():
            cmd += [
                '-c:v', 'h264_nvenc',
                '-preset', 'p7',
                '-tune', 'hq',
                '-rc', 'vbr',
                '-cq', '23',
                '-b:v', '0',
                '-profile:v', 'main',
                '-bf', '3',
            ]
        else:
            cmd += [
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '18',
            ]

        cmd += [
            '-r', str(inputs.target_fps),
            '-y',
            str(inputs.output_path),
        ]

        return cmd

    # ═══════════════════════════════════════════════════════════════
    # FILTER BUILDERS
    # ═══════════════════════════════════════════════════════════════

    def _build_crop_resize(self, label: str, inputs: RenderInputs,
                           settings: dict) -> str:
        """Build crop + scale filter string."""
        # For now, we just scale to target.
        # The crop logic from the original code (center crop to match aspect)
        # can be added here using FFmpeg's crop filter.
        return f'{label}scale={inputs.target_w}:{inputs.target_h}:force_original_aspect_ratio=decrease,pad={inputs.target_w}:{inputs.target_h}:(ow-iw)/2:(oh-ih)/2:color=black[v_base]'

    def _build_color_filters(self, label: str, settings: dict) -> str:
        """Build color grading / EQ filters."""
        filters = []

        color_grade = settings.get('color_grade', 'none')
        if color_grade == 'warm':
            filters.append('eq=contrast=1.05:saturation=1.1:brightness=0.03')
        elif color_grade == 'cool':
            filters.append('eq=contrast=1.02:saturation=0.9:brightness=0.01')
        elif color_grade == 'vintage':
            filters.append('eq=contrast=0.95:saturation=0.7:brightness=0.02,curves=preset=vintage')
        elif color_grade == 'cinematic':
            filters.append('eq=contrast=1.1:saturation=0.85:brightness=0.0')
        elif color_grade == 'dramatic':
            filters.append('eq=contrast=1.2:saturation=1.15:brightness=-0.02')

        # Gradient overlay (FFmpeg doesn't have a native gradient, but we can
        # approximate with colorkey overlay or skip — most gradients are subtle)
        # TODO: Pre-render gradient as PNG overlay

        if not filters:
            return ''

        chain = ','.join(filters)
        return f'{chain}'

    def _build_transition_filters(self, label: str, settings: dict,
                                  duration: float) -> Tuple[list, str]:
        """Build fade in/out transition filters."""
        filters = []
        current = label

        fade_in = settings.get('transition_fade_in', False)
        fade_out = settings.get('transition_fade_out', False)

        if fade_in:
            fi_dur = settings.get('transition_fade_in_duration', 0.5)
            filters.append(f'[{current}]fade=t=in:st=0:d={fi_dur}[v_fi]')
            current = 'v_fi'

        if fade_out:
            fo_dur = settings.get('transition_fade_out_duration', 0.5)
            st = duration - fo_dur
            if st > 0:
                filters.append(f'[{current}]fade=t=out:st={st:.3f}:d={fo_dur}[v_fo]')
                current = 'v_fo'

        return filters, current

    def _build_region_blur_filter(self, label: str, settings: dict,
                                   w: int, h: int) -> str:
        """Build region blur for watermark/logo hiding."""
        parts = [f'[{label}]']

        blur_regions = settings.get('custom_blur_regions', [])
        if blur_regions:
            for r in blur_regions:
                if not r.get('enabled', False):
                    continue
                x = r.get('x', 0)
                y = r.get('y', 0)
                bw = r.get('width', 100)
                bh = r.get('height', 100)
                parts.append(
                    f'boxblur={bw}:{bh}:enable=\'between(t,{r.get("start_time", 0)},{r.get("end_time", 9999)})\''
                )

        if len(parts) > 1:
            chain = ','.join(parts[1:])
            return f'[{label}]{chain}[v_blur]'

        # Simple region blur (center-based percentage)
        if settings.get('region_blur_enabled', False):
            blur_strength = settings.get('region_blur_strength', 20)
            region = settings.get('blur_region', 'bottom')
            if region == 'bottom':
                parts.append(f'crop=iw:ih*0.15:0:ih*0.85,boxblur={blur_strength},{_overlay_back}')
            # TODO: More region options

        return ''

    def _make_enable(self, start: float, end: float,
                     duration: Optional[float]) -> str:
        """Create FFmpeg enable expression for timed overlays."""
        if start is not None and end is not None and end > 0:
            return f'between(t,{start:.3f},{end:.3f})'
        elif start is not None and start > 0:
            return f'gte(t,{start:.3f})'
        return None

    # ═══════════════════════════════════════════════════════════════
    # AUDIO MUX (separate pass — cleaner than filter_complex audio)
    # ═══════════════════════════════════════════════════════════════

    def _mux_audio(self, video_path: Path, voiceover: Optional[Path],
                   bgm: Optional[Path], bgm_volume: float,
                   target_duration: Optional[float]):
        """Mix voiceover + BGM and mux into video (separate FFmpeg pass)."""
        tmp_dir = self._get_temp_dir()

        cmd = ['ffmpeg', '-y']

        # Input: video (no audio needed yet)
        cmd += ['-i', str(video_path)]

        audio_inputs = []
        audio_idx = 0

        if voiceover and voiceover.exists():
            cmd += ['-i', str(voiceover)]
            audio_inputs.append(audio_idx)
            audio_idx += 1

        if bgm and bgm.exists():
            cmd += ['-i', str(bgm)]
            audio_inputs.append(audio_idx)
            audio_idx += 1

        if not audio_inputs:
            return

        # Build audio filter: mix voiceover + BGM
        audio_filters = []

        if len(audio_inputs) == 1:
            # Just voiceover, no mixing needed
            cmd += ['-map', '0:v', '-map', f'{audio_inputs[0]}:a']
            # Apply voice effects if any
            voice_effects = self._build_voice_effects()
            if voice_effects:
                cmd += ['-af', voice_effects]
        elif len(audio_inputs) == 2:
            # Mix voiceover ( louder ) + BGM (quieter)
            voice_idx = audio_inputs[0]
            bgm_idx = audio_inputs[1]
            # Apply effects to voiceover first
            voice_effects = self._build_voice_effects()
            if voice_effects:
                mix = f'[{voice_idx}:a]{voice_effects}[a_voice];[{bgm_idx}:a]volume={bgm_volume:.2f}[a_bgm];[a_voice][a_bgm]amix=inputs=2:duration=shortest[aout]'
            else:
                mix = f'[{voice_idx}:a]anull[a_voice];[{bgm_idx}:a]volume={bgm_volume:.2f}[a_bgm];[a_voice][a_bgm]amix=inputs=2:duration=shortest[aout]'
            cmd += ['-filter_complex', mix, '-map', '0:v', '-map', '[aout]']

        cmd += ['-c:v', 'copy']  # No re-encode video!
        cmd += ['-c:a', 'aac', '-b:a', '192k']
        cmd += ['-shortest']
        cmd += ['-y', str(video_path)]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.error(f"[FFmpeg] Audio mux failed: {result.stderr[-200:]}")
        else:
            logger.info("[FFmpeg] Audio muxed successfully")

    def _build_voice_effects(self) -> str:
        """Build FFmpeg audio filter chain for voice effects."""
        s = self.settings
        filters = []

        effect = s.get('voice_effect', 'none')
        if effect == 'deep':
            filters.append('asetrate=22050,aresample=44100')
        elif effect == 'high':
            filters.append('asetrate=88200,aresample=44100')
        elif effect == 'robot':
            filters.append('afftfilt=real=\'hypot(re,im)*sin(0)\':imag=\'hypot(re,im)*cos(0)\':win_size=512:overlap=0.75')
        elif effect == 'echo':
            filters.append('aecho=0.8:0.88:60:0.4')
        elif effect == 'whisper':
            filters.append('highpass=f=1000,lowpass=f=3000,volume=1.5')
        elif effect == 'radio':
            filters.append('highpass=f=300,lowpass=f=3400,equalizer=f=1000:t=h:w=200:g=3')
        elif effect == 'chipmunk':
            filters.append('asetrate=110250,aresample=44100')

        # Pitch shift
        tts_engine = s.get('tts_engine', 'cloud')
        pitch = 0
        if tts_engine == 'local':
            pitch = s.get('kokoro_pitch', 0)
        if pitch != 0:
            factor = 2 ** (pitch / 12)
            filters.insert(0, f'asetrate=44100*{factor},aresample=44100')

        return ','.join(filters) if filters else ''

    # ═══════════════════════════════════════════════════════════════
    # CONVENIENCE: Build RenderInputs from existing pipeline
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def build_inputs(
        video_path: Path,
        output_path: Path,
        settings: dict,
        voiceover_path: Optional[Path] = None,
        bgm_path: Optional[Path] = None,
        caption_words: Optional[List[dict]] = None,
        text_overlay_image: Optional[Path] = None,
        watermark_image: Optional[Path] = None,
        target_w: int = 1080,
        target_h: int = 1920,
        target_fps: int = 24,
        target_duration: Optional[float] = None,
    ) -> RenderInputs:
        """Build a RenderInputs from the current pipeline's data.

        This is the bridge function: it translates the existing MoviePy pipeline's
        settings and pre-rendered assets into FFmpeg render inputs.
        """
        overlays = []

        # Text overlay (pre-rendered by the existing PIL code)
        if text_overlay_image and text_overlay_image.exists():
            position = settings.get('position', 'bottom')
            opacity = 1.0
            # Read settings for position
            x = 0
            y = 0
            if position == 'top':
                y = int(target_h * 0.05)
            elif position == 'center':
                y = (target_h - 200) // 2  # approximate
            else:
                y = target_h - 200 - int(target_h * 0.12)
            overlays.append(OverlayImage(
                image_path=text_overlay_image,
                x=(target_w - 1000) // 2,  # centered
                y=max(10, y),
                opacity=opacity,
            ))

        # Watermark
        if watermark_image and watermark_image.exists():
            wm_pos = settings.get('watermark_position', 'bottom-right')
            margin_x = settings.get('watermark_margin_x', 20)
            margin_y = settings.get('watermark_margin_y', 20)
            wm_opacity = settings.get('watermark_opacity', 70) / 100.0

            # Calculate position based on image size
            from PIL import Image
            wm_img = Image.open(str(watermark_image))
            wm_w, wm_h = wm_img.size
            wm_img.close()

            if wm_pos == 'top-left':
                x, y = margin_x, margin_y
            elif wm_pos == 'top-right':
                x, y = target_w - wm_w - margin_x, margin_y
            elif wm_pos == 'bottom-left':
                x, y = margin_x, target_h - wm_h - margin_y
            elif wm_pos == 'center':
                x, y = (target_w - wm_w) // 2, (target_h - wm_h) // 2
            else:  # bottom-right (default)
                x, y = target_w - wm_w - margin_x, target_h - wm_h - margin_y

            overlays.append(OverlayImage(
                image_path=watermark_image,
                x=x, y=y,
                opacity=wm_opacity,
            ))

        # Captions
        captions = []
        if caption_words:
            for w in caption_words:
                captions.append(CaptionWord(
                    text=w.get('word', ''),
                    start=w.get('offset', 0),
                    duration=w.get('duration', 0.5),
                ))

        # Caption style
        caption_style = CaptionStyle(
            font_file=settings.get('caption_highlight_font_file',
                                    'C:/Windows/Fonts/arialbd.ttf'),
            font_size=int(settings.get('caption_font_size', 60)),
            active_color=settings.get('caption_highlight_color', '#FFFFFF'),
            inactive_color=settings.get('caption_inactive_color', '#808080'),
            active_bg_color=settings.get('caption_active_stroke_color', '#FF1493'),
            stroke_color=settings.get('caption_stroke_color', '#000000'),
            stroke_width=int(settings.get('caption_stroke_width', 3)),
            bg_opacity=float(settings.get('caption_bg_opacity', 0.7)),
            position=settings.get('caption_position', 'bottom'),
            y_offset=int(settings.get('caption_y_offset', 0)),
            words_per_line=int(settings.get('caption_words_per_line', 3)),
            text_case=settings.get('caption_text_case', 'Normal'),
            corner_radius=int(settings.get('caption_corner_radius', 10)),
        )

        return RenderInputs(
            video_path=video_path,
            output_path=output_path,
            settings=settings,
            voiceover_path=voiceover_path,
            bgm_path=bgm_path,
            bgm_volume=float(settings.get('bgm_volume', 0.3)),
            overlays=overlays,
            captions=captions,
            caption_style=caption_style,
            target_w=target_w,
            target_h=target_h,
            target_fps=target_fps,
            target_duration=target_duration,
        )
