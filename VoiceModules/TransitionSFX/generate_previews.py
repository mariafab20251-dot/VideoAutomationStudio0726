"""
Transition Preview Generator
Generates short demo videos (2-3s each) for every transition type and saves
them into the previews/ folder. Useful for showing what each transition looks
like before applying it to a real video.

Run:
    python generate_previews.py
Outputs:
    previews/<name>.mp4
"""

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# MoviePy 1.x and 2.x have different API names. Detect which one is in use.
try:
    from moviepy import ImageClip, VideoClip, concatenate_videoclips, CompositeVideoClip
    _MP2 = True
    def _set_dur(c, d):  return c.with_duration(d)
    def _set_fps(c, f):  return c.with_fps(f)
except ImportError:
    from moviepy.editor import ImageClip, VideoClip, concatenate_videoclips, CompositeVideoClip
    _MP2 = False
    def _set_dur(c, d):  return c.set_duration(d)
    def _set_fps(c, f):  return c.set_fps(f)


HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from youtube_video_automation_enhanced import TransitionEffects, LightLeaksEffects


PREVIEW_DIR = HERE / "previews"
PREVIEW_DIR.mkdir(exist_ok=True)

W, H = 480, 270
FPS = 24
SEG_DURATION = 1.0  # 1 second per segment


def make_demo_clip(label: str, color=(60, 100, 160), duration: float = SEG_DURATION) -> ImageClip:
    """Create a labeled segment clip."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :] = color
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((W - tw) // 2, (H - th) // 2), label, fill=(255, 255, 255), font=font)
    return _set_fps(_set_dur(ImageClip(np.array(pil)), duration), FPS)


def save_preview(name: str, clip, fps=FPS):
    out_path = PREVIEW_DIR / f"{name}.mp4"
    try:
        kwargs = dict(
            fps=fps,
            codec="libx264",
            preset="ultrafast",
            audio=False,
        )
        # MoviePy 1.x accepts verbose/logger; 2.x does not
        if not _MP2:
            kwargs["verbose"] = False
            kwargs["logger"] = None
        clip.write_videofile(str(out_path), **kwargs)
        size_kb = out_path.stat().st_size / 1024
        print(f"  [OK] {name}.mp4 ({size_kb:.1f} KB)")
    except Exception as e:
        print(f"  [WARN] {name}.mp4 failed: {e}")


def gen_fade_in():
    a = make_demo_clip("Scene A", color=(180, 60, 60))
    b = make_demo_clip("Scene B", color=(60, 180, 100))
    full = concatenate_videoclips([a, b])
    return TransitionEffects.apply_fade_transition(full, fade_in_duration=SEG_DURATION, fade_out_duration=0)


def gen_fade_out():
    a = make_demo_clip("Scene A", color=(180, 60, 60))
    b = make_demo_clip("Scene B", color=(60, 180, 100))
    full = concatenate_videoclips([a, b])
    return TransitionEffects.apply_fade_transition(full, fade_in_duration=0, fade_out_duration=SEG_DURATION)


def gen_zoom_in():
    a = make_demo_clip("Scene A", color=(60, 100, 180))
    return TransitionEffects.create_zoom_transition(a, zoom_in=True, duration=SEG_DURATION, zoom_scale=1.5)


def gen_zoom_out():
    a = make_demo_clip("Scene A", color=(60, 100, 180))
    return TransitionEffects.create_zoom_transition(a, zoom_in=False, duration=SEG_DURATION, zoom_scale=1.5)


def gen_blur_in():
    a = make_demo_clip("Scene A", color=(100, 60, 180))
    return TransitionEffects.create_blur_transition(a, blur_in=True, duration=SEG_DURATION, max_blur=20)


def gen_blur_out():
    a = make_demo_clip("Scene A", color=(100, 60, 180))
    return TransitionEffects.create_blur_transition(a, blur_in=False, duration=SEG_DURATION, max_blur=20)


def gen_slide_in():
    a = make_demo_clip("Scene A", color=(180, 140, 60))
    return TransitionEffects.create_slide_transition(a, direction="left", in_transition=True, duration=SEG_DURATION)


def gen_slide_out():
    a = make_demo_clip("Scene A", color=(180, 140, 60))
    return TransitionEffects.create_slide_transition(a, direction="right", in_transition=False, duration=SEG_DURATION)


def gen_wipe_in():
    a = make_demo_clip("Scene A", color=(60, 180, 180))
    return TransitionEffects.create_wipe_transition(a, direction="right", in_transition=True, duration=SEG_DURATION)


def gen_wipe_out():
    a = make_demo_clip("Scene A", color=(60, 180, 180))
    return TransitionEffects.create_wipe_transition(a, direction="left", in_transition=False, duration=SEG_DURATION)


def gen_glitch_start():
    a = make_demo_clip("Scene A", color=(200, 50, 200))
    return TransitionEffects.create_glitch_transition(a, glitch_start=True, duration=SEG_DURATION, intensity=0.7)


def gen_glitch_end():
    a = make_demo_clip("Scene A", color=(200, 50, 200))
    return TransitionEffects.create_glitch_transition(a, glitch_start=False, duration=SEG_DURATION, intensity=0.7)


def gen_cinematic_bars():
    a = make_demo_clip("Cinema", color=(40, 40, 60))
    return TransitionEffects.create_cinematic_bars(a, fade_in=True, duration=SEG_DURATION, bar_height_percent=12)


def gen_lens_flare():
    a = make_demo_clip("Lens", color=(80, 80, 120))
    try:
        flare = LightLeaksEffects.create_lens_flare(W, H, SEG_DURATION, FPS)
    except TypeError:
        # Older signature
        flare = LightLeaksEffects.create_lens_flare(W, H, SEG_DURATION, FPS,
                                                     intensity=0.7, start_time=0,
                                                     flare_duration=SEG_DURATION)
    return _composite_rgba(a, flare)


def gen_light_leak():
    a = make_demo_clip("Leak", color=(80, 80, 120))
    try:
        leak = LightLeaksEffects.create_light_leak(
            W, H, SEG_DURATION, FPS,
            color="warm", intensity=0.7,
            start_time=0, leak_duration=SEG_DURATION,
            direction="top_right"
        )
    except TypeError:
        leak = LightLeaksEffects.create_light_leak(W, H, SEG_DURATION, FPS)
    return _composite_rgba(a, leak)


def gen_film_burn():
    a = make_demo_clip("Burn", color=(80, 80, 120))
    try:
        burn = LightLeaksEffects.create_film_burn(
            W, H, SEG_DURATION, FPS,
            intensity=0.7, start_time=0, burn_duration=SEG_DURATION
        )
    except TypeError:
        burn = LightLeaksEffects.create_film_burn(W, H, SEG_DURATION, FPS)
    return _composite_rgba(a, burn)


def _composite_rgba(base_clip, overlay_clip):
    """Composite an RGBA overlay onto an RGB base, converting to RGB."""
    if overlay_clip is None:
        return base_clip
    def make_rgb(get_frame, t):
        base = get_frame(t)
        # Get overlay frame
        try:
            over = overlay_clip.get_frame(t)
        except Exception:
            return base
        if over.ndim == 3 and over.shape[2] == 4:
            # Alpha-blend onto base
            alpha = over[:, :, 3:4].astype(np.float32) / 255.0
            rgb = over[:, :, :3].astype(np.float32)
            base_f = base.astype(np.float32)
            out = (rgb * alpha + base_f * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
            return out
        return over if over.ndim == 3 else np.stack([over] * 3, axis=-1)
    try:
        return base_clip.transform(make_rgb)
    except AttributeError:
        return base_clip.fl(make_rgb)


def gen_zoom_pulse():
    """Zoom pulse at intervals (3s clip with 1.5s interval so we see 2 pulses)."""
    a = make_demo_clip("Pulse", color=(60, 100, 180), duration=3.0)
    return TransitionEffects.create_zoom_pulse(a, duration=1.0, zoom_scale=1.5, interval=1.5)


def gen_blur_pulse():
    a = make_demo_clip("Pulse", color=(100, 60, 180), duration=3.0)
    return TransitionEffects.create_blur_pulse(a, duration=0.7, max_blur=20, interval=1.5)


def gen_glitch_pulse():
    a = make_demo_clip("Pulse", color=(200, 50, 200), duration=3.0)
    return TransitionEffects.create_glitch_pulse(a, duration=0.7, intensity=0.7, interval=1.5)


def gen_shake_pulse():
    a = make_demo_clip("Pulse", color=(200, 80, 60), duration=3.0)
    return TransitionEffects.create_shake_pulse(a, duration=0.5, intensity=0.08, interval=1.5)


GENERATORS = [
    ("01_fade_in",            gen_fade_in),
    ("02_fade_out",           gen_fade_out),
    ("03_zoom_in",            gen_zoom_in),
    ("04_zoom_out",           gen_zoom_out),
    ("05_blur_in",            gen_blur_in),
    ("06_blur_out",           gen_blur_out),
    ("07_slide_in",           gen_slide_in),
    ("08_slide_out",          gen_slide_out),
    ("09_wipe_in",            gen_wipe_in),
    ("10_wipe_out",           gen_wipe_out),
    ("11_glitch_start",       gen_glitch_start),
    ("12_glitch_end",         gen_glitch_end),
    ("13_cinematic_bars",     gen_cinematic_bars),
    ("14_lens_flare",         gen_lens_flare),
    ("15_light_leak",         gen_light_leak),
    ("16_film_burn",          gen_film_burn),
    ("17_zoom_pulse",         gen_zoom_pulse),
    ("18_blur_pulse",         gen_blur_pulse),
    ("19_glitch_pulse",       gen_glitch_pulse),
    ("20_shake_pulse",        gen_shake_pulse),
]


def main():
    print(f"Generating {len(GENERATORS)} transition previews to {PREVIEW_DIR}")
    print("=" * 60)
    for name, gen in GENERATORS:
        print(f"  Generating {name}...")
        try:
            clip = gen()
            save_preview(name, clip)
            clip.close()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [ERROR] {name}: {e}")
    print("=" * 60)
    files = sorted(PREVIEW_DIR.glob("*.mp4"))
    print(f"Done! {len(files)} preview files in {PREVIEW_DIR}")
    for f in files:
        print(f"  - {f.name} ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
