"""
Wav2Lip integration helper for AI avatar lip-syncing.

Wraps Wav2Lip inference so the main pipeline can generate talking-avatar
videos from a face image + audio without importing Wav2Lip's fragile
dependencies directly into the ChangeGUI process.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────
WAV2LIP_DIR = Path(os.environ.get(
    'WAV2LIP_DIR',
    r'D:\GitHub\Wav2Lip',
))
WAV2LIP_CHECKPOINT = WAV2LIP_DIR / 'checkpoints' / 'wav2lip_gan.pth'
WAV2LIP_INFERENCE = WAV2LIP_DIR / 'inference.py'
TEMP_DIR = WAV2LIP_DIR / 'temp'

# ── Python executable auto-detect ─────────────────────────────
# Wav2Lip's dependencies (librosa, torch, face_detection) may
# not be installed in the same Python that runs the GUI (e.g. GUI
# runs on 3.13 but Wav2Lip deps are in 3.11). Probe known paths
# and cache the first one that can import librosa.
_WAV2LIP_PYTHON: Optional[str] = None


def _find_wav2lip_python() -> str:
    """Return a path to a Python executable that can run Wav2Lip
    (has librosa, torch, face_detection). Cached after first probe."""
    global _WAV2LIP_PYTHON
    if _WAV2LIP_PYTHON is not None:
        return _WAV2LIP_PYTHON

    candidates = [
        sys.executable,  # The GUI's Python (might work)
        # Common alternate installations
        r'C:\Users\shahi\AppData\Local\Programs\Python\Python311\python.exe',
        r'C:\Users\shahi\AppData\Local\Programs\Python\Python312\python.exe',
        'python3',
        'python',
    ]
    import subprocess as _sp
    for cand in candidates:
        if not cand:
            continue
        try:
            r = _sp.run(
                [cand, '-c', 'import librosa'],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                _WAV2LIP_PYTHON = cand
                logger.info(f'Wav2Lip using Python: {cand}')
                return cand
        except (FileNotFoundError, OSError, _sp.TimeoutExpired):
            continue

    # Fallback — try the GUI Python anyway (will likely fail)
    _WAV2LIP_PYTHON = sys.executable or 'python'
    logger.warning(f'Wav2Lip Python auto-detect failed — falling back to {_WAV2LIP_PYTHON}')
    return _WAV2LIP_PYTHON


def _ensure_temp_dir():
    """Ensure the temp directory exists (Wav2Lip writes .avi there)."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


def extract_audio(video_path: Path, output_wav: Path) -> bool:
    """Extract audio from a video file into a WAV (16kHz mono)."""
    import subprocess as sp
    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        str(output_wav),
    ]
    try:
        sp.run(cmd, check=True, capture_output=True, timeout=120)
        return output_wav.exists() and output_wav.stat().st_size > 0
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        return False


def is_available() -> bool:
    """Check whether Wav2Lip model & inference script are present."""
    return WAV2LIP_CHECKPOINT.exists() and WAV2LIP_INFERENCE.exists()


def run_wav2lip(
    face_path: Path,
    audio_path: Path,
    output_path: Path,
    pads: Optional[list] = None,
    resize_factor: int = 2,
    nosmooth: bool = False,
) -> Optional[Path]:
    """Run Wav2Lip inference via subprocess.

    Args:
        face_path: Path to face image (jpg/png) or video.
        audio_path: Path to audio file (wav/mp4).
        output_path: Desired output video path (.mp4).
        pads: Padding [top, bottom, left, right] to include more chin etc.
        resize_factor: Downscale input by this factor (1=full, 2=half, etc.).
        nosmooth: Disable temporal face-box smoothing.

    Returns:
        Path to output video on success, or None on failure.
    """
    if not is_available():
        logger.error("Wav2Lip not available — missing checkpoint or inference.py")
        return None

    if not face_path.exists():
        logger.error(f"Face image not found: {face_path}")
        return None

    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return None

    pads = pads or [0, 20, 0, 0]

    # Ensure output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure temp directory for intermediate .avi
    _ensure_temp_dir()

    cmd = [
        _find_wav2lip_python(),
        str(WAV2LIP_INFERENCE),
        '--checkpoint_path', str(WAV2LIP_CHECKPOINT),
        '--face', str(face_path),
        '--audio', str(audio_path),
        '--outfile', str(output_path),
        '--pads', str(pads[0]), str(pads[1]), str(pads[2]), str(pads[3]),
        '--resize_factor', str(resize_factor),
    ]

    if nosmooth:
        cmd.append('--nosmooth')

    # Use the face image extension to determine static mode
    # (inference.py auto-detects based on file extension)

    logger.info(f"Running Wav2Lip: {' '.join(str(c) for c in cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(WAV2LIP_DIR),  # Wav2Lip's audio.py needs to import from its own dir
            check=True,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
        )
        # Log output for debugging
        for line in result.stdout.splitlines():
            logger.debug(f"[Wav2Lip] {line}")
        if result.stderr:
            for line in result.stderr.splitlines():
                logger.debug(f"[Wav2Lip:stderr] {line}")

        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"Wav2Lip succeeded: {output_path}")
            return output_path
        else:
            logger.error("Wav2Lip ran but produced no output file")
            return None

    except subprocess.CalledProcessError as e:
        logger.error(f"Wav2Lip failed (exit={e.returncode}): {e.stderr}")
        return None
    except subprocess.TimeoutExpired:
        logger.error("Wav2Lip timed out after 5 minutes")
        return None
    except Exception as e:
        logger.error(f"Wav2Lip error: {e}")
        return None


def composite_avatar(
    main_video_path: Path,
    avatar_video_path: Path,
    output_path: Path,
    position: str = 'bottom-right',
    avatar_scale: float = 0.25,
) -> Optional[Path]:
    """Overlay the avatar video onto the main video as picture-in-picture.

    Args:
        main_video_path: Original rendered video.
        avatar_video_path: Wav2Lip lip-synced face video.
        output_path: Where to write the composited video.
        position: One of 'bottom-right', 'bottom-left', 'top-right', 'top-left', 'center'.
        avatar_scale: Fraction of main video width the avatar should occupy.

    Returns:
        Path to composited video on success, or None.
    """
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip
        from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
    except ImportError:
        logger.error("moviepy not available for compositing")
        return None

    try:
        main = VideoFileClip(str(main_video_path))
        avatar = VideoFileClip(str(avatar_video_path))

        # Calculate target size
        av_w = int(main.w * avatar_scale)
        av_h = int(avatar.h * (av_w / avatar.w))
        from moviepy.video.fx.resize import resize as fx_resize
        avatar_resized = avatar.fx(fx_resize, newsize=(av_w, av_h))

        # Calculate position
        margin = 30
        positions = {
            'bottom-right': (main.w - av_w - margin, main.h - av_h - margin),
            'bottom-left': (margin, main.h - av_h - margin),
            'top-right': (main.w - av_w - margin, margin),
            'top-left': (margin, margin),
            'center': ((main.w - av_w) // 2, (main.h - av_h) // 2),
        }
        pos = positions.get(position, positions['bottom-right'])

        final = CompositeVideoClip([main, avatar_resized.set_position(pos)])

        final.write_videofile(
            str(output_path),
            codec='libx264',
            audio_codec='aac',
            fps=24,
            preset='ultrafast',
            threads=8,
            logger='bar',
        )

        main.close()
        avatar.close()
        final.close()

        logger.info(f"Avatar composited: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Avatar compositing failed: {e}")
        import traceback
        traceback.print_exc()
        return None
