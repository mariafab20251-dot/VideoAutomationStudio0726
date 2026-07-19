#!/usr/bin/env python
"""
Vertical Reframe (9:16) with speaking-face tracking
===================================================
Standalone, Tk-free helper that converts a landscape (or any) video into a
vertical 1080x1920 short, keeping the on-screen speaker inside the frame.

Pipeline (single OpenCV decode pass, ffmpeg-piped encode):

    detect faces per sampled frame (Haar cascade — no extra deps)
      → pick the dominant face (largest, most central, temporally stable)
      → build a per-frame crop-centre track
      → smooth the track (EMA + clamp velocity) so it glides, never jitters
      → crop a 9:16 window around it, resize to 1080x1920
      → pipe frames to ffmpeg, then mux the ORIGINAL audio back untouched.

Kept deliberately separate from ``complete_automation_gui.py`` so the huge
main file isn't touched.  Reuses cv2 + numpy that the project already ships.

Public API
----------
    reframe_vertical(src_video, out_video, settings=None, log=None,
                     progress=None) -> Path | None
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

# Target canvas — matches the app's STANDARD_WIDTH/HEIGHT (1080x1920, 9:16).
OUT_W = 1080
OUT_H = 1920


def _noop_log(level, msg):
    print(f"[{level.upper()}] {msg}")


# ── Face detection ──────────────────────────────────────────────────────


def _load_cascade():
    """Load the frontal-face Haar cascade shipped with opencv."""
    import cv2
    path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    clf = cv2.CascadeClassifier(path)
    if clf.empty():
        return None
    return clf


def _pick_face(faces, frame_w, frame_h, prev_cx):
    """Choose the dominant face from a list of (x, y, w, h) detections.

    Scores each face by size and centrality, with a bonus for being near the
    previously-tracked centre so we don't hop between people every frame.
    Returns the chosen face centre-x (float) or ``None``.
    """
    if len(faces) == 0:
        return None
    best = None
    best_score = -1.0
    for (x, y, w, h) in faces:
        cx = x + w / 2.0
        area = (w * h) / float(frame_w * frame_h)          # 0..1
        centrality = 1.0 - abs(cx - frame_w / 2.0) / (frame_w / 2.0)  # 0..1
        score = area * 2.0 + centrality * 0.5
        if prev_cx is not None:
            # Reward continuity with the face we were already following.
            closeness = 1.0 - min(abs(cx - prev_cx) / frame_w, 1.0)
            score += closeness * 1.5
        if score > best_score:
            best_score = score
            best = cx
    return best


# ── Main reframe ────────────────────────────────────────────────────────


def reframe_vertical(src_video, out_video, settings=None, log=None,
                     progress=None, out_w=None, out_h=None):
    """Convert *src_video* to a face-tracked crop at the target aspect.

    Defaults to 9:16 (1080x1920).  Pass ``out_w``/``out_h`` for other
    aspects, e.g. 1080x1080 for a 1:1 square recap.

    Returns the output Path, or ``None`` on failure.  ``progress(done, total,
    note)`` is an optional UI callback.
    """
    import cv2
    import numpy as np

    log = log or _noop_log
    prog = progress or (lambda *a, **k: None)
    settings = settings or {}
    # Target canvas — default to the module 9:16, override per call.
    OUT_W = int(out_w) if out_w else globals()['OUT_W']
    OUT_H = int(out_h) if out_h else globals()['OUT_H']
    src_video = Path(src_video)
    out_video = Path(out_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)

    if not src_video.is_file():
        log('error', f'Reframe: source not found: {src_video}')
        return None

    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        log('error', f'Reframe: cannot open {src_video.name}')
        return None

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    if src_w <= 0 or src_h <= 0:
        log('error', 'Reframe: bad source dimensions')
        cap.release()
        return None

    # If the source is already portrait/near-9:16, a face-track crop buys
    # little — just normalise the size and skip the heavy per-frame work.
    src_ar = src_w / float(src_h)
    target_ar = OUT_W / float(OUT_H)  # 0.5625

    # The crop window taken from the SOURCE is the tallest 9:16 rectangle that
    # fits the source height, then panned horizontally to follow the face.
    crop_h = src_h
    crop_w = int(round(crop_h * target_ar))
    if crop_w > src_w:
        # Source is narrower than 9:16 (already portrait-ish): pillar via width.
        crop_w = src_w
        crop_h = int(round(crop_w / target_ar))
        if crop_h > src_h:
            crop_h = src_h
    max_x = max(0, src_w - crop_w)
    max_y = max(0, src_h - crop_h)
    center_x_default = (src_w - crop_w) / 2.0 + crop_w / 2.0

    log('info',
        f'Reframe: {src_w}x{src_h} @ {fps:.0f}fps → {OUT_W}x{OUT_H} crop '
        f'{crop_w}x{crop_h} (pan range 0..{max_x}px)')

    # ── Pass 1: sample faces to build a smoothed centre-x track ──────────
    cascade = _load_cascade()
    if cascade is None:
        log('warn', 'Reframe: face cascade unavailable — using centre crop')

    # Sample every Nth frame for detection (detection is the slow part); the
    # track is interpolated + smoothed across all frames afterwards.
    sample_stride = max(1, int(round(fps / 6.0)))   # ~6 detections/sec
    samples = []          # (frame_idx, center_x)
    prev_cx = None
    idx = 0
    detect_scale = 640.0 / src_w if src_w > 640 else 1.0  # shrink for speed

    prog(0, 1, 'Analyzing faces…')
    if cascade is not None:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % sample_stride == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                if detect_scale != 1.0:
                    small = cv2.resize(
                        frame, None, fx=detect_scale, fy=detect_scale,
                        interpolation=cv2.INTER_AREA)
                else:
                    small = frame
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5,
                    minSize=(int(30 * detect_scale) or 20,
                             int(30 * detect_scale) or 20))
                cx = _pick_face(faces, small.shape[1], small.shape[0],
                                (prev_cx * detect_scale
                                 if prev_cx is not None else None))
                if cx is not None:
                    cx = cx / detect_scale     # back to source pixels
                    prev_cx = cx
                    samples.append((idx, cx))
            idx += 1
            if total_frames:
                prog(idx, total_frames, 'Analyzing faces…')
        if total_frames == 0:
            total_frames = idx

    cap.release()

    # Build a per-frame centre-x track (default to centre when no faces seen).
    if not samples:
        log('info', 'Reframe: no faces detected — static centre crop')
        track_cx = [center_x_default] * max(total_frames, 1)
    else:
        # Linear-interpolate sampled centres across every frame.
        track_cx = [None] * max(total_frames, 1)
        for (fidx, cx) in samples:
            if 0 <= fidx < len(track_cx):
                track_cx[fidx] = cx
        # Fill leading/trailing/gaps by interpolation.
        known = [i for i, v in enumerate(track_cx) if v is not None]
        first, last = known[0], known[-1]
        for i in range(first):
            track_cx[i] = track_cx[first]
        for i in range(last + 1, len(track_cx)):
            track_cx[i] = track_cx[last]
        ki = 0
        for i in range(first, last + 1):
            if track_cx[i] is not None:
                ki = i
                continue
            # find next known
            nj = next(j for j in known if j > i)
            span = nj - ki
            frac = (i - ki) / float(span) if span else 0.0
            track_cx[i] = track_cx[ki] + (track_cx[nj] - track_cx[ki]) * frac

        # ── Smooth: EMA both directions + clamp per-frame velocity ───────
        alpha = float(settings.get('reframe_smooth', 0.12))  # lower = smoother
        max_vel = float(settings.get('reframe_max_pan_px', 12.0))  # px/frame
        sm = list(track_cx)
        for i in range(1, len(sm)):
            sm[i] = sm[i - 1] * (1 - alpha) + sm[i] * alpha
        for i in range(len(sm) - 2, -1, -1):
            sm[i] = sm[i + 1] * (1 - alpha) + sm[i] * alpha
        # Velocity clamp so a big scene change eases instead of snapping.
        for i in range(1, len(sm)):
            dv = sm[i] - sm[i - 1]
            if dv > max_vel:
                sm[i] = sm[i - 1] + max_vel
            elif dv < -max_vel:
                sm[i] = sm[i - 1] - max_vel
        track_cx = sm
        log('ok', f'Reframe: face track from {len(samples)} detections')

    # Convert centre-x → crop x1 and clamp to frame.
    def _x1(cx):
        x1 = int(round(cx - crop_w / 2.0))
        return max(0, min(max_x, x1))
    crop_y1 = max(0, min(max_y, (src_h - crop_h) // 2))

    # ── Pass 2: crop + resize each frame, pipe to ffmpeg ─────────────────
    tmp_video = out_video.with_suffix('.reframe_noaudio.mp4')
    # Detect NVENC availability (mirrors the app's fast path); fall back to x264.
    use_nvenc = bool(settings.get('reframe_use_nvenc', True))
    vcodec = ['-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '23'] if use_nvenc \
        else ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20']

    ff = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{OUT_W}x{OUT_H}', '-r', f'{fps:.5f}',
        '-i', 'pipe:0',
        *vcodec, '-pix_fmt', 'yuv420p',
        str(tmp_video),
    ]
    proc = subprocess.Popen(ff, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    cap = cv2.VideoCapture(str(src_video))
    prog(0, 1, 'Reframing…')
    written = 0
    fi = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cx = track_cx[fi] if fi < len(track_cx) else center_x_default
            x1 = _x1(cx)
            crop = frame[crop_y1:crop_y1 + crop_h, x1:x1 + crop_w]
            if crop.shape[1] != crop_w or crop.shape[0] != crop_h:
                # Edge frame smaller than expected — pad/resize defensively.
                crop = cv2.resize(crop, (crop_w, crop_h))
            out = cv2.resize(crop, (OUT_W, OUT_H),
                             interpolation=cv2.INTER_AREA)
            try:
                proc.stdin.write(out.tobytes())
            except (BrokenPipeError, OSError):
                break
            written += 1
            fi += 1
            if total_frames:
                prog(fi, total_frames, 'Reframing…')
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except Exception:
            pass
        _err = b''
        try:
            _, _err = proc.communicate(timeout=600)
        except Exception:
            proc.kill()

    if proc.returncode not in (0, None) or not tmp_video.is_file() \
            or tmp_video.stat().st_size == 0:
        log('error',
            f'Reframe: encode failed — {(_err or b"").decode("utf-8", "replace")[-300:]}')
        try:
            tmp_video.unlink(missing_ok=True)
        except Exception:
            pass
        return None

    if written == 0:
        log('error', 'Reframe: no frames written')
        try:
            tmp_video.unlink(missing_ok=True)
        except Exception:
            pass
        return None

    # ── Mux the ORIGINAL audio back over the reframed video ──────────────
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', str(tmp_video), '-i', str(src_video),
             '-map', '0:v:0', '-map', '1:a:0?',
             '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
             '-shortest', str(out_video)],
            capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        _e = (e.stderr or b'').decode('utf-8', 'replace')[-300:]
        log('warn', f'Reframe: audio mux failed ({_e}); keeping video-only')
        try:
            shutil.move(str(tmp_video), str(out_video))
        except Exception:
            return None
        return out_video
    finally:
        try:
            tmp_video.unlink(missing_ok=True)
        except Exception:
            pass

    log('ok', f'Reframe: ✅ vertical short → {out_video.name} '
              f'({written} frames)')
    return out_video
