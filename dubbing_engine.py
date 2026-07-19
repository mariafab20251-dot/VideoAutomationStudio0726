#!/usr/bin/env python
"""
Dubbing Engine
==============
Standalone, Tk-free logic that re-voices a video's ORIGINAL dialogue into a
target language:

    whisper transcribe  →  group into lines  →  translate (Gemini)  →
    TTS each line  →  overlay on the (ducked) original audio  →  mux back
    into a new video file.

Kept deliberately separate from ``complete_automation_gui.py`` so the huge
main file isn't touched.  The heavy dependencies (whisper, TTS engine,
translation) are all reused from the existing project modules at call time,
so this file adds no new install requirements.

Public API
----------
    group_words_into_segments(word_timings, max_gap=0.7, max_words=14) -> list
    transcribe_video(video_path, settings, log) -> list[word_timing]
    build_dubbed_audio(video_path, out_mp3, target_language, settings, log,
                       progress=None) -> Path | None
    dub_video(video_path, out_video, target_language, settings, log,
              progress=None) -> Path | None
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

SAMPLE_RATE = 44100


def _noop_log(level, msg):
    print(f"[{level.upper()}] {msg}")


def _atempo_stretch(data, factor, log=_noop_log):
    """Speed up ``data`` (float32 mono @ SAMPLE_RATE) by ``factor`` WITHOUT
    changing pitch, using ffmpeg's WSOLA ``atempo`` filter.

    ``factor`` > 1 makes it shorter/faster.  Values are chained so we stay in
    atempo's stable 0.5–2.0 range even for a factor like 1.3 (single pass).
    Returns the stretched float32 array, or the original on any failure.
    """
    import numpy as np
    import soundfile as sf
    import tempfile, subprocess
    from pathlib import Path

    if factor <= 1.001:
        return data
    # Build a chain of atempo values each within [0.5, 2.0].
    chain = []
    f = factor
    while f > 2.0:
        chain.append(2.0)
        f /= 2.0
    chain.append(f)
    filt = ','.join(f'atempo={c:.5f}' for c in chain)

    tmp_in = tmp_out = None
    try:
        tmp_in = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp_in.close()
        tmp_out = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp_out.close()
        sf.write(tmp_in.name, data, SAMPLE_RATE)
        subprocess.run(
            ['ffmpeg', '-y', '-i', tmp_in.name, '-filter:a', filt,
             '-ar', str(SAMPLE_RATE), '-ac', '1', tmp_out.name],
            capture_output=True, check=True)
        out, _ = sf.read(tmp_out.name)
        if out.ndim == 2:
            out = out.mean(axis=1)
        return out.astype(np.float32)
    except Exception as e:
        log('warn', f'Dub: atempo stretch failed ({e}); using natural speed')
        return data
    finally:
        for _p in (tmp_in, tmp_out):
            if _p is not None:
                try:
                    Path(_p.name).unlink(missing_ok=True)
                except Exception:
                    pass


# ── Word grouping ───────────────────────────────────────────────────────


def group_words_into_segments(word_timings: list,
                              max_gap: float = 0.7,
                              max_words: int = 14) -> list:
    """Group whisper word timings into speakable dialogue segments.

    A new segment starts when there's a silence gap > ``max_gap`` between
    words, when the previous word ends a sentence (. ! ? …), or when the
    current segment reaches ``max_words`` words.

    Returns ``[{'start': float, 'end': float, 'text': str}, ...]``.
    """
    segs = []
    cur = []            # list of (word, offset)
    cur_start = None
    prev_end = None
    for w in word_timings:
        try:
            off = float(w.get('offset', 0))
            dur = float(w.get('duration', 0))
        except Exception:
            continue
        word = str(w.get('word', '')).strip()
        if not word:
            continue
        gap = (off - prev_end) if prev_end is not None else 0.0
        ends_sentence = bool(cur) and cur[-1][0][-1:] in '.!?…'
        if cur and (gap > max_gap or ends_sentence or len(cur) >= max_words):
            segs.append({
                'start': cur_start,
                'end': prev_end,
                'text': ' '.join(t for t, _ in cur).strip(),
            })
            cur = []
            cur_start = None
        if cur_start is None:
            cur_start = off
        cur.append((word, off))
        prev_end = off + dur
    if cur:
        segs.append({
            'start': cur_start,
            'end': prev_end,
            'text': ' '.join(t for t, _ in cur).strip(),
        })
    return [s for s in segs if s['text']]


# ── Transcription ───────────────────────────────────────────────────────


def _resolve_whisper_python(settings: dict, log) -> str | None:
    """Find a Python interpreter that has whisper installed."""
    import subprocess, sys as _sys
    from pathlib import Path

    _candidates = []
    try:
        _candidates.append(str(Path(_sys.executable).resolve()))
    except Exception:
        pass
    _from_setting = str(settings.get('whisper_python_path', '') or '').strip()
    if _from_setting:
        _candidates.append(str(Path(_from_setting).resolve()))
    _candidates.append(
        str(Path(r'D:\GitHub\pythonprojects\VideoTextExtractor\venv\Scripts\python.exe').resolve()))

    _seen = set()
    for c in _candidates:
        c = str(Path(c).resolve())
        if c in _seen or not Path(c).is_file():
            continue
        _seen.add(c)
        try:
            r = subprocess.run([c, '-c', 'import whisper; print(whisper.__version__)'],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0:
                log('info', f'Dub: whisper Python = {c} (v{r.stdout.strip()})')
                return c
        except Exception:
            pass
    log('warn', 'Dub: no Python with whisper found')
    return None


def transcribe_video(video_path: Path, settings: dict, log=None,
                     source_language: str | None = None) -> list:
    """Transcribe the video's own audio → word-level timings via whisper.

    Calls ``_whisper_word_timestamps.py`` directly with the correct
    ``--language`` flag so non-English source videos (Urdu, Russian, etc.)
    transcribe correctly.  ``source_language`` should be a 2-letter ISO 639-1
    code (e.g. 'ur', 'ru', 'hi') or a full language name; defaults to 'en'.
    """
    import subprocess, json, tempfile, sys as _sys
    from pathlib import Path

    log = log or _noop_log

    # --- Resolve a python that has whisper installed -------------------------
    venv_python = _resolve_whisper_python(settings, log)
    if not venv_python:
        return []

    # --- Extract audio from video to a temp WAV ------------------------------
    cleanup_temp = False
    try:
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.close()
        audio_to_transcribe = tmp.name
        cleanup_temp = True
        log('info', f'Dub: extracting audio from {video_path.name} …')
        subprocess.run(
            ['ffmpeg', '-y', '-i', str(video_path),
             '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
             audio_to_transcribe],
            capture_output=True, check=True, timeout=120)
    except Exception as e:
        log('error', f'Dub: could not extract audio: {e}')
        return []

    # --- Run whisper via the venv subprocess ---------------------------------
    whisper_script = Path(__file__).parent / '_whisper_word_timestamps.py'
    if not whisper_script.exists():
        log('error', f'Dub: whisper helper not found: {whisper_script}')
        return []

    # Map source_language to whisper's 2-letter code (default 'en')
    whisper_lang = 'en'
    if source_language:
        sl = source_language.strip().lower()
        # Map full language names to ISO 639-1 codes
        _LANG_MAP = {
            'chinese': 'zh', 'japanese': 'ja', 'korean': 'ko',
            'arabic': 'ar', 'russian': 'ru', 'hindi': 'hi',
            'urdu': 'ur', 'vietnamese': 'vi', 'thai': 'th',
            'telugu': 'te', 'tamil': 'ta', 'malayalam': 'ml',
            'kannada': 'kn', 'bengali': 'bn', 'marathi': 'mr',
            'gujarati': 'gu', 'punjabi': 'pa', 'odia': 'or',
            'english': 'en', 'french': 'fr', 'german': 'de',
            'spanish': 'es', 'portuguese': 'pt', 'italian': 'it',
            'dutch': 'nl', 'turkish': 'tr', 'indonesian': 'id',
            'malay': 'ms', 'tagalog': 'tl', 'swahili': 'sw',
        }
        if sl in _LANG_MAP:
            whisper_lang = _LANG_MAP[sl]
        elif sl not in ('en', 'english', 'en-us', 'en-gb', ''):
            whisper_lang = sl[:2]  # fallback: take first 2 chars

    try:
        result = subprocess.run(
            [venv_python, str(whisper_script), audio_to_transcribe,
             '--model', 'base', '--language', whisper_lang],
            capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            log('error', f'Dub: whisper error (rc={result.returncode}): '
                        f'{result.stderr[:200]}')
            return []
        word_timings = json.loads(result.stdout)
        if not isinstance(word_timings, list):
            log('error', 'Dub: whisper returned non-list output')
            return []
        log('ok', f'Dub: whisper returned {len(word_timings)} word(s)')
        return word_timings
    except Exception as e:
        log('error', f'Dub: whisper subprocess failed: {e}')
        return []
    finally:
        if cleanup_temp:
            try:
                Path(audio_to_transcribe).unlink(missing_ok=True)
            except Exception:
                pass


# ── Full dub-audio build ────────────────────────────────────────────────


def build_dubbed_audio(video_path: Path, out_mp3: Path, target_language: str,
                       settings: dict, log=None, progress=None,
                       source_language: str | None = None):
    """Produce a full-length dubbed audio track as an MP3.

    The original audio is kept as a low background bed and ducked further
    under each spoken line, with the translated TTS voice laid on top at the
    original timing.  Returns the MP3 Path, or ``None`` on failure.

    ``progress(done, total, note)`` — optional callback for UI updates.
    """
    import numpy as np
    import soundfile as sf

    log = log or _noop_log
    prog = progress or (lambda *a, **k: None)
    video_path = Path(video_path)
    out_mp3 = Path(out_mp3)

    tgt = (target_language or '').strip()
    if not tgt or tgt.lower() in ('english', 'en', 'en-us', 'en-gb'):
        log('warn', 'Dub: target language is English — nothing to translate/dub')
        return None

    try:
        from youtube_video_automation_enhanced import TTSGenerator
    except Exception as e:
        log('error', f'Dub: could not import TTSGenerator: {e}')
        return None

    DUCK_VOL = float(settings.get('dub_original_duck', 0.12))   # original under dub
    BG_VOL = float(settings.get('dub_original_bg', 0.55))       # original elsewhere

    # 1) Transcribe -------------------------------------------------------
    log('info', f'Dub: transcribing original dialogue from {video_path.name} …')
    prog(0, 1, 'Transcribing…')
    word_timings = transcribe_video(video_path, settings, log, source_language)
    if not word_timings:
        log('warn', 'Dub: whisper returned no words — nothing to dub')
        return None

    segments = group_words_into_segments(word_timings)
    if not segments:
        log('warn', 'Dub: no dialogue segments detected')
        return None
    log('ok', f'Dub: {len(word_timings)} words → {len(segments)} dialogue line(s)')

    # 2) Translate all lines -- ONE request per model to avoid 429 limits --
    #    Each Gemini model has its OWN free-tier quota bucket, so if the first
    #    is exhausted we fall through to the next automatically.  Override the
    #    order via settings['dub_translate_models'] (comma-separated or list).
    log('info', f'Dub: translating {len(segments)} line(s) → {tgt} …')
    prog(0, 1, f'Translating {len(segments)} line(s)…')
    try:
        from gemini_api_tts_helper import translate_lines, translate_text
    except Exception as e:
        log('error', f'Dub: could not import translators: {e}')
        translate_lines = translate_text = None

    _cfg = settings.get('dub_translate_models')
    if isinstance(_cfg, str):
        _cfg = [m.strip() for m in _cfg.split(',') if m.strip()]
    model_chain = _cfg or [
        'gemini-2.5-flash',
        'gemini-3.1-flash-preview',
        'gemini-flash-latest',
        'gemini-2.0-flash',
        'gemini-flash-lite-latest',
        'gemini-2.0-flash-lite',
        'gemini-1.5-flash',
        'gemini-1.5-flash-8b',
        'gemini-2.5-pro',
        'gemini-2.5-flash-lite',
    ]

    src_lines = [s['text'] for s in segments]
    batch_ok = False
    if translate_lines is not None:
        last_err = ''
        for mdl in model_chain:
            ok, out = translate_lines(src_lines, tgt, settings, model=mdl,
                                       source_language=source_language)
            if ok:
                for seg, xl in zip(segments, out):
                    seg['xlated'] = xl
                batch_ok = True
                log('ok', f'Dub: translated all {len(segments)} line(s) in one '
                          f'request via {mdl}')
                break
            last_err = str(out)
            if '429' in last_err or 'RESOURCE_EXHAUSTED' in last_err:
                log('warn', f'Dub: {mdl} quota exhausted — trying next model')
            else:
                log('warn', f'Dub: {mdl} translate failed ({last_err[:120]})')
        if not batch_ok:
            log('warn', 'Dub: all batch models failed — falling back to per-line')

    if not batch_ok and translate_text is not None:
        for i, seg in enumerate(segments):
            ok, out = None, None
            for mdl in model_chain:
                ok, out = translate_text(seg['text'], tgt, settings, model=mdl,
                                           source_language=source_language)
                _e = str(out)
                # keep trying next model on quota (429) or missing-model (404),
                # otherwise accept the result (success or a real error)
                if ok or not ('429' in _e or 'RESOURCE_EXHAUSTED' in _e
                              or '404' in _e):
                    break
            seg['xlated'] = out if ok else seg['text']
            if not ok:
                log('warn', f'Dub: translate line {i} failed ({str(out)[:80]}) — kept original')
            prog(i + 1, len(segments), f'Translating {i + 1}/{len(segments)}')
        log('ok', 'Dub: translation complete')

    # 3) Read the original audio -----------------------------------------
    tmp_dir = out_mp3.parent / (out_mp3.stem + '_dubtmp')
    tmp_dir.mkdir(parents=True, exist_ok=True)
    orig_wav = tmp_dir / '_orig.wav'
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', str(video_path),
             '-vn', '-ar', str(SAMPLE_RATE), '-ac', '1',
             '-sample_fmt', 's16', str(orig_wav)],
            capture_output=True, check=True)
        orig_audio, _sr = sf.read(str(orig_wav))
        if orig_audio.ndim == 2:
            orig_audio = orig_audio.mean(axis=1)
        orig_audio = orig_audio.astype(np.float32)
    except Exception as e:
        log('warn', f'Dub: could not read original audio ({e}); using silence bed')
        _dur = max((s['end'] for s in segments), default=0) + 2.0
        orig_audio = np.zeros(int(_dur * SAMPLE_RATE), dtype=np.float32)

    total_samples = len(orig_audio)
    track = orig_audio * BG_VOL
    ducked = np.zeros(total_samples, dtype=bool)  # track which samples have been ducked

    # 4) TTS each translated line + overlay ------------------------------
    #
    # Placement strategy — keep the dub LOCKED to the video timeline so it
    # never drifts behind the scene:
    #   • Anchor each line to its OWN original timestamp (not chained after the
    #     previous line).  This stops lag from cascading down a long video —
    #     line N is always spoken at the moment scene N happens.
    #   • Fit-to-slot stretch — a translated line (Urdu/Hindi are longer than
    #     English) is sped up just enough to fit the gap until the NEXT line's
    #     original start, pitch-preserving via atempo, capped at ``dub_max_speed``
    #     (default 1.6×).  Lines that already fit stay at natural speed.
    #   • If even the cap can't make it fit, a small tail overlaps the next
    #     line — bounded to that one line only (no cascade), which sounds like
    #     brief natural cross-talk and is far better than falling behind.
    MAX_SPEED = float(settings.get('dub_max_speed', 1.6))
    BREATH = int(0.08 * SAMPLE_RATE)          # 80 ms gap target between lines
    placed = 0
    for i, seg in enumerate(segments):
        txt = (seg.get('xlated') or '').strip()
        if not txt:
            continue
        seg_mp3 = tmp_dir / f'dub_{i:04d}.mp3'
        seg_wav = tmp_dir / f'dub_{i:04d}.wav'
        try:
            ok, _ = TTSGenerator.generate_voiceover(txt, seg_mp3, settings)
            if not ok or not seg_mp3.is_file() or seg_mp3.stat().st_size == 0:
                log('warn', f'Dub: TTS line {i} produced no audio — skipped')
                continue
            subprocess.run(
                ['ffmpeg', '-y', '-i', str(seg_mp3),
                 '-ar', str(SAMPLE_RATE), '-ac', '1',
                 '-sample_fmt', 's16', str(seg_wav)],
                capture_output=True, check=True)
            data, sr = sf.read(str(seg_wav))
            if data.ndim == 2:
                data = data.mean(axis=1)
            data = data.astype(np.float32)
        except Exception as e:
            log('warn', f'Dub: TTS line {i} failed ({e})')
            continue

        # Anchor to the ORIGINAL timestamp — this is what keeps the dub in
        # sync with the scene and prevents cumulative drift.
        start = int(float(seg['start']) * SAMPLE_RATE)

        # Slot = time until the next spoken line's original start.  Stretch the
        # line to fit that slot, capped, so it stays on the video timeline.
        next_start = None
        for j in range(i + 1, len(segments)):
            nx = (segments[j].get('xlated') or '').strip()
            if nx:
                next_start = int(float(segments[j]['start']) * SAMPLE_RATE)
                break
        if next_start is not None:
            room = next_start - start - BREATH
            if room > SAMPLE_RATE // 4 and len(data) > room:   # need >0.25s room
                factor = min(MAX_SPEED, len(data) / float(room))
                if factor > 1.01:
                    data = _atempo_stretch(data, factor, log)
                    log('info', f'Dub: line {i} sped {factor:.2f}× to fit slot')

        end = start + len(data)
        if end > total_samples:
            pad = end - total_samples
            track = np.concatenate([track, np.zeros(pad, dtype=np.float32)])
            orig_audio = np.concatenate([orig_audio, np.zeros(pad, dtype=np.float32)])
            ducked = np.concatenate([ducked, np.zeros(pad, dtype=bool)])
            total_samples = len(track)
        # Duck the original only for samples not yet ducked, then add TTS.
        _seg_slice = slice(start, end)
        _unducked = ~ducked[_seg_slice]
        if _unducked.any():
            track[_seg_slice][_unducked] += orig_audio[_seg_slice][_unducked] * (DUCK_VOL - BG_VOL)
            ducked[_seg_slice] = True
        track[_seg_slice] += data
        placed += 1
        prog(i + 1, len(segments), f'Voicing {i + 1}/{len(segments)}')


    if placed == 0:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        log('warn', 'Dub: no lines were voiced — aborting')
        return None

    track = np.clip(track, -1.0, 1.0)

    # 5) Export the dub track → MP3 --------------------------------------
    dub_wav = tmp_dir / '_dub_track.wav'
    sf.write(str(dub_wav), track, SAMPLE_RATE)
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', str(dub_wav), '-b:a', '192k', str(out_mp3)],
            capture_output=True, check=True)
    except Exception as e:
        log('error', f'Dub: could not export dubbed MP3 ({e})')
        shutil.rmtree(str(tmp_dir), ignore_errors=True)
        return None
    shutil.rmtree(str(tmp_dir), ignore_errors=True)
    log('ok', f'Dub: ✅ voiced {placed}/{len(segments)} line(s) → {out_mp3.name}')
    return out_mp3


# ── Full dub-video (mux) ────────────────────────────────────────────────


def dub_video(video_path: Path, out_video: Path, target_language: str,
              settings: dict, log=None, progress=None,
              source_language: str | None = None,
              keep_audio_file: bool = False):
    """Dub the original dialogue and mux the result into a new video file.

    Copies the video stream untouched and replaces the audio with the dubbed
    track.  Returns the output-video Path, or ``None`` on failure.
    """
    log = log or _noop_log
    video_path = Path(video_path)
    out_video = Path(out_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)

    dub_mp3 = out_video.with_suffix('.dub.mp3')
    result = build_dubbed_audio(
        video_path, dub_mp3, target_language, settings, log, progress,
        source_language)
    if result is None:
        return None

    log('info', f'Dub: muxing dubbed audio into {out_video.name} …')
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', str(video_path), '-i', str(dub_mp3),
             '-map', '0:v:0', '-map', '1:a:0',
             '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
             '-shortest', str(out_video)],
            capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        _err = (e.stderr or b'').decode('utf-8', errors='replace')[-400:]
        log('error', f'Dub: mux failed — {_err}')
        return None
    finally:
        if not keep_audio_file:
            try:
                dub_mp3.unlink()
            except Exception:
                pass

    log('ok', f'Dub: ✅ dubbed video written → {out_video}')
    return out_video
