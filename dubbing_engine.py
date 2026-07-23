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


def _resolve_demucs_python(settings: dict, log) -> str | None:
    """Find a Python interpreter that has Demucs installed.

    Demucs needs the same heavy torch stack as diarization, so the whisper
    interpreter (Python311) is the natural home.  Probed in priority order:
    this process's Python, the ``whisper_python_path`` setting, then the known
    system Python311.  Returns None if none has demucs — the caller then falls
    back to the plain ducked-original bed.
    """
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
    # Repo-relative dubbing venv created by setup\install_dubbing.bat — the
    # portable, username-independent home for the whole dub stack.  Probed
    # BEFORE any hardcoded absolute path so a fresh PC just works.
    try:
        _here = Path(__file__).resolve().parent
        _candidates.append(str((_here / 'setup' / 'dub_venv' / 'Scripts' / 'python.exe').resolve()))
    except Exception:
        pass
    for _p in (r'C:\Users\shahi\AppData\Local\Programs\Python\Python311\python.exe',
               r'D:\GitHub\pythonprojects\VideoTextExtractor\venv\Scripts\python.exe'):
        _candidates.append(str(Path(_p).resolve()))

    _seen = set()
    for c in _candidates:
        c = str(Path(c).resolve())
        if c in _seen or not Path(c).is_file():
            continue
        _seen.add(c)
        try:
            r = subprocess.run(
                [c, '-c',
                 'import importlib.util as u\n'
                 'def has(m):\n'
                 ' try: return u.find_spec(m) is not None\n'
                 ' except Exception: return False\n'
                 'print("yes" if has("demucs") else "no")'],
                capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and (r.stdout or '').strip() == 'yes':
                log('info', f'Dub: demucs Python = {c}')
                return c
        except Exception:
            pass
    return None


def _demucs_torch_home(py_exe: str) -> str | None:
    """If ``py_exe`` is the bundled setup\\dub_venv interpreter, return the
    venv-local torch cache dir (where install_dubbing.bat stashed the htdemucs
    weight) so demucs loads it offline. Returns None for any other interpreter,
    leaving TORCH_HOME at its default.
    """
    from pathlib import Path
    try:
        p = Path(py_exe).resolve()
        # <venv>\Scripts\python.exe → <venv>
        venv = p.parent.parent
        if venv.name.lower() == 'dub_venv':
            cache = venv / 'torch_cache'
            if cache.is_dir():
                return str(cache)
    except Exception:
        pass
    return None


def _separate_instrumental(orig_wav, tmp_dir, settings, log, sample_rate):
    """Split ``orig_wav`` into vocals vs. everything-else via Demucs and return
    the NON-vocal stem (music + SFX + ambience) as a mono float32 numpy array
    at ``sample_rate``.

    This is what lets a dub keep the original score and sound effects while
    fully removing the actors' speech — far cleaner than ducking, which leaves
    the original voices audible underneath.

    Returns the instrumental array on success, or ``None`` if demucs isn't
    available / fails, so the caller falls back to the ducked-original bed.
    """
    import subprocess, tempfile
    import os as _os
    import numpy as np
    import soundfile as sf
    from pathlib import Path
    orig_wav = Path(orig_wav)
    tmp_dir = Path(tmp_dir)

    py = _resolve_demucs_python(settings, log)
    if not py:
        log('warn', 'Dub: demucs not installed — keeping original voices '
                    '(ducked) instead of a clean music/SFX bed. Install with '
                    '`pip install demucs` for voice removal.')
        return None

    tmp_root = None
    try:
        tmp_root = Path(tempfile.mkdtemp(prefix='demucs_', dir=str(tmp_dir)))
        model = str(settings.get('dub_demucs_model', 'htdemucs'))
        # When running under the bundled setup\dub_venv, the installer restored
        # the htdemucs weight into <venv>\torch_cache. Point TORCH_HOME there so
        # demucs finds it offline (portable, username-independent) instead of the
        # default ~/.cache/torch. Harmless for any other interpreter.
        env = dict(_os.environ)
        _venv_cache = _demucs_torch_home(py)
        if _venv_cache:
            env['TORCH_HOME'] = _venv_cache
        # --two-stems vocals → produces vocals.wav + no_vocals.wav
        r = subprocess.run(
            [py, '-m', 'demucs', '--two-stems', 'vocals', '-n', model,
             '--float32', '-o', str(tmp_root), str(orig_wav)],
            capture_output=True, text=True, timeout=1800, env=env)
        if r.returncode != 0:
            log('warn', f'Dub: demucs failed (rc={r.returncode}): '
                        f'{(r.stderr or "")[-200:]} — falling back to ducked bed')
            return None

        # Demucs writes <out>/<model>/<input-stem>/no_vocals.wav
        stem = orig_wav.stem
        no_vocals = tmp_root / model / stem / 'no_vocals.wav'
        if not no_vocals.is_file():
            # Be tolerant of layout differences across demucs versions.
            hits = list(tmp_root.rglob('no_vocals.wav'))
            no_vocals = hits[0] if hits else None
        if not no_vocals or not no_vocals.is_file():
            log('warn', 'Dub: demucs produced no music stem — falling back')
            return None

        bed, _sr = sf.read(str(no_vocals))
        if bed.ndim == 2:
            bed = bed.mean(axis=1)
        bed = bed.astype(np.float32)
        # Demucs outputs at 44100; resample to the engine rate if they differ.
        if int(_sr) != int(sample_rate):
            res_wav = tmp_root / '_bed_resampled.wav'
            sf.write(str(res_wav), bed, int(_sr))
            subprocess.run(
                ['ffmpeg', '-y', '-i', str(res_wav),
                 '-ar', str(sample_rate), '-ac', '1', '-sample_fmt', 's16',
                 str(tmp_root / '_bed_out.wav')],
                capture_output=True, check=True, timeout=180)
            bed, _ = sf.read(str(tmp_root / '_bed_out.wav'))
            if bed.ndim == 2:
                bed = bed.mean(axis=1)
            bed = bed.astype(np.float32)
        return bed
    except subprocess.TimeoutExpired:
        log('warn', 'Dub: demucs timed out — falling back to ducked bed')
        return None
    except Exception as e:
        log('warn', f'Dub: vocal separation failed ({e}) — falling back to ducked bed')
        return None
    finally:
        if tmp_root is not None:
            shutil.rmtree(str(tmp_root), ignore_errors=True)


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


# ── Child voices ────────────────────────────────────────────────────────
#
# Gemini/Edge TTS have NO real child voices (all voices sound adult), so a
# "child" is produced by taking a bright, youthful base voice and pitch-
# shifting it UP into child range (~+4 semitones).  The one genuine exception
# is Microsoft Edge's ``en-US-AnaNeural`` — an actual child (girl) voice — so
# the Edge-girl preset uses it directly with no shift.
#
# Each preset maps a dropdown pseudo-key → (engine, base voice, semitones up).
# ``engine`` is 'gemini' (uses gemini_tts_voice) or 'edge' (uses tts_voice).
CHILD_VOICE_PRESETS: dict = {
    # key                     engine    base voice            semitones
    'Child girl (Gemini)':  ('gemini', 'Leda',              4.0),
    'Child boy (Gemini)':   ('gemini', 'Puck',              4.0),
    'Child girl (Edge)':    ('edge',   'ana',               0.0),   # real child voice
    'Child boy (Edge)':     ('edge',   'steffan',           4.0),   # young male, shifted
}


def _is_child_voice(voice_key) -> bool:
    return str(voice_key) in CHILD_VOICE_PRESETS


def _resolve_voice_settings(settings: dict, voice):
    """Return ``(seg_settings, child_shift)`` for a chosen voice key.

    Shared by the main dub loop AND the voice preview so both render a voice
    identically.  ``voice`` may be a plain Gemini voice key (e.g. 'Puck') or a
    child pseudo-key from ``CHILD_VOICE_PRESETS``.  ``child_shift`` is the number
    of semitones to pitch the rendered audio up afterwards (0 for adult / real
    child voices).  ``seg_settings`` is a shallow copy — the caller's dict is
    never mutated.
    """
    if not voice:
        return settings, 0.0
    if _is_child_voice(voice):
        engine, base_voice, semis = CHILD_VOICE_PRESETS[voice]
        seg = dict(settings)
        if engine == 'edge':
            seg['tts_engine'] = 'cloud'          # edge-tts path
            seg['tts_voice'] = base_voice
        else:
            seg['tts_engine'] = 'google_cloud'   # gemini path
            seg['gemini_tts_voice'] = base_voice
        return seg, float(semis)
    seg = dict(settings)
    seg['gemini_tts_voice'] = voice
    return seg, 0.0


def preview_actor_clip(video_path, segments, speaker, out_wav,
                       log=_noop_log, max_dur=10.0):
    """Extract up to ``max_dur`` seconds of ONE actor's ORIGINAL voice from the
    video (their diarized segments, concatenated) → ``out_wav``.

    Lets the user hear who a SPEAKER_xx label actually is (and their real
    gender) before assigning a dubbing voice.  Returns ``out_wav`` on success or
    ``None`` on failure.
    """
    import subprocess
    from pathlib import Path
    video_path = Path(video_path)
    out_wav = Path(out_wav)

    # Collect this speaker's segments in time order until we reach max_dur.
    spans = []
    total = 0.0
    for s in segments:
        if str(s.get('speaker', 'SPEAKER_00')) != str(speaker):
            continue
        st = float(s.get('start', 0.0))
        en = float(s.get('end', st))
        if en <= st:
            continue
        take = min(en - st, max_dur - total)
        spans.append((st, st + take))
        total += take
        if total >= max_dur:
            break
    if not spans:
        log('warn', f'Dub: no audio spans for {speaker} to preview')
        return None

    # Build one ffmpeg command: trim each span, concat, write mono wav.
    try:
        inputs = []
        filters = []
        for idx, (st, en) in enumerate(spans):
            inputs += ['-ss', f'{st:.3f}', '-to', f'{en:.3f}', '-i', str(video_path)]
            filters.append(f'[{idx}:a]')
        filt = ''.join(filters) + f'concat=n={len(spans)}:v=0:a=1[a]'
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        cmd = ['ffmpeg', '-y', *inputs, '-filter_complex', filt,
               '-map', '[a]', '-ar', str(SAMPLE_RATE), '-ac', '1', str(out_wav)]
        subprocess.run(cmd, capture_output=True, check=True, timeout=120)
        return out_wav
    except Exception as e:
        log('warn', f'Dub: actor preview failed ({e})')
        return None


def preview_voice(text, voice, settings, out_mp3, log=_noop_log):
    """Render a short TTS sample of ``voice`` saying ``text`` → ``out_mp3``.

    Uses the SAME voice-resolution + pitch-shift path as the real dub loop, so
    what you hear in the preview is exactly what the dub will use.  Returns
    ``out_mp3`` on success or ``None`` on failure.
    """
    import subprocess
    from pathlib import Path
    out_mp3 = Path(out_mp3)
    try:
        from youtube_video_automation_enhanced import TTSGenerator
    except Exception as e:
        log('error', f'Dub: could not import TTSGenerator for preview: {e}')
        return None

    seg_settings, child_shift = _resolve_voice_settings(settings, voice)
    try:
        ok, _ = TTSGenerator.generate_voiceover(text, out_mp3, seg_settings)
        if not ok or not out_mp3.is_file() or out_mp3.stat().st_size == 0:
            log('warn', 'Dub: voice preview produced no audio')
            return None
        if child_shift > 0.01:
            # Render → shift pitch up → re-encode back to mp3.
            import soundfile as sf
            import numpy as np  # noqa: F401
            tmp_wav = out_mp3.with_suffix('.pre.wav')
            subprocess.run(
                ['ffmpeg', '-y', '-i', str(out_mp3),
                 '-ar', str(SAMPLE_RATE), '-ac', '1', str(tmp_wav)],
                capture_output=True, check=True)
            data, _sr = sf.read(str(tmp_wav))
            if getattr(data, 'ndim', 1) == 2:
                data = data.mean(axis=1)
            data = _pitch_shift_up(data, child_shift, log)
            sf.write(str(tmp_wav), data, SAMPLE_RATE)
            subprocess.run(
                ['ffmpeg', '-y', '-i', str(tmp_wav), '-b:a', '192k', str(out_mp3)],
                capture_output=True, check=True)
            try:
                tmp_wav.unlink(missing_ok=True)
            except Exception:
                pass
        return out_mp3
    except Exception as e:
        log('warn', f'Dub: voice preview failed ({e})')
        return None


def _pitch_shift_up(data, semitones, log=_noop_log):
    """Shift ``data`` (float32 mono @ SAMPLE_RATE) UP by ``semitones`` WITHOUT
    changing its duration, so an adult TTS voice sounds child-like.

    Uses ffmpeg: resample the sample rate up by the pitch factor (raises pitch
    AND speed), then ``atempo`` back down to restore the original duration.
    Returns the shifted float32 array, or the original on any failure.
    """
    import numpy as np
    import soundfile as sf
    import tempfile, subprocess
    from pathlib import Path

    if not semitones or abs(float(semitones)) < 0.01:
        return data
    factor = 2.0 ** (float(semitones) / 12.0)   # e.g. +4 st → ~1.26×
    new_sr = int(round(SAMPLE_RATE * factor))
    # atempo must stay within [0.5, 2.0]; 1/factor for +12 st is 0.5, fine here.
    tempo = 1.0 / factor
    tempo = max(0.5, min(2.0, tempo))
    filt = f'asetrate={new_sr},aresample={SAMPLE_RATE},atempo={tempo:.5f}'

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
        log('warn', f'Dub: pitch shift failed ({e}); using natural pitch')
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
    words, when the previous word ends a sentence (. ! ? …), when the current
    segment reaches ``max_words`` words, or when the speaker changes (only if
    the words carry a ``'speaker'`` key from diarization).

    Each segment gets a ``'speaker'`` label — the majority speaker of the words
    in it (first word wins ties).  When words have no ``'speaker'`` key,
    every segment is ``'SPEAKER_00'`` and grouping is identical to before.

    Returns ``[{'start': float, 'end': float, 'text': str, 'speaker': str}, …]``.
    """
    def _flush(words):
        """Build a segment dict from a list of (word, offset, speaker)."""
        counts = {}
        for _, _, spk in words:
            counts[spk] = counts.get(spk, 0) + 1
        # majority speaker; ties resolved by first appearance (dict is ordered)
        speaker = max(counts, key=counts.get) if counts else 'SPEAKER_00'
        return {
            'start': words[0][1],
            'end': prev_end,
            'text': ' '.join(t for t, _, _ in words).strip(),
            'speaker': speaker,
        }

    segs = []
    cur = []            # list of (word, offset, speaker)
    cur_start = None
    prev_end = None
    prev_speaker = None
    for w in word_timings:
        try:
            off = float(w.get('offset', 0))
            dur = float(w.get('duration', 0))
        except Exception:
            continue
        word = str(w.get('word', '')).strip()
        if not word:
            continue
        speaker = str(w.get('speaker', 'SPEAKER_00')) or 'SPEAKER_00'
        gap = (off - prev_end) if prev_end is not None else 0.0
        ends_sentence = bool(cur) and cur[-1][0][-1:] in '.!?…'
        speaker_changed = bool(cur) and speaker != prev_speaker
        if cur and (gap > max_gap or ends_sentence or len(cur) >= max_words
                    or speaker_changed):
            segs.append(_flush(cur))
            cur = []
            cur_start = None
        if cur_start is None:
            cur_start = off
        cur.append((word, off, speaker))
        prev_end = off + dur
        prev_speaker = speaker
    if cur:
        segs.append(_flush(cur))
    return [s for s in segs if s['text']]


def distinct_speakers(segments: list) -> list:
    """Return the sorted unique speaker labels across ``segments``.

    Segments produced by :func:`group_words_into_segments` always carry a
    ``'speaker'`` key; defaults to ``['SPEAKER_00']`` for non-diarized input.
    """
    return sorted({str(s.get('speaker', 'SPEAKER_00')) for s in segments})


def _frame_f0(frame, sr, fmin=70, fmax=350):
    """Estimate fundamental frequency of one short frame via autocorrelation.

    Returns 0.0 for silent/unvoiced frames.  Pure-numpy so it runs in the GUI
    venv (no librosa needed).
    """
    import numpy as np
    frame = frame - frame.mean()
    rms = float(np.sqrt(np.mean(frame ** 2)))
    if rms < 1e-3:                      # effectively silence
        return 0.0
    corr = np.correlate(frame, frame, 'full')[len(frame) - 1:]
    imin, imax = int(sr / fmax), int(sr / fmin)
    if imax >= len(corr) or imax <= imin:
        return 0.0
    seg = corr[imin:imax]
    if len(seg) == 0:
        return 0.0
    peak = int(np.argmax(seg)) + imin
    if corr[peak] <= 0.3 * corr[0]:     # weak periodicity → unvoiced
        return 0.0
    return sr / peak


def estimate_speaker_genders(video_path, segments, log=None) -> dict:
    """Best-effort per-speaker gender from median voice pitch (F0).

    Returns ``{speaker_label: 'Male'|'Female'}``.  Empty dict on any failure —
    callers must treat gender as optional.  Male voices sit roughly below
    ~165 Hz, female above; a small guard band avoids flip-flopping on the edge.
    """
    log = log or _noop_log
    try:
        import numpy as np
        import soundfile as sf
        import tempfile
        from pathlib import Path
    except Exception:
        return {}

    sr = 16000
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', str(video_path), '-vn', '-ac', '1',
             '-ar', str(sr), '-sample_fmt', 's16', tmp.name],
            capture_output=True, check=True, timeout=120)
        audio, _sr = sf.read(tmp.name)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
    except Exception as e:
        log('warn', f'Dub: gender detect skipped (audio read failed: {e})')
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except Exception:
            pass
        return {}

    win = int(0.040 * sr)     # 40 ms analysis frame
    hop = int(0.020 * sr)     # 20 ms hop
    per_spk_f0: dict = {}
    per_spk_budget: dict = {}   # cap analysed audio per speaker (seconds)
    MAX_SEC = 10.0
    for seg in segments:
        spk = str(seg.get('speaker', 'SPEAKER_00'))
        if per_spk_budget.get(spk, 0.0) >= MAX_SEC:
            continue
        s = int(float(seg.get('start', 0)) * sr)
        e = int(float(seg.get('end', 0)) * sr)
        s = max(0, s)
        e = min(len(audio), e)
        if e - s < win:
            continue
        clip = audio[s:e]
        for off in range(0, len(clip) - win, hop):
            f0 = _frame_f0(clip[off:off + win], sr)
            if f0 > 0:
                per_spk_f0.setdefault(spk, []).append(f0)
        per_spk_budget[spk] = per_spk_budget.get(spk, 0.0) + (e - s) / sr

    try:
        from pathlib import Path
        Path(tmp.name).unlink(missing_ok=True)
    except Exception:
        pass

    # Median F0 per speaker (needs a few voiced frames to be meaningful).
    med_f0: dict = {}
    for spk, f0s in per_spk_f0.items():
        if len(f0s) >= 5:
            med_f0[spk] = float(np.median(f0s))
    for spk, m in med_f0.items():
        log('info', f'Dub: {spk} median pitch ≈ {m:.0f} Hz')

    genders: dict = {}
    for spk, med in med_f0.items():
        # Clear cases: below ~155 Hz reads male, above ~180 Hz reads female.
        if med < 155:
            genders[spk] = 'Male'
        elif med > 180:
            genders[spk] = 'Female'
        else:
            # Ambiguous 155–180 Hz band — decide by frame distribution.
            f0s = per_spk_f0[spk]
            low = sum(1 for f in f0s if f < 165)
            genders[spk] = 'Male' if low >= len(f0s) / 2 else 'Female'

    # Two-speaker relative split: if exactly two speakers land on the SAME
    # gender but their pitches differ meaningfully (>25 Hz), the lower one is
    # very likely the opposite gender that the absolute threshold missed on a
    # short/noisy clip.  Nudge the lower speaker down a gender so a mixed-gender
    # pair doesn't collapse to two identical voices.
    if len(med_f0) == 2:
        (a, fa), (b, fb) = sorted(med_f0.items(), key=lambda kv: kv[1])
        if genders.get(a) == genders.get(b) and (fb - fa) > 25:
            if genders[a] == 'Female':      # both read female → lower is male
                genders[a] = 'Male'
            elif genders[b] == 'Male':      # both read male → higher is female
                genders[b] = 'Female'
    return genders


# ── Transcription ───────────────────────────────────────────────────────


def _resolve_whisper_python(settings: dict, log, require_pyannote: bool = False) -> str | None:
    """Find a Python interpreter that has whisper installed.

    When ``require_pyannote`` is True (multi-speaker dubbing), only accept an
    interpreter that ALSO has ``pyannote.audio`` — otherwise the whisper
    subprocess can't diarize locally and falls back to the gated HF hub,
    which then demands a token.  The interpreters are probed in priority order
    (this process's own Python first), so a fully-equipped one wins.
    """
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
    # Repo-relative dubbing venv created by setup\install_dubbing.bat — the
    # portable, username-independent home for whisper + pyannote + demucs.
    # Probed BEFORE any hardcoded absolute path so a fresh PC just works.
    try:
        _here = Path(__file__).resolve().parent
        _candidates.append(str((_here / 'setup' / 'dub_venv' / 'Scripts' / 'python.exe').resolve()))
    except Exception:
        pass
    # Known-good interpreters, in priority order.  The system Python311 carries
    # the full faster-whisper + pyannote stack (see MULTISPEAKER_DUBBING_PLAN.md);
    # the VTE venv is a whisper-only fallback for plain (non-diarized) dubbing.
    for _p in (
        r'C:\Users\shahi\AppData\Local\Programs\Python\Python311\python.exe',
        r'D:\GitHub\pythonprojects\VideoTextExtractor\venv\Scripts\python.exe',
    ):
        try:
            _candidates.append(str(Path(_p).resolve()))
        except Exception:
            pass

    _seen = set()
    _whisper_only = None   # remember a whisper-capable (no pyannote) fallback
    for c in _candidates:
        c = str(Path(c).resolve())
        if c in _seen or not Path(c).is_file():
            continue
        _seen.add(c)
        # Accept either faster-whisper (preferred) or openai-whisper, and
        # report whether pyannote.audio is present for diarization.
        try:
            # NOTE: find_spec("pyannote.audio") RAISES ModuleNotFoundError (it
            # doesn't return None) when the parent "pyannote" package is missing,
            # which would crash the probe and wrongly reject a good whisper-only
            # interpreter.  Guard every lookup with a helper that swallows that.
            r = subprocess.run(
                [c, '-c',
                 'import importlib.util as u\n'
                 'def has(m):\n'
                 ' try: return u.find_spec(m) is not None\n'
                 ' except Exception: return False\n'
                 'fw=has("faster_whisper"); ow=has("whisper"); pa=has("pyannote.audio")\n'
                 'print(("fw" if fw else ("ow" if ow else "none"))+("+pa" if pa else ""))'],
                capture_output=True, text=True, timeout=10)
            tag = (r.stdout or '').strip()
            has_whisper = r.returncode == 0 and tag.split('+')[0] in ('fw', 'ow')
            has_pyannote = tag.endswith('+pa')
            if not has_whisper:
                continue
            engine = 'faster-whisper' if tag.startswith('fw') else 'openai-whisper'
            if require_pyannote and not has_pyannote:
                # Keep as a last-resort fallback but keep looking for one with pyannote
                if _whisper_only is None:
                    _whisper_only = c
                continue
            suffix = ' + pyannote' if has_pyannote else ''
            log('info', f'Dub: whisper Python = {c} ({engine}{suffix})')
            return c
        except Exception:
            pass

    if require_pyannote and _whisper_only:
        log('warn', 'Dub: no Python with BOTH whisper and pyannote found — '
                    f'using {_whisper_only} (diarization will fall back to the '
                    'HF hub and need a token). Install pyannote.audio into that '
                    'interpreter to diarize offline.')
        return _whisper_only
    log('warn', 'Dub: no Python with whisper found')
    return None


def transcribe_video(video_path: Path, settings: dict, log=None,
                     source_language: str | None = None,
                     diarize: bool = False, hf_token: str | None = None,
                     min_spk: int | None = None,
                     max_spk: int | None = None) -> list:
    """Transcribe the video's own audio → word-level timings via whisper.

    Calls ``_whisper_word_timestamps.py`` directly with the correct
    ``--language`` flag so non-English source videos (Urdu, Russian, etc.)
    transcribe correctly.  ``source_language`` should be a 2-letter ISO 639-1
    code (e.g. 'ur', 'ru', 'hi') or a full language name; defaults to 'en'.

    When ``diarize`` is True, appends the diarization flags so each returned
    word carries a ``'speaker'`` label (used for multi-speaker dubbing).
    """
    import subprocess, json, tempfile, sys as _sys
    from pathlib import Path

    log = log or _noop_log

    # --- Resolve a python that has whisper (and pyannote if diarizing) -------
    venv_python = _resolve_whisper_python(settings, log, require_pyannote=diarize)
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

    cmd = [venv_python, str(whisper_script), audio_to_transcribe,
           '--model', str(settings.get('dub_whisper_model', 'medium')),
           '--language', whisper_lang]
    if diarize:
        cmd.append('--diarize')
        if hf_token:
            cmd += ['--hf-token', str(hf_token)]
        if min_spk:
            cmd += ['--min-speakers', str(int(min_spk))]
        if max_spk:
            cmd += ['--max-speakers', str(int(max_spk))]
        log('info', 'Dub: diarization enabled — detecting speakers …')

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=900 if diarize else 600)
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
    src = (source_language or '').strip().lower()
    if not tgt:
        log('warn', 'Dub: target language is empty — nothing to translate/dub')
        return None
    # Skip if source and target are the same English variant (nothing to translate)
    if tgt.lower() in ('english', 'en', 'en-us', 'en-gb') and src in ('english', 'en', 'en-us', 'en-gb', ''):
        log('warn', 'Dub: source and target are both English — nothing to translate/dub')
        return None

    try:
        from youtube_video_automation_enhanced import TTSGenerator
    except Exception as e:
        log('error', f'Dub: could not import TTSGenerator: {e}')
        return None

    DUCK_VOL = float(settings.get('dub_original_duck', 0.12))   # original under dub
    BG_VOL = float(settings.get('dub_original_bg', 0.55))       # original elsewhere

    # Multi-speaker dubbing: when on, transcription diarizes and each segment's
    # speaker gets its own mapped Gemini voice in the TTS loop.  Falls back to
    # single-voice cleanly when off or unmapped.
    multi = bool(settings.get('dub_multispeaker', False))
    voice_map = settings.get('dub_speaker_voices') or {}
    hf_token = str(settings.get('hf_token', '') or '').strip()
    min_spk = settings.get('dub_min_speakers') or None
    max_spk = settings.get('dub_max_speakers') or None

    # 1) Transcribe -------------------------------------------------------
    log('info', f'Dub: transcribing original dialogue from {video_path.name} …')
    prog(0, 1, 'Transcribing…')
    word_timings = transcribe_video(
        video_path, settings, log, source_language,
        diarize=multi, hf_token=hf_token, min_spk=min_spk, max_spk=max_spk)
    if not word_timings:
        log('warn', 'Dub: whisper returned no words — nothing to dub')
        return None

    segments = group_words_into_segments(word_timings)
    if not segments:
        log('warn', 'Dub: no dialogue segments detected')
        return None
    log('ok', f'Dub: {len(word_timings)} words → {len(segments)} dialogue line(s)')
    if multi:
        spks = distinct_speakers(segments)
        mapped = [s for s in spks if voice_map.get(s)]
        log('info', f'Dub: multi-speaker — {len(spks)} speaker(s) detected '
                    f'{spks}; {len(mapped)} mapped to a voice')

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

    # 3b) Optional vocal removal — keep music & SFX, drop the actors' voices.
    #     When ``dub_keep_music`` is on we run Demucs to split the original into
    #     a "vocals" stem (discarded) and a "no_vocals" instrumental+SFX stem
    #     (kept at full volume as the bed).  The dubbed voice then sits on a
    #     clean music/SFX track with NO original speech underneath — a far more
    #     professional result than ducking the whole mix.  Falls back silently
    #     to the ducking path if Demucs is unavailable or fails.
    keep_music = bool(settings.get('dub_keep_music', False))
    music_bed = None
    if keep_music and orig_audio.any():
        log('info', 'Dub: separating vocals from music/SFX (Demucs) …')
        prog(0, 1, 'Removing original voices…')
        music_bed = _separate_instrumental(
            orig_wav, tmp_dir, settings, log, SAMPLE_RATE)
        if music_bed is not None:
            # Match length to the original bed so downstream indexing is safe.
            if len(music_bed) < len(orig_audio):
                music_bed = np.concatenate(
                    [music_bed, np.zeros(len(orig_audio) - len(music_bed),
                                         dtype=np.float32)])
            else:
                music_bed = music_bed[:len(orig_audio)]
            log('ok', 'Dub: using clean music/SFX bed (original voices removed)')
        else:
            log('warn', 'Dub: vocal removal unavailable — falling back to '
                        'ducking the original mix')

    total_samples = len(orig_audio)
    if music_bed is not None:
        # The bed already has no speech, so keep it at full volume everywhere
        # and DON'T duck it under the dub — there are no actor voices to hide.
        track = music_bed.astype(np.float32)
        ducked = np.ones(total_samples, dtype=bool)   # never duck a clean bed
    else:
        track = orig_audio * BG_VOL
        ducked = np.zeros(total_samples, dtype=bool)  # track which samples ducked

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
        # Per-speaker voice: override this line's TTS voice with the speaker's
        # mapped one.  Shallow-copy settings so the shared dict is never mutated.
        #
        # Child voices are a special pseudo-key (see CHILD_VOICE_PRESETS): they
        # select a bright base voice on the right engine and remember how many
        # semitones to pitch-shift the rendered audio up afterwards.
        seg_settings = settings
        child_shift = 0.0
        if multi and voice_map:
            spk = seg.get('speaker', 'SPEAKER_00')
            voice = voice_map.get(spk)
            if voice:
                seg_settings, child_shift = _resolve_voice_settings(settings, voice)
        try:
            ok, _ = TTSGenerator.generate_voiceover(txt, seg_mp3, seg_settings)
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
            # Pitch the adult voice up into child range (skipped for the real
            # Edge child voice, whose preset shift is 0).
            if child_shift > 0.01:
                data = _pitch_shift_up(data, child_shift, log)
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

    # Guard: a 0-byte or unreadable input (e.g. a corrupt file left by a prior
    # failed run) yields a cryptic "stream received no packets" mux error.
    # Catch it up front with a clear message.
    try:
        if not video_path.is_file() or video_path.stat().st_size == 0:
            log('error', f'Dub: input video is missing or empty — {video_path.name}')
            return None
        _probe = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=codec_name', '-of', 'csv=p=0',
             str(video_path)],
            capture_output=True, text=True)
        if _probe.returncode != 0 or not _probe.stdout.strip():
            log('error', f'Dub: input video is corrupt/unreadable — {video_path.name} '
                        f'({(_probe.stderr or "").strip()[:120]})')
            return None
    except Exception as e:
        log('warn', f'Dub: could not validate input video ({e}) — continuing')

    dub_mp3 = out_video.with_suffix('.dub.mp3')
    result = build_dubbed_audio(
        video_path, dub_mp3, target_language, settings, log, progress,
        source_language)
    if result is None:
        return None

    log('info', f'Dub: muxing dubbed audio into {out_video.name} …')
    try:
        # NOTE: no ``-shortest`` — the dubbed track can legitimately run a beat
        # past the video (a long final Urdu line), and -shortest would chop that
        # tail, cutting off the last word.  Video stream is copied untouched;
        # the container simply lasts as long as the longer stream (the audio),
        # so the last word always plays out.
        subprocess.run(
            ['ffmpeg', '-y', '-i', str(video_path), '-i', str(dub_mp3),
             '-map', '0:v:0', '-map', '1:a:0',
             '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
             str(out_video)],
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
