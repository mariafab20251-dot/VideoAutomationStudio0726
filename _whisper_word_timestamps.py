#!/usr/bin/env python
"""
Whisper Word Timestamp Extractor
=================================
Extracts word-level timestamps from an audio file.

Prefers **faster-whisper** (CTranslate2) when available — it is ~4x faster,
uses less memory, auto-downloads its own models, and runs well on CPU-only
machines (int8) as well as GPUs (float16).  Falls back to **openai-whisper**
if faster-whisper is not installed, so this script works on any PC.

Usage:
  python _whisper_word_timestamps.py audio.mp3 --model medium --language en

Outputs JSON in the standard word_timings format:
  [{"word": "A", "offset": 0.15, "duration": 0.12}, ...]
"""

import json
import os
import sys
import argparse

# All info/debug logging goes to stderr so stdout stays pure JSON
_log = lambda *a, **kw: print(*a, file=sys.stderr, **kw)


def _resolve_local_model(model_size: str):
    """Return a path to a bundled faster-whisper model folder if one ships
    next to this script, else ``None`` (→ let faster-whisper auto-download).

    Looked-up locations (first hit wins), so the model travels with the
    portable folder and never needs internet on a fresh PC::

        <script_dir>/models/whisper/faster-whisper-<size>/
        <script_dir>/models/whisper/<size>/
    """
    from pathlib import Path
    here = Path(__file__).resolve().parent
    candidates = [
        here / "models" / "whisper" / f"faster-whisper-{model_size}",
        here / "models" / "whisper" / model_size,
    ]
    for c in candidates:
        # A valid CTranslate2 model folder contains model.bin + config.json
        if (c / "model.bin").is_file() and (c / "config.json").is_file():
            return str(c)
    return None


def _extract_faster_whisper(audio_path: str, model_size: str, language: str):
    """Transcribe with faster-whisper (CTranslate2). Returns word_timings or
    raises ImportError if the package isn't installed."""
    from faster_whisper import WhisperModel  # ImportError → caller falls back

    # Auto-pick the best device/precision available.  GPU → float16 (fast,
    # accurate); CPU → int8 (low RAM, still usable on machines with no GPU).
    device, compute_type = "cpu", "int8"
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            device, compute_type = "cuda", "float16"
    except Exception:
        pass

    # Prefer a bundled local model folder (offline / portable); otherwise
    # pass the size name so faster-whisper downloads + caches it once.
    local = _resolve_local_model(model_size)
    model_ref = local or model_size
    _src = f"local:{local}" if local else f"hub:{model_size}"
    _log(f"[FASTER-WHISPER] Loading model '{model_size}' from {_src} "
         f"(device={device}, compute_type={compute_type}) ...")
    model = WhisperModel(model_ref, device=device, compute_type=compute_type)

    _log(f"[FASTER-WHISPER] Transcribing: {audio_path}")
    segments, _info = model.transcribe(
        audio_path,
        word_timestamps=True,
        language=language,
    )

    word_timings = []
    for segment in segments:
        for word_info in (segment.words or []):
            word = (word_info.word or "").strip()
            if not word:
                continue
            offset = float(word_info.start or 0)
            end = float(word_info.end or 0)
            duration = max(0.01, end - offset)
            word_timings.append({
                "word": word,
                "offset": offset,
                "duration": duration,
            })

    _log(f"[FASTER-WHISPER] Extracted {len(word_timings)} word timestamps "
         f"(audio: {audio_path})")
    return word_timings


def _extract_openai_whisper(audio_path: str, model_size: str, language: str):
    """Transcribe with the original openai-whisper (PyTorch). Fallback path."""
    _log(f"[WHISPER] Loading model '{model_size}' ...")
    import whisper

    model = whisper.load_model(model_size)
    _log(f"[WHISPER] Transcribing: {audio_path}")

    result = model.transcribe(
        audio_path,
        word_timestamps=True,
        language=language,
        verbose=False,
    )

    word_timings = []
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            word = word_info.get("word", "").strip()
            if not word:
                continue
            offset = float(word_info.get("start", 0))
            end = float(word_info.get("end", 0))
            duration = max(0.01, end - offset)
            word_timings.append({
                "word": word,
                "offset": offset,
                "duration": duration,
            })

    _log(f"[WHISPER] Extracted {len(word_timings)} word timestamps "
         f"(audio: {audio_path})")
    return word_timings


def _model_fallback_chain(model_size: str) -> list:
    """Ordered list of models to try: the requested one first, then lighter
    fallbacks so a missing/broken model never leaves dubbing stuck.

    e.g. 'medium' → ['medium', 'base'];  'base' → ['base'].
    """
    order = ["large-v3", "medium", "small", "base", "tiny"]
    chain = [model_size]
    # Always keep 'base' as a lightweight last resort (unless already chosen).
    if model_size in order:
        for m in order[order.index(model_size) + 1:]:
            if m == "base":
                chain.append(m)
                break
    if "base" not in chain:
        chain.append("base")
    # De-dupe while preserving order.
    seen, out = set(), []
    for m in chain:
        if m not in seen:
            seen.add(m); out.append(m)
    return out


def extract_word_timestamps(audio_path: str, model_size: str = "medium",
                            language: str = None):
    """Transcribe audio and return word-level timestamps.

    Model resolution, in order (first that works wins):
      1. requested faster-whisper model (e.g. 'medium'), local folder or hub
      2. lighter faster-whisper fallback ('base') if the first can't load
      3. openai-whisper if faster-whisper isn't installed at all

    This means: prefer 'medium' for accuracy, but if it's missing, fails to
    download, or errors out, automatically drop to 'base' so dubbing still
    runs instead of returning nothing.

    Args:
        audio_path: Path to audio file (mp3, wav, m4a, etc.)
        model_size: Preferred model size (tiny, base, small, medium, large-v3)
        language: Language code (e.g., 'en') or None for auto-detect

    Returns:
        list of dicts: [{"word": str, "offset": float, "duration": float}, ...]
    """
    # --- faster-whisper path (preferred), with size fallback chain ----------
    try:
        import faster_whisper  # noqa: F401  (presence check only)
    except ImportError:
        _log("[WHISPER] faster-whisper not installed — falling back to "
             "openai-whisper")
        return _extract_openai_whisper(audio_path, model_size, language)

    last_err = None
    for m in _model_fallback_chain(model_size):
        try:
            if m != model_size:
                _log(f"[WHISPER] Falling back to lighter model '{m}' …")
            return _extract_faster_whisper(audio_path, m, language)
        except Exception as e:  # model missing / download failed / load error
            last_err = e
            _log(f"[WHISPER] Model '{m}' failed: {e}")
            continue

    # --- last resort: openai-whisper 'base' ---------------------------------
    _log("[WHISPER] All faster-whisper models failed — trying openai-whisper "
         "'base' as a last resort")
    try:
        return _extract_openai_whisper(audio_path, "base", language)
    except Exception as e:
        _log(f"[WHISPER] openai-whisper fallback also failed: {e}")
        raise last_err or e


def main():
    parser = argparse.ArgumentParser(
        description="Extract word timestamps from audio using whisper")
    parser.add_argument("audio_path", help="Path to audio file")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--model", "-m", default="medium",
                        help="Whisper model size (tiny, base, small, medium, "
                             "large-v3). Default: medium")
    parser.add_argument("--language", "-l", default=None,
                        help="Language code (e.g., 'en'). Auto-detected if omitted.")
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"[ERROR] Audio file not found: {args.audio_path}", file=sys.stderr)
        sys.exit(1)

    word_timings = extract_word_timestamps(
        args.audio_path,
        model_size=args.model,
        language=args.language,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(word_timings, f, indent=2)
        _log(f"[OK] Saved {len(word_timings)} word timestamps to: {args.output}")
    else:
        print(json.dumps(word_timings, indent=2))

    return word_timings


if __name__ == "__main__":
    main()
