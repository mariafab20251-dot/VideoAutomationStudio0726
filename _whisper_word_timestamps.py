#!/usr/bin/env python
"""
Whisper Word Timestamp Extractor
=================================
Extracts word-level timestamps from an audio file.

Prefers **faster-whisper** (CTranslate2) when available — it is ~4x faster,
uses less memory, auto-downloads its own models, and runs well on CPU-only
machines (int8) as well as GPUs (float16).  Falls back to **openai-whisper**
if faster-whisper is not installed, so this script works on any PC.

When ``--diarize`` is passed, uses **faster-whisper** + **pyannote.audio** to
assign speaker labels to each word (word timestamps from faster-whisper,
speaker turns from pyannote, assigned by temporal overlap — no whisperx, so it
stays on the cuDNN 9 GPU stack).  The pyannote models are loaded from the
bundled ``models/pyannote/`` folder when available (offline/portable), falling
back to Hugging Face hub download.

Usage:
  python _whisper_word_timestamps.py audio.mp3 --model medium --language en
  python _whisper_word_timestamps.py audio.mp3 --diarize --hf-token hf_xxx

Outputs JSON in the standard word_timings format:
  [{"word": "A", "offset": 0.15, "duration": 0.12}, ...]
  (with optional "speaker" key when diarized)
"""

import json
import os
import sys
import argparse


# ── Disable TensorFlow (not used by this torch-only pipeline) ────────────
# thinc (a transitive dep via spaCy/pyannote) eagerly imports its Keras shim.
# The installed TF ships protobuf gencode 6.31.x, but the runtime is pinned to
# protobuf 5.29.x by google-generativeai (Gemini TTS), so that import crashes
# with a "Protobuf Gencode/Runtime versions" error mid-way — which thinc
# doesn't catch (it only handles ImportError).  We never need TF here, so
# disable it two ways:
#   1. USE_TF=0 → any transformers-style is_tf_available() check reports False
#      so it never even attempts the import (an env-var check, not an import).
#   2. a meta_path finder that fails the import cleanly → thinc catches the
#      ImportError and takes its no-TF path (thinc has no env-var switch).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


class _BlockTensorFlow:
    def find_module(self, name, path=None):
        if name == "tensorflow" or name.startswith("tensorflow."):
            return self
        return None

    def load_module(self, name):
        raise ImportError("tensorflow is intentionally disabled in the "
                          "whisper/dubbing subprocess (not required)")


if not any(isinstance(f, _BlockTensorFlow) for f in sys.meta_path):
    sys.meta_path.insert(0, _BlockTensorFlow())

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


# ─────────────────────────────────────────────────────────────────────
# Speaker diarization — pyannote.audio (Phase 1 — multi-speaker dubbing)
# ─────────────────────────────────────────────────────────────────────

def _rewrite_pyannote_config(bundle):
    """Rewrite the pipeline config's embedding/segmentation fields (which
    normally point at gated HF model IDs) to the bundled local file paths, so
    the pipeline loads fully offline.  Returns a temp .yaml path, or ``None``
    if the local weights aren't present.

    The keys live under ``pipeline.params`` in the diarization-3.1 config::

        pipeline:
          params:
            embedding: pyannote/wespeaker-voxceleb-resnet34-LM
            segmentation: pyannote/segmentation-3.0
    """
    import yaml
    pipeline_cfg = bundle / "speaker-diarization-3.1" / "config.yaml"
    with open(pipeline_cfg, "r") as f:
        cfg = yaml.safe_load(f)

    params = cfg.get("pipeline", {}).get("params", {})

    # segmentation: point at the local .bin weight file
    seg_bin = (bundle / "segmentation-3.0" / "pytorch_model.bin").resolve()
    if seg_bin.is_file():
        params["segmentation"] = str(seg_bin)
    else:
        _log(f"[DIARIZE] Missing local segmentation weights: {seg_bin}")
        return None

    # embedding: point at the local wespeaker folder (pyannote loads the .bin)
    emb_dir = (bundle / "wespeaker-voxceleb-resnet34-LM").resolve()
    emb_bin = emb_dir / "pytorch_model.bin"
    if emb_bin.is_file():
        params["embedding"] = str(emb_bin)
    else:
        _log(f"[DIARIZE] Missing local embedding weights: {emb_bin}")
        return None

    cfg["pipeline"]["params"] = params

    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w")
    yaml.dump(cfg, tmp)
    tmp.close()
    return tmp.name


def _build_diarization_pipeline(hf_token: str):
    """Build a pyannote ``SpeakerDiarization`` pipeline from the bundled local
    models (``models/pyannote/``) when available, falling back to HF hub.

    Returns the pipeline, or ``None`` if diarization is unavailable.
    """
    from pyannote.audio import Pipeline
    from pathlib import Path

    here = Path(__file__).resolve().parent
    bundle = here / "models" / "pyannote"
    pipeline_cfg = bundle / "speaker-diarization-3.1" / "config.yaml"

    if pipeline_cfg.is_file():
        # --- Local bundle path: rewrite config to point at local sub-folders ---
        _log(f"[DIARIZE] Loading pyannote from local bundle: {bundle}")
        cfg_path = _rewrite_pyannote_config(bundle)
        if cfg_path is None:
            _log("[DIARIZE] Local bundle config rewrite failed — "
                 "falling back to HF hub")
        else:
            try:
                pipeline = Pipeline.from_pretrained(cfg_path)
                _log("[DIARIZE] Local pyannote pipeline loaded OK")
                return pipeline
            except Exception as e:
                _log(f"[DIARIZE] Local pipeline load failed: {e}")
                try:
                    os.unlink(cfg_path)
                except Exception:
                    pass
                # Fall through to HF hub path

    # --- Fallback: HF hub (needs token + internet) ---------------------------
    if not hf_token:
        _log("[DIARIZE] No HF token provided and local bundle unavailable — "
             "diarization disabled")
        return None

    _log("[DIARIZE] Loading pyannote from Hugging Face hub (needs internet) ...")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        _log("[DIARIZE] HF hub pipeline loaded OK")
        return pipeline
    except Exception as e:
        _log(f"[DIARIZE] HF hub pipeline failed: {e}")
        return None


def _diarize_turns(audio_path: str, hf_token: str,
                   min_spk: int = None, max_spk: int = None):
    """Run pyannote speaker diarization and return a list of speaker turns.

    Uses the bundled local pyannote pipeline (offline) when available, else
    the HF hub.  pyannote is torch-based (cuDNN 9), so it runs on the same
    GPU stack as faster-whisper without the cuDNN-8 conflict that whisperx's
    ctranslate2 pin introduces.

    Returns:
        list of (start, end, speaker) tuples sorted by start, or ``None`` if
        diarization is unavailable.
    """
    import torch

    pipeline = _build_diarization_pipeline(hf_token)
    if pipeline is None:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        pipeline.to(device)
    except Exception as e:
        _log(f"[DIARIZE] Could not move pipeline to {device}: {e} — using CPU")

    _log(f"[DIARIZE] Running diarization on {device} "
         f"(min_spk={min_spk}, max_spk={max_spk}) ...")
    kwargs = {}
    if min_spk is not None:
        kwargs["min_speakers"] = min_spk
    if max_spk is not None:
        kwargs["max_speakers"] = max_spk

    annotation = pipeline(audio_path, **kwargs)

    turns = [
        (float(turn.start), float(turn.end), str(speaker))
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: t[0])
    n_spk = len({t[2] for t in turns})
    _log(f"[DIARIZE] Found {len(turns)} speaker turns across {n_spk} speaker(s)")
    return turns


def _assign_speakers(word_timings, turns):
    """Assign a speaker label to each word by max temporal overlap with the
    diarization turns.

    For each word's [offset, offset+duration] interval, pick the turn with the
    greatest overlap.  If a word overlaps no turn (silence gaps, boundary
    rounding), fall back to the nearest turn by midpoint distance.  Mutates and
    returns ``word_timings`` with a ``speaker`` key added to each entry.
    """
    if not turns:
        for w in word_timings:
            w["speaker"] = "SPEAKER_00"
        return word_timings

    for w in word_timings:
        w_start = w["offset"]
        w_end = w["offset"] + w["duration"]
        w_mid = (w_start + w_end) / 2.0

        best_spk, best_overlap = None, 0.0
        nearest_spk, nearest_dist = None, float("inf")

        for t_start, t_end, spk in turns:
            overlap = min(w_end, t_end) - max(w_start, t_start)
            if overlap > best_overlap:
                best_overlap, best_spk = overlap, spk

            # nearest-turn fallback: distance from word midpoint to the turn
            if t_start <= w_mid <= t_end:
                dist = 0.0
            else:
                dist = min(abs(w_mid - t_start), abs(w_mid - t_end))
            if dist < nearest_dist:
                nearest_dist, nearest_spk = dist, spk

        w["speaker"] = best_spk if best_spk is not None else nearest_spk

    return word_timings


def _extract_diarized(audio_path: str, model_size: str, language: str,
                      hf_token: str, min_spk: int = None, max_spk: int = None):
    """Transcribe with faster-whisper and label each word with a speaker.

    Pipeline (no whisperx — avoids its cuDNN-8 ctranslate2 pin):
      1. pyannote diarization → speaker turns
      2. faster-whisper ASR → word timestamps (with size-fallback chain)
      3. overlap-based assignment → per-word speaker labels

    Diarization runs *before* ASR on purpose: pyannote (torch) must load
    cuDNN 9 into the process first.  If ctranslate2 (faster-whisper) loads its
    own bundled cuDNN first, torch's later cuDNN symbol lookups fail with
    "Could not load symbol cudnnGetLibConfig".  Loading torch's cuDNN first
    lets both libraries coexist in the same process.

    Falls back to word timestamps without speaker labels if diarization is
    unavailable, and re-raises ASR failures so the caller's fallback chain
    (lighter model / openai-whisper) can take over.

    Returns:
        list of dicts: [{"word", "offset", "duration", "speaker"}, ...]
    """
    # --- 1. Diarize FIRST (loads torch cuDNN 9 before ctranslate2) ----------
    try:
        turns = _diarize_turns(audio_path, hf_token, min_spk, max_spk)
    except Exception as e:
        _log(f"[DIARIZE] Diarization failed: {e} — returning words w/o speaker")
        turns = None

    # --- 2. ASR: reuse the faster-whisper size-fallback chain ---------------
    last_err = None
    word_timings = None
    for m in _model_fallback_chain(model_size):
        try:
            if m != model_size:
                _log(f"[DIARIZE] Falling back to lighter ASR model '{m}' …")
            word_timings = _extract_faster_whisper(audio_path, m, language)
            break
        except Exception as e:
            last_err = e
            _log(f"[DIARIZE] ASR model '{m}' failed: {e}")
            continue
    if word_timings is None:
        # Let the caller's own openai-whisper last resort handle it
        raise last_err or RuntimeError("faster-whisper ASR failed")

    if turns is None:
        _log("[DIARIZE] No speaker turns — returning words without labels")
        return word_timings

    # --- 3. Assign speakers by overlap --------------------------------------
    word_timings = _assign_speakers(word_timings, turns)
    labelled = len({w.get("speaker") for w in word_timings})
    _log(f"[DIARIZE] Labelled {len(word_timings)} words across "
         f"{labelled} speaker(s)")
    return word_timings


def extract_word_timestamps(audio_path: str, model_size: str = "medium",
                            language: str = None,
                            diarize: bool = False,
                            hf_token: str = None,
                            min_speakers: int = None,
                            max_speakers: int = None):
    """Transcribe audio and return word-level timestamps.

    Model resolution, in order (first that works wins):
      1. requested faster-whisper model (e.g. 'medium'), local folder or hub
      2. lighter faster-whisper fallback ('base') if the first can't load
      3. openai-whisper if faster-whisper isn't installed at all

    When ``diarize=True``, uses faster-whisper + pyannote to assign speaker
    labels to each word.  Falls back gracefully to non-diarized output on any
    error.

    Args:
        audio_path: Path to audio file (mp3, wav, m4a, etc.)
        model_size: Preferred model size (tiny, base, small, medium, large-v3)
        language: Language code (e.g., 'en') or None for auto-detect
        diarize: Whether to run speaker diarization
        hf_token: Hugging Face read token (needed for pyannote hub fallback)
        min_speakers: Hint to diarizer — minimum expected speakers
        max_speakers: Hint to diarizer — maximum expected speakers

    Returns:
        list of dicts: [{"word": str, "offset": float, "duration": float}, ...]
        When diarized, each dict also includes ``"speaker"`` (e.g. "SPEAKER_00").
    """
    # --- Diarized path (faster-whisper + pyannote) --------------------------
    if diarize:
        try:
            _log("[DIARIZE] Diarization requested — faster-whisper + pyannote")
            return _extract_diarized(
                audio_path, model_size, language,
                hf_token=hf_token or "",
                min_spk=min_speakers, max_spk=max_speakers,
            )
        except Exception as e:
            _log(f"[DIARIZE] Diarized path failed: {e} — "
                 f"falling back to non-diarized transcription")

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
    # Diarization flags
    parser.add_argument("--diarize", action="store_true",
                        help="Enable speaker diarization via faster-whisper + pyannote")
    parser.add_argument("--hf-token", default=None,
                        help="Hugging Face read token (needed for pyannote hub "
                             "fallback; not required when local bundle exists)")
    parser.add_argument("--min-speakers", type=int, default=None,
                        help="Hint: minimum expected speakers")
    parser.add_argument("--max-speakers", type=int, default=None,
                        help="Hint: maximum expected speakers")
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"[ERROR] Audio file not found: {args.audio_path}", file=sys.stderr)
        sys.exit(1)

    word_timings = extract_word_timestamps(
        args.audio_path,
        model_size=args.model,
        language=args.language,
        diarize=args.diarize,
        hf_token=args.hf_token,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
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
