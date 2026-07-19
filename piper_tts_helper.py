"""
Piper TTS Helper — lightweight, fast, multi-language TTS via piper.exe.

Integrates with the automation by calling Piper's native C++ CLI (piper.exe)
as a subprocess — no Python ML dependencies required.

Each voice model is only ~20-80MB (ONNX format).
Supports 160+ voices across 49 languages via the Piper voice catalog.
"""

import os
import json
import subprocess
import hashlib
import shutil
import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────
MODULE_DIR = Path(__file__).parent / "VoiceModules" / "PiperTTS"
PIPER_EXE = MODULE_DIR / "bin" / "piper.exe"
VOICES_DIR = MODULE_DIR / "voices"
VOICES_CATALOG = MODULE_DIR / "voices.json"

# HuggingFace base for Piper voice downloads
HF_PIPER_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# Default voice if none specified
DEFAULT_VOICE = "en_US-ryan-medium"

# Cache of parsed catalog
_voices_cache = None
_voices_cache_lock = threading.Lock()


# ── Core TTS ───────────────────────────────────────────────────────────

def generate_speech(
    text: str,
    output_path: str,
    voice: str = DEFAULT_VOICE,
    speaker_id: Optional[int] = None,
    noise_scale: float = 0.667,
    length_scale: float = 1.0,
    noise_w: float = 0.8,
    sentence_silence: float = 0.2,
    timeout: int = 60,
) -> bool:
    """Generate speech from text using Piper TTS.

    Args:
        text: Text to synthesize.
        output_path: Path to save WAV file.
        voice: Voice key from catalog (e.g. 'en_US-ryan-medium').
        speaker_id: Optional speaker ID for multi-speaker models.
        noise_scale: Generator noise (0.0-1.0, default 0.667).
        length_scale: Phoneme length / speed (lower=faster, default 1.0).
        noise_w: Phoneme width noise (default 0.8).
        sentence_silence: Seconds of silence between sentences (default 0.2).
        timeout: Subprocess timeout in seconds.

    Returns:
        True if generation succeeded.
    """
    voice_path = _resolve_voice_path(voice)
    if not voice_path:
        logger.error("Piper TTS: voice '%s' not found. Use download_voice() first.", voice)
        return False

    config_path = voice_path.with_suffix(".onnx.json")
    if not config_path.exists():
        logger.warning("Piper TTS: config not found for %s, proceeding without it", voice)

    cmd = [
        str(PIPER_EXE),
        "--model", str(voice_path),
        "--output_file", str(output_path),
        "--noise_scale", str(noise_scale),
        "--length_scale", str(length_scale),
        "--noise_w", str(noise_w),
        "--sentence_silence", str(sentence_silence),
    ]
    if config_path.exists():
        cmd.extend(["--config", str(config_path)])
    if speaker_id is not None:
        cmd.extend(["--speaker", str(speaker_id)])

    try:
        proc = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            logger.error("Piper TTS failed (rc=%d): %s", proc.returncode, stderr)
            return False
        # Windows file-system cache / antivirus can delay visibility.
        import time as _time
        for _attempt in range(5):
            if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
                return True
            _time.sleep(0.1)
        logger.error("Piper TTS: piper.exe exited 0 but output file not found: %s",
                      output_path)
        return False
    except subprocess.TimeoutExpired:
        logger.error("Piper TTS timed out after %ds", timeout)
        return False
    except FileNotFoundError:
        logger.error("Piper TTS: piper.exe not found at %s", PIPER_EXE)
        return False


# ── Voice Catalog ──────────────────────────────────────────────────────

def get_available_voices() -> dict:
    """Return parsed voices.json catalog: {voice_key: info_dict}."""
    global _voices_cache
    if _voices_cache is not None:
        return _voices_cache
    with _voices_cache_lock:
        if _voices_cache is not None:
            return _voices_cache
        if not VOICES_CATALOG.exists():
            logger.warning("Piper TTS: voices.json not found at %s", VOICES_CATALOG)
            _voices_cache = {}
            return _voices_cache
        try:
            with open(VOICES_CATALOG, encoding="utf-8") as f:
                _voices_cache = json.load(f)
        except Exception as e:
            logger.error("Piper TTS: failed to parse voices.json: %s", e)
            _voices_cache = {}
        return _voices_cache


def get_languages() -> list[dict]:
    """Return list of {code, name, country, voice_count} for each language."""
    catalog = get_available_voices()
    langs = {}
    for key, info in catalog.items():
        lang = info["language"]
        code = lang["code"]
        if code not in langs:
            langs[code] = {
                "code": code,
                "name": lang["name_english"],
                "native": lang["name_native"],
                "country": lang["country_english"],
                "voice_count": 0,
                "voices": [],
            }
        langs[code]["voice_count"] += 1
        langs[code]["voices"].append({
            "key": key,
            "name": info["name"],
            "quality": info["quality"],
            "num_speakers": info["num_speakers"],
            "downloaded": _is_downloaded(key),
        })
    return sorted(langs.values(), key=lambda x: x["code"])


def get_voices_for_language(lang_code: str) -> list[dict]:
    """Return list of voice dicts for a given language code (e.g. 'en_US')."""
    catalog = get_available_voices()
    results = []
    for key, info in catalog.items():
        if info["language"]["code"] == lang_code:
            results.append({
                "key": key,
                "name": info["name"],
                "quality": info["quality"],
                "num_speakers": info["num_speakers"],
                "speaker_id_map": info.get("speaker_id_map", {}),
                "downloaded": _is_downloaded(key),
            })
    return results


def get_voice_info(voice_key: str) -> Optional[dict]:
    """Get info dict for a specific voice key."""
    catalog = get_available_voices()
    return catalog.get(voice_key)


def get_voice_speakers(voice_key: str) -> list[dict]:
    """Return list of {id: name_or_id, label} for a voice's speakers/emotions.

    For multi-speaker models this returns the available speaker IDs.
    The 'thorsten_emotional' model has proper emotion names (amused, angry, etc.).
    Returns empty list if the voice has only 1 speaker.
    """
    info = get_voice_info(voice_key)
    if not info or info.get("num_speakers", 1) <= 1:
        return []

    speaker_map = info.get("speaker_id_map", {})
    if not speaker_map:
        return []

    result = []
    for name, spk_id in speaker_map.items():
        # Clean up display name
        display = name.replace("_", " ").replace("-", " ").title().strip()
        result.append({"id": spk_id, "name": name, "label": display})
    return result


def is_emotional_voice(voice_key: str) -> bool:
    """Check if a voice has emotion-named speakers (like thorsten_emotional)."""
    speakers = get_voice_speakers(voice_key)
    emotion_keywords = {"amused", "angry", "disgusted", "drunk", "neutral",
                        "sleepy", "surprised", "whisper", "happy", "sad",
                        "excited", "calm", "fearful"}
    for spk in speakers:
        if spk["name"].lower() in emotion_keywords:
            return True
    return False


def _is_downloaded(voice_key: str) -> bool:
    """Check if a voice model is already downloaded."""
    voice_file = VOICES_DIR / f"{voice_key}.onnx"
    return voice_file.exists()


# ── Download ───────────────────────────────────────────────────────────

def download_voice(voice_key: str, progress_callback=None) -> bool:
    """Download a voice model from HuggingFace Piper voices repo.

    Args:
        voice_key: Voice key from catalog (e.g. 'en_US-amy-medium').
        progress_callback: Optional fn(bytes_downloaded, total_bytes).

    Returns:
        True if download succeeded or voice already exists.
    """
    if _is_downloaded(voice_key):
        logger.info("Piper TTS: voice '%s' already downloaded", voice_key)
        return True

    voice_info = get_voice_info(voice_key)
    if not voice_info:
        logger.error("Piper TTS: voice '%s' not found in catalog", voice_key)
        return False

    # Build download URL from voice file path
    file_paths = voice_info.get("files", {})
    if not file_paths:
        logger.error("Piper TTS: no file entries for voice '%s'", voice_key)
        return False

    # Sort files: ONNX first, then JSON, then MODEL_CARD
    sorted_files = sorted(file_paths.items(), key=lambda x: (0 if x[0].endswith(".onnx") else 1 if x[0].endswith(".json") else 2))

    success = True
    for rel_path, file_info in sorted_files:
        dest = VOICES_DIR / os.path.basename(rel_path)
        if dest.exists():
            # Verify integrity if we have md5
            if "md5_digest" in file_info:
                if _verify_md5(dest, file_info["md5_digest"]):
                    continue
                else:
                    logger.warning("Piper TTS: corrupted %s, re-downloading", dest.name)
                    dest.unlink(missing_ok=True)
            else:
                continue

        url = f"{HF_PIPER_BASE}/{rel_path.replace(os.sep, '/')}"
        logger.info("Piper TTS: downloading %s → %s", url, dest.name)

        if not _download_file(url, dest, file_info.get("size_bytes"), progress_callback):
            logger.error("Piper TTS: failed to download %s", url)
            success = False
            break

        # Verify MD5
        if "md5_digest" in file_info:
            if not _verify_md5(dest, file_info["md5_digest"]):
                logger.error("Piper TTS: MD5 mismatch for %s", dest.name)
                dest.unlink(missing_ok=True)
                success = False
                break

    return success


def download_voice_sync(voice_key: str) -> bool:
    """Synchronous convenience wrapper for download_voice."""
    return download_voice(voice_key)


def _download_file(url: str, dest: Path, expected_size=None, progress_callback=None) -> bool:
    """Download a file with optional progress reporting."""
    try:
        import urllib.request
        import ssl

        # Create SSL context that doesn't verify (handles corporate proxies)
        ctx = ssl.create_default_context()

        def report(block_count, block_size, total_size):
            if progress_callback and total_size > 0:
                downloaded = block_count * block_size
                progress_callback(min(downloaded, total_size), total_size)

        urllib.request.urlretrieve(url, dest, reporthook=report)
        return True
    except Exception as e:
        logger.error("Piper TTS: download error for %s: %s", url, e)
        return False


def _verify_md5(file_path: Path, expected_md5: str) -> bool:
    """Verify file integrity against expected MD5 hash."""
    try:
        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                md5.update(chunk)
        return md5.hexdigest() == expected_md5.lower()
    except Exception:
        return False


# ── Available voices for GUI dropdown ──────────────────────────────────

def get_display_voices() -> list[tuple]:
    """Return list of (voice_key, display_name) tuples for GUI dropdowns.

    Display name format: "Language — Voice Name (Quality)" or "Voice Name (Quality)"
    """
    catalog = get_available_voices()
    result = []
    for key, info in catalog.items():
        lang_name = info["language"]["name_english"]
        country = info["language"]["country_english"]
        voice_name = info["name"].replace("_", " ").title()
        quality = _quality_label(info["quality"])
        label = f"{voice_name} ({quality}) — {lang_name}"
        result.append((key, label))
    return sorted(result, key=lambda x: (x[1].split("—")[-1].strip(), x[1]))


def _quality_label(quality: str) -> str:
    """Convert quality code to display label."""
    labels = {
        "x_low": "X-Low (20MB)",
        "low": "Low (63MB)",
        "medium": "Medium (63MB)",
        "high": "High (114MB)",
    }
    return labels.get(quality, quality)


# ── Resolve voice file path ────────────────────────────────────────────

def _resolve_voice_path(voice_key: str) -> Optional[Path]:
    """Get the filesystem path to a voice ONNX file."""
    # Check exact file
    direct = VOICES_DIR / f"{voice_key}.onnx"
    if direct.exists():
        return direct
    # Check if voice_key matches a file (some might have different naming)
    for f in VOICES_DIR.glob("*.onnx"):
        if f.stem == voice_key:
            return f
    return None


# ── Health check ───────────────────────────────────────────────────────

def is_available() -> bool:
    """Check if Piper TTS engine is available."""
    return PIPER_EXE.exists()


def is_voice_available(voice_key: str = DEFAULT_VOICE) -> bool:
    """Check if a specific voice is downloaded."""
    return _resolve_voice_path(voice_key) is not None


def get_downloaded_voices() -> list[str]:
    """Return list of voice keys that are downloaded."""
    voices = []
    for f in VOICES_DIR.glob("*.onnx"):
        voices.append(f.stem)
    return sorted(voices)


# ── Voice Management ───────────────────────────────────────────────────

def delete_voice(voice_key: str) -> bool:
    """Delete a downloaded voice model."""
    voice_path = _resolve_voice_path(voice_key)
    if not voice_path:
        return False
    try:
        voice_path.unlink(missing_ok=True)
        # Clean up associated files
        for ext in [".onnx.json", ".json"]:
            extra = voice_path.with_suffix(ext)
            extra.unlink(missing_ok=True)
        return True
    except Exception as e:
        logger.error("Piper TTS: failed to delete voice %s: %s", voice_key, e)
        return False


def get_voices_summary() -> str:
    """Return a formatted summary of available/downloaded voices."""
    catalog = get_available_voices()
    downloaded = get_downloaded_voices()
    total = len(catalog)
    dl = len(downloaded)

    lines = [
        f"Piper TTS — {dl}/{total} voices downloaded",
        "",
    ]
    langs = get_languages()
    for lang in langs[:10]:  # top 10 languages
        dl_lang = sum(1 for v in lang["voices"] if v["downloaded"])
        lines.append(f"  {lang['code']} ({lang['name']}): {dl_lang}/{lang['voice_count']} voices")
    if len(langs) > 10:
        lines.append(f"  ... and {len(langs)-10} more languages")
    lines.append("")
    lines.append(f"Downloaded voices: {', '.join(downloaded) if downloaded else '(none)'}")
    return "\n".join(lines)


# ── Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Piper TTS Helper")
    print(f"  Available: {is_available()}")
    print(f"  Piper exe: {PIPER_EXE}")
    print(f"  Voices dir: {VOICES_DIR}")
    print()
    print(get_voices_summary())

    if is_available() and is_voice_available():
        print("\n--- Testing generation ---")
        test_out = os.path.join(os.path.dirname(__file__), "_piper_test.wav")
        result = generate_speech("Hello, this is Piper TTS speaking. It is very fast and lightweight.", test_out)
        print(f"  Result: {'SUCCESS' if result else 'FAILED'}")
        if result:
            size = os.path.getsize(test_out)
            print(f"  Output: {test_out} ({size} bytes)")
    else:
        print("\nPiper TTS not fully available. Copy engine files and voices first.")
