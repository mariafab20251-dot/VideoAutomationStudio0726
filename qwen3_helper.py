"""
Qwen3-TTS integration helper for AI voiceover generation.

Wraps Qwen3-TTS (Alibaba's CustomVoice model) so the main pipeline can
generate emotional, natural-sounding voiceovers with predefined speakers
or voice cloning without importing fragile dependencies directly.

Model files (from HuggingFace Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
should be placed in:
    VoiceModules/Qwen3-TTS/models/CustomVoice/

Python dependency (install into the Python 3.11 env):
    pip install qwen-tts
"""

import os
import sys
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Last-error tracking for GUI diagnostics ───────────────────────
_last_error: str = ""

def _set_last_error(msg: str):
    global _last_error
    _last_error = msg[:500]

def get_last_error() -> str:
    """Return the last recorded error from any generate_* call."""
    return _last_error

# ── Paths ──────────────────────────────────────────────────────────
QWEN3_DIR = Path(__file__).parent / 'VoiceModules' / 'Qwen3-TTS'
MODEL_DIR_0_6B = QWEN3_DIR / 'models' / 'CustomVoice'   # 0.6B CustomVoice
MODEL_DIR_1_7B = QWEN3_DIR / 'models' / '1_7B' / 'CustomVoice'  # 1.7B CustomVoice
MODEL_DIR = MODEL_DIR_0_6B  # default for backward compat
BASE_DIR = QWEN3_DIR / 'models' / 'Base'
BASE_DIR_1_7B = QWEN3_DIR / 'models' / '1_7B' / 'Base'
FASTER_DIR = QWEN3_DIR / 'faster_qwen3_tts'  # faster-qwen3-tts CUDA backend

# ── Model size switching (0.6B vs 1.7B) ─────────────────────────
_ACTIVE_MODEL_DIR: Optional[Path] = None  # None = use MODEL_DIR (0.6B default)
_ACTIVE_BASE_DIR: Optional[Path] = None   # None = use BASE_DIR (0.6B Base default)

def set_model_size(size: str):
    """Switch between '0.6B' and '1.7B' model directories.

    The 1.7B model supports emotion/style instructions;
    0.6B does not (emotion instruct is silently ignored).

    Affects both CustomVoice (speaker-based TTS) and
    Base (voice cloning) model paths.
    Call before generate_speech / generate_speech_batch.
    """
    global _ACTIVE_MODEL_DIR, _ACTIVE_BASE_DIR
    if size == '1.7B':
        _ACTIVE_MODEL_DIR = MODEL_DIR_1_7B
        _ACTIVE_BASE_DIR = BASE_DIR_1_7B
        logger.info('Qwen3: switched to 1.7B model')
    else:
        _ACTIVE_MODEL_DIR = None  # back to 0.6B
        _ACTIVE_BASE_DIR = None
        logger.info('Qwen3: switched to 0.6B model')

def get_active_model_dir() -> Path:
    """Return the CustomVoice model directory for the active size."""
    return _ACTIVE_MODEL_DIR if _ACTIVE_MODEL_DIR is not None else MODEL_DIR

def get_active_base_dir() -> Path:
    """Return the Base (voice cloning) model directory for the active size."""
    return _ACTIVE_BASE_DIR if _ACTIVE_BASE_DIR is not None else BASE_DIR

# ── Emotion → instruct mapping ───────────────────────────────────
# Maps emotion names (used in script tags like [excited]) to instruct
# strings that the 1.7B CustomVoice model understands.
EMOTION_INSTRUCT_MAP: dict = {
    'neutral': '',
    'happy': 'Happy, cheerful, bright and uplifting tone with a smile in the voice',
    'sad': 'Sad, melancholic, sorrowful, soft and gentle with a touch of longing',
    'excited': 'Excited, energetic, enthusiastic, dynamic and passionate delivery',
    'angry': 'Angry, aggressive, intense, forceful and sharp with strong emphasis',
    'calm': 'Calm, soothing, gentle, relaxed and peaceful like a quiet evening',
    'whisper': 'Whispering, soft, intimate, quiet and close like sharing a secret',
    'deep': 'Deep, authoritative, powerful, commanding and resonant like a narrator',
    'dramatic': 'Dramatic, intense, impactful, theatrical and grand with rising tension',
    'storytelling': 'Narrative, storytelling, engaging, warm and inviting like a favorite story',
    'sarcastic': 'Sarcastic, ironic, dry wit, teasing and playful with a knowing tone',
    'urgent': 'Urgent, pressing, rapid, serious and tense with a sense of immediacy',
    'whimsical': 'Whimsical, playful, lighthearted, dreamy and fanciful like a fairytale',
    'authoritative': 'Authoritative, commanding, confident, firm and decisive like an expert',
}

def resolve_emotion_instruct(emotion: str) -> str:
    """Convert an emotion tag name (e.g. 'excited', 'sad') to an instruct string.

    If the emotion is not in the map, returns it as-is (so custom instruct
    strings can be used directly as tags).
    """
    if not emotion or emotion.lower() == 'neutral':
        return ''
    return EMOTION_INSTRUCT_MAP.get(emotion.lower(), emotion)

# ── Supported speakers (from Qwen3-TTS) ────────────────────────────
# (display_name, model_name) — language labels help identify each speaker
PREDEFINED_SPEAKERS = [
    ("Vivian (Chinese)", "Vivian"),
    ("Serena (Chinese)", "Serena"),
    ("Uncle_Fu (Chinese)", "Uncle_Fu"),
    ("Dylan (Chinese-Beijing)", "Dylan"),
    ("Eric (Chinese-Sichuan)", "Eric"),
    ("Ryan (English)", "Ryan"),
    ("Aiden (English)", "Aiden"),
    ("Ono_Anna (Japanese)", "Ono_Anna"),
    ("Sohee (Korean)", "Sohee"),
]

SUPPORTED_LANGUAGES = [
    "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese",
    "Spanish", "Italian",
    # Additional — may work, not officially tested
    "Urdu", "Hindi", "Arabic",
]

# ── Python executable auto-detect ─────────────────────────────
# Qwen3-TTS needs torch + qwen_tts, which may not be in the GUI's
# Python (3.13). Probe known paths and cache the first one that
# can import qwen_tts.
_QWEN3_PYTHON: Optional[str] = None


def _find_qwen3_python() -> str:
    """Return a path to a Python that can import qwen_tts + torch.
    Cached after first probe."""
    global _QWEN3_PYTHON
    if _QWEN3_PYTHON is not None:
        return _QWEN3_PYTHON
    return _probe_qwen3_python()


def _probe_qwen3_python() -> str:
    """Probe all candidate Python executables for qwen_tts availability."""
    global _QWEN3_PYTHON
    candidates = [
        sys.executable,
        r'C:\Users\shahi\AppData\Local\Programs\Python\Python311\python.exe',
        r'C:\Users\shahi\AppData\Local\Programs\Python\Python312\python.exe',
        'python3',
        'python',
    ]
    for cand in candidates:
        if not cand:
            continue
        try:
            r = subprocess.run(
                [cand, '-c', 'import os; os.environ["USE_TF"]="0"; import qwen_tts; import torch; print(torch.cuda.is_available())'],
                capture_output=True, text=True, timeout=60,
            )
            if r.returncode == 0:
                _QWEN3_PYTHON = cand
                # flash-attn / SoX warnings print to stdout, not stderr.
                # Take the LAST non-empty line of output to find the boolean.
                _out_lines = [ln.strip() for ln in r.stdout.strip().splitlines() if ln.strip()]
                has_cuda = _out_lines[-1] == 'True' if _out_lines else False
                logger.info(f'Qwen3-TTS using Python: {cand} (CUDA={has_cuda})')
                return cand
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            continue

    _QWEN3_PYTHON = sys.executable or 'python'
    logger.warning(f'Qwen3-TTS Python auto-detect failed — falling back to {_QWEN3_PYTHON}')
    return _QWEN3_PYTHON


def reprobe_python() -> str:
    """Clear the cached Python path and re-probe. Call this when
    Qwen3 generation fails — it forces fresh detection."""
    global _QWEN3_PYTHON
    _QWEN3_PYTHON = None
    return _probe_qwen3_python()


def get_python_path() -> str:
    """Return the currently cached Python path (or None if not probed yet)."""
    return _QWEN3_PYTHON


def is_available() -> bool:
    """Check whether the model files exist on disk.

    Checks the active model directory (set via set_model_size).
    Returns True if config.json is found there.
    """
    return (get_active_model_dir() / 'config.json').exists()


def get_available_speakers() -> list:
    """Return the list of predefined speakers (display names with language)."""
    return [s[0] for s in PREDEFINED_SPEAKERS]


def _resolve_speaker_name(display_name: str) -> str:
    """Convert a display name like 'Ryan (English)' back to the model name 'Ryan'."""
    for display, model in PREDEFINED_SPEAKERS:
        if display == display_name:
            return model
    return display_name  # fallback: use as-is


def is_base_available() -> bool:
    """Check whether the Base model (voice cloning) files exist on disk.

    Checks the active base directory (set via set_model_size).
    """
    _bd = get_active_base_dir()
    return (_bd / 'config.json').exists() and (_bd / 'model.safetensors').exists()


def generate_voice_clone(
    text: str,
    output_path: Path,
    ref_audio: str,
    ref_text: str = "",
    language: str = "English",
    instruct: str = "",
    speed: float = 1.0,
) -> Optional[Path]:
    """Generate speech using Qwen3-TTS Base model (voice cloning) via subprocess.

    Uses a temp script + stdin JSON piping — same pattern as generate_speech().

    The 1.7B Base model supports emotion/style instruct alongside voice cloning,
    so you can clone a voice AND give it a specific emotional delivery.

    Args:
        text: Text to convert to speech.
        output_path: Desired output path (.wav).
        ref_audio: Path to reference audio file (10-30s, .wav or .mp3).
        ref_text: Transcription of what is spoken in ref_audio (optional but
                  recommended for better quality).
        language: One of SUPPORTED_LANGUAGES.
        instruct: Style/emotion instruction (e.g. "Excited, energetic").
                  Only effective with 1.7B Base model.
        speed: Speech speed multiplier (0.5-2.0).

    Returns:
        Path to output WAV on success, or None on failure.
    """
    _bd = get_active_base_dir()
    if not is_base_available():
        logger.error(f"Qwen3-TTS Base model not found at {_bd}")
        return None

    if not text.strip():
        logger.error("Qwen3-TTS voice clone: empty text")
        return None

    ref_audio_path = Path(ref_audio)
    if not ref_audio_path.exists():
        logger.error(f"Qwen3-TTS voice clone: reference audio not found: {ref_audio}")
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    is_mp3 = output_path.suffix.lower() == '.mp3'
    wav_path = output_path.with_suffix('.wav') if is_mp3 else output_path

    params = {
        'model_dir': str(_bd),
        'faster_path': str(FASTER_DIR),
        'text': text,
        'ref_audio': str(ref_audio_path.resolve()),
        'ref_text': ref_text,
        'language': language,
        'instruct': instruct,
        'speed': speed,
        'wav_path': str(wav_path),
    }

    script_body = r'''import sys, json, torch, soundfile as sf, traceback
import os, warnings, gc
os.environ['USE_TF'] = '0'
warnings.filterwarnings("ignore")
# Diagnostic: show which Python we are running
_with_exe = getattr(sys, "executable", "unknown")
print(f"PYTHON: {_with_exe} (v{sys.version_info.major}.{sys.version_info.minor})")
try:
    from qwen_tts import Qwen3TTSModel
except ImportError as _ie:
    print(f"IMPORT_ERR: qwen_tts not found at {_with_exe}")
    sys.exit(1)

params = json.loads(sys.stdin.read())

model_dir = params["model_dir"]
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# fp16 on CUDA: 1.7B model is ~7GB in float32 (won't fit 6-8GB cards) but
# ~3.5GB in fp16.  Turing+ has native fp16 tensor cores, so it's also faster.
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
attn_impl = "eager"

try:
    # Clear any stale CUDA state before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # -- Try faster-qwen3-tts CUDA backend --
    _faster_path = params.get("faster_path", "")
    _faster_parent = os.path.dirname(_faster_path) if _faster_path else ""
    if _faster_path and os.path.isdir(_faster_path):
        if _faster_parent and _faster_parent not in sys.path:
            sys.path.insert(0, _faster_parent)
        if _faster_path not in sys.path:
            sys.path.insert(0, _faster_path)
    _use_faster = False
    if torch.cuda.is_available() and _faster_path:
        try:
            from faster_qwen3_tts import FasterQwen3TTS
            model = FasterQwen3TTS.from_pretrained(
                model_dir, device="cuda", dtype=dtype,
                attn_implementation=attn_impl, max_seq_len=4096,
            )
            _use_faster = True
        except Exception:
            pass
    if not _use_faster:
        model = Qwen3TTSModel.from_pretrained(
            model_dir, device_map=device, dtype=dtype,
            attn_implementation=attn_impl,
        )
    ref_text = params.get("ref_text", "") or ""
    x_vector_only = params.get("x_vector_only", False) or not ref_text.strip()
    instruct = params.get("instruct", "") or ""
    wavs, sr = model.generate_voice_clone(
        text=params["text"],
        language=params["language"],
        ref_audio=params["ref_audio"],
        ref_text=ref_text,
        instruct=instruct,
        x_vector_only_mode=x_vector_only,
        do_sample=True,
        temperature=0.75,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.05,
    )
    audio = wavs[0]
    sr_out = int(sr)
    sp = params.get("speed", 1.0)
    if sp != 1.0 and sp > 0:
        import numpy as np
        orig_len = len(audio)
        new_len = int(orig_len / sp)
        x_old = np.linspace(0, 1, orig_len)
        x_new = np.linspace(0, 1, new_len)
        try:
            audio = np.interp(x_new, x_old, audio)
        except Exception:
            pass
    # Upsample to 44100 Hz for consistent compositing with GUI
    if sr_out != 44100:
        import numpy as _rs
        _nl = int(len(audio) * 44100 / sr_out)
        _xo = _rs.linspace(0, 1, len(audio))
        _xn = _rs.linspace(0, 1, _nl)
        audio = _rs.interp(_xn, _xo, audio)
        sr_out = 44100
    sf.write(params["wav_path"], audio, sr_out)
    # Clean up to free VRAM
    del model, wavs, audio
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("OK:" + params["wav_path"])
except Exception as e:
    tb = traceback.format_exc()[:500]
    print("ERROR:" + str(e) + "|TRACEBACK:" + tb)
    # Clean up on error too
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    sys.exit(1)
'''

    script_path = None
    for attempt in range(2):
        python_exe = _find_qwen3_python() if attempt == 0 else reprobe_python()
        try:
            if script_path is None:
                with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.py', delete=False, encoding='utf-8',
                ) as tf:
                    tf.write(script_body)
                    script_path = tf.name

            params_json = json.dumps(params, ensure_ascii=False)
            result = subprocess.run(
                [python_exe, script_path],
                input=params_json,
                capture_output=True, text=True, timeout=300,
            )

            for line in result.stdout.splitlines():
                if line.startswith('OK:'):
                    logger.info(f"Qwen3-TTS voice clone output: {line}")
                else:
                    logger.debug(f"[Qwen3-TTS clone] {line}")
            if result.stderr:
                for line in result.stderr.splitlines():
                    logger.debug(f"[Qwen3-TTS clone:stderr] {line}")

            if result.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 0:
                logger.info(f"Qwen3-TTS voice clone succeeded: {wav_path} ({wav_path.stat().st_size:,} bytes)")

                if is_mp3:
                    try:
                        subprocess.run([
                            'ffmpeg', '-y', '-i', str(wav_path),
                            '-acodec', 'libmp3lame', '-q:a', '2',
                            str(output_path),
                        ], capture_output=True, check=True, timeout=60)
                        if wav_path.exists():
                            wav_path.unlink()
                        logger.info(f"Qwen3-TTS voice clone MP3: {output_path}")
                    except Exception as e:
                        logger.error(f"Qwen3-TTS voice clone MP3 conversion failed: {e}")
                        return wav_path

                return output_path
            else:
                # Try to extract a meaningful error from stdout
                err_msg = 'unknown error'
                py_info = ''
                is_import_err = False
                for line in result.stdout.splitlines():
                    if line.startswith('ERROR:'):
                        err_msg = line[6:].strip()[:200]
                    elif line.startswith('IMPORT_ERR:'):
                        err_msg = line[6:].strip()[:200]
                        is_import_err = True
                    elif line.startswith('PYTHON:'):
                        py_info = line[7:].strip()
                if err_msg == 'unknown error':
                    err_msg = (result.stderr.strip() or result.stdout.strip() or 'unknown error')[:200]
                if py_info:
                    err_msg = f"[{py_info}] {err_msg}"
                logger.error(f"Qwen3-TTS voice clone failed: {err_msg}")

                # Retry once if import error — cache might be stale
                if is_import_err and attempt == 0:
                    logger.warning(f"Import error detected, re-probing Python and retrying...")
                    continue

                # Store last error for diagnostics
                _set_last_error(f"clone: {err_msg}")
                return None

        except subprocess.TimeoutExpired:
            logger.error("Qwen3-TTS voice clone timed out after 5 minutes")
            _set_last_error("clone: timed out after 5 minutes")
            return None
        except Exception as e:
            logger.error(f"Qwen3-TTS voice clone error: {e}")
            _set_last_error(f"clone: {e}")
            return None
        finally:
            if script_path is not None:
                try:
                    Path(script_path).unlink(missing_ok=True)
                except Exception:
                    pass
            script_path = None  # reset for possible retry


def generate_speech(
    text: str,
    output_path: Path,
    speaker: str = "Ryan (English)",
    language: str = "English",
    instruct: str = "",
    speed: float = 1.0,
) -> Optional[Path]:
    """Generate speech using Qwen3-TTS CustomVoice model via subprocess.

    Uses a temp JSON file to pass data to the subprocess, avoiding any
    quoting/shell-injection issues from embedding text in command-line args.

    Args:
        text: Text to convert to speech.
        output_path: Desired output path (.wav).
        speaker: One of PREDEFINED_SPEAKERS.
        language: One of SUPPORTED_LANGUAGES.
        instruct: Style/emotion instruction (e.g. "Happy, excited").
        speed: Speech speed multiplier (0.5-2.0).

    Returns:
        Path to output WAV on success, or None on failure.
    """
    if not is_available():
        _md = get_active_model_dir()
        logger.error(f"Qwen3-TTS model not found at {_md}")
        return None

    if not text.strip():
        logger.error("Qwen3-TTS: empty text")
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # If the caller wants MP3, generate WAV first then convert
    is_mp3 = output_path.suffix.lower() == '.mp3'
    wav_path = output_path.with_suffix('.wav') if is_mp3 else output_path

    # Resolve display name (e.g. "Ryan (English)") to model name ("Ryan")
    speaker = _resolve_speaker_name(speaker)

    # Write a temp script file and a temp params JSON.
    # Using a file avoids Windows command-line length limits (~8191 chars).
    params = {
        'model_dir': str(get_active_model_dir()),
        'faster_path': str(FASTER_DIR),
        'text': text,
        'instruct': instruct,
        'language': language,
        'speaker': speaker,
        'speed': speed,
        'wav_path': str(wav_path),
    }

    script_body = r'''import sys, json, torch, soundfile as sf, traceback
import os, warnings, gc
# Prevent transformers from importing tensorflow (protobuf conflict)
os.environ['USE_TF'] = '0'
warnings.filterwarnings("ignore")
# Diagnostic: show which Python we are running
_with_exe = getattr(sys, "executable", "unknown")
print(f"PYTHON: {_with_exe} (v{sys.version_info.major}.{sys.version_info.minor})")
try:
    from qwen_tts import Qwen3TTSModel
except ImportError as _ie:
    print(f"IMPORT_ERR: qwen_tts not found at {_with_exe}")
    sys.exit(1)

params = json.loads(sys.stdin.read())

model_dir = params["model_dir"]
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# fp16 on CUDA: 1.7B model is ~7GB in float32 (won't fit 6-8GB cards) but
# ~3.5GB in fp16.  Turing+ has native fp16 tensor cores, so it's also faster.
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
attn_impl = "eager"

try:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    # -- Try faster-qwen3-tts CUDA backend --
    _faster_path = params.get("faster_path", "")
    _faster_parent = os.path.dirname(_faster_path) if _faster_path else ""
    if _faster_path and os.path.isdir(_faster_path):
        if _faster_parent and _faster_parent not in sys.path:
            sys.path.insert(0, _faster_parent)
        if _faster_path not in sys.path:
            sys.path.insert(0, _faster_path)
    _use_faster = False
    if torch.cuda.is_available() and _faster_path:
        try:
            from faster_qwen3_tts import FasterQwen3TTS
            model = FasterQwen3TTS.from_pretrained(
                model_dir, device="cuda", dtype=dtype,
                attn_implementation=attn_impl, max_seq_len=4096,
            )
            _use_faster = True
        except Exception:
            pass
    if not _use_faster:
        model = Qwen3TTSModel.from_pretrained(
            model_dir, device_map=device, dtype=dtype,
            attn_implementation=attn_impl,
        )
    wavs, sr = model.generate_custom_voice(
        text=params["text"],
        language=params["language"],
        speaker=params["speaker"],
        instruct=params.get("instruct", "") or "",
        do_sample=True,
        temperature=0.75,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.05,
    )
    audio = wavs[0]
    sr_out = int(sr)
    orig_dur = len(audio) / sr_out
    sp = params.get("speed", 1.0)
    if sp != 1.0 and sp > 0:
        import numpy as np
        orig_len = len(audio)
        new_len = int(orig_len / sp)
        x_old = np.linspace(0, 1, orig_len)
        x_new = np.linspace(0, 1, new_len)
        try:
            audio = np.interp(x_new, x_old, audio)
        except Exception:
            pass
    adj_dur = len(audio) / sr_out
    print(f"DIAG: orig={orig_dur:.2f}s adj={adj_dur:.2f}s sp={sp}")
    # Upsample to 44100 Hz for consistent compositing with GUI
    if sr_out != 44100:
        import numpy as _resamp_np
        _new_len = int(len(audio) * 44100 / sr_out)
        _x_old = _resamp_np.linspace(0, 1, len(audio))
        _x_new = _resamp_np.linspace(0, 1, _new_len)
        audio = _resamp_np.interp(_x_new, _x_old, audio)
        sr_out = 44100
    sf.write(params["wav_path"], audio, sr_out)
    del model, wavs, audio
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("OK:" + params["wav_path"])
except Exception as e:
    tb = traceback.format_exc()[:500]
    print("ERROR:" + str(e) + "|TRACEBACK:" + tb)
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    sys.exit(1)
'''

    script_path = None
    for attempt in range(2):
        python_exe = _find_qwen3_python() if attempt == 0 else reprobe_python()
        try:
            if script_path is None:
                # Write the runner script to a temp file (avoids cmdline length limits)
                with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.py', delete=False, encoding='utf-8',
                ) as tf:
                    tf.write(script_body)
                    script_path = tf.name

            # Pipe the JSON params via stdin — avoids any quoting issues
            params_json = json.dumps(params, ensure_ascii=False)
            result = subprocess.run(
                [python_exe, script_path],
                input=params_json,
                capture_output=True, text=True, timeout=300,  # 5 min
            )

            # Log output for debugging
            for line in result.stdout.splitlines():
                if line.startswith('OK:'):
                    logger.info(f"Qwen3-TTS output: {line}")
                else:
                    logger.debug(f"[Qwen3-TTS] {line}")
            if result.stderr:
                for line in result.stderr.splitlines():
                    logger.debug(f"[Qwen3-TTS:stderr] {line}")

            if result.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 0:
                logger.info(f"Qwen3-TTS succeeded: {wav_path} ({wav_path.stat().st_size:,} bytes)")

                # Convert to MP3 if needed
                if is_mp3:
                    try:
                        subprocess.run([
                            'ffmpeg', '-y', '-i', str(wav_path),
                            '-acodec', 'libmp3lame', '-q:a', '2',
                            str(output_path),
                        ], capture_output=True, check=True, timeout=60)
                        if wav_path.exists():
                            wav_path.unlink()
                        logger.info(f"Qwen3-TTS MP3: {output_path}")
                    except Exception as e:
                        logger.error(f"Qwen3-TTS MP3 conversion failed: {e}")
                        return wav_path  # return WAV as fallback

                return output_path
            else:
                # Try to extract a meaningful error from stdout
                err_msg = 'unknown error'
                py_info = ''
                is_import_err = False
                for line in result.stdout.splitlines():
                    if line.startswith('ERROR:'):
                        err_msg = line[6:].strip()[:300]
                    elif line.startswith('IMPORT_ERR:'):
                        err_msg = line[6:].strip()[:300]
                        is_import_err = True
                    elif line.startswith('PYTHON:'):
                        py_info = line[7:].strip()
                if err_msg == 'unknown error':
                    err_msg = (result.stderr.strip() or result.stdout.strip() or 'unknown error')[:300]
                if py_info:
                    err_msg = f"[{py_info}] {err_msg}"
                logger.error(f"Qwen3-TTS failed: {err_msg}")

                # Retry once if import error — cache might be stale
                if is_import_err and attempt == 0:
                    logger.warning(f"Import error detected, re-probing Python and retrying...")
                    continue

                _set_last_error(f"cv: {err_msg}")
                return None

        except subprocess.TimeoutExpired:
            logger.error("Qwen3-TTS timed out after 5 minutes")
            _set_last_error("cv: timed out after 5 minutes")
            return None
        except Exception as e:
            logger.error(f"Qwen3-TTS error: {e}")
            _set_last_error(f"cv: {e}")
            return None
        finally:
            # Clean up temp script file on each attempt
            if script_path is not None:
                try:
                    Path(script_path).unlink(missing_ok=True)
                except Exception:
                    pass
            script_path = None  # reset for possible retry




def generate_speech_batch(
    segments: list,
    output_dir: Path,
    speaker: str = "Ryan (English)",
    language: str = "English",
    instruct: str = "",
    speed: float = 1.0,
) -> list:
    """Generate speech for MULTIPLE text segments in ONE subprocess.

    Model loads ONCE, generates all segments, then exits.
    Each segment is a dict: {'text': str, 'wav_name': str}
    Returns list of (wav_path, success_bool).

    This is 10-15x faster than calling generate_speech() N times
    because model loading (~30-50s) happens only once.
    """
    if not is_available():
        _md = get_active_model_dir()
        logger.error("Qwen3-TTS model not found at %s", _md)
        return [(None, False)] * len(segments)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    speaker = _resolve_speaker_name(speaker)

    # Build list of generation requests
    # Each segment dict can optionally include its own 'instruct', 'speaker',
    # 'language', 'speed' fields to override the global defaults.
    requests = []
    for seg in segments:
        wav_path = output_dir / seg["wav_name"]
        requests.append({
            "text": seg["text"],
            "wav_path": str(wav_path),
            "speaker": seg.get("speaker", speaker),
            "language": seg.get("language", language),
            "instruct": seg.get("instruct", instruct),
            "speed": seg.get("speed", speed),
        })

    params = {
        "model_dir": str(get_active_model_dir()),
        "faster_path": str(FASTER_DIR),
        "requests": requests,
    }

    # Subprocess script -- same as generate_speech but loops through requests
    script_body = (
        'import sys, json, torch, soundfile as sf, traceback\n'
        'import os, warnings, gc\n'
        'os.environ["USE_TF"] = "0"\n'
        'warnings.filterwarnings("ignore")\n'
        '_with_exe = getattr(sys, "executable", "unknown")\n'
        'print(f"PYTHON: {_with_exe} (v{sys.version_info.major}.{sys.version_info.minor})")\n'
        'try:\n'
        '    from qwen_tts import Qwen3TTSModel\n'
        'except ImportError as _ie:\n'
        '    print("IMPORT_ERR: qwen_tts not found at", _with_exe)\n'
        '    sys.exit(1)\n'
        'params = json.loads(sys.stdin.read())\n'
        'model_dir = params["model_dir"]\n'
        'device = "cuda:0" if torch.cuda.is_available() else "cpu"\n'
        'dtype = torch.float16 if torch.cuda.is_available() else torch.float32\n'
        'attn_impl = "eager"\n'
        'try:\n'
        '    if torch.cuda.is_available():\n'
        '        torch.cuda.empty_cache()\n'
        '        torch.cuda.synchronize()\n'
        '    _faster_path = params.get("faster_path", "")\n'
        '    _faster_parent = os.path.dirname(_faster_path) if _faster_path else ""\n'
        '    if _faster_path and os.path.isdir(_faster_path):\n'
        '        if _faster_parent and _faster_parent not in sys.path:\n'
        '            sys.path.insert(0, _faster_parent)\n'
        '        if _faster_path not in sys.path:\n'
        '            sys.path.insert(0, _faster_path)\n'
        '    _use_faster = False\n'
        '    if torch.cuda.is_available() and _faster_path:\n'
        '        try:\n'
        '            from faster_qwen3_tts import FasterQwen3TTS\n'
        '            model = FasterQwen3TTS.from_pretrained(\n'
        '                model_dir, device="cuda", dtype=dtype,\n'
        '                attn_implementation=attn_impl, max_seq_len=4096,\n'
        '            )\n'
        '            _use_faster = True\n'
        '        except Exception:\n'
        '            print("DIAG: faster backend failed")\n'
        '            pass\n'
        '    if not _use_faster:\n'
        '        model = Qwen3TTSModel.from_pretrained(\n'
        '            model_dir, device_map=device, dtype=dtype,\n'
        '            attn_implementation=attn_impl,\n'
        '        )\n'
        '    print("DIAG: model_loaded=" + ("faster" if _use_faster else "standard"))\n'
        '    for i, req in enumerate(params["requests"]):\n'
        '        try:\n'
        '            wavs, sr = model.generate_custom_voice(\n'
        '                text=req["text"],\n'
        '                language=req["language"],\n'
        '                speaker=req["speaker"],\n'
        '                instruct=req.get("instruct", "") or "",\n'
        '                do_sample=True,\n'
        '                temperature=0.75,\n'
        '                top_k=50,\n'
        '                top_p=0.9,\n'
        '                repetition_penalty=1.05,\n'
        '            )\n'
        '            audio = wavs[0]\n'
        '            sr_out = int(sr)\n'
        '            orig_dur = len(audio) / sr_out\n'
        '            sp = req.get("speed", 1.0)\n'
        '            if sp != 1.0 and sp > 0:\n'
        '                import numpy as np\n'
        '                orig_len = len(audio)\n'
        '                new_len = int(orig_len / sp)\n'
        '                x_old = np.linspace(0, 1, orig_len)\n'
        '                x_new = np.linspace(0, 1, new_len)\n'
        '                try:\n'
        '                    audio = np.interp(x_new, x_old, audio)\n'
        '                except Exception:\n'
        '                    pass\n'
        '            adj_dur = len(audio) / sr_out\n'
        '            print(f"DIAG: seg i={i}: orig={orig_dur:.2f}s adj={adj_dur:.2f}s sp={sp}")\n'
        '            # Upsample to 44100 Hz for consistent compositing with GUI\n'
        '            if sr_out != 44100:\n'
        '                import numpy as _resamp_np\n'
        '                _new_len = int(len(audio) * 44100 / sr_out)\n'
        '                _x_old = _resamp_np.linspace(0, 1, len(audio))\n'
        '                _x_new = _resamp_np.linspace(0, 1, _new_len)\n'
        '                audio = _resamp_np.interp(_x_new, _x_old, audio)\n'
        '                sr_out = 44100\n'
        '            sf.write(req["wav_path"], audio, sr_out)\n'
        '            print("OK:" + req["wav_path"])\n'
        '        except Exception as e:\n'
        '            tb = traceback.format_exc()[:500]\n'
        '            print("ERROR:" + str(e) + "|" + tb[:200])\n'
        '    del model\n'
        '    if torch.cuda.is_available():\n'
        '        torch.cuda.empty_cache()\n'
        '    gc.collect()\n'
        'except Exception as e:\n'
        '    tb = traceback.format_exc()[:500]\n'
        '    print("FATAL:" + tb)\n'
        '    sys.exit(1)\n'
    )

    import subprocess as _sp
    import tempfile as _tf

    script_path = None
    for attempt in range(2):
        python_exe = _find_qwen3_python() if attempt == 0 else reprobe_python()
        try:
            if script_path is None:
                with _tf.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
                    tf.write(script_body)
                    script_path = tf.name

            params_json = json.dumps(params, ensure_ascii=False)
            result = _sp.run(
                [python_exe, script_path],
                input=params_json,
                capture_output=True, text=True, timeout=600,
            )

            # Log all diagnostic lines from the subprocess
            _last_diag = ""
            for line in result.stdout.splitlines():
                if line.startswith("PYTHON:"):
                    logger.info("[batch] %s", line.strip())
                if line.startswith("DIAG:"):
                    _last_diag = line[5:].strip()
                    logger.info("[batch] %s", line.strip())

            # Parse results
            results = []
            _batch_errors = []
            for line in result.stdout.splitlines():
                if line.startswith("OK:"):
                    wav = line[3:].strip()
                    results.append((Path(wav), True))
                elif line.startswith("ERROR:"):
                    results.append((None, False))
                    _err_msg = line[6:].strip()[:200]
                    _batch_errors.append(_err_msg)

            if results:
                if not any(r[1] for r in results):
                    _first = _batch_errors[0] if _batch_errors else ("unknown error; " + _last_diag if _last_diag else "unknown error")
                    logger.error("Qwen3 batch: all %d segments failed — first error: %s", len(results), _first)
                    _set_last_error("batch: " + _first)
                return results

            err_msg = "unknown error"
            for line in result.stdout.splitlines():
                if line.startswith("IMPORT_ERR:") and attempt == 0:
                    continue
                if line.startswith("ERROR:") or line.startswith("FATAL:"):
                    err_msg = line[6:].strip()[:300]
            logger.error("Qwen3 batch failed: %s", err_msg)
            _set_last_error("batch: " + err_msg)
            return [(None, False)] * len(segments)

        except _sp.TimeoutExpired:
            logger.error("Qwen3 batch timed out")
            _set_last_error("batch: timed out")
            return [(None, False)] * len(segments)
        except Exception as e:
            logger.error("Qwen3 batch error: %s", e)
            _set_last_error("batch: " + str(e))
            return [(None, False)] * len(segments)
        finally:
            if script_path is not None:
                try:
                    Path(script_path).unlink(missing_ok=True)
                except Exception:
                    pass
            script_path = None
def generate_with_emotion(
    text: str,
    output_path: Path,
    speaker: str = "Ryan (English)",
    language: str = "English",
    emotion: str = "neutral",
    speed: float = 1.0,
) -> Optional[Path]:
    """Generate speech with an emotion/style instruction.

    Emotion presets map to instruct strings for the Qwen3-TTS model:

        neutral     → ""
        happy       → "Happy, cheerful"
        sad         → "Sad, melancholic"
        excited     → "Excited, energetic"
        angry       → "Angry, aggressive"
        calm        → "Calm, soothing"
        whisper     → "Whispering, soft"
        deep        → "Deep, authoritative"
        dramatic    → "Dramatic, intense"
        storytelling → "Narrative, storytelling"
    """
    emotion_map = {
        'neutral': '',
        'happy': 'Happy, cheerful, bright and uplifting tone with a smile in the voice',
        'sad': 'Sad, melancholic, sorrowful, soft and gentle with a touch of longing',
        'excited': 'Excited, energetic, enthusiastic, dynamic and passionate delivery',
        'angry': 'Angry, aggressive, intense, forceful and sharp with strong emphasis',
        'calm': 'Calm, soothing, gentle, relaxed and peaceful like a quiet evening',
        'whisper': 'Whispering, soft, intimate, quiet and close like sharing a secret',
        'deep': 'Deep, authoritative, powerful, commanding and resonant like a narrator',
        'dramatic': 'Dramatic, intense, impactful, theatrical and grand with rising tension',
        'storytelling': 'Narrative, storytelling, engaging, warm and inviting like a favorite story',
    }
    instruct = emotion_map.get(emotion.lower(), emotion)
    return generate_speech(
        text=text,
        output_path=output_path,
        speaker=speaker,
        language=language,
        instruct=instruct,
        speed=speed,
    )
