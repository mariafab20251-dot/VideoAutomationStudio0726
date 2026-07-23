"""Gemini API Text-to-Speech via ai.google.dev (Developer API).

Supports THREE auth methods (tried in order):
  1. Service Account JSON — OAuth2 Bearer token → Cloud TTS + Gemini TTS
  2. API key (AIza... from Cloud Console / AQ... from aistudio) → Gemini TTS only

API key format note: AIza keys (Cloud Console) have TTS quota; AQ keys (aistudio)
may have limit=0 for TTS models. Use a Cloud Console key or service account
for TTS access.

Emotion tags (inline in text, works with Gemini TTS):
    [whispers], [excited], [laughs], [sighs], [shouting], [sarcastic],
    [serious], [tired], [gasp], [amazed], [crying], [curious],
    [very fast], [very slow], etc — experiment freely.

Cloud TTS voices (used when service_account_path provided):
    Uses Google Cloud Text-to-Speech API for higher quality voices
    (Wavenet, Studio, Neural2). Pass use_cloud_tts=True in settings.
"""

import base64
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────
API_BASE = "https://generativelanguage.googleapis.com/v1beta"
CLOUD_TTS_BASE = "https://texttospeech.googleapis.com/v1"

# Gemini TTS has a 4000 byte limit on input.text + input.prompt combined.
# Leave a safety margin for JSON overhead and prompt text.
MAX_INPUT_BYTES = 3800

# Gemini TTS models (esp. *-pro-tts / *-flash-tts-preview) can take 70-120s
# to return audio for a full chunk. A 60s read timeout was tripping before
# the server ever replied, producing "read operation timed out" failures.
TTS_TIMEOUT = 240


def _truncate_to_byte_limit(text: str, limit: int) -> str:
    """Truncate text so its UTF-8 encoding fits within `limit` bytes."""
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(text[:mid].encode("utf-8")) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def _split_text_to_chunks(text: str, limit: int) -> list[str]:
    """Split text into chunks each <= `limit` UTF-8 bytes, breaking on
    sentence / clause / word boundaries so nothing is dropped.

    Used to work around the Gemini/Cloud TTS ~4000 byte per-request cap:
    long scripts are synthesized as multiple requests and concatenated.
    """
    text = text.strip()
    if not text:
        return []
    if len(text.encode("utf-8")) <= limit:
        return [text]

    # Split into sentences first (keep the delimiter with the sentence).
    sentences = re.split(r"(?<=[.!?…])\s+", text)

    chunks: list[str] = []
    current = ""

    def _flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for sentence in sentences:
        if not sentence:
            continue
        # A single sentence bigger than the limit → split on words.
        if len(sentence.encode("utf-8")) > limit:
            _flush()
            words = sentence.split()
            piece = ""
            for w in words:
                candidate = (piece + " " + w).strip()
                if len(candidate.encode("utf-8")) > limit:
                    if piece.strip():
                        chunks.append(piece.strip())
                    # Word alone longer than limit — hard-truncate it.
                    if len(w.encode("utf-8")) > limit:
                        piece = _truncate_to_byte_limit(w, limit)
                    else:
                        piece = w
                else:
                    piece = candidate
            if piece.strip():
                current = piece
            continue

        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate.encode("utf-8")) > limit:
            _flush()
            current = sentence
        else:
            current = candidate

    _flush()
    return chunks


def _extract_pcm(audio_bytes: bytes) -> tuple[bytes, int, int, int]:
    """Return (pcm_frames, sample_rate, channels, sample_width) from either a
    RIFF-WAV byte blob or raw PCM (assumed 24kHz/mono/16-bit)."""
    if audio_bytes[:4] == b"RIFF":
        import io
        import wave as _wave
        with _wave.open(io.BytesIO(audio_bytes), "rb") as rf:
            return (
                rf.readframes(rf.getnframes()),
                rf.getframerate(),
                rf.getnchannels(),
                rf.getsampwidth(),
            )
    # Raw PCM fallback (Gemini Interactions API default format)
    return audio_bytes, 24000, 1, 2

# ── Gemini TTS voices (30) ─────────────────────────────────────────────
GEMINI_TTS_VOICES: dict[str, str] = {
    "Zephyr": "Zephyr — Bright",
    "Charon": "Charon — Informative",
    "Fenrir": "Fenrir — Excitable",
    "Puck": "Puck — Upbeat",
    "Kore": "Kore — Firm",
    "Leda": "Leda — Youthful",
    "Orus": "Orus — Firm",
    "Aoede": "Aoede — Breezy",
    "Callirrhoe": "Callirrhoe — Easy-going",
    "Autonoe": "Autonoe — Bright",
    "Enceladus": "Enceladus — Breathy",
    "Iapetus": "Iapetus — Clear",
    "Umbriel": "Umbriel — Easy-going",
    "Algieba": "Algieba — Smooth",
    "Despina": "Despina — Smooth",
    "Erinome": "Erinome — Clear",
    "Algenib": "Algenib — Gravelly",
    "Rasalgethi": "Rasalgethi — Informative",
    "Laomedeia": "Laomedeia — Upbeat",
    "Achernar": "Achernar — Soft",
    "Alnilam": "Alnilam — Firm",
    "Schedar": "Schedar — Even",
    "Gacrux": "Gacrux — Mature",
    "Pulcherrima": "Pulcherrima — Forward",
    "Achird": "Achird — Friendly",
    "Zubenelgenubi": "Zubenelgenubi — Casual",
    "Vindemiatrix": "Vindemiatrix — Gentle",
    "Sadachbia": "Sadachbia — Lively",
    "Sadaltager": "Sadaltager — Knowledgeable",
    "Sulafat": "Sulafat — Warm",
}

# Gender per Gemini TTS voice (from Google's Gemini-TTS voice table).
GEMINI_VOICE_GENDER: dict[str, str] = {
    "Zephyr": "Female", "Charon": "Male", "Fenrir": "Male", "Puck": "Male",
    "Kore": "Female", "Leda": "Female", "Orus": "Male", "Aoede": "Female",
    "Callirrhoe": "Female", "Autonoe": "Female", "Enceladus": "Male",
    "Iapetus": "Male", "Umbriel": "Male", "Algieba": "Male", "Despina": "Female",
    "Erinome": "Female", "Algenib": "Male", "Rasalgethi": "Male",
    "Laomedeia": "Female", "Achernar": "Female", "Alnilam": "Male",
    "Schedar": "Male", "Gacrux": "Female", "Pulcherrima": "Female",
    "Achird": "Male", "Zubenelgenubi": "Male", "Vindemiatrix": "Female",
    "Sadachbia": "Male", "Sadaltager": "Male", "Sulafat": "Female",
}


def get_voice_gender(key: str) -> str:
    """Return 'Male'/'Female' for a Gemini voice key, or '' if unknown."""
    return GEMINI_VOICE_GENDER.get(key, "")


# Best-use hint per Gemini voice — a short "what this voice is good for" tag,
# derived from Google's own one-word descriptor (see GEMINI_TTS_VOICES).  Shown
# in the voice dropdowns so voices aren't picked blind.  Tone/emotion itself can
# still be steered per line via the ``gemini_tts_prompt`` style field.
GEMINI_VOICE_USE: dict[str, str] = {
    "Zephyr": "bright · lifestyle/cheerful",
    "Charon": "informative · news/explainer",
    "Fenrir": "excitable · hype/energetic",
    "Puck": "upbeat · ads/reactions",
    "Kore": "firm · authority/motivation",
    "Leda": "youthful · lifestyle/bright",
    "Orus": "firm · authority/corporate",
    "Aoede": "breezy · casual/podcast",
    "Callirrhoe": "easy-going · podcast/chat",
    "Autonoe": "bright · cheerful/upbeat",
    "Enceladus": "breathy · calm/soft",
    "Iapetus": "clear · explainer/tutorial",
    "Umbriel": "easy-going · podcast/chat",
    "Algieba": "smooth · luxury/late-night",
    "Despina": "smooth · luxury/late-night",
    "Erinome": "clear · news/explainer",
    "Algenib": "gravelly · documentary/gravitas",
    "Rasalgethi": "informative · news/anchor",
    "Laomedeia": "upbeat · energetic/ads",
    "Achernar": "soft · calm/meditation/ASMR",
    "Alnilam": "firm · motivation/authority",
    "Schedar": "even · documentary/steady",
    "Gacrux": "mature · story/documentary",
    "Pulcherrima": "forward · punchy/motivation",
    "Achird": "friendly · podcast/conversational",
    "Zubenelgenubi": "casual · podcast/vlog",
    "Vindemiatrix": "gentle · storytelling/calm",
    "Sadachbia": "lively · energetic/fun",
    "Sadaltager": "knowledgeable · explainer/educational",
    "Sulafat": "warm · storytelling/narration",
}


def get_voice_use(key: str) -> str:
    """Return the best-use hint for a Gemini voice key, or '' if unknown."""
    return GEMINI_VOICE_USE.get(key, "")

# ── Cloud TTS voices (popular English) ─────────────────────────────────
CLOUD_TTS_VOICES: dict[str, str] = {
    "en-US-Studio-Q": "Studio Q (US English, Male, Warm)",
    "en-US-Studio-O": "Studio O (US English, Female, Warm)",
    "en-US-Wavenet-J": "Wavenet J (US English, Male)",
    "en-US-Wavenet-D": "Wavenet D (US English, Male)",
    "en-US-Wavenet-F": "Wavenet F (US English, Female)",
    "en-US-Wavenet-G": "Wavenet G (US English, Female)",
    "en-US-Wavenet-H": "Wavenet H (US English, Female)",
    "en-US-Wavenet-I": "Wavenet I (US English, Male)",
    "en-US-Neural2-J": "Neural2 J (US English, Male)",
    "en-US-Neural2-D": "Neural2 D (US English, Male)",
    "en-US-Neural2-F": "Neural2 F (US English, Female)",
    "en-US-Neural2-G": "Neural2 G (US English, Female)",
    "en-US-Neural2-H": "Neural2 H (US English, Female)",
    "en-US-Neural2-I": "Neural2 I (US English, Male)",
    "en-US-Journey-D": "Journey D (US English, Male)",
    "en-US-Journey-F": "Journey F (US English, Female)",
    "en-GB-Wavenet-A": "Wavenet A (UK English, Female)",
    "en-GB-Wavenet-B": "Wavenet B (UK English, Male)",
    "en-GB-Wavenet-C": "Wavenet C (UK English, Female)",
    "en-GB-Wavenet-D": "Wavenet D (UK English, Male)",
    "en-GB-Neural2-A": "Neural2 A (UK English, Female)",
    "en-GB-Neural2-B": "Neural2 B (UK English, Male)",
    "en-IN-Wavenet-A": "Wavenet A (IN English, Female)",
    "en-IN-Wavenet-B": "Wavenet B (IN English, Male)",
    "en-IN-Neural2-A": "Neural2 A (IN English, Female)",
    "en-IN-Neural2-B": "Neural2 B (IN English, Male)",
    "en-AU-Wavenet-A": "Wavenet A (AU English, Female)",
    "en-AU-Wavenet-B": "Wavenet B (AU English, Male)",
    "en-AU-Neural2-A": "Neural2 A (AU English, Female)",
    "en-AU-Neural2-B": "Neural2 B (AU English, Male)",
}


def get_all_voices(cloud: bool = False) -> dict[str, str]:
    """Return voice dict. cloud=True returns Cloud TTS voices."""
    return dict(CLOUD_TTS_VOICES) if cloud else dict(GEMINI_TTS_VOICES)


def get_voice_keys(cloud: bool = False) -> list[str]:
    """Return voice name list. cloud=True returns Cloud TTS voices."""
    return list(CLOUD_TTS_VOICES.keys()) if cloud else list(GEMINI_TTS_VOICES.keys())


# ── Auth helpers ───────────────────────────────────────────────────────


_TOKEN_CACHE: dict[str, tuple[str, float]] = {}  # path → (token, expiry)

def _get_bearer_token(service_account_path: str) -> str:
    """Get an OAuth2 Bearer token from a service account JSON file.

    Tokens are cached per path and auto-refreshed when they expire (1h).
    Requires: pip install google-auth
    """
    import time as _time
    _now = _time.time()
    _cached = _TOKEN_CACHE.get(service_account_path)
    # Tokens are valid ~1h; refresh after 50 min to be safe.
    if _cached and (_now - _cached[1]) < 3000:
        return _cached[0]

    try:
        from google.oauth2 import service_account
        from google.auth.transport.requests import Request
    except ImportError:
        raise ImportError(
            "google-auth library required for service accounts. "
            "Run: pip install google-auth"
        )

    creds = service_account.Credentials.from_service_account_file(
        service_account_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    creds.refresh(Request())
    _TOKEN_CACHE[service_account_path] = (creds.token, _now)
    return creds.token


def _get_project_id(service_account_path: str) -> str:
    """Read the project_id out of a service account JSON file."""
    try:
        with open(service_account_path, "r", encoding="utf-8") as fh:
            return json.load(fh).get("project_id", "")
    except Exception:
        return ""


# ── HTTP helpers ───────────────────────────────────────────────────────


def _try_requests(
    url: str, headers: dict, payload: dict, timeout: int = 60
) -> tuple[bool, Any, str]:
    """Try using the `requests` library for the HTTP POST."""
    try:
        import requests as _req
    except ImportError:
        return False, None, "requests library not available"

    try:
        resp = _req.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 200:
            return True, resp.json(), ""
        else:
            body_preview = resp.text[:500]
            return (
                False,
                None,
                f"HTTP {resp.status_code}: {body_preview}",
            )
    except Exception as exc:
        return False, None, str(exc)


def _try_urllib(
    url: str, headers: dict, payload: dict, timeout: int = 60
) -> tuple[bool, Any, str]:
    """Fallback using urllib from stdlib."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return True, json.loads(body), ""
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        return False, None, f"HTTP {exc.code}: {body}"
    except Exception as exc:
        return False, None, str(exc)


def _http_post(
    url: str, payload: dict, headers: dict | None = None, timeout: int = 60
) -> tuple[bool, Any, str]:
    """POST JSON payload, return (ok, data_or_None, error_or_empty)."""
    h = headers or {"Content-Type": "application/json"}
    ok, data, err = _try_requests(url, h, payload, timeout)
    if not ok:
        ok, data, err = _try_urllib(url, h, payload, timeout)
    return ok, data, err


# ── Text translation (Gemini generateContent) ───────────────────────────


def translate_text(
    text: str,
    target_language: str,
    settings: dict | None = None,
    model: str = "gemini-2.5-flash",
    timeout: int = 120,
    source_language: str | None = None,
) -> tuple[bool, str]:
    """Translate `text` into `target_language` using Gemini generateContent.

    Reuses the same API-key / service-account auth as the TTS calls so no
    extra SDK or credential is needed. Returns (ok, translated_text). On any
    failure returns (False, error_reason) so the caller can surface it.

    The prompt is deliberately strict: return ONLY the translation, no
    quotes, no notes, preserve meaning and tone for spoken delivery.

    ``source_language`` — if provided and non-English, translation will run
    even when the target is English (needed for dubbing non-English videos).
    """
    s = settings or {}
    text = (text or "").strip()
    if not text:
        return False, "empty text"
    tgt = (target_language or "").strip()
    src = (source_language or "").strip().lower()
    _is_eng = lambda v: v.lower() in ("english", "en", "en-us", "en-gb")
    if not tgt:
        return False, "empty target language"
    if _is_eng(tgt) and (not src or _is_eng(src)):
        # No translation needed — English source stays as-is.
        return True, text

    api_key = (s.get("gemini_api_key") or "").strip()
    sa_path = (s.get("gemini_service_account") or
               s.get("service_account_path") or "").strip()

    prompt = (
        f"Translate the following text from {'English' if _is_eng(src) else src.title()} "
        f"into {tgt}. This is spoken dialogue for a "
        f"voiceover dub, so use natural, conversational {tgt} that sounds good when "
        f"read aloud. Keep it roughly the same length as the original.\n"
        f"Return ONLY the translated text — no quotes, no notes, no transliteration, "
        f"no original text.\n\n"
        f"TEXT:\n{text}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3},
    }

    # Prefer API key (simplest); fall back to service-account bearer token.
    # Service-account tokens ONLY work with the Vertex AI endpoint, not the
    # free Gemini API — so we switch to the Vertex AI host when using one.
    headers = {"Content-Type": "application/json"}
    is_vertex = False
    if api_key:
        url = f"{API_BASE}/models/{model}:generateContent"
        url = f"{url}?key={api_key}"
    elif sa_path:
        try:
            token = _get_bearer_token(sa_path)
            headers["Authorization"] = f"Bearer {token}"
            # Read the Vertex project-id from the service-account JSON.
            _project_id = _get_project_id(sa_path)
            if _project_id:
                url = (f"https://us-central1-aiplatform.googleapis.com"
                       f"/v1/projects/{_project_id}/locations/us-central1"
                       f"/publishers/google/models/{model}:generateContent")
                is_vertex = True
            else:
                return False, "service-account JSON has no project_id"
        except Exception as exc:
            return False, f"auth failed: {exc}"
    else:
        return False, "no gemini_api_key or service account configured"

    # Vertex AI requires role: "user" on content parts
    if is_vertex:
        payload["contents"][0]["role"] = "user"

    ok, data, err = _http_post(url, payload, headers, timeout)
    if not ok:
        return False, err or "translation request failed"
    try:
        parts = data["candidates"][0]["content"]["parts"]
        out = "".join(p.get("text", "") for p in parts).strip()
        # Strip stray wrapping quotes the model sometimes adds.
        if len(out) >= 2 and out[0] in "\"'" and out[-1] == out[0]:
            out = out[1:-1].strip()
        if not out:
            return False, "translation returned empty"
        return True, out
    except Exception as exc:
        # Surface API-level error messages (quota, bad key, safety block).
        try:
            _msg = data.get("error", {}).get("message", "")
        except Exception:
            _msg = ""
        return False, _msg or f"could not parse translation: {exc}"


def translate_lines(
    lines: list,
    target_language: str,
    settings: dict | None = None,
    model: str = "gemini-2.5-flash",
    timeout: int = 180,
    source_language: str | None = None,
) -> tuple[bool, list]:
    """Translate MANY lines in a SINGLE request (one API call, not one-per-line).

    This exists to avoid free-tier rate limits (HTTP 429): dubbing a video
    produces dozens of short lines, and firing one generateContent request per
    line quickly exceeds the 5-20 req/min free quota. Here every line is sent in
    one numbered prompt and the model returns a JSON array of translations.

    Returns (ok, translated_lines) where translated_lines is aligned 1:1 with
    the input. On any failure returns (False, error_reason) so the caller can
    fall back to per-line translation or keep the originals.

    ``source_language`` — if provided and non-English, translation will run
    even when the target is English (needed for dubbing non-English videos).
    """
    s = settings or {}
    lines = [(l or "").strip() for l in (lines or [])]
    if not lines:
        return False, "no lines"
    tgt = (target_language or "").strip()
    src = (source_language or "").strip().lower()
    _is_eng = lambda v: v.lower() in ("english", "en", "en-us", "en-gb")
    if not tgt:
        return False, "empty target language"
    if _is_eng(tgt) and (not src or _is_eng(src)):
        return True, list(lines)  # English source stays as-is.

    api_key = (s.get("gemini_api_key") or "").strip()
    sa_path = (s.get("gemini_service_account") or
               s.get("service_account_path") or "").strip()

    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(lines))
    prompt = (
        f"Translate each of the following numbered lines from "
        f"{'English' if _is_eng(src) else src.title()} into {tgt}. These are "
        f"lines of spoken dialogue for a voiceover dub that must fit the SAME "
        f"time slot as the original speech.\n"
        f"CRITICAL: keep each translation SHORT — it must take no longer to say "
        f"aloud than the original line. {tgt} often runs longer than English, "
        f"so prefer the most concise natural phrasing, drop filler words, and "
        f"never pad or add words. If a literal translation would be longer, "
        f"shorten it while keeping the meaning. Use natural, conversational "
        f"{tgt} that sounds good read aloud.\n"
        f"Return ONLY a JSON array of objects, one per input line, each shaped "
        f'{{"n": <the line number>, "t": "<the {tgt} translation>"}}. Include an '
        f"object for EVERY line number from 1 to {len(lines)} — do not skip, "
        f"merge, or renumber any line. No notes, no transliteration.\n\n"
        f"LINES:\n{numbered}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "responseMimeType": "application/json",
        },
    }

    headers = {"Content-Type": "application/json"}
    is_vertex = False
    if api_key:
        url = f"{API_BASE}/models/{model}:generateContent"
        url = f"{url}?key={api_key}"
    elif sa_path:
        try:
            token = _get_bearer_token(sa_path)
            headers["Authorization"] = f"Bearer {token}"
            _project_id = _get_project_id(sa_path)
            if _project_id:
                url = (f"https://us-central1-aiplatform.googleapis.com"
                       f"/v1/projects/{_project_id}/locations/us-central1"
                       f"/publishers/google/models/{model}:generateContent")
                is_vertex = True
            else:
                return False, "service-account JSON has no project_id"
        except Exception as exc:
            return False, f"auth failed: {exc}"
    else:
        return False, "no gemini_api_key or service account configured"

    # Vertex AI requires role: "user" on content parts
    if is_vertex:
        payload["contents"][0]["role"] = "user"

    ok, data, err = _http_post(url, payload, headers, timeout)
    # Free-tier per-minute quota (HTTP 429) — honor Gemini's "retry in Xs" hint
    # and retry a couple of times, since a single batch request may still land
    # in a throttled window right after a prior run.
    import re as _re
    import time as _time
    _attempts = 0
    while (not ok) and ("429" in str(err)) and _attempts < 3:
        m = _re.search(r'retry in ([\d.]+)s', str(err))
        wait = min(float(m.group(1)) + 1.0, 65.0) if m else 20.0
        _time.sleep(wait)
        _attempts += 1
        ok, data, err = _http_post(url, payload, headers, timeout)
    if not ok:
        return False, err or "translation request failed"
    try:
        parts = data["candidates"][0]["content"]["parts"]
        raw = "".join(p.get("text", "") for p in parts).strip()
    except Exception as exc:
        try:
            _msg = data.get("error", {}).get("message", "")
        except Exception:
            _msg = ""
        return False, _msg or f"could not parse translation: {exc}"

    # Model was asked for a JSON array; parse defensively.
    import json as _json
    out = None
    try:
        out = _json.loads(raw)
    except Exception:
        # Strip a ```json ... ``` fence if the model added one, then retry.
        _r = raw.strip()
        if _r.startswith("```"):
            _r = _r.split("\n", 1)[-1]
            if _r.rstrip().endswith("```"):
                _r = _r.rstrip()[:-3]
        try:
            out = _json.loads(_r)
        except Exception:
            out = None

    if not isinstance(out, list) or not out:
        return False, (
            f"batch translation returned "
            f"{'non-list' if not isinstance(out, list) else 'empty list'}, "
            f"expected {len(lines)} items"
        )

    def _unquote(t: str) -> str:
        t = str(t).strip()
        if len(t) >= 2 and t[0] in "\"'" and t[-1] == t[0]:
            t = t[1:-1].strip()
        return t

    # Two accepted shapes: indexed objects [{"n":1,"t":"…"}, …] (preferred, lets
    # us slot each translation to its line even if the model drops or reorders
    # one) or a bare array of strings (legacy).  Missing lines fall back to the
    # original text rather than nuking the whole batch — one short line the model
    # merged shouldn't cost us all 61 translations.
    cleaned = list(lines)
    matched = 0
    if all(isinstance(o, dict) for o in out):
        for o in out:
            try:
                n = int(o.get("n"))
            except (TypeError, ValueError):
                continue
            if 1 <= n <= len(lines):
                t = _unquote(o.get("t", ""))
                if t:
                    cleaned[n - 1] = t
                    matched += 1
    else:
        for i, item in enumerate(out[:len(lines)]):
            t = _unquote(item)
            if t:
                cleaned[i] = t
                matched += 1

    if matched == 0:
        return False, "batch translation produced no usable items"
    if matched < len(lines):
        # Partial success is fine — kept originals for the gaps.  Signal ok so
        # the caller doesn't burn through the (Vertex-404) fallback model chain.
        return True, cleaned
    return True, cleaned


# ── Cloud TTS API (requires service account) ────────────────────────────
def _call_cloud_tts(
    service_account_path: str,
    text: str,
    voice_name: str,
    timeout: int = 60,
) -> tuple[bool, bytes, str]:
    """Call Cloud Text-to-Speech API with a service account Bearer token.

    POST https://texttospeech.googleapis.com/v1/text:synthesize
    Authorization: Bearer <token>
    """
    try:
        token = _get_bearer_token(service_account_path)
    except Exception as exc:
        return False, b"", f"Service account auth failed: {exc}"

    # Parse voice name → language_code + voice name
    # Format: "en-US-Studio-Q" → lang="en-US", name="en-US-Studio-Q"
    parts = voice_name.split("-")
    lang_code = "-".join(parts[:2]) if len(parts) >= 2 else "en-US"

    payload = {
        "input": {"text": text},
        "voice": {
            "languageCode": lang_code,
            "name": voice_name,
        },
        "audioConfig": {
            "audioEncoding": "LINEAR16",
            "speakingRate": 1.0,
        },
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }

    url = f"{CLOUD_TTS_BASE}/text:synthesize"
    ok, data, err = _http_post(url, payload, headers, timeout)
    if not ok:
        return False, b"", err

    try:
        audio_b64 = data.get("audioContent", "")
        if not audio_b64:
            return False, b"", "No audioContent in Cloud TTS response"
        audio_bytes = base64.b64decode(audio_b64)
        return True, audio_bytes, ""
    except Exception as exc:
        return False, b"", f"Cloud TTS decode failed: {exc}"


# ── Gemini TTS via Cloud text:synthesize (service account) ──────────────
# These are the star-named expressive voices (Achernar, Kore, Puck…) with
# natural-language style/emotion control via the `prompt` field. They run
# through the SAME Cloud endpoint as Studio/Wavenet — using the paid/trial
# project on the service account — so they are NOT subject to the
# Developer-API free-tier `limit: 0` quota.

GEMINI_TTS_MODELS: list[str] = [
    "gemini-2.5-flash-tts",
    "gemini-2.5-pro-tts",
    "gemini-2.5-flash-lite-tts",
    "gemini-3.1-flash-tts-preview",
]


def _call_gemini_cloud_tts(
    service_account_path: str,
    model_id: str,
    text: str,
    voice: str,
    prompt: str = "",
    language_code: str = "en-US",
    speaking_rate: float = 1.0,
    timeout: int = 60,
) -> tuple[bool, bytes, str]:
    """Synthesize expressive Gemini TTS via the Cloud text:synthesize endpoint.

    POST https://texttospeech.googleapis.com/v1/text:synthesize
    {
      "input": {"prompt": "<style>", "text": "<narration>"},
      "voice": {"languageCode": "en-US", "modelName": "<model>", "name": "<voice>"},
      "audioConfig": {"audioEncoding": "LINEAR16", "speakingRate": 1.0}
    }
    """
    try:
        token = _get_bearer_token(service_account_path)
    except Exception as exc:
        return False, b"", f"Service account auth failed: {exc}"

    # Truncate text to stay within the 4000 byte API limit
    if prompt:
        prompt_bytes = len(prompt.encode("utf-8"))
        text_budget = MAX_INPUT_BYTES - prompt_bytes
    else:
        text_budget = MAX_INPUT_BYTES
    if text_budget < 100:
        text = ""
    else:
        text = _truncate_to_byte_limit(text, text_budget)

    input_block: dict[str, Any] = {"text": text}
    if prompt:
        input_block["prompt"] = prompt

    payload = {
        "input": input_block,
        "voice": {
            "languageCode": language_code,
            "modelName": model_id,
            "name": voice,
        },
        "audioConfig": {
            "audioEncoding": "LINEAR16",
            "speakingRate": speaking_rate,
        },
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    # Gemini TTS models require the billing/quota project header
    project_id = _get_project_id(service_account_path)
    if project_id:
        headers["x-goog-user-project"] = project_id

    url = f"{CLOUD_TTS_BASE}/text:synthesize"
    ok, data, err = _http_post(url, payload, headers, timeout)
    if not ok:
        return False, b"", err

    try:
        audio_b64 = data.get("audioContent", "")
        if not audio_b64:
            return False, b"", "No audioContent in Gemini TTS response"
        return True, base64.b64decode(audio_b64), ""
    except Exception as exc:
        return False, b"", f"Gemini TTS decode failed: {exc}"


# ── Gemini TTS via Interactions API ────────────────────────────────────


def _call_gemini_tts(
    api_key: str | None,
    model_id: str,
    text: str,
    voice: str,
    service_account_path: str | None = None,
    timeout: int = 60,
) -> tuple[bool, bytes, str]:
    """Call the Gemini Interactions API for TTS.

    Uses Bearer token if service_account_path provided, else ?key= query param.

    Returns:
        (success, audio_bytes, error_message)
    """
    # Build auth
    headers = {"Content-Type": "application/json"}
    url = f"{API_BASE}/interactions"

    if service_account_path:
        try:
            token = _get_bearer_token(service_account_path)
            headers["Authorization"] = f"Bearer {token}"
        except Exception as exc:
            logger.warning("SA auth failed (falling back to API key): %s", exc)
            # Fall through to API key if available
            if api_key:
                url = f"{API_BASE}/interactions?key={api_key}"
            else:
                return False, b"", f"Service account auth failed: {exc}"
    elif api_key:
        url = f"{API_BASE}/interactions?key={api_key}"
    else:
        return False, b"", "No API key or service account provided"

    payload: dict[str, Any] = {
        "model": model_id,
        "input": text,
        "response_format": {"type": "audio"},
        "generation_config": {
            "speech_config": [{"voice": voice}],
        },
    }

    ok, data, err = _http_post(url, payload, headers, timeout)
    if not ok:
        return False, b"", err

    # Parse response
    try:
        output_audio = data.get("output_audio") or data.get("outputAudio") or {}

        b64_data = (
            output_audio.get("data", "")
            or output_audio.get("audioData", "")
            or data.get("data", "")
        )

        if b64_data:
            try:
                audio_bytes = base64.b64decode(b64_data)
                return True, audio_bytes, ""
            except Exception as exc:
                return False, b"", f"Base64 decode failed: {exc}"

        # Fallback: candidates format
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                inline_data = part.get("inlineData", {})
                if inline_data.get("mimeType", "").startswith("audio/"):
                    b64 = inline_data.get("data", "")
                    if b64:
                        return True, base64.b64decode(b64), ""

        # Error check
        err_block = data.get("error", {})
        if err_block:
            msg = err_block.get("message", json.dumps(err_block))
            return False, b"", msg

        return False, b"", f"No audio in response. Keys: {list(data.keys())}"

    except Exception as exc:
        return False, b"", str(exc)


# ── Public generate_speech ─────────────────────────────────────────────


def generate_speech(
    text: str,
    output_path: str | Path,
    settings: dict | None = None,
) -> tuple[bool, list[dict]]:
    """Generate speech via Gemini API TTS or Cloud TTS.

    Settings:
        gemini_api_key         — API key (AIza... or AQ...)
        service_account_path   — Path to SA JSON key file (optional)
        use_cloud_tts          — True=Cloud TTS, False=Gemini TTS (default)
        gemini_tts_voice       — Gemini TTS voice (default Zephyr)
        gemini_tts_model       — Gemini model (default gemini-2.5-flash-tts)
        gemini_tts_speed       — Speed multiplier (1.0)
        cloud_tts_voice        — Cloud TTS voice (default en-US-Studio-Q)

    Auth priority:
      1. service_account_path — gets OAuth2 token, full access
      2. gemini_api_key       — API key (AIza recommended)

    Returns:
        (success, details)
          success=True  → details is word_timings list
          success=False → details is [error_message_str]
    """
    s = settings or {}
    api_key = (s.get("gemini_api_key") or "").strip()
    sa_path = (s.get("service_account_path") or "").strip()
    use_cloud = s.get("use_cloud_tts", False)

    voice = s.get("gemini_tts_voice", "Achernar") if not use_cloud \
            else s.get("cloud_tts_voice", "en-US-Studio-Q")
    model_id = s.get("gemini_tts_model", "gemini-3.1-flash-tts-preview")
    speed = float(s.get("gemini_tts_speed", 1.0))
    # Natural-language style/emotion instruction for Gemini TTS (prompt field)
    style_prompt = (s.get("gemini_tts_prompt") or "").strip()
    lang_code = (s.get("gemini_tts_language") or "en-US").strip()

    if not api_key and not sa_path:
        msg = "No API key or service account path provided"
        logger.error("Gemini/Cloud TTS: %s", msg)
        return False, [msg]

    # ── Clean text ────────────────────────────────────────────────
    clean_text = re.sub(
        r"[\U0001F300-\U0001F9FF\U0001F600-\U0001F64F"
        r"\U0001F680-\U0001F6FF\U00002600-\U000027BF"
        r"\U0001F1E0-\U0001F1FF]+",
        "",
        text,
    )
    clean_text = clean_text.strip()
    if not clean_text:
        msg = "Empty text after cleaning"
        logger.warning("TTS: %s", msg)
        return False, [msg]

    # ── Speed handling ────────────────────────────────────────────
    # Cloud + Gemini TTS both support audioConfig.speakingRate, so no
    # need to inject [very slow]/[very fast] text hacks. Keep text clean.
    final_text = clean_text

    logger.info(
        "TTS: method=%s voice=%s model=%s chars=%d",
        "cloud_tts" if use_cloud else "gemini_tts",
        voice, model_id if not use_cloud else "-", len(final_text),
    )

    # ── Split into request-sized chunks ───────────────────────────
    # Gemini/Cloud TTS cap input at ~4000 bytes per request. Long scripts
    # must be synthesized in pieces and stitched back together, otherwise
    # only the first ~3800 bytes get voiced and the rest is silently lost.
    if use_cloud:
        # Cloud TTS (Studio/Wavenet/Neural2) allows up to 5000 bytes.
        chunk_limit = 4800
    elif style_prompt:
        chunk_limit = MAX_INPUT_BYTES - len(style_prompt.encode("utf-8")) - 50
    else:
        chunk_limit = MAX_INPUT_BYTES
    chunk_limit = max(chunk_limit, 200)

    chunks = _split_text_to_chunks(final_text, chunk_limit)
    if not chunks:
        return False, ["No text to synthesize after chunking"]
    logger.info("TTS: %d chunk(s) to synthesize (limit=%d bytes)", len(chunks), chunk_limit)

    # ── Call API per chunk, with retry ────────────────────────────
    pcm_parts: list[bytes] = []
    out_rate, out_channels, out_width = 24000, 1, 2

    for chunk_idx, chunk_text in enumerate(chunks):
        last_error = ""
        audio_bytes = None

        for attempt in range(3):
            if attempt > 0:
                wait = 10 * (2 ** attempt)
                logger.info("Retry %d/3 after %ds...", attempt + 1, wait)
                time.sleep(wait)

            if use_cloud:
                ok, ab, err = _call_cloud_tts(sa_path, chunk_text, voice, timeout=TTS_TIMEOUT)
            elif sa_path:
                # Preferred: expressive Gemini voices via Cloud endpoint (service account)
                ok, ab, err = _call_gemini_cloud_tts(
                    sa_path, model_id, chunk_text, voice,
                    prompt=style_prompt, language_code=lang_code,
                    speaking_rate=speed, timeout=TTS_TIMEOUT,
                )
            else:
                # Fallback: Developer-API key path (limited free quota)
                ok, ab, err = _call_gemini_tts(
                    api_key, model_id, chunk_text, voice,
                    service_account_path=None,
                    timeout=TTS_TIMEOUT,
                )

            if ok:
                audio_bytes = ab
                break
            last_error = err
            logger.warning(
                "TTS chunk %d/%d attempt %d/3 failed: %s",
                chunk_idx + 1, len(chunks), attempt + 1, err,
            )
            if "429" in err:
                logger.info("Rate limited (429) — waiting extra 20s before retry")
                time.sleep(20)

        if audio_bytes is None:
            logger.error(
                "TTS failed on chunk %d/%d after 3 retries: %s",
                chunk_idx + 1, len(chunks), last_error,
            )
            return False, [last_error]

        pcm, rate, channels, width = _extract_pcm(audio_bytes)
        if chunk_idx == 0:
            out_rate, out_channels, out_width = rate, channels, width
        pcm_parts.append(pcm)

    # ── Save concatenated PCM as one WAV ──────────────────────────
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    combined_pcm = b"".join(pcm_parts)
    import wave as _wave
    with _wave.open(str(out), "wb") as wf:
        wf.setnchannels(out_channels)
        wf.setsampwidth(out_width)
        wf.setframerate(out_rate)
        wf.writeframes(combined_pcm)

    bytes_per_frame = out_width * out_channels
    duration = (len(combined_pcm) / (out_rate * bytes_per_frame)
                if out_rate and bytes_per_frame else 0.0)

    logger.info("TTS: saved %s (%.1fs, %d bytes)", out.name, duration, out.stat().st_size)

    # ── Approximate word timings ──────────────────────────────────
    word_timings: list[dict] = []
    if duration > 0:
        words = clean_text.split()
        wc = len(words)
        if wc > 0:
            t_per_word = duration / wc
            word_timings = [
                {"word": w, "start": i * t_per_word, "end": (i + 1) * t_per_word}
                for i, w in enumerate(words)
            ]

    return True, word_timings
