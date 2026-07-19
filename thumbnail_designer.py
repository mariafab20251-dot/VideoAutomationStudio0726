#!/usr/bin/env python
"""
Thumbnail Designer — Gemini image generation (Nano Banana / Nano Banana Pro)
============================================================================
Standalone, Tk-free helper that turns a video frame + a text prompt into a
finished YouTube/Shorts thumbnail using Google's Gemini image models.

Auth is shared with the TTS pipeline: it reuses ``_get_bearer_token`` from
``gemini_api_tts_helper`` so a service-account JSON grants image access with
no extra key.  Falls back to an ``AIza…`` API key if that's all that's set.

Model names (community codename → real id):
    Nano Banana        → gemini-2.5-flash-image        (cheap, fast)
    Nano Banana Pro    → gemini-3-pro-image-preview     (crisper text, pricier)

Public API
----------
    extract_frame(src_video, timestamp, out_img, log=None) -> Path | None
    generate_thumbnail(prompt, out_img, model=..., frame_path=None,
                       aspect='16:9', settings=None, log=None) -> Path | None
"""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path

# Reuse the exact same endpoint + auth the TTS helper already uses.
try:
    from gemini_api_tts_helper import (
        API_BASE, _get_bearer_token,
    )
except Exception:  # pragma: no cover — allow standalone import
    API_BASE = "https://generativelanguage.googleapis.com/v1beta"
    _get_bearer_token = None


# Friendly label → model id.  Order matters for the UI dropdown.
MODELS: dict[str, str] = {
    "Nano Banana (2.5 Flash Image)": "gemini-2.5-flash-image",
    "Nano Banana Pro (3 Pro Image)": "gemini-3-pro-image-preview",
}
DEFAULT_MODEL = "gemini-2.5-flash-image"


def _noop_log(level, msg):
    print(f"[{level.upper()}] {msg}")


# ── Frame extraction ────────────────────────────────────────────────────


def extract_frame(src_video, timestamp, out_img, log=None):
    """Grab a single high-quality JPEG frame from *src_video* at *timestamp*.

    *timestamp* is seconds (float) or an ffmpeg time string ('00:01:23').
    Returns the output Path, or ``None`` on failure.
    """
    log = log or _noop_log
    src_video = Path(src_video)
    out_img = Path(out_img)
    out_img.parent.mkdir(parents=True, exist_ok=True)
    if not src_video.is_file():
        log('error', f'Thumbnail: source not found: {src_video}')
        return None
    ts = timestamp if isinstance(timestamp, str) else f'{float(timestamp):.3f}'
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-loglevel', 'error',
             '-ss', ts, '-i', str(src_video),
             '-frames:v', '1', '-q:v', '2', str(out_img)],
            check=True, capture_output=True, timeout=60)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        _e = getattr(e, 'stderr', b'') or b''
        log('error', f'Thumbnail: frame grab failed — '
                     f'{_e.decode("utf-8", "replace")[-200:]}')
        return None
    if not out_img.is_file() or out_img.stat().st_size == 0:
        log('error', 'Thumbnail: frame grab produced no image')
        return None
    log('ok', f'Thumbnail: grabbed frame @ {ts}s → {out_img.name}')
    return out_img


# ── Gemini image generation ─────────────────────────────────────────────


def _auth_url_headers(model, settings, log):
    """Build the request URL + headers, mirroring the TTS auth priority.

    Service accounts work with the Vertex AI endpoint
    (us-central1-aiplatform.googleapis.com), not the free Gemini endpoint.
    API keys work with the free Gemini endpoint
    (generativelanguage.googleapis.com).

    Returns (url, headers, is_vertex_ai, error)  — is_vertex_ai is True when
    the URL targets the Vertex AI endpoint (caller must add ``role: "user"``
    to content parts).
    """
    s = settings or {}
    sa_path = (s.get('gemini_service_account')
               or s.get('service_account_path') or '').strip()
    api_key = (s.get('gemini_api_key') or '').strip()
    headers = {'Content-Type': 'application/json'}

    # Image models (gemini-2.5-flash-image, gemini-3-pro-image-preview) DO
    # work on Vertex AI, and that is where paid/trial credit lives — the free
    # API-key endpoint has a tiny image quota that 429s almost immediately.
    # So we PREFER the service account (Vertex AI) for image generation.
    _is_image_model = ('-image' in model)
    # gemini-3-pro-image-preview is only served from the ``global`` location;
    # gemini-2.5-flash-image works on both us-central1 and global, so we use
    # ``global`` for all image models to keep things simple and correct.
    _image_region = 'global'

    # Priority 1: Service account → Vertex AI endpoint (uses trial/paid credit)
    if sa_path and _get_bearer_token is not None:
        try:
            token = _get_bearer_token(sa_path)
            # Read project_id from the service account JSON
            _project_id = ''
            try:
                import json as _json
                _project_id = _json.loads(
                    Path(sa_path).read_text(encoding='utf-8')
                ).get('project_id', '')
            except Exception:
                pass
            if _project_id:
                if _is_image_model:
                    _region = _image_region
                else:
                    _region = s.get('gcp_region', 'us-central1').strip() or 'us-central1'
                # The ``global`` location uses the plain aiplatform host,
                # regional locations use the ``<region>-aiplatform`` host.
                _host = ('aiplatform.googleapis.com' if _region == 'global'
                         else f'{_region}-aiplatform.googleapis.com')
                url = (
                    f'https://{_host}'
                    f'/v1/projects/{_project_id}/locations/{_region}'
                    f'/publishers/google/models/{model}:generateContent'
                )
            else:
                log('warn', 'Thumbnail: service-account JSON has no project_id; '
                            'falling back to API key')
                raise ValueError('no project_id')
            headers['Authorization'] = f'Bearer {token}'
            return url, headers, True, None
        except Exception as e:
            log('warn', f'Thumbnail: Vertex AI service-account failed ({e}); '
                        f'falling back to API key')

    # Priority 2: API key → free Gemini endpoint (limited image quota)
    if api_key:
        url = f'{API_BASE}/models/{model}:generateContent'
        return f'{url}?key={api_key}', headers, False, None
    return None, None, False, 'no service account or API key configured'


def _extract_image_bytes(resp_json):
    """Pull the first inline image out of a generateContent response."""
    for cand in (resp_json.get('candidates') or []):
        parts = (cand.get('content') or {}).get('parts') or []
        for p in parts:
            inline = p.get('inlineData') or p.get('inline_data')
            if inline and inline.get('data'):
                return base64.b64decode(inline['data'])
    return None


def _text_reason(resp_json):
    """Best-effort human reason when no image came back (safety block, etc.)."""
    cands = resp_json.get('candidates') or []
    if cands:
        fr = cands[0].get('finishReason') or cands[0].get('finish_reason')
        if fr and fr not in ('STOP', 'MAX_TOKENS'):
            return f'finishReason={fr}'
        for p in (cands[0].get('content') or {}).get('parts') or []:
            if p.get('text'):
                return p['text'][:200]
    pf = resp_json.get('promptFeedback') or {}
    if pf.get('blockReason'):
        return f'blocked: {pf["blockReason"]}'
    return 'no image in response'


def generate_thumbnail(prompt, out_img, model=DEFAULT_MODEL, frame_path=None,
                       aspect='16:9', settings=None, log=None, timeout=180):
    """Generate a thumbnail image from *prompt* (+ optional reference frame).

    Returns the saved Path, or ``None`` on failure.
    """
    import requests

    log = log or _noop_log
    settings = settings or {}
    out_img = Path(out_img)
    out_img.parent.mkdir(parents=True, exist_ok=True)

    url, headers, is_vertex, err = _auth_url_headers(model, settings, log)
    if err:
        log('error', f'Thumbnail: {err}')
        return None

    # Vertex AI serves the image models under their FULL name (verified:
    # gemini-2.5-flash-image and gemini-3-pro-image-preview both return
    # images on Vertex AI).  Do NOT strip the ``-image`` suffix — doing so
    # turns the request into a text-only model that cannot output images.
    _model = model

    parts = [{'text': prompt}]
    if frame_path:
        fp = Path(frame_path)
        if fp.is_file():
            mime = 'image/png' if fp.suffix.lower() == '.png' else 'image/jpeg'
            parts.append({'inline_data': {
                'mime_type': mime,
                'data': base64.b64encode(fp.read_bytes()).decode('ascii'),
            }})
        else:
            log('warn', f'Thumbnail: reference frame missing ({fp}); '
                        f'generating from prompt only')

    # Vertex AI requires ``role: "user"`` on the content; the free API
    # doesn't care (and may reject it on some models).
    _content = {'parts': parts}
    if is_vertex:
        _content['role'] = 'user'
    payload = {
        'contents': [_content],
        'generationConfig': {
            'responseModalities': ['IMAGE'],
            'imageConfig': {'aspectRatio': aspect},
        },
    }

    log('info', f'Thumbnail: calling {_model} (aspect {aspect})…')
    try:
        resp = requests.post(url, headers=headers, json=payload,
                             timeout=timeout)
    except Exception as e:
        log('error', f'Thumbnail: request failed — {e}')
        return None

    # Some model revisions reject imageConfig/aspectRatio — retry once bare.
    if resp.status_code == 400 and 'imageConfig' in resp.text:
        log('warn', 'Thumbnail: model rejected imageConfig — retrying without '
                    'aspect ratio')
        payload['generationConfig'].pop('imageConfig', None)
        try:
            resp = requests.post(url, headers=headers, json=payload,
                                 timeout=timeout)
        except Exception as e:
            log('error', f'Thumbnail: retry failed — {e}')
            return None

    # If Vertex AI returned 403 (insufficient scopes / API not enabled) and
    # we have an API key, retry with the free Gemini endpoint.
    if resp.status_code == 403 and 'ACCESS_TOKEN_SCOPE' in resp.text:
        _api_key = (settings or {}).get('gemini_api_key', '') or ''
        if _api_key:
            log('warn', 'Thumbnail: Vertex AI returned 403 — retrying with '
                        'API key on free endpoint')
            _new_url = (f'{API_BASE}/models/{model}:generateContent'
                        f'?key={_api_key}')
            payload['generationConfig'].pop('imageConfig', None)
            try:
                resp = requests.post(_new_url, json=payload, timeout=timeout)
            except Exception as e:
                log('error', f'Thumbnail: free-endpoint retry failed — {e}')
                return None
        else:
            log('error', 'Thumbnail: Vertex AI failed (403) and no API key '
                         'configured. Enable Vertex AI API in your GCP project '
                         'or add a Gemini API key to settings.')
            return None

    # Vertex AI 404 — model not found; retry with the original (-image) name
    # if it was stripped, else fall back to API key.
    if resp.status_code == 404 and is_vertex:
        _alt_model = model  # original model name (with -image suffix)
        if _alt_model != _model:
            log('warn', f'Thumbnail: Vertex AI model {_model} not found — '
                        f'retrying as {_alt_model}')
            _alt_url = url.replace(f'models/{_model}:', f'models/{_alt_model}:')
            try:
                resp = requests.post(_alt_url, headers=headers, json=payload,
                                     timeout=timeout)
            except Exception as e:
                log('error', f'Thumbnail: Vertex AI retry failed — {e}')
                return None
        else:
            _api_key = (settings or {}).get('gemini_api_key', '') or ''
            if _api_key:
                log('warn', 'Thumbnail: Vertex AI 404 — retrying with API key')
                _new_url = (f'{API_BASE}/models/{model}:generateContent'
                            f'?key={_api_key}')
                try:
                    resp = requests.post(_new_url, json=payload, timeout=timeout)
                except Exception as e:
                    log('error', f'Thumbnail: free-endpoint retry failed — {e}')
                    return None
            else:
                log('error', 'Thumbnail: Vertex AI model not found and no API '
                             'key configured. Try a different model or add a '
                             'Gemini API key.')
                return None

    if resp.status_code != 200:
        log('error', f'Thumbnail: HTTP {resp.status_code} — '
                     f'{resp.text[:300]}')
        return None

    try:
        data = resp.json()
    except Exception as e:
        log('error', f'Thumbnail: bad JSON response — {e}')
        return None

    img = _extract_image_bytes(data)
    if not img:
        log('error', f'Thumbnail: {_text_reason(data)}')
        return None

    out_img.write_bytes(img)
    log('ok', f'Thumbnail: ✅ saved → {out_img.name} ({len(img):,} bytes)')
    return out_img


# ── CLI smoke test ──────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    import json as _json

    ap = argparse.ArgumentParser(description='Generate a thumbnail via Gemini')
    ap.add_argument('prompt')
    ap.add_argument('-o', '--out', default='thumbnail.png')
    ap.add_argument('-m', '--model', default=DEFAULT_MODEL)
    ap.add_argument('-f', '--frame', default=None,
                    help='reference image, or a video to grab a frame from')
    ap.add_argument('-t', '--time', default='2', help='frame timestamp (s)')
    ap.add_argument('-a', '--aspect', default='16:9')
    ap.add_argument('--settings', default=None,
                    help='JSON file with gemini_service_account / gemini_api_key')
    a = ap.parse_args()

    s = {}
    if a.settings and Path(a.settings).is_file():
        s = _json.loads(Path(a.settings).read_text(encoding='utf-8'))

    ref = a.frame
    if ref and Path(ref).suffix.lower() in ('.mp4', '.mov', '.mkv', '.webm'):
        ref = extract_frame(ref, a.time, 'thumb_frame.jpg') or None

    generate_thumbnail(a.prompt, a.out, model=a.model, frame_path=ref,
                       aspect=a.aspect, settings=s)
