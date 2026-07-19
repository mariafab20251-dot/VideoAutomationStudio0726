# Multi-Speaker Dubbing — Implementation Plan (WhisperX + Diarization)

> **For the implementing model:** Read this whole file first. Work **one phase at a time**,
> in order. After each phase, run the stated verification and **stop for the user to test**
> before moving on. Do **not** commit or push — the user does that manually after testing.
> The user's workflow rule: *always ask before commit/push, and ask the user to verify each
> feature works before moving on.*

---

## 0. Goal (what we're building)

Right now the Dubbing tab re-voices a whole video with **one** TTS voice. This feature makes it
use **a different voice per speaker** — e.g. a video with a man, a woman, and a child gets three
distinct dubbed voices, each mapped by the user.

The "who is speaking" decision uses **real audio diarization via WhisperX** (clusters voices by
acoustic fingerprint), NOT text guessing. This is the accurate, stable approach.

### Pipeline change
```
BEFORE:  faster-whisper transcribe → group lines → translate → TTS (1 voice) → mux
AFTER:   WhisperX transcribe+align+diarize → words tagged with speaker
         → group lines (split on speaker change, carry speaker)
         → translate → map each speaker → a voice
         → TTS per-line with that speaker's voice → overlay/mux
```

### Non-negotiable design rule: graceful fallback
If WhisperX / the HF token / the pyannote license / the models are missing or fail, dubbing
**must fall back to today's single-voice behavior** and still produce a dubbed video. The feature
is additive; it must never break the existing working dubbing path.

---

## 1. Current code map (verified — read these before editing)

Repo root: `d:\GitHub\ChangeGUI\`

| File | What's there now | Line anchors |
|---|---|---|
| `_whisper_word_timestamps.py` | Standalone transcription script. Prefers **faster-whisper** (local models in `models/whisper/faster-whisper-<size>/`), falls back to openai-whisper. Emits JSON `[{"word","offset","duration"}, ...]` to stdout. | `extract_word_timestamps()` ~L161; `_extract_faster_whisper()` ~L51; `_resolve_local_model()` ~L28; `main()` ~L211 |
| `dubbing_engine.py` | The Tk-free pipeline. `transcribe_video()` calls the script above as a subprocess. `group_words_into_segments()` groups words into `{'start','end','text'}`. `build_dubbed_audio()` does translate → TTS loop → overlay. | `transcribe_video()` ~L186; `group_words_into_segments()` ~L94; translate block ~L336; **TTS loop ~L431 (call site ~L438)** |
| `dubbing_tab.py` | The Tk UI (mixin). Builds cards (video, source lang, target lang, options), Run button, log. Worker calls `dubbing_engine.dub_video()`. | `_dub_card()` helper; card build ~L90+; worker `_dub_worker()` ~L388 |
| `gemini_api_tts_helper.py` | `generate_speech()` (TTS) reads `settings['gemini_tts_voice']`. 30 Gemini voices in `GEMINI_TTS_VOICES` (~L140-171). `get_all_voices()` / `get_voice_keys()` return them. `translate_lines()` / `translate_text()` for translation. | `GEMINI_TTS_VOICES` ~L140; `get_all_voices()` ~L208 |
| `youtube_video_automation_enhanced.py` | `TTSGenerator.generate_voiceover(text, out, settings)` — dispatches by `settings['tts_engine']`. For dubbing it's `'google_cloud'` → Gemini TTS. Voice comes from `settings['gemini_tts_voice']`. | `generate_voiceover()` ~L5883; `_generate_google_cloud_voiceover()` ~L5821 |

### Segment data shape (important)
`group_words_into_segments()` returns a list of dicts:
```python
{'start': float, 'end': float, 'text': str}
```
Later `build_dubbed_audio()` adds `seg['xlated']` (translated text). **We will add `seg['speaker']`.**

### The TTS call site (the surgical point for per-voice)
In `build_dubbed_audio()`, ~line 438:
```python
ok, _ = TTSGenerator.generate_voiceover(txt, seg_mp3, settings)
```
`settings['gemini_tts_voice']` decides the voice. To get per-speaker voices we pass a **per-segment
copy of settings** with `gemini_tts_voice` overridden. Nothing downstream changes.

### The 30 Gemini voices (for the UI dropdown + auto-assign)
Keys from `GEMINI_TTS_VOICES` with tone labels. Useful for auto-suggesting by gender/age:
- Firm/male-leaning: `Alnilam` (Firm), `Algenib` (Gravelly), `Schedar` (Even), `Rasalgethi` (Informative)
- Soft/female-leaning: `Achernar` (Soft), `Vindemiatrix` (Gentle), `Despina` (Smooth), `Sulafat` (Warm)
- Bright/young-leaning: `Autonoe` (Bright), `Laomedeia` (Upbeat), `Sadachbia` (Lively)
- (Full list is authoritative in the code — read `GEMINI_TTS_VOICES` at runtime; don't hardcode a stale copy.)

---

## 2. One-time setup the USER must do (blocker for diarization)

Diarization models are **gated** on Hugging Face. Before Phase 1 diarization can run, the user must:

1. Create a free HF account: https://huggingface.co/join
2. Generate a **read** token: https://huggingface.co/settings/tokens → "New token" → type **Read**.
3. Accept the license on **both** gated model pages (must be logged in, click "Agree"):
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-3.1
4. The token gets stored in settings as `hf_token` (Phase 3 adds a field; until then it can be
   set manually in `overlay_settings.json` or via env var `HF_TOKEN`).

**If the user hasn't done this, diarization returns HTTP 401/403 → we fall back to single-voice.**

### Downloads (unstable internet → prefer manual)
- `pip install whisperx` pulls the code + `pyannote.audio`, `faster-whisper` (already have), `ctranslate2`, `silero-vad`. See §6 for the install command.
- The pyannote **model weights** (~1 GB total) are gated, so a plain browser/IDM URL returns **401** —
  they can't be grabbed anonymously. Two options for the model files:
  - **Preferred:** let `huggingface_hub` download them on first diarization run with the token set
    (`HF_TOKEN`). It **resumes** partial downloads, so unstable internet is survivable.
  - **IDM with auth header:** IDM → Add URL → add HTTP header `Authorization: Bearer hf_xxx`.
    Files:
    - `https://huggingface.co/pyannote/segmentation-3.0/resolve/main/pytorch_model.bin`
    - `https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/resolve/main/pytorch_model.bin` (ungated, direct)
    - plus the small `config.yaml` from `speaker-diarization-3.1` and `segmentation-3.0`.
  - Files land in `C:\Users\shahi\.cache\huggingface\hub\`. Once cached, works offline.

> **Implementing model:** do NOT try to auto-download these in code without a token. Detect absence,
> log a clear message telling the user to do §2, and fall back.

---

## 3. Phases

### Phase 1 — WhisperX transcription + diarization (`_whisper_word_timestamps.py`)
**Goal:** add an optional diarized transcription mode that emits the same JSON word format **plus**
a `"speaker"` field on each word.

**Do:**
1. Add CLI flags: `--diarize` (bool), `--hf-token <token>`, `--min-speakers`, `--max-speakers` (optional ints).
2. New function `_extract_whisperx(audio_path, model_size, language, hf_token, min_spk, max_spk)`:
   - `import whisperx` (ImportError → raise so caller falls back).
   - Load ASR: `model = whisperx.load_model(model_size, device, compute_type=...)`
     (reuse device/precision logic already in `_extract_faster_whisper`: cuda/float16 else cpu/int8).
   - `result = model.transcribe(audio, language=language)`
   - Align: `model_a, meta = whisperx.load_align_model(language_code=result["language"], device=device)`
     then `result = whisperx.align(result["segments"], model_a, meta, audio, device)`
   - Diarize: `dia = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_token, device=device)`
     then `diar = dia(audio, min_speakers=min_spk, max_speakers=max_spk)`
     then `result = whisperx.assign_word_speakers(diar, result)`
   - Flatten to the standard word list, adding `"speaker": word.get("speaker", "SPEAKER_00")`.
     Keep `word/offset/duration` exactly as the existing format so nothing else breaks.
   - **Note:** whisperx API has shifted across versions. If `whisperx.diarize.DiarizationPipeline`
     isn't found, try `from whisperx import DiarizationPipeline`. Pin version per §6 to avoid this.
3. In `extract_word_timestamps(...)`, add params `diarize=False, hf_token=None, min_spk=None, max_spk=None`.
   - If `diarize` and `hf_token`: `try: return _extract_whisperx(...)` — on **any** exception, log the
     reason and fall through to the existing faster-whisper chain (words with no speaker).
   - If not diarizing, behavior is unchanged.
4. In `main()`, wire the new flags; when diarize path is used, each JSON word includes `"speaker"`.

**Verify:**
- `python _whisper_word_timestamps.py <clip.wav> --model medium --language en --diarize --hf-token hf_xxx`
  prints JSON where words carry a `"speaker"` field (e.g. `SPEAKER_00`, `SPEAKER_01`).
- Without `--diarize`, output is byte-for-byte the old format (no `speaker` key) — regression check.
- With a bad/empty token, it logs a clear error to stderr and still returns words (fallback).

**STOP — user tests Phase 1 on a real multi-speaker clip before Phase 2.**

---

### Phase 2 — Carry speaker through the pipeline (`dubbing_engine.py`)
**Goal:** transcription → segments now preserve which speaker said each line.

**Do:**
1. `transcribe_video()` (~L186): add optional params `diarize=False, hf_token=None,
   min_spk=None, max_spk=None`. When diarizing, append the new flags to the subprocess `cmd`
   that runs `_whisper_word_timestamps.py`. Parse JSON as today (words may now have `"speaker"`).
2. `group_words_into_segments()` (~L94): add a rule — **start a new segment when the speaker
   changes** (in addition to the existing gap / sentence-end / max-words rules). Attach the
   segment's speaker: `seg['speaker'] = <majority or first speaker of the words in it>`.
   If words have no speaker key, default `'SPEAKER_00'` (keeps single-voice behavior).
3. Add helper `distinct_speakers(segments) -> list[str]` returning sorted unique speaker labels.

**Verify:**
- Feed a diarized word list (from Phase 1) → segments each have a `speaker`, and a new segment
  begins whenever the speaker changes.
- Feed a non-diarized word list → every segment is `SPEAKER_00`, grouping identical to before.

**STOP — user confirms segments split correctly by speaker.**

---

### Phase 3 — Voice-mapping UI (`dubbing_tab.py`)
**Goal:** user picks which Gemini voice each detected speaker gets.

**Do:**
1. Add settings fields + a UI card **"🎭 Speaker Voices"** below the Target Language card:
   - A checkbox **"Multi-speaker dubbing (detect & assign voices)"** → `settings['dub_multispeaker']`.
   - An **"HF Token"** entry (password-style) → `settings['hf_token']` (needed for diarization).
     Add a small hint linking to §2 setup.
   - A **"Detect Speakers"** button. On click (in a thread, like `_dub_worker`):
     - Requires a chosen video. Runs `transcribe_video(video, settings, diarize=True,
       hf_token=...)` then `group_words_into_segments` + `distinct_speakers`.
     - Populates a row per speaker: label (`SPEAKER_00`) + a `ttk.Combobox` of Gemini voice keys
       (from `gemini_api_tts_helper.get_voice_keys()`), pre-filled with an auto-suggested voice.
   - Persist the mapping to `settings['dub_speaker_voices']` = `{"SPEAKER_00": "Alnilam", ...}`
     whenever a dropdown changes.
2. Auto-suggest: cycle distinct voices so no two speakers default to the same one.
   (Optional nicety: a "▶" button per speaker to TTS a 2-sec sample in that voice.)
3. Layout note: the tab already uses a scrollable canvas — add the card into `scrollable`,
   consistent with the existing cards. Keep the log area visibility intact.

**Verify:**
- Enabling the checkbox + Detect Speakers on a 3-speaker clip lists 3 rows with distinct default voices.
- Changing a dropdown persists to `overlay_settings.json` under `dub_speaker_voices`.
- With the checkbox off, the tab behaves exactly as today.

**STOP — user confirms the UI detects speakers and saves the mapping.**

---

### Phase 4 — Per-line voice in the TTS loop (`dubbing_engine.py`)
**Goal:** actually voice each line with its speaker's mapped voice.

**Do:**
1. In `build_dubbed_audio()`, plumb the same new params (`diarize`, `hf_token`, min/max) into the
   `transcribe_video(...)` call so segments carry speakers when multi-speaker is on.
2. Read `voice_map = settings.get('dub_speaker_voices') or {}` and
   `multi = settings.get('dub_multispeaker', False)`.
3. In the TTS loop (~L431), before the `generate_voiceover` call (~L438):
   ```python
   if multi and voice_map:
       spk = seg.get('speaker', 'SPEAKER_00')
       voice = voice_map.get(spk)
       if voice:
           seg_settings = dict(settings)          # shallow copy — don't mutate shared settings
           seg_settings['gemini_tts_voice'] = voice
       else:
           seg_settings = settings
   else:
       seg_settings = settings
   ok, _ = TTSGenerator.generate_voiceover(txt, seg_mp3, seg_settings)
   ```
4. `dub_video()` / `_dub_worker` (`dubbing_tab.py` ~L388): pass `diarize=multi`,
   `hf_token=settings['hf_token']` down so the full render (not just Detect) diarizes.

**Verify:**
- Full dub of a 3-speaker clip → each speaker audibly uses its mapped voice; timing stays synced
  to the original (the existing anchor/stretch logic is untouched).
- Turn multi-speaker off → identical to today's single-voice dub (regression check).

**STOP — user does a full end-to-end multi-speaker dub and confirms.**

---

### Phase 5 — Polish, deps, docs
**Do:**
1. Create `setup/requirements_diarize.txt` (see §6) — keep whisperx OUT of `requirements_core.txt`
   so the core install stays lean.
2. Add a `setup/install_diarize.bat` mirroring the other installers (skip-aware, quiet).
3. Update `todo_multi_speaker_dubbing.md` / this file's status to "done".
4. Confirm `models/whisper/` stays gitignored; whisperx caches pyannote under
   `~/.cache/huggingface` (already outside the repo).

**Verify:** fresh `pip install -r setup/requirements_diarize.txt` in a clean venv resolves.

---

## 4. Files to touch (summary)
- `_whisper_word_timestamps.py` — Phase 1 (diarized mode + speaker field)
- `dubbing_engine.py` — Phase 2 (carry speaker), Phase 4 (per-voice TTS), plumb params
- `dubbing_tab.py` — Phase 3 (UI card + Detect Speakers), Phase 4 (pass params to worker)
- `gemini_api_tts_helper.py` — read-only (use `get_voice_keys()` / `GEMINI_TTS_VOICES`)
- `setup/requirements_diarize.txt`, `setup/install_diarize.bat` — Phase 5 (new)

---

## 5. Settings keys introduced
| Key | Type | Meaning |
|---|---|---|
| `dub_multispeaker` | bool | Master toggle for the feature |
| `hf_token` | str | Hugging Face read token (for pyannote) |
| `dub_speaker_voices` | dict[str,str] | `{"SPEAKER_00": "Alnilam", ...}` speaker→Gemini voice |
| `dub_min_speakers` / `dub_max_speakers` | int (optional) | Hints to diarizer; blank = auto |

---

## 6. Install (Phase 5 content)

`setup/requirements_diarize.txt`:
```
# Multi-speaker dubbing (speaker diarization). Optional — only needed for the
# "Multi-speaker dubbing" toggle in the Dubbing tab. Requires a Hugging Face
# token + accepting the pyannote license (see MULTISPEAKER_DUBBING_PLAN.md §2).
whisperx>=3.1.1
# whisperx pulls pyannote.audio, faster-whisper, ctranslate2, silero-vad.
```
Install command (documented in the .bat):
```
pip install -r setup/requirements_diarize.txt
```
> Version note: whisperx's diarization import path has changed between releases.
> If `whisperx.diarize.DiarizationPipeline` is missing at runtime, try
> `from whisperx import DiarizationPipeline`. Pin a known-good version if needed.

---

## 7. Guardrails / gotchas for the implementer
- **Never mutate the shared `settings` dict** inside the TTS loop — copy per segment (Phase 4).
- **Always fall back** to single-voice on any diarization failure; log *why*.
- Keep the **JSON word format stable** — only ADD a `speaker` key; don't rename existing keys,
  or `group_words_into_segments` and the whole pipeline break.
- The faster-whisper local models (`models/whisper/faster-whisper-medium|base/`) are already
  installed and gitignored — reuse them; whisperx can load the same sizes.
- Diarization runs on the **original audio**, before translation. TTS/translation still use Gemini.
- Test the **single-speaker regression** at the end of every phase — the existing dubbing path
  must keep working unchanged.
- Do **not** commit/push. After each phase, stop and ask the user to test.

## 8. Progress log (implementer: update as you go)
- [ ] Phase 1 — WhisperX diarized transcription
- [ ] Phase 2 — carry speaker through segments
- [ ] Phase 3 — voice-mapping UI
- [ ] Phase 4 — per-line voice in TTS loop
- [ ] Phase 5 — deps, installer, docs
