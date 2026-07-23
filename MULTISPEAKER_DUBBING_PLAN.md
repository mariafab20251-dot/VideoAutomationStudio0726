# Multi-Speaker Dubbing — Implementation Plan (Diarization)

> **For the implementing model:** Read this whole file first. Work **one phase at a time**,
> in order. After each phase, run the stated verification and **stop for the user to test**
> before moving on. Do **not** commit or push — the user does that manually after testing.
> The user's workflow rule: *always ask before commit/push, and ask the user to verify each
> feature works before moving on.*

> **⚠️ Build note (architecture change from the original plan):** This plan was written around
> **WhisperX** as the transcribe+align+diarize wrapper. In practice WhisperX could not be used on
> this machine: whisperx pins `ctranslate2<4.5.0`, which loads **cuDNN 8**, but the working GPU
> stack (torch 2.5.1, faster-whisper 1.2.1, ctranslate2 4.8.0) runs on **cuDNN 9**. Installing
> whisperx broke GPU ASR (`Could not locate cudnn_ops_infer64_8.dll`) and its pyannote 4.x
> requirement clashed with the bundled pyannote 3.x models. **Phase 1 was therefore implemented
> without whisperx**: word timestamps come from **faster-whisper** directly, speaker turns from
> **pyannote.audio** directly, and words are assigned to speakers by **temporal overlap** in our
> own code (`_extract_diarized` / `_diarize_turns` / `_assign_speakers`). Diarization runs *before*
> ASR so torch's cuDNN 9 loads before ctranslate2's copy (otherwise torch cuDNN symbol lookups
> fail). The external behavior (per-word `speaker` labels, graceful fallback) is unchanged.

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

## 2. One-time setup — ✅ ALREADY DONE (do NOT redo)

All of the user-side setup below is **complete as of 2026-07-19**. The implementing model does
**not** need to ask the user to do any of this — it's finished and verified.

- ✅ Free HF account created.
- ✅ **Read** token generated and stored in `overlay_settings.json` under key `hf_token`
  (gitignored — never commit it, never echo its value).
- ✅ Licenses **accepted** on both gated pages (verified: both return HTTP 200 with the token):
  - https://huggingface.co/pyannote/segmentation-3.0
  - https://huggingface.co/pyannote/speaker-diarization-3.1
- ✅ All diarization model files **downloaded and staged locally** (see below).

### Models are BUNDLED LOCALLY — do NOT download, do NOT use the HF cache

The diarization models (**~32 MB total**, not 1 GB — earlier estimate was wrong) are already
staged in the repo under `models/pyannote/` (gitignored, travels with the portable folder like
`models/whisper/`):

```
models/pyannote/segmentation-3.0/config.yaml
models/pyannote/segmentation-3.0/pytorch_model.bin          (5.9 MB)
models/pyannote/speaker-diarization-3.1/config.yaml         (pipeline config; no weights)
models/pyannote/wespeaker-voxceleb-resnet34-LM/config.yaml
models/pyannote/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin   (26.6 MB)
```

**CRITICAL implementation detail — load pyannote from these LOCAL paths, not by model-ID.**
pyannote's default `Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=...)`
downloads from HF by ID into the user cache. We do NOT want that (portability + offline). Instead:

1. Locate the bundled dir: `<repo>/models/pyannote/` (resolve relative to the script file).
2. If it exists, load from the **local** `speaker-diarization-3.1/config.yaml`, and REWRITE that
   config's `embedding:` and `segmentation:` fields (currently the HF IDs
   `pyannote/wespeaker-voxceleb-resnet34-LM` and `pyannote/segmentation-3.0`) to point at the local
   sub-folders before instantiating the pipeline. (WhisperX's `DiarizationPipeline` wraps
   `pyannote.audio.Pipeline`; you may need to build the `pyannote.audio.Pipeline` directly from the
   local config and hand it to WhisperX, or call pyannote diarization directly then map speakers to
   words with `whisperx.assign_word_speakers`.)
3. Only if the bundled dir is ABSENT: fall back to `from_pretrained(<id>, use_auth_token=hf_token)`
   (which will download to cache) — and log that this needs internet + the token.

**If anything diarization-related fails (missing bundle AND no token, load error, etc.) → log a
clear reason and fall back to single-voice transcription. Never crash the dub.**

### Required packages (whisperx dropped — see build note at top)

The models are staged. For the code, **do not install whisperx** (its cuDNN-8 ctranslate2 pin
breaks the GPU stack — see the build note at the top). The pieces actually used are already
present: `faster-whisper` (ASR) and `pyannote.audio==3.3.2` (diarization, matching the bundled
3.x models). If pyannote is missing, `pip install "pyannote.audio==3.3.2"`.

---

## 3. Phases

### Phase 1 — diarized transcription (`_whisper_word_timestamps.py`) ✅ IMPLEMENTED
**Goal:** add an optional diarized transcription mode that emits the same JSON word format **plus**
a `"speaker"` field on each word.

**As built (faster-whisper + pyannote, no whisperx — see build note at top):**
1. CLI flags added: `--diarize` (bool), `--hf-token <token>`, `--min-speakers`, `--max-speakers`.
2. `_extract_diarized(audio_path, model_size, language, hf_token, min_spk, max_spk)`:
   - **Diarize first** (`_diarize_turns`): builds the pyannote pipeline from the local bundle
     (`_build_diarization_pipeline`), moves it to cuda, runs it, returns `(start, end, speaker)`
     turns. Runs before ASR so torch's cuDNN 9 loads before ctranslate2's copy.
   - **ASR** (`_extract_faster_whisper` via the existing size-fallback chain) → word timestamps.
   - **Assign** (`_assign_speakers`): label each word by max temporal overlap with the turns, with
     a nearest-turn fallback for words that overlap no turn. Adds `"speaker"` to each word dict;
     keeps `word/offset/duration` exactly as the existing format.
3. `extract_word_timestamps(...)` has `diarize`, `hf_token`, `min_speakers`, `max_speakers`.
   - If `diarize`: `try: return _extract_diarized(...)` — on **any** exception, log and fall through
     to the existing faster-whisper chain (words with no speaker). If diarization alone fails, words
     are returned without a `speaker` key.
   - If not diarizing, behavior is unchanged.
4. `main()` wires the flags; diarized output includes `"speaker"` on each JSON word.

**Also required (`_whisper_word_timestamps.py` top):** TensorFlow is disabled via `USE_TF=0` +
a meta_path blocker, because thinc (transitive via pyannote) eagerly imports TF, whose bundled
protobuf gencode (6.31.x) clashes with the runtime protobuf 5.29.x pinned by google-generativeai.

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
#
# NOTE: do NOT add whisperx here. whisperx pins ctranslate2<4.5.0 (cuDNN 8),
# which conflicts with the working cuDNN-9 GPU stack (torch 2.5.1 /
# faster-whisper 1.2.1 / ctranslate2 4.8.0). Diarization uses pyannote directly.
pyannote.audio==3.3.2   # matches the bundled models/pyannote/ 3.x weights
# faster-whisper is already required by the core install (ASR).
```
Install command (documented in the .bat):
```
pip install -r setup/requirements_diarize.txt
```
> Version note: keep `pyannote.audio` on the **3.x** line to match the bundled models. pyannote 4.x
> needs torch 2.8+ and a different config format; it will not load the staged 3.x weights.

---

## 7. Guardrails / gotchas for the implementer
- **Never mutate the shared `settings` dict** inside the TTS loop — copy per segment (Phase 4).
- **Always fall back** to single-voice on any diarization failure; log *why*.
- Keep the **JSON word format stable** — only ADD a `speaker` key; don't rename existing keys,
  or `group_words_into_segments` and the whole pipeline break.
- The faster-whisper local models (`models/whisper/faster-whisper-medium|base/`) are already
  installed and gitignored — reuse them for ASR.
- **cuDNN load order:** run diarization (pyannote/torch, cuDNN 9) *before* faster-whisper
  (ctranslate2 loads its own cuDNN). Reverse order → `Could not load symbol cudnnGetLibConfig`.
- Diarization runs on the **original audio**, before translation. TTS/translation still use Gemini.
- Test the **single-speaker regression** at the end of every phase — the existing dubbing path
  must keep working unchanged.
- Do **not** commit/push. After each phase, stop and ask the user to test.

## 8. Progress log (implementer: update as you go)
- [x] Phase 1 — diarized transcription (faster-whisper + pyannote; whisperx dropped, see build note)
- [x] Phase 2 — carry speaker through segments (`group_words_into_segments` splits on speaker change + `speaker` key; `distinct_speakers` helper; `transcribe_video` diarize params)
- [x] Phase 3 — voice-mapping UI (🎭 Speaker Voices card in `dubbing_tab.py`: toggle, HF token, Detect Speakers button, per-speaker voice dropdowns; persists `dub_speaker_voices`)
- [x] Phase 4 — per-line voice in TTS loop (`build_dubbed_audio` reads `dub_multispeaker`/`dub_speaker_voices`, diarizes transcription, shallow-copies settings per segment to override `gemini_tts_voice`)
- [x] Phase 5 — deps, installer, docs (`setup/requirements_diarize.txt` + `setup/install_diarize.bat`, whisperx-free; `models/whisper` + `models/pyannote` confirmed gitignored)
