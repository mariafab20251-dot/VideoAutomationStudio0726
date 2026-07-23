# Fresh PC Setup Guide — Video Automation Studio

This guide walks through setting up **Video Automation Studio** (ChangeGUI) on a completely fresh Windows PC, starting from a git clone. Follow the steps in order.

---

## Prerequisites

Install these **before** cloning the repo:

1. **Git for Windows** — [https://git-scm.com/download/win](https://git-scm.com/download/win)
2. **Python 3.13** (main app) — [https://www.python.org/downloads/](https://www.python.org/downloads/)
   - ✅ Check "Add Python to PATH" during install
3. **Python 3.11** (dubbing only) — [https://www.python.org/downloads/](https://www.python.org/downloads/)
   - Install to a **separate** location (e.g., `C:\Python311\`)
   - ✅ Check "Add Python to PATH" during install
4. **FFmpeg** — [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)
   - Download the **essentials** build (zip)
   - Extract to `C:\ffmpeg\` (or any folder)
   - Add `C:\ffmpeg\bin\` to System PATH
   - Verify: open CMD, type `ffmpeg -version`
5. **(Optional) CUDA Toolkit 12.1+** — for dubbing with GPU acceleration
   - [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
   - Only needed if you have a dedicated NVIDIA GPU

---

## Step 1: Clone the Repository

### Option A: Public Repo (default)

```bash
cd D:\  # or wherever you keep projects
git clone https://github.com/mariafab20251-dot/VideoAutomationStudio0726.git ChangeGUI
cd ChangeGUI
```

### Option B: Private Repo (if you make it private later)

**First time only** — authenticate once:

1. Generate a **Personal Access Token (PAT)** on GitHub:
   - Go to [https://github.com/settings/tokens](https://github.com/settings/tokens)
   - Click **Generate new token (classic)**
   - Scopes: check `repo` (full access to private repos)
   - Copy the token (you won't see it again)

2. Clone using the token:
   ```bash
   git clone https://<YOUR_GITHUB_USERNAME>:<YOUR_PAT_TOKEN>@github.com/mariafab20251-dot/VideoAutomationStudio0726.git ChangeGUI
   cd ChangeGUI
   ```

   Windows will cache your credentials — future `git pull` works without re-entering the token.

**Alternative:** Use **GitHub Desktop** ([https://desktop.github.com](https://desktop.github.com)) — it handles auth automatically for private repos.

---

## Step 2: Copy Gitignored Files (Manual Transfer)

The repo clone is **missing** these files because they're gitignored (credentials, API keys, large model weights). You must copy them from your **source PC** (the one that already has a working setup).

### What to Copy

On your **source PC**, zip these folders/files:

```
ChangeGUI/
├── overlay_settings.json                  ← MAIN settings + ALL API keys (gemini_api_key etc.)
├── automation_settings.json               ← dashboard settings (ElevenLabs key etc.)
├── processing_paths.json                  ← your saved folder paths
├── google_credentials/                    ← Google Cloud service account JSON
│   └── cloud-tts-sa.json
├── .env                                   ← (only if present) extra env vars
└── setup/
    └── models/                            ← 1.7 GB model bundle
        ├── demucs/                        ← 81 MB (already in git clone ✓)
        ├── pyannote/                      ← 31 MB (COPY THIS)
        └── whisper/                       ← 1.6 GB (COPY THIS)
            ├── faster-whisper-base/
            └── faster-whisper-medium/
```

> **⚠️ IMPORTANT — where the API keys actually live:**
> This app does **NOT** read keys from a `.env` file. All API keys (Gemini,
> etc.) are stored inside **`overlay_settings.json`** (key `gemini_api_key`),
> and the dashboard's ElevenLabs key lives in **`automation_settings.json`**.
> Copying these two JSON files brings your keys across. Alternatively, skip
> them and re-enter the keys in the app's **Settings** panel after first launch.

**Transfer method:**
- USB drive, network share, cloud storage — your choice.
- On the **fresh PC**, extract/copy these into your cloned `ChangeGUI\` folder (overwrite if asked).

**Automated option — `copy_from_source_pc.bat`:**
If your source PC's folder is reachable (network share, USB, or mounted drive),
skip the manual copying and run this helper from the clone. It pulls in all the
gitignored settings, credentials, and the model bundle for you:

```powershell
.\setup\copy_from_source_pc.bat "\\OLDPC\share\ChangeGUI"
# or a USB/backup path:
.\setup\copy_from_source_pc.bat "E:\Backup\ChangeGUI"
```

It copies `overlay_settings.json`, `automation_settings.json`, `processing_paths.json`,
`.env`, `google_credentials\`, and `setup\models\whisper` + `pyannote` (skipping the
demucs weight that's already in the clone). It also *offers* to copy `VoiceModules\`.
Run it **before** `setup_new_pc.bat`.

**Why these aren't in git:**
- `overlay_settings.json` / `automation_settings.json` — contain your API keys + machine-specific settings
- `google_credentials/` — contains the service-account private key (never commit)
- `setup/models/pyannote/` + `whisper/` — too large for GitHub (1.6 GB)
- `processing_paths.json` — machine-specific folder paths

---

## Step 3: Run the Main Installer

Open **PowerShell** (or CMD) as **Administrator**, navigate to the repo, and run:

```powershell
cd D:\ChangeGUI  # adjust to your clone path
.\setup\setup_new_pc.bat
```

### What this does:

1. **Creates `.venv`** (Python 3.13 virtual environment for the main app)
2. **Installs core dependencies** from `setup\requirements_core.txt`
3. **Asks about TTS engines** — the menu offers:
   - **[1] Kokoro** (~500 MB) — fast, GPU/CPU
   - **[2] NeuTTS** (~2 GB) — neural, GPU recommended
   - **[3] Qwen3-TTS** (~15 GB) — highest quality, **requires NVIDIA GPU**
   - **[4] ALL** / **[0] Skip**
   - *Piper* is not in this menu — install it separately with `setup\install_piper.bat` if you want it.
4. **Asks about dubbing** — say **Y** to install the dubbing tab

The installer will:
- Detect your GPU (CUDA vs CPU)
- Create `setup\dub_venv` (Python 3.11 dedicated venv)
- Restore bundled models from `setup\models\` to `models\` (runtime location)
- Install torch 2.5.1 (GPU or CPU build)
- Install pyannote, faster-whisper, demucs

**Offline install:** If you have `setup\wheels\` pre-downloaded (see "Offline Setup" below), the installer uses those instead of hitting PyPI.

---

## Step 4: Verify the Installation

After the installer finishes:

```powershell
# activate the main venv
.\.venv\Scripts\Activate.ps1

# verify core imports
python -c "import tkinter, google.generativeai, cv2, moviepy.editor; print('Core OK')"

# verify dubbing stack (if installed)
.\setup\dub_venv\Scripts\python.exe -c "import torch, faster_whisper, pyannote.audio, demucs; print('Dubbing OK')"
```

If all print "OK", you're ready.

---

## Step 5: Run the App

```powershell
cd D:\ChangeGUI
python complete_automation_gui.py
```

Or double-click `run.bat`.

**First launch:**
- The app will read `.env` for API keys
- Go to **Settings** (⚙️ icon) to verify paths (downloads, temp folder, output folder)
- Test TTS: Script Studio tab → Write Story → pick a TTS voice → Generate

---

## Updating from the Repo (git pull)

When you make changes on your **source PC** and push them:

```powershell
# on the fresh PC
cd D:\ChangeGUI
git pull origin main
```

If the repo is **private**, the first `git pull` will ask for credentials (use the same PAT you used for clone). Windows caches it after the first time.

**After pulling code changes:**
- If `requirements_core.txt` changed → re-run `setup\install_core.bat`
- If `requirements_dubbing.txt` changed → re-run `setup\install_dubbing.bat`
- If new model files were added to `setup\models\` → copy them manually (they're gitignored)

---

## Offline Setup (No Internet on Fresh PC)

If the fresh PC has **zero internet**, you need to pre-download all wheels on the source PC.

### On the Source PC (with internet):

```powershell
cd D:\ChangeGUI
.\setup\download_dubbing_wheels.bat
```

This creates `setup\wheels\` (~500 MB) with all pip packages for dubbing (torch CUDA 12.1 + all dependencies).

**Then:**
1. Copy the **entire** `setup\` folder (includes `wheels\`, `models\`, all `.bat` files) to USB/network
2. On the fresh PC, `setup_new_pc.bat` auto-detects `setup\wheels\` and installs **offline** (no PyPI downloads)

---

## CPU-Only Setup (No NVIDIA GPU)

If the fresh PC has **integrated GPU** (Intel) or **AMD shared GPU** (no CUDA), use the **CPU variant** of dubbing:

Instead of running `setup\install_dubbing.bat`, run:

```powershell
.\setup\dubbing_cpu\install_dubbing_cpu.bat
```

This installs:
- **torch 2.5.1 CPU build** (no CUDA)
- **CPU-tuned dependencies** (from `setup\dubbing_cpu\requirements_dubbing_cpu.txt`)

Everything else (main app, TTS engines) works identically.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'google.generativeai'`

The venv isn't activated. Run:
```powershell
.\.venv\Scripts\Activate.ps1
python complete_automation_gui.py
```

Or use `run.bat` which auto-activates.

### `ffmpeg is not recognized`

FFmpeg isn't in PATH. Add `C:\ffmpeg\bin` to System PATH (Google "add to PATH Windows 11"), then restart PowerShell.

### Dubbing tab: `Pipeline.from_pretrained() returns None`

`matplotlib` is missing (pyannote silently fails without it). The installer should've caught this — re-run:
```powershell
.\setup\dub_venv\Scripts\python.exe -m pip install matplotlib==3.11.0
```

### GPU dubbing fails with `cuDNN 8 vs 9 mismatch`

You installed whisperx (cuDNN 8) by mistake. It's incompatible. Delete `setup\dub_venv\`, re-run `install_dubbing.bat`. **Never** `pip install whisperx` in this venv.

### App launches but all tabs are empty / crashes on "Write Story"

Your API keys are missing. This app stores keys in **`overlay_settings.json`**, not `.env`.
Either copy `overlay_settings.json` (and `automation_settings.json`) from the source PC, or
open the app → **Settings** panel → paste your Gemini API key into the "Gemini API Key" field
and save. The key is stored as `gemini_api_key` inside `overlay_settings.json`.

---

## File Structure Reference

After a complete setup, your clone should look like this:

```
ChangeGUI/
├── .env                              ← API keys (gitignored)
├── .venv/                            ← main app venv (gitignored, rebuilt per PC)
├── google_credentials/               ← service account JSON (gitignored)
├── models/                           ← runtime model location (gitignored)
│   ├── demucs/                       ← htdemucs weight (restored by installer)
│   ├── pyannote/                     ← diarization models (restored by installer)
│   └── whisper/                      ← faster-whisper base + medium (restored by installer)
├── setup/
│   ├── models/                       ← bundled models (gitignored except demucs)
│   │   ├── demucs/                   ← 81 MB (tracked in git ✓)
│   │   ├── pyannote/                 ← 31 MB (gitignored, copy manually)
│   │   └── whisper/                  ← 1.6 GB (gitignored, copy manually)
│   ├── wheels/                       ← offline pip cache (gitignored, optional)
│   ├── dub_venv/                     ← dubbing venv (gitignored, rebuilt per PC)
│   ├── dubbing_cpu/                  ← CPU variant installer + requirements
│   ├── setup_new_pc.bat              ← master installer
│   ├── install_dubbing.bat           ← GPU dubbing installer
│   ├── download_dubbing_wheels.bat   ← wheel downloader for offline
│   ├── requirements_core.txt         ← main app deps
│   ├── requirements_dubbing.txt      ← GPU dubbing deps
│   └── ...                           ← other TTS installers
├── VoiceModules/                     ← TTS engine data (gitignored, downloaded by installers)
│   ├── KokoroTTS/                    ← 354 MB
│   ├── PiperTTS/                     ← 814 MB
│   ├── NeuTTS/                       ← 3.7 GB
│   └── Qwen3-TTS/                    ← 13 GB
├── complete_automation_gui.py        ← main entry point
├── run.bat                           ← launcher (auto-activates venv)
└── FRESH_PC_SETUP.md                 ← this guide

Total size: ~3.5 GB (minimal, no TTS) to ~22 GB (all TTS engines + dubbing)
```

---

## Summary Checklist

- [ ] Install Git, Python 3.13, Python 3.11, FFmpeg, (optional) CUDA Toolkit
- [ ] Clone the repo (public HTTPS or private PAT)
- [ ] Copy `.env`, `google_credentials/`, `setup/models/pyannote/`, `setup/models/whisper/` from source PC
- [ ] Run `setup\setup_new_pc.bat` (installs main app + dubbing)
- [ ] Verify: `python -c "import tkinter, google.generativeai, cv2, moviepy.editor; print('OK')"`
- [ ] Launch: `python complete_automation_gui.py` or `run.bat`
- [ ] Pull updates: `git pull origin main` (authenticate once if private)

---

**Need help?** Check the [SETUP_GUIDE.md](SETUP_GUIDE.md) (original setup guide) or [MULTISPEAKER_DUBBING_PLAN.md](MULTISPEAKER_DUBBING_PLAN.md) (dubbing architecture).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
