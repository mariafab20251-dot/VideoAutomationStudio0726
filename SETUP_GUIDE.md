# Video Automation Studio — Setup Guide

Two ways to get the tool on a new machine:

- **Portable folder** (quick copy — no git needed, but can't `git pull`)
- **Git clone** (full repo — enables `git pull` for updates)

---

## 1️⃣ Portable Folder Copy (Quick Start)

Use this to move the tool to a new laptop immediately without setting up git.

### On the source machine (current PC)

1. Copy the entire `D:\MyAutomations\VideoAutomationStudio` folder to a USB drive or network share.
2. Or use `robocopy` for faster/resumable copy:
   ```
   robocopy "D:\MyAutomations\VideoAutomationStudio" "X:\VideoAutomationStudio" /E /COPY:DAT /R:2 /W:2
   ```

### On the new laptop

Just run the setup script — it handles everything:

```
cd /d D:\MyAutomations\VideoAutomationStudio
setup\setup_new_pc.bat
```

That's it. It will:
1. Rename the old `.venv` to `.venv_old`
2. Create a fresh virtual environment
3. Install core dependencies
4. Ask which TTS engines you want and run the right installer
4. Install the TTS engine(s) you need:
   - `setup\install_kokoro.bat` — Kokoro TTS
   - `setup\install_neutts.bat` — NeuTTS  
   - `setup\install_qwen.bat` — Qwen3-TTS
5. Launch:
   ```
   run.bat
   ```

> **Limitation:** With the portable copy, you cannot `git pull` updates. You'd need to re-copy the whole folder each time.

---

## 2️⃣ Git Clone (Full Repo — Enables Updates via `git pull`)

### First time setup on new laptop

```bash
# Install Git for Windows if not already installed
# https://git-scm.com/downloads/win

# Clone the repository
cd D:\MyAutomations
git clone https://github.com/shahi/VideoAutomationStudio.git

# Rename the folder (optional)
ren VideoAutomationStudio VideoAutomationStudio_repo

# Create virtual environment  
cd VideoAutomationStudio_repo
python -m venv .venv

# Activate and install
.venv\Scripts\activate.bat
pip install -r setup\requirements_core.txt

# Install TTS engines you need
setup\install_kokoro.bat
setup\install_neutts.bat
setup\install_qwen.bat

# Launch
run.bat
```

### Getting updates later

```bash
# Navigate to the repo folder
cd D:\MyAutomations\VideoAutomationStudio_repo

# Activate venv
.venv\Scripts\activate.bat

# Pull latest code from GitHub
git pull origin main

# Update dependencies (if any changed)
pip install -r setup\requirements_core.txt --upgrade

# Launch
run.bat
```

> `git pull` downloads only the changed files — much faster than re-copying everything.

---

## 3️⃣ Private Repository Setup

When you make the repo private, authentication is required.

### Step 1: Make repo private on GitHub
- Go to https://github.com/shahi/VideoAutomationStudio
- **Settings → General → Change repository visibility → Make private**

### Step 2: Update your local clone

#### Option A: Personal Access Token (recommended)
```bash
# Generate a token at: GitHub → Settings → Developer settings → Personal access tokens
#   - Scopes: repo (full control) + optionally workflow if needed
#   - Copy the token immediately (it's shown once)

# Update the remote URL to include your token:
git remote set-url origin https://YOUR_TOKEN@github.com/shahi/VideoAutomationStudio.git

# Now pull works without password prompts:
git pull origin main
```

#### Option B: GitHub CLI
```bash
# Install: https://cli.github.com/
gh auth login
# The clone will work automatically after auth.
```

#### Option C: SSH Key
```bash
# Generate SSH key (if you haven't):
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to GitHub: Settings → SSH and GPG keys → New SSH key

# Update remote URL:
git remote set-url origin git@github.com:shahi/VideoAutomationStudio.git

# Now pull works:
git pull origin main
```

> **Security note:** If using Option A with a token in the URL, the token is stored in `.git/config`. Never share this file. Option B (GitHub CLI) or C (SSH) are more secure for long-term use.

---

## 4️⃣ Workflow Summary

| Action | Command |
|--------|---------|
| **Clone fresh** | `git clone https://github.com/shahi/VideoAutomationStudio.git` |
| **Pull updates** | `git pull origin main` |
| **Check status** | `git status` |
| **See recent changes** | `git log --oneline -10` |
| **Launch tool** | `run.bat` |

### Sync from portable to repo (after making local changes)

If you've been working in the portable folder and want to push those changes to git:

```bash
# From inside the portable folder:
git init
git remote add origin https://github.com/shahi/VideoAutomationStudio.git
git add -A
git commit -m "sync portable folder"
git push -u origin main
```

---

## 5️⃣ TTS Engine Setup (Per Laptop)

Each TTS engine is separate because they have large dependencies (~10-15 GB for Qwen3):

| TTS Engine | Install Script | Size | Notes |
|-----------|---------------|------|-------|
| Kokoro | `setup\install_kokoro.bat` | ~500 MB | Fastest, good quality |
| NeuTTS | `setup\install_neutts.bat` | ~2 GB | Medium quality |
| Qwen3-TTS | `setup\install_qwen.bat` | ~15 GB | Best quality, needs CUDA GPU |

> **Note:** PiperTTS and Qwen3-TTS engine files are NOT stored in the git repo (~10-15 GB of model binaries). They're installed per-machine via the setup scripts above. If you use the portable folder copy, those engines come with it.

> **CUDA Note:** Qwen3 TTS checks for an NVIDIA GPU automatically. If `CUDA=False` appears in the log, install CUDA-capable PyTorch:
> ```
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```
