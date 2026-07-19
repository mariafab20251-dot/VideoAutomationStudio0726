#!/usr/bin/env python3
"""
Download and convert all Kokoro TTS voices to voices.bin format.
This script downloads voice files from Hugging Face and creates a combined voices.bin file.
"""

import os
import sys
import numpy as np
from pathlib import Path

def main():
    # Output directory
    output_dir = Path(__file__).parent / "VoiceModules" / "KokoroTTS"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Kokoro TTS Multilingual Voices Downloader")
    print("=" * 60)

    # Check for required packages
    try:
        import torch
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        print(f"\n[ERROR] Missing required package: {e}")
        print("\nPlease install required packages:")
        print("  pip install torch huggingface_hub")
        sys.exit(1)

    # All available voices from hexgrad/Kokoro-82M
    voices = [
        # American English Female
        "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
        "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        # American English Male
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
        "am_michael", "am_onyx", "am_puck",
        # British English Female
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        # British English Male
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
        # Spanish
        "ef_dora", "em_alex", "em_santa",
        # French
        "ff_siwis",
        # Hindi
        "hf_alpha", "hf_beta",
        # Italian
        "if_sara", "im_nicola",
        # Japanese
        "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
        # Brazilian Portuguese
        "pf_dora", "pm_alex", "pm_santa",
        # Mandarin Chinese
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zm_yunjian", "zm_yunxi", "zm_yunyang",
    ]

    print(f"\nDownloading {len(voices)} voices from Hugging Face...")
    print("This may take a few minutes...\n")

    voice_data = {}
    failed = []

    for i, voice_name in enumerate(voices, 1):
        try:
            print(f"[{i}/{len(voices)}] Downloading {voice_name}...", end=" ", flush=True)

            # Download the .pt file from Hugging Face
            file_path = hf_hub_download(
                repo_id="hexgrad/Kokoro-82M",
                filename=f"voices/{voice_name}.pt",
                cache_dir=None
            )

            # Load the PyTorch tensor
            voice_tensor = torch.load(file_path, map_location="cpu", weights_only=True)

            # Convert to numpy array
            voice_array = voice_tensor.numpy()

            # Store in dictionary
            voice_data[voice_name] = voice_array

            print("OK")

        except Exception as e:
            print(f"FAILED ({e})")
            failed.append(voice_name)

    if not voice_data:
        print("\n[ERROR] No voices were downloaded successfully!")
        sys.exit(1)

    # Save as .npz file (numpy compressed archive)
    output_file = output_dir / "voices-multilingual.bin"

    print(f"\nSaving {len(voice_data)} voices to {output_file}...")

    # Save as npz format
    np.savez(output_file, **voice_data)

    # Rename to .bin
    npz_file = output_dir / "voices-multilingual.bin.npz"
    if npz_file.exists():
        npz_file.rename(output_file)

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"\nSuccessfully downloaded: {len(voice_data)} voices")
    if failed:
        print(f"Failed to download: {len(failed)} voices")
        print(f"  Failed voices: {', '.join(failed)}")

    print(f"\nVoices file saved to:")
    print(f"  {output_file}")

    print("\n" + "-" * 60)
    print("NEXT STEPS:")
    print("-" * 60)
    print("1. Rename the file to replace your current voices file:")
    print(f"   Rename: voices-multilingual.bin -> voices-v1.0.bin")
    print("\n2. Or update the code to look for 'voices-multilingual.bin'")
    print("\n3. Restart the GUI and test the multilingual voices!")
    print("=" * 60)

if __name__ == "__main__":
    main()
