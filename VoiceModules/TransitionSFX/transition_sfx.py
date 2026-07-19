"""
Transition Sound Effects Generator
==================================
Generates cinematic, "viral"-quality transition SFX procedurally with numpy/scipy.
Key techniques used:
  - Exponential frequency sweeps (the unmistakable whoosh shape)
  - Stereo L<R delay (creates the "passing by" sensation)
  - Bandpass-filtered noise tail (the airy wind)
  - Pitched transients + lowpass envelopes (impact, boom, bass drop)
  - FM/AM synthesis (zaps, glitches, risers)
  - Vinyl-style lowpass sweep with pitch drop (vinyl brake)
  - Layered sub + thump + click (bass drop)

Also loads user-provided SFX from the sfx_library folder if present.

Usage:
    from transition_sfx import TransitionSFX
    sfx = TransitionSFX(sr=44100)
    audio = sfx.whoosh(duration=0.5)  # returns mono float32 array
"""

import os
import math
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from scipy.signal import butter, sosfilt, sosfiltfilt
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


SFX_LIBRARY_DIR = Path(__file__).parent / "sfx_library"
SFX_SAMPLES_DIR = Path(__file__).parent / "samples"


# ---------- DSP primitives ------------------------------------------------

def _normalize(arr: np.ndarray, peak: float = 0.9) -> np.ndarray:
    if arr.size == 0:
        return arr
    m = np.max(np.abs(arr))
    if m < 1e-9:
        return arr
    return arr * (peak / m)


def _adsr(n: int, attack: float, decay: float, sustain_level: float, release: float,
          hold: float = 0.0) -> np.ndarray:
    """ADSR envelope. attack/decay/release are fractions of n (or seconds if hold>0)."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    if hold > 0:
        a_n = int(attack * _SR)
        d_n = int(decay * _SR)
        h_n = int(hold * _SR)
        r_n = int(release * _SR)
    else:
        a_n = max(1, int(attack * n))
        d_n = max(1, int(decay * n))
        h_n = max(0, int(hold * n))
        r_n = max(1, int(release * n))

    total = a_n + d_n + h_n + r_n
    if total > n:
        # Scale down to fit
        scale = n / total
        a_n = max(1, int(a_n * scale))
        d_n = max(1, int(d_n * scale))
        h_n = max(0, int(h_n * scale))
        r_n = max(1, int(r_n * scale))

    env = np.zeros(n, dtype=np.float32)
    pos = 0
    # Attack
    end = min(pos + a_n, n)
    seg = np.linspace(0, 1, end - pos, endpoint=False, dtype=np.float32)
    env[pos:end] = seg
    pos = end
    # Decay
    end = min(pos + d_n, n)
    seg = np.linspace(1, sustain_level, end - pos, endpoint=False, dtype=np.float32)
    env[pos:end] = seg
    pos = end
    # Hold
    end = min(pos + h_n, n)
    env[pos:end] = sustain_level
    pos = end
    # Release
    end = min(pos + r_n, n)
    seg = np.linspace(sustain_level, 0, end - pos, endpoint=False, dtype=np.float32)
    env[pos:end] = seg
    pos = end
    return env


def _lowpass(arr: np.ndarray, cutoff_norm: float) -> np.ndarray:
    if not _HAS_SCIPY:
        k = max(3, int(1 / max(cutoff_norm, 0.01)))
        kernel = np.ones(k, dtype=np.float32) / k
        return np.convolve(arr, kernel, mode='same')
    sos = butter(4, cutoff_norm, btype='low', output='sos')
    return sosfilt(sos, arr).astype(np.float32)


def _highpass(arr: np.ndarray, cutoff_norm: float) -> np.ndarray:
    if not _HAS_SCIPY:
        return arr - _lowpass(arr, cutoff_norm)
    sos = butter(4, cutoff_norm, btype='high', output='sos')
    return sosfilt(sos, arr).astype(np.float32)


def _bandpass(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    if not _HAS_SCIPY:
        return _highpass(_lowpass(arr, high), low)
    sos = butter(4, [low, high], btype='band', output='sos')
    return sosfilt(sos, arr).astype(np.float32)


def _noise(n: int) -> np.ndarray:
    return np.random.uniform(-1.0, 1.0, n).astype(np.float32)


def _sine(freq: float, n: int, sr: int, phase: float = 0.0) -> np.ndarray:
    t = np.linspace(0, n / sr, n, endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * freq * t + phase).astype(np.float32)


def _exp_sweep(f0: float, f1: float, n: int, sr: int) -> np.ndarray:
    """Exponential frequency sweep — the foundation of a great whoosh."""
    if f0 <= 0:
        f0 = 20.0
    t = np.linspace(0, n / sr, n, endpoint=False, dtype=np.float32)
    k = math.log(f1 / f0) / (n / sr)
    phase = 2 * np.pi * f0 * (np.exp(k * t) - 1) / k
    return np.sin(phase).astype(np.float32)


def _linear_sweep(f0: float, f1: float, n: int, sr: int) -> np.ndarray:
    t = np.linspace(0, n / sr, n, endpoint=False, dtype=np.float32)
    phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / (n / sr) * t ** 2)
    return np.sin(phase).astype(np.float32)


def _stereoize(mono: np.ndarray, delay_samples: int = 0,
               pan: float = 0.0) -> np.ndarray:
    """Convert mono to stereo with optional delay (the "passing by" feel)."""
    n = len(mono)
    if delay_samples <= 0 and pan == 0.0:
        return np.stack([mono, mono], axis=-1).astype(np.float32)

    delay_samples = max(0, int(delay_samples))
    right = np.zeros(n + delay_samples, dtype=np.float32)
    right[delay_samples:delay_samples + n] = mono
    right = right[:n]

    if pan != 0.0:
        # pan in [-1, 1]; -1 = full left, +1 = full right
        l_gain = math.cos((pan + 1) * math.pi / 4)
        r_gain = math.sin((pan + 1) * math.pi / 4)
    else:
        l_gain = r_gain = 1.0

    left = mono * l_gain
    return np.stack([left, right * r_gain], axis=-1).astype(np.float32)


# Cache sr for _adsr fraction-mode
_SR = 44100


class TransitionSFX:
    """Generates transition sound effects. Loads from sfx_library if file exists, else generates."""

    SFX_MAP = {
        "fade": "shimmer.wav",
        "zoom": "whoosh.wav",
        "blur": "boom.wav",
        "slide": "whoosh.wav",
        "wipe": "swoosh.wav",
        "glitch": "click.wav",
        "cinematic_bars": "chime.wav",
        "lens_flare": "sparkle.wav",
        "light_leak": "hiss.wav",
        "film_burn": "rumble.wav",
        # New viral SFX
        "bass_drop": "bass_drop.wav",
        "riser": "riser.wav",
        "impact": "impact.wav",
        "vinyl_brake": "vinyl_brake.wav",
        "notification": "notification.wav",
        "cinematic_whoosh": "cinematic_whoosh.wav",
        "sub_boom": "sub_boom.wav",
        "horn_stab": "horn_stab.wav",
        # CapCut-style transitions
        "radial_wipe": "swoosh.wav",
        "color_dissolve": "shimmer.wav",
        "split_wipe": "swoosh.wav",
        "luma_wipe": "shimmer.wav",
        # SFX name → filename aliases (so get() can load the correct sample)
        "shimmer":   "shimmer.wav",
        "whoosh":    "whoosh.wav",
        "swoosh":    "swoosh.wav",
        "boom":      "boom.wav",
        "chime":     "chime.wav",
        "click":     "click.wav",
        "zap":       "zap.wav",
        "sparkle":   "sparkle.wav",
        "hiss":      "hiss.wav",
        "rumble":    "rumble.wav",
    }

    def __init__(self, sr: int = 44100, library_dir: Optional[str] = None):
        global _SR
        self.sr = sr
        _SR = sr
        self.library_dir = Path(library_dir) if library_dir else SFX_LIBRARY_DIR
        self.library_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir = SFX_SAMPLES_DIR
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict = {}

    # ---- loader ----------------------------------------------------------

    def get(self, name: str, duration: float = 0.5) -> np.ndarray:
        """Get SFX by transition name. Returns mono float32 array.
        Lookup order:
          1. cache
          2. <library_dir>/<name>.wav/.mp3/.ogg   (user-named file)
          3. <library_dir>/<SFX_MAP_filename>      (user sfx_library folder)
          4. <samples_dir>/<SFX_MAP_filename>      (the bundled samples/ folder)
          5. fallback to procedurally generated SFX
        """
        cache_key = (name, round(duration, 3))
        if cache_key in self._cache:
            return self._cache[cache_key]

        def _try_load(path):
            if not path.exists():
                return None
            try:
                import soundfile as sf
                data, file_sr = sf.read(str(path), dtype='float32')
                if data.ndim > 1:
                    data = data.mean(axis=1)
                if file_sr != self.sr:
                    n_target = int(len(data) * self.sr / file_sr)
                    data = np.interp(
                        np.linspace(0, len(data), n_target),
                        np.arange(len(data)),
                        data
                    ).astype(np.float32)
                target_n = int(duration * self.sr)
                if len(data) > target_n:
                    data = data[:target_n]
                elif len(data) < target_n:
                    pad = np.zeros(target_n - len(data), dtype=np.float32)
                    data = np.concatenate([data, pad])
                self._cache[cache_key] = data
                return data
            except Exception as e:
                print(f"[SFX] Failed to load {path}: {e}")
                return None

        # Step 1: Try <library_dir>/<name>.wav/.mp3/.ogg (user-named custom files)
        for ext in ('.wav', '.mp3', '.ogg'):
            result = _try_load(self.library_dir / f'{name}{ext}')
            if result is not None:
                return result

        # Step 2: Map name -> filename via SFX_MAP, then try library_dir + samples_dir
        filename = self.SFX_MAP.get(name, "whoosh.wav")
        for base_dir in (self.library_dir, self.samples_dir):
            result = _try_load(base_dir / filename)
            if result is not None:
                return result

        # Step 3: No file found anywhere — generate procedurally
        gen = getattr(self, name, None) or self.whoosh
        data = gen(duration=duration)
        self._cache[cache_key] = data
        return data

    # ---- SFX definitions -------------------------------------------------

    def whoosh(self, duration: float = 0.5) -> np.ndarray:
        """
        Cinematic whoosh — exponential frequency sweep (300→4000Hz) layered with
        bandpass noise. Sharp attack, smooth decay. The "viral" whoosh.
        """
        n = int(duration * self.sr)

        # Layer 1: pitch sweep (the tonal body)
        sweep = _exp_sweep(300, 4000, n, self.sr)

        # Layer 2: bandpass-filtered noise moving with the sweep
        noise = _noise(n)
        steps = 12
        chunk = max(1, n // steps)
        noise_sweep = np.zeros(n, dtype=np.float32)
        for i in range(steps):
            progress = i / steps
            f_center = 300 + 3700 * progress  # match the sweep
            f_low = max(50, f_center - 400)
            f_high = min(self.sr // 2 - 100, f_center + 400)
            seg = _bandpass(noise[i * chunk:(i + 1) * chunk],
                            f_low / self.sr, f_high / self.sr)
            noise_sweep[i * chunk:(i + 1) * chunk] = seg

        # Mix tonal + noise
        out = sweep * 0.55 + noise_sweep * 0.7

        # ADSR: fast attack, soft release
        env = _adsr(n, attack=0.05, decay=0.15, sustain_level=0.7, release=0.6)
        return _normalize(out * env, peak=0.85)

    def swoosh(self, duration: float = 0.6) -> np.ndarray:
        """
        Soft swoosh — the "air moved" sound. A downward sweep (4000→200Hz) blended
        with a lowpass-noise bed. Used for wipes and gentle slides.
        """
        n = int(duration * self.sr)

        # Downward sweep (signature wipe sound)
        sweep = _exp_sweep(4000, 200, n, self.sr)

        # Noise bed that closes down with the sweep
        noise = _noise(n)
        steps = 10
        chunk = max(1, n // steps)
        noise_bed = np.zeros(n, dtype=np.float32)
        for i in range(steps):
            progress = i / steps
            cutoff = 0.5 - 0.45 * progress  # close the lowpass as time passes
            seg = _lowpass(noise[i * chunk:(i + 1) * chunk], max(0.02, cutoff))
            noise_bed[i * chunk:(i + 1) * chunk] = seg

        out = sweep * 0.5 + noise_bed * 0.6

        # Soft attack, slow release
        env = _adsr(n, attack=0.08, decay=0.2, sustain_level=0.6, release=0.5)
        return _normalize(out * env, peak=0.8)

    def boom(self, duration: float = 0.5) -> np.ndarray:
        """
        Cinematic boom — sub-bass thump (60Hz) with fast attack, layered with a
        high transient click. The "reveal" sound.
        """
        n = int(duration * self.sr)

        # Sub bass
        sub = _sine(60, n, self.sr) * np.exp(-np.linspace(0, 1, n) * 5)
        # Pitch drop on the sub for extra weight
        drop = _exp_sweep(120, 40, n, self.sr) * np.exp(-np.linspace(0, 1, n) * 4) * 0.5

        # Noise burst (the "ka" part of "boom")
        noise = _noise(n) * np.exp(-np.linspace(0, 1, n) * 12)
        noise = _lowpass(noise, 0.15) * 0.4

        # Initial transient click
        click_n = int(0.008 * self.sr)
        click = _highpass(_noise(click_n), 0.3) * np.exp(-np.linspace(0, 1, click_n) * 25) * 0.7

        out = np.zeros(n, dtype=np.float32)
        out += sub * 0.9
        out += drop
        out += noise
        out[:click_n] += click

        return _normalize(out, peak=0.9)

    def chime(self, duration: float = 0.7) -> np.ndarray:
        """Bright bell-like multi-sine — C5+E5+G5 with bell envelope."""
        n = int(duration * self.sr)
        t = np.linspace(0, n / self.sr, n, endpoint=False, dtype=np.float32)
        freqs = [523.25, 659.25, 783.99, 1046.5]  # C5, E5, G5, C6
        out = np.zeros(n, dtype=np.float32)
        decay = np.exp(-np.linspace(0, 1, n) * 3.0)
        for i, f in enumerate(freqs):
            out += np.sin(2 * np.pi * f * t).astype(np.float32) * decay * (0.7 - i * 0.1)
        out /= len(freqs)
        env = _adsr(n, attack=0.005, decay=0.1, sustain_level=0.5, release=0.4)
        return _normalize(out * env, peak=0.7)

    def click(self, duration: float = 0.12) -> np.ndarray:
        """Short noise burst — sharp, digital. Glitch/cut."""
        n = int(duration * self.sr)
        noise = _noise(n) * np.exp(-np.linspace(0, 1, n) * 35)
        out = _highpass(noise, 0.25) * 0.8 + noise * 0.2
        return _normalize(out, peak=0.9)

    def zap(self, duration: float = 0.25) -> np.ndarray:
        """Frequency sweep 1200→200Hz with noise. Sci-fi zap."""
        n = int(duration * self.sr)
        sine = _exp_sweep(1200, 200, n, self.sr, )
        noise = _noise(n) * 0.35
        out = sine * 0.6 + _highpass(noise, 0.3) * 0.4
        env = _adsr(n, attack=0.001, decay=0.05, sustain_level=0.5, release=0.3)
        return _normalize(out * env, peak=0.85)

    def sparkle(self, duration: float = 0.5) -> np.ndarray:
        """High sine with vibrato + bell ping. Lens-flare shimmer."""
        n = int(duration * self.sr)
        t = np.linspace(0, n / self.sr, n, endpoint=False, dtype=np.float32)
        base = 2000
        vibrato = 200 * np.sin(2 * np.pi * 6 * t)
        out = np.sin(2 * np.pi * (base + vibrato) * t).astype(np.float32)
        out *= np.exp(-np.linspace(0, 1, n) * 4)

        # Add a small bell ping
        ping_n = int(0.08 * self.sr)
        tp = np.linspace(0, 0.08, ping_n, endpoint=False, dtype=np.float32)
        ping = np.sin(2 * np.pi * 2400 * tp).astype(np.float32) * np.exp(-np.linspace(0, 1, ping_n) * 12)
        if ping_n < n:
            out[:ping_n] += ping * 0.5

        env = _adsr(n, attack=0.01, decay=0.1, sustain_level=0.5, release=0.3)
        return _normalize(out * env, peak=0.6)

    def hiss(self, duration: float = 0.6) -> np.ndarray:
        """Highpass noise with AM modulation. Soft airy."""
        n = int(duration * self.sr)
        out = _highpass(_noise(n), 0.1) * 0.4
        mod = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * np.linspace(0, n / self.sr, n, endpoint=False))
        out *= mod.astype(np.float32)
        env = _adsr(n, attack=0.15, decay=0.1, sustain_level=0.6, release=0.3)
        return _normalize(out * env, peak=0.4)

    def rumble(self, duration: float = 0.7) -> np.ndarray:
        """Low sine + filtered noise. Deep ominous film-burn reveal."""
        n = int(duration * self.sr)
        sine = _sine(50, n, self.sr) * np.exp(-np.linspace(0, 1, n) * 2.5)
        noise = _lowpass(_noise(n), 0.1) * np.exp(-np.linspace(0, 1, n) * 3.5)
        out = sine * 0.6 + noise * 0.4
        env = _adsr(n, attack=0.05, decay=0.1, sustain_level=0.7, release=0.4)
        return _normalize(out * env, peak=0.8)

    def shimmer(self, duration: float = 0.5) -> np.ndarray:
        """Highpass noise + bell ping. Soft fade sparkle."""
        n = int(duration * self.sr)
        out = _highpass(_noise(n), 0.15) * 0.5
        ping_n = int(0.1 * self.sr)
        t = np.linspace(0, 0.1, ping_n, endpoint=False, dtype=np.float32)
        ping = np.sin(2 * np.pi * 1500 * t).astype(np.float32) * np.exp(-np.linspace(0, 1, ping_n) * 10)
        if ping_n < n:
            out[:ping_n] += ping * 0.4
        env = _adsr(n, attack=0.05, decay=0.1, sustain_level=0.6, release=0.3)
        return _normalize(out * env, peak=0.55)

    # ---- NEW: viral SFX --------------------------------------------------

    def bass_drop(self, duration: float = 0.8) -> np.ndarray:
        """
        TikTok-style bass drop — downward sine sweep from 250Hz to 35Hz with
        a sub layer and a quick lowpass noise tail. The "viral hook" sound.
        """
        n = int(duration * self.sr)

        # Main pitch drop
        main = _exp_sweep(250, 35, n, self.sr)

        # Sub layer (sine below the sweep)
        sub = _sine(40, n, self.sr) * np.exp(-np.linspace(0, 1, n) * 2.5) * 0.6

        # Lowpass noise burst (the "thud")
        noise = _lowpass(_noise(n), 0.08) * np.exp(-np.linspace(0, 1, n) * 5) * 0.3

        out = main * 0.7 + sub + noise

        # Sharp attack, long release
        env = _adsr(n, attack=0.005, decay=0.05, sustain_level=0.8, release=0.5)
        return _normalize(out * env, peak=0.95)

    def riser(self, duration: float = 1.0) -> np.ndarray:
        """
        Tension riser — upward sweep 80→2000Hz with increasing noise. The sound
        right before a reveal/climax.
        """
        n = int(duration * self.sr)

        # Upward sweep
        sweep = _exp_sweep(80, 2000, n, self.sr)

        # Increasing noise
        t = np.linspace(0, 1, n, dtype=np.float32)
        noise = _noise(n) * np.linspace(0.1, 1.0, n).astype(np.float32)
        noise = _highpass(noise, 0.05) * 0.4

        out = sweep * 0.5 + noise

        # Slow build, then sharp end
        env = _adsr(n, attack=0.5, decay=0.3, sustain_level=0.7, release=0.05)
        return _normalize(out * env, peak=0.8)

    def impact(self, duration: float = 0.4) -> np.ndarray:
        """
        Trailer-style impact — a hard transient (filtered click) + low boom
        + a short metallic ring. The "drop frame" sound.
        """
        n = int(duration * self.sr)

        # Hard transient
        click_n = int(0.005 * self.sr)
        click = _noise(click_n) * np.exp(-np.linspace(0, 1, click_n) * 30)
        click = _bandpass(click, 200 / self.sr, 4000 / self.sr) * 1.5

        # Low boom
        boom = _sine(80, n, self.sr) * np.exp(-np.linspace(0, 1, n) * 8) * 0.8

        # Noise tail
        noise = _lowpass(_noise(n), 0.1) * np.exp(-np.linspace(0, 1, n) * 6) * 0.4

        # Metallic ring
        ring_n = int(0.3 * self.sr)
        tr = np.linspace(0, 0.3, ring_n, endpoint=False, dtype=np.float32)
        ring = (np.sin(2 * np.pi * 1800 * tr) * 0.3
                + np.sin(2 * np.pi * 2400 * tr) * 0.2).astype(np.float32)
        ring *= np.exp(-np.linspace(0, 1, ring_n) * 4)
        ring_full = np.zeros(n, dtype=np.float32)
        ring_full[:ring_n] += ring

        out = np.zeros(n, dtype=np.float32)
        out[:click_n] += click
        out += boom + noise + ring_full

        return _normalize(out, peak=0.95)

    def vinyl_brake(self, duration: float = 0.7) -> np.ndarray:
        """
        Record scratch / vinyl brake — fast downward sweep (2000→80Hz) with
        a high-frequency noise crackle. Perfect for "rewind" or "wait what" moments.
        """
        n = int(duration * self.sr)

        # Down sweep
        sweep = _exp_sweep(2000, 80, n, self.sr)

        # Crackle noise (vinyl surface)
        noise = _highpass(_noise(n), 0.15) * 0.4

        # Amplitude wobble (the "brake" stutter)
        t = np.linspace(0, 1, n, dtype=np.float32)
        wobble = 0.7 + 0.3 * np.sin(2 * np.pi * 20 * t)
        out = (sweep + noise) * wobble.astype(np.float32)

        env = _adsr(n, attack=0.01, decay=0.1, sustain_level=0.7, release=0.2)
        return _normalize(out * env, peak=0.85)

    def notification(self, duration: float = 0.5) -> np.ndarray:
        """
        Two-tone notification "ding-dong" — A5 then E5 with bell envelope.
        Great for "new message" or "info pop" transitions.
        """
        n = int(duration * self.sr)
        half = n // 2

        t1 = np.linspace(0, half / self.sr, half, endpoint=False, dtype=np.float32)
        t2 = np.linspace(0, (n - half) / self.sr, n - half, endpoint=False, dtype=np.float32)

        decay1 = np.exp(-np.linspace(0, 1, half) * 6)
        decay2 = np.exp(-np.linspace(0, 1, n - half) * 6)

        tone1 = (np.sin(2 * np.pi * 880 * t1) * 0.6
                 + np.sin(2 * np.pi * 1760 * t1) * 0.3).astype(np.float32) * decay1
        tone2 = (np.sin(2 * np.pi * 659.25 * t2) * 0.6
                 + np.sin(2 * np.pi * 1318.5 * t2) * 0.3).astype(np.float32) * decay2

        out = np.concatenate([tone1, tone2]).astype(np.float32)
        return _normalize(out, peak=0.7)

    def cinematic_whoosh(self, duration: float = 0.8) -> np.ndarray:
        """
        Trailer-grade cinematic whoosh — heavy layered sweep (200→5000Hz) with
        stereo delay (left-to-right pass-by) and an air tail. The "epic" sound.
        """
        n = int(duration * self.sr)

        # Main sweep
        sweep = _exp_sweep(200, 5000, n, self.sr)

        # Noise layer moving with the sweep
        noise = _noise(n)
        steps = 16
        chunk = max(1, n // steps)
        noise_sweep = np.zeros(n, dtype=np.float32)
        for i in range(steps):
            progress = i / steps
            f_center = 200 + 4800 * progress
            f_low = max(50, f_center - 600)
            f_high = min(self.sr // 2 - 100, f_center + 600)
            seg = _bandpass(noise[i * chunk:(i + 1) * chunk],
                            f_low / self.sr, f_high / self.sr)
            noise_sweep[i * chunk:(i + 1) * chunk] = seg

        # Air tail (high noise fading out)
        air = _highpass(_noise(n), 0.3) * 0.3

        out_mono = sweep * 0.5 + noise_sweep * 0.7 + air

        # ADSR
        env = _adsr(n, attack=0.03, decay=0.1, sustain_level=0.7, release=0.4)
        out_mono = out_mono * env

        # This is mono (caller will mono→stereo via duplication if needed)
        return _normalize(out_mono, peak=0.9)

    def sub_boom(self, duration: float = 0.6) -> np.ndarray:
        """
        Pure sub-bass boom — 45Hz sine with fast attack, used under a cinematic
        reveal so you feel the weight.
        """
        n = int(duration * self.sr)

        # Sub tone
        sub = _sine(45, n, self.sr) * np.exp(-np.linspace(0, 1, n) * 4)

        # Pitch drop adds weight
        drop = _exp_sweep(90, 35, n, self.sr) * np.exp(-np.linspace(0, 1, n) * 3) * 0.4

        out = sub * 0.9 + drop

        env = _adsr(n, attack=0.005, decay=0.05, sustain_level=0.7, release=0.5)
        return _normalize(out * env, peak=0.95)

    def horn_stab(self, duration: float = 0.5) -> np.ndarray:
        """
        Air horn / hype stab — sawtooth-like layered sines (200, 400, 800Hz)
        with a square envelope. The "LET'S GO" sound.
        """
        n = int(duration * self.sr)
        t = np.linspace(0, n / self.sr, n, endpoint=False, dtype=np.float32)

        # Build a sawtooth-ish tone with odd harmonics
        out = np.zeros(n, dtype=np.float32)
        harmonics = [(200, 1.0), (400, 0.6), (600, 0.4), (800, 0.3), (1000, 0.2)]
        for f, amp in harmonics:
            out += np.sin(2 * np.pi * f * t).astype(np.float32) * amp

        # Hard envelope: instant on, instant off
        env = _adsr(n, attack=0.005, decay=0.02, sustain_level=0.9, release=0.05)
        out *= env
        return _normalize(out, peak=0.9)

    # ---- aliases --------------------------------------------------------

    def fade(self, duration: float = 0.5) -> np.ndarray:
        return self.shimmer(duration)

    def zoom(self, duration: float = 0.5) -> np.ndarray:
        return self.whoosh(duration)

    def blur(self, duration: float = 0.5) -> np.ndarray:
        return self.boom(duration)

    def slide(self, duration: float = 0.5) -> np.ndarray:
        return self.whoosh(duration)

    def wipe(self, duration: float = 0.5) -> np.ndarray:
        return self.swoosh(duration)

    def glitch(self, duration: float = 0.5) -> np.ndarray:
        c1 = self.click(duration / 3)
        c2 = self.zap(duration / 3)
        c3 = self.click(duration / 3) * 0.6
        out = np.concatenate([c1, c2, c3])
        if len(out) < int(duration * self.sr):
            out = np.concatenate([out, np.zeros(int(duration * self.sr) - len(out), dtype=np.float32)])
        else:
            out = out[:int(duration * self.sr)]
        return _normalize(out, peak=0.8)

    def cinematic_bars(self, duration: float = 0.5) -> np.ndarray:
        return self.chime(duration)

    def lens_flare(self, duration: float = 0.5) -> np.ndarray:
        return self.sparkle(duration)

    def light_leak(self, duration: float = 0.5) -> np.ndarray:
        return self.hiss(duration)

    def film_burn(self, duration: float = 0.5) -> np.ndarray:
        return self.rumble(duration)

    # ---- output ---------------------------------------------------------

    def make_audio_array(self, name: str, duration: float = 0.5, volume: float = 1.0,
                         channels: int = 2) -> np.ndarray:
        """
        Returns a MoviePy-compatible audio array with shape (n_samples, channels).
        """
        mono = self.get(name, duration=duration)
        mono = mono * float(volume)
        if channels == 1:
            return mono.reshape(-1, 1).astype(np.float32)
        return np.stack([mono, mono], axis=-1).astype(np.float32)


# Map each transition name to the SFX it should trigger
SFX_MAP = {
    # Main transitions
    "fade": "shimmer",
    "zoom_in": "whoosh",
    "zoom_out": "whoosh",
    "blur_in": "boom",
    "blur_out": "boom",
    "slide_in": "whoosh",
    "slide_out": "whoosh",
    "wipe_in": "swoosh",
    "wipe_out": "swoosh",
    "glitch_start": "glitch",
    "glitch_end": "glitch",
    "cinematic_bars": "chime",
    # CapCut-style transitions
    "mask_reveal": "sparkle",
    "bounce": "impact",
    "bounce_mask": "impact",
    "split": "swoosh",
    "radial_wipe": "swoosh",
    "color_dissolve": "shimmer",
    "split_wipe": "swoosh",
    "luma_wipe": "shimmer",
    # Cinematic effects
    "lens_flare": "sparkle",
    "light_leak": "hiss",
    "film_burn": "rumble",
    # New viral SFX (callable by name)
    "bass_drop": "bass_drop",
    "riser": "riser",
    "impact": "impact",
    "vinyl_brake": "vinyl_brake",
    "notification": "notification",
    "cinematic_whoosh": "cinematic_whoosh",
    "sub_boom": "sub_boom",
    "horn_stab": "horn_stab",
}


def get_sfx_name(transition_name: str) -> Optional[str]:
    """Returns the SFX name for a given transition, or None if not mapped."""
    return SFX_MAP.get(transition_name)


def list_sfx_names() -> list:
    """Returns all available SFX names."""
    return list(SFX_MAP.keys())


if __name__ == "__main__":
    # Quick test - generate a sample of each
    sfx = TransitionSFX()
    out_dir = Path(__file__).parent / "samples"
    out_dir.mkdir(exist_ok=True)
    try:
        import soundfile as sf
        all_names = list(SFX_MAP.values())
        # de-dup while preserving order
        seen = set()
        unique = []
        for n in all_names:
            if n not in seen:
                seen.add(n)
                unique.append(n)
        for name in unique:
            duration = 0.6 if name not in ("notification", "bass_drop", "riser") else 0.8
            data = getattr(sfx, name)(duration=duration)
            sf.write(str(out_dir / f"{name}.wav"), data, sfx.sr)
            print(f"Generated {name}.wav ({len(data)} samples, peak={np.max(np.abs(data)):.2f})")
    except ImportError:
        print("soundfile not available, skipping write test")
