import pygame
import random
import numpy as np
import sys
import math
import threading
import time
import importlib.util
import os
import json
import zipfile
import tempfile

try:
    import ctypes
    HAS_CTYPES = True
except ImportError:
    HAS_CTYPES = False

try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TK = True
except ImportError:
    HAS_TK = False

# ============================================================
# ===================== USER SETTINGS ========================
# ============================================================

WINDOW_WIDTH   = 1100
WINDOW_HEIGHT  = 680
MAX_ARRAY_SIZE = 128
FPS            = 120

BACKGROUND_COLOR = (5, 5, 10)
ACTIVE_COLOR     = (255, 60, 60)
BAR_SPACING      = 1

ENABLE_SOUND  = True
FREQ_LOW      = 24.0
FREQ_HIGH     = 480.0
FREQ_ABS_LOW  = 12.0
FREQ_ABS_HIGH = 480.0
FREQ_SNAP     = 12
SAMPLE_RATE   = 44100
CHUNK_SIZE    = 512
_last_trigger_time: float = 0.0
TRIGGER_MIN_INTERVAL = 0.035

# ============================================================
# ====================== SOUND SETTINGS ======================
# ============================================================
#
# SOUND_SUSTAIN — how long each tone rings out, in seconds.
#   Higher = longer sustain / more overlap between notes.
#   Good range: 0.08 (snappy) to 0.25 (lush/washy).
SOUND_SUSTAIN = 0.18
#
# SOUND_ATTACK — fade-in time in seconds.
#   A raised-cosine (Hann) window is used instead of a linear ramp,
#   which eliminates the "click" you get at note onset.
#   Formula: env[t] = 0.5 * (1 - cos(pi * t / attack_samples))
SOUND_ATTACK  = 0.012
#
# SOUND_RELEASE — fade-out time in seconds.
#   Same raised-cosine shape applied in reverse at note end.
#   Longer release = smoother, more piano-like tail.
SOUND_RELEASE = 0.060
#
# HARMONIC_BLEND — amount of subtle 2nd harmonic (octave above) mixed in.
#   0.0 = pure sine (very clean). 0.08 = a touch of warmth without buzz.
#   The 2nd harmonic is also a sine, so it adds warmth without harshness.
#   Formula: wave = sin(2pi*f*t) + HARMONIC_BLEND * sin(4pi*f*t)
HARMONIC_BLEND = 0.08
#
# MAX_VOICES — maximum simultaneous oscillators before oldest are culled.
#   Prevents CPU overload when many comparisons fire at once (e.g. bubble sort).
MAX_VOICES = 24
#
# VOICE_STEAL_FADE — when MAX_VOICES is exceeded, stolen voices get this many
#   samples to fade to zero before being removed (avoids clicks on cull).
VOICE_STEAL_FADE = 64

# JSON file saved next to the script for autoloading custom sorters
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
AUTOLOAD_JSON = os.path.join(_SCRIPT_DIR, "supersorter_custom.json")

# ============================================================
# ========================= UI THEME =========================
# ============================================================

UI_BG         = (8,   8,  14)
UI_PANEL      = (14, 14,  22)
UI_PANEL2     = (22, 22,  36)
UI_ACCENT     = (255, 55,  55)
UI_TEXT       = (215, 215, 228)
UI_SUBTEXT    = (105, 105, 130)
UI_HOVER      = (30,  22,  38)
UI_SEL_BG     = (50,  12,  12)
UI_BORDER     = (38,  38,  58)
UI_SEL_BORDER = (255, 55,  55)
UI_DIM        = (60,  60,  80)
UI_HATCH      = (48,  36,  52)
UI_GREEN      = (60, 200, 100)

ALGORITHMS = [
    ("Bubble Sort",     "bubble"),
    ("Insertion Sort",  "insertion"),
    ("Selection Sort",  "selection"),
    ("Quick Sort",      "quick"),
    ("Merge Sort",      "merge"),
    ("Heap Sort",       "heap"),
    ("Shell Sort",      "shell"),
    ("Cocktail Shaker", "cocktail"),
    ("Gnome Sort",      "gnome"),
    ("Comb Sort",       "comb"),
    ("Cycle Sort",      "cycle"),
    ("Pancake Sort",    "pancake"),
    ("Odd-Even Sort",   "oddeven"),
    ("LSD Radix Sort",  "lsd_radix"),
    ("MSD Radix Sort",  "msd_radix"),
]

_custom_generators: dict = {}

# ============================================================
# =================== AUTOLOAD JSON ==========================
# ============================================================

def _save_autoload_json():
    paths = [info["path"] for info in _custom_generators.values() if "path" in info]
    try:
        with open(AUTOLOAD_JSON, "w") as f:
            json.dump({"custom_sorters": paths}, f, indent=2)
    except Exception:
        pass

def _read_autoload_json() -> list:
    if not os.path.exists(AUTOLOAD_JSON):
        return []
    try:
        with open(AUTOLOAD_JSON) as f:
            return json.load(f).get("custom_sorters", [])
    except Exception:
        return []

# ============================================================
# ====================== SOUND ENGINE ========================
# ============================================================
#
# HOW THE SILKY SINE ENGINE WORKS
# ================================
#
# Each note trigger creates an _Osc (oscillator) object.
# Per chunk, all active oscillators are mixed together into one buffer.
#
# WAVEFORM — pure sine + tiny 2nd harmonic (both sines):
#   phase advances by (freq / sample_rate) each sample.
#   wave[t] = sin(2pi * phase[t]) + HARMONIC_BLEND * sin(4pi * phase[t])
#   Using numpy: np.sin(TWO_PI * phases) + HARMONIC_BLEND * np.sin(TWO_PI * 2 * phases)
#
# ENVELOPE — raised-cosine (Hann) shape for attack and release:
#   Attack:  env[t] = 0.5 * (1 - cos(pi * t / A))         t in [0, A)
#   Sustain: env[t] = 1.0                                   t in [A, max_age - R)
#   Release: env[t] = 0.5 * (1 + cos(pi * (t-start) / R))  t in [max_age-R, max_age)
#   This is smoother than a linear ramp and eliminates clicking entirely.
#
# VOICE STEALING — when MAX_VOICES is exceeded:
#   The oldest oscillator is flagged for "early release" — its remaining
#   life is clamped to VOICE_STEAL_FADE samples. The raised-cosine release
#   still applies, so the stolen note fades cleanly with no click.
#
# NORMALIZATION — total output is divided by sqrt(n_voices) (RMS-aware),
#   which keeps perceived loudness roughly constant regardless of how many
#   voices are playing at once. Pure /n gives too-quiet results with few voices;
#   sqrt splits the difference perceptually.

TWO_PI = 2.0 * math.pi


class _Osc:
    """
    Single oscillator voice.

    Attributes
    ----------
    freq      : float  — frequency in Hz
    phase     : float  — current phase in [0, 1), advances by freq/sr each sample
    age       : int    — samples rendered so far
    max_age   : int    — total lifetime in samples
    attack    : int    — attack length in samples (raised-cosine fade-in)
    release   : int    — release length in samples (raised-cosine fade-out)
    """
    __slots__ = ('freq', 'phase', 'age', 'max_age', 'attack', 'release')

    def __init__(self, freq, max_age, attack, release):
        self.freq    = freq
        self.phase   = 0.0
        self.age     = 0
        self.max_age = max_age
        self.attack  = attack
        self.release = release


class SoundEngine:
    def __init__(self):
        self.sample_rate  = SAMPLE_RATE
        self.chunk_size   = CHUNK_SIZE
        self.sustain_smp  = int(SOUND_SUSTAIN  * SAMPLE_RATE)
        self.attack_smp   = max(1, int(SOUND_ATTACK  * SAMPLE_RATE))
        self.release_smp  = max(1, int(SOUND_RELEASE * SAMPLE_RATE))
        self._oscs        = []          # list of active _Osc instances
        self._lock        = threading.Lock()
        self._running     = False
        self._thread      = None
        self._channel     = None

    def start(self):
        self._channel = pygame.mixer.Channel(1)
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._channel:
            self._channel.stop()

    def trigger(self, value: int, max_value: int):
        """
        Fire a new note for the given array value.
        Maps value -> frequency linearly in [FREQ_LOW, FREQ_HIGH].
        If we're at MAX_VOICES, the oldest voice is voice-stolen.
        """
        ratio = value / max_value
        freq  = FREQ_LOW + ratio * (FREQ_HIGH - FREQ_LOW)
        osc   = _Osc(freq, self.sustain_smp, self.attack_smp, self.release_smp)
        with self._lock:
            # Voice steal: clamp oldest oscillator to a fast fade-out
            if len(self._oscs) >= MAX_VOICES:
                oldest = self._oscs[0]
                steal_release   = min(VOICE_STEAL_FADE, oldest.release)
                oldest.max_age  = oldest.age + steal_release
                oldest.release  = steal_release
            self._oscs.append(osc)

    def _gen_chunk(self) -> np.ndarray:
        """
        Synthesise one chunk of audio and return it as a float64 array.

        Step-by-step per oscillator:
          1. Compute per-sample phase array (vectorised with numpy)
          2. Compute sine wave + optional 2nd harmonic
          3. Compute raised-cosine envelope (attack / sustain / release)
          4. Accumulate wave * envelope into the output buffer
          5. Advance oscillator state (phase, age)
          6. Remove finished oscillators
        """
        buf = np.zeros(self.chunk_size, dtype=np.float64)
        # Local sample indices within this chunk: [0, 1, 2, ..., CHUNK_SIZE-1]
        idx = np.arange(self.chunk_size, dtype=np.float64)

        with self._lock:
            alive = []
            for o in self._oscs:
                # Absolute sample age for each position in this chunk
                abs_age = idx + o.age

                # ---- WAVEFORM ----
                # Phase at each sample: (current_phase + sample_offset * freq/sr) mod 1
                phases = (o.phase + idx * (o.freq / self.sample_rate)) % 1.0
                # Pure sine (phase in [0,1] maps to angle in [0, 2pi])
                wave = np.sin(TWO_PI * phases)
                # Subtle 2nd harmonic — still a sine, adds warmth not buzz
                if HARMONIC_BLEND > 0.0:
                    wave += HARMONIC_BLEND * np.sin(TWO_PI * 2.0 * phases)

                # ---- ENVELOPE ----
                env = np.ones(self.chunk_size, dtype=np.float64)

                # Attack: raised-cosine fade-in
                #   env = 0.5 * (1 - cos(pi * t / A))
                #   At t=0: env=0, at t=A: env=1 — smooth, click-free onset
                a_mask = abs_age < o.attack
                if np.any(a_mask):
                    env[a_mask] = 0.5 * (1.0 - np.cos(math.pi * abs_age[a_mask] / o.attack))

                # Release: raised-cosine fade-out (mirror of attack)
                #   env = 0.5 * (1 + cos(pi * (t - rel_start) / R))
                #   At t=rel_start: env=1, at t=max_age: env=0 — smooth tail
                rel_start = o.max_age - o.release
                r_mask = abs_age >= rel_start
                if np.any(r_mask):
                    env[r_mask] = 0.5 * (1.0 + np.cos(
                        math.pi * (abs_age[r_mask] - rel_start) / o.release
                    ))
                    env[r_mask] = np.maximum(0.0, env[r_mask])

                # Zero out anything past the end (safety net)
                env[abs_age >= o.max_age] = 0.0

                buf += wave * env

                # Advance oscillator state for next chunk
                o.phase = (o.phase + self.chunk_size * (o.freq / self.sample_rate)) % 1.0
                o.age  += self.chunk_size

                if o.age < o.max_age:
                    alive.append(o)

            self._oscs = alive
            n_voices = max(1, len(alive))

        # RMS-aware normalisation: divide by sqrt(n) keeps perceived loudness stable
        # (pure /n is too quiet with few voices; sqrt is a perceptual sweet spot)
        buf /= math.sqrt(n_voices) * (1.0 + HARMONIC_BLEND)
        return buf

    def _loop(self):
        """
        Audio thread: generates chunks and queues them to the pygame mixer channel.
        Runs slightly ahead of playback to avoid gaps (queue up to 4 chunks).
        """
        chunk_secs = self.chunk_size / self.sample_rate
        while self._running:
            mono   = self._gen_chunk()
            # Convert float [-1,1] to int16, scale to 85% of max for headroom
            pcm    = (np.clip(mono, -1.0, 1.0) * 32767 * 0.85).astype(np.int16)
            # Duplicate mono to stereo (L/R identical)
            stereo = np.column_stack((pcm, pcm))
            snd    = pygame.mixer.Sound(buffer=stereo.tobytes())
            deadline = time.monotonic() + chunk_secs * 4
            while self._channel.get_queue() is not None and self._running:
                time.sleep(0.001)
                if time.monotonic() > deadline:
                    break
            if self._running:
                self._channel.queue(snd)
            time.sleep(chunk_secs * 0.75)


_engine: SoundEngine | None = None


def init_sound():
    global _engine
    pygame.mixer.pre_init(SAMPLE_RATE, -16, 2, CHUNK_SIZE)
    pygame.mixer.init()
    _engine = SoundEngine()
    _engine.start()


def stop_sound():
    global _engine
    if _engine:
        _engine.stop()
        _engine = None


def trigger_tone(value: int, max_value: int):
    global _last_trigger_time
    if not (_engine and ENABLE_SOUND):
        return
    now = time.monotonic()
    if now - _last_trigger_time >= TRIGGER_MIN_INTERVAL:
        _engine.trigger(value, max_value)
        _last_trigger_time = now


# ============================================================
# ======================= COLOR / DRAW =======================
# ============================================================

def value_to_color(value, max_value):
    r = value / max_value
    if r < 0.25: return (0, int(255 * r * 4), 255)
    if r < 0.5:  return (0, 255, int(255 * (1 - (r - 0.25) * 4)))
    if r < 0.75: return (int(255 * (r - 0.5) * 4), 255, 0)
    return (255, int(255 * (1 - (r - 0.75) * 4)), 0)


def draw_bars(screen, array, active_indices, label=""):
    screen.fill(BACKGROUND_COLOR)
    n  = len(array)
    bw = WINDOW_WIDTH / n
    for i, v in enumerate(array):
        h = (v / n) * (WINDOW_HEIGHT - 60)
        c = ACTIVE_COLOR if i in active_indices else value_to_color(v, n)
        pygame.draw.rect(screen, c, (i * bw, WINDOW_HEIGHT - h, bw - BAR_SPACING, h))
    if label:
        f = pygame.font.SysFont("consolas", 18)
        screen.blit(f.render(label, True, (140, 140, 160)), (12, 10))
    pygame.display.flip()

# ============================================================
# ===================== SORTING ALGORITHMS ===================
# ============================================================

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            yield arr, [j, j+1]
            if arr[j] > arr[j+1]: arr[j], arr[j+1] = arr[j+1], arr[j]; yield arr, [j, j+1]

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]; j = i - 1
        while j >= 0 and arr[j] > key:
            yield arr, [j, j+1]; arr[j+1] = arr[j]; j -= 1; yield arr, [j+1]
        arr[j+1] = key; yield arr, [j+1]

def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        mi = i
        for j in range(i+1, n):
            yield arr, [mi, j]
            if arr[j] < arr[mi]: mi = j; yield arr, [mi]
        arr[i], arr[mi] = arr[mi], arr[i]; yield arr, [i, mi]

def quick_sort(arr):
    def _q(lo, hi):
        if lo >= hi: return
        pivot = arr[hi]; i = lo - 1
        for j in range(lo, hi):
            yield arr, [j, hi]
            if arr[j] <= pivot: i += 1; arr[i], arr[j] = arr[j], arr[i]; yield arr, [i, j]
        arr[i+1], arr[hi] = arr[hi], arr[i+1]; yield arr, [i+1, hi]
        yield from _q(lo, i); yield from _q(i+2, hi)
    yield from _q(0, len(arr)-1)

def merge_sort(arr):
    def _m(lo, mid, hi):
        L = arr[lo:mid+1][:]; R = arr[mid+1:hi+1][:]
        i = j = 0; k = lo
        while i < len(L) and j < len(R):
            yield arr, [lo+i, mid+1+j]
            if L[i] <= R[j]: arr[k] = L[i]; i += 1
            else:             arr[k] = R[j]; j += 1
            yield arr, [k]; k += 1
        while i < len(L): arr[k] = L[i]; yield arr, [k]; i += 1; k += 1
        while j < len(R): arr[k] = R[j]; yield arr, [k]; j += 1; k += 1
    def _ms(lo, hi):
        if lo < hi:
            mid = (lo+hi)//2; yield from _ms(lo, mid); yield from _ms(mid+1, hi)
            yield from _m(lo, mid, hi)
    yield from _ms(0, len(arr)-1)

def heap_sort(arr):
    def hfy(n, i):
        lg, l, r = i, 2*i+1, 2*i+2
        if l < n and arr[l] > arr[lg]: lg = l
        if r < n and arr[r] > arr[lg]: lg = r
        if lg != i: arr[i], arr[lg] = arr[lg], arr[i]; yield arr, [i, lg]; yield from hfy(n, lg)
        else: yield arr, [i]
    n = len(arr)
    for i in range(n//2-1, -1, -1): yield from hfy(n, i)
    for i in range(n-1, 0, -1): arr[0], arr[i] = arr[i], arr[0]; yield arr, [0, i]; yield from hfy(i, 0)

def shell_sort(arr):
    n, gap = len(arr), len(arr)//2
    while gap > 0:
        for i in range(gap, n):
            t = arr[i]; j = i
            while j >= gap and arr[j-gap] > t: yield arr, [j, j-gap]; arr[j] = arr[j-gap]; j -= gap
            arr[j] = t; yield arr, [j]
        gap //= 2

def cocktail_sort(arr):
    lo, hi = 0, len(arr)-1
    while lo < hi:
        for i in range(lo, hi):
            yield arr, [i, i+1]
            if arr[i] > arr[i+1]: arr[i], arr[i+1] = arr[i+1], arr[i]; yield arr, [i, i+1]
        hi -= 1
        for i in range(hi, lo, -1):
            yield arr, [i, i-1]
            if arr[i] < arr[i-1]: arr[i], arr[i-1] = arr[i-1], arr[i]; yield arr, [i, i-1]
        lo += 1

def gnome_sort(arr):
    i = 0
    while i < len(arr):
        yield arr, [i]
        if i == 0 or arr[i] >= arr[i-1]: i += 1
        else: arr[i], arr[i-1] = arr[i-1], arr[i]; yield arr, [i, i-1]; i -= 1

def comb_sort(arr):
    n, gap, shrink = len(arr), len(arr), 1.3; s = False
    while not s:
        gap = int(gap/shrink)
        if gap <= 1: gap = 1; s = True
        for i in range(n-gap):
            yield arr, [i, i+gap]
            if arr[i] > arr[i+gap]: arr[i], arr[i+gap] = arr[i+gap], arr[i]; s = False; yield arr, [i, i+gap]

def cycle_sort(arr):
    n = len(arr)
    for cs in range(n-1):
        item = arr[cs]; pos = cs
        for i in range(cs+1, n):
            yield arr, [i, cs]
            if arr[i] < item: pos += 1
        if pos == cs: continue
        while item == arr[pos]: pos += 1
        arr[pos], item = item, arr[pos]; yield arr, [pos, cs]
        while pos != cs:
            pos = cs
            for i in range(cs+1, n):
                yield arr, [i, cs]
                if arr[i] < item: pos += 1
            while item == arr[pos]: pos += 1
            arr[pos], item = item, arr[pos]; yield arr, [pos]

def pancake_sort(arr):
    def flip(k):
        lo, hi = 0, k
        while lo < hi: arr[lo], arr[hi] = arr[hi], arr[lo]; yield arr, [lo, hi]; lo += 1; hi -= 1
    for sz in range(len(arr), 1, -1):
        mi = arr.index(max(arr[:sz]))
        if mi != sz-1:
            if mi != 0: yield from flip(mi)
            yield from flip(sz-1)

def odd_even_sort(arr):
    n = len(arr); s = False
    while not s:
        s = True
        for i in range(1, n-1, 2):
            yield arr, [i, i+1]
            if arr[i] > arr[i+1]: arr[i], arr[i+1] = arr[i+1], arr[i]; s = False; yield arr, [i, i+1]
        for i in range(0, n-1, 2):
            yield arr, [i, i+1]
            if arr[i] > arr[i+1]: arr[i], arr[i+1] = arr[i+1], arr[i]; s = False; yield arr, [i, i+1]

def _counting_radix(arr, exp, base):
    n = len(arr); out = [0]*n; cnt = [0]*base
    for i in range(n): cnt[(arr[i]//exp)%base] += 1; yield arr, [i]
    for i in range(1, base): cnt[i] += cnt[i-1]
    for i in range(n-1, -1, -1):
        idx = (arr[i]//exp)%base; out[cnt[idx]-1] = arr[i]; cnt[idx] -= 1; yield arr, [i]
    for i in range(n): arr[i] = out[i]; yield arr, [i]

def lsd_radix_sort(arr, base=10):
    mv, exp = max(arr), 1
    while mv//exp > 0: yield from _counting_radix(arr, exp, base); exp *= base

def msd_radix_sort(arr, base=10):
    def helper(lo, hi, exp):
        if hi-lo <= 1 or exp == 0: return
        bkts = [[] for _ in range(base)]
        for i in range(lo, hi): bkts[(arr[i]//exp)%base].append(arr[i]); yield arr, [i]
        i = lo
        for bk in bkts:
            for v in bk: arr[i] = v; i += 1
            if bk: yield arr, list(range(i-len(bk), i))
        i = lo
        for bk in bkts:
            if len(bk) > 1: yield from helper(i, i+len(bk), exp//base)
            i += len(bk)
    if not arr: return
    mv, exp = max(arr), 1
    while mv//exp >= base: exp *= base
    yield from helper(0, len(arr), exp)

def get_generator(key, arr, radix_base):
    builtins = {
        "bubble":    lambda: bubble_sort(arr),
        "insertion": lambda: insertion_sort(arr),
        "selection": lambda: selection_sort(arr),
        "quick":     lambda: quick_sort(arr),
        "merge":     lambda: merge_sort(arr),
        "heap":      lambda: heap_sort(arr),
        "shell":     lambda: shell_sort(arr),
        "cocktail":  lambda: cocktail_sort(arr),
        "gnome":     lambda: gnome_sort(arr),
        "comb":      lambda: comb_sort(arr),
        "cycle":     lambda: cycle_sort(arr),
        "pancake":   lambda: pancake_sort(arr),
        "oddeven":   lambda: odd_even_sort(arr),
        "lsd_radix": lambda: lsd_radix_sort(arr, radix_base),
        "msd_radix": lambda: msd_radix_sort(arr, radix_base),
    }
    if key in builtins: return builtins[key]()
    if key in _custom_generators: return _custom_generators[key]["fn"](arr)
    raise KeyError(f"Unknown key: {key}")

# ============================================================
# ==================== CUSTOM SORTER LOADER ==================
# ============================================================

def load_custom_sorter(filepath: str):
    """
    Load a .py file as a custom sorter.
    Must define: NAME (str, optional) and sort(arr) generator.
    Returns ((display_name, key), None) on success, (None, error_str) on failure.
    """
    try:
        filepath = os.path.abspath(filepath)
        spec   = importlib.util.spec_from_file_location("_cs", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "sort"):
            return None, "No sort(arr) function found"
        fn   = module.sort
        name = getattr(module, "NAME", os.path.splitext(os.path.basename(filepath))[0])
        key  = f"custom_{len(_custom_generators)}"
        _custom_generators[key] = {"fn": fn, "path": filepath}
        return (name, key), None
    except Exception as e:
        return None, str(e)

def open_file_dialog():
    if not HAS_TK: return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Load Custom Sorter / SortPack",
        filetypes=[
            ("Python files", "*.py"),
            ("SortPack ZIP files", "*.zip"),
            ("All files", "*.*")
        ]
    )
    root.destroy()
    return path if path else None

def autoload_custom_sorters():
    """On startup: load all paths listed in the JSON, silently skip missing/broken ones."""
    paths = _read_autoload_json()
    for path in paths:
        if os.path.exists(path):
            result, err = load_custom_sorter(path)
            if result:
                name, key = result
                if not any(k == key for _, k in ALGORITHMS):
                    ALGORITHMS.append((name, key))

def load_sortpack(zip_path):
    loaded = []
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(tmpdir)
            for root, dirs, files in os.walk(tmpdir):
                for fname in files:
                    if fname.endswith(".py"):
                        fpath = os.path.join(root, fname)
                        result, err = load_custom_sorter(fpath)
                        if result:
                            name, key = result
                            loaded.append((name, key))
    except Exception as e:
        print(f"SortPack load error: {e}")
    return loaded

if sys.platform == "win32":
    APPDATA = os.getenv("APPDATA") or os.path.expanduser("~\\AppData\\Roaming")
    SORTPACK_DIR = os.path.join(APPDATA, "SuperSorter", "SortPacks")
else:
    SORTPACK_DIR = os.path.expanduser("~/.supersorter/sortpacks")

os.makedirs(SORTPACK_DIR, exist_ok=True)

# ============================================================
# ========================= LAYOUT CONSTANTS =================
# ============================================================

COLS    = 3
BTN_W   = 220
BTN_H   = 42
BTN_GAP = 5
COL_GAP = 8
PAD     = 16
GRID_W  = COLS * BTN_W + (COLS-1) * COL_GAP
RX      = PAD + GRID_W + 18
RW      = WINDOW_WIDTH - RX - PAD

_Y_SETTINGS = 88
_Y_SIZE     = 108
_Y_SPEED    = 162
_Y_FREQ     = 216
_Y_RADIX    = 284

# ============================================================
# ========================= UI WIDGETS =======================
# ============================================================

class Slider:
    """Single-knob slider with snapping and smaller knob."""
    KNOB_RADIUS = 6

    def __init__(self, x, y, w, lo, hi, val, label, is_int=False, snap=None):
        self.x, self.y, self.w = x, y, w
        self.lo, self.hi = lo, hi
        self.value = val
        self.label = label
        self.is_int = is_int
        self.drag = False
        self.snap = snap
        self.track = pygame.Rect(x, y+18, w, 4)
        self.hit = pygame.Rect(x-5, y, w+10, 38)

    def _r(self):
        return (self.value - self.lo) / (self.hi - self.lo)

    def _kx(self):
        return int(self.x + self._r() * self.w)

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if math.hypot(ev.pos[0]-self._kx(), ev.pos[1]-self.track.centery) < 14 \
               or self.hit.collidepoint(ev.pos):
                self.drag = True; self._set(ev.pos[0])
        elif ev.type == pygame.MOUSEBUTTONUP:
            self.drag = False
        elif ev.type == pygame.MOUSEMOTION and self.drag:
            self._set(ev.pos[0])

    def _set(self, mx):
        r = max(0.0, min(1.0, (mx - self.x) / self.w))
        raw = self.lo + r * (self.hi - self.lo)
        if self.snap: raw = round(raw / self.snap) * self.snap
        self.value = int(round(raw)) if self.is_int else round(raw * 4) / 4

    def draw(self, s, fonts):
        vs = str(self.value) if self.is_int else f"{self.value:.2f}x"
        s.blit(fonts['small'].render(f"{self.label}:  {vs}", True, UI_SUBTEXT), (self.x, self.y))
        pygame.draw.rect(s, UI_BORDER, self.track, border_radius=2)
        fw = int(self._r() * self.w)
        if fw > 0: pygame.draw.rect(s, UI_ACCENT, (self.x, self.track.y, fw, 4), border_radius=2)
        kx, ky = self._kx(), self.track.centery
        pygame.draw.circle(s, UI_PANEL2, (kx, ky), self.KNOB_RADIUS)
        pygame.draw.circle(s, UI_ACCENT, (kx, ky), self.KNOB_RADIUS, 2)
        pygame.draw.circle(s, UI_ACCENT, (kx, ky), 2)


class FreqRangeSlider:
    """Dual-knob frequency slider with snap and smaller knobs."""
    KNOB_RADIUS = 6

    def __init__(self, x, y, w, lo_val, hi_val):
        self.x, self.y, self.w = x, y, w
        self.lo_val = lo_val
        self.hi_val = hi_val
        self.drag_lo = False
        self.drag_hi = False
        self.track = pygame.Rect(x, y+18, w, 4)

    @property
    def height(self): return 44

    def _snap(self, freq):
        return round(freq / FREQ_SNAP) * FREQ_SNAP

    def _to_x(self, freq):
        r = (freq - FREQ_ABS_LOW) / (FREQ_ABS_HIGH - FREQ_ABS_LOW)
        return int(self.x + r * self.w)

    def _to_freq(self, px):
        r = max(0.0, min(1.0, (px - self.x) / self.w))
        raw = FREQ_ABS_LOW + r * (FREQ_ABS_HIGH - FREQ_ABS_LOW)
        return float(self._snap(raw))

    @property
    def lo_x(self): return self._to_x(self.lo_val)
    @property
    def hi_x(self): return self._to_x(self.hi_val)

    def handle(self, ev):
        ky = self.track.centery
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            lx, hx = self.lo_x, self.hi_x
            dl = math.hypot(ev.pos[0]-lx, ev.pos[1]-ky)
            dh = math.hypot(ev.pos[0]-hx, ev.pos[1]-ky)
            if dl < 14 and dl <= dh: self.drag_lo = True
            elif dh < 14:            self.drag_hi = True
        elif ev.type == pygame.MOUSEBUTTONUP:
            self.drag_lo = self.drag_hi = False
        elif ev.type == pygame.MOUSEMOTION:
            if self.drag_lo:
                self.lo_val = max(FREQ_ABS_LOW,
                                  min(self._to_freq(ev.pos[0]), self.hi_val - FREQ_SNAP))
            if self.drag_hi:
                self.hi_val = min(FREQ_ABS_HIGH,
                                  max(self._to_freq(ev.pos[0]), self.lo_val + FREQ_SNAP))

    def draw(self, s, fonts):
        tx, ty, tw = self.track.x, self.track.y, self.track.width
        lx, hx = self.lo_x, self.hi_x
        pygame.draw.rect(s, UI_BORDER, self.track, border_radius=2)
        if lx > tx:
            pygame.draw.rect(s, (20,15,28), pygame.Rect(tx, ty-2, lx-tx, 8))
        if hx > lx: pygame.draw.rect(s, UI_ACCENT, (lx, ty, hx-lx, 4))
        for kx2 in (lx, hx):
            pygame.draw.circle(s, UI_PANEL2, (kx2, ty+2), self.KNOB_RADIUS)
            pygame.draw.circle(s, UI_ACCENT, (kx2, ty+2), self.KNOB_RADIUS, 2)
            pygame.draw.circle(s, UI_ACCENT, (kx2, ty+2), 2)
        lbl  = fonts['small'].render("Freq Range", True, UI_SUBTEXT)
        info = fonts['mono_sm'].render(f"Min: {self.lo_val:.0f} Hz   Max: {self.hi_val:.0f} Hz", True, UI_SUBTEXT)
        s.blit(lbl, (self.x, self.y))
        s.blit(info, (self.x, self.y + 30))


class AlgoBtn:
    H = BTN_H
    def __init__(self, x, y, w, name, key, idx):
        self.rect = pygame.Rect(x, y, w, self.H)
        self.name, self.key, self.idx = name, key, idx

    def draw(self, s, fonts, sel, hov):
        bg = UI_SEL_BG if sel else (UI_HOVER if hov else UI_PANEL)
        br = UI_SEL_BORDER if sel else (UI_DIM if hov else UI_BORDER)
        pygame.draw.rect(s, bg, self.rect, border_radius=5)
        pygame.draw.rect(s, br, self.rect, 1, border_radius=5)
        nc = UI_ACCENT if sel else UI_SUBTEXT
        tc = UI_TEXT   if sel else (UI_TEXT if hov else (150, 150, 170))
        s.blit(fonts['mono_sm'].render(f"{self.idx+1:02d}", True, nc),
               (self.rect.x+10, self.rect.y+14))
        nm = self.name if len(self.name) <= 18 else self.name[:16]+".."
        s.blit(fonts['mid'].render(nm, True, tc), (self.rect.x+40, self.rect.y+12))


class SmBtn:
    def __init__(self, x, y, w, h, lbl):
        self.rect = pygame.Rect(x, y, w, h); self.label = lbl
    def draw(self, s, fonts, act=False, hov=False):
        bg = UI_ACCENT if act else (UI_HOVER if hov else UI_PANEL2)
        fc = (0, 0, 0) if act else UI_TEXT
        pygame.draw.rect(s, bg,        self.rect, border_radius=5)
        pygame.draw.rect(s, UI_BORDER, self.rect, 1, border_radius=5)
        t = fonts['small'].render(self.label, True, fc)
        s.blit(t, t.get_rect(center=self.rect.center))


# ============================================================
# ========================= MENU =============================
# ============================================================

class Menu:
    def __init__(self, screen, fonts):
        self.screen    = screen
        self.fonts     = fonts
        self.sel       = 0
        self.hov       = -1
        self.sound_on  = ENABLE_SOUND
        self.msg       = ""
        self.msg_until = 0
        self.msg_ok    = True

        self._build_btns()

        self.sl_size  = Slider(RX, _Y_SIZE,  RW, 4, MAX_ARRAY_SIZE, 32, "Array Size", is_int=True)
        self.sl_speed = Slider(RX, _Y_SPEED, RW, 0.25, 8.0, 1.0, "Speed")
        self.sl_freq  = FreqRangeSlider(RX, _Y_FREQ, RW, FREQ_LOW, FREQ_HIGH)

        bases = [2, 4, 8, 10, 16]
        bw2   = (RW - (len(bases)-1)*5) // len(bases)
        self.radix_base = 10
        self.base_btns  = []
        for i, b in enumerate(bases):
            self.base_btns.append((SmBtn(RX + i*(bw2+5), _Y_RADIX+16, bw2, 26, f"B{b}"), b))

        self.start_rect = pygame.Rect(RX, WINDOW_HEIGHT-62, RW, 46)
        self.start_hov  = False

    def _build_btns(self):
        gx, gy = PAD, 80
        self.btns = []
        for i, (nm, ky) in enumerate(ALGORITHMS):
            col = i % COLS; row = i // COLS
            x = gx + col*(BTN_W+COL_GAP)
            y = gy + row*(BTN_H+BTN_GAP)
            self.btns.append(AlgoBtn(x, y, BTN_W, nm, ky, i))

        rows_used  = (len(ALGORITHMS) + COLS - 1) // COLS
        load_y     = gy + rows_used*(BTN_H+BTN_GAP) + 4
        self.load_btn = SmBtn(gx, load_y, GRID_W, 30, "[+] Load Custom Sorter")

    def _is_radix(self):
        return ALGORITHMS[self.sel][1] in ("lsd_radix", "msd_radix")

    def _notify(self, msg, ok=True):
        self.msg = msg; self.msg_ok = ok
        self.msg_until = pygame.time.get_ticks() + 4000

    def _y_sound(self):
        base = _Y_FREQ + self.sl_freq.height + 10
        if self._is_radix():
            base = _Y_RADIX + 16 + 26 + 10
        return base

    def handle(self, ev):
        self.sl_size.handle(ev)
        self.sl_speed.handle(ev)
        self.sl_freq.handle(ev)

        if ev.type == pygame.MOUSEMOTION:
            self.hov = -1
            self.start_hov = self.start_rect.collidepoint(ev.pos)
            for b in self.btns:
                if b.rect.collidepoint(ev.pos): self.hov = b.idx

        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for b in self.btns:
                if b.rect.collidepoint(ev.pos): self.sel = b.idx

            if self._is_radix():
                for sb, base in self.base_btns:
                    if sb.rect.collidepoint(ev.pos): self.radix_base = base

            snd_rect = pygame.Rect(RX, self._y_sound(), RW, 30)
            if snd_rect.collidepoint(ev.pos):
                self.sound_on = not self.sound_on

            if self.load_btn.rect.collidepoint(ev.pos):
                self._do_load()

            if self.start_rect.collidepoint(ev.pos):
                return "start"

        return None

    def _do_load(self):
        path = open_file_dialog()
        if not path: return

        if path.lower().endswith(".py"):
            result, err = load_custom_sorter(path)
            if err:
                self._notify(f"Error: {err[:55]}", ok=False)
                return
            name, key = result
            if not any(k == key for _, k in ALGORITHMS):
                ALGORITHMS.append((name, key))
            _save_autoload_json()
            self._build_btns()
            self.sel = len(ALGORITHMS) - 1
            self._notify(f"Loaded: {name}", ok=True)

        elif path.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(path, "r") as z:
                    md_files = [f for f in z.namelist() if f.lower().endswith(".txt")]
                    sortpack_name = os.path.splitext(os.path.basename(path))[0]
                    if md_files:
                        md_content = z.read(md_files[0]).decode("utf-8")
                        first_line = md_content.strip().splitlines()[0]
                        if first_line:
                            sortpack_name = first_line.strip()

                    temp_dir = os.path.join(_SCRIPT_DIR, "__sortpack_temp__")
                    os.makedirs(temp_dir, exist_ok=True)
                    py_files = [f for f in z.namelist() if f.lower().endswith(".py")]
                    for f in py_files:
                        z.extract(f, temp_dir)
                        full_path = os.path.join(temp_dir, f)
                        result, err = load_custom_sorter(full_path)
                        if result:
                            name, key = result
                            if not any(k == key for _, k in ALGORITHMS):
                                ALGORITHMS.append((name, key))

                    _save_autoload_json()
                    self._build_btns()
                    self.sel = len(ALGORITHMS) - 1
                    self._notify(f"Loaded SortPack: {sortpack_name}", ok=True)

            except Exception as e:
                self._notify(f"Failed to load SortPack: {str(e)[:50]}", ok=False)

    def draw(self):
        s  = self.screen
        mp = pygame.mouse.get_pos()
        s.fill(UI_BG)

        t1 = self.fonts['title'].render("SuperSorter", True, UI_TEXT)
        t2 = self.fonts['title'].render("SuperSorter", True, UI_ACCENT)
        s.blit(t2, (PAD+1, 23)); s.blit(t1, (PAD, 22))
        s.blit(self.fonts['small'].render(f"{len(ALGORITHMS)} algorithms", True, UI_SUBTEXT),
               (PAD + t1.get_width() + 12, 31))
        pygame.draw.line(s, UI_BORDER, (PAD, 68), (WINDOW_WIDTH-PAD, 68), 1)

        for b in self.btns: b.draw(s, self.fonts, b.idx==self.sel, b.idx==self.hov)

        self.load_btn.draw(s, self.fonts, False, self.load_btn.rect.collidepoint(mp))

        now = pygame.time.get_ticks()
        if self.msg and now < self.msg_until:
            col = UI_GREEN if self.msg_ok else (255, 90, 90)
            s.blit(self.fonts['small'].render(self.msg, True, col),
                   (PAD, self.load_btn.rect.bottom + 6))
        elif now >= self.msg_until:
            self.msg = ""

        panel = pygame.Rect(RX-10, 76, RW+20, WINDOW_HEIGHT-82)
        pygame.draw.rect(s, UI_PANEL,  panel, border_radius=7)
        pygame.draw.rect(s, UI_BORDER, panel, 1, border_radius=7)
        s.blit(self.fonts['small'].render("SETTINGS", True, UI_SUBTEXT), (RX, _Y_SETTINGS))

        self.sl_size.draw(s, self.fonts)
        self.sl_speed.draw(s, self.fonts)
        self.sl_freq.draw(s, self.fonts)

        if self._is_radix():
            s.blit(self.fonts['small'].render("Radix Base:", True, UI_SUBTEXT),
                   (RX, _Y_RADIX))
            for sb, base in self.base_btns:
                sb.draw(s, self.fonts, self.radix_base==base, sb.rect.collidepoint(mp))

        y_snd = self._y_sound()
        snd_rect = pygame.Rect(RX, y_snd, RW, 30)
        snd_lbl  = "Sound: ON" if self.sound_on else "Sound: OFF"
        bg  = UI_ACCENT if self.sound_on else UI_PANEL2
        fc  = (0, 0, 0) if self.sound_on else UI_TEXT
        hov = snd_rect.collidepoint(mp)
        if not self.sound_on and hov: bg = UI_HOVER
        pygame.draw.rect(s, bg,        snd_rect, border_radius=5)
        pygame.draw.rect(s, UI_BORDER, snd_rect, 1, border_radius=5)
        st = self.fonts['small'].render(snd_lbl, True, fc)
        s.blit(st, st.get_rect(center=snd_rect.center))

        nm, _ = ALGORITHMS[self.sel]
        y_sel = y_snd + 38
        s.blit(self.fonts['small'].render("Selected:", True, UI_SUBTEXT), (RX, y_sel))
        s.blit(self.fonts['mid'].render(nm, True, UI_ACCENT), (RX, y_sel+16))
        s.blit(self.fonts['small'].render("ESC during sort returns to menu", True, UI_DIM),
               (RX, y_sel+36))

        sh = self.start_hov
        pygame.draw.rect(s, (240,50,50) if sh else (200,35,35), self.start_rect, border_radius=7)
        pygame.draw.rect(s, (255,90,90) if sh else UI_ACCENT,   self.start_rect, 2, border_radius=7)
        st2 = self.fonts['big'].render("> START", True, (255, 255, 255))
        s.blit(st2, st2.get_rect(center=self.start_rect.center))

        pygame.display.flip()

    def config(self):
        nm, ky = ALGORITHMS[self.sel]
        return dict(
            name=nm, key=ky,
            size=self.sl_size.value,
            speed=self.sl_speed.value,
            freq_lo=self.sl_freq.lo_val,
            freq_hi=self.sl_freq.hi_val,
            radix_base=self.radix_base,
            sound=self.sound_on,
        )

# ============================================================
# ========================= MAIN =============================
# ============================================================

def force_top():
    if HAS_CTYPES and sys.platform == "win32":
        try:
            hwnd = pygame.display.get_wm_info()['window']
            ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001|0x0002)
        except Exception: pass

def build_fonts():
    def tf(names, sz):
        for n in names:
            try: return pygame.font.SysFont(n, sz)
            except: pass
        return pygame.font.SysFont(None, sz)
    mono = ["Consolas", "Courier New", "Lucida Console"]
    sans = ["Segoe UI", "Tahoma", "Arial"]
    return dict(title=tf(mono, 26), big=tf(sans, 22), mid=tf(sans, 17),
                small=tf(sans, 13), mono_sm=tf(mono, 12))

def run_sort(screen, fonts, cfg):
    global ENABLE_SOUND, FREQ_LOW, FREQ_HIGH
    ENABLE_SOUND = cfg["sound"]
    FREQ_LOW     = cfg["freq_lo"]
    FREQ_HIGH    = cfg["freq_hi"]

    arr = list(range(1, cfg["size"]+1)); random.shuffle(arr)
    gen = get_generator(cfg["key"], arr, cfg["radix_base"])
    clock = pygame.time.Clock(); label = cfg["name"]

    while True:
        clock.tick(FPS * cfg["speed"])
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: stop_sound(); pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE: return
        try:
            state, active = next(gen)
            draw_bars(screen, state, active, label)
            if active and ENABLE_SOUND:
                trigger_tone(state[active[0]], cfg["size"])
        except StopIteration:
            draw_bars(screen, arr, [], label + "  [SORTED]")
            pygame.time.wait(1800); return

def main():
    pygame.init(); init_sound()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("SuperSorter")
    force_top()
    fonts = build_fonts(); clock = pygame.time.Clock()

    autoload_custom_sorters()

    while True:
        menu = Menu(screen, fonts)
        while True:
            clock.tick(60)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT: stop_sound(); pygame.quit(); sys.exit()
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    stop_sound(); pygame.quit(); sys.exit()
                if menu.handle(ev) == "start":
                    run_sort(screen, fonts, menu.config()); break
            else:
                menu.draw(); continue
            break

if __name__ == "__main__":
    main()