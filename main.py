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

# ============================================================
# ====================== SOUND SETTINGS ======================
# ============================================================
#
# SOUND_SUSTAIN — how long each triggered tone rings out, in seconds.
#   This is the note's total lifetime from trigger to silence.
#   Good range: 0.15 (snappy) to 0.40 (lush, washy overlap).
SOUND_SUSTAIN  = 0.25
#
# SOUND_ATTACK — fade-in time in seconds using a raised-cosine curve.
#   Raised-cosine (Hann) means the waveform eases in smoothly rather
#   than jumping from zero — eliminates the sharp "click" on note onset.
#   Formula per sample t in [0, A): env = 0.5 * (1 - cos(pi * t / A))
SOUND_ATTACK   = 0.015
#
# SOUND_RELEASE — fade-out time in seconds, same raised-cosine shape.
#   The tail of each note eases out rather than cutting abruptly.
#   Formula per sample t in [max_age-R, max_age): env = 0.5*(1+cos(pi*(t-rel_start)/R))
SOUND_RELEASE  = 0.120
#
# HARMONIC_BLEND — mix of a subtle 2nd harmonic (one octave up) into the sine.
#   0.0 = pure crystal sine. 0.12 = gentle warmth. 0.25 = noticeably richer.
#   Both fundamentals and harmonic are pure sines so no harshness is introduced.
HARMONIC_BLEND  = 0.12
#
# HARMONIC_BLEND_3 — mix of a 3rd harmonic (octave + a fifth up).
#   This is what adds "crunch" — the same partial that makes organs and
#   music boxes sound gritty and interesting rather than pure and glassy.
#   0.0 = off. 0.06 = subtle grit. 0.15 = noticeably crunchy.
HARMONIC_BLEND_3 = 0.07
#
# MAX_VOICES — hard cap on simultaneous oscillators.
#   When exceeded the oldest voice is smoothly faded out (voice stealing).
MAX_VOICES     = 32
#
# TRIGGER_RATE_LIMIT — minimum seconds between accepting new note triggers.
#   This is the key to sustain feeling: without it, the sort loop fires hundreds
#   of triggers per second, each note barely starting before the next crowds it.
#   With a gate, notes have breathing room to bloom, sustain, and release.
#   0.030 = busy and detailed. 0.055 = lush and musical. 0.080 = slow and meditative.
TRIGGER_RATE_LIMIT = 0.040

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
# WHY PYAUDIO INSTEAD OF PYGAME MIXER
# =====================================
# pygame.mixer works by queueing discrete Sound objects onto a channel.
# The OS plays one, then the next queues up — there is always a seam between
# chunks and the timing of when new oscillators appear is at the mercy of the
# queue drain speed. This causes the "piercing sharp bursts" feeling: each note
# starts as a new Sound object with no connection to the previous one.
#
# PyAudio uses a PULL CALLBACK model instead:
#   - We open a continuous output stream at 44100 Hz
#   - The OS audio driver calls our _callback() function whenever it needs
#     more samples — typically every ~10ms
#   - We synthesise exactly the samples requested and return them
#   - There are NO seams, NO queues, NO timing fights — it's one unbroken stream
#   - New oscillators added between callbacks blend seamlessly into the next chunk
#
# THE OSCILLATOR MODEL
# =====================
# Each trigger() call creates one _Osc. Every _Osc has:
#   freq      — target frequency in Hz (maps from array value)
#   phase     — current oscillator phase in [0,1), advances by freq/sr each sample
#   age       — how many samples have been rendered for this voice
#   max_age   — total lifetime in samples (= SOUND_SUSTAIN * SAMPLE_RATE)
#   attack    — samples for the fade-in  (= SOUND_ATTACK  * SAMPLE_RATE)
#   release   — samples for the fade-out (= SOUND_RELEASE * SAMPLE_RATE)
#
# WAVEFORM
# =========
# Pure sine + optional 2nd harmonic (both sines, so no harshness):
#   phase array: phases[t] = (phase0 + t * freq/sr) mod 1.0
#   wave[t]    = sin(2pi * phases[t])
#              + HARMONIC_BLEND * sin(2pi * 2 * phases[t])
#
# ENVELOPE — raised-cosine (Hann window) shape
# =============================================
# A linear ramp (the old approach) creates an audible click because the
# waveform's *slope* is discontinuous at the start even if amplitude is zero.
# Raised-cosine has zero slope at both endpoints — completely click-free.
#
#   Attack  (t in [0, A)):           env = 0.5 * (1 - cos(pi * t / A))
#   Sustain (t in [A, max-R)):       env = 1.0
#   Release (t in [max-R, max)):     env = 0.5 * (1 + cos(pi * (t-(max-R)) / R))
#
# TRIGGER RATE LIMITING
# ======================
# Sort algorithms call trigger_tone() on every comparison/swap — easily
# hundreds of times per second. Without a gate, no individual note ever gets
# to develop: the voice pool saturates and everything blurs into noise.
# TRIGGER_RATE_LIMIT sets the minimum gap between accepted triggers.
# Notes that arrive too soon are simply dropped. The ones that do fire
# get their full sustain time uninterrupted.
#
# VOICE STEALING
# ===============
# If MAX_VOICES is exceeded despite the rate limiter (e.g. slow sort + long
# sustain), the oldest oscillator has its remaining life clamped to a short
# cosine fade so it exits cleanly without clicking.

try:
    import sounddevice as sd
    HAS_SD = True
except ImportError:
    HAS_SD = False

TWO_PI = 2.0 * math.pi


class _Osc:
    """
    A single synthesiser voice.

    phase   : oscillator phase in [0, 1) — advanced by freq/sample_rate each sample
    age     : samples rendered so far this lifetime
    max_age : total lifetime in samples
    attack  : length of raised-cosine fade-in in samples
    release : length of raised-cosine fade-out in samples
    """
    __slots__ = ('freq', 'phase', 'age', 'max_age', 'attack', 'release')

    def __init__(self, freq, max_age, attack, release):
        self.freq    = freq
        # Random initial phase in [0,1) prevents phase-discontinuity clicks.
        # phase=0 always means every new voice snaps the waveform slope at
        # onset — audible as a tiny pop even with a smooth amplitude envelope.
        # A random start phase blends the new voice into the stream naturally.
        self.phase   = random.random()
        self.age     = 0
        self.max_age = max_age
        self.attack  = attack
        self.release = release


class SoundEngine:
    """
    Continuous-stream additive sine synthesiser using sounddevice's callback API.

    sounddevice wraps PortAudio and ships it as a binary — no compilation needed.
    The OS pulls audio from _callback() in real time. We mix all live oscillators
    into each requested chunk and return it instantly — no buffering, no queuing,
    no seams between notes.

    Install with:  pip install sounddevice
    """

    def __init__(self):
        self._sustain_smp = int(SOUND_SUSTAIN * SAMPLE_RATE)
        self._attack_smp  = max(1, int(SOUND_ATTACK  * SAMPLE_RATE))
        self._release_smp = max(1, int(SOUND_RELEASE * SAMPLE_RATE))
        self._oscs        = []      # list[_Osc] — all currently live voices
        self._lock        = threading.Lock()
        self._stream      = None    # sd.OutputStream
        self._last_t      = 0.0     # wall-clock time of last accepted trigger

    def start(self):
        if not HAS_SD:
            return
        # sounddevice.OutputStream with a callback — same pull model as PyAudio
        # but ships PortAudio as a bundled binary so no compilation is needed.
        # blocksize=CHUNK_SIZE means the callback fires every ~12ms at 44100 Hz.
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=CHUNK_SIZE,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def trigger(self, value: int, max_value: int):
        """
        Accept a new note trigger only if enough time has passed since the last
        one (TRIGGER_RATE_LIMIT gate). This is what gives notes room to breathe —
        without it, sort algorithms fire thousands of triggers per second and no
        individual note ever develops past its first few milliseconds of attack.
        """
        now = time.monotonic()
        if now - self._last_t < TRIGGER_RATE_LIMIT:
            return                      # too soon — drop this trigger
        self._last_t = now

        ratio = value / max_value
        freq  = FREQ_LOW + ratio * (FREQ_HIGH - FREQ_LOW)
        osc   = _Osc(freq, self._sustain_smp, self._attack_smp, self._release_smp)

        with self._lock:
            # Voice steal: if at capacity, clamp oldest voice to a fast cosine fade
            if len(self._oscs) >= MAX_VOICES:
                old        = self._oscs[0]
                fade       = min(256, old.release)
                old.max_age = old.age + fade
                old.release = fade
            self._oscs.append(osc)

    def _callback(self, outdata, frames, time_info, status):
        """
        sounddevice pull callback — called by the OS audio driver on a real-time
        thread. outdata is a (frames, 1) float32 numpy array we fill in place.

        We synthesise all live oscillators sample-accurately, apply raised-cosine
        envelopes, mix them down, and write the result into outdata.

        Note: globals are captured as locals at the top of this function.
        cffi C-thread callbacks can fail to resolve module globals reliably,
        so we snapshot them into locals on every call to guarantee availability.
        """
        # Snapshot globals into locals — safe against cffi thread scoping issues
        h2   = HARMONIC_BLEND
        h3   = HARMONIC_BLEND_3
        sr   = SAMPLE_RATE
        n    = frames
        buf  = np.zeros(n, dtype=np.float64)
        idx  = np.arange(n, dtype=np.float64)    # [0, 1, 2, ..., n-1]

        with self._lock:
            alive = []
            for o in self._oscs:
                t = idx + o.age

                # --- WAVEFORM ---
                phases = (o.phase + idx * (o.freq / sr)) % 1.0
                wave   = np.sin(TWO_PI * phases)
                if h2 > 0.0:
                    wave += h2 * np.sin(TWO_PI * 2.0 * phases)
                if h3 > 0.0:
                    wave += h3 * np.sin(TWO_PI * 3.0 * phases)

                # --- ENVELOPE (raised-cosine across all three regions) ---
                env = np.ones(n, dtype=np.float64)

                a_mask = t < o.attack
                if a_mask.any():
                    env[a_mask] = 0.5 * (1.0 - np.cos(math.pi * t[a_mask] / o.attack))

                rs     = o.max_age - o.release
                r_mask = t >= rs
                if r_mask.any():
                    env[r_mask] = 0.5 * (1.0 + np.cos(math.pi * (t[r_mask] - rs) / o.release))
                    np.maximum(env, 0.0, out=env)

                env[t >= o.max_age] = 0.0

                buf += wave * env

                o.phase = (o.phase + n * o.freq / sr) % 1.0
                o.age  += n
                if o.age < o.max_age:
                    alive.append(o)

            self._oscs = alive
            n_voices   = len(alive)

        peak_per_voice = 1.0 + h2 + h3
        if n_voices > 0:
            buf /= (math.sqrt(n_voices) * peak_per_voice)

        DRIVE = 1.0 / 0.85
        outdata[:] = (np.tanh(buf * DRIVE) * 0.85).astype(np.float32).reshape(-1, 1)


# --------------- Pygame mixer fallback ---------------
# Used only when sounddevice is not installed.

class _PygameFallbackEngine:
    """
    Minimal pygame.mixer fallback. Sounds noticeably worse than the sounddevice
    engine (choppy, no true sustain) but at least something plays.
    Install sounddevice for the full silky experience:  pip install sounddevice
    """
    def __init__(self):
        self._last_t = 0.0

    def start(self):
        pygame.mixer.pre_init(SAMPLE_RATE, -16, 1, CHUNK_SIZE)
        pygame.mixer.init()

    def stop(self):
        pygame.mixer.quit()

    def trigger(self, value: int, max_value: int):
        now = time.monotonic()
        if now - self._last_t < TRIGGER_RATE_LIMIT:
            return
        self._last_t = now

        ratio  = value / max_value
        freq   = FREQ_LOW + ratio * (FREQ_HIGH - FREQ_LOW)
        n      = int(SOUND_SUSTAIN * SAMPLE_RATE)
        t      = np.arange(n, dtype=np.float64)
        wave   = np.sin(TWO_PI * freq * t / SAMPLE_RATE)
        env    = 0.5 * (1.0 - np.cos(TWO_PI * t / n))
        pcm    = (wave * env * 32767 * 0.7).astype(np.int16)
        snd    = pygame.mixer.Sound(buffer=pcm.tobytes())
        snd.play()


# ---- global engine instance ----

_engine = None


def init_sound():
    global _engine
    if HAS_SD:
        _engine = SoundEngine()
    else:
        print("[SuperSolver] sounddevice not found — using pygame fallback. "
              "Install with:  pip install sounddevice")
        _engine = _PygameFallbackEngine()
        pygame.mixer.pre_init(SAMPLE_RATE, -16, 1, CHUNK_SIZE)
        pygame.mixer.init()
    _engine.start()


def stop_sound():
    global _engine
    if _engine:
        _engine.stop()
        _engine = None


def trigger_tone(value: int, max_value: int):
    if _engine and ENABLE_SOUND:
        _engine.trigger(value, max_value)


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
    SORTPACK_DIR = os.path.join(APPDATA, "SuperSolver", "SortPacks")
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
# ========================= TAB BAR ==========================
# ============================================================
#
# TAB_SORT and TAB_MAZE are the two top-level modes.
# The TabBar widget draws two pill-shaped buttons at the top of the window
# and returns which tab is active. The main loop swaps between SortMenu
# and MazeMenu based on this selection.

TAB_SORT = 0
TAB_MAZE = 1
TAB_LABELS = ["  SORT  ", "  MAZE  "]

class TabBar:
    """
    Two-tab switcher rendered to the RIGHT of the title text.
    TAB_X is set to clear "SuperSolver  20 algorithms" comfortably.
    H and Y are chosen to sit neatly centred in the 48px title row.
    """
    H    = 28
    BTNW = 100
    TAB_X = 320   # left edge of first tab — clears the title + subtitle
    Y     = 14    # vertically centred in the ~48px title row

    def __init__(self):
        self._rects = []
        for i, lbl in enumerate(TAB_LABELS):
            x = self.TAB_X + i * (self.BTNW + 6)
            self._rects.append((pygame.Rect(x, self.Y, self.BTNW, self.H), lbl))

    def handle(self, ev, active_tab):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for i, (r, _) in enumerate(self._rects):
                if r.collidepoint(ev.pos):
                    return i
        return None

    def draw(self, s, fonts, active_tab):
        mp = pygame.mouse.get_pos()
        for i, (r, lbl) in enumerate(self._rects):
            sel = (i == active_tab)
            hov = r.collidepoint(mp) and not sel
            bg  = UI_ACCENT  if sel else (UI_HOVER if hov else UI_PANEL2)
            fc  = (0, 0, 0)  if sel else (UI_TEXT  if hov else UI_SUBTEXT)
            pygame.draw.rect(s, bg,        r, border_radius=8)
            pygame.draw.rect(s, UI_BORDER if not sel else UI_ACCENT, r, 1, border_radius=8)
            t = fonts['small'].render(lbl, True, fc)
            s.blit(t, t.get_rect(center=r.center))

# ============================================================
# ========================= SORT MENU ========================
# ============================================================

class SortMenu:
    """
    The original sorting algorithm selection menu.
    Moved into its own class so it can live alongside MazeMenu under a tab bar.
    """
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

        # Content starts below tab bar (TAB_H) + title area
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

        t1 = self.fonts['title'].render("SuperSolver", True, UI_TEXT)
        t2 = self.fonts['title'].render("SuperSolver", True, UI_ACCENT)
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
            s.blit(self.fonts['small'].render("Radix Base:", True, UI_SUBTEXT), (RX, _Y_RADIX))
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
# ====================== MAZE ALGORITHMS =====================
# ============================================================
#
# HOW MAZE PATHFINDING GENERATORS WORK
# ======================================
# Every maze algorithm is a Python generator that yields tuples describing
# the current visualisation state after each logical step:
#
#   yield (visited_set, frontier_set, path_list, current_cell)
#
#   visited_set   : set of (row, col) cells already fully explored
#   frontier_set  : set of (row, col) cells currently being considered
#   path_list     : list of (row, col) cells on the best path found so far
#   current_cell  : (row, col) of the cell being processed this step
#
# The maze itself is a 2-D grid of 0 (open) or 1 (wall).
# Cells outside the grid and cells with value 1 are impassable.
#
# MAZE GENERATION — Recursive Backtracker (perfect maze)
# =======================================================
# Starts at top-left, carves passages by choosing a random unvisited
# neighbour 2 cells away (so walls stay between cells), removing the
# wall between them, and recursing. This guarantees exactly one path
# between any two cells (a "perfect" maze with no loops).
#
# CUSTOM MAZE LOADER
# ===================
# A custom maze .py file may define:
#   NAME  : str    — display name
#   solve(grid, start, end) -> generator  yielding (visited, frontier, path, current)
# Drop it in via "[+] Load Custom Solver" just like custom sorters.

import heapq
from collections import deque

_custom_maze_solvers: dict = {}
MAZE_AUTOLOAD_JSON = os.path.join(_SCRIPT_DIR, "supersorter_maze_custom.json")


def _save_maze_autoload_json():
    paths = [info["path"] for info in _custom_maze_solvers.values() if "path" in info]
    try:
        with open(MAZE_AUTOLOAD_JSON, "w") as f:
            json.dump({"custom_solvers": paths}, f, indent=2)
    except Exception:
        pass


def load_custom_maze_solver(filepath: str):
    """
    Load a .py file as a custom maze solver.
    Must define: solve(grid, start, end) generator.
    Optional: NAME str.
    Returns ((name, key), None) on success, (None, err_str) on failure.
    """
    try:
        filepath = os.path.abspath(filepath)
        spec     = importlib.util.spec_from_file_location("_cms", filepath)
        module   = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "solve"):
            return None, "No solve(grid, start, end) function found"
        fn   = module.solve
        name = getattr(module, "NAME", os.path.splitext(os.path.basename(filepath))[0])
        key  = f"maze_custom_{len(_custom_maze_solvers)}"
        _custom_maze_solvers[key] = {"fn": fn, "path": filepath}
        return (name, key), None
    except Exception as e:
        return None, str(e)


def _autoload_maze_solvers():
    if not os.path.exists(MAZE_AUTOLOAD_JSON):
        return
    try:
        with open(MAZE_AUTOLOAD_JSON) as f:
            paths = json.load(f).get("custom_solvers", [])
    except Exception:
        return
    for path in paths:
        if os.path.exists(path):
            result, _ = load_custom_maze_solver(path)
            if result:
                name, key = result
                MAZE_ALGORITHMS.append((name, key))


def generate_maze(rows, cols):
    """
    Recursive-backtracker perfect maze generator.
    Returns a 2-D list (rows x cols) of 0 (open) / 1 (wall).
    Cells are on odd indices; walls between them on even indices.
    The grid is always (2*rows+1) x (2*cols+1) in size.
    Start = (1,1), End = (2*rows-1, 2*cols-1).
    """
    R, C = 2 * rows + 1, 2 * cols + 1
    grid = [[1] * C for _ in range(R)]

    def carve(r, c):
        grid[r][c] = 0
        dirs = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 < nr < R and 0 < nc < C and grid[nr][nc] == 1:
                grid[r + dr//2][c + dc//2] = 0   # knock out wall between
                carve(nr, nc)

    sys.setrecursionlimit(10000)
    carve(1, 1)
    # Ensure end cell is open
    grid[R-2][C-2] = 0
    return grid


def _neighbours(grid, r, c):
    """4-connected passable neighbours of (r, c)."""
    R, C = len(grid), len(grid[0])
    for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
        nr, nc = r+dr, c+dc
        if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == 0:
            yield nr, nc


# ---- A* (Manhattan heuristic) ----
def maze_astar(grid, start, end):
    """
    A* search with Manhattan distance heuristic.
    Explores the cell that minimises f = g (cost so far) + h (heuristic).
    h = |row_diff| + |col_diff| — admissible for 4-connected grids so A*
    is guaranteed to find the shortest path.

    Yields (visited, frontier, path, current) after each cell pop.
    """
    er, ec   = end
    g_score  = {start: 0}
    came_from = {}
    # heap entries: (f_score, g_score, cell)
    heap     = [(abs(start[0]-er)+abs(start[1]-ec), 0, start)]
    visited  = set()
    frontier = {start}

    def reconstruct(node):
        path = []
        while node in came_from:
            path.append(node)
            node = came_from[node]
        path.append(start)
        return path[::-1]

    while heap:
        _, g, cur = heapq.heappop(heap)
        if cur in visited:
            continue
        visited.add(cur)
        frontier.discard(cur)
        yield visited.copy(), frontier.copy(), reconstruct(cur) if cur == end else [], cur
        if cur == end:
            return
        for nb in _neighbours(grid, *cur):
            ng = g + 1
            if ng < g_score.get(nb, 10**9):
                g_score[nb]   = ng
                came_from[nb] = cur
                h = abs(nb[0]-er) + abs(nb[1]-ec)
                heapq.heappush(heap, (ng + h, ng, nb))
                frontier.add(nb)


# ---- BFS (Breadth-First Search) ----
def maze_bfs(grid, start, end):
    """
    BFS explores in expanding rings of equal distance from start.
    Guarantees shortest path on unweighted grids.
    Frontier = the current wave front (cells at the same distance).

    Yields (visited, frontier, path, current) after each cell dequeue.
    """
    queue      = deque([start])
    came_from  = {start: None}
    visited    = set()

    def reconstruct(node):
        path = []
        while node is not None:
            path.append(node)
            node = came_from[node]
        return path[::-1]

    while queue:
        cur = queue.popleft()
        if cur in visited:
            continue
        visited.add(cur)
        frontier = set(queue)
        yield visited.copy(), frontier, reconstruct(cur) if cur == end else [], cur
        if cur == end:
            return
        for nb in _neighbours(grid, *cur):
            if nb not in came_from:
                came_from[nb] = cur
                queue.append(nb)


# ---- DFS (Depth-First Search) ----
def maze_dfs(grid, start, end):
    """
    DFS dives as deep as possible before backtracking.
    Does NOT guarantee shortest path but explores dramatically —
    it often finds a winding, scenic route.
    Uses an explicit stack to avoid Python recursion limits on large mazes.

    Yields (visited, frontier, path, current) after each cell pop.
    """
    stack      = [start]
    came_from  = {start: None}
    visited    = set()

    def reconstruct(node):
        path = []
        while node is not None:
            path.append(node)
            node = came_from[node]
        return path[::-1]

    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        frontier = set(stack)
        yield visited.copy(), frontier, reconstruct(cur) if cur == end else [], cur
        if cur == end:
            return
        for nb in _neighbours(grid, *cur):
            if nb not in visited:
                if nb not in came_from:
                    came_from[nb] = cur
                stack.append(nb)


# ---- Dijkstra ----
def maze_dijkstra(grid, start, end):
    """
    Dijkstra's algorithm — like A* but with h=0 (no heuristic).
    On unweighted grids this is equivalent to BFS but uses a priority
    queue, making it easy to extend to weighted edges.

    Yields (visited, frontier, path, current) after each cell pop.
    """
    dist       = {start: 0}
    came_from  = {}
    heap       = [(0, start)]
    visited    = set()
    frontier   = {start}

    def reconstruct(node):
        path = []
        while node in came_from:
            path.append(node)
            node = came_from[node]
        path.append(start)
        return path[::-1]

    while heap:
        d, cur = heapq.heappop(heap)
        if cur in visited:
            continue
        visited.add(cur)
        frontier.discard(cur)
        yield visited.copy(), frontier.copy(), reconstruct(cur) if cur == end else [], cur
        if cur == end:
            return
        for nb in _neighbours(grid, *cur):
            nd = d + 1
            if nd < dist.get(nb, 10**9):
                dist[nb]      = nd
                came_from[nb] = cur
                heapq.heappush(heap, (nd, nb))
                frontier.add(nb)


# ---- Greedy Best-First ----
def maze_greedy(grid, start, end):
    """
    Greedy Best-First: always expands the cell closest to the end
    by heuristic alone (ignores actual cost so far).
    Very fast to find *a* path but not guaranteed to be shortest.
    Visually dramatic — it beelines toward the goal aggressively.

    Yields (visited, frontier, path, current) after each cell pop.
    """
    er, ec     = end
    came_from  = {}
    heap       = [(abs(start[0]-er)+abs(start[1]-ec), start)]
    visited    = set()
    frontier   = {start}

    def reconstruct(node):
        path = []
        while node in came_from:
            path.append(node)
            node = came_from[node]
        path.append(start)
        return path[::-1]

    while heap:
        _, cur = heapq.heappop(heap)
        if cur in visited:
            continue
        visited.add(cur)
        frontier.discard(cur)
        yield visited.copy(), frontier.copy(), reconstruct(cur) if cur == end else [], cur
        if cur == end:
            return
        for nb in _neighbours(grid, *cur):
            if nb not in visited:
                came_from.setdefault(nb, cur)
                h = abs(nb[0]-er) + abs(nb[1]-ec)
                heapq.heappush(heap, (h, nb))
                frontier.add(nb)


# ---- Wall Follower (Right-Hand Rule) ----
def maze_wall_follower(grid, start, end):
    """
    Right-hand rule: always try to turn right first, then go straight,
    then left, then back. Works on mazes where the start and end are
    connected to the outer wall (i.e. simply-connected mazes).
    Will NOT always solve mazes with islands (loops).

    Facing: 0=up 1=right 2=down 3=left
    Yields (visited, frontier, path, current) after each step.
    """
    # Direction vectors for up/right/down/left
    DR = [-1, 0, 1,  0]
    DC = [ 0, 1, 0, -1]
    r, c    = start
    facing  = 2          # start facing down (into the maze)
    visited = set()
    path    = [start]
    steps   = 0
    max_steps = len(grid) * len(grid[0]) * 8

    while (r, c) != end and steps < max_steps:
        visited.add((r, c))
        yield visited.copy(), set(), path[:], (r, c)
        # Try: right, straight, left, back
        for turn in (-1, 0, 1, 2):
            nf = (facing + 1 + turn) % 4     # right = facing+1, straight = facing+0, left = facing-1
            # Re-map: turn=-1→right, turn=0→straight, turn=1→left, turn=2→back
            nf = (facing + [1, 0, -1, 2][turn + 1]) % 4
            nr, nc = r + DR[nf], c + DC[nf]
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] == 0:
                facing = nf
                r, c   = nr, nc
                path.append((r, c))
                break
        steps += 1

    visited.add(end)
    yield visited.copy(), set(), path if (r,c)==end else [], end


# ---- Dead-End Filling ----
def maze_dead_end_fill(grid, start, end):
    """
    Dead-end filling: repeatedly find cells with exactly one open
    neighbour (dead ends) and fill them in as walls — except start/end.
    What remains is the solution path(s).
    This is a reduction algorithm, not a search — it reveals the path
    by eliminating everything that ISN'T the path.

    Yields (visited/filled set, frontier, path, current) each fill step.
    """
    R, C   = len(grid), len(grid[0])
    filled = [[grid[r][c] for c in range(C)] for r in range(R)]
    filled_cells = set()

    def open_count(r, c):
        return sum(1 for nr, nc in _neighbours(filled, r, c))

    changed = True
    while changed:
        changed = False
        for r in range(R):
            for c in range(C):
                if filled[r][c] == 0 and (r, c) not in (start, end):
                    if open_count(r, c) == 1:
                        filled[r][c] = 1
                        filled_cells.add((r, c))
                        changed = True
                        # Build remaining path = open cells minus filled
                        path = [(rr,cc) for rr in range(R) for cc in range(C)
                                if filled[rr][cc]==0]
                        yield filled_cells.copy(), set(), path, (r, c)

    path = [(r,c) for r in range(R) for c in range(C) if filled[r][c]==0]
    yield filled_cells.copy(), set(), path, end


MAZE_ALGORITHMS = [
    ("A* (Manhattan)",    "astar"),
    ("BFS",               "bfs"),
    ("DFS",               "dfs"),
    ("Dijkstra",          "dijkstra"),
    ("Greedy Best-First", "greedy"),
    ("Wall Follower",     "wall_follower"),
    ("Dead-End Fill",     "dead_end_fill"),
]

def get_maze_generator(key, grid, start, end):
    builtins = {
        "astar":         lambda: maze_astar(grid, start, end),
        "bfs":           lambda: maze_bfs(grid, start, end),
        "dfs":           lambda: maze_dfs(grid, start, end),
        "dijkstra":      lambda: maze_dijkstra(grid, start, end),
        "greedy":        lambda: maze_greedy(grid, start, end),
        "wall_follower": lambda: maze_wall_follower(grid, start, end),
        "dead_end_fill": lambda: maze_dead_end_fill(grid, start, end),
    }
    if key in builtins: return builtins[key]()
    if key in _custom_maze_solvers: return _custom_maze_solvers[key]["fn"](grid, start, end)
    raise KeyError(f"Unknown maze key: {key}")


# ============================================================
# ========================= MAZE MENU ========================
# ============================================================
#
# Layout:
#   Left panel  — algorithm buttons (2 cols)
#   Right panel — settings + start button (same RX/RW as sort menu)
#
# Maze size slider controls the N in an N×N logical cell grid.
# The rendered grid is (2N+1)×(2N+1) pixels mapped to the window.

_MZ_COLS   = 2
_MZ_BTN_W  = (GRID_W - (_MZ_COLS-1)*COL_GAP) // _MZ_COLS
_MZ_BTN_H  = 38

# Right-panel Y anchors for maze settings
_MY_SETTINGS = 88
_MY_SIZE     = 108
_MY_SPEED    = 162
_MY_GEN      = 216   # maze generation style label+buttons


class MazeMenu:
    """
    Algorithm selection and settings panel for Maze mode.
    Mirrors SortMenu's structure so the tab switch feels seamless.
    """
    GEN_STYLES = [("Recursive", "recurse"), ("Prim's", "prims"), ("Empty", "empty")]

    def __init__(self, screen, fonts):
        self.screen    = screen
        self.fonts     = fonts
        self.sel       = 0
        self.hov       = -1
        self.msg       = ""
        self.msg_until = 0
        self.msg_ok    = True
        self.gen_style = "recurse"

        self._build_btns()

        self.sl_size  = Slider(RX, _MY_SIZE,  RW, 5, 40, 15, "Maze Size (N)", is_int=True)
        self.sl_speed = Slider(RX, _MY_SPEED, RW, 0.25, 16.0, 2.0, "Speed")

        # Generation style buttons
        gsw = (RW - (len(self.GEN_STYLES)-1)*5) // len(self.GEN_STYLES)
        self.gen_btns = []
        for i, (lbl, key) in enumerate(self.GEN_STYLES):
            self.gen_btns.append((SmBtn(RX + i*(gsw+5), _MY_GEN+16, gsw, 26, lbl), key))

        self.start_rect = pygame.Rect(RX, WINDOW_HEIGHT-62, RW, 46)
        self.start_hov  = False

    def _build_btns(self):
        gx, gy = PAD, 80
        self.btns = []
        for i, (nm, ky) in enumerate(MAZE_ALGORITHMS):
            col = i % _MZ_COLS; row = i // _MZ_COLS
            x   = gx + col*(_MZ_BTN_W + COL_GAP)
            y   = gy + row*(_MZ_BTN_H + BTN_GAP)
            self.btns.append(AlgoBtn(x, y, _MZ_BTN_W, nm, ky, i))

        rows_used = (len(MAZE_ALGORITHMS) + _MZ_COLS - 1) // _MZ_COLS
        load_y    = gy + rows_used*(_MZ_BTN_H + BTN_GAP) + 4
        self.load_btn = SmBtn(gx, load_y, GRID_W, 30, "[+] Load Custom Solver")

    def _notify(self, msg, ok=True):
        self.msg = msg; self.msg_ok = ok
        self.msg_until = pygame.time.get_ticks() + 4000

    def handle(self, ev):
        self.sl_size.handle(ev)
        self.sl_speed.handle(ev)

        if ev.type == pygame.MOUSEMOTION:
            self.hov = -1
            self.start_hov = self.start_rect.collidepoint(ev.pos)
            for b in self.btns:
                if b.rect.collidepoint(ev.pos): self.hov = b.idx

        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for b in self.btns:
                if b.rect.collidepoint(ev.pos): self.sel = b.idx

            for sb, key in self.gen_btns:
                if sb.rect.collidepoint(ev.pos): self.gen_style = key

            if self.load_btn.rect.collidepoint(ev.pos):
                self._do_load()

            if self.start_rect.collidepoint(ev.pos):
                return "start"

        return None

    def _do_load(self):
        path = open_file_dialog()
        if not path or not path.lower().endswith(".py"): return
        result, err = load_custom_maze_solver(path)
        if err:
            self._notify(f"Error: {err[:55]}", ok=False)
            return
        name, key = result
        MAZE_ALGORITHMS.append((name, key))
        _save_maze_autoload_json()
        self._build_btns()
        self.sel = len(MAZE_ALGORITHMS) - 1
        self._notify(f"Loaded: {name}", ok=True)

    def draw(self):
        s  = self.screen
        mp = pygame.mouse.get_pos()

        # Title
        t1 = self.fonts['title'].render("SuperSolver", True, UI_TEXT)
        t2 = self.fonts['title'].render("SuperSolver", True, UI_ACCENT)
        s.blit(t2, (PAD+1, 23)); s.blit(t1, (PAD, 22))
        s.blit(self.fonts['small'].render(f"{len(MAZE_ALGORITHMS)} solvers", True, UI_SUBTEXT),
               (PAD + t1.get_width() + 12, 31))
        pygame.draw.line(s, UI_BORDER, (PAD, 68), (WINDOW_WIDTH-PAD, 68), 1)

        # Algorithm buttons
        for b in self.btns:
            b.draw(s, self.fonts, b.idx==self.sel, b.idx==self.hov)
        self.load_btn.draw(s, self.fonts, False, self.load_btn.rect.collidepoint(mp))

        now = pygame.time.get_ticks()
        if self.msg and now < self.msg_until:
            col = UI_GREEN if self.msg_ok else (255, 90, 90)
            s.blit(self.fonts['small'].render(self.msg, True, col),
                   (PAD, self.load_btn.rect.bottom + 6))
        elif now >= self.msg_until:
            self.msg = ""

        # Right panel
        panel = pygame.Rect(RX-10, 76, RW+20, WINDOW_HEIGHT-82)
        pygame.draw.rect(s, UI_PANEL,  panel, border_radius=7)
        pygame.draw.rect(s, UI_BORDER, panel, 1, border_radius=7)
        s.blit(self.fonts['small'].render("SETTINGS", True, UI_SUBTEXT), (RX, _MY_SETTINGS))

        self.sl_size.draw(s, self.fonts)
        self.sl_speed.draw(s, self.fonts)

        # Maze generation style
        s.blit(self.fonts['small'].render("Generation:", True, UI_SUBTEXT), (RX, _MY_GEN))
        for sb, key in self.gen_btns:
            sb.draw(s, self.fonts, self.gen_style==key, sb.rect.collidepoint(mp))

        # Selected info
        nm, _ = MAZE_ALGORITHMS[self.sel]
        y_sel  = _MY_GEN + 16 + 26 + 16
        s.blit(self.fonts['small'].render("Selected:", True, UI_SUBTEXT), (RX, y_sel))
        s.blit(self.fonts['mid'].render(nm, True, UI_ACCENT), (RX, y_sel+16))
        s.blit(self.fonts['small'].render("ESC during solve returns here", True, UI_DIM),
               (RX, y_sel+36))

        # Start button
        sh = self.start_hov
        pygame.draw.rect(s, (240,50,50) if sh else (200,35,35), self.start_rect, border_radius=7)
        pygame.draw.rect(s, (255,90,90) if sh else UI_ACCENT,   self.start_rect, 2, border_radius=7)
        st2 = self.fonts['big'].render("> START", True, (255, 255, 255))
        s.blit(st2, st2.get_rect(center=self.start_rect.center))

    def config(self):
        nm, ky = MAZE_ALGORITHMS[self.sel]
        return dict(name=nm, key=ky,
                    size=self.sl_size.value,
                    speed=self.sl_speed.value,
                    gen_style=self.gen_style)


# ============================================================
# ====================== MAZE GENERATION =====================
# ============================================================

def _gen_prims(rows, cols):
    """
    Prim's randomised maze generation.
    Starts with a grid of walls, picks a random cell, adds its neighbours
    to a frontier list, and repeatedly picks a random frontier cell to
    connect to the already-visited region — carving a passage as it goes.
    Produces mazes with shorter average corridors and more branching than
    the recursive backtracker.
    """
    R, C  = 2*rows+1, 2*cols+1
    grid  = [[1]*C for _ in range(R)]
    start = (1, 1)
    grid[1][1] = 0

    def cell_neighbours_2(r, c):
        for dr, dc in [(0,2),(0,-2),(2,0),(-2,0)]:
            nr, nc = r+dr, c+dc
            if 0 < nr < R and 0 < nc < C:
                yield nr, nc

    frontier = list(cell_neighbours_2(1, 1))
    in_maze  = {(1,1)}

    while frontier:
        r, c = frontier.pop(random.randrange(len(frontier)))
        if (r, c) in in_maze:
            continue
        # Find a neighbour already in maze and connect
        nbrs_in = [(r+dr,c+dc) for dr,dc in [(-2,0),(2,0),(0,-2),(0,2)]
                   if 0<r+dr<R and 0<c+dc<C and (r+dr,c+dc) in in_maze]
        if nbrs_in:
            nr, nc     = random.choice(nbrs_in)
            grid[r][c] = 0
            grid[(r+nr)//2][(c+nc)//2] = 0   # wall between them
            in_maze.add((r, c))
            for nb in cell_neighbours_2(r, c):
                if nb not in in_maze:
                    frontier.append(nb)

    grid[R-2][C-2] = 0
    return grid


def _gen_empty(rows, cols):
    """Open grid — no walls at all. Good for showing how search fans out."""
    R, C = 2*rows+1, 2*cols+1
    return [[0]*C for _ in range(R)]


def build_maze_grid(size, style):
    """
    Build a maze grid of the given logical size using the chosen style.
    Returns (grid, start_cell, end_cell).
    start = (1,1), end = (2*size-1, 2*size-1) (bottom-right corner).
    """
    if style == "recurse":
        grid = generate_maze(size, size)
    elif style == "prims":
        grid = _gen_prims(size, size)
    else:
        grid = _gen_empty(size, size)

    R, C  = len(grid), len(grid[0])
    start = (1, 1)
    end   = (R-2, C-2)
    return grid, start, end


# ============================================================
# ====================== MAZE VISUALISER =====================
# ============================================================
#
# CELL COLOURS
# =============
# WALL        : near-black  (5, 5, 10)
# OPEN        : dark grey   (28, 28, 40)
# VISITED     : deep blue   (30, 80, 180) — cells already explored
# FRONTIER    : cyan        (0, 200, 200) — cells currently queued
# PATH        : bright blue (60, 160, 255) — solution path so far
# CURRENT     : red accent  (255, 60, 60) — cell being processed this step
# START       : green       (60, 200, 80)
# END         : red         (220, 50, 50)

MAZE_C_WALL     = (5,   5,  10)
MAZE_C_OPEN     = (28,  28,  40)
MAZE_C_VISITED  = (20,  60, 160)
MAZE_C_FRONTIER = (0,  180, 200)
MAZE_C_PATH     = (60, 160, 255)
MAZE_C_CURRENT  = (255, 60,  60)
MAZE_C_START    = (60, 200,  80)
MAZE_C_END      = (220,  50,  50)


def draw_maze(screen, grid, start, end, visited, frontier, path, current, label=""):
    """
    Render the maze onto the full screen.
    Maps the grid cells to screen pixels so the maze fills most of the window.
    """
    screen.fill(MAZE_C_WALL)
    R, C    = len(grid), len(grid[0])
    margin  = 36           # pixels of margin on each side
    cell_w  = (WINDOW_WIDTH  - 2*margin) / C
    cell_h  = (WINDOW_HEIGHT - 2*margin - 20) / R

    path_set     = set(path)
    frontier_set = set(frontier)

    for r in range(R):
        for c in range(C):
            x = int(margin + c * cell_w)
            y = int(margin + r * cell_h)
            w = max(1, int(cell_w))
            h = max(1, int(cell_h))

            if grid[r][c] == 1:
                col = MAZE_C_WALL
            elif (r, c) == current:
                col = MAZE_C_CURRENT
            elif (r, c) == start:
                col = MAZE_C_START
            elif (r, c) == end:
                col = MAZE_C_END
            elif (r, c) in path_set:
                col = MAZE_C_PATH
            elif (r, c) in frontier_set:
                col = MAZE_C_FRONTIER
            elif (r, c) in visited:
                col = MAZE_C_VISITED
            else:
                col = MAZE_C_OPEN

            pygame.draw.rect(screen, col, (x, y, w-1, h-1))

    if label:
        f = pygame.font.SysFont("consolas", 16)
        screen.blit(f.render(label, True, (140, 140, 160)), (margin, 8))

    pygame.display.flip()


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


def run_maze(screen, fonts, cfg):
    """
    Main maze solving loop.
    Builds the grid, runs the chosen solver generator, renders each step.
    Speed slider maps directly to steps-per-frame (1x = 2 steps/frame,
    16x = 32 steps/frame) so fast speeds skip visual frames for big mazes.

    SOUND IN MAZE MODE
    ===================
    We reuse the same sine engine as the sort visualiser.
    The current cell's row position is mapped to a frequency:
        freq = FREQ_LOW + (row / max_row) * (FREQ_HIGH - FREQ_LOW)
    This means the tone rises as the solver moves down the maze and
    drops as it backtracks upward — creating a natural musical contour
    that mirrors the visual search pattern.
    trigger_tone() already applies the TRIGGER_RATE_LIMIT gate so we
    can call it every step without worrying about flooding.
    """
    grid, start, end = build_maze_grid(cfg["size"], cfg["gen_style"])
    gen   = get_maze_generator(cfg["key"], grid, start, end)
    clock = pygame.time.Clock()
    label = cfg["name"]
    steps_per_frame = max(1, int(cfg["speed"] * 2))
    max_row = max(1, len(grid) - 1)   # used for row → frequency mapping

    visited  = set()
    frontier = set()
    path     = []
    current  = start

    solved = False
    while True:
        clock.tick(FPS)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:   stop_sound(); pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE: return

        if not solved:
            for _ in range(steps_per_frame):
                try:
                    visited, frontier, path, current = next(gen)
                    # Map current row to a tone value in [0, max_row]
                    # trigger_tone expects (value, max_value) and maps linearly
                    # to [FREQ_LOW, FREQ_HIGH], so passing row/max_row gives us
                    # a pitch that follows vertical position through the maze.
                    trigger_tone(current[0], max_row)
                except StopIteration:
                    solved = True
                    break

        suffix = "  [SOLVED]" if solved else ""
        draw_maze(screen, grid, start, end, visited, frontier, path, current,
                  label=f"{label}{suffix}  —  visited: {len(visited)}")

        if solved:
            pygame.time.wait(2200)
            return


def main():
    pygame.init(); init_sound()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("SuperSolver")
    force_top()
    fonts = build_fonts(); clock = pygame.time.Clock()

    autoload_custom_sorters()
    _autoload_maze_solvers()

    # Active tab: TAB_SORT or TAB_MAZE
    active_tab  = TAB_SORT
    tab_bar     = TabBar()
    sort_menu   = SortMenu(screen, fonts)
    maze_menu   = MazeMenu(screen, fonts)

    while True:
        clock.tick(60)

        # Draw background then tab bar on top of both menus
        screen.fill(UI_BG)
        if active_tab == TAB_SORT:
            sort_menu.draw()
        else:
            maze_menu.draw()
        tab_bar.draw(screen, fonts, active_tab)
        pygame.display.flip()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                stop_sound(); pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                stop_sound(); pygame.quit(); sys.exit()

            # Tab bar gets first pick of clicks
            new_tab = tab_bar.handle(ev, active_tab)
            if new_tab is not None:
                active_tab = new_tab
                continue

            if active_tab == TAB_SORT:
                result = sort_menu.handle(ev)
                if result == "start":
                    run_sort(screen, fonts, sort_menu.config())
            else:
                result = maze_menu.handle(ev)
                if result == "start":
                    run_maze(screen, fonts, maze_menu.config())

if __name__ == "__main__":
    main()