import pygame
import random
import numpy as np
import sys
import math
import threading
import time
import importlib.util
import os

try:
    import ctypes
    HAS_CTYPES = True
except ImportError:
    HAS_CTYPES = False

# tkinter for file dialog (stdlib)
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
FREQ_LOW      = 120.0    # default min (user-adjustable)
FREQ_HIGH     = 1212.0   # default max (user-adjustable)
FREQ_ABS_LOW  = 12.0     # absolute slider floor
FREQ_ABS_HIGH = 4800.0   # absolute slider ceiling
SOUND_SUSTAIN = 0.10
SAMPLE_RATE   = 44100
CHUNK_SIZE    = 512

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
UI_HATCH      = (50,  40,  55)   # hatched dead-zone colour

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

# Custom-loaded algorithms are appended here at runtime:
# ("Custom: filename", "custom_N")
_custom_generators: dict = {}   # key -> callable(arr)

# ============================================================
# ============== SoS-STYLE SOUND ENGINE ======================
# ============================================================

class _Osc:
    __slots__ = ('freq', 'phase', 'age', 'max_age', 'attack', 'release')
    def __init__(self, freq, max_age, attack, release):
        self.freq = freq; self.phase = 0.0; self.age = 0
        self.max_age = max_age; self.attack = attack; self.release = release


class SoundEngine:
    def __init__(self):
        self.sample_rate  = SAMPLE_RATE
        self.chunk_size   = CHUNK_SIZE
        self.sustain_smp  = int(SOUND_SUSTAIN * SAMPLE_RATE)
        self.attack_smp   = max(1, int(0.004 * SAMPLE_RATE))
        self.release_smp  = max(1, int(0.020 * SAMPLE_RATE))
        self._oscs        = []
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
        if self._channel: self._channel.stop()

    def trigger(self, value: int, max_value: int):
        ratio = value / max_value
        freq  = FREQ_LOW + ratio * (FREQ_HIGH - FREQ_LOW)
        osc   = _Osc(freq, self.sustain_smp, self.attack_smp, self.release_smp)
        with self._lock: self._oscs.append(osc)

    def set_sustain(self, seconds: float):
        self.sustain_smp = int(seconds * self.sample_rate)

    def _gen_chunk(self) -> np.ndarray:
        buf = np.zeros(self.chunk_size, dtype=np.float64)
        idx = np.arange(self.chunk_size, dtype=np.float64)
        with self._lock:
            alive = []
            for o in self._oscs:
                abs_age = idx + o.age
                phases  = (o.phase + idx * (o.freq / self.sample_rate)) % 1.0
                wave    = 2.0 * np.abs(2.0 * phases - 1.0) - 1.0
                env     = np.ones(self.chunk_size)
                a_mask  = abs_age < o.attack
                env[a_mask] = abs_age[a_mask] / o.attack
                rel_start   = o.max_age - o.release
                r_mask      = abs_age >= rel_start
                env[r_mask] = np.maximum(0.0, (o.max_age - abs_age[r_mask]) / o.release)
                env[abs_age >= o.max_age] = 0.0
                buf += wave * env
                o.phase = (o.phase + self.chunk_size * o.freq / self.sample_rate) % 1.0
                o.age  += self.chunk_size
                if o.age < o.max_age: alive.append(o)
            self._oscs = alive
            n = max(1, len(alive))
        buf /= (1.0 + n * 0.40)
        return buf

    def _loop(self):
        chunk_secs = self.chunk_size / self.sample_rate
        while self._running:
            mono   = self._gen_chunk()
            pcm    = (np.clip(mono, -1.0, 1.0) * 32767 * 0.85).astype(np.int16)
            stereo = np.column_stack((pcm, pcm))
            snd    = pygame.mixer.Sound(buffer=stereo.tobytes())
            deadline = time.monotonic() + chunk_secs * 4
            while self._channel.get_queue() is not None and self._running:
                time.sleep(0.001)
                if time.monotonic() > deadline: break
            if self._running: self._channel.queue(snd)
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
    if _engine: _engine.stop(); _engine = None

def trigger_tone(value: int, max_value: int):
    if _engine and ENABLE_SOUND: _engine.trigger(value, max_value)

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
            yield arr, [j, j + 1]
            if arr[j] > arr[j + 1]: arr[j], arr[j+1] = arr[j+1], arr[j]; yield arr, [j, j+1]

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
        for i in range(cs+1, n): yield arr, [i, cs]; (pos := pos+1) if arr[i] < item else None
        if pos == cs: continue
        while item == arr[pos]: pos += 1
        arr[pos], item = item, arr[pos]; yield arr, [pos, cs]
        while pos != cs:
            pos = cs
            for i in range(cs+1, n): yield arr, [i, cs]; (pos := pos+1) if arr[i] < item else None
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
    for i in range(n-1, -1, -1): idx=(arr[i]//exp)%base; out[cnt[idx]-1]=arr[i]; cnt[idx]-=1; yield arr,[i]
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
    if key in _custom_generators: return _custom_generators[key](arr)
    raise KeyError(f"Unknown algorithm key: {key}")

# ============================================================
# ===================== CUSTOM SORTER LOADER =================
# ============================================================

def load_custom_sorter(filepath: str) -> tuple[str, str] | None:
    """
    Load a Python file as a custom sorter.
    The file must define a function:  sort(arr)  ->  generator yielding (arr, [indices])
    Returns (display_name, key) on success, None on failure.
    """
    try:
        spec   = importlib.util.spec_from_file_location("custom_sort", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "sort"):
            return None, "Missing `sort(arr)` function"
        fn    = module.sort
        name  = getattr(module, "NAME", os.path.splitext(os.path.basename(filepath))[0])
        key   = f"custom_{len(_custom_generators)}"
        _custom_generators[key] = fn
        return (f"⚡ {name}", key), None
    except Exception as e:
        return None, str(e)

def open_file_dialog() -> str | None:
    """Open a native file-picker and return chosen path (or None)."""
    if not HAS_TK:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Load Custom Sorter",
        filetypes=[("Python files", "*.py"), ("All files", "*.*")]
    )
    root.destroy()
    return path if path else None

# ============================================================
# ========================= UI CLASSES =======================
# ============================================================

class Slider:
    """Standard single-knob slider."""
    def __init__(self, x, y, w, lo, hi, val, label, is_int=False):
        self.x, self.y, self.w = x, y, w
        self.lo, self.hi = lo, hi
        self.value = val; self.label = label; self.is_int = is_int
        self.drag  = False
        self.track = pygame.Rect(x, y+18, w, 5)
        self.hit   = pygame.Rect(x-5, y, w+10, 42)

    def _r(self):  return (self.value - self.lo) / (self.hi - self.lo)
    def _kx(self): return int(self.x + self._r() * self.w)

    def handle(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if math.hypot(ev.pos[0]-self._kx(), ev.pos[1]-self.track.centery) < 14 \
               or self.hit.collidepoint(ev.pos): self.drag = True; self._set(ev.pos[0])
        elif ev.type == pygame.MOUSEBUTTONUP: self.drag = False
        elif ev.type == pygame.MOUSEMOTION and self.drag: self._set(ev.pos[0])

    def _set(self, mx):
        r = max(0.0, min(1.0, (mx - self.x) / self.w))
        raw = self.lo + r * (self.hi - self.lo)
        self.value = int(round(raw)) if self.is_int else round(raw * 4) / 4

    def draw(self, s, fonts):
        vs = str(self.value) if self.is_int else f"{self.value:.2f}x"
        s.blit(fonts['small'].render(f"{self.label}:  {vs}", True, UI_SUBTEXT), (self.x, self.y))
        pygame.draw.rect(s, UI_BORDER, self.track, border_radius=3)
        fw = int(self._r() * self.w)
        if fw > 0: pygame.draw.rect(s, UI_ACCENT, (self.x, self.track.y, fw, 5), border_radius=3)
        kx, ky = self._kx(), self.track.centery
        pygame.draw.circle(s, UI_PANEL2, (kx, ky), 10)
        pygame.draw.circle(s, UI_ACCENT, (kx, ky), 10, 2)
        pygame.draw.circle(s, UI_ACCENT, (kx, ky), 4)


class FreqRangeSlider:
    """
    Dual-knob frequency range slider matching the sketch:
      - Left knob  = min freq  (FREQ_LOW)
      - Right knob = max freq  (FREQ_HIGH)
      - Hatched fill left of left knob  (dead-zone below min)
      - Solid accent fill between knobs (active band)
      - Plain track right of right knob
    Absolute limits: FREQ_ABS_LOW .. FREQ_ABS_HIGH
    """
    KNOB_R    = 10
    HATCH_GAP = 6    # pixels between hatch lines

    def __init__(self, x, y, w, lo_val, hi_val):
        self.x, self.y, self.w = x, y, w
        self.lo_val = lo_val   # current min freq
        self.hi_val = hi_val   # current max freq
        self.drag_lo = False
        self.drag_hi = False
        self.track   = pygame.Rect(x, y + 22, w, 5)
        self._hatch_surf = None   # cached

    # --- helpers ---
    def _to_x(self, freq):
        r = (freq - FREQ_ABS_LOW) / (FREQ_ABS_HIGH - FREQ_ABS_LOW)
        return int(self.x + r * self.w)

    def _to_freq(self, px):
        r = max(0.0, min(1.0, (px - self.x) / self.w))
        return FREQ_ABS_LOW + r * (FREQ_ABS_HIGH - FREQ_ABS_LOW)

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
                f = self._to_freq(ev.pos[0])
                self.lo_val = max(FREQ_ABS_LOW, min(f, self.hi_val - 10))
            if self.drag_hi:
                f = self._to_freq(ev.pos[0])
                self.hi_val = min(FREQ_ABS_HIGH, max(f, self.lo_val + 10))

    def draw(self, s, fonts):
        tx, ty = self.track.x, self.track.y
        tw, th = self.track.width, self.track.height
        ky     = self.track.centery
        lx     = self.lo_x
        hx     = self.hi_x

        # --- label row ---
        lbl = fonts['small'].render("Freq Range:", True, UI_SUBTEXT)
        s.blit(lbl, (self.x, self.y))

        lo_lbl = fonts['mono_sm'].render(f"{self.lo_val:.0f} Hz", True, UI_ACCENT)
        hi_lbl = fonts['mono_sm'].render(f"{self.hi_val:.0f} Hz", True, UI_ACCENT)
        abs_lo = fonts['mono_sm'].render(f"{FREQ_ABS_LOW:.0f}", True, UI_DIM)
        abs_hi = fonts['mono_sm'].render(f"{FREQ_ABS_HIGH:.0f}", True, UI_DIM)

        # abs limits
        s.blit(abs_lo, (self.x,              self.y + 2))
        s.blit(abs_hi, (self.x + self.w - abs_hi.get_width(), self.y + 2))

        # min/max freq labels above knobs
        lo_lw = lo_lbl.get_width()
        hi_lw = hi_lbl.get_width()
        lo_lx = max(self.x, min(lx - lo_lw//2, self.x + self.w - lo_lw))
        hi_lx = max(self.x, min(hx - hi_lw//2, self.x + self.w - hi_lw))
        s.blit(lo_lbl, (lo_lx, self.y + 2))
        s.blit(hi_lbl, (hi_lx, self.y + 2))

        # --- track base (dark full width) ---
        pygame.draw.rect(s, UI_BORDER, self.track, border_radius=3)

        # --- hatched dead-zone (left of lo knob) ---
        # draw diagonal hatch lines clipped to [tx, lx]
        if lx > tx:
            clip_rect = pygame.Rect(tx, ty - 2, lx - tx, th + 4)
            # darker fill first
            pygame.draw.rect(s, (25, 18, 32), clip_rect)
            # diagonal hatch lines
            gap = self.HATCH_GAP
            span = (lx - tx) + th * 2
            for off in range(0, span, gap):
                x1 = tx + off;       y1 = ty + th
                x2 = tx + off - th;  y2 = ty
                # clamp to clip_rect
                x1 = max(tx, min(lx, x1)); x2 = max(tx, min(lx, x2))
                pygame.draw.line(s, UI_HATCH, (x1, y1), (x2, y2), 1)

        # --- active band fill (between knobs) ---
        band_w = hx - lx
        if band_w > 0:
            pygame.draw.rect(s, UI_ACCENT, (lx, ty, band_w, th))

        # --- knobs ---
        for kx, is_lo in ((lx, True), (hx, False)):
            pygame.draw.circle(s, UI_PANEL2, (kx, ky), self.KNOB_R)
            pygame.draw.circle(s, UI_ACCENT, (kx, ky), self.KNOB_R, 2)
            pygame.draw.circle(s, UI_ACCENT, (kx, ky), 4)

        # --- abs limit ticks ---
        for tx2 in (tx, tx + tw):
            pygame.draw.line(s, UI_DIM, (tx2, ky - 7), (tx2, ky + 7), 1)


class AlgoBtn:
    H = 44
    def __init__(self, x, y, w, name, key, idx):
        self.rect = pygame.Rect(x, y, w, self.H)
        self.name, self.key, self.idx = name, key, idx

    def draw(self, s, fonts, sel, hov):
        bg = UI_SEL_BG if sel else (UI_HOVER if hov else UI_PANEL)
        br = UI_SEL_BORDER if sel else (UI_DIM if hov else UI_BORDER)
        pygame.draw.rect(s, bg, self.rect, border_radius=6)
        pygame.draw.rect(s, br, self.rect, 1, border_radius=6)
        nc = UI_ACCENT if sel else UI_SUBTEXT
        tc = UI_TEXT if sel else (UI_TEXT if hov else (160, 160, 180))
        s.blit(fonts['mono_sm'].render(f"{self.idx+1:02d}", True, nc), (self.rect.x+10, self.rect.y+14))
        # Truncate long names
        name = self.name if len(self.name) <= 20 else self.name[:18] + "…"
        s.blit(fonts['mid'].render(name, True, tc), (self.rect.x+42, self.rect.y+13))


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
# ====================== MENU SCREEN =========================
# ============================================================

class Menu:
    PAD = 18; BTN_W = 226; BTN_GAP = 5; COL_GAP = 10; COLS = 3

    def __init__(self, screen, fonts):
        self.screen = screen; self.fonts = fonts
        self.sel = 0; self.hov = -1; self.sound_on = ENABLE_SOUND
        self.err_msg = ""; self.err_timer = 0

        self._rebuild_buttons()

        rx = self.PAD + self.COLS * (self.BTN_W + self.COL_GAP) + 20
        self.rx = rx
        rw = WINDOW_WIDTH - rx - self.PAD

        self.sl_size    = Slider(rx, 105, rw, 4, MAX_ARRAY_SIZE, 32, "Array Size", is_int=True)
        self.sl_speed   = Slider(rx, 175, rw, 0.25, 8.0, 1.0, "Speed")
        self.sl_sustain = Slider(rx, 245, rw, 0.02, 0.40, 0.10, "Sound Sustain")
        self.sl_freq    = FreqRangeSlider(rx, 315, rw, FREQ_LOW, FREQ_HIGH)

        bases = [2, 4, 8, 10, 16]
        bw = (rw - (len(bases)-1)*6) // len(bases)
        self.radix_base = 10; self.base_btns = []
        for i, b in enumerate(bases):
            self.base_btns.append((SmBtn(rx+i*(bw+6), 405, bw, 28, f"Base {b}"), b))

        self.snd_btn     = SmBtn(rx, 452, rw//2 - 4, 32, "")
        self.load_btn    = SmBtn(rx + rw//2 + 4, 452, rw//2 - 4, 32, "⚡ Load Sorter")
        self.start_rect  = pygame.Rect(rx, WINDOW_HEIGHT-75, rw, 50)
        self.start_hov   = False

    def _rebuild_buttons(self):
        """Rebuild button list from ALGORITHMS + any loaded customs."""
        gx, gy = self.PAD, 80
        self.btns = []
        all_algos = list(ALGORITHMS)
        for key, fn in _custom_generators.items():
            # find existing entry if already added
            found = any(k == key for _, k in all_algos)
            if not found:
                # look it up in the global ALGORITHMS list (we append there on load)
                pass
        # Just use the global ALGORITHMS list which we mutate on load
        for i, (nm, ky) in enumerate(ALGORITHMS):
            col = i % self.COLS; row = i // self.COLS
            x = gx + col*(self.BTN_W+self.COL_GAP)
            y = gy + row*(AlgoBtn.H+self.BTN_GAP)
            self.btns.append(AlgoBtn(x, y, self.BTN_W, nm, ky, i))

    def _radix(self):
        key = ALGORITHMS[self.sel][1]
        return key in ("lsd_radix", "msd_radix")

    def handle(self, ev):
        self.sl_size.handle(ev); self.sl_speed.handle(ev)
        self.sl_sustain.handle(ev); self.sl_freq.handle(ev)

        if ev.type == pygame.MOUSEMOTION:
            self.hov = -1; self.start_hov = self.start_rect.collidepoint(ev.pos)
            for b in self.btns:
                if b.rect.collidepoint(ev.pos): self.hov = b.idx

        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            for b in self.btns:
                if b.rect.collidepoint(ev.pos): self.sel = b.idx

            if self._radix():
                for sb, base in self.base_btns:
                    if sb.rect.collidepoint(ev.pos): self.radix_base = base

            if self.snd_btn.rect.collidepoint(ev.pos):
                self.sound_on = not self.sound_on

            if self.load_btn.rect.collidepoint(ev.pos):
                self._do_load()

            if self.start_rect.collidepoint(ev.pos):
                return "start"

        return None

    def _do_load(self):
        path = open_file_dialog()
        if not path: return
        result, err = load_custom_sorter(path)
        if err:
            self.err_msg = f"Load error: {err[:60]}"
            self.err_timer = pygame.time.get_ticks() + 4000
            return
        name, key = result
        # Append to global list if not already there
        if not any(k == key for _, k in ALGORITHMS):
            ALGORITHMS.append((name, key))
        # Rebuild buttons & select the new one
        self._rebuild_buttons()
        self.sel = len(ALGORITHMS) - 1
        self.err_msg = f"Loaded: {name}"
        self.err_timer = pygame.time.get_ticks() + 3000

    def draw(self):
        s = self.screen; rx = self.rx; rw = WINDOW_WIDTH - rx - self.PAD
        s.fill(UI_BG)

        # Title
        t  = self.fonts['title'].render("SuperSorter", True, UI_TEXT)
        t2 = self.fonts['title'].render("SuperSorter", True, UI_ACCENT)
        # slight glow by drawing accent version offset
        s.blit(t2, (self.PAD+1, 23))
        s.blit(t,  (self.PAD,   22))
        s.blit(self.fonts['small'].render(f"{len(ALGORITHMS)} algorithms  •  SoS audio", True, UI_SUBTEXT),
               (self.PAD + t.get_width() + 14, 31))
        pygame.draw.line(s, UI_BORDER, (self.PAD, 70), (WINDOW_WIDTH-self.PAD, 70), 1)

        # Algorithm buttons
        for b in self.btns: b.draw(s, self.fonts, b.idx==self.sel, b.idx==self.hov)

        # Right panel
        panel = pygame.Rect(rx-12, 78, rw+24, WINDOW_HEIGHT-93)
        pygame.draw.rect(s, UI_PANEL,  panel, border_radius=8)
        pygame.draw.rect(s, UI_BORDER, panel, 1, border_radius=8)
        s.blit(self.fonts['small'].render("SETTINGS", True, UI_SUBTEXT), (rx, 90))

        self.sl_size.draw(s, self.fonts)
        self.sl_speed.draw(s, self.fonts)
        self.sl_sustain.draw(s, self.fonts)
        self.sl_freq.draw(s, self.fonts)

        # Radix base
        mp = pygame.mouse.get_pos()
        if self._radix():
            s.blit(self.fonts['small'].render("Radix Base:", True, UI_SUBTEXT), (rx, 392))
            for sb, base in self.base_btns:
                sb.draw(s, self.fonts, self.radix_base==base, sb.rect.collidepoint(mp))
        else:
            s.blit(self.fonts['small'].render("Radix base: N/A for this algorithm", True, UI_DIM), (rx, 400))

        # Sound + Load buttons
        self.snd_btn.label = "♪ Sound ON" if self.sound_on else "✕ Sound OFF"
        self.snd_btn.draw(s, self.fonts, self.sound_on, self.snd_btn.rect.collidepoint(mp))
        self.load_btn.draw(s, self.fonts, False, self.load_btn.rect.collidepoint(mp))

        # Selected name
        nm, ky = ALGORITHMS[self.sel]
        s.blit(self.fonts['small'].render("Selected:", True, UI_SUBTEXT), (rx, 502))
        s.blit(self.fonts['mid'].render(nm, True, UI_ACCENT), (rx, 518))

        # Error / status message
        now = pygame.time.get_ticks()
        if self.err_msg and now < self.err_timer:
            is_err = self.err_msg.startswith("Load error")
            col    = (255, 90, 90) if is_err else (90, 220, 130)
            s.blit(self.fonts['small'].render(self.err_msg, True, col), (rx, 542))
        elif now >= self.err_timer:
            self.err_msg = ""

        s.blit(self.fonts['small'].render("ESC during sort returns to menu", True, UI_DIM), (rx, 560))

        # Start button
        sh = self.start_hov
        pygame.draw.rect(s, (240,50,50) if sh else (200,35,35), self.start_rect, border_radius=8)
        pygame.draw.rect(s, (255,90,90) if sh else UI_ACCENT,   self.start_rect, 2, border_radius=8)
        st = self.fonts['big'].render("▶  START", True, (255,255,255))
        s.blit(st, st.get_rect(center=self.start_rect.center))

        pygame.display.flip()

    def config(self):
        nm, ky = ALGORITHMS[self.sel]
        return dict(
            name=nm, key=ky,
            size=self.sl_size.value,
            speed=self.sl_speed.value,
            sustain=self.sl_sustain.value,
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
    sans = ["Segoe UI", "Helvetica Neue", "Arial"]
    return dict(title=tf(mono, 26), big=tf(sans, 24), mid=tf(sans, 18),
                small=tf(sans, 14), mono_sm=tf(mono, 13))

def run_sort(screen, fonts, cfg):
    global ENABLE_SOUND, FREQ_LOW, FREQ_HIGH
    ENABLE_SOUND = cfg["sound"]
    FREQ_LOW     = cfg["freq_lo"]
    FREQ_HIGH    = cfg["freq_hi"]
    if _engine: _engine.set_sustain(cfg["sustain"])

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
            draw_bars(screen, arr, [], label + "  ✓  SORTED")
            pygame.time.wait(1800); return

def main():
    pygame.init(); init_sound()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("SuperSorter")
    force_top()
    fonts = build_fonts(); clock = pygame.time.Clock()

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