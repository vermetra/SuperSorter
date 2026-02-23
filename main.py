import pygame
import random
import numpy as np
import sys
import math

try:
    import ctypes
    HAS_CTYPES = True
except ImportError:
    HAS_CTYPES = False

# ============================================================
# ===================== USER SETTINGS ========================
# ============================================================

WINDOW_WIDTH  = 1100
WINDOW_HEIGHT = 680
MAX_ARRAY_SIZE = 128
FPS = 120

BACKGROUND_COLOR = (5, 5, 10)
ACTIVE_COLOR     = (255, 60, 60)
BAR_SPACING      = 1

ENABLE_SOUND   = True
MIN_FREQUENCY  = 60
MAX_FREQUENCY  = 900
SOUND_DURATION = 0.025

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

ALGORITHMS = [
    ("Bubble Sort",       "bubble"),
    ("Insertion Sort",    "insertion"),
    ("Selection Sort",    "selection"),
    ("Quick Sort",        "quick"),
    ("Merge Sort",        "merge"),
    ("Heap Sort",         "heap"),
    ("Shell Sort",        "shell"),
    ("Cocktail Shaker",   "cocktail"),
    ("Gnome Sort",        "gnome"),
    ("Comb Sort",         "comb"),
    ("Cycle Sort",        "cycle"),
    ("Pancake Sort",      "pancake"),
    ("Odd-Even Sort",     "oddeven"),
    ("LSD Radix Sort",    "lsd_radix"),
    ("MSD Radix Sort",    "msd_radix"),
]

# ============================================================
# ======================= SOUND SYSTEM =======================
# ============================================================

def play_tone(value, max_value):
    if not ENABLE_SOUND:
        return
    sample_rate = 44100
    frequency = MIN_FREQUENCY + ((value / max_value) * (MAX_FREQUENCY - MIN_FREQUENCY))
    t = np.linspace(0, SOUND_DURATION, int(sample_rate * SOUND_DURATION), endpoint=False)
    wave = np.sin(2 * np.pi * frequency * t)
    fade_len = int(0.005 * sample_rate)
    envelope = np.ones_like(wave)
    envelope[:fade_len]  = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    wave *= envelope
    mono   = (wave * 32767).astype(np.int16)
    stereo = np.column_stack((mono, mono))
    sound  = pygame.mixer.Sound(buffer=stereo.tobytes())
    sound.play()

# ============================================================
# ======================= COLOR / DRAW =======================
# ============================================================

def value_to_color(value, max_value):
    ratio = value / max_value
    if ratio < 0.25:
        return (0, int(255 * ratio * 4), 255)
    elif ratio < 0.5:
        return (0, 255, int(255 * (1 - (ratio - 0.25) * 4)))
    elif ratio < 0.75:
        return (int(255 * (ratio - 0.5) * 4), 255, 0)
    else:
        return (255, int(255 * (1 - (ratio - 0.75) * 4)), 0)

def draw_bars(screen, array, active_indices, label=""):
    screen.fill(BACKGROUND_COLOR)
    n = len(array)
    bar_width = WINDOW_WIDTH / n
    for idx, val in enumerate(array):
        x      = idx * bar_width
        height = (val / n) * (WINDOW_HEIGHT - 60)
        color  = ACTIVE_COLOR if idx in active_indices else value_to_color(val, n)
        pygame.draw.rect(screen, color,
                         (x, WINDOW_HEIGHT - height, bar_width - BAR_SPACING, height))
    if label:
        font = pygame.font.SysFont("consolas", 18)
        surf = font.render(label, True, (140, 140, 160))
        screen.blit(surf, (12, 10))
    pygame.display.flip()

# ============================================================
# ===================== SORTING ALGORITHMS ===================
# ============================================================

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            yield arr, [j, j + 1]
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                yield arr, [j, j + 1]

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            yield arr, [j, j + 1]
            arr[j + 1] = arr[j]
            j -= 1
            yield arr, [j + 1]
        arr[j + 1] = key
        yield arr, [j + 1]

def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            yield arr, [min_idx, j]
            if arr[j] < arr[min_idx]:
                min_idx = j
                yield arr, [min_idx]
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
        yield arr, [i, min_idx]

def quick_sort(arr):
    def _quick(lo, hi):
        if lo >= hi:
            return
        pivot = arr[hi]
        i = lo - 1
        for j in range(lo, hi):
            yield arr, [j, hi]
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                yield arr, [i, j]
        arr[i + 1], arr[hi] = arr[hi], arr[i + 1]
        yield arr, [i + 1, hi]
        yield from _quick(lo, i)
        yield from _quick(i + 2, hi)
    yield from _quick(0, len(arr) - 1)

def merge_sort(arr):
    def _merge(lo, mid, hi):
        left  = arr[lo:mid + 1]
        right = arr[mid + 1:hi + 1]
        i = j = 0
        k = lo
        while i < len(left) and j < len(right):
            yield arr, [lo + i, mid + 1 + j]
            if left[i] <= right[j]:
                arr[k] = left[i]; i += 1
            else:
                arr[k] = right[j]; j += 1
            yield arr, [k]
            k += 1
        while i < len(left):
            arr[k] = left[i]; yield arr, [k]; i += 1; k += 1
        while j < len(right):
            arr[k] = right[j]; yield arr, [k]; j += 1; k += 1

    def _ms(lo, hi):
        if lo < hi:
            mid = (lo + hi) // 2
            yield from _ms(lo, mid)
            yield from _ms(mid + 1, hi)
            yield from _merge(lo, mid, hi)
    yield from _ms(0, len(arr) - 1)

def heap_sort(arr):
    def heapify(n, i):
        largest, l, r = i, 2 * i + 1, 2 * i + 2
        if l < n and arr[l] > arr[largest]: largest = l
        if r < n and arr[r] > arr[largest]: largest = r
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            yield arr, [i, largest]
            yield from heapify(n, largest)
        else:
            yield arr, [i]

    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        yield from heapify(n, i)
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        yield arr, [0, i]
        yield from heapify(i, 0)

def shell_sort(arr):
    n, gap = len(arr), len(arr) // 2
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                yield arr, [j, j - gap]
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
            yield arr, [j]
        gap //= 2

def cocktail_sort(arr):
    lo, hi = 0, len(arr) - 1
    while lo < hi:
        for i in range(lo, hi):
            yield arr, [i, i + 1]
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                yield arr, [i, i + 1]
        hi -= 1
        for i in range(hi, lo, -1):
            yield arr, [i, i - 1]
            if arr[i] < arr[i - 1]:
                arr[i], arr[i - 1] = arr[i - 1], arr[i]
                yield arr, [i, i - 1]
        lo += 1

def gnome_sort(arr):
    i = 0
    while i < len(arr):
        yield arr, [i]
        if i == 0 or arr[i] >= arr[i - 1]:
            i += 1
        else:
            arr[i], arr[i - 1] = arr[i - 1], arr[i]
            yield arr, [i, i - 1]
            i -= 1

def comb_sort(arr):
    n, gap, shrink = len(arr), len(arr), 1.3
    sorted_ = False
    while not sorted_:
        gap = int(gap / shrink)
        if gap <= 1:
            gap = 1
            sorted_ = True
        for i in range(n - gap):
            yield arr, [i, i + gap]
            if arr[i] > arr[i + gap]:
                arr[i], arr[i + gap] = arr[i + gap], arr[i]
                sorted_ = False
                yield arr, [i, i + gap]

def cycle_sort(arr):
    n = len(arr)
    for cs in range(n - 1):
        item = arr[cs]
        pos  = cs
        for i in range(cs + 1, n):
            yield arr, [i, cs]
            if arr[i] < item:
                pos += 1
        if pos == cs:
            continue
        while item == arr[pos]:
            pos += 1
        arr[pos], item = item, arr[pos]
        yield arr, [pos, cs]
        while pos != cs:
            pos = cs
            for i in range(cs + 1, n):
                yield arr, [i, cs]
                if arr[i] < item:
                    pos += 1
            while item == arr[pos]:
                pos += 1
            arr[pos], item = item, arr[pos]
            yield arr, [pos]

def pancake_sort(arr):
    def flip(k):
        lo, hi = 0, k
        while lo < hi:
            arr[lo], arr[hi] = arr[hi], arr[lo]
            yield arr, [lo, hi]
            lo += 1; hi -= 1

    for size in range(len(arr), 1, -1):
        max_idx = arr.index(max(arr[:size]))
        if max_idx != size - 1:
            if max_idx != 0:
                yield from flip(max_idx)
            yield from flip(size - 1)

def odd_even_sort(arr):
    n = len(arr)
    sorted_ = False
    while not sorted_:
        sorted_ = True
        for i in range(1, n - 1, 2):
            yield arr, [i, i + 1]
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                sorted_ = False; yield arr, [i, i + 1]
        for i in range(0, n - 1, 2):
            yield arr, [i, i + 1]
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                sorted_ = False; yield arr, [i, i + 1]

def counting_sort_radix(arr, exp, base):
    n      = len(arr)
    output = [0] * n
    count  = [0] * base
    for i in range(n):
        count[(arr[i] // exp) % base] += 1
        yield arr, [i]
    for i in range(1, base):
        count[i] += count[i - 1]
    for i in range(n - 1, -1, -1):
        idx = (arr[i] // exp) % base
        output[count[idx] - 1] = arr[i]
        count[idx] -= 1
        yield arr, [i]
    for i in range(n):
        arr[i] = output[i]
        yield arr, [i]

def lsd_radix_sort(arr, base=10):
    max_val, exp = max(arr), 1
    while max_val // exp > 0:
        yield from counting_sort_radix(arr, exp, base)
        exp *= base

def msd_radix_sort(arr, base=10):
    def msd_helper(lo, hi, exp):
        if hi - lo <= 1 or exp == 0:
            return
        buckets = [[] for _ in range(base)]
        for i in range(lo, hi):
            buckets[(arr[i] // exp) % base].append(arr[i])
            yield arr, [i]
        i = lo
        for bucket in buckets:
            for num in bucket:
                arr[i] = num; i += 1
            if bucket:
                yield arr, list(range(i - len(bucket), i))
        i = lo
        for bucket in buckets:
            if len(bucket) > 1:
                yield from msd_helper(i, i + len(bucket), exp // base)
            i += len(bucket)

    if not arr:
        return
    max_val, exp = max(arr), 1
    while max_val // exp >= base:
        exp *= base
    yield from msd_helper(0, len(arr), exp)

def get_sort_generator(key, arr, radix_base=10):
    return {
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
    }[key]()

# ============================================================
# ========================= UI CLASSES =======================
# ============================================================

class Slider:
    def __init__(self, x, y, w, min_val, max_val, initial, label, is_int=False):
        self.x, self.y, self.w = x, y, w
        self.min_val  = min_val
        self.max_val  = max_val
        self.value    = initial
        self.label    = label
        self.is_int   = is_int
        self.dragging = False
        self.track    = pygame.Rect(x, y + 18, w, 5)
        self.hit_area = pygame.Rect(x - 5, y, w + 10, 42)

    def _ratio(self):
        return (self.value - self.min_val) / (self.max_val - self.min_val)

    def _knob_x(self):
        return int(self.x + self._ratio() * self.w)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            kx, ky = self._knob_x(), self.track.centery
            if math.hypot(event.pos[0] - kx, event.pos[1] - ky) < 14 \
               or self.hit_area.collidepoint(event.pos):
                self.dragging = True
                self._set(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set(event.pos[0])

    def _set(self, mx):
        ratio = max(0.0, min(1.0, (mx - self.x) / self.w))
        raw = self.min_val + ratio * (self.max_val - self.min_val)
        self.value = int(round(raw)) if self.is_int else round(raw * 4) / 4

    def draw(self, screen, fonts):
        # Label + value
        val_str = str(self.value) if self.is_int else f"{self.value:.2f}x"
        lbl = fonts['small'].render(f"{self.label}:  {val_str}", True, UI_SUBTEXT)
        screen.blit(lbl, (self.x, self.y))
        # Track background
        pygame.draw.rect(screen, UI_BORDER, self.track, border_radius=3)
        # Fill
        fill_w = int(self._ratio() * self.w)
        if fill_w > 0:
            pygame.draw.rect(screen, UI_ACCENT,
                             (self.x, self.track.y, fill_w, 5), border_radius=3)
        # Knob
        kx = self._knob_x()
        ky = self.track.centery
        pygame.draw.circle(screen, UI_PANEL2, (kx, ky), 10)
        pygame.draw.circle(screen, UI_ACCENT, (kx, ky), 10, 2)
        pygame.draw.circle(screen, UI_ACCENT, (kx, ky), 4)


class AlgoButton:
    HEIGHT = 44
    def __init__(self, x, y, w, name, key, index):
        self.rect  = pygame.Rect(x, y, w, self.HEIGHT)
        self.name  = name
        self.key   = key
        self.index = index

    def draw(self, screen, fonts, selected, hovered):
        bg     = UI_SEL_BG  if selected else (UI_HOVER if hovered else UI_PANEL)
        border = UI_SEL_BORDER if selected else (UI_DIM if hovered else UI_BORDER)
        pygame.draw.rect(screen, bg,     self.rect, border_radius=6)
        pygame.draw.rect(screen, border, self.rect, 1, border_radius=6)

        num_col  = UI_ACCENT  if selected else UI_SUBTEXT
        name_col = UI_TEXT    if selected else (UI_TEXT if hovered else (160, 160, 180))

        num_surf  = fonts['mono_sm'].render(f"{self.index + 1:02d}", True, num_col)
        name_surf = fonts['mid'].render(self.name, True, name_col)

        screen.blit(num_surf,  (self.rect.x + 10, self.rect.y + 14))
        screen.blit(name_surf, (self.rect.x + 42, self.rect.y + 13))


class SmallButton:
    def __init__(self, x, y, w, h, label):
        self.rect  = pygame.Rect(x, y, w, h)
        self.label = label

    def draw(self, screen, fonts, active=False, hovered=False):
        bg  = UI_ACCENT if active else (UI_HOVER if hovered else UI_PANEL2)
        col = (0, 0, 0)  if active else UI_TEXT
        pygame.draw.rect(screen, bg,       self.rect, border_radius=5)
        pygame.draw.rect(screen, UI_BORDER, self.rect, 1, border_radius=5)
        surf = fonts['small'].render(self.label, True, col)
        screen.blit(surf, surf.get_rect(center=self.rect.center))

# ============================================================
# ====================== MENU SCREEN =========================
# ============================================================

class MenuScreen:
    PAD        = 18
    BTN_W      = 226
    BTN_GAP    = 6
    COL_GAP    = 10
    COLS       = 3
    ROWS       = 5          # 3 * 5 = 15 exactly

    def __init__(self, screen, fonts):
        self.screen    = screen
        self.fonts     = fonts
        self.selected  = 0
        self.hovered   = -1
        self.sound_on  = ENABLE_SOUND

        # Layout: algorithm grid on the left, settings on the right
        grid_x = self.PAD
        grid_y = 82

        self.buttons = []
        for idx, (name, key) in enumerate(ALGORITHMS):
            col = idx % self.COLS
            row = idx // self.COLS
            x = grid_x + col * (self.BTN_W + self.COL_GAP)
            y = grid_y + row * (AlgoButton.HEIGHT + self.BTN_GAP)
            self.buttons.append(AlgoButton(x, y, self.BTN_W, name, key, idx))

        # Right panel
        rx = grid_x + self.COLS * (self.BTN_W + self.COL_GAP) + 20
        self.right_x = rx
        rw = WINDOW_WIDTH - rx - self.PAD

        self.slider_size  = Slider(rx, 130, rw, 4,  MAX_ARRAY_SIZE, 32,  "Array Size", is_int=True)
        self.slider_speed = Slider(rx, 200, rw, 0.25, 8.0,          1.0, "Speed")

        # Radix base selector (only shown for radix sorts)
        self.radix_base   = 10
        self.base_btns    = []
        bases = [2, 4, 8, 10, 16]
        bw = (rw - (len(bases)-1)*6) // len(bases)
        for i, b in enumerate(bases):
            bx = rx + i * (bw + 6)
            self.base_btns.append((SmallButton(bx, 300, bw, 30, f"Base {b}"), b))

        # Sound toggle
        self.sound_btn = SmallButton(rx, 365, rw, 34, "")

        # Start button
        sy = WINDOW_HEIGHT - 85
        self.start_rect    = pygame.Rect(rx, sy, rw, 52)
        self.start_hovered = False

    def _is_radix(self):
        return ALGORITHMS[self.selected][1] in ("lsd_radix", "msd_radix")

    def handle_event(self, event):
        self.slider_size.handle_event(event)
        self.slider_speed.handle_event(event)

        if event.type == pygame.MOUSEMOTION:
            self.hovered      = -1
            self.start_hovered = self.start_rect.collidepoint(event.pos)
            for btn in self.buttons:
                if btn.rect.collidepoint(event.pos):
                    self.hovered = btn.index

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for btn in self.buttons:
                if btn.rect.collidepoint(event.pos):
                    self.selected = btn.index

            if self._is_radix():
                for sb, base in self.base_btns:
                    if sb.rect.collidepoint(event.pos):
                        self.radix_base = base

            if self.sound_btn.rect.collidepoint(event.pos):
                self.sound_on = not self.sound_on

            if self.start_rect.collidepoint(event.pos):
                return "start"

        return None

    def draw(self):
        s = self.screen
        s.fill(UI_BG)

        # ---- Title bar ----
        title = self.fonts['title'].render("SORTING VISUALIZER", True, UI_TEXT)
        s.blit(title, (self.PAD, 22))
        ver = self.fonts['small'].render("15 ALGORITHMS", True, UI_SUBTEXT)
        s.blit(ver, (self.PAD + title.get_width() + 14, 31))

        # Divider
        pygame.draw.line(s, UI_BORDER, (self.PAD, 72), (WINDOW_WIDTH - self.PAD, 72), 1)

        # ---- Algorithm buttons ----
        for btn in self.buttons:
            btn.draw(s, self.fonts, btn.index == self.selected, btn.index == self.hovered)

        # ---- Right panel background ----
        rx  = self.right_x
        rw  = WINDOW_WIDTH - rx - self.PAD
        panel = pygame.Rect(rx - 12, 80, rw + 24, WINDOW_HEIGHT - 95)
        pygame.draw.rect(s, UI_PANEL, panel, border_radius=8)
        pygame.draw.rect(s, UI_BORDER, panel, 1, border_radius=8)

        # Settings label
        hdr = self.fonts['small'].render("SETTINGS", True, UI_SUBTEXT)
        s.blit(hdr, (rx, 92))

        # Sliders
        self.slider_size.draw(s, self.fonts)
        self.slider_speed.draw(s, self.fonts)

        # Radix options
        if self._is_radix():
            lbl = self.fonts['small'].render("Radix Base:", True, UI_SUBTEXT)
            s.blit(lbl, (rx, 278))
            for sb, base in self.base_btns:
                sb.draw(s, self.fonts,
                        active=self.radix_base == base,
                        hovered=sb.rect.collidepoint(pygame.mouse.get_pos()))
        else:
            # Dim the radix area with a note
            note = self.fonts['small'].render("Radix base: N/A for this algorithm", True, UI_DIM)
            s.blit(note, (rx, 288))

        # Sound toggle
        sy_lbl = 352
        lbl = self.fonts['small'].render("Sound:", True, UI_SUBTEXT)
        s.blit(lbl, (rx, sy_lbl))
        self.sound_btn.rect.y = sy_lbl + 20
        self.sound_btn.label  = "ON  ♪" if self.sound_on else "OFF ✕"
        self.sound_btn.draw(s, self.fonts,
                            active=self.sound_on,
                            hovered=self.sound_btn.rect.collidepoint(pygame.mouse.get_pos()))

        # Selected algorithm info
        name, key = ALGORITHMS[self.selected]
        info_y = 440
        info_lbl = self.fonts['small'].render("Selected:", True, UI_SUBTEXT)
        info_val = self.fonts['mid'].render(name, True, UI_ACCENT)
        s.blit(info_lbl, (rx, info_y))
        s.blit(info_val, (rx, info_y + 18))

        # START button
        sh = self.start_hovered
        start_col  = (240, 50, 50) if sh else (200, 35, 35)
        border_col = (255, 90, 90) if sh else UI_ACCENT
        pygame.draw.rect(s, start_col, self.start_rect, border_radius=8)
        pygame.draw.rect(s, border_col, self.start_rect, 2, border_radius=8)
        start_txt = self.fonts['big'].render("▶  START", True, (255, 255, 255))
        s.blit(start_txt, start_txt.get_rect(center=self.start_rect.center))

        pygame.display.flip()

    def get_config(self):
        name, key = ALGORITHMS[self.selected]
        return {
            "name":       name,
            "key":        key,
            "size":       self.slider_size.value,
            "speed":      self.slider_speed.value,
            "radix_base": self.radix_base,
            "sound":      self.sound_on,
        }

# ============================================================
# ========================= MAIN LOOP ========================
# ============================================================

def force_window_on_top(screen):
    if HAS_CTYPES and sys.platform == "win32":
        try:
            hwnd = pygame.display.get_wm_info()['window']
            ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)
        except Exception:
            pass

def build_fonts():
    def try_font(names, size):
        for n in names:
            try:
                f = pygame.font.SysFont(n, size)
                return f
            except Exception:
                pass
        return pygame.font.SysFont(None, size)

    mono_candidates = ["Consolas", "Courier New", "Lucida Console", "monospace"]
    sans_candidates = ["Segoe UI", "Helvetica Neue", "Arial", "sans-serif"]

    return {
        "title":   try_font(mono_candidates, 26),
        "big":     try_font(sans_candidates, 24),
        "mid":     try_font(sans_candidates, 18),
        "small":   try_font(sans_candidates, 14),
        "mono_sm": try_font(mono_candidates, 13),
    }

def run_sort(screen, fonts, config):
    global ENABLE_SOUND
    ENABLE_SOUND = config["sound"]

    arr       = list(range(1, config["size"] + 1))
    random.shuffle(arr)

    gen   = get_sort_generator(config["key"], arr, config["radix_base"])
    clock = pygame.time.Clock()
    label = config["name"]

    while True:
        clock.tick(FPS * config["speed"])
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return  # back to menu

        try:
            arr_state, active = next(gen)
            draw_bars(screen, arr_state, active, label)
            if active and ENABLE_SOUND:
                play_tone(arr_state[active[0]], config["size"])
        except StopIteration:
            draw_bars(screen, arr, [], label + "  ✓ SORTED")
            pygame.time.wait(1600)
            return

def main():
    pygame.init()
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Sorting Visualizer")
    force_window_on_top(screen)

    fonts = build_fonts()
    menu  = MenuScreen(screen, fonts)

    clock = pygame.time.Clock()

    while True:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit()

            result = menu.handle_event(event)
            if result == "start":
                config = menu.get_config()
                run_sort(screen, fonts, config)
                # After sorting, redraw menu
                menu = MenuScreen(screen, fonts)

        menu.draw()

if __name__ == "__main__":
    main()