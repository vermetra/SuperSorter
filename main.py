import pygame
import random
import numpy as np
import sys
import ctypes

# ============================================================
# ===================== USER SETTINGS ========================
# ============================================================

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 600
MAX_ARRAY_SIZE = 64
FPS = 120

BACKGROUND_COLOR = (0, 0, 0)
ACTIVE_COLOR = (255, 60, 60)
BAR_SPACING = 2

ENABLE_SOUND = True
MIN_FREQUENCY = 40
MAX_FREQUENCY = 360
SOUND_DURATION = 0.025

# ============================================================

def generate_array(size):
    arr = list(range(1, size + 1))
    random.shuffle(arr)
    return arr

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
    envelope[:fade_len] = np.linspace(0, 1, fade_len)
    envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    wave *= envelope

    mono = (wave * 32767).astype(np.int16)
    stereo = np.column_stack((mono, mono))
    sound = pygame.mixer.Sound(buffer=stereo.tobytes())
    sound.play()

# ============================================================
# ======================= COLOR SYSTEM =======================
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

def draw_bars(screen, array, active_indices):
    screen.fill(BACKGROUND_COLOR)
    bar_width = WINDOW_WIDTH / len(array)
    for index, value in enumerate(array):
        x = index * bar_width
        height = (value / len(array)) * (WINDOW_HEIGHT - 50)
        color = ACTIVE_COLOR if index in active_indices else value_to_color(value, len(array))
        pygame.draw.rect(screen, color, (x, WINDOW_HEIGHT - height, bar_width - BAR_SPACING, height))
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

# ====================== RADIX SORT ===========================

def counting_sort_radix(arr, exp, base):
    n = len(arr)
    output = [0] * n
    count = [0] * base

    for i in range(n):
        index = (arr[i] // exp) % base
        count[index] += 1
        yield arr, [i]

    for i in range(1, base):
        count[i] += count[i - 1]

    for i in range(n - 1, -1, -1):
        index = (arr[i] // exp) % base
        output[count[index] - 1] = arr[i]
        count[index] -= 1
        yield arr, [i]

    for i in range(n):
        arr[i] = output[i]
        yield arr, [i]

def lsd_radix_sort(arr, base=10):
    max_val = max(arr)
    exp = 1
    while max_val // exp > 0:
        yield from counting_sort_radix(arr, exp, base)
        exp *= base

def msd_radix_sort(arr, base=10):
    def msd_helper(arr, lo, hi, exp):
        if hi - lo <= 1 or exp == 0:
            return

        # Collect buckets from the slice arr[lo:hi]
        buckets = [[] for _ in range(base)]
        for i in range(lo, hi):
            digit = (arr[i] // exp) % base
            buckets[digit].append(arr[i])
            yield arr, [i]

        # Write buckets back into arr[lo:hi] in order
        i = lo
        for bucket in buckets:
            for num in bucket:
                arr[i] = num
                i += 1
            if bucket:
                yield arr, list(range(i - len(bucket), i))

        # Recurse on each bucket's sub-range within arr
        i = lo
        for bucket in buckets:
            if len(bucket) > 1:
                yield from msd_helper(arr, i, i + len(bucket), exp // base)
            i += len(bucket)

    if not arr:
        return

    max_val = max(arr)
    exp = 1
    while max_val // exp >= base:
        exp *= base

    yield from msd_helper(arr, 0, len(arr), exp)

# ============================================================
# ====================== USER INPUT ==========================
# ============================================================

def get_user_choice():
    print("\nChoose sorting algorithm:")
    print("1. Bubble Sort")
    print("2. Insertion Sort")
    print("3. Selection Sort")
    print("4. Radix Sort")
    print("5. Quit")

    while True:
        choice = input("Enter option number: ")
        if choice in ["1", "2", "3", "4", "5"]:
            break
        print("Invalid choice.")

    if choice == "5":
        return choice, None, None, None

    while True:
        try:
            size = int(input(f"Enter array size (1 - {MAX_ARRAY_SIZE}): "))
            if 1 <= size <= MAX_ARRAY_SIZE:
                break
        except:
            pass
        print("Invalid size.")

    while True:
        try:
            speed = float(input("Enter speed multiplier (1=normal, 2=faster, 0.5=slower): "))
            if speed > 0:
                break
        except:
            pass
        print("Invalid speed.")

    radix_type = None
    if choice == "4":
        print("Choose radix sort type:")
        print("1. LSD Base-10")
        print("2. MSD Base-10")
        print("3. LSD Base-N (custom)")
        while True:
            radix_choice = input("Enter option number: ")
            if radix_choice in ["1", "2", "3"]:
                break
            print("Invalid choice.")
        if radix_choice == "3":
            while True:
                try:
                    base = int(input("Enter base (>=2): "))
                    if base >= 2:
                        break
                except:
                    pass
                print("Invalid base.")
            radix_type = ("LSD", base)
        elif radix_choice == "2":
            radix_type = ("MSD", 10)
        else:
            radix_type = ("LSD", 10)

    return choice, size, speed, radix_type

# ============================================================
# ========================= MAIN =============================
# ============================================================

def force_window_on_top():
    hwnd = pygame.display.get_wm_info()['window']
    ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0001 | 0x0002)

def run_sort(choice, size, speed, radix_type):
    pygame.init()
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Sorting Visualizer")
    force_window_on_top()

    array = generate_array(size)

    if choice == "1":
        sort_generator = bubble_sort(array)
    elif choice == "2":
        sort_generator = insertion_sort(array)
    elif choice == "3":
        sort_generator = selection_sort(array)
    else:
        if radix_type[0] == "LSD":
            sort_generator = lsd_radix_sort(array, radix_type[1])
        else:
            sort_generator = msd_radix_sort(array, radix_type[1])

    clock = pygame.time.Clock()
    running = True

    while running:
        clock.tick(FPS * speed)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                sys.exit()

        try:
            arr_state, active = next(sort_generator)
            draw_bars(screen, arr_state, active)
            if active:
                play_tone(arr_state[active[0]], size)
        except StopIteration:
            draw_bars(screen, array, [])
            pygame.time.wait(1200)
            running = False

    pygame.quit()

def main():
    while True:
        choice, size, speed, radix_type = get_user_choice()
        if choice == "5":
            print("Exiting program.")
            break
        run_sort(choice, size, speed, radix_type)

if __name__ == "__main__":
    main()