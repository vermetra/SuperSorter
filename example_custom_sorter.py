# ============================================================
# SuperSorter - Custom Sorter Template
# ============================================================
#
# Rules:
#   1. Define a function called  sort(arr)
#   2. It must be a generator that yields (arr, [active_indices])
#      on every "interesting" step (compare, swap, write).
#   3. Mutate `arr` in-place - do NOT return a new list.
#   4. Optionally set NAME = "My Algorithm"  (used as display name)
#
# Load this file from SuperSorter via the "⚡ Load Sorter" button.
# ============================================================

NAME = "Stooge Sort"   # <-- change this to whatever you like


def sort(arr):
    """Stooge Sort - O(n^2.7) - famously terrible, famously entertaining."""

    def stooge(lo, hi):
        if arr[lo] > arr[hi]:
            arr[lo], arr[hi] = arr[hi], arr[lo]
            yield arr, [lo, hi]

        if hi - lo + 1 > 2:
            t = (hi - lo + 1) // 3
            yield from stooge(lo, hi - t)
            yield from stooge(lo + t, hi)
            yield from stooge(lo, hi - t)

    yield from stooge(0, len(arr) - 1)
