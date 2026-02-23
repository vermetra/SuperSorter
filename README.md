# SuperSolver

A visual sorting and maze-solving algorithm playground built with Python and pygame-ce. Watch algorithms come to life with a colour-coded bar visualiser, a fully synthesised sound engine, and an interactive maze pathfinder — all in a sleek dark-themed GUI.

---

## Features

### Sorting Visualiser
- **15 built-in sorting algorithms** including Bubble, Quick, Merge, Heap, Shell, Radix (LSD + MSD), Cycle, Pancake, and more
- Colour-coded bars mapped to value (blue → cyan → green → yellow → red)
- Adjustable **array size** (4–128 elements) and **playback speed** (0.25×–8×)
- Configurable **frequency range** for the sound engine (dual-knob slider)
- Selectable **radix base** (2, 4, 8, 10, 16) for radix sort variants
- **Custom sorter loader** — drop in any `.py` file or a `.zip` SortPack

### Maze Visualiser
- **7 built-in pathfinding algorithms**: A* (Manhattan), BFS, DFS, Dijkstra, Greedy Best-First, Wall Follower (right-hand rule), Dead-End Filling
- Three **maze generation styles**: Recursive Backtracker (perfect maze), Prim's algorithm, and Open Grid
- Adjustable **maze size** (5–40 logical cells) and **solve speed** (0.25×–16×)
- Live colour feedback — walls, open cells, visited, frontier, path, current cell, start, and end all distinctly coloured
- **Custom solver loader** — plug in your own `.py` pathfinder

### Sound Engine
- Continuous-stream additive sine synthesiser using **sounddevice** (PortAudio) — no choppy queuing, no seams
- Raised-cosine (Hann) attack and release envelopes — completely click-free note transitions
- 2nd and 3rd harmonic blending for warmth and character
- Voice stealing with smooth cosine fade when the voice pool is full
- Trigger rate limiter so notes have room to breathe during fast sorts
- Automatic fallback to **pygame mixer** if sounddevice is not installed

---

## Requirements

- **Python 3.8+**
- **pygame-ce** (Community Edition — *not* the original pygame)
- **numpy**
- **sounddevice** *(optional but strongly recommended for quality audio)*

> ⚠️ This project uses **pygame-ce**, not `pygame`. Do not install both — they conflict. Uninstall regular `pygame` first if you have it.

---

## Installation

```bash
# 1. Uninstall regular pygame if present
pip uninstall pygame

# 2. Install pygame-ce and numpy
pip install pygame-ce numpy

# 3. (Recommended) Install sounddevice for the full audio experience
pip install sounddevice
```

Then run:

```bash
python supersorter.py
```

---

## Controls

| Action | Input |
|---|---|
| Select algorithm | Click any button in the left panel |
| Adjust settings | Drag the sliders in the right panel |
| Start visualisation | Click **> START** |
| Return to menu | Press **ESC** |
| Quit | Press **ESC** from the menu, or close the window |

---

## Custom Sorters

You can load your own sorting algorithm as a `.py` file. The file must define a `sort(arr)` generator function that yields `(array, active_indices)` at each step. Optionally define a `NAME` string for the display label.

**Example** (`my_sorter.py`):

```python
NAME = "My Custom Sort"

def sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            yield arr, [j, j + 1]
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                yield arr, [j, j + 1]
```

Load via **[+ Load Custom Sorter]** in the Sort menu. Custom sorters are saved to `supersorter_custom.json` and auto-loaded on next launch.

### SortPacks

Bundle multiple sorters into a `.zip` file and load them all at once. The pack name is read from the first line of any `.txt` file inside the archive.

---

## Custom Maze Solvers

Create a `.py` file with a `solve(grid, start, end)` generator. The `grid` is a 2D list of `0` (open) and `1` (wall). Yield `(visited_set, frontier_set, path_list, current_cell)` at each step.

**Example** (`my_solver.py`):

```python
NAME = "My Custom Solver"
from collections import deque

def solve(grid, start, end):
    queue = deque([start])
    came_from = {start: None}
    visited = set()

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
        yield visited.copy(), set(queue), reconstruct(cur) if cur == end else [], cur
        if cur == end:
            return
        r, c = cur
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] == 0:
                if (nr, nc) not in came_from:
                    came_from[(nr, nc)] = cur
                    queue.append((nr, nc))
```

Load via **[+ Load Custom Solver]** in the Maze menu. Custom solvers are saved to `supersorter_maze_custom.json` and auto-loaded on next launch.

---

## Sound Engine Details

SuperSolver uses a **pull-callback audio model** via sounddevice/PortAudio — a single unbroken output stream that the OS audio driver fills on demand. This is fundamentally different from pygame.mixer's queued Sound objects, which create audible seams between notes.

Each triggered tone is an `_Osc` voice with:
- A sine wave fundamental plus optional 2nd and 3rd harmonics
- A raised-cosine (Hann) envelope for click-free attack and release
- A random initial phase to prevent onset pops
- A configurable lifetime (sustain, attack, release all tunable in the source)

Key tuneable constants at the top of the file:

| Constant | Default | Description |
|---|---|---|
| `SOUND_SUSTAIN` | `0.25` | Total note lifetime in seconds |
| `SOUND_ATTACK` | `0.015` | Fade-in time (raised-cosine) |
| `SOUND_RELEASE` | `0.120` | Fade-out time (raised-cosine) |
| `HARMONIC_BLEND` | `0.12` | 2nd harmonic mix (warmth) |
| `HARMONIC_BLEND_3` | `0.07` | 3rd harmonic mix (grit) |
| `MAX_VOICES` | `32` | Max simultaneous oscillators |
| `TRIGGER_RATE_LIMIT` | `0.040` | Min seconds between note triggers |
| `FREQ_LOW` | `120.0` | Frequency for lowest array value (Hz) |
| `FREQ_HIGH` | `360.0` | Frequency for highest array value (Hz) |

---

## Project Structure

```
supersorter.py            — Main application (single file)
supersorter_custom.json   — Auto-generated: paths to loaded custom sorters
supersorter_maze_custom.json — Auto-generated: paths to loaded maze solvers
```

---

## Maze Colour Key

| Colour | Meaning |
|---|---|
| Near-black | Wall |
| Dark grey | Open passage |
| Deep blue | Visited cell |
| Cyan | Frontier (queued) |
| Bright blue | Solution path |
| Red | Current cell |
| Green | Start |
| Red (bright) | End |

---

## Troubleshooting

**No sound / import error for sounddevice**
Run `pip install sounddevice`. The app will fall back to the pygame mixer if sounddevice is unavailable, but audio quality will be reduced.

**`pygame` and `pygame-ce` conflict**
Only one can be installed at a time. Run `pip uninstall pygame` then `pip install pygame-ce`.

**Maze solver hangs or is very slow**
Reduce the maze size via the slider. Large mazes (N=40) with DFS can explore a very large number of cells before finding the exit.

**Custom sorter not appearing after reload**
Check that your `.py` file defines a `sort(arr)` generator (uses `yield`). Regular functions will fail silently with a descriptive error message shown in the menu.

---

## License

MIT — do whatever you like with it.
