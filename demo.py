"""
demo.py — Generate combined and animated visualizations for d-ary cuckoo hashing.

Outputs four chart types instead of one-chart-per-parameter:

  demo/compare_d/           All d values on one chart (fixed size + load_factor).
                            Best for showing how more hash functions reduce displacements.

  demo/load_progression/    All load factors on one chart (fixed size + d).
                            Best for showing the phase transition as the table fills up.

  demo/animated/per_d/      Animated GIF per (size, d): curve evolves as load_factor rises.

  demo/animated/d_compare/  Animated GIF per size: all d curves evolve together.
                            The headline demo — shows d=2 struggling while d=8 stays flat.

Simulation results are cached as JSON in demo/data/ so subsequent runs are fast.
Delete demo/data/ to force a full re-run.

---- Tweak these to control what gets generated ----
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import numpy as np

from config import N as CONFIG_N, PARTITIONED_HASH_TABLE
from hash_tables.hash_table import MaxDisplacementsExceededError
from hash_tables.random_walk_dary_hash_table import RandomWalkDaryHashTable
from utils import get_random_key

# ── Parameters ──────────────────────────────────────────────────────────────
DEMO_SIZES      = [10_000]                         # table sizes to include
DEMO_D          = [2, 3, 4, 5, 6, 8]              # d values (hash function counts)
DEMO_LF         = [0.4, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]  # load factors
N               = CONFIG_N                         # insertions per simulation (set in config.py)
ANIMATION_FPS   = 1.2                              # frames per second in GIFs
FRAME_PAUSE_MS  = int(1000 / ANIMATION_FPS)       # derived, don't change

DEMO_DIR        = Path("demo")
DATA_DIR        = DEMO_DIR / "data"
# ────────────────────────────────────────────────────────────────────────────


# ── Simulation + caching ─────────────────────────────────────────────────────

def run_or_load(d: int, size: int, lf: float) -> list[int]:
    cache = DATA_DIR / str(size) / str(d) / f"{lf}.json"

    if cache.exists():
        with open(cache) as f:
            return json.load(f)

    max_disp = size * 2
    table = RandomWalkDaryHashTable(
        size=size, d=d, max_displacements=max_disp,
        partitioned=PARTITIONED_HASH_TABLE,
    )

    try:
        table.fill_random(lf)
    except MaxDisplacementsExceededError:
        print(f"  FAILED to fill: size={size}, d={d}, lf={lf}")
        return []

    original = table.slots.copy()
    result = []

    for key in (get_random_key() for _ in range(N)):
        table.slots = original.copy()
        try:
            result.append(table.insert_key(key))
        except MaxDisplacementsExceededError:
            result.append(max_disp)

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(result, f)

    print(f"  Simulated  size={size:>7,}  d={d}  lf={lf:.4f}  mean={np.mean(result):.1f}")
    return result


def collect_all() -> dict:
    """Return data[size][d][lf] = displacement list."""
    print("Collecting simulation data (cached runs are instant)...")
    data: dict = {}
    for size in DEMO_SIZES:
        data[size] = {}
        for d in DEMO_D:
            data[size][d] = {}
            for lf in DEMO_LF:
                arr = run_or_load(d, size, lf)
                if arr:
                    data[size][d][lf] = arr
    return data


# ── Plot helpers ─────────────────────────────────────────────────────────────

def _style(ax, title, max_y=None):
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Insertions (sorted by displacement count)")
    ax.set_ylabel("Displacements per insertion")
    ax.grid(True, alpha=0.3)
    if max_y is not None:
        ax.set_ylim(0, max_y)
    ax.set_xlim(0, N)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  {path}")


# ── Chart type 1: d comparison (fixed size + load_factor) ────────────────────

def plot_compare_d(data: dict):
    print("\nGenerating d-comparison plots...")
    cmap = plt.get_cmap("tab10")

    for size in DEMO_SIZES:
        for lf in DEMO_LF:
            fig, ax = plt.subplots(figsize=(10, 6))
            plotted = False

            for i, d in enumerate(DEMO_D):
                arr = data.get(size, {}).get(d, {}).get(lf)
                if not arr:
                    continue
                ax.plot(
                    sorted(arr),
                    label=f"d={d}  (mean {np.mean(arr):.1f})",
                    color=cmap(i / len(DEMO_D)),
                    lw=2,
                )
                plotted = True

            if not plotted:
                plt.close(fig)
                continue

            _style(ax, f"Effect of d — size={size:,}, load_factor={lf}")
            ax.legend(loc="upper left")
            _save(fig, DEMO_DIR / "compare_d" / str(size) / f"lf{lf}.png")


# ── Chart type 2: load-factor progression (fixed size + d) ───────────────────

def plot_load_progression(data: dict):
    print("\nGenerating load-factor progression plots...")
    cmap = plt.get_cmap("plasma")

    for size in DEMO_SIZES:
        for d in DEMO_D:
            lf_data = data.get(size, {}).get(d, {})
            if not lf_data:
                continue

            lfs = sorted(lf_data)
            fig, ax = plt.subplots(figsize=(10, 6))

            for i, lf in enumerate(lfs):
                color = cmap(i / max(len(lfs) - 1, 1))
                ax.plot(sorted(lf_data[lf]), label=f"lf={lf}", color=color, lw=2)

            _style(ax, f"Load factor progression — size={size:,}, d={d}")
            ax.legend(loc="upper left", fontsize=9)
            _save(fig, DEMO_DIR / "load_progression" / str(size) / f"d{d}.png")


# ── Chart type 3: animated GIF per (size, d) — load factor evolves ───────────

def animate_per_d(data: dict):
    print("\nGenerating per-d animated GIFs...")

    for size in DEMO_SIZES:
        for d in DEMO_D:
            lf_data = data.get(size, {}).get(d, {})
            if not lf_data:
                continue

            lfs    = sorted(lf_data)
            arrays = [sorted(lf_data[lf]) for lf in lfs]
            max_y  = max(max(a) for a in arrays) * 1.1

            fig, ax = plt.subplots(figsize=(10, 6))
            (line,) = ax.plot([], [], lw=2.5, color="steelblue")
            title   = ax.set_title("")
            info    = ax.text(
                0.98, 0.95, "", transform=ax.transAxes,
                ha="right", va="top", fontsize=11,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
            )
            _style(ax, "", max_y=max_y)

            def update(frame):
                arr = arrays[frame]
                lf  = lfs[frame]
                xs  = list(range(len(arr)))
                line.set_data(xs, arr)

                # Rebuild fill_between each frame
                for coll in ax.collections:
                    coll.remove()
                ax.fill_between(xs, arr, alpha=0.15, color="steelblue")

                title.set_text(f"d={d}, size={size:,} — load factor: {lf:.4f}")
                info.set_text(f"mean: {np.mean(arr):.1f}   max: {max(arr)}")
                return line, title, info

            ani = animation.FuncAnimation(
                fig, update, frames=len(lfs),
                interval=FRAME_PAUSE_MS, blit=False, repeat=True,
            )

            out = DEMO_DIR / "animated" / "per_d" / str(size) / f"d{d}.gif"
            out.parent.mkdir(parents=True, exist_ok=True)
            ani.save(str(out), writer="pillow", fps=ANIMATION_FPS)
            plt.close(fig)
            print(f"  Saved  {out}")


# ── Chart type 4: animated GIF per size — all d curves evolve together ────────

def animate_d_compare(data: dict):
    print("\nGenerating d-comparison animated GIFs (headline demo)...")
    cmap = plt.get_cmap("tab10")

    for size in DEMO_SIZES:
        lfs = sorted(DEMO_LF)

        all_vals = [
            v
            for d in DEMO_D
            for lf in lfs
            for v in data.get(size, {}).get(d, {}).get(lf, [])
        ]
        if not all_vals:
            continue

        max_y  = max(all_vals) * 1.1
        colors = {d: cmap(i / len(DEMO_D)) for i, d in enumerate(DEMO_D)}

        fig, ax = plt.subplots(figsize=(12, 7))
        lines = {}
        for d in DEMO_D:
            (l,) = ax.plot([], [], lw=2.5, label=f"d={d}", color=colors[d])
            lines[d] = l

        title = ax.set_title("")
        _style(ax, "", max_y=max_y)
        ax.legend(loc="upper left", fontsize=10)

        # Mean annotations per d
        annotations = {
            d: ax.text(
                N * 0.98, -9999, "",
                color=colors[d], fontsize=8, ha="right", va="bottom",
            )
            for d in DEMO_D
        }

        def update(frame):
            lf = lfs[frame]
            artists = [title]
            for d in DEMO_D:
                arr = data.get(size, {}).get(d, {}).get(lf)
                if arr:
                    sarr = sorted(arr)
                    lines[d].set_data(range(len(sarr)), sarr)
                    annotations[d].set_position((len(sarr) - 5, sarr[-1] + max_y * 0.01))
                    annotations[d].set_text(f"mean {np.mean(arr):.1f}")
                else:
                    lines[d].set_data([], [])
                    annotations[d].set_text("")
                artists += [lines[d], annotations[d]]
            title.set_text(f"d comparison — size={size:,}    load factor: {lf:.4f}")
            return artists

        ani = animation.FuncAnimation(
            fig, update, frames=len(lfs),
            interval=FRAME_PAUSE_MS, blit=False, repeat=True,
        )

        out = DEMO_DIR / "animated" / "d_compare" / str(size) / "d_comparison.gif"
        out.parent.mkdir(parents=True, exist_ok=True)
        ani.save(str(out), writer="pillow", fps=ANIMATION_FPS)
        plt.close(fig)
        print(f"  Saved  {out}")


# ── Chart type 5: mean displacement heatmap (d × load_factor) ────────────────

def plot_heatmap(data: dict):
    """One chart showing everything: rows=d, cols=load_factor, color=mean displacements.

    Cells where insertions regularly hit max_displacements are marked with an X —
    that is the 'phase transition' where the algorithm breaks down.
    """
    print("\nGenerating heatmap...")
    max_disp_sentinel = max(DEMO_SIZES) * 2  # value used for failed insertions

    for size in DEMO_SIZES:
        ds  = sorted(DEMO_D)
        lfs = sorted(DEMO_LF)

        grid      = np.full((len(ds), len(lfs)), np.nan)
        fail_mask = np.zeros((len(ds), len(lfs)), dtype=bool)

        for i, d in enumerate(ds):
            for j, lf in enumerate(lfs):
                arr = data.get(size, {}).get(d, {}).get(lf)
                if arr:
                    grid[i, j] = np.mean(arr)
                    # Mark cell as unstable if >5% of insertions hit the displacement limit
                    fail_rate = sum(1 for v in arr if v >= max_disp_sentinel) / len(arr)
                    if fail_rate > 0.05:
                        fail_mask[i, j] = True

        # Cap display at 99th percentile so one extreme cell doesn't wash out the rest
        finite = grid[np.isfinite(grid) & ~fail_mask]
        vmax   = np.percentile(finite, 99) if len(finite) else 1
        vmin   = max(np.nanmin(grid[np.isfinite(grid)]), 0.1)

        fig, ax = plt.subplots(figsize=(max(10, len(lfs) * 1.2), max(5, len(ds) * 0.8)))

        im = ax.imshow(
            np.clip(grid, vmin, vmax),
            aspect="auto",
            cmap="YlOrRd",
            norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
            interpolation="nearest",
        )

        ax.set_xticks(range(len(lfs)))
        ax.set_xticklabels([str(lf) for lf in lfs], rotation=40, ha="right", fontsize=9)
        ax.set_yticks(range(len(ds)))
        ax.set_yticklabels([f"d = {d}" for d in ds], fontsize=9)
        ax.set_xlabel("Load Factor  (how full the table is)", fontsize=11)
        ax.set_ylabel("Hash Functions (d)", fontsize=11)
        ax.set_title(
            f"Mean Displacements per Insertion — size = {size:,}\n"
            "(lower = better;  ✗ = algorithm breaks down at this load)",
            fontsize=12,
        )

        # Annotate each cell with the mean value
        for i in range(len(ds)):
            for j in range(len(lfs)):
                val = grid[i, j]
                if np.isnan(val):
                    continue
                if fail_mask[i, j]:
                    ax.text(j, i, "✗", ha="center", va="center",
                            fontsize=13, fontweight="bold", color="#cc0000")
                else:
                    cell_color = "#ffffff" if val > vmax * 0.4 else "#222222"
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                            fontsize=8, color=cell_color)

        plt.colorbar(im, ax=ax, label="Mean displacements (log scale)", shrink=0.8)
        fig.tight_layout()
        _save(fig, DEMO_DIR / "heatmap" / f"{size}.png")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    data = collect_all()

    plot_compare_d(data)
    plot_load_progression(data)
    animate_per_d(data)
    animate_d_compare(data)
    plot_heatmap(data)

    print(f"\nDone. All output in: {DEMO_DIR}/")
    print("""
Output layout:
  demo/compare_d/          static: all d values on one chart per load_factor
  demo/load_progression/   static: all load_factors on one chart per d
  demo/animated/per_d/     GIF: displacement curve evolves as load_factor rises (one per d)
  demo/animated/d_compare/ GIF: all d curves side-by-side evolving (headline demo)
  demo/heatmap/            static: mean displacements as a d × load_factor grid
  demo/data/               cached simulation JSON (delete to force re-run)
""")


if __name__ == "__main__":
    main()
