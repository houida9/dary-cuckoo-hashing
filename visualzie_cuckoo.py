"""
visualize_cuckoo.py — Animated walkthrough of a single d=2 cuckoo hash insertion.

Shows the "cuckoo" eviction chain step by step: a new key displaces a resident,
the resident displaces another, until an empty slot is found.

Output: demo/cuckoo_insertion.gif

No simulation needed — this uses a fixed, hand-designed example so the chain
is always clear and reproducible. Run it standalone:

    python visualize_cuckoo.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

# ── Color palette ─────────────────────────────────────────────────────────────
C_EMPTY     = "#f0f0f0"   # empty slot
C_OCCUPIED  = "#7eb8d4"   # normal occupied slot
C_CANDIDATE = "#f7c948"   # highlighted candidate position
C_EVICTED   = "#e06c75"   # key being kicked out this step
C_PLACED    = "#61af71"   # key just landed here
C_SETTLED   = "#b8d4b8"   # key placed in an earlier step (settled)
C_BORDER    = "#444444"
C_DIM       = "#999999"
# ─────────────────────────────────────────────────────────────────────────────

N_SLOTS  = 8        # slots per table
FPS      = 0.6      # frames per second (slow so viewers can read each step)
HOLD_MS  = int(1000 / FPS)

# ── Scenario ──────────────────────────────────────────────────────────────────
#
# We insert "nit" into a table that is already ~87% full.
# This triggers a 2-displacement chain:
#
#   nit  → T1[3]  (evicts fox)
#   fox  → T2[1]  (evicts pot)
#   pot  → T1[6]  (empty — chain ends)
#
# Hash function assignments (hard-coded for the demo):
#   h₁(nit) = 3,  h₂(nit) = 4
#   h₁(fox) = 3,  h₂(fox) = 1      ← T1[3] excluded (we came from there)
#   h₁(pot) = 6,  h₂(pot) = 1      ← T2[1] excluded (we came from there)
# ─────────────────────────────────────────────────────────────────────────────

_T1_init  = ["ant", "bee",  None,  "fox",  "dog", "eel",  None,  "gnu"]
_T2_init  = ["hog", "pot",  "ivy", "jay",  "koi", "lox",  "mud", "mew"]

_T1_s3    = ["ant", "bee",  None,  "nit",  "dog", "eel",  None,  "gnu"]  # fox evicted
_T2_s5    = ["hog", "fox",  "ivy", "jay",  "koi", "lox",  "mud", "mew"]  # pot evicted
_T1_done  = ["ant", "bee",  None,  "nit",  "dog", "eel",  "pot", "gnu"]  # pot placed

# Each scene: (t1, t2, hl_t1, hl_t2, explanation_lines, status_line)
# hl_t1/hl_t2 = {slot_index: color}
SCENES = [
    (
        _T1_init, _T2_init,
        {}, {},
        [
            "Initial state — table is ~87% full.",
            "",
            "We want to insert the key  \"nit\".",
            "Every key has exactly 2 candidate slots,",
            "one in each table, determined by hash functions h₁ and h₂.",
        ],
        "Goal: insert  \"nit\"",
    ),
    (
        _T1_init, _T2_init,
        {3: C_CANDIDATE}, {4: C_CANDIDATE},
        [
            "Find candidate slots for  \"nit\":",
            "",
            "  h₁(nit)  →  Table 1, slot 3   [occupied by \"fox\"]",
            "  h₂(nit)  →  Table 2, slot 4   [occupied by \"koi\"]",
            "",
            "Both slots are taken — we must evict one.",
        ],
        "Inserting: \"nit\"   |   both candidates occupied",
    ),
    (
        _T1_init, _T2_init,
        {3: C_EVICTED}, {},
        [
            "Randomly choose to evict from Table 1, slot 3.",
            "",
            "\"fox\" is displaced  (like a cuckoo chick",
            "kicking an egg out of the nest).",
            "",
            "\"nit\" will take that vacant slot.",
        ],
        "Inserting: \"nit\"   →   evicting \"fox\" from T1[3]",
    ),
    (
        _T1_s3, _T2_init,
        {3: C_PLACED}, {1: C_CANDIDATE},
        [
            "\"nit\" placed at Table 1, slot 3.  ✓   (displacement #1)",
            "",
            "Now \"fox\" needs a new home.",
            "  h₂(fox)  →  Table 2, slot 1   [occupied by \"pot\"]",
            "  (Table 1 slot 3 is excluded — we just came from there)",
            "",
            "Slot 1 is occupied. Evict again.",
        ],
        "Placed: \"nit\" → T1[3]   |   Now inserting: \"fox\"",
    ),
    (
        _T1_s3, _T2_init,
        {3: C_SETTLED}, {1: C_EVICTED},
        [
            "Evict \"pot\" from Table 2, slot 1.",
            "",
            "\"pot\" is displaced.",
            "\"fox\" will take Table 2, slot 1.",
        ],
        "Placed: \"nit\" → T1[3]   |   Inserting: \"fox\"  →  evicting \"pot\" from T2[1]",
    ),
    (
        _T1_s3, _T2_s5,
        {3: C_SETTLED, 6: C_CANDIDATE}, {1: C_PLACED},
        [
            "\"fox\" placed at Table 2, slot 1.  ✓   (displacement #2)",
            "",
            "Now \"pot\" needs a new home.",
            "  h₁(pot)  →  Table 1, slot 6   [empty!]",
            "  (Table 2 slot 1 is excluded — we just came from there)",
            "",
            "Empty slot found — chain ends here.",
        ],
        "Placed: \"nit\" → T1[3], \"fox\" → T2[1]   |   Now inserting: \"pot\"",
    ),
    (
        _T1_done, _T2_s5,
        {3: C_SETTLED, 6: C_PLACED}, {1: C_SETTLED},
        [
            "\"pot\" placed at Table 1, slot 6.  ✓",
            "",
            "Insertion complete!   2 displacements total.",
            "",
            "  \"nit\"  →  T1[3]   (evicted fox)",
            "  \"fox\"  →  T2[1]   (evicted pot)",
            "  \"pot\"  →  T1[6]   (empty slot — chain ends)",
        ],
        "Done!   \"nit\" inserted in 2 displacements",
    ),
]


# ── Drawing helpers ───────────────────────────────────────────────────────────

SLOT_W   = 1.0
SLOT_H   = 0.65
SLOT_GAP = 0.2
T1_Y     = 4.1
T2_Y     = 2.6


def _sx(i):
    return i * (SLOT_W + SLOT_GAP)


def _draw_table(ax, table, highlights, y, label):
    ax.text(-0.6, y + SLOT_H / 2, label,
            ha="right", va="center", fontsize=10, fontweight="bold", color="#333333")

    for i, key in enumerate(table):
        x     = _sx(i)
        color = highlights.get(i, C_OCCUPIED if key else C_EMPTY)

        rect = mpatches.FancyBboxPatch(
            (x, y), SLOT_W, SLOT_H,
            boxstyle="round,pad=0.06",
            facecolor=color, edgecolor=C_BORDER, linewidth=1.6, zorder=2,
        )
        ax.add_patch(rect)

        light_bg = color in (C_EMPTY, C_CANDIDATE, C_SETTLED)
        ax.text(
            x + SLOT_W / 2, y + SLOT_H / 2,
            key if key else "—",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="#222222" if light_bg else "#ffffff", zorder=3,
        )
        ax.text(_sx(i) + SLOT_W / 2, y - 0.22, str(i),
                ha="center", va="top", fontsize=8, color=C_DIM)


def _draw_legend(ax):
    items = [
        mpatches.Patch(facecolor=C_OCCUPIED,  edgecolor=C_BORDER, label="occupied"),
        mpatches.Patch(facecolor=C_EMPTY,     edgecolor=C_BORDER, label="empty"),
        mpatches.Patch(facecolor=C_CANDIDATE, edgecolor=C_BORDER, label="candidate slot"),
        mpatches.Patch(facecolor=C_EVICTED,   edgecolor=C_BORDER, label="being evicted"),
        mpatches.Patch(facecolor=C_PLACED,    edgecolor=C_BORDER, label="just placed"),
        mpatches.Patch(facecolor=C_SETTLED,   edgecolor=C_BORDER, label="already placed"),
    ]
    ax.legend(handles=items, loc="upper right", fontsize=8,
              framealpha=0.92, bbox_to_anchor=(1.01, 1.0))


# ── Animation ─────────────────────────────────────────────────────────────────

def _render(ax_top, ax_bot, idx):
    t1, t2, hl1, hl2, lines, status = SCENES[idx]

    ax_top.cla()
    _draw_table(ax_top, t1, hl1, T1_Y, "Table 1\n(h₁)")
    _draw_table(ax_top, t2, hl2, T2_Y, "Table 2\n(h₂)")
    _draw_legend(ax_top)

    w = _sx(N_SLOTS - 1) + SLOT_W
    ax_top.set_xlim(-1.0, w + 0.4)
    ax_top.set_ylim(2.1, 5.4)
    ax_top.axis("off")
    ax_top.set_title(
        f"d = 2 Cuckoo Hashing — Insertion Walkthrough   (step {idx} of {len(SCENES)-1})",
        fontsize=12, fontweight="bold", pad=8,
    )

    ax_bot.cla()
    ax_bot.axis("off")
    explanation = "\n".join(lines)
    ax_bot.text(
        0.5, 0.80, explanation,
        ha="center", va="top", fontsize=10, family="monospace",
        transform=ax_bot.transAxes,
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#fafafa",
                  edgecolor="#cccccc", linewidth=1.4),
        multialignment="left",
    )
    ax_bot.text(
        0.5, 0.05, status,
        ha="center", va="bottom", fontsize=9, color="#555555",
        style="italic", transform=ax_bot.transAxes,
    )


def main():
    out = Path("demo/cuckoo_insertion.gif")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 7.5),
        gridspec_kw={"height_ratios": [3, 2.2]},
    )
    fig.subplots_adjust(hspace=0.04, left=0.08, right=0.96, top=0.93, bottom=0.03)

    def update(frame):
        _render(ax_top, ax_bot, frame)
        return []

    ani = FuncAnimation(
        fig, update,
        frames=len(SCENES),
        interval=HOLD_MS,
        blit=False,
        repeat=True,
    )

    print(f"Saving {out}  ({len(SCENES)} frames @ {HOLD_MS}ms each)...")
    ani.save(str(out), writer="pillow", fps=FPS)
    plt.close(fig)
    print(f"Done.  Open  {out}  to view.")


if __name__ == "__main__":
    main()
