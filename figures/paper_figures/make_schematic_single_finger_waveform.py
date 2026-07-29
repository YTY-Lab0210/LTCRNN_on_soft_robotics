from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent


def make_step_waveform():
    dt = 0.08
    t = np.arange(0, 4.0 + dt, dt)
    y = np.zeros_like(t)

    closing = t <= 2.0
    opening = t > 2.0
    y[closing] = t[closing] / 2.0
    y[opening] = 1.0 - (t[opening] - 2.0) / 2.0
    y = np.clip(y, 0, 1.0)

    return t, y


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t, y = make_step_waveform()

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=300)
    teal = "#1687a7"

    ax.step(t, y, where="post", color=teal, linewidth=3.0)

    ax.set_xlim(-0.12, 4.25)
    ax.set_ylim(-0.08, 1.15)
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(["0", "1", "2", "3", "4"], fontsize=16)
    ax.set_yticks([])

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_position(("data", 0))
    ax.spines["bottom"].set_position(("data", 0))
    ax.spines["left"].set_linewidth(2.2)
    ax.spines["bottom"].set_linewidth(2.2)
    ax.tick_params(axis="x", width=2.0, length=7)

    # Arrowheads for axes.
    ax.annotate(
        "",
        xy=(4.2, 0),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", lw=2.2, color="black", mutation_scale=18),
    )
    ax.annotate(
        "",
        xy=(0, 1.12),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", lw=2.2, color="black", mutation_scale=18),
    )

    ax.text(4.23, -0.02, "t (s)", ha="left", va="top", fontsize=20)
    ax.text(-0.08, 1.12, r"$\kappa$ (a.u.)", ha="right", va="bottom", fontsize=20)

    ax.set_frame_on(False)
    fig.tight_layout(pad=0.4)

    out_png = OUT_DIR / "schematic_single_finger_waveform.png"
    out_transparent = OUT_DIR / "schematic_single_finger_waveform_transparent.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_transparent, dpi=300, bbox_inches="tight", transparent=True)
    print(out_png)


if __name__ == "__main__":
    main()
