from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)


def box(ax, xy, wh, text, fc="#f8fafc", ec="#334155", fs=10, lw=1.4):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
    return patch


def arrow(ax, p1, p2, color="#475569"):
    arr = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.4,
        color=color,
    )
    ax.add_patch(arr)


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def overview():
    fig, ax = plt.subplots(figsize=(11.2, 3.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.94, "DePose-Fi: decomposition-first Wi-Fi HPE", fontsize=13, weight="bold")

    box(ax, (0.03, 0.34), (0.14, 0.34), "Raw CSI frame\n3 x 114 x 10\nmixed multipath", "#e0f2fe", "#0369a1")
    box(ax, (0.22, 0.34), (0.16, 0.34), "Non-negative\nCP decomposition\nrank R=4", "#fef9c3", "#a16207")
    box(ax, (0.43, 0.56), (0.14, 0.18), "Link factor A\nspatial visibility", "#dcfce7", "#15803d", fs=9)
    box(ax, (0.43, 0.34), (0.14, 0.18), "Subcarrier factor B\nfrequency signature", "#dcfce7", "#15803d", fs=9)
    box(ax, (0.43, 0.12), (0.14, 0.18), "Packet factor C\nshort-time activation", "#dcfce7", "#15803d", fs=9)
    box(ax, (0.63, 0.34), (0.15, 0.34), "Compact feature\n508 values\n6.73x smaller", "#fae8ff", "#86198f")
    box(ax, (0.83, 0.34), (0.14, 0.34), "Tiny ML regressor\nRidge / MLP / Trees\n17 keypoints", "#fee2e2", "#b91c1c")

    arrow(ax, (0.17, 0.51), (0.22, 0.51))
    arrow(ax, (0.38, 0.51), (0.43, 0.65))
    arrow(ax, (0.38, 0.51), (0.43, 0.43))
    arrow(ax, (0.38, 0.51), (0.43, 0.21))
    arrow(ax, (0.57, 0.65), (0.63, 0.51))
    arrow(ax, (0.57, 0.43), (0.63, 0.51))
    arrow(ax, (0.57, 0.21), (0.63, 0.51))
    arrow(ax, (0.78, 0.51), (0.83, 0.51))

    ax.text(
        0.515,
        0.88,
        "Main claim: expose pose-relevant wireless components before learning",
        ha="center",
        fontsize=10,
        color="#334155",
    )
    save(fig, "fig1_overview")


def decomposition():
    fig, ax = plt.subplots(figsize=(8.5, 4.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.94, "Rank-one CSI components", fontsize=13, weight="bold")

    # tensor block
    base_x, base_y = 0.06, 0.28
    for i, color in enumerate(["#dbeafe", "#bfdbfe", "#93c5fd"]):
        ax.add_patch(Rectangle((base_x + 0.025 * i, base_y + 0.035 * i), 0.22, 0.28, fc=color, ec="#1d4ed8", lw=1))
    ax.text(base_x + 0.13, base_y + 0.18, "CSI tensor X\nlinks x subcarriers x packets", ha="center", va="center", fontsize=10)
    ax.text(0.35, 0.44, r"$\approx \sum_{r=1}^{R}$", fontsize=18, ha="center")

    colors = ["#dcfce7", "#fef9c3", "#fee2e2", "#fae8ff"]
    xs = [0.48, 0.61, 0.74, 0.87]
    for r, x in enumerate(xs):
        box(ax, (x, 0.68), (0.09, 0.13), f"A$_{r}$\nlink", colors[r], "#334155", fs=9)
        box(ax, (x, 0.44), (0.09, 0.13), f"B$_{r}$\nsubcarrier", colors[r], "#334155", fs=9)
        box(ax, (x, 0.20), (0.09, 0.13), f"C$_{r}$\npacket", colors[r], "#334155", fs=9)
        ax.plot([x + 0.045, x + 0.045], [0.33, 0.44], color="#64748b", lw=1)
        ax.plot([x + 0.045, x + 0.045], [0.57, 0.68], color="#64748b", lw=1)
        ax.text(x + 0.045, 0.08, f"component {r}", ha="center", fontsize=8)

    ax.text(0.52, 0.91, "Each component = link visibility x frequency signature x short-time activation", fontsize=10)
    save(fig, "fig2_decomposition")


def ablation():
    labels = ["drop C0", "drop C1", "drop C2", "drop C3"]
    drops = [13.656, 7.553, 9.047, 18.937]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bars = ax.bar(labels, drops, color=["#60a5fa", "#34d399", "#fbbf24", "#f87171"], edgecolor="#334155")
    ax.set_ylabel("PCK$_{20}$ drop")
    ax.set_title("Pose relevance of decomposed components")
    ax.set_ylim(0, 21.0)
    ax.grid(axis="y", alpha=0.25)
    for b, v in zip(bars, drops):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"-{v:.2f}", ha="center", fontsize=9)
    save(fig, "fig3_component_ablation")


def factor_importance():
    labels = ["Link A", "Subcarrier B", "Packet C"]
    drops = [2.682, 24.902, 0.263]
    importance = [0.045, 0.928, 0.027]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), constrained_layout=True)
    axes[0].bar(labels, drops, color="#93c5fd", edgecolor="#334155")
    axes[0].set_ylabel("PCK$_{20}$ drop")
    axes[0].set_title("Factor-mode ablation")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, importance, color="#86efac", edgecolor="#334155")
    axes[1].set_ylabel("Feature importance")
    axes[1].set_title("ExtraTrees importance")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    save(fig, "fig4_factor_importance")


def full_results_complexity():
    methods = ["Raw\nMLP64", "CP\nMLP64", "CP\nAFF", "CP\nCNN", "CP\nTrees", "CP\nS-AFF", "HPE-Li"]
    pck20 = [44.691, 46.235, 48.640, 49.620, 50.401, 50.802, 52.070]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.25), constrained_layout=True)
    x = range(len(methods))
    axes[0].bar(
        x,
        pck20,
        color=["#cbd5e1", "#60a5fa", "#c4b5fd", "#38bdf8", "#34d399", "#8b5cf6", "#f59e0b"],
        edgecolor="#334155",
    )
    axes[0].set_xticks(list(x), methods)
    axes[0].set_ylabel("PCK$_{20}$ (%)")
    axes[0].set_title("Full MM-Fi protocol-3 accuracy")
    axes[0].set_ylim(40, 54)
    axes[0].grid(axis="y", alpha=0.25)
    for i, v in enumerate(pck20):
        axes[0].text(i, v + 0.35, f"{v:.1f}", ha="center", fontsize=8)

    axes[1].scatter([2.887], [50.802], s=120, color="#8b5cf6", edgecolor="#334155", label="CP+S-AFF")
    axes[1].scatter([2420], [52.070], s=120, color="#f59e0b", edgecolor="#334155", label="HPE-Li")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("FLOPs per frame (M, log scale)")
    axes[1].set_ylabel("PCK$_{20}$ (%)")
    axes[1].set_title("Accuracy-compute trade-off")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].annotate("~838x fewer FLOPs", xy=(2.887, 50.802), xytext=(12, 51.0),
                     arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#475569"},
                     fontsize=8)
    save(fig, "fig6_full_results_complexity")


if __name__ == "__main__":
    overview()
    decomposition()
    ablation()
    factor_importance()
    full_results_complexity()
    print(f"saved figures to {OUT}")
