"""Regenerate the README's conceptual diagram and explicitly simulated example.

Run from any directory with the project's Python environment. No saved research
results are loaded; the simulated profiles illustrate the factorization only.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


OUT = Path(__file__).resolve().parent
INK = "#182f45"
TEAL = "#087f8c"
ORANGE = "#c56527"


def box(ax, xy: tuple[float, float], width: float, height: float,
        title: str, body: str, color: str) -> None:
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.4, edgecolor=color, facecolor="white",
    ))
    ax.text(x + width / 2, y + height * 0.67, title, ha="center", va="center",
            fontsize=12, fontweight="bold", color=color)
    ax.text(x + width / 2, y + height * 0.30, body, ha="center", va="center",
            fontsize=9.4, color=INK, linespacing=1.55)


def arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#667a8d"})


def overview() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.2))
    fig.set_facecolor("#f4f8fb")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.02, 0.94, "One shared profile. Dataset-specific corrections.",
            fontsize=21, weight="bold", color=INK)
    ax.text(0.02, 0.865, "RiboUnmix learns a reference-defined decomposition of ribosome profiles.",
            fontsize=11, color="#516578")
    box(ax, (0.025, 0.52), 0.23, 0.22, "Coding sequence", "Codon-aligned features\nShared across datasets", INK)
    box(ax, (0.355, 0.52), 0.26, 0.22, "Shared branch", "Sequence-only BiGRU\nPositive, mean-one load L", TEAL)
    box(ax, (0.025, 0.10), 0.23, 0.22, "Sequence + dataset ID", "Codon context and position\nDataset-conditioned embeddings", INK)
    box(ax, (0.355, 0.10), 0.26, 0.22, "Dataset branch", "Independent BiGRU\nCorrection gamma; dispersion alpha", ORANGE)
    box(ax, (0.745, 0.32), 0.23, 0.25, "Observation model", "NB2 mean from L × gamma\nScaled by observed target mean", INK)
    arrow(ax, (0.28, 0.63), (0.33, 0.63))
    arrow(ax, (0.28, 0.21), (0.33, 0.21))
    arrow(ax, (0.64, 0.62), (0.72, 0.52))
    arrow(ax, (0.64, 0.22), (0.72, 0.37))
    ax.text(0.735, 0.16, "Reference centering fixes a convention;\nbiological purity is not guaranteed.",
            fontsize=9.2, color="#516578", linespacing=1.6)
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.035, top=0.995)
    fig.savefig(OUT / "ribounmix_overview.svg", metadata={"Date": None})
    plt.close(fig)


def illustration() -> None:
    rng = np.random.default_rng(42)
    positions = np.arange(120)
    x = positions / 119
    shared = 0.7 + 1.8 * np.exp(-((x - 0.57) / 0.055) ** 2)
    shared += 0.5 * np.exp(-((x - 0.22) / 0.045) ** 2)
    shared /= shared.mean()
    log_gamma = 0.55 * np.cos(2 * np.pi * x)
    log_gamma -= log_gamma.mean()
    gamma = np.exp(np.stack([log_gamma, -log_gamma]))
    mean = gamma * shared
    mean = np.array([10.0, 16.0])[:, None] * mean / mean.mean(axis=1, keepdims=True)
    alpha = 0.12
    observed = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mean))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    for i, color in enumerate([TEAL, ORANGE]):
        axes[0].plot(positions, observed[i], color=color, alpha=0.6, lw=0.85)
        axes[0].plot(positions, mean[i], color=color, lw=2, label=f"Dataset {i + 1}")
        axes[2].plot(positions, gamma[i], color=color, lw=2, label=f"Dataset {i + 1}")
    axes[1].plot(positions, shared, color=TEAL, lw=2)
    for ax, title, ylabel in zip(axes,
            ["Observed profiles + generating means", "Generating shared profile", "Generating dataset corrections"],
            ["Counts", "Mean-one load", "Gamma"]):
        ax.set(title=title, xlabel="Codon position", ylabel=ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.16)
    axes[0].legend(frameon=False, fontsize=8)
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle("Illustrative simulation • not fitted results or experimental evidence",
                 fontsize=12, color=INK)
    fig.savefig(OUT / "ribounmix_profiles.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "svg.fonttype": "none", "svg.hashsalt": "ribounmix-readme-v1"})
    overview()
    illustration()
    print(f"Wrote README figures to {OUT}")
