#!/usr/bin/env python3
"""Create the per-dataset observation-fit figure used in the ICLR appendix."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utils.publication_plot_style import publication_rc


REQUIRED_COLUMNS = {
    "arm",
    "N",
    "dataset_id",
    "status",
    "matched",
    "mu_pcc",
    "L_bio_pcc",
    "correction_gain",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "analyses/artifacts/real_data/cumulative_stability/"
            "observed_fit_per_dataset.csv"
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(
            "figures/assets_5_real_datasets_4_panels/"
            "appendix_observation_fit_per_dataset"
        ),
    )
    return parser.parse_args()


def stable_jitter(dataset_id: str, n_datasets: int, width: float = 0.22) -> float:
    key = f"{dataset_id}|{n_datasets}".encode("utf-8")
    unit = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / (2**64 - 1)
    return width * (2.0 * unit - 1.0)


def add_dataset_distribution(
    ax: plt.Axes,
    frame: pd.DataFrame,
    values: str,
    ns: list[int],
    color: str,
) -> None:
    groups = [frame.loc[frame["N"].eq(n), values].to_numpy() for n in ns]
    positions = np.arange(len(ns), dtype=float)
    ax.boxplot(
        groups,
        positions=positions,
        widths=0.48,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.1},
        boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.22},
        whiskerprops={"color": color, "linewidth": 0.9},
        capprops={"color": color, "linewidth": 0.9},
    )
    for position, n in zip(positions, ns, strict=True):
        part = frame.loc[frame["N"].eq(n)].sort_values("dataset_id")
        x = np.array(
            [position + stable_jitter(dataset_id, n) for dataset_id in part.dataset_id]
        )
        ax.scatter(x, part[values], s=10, color=color, alpha=0.72, linewidths=0)
    ax.set_xticks(positions, [str(n) for n in ns])
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.55, alpha=0.75)


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.input)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    valid = data.loc[
        data["matched"].astype(bool)
        & data["status"].eq("validated_predictions")
        & data[["mu_pcc", "L_bio_pcc", "correction_gain"]].notna().all(axis=1)
    ].copy()
    duplicate = valid.duplicated(["arm", "N", "dataset_id"])
    if duplicate.any():
        raise ValueError("Duplicate arm/N/dataset rows in observation-fit input")

    equal = valid.loc[valid["arm"].eq("equal")].copy()
    ns = sorted(equal["N"].astype(int).unique().tolist())
    if ns != [2, 5, 10, 20, 40, 80]:
        raise ValueError(f"Unexpected completed dataset sizes: {ns}")
    observed_counts = equal.groupby("N")["dataset_id"].nunique().to_dict()
    if any(observed_counts[n] != n for n in ns):
        raise ValueError(f"Incomplete per-dataset metrics: {observed_counts}")

    style = publication_rc()
    style.update(
        {
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    plt.rcParams.update(style)
    fig, axes = plt.subplots(1, 3, figsize=(8.3, 2.65), constrained_layout=True)
    specifications = [
        ("mu_pcc", "#286b9b", r"PCC($\mu$, observed)", "A  Fitted mean"),
        ("L_bio_pcc", "#b36a1e", r"PCC($L_{\mathrm{bio}}$, observed)", "B  Shared profile"),
        ("correction_gain", "#178153", r"PCC($\mu$, observed) $-$ PCC($L_{\mathrm{bio}}$, observed)", "C  Correction gain"),
    ]
    for ax, (column, color, ylabel, title) in zip(axes, specifications, strict=True):
        add_dataset_distribution(ax, equal, column, ns, color)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Number of training datasets, $N$")
        ax.set_ylabel(ylabel)
    axes[0].set_ylim(0.0, 0.60)
    axes[1].set_ylim(0.0, 0.60)
    axes[2].set_ylim(0.0, 0.40)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    source_columns = [
        "N",
        "dataset_id",
        "mu_pcc",
        "L_bio_pcc",
        "correction_gain",
    ]
    equal[source_columns].sort_values(["N", "dataset_id"]).to_csv(
        args.output_prefix.with_name(args.output_prefix.name + "_source.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
