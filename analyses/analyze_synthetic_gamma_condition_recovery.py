#!/usr/bin/env python3
"""Condition-resolved gamma recovery for the complete synthetic bias panel.

The input table was produced from saved prediction arrays by
``analyze_synthetic_gamma_depth_recovery.py``.  This script makes no model
predictions and does not reconstruct profiles from summary statistics.  It
uses N=10, the first cumulative panel containing every programmed bias, and
keeps the same 27 validation transcripts across depths and conditions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Utils.publication_plot_style import publication_rc
from analyses.plot_synthetic_read_depth_effect import BIAS_ORDER, DEPTHS, cohort_hash, file_hash


N_DATASETS = 10
BOOTSTRAP_REPEATS = 5_000
BOOTSTRAP_SEED = 42
METRICS = ("gamma_pcc", "gamma_rmse", "amplitude_slope")
DEPTH_LABELS = {
    "0p25_per_codon": "0.25",
    "2_per_codon": "2",
    "20_per_codon": "20",
}
BIAS_LABELS = {
    "3prime_aa": r"$3^\prime$-AA",
    "3prime_cc": r"$3^\prime$-CC",
    "3prime_gg": r"$3^\prime$-GG",
    "3prime_uu": r"$3^\prime$-UU",
    "5prime_aa": r"$5^\prime$-AA",
    "5prime_cc": r"$5^\prime$-CC",
    "5prime_gg": r"$5^\prime$-GG",
    "5prime_uu": r"$5^\prime$-UU",
    "gc_fraction_gt_0p7": "GC-rich",
    "au_fraction_gt_0p7": "AU-rich",
}
STEM = "synthetic_gamma_condition_recovery"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_condition_records(path: Path) -> tuple[pd.DataFrame, list[str]]:
    required = {
        "run",
        "depth",
        "n_datasets",
        "transcript_id",
        "dataset",
        "gamma_pcc",
        "gamma_rmse",
        "pcc_valid",
        "predicted_gamma_variance",
        "reference_gamma_variance",
    }
    frame = pd.read_csv(path)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Gamma detail table lacks {sorted(missing)}: {path}")
    frame = frame.loc[frame["n_datasets"] == N_DATASETS].copy()
    frame["bias_condition"] = frame["dataset"].str.removeprefix("artificial_bias_")
    expected_conditions = set(BIAS_ORDER)
    actual_conditions = set(frame["bias_condition"])
    if actual_conditions != expected_conditions:
        raise ValueError(
            f"N=10 bias conditions differ: missing={sorted(expected_conditions-actual_conditions)}, "
            f"unexpected={sorted(actual_conditions-expected_conditions)}"
        )
    if set(frame["depth"]) != set(DEPTHS):
        raise ValueError("N=10 gamma table does not contain the three declared depths.")
    keys = ["depth", "bias_condition", "transcript_id"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate depth/condition/transcript gamma records.")
    if not frame["pcc_valid"].astype(bool).all():
        raise ValueError("Undefined gamma PCC occurs in the primary N=10 cohort.")
    numeric = frame[
        ["gamma_pcc", "gamma_rmse", "predicted_gamma_variance", "reference_gamma_variance"]
    ].to_numpy(float)
    if not np.isfinite(numeric).all() or np.any(frame["reference_gamma_variance"] <= 0):
        raise ValueError("Non-finite gamma metrics or non-positive reference variance.")
    frame["amplitude_slope"] = frame["gamma_pcc"] * np.sqrt(
        frame["predicted_gamma_variance"] / frame["reference_gamma_variance"]
    )

    identities = []
    for depth in DEPTHS:
        for condition in BIAS_ORDER:
            ids = sorted(
                frame.loc[
                    (frame["depth"] == depth) & (frame["bias_condition"] == condition),
                    "transcript_id",
                ].astype(str)
            )
            identities.append(ids)
    if not identities or any(ids != identities[0] for ids in identities[1:]):
        raise ValueError("Gamma condition cohorts are not identical across depths and conditions.")
    return frame, identities[0]


def summarize(frame: pd.DataFrame, ids: list[str], repeats: int, seed: int) -> pd.DataFrame:
    """Use one transcript bootstrap draw matrix for every displayed cell."""
    draws = np.random.default_rng(seed).integers(len(ids), size=(repeats, len(ids)))
    records: list[dict] = []
    for depth in DEPTHS:
        for condition in BIAS_ORDER:
            cell = frame.loc[
                (frame["depth"] == depth) & (frame["bias_condition"] == condition)
            ].set_index("transcript_id").reindex(ids)
            if cell[list(METRICS)].isna().any().any():
                raise ValueError(f"Incomplete gamma cell: {depth}/{condition}")
            for metric in METRICS:
                values = cell[metric].to_numpy(float)
                bootstrap = values[draws].mean(axis=1)
                records.append(
                    {
                        "depth": depth,
                        "nominal_reads_per_codon": DEPTH_LABELS[depth],
                        "bias_condition": condition,
                        "bias_label": BIAS_LABELS[condition],
                        "metric": metric,
                        "n_transcripts": len(ids),
                        "cohort_hash": cohort_hash(ids),
                        "mean": float(values.mean()),
                        "ci_low": float(np.quantile(bootstrap, 0.025)),
                        "ci_high": float(np.quantile(bootstrap, 0.975)),
                    }
                )
    return pd.DataFrame(records)


def _matrix(summary: pd.DataFrame, metric: str) -> np.ndarray:
    return (
        summary.loc[summary["metric"] == metric]
        .pivot(index="bias_condition", columns="depth", values="mean")
        .reindex(index=BIAS_ORDER, columns=DEPTHS)
        .to_numpy(float)
    )


def plot(summary: pd.DataFrame, output: Path, width: float) -> dict:
    style = publication_rc()
    style.update(
        {
            "font.size": 11.5,
            "font.weight": "bold",
            "axes.labelsize": 12.0,
            "axes.labelweight": "bold",
            "axes.titlesize": 13.0,
            "axes.titleweight": "bold",
            "xtick.labelsize": 11.0,
            "ytick.labelsize": 11.0,
            "axes.linewidth": 1.2,
        }
    )
    if style["text.usetex"]:
        style["text.latex.preamble"] += (
            r"\renewcommand{\seriesdefault}{\bfdefault}\AtBeginDocument{\boldmath}"
        )
    matrices = {metric: _matrix(summary, metric) for metric in METRICS}
    slope = matrices["amplitude_slope"]
    slope_span = max(float(abs(slope - 1).max()), 0.02)
    settings = {
        "gamma_pcc": dict(cmap="viridis", vmin=float(matrices["gamma_pcc"].min()), vmax=1.0),
        "gamma_rmse": dict(cmap="magma_r", vmin=0.0, vmax=float(matrices["gamma_rmse"].max())),
        "amplitude_slope": dict(
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=1 - slope_span, vcenter=1.0, vmax=1 + slope_span),
        ),
    }
    titles = (
        r"A  Multiplier PCC",
        r"B  Multiplier RMSE",
        r"C  Amplitude slope",
    )
    with matplotlib.rc_context(style):
        figure, axes = plt.subplots(1, 3, figsize=(width, width / 2.05))
        figure.subplots_adjust(left=0.17, right=0.985, bottom=0.15, top=0.88, wspace=0.48)
        for axis, metric, title in zip(axes, METRICS, titles):
            values = matrices[metric]
            image = axis.imshow(values, aspect="auto", **settings[metric])
            axis.set_title(title, loc="left", pad=8)
            axis.set_xticks(range(len(DEPTHS)), [DEPTH_LABELS[depth] for depth in DEPTHS])
            axis.set_xlabel("Reads per codon")
            axis.set_yticks(range(len(BIAS_ORDER)))
            if axis is axes[0]:
                axis.set_yticklabels([BIAS_LABELS[condition] for condition in BIAS_ORDER])
                axis.set_ylabel("Injected bias condition")
            else:
                axis.set_yticklabels([])
            for row in range(values.shape[0]):
                for column in range(values.shape[1]):
                    value = values[row, column]
                    label = f"{value:.3f}" if metric != "gamma_pcc" else f"{value:.4f}"
                    rgba = image.cmap(image.norm(value))
                    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    axis.text(
                        column,
                        row,
                        label,
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        fontweight="bold",
                        color="black" if luminance > 0.58 else "white",
                    )
            colorbar = figure.colorbar(image, ax=axis, fraction=0.055, pad=0.035)
            colorbar.ax.tick_params(labelsize=9.5, width=1.0)
        for suffix, dpi in (("pdf", None), ("svg", None), ("png", 600)):
            figure.savefig(output / f"{STEM}.{suffix}", dpi=dpi, bbox_inches="tight")
        plt.close(figure)
    return {
        "text.usetex": style["text.usetex"],
        "font.family": style["font.family"],
        "font.size": style["font.size"],
        "figure_width_inches": width,
    }


def write_text(output: Path, summary: pd.DataFrame, ids: list[str], repeats: int, seed: int) -> None:
    caption = rf"""\textbf{{Correction recovery depends on the programmed bias condition.}}
All ten conditions are evaluated in the complete $N=10$ cumulative panel at nominal
depths 0.25, 2, and 20 reads per codon. \textbf{{(A)}} PCC and \textbf{{(B)}} RMSE
between the learned correction $\widetilde\gamma_{{dt}}$ and the injected multiplier
$b^{{\mathrm{{ref}}}}_{{dt}}$ after applying the same two-way log-centering operator and
ten-codon boundary crop. \textbf{{(C)}} OLS amplitude slope with an intercept,
$\operatorname{{corr}}(\widetilde\gamma,b^{{\mathrm{{ref}}}})
\operatorname{{sd}}(\widetilde\gamma)/\operatorname{{sd}}(b^{{\mathrm{{ref}}}})$;
one indicates matched multiplier amplitude. Cells are equal-transcript means over the same
{len(ids)} validation transcripts. Exact 95\% intervals from {repeats:,} paired
transcript-bootstrap draws (seed {seed}) are supplied in the source table, using one draw
matrix across all conditions, depths, and metrics. The scored targets are gauge-matched
multipliers on the CDS interior, not raw generator coefficients at unavailable UTR context.
High PCC can coexist with RMSE or slope error and therefore does not establish exact
correction recovery.
"""
    (output / "caption.tex").write_text(caption)

    lines = [
        "# Condition-resolved gamma recovery",
        "",
        f"All cells use N=10 and the same {len(ids)} validation transcripts.",
        "The complete panel is used so all ten bias mechanisms are compared without changing N.",
        "",
        "| Depth | PCC range | RMSE range | Amplitude-slope range | Largest-RMSE condition |",
        "|---|---:|---:|---:|---|",
    ]
    for depth in DEPTHS:
        cell = summary.loc[summary["depth"] == depth]
        pcc = cell.loc[cell["metric"] == "gamma_pcc"]
        rmse = cell.loc[cell["metric"] == "gamma_rmse"]
        slope = cell.loc[cell["metric"] == "amplitude_slope"]
        worst = rmse.loc[rmse["mean"].idxmax()]
        lines.append(
            f"| {DEPTH_LABELS[depth]} | {pcc['mean'].min():.4f}–{pcc['mean'].max():.4f} "
            f"| {rmse['mean'].min():.4f}–{rmse['mean'].max():.4f} "
            f"| {slope['mean'].min():.4f}–{slope['mean'].max():.4f} "
            f"| {BIAS_LABELS[worst.bias_condition]} ({worst['mean']:.4f}) |"
        )
    lines.extend(
        [
            "",
            "PCC is almost saturated for every condition, but RMSE and amplitude slope expose condition-specific errors.",
            "The 3-prime-CC condition has the largest mean RMSE at 0.25 and 2 reads/codon; the ordering changes at depth 20.",
            "This is one training seed and a 27-transcript cross-depth intersection; intervals are conditional on the fitted models.",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = ROOT / "analyses/artifacts/synthetic/read_depth/depth_recovery_overview/gamma_recovery"
    parser.add_argument("--input", type=Path, default=default_root / "gamma_per_transcript_dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=default_root / "condition_recovery")
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--figure-width", type=float, default=10.2)
    args = parser.parse_args()
    if args.bootstrap_repeats < 100:
        parser.error("Use at least 100 bootstrap draws.")
    if args.figure_width <= 0:
        parser.error("--figure-width must be positive.")

    source = args.input.resolve()
    output = args.output_dir.resolve()
    frame, ids = load_condition_records(source)
    summary = summarize(frame, ids, args.bootstrap_repeats, args.bootstrap_seed)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "gamma_condition_per_transcript.csv", index=False)
    summary.to_csv(output / "gamma_condition_summary.csv", index=False)
    typography = plot(summary, output, args.figure_width)
    write_text(output, summary, ids, args.bootstrap_repeats, args.bootstrap_seed)
    command = shlex.join([sys.executable, str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]])
    provenance = {
        "command": command,
        "input": str(source),
        "input_sha256": sha256(source),
        "source_analysis": "analyses/analyze_synthetic_gamma_depth_recovery.py",
        "n_datasets": N_DATASETS,
        "depths": list(DEPTHS),
        "bias_conditions": list(BIAS_ORDER),
        "cohort": {"n": len(ids), "sha256": cohort_hash(ids)},
        "gamma_target": "same two-way interior log-centering for prediction and injected multiplier",
        "amplitude_slope": "PCC * sqrt(predicted variance / reference variance); OLS slope with intercept",
        "bootstrap": {
            "repeats": args.bootstrap_repeats,
            "seed": args.bootstrap_seed,
            "unit": "transcript",
            "pairing": "one resample matrix across all depths, conditions, and metrics",
            "interval": "pointwise percentile 95%",
        },
        "training_seed": 42,
        "fitting_or_inference": False,
        "typography": typography,
        "analysis_source_sha256": {
            str(Path(__file__).relative_to(ROOT)): file_hash(Path(__file__)),
            "analyses/analyze_synthetic_gamma_depth_recovery.py": file_hash(
                ROOT / "analyses/analyze_synthetic_gamma_depth_recovery.py"
            ),
        },
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(summary.pivot_table(index=["depth", "bias_condition"], columns="metric", values="mean"))
    print(output / f"{STEM}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
