#!/usr/bin/env python3
"""Observed-profile reconstruction across the complete synthetic N-by-depth grid.

This is a descriptive validation analysis of the frozen ``best_pcc`` models.
For each transcript, metrics were first computed separately for every dataset in
the cumulative panel and then averaged equally over datasets.  This script
averages those transcript-level values with equal transcript weights.

The primary curves use the complete 27-transcript intersection shared by all
three depth-specific validation lists and all N=2,...,10 models.  Larger
depth-specific validation cohorts are retained as a sensitivity table; they are
not mixed in the primary cross-depth plot.
"""
from __future__ import annotations

import argparse
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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Utils.publication_plot_style import publication_rc
from analyses.plot_synthetic_read_depth_effect import (
    COUNTS,
    DEPTHS,
    cohort_hash,
    complete_cohort,
    file_hash,
    mean_stats,
    verified_runs,
)


RAW_METRICS = ("mu_target_pcc_trim10", "mu_target_rmse_trim10")
# Backward-compatible name for the two metrics read from the original audit.
METRICS = RAW_METRICS
STANDARDIZED_METRIC = "mu_target_nb_standardized_rmse_trim10"
COMMON_COHORT_METRICS = (*RAW_METRICS, STANDARDIZED_METRIC)
PLOT_METRICS = ("mu_target_pcc_trim10", STANDARDIZED_METRIC)
METRIC_LABELS = {
    "mu_target_pcc_trim10": r"Mean PCC$(\mu_{dt},\overline{Y}_{dt})$",
    "mu_target_rmse_trim10": r"Mean RMSE$(\mu_{dt},\overline{Y}_{dt})$",
    STANDARDIZED_METRIC: "Std. RMS residual",
}
DEPTH_LABELS = {
    "0p25_per_codon": "0.25 reads/codon",
    "2_per_codon": "2 reads/codon",
    "20_per_codon": "20 reads/codon",
}
DEPTH_STYLES = {
    "0p25_per_codon": dict(color="#0072B2", marker="o"),
    "2_per_codon": dict(color="#E69F00", marker="s"),
    "20_per_codon": dict(color="#009E73", marker="^"),
}
STEM = "synthetic_multidataset_mu_reconstruction"


def load_records(path: Path, runs: list[dict]) -> pd.DataFrame:
    """Load only the audited scalar records required by this analysis."""
    columns = [
        "run",
        "depth",
        "n_datasets",
        "variant",
        "transcript_id",
        "sense_length",
        "mu_target_pcc_trim10",
        "mu_target_rmse_trim10",
        "mu_target_valid_datasets",
    ]
    run_names = {run["run"] for run in runs}
    parts: list[pd.DataFrame] = []
    with pd.read_csv(path, usecols=columns, chunksize=10_000) as reader:
        for chunk in reader:
            selected = (
                (chunk["variant"] == "best_pcc")
                & chunk["run"].isin(run_names)
                & chunk["depth"].isin(DEPTHS)
                & chunk["n_datasets"].isin(COUNTS)
            )
            parts.append(chunk.loc[selected].copy())
    if not parts:
        raise ValueError("No matching best-PCC records were found.")
    frame = pd.concat(parts, ignore_index=True)
    keys = ["transcript_id", "depth", "n_datasets"]
    if frame.duplicated(keys).any():
        examples = frame.loc[frame.duplicated(keys, keep=False), keys].head().to_dict("records")
        raise ValueError(f"Duplicate transcript/model records: {examples}")
    expected_cells = {(depth, n) for depth in DEPTHS for n in COUNTS}
    actual_cells = set(zip(frame["depth"], frame["n_datasets"]))
    if actual_cells != expected_cells:
        raise ValueError(
            f"Incomplete N-by-depth grid; missing={sorted(expected_cells-actual_cells)}, "
            f"unexpected={sorted(actual_cells-expected_cells)}"
        )
    # A transcript-level PCC is admissible only if every selected dataset had a
    # defined within-profile correlation.  RMSE is likewise required to be
    # finite; no undefined values are converted to zero.
    incomplete = frame["mu_target_valid_datasets"] != frame["n_datasets"]
    frame.loc[incomplete, "mu_target_pcc_trim10"] = np.nan
    return frame


def attach_standardized_error(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Attach the independent simulator-NB2 variance-standardized residual.

    The source table is calculated from the two saved raw replicas.  Its
    consensus statistic is

        (mu - mean_r Y_r)^2 / [sum_r(mu_r + 0.1 mu_r^2) / R^2].

    It therefore removes the leading depth-dependent count scale without
    using the fitted alpha.  We display its square root so that zero remains
    perfect fit and one is the heuristic sampling-scale reference.
    """
    required = {
        "run",
        "depth",
        "n_datasets",
        "transcript_id",
        "domain",
        "consensus_reference_standardized_error",
    }
    table = pd.read_csv(path, usecols=lambda column: column in required)
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Standardized-error table lacks {sorted(missing)}: {path}")
    table = table.loc[table["domain"] == "interior10"].copy()
    keys = ["run", "depth", "n_datasets", "transcript_id"]
    if table.duplicated(keys).any():
        raise ValueError(f"Duplicate standardized-error records in {path}")
    value = table["consensus_reference_standardized_error"].to_numpy(float)
    if np.any(~np.isfinite(value)) or np.any(value < 0):
        raise ValueError(f"Non-finite or negative standardized errors in {path}")
    table[STANDARDIZED_METRIC] = np.sqrt(value)
    merged = frame.merge(table[keys + [STANDARDIZED_METRIC]], on=keys, how="left", validate="one_to_one")
    return merged


def fixed_cohorts(
    frame: pd.DataFrame,
    validation: dict[str, set[str]],
) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:
    """Return one cross-depth cohort and fixed within-depth sensitivities."""
    common_candidates = set.intersection(*(validation[depth] for depth in DEPTHS))
    primary, primary_excluded = complete_cohort(
        frame, common_candidates, DEPTHS, COMMON_COHORT_METRICS
    )
    if len(primary) < 2:
        raise ValueError("Fewer than two complete transcripts remain in the cross-depth cohort.")
    depth_cohorts: dict[str, list[str]] = {}
    exclusions = {"common_three_depths": primary_excluded}
    for depth in DEPTHS:
        included, excluded = complete_cohort(frame, validation[depth], (depth,), RAW_METRICS)
        if len(included) < 2:
            raise ValueError(f"Fewer than two complete validation transcripts at {depth}.")
        depth_cohorts[depth] = included
        exclusions[f"own_validation:{depth}"] = excluded
    return primary, depth_cohorts, exclusions


def summarize(
    frame: pd.DataFrame,
    primary: list[str],
    depth_cohorts: dict[str, list[str]],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Equal-transcript means with transcript-paired percentile intervals."""
    lookup = frame.set_index(["depth", "n_datasets", "transcript_id"]).sort_index()
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    # One resample is reused across all depths, N values and metrics.
    primary_draws = rng.integers(len(primary), size=(repeats, len(primary)))
    for depth in DEPTHS:
        for n in COUNTS:
            cell = lookup.loc[(depth, n)].reindex(primary)
            for metric in COMMON_COHORT_METRICS:
                records.append(
                    dict(
                        cohort="common_three_depths",
                        depth=depth,
                        n_datasets=n,
                        metric=metric,
                        n_transcripts=len(primary),
                        cohort_hash=cohort_hash(primary),
                        **mean_stats(cell[metric].to_numpy(), primary_draws),
                    )
                )

    # These estimates use all eligible transcripts at each depth but are not a
    # matched cross-depth comparison.  Each cohort remains fixed over N.
    for depth in DEPTHS:
        ids = depth_cohorts[depth]
        draws = rng.integers(len(ids), size=(repeats, len(ids)))
        for n in COUNTS:
            cell = lookup.loc[(depth, n)].reindex(ids)
            for metric in RAW_METRICS:
                records.append(
                    dict(
                        cohort="own_depth_validation",
                        depth=depth,
                        n_datasets=n,
                        metric=metric,
                        n_transcripts=len(ids),
                        cohort_hash=cohort_hash(ids),
                        **mean_stats(cell[metric].to_numpy(), draws),
                    )
                )
    return pd.DataFrame(records)


def endpoint_changes(
    frame: pd.DataFrame,
    primary: list[str],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Descriptive N=10 minus N=2 changes on paired transcript identities."""
    lookup = frame.set_index(["depth", "n_datasets", "transcript_id"]).sort_index()
    draws = np.random.default_rng(seed).integers(len(primary), size=(repeats, len(primary)))
    records = []
    for depth in DEPTHS:
        for metric in COMMON_COHORT_METRICS:
            first = lookup.loc[(depth, COUNTS[0]), metric].reindex(primary).to_numpy()
            last = lookup.loc[(depth, COUNTS[-1]), metric].reindex(primary).to_numpy()
            delta = last - first
            boot = delta[draws].mean(axis=1)
            records.append(
                dict(
                    depth=depth,
                    metric=metric,
                    contrast=f"N={COUNTS[-1]} minus N={COUNTS[0]}",
                    n_transcripts=len(primary),
                    cohort_hash=cohort_hash(primary),
                    estimate=float(delta.mean()),
                    ci_low=float(np.quantile(boot, 0.025)),
                    ci_high=float(np.quantile(boot, 0.975)),
                )
            )
    return pd.DataFrame(records)


def plot(summary: pd.DataFrame, output: Path, width: float) -> dict:
    """Render the two requested reconstruction metrics at manuscript size."""
    primary = summary[summary["cohort"] == "common_three_depths"]
    style = publication_rc()
    style.update(
        {
            "font.size": 12.0,
            "font.weight": "bold",
            "axes.labelsize": 12.0,
            "axes.labelweight": "bold",
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 11.0,
            "axes.titlesize": 13.0,
            "axes.titleweight": "bold",
            "axes.labelpad": 3.0,
            "axes.linewidth": 1.1,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "grid.linewidth": 0.7,
        }
    )
    if style["text.usetex"]:
        style["text.latex.preamble"] += (
            r"\renewcommand{\seriesdefault}{\bfdefault}"
            r"\AtBeginDocument{\boldmath}"
        )
    with matplotlib.rc_context(style):
        figure, axes = plt.subplots(1, 2, figsize=(width, width / 2.45))
        figure.subplots_adjust(left=0.09, right=0.985, bottom=0.20, top=0.79, wspace=0.30)
        for axis, metric, title in zip(
            axes,
            PLOT_METRICS,
            ("A  Observed-profile correlation", "B  Scale-adjusted residual"),
        ):
            for depth in DEPTHS:
                cell = (
                    primary[(primary["depth"] == depth) & (primary["metric"] == metric)]
                    .set_index("n_datasets")
                    .loc[list(COUNTS)]
                )
                style_for_depth = DEPTH_STYLES[depth]
                axis.errorbar(
                    COUNTS,
                    cell["mean"],
                    yerr=[cell["mean"] - cell["ci_low"], cell["ci_high"] - cell["mean"]],
                    color=style_for_depth["color"],
                    marker=style_for_depth["marker"],
                    linewidth=2.4,
                    markersize=6.0,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    capsize=2.5,
                    elinewidth=1.2,
                    zorder=3,
                )
            axis.set_title(title, loc="left", pad=7)
            axis.set_xlim(1.65, 10.35)
            axis.set_xticks(COUNTS)
            axis.set_xlabel("Number of training datasets")
            axis.set_ylabel(METRIC_LABELS[metric])
            axis.grid(axis="y")
            axis.set_axisbelow(True)
        axes[0].set_ylim(0.45, 0.96)
        standardized = primary[primary["metric"] == STANDARDIZED_METRIC]
        axes[1].set_ylim(
            max(0.0, float(standardized["ci_low"].min()) - 0.06),
            float(standardized["ci_high"].max()) + 0.06,
        )
        axes[1].axhline(1.0, color="#666666", linestyle=":", linewidth=1.5, zorder=1)
        handles = [
            Line2D(
                [],
                [],
                color=DEPTH_STYLES[depth]["color"],
                marker=DEPTH_STYLES[depth]["marker"],
                linewidth=2.4,
                markersize=6.0,
                label=DEPTH_LABELS[depth],
            )
            for depth in DEPTHS
        ]
        figure.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.54, 0.995),
            ncol=3,
            handlelength=1.7,
            columnspacing=1.25,
            handletextpad=0.45,
        )
        figure.savefig(output / f"{STEM}.pdf")
        figure.savefig(output / f"{STEM}.svg")
        figure.savefig(output / f"{STEM}.png", dpi=600)
        plt.close(figure)
    return {
        "text.usetex": style["text.usetex"],
        "font.family": style["font.family"],
        "font.size": style["font.size"],
        "axes.titlesize": style["axes.titlesize"],
        "figure_width_inches": width,
    }


def write_caption(output: Path, n: int, repeats: int, seed: int) -> None:
    caption = rf"""\textbf{{Observed-profile reconstruction across synthetic panel sizes and read depths.}}
Models use the cumulative bias panels, uniform reference weights $\pi_d=1/N$, training
seed 42, and the checkpoint maximizing validation $\mu$ PCC. For dataset $d$ and transcript
$t$, PCC and RMSE compare the fitted count mean $\boldsymbol\mu_{{dt}}$ with the arithmetic
two-replicate consensus $\overline{{\mathbf Y}}_{{dt}}$ over the sense-CDS interior after
removing the appended terminal entry and the first and last ten codons. Metrics are computed
within each transcript--dataset profile, averaged equally across the $N$ datasets within a
transcript, and then averaged equally across the same {n} transcripts held out at all three
depths and all $N=2,\ldots,10$ models. \textbf{{(A)}} Mean profile PCC. \textbf{{(B)}} Root
mean squared residual after division of each squared consensus residual by its variance under
the simulator NB2 model, $\sum_r(\mu_{{dtri}}+0.1\mu_{{dtri}}^2)/R^2$ for $R=2$ replicas.
The fitted dispersion is not used in this normalization; the dotted line at one is a
heuristic sampling-scale reference, not an exact calibration null. Error bars are pointwise
95\% percentile intervals from {repeats:,} paired
transcript-bootstrap draws (seed {seed}), with each sampled transcript carrying every depth,
panel size, and metric. Intervals condition on the fitted models. The observed consensus is a
noise-affected target, and it was also used by the checkpoint-selection metric; these are
descriptive validation-reconstruction scores, not independent generalization estimates or
latent biological-recovery metrics. Raw count RMSE is retained in the source tables but is
not compared visually across depths. Increasing $N$ changes both the observations and bias
composition, so slopes across $N$ do not isolate a sample-size effect.
"""
    (output / "caption.tex").write_text(caption)


def write_report(
    output: Path,
    summary: pd.DataFrame,
    changes: pd.DataFrame,
    primary: list[str],
    depth_cohorts: dict[str, list[str]],
) -> None:
    primary_summary = summary[summary["cohort"] == "common_three_depths"]
    lines = [
        "# Multi-dataset observed-profile reconstruction",
        "",
        f"Primary cohort: {len(primary)} transcripts shared by all depths and N values.",
        "The checkpoint and evaluation cohort are both validation-based, so the results are descriptive.",
        "",
        "## Actual ranges across N=2,...,10",
        "",
        "| Depth | PCC range | Standardized RMS range | Raw RMSE range | N=10 − N=2 PCC | N=10 − N=2 standardized RMS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for depth in DEPTHS:
        pcc = primary_summary[
            (primary_summary["depth"] == depth)
            & (primary_summary["metric"] == "mu_target_pcc_trim10")
        ]
        rmse = primary_summary[
            (primary_summary["depth"] == depth)
            & (primary_summary["metric"] == "mu_target_rmse_trim10")
        ]
        standardized = primary_summary[
            (primary_summary["depth"] == depth)
            & (primary_summary["metric"] == STANDARDIZED_METRIC)
        ]
        dpcc = changes[
            (changes["depth"] == depth) & (changes["metric"] == "mu_target_pcc_trim10")
        ].iloc[0]
        drmse = changes[
            (changes["depth"] == depth) & (changes["metric"] == "mu_target_rmse_trim10")
        ].iloc[0]
        dstandardized = changes[
            (changes["depth"] == depth) & (changes["metric"] == STANDARDIZED_METRIC)
        ].iloc[0]
        lines.append(
            f"| {DEPTH_LABELS[depth]} | {pcc['mean'].min():.4f}–{pcc['mean'].max():.4f} "
            f"| {standardized['mean'].min():.4f}–{standardized['mean'].max():.4f} "
            f"| {rmse['mean'].min():.4f}–{rmse['mean'].max():.4f} "
            f"| {dpcc.estimate:+.4f} [{dpcc.ci_low:+.4f}, {dpcc.ci_high:+.4f}] "
            f"| {dstandardized.estimate:+.4f} [{dstandardized.ci_low:+.4f}, "
            f"{dstandardized.ci_high:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Higher-depth consensuses have substantially higher PCC with fitted means, but the target itself is less noisy.",
            "- Raw RMSE grows with depth because it is measured in count units; it is retained for audit but not used for the cross-depth visual conclusion.",
            "- The NB2-standardized RMS residual removes the leading mean/variance scale. It is modestly higher at depth 20 even though PCC is highest, so high depth improves shape agreement without making every standardized residual smaller.",
            "- Curves are not monotone in N. Each increment adds a prespecified bias condition, so N and bias composition are inseparable.",
            "- The larger within-depth sensitivity cohorts contain "
            + ", ".join(f"{DEPTH_LABELS[d]}: {len(depth_cohorts[d]):,}" for d in DEPTHS)
            + " transcripts. They are saved in `summary.csv`, but are not mixed in the primary cross-depth plot.",
            "- These metrics evaluate observed-profile reconstruction. They should be discussed separately from recovery of K, H, gamma, or alpha.",
            "",
            "## Files",
            "",
            "- `per_transcript.csv`: exact audited scalar records and cohort membership.",
            "- `summary.csv`: primary and depth-specific estimates and bootstrap intervals.",
            "- `endpoint_changes.csv`: paired descriptive N=10 minus N=2 changes.",
            "- `caption.tex`: publication caption with estimand and limitations.",
            "- `provenance.json`: inputs, hashes, run identities, bootstrap, and typography.",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=ROOT / "analyses/artifacts/synthetic/read_depth",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--standardized-profile-table",
        type=Path,
        help="Actual per-transcript NB2-standardized residual table; defaults to the alpha audit output.",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=5_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--figure-width", type=float, default=6.6)
    args = parser.parse_args()
    if args.bootstrap_repeats < 100:
        parser.error("Use at least 100 bootstrap draws.")
    if args.figure_width <= 0:
        parser.error("--figure-width must be positive.")

    audit_dir = args.audit_dir.resolve()
    output = (args.output_dir or audit_dir / "multidataset_mu_reconstruction").resolve()
    runs, validation = verified_runs(audit_dir)
    source = audit_dir / "recovery_per_transcript.csv"
    frame = load_records(source, runs)
    standardized_source = (
        args.standardized_profile_table
        or audit_dir / "alpha_recovery/position_error/profile_per_transcript.csv"
    ).resolve()
    frame = attach_standardized_error(frame, standardized_source)
    primary, depth_cohorts, exclusions = fixed_cohorts(frame, validation)
    summary = summarize(
        frame,
        primary,
        depth_cohorts,
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )
    changes = endpoint_changes(
        frame,
        primary,
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )

    output.mkdir(parents=True, exist_ok=True)
    frame["in_common_three_depths"] = frame["transcript_id"].isin(primary)
    frame["in_complete_own_depth_validation"] = [
        transcript_id in set(depth_cohorts[depth])
        for transcript_id, depth in zip(frame["transcript_id"], frame["depth"])
    ]
    frame.to_csv(output / "per_transcript.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    changes.to_csv(output / "endpoint_changes.csv", index=False)
    pd.DataFrame(
        [
            dict(cohort="common_three_depths", depth="all", transcript_id=transcript_id)
            for transcript_id in primary
        ]
        + [
            dict(cohort="own_depth_validation", depth=depth, transcript_id=transcript_id)
            for depth in DEPTHS
            for transcript_id in depth_cohorts[depth]
        ]
    ).to_csv(output / "cohort_ids.csv", index=False)

    typography = plot(summary, output, args.figure_width)
    write_caption(output, len(primary), args.bootstrap_repeats, args.bootstrap_seed)
    write_report(output, summary, changes, primary, depth_cohorts)
    command = " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv])
    provenance = {
        "command": command,
        "analysis": "observed-profile reconstruction; not latent recovery",
        "source": str(source),
        "source_sha256": file_hash(source),
        "standardized_error_source": str(standardized_source),
        "standardized_error_source_sha256": file_hash(standardized_source),
        "source_audit_provenance": str(audit_dir / "provenance.json"),
        "source_audit_provenance_sha256": file_hash(audit_dir / "provenance.json"),
        "checkpoint_variant": "best_pcc",
        "training_seed": 42,
        "depths": list(DEPTHS),
        "dataset_counts": list(COUNTS),
        "reference_weighting": "equal",
        "position_domain": "sense CDS after terminal removal; first/last 10 codons excluded",
        "aggregation": "metric within transcript-dataset; equal datasets within transcript; equal transcripts",
        "raw_rmse_units": "consensus count units; not scale-comparable across depths",
        "plotted_standardized_error": {
            "metric": STANDARDIZED_METRIC,
            "definition": "sqrt(mean over positions and equal-weight datasets of (mu-mean_r Y_r)^2 / [sum_r(mu_r+0.1*mu_r^2)/R^2])",
            "simulator_alpha": 0.1,
            "replicates": 2,
            "uses_fitted_alpha": False,
            "reference_value_one": "heuristic sampling-scale reference, not exact calibration null",
        },
        "bootstrap": {
            "repeats": args.bootstrap_repeats,
            "seed": args.bootstrap_seed,
            "unit": "transcript",
            "pairing": "one common resample across every depth, N, and metric",
            "interval": "pointwise percentile 95%",
            "training_seed_uncertainty_included": False,
        },
        "cohorts": {
            "common_three_depths": {
                "n": len(primary),
                "sha256": cohort_hash(primary),
            },
            "own_depth_validation": {
                depth: {"n": len(ids), "sha256": cohort_hash(ids)}
                for depth, ids in depth_cohorts.items()
            },
        },
        "exclusions": exclusions,
        "runs": runs,
        "typography": typography,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    print(f"Primary common cohort: {len(primary)} transcripts")
    print(
        summary[summary["cohort"] == "common_three_depths"]
        .pivot_table(index=["depth", "n_datasets"], columns="metric", values="mean")
        .to_string(float_format=lambda value: f"{value:.6f}")
    )
    print(output / f"{STEM}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
