"""Evaluate synthetic shared-profile recovery and observed-profile agreement.

The primary synthetic metric is the shared mean-one ``L_bio`` profile against
the deterministic mean-one latent kinetic profile ``K``.  It uses only the
CDS-observable interior, ``5 <= i < L - 5``.  Comparisons against held-out
``target`` profiles remain useful secondary diagnostics, but those targets are
both stochastic and deliberately dataset-biased; they must not be interpreted
as direct recovery of the shared latent profile.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    # Direct-file invocation (``python results/13_...py``) puts ``results``
    # on ``sys.path``; module invocation (``python -m results.13_...``) puts
    # the repository root on it.  Support both forms so result analyses do
    # not depend on the caller's working directory or launch style.
    from gamma_ablation.common import (
        DEFAULT_CONFIG,
        DEFAULT_DATASET_ENCODING,
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_RESULTS_ROOT,
        PredictionFile,
        RunRecord,
        bootstrap_summary,
        discover_runs,
        filter_runs,
        fisher_summary,
        inventory_dataframe,
        iter_rows,
        load_config,
        load_dataset_names,
        array_or_none,
        normalize_profile,
        profile_metrics,
        resolve_reference,
        select_latest_runs,
    )
except ModuleNotFoundError:  # pragma: no cover - depends on invocation form
    from analyses.gamma_ablation.common import (
        DEFAULT_CONFIG,
        DEFAULT_DATASET_ENCODING,
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_RESULTS_ROOT,
        PredictionFile,
        RunRecord,
        bootstrap_summary,
        discover_runs,
        filter_runs,
        fisher_summary,
        inventory_dataframe,
        iter_rows,
        load_config,
        load_dataset_names,
        array_or_none,
        normalize_profile,
        profile_metrics,
        resolve_reference,
        select_latest_runs,
    )

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None


COMPONENTS = ("L_bio", "mu")
METRICS = ("pearson", "spearman", "shape_rmse")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LATENT_TRUTH = (
    REPOSITORY_ROOT
    / "Datasets"
    / "Synthetic_data"
    / "artificial_ground_truth_kinetics_target_mean_one.parquet"
)
BOUNDARY_TRIM_CODONS = 5

# Compact, colour-blind-safe defaults suitable for a two-column ICLR/NeurIPS
# paper figure: restrained grid, no top/right box, embedded editable text in
# vector outputs, and enough marker contrast to remain readable in grayscale.
ICLR_NEURIPS_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.titleweight": "semibold",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.fontsize": 8.5,
    "legend.title_fontsize": 8.5,
    "legend.frameon": False,
    "lines.linewidth": 2.1,
    "lines.markersize": 5.5,
    "grid.color": "#D0D0D0",
    "grid.linewidth": 0.55,
    "grid.alpha": 0.55,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 300,
}

READ_DEPTH_ORDER = ("0p25_per_codon", "2_per_codon", "20_per_codon")
READ_DEPTH_LABELS = {
    "0p25_per_codon": "0.25 reads/codon",
    "2_per_codon": "2 reads/codon",
    "20_per_codon": "20 reads/codon",
}
# Okabe--Ito colours: reliable for common forms of colour-vision deficiency.
READ_DEPTH_STYLES = {
    "0p25_per_codon": {"color": "#0072B2", "marker": "o"},
    "2_per_codon": {"color": "#E69F00", "marker": "s"},
    "20_per_codon": {"color": "#009E73", "marker": "^"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "recovery")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-encoding", type=Path, default=DEFAULT_DATASET_ENCODING)
    parser.add_argument(
        "--latent-truth",
        type=Path,
        default=DEFAULT_LATENT_TRUTH,
        help="Deterministic synthetic K profile parquet (transcript_id, rib_profile).",
    )
    parser.add_argument(
        "--boundary-trim-codons",
        type=int,
        default=BOUNDARY_TRIM_CODONS,
        help=(
            "Exclude this many codons from each CDS end in every profile metric "
            "(default: 5)."
        ),
    )
    parser.add_argument("--strategy", action="append", default=None)
    parser.add_argument("--feature-preset", action="append", default=None)
    parser.add_argument("--seed", action="append", type=int, default=None)
    parser.add_argument("--n-datasets", action="append", type=int, default=None)
    parser.add_argument(
        "--exclude-n-datasets",
        action="append",
        type=int,
        default=None,
        help="Exclude these training dataset counts (repeatable), e.g. --exclude-n-datasets 80.",
    )
    parser.add_argument("--quality-power", action="append", type=float, default=None)
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Restrict per-dataset component metrics; shared consensus still uses all included datasets.",
    )
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--batch-rows", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help=(
            "Regenerate the compact primary-recovery figure from the existing "
            "recovery_by_condition.csv without reopening prediction parquets."
        ),
    )
    return parser.parse_args()


def dataset_name(value: Any, names: dict[int, str]) -> str:
    try:
        identifier = int(value)
    except (TypeError, ValueError):
        return "unknown"
    return names.get(identifier, f"dataset_{identifier}")


def load_latent_truth(path: Path) -> dict[str, np.ndarray]:
    """Load deterministic synthetic K profiles keyed by versioned transcript ID."""
    frame = pd.read_parquet(path, columns=["transcript_id", "rib_profile"])
    truth: dict[str, np.ndarray] = {}
    for row in frame.itertuples(index=False):
        transcript_id = str(row.transcript_id)
        profile = array_or_none(row.rib_profile)
        if profile is None or profile.size < 2 or not np.isfinite(profile).all():
            raise ValueError(
                f"Latent truth for {transcript_id!r} is not a finite one-dimensional profile."
            )
        if transcript_id in truth:
            raise ValueError(f"Duplicate latent transcript ID: {transcript_id}")
        truth[transcript_id] = profile
    if not truth:
        raise ValueError(f"No latent profiles found in {path}")
    return truth


def _valid_interior_mask(
    raw_mask: Any | None,
    length: int,
    boundary_trim_codons: int,
) -> np.ndarray:
    """Return valid physical CDS coordinates after padding and boundary masking."""
    if length <= 0:
        return np.zeros(0, dtype=bool)
    if boundary_trim_codons < 0:
        raise ValueError("boundary_trim_codons must be non-negative.")
    if raw_mask is None:
        valid = np.ones(length, dtype=bool)
    else:
        mask = array_or_none(raw_mask, dtype=bool)
        if mask is None:
            return np.zeros(length, dtype=bool)
        valid = np.zeros(length, dtype=bool)
        valid[: min(length, mask.size)] = mask[:length]
    trim = int(boundary_trim_codons)
    if trim:
        valid[:trim] = False
        valid[max(0, length - trim) :] = False
    return valid


def profile_pair_on_interior(
    predicted_value: Any,
    reference_value: Any,
    raw_mask: Any | None,
    boundary_trim_codons: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Align two profiles on the same valid, interior physical coordinates."""
    predicted = array_or_none(predicted_value)
    reference = array_or_none(reference_value)
    if predicted is None or reference is None:
        return None
    length = min(predicted.size, reference.size)
    if length < 2:
        return None
    predicted = predicted[:length]
    reference = reference[:length]
    valid = _valid_interior_mask(raw_mask, length, boundary_trim_codons)
    valid &= np.isfinite(predicted) & np.isfinite(reference)
    if int(valid.sum()) < 2:
        return None
    return predicted[valid], reference[valid]


def normalized_profile_on_interior(
    value: Any,
    raw_mask: Any | None,
    boundary_trim_codons: int,
) -> np.ndarray | None:
    """Mean-normalize a profile on the valid interior only."""
    profile = array_or_none(value)
    if profile is None:
        return None
    return normalize_profile(
        profile,
        _valid_interior_mask(raw_mask, profile.size, boundary_trim_codons),
    )


def summarize_transcript_rows(
    rows: list[dict[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    keys = [
        "run_id",
        "strategy",
        "training_scope",
        "depth",
        "mass_condition",
        "n_datasets",
        "quality_rank_power",
        "gamma_weighting",
        "feature_preset",
        "seed",
        "model_id",
        "split",
        "dataset",
        "component",
        "reference_column",
        "reference_kind",
        "scope",
    ]
    output: list[dict[str, Any]] = []
    for key, group in frame.groupby(keys, dropna=False, sort=False):
        record = dict(zip(keys, key))
        record["n_transcripts"] = int(group["transcript_id"].nunique())
        for metric in METRICS:
            summary = bootstrap_summary(
                group[metric], n_bootstrap=n_bootstrap, seed=seed
            )
            for name, value in summary.items():
                record[f"{metric}_{name}"] = value
        record.update(
            {
                f"pearson_{name}": value
                for name, value in fisher_summary(
                    group["pearson"], group["n_positions"]
                ).items()
            }
        )
        output.append(record)
    return output


def consensus_rows(
    run: RunRecord,
    prediction: PredictionFile,
    states: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    inconsistent = 0
    compared = 0
    maximum_difference = 0.0
    for transcript_id, state in states.items():
        predicted = state.get("predicted")
        if predicted is None:
            continue
        inconsistent += int(state.get("inconsistent", False))
        compared += int(state.get("seen", 0) > 1)
        maximum_difference = max(maximum_difference, float(state.get("max_abs_difference", 0.0)))
        for weighting, sums_key, counts_key in (
            ("equal", "equal_sum", "equal_count"),
            ("dataset_quality", "quality_sum", "quality_count"),
        ):
            sums = state.get(sums_key)
            counts = state.get(counts_key)
            if sums is None or counts is None:
                continue
            length = min(predicted.size, sums.size, counts.size)
            valid = (
                np.isfinite(predicted[:length])
                & np.isfinite(sums[:length])
                & (counts[:length] > 0.0)
            )
            if int(valid.sum()) < 4:
                continue
            reference = sums[:length][valid] / counts[:length][valid]
            metrics = profile_metrics(predicted[:length][valid], reference)
            rows.append(
                {
                    **run.metadata(),
                    "model_id": run.run_id,
                    "split": prediction.split,
                    "dataset": "__shared_consensus__",
                    "component": "L_bio",
                    "reference_column": f"target_consensus_{weighting}_interior",
                    "reference_kind": "observed_profile_proxy",
                    "scope": f"shared_consensus_{weighting}_interior",
                    "transcript_id": transcript_id,
                    **metrics,
                }
            )
    audit = {
        **run.metadata(),
        "split": prediction.split,
        "n_shared_transcripts": len(states),
        "n_compared_across_datasets": compared,
        "n_inconsistent_L_bio": inconsistent,
        "fraction_inconsistent_L_bio": inconsistent / compared if compared else np.nan,
        "maximum_absolute_L_bio_difference": maximum_difference,
    }
    return rows, audit


def process_file(
    run: RunRecord,
    prediction: PredictionFile,
    *,
    references: dict[str, list[str]],
    latent_truth: dict[str, np.ndarray],
    dataset_names: dict[int, str],
    batch_rows: int,
    n_bootstrap: int,
    bootstrap_seed: int,
    included_datasets: set[str] | None,
    boundary_trim_codons: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    resolved = {
        component: resolve_reference(prediction.columns, references[component])
        for component in COMPONENTS
    }
    requested = {
        "transcript_id",
        "dataset_id",
        "mask",
        "dataset_quality_weight",
        *COMPONENTS,
        *(reference for reference, _ in resolved.values() if reference),
    }
    transcript_rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []
    latent_seen: set[str] = set()
    latent_missing = 0
    shared_states: dict[str, dict[str, Any]] = {}
    model_id = prediction.experiment if run.training_scope == "individual_dataset" else run.run_id

    for row in iter_rows(prediction, requested, batch_rows=batch_rows):
        transcript_id = str(row.get("transcript_id", "unknown"))
        current_dataset = dataset_name(row.get("dataset_id"), dataset_names)
        common = {
            **run.metadata(),
            "model_id": model_id,
            "split": prediction.split,
            "dataset": current_dataset,
            "transcript_id": transcript_id,
            "scope": "per_dataset_interior",
        }

        # ``L_bio`` is shared across dataset rows.  Score it once per
        # transcript against deterministic K on the physical CDS interior.
        # This is the primary synthetic-recovery statistic; unlike ``target``,
        # K contains neither sampling noise nor a programmed dataset bias.
        if transcript_id not in latent_seen:
            latent_seen.add(transcript_id)
            truth = latent_truth.get(transcript_id)
            if truth is None:
                latent_missing += 1
            else:
                pair = profile_pair_on_interior(
                    row.get("L_bio"),
                    truth,
                    row.get("mask"),
                    boundary_trim_codons,
                )
                if pair is not None:
                    latent_rows.append(
                        {
                            **run.metadata(),
                            "model_id": model_id,
                            "split": prediction.split,
                            "dataset": "__latent_truth__",
                            "transcript_id": transcript_id,
                            "component": "L_bio",
                            "reference_column": "rib_profile",
                            "reference_kind": "latent_ground_truth",
                            "scope": "shared_latent_truth_interior",
                            **profile_metrics(*pair),
                        }
                    )
        if not included_datasets or current_dataset in included_datasets:
            for component in COMPONENTS:
                reference, reference_kind = resolved[component]
                if reference is None:
                    continue
                pair = profile_pair_on_interior(
                    row.get(component),
                    row.get(reference),
                    row.get("mask"),
                    boundary_trim_codons,
                )
                if pair is None:
                    continue
                transcript_rows.append(
                    {
                        **common,
                        "component": component,
                        "reference_column": reference,
                        "reference_kind": reference_kind,
                        **profile_metrics(*pair),
                    }
                )

        # A multi-dataset L_bio is shared. Build two transcript-level observed
        # references without counting the repeated L_bio as independent data.
        if run.training_scope != "multi_dataset" or resolved["L_bio"][0] != "target":
            continue
        target = normalized_profile_on_interior(
            row.get("target"), row.get("mask"), boundary_trim_codons
        )
        if target is None:
            continue
        state = shared_states.setdefault(transcript_id, {"seen": 0})
        if "predicted" not in state:
            predicted = normalized_profile_on_interior(
                row.get("L_bio"), row.get("mask"), boundary_trim_codons
            )
            if predicted is None:
                continue
            length = min(predicted.size, target.size)
            state["predicted"] = predicted[:length]
            state["max_abs_difference"] = 0.0
            for prefix in ("equal", "quality"):
                state[f"{prefix}_sum"] = np.zeros(length, dtype=np.float64)
                state[f"{prefix}_count"] = np.zeros(length, dtype=np.float64)
        elif int(state["seen"]) < 2:
            # One independent repeat per transcript is enough to verify the
            # architectural sharing invariant without re-normalizing L_bio for
            # every dataset in large N=80/114 bundles.
            predicted = normalized_profile_on_interior(
                row.get("L_bio"), row.get("mask"), boundary_trim_codons
            )
            if predicted is None:
                continue
            length = min(predicted.size, target.size, state["predicted"].size)
            overlap_valid = np.isfinite(predicted[:length]) & np.isfinite(state["predicted"][:length])
            difference = (
                float(np.max(np.abs(predicted[:length][overlap_valid] - state["predicted"][:length][overlap_valid])))
                if np.any(overlap_valid)
                else np.nan
            )
            if np.isfinite(difference):
                state["max_abs_difference"] = max(float(state["max_abs_difference"]), difference)
            state["inconsistent"] = bool(
                state.get("inconsistent", False)
                or not np.allclose(
                    predicted[:length],
                    state["predicted"][:length],
                    rtol=1.0e-5,
                    atol=1.0e-7,
                    equal_nan=True,
                )
            )
        else:
            length = min(target.size, state["predicted"].size)
        state["seen"] += 1
        valid = np.isfinite(target[:length])
        state["equal_sum"][:length][valid] += target[:length][valid]
        state["equal_count"][:length][valid] += 1.0
        quality_weight = float(row.get("dataset_quality_weight", 1.0) or 1.0)
        if not np.isfinite(quality_weight) or quality_weight <= 0.0:
            quality_weight = 1.0
        state["quality_sum"][:length][valid] += quality_weight * target[:length][valid]
        state["quality_count"][:length][valid] += quality_weight

    dataset_summary = summarize_transcript_rows(
        transcript_rows, n_bootstrap=n_bootstrap, seed=bootstrap_seed
    )
    shared_rows, audit = consensus_rows(run, prediction, shared_states)
    shared_rows = latent_rows + shared_rows
    audit.update(
        {
            "n_latent_truth_compared": int(len(latent_rows)),
            "n_latent_truth_missing": int(latent_missing),
            "boundary_trim_codons": int(boundary_trim_codons),
        }
    )
    shared_summary = summarize_transcript_rows(
        shared_rows, n_bootstrap=n_bootstrap, seed=bootstrap_seed
    )
    return dataset_summary, shared_summary, audit


def common_anchor_sets(dataset_summary: pd.DataFrame) -> dict[tuple[Any, ...], set[str]]:
    anchors: dict[tuple[Any, ...], set[str]] = {}
    if dataset_summary.empty:
        return anchors
    keys = [
        "strategy",
        "depth",
        "mass_condition",
        "feature_preset",
        "seed",
        "split",
        "component",
    ]
    for key, group in dataset_summary.groupby(keys, dropna=False):
        sets = [set(run_group["dataset"]) for _, run_group in group.groupby("run_id")]
        anchors[key] = set.intersection(*sets) if sets else set()
    return anchors


def condition_summary(dataset_summary: pd.DataFrame, shared_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    anchors = common_anchor_sets(dataset_summary)
    run_keys = [
        "run_id",
        "strategy",
        "training_scope",
        "depth",
        "mass_condition",
        "n_datasets",
        "quality_rank_power",
        "gamma_weighting",
        "feature_preset",
        "seed",
        "split",
        "component",
        "reference_column",
        "reference_kind",
    ]
    for key, full_group in dataset_summary.groupby(run_keys, dropna=False, sort=False):
        base = dict(zip(run_keys, key))
        anchor_key = tuple(
            base[name]
            for name in (
                "strategy",
                "depth",
                "mass_condition",
                "feature_preset",
                "seed",
                "split",
                "component",
            )
        )
        for scope, group in (
            ("all_included", full_group),
            ("strategy_common_anchor", full_group[full_group["dataset"].isin(anchors.get(anchor_key, set()))]),
        ):
            if group.empty:
                continue
            rows.append(
                {
                    **base,
                    "comparison_scope": scope,
                    "n_observed_datasets": int(group["dataset"].nunique()),
                    "n_transcripts_sum": int(group["n_transcripts"].sum()),
                    "pearson_dataset_macro_mean": float(group["pearson_fisher"].mean()),
                    "pearson_worst_dataset": float(group["pearson_fisher"].min()),
                    "spearman_dataset_macro_mean": float(group["spearman_mean"].mean()),
                    "shape_rmse_dataset_macro_mean": float(group["shape_rmse_mean"].mean()),
                }
            )
    for record in shared_summary.to_dict("records"):
        rows.append(
            {
                **{name: record.get(name) for name in run_keys},
                "comparison_scope": record["scope"],
                "n_observed_datasets": np.nan,
                "n_transcripts_sum": record["n_transcripts"],
                "pearson_dataset_macro_mean": record["pearson_fisher"],
                "pearson_worst_dataset": np.nan,
                "spearman_dataset_macro_mean": record["spearman_mean"],
                "shape_rmse_dataset_macro_mean": record["shape_rmse_mean"],
            }
        )
    return pd.DataFrame(rows)


def _depth_sort_key(depth: Any) -> tuple[int, str]:
    value = str(depth)
    try:
        return (READ_DEPTH_ORDER.index(value), value)
    except ValueError:
        return (len(READ_DEPTH_ORDER), value)


def _safe_plot_token(value: Any) -> str:
    return str(value).replace("/", "_").replace(" ", "_").replace(".", "p")


def plot_primary_latent_recovery_by_depth(summary: pd.DataFrame, output: Path) -> None:
    """Plot only the shared-L-versus-deterministic-K recovery statistics.

    ``pearson_dataset_macro_mean`` is named for historical compatibility in
    ``recovery_by_condition.csv``.  For the shared latent-truth scope it is
    exactly the transcript-length Fisher-weighted PCC; the RMSE column is the
    mean of transcript-level mean-one shape RMSEs.  These are the two primary
    deterministic-recovery views, so observed-profile diagnostics are kept in
    the CSVs but deliberately excluded from this paper figure.
    """
    if plt is None or summary.empty:
        return
    required = {
        "depth",
        "n_datasets",
        "pearson_dataset_macro_mean",
        "shape_rmse_dataset_macro_mean",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise KeyError(f"Recovery summary is missing plot columns: {missing}")

    data = summary[
        (summary["split"] == "main_val")
        & (summary["component"] == "L_bio")
        & (summary["comparison_scope"] == "shared_latent_truth_interior")
    ].copy()
    if data.empty:
        return
    if data.duplicated(["depth", "n_datasets"]).any():
        duplicate = data.loc[
            data.duplicated(["depth", "n_datasets"], keep=False),
            ["depth", "n_datasets", "run_id"],
        ]
        raise ValueError(
            "Expected one primary latent-recovery row per read-depth/dataset-count; "
            f"found duplicates: {duplicate.to_dict('records')[:5]}"
        )

    depths = sorted(data["depth"].dropna().unique(), key=_depth_sort_key)
    dataset_counts = sorted(int(value) for value in data["n_datasets"].dropna().unique())
    if not depths or not dataset_counts:
        return

    with matplotlib.rc_context(ICLR_NEURIPS_RC):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(8.4, 3.35),
            sharex=True,
            constrained_layout=True,
        )
        panel_specs = (
            (
                "pearson_dataset_macro_mean",
                "Fisher-weighted transcript PCC",
                "a  Shared-profile shape recovery",
            ),
            (
                "shape_rmse_dataset_macro_mean",
                r"Mean transcript RMSE of $L_{\mathrm{bio}}$ vs $K$",
                "b  Shared-profile error",
            ),
        )
        handles = []
        for depth in depths:
            line = data[data["depth"] == depth].sort_values("n_datasets")
            style = READ_DEPTH_STYLES.get(
                str(depth),
                {"color": "#6C6C6C", "marker": "D"},
            )
            label = READ_DEPTH_LABELS.get(str(depth), str(depth).replace("_", " "))
            for axis_index, (metric, ylabel, title) in enumerate(panel_specs):
                line_handle = axes[axis_index].plot(
                    line["n_datasets"],
                    line[metric],
                    label=label,
                    color=style["color"],
                    marker=style["marker"],
                    markeredgecolor="white",
                    markeredgewidth=0.65,
                    zorder=3,
                )[0]
                if axis_index == 0:
                    handles.append(line_handle)
                axes[axis_index].set_ylabel(ylabel)
                axes[axis_index].set_title(title, loc="left", pad=8)
                axes[axis_index].grid(axis="y")
                axes[axis_index].set_axisbelow(True)
                axes[axis_index].set_xlabel("Number of training datasets")
                axes[axis_index].set_xticks(dataset_counts)
                axes[axis_index].set_xlim(min(dataset_counts) - 0.25, max(dataset_counts) + 0.25)

        # The upper-right region of the RMSE panel is unused after the first
        # few dataset counts, making it a compact paper-friendly home for the
        # one shared legend without consuming a separate title row.
        axes[1].legend(
            handles=handles,
            labels=[handle.get_label() for handle in handles],
            title="Read depth",
            loc="upper right",
            handlelength=2.2,
        )
        fig.text(
            0.5,
            -0.015,
            r"Shared $L_{\mathrm{bio}}$ evaluated against deterministic latent $K$; "
            r"interior codons only ($5 \leq i < L - 5$).",
            ha="center",
            va="top",
            fontsize=8.3,
            color="#404040",
        )
        for suffix in (".png", ".pdf", ".svg"):
            fig.savefig(output.with_suffix(suffix), bbox_inches="tight")
        plt.close(fig)


def write_primary_recovery_plots(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Write one two-panel depth-comparison figure per non-depth condition."""
    output_dir.mkdir(parents=True, exist_ok=True)
    primary = summary[
        (summary["split"] == "main_val")
        & (summary["component"] == "L_bio")
        & (summary["comparison_scope"] == "shared_latent_truth_interior")
    ].copy()
    if primary.empty:
        return []
    controls = (
        "strategy",
        "training_scope",
        "mass_condition",
        "quality_rank_power",
        "gamma_weighting",
        "feature_preset",
        "seed",
    )
    missing_controls = [column for column in controls if column not in primary.columns]
    if missing_controls:
        raise KeyError(f"Recovery summary is missing control columns: {missing_controls}")

    written: list[Path] = []
    for key, group in primary.groupby(list(controls), dropna=False, sort=True):
        values = dict(zip(controls, key))
        filename = (
            "recovery_main_val_by_read_depth"
            f"__{_safe_plot_token(values['strategy'])}"
            f"__{_safe_plot_token(values['mass_condition'])}"
            f"__rankp{_safe_plot_token(values['quality_rank_power'])}.png"
        )
        output = output_dir / filename
        plot_primary_latent_recovery_by_depth(group, output)
        written.append(output)
    return written


def clear_recovery_plot_outputs(output_dir: Path) -> None:
    """Remove only figures generated by this recovery script, never summary tables."""
    (output_dir / "recovery_main_val.png").unlink(missing_ok=True)
    plot_dir = output_dir / "plots_by_depth"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for pattern in (
        "recovery_main_val__*",
        "recovery_main_val_by_read_depth__*",
    ):
        for old_plot in plot_dir.glob(pattern):
            if old_plot.is_file():
                old_plot.unlink()


def refresh_plot_settings(output_dir: Path) -> None:
    """Keep the generated provenance explicit when ``--plot-only`` is used."""
    path = output_dir / "analysis_settings.json"
    if not path.is_file():
        return
    try:
        settings = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    settings.pop("read_depths_are_never_pooled_in_plots", None)
    settings.update(
        {
            "plot_partition_keys": [
                "strategy",
                "training_scope",
                "mass_condition",
                "quality_rank_power",
                "gamma_weighting",
                "feature_preset",
                "seed",
            ],
            "read_depths_combined_in_primary_plots": True,
            "primary_figure": (
                "One row of two panels: Fisher-weighted transcript PCC and "
                "mean per-transcript L_bio-vs-K RMSE; colour/marker encodes read depth."
            ),
            "primary_figure_style": "compact ICLR/NeurIPS-inspired publication style",
        }
    )
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.boundary_trim_codons < 0:
        raise ValueError("--boundary-trim-codons must be non-negative.")
    if args.plot_only:
        summary_path = args.output_dir / "recovery_by_condition.csv"
        if not summary_path.is_file():
            raise FileNotFoundError(
                "--plot-only requires an existing recovery summary: "
                f"{summary_path}"
            )
        conditions = pd.read_csv(summary_path)
        clear_recovery_plot_outputs(args.output_dir)
        written = write_primary_recovery_plots(
            conditions, args.output_dir / "plots_by_depth"
        )
        refresh_plot_settings(args.output_dir)
        if not written:
            raise RuntimeError(
                "No main-validation shared-L-versus-K rows were available for plotting."
            )
        print("Regenerated compact primary-recovery plot(s):")
        for path in written:
            print(f"  {path}")
        return
    config = load_config(args.config)
    statistics = config.get("statistics", {})
    batch_rows = args.batch_rows or int(statistics.get("batch_rows", 256))
    n_bootstrap = args.bootstrap if args.bootstrap is not None else int(statistics.get("bootstrap_replicates", 500))
    bootstrap_seed = args.bootstrap_seed if args.bootstrap_seed is not None else int(statistics.get("bootstrap_seed", 42))
    splits = set(args.split or config.get("splits", {}).get("recovery", ["main_val", "css_benchmark"]))
    # Older analysis configs had an explicit ``reference_columns`` block.  The
    # current synthetic config intentionally keeps the resolved prediction
    # schema minimal, so use its stable observed-target fallback when that
    # legacy block is absent.  ``resolve_reference`` still promotes any latent
    # column present in a future prediction parquet.
    references = config.get("reference_columns", {})
    if not isinstance(references, dict):
        references = {}
    references = {
        "L_bio": references.get(
            "L_bio", ["L_bio_true", "latent_L_bio", "target"]
        ),
        "mu": references.get(
            "mu", ["mu_true", "latent_mu", "target"]
        ),
    }
    latent_truth_path = args.latent_truth.expanduser()
    if not latent_truth_path.is_absolute():
        latent_truth_path = REPOSITORY_ROOT / latent_truth_path
    latent_truth = load_latent_truth(latent_truth_path)

    all_runs = discover_runs(args.results_root)
    latest = select_latest_runs(all_runs)
    selected = filter_runs(
        latest,
        strategies=set(args.strategy) if args.strategy else None,
        feature_presets=set(args.feature_preset) if args.feature_preset else None,
        seeds=set(args.seed) if args.seed else None,
        dataset_counts=set(args.n_datasets) if args.n_datasets else None,
        quality_powers=set(args.quality_power) if args.quality_power else None,
        max_runs=args.max_runs,
    )
    excluded_dataset_counts = set(args.exclude_n_datasets or [])
    selected = [
        run for run in selected if run.n_datasets not in excluded_dataset_counts
    ]
    if not selected:
        raise RuntimeError("No readable prediction runs matched the requested filters.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis_settings.json").write_text(
        json.dumps(
            {
                "results_root": str(args.results_root.resolve()),
                "config": str(args.config.resolve()),
                "latent_truth": str(latent_truth_path.resolve()),
                "primary_shared_profile_metric": (
                    "L_bio versus deterministic rib_profile (K), one score per "
                    "transcript, using 5 <= i < L-5"
                ),
                "boundary_trim_codons": int(args.boundary_trim_codons),
                "observed_profile_metrics_are_secondary": True,
                "strategies": args.strategy,
                "feature_presets": args.feature_preset,
                "seeds": args.seed,
                "dataset_counts": args.n_datasets,
                "excluded_dataset_counts": sorted(excluded_dataset_counts),
                "quality_powers": args.quality_power,
                "splits": sorted(splits),
                "per_dataset_filter": args.dataset,
                "shared_consensus_uses_all_included_datasets": True,
                "plot_partition_keys": [
                    "strategy",
                    "training_scope",
                    "mass_condition",
                    "quality_rank_power",
                    "gamma_weighting",
                    "feature_preset",
                    "seed",
                ],
                "read_depths_combined_in_primary_plots": True,
                "primary_figure": (
                    "One row of two panels: Fisher-weighted transcript PCC and "
                    "mean per-transcript L_bio-vs-K RMSE; colour/marker encodes read depth."
                ),
                "primary_figure_style": "compact ICLR/NeurIPS-inspired publication style",
                "bootstrap_replicates": n_bootstrap,
                "bootstrap_seed": bootstrap_seed,
                "batch_rows": batch_rows,
                "selected_run_ids": [run.run_id for run in selected],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    inventory_dataframe(all_runs, selected).to_csv(args.output_dir / "run_inventory.csv", index=False)
    dataset_names = load_dataset_names(args.dataset_encoding)
    dataset_records: list[dict[str, Any]] = []
    shared_records: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for run_index, run in enumerate(selected, start=1):
        files = [file for file in run.usable_files if file.split in splits]
        print(f"[{run_index}/{len(selected)}] {run.run_id}: {len(files)} file(s)")
        for prediction in files:
            dataset_rows, shared_rows, audit = process_file(
                run,
                prediction,
                references=references,
                latent_truth=latent_truth,
                dataset_names=dataset_names,
                batch_rows=batch_rows,
                n_bootstrap=n_bootstrap,
                bootstrap_seed=bootstrap_seed,
                included_datasets=set(args.dataset) if args.dataset else None,
                boundary_trim_codons=int(args.boundary_trim_codons),
            )
            dataset_records.extend(dataset_rows)
            shared_records.extend(shared_rows)
            audits.append(audit)

    dataset_summary = pd.DataFrame(dataset_records)
    shared_summary = pd.DataFrame(shared_records)
    dataset_summary.to_csv(args.output_dir / "recovery_by_dataset.csv", index=False)
    shared_summary.to_csv(
        args.output_dir / "L_bio_shared_reference_comparisons.csv", index=False
    )
    # Retain the historical filename for scripts that consume only the observed
    # consensus diagnostic. The primary latent-recovery table above contains
    # both latent and observed shared-profile references.
    if shared_summary.empty:
        consensus_only = shared_summary.copy()
    else:
        consensus_only = shared_summary[
            shared_summary["scope"].astype(str).str.startswith("shared_consensus_")
        ]
    consensus_only.to_csv(
        args.output_dir / "L_bio_shared_consensus.csv", index=False
    )
    pd.DataFrame(audits).to_csv(args.output_dir / "L_bio_sharing_audit.csv", index=False)
    conditions = condition_summary(dataset_summary, shared_summary)
    conditions.to_csv(args.output_dir / "recovery_by_condition.csv", index=False)
    # The prior four-row figures plotted every read depth separately and mixed
    # primary recovery with secondary observed-profile diagnostics.  The paper
    # figure instead compares read depths directly and reports only the two
    # deterministic shared-profile metrics.
    clear_recovery_plot_outputs(args.output_dir)
    written = write_primary_recovery_plots(
        conditions, args.output_dir / "plots_by_depth"
    )
    if written:
        print("Saved compact primary-recovery plot(s):")
        for path in written:
            print(f"  {path}")
    print(f"Saved recovery analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
