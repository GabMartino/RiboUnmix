"""Measure biological quality of L_bio across gamma-centering settings.

The analysis includes CSS recall at configurable z-score thresholds with exact
and +/-1-codon matching, P/PP/PPP enrichment, and transcript-normalized 5' and
stop-proximal meta-profiles. L_bio is de-duplicated across datasets because it
is the shared branch; target, mu, and gamma are retained per dataset to help
distinguish biological structure from dataset-specific technical structure.

For terminal plots, "transcript-normalized" means amplitude normalization:
each source profile is divided by its own valid-codon mean. It does not mean
that transcript positions are rescaled. Stop-aligned profiles are reversed, so
position 0 is the stop codon, position 1 is the preceding codon, and increasing
position moves upstream toward the 5' end. The plotted target is the per-dataset
codon-wise replica mean; targets from different datasets are pooled as separate
dataset-transcript profiles and are never averaged across datasets.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from gamma_ablation.common import (
        DEFAULT_CONFIG,
        DEFAULT_DATASET_ENCODING,
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_RESULTS_ROOT,
        PredictionFile,
        RunRecord,
        array_or_none,
        bootstrap_summary,
        discover_runs,
        filter_runs,
        inventory_dataframe,
        load_codon_to_amino_acid,
        load_config,
        load_dataset_names,
        normalize_profile,
        select_latest_runs,
        wilson_interval,
    )
except ModuleNotFoundError:  # pragma: no cover - depends on invocation form
    from analyses.gamma_ablation.common import (
        DEFAULT_CONFIG,
        DEFAULT_DATASET_ENCODING,
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_RESULTS_ROOT,
        PredictionFile,
        RunRecord,
        array_or_none,
        bootstrap_summary,
        discover_runs,
        filter_runs,
        inventory_dataframe,
        load_codon_to_amino_acid,
        load_config,
        load_dataset_names,
        normalize_profile,
        select_latest_runs,
        wilson_interval,
    )

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


SIGNALS = ("L_bio", "mu", "target", "gamma")

TERMINAL_SIGNAL_TITLES = {
    "L_bio": (
        r"Shared biological load $L_{\mathrm{bio}}$"
        "\n(one profile per transcript; dataset-deduplicated)"
    ),
    "mu": (
        r"Predicted mean $\mu_{dt}$"
        "\n(dataset–transcript profiles pooled)"
    ),
    "target": (
        r"Observed target $y_{dt}$"
        "\n(per-dataset replica mean; datasets pooled, not averaged)"
    ),
}

TERMINAL_PROFILE_NOTE = (
    "Normalization: each nonzero-mean source profile is divided by its own mean over valid "
    "codons before aggregation (1 = that profile's transcript mean); zero-mean profiles are excluded.\n"
    r"Target: $y_{dt}$ is the codon-wise arithmetic mean of biological replicas within dataset $d$. "
    "It is not averaged across datasets; target and μ pool dataset–transcript profiles, while "
    r"shared $L_{\mathrm{bio}}$ contributes once per transcript."
    "\n"
    "Curves are means, shaded bands are ±1 SEM, and the box in each panel gives the range of "
    "contributing profiles across positions and p settings."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "biological_quality")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-encoding", type=Path, default=DEFAULT_DATASET_ENCODING)
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
    parser.add_argument("--dataset", action="append", default=None, help="Restrict dataset-specific signals; shared L_bio is unaffected.")
    parser.add_argument("--signal", action="append", choices=SIGNALS, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--batch-rows", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    parser.add_argument(
        "--terminal-profiles-only",
        action="store_true",
        help=(
            "Recompute only main-validation start/stop metaprofiles and their plots. "
            "This skips motif, CSS, and bootstrap work and leaves their existing CSV/PNG "
            "outputs unchanged."
        ),
    )
    return parser.parse_args()


def dataset_name(value: Any, names: dict[int, str]) -> str:
    try:
        identifier = int(value)
    except (TypeError, ValueError):
        return "unknown"
    return names.get(identifier, f"dataset_{identifier}")


def iter_relevant_rows(
    prediction: PredictionFile,
    requested_columns: Iterable[str],
    *,
    batch_rows: int,
    dataset_names: dict[int, str],
    included_datasets: set[str] | None,
    select_one_shared_row: bool,
) -> Iterable[dict[str, Any]]:
    """Materialize array columns only for needed dataset rows and one shared row.

    This is important for N=80/114 Parquets: converting codon/profile arrays for
    every repeated dataset row would dominate both memory traffic and runtime.
    """
    columns = [column for column in requested_columns if column in prediction.columns]
    parquet = pq.ParquetFile(prediction.path)
    selected_shared_transcripts: set[str] = set()
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns):
        names = batch.schema.names
        transcript_values = batch.column(names.index("transcript_id")).to_pylist()
        dataset_values = batch.column(names.index("dataset_id")).to_pylist()
        keep: list[int] = []
        shared_flags: list[bool] = []
        for index, (transcript_value, dataset_value) in enumerate(zip(transcript_values, dataset_values)):
            transcript_id = str(transcript_value)
            dataset = dataset_name(dataset_value, dataset_names)
            keep_dataset = not included_datasets or dataset in included_datasets
            keep_shared = select_one_shared_row and transcript_id not in selected_shared_transcripts
            if keep_shared:
                selected_shared_transcripts.add(transcript_id)
            if keep_dataset or keep_shared:
                keep.append(index)
                shared_flags.append(keep_shared)
        if not keep:
            continue
        selected = batch.take(pa.array(keep, type=pa.int64()))
        values = selected.to_pydict()
        for index in range(selected.num_rows):
            row = {column: values[column][index] for column in columns}
            row["__shared_candidate"] = shared_flags[index]
            yield row


def css_positions(value: Any, length: int) -> np.ndarray:
    positions = array_or_none(value)
    if positions is None:
        return np.asarray([], dtype=np.int64)
    positions = np.rint(positions).astype(np.int64, copy=False)
    return np.unique(positions[(positions >= 0) & (positions < length)])


def zscore(profile: np.ndarray) -> np.ndarray | None:
    valid = np.isfinite(profile)
    values = profile[valid]
    if values.size < 4:
        return None
    standard_deviation = float(np.std(values))
    if standard_deviation <= 1.0e-12:
        return None
    output = np.full(profile.size, np.nan, dtype=np.float64)
    output[valid] = (values - float(np.mean(values))) / standard_deviation
    return output


def amino_acids(codon_ids: Any, id_to_aa: dict[int, str], length: int) -> np.ndarray | None:
    identifiers = array_or_none(codon_ids, dtype=np.int64)
    if identifiers is None:
        return None
    usable = min(length, identifiers.size)
    return np.asarray([id_to_aa.get(int(value), "") for value in identifiers[:usable]], dtype=object)


def motif_mask(sequence: np.ndarray, motif: str) -> np.ndarray:
    result = np.zeros(sequence.size, dtype=bool)
    width = len(motif)
    if width == 0 or sequence.size < width:
        return result
    matches = np.ones(sequence.size - width + 1, dtype=bool)
    for offset, residue in enumerate(motif):
        matches &= sequence[offset : offset + matches.size] == residue
    # Mark every residue participating in the motif. This matches the existing
    # biological-profile analysis and is stated in the output documentation.
    for offset in range(width):
        result[offset : offset + matches.size] |= matches
    return result


def log2_enrichment(profile: np.ndarray, selected: np.ndarray, background: np.ndarray) -> float:
    selected_values = profile[selected & np.isfinite(profile)]
    background_values = profile[background & np.isfinite(profile)]
    if selected_values.size < 1 or background_values.size < 20:
        return np.nan
    numerator = float(np.mean(selected_values))
    denominator = float(np.mean(background_values))
    if numerator <= 1.0e-12 or denominator <= 1.0e-12:
        return np.nan
    return float(np.log2(numerator / denominator))


def window_log2_ratio(profile: np.ndarray, near: tuple[int, int], baseline: tuple[int, int]) -> float:
    near_values = profile[slice(*near)]
    baseline_values = profile[slice(*baseline)]
    near_values = near_values[np.isfinite(near_values)]
    baseline_values = baseline_values[np.isfinite(baseline_values)]
    if near_values.size < 5 or baseline_values.size < 5:
        return np.nan
    numerator = float(np.mean(near_values))
    denominator = float(np.mean(baseline_values))
    if numerator <= 1.0e-12 or denominator <= 1.0e-12:
        return np.nan
    return float(np.log2(numerator / denominator))


class ProfileAccumulator:
    def __init__(self, size: int) -> None:
        self.sum = np.zeros(size, dtype=np.float64)
        self.sumsq = np.zeros(size, dtype=np.float64)
        self.count = np.zeros(size, dtype=np.int64)

    def add(self, values: np.ndarray) -> None:
        length = min(values.size, self.sum.size)
        valid = np.isfinite(values[:length])
        self.sum[:length][valid] += values[:length][valid]
        self.sumsq[:length][valid] += values[:length][valid] ** 2
        self.count[:length][valid] += 1

    def records(self) -> list[dict[str, Any]]:
        rows = []
        for position in np.flatnonzero(self.count > 0):
            n = int(self.count[position])
            mean = float(self.sum[position] / n)
            sem = np.nan
            if n > 1:
                variance = max(
                    float((self.sumsq[position] - self.sum[position] ** 2 / n) / (n - 1)),
                    0.0,
                )
                sem = float(np.sqrt(variance / n))
            rows.append({"position_codon": int(position), "mean": mean, "sem": sem, "n": n})
        return rows


def summarize_metric_values(
    run: RunRecord,
    prediction: PredictionFile,
    model_id: str,
    values: dict[tuple[str, str, str], list[float]],
    *,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for (dataset, signal, metric), metric_values in sorted(values.items()):
        summary = bootstrap_summary(metric_values, n_bootstrap=n_bootstrap, seed=seed)
        rows.append(
            {
                **run.metadata(),
                "model_id": model_id,
                "split": prediction.split,
                "dataset": dataset,
                "signal": signal,
                "metric": metric,
                **summary,
            }
        )
    return rows


def process_profile(
    *,
    profile: np.ndarray,
    sequence: np.ndarray | None,
    css: np.ndarray,
    dataset: str,
    signal: str,
    split: str,
    motifs: Iterable[str],
    thresholds: Iterable[float],
    margins: Iterable[int],
    start_window: tuple[int, int],
    start_baseline: tuple[int, int],
    stop_window: tuple[int, int],
    stop_baseline: tuple[int, int],
    metric_values: dict[tuple[str, str, str], list[float]],
    css_counts: dict[tuple[str, str, float, int], list[int]],
    terminal_profiles: dict[tuple[str, str], ProfileAccumulator],
    profile_codons: int,
) -> None:
    if split == "main_val":
        if sequence is not None:
            usable = min(profile.size, sequence.size)
            valid = np.isfinite(profile[:usable])
            for motif in motifs:
                selected = motif_mask(sequence[:usable], motif) & valid
                value = log2_enrichment(profile[:usable], selected, (~selected) & valid)
                if np.isfinite(value):
                    metric_values[(dataset, signal, f"motif_{motif}")].append(value)

        start_value = window_log2_ratio(profile, start_window, start_baseline)
        if np.isfinite(start_value):
            metric_values[(dataset, signal, "start_ramp")].append(start_value)
        reversed_profile = profile[::-1]
        stop_value = window_log2_ratio(reversed_profile, stop_window, stop_baseline)
        if np.isfinite(stop_value):
            metric_values[(dataset, signal, "stop_ramp")].append(stop_value)
        terminal_profiles[(signal, "start")].add(profile[:profile_codons])
        terminal_profiles[(signal, "stop")].add(reversed_profile[:profile_codons])

    if split != "css_benchmark" or css.size == 0:
        return
    standardized = zscore(profile)
    if standardized is None:
        return
    for threshold in thresholds:
        for margin in margins:
            counts = css_counts[(dataset, signal, float(threshold), int(margin))]
            for center in css:
                left = max(0, int(center) - int(margin))
                right = min(standardized.size, int(center) + int(margin) + 1)
                window = standardized[left:right]
                if not np.isfinite(window).any():
                    continue
                counts[1] += 1
                counts[0] += int(np.nanmax(window) >= float(threshold))


def process_file(
    run: RunRecord,
    prediction: PredictionFile,
    *,
    dataset_names: dict[int, str],
    id_to_aa: dict[int, str],
    motifs: tuple[str, ...],
    thresholds: tuple[float, ...],
    margins: tuple[int, ...],
    ramps: dict[str, Any],
    batch_rows: int,
    n_bootstrap: int,
    bootstrap_seed: int,
    included_datasets: set[str] | None,
    included_signals: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    requested = {
        "transcript_id",
        "dataset_id",
        "length",
        "lengths",
        "mask",
        "codon_ids",
        "css",
        *SIGNALS,
    }
    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    css_counts: dict[tuple[str, str, float, int], list[int]] = defaultdict(lambda: [0, 0])
    profile_codons = int(ramps.get("profile_codons", 200))
    terminal_profiles: dict[tuple[str, str], ProfileAccumulator] = defaultdict(
        lambda: ProfileAccumulator(profile_codons)
    )
    model_id = prediction.experiment if run.training_scope == "individual_dataset" else run.run_id
    start_window = tuple(map(int, ramps.get("start_window", [0, 50])))
    start_baseline = tuple(map(int, ramps.get("start_baseline", [100, 200])))
    stop_window = tuple(map(int, ramps.get("stop_window", [0, 50])))
    stop_baseline = tuple(map(int, ramps.get("stop_baseline", [100, 200])))

    for row in iter_relevant_rows(
        prediction,
        requested,
        batch_rows=batch_rows,
        dataset_names=dataset_names,
        included_datasets=included_datasets,
        select_one_shared_row=(
            "L_bio" in included_signals and run.training_scope == "multi_dataset"
        ),
    ):
        transcript_id = str(row.get("transcript_id", "unknown"))
        length_value = row.get("length", row.get("lengths", 0))
        try:
            length = int(length_value)
        except (TypeError, ValueError):
            continue
        if length <= 0:
            continue
        dataset = dataset_name(row.get("dataset_id"), dataset_names)
        sequence = amino_acids(row.get("codon_ids"), id_to_aa, length)
        css = css_positions(row.get("css"), length)
        for signal in SIGNALS:
            if signal not in included_signals:
                continue
            if signal not in row:
                continue
            is_shared_lbio = signal == "L_bio" and run.training_scope == "multi_dataset"
            if not is_shared_lbio and included_datasets and dataset not in included_datasets:
                continue
            if is_shared_lbio and not bool(row.get("__shared_candidate", False)):
                continue
            profile = normalize_profile(row.get(signal), row.get("mask"))
            if profile is None:
                continue
            signal_dataset = "__shared__" if is_shared_lbio else dataset
            process_profile(
                profile=profile,
                sequence=sequence,
                css=css,
                dataset=signal_dataset,
                signal=signal,
                split=prediction.split,
                motifs=motifs,
                thresholds=thresholds,
                margins=margins,
                start_window=start_window,
                start_baseline=start_baseline,
                stop_window=stop_window,
                stop_baseline=stop_baseline,
                metric_values=values,
                css_counts=css_counts,
                terminal_profiles=terminal_profiles,
                profile_codons=profile_codons,
            )

    metric_rows = summarize_metric_values(
        run,
        prediction,
        model_id,
        values,
        n_bootstrap=n_bootstrap,
        seed=bootstrap_seed,
    )
    css_rows = []
    for (dataset, signal, threshold, margin), (recovered, total) in sorted(css_counts.items()):
        lower, upper = wilson_interval(recovered, total)
        css_rows.append(
            {
                **run.metadata(),
                "model_id": model_id,
                "split": prediction.split,
                "dataset": dataset,
                "signal": signal,
                "z_threshold": threshold,
                "margin_codons": margin,
                "recovered_sites": recovered,
                "evaluable_sites": total,
                "recall": recovered / total if total else np.nan,
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )
    profile_rows = []
    for (signal, orientation), accumulator in terminal_profiles.items():
        for record in accumulator.records():
            profile_rows.append(
                {
                    **run.metadata(),
                    "model_id": model_id,
                    "split": prediction.split,
                    "signal": signal,
                    "orientation": orientation,
                    **record,
                }
            )
    return metric_rows, css_rows, profile_rows


def process_terminal_profiles_file(
    run: RunRecord,
    prediction: PredictionFile,
    *,
    dataset_names: dict[int, str],
    ramps: dict[str, Any],
    batch_rows: int,
    included_datasets: set[str] | None,
    included_signals: set[str],
) -> list[dict[str, Any]]:
    """Compute only start/stop metaprofiles without motif, CSS, or bootstrap work."""
    terminal_signals = included_signals & {"L_bio", "mu", "target"}
    if not terminal_signals:
        return []
    requested = {
        "transcript_id",
        "dataset_id",
        "mask",
        *terminal_signals,
    }
    profile_codons = int(ramps.get("profile_codons", 200))
    terminal_profiles: dict[tuple[str, str], ProfileAccumulator] = defaultdict(
        lambda: ProfileAccumulator(profile_codons)
    )
    model_id = prediction.experiment if run.training_scope == "individual_dataset" else run.run_id

    for row in iter_relevant_rows(
        prediction,
        requested,
        batch_rows=batch_rows,
        dataset_names=dataset_names,
        included_datasets=included_datasets,
        select_one_shared_row=(
            "L_bio" in terminal_signals and run.training_scope == "multi_dataset"
        ),
    ):
        dataset = dataset_name(row.get("dataset_id"), dataset_names)
        for signal in ("L_bio", "mu", "target"):
            if signal not in terminal_signals or signal not in row:
                continue
            is_shared_lbio = signal == "L_bio" and run.training_scope == "multi_dataset"
            if not is_shared_lbio and included_datasets and dataset not in included_datasets:
                continue
            if is_shared_lbio and not bool(row.get("__shared_candidate", False)):
                continue
            profile = normalize_profile(row.get(signal), row.get("mask"))
            if profile is None:
                continue
            terminal_profiles[(signal, "start")].add(profile[:profile_codons])
            terminal_profiles[(signal, "stop")].add(profile[::-1][:profile_codons])

    rows: list[dict[str, Any]] = []
    for (signal, orientation), accumulator in terminal_profiles.items():
        for record in accumulator.records():
            rows.append(
                {
                    **run.metadata(),
                    "model_id": model_id,
                    "split": prediction.split,
                    "signal": signal,
                    "orientation": orientation,
                    **record,
                }
            )
    return rows


def condition_summaries(metrics: pd.DataFrame, css: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
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
        "split",
        "signal",
    ]
    if not metrics.empty:
        for key, group in metrics.groupby(keys + ["metric"], dropna=False, sort=False):
            base = dict(zip(keys + ["metric"], key))
            rows.append(
                {
                    **base,
                    "z_threshold": np.nan,
                    "margin_codons": np.nan,
                    "n_groups": int(len(group)),
                    "n_observations": int(group["n"].sum()),
                    "value": float(group["mean"].mean()),
                    "worst_dataset": float(group["mean"].min()),
                }
            )
    if not css.empty:
        for key, group in css.groupby(keys + ["z_threshold", "margin_codons"], dropna=False, sort=False):
            base = dict(zip(keys + ["z_threshold", "margin_codons"], key))
            total = int(group["evaluable_sites"].sum())
            recovered = int(group["recovered_sites"].sum())
            rows.append(
                {
                    **base,
                    "metric": "css_recall",
                    "n_groups": int(len(group)),
                    "n_observations": total,
                    "value": recovered / total if total else np.nan,
                    "worst_dataset": float(group["recall"].min()),
                }
            )
    return pd.DataFrame(rows)


def plot_biological_summary(summary: pd.DataFrame, output: Path) -> None:
    if plt is None or summary.empty:
        return
    requested_metrics = ("motif_P", "motif_PP", "motif_PPP", "start_ramp", "stop_ramp")
    data = summary[
        (summary["signal"] == "L_bio")
        & (summary["metric"].isin(requested_metrics))
    ]
    if data.empty:
        return
    plot_conditions = data[["depth", "mass_condition"]].drop_duplicates()
    if len(plot_conditions) != 1:
        raise ValueError(
            "plot_biological_summary requires exactly one read depth and mass "
            f"condition; received {plot_conditions.to_dict('records')}."
        )
    depth = str(plot_conditions.iloc[0]["depth"])
    mass_condition = str(plot_conditions.iloc[0]["mass_condition"])
    strategies = sorted(data["strategy"].unique())
    fig, axes = plt.subplots(len(requested_metrics), len(strategies), figsize=(6.2 * len(strategies), 3.1 * len(requested_metrics)), squeeze=False)
    colors = {0.0: "#4C78A8", 0.5: "#59A14F", 1.0: "#F28E2B", 2.0: "#E15759"}
    for row_index, metric in enumerate(requested_metrics):
        for column, strategy in enumerate(strategies):
            ax = axes[row_index, column]
            subset = data[(data["metric"] == metric) & (data["strategy"] == strategy)]
            for power, line in subset.groupby("quality_rank_power"):
                line = line.sort_values("n_datasets")
                ax.plot(line["n_datasets"], line["value"], marker="o", color=colors.get(float(power)), label=f"p={float(power):g}")
            ax.axhline(0.0, color="#555555", linewidth=0.8)
            ax.set_xscale("log", base=2)
            dataset_counts = sorted(
                int(value) for value in subset["n_datasets"].dropna().unique()
            )
            ax.set_xticks(
                dataset_counts,
                labels=[str(value) for value in dataset_counts],
            )
            if strategy == "top_quality" and dataset_counts:
                quality_axis = ax.secondary_xaxis("top")
                quality_axis.set_xscale("log", base=2)
                quality_axis.set_xticks(dataset_counts)
                quality_axis.set_xticklabels(
                    [f"1–{value}" for value in dataset_counts], fontsize=8
                )
                quality_axis.set_xlabel(
                    "Quality ranks included (1 = best; rightward adds lower-ranked data)",
                    fontsize=8,
                )
                for index in range(len(dataset_counts) - 1):
                    ax.axvspan(
                        dataset_counts[index],
                        dataset_counts[index + 1],
                        color="#E15759",
                        alpha=0.018 + 0.018 * index,
                        zorder=0,
                    )
            ax.set_xlabel("Exact number of training datasets")
            ax.set_ylabel("Mean log2 enrichment")
            ax.set_title(f"{strategy.replace('_', ' ')} — {metric}")
            ax.grid(alpha=0.25)
            if row_index == 0:
                ax.legend(title="quality-rank power")
    fig.suptitle(
        f"Biological-profile summaries — read depth: {depth}; "
        f"mass mode: {mass_condition.replace('_', ' ')}",
        fontsize=14,
        y=0.998,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.982))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_css(summary: pd.DataFrame, output: Path) -> None:
    if plt is None or summary.empty:
        return
    data = summary[
        (summary["metric"] == "css_recall")
        & (summary["signal"] == "L_bio")
        & (summary["z_threshold"] == 3.0)
    ]
    if data.empty:
        return
    plot_conditions = data[["depth", "mass_condition"]].drop_duplicates()
    if len(plot_conditions) != 1:
        raise ValueError(
            "plot_css requires exactly one read depth and mass condition; "
            f"received {plot_conditions.to_dict('records')}."
        )
    depth = str(plot_conditions.iloc[0]["depth"])
    mass_condition = str(plot_conditions.iloc[0]["mass_condition"])
    strategies = sorted(data["strategy"].unique())
    fig, axes = plt.subplots(2, len(strategies), figsize=(6.2 * len(strategies), 7.5), squeeze=False)
    colors = {0.0: "#4C78A8", 0.5: "#59A14F", 1.0: "#F28E2B", 2.0: "#E15759"}
    for row_index, margin in enumerate((0, 1)):
        for column, strategy in enumerate(strategies):
            ax = axes[row_index, column]
            subset = data[(data["margin_codons"] == margin) & (data["strategy"] == strategy)]
            for power, line in subset.groupby("quality_rank_power"):
                line = line.sort_values("n_datasets")
                ax.plot(line["n_datasets"], line["value"], marker="o", color=colors.get(float(power)), label=f"p={float(power):g}")
            ax.set_xscale("log", base=2)
            dataset_counts = sorted(
                int(value) for value in subset["n_datasets"].dropna().unique()
            )
            ax.set_xticks(
                dataset_counts,
                labels=[str(value) for value in dataset_counts],
            )
            if strategy == "top_quality" and dataset_counts:
                quality_axis = ax.secondary_xaxis("top")
                quality_axis.set_xscale("log", base=2)
                quality_axis.set_xticks(dataset_counts)
                quality_axis.set_xticklabels(
                    [f"ranks 1–{value}" for value in dataset_counts],
                    fontsize=8,
                )
                quality_axis.set_xlabel(
                    "Cumulative quality ranks (1 = best; rightward adds lower-ranked datasets)",
                    fontsize=9,
                )
                for index in range(len(dataset_counts) - 1):
                    ax.axvspan(
                        dataset_counts[index],
                        dataset_counts[index + 1],
                        color="#E15759",
                        alpha=0.018 + 0.018 * index,
                        zorder=0,
                    )
            ax.set_xlabel("Exact number of training datasets")
            ax.set_ylim(bottom=0.0)
            ax.set_ylabel("CSS recall")
            ax.set_title(f"{strategy.replace('_', ' ')} — z>=3, margin +/-{margin}")
            ax.grid(alpha=0.25)
            ax.legend(title="quality-rank power")
    fig.suptitle(
        f"CSS recall — read depth: {depth}; "
        f"mass mode: {mass_condition.replace('_', ' ')}",
        fontsize=14,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_terminal_profiles(profiles: pd.DataFrame, output_dir: Path) -> None:
    if plt is None or profiles.empty:
        return
    data = profiles[profiles["split"] == "main_val"]
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_plot in output_dir.glob("*.png"):
        old_plot.unlink()
    colors = {0.0: "#4C78A8", 0.5: "#59A14F", 1.0: "#F28E2B", 2.0: "#E15759"}
    for (depth, mass_condition, strategy, n_datasets), group in data.groupby(
        ["depth", "mass_condition", "strategy", "n_datasets"],
        dropna=False,
        sort=True,
    ):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), squeeze=False)
        for row_index, orientation in enumerate(("start", "stop")):
            for column, signal in enumerate(("L_bio", "mu", "target")):
                ax = axes[row_index, column]
                subset = group[(group["orientation"] == orientation) & (group["signal"] == signal)]
                for power, line in subset.groupby("quality_rank_power"):
                    line = line.sort_values("position_codon")
                    color = colors.get(float(power), "#555555")
                    x = line["position_codon"].to_numpy(dtype=float)
                    mean = line["mean"].to_numpy(dtype=float)
                    sem = line["sem"].to_numpy(dtype=float)
                    ax.plot(x, mean, color=color, label=f"p={float(power):g}")
                    ax.fill_between(
                        x,
                        mean - sem,
                        mean + sem,
                        color=color,
                        alpha=0.12,
                        linewidth=0.0,
                    )
                ax.axhline(1.0, color="#555555", linewidth=0.9, linestyle="--")
                if orientation == "start":
                    ax.set_xlabel(
                        "Position from CDS start (codons)\n"
                        "0 = start codon (ATG); rightward follows 5′ → 3′"
                    )
                else:
                    ax.set_xlabel(
                        "Distance upstream from stop codon (codons)\n"
                        "0 = stop; 1 = preceding codon; rightward moves 3′ → 5′"
                    )
                ax.set_ylabel(
                    "Mean within-transcript–normalized value\n"
                    "(source profile ÷ its valid-codon mean)"
                )
                ax.set_title(TERMINAL_SIGNAL_TITLES[signal], fontsize=11)
                ax.grid(alpha=0.2)
                if not subset.empty:
                    n_min = int(subset["n"].min())
                    n_max = int(subset["n"].max())
                    n_text = (
                        f"contributing profiles/position: {n_min:,}"
                        if n_min == n_max
                        else f"contributing profiles/position: {n_min:,}–{n_max:,}"
                    )
                    ax.text(
                        0.985,
                        0.025,
                        n_text,
                        transform=ax.transAxes,
                        ha="right",
                        va="bottom",
                        fontsize=8,
                        color="#444444",
                        bbox={
                            "boxstyle": "round,pad=0.25",
                            "facecolor": "white",
                            "edgecolor": "#BBBBBB",
                            "alpha": 0.88,
                        },
                    )
                if row_index == 0 and column == 0:
                    ax.legend(title="gamma-centering rank power p")
        fig.suptitle(
            f"Stop- and start-aligned validation metaprofiles — "
            f"{strategy.replace('_', ' ')}, exactly {int(n_datasets)} training datasets\n"
            f"read depth: {depth}; mass mode: {str(mass_condition).replace('_', ' ')}",
            fontsize=15,
            y=0.985,
        )
        fig.text(
            0.5,
            0.018,
            TERMINAL_PROFILE_NOTE,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#303030",
            linespacing=1.35,
        )
        fig.tight_layout(rect=(0.0, 0.115, 1.0, 0.95))
        safe = f"{depth}__{mass_condition}".replace("/", "_").replace(" ", "_")
        fig.savefig(
            output_dir / f"{safe}__{strategy}_n{int(n_datasets)}.png",
            dpi=170,
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    statistics = config.get("statistics", {})
    batch_rows = args.batch_rows or int(statistics.get("batch_rows", 256))
    n_bootstrap = args.bootstrap if args.bootstrap is not None else int(statistics.get("bootstrap_replicates", 500))
    bootstrap_seed = args.bootstrap_seed if args.bootstrap_seed is not None else int(statistics.get("bootstrap_seed", 42))
    splits = set(args.split or config.get("splits", {}).get("biological", ["main_val", "css_benchmark"]))
    if args.terminal_profiles_only:
        splits = {"main_val"}
    css_config = config.get("css", {})
    thresholds = tuple(map(float, css_config.get("z_thresholds", [1, 2, 3, 4, 5])))
    margins = tuple(map(int, css_config.get("margins_codons", [0, 1])))
    motifs = tuple(map(str, config.get("motifs", ["P", "PP", "PPP"])))

    all_runs = discover_runs(args.results_root)
    selected = filter_runs(
        select_latest_runs(all_runs),
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
    included_signals = set(args.signal or SIGNALS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis_settings.json").write_text(
        json.dumps(
            {
                "results_root": str(args.results_root.resolve()),
                "config": str(args.config.resolve()),
                "strategies": args.strategy,
                "feature_presets": args.feature_preset,
                "seeds": args.seed,
                "dataset_counts": args.n_datasets,
                "excluded_dataset_counts": sorted(excluded_dataset_counts),
                "quality_powers": args.quality_power,
                "splits": sorted(splits),
                "dataset_specific_signal_filter": args.dataset,
                "signals": sorted(included_signals),
                "terminal_profiles_only": bool(args.terminal_profiles_only),
                "L_bio_is_deduplicated_across_datasets": True,
                "terminal_profile_normalization": (
                    "Each nonzero-mean source profile is divided by its own mean over "
                    "valid codons before profiles are averaged; zero-mean profiles are excluded."
                ),
                "terminal_start_axis": (
                    "position 0 is the ATG start codon; position increases 5-prime to 3-prime"
                ),
                "terminal_stop_axis": (
                    "profile order is reversed: position 0 is the stop codon, position 1 is "
                    "the preceding codon, and position increases upstream toward 5-prime"
                ),
                "terminal_target_definition": (
                    "For each dataset-transcript pair, target is the codon-wise arithmetic "
                    "mean of that dataset's biological replicas. Dataset-specific targets are "
                    "pooled as separate profiles and are not averaged across datasets."
                ),
                "plot_partition_keys": ["depth", "mass_condition"],
                "read_depths_are_never_pooled_in_plots": True,
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
    id_to_aa = load_codon_to_amino_acid()

    metric_rows: list[dict[str, Any]] = []
    css_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    for run_index, run in enumerate(selected, start=1):
        files = [file for file in run.usable_files if file.split in splits]
        print(f"[{run_index}/{len(selected)}] {run.run_id}: {len(files)} file(s)")
        for prediction in files:
            if args.terminal_profiles_only:
                metrics = []
                css = []
                profiles = process_terminal_profiles_file(
                    run,
                    prediction,
                    dataset_names=dataset_names,
                    ramps=config.get("ramps", {}),
                    batch_rows=batch_rows,
                    included_datasets=set(args.dataset) if args.dataset else None,
                    included_signals=included_signals,
                )
            else:
                metrics, css, profiles = process_file(
                    run,
                    prediction,
                    dataset_names=dataset_names,
                    id_to_aa=id_to_aa,
                    motifs=motifs,
                    thresholds=thresholds,
                    margins=margins,
                    ramps=config.get("ramps", {}),
                    batch_rows=batch_rows,
                    n_bootstrap=n_bootstrap,
                    bootstrap_seed=bootstrap_seed,
                    included_datasets=set(args.dataset) if args.dataset else None,
                    included_signals=included_signals,
                )
            metric_rows.extend(metrics)
            css_rows.extend(css)
            profile_rows.extend(profiles)

    metrics_frame = pd.DataFrame(metric_rows)
    css_frame = pd.DataFrame(css_rows)
    profiles_frame = pd.DataFrame(profile_rows)
    if args.terminal_profiles_only:
        profiles_frame.to_csv(args.output_dir / "terminal_metaprofiles.csv", index=False)
        plot_terminal_profiles(profiles_frame, args.output_dir / "terminal_profiles")
        print(f"Saved terminal biological-quality profiles to {args.output_dir}")
        return
    metrics_frame.to_csv(args.output_dir / "motif_and_ramp_by_dataset.csv", index=False)
    css_frame.to_csv(args.output_dir / "css_recall_by_dataset.csv", index=False)
    profiles_frame.to_csv(args.output_dir / "terminal_metaprofiles.csv", index=False)
    summary = condition_summaries(metrics_frame, css_frame)
    summary.to_csv(args.output_dir / "biological_quality_by_condition.csv", index=False)
    (args.output_dir / "L_bio_motif_and_ramp_summary.png").unlink(missing_ok=True)
    (args.output_dir / "L_bio_css_recall_z3.png").unlink(missing_ok=True)
    plot_dir = args.output_dir / "plots_by_depth"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for old_plot in plot_dir.glob("*.png"):
        old_plot.unlink()
    for (depth, mass_condition), group in summary.groupby(
        ["depth", "mass_condition"], dropna=False, sort=True
    ):
        safe = f"{depth}__{mass_condition}".replace("/", "_").replace(" ", "_")
        plot_biological_summary(
            group,
            plot_dir / f"L_bio_motif_and_ramp_summary__{safe}.png",
        )
        plot_css(group, plot_dir / f"L_bio_css_recall_z3__{safe}.png")
    plot_terminal_profiles(profiles_frame, args.output_dir / "terminal_profiles")
    print(f"Saved biological-quality analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
