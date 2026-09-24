#!/usr/bin/env python3
"""Plot matched, profile-level examples of strong synthetic gamma misses.

This is a post-processing utility.  It selects representative coordinates from
the persisted gamma-compensation worker Parquets, reads only the selected rows
from existing best-validation-loss prediction and observation Parquets, and
never loads a checkpoint or runs model inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyses.analyze_synthetic_gamma_compensation import (
    CANONICAL_BIASES,
    DEFAULT_BIAS_ROOT,
    DEFAULT_OUTPUT,
    EPS,
    _historical_config,
    _matched_mass_pairs,
    _mean_one_positive,
)
from analyses.analyze_synthetic_gamma_recovery import (
    _base_bias_name,
    joint_log_gamma_gauge,
)
from analyses.analyze_synthetic_recovery import (
    BOUNDARY_TRIM_CODONS,
    REPOSITORY_ROOT,
    _encoding_from_config,
    _load_latent_truth,
    _read_yaml,
    cds_interior_mask,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from tqdm.auto import tqdm


def _representative_sites(conserved: pd.DataFrame, free: pd.DataFrame) -> pd.DataFrame:
    for label, frame in (("mass conserved", conserved), ("mass free", free)):
        if "is_interior" not in frame.columns:
            raise ValueError(
                f"{label} site diagnostics predate the CDS-interior mask; "
                "regenerate gamma compensation diagnostics first."
            )
    conserved = conserved.loc[conserved["is_interior"]].copy()
    free = free.loc[free["is_interior"]].copy()
    keys = ["bias_name", "dataset", "transcript_id", "position"]
    columns = [
        *keys, "g_true", "g_learned", "gamma_underestimate", "gamma_abs_error",
        "L_true", "L_learned", "L_log_error", "alpha", "log_alpha",
        "alpha_log_excess", "consensus", "replicate_cv", "is_missed_strong",
        "joint_compensation",
    ]
    paired = conserved[columns].merge(
        free[columns], on=keys, suffixes=("_mass_conserved", "_mass_free"),
        validate="one_to_one",
    )
    paired = paired[
        paired["is_missed_strong_mass_conserved"]
        | paired["is_missed_strong_mass_free"]
    ].copy()
    paired["missed_in_both"] = (
        paired["is_missed_strong_mass_conserved"]
        & paired["is_missed_strong_mass_free"]
    )
    paired["robust_underestimate"] = paired[
        ["gamma_underestimate_mass_conserved", "gamma_underestimate_mass_free"]
    ].min(axis=1)
    paired["selection_score"] = (
        100.0 * paired["missed_in_both"].astype(float)
        + paired["robust_underestimate"]
        + 0.05 * paired[
            ["alpha_log_excess_mass_conserved", "alpha_log_excess_mass_free"]
        ].max(axis=1).clip(lower=0.0)
    )
    order = {name: index for index, name in enumerate(CANONICAL_BIASES)}
    chosen = (
        paired.sort_values(["bias_name", "selection_score"], ascending=[True, False])
        .drop_duplicates("bias_name", keep="first")
        .sort_values("bias_name", key=lambda value: value.map(order))
        .reset_index(drop=True)
    )
    return chosen


def _dataset_id_map(config: dict[str, Any]) -> dict[int, str]:
    datasets = [str(value) for value in config["experiment"]["dataset"]]
    mapping = _encoding_from_config(config, REPOSITORY_ROOT)
    if not set(datasets).issubset(mapping.values()):
        return {index: dataset for index, dataset in enumerate(datasets)}
    return mapping


def _prediction_rows(
    path: Path,
    transcript_ids: set[str],
    dataset_id_to_name: dict[int, str],
) -> dict[tuple[str, str], dict[str, Any]]:
    columns = [
        "transcript_id", "dataset_id", "length", "L_bio", "log_gamma",
        "gamma_centering_reliability", "log_sigma", "mu", "target",
    ]
    # The prediction parquet has one large row group. Reading it in tiny Python
    # batches repeatedly decoded every long list column. Let Arrow apply the ID
    # predicate and decode the selected transcript rows in one vectorized pass.
    table = pq.read_table(
        path, columns=columns,
        filters=[("transcript_id", "in", sorted(transcript_ids))],
    )
    values = table.to_pydict()
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for index in range(table.num_rows):
        transcript_id = str(values["transcript_id"][index])
        dataset = dataset_id_to_name[int(values["dataset_id"][index])]
        length = int(values["length"][index])
        result[(transcript_id, dataset)] = {
            "length": length,
            "L_bio": np.asarray(values["L_bio"][index], dtype=float).reshape(-1)[:length],
            "log_gamma": np.asarray(values["log_gamma"][index], dtype=float).reshape(-1)[:length],
            "reliability": np.asarray(values["gamma_centering_reliability"][index], dtype=float).reshape(-1)[:length],
            "log_sigma": np.asarray(values["log_sigma"][index], dtype=float).reshape(-1)[:length],
            "mu": np.asarray(values["mu"][index], dtype=float).reshape(-1)[:length],
            "target": np.asarray(values["target"][index], dtype=float).reshape(-1)[:length],
        }
    return result


def _targeted_bias_profiles(
    dataset: str,
    bias_root: Path,
    transcript_ids: set[str],
    expected_lengths: dict[str, int],
) -> dict[str, np.ndarray]:
    path = bias_root / f"{_base_bias_name(dataset)}_compendium_added_bias_only.parquet"
    table = pq.read_table(
        path, columns=["sample", "transcript_id", "added_bias"],
        filters=[("transcript_id", "in", sorted(transcript_ids))],
    )
    profiles: dict[str, np.ndarray] = {}
    for row in table.to_pylist():
        transcript_id = str(row["transcript_id"])
        if not str(row["sample"]).endswith("_mean"):
            continue
        profile = np.asarray(row["added_bias"], dtype=float).reshape(-1)
        if profile.size != expected_lengths[transcript_id]:
            raise ValueError(f"Bias length mismatch for {dataset}/{transcript_id}.")
        profiles[transcript_id] = profile
    missing = transcript_ids - profiles.keys()
    if missing:
        raise KeyError(f"Bias truth missing {len(missing)} selected transcript(s) for {dataset}.")
    return profiles


def _targeted_observations(
    path: Path,
    transcript_ids: set[str],
    expected_lengths: dict[str, int],
) -> dict[str, dict[str, Any]]:
    columns = ["id", "ribo", "ribo_cds_replicas", "replica_ids"]
    table = pq.read_table(path, columns=columns, filters=[("id", "in", sorted(transcript_ids))])
    result: dict[str, dict[str, Any]] = {}
    for row in table.to_pylist():
        transcript_id = str(row["id"])
        length = expected_lengths[transcript_id]
        raw_consensus = np.asarray(row["ribo"], dtype=float).reshape(-1)
        consensus = np.full(length, np.nan, dtype=float)
        consensus[: min(length, raw_consensus.size)] = raw_consensus[:length]
        replicas = []
        for value in row["ribo_cds_replicas"] or []:
            raw_replica = np.asarray(value, dtype=float).reshape(-1)
            replica = np.full(length, np.nan, dtype=float)
            replica[: min(length, raw_replica.size)] = raw_replica[:length]
            replicas.append(replica)
        result[transcript_id] = {
            "consensus": consensus,
            "replicas": np.stack(replicas) if replicas else None,
            "replica_ids": list(row["replica_ids"] or []),
        }
    missing = transcript_ids - result.keys()
    if missing:
        raise KeyError(f"Observation parquet {path} is missing {len(missing)} selected transcript(s).")
    return result


def _resolve_dataset_path(config: dict[str, Any], dataset: str) -> Path:
    raw = Path(str(config["dataset_config"]["dataset_path"][dataset]))
    return raw.resolve() if raw.is_absolute() else (REPOSITORY_ROOT / raw).resolve()


def _build_run_profiles(
    config: dict[str, Any],
    prediction_path: Path,
    selected: pd.DataFrame,
    truth: dict[str, np.ndarray],
    bias_root: Path,
    bias_cache: dict[str, dict[str, np.ndarray]],
    *,
    load_observations: bool,
    progress_label: str,
) -> tuple[dict[tuple[str, str], dict[str, np.ndarray]], dict[tuple[str, str], dict[str, Any]]]:
    datasets = [str(value) for value in config["experiment"]["dataset"]]
    transcript_ids = set(selected["transcript_id"].astype(str))
    rows = _prediction_rows(prediction_path, transcript_ids, _dataset_id_map(config))
    expected_lengths = {identifier: int(truth[identifier].size) for identifier in transcript_ids}
    # Synthetic predictions may retain one masked terminal tensor position.
    # The established recovery coordinate system is the deterministic truth
    # length, so crop every prediction vector to that same P-site axis.
    for (transcript_id, _dataset), row in rows.items():
        length = expected_lengths[transcript_id]
        for column in ("L_bio", "log_gamma", "reliability", "log_sigma", "mu", "target"):
            row[column] = row[column][:length]
        row["length"] = length
    for dataset in tqdm(datasets, desc=f"bias truth: {progress_label}", unit="dataset", leave=False):
        if dataset not in bias_cache:
            bias_cache[dataset] = _targeted_bias_profiles(
                dataset, bias_root, transcript_ids, expected_lengths,
            )
    programmed = {dataset: bias_cache[dataset] for dataset in datasets}
    profiles: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for transcript_id in transcript_ids:
        participants = [dataset for dataset in datasets if (transcript_id, dataset) in rows]
        if len(participants) < 2:
            continue
        weights = np.asarray([
            np.median(rows[(transcript_id, dataset)]["reliability"][rows[(transcript_id, dataset)]["reliability"] > 0])
            for dataset in participants
        ])
        raw_learned = np.stack(
            [rows[(transcript_id, dataset)]["log_gamma"] for dataset in participants]
        )
        raw_true = np.stack(
            [np.log1p(programmed[dataset][transcript_id]) for dataset in participants]
        )
        interior_mask = cds_interior_mask(raw_true.shape[1])
        learned = joint_log_gamma_gauge(raw_learned, weights)
        true = joint_log_gamma_gauge(raw_true, weights)
        learned[:, interior_mask] = joint_log_gamma_gauge(
            raw_learned, weights, position_mask=interior_mask
        )
        true[:, interior_mask] = joint_log_gamma_gauge(
            raw_true, weights, position_mask=interior_mask
        )
        for index, dataset in enumerate(participants):
            row = rows[(transcript_id, dataset)]
            loss = config.get("loss", {})
            alpha = np.exp(np.clip(
                row["log_sigma"],
                float(loss.get("nb_log_alpha_min", -5.0)),
                float(loss.get("nb_log_alpha_max", 3.0)),
            ))
            profiles[(transcript_id, dataset)] = {
                "K": truth[transcript_id].astype(float),
                "L": _mean_one_positive(row["L_bio"], row["length"], label=f"L {transcript_id}/{dataset}"),
                "g_true": true[index], "g_learned": learned[index],
                "alpha": alpha, "mu": row["mu"], "target": row["target"],
            }
    observations: dict[tuple[str, str], dict[str, Any]] = {}
    if load_observations:
        for dataset, part in tqdm(
            selected.groupby("dataset", sort=False),
            total=selected["dataset"].nunique(), desc=f"observations: {progress_label}",
            unit="dataset", leave=False,
        ):
            ids = set(part["transcript_id"].astype(str))
            loaded = _targeted_observations(
                _resolve_dataset_path(config, str(dataset)), ids,
                {identifier: expected_lengths[identifier] for identifier in ids},
            )
            observations.update({(identifier, str(dataset)): value for identifier, value in loaded.items()})
    return profiles, observations


def _plot_example(
    row: Any,
    conserved: dict[str, np.ndarray],
    free: dict[str, np.ndarray],
    observation: dict[str, Any],
    output: Path,
    radius: int,
) -> None:
    anchor = int(row.position)
    length = len(conserved["K"])
    start = max(BOUNDARY_TRIM_CODONS, anchor - radius)
    stop = min(length - BOUNDARY_TRIM_CODONS, anchor + radius + 1)
    if not start <= anchor < stop:
        raise ValueError(
            f"Selected position {anchor} is outside the CDS-interior domain."
        )
    x = np.arange(start, stop)
    fig, axes = plt.subplots(5, 2, figsize=(16, 15), sharex=True, constrained_layout=True)
    for column, (condition, profile) in enumerate((("mass conserved", conserved), ("mass free", free))):
        axes[0, column].plot(x, profile["K"][start:stop], color="#111827", lw=1.7, label="K true")
        axes[0, column].plot(x, profile["L"][start:stop], color="#2563eb", lw=1.4, label="L learned")
        axes[0, column].set_ylabel("mean-one load")

        axes[1, column].plot(x, profile["g_true"][start:stop], color="#111827", lw=1.7, label="g true")
        axes[1, column].plot(x, profile["g_learned"][start:stop], color="#dc2626", lw=1.4, label="g learned")
        strong = profile["g_true"][start:stop] >= 1.0
        axes[1, column].scatter(x[strong], profile["g_true"][start:stop][strong], s=9, color="#f59e0b", label="true strong")
        axes[1, column].axhline(0, color="black", lw=.7)
        axes[1, column].set_ylabel("log gamma")

        axes[2, column].plot(x, np.log(np.maximum(profile["alpha"][start:stop], EPS)), color="#7c3aed", lw=1.5, label="log alpha")
        axes[2, column].set_ylabel("log alpha")

        axes[3, column].plot(x, np.log1p(profile["mu"][start:stop]), color="#2563eb", lw=1.5, label="log1p(mu)")
        axes[3, column].plot(x, np.log1p(observation["consensus"][start:stop]), color="#111827", lw=1.2, label="log1p(consensus)")
        replicas = observation["replicas"]
        if replicas is not None:
            for replica_index, replica in enumerate(replicas):
                replica_id = observation["replica_ids"][replica_index] if replica_index < len(observation["replica_ids"]) else f"replica {replica_index + 1}"
                axes[3, column].plot(x, np.log1p(replica[start:stop]), lw=.8, alpha=.55, label=str(replica_id))
        axes[3, column].set_ylabel("log1p counts")

        variance = profile["mu"] + profile["alpha"] * profile["mu"] ** 2
        residual = (observation["consensus"] - profile["mu"]) / np.sqrt(np.maximum(variance, EPS))
        axes[4, column].plot(x, residual[start:stop], color="#059669", lw=1.2)
        axes[4, column].axhline(0, color="black", lw=.7)
        axes[4, column].axhline(2, color="#9ca3af", lw=.7, ls="--")
        axes[4, column].axhline(-2, color="#9ca3af", lw=.7, ls="--")
        axes[4, column].set_ylabel("NB-standardized\nconsensus residual")
        axes[4, column].set_xlabel("codon position")

        for axis in axes[:, column]:
            axis.axvline(anchor, color="#dc2626", lw=1.0, ls="--")
            axis.grid(alpha=.18)
        axes[0, column].set_title(condition, fontweight="bold")
        for axis in axes[:4, column]:
            axis.legend(frameon=False, fontsize=7, ncol=2, loc="best")

    for row_index in range(5):
        limits = [axes[row_index, column].get_ylim() for column in range(2)]
        axes[row_index, 0].set_ylim(min(value[0] for value in limits), max(value[1] for value in limits))
        axes[row_index, 1].set_ylim(axes[row_index, 0].get_ylim())
    fig.suptitle(
        f"Representative strong gamma miss: {str(row.bias_name).replace('artificial_bias_', '')}\n"
        f"{row.transcript_id}, codon {anchor}; red dashed line is the selected missed site",
        fontweight="bold",
    )
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bias-root", type=Path, default=DEFAULT_BIAS_ROOT)
    parser.add_argument("--radius", type=int, default=75)
    parser.add_argument(
        "--overwrite-figures", action="store_true",
        help="Regenerate representative figures that already have PNG outputs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.radius < 1:
        raise ValueError("--radius must be positive.")
    analysis = args.analysis_dir.expanduser().resolve()
    run_summary = pd.read_csv(analysis / "run_summary.tsv", sep="\t")
    primary = pd.read_csv(analysis / "primary_runs.tsv", sep="\t")
    matched = _matched_mass_pairs(primary)
    if matched.empty:
        raise ValueError("No fair mass-conserved/mass-free run pair is available.")
    figure_dir = analysis / "figures" / "representative_missed_profiles"
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[pd.DataFrame] = []

    for pair_index, pair in matched.iterrows():
        conserved_run, free_run = pair["mass_conserved_run"], pair["mass_free_run"]
        missed = pd.read_csv(analysis / "missed_strong_site_diagnostics.tsv.gz", sep="\t")
        conserved_candidates = missed[missed["run"] == conserved_run]
        free_candidates = missed[missed["run"] == free_run]
        selected = _representative_sites(conserved_candidates, free_candidates)
        if selected.empty:
            continue
        conserved_summary = run_summary.loc[run_summary["run"] == conserved_run].iloc[0]
        free_summary = run_summary.loc[run_summary["run"] == free_run].iloc[0]
        conserved_dir = Path(str(conserved_summary.prediction_path)).parents[3]
        free_dir = Path(str(free_summary.prediction_path)).parents[3]
        conserved_config = _read_yaml(_historical_config(conserved_dir))
        free_config = _read_yaml(_historical_config(free_dir))
        truth = _load_latent_truth(
            REPOSITORY_ROOT / "Datasets/Synthetic_data/artificial_ground_truth_kinetics_target_mean_one.parquet"
        )
        bias_cache: dict[str, dict[str, np.ndarray]] = {}
        conserved_profiles, observations = _build_run_profiles(
            conserved_config, Path(str(conserved_summary.prediction_path)), selected,
            truth, args.bias_root.expanduser().resolve(), bias_cache,
            load_observations=True, progress_label="mass conserved",
        )
        free_profiles, _ = _build_run_profiles(
            free_config, Path(str(free_summary.prediction_path)), selected,
            truth, args.bias_root.expanduser().resolve(), bias_cache,
            load_observations=False, progress_label="mass free",
        )
        for row in tqdm(selected.itertuples(index=False), total=len(selected), desc=f"profile figures: {pair.depth}", unit="bias"):
            key = (str(row.transcript_id), str(row.dataset))
            figure_path = figure_dir / f"{pair.depth}__{row.bias_name}.png"
            if figure_path.is_file() and not args.overwrite_figures:
                continue
            _plot_example(
                row, conserved_profiles[key], free_profiles[key], observations[key],
                figure_path, args.radius,
            )
        selected.insert(0, "pair_index", pair_index)
        selected.insert(1, "depth", pair["depth"])
        selected.insert(2, "dataset_count", pair["dataset_count"])
        output_rows.append(selected)

    result = pd.concat(output_rows, ignore_index=True) if output_rows else pd.DataFrame()
    result.to_csv(analysis / "representative_missed_sites.tsv", sep="\t", index=False)
    report_lines = [
        "# Representative strong gamma misses", "",
        "Each figure compares the same transcript–dataset–codon coordinate between a strictly matched mass-conserved and mass-free run. Examples prefer sites missed in both modes, then rank by the smaller of the two gamma-underestimation errors. Selection therefore illustrates robust misses rather than estimating their population frequency.", "",
        "The first row compares deterministic `K` with learned `L`; the second compares gauge-fixed true and learned log-gamma; the third shows learned log-alpha; the fourth shows predicted mean, consensus, and every raw replicate on a `log1p` display; the fifth shows the consensus residual divided by the learned NB2 standard deviation. If a visibly wrong mean has a modest standardized residual only because alpha is large, that is direct visual evidence of dispersion accommodation.", "",
        f"Generated {len(result)} figures from existing prediction/observation Parquets without checkpoint loading or inference.", "",
        "See `representative_missed_sites.tsv` for the exact selected coordinates and `figures/representative_missed_profiles/` for PNG figures.",
    ]
    (analysis / "REPRESENTATIVE_MISSES.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(result)} representative missed-site figures to {figure_dir}")


if __name__ == "__main__":
    main()
