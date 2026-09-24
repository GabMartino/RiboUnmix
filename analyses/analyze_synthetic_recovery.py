#!/usr/bin/env python3
"""Summarize recovery of the latent synthetic kinetics across experiment runs.

The synthetic model has two different profile-level outputs with deliberately
different meanings:

* ``L_bio`` is the shared, dataset-independent signal and is compared with the
  deterministic mean-one kinetic ground truth.
* ``mu`` is the reconstructed dataset observation, so it is expected to retain
  the artificial dataset bias.  It is reported only as a factorization
  diagnostic, not as the primary recovery score.

Only immediate child directories whose names start with
``riboai_synthetic_within_`` are analyzed by default. Run selection never uses
the synthetic ground truth. By default this utility
uses the same ``val_mu_pcc`` maximum as the prediction checkpoint selected by
the training entry point. Profile errors preserve the historical full-profile
mean-one normalization, then use only 0-based P-site coordinates
``5 <= i < L-5``. They are averaged equally across validation transcripts.
The first/last five codons remain available only as explicitly labelled
historical/boundary diagnostics because their programmed 30-nt RPF features
can require UTR sequence absent from the CDS-only model input.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto import event_pb2

matplotlib.use("Agg")
from matplotlib import pyplot as plt


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPOSITORY_ROOT / "results" / "riboai_synthetic_experiments"
DEFAULT_RUN_PREFIX = "riboai_synthetic_within_"
DEFAULT_OUTPUT_DIRECTORY_NAME = "recovery_report_within"
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "analyses"
    / "artifacts"
    / "synthetic"
    / "shared_profile_recovery"
)
DEFAULT_LATENT_TRUTH = Path(
    "Datasets/Synthetic_data/artificial_ground_truth_kinetics_target_mean_one.parquet"
)
BOUNDARY_TRIM_CODONS = 5
EVALUATION_DOMAIN = "cds_interior"
SCALAR_TAGS = (
    "val_loss",
    "val_mu_pcc",
    "val/synthetic_ground_truth/latent_mse",
    "val/synthetic_ground_truth/latent_pcc",
    "val/synthetic_ground_truth/transcripts",
)


@dataclass(frozen=True)
class ProfileMetrics:
    pcc: float
    mse: float
    rmse: float
    mae: float


def cds_interior_mask(
    profile_length: int,
    boundary_trim_codons: int = BOUNDARY_TRIM_CODONS,
) -> np.ndarray:
    """Return the fair CDS-only evaluation mask for 0-based P-site codons.

    Synthetic profiles already omit the terminal stop boundary.  Therefore a
    profile of length ``L`` represents physical modeled coordinates
    ``0, ..., L-1`` and the observable interior is ``5 <= i < L-5``.  Short
    profiles with no such coordinate safely return an all-false mask.
    """
    length = int(profile_length)
    trim = int(boundary_trim_codons)
    if length < 0:
        raise ValueError("profile_length must be non-negative.")
    if trim < 0:
        raise ValueError("boundary_trim_codons must be non-negative.")
    positions = np.arange(length, dtype=np.int64)
    return (positions >= trim) & (positions < length - trim)


def evaluation_domain_metadata() -> dict[str, Any]:
    """Metadata attached to tables whose primary metrics use the interior."""
    return {
        "boundary_trim_codons": BOUNDARY_TRIM_CODONS,
        "evaluation_domain": EVALUATION_DOMAIN,
        "boundary_positions_excluded": True,
    }


def discover_run_directories(results_root: Path, run_prefix: str) -> list[Path]:
    """Return matching immediate run directories in deterministic name order."""
    if not results_root.is_dir():
        raise FileNotFoundError(f"Synthetic results root does not exist: {results_root}")
    if not run_prefix:
        raise ValueError("run_prefix must be non-empty.")
    return sorted(
        path
        for path in results_root.iterdir()
        if path.is_dir() and path.name.startswith(run_prefix)
    )


def synthetic_mass_conservation(
    config: dict[str, Any], run_name: str = ""
) -> tuple[bool, str]:
    """Return the resolved mass-conservation setting and a plot/table label."""
    value = config.get("model", {}).get("mass_conservation")
    enabled = ("massfree" not in run_name.lower()) if value is None else bool(value)
    return enabled, "mass_conserved" if enabled else "mass_free"


def _normalize_mean_one(profile: Any, *, label: str) -> np.ndarray:
    values = np.asarray(profile, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError(f"{label} must contain at least two positions.")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains a non-finite value.")
    if bool((values < 0.0).any()):
        raise ValueError(f"{label} contains a negative value.")
    mean = float(values.mean())
    if not math.isfinite(mean) or mean <= 0.0:
        raise ValueError(f"{label} must have a finite positive mean.")
    return values / mean


def profile_recovery_metrics(
    prediction: Any,
    reference: Any,
    *,
    label: str = "profile",
    position_mask: Any | None = None,
) -> ProfileMetrics:
    """Compare profile shapes after full-profile mean-one normalization.

    When ``position_mask`` is supplied, the same physical coordinates are
    selected from prediction and truth *after* preserving the historical
    whole-CDS normalization.  An empty (very short transcript) evaluation
    domain returns NaNs rather than fabricating a recovery score.
    """
    pred = _normalize_mean_one(prediction, label=f"{label} prediction")
    truth = _normalize_mean_one(reference, label=f"{label} reference")
    if pred.size != truth.size:
        raise ValueError(
            f"{label} length mismatch: prediction={pred.size}, reference={truth.size}."
        )
    if position_mask is not None:
        selected = np.asarray(position_mask, dtype=bool).reshape(-1)
        if selected.shape != pred.shape:
            raise ValueError(
                f"{label} position-mask mismatch: mask={selected.size}, "
                f"profiles={pred.size}."
            )
        if not bool(selected.any()):
            return ProfileMetrics(*([float("nan")] * 4))
        pred = pred[selected]
        truth = truth[selected]
    residual = pred - truth
    mse = float(np.mean(residual**2))
    pred_centered = pred - pred.mean()
    truth_centered = truth - truth.mean()
    denominator = float(
        np.sqrt(np.sum(pred_centered**2) * np.sum(truth_centered**2))
    )
    pcc = (
        float(np.sum(pred_centered * truth_centered) / denominator)
        if denominator > 0.0
        else float("nan")
    )
    return ProfileMetrics(
        pcc=pcc,
        mse=mse,
        rmse=math.sqrt(mse),
        mae=float(np.mean(np.abs(residual))),
    )


def _load_latent_truth(path: Path) -> dict[str, np.ndarray]:
    frame = pd.read_parquet(path, columns=["transcript_id", "rib_profile"])
    truth: dict[str, np.ndarray] = {}
    for row in frame.itertuples(index=False):
        transcript_id = str(row.transcript_id)
        if transcript_id in truth:
            raise ValueError(f"Duplicate latent transcript ID: {transcript_id}")
        truth[transcript_id] = _normalize_mean_one(
            row.rib_profile,
            label=f"latent ground truth {transcript_id}",
        ).astype(np.float32)
    if not truth:
        raise ValueError(f"No latent profiles found in {path}.")
    return truth


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return value


def _read_scalar_history(run_dir: Path) -> dict[str, dict[int, tuple[float, float]]]:
    """Read only scalar summaries without retaining large TensorBoard images."""
    histories: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
    for event_path in sorted(run_dir.rglob("events.out.tfevents.*")):
        for raw_event in RawEventFileLoader(str(event_path)).Load():
            event = event_pb2.Event.FromString(raw_event)
            if not event.HasField("summary"):
                continue
            for value in event.summary.value:
                if value.tag not in SCALAR_TAGS or not value.HasField("simple_value"):
                    continue
                step = int(event.step)
                previous = histories[value.tag].get(step)
                current = (float(event.wall_time), float(value.simple_value))
                if previous is None or current[0] >= previous[0]:
                    histories[value.tag][step] = current
            del event
    return histories


def _select_step(
    histories: dict[str, dict[int, tuple[float, float]]],
    selection_metric: str,
) -> int:
    history = histories.get(selection_metric, {})
    if not history:
        raise ValueError(f"TensorBoard history has no {selection_metric!r} values.")
    if "pcc" in selection_metric.lower():
        # ModelCheckpoint keeps the first strict maximum when later values tie.
        return max(history, key=lambda step: (history[step][1], -step))
    return min(history, key=lambda step: (history[step][1], step))


def _scalar_at(
    histories: dict[str, dict[int, tuple[float, float]]],
    tag: str,
    step: int,
) -> float:
    try:
        return float(histories[tag][step][1])
    except KeyError as exc:
        raise ValueError(f"Missing TensorBoard scalar {tag!r} at step {step}.") from exc


def _validation_manifest(run_dir: Path) -> tuple[list[str], str]:
    manifests = sorted(run_dir.rglob("split_manifest_*.json"))
    if len(manifests) != 1:
        raise ValueError(
            f"Expected one split manifest below {run_dir}, found {len(manifests)}."
        )
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    ids = [str(value) for value in payload.get("validation_ids", [])]
    digest = hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()[:12]
    return ids, digest


def _encoding_from_config(config: dict[str, Any], repository_root: Path) -> dict[int, str]:
    raw_path = config.get("paths", {}).get("encodings", {}).get("datasets")
    path = None if raw_path is None else Path(str(raw_path)).expanduser()
    if path is not None and not path.is_absolute():
        path = repository_root / path

    # The cumulative inter-bias launcher creates a temporary encoding whose
    # IDs follow the fixed 30-dataset split universe.  That temporary file is
    # intentionally deleted when the training task exits, but the complete
    # ordered universe is retained in the resolved config.  Reconstruct the
    # exact mapping from that list; falling back to the ordinary 11-dataset
    # synthetic encoding would silently attach the wrong bias/depth name to
    # every prediction row.
    if (
        (path is None or not path.is_file())
        and "synthetic_cross_depth_dataset_encoding" in str(raw_path)
    ):
        universe = config.get("split", {}).get("master_dataset_universe")
        if not isinstance(universe, (list, tuple)) or not universe:
            raise ValueError(
                "The temporary cross-depth dataset encoding is unavailable and "
                "the resolved split.master_dataset_universe does not contain "
                "the ordered dataset names needed to reconstruct it."
            )
        names = [str(value) for value in universe]
        if len(names) != len(set(names)):
            raise ValueError(
                "split.master_dataset_universe contains duplicate cross-depth "
                "dataset names, so dataset IDs cannot be reconstructed safely."
            )
        return {dataset_id: name for dataset_id, name in enumerate(names)}

    # Completed synthetic runs can retain a temporary Hydra-resolved absolute
    # path (for example /tmp/.../synthetic_cross_depth_dataset_encoding.yaml)
    # that no longer exists after the run.  The repository copy is canonical
    # for ordinary within-depth runs; use it only for those synthetic configs.
    if (
        (path is None or not path.is_file())
        and (
            "synthetic" in str(raw_path).lower()
            or "synthetic" in str(config.get("dataset_config", {})).lower()
        )
    ):
        path = repository_root / "Datasets" / "encodings" / "synthetic_dataset_encoding.yaml"
    if path is None:
        return {}
    if not path.is_file():
        return {}
    mapping = _read_yaml(path)
    return {int(dataset_id): str(name) for name, dataset_id in mapping.items()}


def synthetic_depth_label(config: dict[str, Any]) -> str:
    """Return the depth represented by the selected synthetic dataset panel.

    Inter-bias experiments deliberately combine the 0.25, 2, and 20 reads per
    codon files, while the resolved Hydra ``dataset_config._name_`` reflects
    only the config group used to construct the paths.  Dataset names are the
    authoritative source for the panel depth label.
    """
    datasets = [
        str(value) for value in config.get("experiment", {}).get("dataset", [])
    ]
    suffixes = ("_0p25_per_codon", "_2_per_codon", "_20_per_codon")
    depths = sorted(
        {
            suffix.removeprefix("_").removesuffix("_per_codon")
            for dataset in datasets
            for suffix in suffixes
            if dataset.endswith(suffix)
        }
    )
    if len(depths) == 1:
        return f"{depths[0]}_per_codon"
    if len(depths) > 1:
        return "cross_depth"
    name = str(config.get("dataset_config", {}).get("_name_", "unknown"))
    return name.removeprefix("synthetic_")


def _find_prediction(
    run_dir: Path,
    checkpoint_variant: str = "best_pcc",
) -> Path | None:
    """Find one prediction artifact, preferring an explicit checkpoint variant.

    Historical runs contain one unsuffixed prediction generated from the
    best-PCC checkpoint. New runs contain independent best-PCC and
    best-validation-loss predictions. Existing synthetic analyses retain their
    historical best-PCC meaning unless a caller explicitly requests otherwise.
    """
    explicit = sorted(
        run_dir.rglob(
            f"predictions_main_val_{checkpoint_variant}_*.parquet"
        )
    )
    if len(explicit) > 1:
        raise ValueError(
            f"Multiple {checkpoint_variant} prediction parquets found below {run_dir}."
        )
    if explicit:
        return explicit[0]
    if checkpoint_variant != "best_pcc":
        return None

    legacy = sorted(run_dir.rglob("predictions_main_val_*.parquet"))
    legacy = [
        path
        for path in legacy
        if "predictions_main_val_best_val_loss_" not in path.name
        and "predictions_main_val_best_pcc_" not in path.name
    ]
    if len(legacy) > 1:
        raise ValueError(f"Multiple legacy prediction parquets found below {run_dir}.")
    return legacy[0] if legacy else None


def _bootstrap_mean_interval(
    values: Iterable[float],
    *,
    seed: int,
    replicates: int = 2_000,
) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    # Chunking avoids materializing a replicates x transcripts index matrix.
    for index in range(replicates):
        means[index] = float(array[rng.integers(0, array.size, array.size)].mean())
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _metrics_dict(metrics: ProfileMetrics, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_pcc": metrics.pcc,
        f"{prefix}_mse": metrics.mse,
        f"{prefix}_rmse": metrics.rmse,
        f"{prefix}_mae": metrics.mae,
    }


def _analyze_prediction(
    prediction_path: Path,
    *,
    truth: dict[str, np.ndarray],
    dataset_id_to_name: dict[int, str],
    run_label: str,
    depth: str,
    dataset_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    required_columns = (
        "transcript_id",
        "dataset_id",
        "length",
        "L_bio",
        "target",
        "mu",
    )
    parquet_file = pq.ParquetFile(prediction_path)
    missing = sorted(set(required_columns) - set(parquet_file.schema_arrow.names))
    if missing:
        raise ValueError(f"{prediction_path} is missing columns: {missing}")

    first_l_bio: dict[str, np.ndarray] = {}
    transcript_rows: dict[str, dict[str, Any]] = {}
    pair_rows: list[dict[str, Any]] = []
    maximum_duplicate_difference = 0.0

    for batch in parquet_file.iter_batches(batch_size=16, columns=list(required_columns)):
        columns = batch.to_pydict()
        for row_index in range(batch.num_rows):
            transcript_id = str(columns["transcript_id"][row_index])
            reference = truth.get(transcript_id)
            if reference is None:
                raise KeyError(
                    f"Prediction transcript {transcript_id} is absent from latent ground truth."
                )
            expected_length = int(reference.size)
            sequence_length = int(columns["length"][row_index])
            if sequence_length < expected_length:
                raise ValueError(
                    f"Prediction for {transcript_id} is shorter than latent truth: "
                    f"sequence={sequence_length}, latent={expected_length}."
                )

            profiles: dict[str, np.ndarray] = {}
            for profile_name in ("L_bio", "target", "mu"):
                full_profile = np.asarray(
                    columns[profile_name][row_index], dtype=np.float64
                ).reshape(-1)
                if full_profile.size < expected_length:
                    raise ValueError(
                        f"{profile_name} for {transcript_id} has {full_profile.size} "
                        f"positions; expected at least {expected_length}."
                    )
                profiles[profile_name] = full_profile[:expected_length]

            l_bio = profiles["L_bio"]
            if transcript_id not in first_l_bio:
                first_l_bio[transcript_id] = l_bio.astype(np.float32, copy=True)
                position_mask = cds_interior_mask(expected_length)
                l_bio_metrics_full = profile_recovery_metrics(
                    l_bio,
                    reference,
                    label=f"L_bio {transcript_id}",
                )
                l_bio_metrics = profile_recovery_metrics(
                    l_bio,
                    reference,
                    label=f"L_bio {transcript_id}",
                    position_mask=position_mask,
                )
                transcript_rows[transcript_id] = {
                    "run": run_label,
                    "depth": depth,
                    "dataset_count": dataset_count,
                    "transcript_id": transcript_id,
                    "profile_length": expected_length,
                    "interior_positions": int(position_mask.sum()),
                    **evaluation_domain_metadata(),
                    **_metrics_dict(l_bio_metrics, "l_bio_latent"),
                    **_metrics_dict(
                        l_bio_metrics,
                        "l_bio_latent_interior",
                    ),
                    **_metrics_dict(
                        l_bio_metrics_full,
                        "l_bio_latent_full_previous_definition",
                    ),
                }
            else:
                difference = float(
                    np.max(np.abs(first_l_bio[transcript_id].astype(np.float64) - l_bio))
                )
                maximum_duplicate_difference = max(maximum_duplicate_difference, difference)

            dataset_id = int(columns["dataset_id"][row_index])
            dataset_name = dataset_id_to_name.get(dataset_id, f"dataset_id_{dataset_id}")
            position_mask = cds_interior_mask(expected_length)
            target_metrics_full = profile_recovery_metrics(
                profiles["target"],
                reference,
                label=f"target {dataset_name}/{transcript_id}",
            )
            target_metrics = profile_recovery_metrics(
                profiles["target"],
                reference,
                label=f"target {dataset_name}/{transcript_id}",
                position_mask=position_mask,
            )
            mu_metrics_full = profile_recovery_metrics(
                profiles["mu"],
                reference,
                label=f"mu {dataset_name}/{transcript_id}",
            )
            mu_metrics = profile_recovery_metrics(
                profiles["mu"],
                reference,
                label=f"mu {dataset_name}/{transcript_id}",
                position_mask=position_mask,
            )
            pair_rows.append(
                {
                    "run": run_label,
                    "depth": depth,
                    "dataset_count": dataset_count,
                    "transcript_id": transcript_id,
                    "dataset": dataset_name,
                    "profile_length": expected_length,
                    "interior_positions": int(position_mask.sum()),
                    **evaluation_domain_metadata(),
                    **_metrics_dict(target_metrics, "target_latent"),
                    **_metrics_dict(target_metrics, "target_latent_interior"),
                    **_metrics_dict(
                        target_metrics_full,
                        "target_latent_full_previous_definition",
                    ),
                    **_metrics_dict(mu_metrics, "mu_latent"),
                    **_metrics_dict(mu_metrics, "mu_latent_interior"),
                    **_metrics_dict(
                        mu_metrics_full,
                        "mu_latent_full_previous_definition",
                    ),
                }
            )
        del columns, batch

    if maximum_duplicate_difference > 1.0e-5:
        raise AssertionError(
            "The dataset-independent L_bio profile changed across dataset rows; "
            f"maximum absolute difference={maximum_duplicate_difference:.3e}."
        )
    return (
        pd.DataFrame(transcript_rows.values()),
        pd.DataFrame(pair_rows),
        maximum_duplicate_difference,
    )


def _aggregate_case_metrics(
    pairs: pd.DataFrame,
    transcript_metrics: pd.DataFrame,
) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    l_bio_lookup = transcript_metrics.set_index(["run", "transcript_id"])
    rows: list[dict[str, Any]] = []
    metric_columns = [
        "target_latent_pcc",
        "target_latent_mse",
        "target_latent_rmse",
        "target_latent_mae",
        "mu_latent_pcc",
        "mu_latent_mse",
        "mu_latent_rmse",
        "mu_latent_mae",
        "target_latent_full_previous_definition_pcc",
        "target_latent_full_previous_definition_rmse",
        "mu_latent_full_previous_definition_pcc",
        "mu_latent_full_previous_definition_rmse",
    ]
    for (run, depth, mass_condition, dataset_count, dataset), frame in pairs.groupby(
        ["run", "depth", "mass_condition", "dataset_count", "dataset"], sort=True
    ):
        ids = [(run, str(value)) for value in frame["transcript_id"]]
        l_bio = l_bio_lookup.loc[ids]
        row: dict[str, Any] = {
            "run": run,
            "depth": depth,
            "mass_condition": mass_condition,
            "dataset_count": int(dataset_count),
            "dataset": dataset,
            "transcripts": int(frame["transcript_id"].nunique()),
        }
        for column in metric_columns:
            row[f"{column}_mean"] = float(frame[column].mean())
            row[f"{column}_median"] = float(frame[column].median())
        row["l_bio_latent_pcc_mean"] = float(l_bio["l_bio_latent_pcc"].mean())
        row["l_bio_latent_rmse_mean"] = float(l_bio["l_bio_latent_rmse"].mean())
        row["l_bio_latent_full_previous_definition_pcc_mean"] = float(
            l_bio["l_bio_latent_full_previous_definition_pcc"].mean()
        )
        row["l_bio_latent_full_previous_definition_rmse_mean"] = float(
            l_bio["l_bio_latent_full_previous_definition_rmse"].mean()
        )
        row.update(evaluation_domain_metadata())
        row["l_bio_minus_target_pcc"] = (
            row["l_bio_latent_pcc_mean"] - row["target_latent_pcc_mean"]
        )
        row["target_minus_l_bio_rmse"] = (
            row["target_latent_rmse_mean"] - row["l_bio_latent_rmse_mean"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _short_case_label(dataset: str) -> str:
    replacements = {
        "artificial_bias_3prime_": "3′-",
        "artificial_bias_5prime_": "5′-",
        "artificial_bias_": "",
        "_fraction_gt_0p7": ">0.7",
        "artificial_ground_truth": "unbiased",
    }
    label = dataset
    for source, target in replacements.items():
        label = label.replace(source, target)
    return label.upper().replace("_", " ")


def _make_overview_plot(
    summary: pd.DataFrame,
    case_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    completed = summary[summary["predictions_available"]].copy()
    if completed.empty:
        raise ValueError(
            "Interior recovery plots require detailed prediction parquets; "
            "historical TensorBoard aggregates use the full-CDS definition."
        )
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.25), constrained_layout=True)
    mass_conditions = summary["mass_condition"].unique()
    if len(mass_conditions) != 1:
        raise ValueError("Each recovery figure must contain exactly one mass condition.")
    mass_condition = str(mass_conditions[0])
    summary = completed.copy()
    summary["plot_series"] = summary["depth"].astype(str)
    series_names = sorted(summary["plot_series"].unique())
    palette = plt.get_cmap("tab10")
    colors = {name: palette(index % 10) for index, name in enumerate(series_names)}

    for series, frame in summary.groupby("plot_series", sort=True):
        frame = frame.sort_values("dataset_count")
        color = colors[series]
        axes[0].plot(
            frame["dataset_count"],
            frame["L_vs_K_PCC_interior"],
            color=color,
            linewidth=2.2,
            label=f"L_bio, {series}",
        )
        axes[1].plot(
            frame["dataset_count"],
            frame["L_vs_K_RMSE_interior"],
            color=color,
            linewidth=2.2,
            label=series,
        )
        detailed = frame[frame["predictions_available"]]
        axes[0].scatter(
            detailed["dataset_count"], detailed["L_vs_K_PCC_interior"],
            color=color, s=58, zorder=3,
        )
        axes[1].scatter(
            detailed["dataset_count"], detailed["L_vs_K_RMSE_interior"],
            color=color, s=58, zorder=3,
        )
        if not detailed.empty and detailed["l_bio_pcc_ci_low"].notna().any():
            axes[0].fill_between(
                detailed["dataset_count"].to_numpy(),
                detailed["l_bio_pcc_ci_low"].to_numpy(),
                detailed["l_bio_pcc_ci_high"].to_numpy(),
                color=color,
                alpha=0.16,
                linewidth=0,
                label="95% transcript bootstrap CI" if series == series_names[0] else None,
            )

    axes[0].set_title("A. Shared-signal shape recovery", loc="left", fontweight="bold")
    axes[0].set_xlabel("Number of biased datasets used for training")
    axes[0].set_ylabel("Interior PCC: L_bio vs latent truth")
    axes[0].set_ylim(0.70, 1.005)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[1].set_title("B. Shared-signal magnitude error", loc="left", fontweight="bold")
    axes[1].set_xlabel("Number of biased datasets used for training")
    axes[1].set_ylabel("Interior RMSE of full-CDS-normalized L_bio")
    axes[1].legend(frameon=False, fontsize=8)

    if not case_summary.empty:
        representative = completed.sort_values(
            ["dataset_count", "depth", "run"]
        ).iloc[-1]
        max_completed = int(representative["dataset_count"])
        depth = str(representative["depth"])
        run = str(representative["run"])
        cases = case_summary[
            (case_summary["dataset_count"] == max_completed)
            & (case_summary["depth"] == depth)
            & (case_summary["run"] == run)
        ].copy()
        cases = cases.sort_values("dataset")
        x = np.arange(len(cases), dtype=np.float64)
        width = 0.25
        axes[2].bar(
            x - width,
            cases["target_latent_pcc_mean"],
            width,
            color="#9aa1a8",
            label="observed biased consensus",
        )
        axes[2].bar(
            x,
            cases["mu_latent_pcc_mean"],
            width,
            color="#d97904",
            label="dataset reconstruction μ",
        )
        axes[2].bar(
            x + width,
            cases["l_bio_latent_pcc_mean"],
            width,
            color="#2468a2",
            label="shared L_bio",
        )
        axes[2].set_xticks(x, [_short_case_label(value) for value in cases["dataset"]])
        axes[2].tick_params(axis="x", rotation=32)
        axes[2].set_ylim(0.0, 1.0)
        axes[2].set_ylabel("Mean per-transcript PCC vs latent truth")
        axes[2].set_title(
            f"C. Factorization by bias case (N={max_completed})",
            loc="left",
            fontweight="bold",
        )
        axes[2].legend(frameon=False, fontsize=8, loc="lower right")
    else:
        axes[2].text(0.5, 0.5, "No prediction parquets available", ha="center", va="center")
        axes[2].set_axis_off()

    for axis in axes[:2]:
        axis.grid(axis="y", alpha=0.22)
        axis.set_xticks(sorted(summary["dataset_count"].unique()))
    fig.suptitle(
        f"Synthetic latent-kinetics recovery ({mass_condition})",
        fontsize=15,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
    header = "| " + " | ".join(label for _, label, _ in columns) + " |"
    divider = "| " + " | ".join("---:" if fmt != "s" else "---" for _, _, fmt in columns) + " |"
    rows = [header, divider]
    for row in frame.itertuples(index=False):
        values: list[str] = []
        data = row._asdict()
        for name, _, fmt in columns:
            value = data[name]
            if fmt == "s":
                values.append(str(value))
            elif pd.isna(value):
                values.append("—")
            else:
                values.append(format(value, fmt))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _write_report(
    summary: pd.DataFrame,
    case_summary: pd.DataFrame,
    *,
    selection_metric: str,
    output_path: Path,
) -> None:
    ordered = summary.sort_values(
        ["mass_condition", "depth", "dataset_count"]
    ).copy()
    completed = ordered[ordered["predictions_available"]].copy()
    if completed.empty:
        raise ValueError(
            "The primary interior report requires detailed prediction parquets."
        )
    best = completed.loc[completed["L_vs_K_PCC_interior"].idxmax()]
    common_hashes = sorted(ordered["validation_id_hash"].unique())
    common_statement = (
        f"All runs use the same validation-ID hash `{common_hashes[0]}`."
        if len(common_hashes) == 1
        else f"WARNING: validation-ID hashes differ: {common_hashes}."
    )

    table = _markdown_table(
        ordered,
        [
            ("mass_condition", "mass", "s"),
            ("depth", "depth", "s"),
            ("dataset_count", "N", ".0f"),
            ("validation_transcripts", "validation transcripts", ".0f"),
            ("L_vs_K_PCC_interior", "L_bio PCC (interior)", ".4f"),
            ("L_vs_K_RMSE_interior", "L_bio RMSE (interior)", ".4f"),
            (
                "L_vs_K_PCC_full_previous_definition",
                "L_bio PCC (old full CDS)",
                ".4f",
            ),
            ("latent_mae_posthoc", "L_bio MAE", ".4f"),
            ("val_mu_pcc", "selection val μ PCC", ".4f"),
            ("artifact_status", "evidence", "s"),
        ],
    )

    case_text = "No prediction parquets were available for case-level analysis."
    case_context = ""
    if not case_summary.empty and not completed.empty:
        representative = completed.sort_values(
            ["dataset_count", "depth", "run"]
        ).iloc[-1]
        max_count = int(representative["dataset_count"])
        max_depth = str(representative["depth"])
        representative_run = str(representative["run"])
        cases = case_summary[
            (case_summary["dataset_count"] == max_count)
            & (case_summary["depth"] == max_depth)
            & (case_summary["run"] == representative_run)
        ].copy()
        case_context = (
            f"Shown for `{max_depth}`, N={max_count}, run "
            f"`{representative_run}`.\n\n"
        )
        cases["case"] = cases["dataset"].map(_short_case_label)
        case_text = _markdown_table(
            cases,
            [
                ("case", "bias case", "s"),
                ("target_latent_pcc_mean", "biased target PCC", ".4f"),
                ("mu_latent_pcc_mean", "μ PCC", ".4f"),
                ("l_bio_latent_pcc_mean", "L_bio PCC", ".4f"),
                ("l_bio_minus_target_pcc", "L_bio − target", "+.4f"),
            ],
        )

    depth_results: list[str] = []
    for (mass_condition, depth), frame in completed.groupby(
        ["mass_condition", "depth"], sort=True
    ):
        frame = frame.sort_values("dataset_count")
        first = frame.iloc[0]
        last = frame.iloc[-1]
        pcc_gain = float(last["L_vs_K_PCC_interior"] - first["L_vs_K_PCC_interior"])
        rmse_reduction = 100.0 * (
            1.0
            - float(
                last["L_vs_K_RMSE_interior"]
                / first["L_vs_K_RMSE_interior"]
            )
        )
        depth_results.append(
            f"- `{mass_condition}`, `{depth}`: "
            f"N={int(first['dataset_count'])} to "
            f"N={int(last['dataset_count'])}; PCC "
            f"{first['L_vs_K_PCC_interior']:.4f} to "
            f"{last['L_vs_K_PCC_interior']:.4f} "
            f"({pcc_gain:+.4f}); RMSE "
            f"{first['L_vs_K_RMSE_interior']:.4f} to "
            f"{last['L_vs_K_RMSE_interior']:.4f} "
            f"({rmse_reduction:.1f}% reduction)."
        )
    depth_result_text = "\n".join(depth_results)

    aggregate_only = ordered.loc[~ordered["predictions_available"], "dataset_count"]
    aggregate_limit = (
        "Some panels have only checkpoint scalars (N="
        + ", ".join(str(int(value)) for value in sorted(aggregate_only.unique()))
        + "); they have no fabricated transcript or case distribution."
        if not aggregate_only.empty
        else "Every analyzed panel has a detailed prediction parquet."
    )
    analyzed_depths = ", ".join(f"`{value}`" for value in sorted(ordered["depth"].unique()))
    analyzed_mass_conditions = ", ".join(
        f"`{value}`" for value in sorted(ordered["mass_condition"].unique())
    )

    plot_links = "\n\n".join(
        f"![Recovery overview: {condition}]"
        f"(synthetic_recovery_overview_{condition}.png)"
        for condition in sorted(ordered["mass_condition"].unique())
    )
    output_path.write_text(
        f"""# Synthetic shared-signal recovery report

{plot_links}

## Result

Changes across the cumulative panels, calculated separately within each read
depth and mass-conservation condition:

{depth_result_text}

The best currently observed shared-signal PCC is
**{best['L_vs_K_PCC_interior']:.4f}** at N={int(best['dataset_count'])} in
`{best['depth']}`.

This is positive evidence that the model recovers the programmed shared shape
as more independently biased observations are supplied. It is not yet a clean
causal estimate of dataset count: panels are cumulative, so N and the identity
of the newly added bias are confounded, and only one seed is present.

## Dataset-count comparison

{table}

`detailed prediction parquet` means the metric was recomputed position by
position and has per-transcript/per-case support. `selected-checkpoint scalar`
means the run has a checkpoint and TensorBoard aggregate at the selected step,
but prediction did not finish, so no distribution or case breakdown is claimed.

## Bias-case comparison

{case_context}
{case_text}

`mu` should retain dataset bias because it reconstructs each observed dataset;
the ground-truth recovery target is `L_bio`. The target and `mu` columns are
included to show that a high `L_bio` score is not merely the observed biased
profile being relabeled.

## Dataset-specific gamma recovery

The complementary [gamma-bias recovery report](GAMMA_RECOVERY.md) compares
each learned dataset-specific `gamma` profile with its programmed
`bias_profile` annotation after applying the same identifiable two-way gauge.
It includes a per-bias figure and strict multiplicative-error thresholds, not
only correlations.

## Exact comparison protocol

- Checkpoint selection: **{selection_metric}**, matching the prediction path;
  latent ground truth is never used to choose an epoch.
- Ground truth: deterministic collision-free
  `artificial_ground_truth_kinetics_target_mean_one.parquet`.
- Both profiles are independently divided by their original full modeled-CDS
  mean, preserving the historical scaling definition.
- Primary metrics then select exactly the 0-based positions
  `5 <= i < L-5` from both profiles. The first/last five codons are unavailable
  fairly to a CDS-only model because the programmed 30-nt RPF feature may use
  UTR sequence there.
- PCC measures interior shape agreement; RMSE and MAE measure error on those
  same interior positions after full-profile normalization.
- Metrics are calculated per transcript and then averaged equally, so long
  transcripts do not receive extra outer weight.
- The terminal stop is already absent from the latent synthetic P-site profile;
  predictions are truncated to that exact coordinate length before the
  boundary mask is constructed.
- Historical full-CDS and corrected interior values are both exported in
  `synthetic_recovery_full_vs_interior.tsv`, with per-bias comparisons in
  `synthetic_recovery_by_bias_full_vs_interior.tsv`.
- {common_statement}
- The selected validation panel has {int(ordered['validation_transcripts'].min())}
  transcripts in every run.

## Limits of the current evidence

1. {aggregate_limit}
2. Analyzed read-depth labels: {analyzed_depths}; mass conditions:
   {analyzed_mass_conditions}. Missing conditions are not inferred or plotted.
3. Cumulative panels change both N and bias composition. Multiple random panel
   orders and seeds are needed for uncertainty about the dataset-count effect.
4. Checkpoint selection uses `{selection_metric}` (minimum for loss, maximum
   for PCC); latent ground-truth recovery is never used to choose an epoch.
5. The resolved 0.25-read/codon configs still point the legacy TensorBoard
   `observed_*` diagnostic at the 20-read/codon unbiased file. Those scalars
   are intentionally excluded here; target/`mu` case baselines are recomputed
   directly from each run's own prediction parquet.
""",
        encoding="utf-8",
    )


def analyze(args: argparse.Namespace) -> dict[str, Path]:
    repository_root = REPOSITORY_ROOT
    results_root = Path(args.results_root).expanduser().resolve()
    latent_path = Path(args.latent_ground_truth).expanduser()
    if not latent_path.is_absolute():
        latent_path = repository_root / latent_path
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    truth = _load_latent_truth(latent_path)
    run_rows: list[dict[str, Any]] = []
    transcript_frames: list[pd.DataFrame] = []
    pair_frames: list[pd.DataFrame] = []
    expected_validation_ids: dict[str, set[str]] = {}
    skipped_rows: list[dict[str, str]] = []

    run_dirs = discover_run_directories(results_root, args.run_prefix)
    if not run_dirs:
        raise RuntimeError(
            f"No run directories beginning {args.run_prefix!r} were found "
            f"directly below {results_root}."
        )
    print(
        f"Found {len(run_dirs)} run directories beginning "
        f"{args.run_prefix!r} below {results_root}."
    )
    for run_index, run_dir in enumerate(run_dirs):
        config_paths = sorted(run_dir.rglob("config.yaml"))
        if len(config_paths) != 1:
            reason = f"expected one config.yaml, found {len(config_paths)}"
            if args.strict:
                raise ValueError(f"{run_dir.name}: {reason}.")
            print(f"[skip] {run_dir.name}: {reason}.")
            skipped_rows.append({"run": run_dir.name, "reason": reason})
            continue
        try:
            config = _read_yaml(config_paths[0])
            datasets = [
                str(value)
                for value in config.get("experiment", {}).get("dataset", [])
            ]
            if not datasets:
                raise ValueError("resolved config contains no experiment datasets")
            depth = synthetic_depth_label(config)
            mass_conservation, mass_condition = synthetic_mass_conservation(
                config, run_dir.name
            )
            validation_ids, validation_hash = _validation_manifest(run_dir)
            histories = _read_scalar_history(run_dir)
            selected_step = _select_step(histories, args.selection_metric)
            latent_mse = _scalar_at(
                histories, "val/synthetic_ground_truth/latent_mse", selected_step
            )
            latent_pcc = _scalar_at(
                histories, "val/synthetic_ground_truth/latent_pcc", selected_step
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            if args.strict:
                raise
            reason = str(exc).replace("\n", " ")
            print(f"[skip] {run_dir.name}: {reason}")
            skipped_rows.append({"run": run_dir.name, "reason": reason})
            continue

        validation_id_set = set(validation_ids)
        validation_key = f"{depth}/{mass_condition}"
        previous_ids = expected_validation_ids.setdefault(validation_key, validation_id_set)
        if validation_id_set != previous_ids and not bool(
            getattr(args, "allow_validation_id_variation", False)
        ):
            raise AssertionError(
                f"Validation IDs differ within condition {validation_key}: {run_dir.name}."
            )
        prediction_checkpoint_variant = (
            "best_val_loss" if args.selection_metric == "val_loss" else "best_pcc"
        )
        prediction_path = _find_prediction(
            run_dir,
            checkpoint_variant=prediction_checkpoint_variant,
        )
        row: dict[str, Any] = {
            "run": run_dir.name,
            "depth": depth,
            "mass_conservation": mass_conservation,
            "mass_condition": mass_condition,
            "dataset_count": len(datasets),
            "datasets": ",".join(datasets),
            "selection_metric": args.selection_metric,
            "prediction_checkpoint_variant": prediction_checkpoint_variant,
            "selection_step": selected_step,
            "val_loss": _scalar_at(histories, "val_loss", selected_step),
            "val_mu_pcc": _scalar_at(histories, "val_mu_pcc", selected_step),
            "validation_transcripts": int(
                round(_scalar_at(
                    histories,
                    "val/synthetic_ground_truth/transcripts",
                    selected_step,
                ))
            ),
            "validation_id_hash": validation_hash,
            "latent_pcc_selected": latent_pcc,
            "latent_mse_selected": latent_mse,
            "latent_rmse_selected": math.sqrt(latent_mse),
            "legacy_logged_metric_definition": "full_cds_previous_definition",
            # TensorBoard values predate the boundary correction and are kept
            # only as explicitly labelled historical whole-CDS diagnostics.
            "L_vs_K_PCC_full_previous_definition": latent_pcc,
            "L_vs_K_MSE_full_previous_definition": latent_mse,
            "L_vs_K_RMSE_full_previous_definition": math.sqrt(latent_mse),
            "L_vs_K_PCC_interior": float("nan"),
            "L_vs_K_MSE_interior": float("nan"),
            "L_vs_K_RMSE_interior": float("nan"),
            **evaluation_domain_metadata(),
            "predictions_available": prediction_path is not None,
            "prediction_path": str(prediction_path) if prediction_path else "",
            "artifact_status": (
                "detailed prediction parquet"
                if prediction_path is not None
                else "selected-checkpoint scalar"
            ),
            "l_bio_pcc_posthoc": float("nan"),
            "l_bio_pcc_median_posthoc": float("nan"),
            "l_bio_pcc_ci_low": float("nan"),
            "l_bio_pcc_ci_high": float("nan"),
            "l_bio_mse_posthoc": float("nan"),
            "l_bio_rmse_posthoc": float("nan"),
            "latent_mae_posthoc": float("nan"),
            "l_bio_dataset_row_max_abs_difference": float("nan"),
        }

        if prediction_path is not None:
            transcript_frame, pair_frame, duplicate_difference = _analyze_prediction(
                prediction_path,
                truth=truth,
                dataset_id_to_name=_encoding_from_config(config, repository_root),
                run_label=run_dir.name,
                depth=depth,
                dataset_count=len(datasets),
            )
            predicted_ids = set(transcript_frame["transcript_id"].astype(str))
            if predicted_ids != validation_id_set:
                raise AssertionError(
                    f"Prediction IDs do not equal validation IDs for {run_dir.name}: "
                    f"prediction={len(predicted_ids)}, validation={len(validation_id_set)}."
                )
            pcc_low, pcc_high = _bootstrap_mean_interval(
                transcript_frame["l_bio_latent_pcc"],
                seed=10_000 + run_index,
            )
            row.update(
                {
                    "l_bio_pcc_posthoc": float(
                        transcript_frame["l_bio_latent_pcc"].mean()
                    ),
                    "l_bio_pcc_median_posthoc": float(
                        transcript_frame["l_bio_latent_pcc"].median()
                    ),
                    "l_bio_pcc_ci_low": pcc_low,
                    "l_bio_pcc_ci_high": pcc_high,
                    "l_bio_mse_posthoc": float(
                        transcript_frame["l_bio_latent_mse"].mean()
                    ),
                    # sqrt(mean transcript MSE), matching the logged aggregate.
                    "l_bio_rmse_posthoc": math.sqrt(
                        float(transcript_frame["l_bio_latent_mse"].mean())
                    ),
                    "latent_mae_posthoc": float(
                        transcript_frame["l_bio_latent_mae"].mean()
                    ),
                    "l_bio_dataset_row_max_abs_difference": duplicate_difference,
                    "L_vs_K_PCC_interior": float(
                        transcript_frame["l_bio_latent_interior_pcc"].mean()
                    ),
                    "L_vs_K_MSE_interior": float(
                        transcript_frame["l_bio_latent_interior_mse"].mean()
                    ),
                    "L_vs_K_RMSE_interior": math.sqrt(
                        float(transcript_frame["l_bio_latent_interior_mse"].mean())
                    ),
                    "L_vs_K_PCC_full_previous_definition": float(
                        transcript_frame[
                            "l_bio_latent_full_previous_definition_pcc"
                        ].mean()
                    ),
                    "L_vs_K_MSE_full_previous_definition": float(
                        transcript_frame[
                            "l_bio_latent_full_previous_definition_mse"
                        ].mean()
                    ),
                    "L_vs_K_RMSE_full_previous_definition": math.sqrt(
                        float(
                            transcript_frame[
                                "l_bio_latent_full_previous_definition_mse"
                            ].mean()
                        )
                    ),
                }
            )
            if not math.isclose(
                row["L_vs_K_PCC_full_previous_definition"],
                latent_pcc,
                rel_tol=0.0,
                abs_tol=2.0e-5,
            ):
                raise AssertionError(
                    f"Full-CDS post-hoc and logged latent PCC disagree for "
                    f"{run_dir.name}: "
                    f"{row['L_vs_K_PCC_full_previous_definition']} vs {latent_pcc}."
                )
            if not math.isclose(
                row["L_vs_K_MSE_full_previous_definition"],
                latent_mse,
                rel_tol=0.0,
                abs_tol=2.0e-5,
            ):
                raise AssertionError(
                    f"Full-CDS post-hoc and logged latent MSE disagree for "
                    f"{run_dir.name}: "
                    f"{row['L_vs_K_MSE_full_previous_definition']} vs {latent_mse}."
                )
            transcript_frames.append(transcript_frame)
            pair_frames.append(pair_frame)
            transcript_frame["mass_conservation"] = mass_conservation
            transcript_frame["mass_condition"] = mass_condition
            pair_frame["mass_conservation"] = mass_conservation
            pair_frame["mass_condition"] = mass_condition
        run_rows.append(row)
        gc.collect()

    if not run_rows:
        raise RuntimeError(f"No analyzable synthetic runs found below {results_root}.")

    summary = pd.DataFrame(run_rows).sort_values(
        ["mass_condition", "depth", "dataset_count"]
    )
    transcript_metrics = (
        pd.concat(transcript_frames, ignore_index=True)
        if transcript_frames
        else pd.DataFrame()
    )
    pair_metrics = (
        pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    )
    case_summary = _aggregate_case_metrics(pair_metrics, transcript_metrics)

    comparison_rows: list[dict[str, Any]] = []
    for row in summary[summary["predictions_available"]].to_dict(orient="records"):
        for metric, full_column, interior_column in (
            (
                "L_vs_K_PCC",
                "L_vs_K_PCC_full_previous_definition",
                "L_vs_K_PCC_interior",
            ),
            (
                "L_vs_K_RMSE",
                "L_vs_K_RMSE_full_previous_definition",
                "L_vs_K_RMSE_interior",
            ),
        ):
            comparison_rows.append(
                {
                    "run": row["run"],
                    "depth": row["depth"],
                    "mass_condition": row["mass_condition"],
                    "dataset_count": row["dataset_count"],
                    "metric": metric,
                    "original_full_cds": row[full_column],
                    "interior_only": row[interior_column],
                    **evaluation_domain_metadata(),
                }
            )

    case_comparison_rows: list[dict[str, Any]] = []
    for row in case_summary.to_dict(orient="records"):
        for signal, full_prefix, interior_prefix in (
            ("L_vs_K", "l_bio_latent_full_previous_definition", "l_bio_latent"),
            ("target_vs_K", "target_latent_full_previous_definition", "target_latent"),
            ("mu_vs_K", "mu_latent_full_previous_definition", "mu_latent"),
        ):
            for metric in ("pcc", "rmse"):
                case_comparison_rows.append(
                    {
                        "run": row["run"],
                        "depth": row["depth"],
                        "mass_condition": row["mass_condition"],
                        "dataset_count": row["dataset_count"],
                        "dataset": row["dataset"],
                        "metric": f"{signal}_{metric.upper()}",
                        "original_full_cds": row[f"{full_prefix}_{metric}_mean"],
                        "interior_only": row[f"{interior_prefix}_{metric}_mean"],
                        **evaluation_domain_metadata(),
                    }
                )

    paths = {
        "summary": output_dir / "synthetic_recovery_by_panel.tsv",
        "cases": output_dir / "synthetic_recovery_by_bias_case.tsv",
        "transcripts": output_dir / "synthetic_recovery_by_transcript.tsv.gz",
        "report": output_dir / "README.md",
        "skipped": output_dir / "synthetic_recovery_skipped_runs.tsv",
        "domain_comparison": output_dir
        / "synthetic_recovery_full_vs_interior.tsv",
        "case_domain_comparison": output_dir
        / "synthetic_recovery_by_bias_full_vs_interior.tsv",
    }
    summary.to_csv(paths["summary"], sep="\t", index=False)
    case_summary.to_csv(paths["cases"], sep="\t", index=False)
    transcript_metrics.to_csv(
        paths["transcripts"], sep="\t", index=False, compression="gzip"
    )
    pd.DataFrame(comparison_rows).to_csv(
        paths["domain_comparison"], sep="\t", index=False
    )
    pd.DataFrame(case_comparison_rows).to_csv(
        paths["case_domain_comparison"], sep="\t", index=False
    )
    pd.DataFrame(skipped_rows, columns=["run", "reason"]).to_csv(
        paths["skipped"], sep="\t", index=False
    )
    for mass_condition, condition_summary in summary.groupby(
        "mass_condition", sort=True
    ):
        condition_cases = (
            case_summary[case_summary["mass_condition"] == mass_condition]
            if not case_summary.empty
            else case_summary.copy()
        )
        plot_path = output_dir / f"synthetic_recovery_overview_{mass_condition}.png"
        _make_overview_plot(condition_summary.copy(), condition_cases, plot_path)
        paths[f"plot_{mass_condition}"] = plot_path
    _write_report(
        summary,
        case_summary,
        selection_metric=args.selection_metric,
        output_path=paths["report"],
    )
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument(
        "--run-prefix",
        default=DEFAULT_RUN_PREFIX,
        help=(
            "Analyze only immediate result directories beginning with this "
            f"prefix (default: {DEFAULT_RUN_PREFIX!r})."
        ),
    )
    parser.add_argument("--latent-ground-truth", default=str(DEFAULT_LATENT_TRUTH))
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("val_mu_pcc", "val_loss"),
        default="val_mu_pcc",
        help="Select max val_mu_pcc (prediction default) or min val_loss.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of recording and skipping incomplete run directories.",
    )
    return parser


def main() -> None:
    paths = analyze(build_parser().parse_args())
    print("Synthetic recovery report written:")
    for label, path in paths.items():
        print(f"  {label:11s} {path}")


if __name__ == "__main__":
    main()
