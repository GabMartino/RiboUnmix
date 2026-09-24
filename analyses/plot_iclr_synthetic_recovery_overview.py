#!/usr/bin/env python3
"""Create the compact synthetic-recovery figures for an ICLR paper.

The first panel reports shared biological-profile recovery: the interior,
Fisher-weighted Pearson correlation of the learned ``L_bio`` profile with the
deterministic synthetic latent kinetics ``K``.  The second uses the same
within-depth, mass-free, equal-reference configuration to report recovery of
the learned dataset-bias ``gamma`` profiles against their programmed truth.
Its values therefore match
``gamma_recovery_main_val_by_read_depth__within__mass_free__rankp0p0``.
The third panel is a horizontal dot-and-confidence-interval version of the
single-dataset ``mu`` PCC-by-bias-and-depth plot.  This layout makes the
variation across bias datasets at one fixed read depth readable in the compact
third column.

The established overview is intentionally retained unchanged.  A second,
separate N=2 read-depth diagnostic explains why finite-count agreement rises
strongly with depth while latent-K recovery is almost flat.  It distinguishes
three quantities:

* estimation of the identifiable equal-gauge target;
* total biological recovery against deterministic K; and
* agreement with an unbiased, finite-depth count observation and its
  latent-to-count noise benchmark.

The defaults read the summary tables produced by
``13_gamma_ablation_recovery.py``, ``20_gamma_ablation_gamma_recovery.py``,
and ``analyze_synthetic_single_dataset_mu_pcc.py``.  They only re-render the
established overview.  The additional diagnostic reads the selected N=2
best-validation-loss prediction parquets and the synthetic evaluation-only
truth artifacts; it never changes a model or checkpoint.

Example
-------
::

    .venv/bin/python analyses/plot_iclr_synthetic_recovery_overview.py
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_ROOT = (
    REPOSITORY_ROOT / "analyses" / "artifacts" / "synthetic" / "gamma_ablation"
)
DEFAULT_RECOVERY_SUMMARY = DEFAULT_ANALYSIS_ROOT / "recovery" / "recovery_by_condition.csv"
DEFAULT_GAMMA_SUMMARY = (
    DEFAULT_ANALYSIS_ROOT / "gamma_recovery" / "gamma_recovery_by_condition.csv"
)
DEFAULT_SINGLE_DATASET_SUMMARY = (
    REPOSITORY_ROOT
    / "analyses"
    / "artifacts"
    / "synthetic"
    / "single_dataset_mu"
    / "single_20260830_205110"
    / "best_pcc"
    / "single_dataset_mu_pcc_summary.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_ANALYSIS_ROOT / "iclr_figures"
DEFAULT_OUTPUT_STEM = "synthetic_recovery_overview"
DEFAULT_DIAGNOSTIC_OUTPUT_STEM = "synthetic_read_depth_diagnostic_n2"
DEFAULT_RESULTS_ROOT = REPOSITORY_ROOT / "results" / "riboai_synthetic_experiments"
DEFAULT_LATENT_TRUTH = (
    REPOSITORY_ROOT
    / "Datasets"
    / "Synthetic_data"
    / "artificial_ground_truth_kinetics_target_mean_one.parquet"
)
DEFAULT_BIAS_ROOT = REPOSITORY_ROOT / "Datasets" / "Synthetic_data" / "bias_profile"
DEFAULT_OBSERVED_COUNTS_ROOT = REPOSITORY_ROOT / "Datasets" / "Synthetic_data"
DEFAULT_DIAGNOSTIC_N_DATASETS = 2
DEFAULT_BOUNDARY_TRIM_CODONS = 5


# Latin Modern Roman is the modern Computer Modern family distributed with
# TeX.  ``mathtext.fontset=cm`` gives labels such as L_bio, mu, and gamma the
# matching Computer Modern mathematical glyphs without requiring a LaTeX
# installation on the machine that renders the figure.  Type 42 keeps PDF text
# editable.
ICLR_RC = {
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8.5,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "axes.titleweight": "semibold",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.fontsize": 8.5,
    "legend.title_fontsize": 8.5,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 5.0,
    "grid.color": "#D0D0D0",
    "grid.linewidth": 0.55,
    "grid.alpha": 0.55,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 300,
}

DEPTH_ORDER = ("0p25_per_codon", "2_per_codon", "20_per_codon")
DEPTH_LABELS = {
    "0p25_per_codon": "0.25 reads/codon",
    "2_per_codon": "2 reads/codon",
    "20_per_codon": "20 reads/codon",
}
DEPTH_STYLES = {
    "0p25_per_codon": {"color": "#0072B2", "marker": "o"},
    "2_per_codon": {"color": "#E69F00", "marker": "s"},
    "20_per_codon": {"color": "#009E73", "marker": "^"},
}
SINGLE_DEPTH_TO_RECOVERY_DEPTH = {
    "0p25": "0p25_per_codon",
    "2": "2_per_codon",
    "20": "20_per_codon",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recovery-summary",
        type=Path,
        default=DEFAULT_RECOVERY_SUMMARY,
        help="CSV from 13_gamma_ablation_recovery.py.",
    )
    parser.add_argument(
        "--gamma-summary",
        type=Path,
        default=DEFAULT_GAMMA_SUMMARY,
        help="CSV from 20_gamma_ablation_gamma_recovery.py.",
    )
    parser.add_argument(
        "--single-dataset-summary",
        type=Path,
        default=DEFAULT_SINGLE_DATASET_SUMMARY,
        help="CSV from analyze_synthetic_single_dataset_mu_pcc.py.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument(
        "--diagnostic-output-stem",
        default=DEFAULT_DIAGNOSTIC_OUTPUT_STEM,
        help="Stem for the separate read-depth diagnostic figure and TSV.",
    )
    parser.add_argument(
        "--skip-read-depth-diagnostic",
        action="store_true",
        help="Render only the unchanged three-panel overview.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory containing the selected synthetic run directories.",
    )
    parser.add_argument(
        "--latent-truth",
        type=Path,
        default=DEFAULT_LATENT_TRUTH,
        help="Deterministic mean-one K parquet used only for evaluation.",
    )
    parser.add_argument(
        "--bias-root",
        type=Path,
        default=DEFAULT_BIAS_ROOT,
        help="Directory containing programmed added-bias compendia.",
    )
    parser.add_argument(
        "--observed-counts-root",
        type=Path,
        default=DEFAULT_OBSERVED_COUNTS_ROOT,
        help="Root containing the three unbiased finite-count references.",
    )
    parser.add_argument(
        "--diagnostic-n-datasets",
        type=int,
        default=DEFAULT_DIAGNOSTIC_N_DATASETS,
        help="Cumulative panel size used by the read-depth diagnostic (default: 2).",
    )
    parser.add_argument(
        "--boundary-trim-codons",
        type=int,
        default=DEFAULT_BOUNDARY_TRIM_CODONS,
        help="Number of codons excluded from each CDS end in diagnostic metrics.",
    )
    parser.add_argument(
        "--strategy",
        default="within",
        help="Synthetic panel strategy to show (default: within).",
    )
    parser.add_argument(
        "--mass-condition",
        default="mass_free",
        help="Mass condition to show (default: mass_free).",
    )
    parser.add_argument(
        "--gamma-weighting",
        default="equal",
        help="Gamma-reference weighting to show (default: equal).",
    )
    parser.add_argument(
        "--quality-rank-power",
        type=float,
        default=0.0,
        help="Gamma quality-rank power to show (default: 0.0).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf", "svg"),
        help="File formats to save (default: png pdf svg).",
    )
    return parser.parse_args()


def _read_csv(path: Path, description: str) -> pd.DataFrame:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{description} is empty: {path}")
    return frame


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], description: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise KeyError(f"{description} is missing required columns: {', '.join(missing)}")


def _filter_equal_reference_rows(
    frame: pd.DataFrame,
    *,
    strategy: str,
    mass_condition: str,
    gamma_weighting: str,
    quality_rank_power: float,
    seed: int,
    description: str,
) -> pd.DataFrame:
    """Keep the scientifically matched multi-dataset recovery configuration."""
    common = {
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
    }
    _require_columns(frame, common, description)
    quality = pd.to_numeric(frame["quality_rank_power"], errors="coerce")
    selected = frame.loc[
        (frame["strategy"].astype(str) == strategy)
        & (frame["training_scope"].astype(str) == "multi_dataset")
        & (frame["mass_condition"].astype(str) == mass_condition)
        & (frame["gamma_weighting"].astype(str) == gamma_weighting)
        & np.isclose(quality, quality_rank_power, equal_nan=False)
        & (frame["feature_preset"].astype(str) == "Baseline")
        & (pd.to_numeric(frame["seed"], errors="coerce") == seed)
        & (frame["split"].astype(str) == "main_val")
    ].copy()
    if selected.empty:
        raise ValueError(
            f"No {description} rows match strategy={strategy!r}, "
            f"mass_condition={mass_condition!r}, gamma_weighting={gamma_weighting!r}, "
            f"quality_rank_power={quality_rank_power}, and seed={seed}."
        )
    return selected


def select_biological_recovery(
    frame: pd.DataFrame, args: argparse.Namespace
) -> pd.DataFrame:
    """Select L_bio-vs-K PCC values for the first panel."""
    _require_columns(
        frame,
        {
            "component",
            "reference_kind",
            "comparison_scope",
            "pearson_dataset_macro_mean",
        },
        "biological recovery summary",
    )
    selected = _filter_equal_reference_rows(
        frame,
        strategy=args.strategy,
        mass_condition=args.mass_condition,
        gamma_weighting=args.gamma_weighting,
        quality_rank_power=args.quality_rank_power,
        seed=args.seed,
        description="biological recovery summary",
    )
    selected = selected.loc[
        (selected["component"].astype(str) == "L_bio")
        & (selected["reference_kind"].astype(str) == "latent_ground_truth")
        & (selected["comparison_scope"].astype(str) == "shared_latent_truth_interior")
    ].copy()
    selected.rename(columns={"pearson_dataset_macro_mean": "pcc"}, inplace=True)
    return _validate_depth_panel(selected, "biological recovery")


def select_gamma_recovery(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Select the same gamma PCC series used by the existing gamma figure."""
    _require_columns(
        frame,
        {
            "checkpoint_variant",
            "component",
            "reference_kind",
            "scope",
            "log_gamma_pearson_fisher",
        },
        "gamma recovery summary",
    )
    selected = _filter_equal_reference_rows(
        frame,
        strategy=args.strategy,
        mass_condition=args.mass_condition,
        gamma_weighting=args.gamma_weighting,
        quality_rank_power=args.quality_rank_power,
        seed=args.seed,
        description="gamma recovery summary",
    )
    selected = selected.loc[
        (selected["checkpoint_variant"].astype(str) == "best_val_loss")
        & (selected["component"].astype(str) == "gamma")
        & (selected["reference_kind"].astype(str) == "programmed_dataset_bias")
        & (selected["scope"].astype(str) == "gauged_log_gamma_interior")
    ].copy()
    selected.rename(columns={"log_gamma_pearson_fisher": "pcc"}, inplace=True)
    return _validate_depth_panel(selected, "gamma recovery")


def _validate_depth_panel(frame: pd.DataFrame, description: str) -> pd.DataFrame:
    if frame.empty:
        raise ValueError(f"No selected rows remain in the {description} summary.")
    frame = frame.copy()
    frame["n_datasets"] = pd.to_numeric(frame["n_datasets"], errors="raise").astype(int)
    frame["pcc"] = pd.to_numeric(frame["pcc"], errors="coerce")
    frame = frame[frame["pcc"].notna()].copy()
    if frame.empty:
        raise ValueError(f"The selected {description} PCC values are all missing.")
    duplicates = frame.duplicated(["depth", "n_datasets"], keep=False)
    if duplicates.any():
        duplicate_rows = frame.loc[duplicates, ["depth", "n_datasets"]].to_dict("records")
        raise ValueError(
            f"Expected one {description} row per depth/dataset count; "
            f"found duplicates: {duplicate_rows[:5]}"
        )
    present_depths = set(frame["depth"].astype(str))
    missing_depths = set(DEPTH_ORDER).difference(present_depths)
    if missing_depths:
        raise ValueError(
            f"{description} is missing requested read depths: {sorted(missing_depths)}"
        )
    return frame


def select_single_dataset_mu(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the 10-bias × 3-depth matrix used in the compact dot panel."""
    required = {
        "bias",
        "bias_label",
        "read_depth",
        "mean_mu_pcc",
        "bootstrap_ci95_low",
        "bootstrap_ci95_high",
    }
    _require_columns(frame, required, "single-dataset mu summary")
    selected = frame.copy()
    selected["recovery_depth"] = selected["read_depth"].astype(str).map(
        SINGLE_DEPTH_TO_RECOVERY_DEPTH
    )
    if selected["recovery_depth"].isna().any():
        unknown = sorted(
            selected.loc[selected["recovery_depth"].isna(), "read_depth"]
            .astype(str)
            .unique()
        )
        raise ValueError(f"Unexpected single-dataset read-depth labels: {unknown}")
    for column in ("mean_mu_pcc", "bootstrap_ci95_low", "bootstrap_ci95_high"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    if selected[["mean_mu_pcc", "bootstrap_ci95_low", "bootstrap_ci95_high"]].isna().any().any():
        raise ValueError("Single-dataset mu summary has non-numeric PCC or CI values.")
    duplicates = selected.duplicated(["bias", "recovery_depth"], keep=False)
    if duplicates.any():
        raise ValueError("Single-dataset mu summary has duplicate bias/read-depth cells.")
    expected_cells = len(selected["bias"].unique()) * len(DEPTH_ORDER)
    if len(selected) != expected_cells:
        raise ValueError(
            "Single-dataset mu summary is not a complete bias × three-read-depth matrix."
        )
    return selected


DEPTH_SUFFIXES = tuple(f"_{depth}" for depth in DEPTH_ORDER)
DIAGNOSTIC_DEPTH_LABELS = {
    "0p25_per_codon": "0.25",
    "2_per_codon": "2",
    "20_per_codon": "20",
}
DIAGNOSTIC_STYLES = {
    "learned": {"color": "#0072B2", "marker": "o"},
    "oracle": {"color": "#D55E00", "marker": "s"},
    "bias_sites": {"color": "#CC79A7", "marker": "^"},
}


def _one_path(paths: Iterable[Path], description: str) -> Path:
    matches = sorted(Path(path) for path in paths)
    if len(matches) != 1:
        raise ValueError(f"Expected one {description}, found {len(matches)}: {matches[:3]}")
    return matches[0]


def _base_bias_name(dataset: str) -> str:
    for suffix in DEPTH_SUFFIXES:
        if dataset.endswith(suffix):
            return dataset[: -len(suffix)]
    return dataset


def _load_prediction_l_bio(path: Path) -> dict[str, np.ndarray]:
    """Load one dataset-independent L_bio vector per validation transcript."""
    parquet = pq.ParquetFile(path)
    required = {"transcript_id", "L_bio", "length"}
    missing = sorted(required.difference(parquet.schema_arrow.names))
    if missing:
        raise KeyError(f"Prediction parquet {path} is missing columns: {missing}")

    profiles: dict[str, np.ndarray] = {}
    maximum_duplicate_difference = 0.0
    for batch in parquet.iter_batches(
        batch_size=64,
        columns=["transcript_id", "L_bio", "length"],
    ):
        columns = batch.to_pydict()
        for transcript_id, value, raw_length in zip(
            columns["transcript_id"], columns["L_bio"], columns["length"]
        ):
            transcript_id = str(transcript_id)
            length = int(raw_length)
            profile = np.asarray(value, dtype=np.float64).reshape(-1)[:length]
            if transcript_id in profiles:
                previous = profiles[transcript_id]
                overlap = min(previous.size, profile.size)
                if overlap:
                    maximum_duplicate_difference = max(
                        maximum_duplicate_difference,
                        float(np.max(np.abs(previous[:overlap] - profile[:overlap]))),
                    )
                continue
            profiles[transcript_id] = profile
    if maximum_duplicate_difference > 1.0e-5:
        raise ValueError(
            "Dataset-independent L_bio differs across duplicate dataset rows in "
            f"{path}: maximum absolute difference={maximum_duplicate_difference:.3e}."
        )
    if not profiles:
        raise ValueError(f"No L_bio profiles found in {path}")
    return profiles


def _load_latent_truth(path: Path, requested_ids: set[str]) -> dict[str, np.ndarray]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Latent truth does not exist: {path}")
    profiles: dict[str, np.ndarray] = {}
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=256,
        columns=["transcript_id", "rib_profile"],
    ):
        columns = batch.to_pydict()
        for transcript_id, value in zip(columns["transcript_id"], columns["rib_profile"]):
            transcript_id = str(transcript_id)
            if transcript_id in requested_ids:
                profiles[transcript_id] = np.asarray(value, dtype=np.float64).reshape(-1)
    missing = sorted(requested_ids.difference(profiles))
    if missing:
        raise KeyError(f"Latent truth is missing {len(missing)} requested IDs, e.g. {missing[:3]}")
    return profiles


def _load_added_bias(
    *,
    dataset: str,
    bias_root: Path,
    requested_ids: set[str],
    latent: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    base_name = _base_bias_name(dataset)
    if base_name == "artificial_ground_truth":
        return {
            transcript_id: np.zeros_like(latent[transcript_id], dtype=np.float64)
            for transcript_id in requested_ids
        }
    path = bias_root.expanduser().resolve() / f"{base_name}_compendium_added_bias_only.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Programmed bias profile does not exist: {path}")
    profiles: dict[str, np.ndarray] = {}
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=256,
        columns=["sample", "transcript_id", "added_bias"],
    ):
        columns = batch.to_pydict()
        for sample, transcript_id, value in zip(
            columns["sample"], columns["transcript_id"], columns["added_bias"]
        ):
            transcript_id = str(transcript_id)
            if transcript_id not in requested_ids or not str(sample).endswith("_mean"):
                continue
            if transcript_id in profiles:
                raise ValueError(f"Duplicate mean bias profile for {dataset}/{transcript_id}")
            profiles[transcript_id] = np.asarray(value, dtype=np.float64).reshape(-1)
    missing = sorted(requested_ids.difference(profiles))
    if missing:
        raise KeyError(f"Bias profile {dataset} is missing {len(missing)} IDs, e.g. {missing[:3]}")
    return profiles


def _load_unbiased_observation(
    *,
    depth: str,
    observed_counts_root: Path,
    requested_ids: set[str],
) -> dict[str, np.ndarray]:
    path = (
        observed_counts_root.expanduser().resolve()
        / depth
        / f"artificial_ground_truth_psite_counts_{depth}.parquet"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Unbiased finite-count reference does not exist: {path}")
    profiles: dict[str, np.ndarray] = {}
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=256,
        columns=["sample", "transcript_id", "rib_profile"],
    ):
        columns = batch.to_pydict()
        for sample, transcript_id, value in zip(
            columns["sample"], columns["transcript_id"], columns["rib_profile"]
        ):
            transcript_id = str(transcript_id)
            # The stored mean is the integerized arithmetic mean of the two
            # ground-truth replicas, not an independent third replicate.
            if str(sample) != "mean" or transcript_id not in requested_ids:
                continue
            profiles[transcript_id] = np.asarray(value, dtype=np.float64).reshape(-1)
    missing = sorted(requested_ids.difference(profiles))
    if missing:
        raise KeyError(
            f"Unbiased observation at {depth} is missing {len(missing)} IDs, e.g. {missing[:3]}"
        )
    return profiles


def _interior_mask(length: int, trim: int) -> np.ndarray:
    if trim < 0:
        raise ValueError("boundary-trim-codons must be non-negative.")
    mask = np.ones(length, dtype=bool)
    if trim:
        mask[:trim] = False
        mask[max(0, length - trim) :] = False
    return mask


def _profile_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    normalization_mask: np.ndarray,
    evaluation_mask: np.ndarray,
) -> tuple[float, float, int] | None:
    """Return PCC/RMSE after one common interior mean-one normalization."""
    length = min(
        prediction.size,
        reference.size,
        normalization_mask.size,
        evaluation_mask.size,
    )
    if length < 2:
        return None
    prediction = prediction[:length]
    reference = reference[:length]
    normalization = normalization_mask[:length].copy()
    normalization &= np.isfinite(prediction) & np.isfinite(reference)
    if int(normalization.sum()) < 2:
        return None
    prediction_mean = float(prediction[normalization].mean())
    reference_mean = float(reference[normalization].mean())
    if prediction_mean <= 0.0 or reference_mean <= 0.0:
        return None
    prediction = prediction / prediction_mean
    reference = reference / reference_mean
    valid = evaluation_mask[:length].copy()
    valid &= np.isfinite(prediction) & np.isfinite(reference)
    count = int(valid.sum())
    if count < 2:
        return None
    predicted_values = prediction[valid]
    reference_values = reference[valid]
    centered_prediction = predicted_values - predicted_values.mean()
    centered_reference = reference_values - reference_values.mean()
    denominator = float(
        np.sqrt(
            np.sum(centered_prediction**2)
            * np.sum(centered_reference**2)
        )
    )
    pcc = (
        float(np.sum(centered_prediction * centered_reference) / denominator)
        if denominator > 0.0
        else float("nan")
    )
    rmse = float(np.sqrt(np.mean((predicted_values - reference_values) ** 2)))
    return pcc, rmse, count


def _aggregate_profile_metrics(
    values: list[tuple[float, float, int]],
) -> dict[str, float | int]:
    if not values:
        return {"pcc": float("nan"), "rmse": float("nan"), "transcripts": 0}
    pcc = np.asarray([value[0] for value in values], dtype=np.float64)
    rmse = np.asarray([value[1] for value in values], dtype=np.float64)
    counts = np.asarray([value[2] for value in values], dtype=np.float64)
    finite_pcc = np.isfinite(pcc)
    if bool(finite_pcc.any()):
        fisher = np.arctanh(np.clip(pcc[finite_pcc], -1.0 + 1.0e-12, 1.0 - 1.0e-12))
        weights = np.maximum(counts[finite_pcc] - 3.0, 1.0)
        pooled_pcc = float(np.tanh(np.sum(weights * fisher) / np.sum(weights)))
    else:
        pooled_pcc = float("nan")
    return {
        "pcc": pooled_pcc,
        "rmse": float(np.nanmean(rmse)),
        "transcripts": int(len(values)),
    }


def build_read_depth_diagnostic(
    biological: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Calculate the N=2 depth diagnostic from selected prediction artifacts."""
    _require_columns(biological, {"run_id", "depth", "n_datasets", "pcc"}, "biological recovery")
    selected = biological.loc[
        pd.to_numeric(biological["n_datasets"], errors="coerce")
        == int(args.diagnostic_n_datasets)
    ].copy()
    if int(args.diagnostic_n_datasets) < 2:
        raise ValueError("The gauge-specific depth diagnostic requires at least two datasets.")
    duplicates = selected.duplicated("depth", keep=False)
    if duplicates.any() or set(selected["depth"].astype(str)) != set(DEPTH_ORDER):
        raise ValueError(
            "Read-depth diagnostic requires exactly one selected run at every depth; "
            f"got {selected[['depth', 'run_id']].to_dict('records')}."
        )

    results_root = args.results_root.expanduser().resolve()
    run_infos: list[dict[str, Any]] = []
    all_validation_ids: set[str] = set()
    for depth in DEPTH_ORDER:
        recovery_row = selected.loc[selected["depth"].astype(str) == depth].iloc[0]
        run_id = str(recovery_row["run_id"])
        direct = results_root / run_id
        run_dir = (
            direct
            if direct.is_dir()
            else _one_path(results_root.rglob(run_id), f"run {run_id}")
        )
        prediction_path = _one_path(
            run_dir.rglob("predictions_main_val_best_val_loss_*.parquet"),
            f"best-validation-loss prediction below {run_dir}",
        )
        config_path = _one_path(run_dir.rglob("config.yaml"), f"resolved config below {run_dir}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        datasets = [str(value) for value in config["experiment"]["dataset"]]
        reference_config = config["model"]["gamma_centering"]["reference"]
        reference_names = reference_config.get("dataset_names")
        references = (
            datasets
            if reference_names is None
            else [str(value) for value in reference_names]
        )
        if not references:
            raise ValueError(f"Run {run_id} has no gamma-centering reference datasets.")
        weighting = str(reference_config.get("weighting", "equal")).lower()
        quality_power = float(reference_config.get("quality_rank_power", 0.0))
        if weighting != "equal" and not np.isclose(quality_power, 0.0):
            raise ValueError(
                "The diagnostic oracle currently supports equal effective gamma-reference "
                f"weights; {run_id} uses weighting={weighting}, p={quality_power}."
            )
        predictions = _load_prediction_l_bio(prediction_path)
        validation_ids = set(predictions)
        all_validation_ids.update(validation_ids)
        run_infos.append(
            {
                "depth": depth,
                "run_id": run_id,
                "run_dir": run_dir,
                "datasets": datasets,
                "references": references,
                "predictions": predictions,
                "validation_ids": validation_ids,
                "reported_l_vs_k_pcc": float(recovery_row["pcc"]),
            }
        )

    latent = _load_latent_truth(args.latent_truth, all_validation_ids)
    unique_references = sorted(
        {dataset for info in run_infos for dataset in info["references"]}
    )
    bias_cache = {
        dataset: _load_added_bias(
            dataset=dataset,
            bias_root=args.bias_root,
            requested_ids=all_validation_ids,
            latent=latent,
        )
        for dataset in unique_references
    }

    rows: list[dict[str, Any]] = []
    for info in run_infos:
        depth = str(info["depth"])
        validation_ids = set(info["validation_ids"])
        observed = _load_unbiased_observation(
            depth=depth,
            observed_counts_root=args.observed_counts_root,
            requested_ids=validation_ids,
        )
        metric_values: dict[str, list[tuple[float, float, int]]] = {
            "l_vs_k": [],
            "l_vs_gauge_oracle": [],
            "l_vs_gauge_oracle_bias_sites": [],
            "gauge_oracle_vs_k": [],
            "l_vs_observed": [],
            "k_vs_observed": [],
        }
        reference_weight = 1.0 / len(info["references"])
        for transcript_id in sorted(validation_ids):
            arrays = [
                info["predictions"][transcript_id],
                latent[transcript_id],
                observed[transcript_id],
                *(bias_cache[dataset][transcript_id] for dataset in info["references"]),
            ]
            length = min(array.size for array in arrays)
            if length < 2:
                continue
            learned = info["predictions"][transcript_id][:length]
            kinetics = latent[transcript_id][:length]
            observation = observed[transcript_id][:length]
            added_biases = np.stack(
                [bias_cache[dataset][transcript_id][:length] for dataset in info["references"]]
            )
            gauge_oracle = kinetics * np.exp(
                reference_weight * np.log1p(added_biases).sum(axis=0)
            )
            interior = _interior_mask(length, int(args.boundary_trim_codons))
            bias_sites = interior & np.any(added_biases > 0.0, axis=0)
            comparisons = {
                "l_vs_k": (learned, kinetics, interior),
                "l_vs_gauge_oracle": (learned, gauge_oracle, interior),
                "l_vs_gauge_oracle_bias_sites": (learned, gauge_oracle, bias_sites),
                "gauge_oracle_vs_k": (gauge_oracle, kinetics, interior),
                "l_vs_observed": (learned, observation, interior),
                "k_vs_observed": (kinetics, observation, interior),
            }
            for name, (prediction, reference, evaluation_mask) in comparisons.items():
                result = _profile_metrics(
                    prediction,
                    reference,
                    normalization_mask=interior,
                    evaluation_mask=evaluation_mask,
                )
                if result is not None:
                    metric_values[name].append(result)

        summaries = {
            name: _aggregate_profile_metrics(values)
            for name, values in metric_values.items()
        }
        validation_hash = hashlib.sha256(
            "\n".join(sorted(validation_ids)).encode("utf-8")
        ).hexdigest()[:12]
        observed_ceiling = float(summaries["k_vs_observed"]["pcc"])
        learned_observed = float(summaries["l_vs_observed"]["pcc"])
        rows.append(
            {
                "run_id": info["run_id"],
                "depth": depth,
                "n_datasets": len(info["datasets"]),
                "datasets": ",".join(info["datasets"]),
                "validation_transcripts": len(validation_ids),
                "validation_id_hash": validation_hash,
                "reported_l_vs_k_pcc": info["reported_l_vs_k_pcc"],
                "l_vs_k_pcc": summaries["l_vs_k"]["pcc"],
                "l_vs_k_rmse": summaries["l_vs_k"]["rmse"],
                "l_vs_gauge_oracle_pcc": summaries["l_vs_gauge_oracle"]["pcc"],
                "l_vs_gauge_oracle_rmse": summaries["l_vs_gauge_oracle"]["rmse"],
                "l_vs_gauge_oracle_bias_sites_pcc": summaries[
                    "l_vs_gauge_oracle_bias_sites"
                ]["pcc"],
                "l_vs_gauge_oracle_bias_sites_rmse": summaries[
                    "l_vs_gauge_oracle_bias_sites"
                ]["rmse"],
                "gauge_oracle_vs_k_pcc": summaries["gauge_oracle_vs_k"]["pcc"],
                "gauge_oracle_vs_k_rmse": summaries["gauge_oracle_vs_k"]["rmse"],
                "l_vs_observed_pcc": learned_observed,
                "l_vs_observed_rmse": summaries["l_vs_observed"]["rmse"],
                "k_vs_observed_pcc": observed_ceiling,
                "k_vs_observed_rmse": summaries["k_vs_observed"]["rmse"],
                "observed_ceiling_ratio": (
                    learned_observed / observed_ceiling
                    if np.isfinite(observed_ceiling) and observed_ceiling != 0.0
                    else float("nan")
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["depth_order"] = result["depth"].map(
        {depth: index for index, depth in enumerate(DEPTH_ORDER)}
    )
    return result.sort_values("depth_order").drop(columns="depth_order").reset_index(drop=True)


def _pcc_limits(values: pd.Series) -> tuple[float, float]:
    """Use a tight, stable y-scale while retaining the physical PCC ceiling."""
    minimum = float(values.min())
    maximum = float(values.max())
    padding = max(0.0005, 0.14 * (maximum - minimum))
    lower = max(-1.0, np.floor((minimum - padding) * 1000.0) / 1000.0)
    upper = 1.0
    return lower, upper


def _plot_recovery_lines(
    axis: matplotlib.axes.Axes,
    frame: pd.DataFrame,
    *,
    title: str,
) -> list[matplotlib.lines.Line2D]:
    handles: list[matplotlib.lines.Line2D] = []
    for depth in DEPTH_ORDER:
        line = frame.loc[frame["depth"].astype(str) == depth].sort_values("n_datasets")
        if line.empty:
            continue
        style = DEPTH_STYLES[depth]
        handle = axis.plot(
            line["n_datasets"],
            line["pcc"],
            label=DEPTH_LABELS[depth],
            color=style["color"],
            marker=style["marker"],
            markeredgecolor="white",
            markeredgewidth=0.65,
            zorder=3,
        )[0]
        handles.append(handle)
    counts = sorted(frame["n_datasets"].unique())
    axis.set_title(title, loc="left", pad=7)
    axis.set_xlabel("Number of training datasets")
    axis.set_ylabel("Fisher-weighted transcript PCC")
    axis.set_xticks(counts)
    axis.set_xlim(min(counts) - 0.23, max(counts) + 0.23)
    axis.set_ylim(*_pcc_limits(frame["pcc"]))
    axis.grid(axis="y")
    axis.set_axisbelow(True)
    return handles


def _compact_bias_label(value: str) -> str:
    replacements = {
        "GC fraction > 0.7": "GC\n> 0.7",
        "AU fraction > 0.7": "AU\n> 0.7",
    }
    return replacements.get(value, value)


def _single_dataset_bias_order_and_labels(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    if "task_index" in frame.columns:
        order = (
            frame.groupby("bias", sort=False)["task_index"]
            .min()
            .sort_values()
            .index.tolist()
        )
    else:
        order = sorted(frame["bias"].unique())
    labels = (
        frame.drop_duplicates("bias")
        .set_index("bias")
        .loc[order, "bias_label"]
        .astype(str)
        .map(_compact_bias_label)
        .tolist()
    )
    return order, labels


def _plot_single_dataset_dots(axis: matplotlib.axes.Axes, frame: pd.DataFrame) -> None:
    """Show bias-to-bias variation clearly with one horizontal PCC point per depth."""
    order, labels = _single_dataset_bias_order_and_labels(frame)
    y = np.arange(len(order), dtype=float)
    offsets = (-0.18, 0.0, 0.18)
    for index, depth in enumerate(DEPTH_ORDER):
        rows = (
            frame.loc[frame["recovery_depth"] == depth]
            .set_index("bias")
            .reindex(order)
        )
        heights = rows["mean_mu_pcc"].to_numpy(dtype=float)
        lower = rows["bootstrap_ci95_low"].to_numpy(dtype=float)
        upper = rows["bootstrap_ci95_high"].to_numpy(dtype=float)
        errors = np.vstack(
            (np.maximum(heights - lower, 0.0), np.maximum(upper - heights, 0.0))
        )
        style = DEPTH_STYLES[depth]
        axis.errorbar(
            heights,
            y + offsets[index],
            xerr=errors,
            fmt=style["marker"],
            color=DEPTH_STYLES[depth]["color"],
            markersize=5.6,
            markeredgecolor="white",
            markeredgewidth=0.7,
            elinewidth=0.85,
            capsize=1.9,
            zorder=3,
        )
    minimum = float(frame["bootstrap_ci95_low"].min())
    maximum = float(frame["bootstrap_ci95_high"].max())
    axis.set_title("c  Single-dataset $\\mu$ agreement", loc="left", pad=7)
    axis.set_xlabel("Mean validation PCC (95% CI)")
    axis.set_ylabel("Synthetic artificial bias")
    axis.set_yticks(y, labels)
    axis.tick_params(axis="y", labelsize=7.8, pad=2)
    axis.invert_yaxis()
    axis.set_xlim(
        max(0.0, np.floor((minimum - 0.035) * 20.0) / 20.0),
        min(1.0, np.ceil((maximum + 0.035) * 20.0) / 20.0),
    )
    axis.grid(axis="x")
    axis.set_axisbelow(True)


def build_figure(
    biological: pd.DataFrame,
    gamma: pd.DataFrame,
    single_dataset: pd.DataFrame,
):
    """Return the paper-ready, one-row/three-column Matplotlib figure."""
    with matplotlib.rc_context(ICLR_RC):
        fig = plt.figure(figsize=(13.45, 3.72), layout="constrained")
        grid = fig.add_gridspec(
            2,
            3,
            height_ratios=(0.14, 1.0),
            width_ratios=(1.0, 1.0, 1.38),
        )
        legend_axis = fig.add_subplot(grid[0, :])
        axes = [
            fig.add_subplot(grid[1, index])
            for index in range(3)
        ]
        handles = _plot_recovery_lines(
            axes[0], biological, title="a  Biological-profile recovery"
        )
        _plot_recovery_lines(axes[1], gamma, title="b  Bias-profile recovery")
        _plot_single_dataset_dots(axes[2], single_dataset)
        legend_axis.axis("off")
        legend_axis.legend(
            handles=handles,
            labels=[handle.get_label() for handle in handles],
            title="Read depth",
            ncol=3,
            loc="center",
            handlelength=2.1,
            columnspacing=1.7,
        )
        return fig


def _plot_diagnostic_series(
    axis: matplotlib.axes.Axes,
    frame: pd.DataFrame,
    series: tuple[tuple[str, str, str, str], ...],
) -> None:
    """Draw a small categorical depth panel with consistent visual grammar."""
    x = np.arange(len(DEPTH_ORDER), dtype=float)
    for column, label, style_name, linestyle in series:
        values = frame.set_index("depth").loc[list(DEPTH_ORDER), column].to_numpy(dtype=float)
        style = DIAGNOSTIC_STYLES[style_name]
        axis.plot(
            x,
            values,
            color=style["color"],
            marker=style["marker"],
            linestyle=linestyle,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=label,
            zorder=3,
        )
    axis.set_xticks(x, [DIAGNOSTIC_DEPTH_LABELS[depth] for depth in DEPTH_ORDER])
    axis.set_xlim(-0.16, len(DEPTH_ORDER) - 0.84)
    axis.set_xlabel("Read depth (reads/codon)")
    axis.grid(axis="y")
    axis.set_axisbelow(True)
    axis.legend(loc="best", handlelength=2.25)


def build_read_depth_diagnostic_figure(
    diagnostic: pd.DataFrame,
    args: argparse.Namespace,
):
    """Return the companion figure separating estimation, gauge, and count noise."""
    with matplotlib.rc_context(ICLR_RC):
        fig = plt.figure(figsize=(13.45, 4.05), layout="constrained")
        grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 0.16))
        axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
        note_axis = fig.add_subplot(grid[1, :])

        _plot_diagnostic_series(
            axes[0],
            diagnostic,
            (
                (
                    "l_vs_gauge_oracle_rmse",
                    "All interior codons",
                    "learned",
                    "-",
                ),
                (
                    "l_vs_gauge_oracle_bias_sites_rmse",
                    "Programmed-bias codons",
                    "bias_sites",
                    "--",
                ),
            ),
        )
        axes[0].set_title("a  Low-depth gauge-target error is larger", loc="left", pad=7)
        axes[0].set_ylabel("Mean transcript shape RMSE\n(lower is better)")
        rmse_columns = [
            "l_vs_gauge_oracle_rmse",
            "l_vs_gauge_oracle_bias_sites_rmse",
        ]
        rmse_maximum = float(diagnostic[rmse_columns].max().max())
        axes[0].set_ylim(0.0, max(0.01, 1.18 * rmse_maximum))

        _plot_diagnostic_series(
            axes[1],
            diagnostic,
            (
                ("l_vs_k_pcc", "Learned $L_{\\rm bio}$ vs. latent $K$", "learned", "-"),
                (
                    "gauge_oracle_vs_k_pcc",
                    "Equal-gauge target $L^*_{\\rm eq}$ vs. $K$",
                    "oracle",
                    "--",
                ),
            ),
        )
        axes[1].set_title("b  Latent recovery is nearly depth-invariant", loc="left", pad=7)
        axes[1].set_ylabel("Fisher-weighted transcript PCC")
        axes[1].set_ylim(0.0, 1.0)

        _plot_diagnostic_series(
            axes[2],
            diagnostic,
            (
                (
                    "l_vs_observed_pcc",
                    "Learned $L_{\\rm bio}$ vs. observed counts",
                    "learned",
                    "-",
                ),
                (
                    "k_vs_observed_pcc",
                    "Latent $K$ vs. counts (noise benchmark)",
                    "oracle",
                    "--",
                ),
            ),
        )
        axes[2].set_title("c  Finite-count agreement rises with depth", loc="left", pad=7)
        axes[2].set_ylabel("Fisher-weighted transcript PCC")
        axes[2].set_ylim(0.0, 1.0)

        fig.suptitle(
            f"Read-depth diagnostic: N={args.diagnostic_n_datasets}, equal gamma reference, "
            f"seed {args.seed}",
            fontsize=10.5,
            fontweight="semibold",
        )
        note_axis.axis("off")
        different_validation_panels = diagnostic["validation_id_hash"].nunique() > 1
        validation_note = (
            "Historical depth runs use different validation panels; comparisons are "
            "descriptive (one seed), not a controlled depth effect. "
            if different_validation_panels
            else "All depth runs use the same validation panel; only one seed is shown. "
        )
        note_axis.text(
            0.0,
            0.58,
            validation_note
            + "Panel c deliberately uses the matched-depth unbiased count profile, so both "
            "training depth and reference noise change. $L^*_{\\rm eq}$ is the latent profile "
            "under the model's equal-reference gamma gauge.",
            ha="left",
            va="center",
            fontsize=7.6,
            color="#444444",
            wrap=True,
        )
        return fig


def main() -> None:
    args = parse_args()
    recovery = select_biological_recovery(
        _read_csv(args.recovery_summary, "Biological recovery summary"), args
    )
    gamma = select_gamma_recovery(
        _read_csv(args.gamma_summary, "Gamma recovery summary"), args
    )
    single_dataset = select_single_dataset_mu(
        _read_csv(args.single_dataset_summary, "Single-dataset mu summary")
    )
    figure = build_figure(recovery, gamma, single_dataset)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in args.formats:
        output = output_dir / f"{args.output_stem}.{suffix}"
        figure.savefig(output, bbox_inches="tight")
        outputs.append(output)
    plt.close(figure)

    diagnostic_outputs: list[Path] = []
    diagnostic_table: Path | None = None
    if not args.skip_read_depth_diagnostic:
        diagnostic = build_read_depth_diagnostic(recovery, args)
        diagnostic_table = output_dir / f"{args.diagnostic_output_stem}.tsv"
        diagnostic.to_csv(diagnostic_table, sep="\t", index=False, float_format="%.8g")
        diagnostic_figure = build_read_depth_diagnostic_figure(diagnostic, args)
        for suffix in args.formats:
            output = output_dir / f"{args.diagnostic_output_stem}.{suffix}"
            diagnostic_figure.savefig(output, bbox_inches="tight")
            diagnostic_outputs.append(output)
        plt.close(diagnostic_figure)

    print("Wrote ICLR synthetic-recovery figure:")
    for output in outputs:
        print(output)
    print(
        f"Biological points: {len(recovery)}; gamma points: {len(gamma)}; "
        f"single-dataset PCC cells: {len(single_dataset)}."
    )
    if diagnostic_table is not None:
        print("Wrote N=2 read-depth diagnostic:")
        for output in diagnostic_outputs:
            print(output)
        print(diagnostic_table)


if __name__ == "__main__":
    main()
