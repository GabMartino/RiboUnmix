"""Common result-bundle discovery and profile statistics for synthetic reports.

The synthetic launchers have changed the on-disk layout a few times.  Current
bundles live below ``results/riboai_synthetic_experiments/<run>`` and contain a
resolved ``config.yaml`` plus prediction files named
``predictions_main_val_best_val_loss_*.parquet`` and/or
``predictions_main_val_best_pcc_*.parquet``.  This module deliberately treats
the resolved files as the source of truth and also handles bundles copied from
another host (whose manifests contain stale absolute paths).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPOSITORY_ROOT / "results" / "riboai_synthetic_experiments"
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "analyses" / "artifacts" / "synthetic" / "gamma_ablation"
)
DEFAULT_CONFIG = REPOSITORY_ROOT / "config" / "config_ribounmix_synthetic.yaml"
DEFAULT_DATASET_ENCODING = REPOSITORY_ROOT / "Datasets" / "encodings" / "synthetic_dataset_encoding.yaml"


def load_config(path: Path | str) -> dict[str, Any]:
    """Load a resolved or base YAML configuration and validate its shape."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Synthetic config does not exist: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return value


def load_dataset_names(path: Path | str = DEFAULT_DATASET_ENCODING) -> dict[int, str]:
    """Return the persisted integer-ID to dataset-name mapping."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    if not path.is_file():
        # Some copied resolved configs refer to a temporary encoding file.  The
        # repository synthetic encoding is stable and is the canonical fallback.
        path = DEFAULT_DATASET_ENCODING
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected dataset-name mapping in {path}.")
    output: dict[int, str] = {}
    for name, identifier in value.items():
        try:
            output[int(identifier)] = str(name)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid dataset ID for {name!r}: {identifier!r}") from exc
    return output


def load_codon_to_amino_acid(path: Path | str | None = None) -> dict[int, str]:
    """Load the codon-ID to amino-acid map used by the biological plots.

    The project encoding is a codon-name to integer map and the codon2aa YAML
    is a codon-name to amino-acid map, so the two are joined when available.
    Unknown IDs are left unmapped rather than fabricating a residue.
    """
    codon_path = REPOSITORY_ROOT / "Datasets" / "encodings" / "codon_encoding.yaml"
    aa_path = REPOSITORY_ROOT / "Datasets" / "encodings" / "codon2aa.yaml"
    if path is not None:
        candidate = Path(path).expanduser()
        if candidate.suffix in {".yaml", ".yml"}:
            aa_path = candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate
    try:
        codon_value = yaml.safe_load(codon_path.read_text(encoding="utf-8"))
        aa_value = yaml.safe_load(aa_path.read_text(encoding="utf-8"))
    except OSError:
        return {}
    if not isinstance(codon_value, dict) or not isinstance(aa_value, dict):
        return {}
    # Encodings in this repository are normally {token: integer}; tolerate the
    # inverse representation as well because old generated tables used it.
    codon_to_id: dict[str, int] = {}
    for token, identifier in codon_value.items():
        try:
            if isinstance(identifier, (int, float)):
                codon_to_id[str(token).upper()] = int(identifier)
            elif isinstance(token, (int, float)):
                codon_to_id[str(identifier).upper()] = int(token)
        except (TypeError, ValueError):
            continue
    mapping: dict[int, str] = {}
    for codon, amino_acid in aa_value.items():
        try:
            identifier = codon_to_id.get(str(codon).upper())
            if identifier is not None:
                mapping[identifier] = str(amino_acid)
        except (TypeError, ValueError):
            continue
    return mapping


def array_or_none(value: Any, dtype: Any = np.float64) -> np.ndarray | None:
    """Convert an Arrow/Pandas list cell to a one-dimensional NumPy array."""
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=dtype).reshape(-1)
    except (TypeError, ValueError):
        return None
    return array


def masked_array(value: Any, mask: Any | None = None) -> np.ndarray | None:
    """Return finite values selected by a profile mask, or ``None``."""
    values = array_or_none(value)
    if values is None:
        return None
    if mask is None:
        selected = np.ones(values.size, dtype=bool)
    else:
        raw_mask = array_or_none(mask, dtype=bool)
        if raw_mask is None:
            return None
        selected = np.zeros(values.size, dtype=bool)
        selected[: min(values.size, raw_mask.size)] = raw_mask[: values.size]
    selected &= np.isfinite(values)
    return values[selected]


def normalize_profile(value: Any, mask: Any | None = None) -> np.ndarray | None:
    """Mean-normalize a profile while retaining only valid codons.

    The returned vector is compact (invalid/masked positions are omitted),
    matching the metrics used by the numbered reports.
    """
    values = array_or_none(value)
    if values is None:
        return None
    if mask is None:
        selected = np.ones(values.size, dtype=bool)
    else:
        raw_mask = array_or_none(mask, dtype=bool)
        if raw_mask is None:
            return None
        selected = np.zeros(values.size, dtype=bool)
        selected[: min(values.size, raw_mask.size)] = raw_mask[: values.size]
    selected &= np.isfinite(values)
    selected_values = values[selected]
    if selected_values.size < 2:
        return None
    mean = float(selected_values.mean())
    if not math.isfinite(mean) or mean <= 0.0:
        return None
    return selected_values / mean


def paired_arrays(row: dict[str, Any], component: str, reference: str | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract two identically masked vectors from a prediction row."""
    if reference is None or component not in row or reference not in row:
        return None
    prediction = array_or_none(row.get(component))
    truth = array_or_none(row.get(reference))
    if prediction is None or truth is None:
        return None
    length = min(prediction.size, truth.size)
    prediction = prediction[:length]
    truth = truth[:length]
    mask = row.get("mask")
    if mask is None:
        valid = np.ones(length, dtype=bool)
    else:
        raw_mask = array_or_none(mask, dtype=bool)
        if raw_mask is None:
            return None
        valid = np.zeros(length, dtype=bool)
        valid[: min(length, raw_mask.size)] = raw_mask[:length]
    valid &= np.isfinite(prediction) & np.isfinite(truth)
    if int(valid.sum()) < 2:
        return None
    return prediction[valid], truth[valid]


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size != x.size:
        return float("nan")
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    if float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def profile_metrics(prediction: Any, reference: Any) -> dict[str, float | int]:
    """Calculate shape PCC, rank correlation, and mean-one RMSE."""
    x = np.asarray(prediction, dtype=np.float64).reshape(-1)
    y = np.asarray(reference, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2:
        return {"pearson": np.nan, "spearman": np.nan, "shape_rmse": np.nan, "n_positions": int(x.size)}
    x_mean, y_mean = float(x.mean()), float(y.mean())
    x_norm = x / x_mean if x_mean > 0.0 else x
    y_norm = y / y_mean if y_mean > 0.0 else y
    residual = x_norm - y_norm
    pearson = _correlation(x_norm, y_norm)
    # Avoid a SciPy dependency for this lightweight report: average ranks with
    # pandas' tie semantics, then correlate those ranks.
    spearman = _correlation(pd.Series(x_norm).rank(method="average").to_numpy(), pd.Series(y_norm).rank(method="average").to_numpy())
    return {
        "pearson": pearson,
        "spearman": spearman,
        "shape_rmse": float(np.sqrt(np.mean(residual**2))),
        "n_positions": int(x.size),
    }


def bootstrap_summary(values: Iterable[Any], *, n_bootstrap: int = 500, seed: int = 42) -> dict[str, float | int]:
    """Summarize finite values and a deterministic percentile bootstrap CI."""
    array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "std": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    mean = float(array.mean())
    median = float(np.median(array))
    std = float(array.std(ddof=1)) if array.size > 1 else np.nan
    replicates = max(0, int(n_bootstrap))
    if replicates and array.size:
        rng = np.random.default_rng(seed)
        # Chunking avoids a potentially huge (replicates × observations) matrix.
        means = np.empty(replicates, dtype=np.float64)
        for index in range(replicates):
            means[index] = float(array[rng.integers(0, array.size, array.size)].mean())
        ci_low, ci_high = (float(v) for v in np.quantile(means, [0.025, 0.975]))
    else:
        ci_low = ci_high = mean
    return {"n": int(array.size), "mean": mean, "median": median, "std": std, "ci_low": ci_low, "ci_high": ci_high}


def fisher_summary(values: Iterable[Any], n_positions: Iterable[Any] | None = None) -> dict[str, float]:
    """Combine per-profile correlations with Fisher's z transform.

    If profile lengths are provided, ``n-3`` is used as the standard Fisher
    weight. The function also returns the ordinary mean for transparent
    downstream tables.
    """
    correlations = np.asarray(list(values), dtype=np.float64)
    valid = np.isfinite(correlations)
    correlations = np.clip(correlations[valid], -0.999999, 0.999999)
    if n_positions is None:
        weights = np.ones(correlations.size, dtype=np.float64)
    else:
        lengths = np.asarray(list(n_positions), dtype=np.float64)
        weights = np.maximum(lengths[valid] - 3.0, 1.0) if lengths.size == valid.size else np.ones(correlations.size)
    if correlations.size == 0:
        return {"fisher": np.nan, "mean": np.nan, "n": 0}
    z = np.arctanh(correlations)
    fisher = float(np.tanh(np.average(z, weights=weights)))
    return {"fisher": fisher, "mean": float(correlations.mean()), "n": int(correlations.size)}


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson binomial interval used by CSS summaries."""
    n = int(total)
    k = int(successes)
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def resolve_reference(columns: Iterable[str], candidates: Sequence[str] | None) -> tuple[str | None, str]:
    """Resolve a configured reference column and label its provenance."""
    available = set(columns)
    if isinstance(candidates, str):
        candidates = [candidates]
    for candidate in candidates or ():
        if candidate in available:
            lowered = candidate.lower()
            kind = "latent_ground_truth" if any(token in lowered for token in ("true", "ground_truth", "latent")) else "observed_profile_proxy"
            return candidate, kind
    return None, "unavailable"


def iter_rows(prediction: "PredictionFile", requested_columns: Iterable[str], *, batch_rows: int = 256) -> Iterable[dict[str, Any]]:
    """Stream rows from a prediction Parquet without loading the whole file."""
    columns = [column for column in requested_columns if column in prediction.columns]
    parquet = pq.ParquetFile(prediction.path)
    for batch in parquet.iter_batches(batch_size=max(1, int(batch_rows)), columns=columns):
        values = batch.to_pydict()
        for index in range(batch.num_rows):
            yield {column: values[column][index] for column in columns}


@dataclass(frozen=True)
class PredictionFile:
    path: Path
    split: str
    experiment: str
    checkpoint_variant: str
    columns: tuple[str, ...]


@dataclass
class RunRecord:
    run_id: str
    path: Path
    config_path: Path
    config: dict[str, Any]
    usable_files: tuple[PredictionFile, ...] = field(default_factory=tuple)

    @property
    def n_datasets(self) -> int:
        datasets = self.config.get("experiment", {}).get("dataset", [])
        if isinstance(datasets, (list, tuple)):
            return len(datasets)
        return 0 if datasets in (None, "") else 1

    @property
    def training_scope(self) -> str:
        return "multi_dataset" if self.n_datasets > 1 else "individual_dataset"

    @property
    def strategy(self) -> str:
        name = self.run_id.lower()
        explicit = self.config.get("data", {}).get("train_sampling_strategy")
        # Keep the experiment family visible; this is more informative than
        # treating every current ``within`` run as a historical rank-stratified
        # ablation.
        if "inter_" in name:
            return "inter"
        if "top_quality" in name or "topquality" in name:
            return "top_quality"
        if "rank_stratified" in name or "rankstratified" in name:
            return "rank_stratified"
        if name.startswith("riboai_synthetic_within_"):
            return "within"
        return str(explicit or "unknown")

    @property
    def feature_preset(self) -> str:
        text = self.run_id
        match = re.search(r"FeatPreset([^_]+)", text)
        if match:
            return match.group(1) or "unknown"
        value = self.config.get("model", {}).get("feature_preset")
        return str(value) if value not in (None, "") else "Baseline"

    @property
    def quality_rank_power(self) -> float:
        value = self.config.get("model", {}).get("gamma_centering", {}).get("reference", {}).get("quality_rank_power", 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    @property
    def gamma_weighting(self) -> str:
        value = self.config.get("model", {}).get("gamma_centering", {}).get("reference", {}).get("weighting", "equal")
        return str(value)

    @property
    def seed(self) -> int:
        try:
            return int(self.config.get("experiment", {}).get("seed", 0))
        except (TypeError, ValueError):
            return 0

    def metadata(self) -> dict[str, Any]:
        datasets = self.config.get("experiment", {}).get("dataset", [])
        if not isinstance(datasets, (list, tuple)):
            datasets = [] if datasets in (None, "") else [datasets]
        return {
            "run_id": self.run_id,
            "strategy": self.strategy,
            "training_scope": self.training_scope,
            "n_datasets": self.n_datasets,
            "quality_rank_power": self.quality_rank_power,
            "gamma_weighting": self.gamma_weighting,
            "feature_preset": self.feature_preset,
            "seed": self.seed,
            "depth": _depth_from_config(self.config),
            "mass_condition": _mass_condition(self.config, self.run_id),
            "datasets": ",".join(map(str, datasets)),
        }


def _mass_condition(config: dict[str, Any], run_id: str) -> str:
    value = config.get("model", {}).get("mass_conservation")
    enabled = ("massfree" not in run_id.lower()) if value is None else bool(value)
    return "mass_conserved" if enabled else "mass_free"


def _depth_from_config(config: dict[str, Any]) -> str:
    datasets = config.get("experiment", {}).get("dataset", [])
    if not isinstance(datasets, (list, tuple)):
        datasets = [] if datasets in (None, "") else [datasets]
    suffixes = ("_0p25_per_codon", "_2_per_codon", "_20_per_codon")
    depths = {
        suffix.removeprefix("_").removesuffix("_per_codon")
        for dataset in datasets
        for suffix in suffixes
        if str(dataset).endswith(suffix)
    }
    if len(depths) == 1:
        return f"{next(iter(depths))}_per_codon"
    if len(depths) > 1:
        return "cross_depth"
    name = str(config.get("dataset_config", {}).get("_name_", "unknown"))
    return name.removeprefix("synthetic_")


def _prediction_split_variant(path: Path) -> tuple[str, str]:
    name = path.name
    match = re.match(r"predictions_(?P<split>.+?)_(?P<variant>best_val_loss|best_pcc)(?:_|\.).*\.parquet$", name)
    if match:
        return match.group("split"), match.group("variant")
    match = re.match(r"predictions_(?P<split>.+?)_.*\.parquet$", name)
    if match:
        return match.group("split"), "legacy"
    return "unknown", "legacy"


def _prediction_experiment(path: Path, run_path: Path) -> str:
    try:
        relative = path.relative_to(run_path)
        parts = relative.parts
        if parts and parts[0] in {"results", "predictions"}:
            parts = parts[1:]
        return parts[0] if parts else path.parent.name
    except ValueError:
        return path.parent.name


def _prediction_files(run_path: Path) -> list[PredictionFile]:
    candidates = sorted(run_path.rglob("predictions_*.parquet"))
    if not candidates:
        return []
    parsed: list[tuple[Path, str, str]] = []
    for path in candidates:
        split, variant = _prediction_split_variant(path)
        parsed.append((path, split, variant))
    # One artifact per split is exposed. Prefer the explicit minimum-loss
    # artifact, then best-PCC, then a legacy unsuffixed artifact. This prevents
    # numbered reports from double-counting the same validation rows when a run
    # exports both checkpoint variants.
    selected: list[PredictionFile] = []
    for split in sorted({split for _, split, _ in parsed}):
        group = [(path, variant) for path, candidate_split, variant in parsed if candidate_split == split]
        priority = {"best_val_loss": 0, "best_pcc": 1, "legacy": 2}
        group.sort(key=lambda item: (priority.get(item[1], 9), str(item[0])))
        path, variant = group[0]
        try:
            columns = tuple(pq.read_schema(path).names)
        except Exception:
            continue
        selected.append(PredictionFile(path, split, _prediction_experiment(path, run_path), variant, columns))
    return selected


def _config_for_run(run_path: Path) -> Path | None:
    configs = sorted(run_path.rglob("config.yaml"))
    if not configs:
        return None
    # Current bundles have one resolved config. If a copied bundle contains
    # multiple versions, choose the newest config under logs and retain the
    # ambiguity in inventory rather than failing all other runs.
    return max(configs, key=lambda path: path.stat().st_mtime)


def discover_runs(results_root: Path | str) -> list[RunRecord]:
    """Discover current synthetic run directories under one result root."""
    root = Path(results_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Synthetic results root does not exist: {root}")
    records: list[RunRecord] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not path.name.startswith("riboai_synthetic_"):
            continue
        config_path = _config_for_run(path)
        if config_path is None:
            continue
        try:
            config = load_config(config_path)
        except (OSError, ValueError, yaml.YAMLError):
            continue
        records.append(RunRecord(path.name, path, config_path, config, tuple(_prediction_files(path))))
    return records


def select_latest_runs(runs: Iterable[RunRecord]) -> list[RunRecord]:
    """Keep the newest run for duplicate condition signatures.

    Run names encode timestamps, but filesystem modification time is more
    reliable after result trees have been copied. The full dataset list is part
    of the signature so cumulative panels are never collapsed together.
    """
    selected: dict[tuple[Any, ...], RunRecord] = {}
    for run in runs:
        datasets = run.config.get("experiment", {}).get("dataset", [])
        if not isinstance(datasets, (list, tuple)):
            datasets = [datasets]
        key = (
            run.strategy,
            tuple(map(str, datasets)),
            run.quality_rank_power,
            run.gamma_weighting,
            run.feature_preset,
            run.seed,
            _depth_from_config(run.config),
            _mass_condition(run.config, run.run_id),
        )
        previous = selected.get(key)
        # A partially copied newer folder should not hide an older completed
        # run with the same condition signature.  Completion therefore wins
        # before timestamp; only then do we select the newest copy.
        current_key = (bool(run.usable_files), run.path.stat().st_mtime, run.run_id)
        previous_key = (
            bool(previous.usable_files),
            previous.path.stat().st_mtime,
            previous.run_id,
        ) if previous is not None else None
        if previous is None or current_key > previous_key:
            selected[key] = run
    return sorted(selected.values(), key=lambda run: run.run_id)


def filter_runs(
    runs: Iterable[RunRecord],
    *,
    strategies: set[str] | None = None,
    feature_presets: set[str] | None = None,
    seeds: set[int] | None = None,
    dataset_counts: set[int] | None = None,
    quality_powers: set[float] | None = None,
    max_runs: int | None = None,
) -> list[RunRecord]:
    output = []
    for run in runs:
        if not run.usable_files:
            continue
        if strategies and run.strategy not in strategies:
            continue
        if feature_presets and run.feature_preset not in feature_presets:
            continue
        if seeds and run.seed not in seeds:
            continue
        if dataset_counts and run.n_datasets not in dataset_counts:
            continue
        if quality_powers and not any(math.isclose(run.quality_rank_power, float(value), rel_tol=0.0, abs_tol=1e-9) for value in quality_powers):
            continue
        output.append(run)
    output.sort(key=lambda run: run.run_id)
    return output[: int(max_runs)] if max_runs is not None and max_runs >= 0 else output


def inventory_dataframe(all_runs: Iterable[RunRecord], selected: Iterable[RunRecord]) -> pd.DataFrame:
    selected_ids = {run.run_id for run in selected}
    rows: list[dict[str, Any]] = []
    for run in all_runs:
        rows.append(
            {
                **run.metadata(),
                "run_path": str(run.path),
                "config_path": str(run.config_path),
                "prediction_files": len(run.usable_files),
                "prediction_variants": ",".join(file.checkpoint_variant for file in run.usable_files),
                "selected": run.run_id in selected_ids,
            }
        )
    return pd.DataFrame(rows)
