from __future__ import annotations

import math
import re
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ablation_utils import (
    MIX_EXPERIMENT,
    metadata_from_prediction_path,
    run_type_for_experiment,
)

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


COMPONENT_COLUMNS = {
    "mu": "mu",
    "L_bio": "L_bio",
}
# Components compared in the configured all-vs-individual analysis, and their
# axis labels. L_bio is the shared, dataset-blind biological signal: tracking its
# per-dataset PCC under joint (all-dataset) vs single-dataset (individual)
# training is the direct test for whether the shared branch loses information
# ("collapses") when it has to serve every dataset at once.
CONFIGURED_COMPONENTS = ("mu", "L_bio")
COMPONENT_AXIS_LABELS = {
    "mu": "μ",
    "L_bio": "L_bio",
}
FEATURE_PRESET_ORDER = ("baseline", "biological", "dataset_bias", "both", "core_bio")
FEATURE_PRESET_LABELS = {
    "baseline": "baseline",
    "biological": "biological",
    "dataset_bias": "dataset bias",
    "both": "both",
    "core_bio": "core bio",
}
CONFIGURED_RUN_PATTERN = re.compile(
    r"^riboai_(?P<training_mode>all|individual)_(?P<feature_preset>.+)"
    r"_seed(?P<seed>[^_]+)_(?P<job_id>[^_]+)$"
)
PREDICTION_PATTERNS = (
    "predictions_*.parquet",
    "comprehensive_predictions_rank*.parquet",
)
COMBINED_VALIDATION_SPLIT = "main_css_val"
COMBINED_VALIDATION_SOURCE_SPLITS = ("main_val", "css_benchmark")
COMBINED_VALIDATION_SPLIT_ORDER = {
    split: idx for idx, split in enumerate(COMBINED_VALIDATION_SOURCE_SPLITS)
}
TRANSCRIPT_KEY_CANDIDATES = (
    ("dataset_id", "transcript_id"),
    ("transcript_id",),
    ("ids",),
)

SINGLE_RUN_TYPE = "single dataset"
MIX_RUN_TYPE = "30-dataset mix"
RUN_TYPE_ORDER = (SINGLE_RUN_TYPE, MIX_RUN_TYPE)
RUN_TYPE_COLORS = {
    SINGLE_RUN_TYPE: "#4C78A8",
    MIX_RUN_TYPE: "#F58518",
}
MIX_ABLATION_COLORS = (
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
    "#4C78A8",
    "#EECA3B",
    "#8CD17D",
    "#B6992D",
    "#499894",
    "#D37295",
    "#FABFD2",
    "#79706E",
)


def sequence_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None

    if isinstance(value, float) and np.isnan(value):
        return None

    try:
        arr = np.asarray(value)
    except Exception:
        return None

    if arr.ndim == 0:
        if pd.isna(arr.item()):
            return None
        arr = arr.reshape(1)

    try:
        arr = arr.astype(np.float64, copy=False).reshape(-1)
    except (TypeError, ValueError):
        return None

    return arr if arr.size > 0 else None


def safe_pearsonr(x: Any, y: Any, eps: float = 1.0e-12) -> float:
    x_arr = sequence_or_none(x)
    y_arr = sequence_or_none(y)
    if x_arr is None or y_arr is None:
        return np.nan

    length = min(x_arr.size, y_arr.size)
    if length < 4:
        return np.nan

    x_arr = x_arr[:length]
    y_arr = y_arr[:length]
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    if x_arr.size < 4:
        return np.nan

    if np.std(x_arr) <= eps or np.std(y_arr) <= eps:
        return np.nan

    x_arr = x_arr - x_arr.mean()
    y_arr = y_arr - y_arr.mean()
    denom = np.sqrt(np.sum(x_arr**2) * np.sum(y_arr**2))
    if denom <= eps:
        return np.nan

    return float(np.sum(x_arr * y_arr) / denom)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Tie-aware average ranks (matches scipy.stats.rankdata 'average')."""
    a = np.asarray(a, dtype=np.float64)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size)
    a_sorted = a[sorter]
    obs = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))
    dense = obs.cumsum()[inv]
    count = np.concatenate((np.flatnonzero(obs), [a.size]))
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def safe_spearmanr(x: Any, y: Any, eps: float = 1.0e-12) -> float:
    """Spearman rank correlation = Pearson of tie-aware ranks."""
    x_arr = sequence_or_none(x)
    y_arr = sequence_or_none(y)
    if x_arr is None or y_arr is None:
        return np.nan

    length = min(x_arr.size, y_arr.size)
    if length < 4:
        return np.nan

    x_arr = x_arr[:length]
    y_arr = y_arr[:length]
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(valid.sum()) < 4:
        return np.nan

    return safe_pearsonr(_rankdata(x_arr[valid]), _rankdata(y_arr[valid]), eps=eps)


def wilcoxon_signed_rank_p(deltas: np.ndarray, eps: float = 1.0e-12) -> float:
    """Two-sided Wilcoxon signed-rank p-value.

    Uses scipy when available; otherwise a normal approximation with tie and
    zero handling. Returns NaN when there is no usable signed sample.
    """
    d = np.asarray(deltas, dtype=np.float64)
    d = d[np.isfinite(d) & (np.abs(d) > eps)]
    n = int(d.size)
    if n < 1:
        return np.nan

    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        pass

    ranks = _rankdata(np.abs(d))
    w_plus = float(ranks[d > 0].sum())
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w <= eps:
        return np.nan
    z = (w_plus - mean_w) / math.sqrt(var_w)
    return float(2.0 * (1.0 - NormalDist().cdf(abs(z))))


def empty_metrics() -> dict[str, float]:
    return {
        "pcc": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "n_transcripts": 0,
        "median_pcc": np.nan,
        "unweighted_mean_pcc": np.nan,
    }


def fisher_weighted_pcc(pccs: list[float], lengths: list[int]) -> dict[str, float]:
    if len(pccs) == 0:
        return empty_metrics()

    pcc_arr = np.asarray(pccs, dtype=np.float64)
    length_arr = np.asarray(lengths, dtype=np.float64)
    z = np.arctanh(np.clip(pcc_arr, -0.9999, 0.9999))
    weights = np.maximum(length_arr - 3.0, 1.0)

    z_mean = np.average(z, weights=weights)
    z_se = 1.0 / np.sqrt(np.sum(weights))
    z_crit = NormalDist().inv_cdf(0.975)

    return {
        "pcc": float(np.tanh(z_mean)),
        "ci_lower": float(np.tanh(z_mean - z_crit * z_se)),
        "ci_upper": float(np.tanh(z_mean + z_crit * z_se)),
        "n_transcripts": int(len(pccs)),
        "median_pcc": float(np.median(pcc_arr)),
        "unweighted_mean_pcc": float(np.mean(pcc_arr)),
    }


def discover_prediction_files(base_path: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in PREDICTION_PATTERNS:
        files.extend(base_path.rglob(pattern))
    return sorted(set(files))


def prediction_metadata(path: Path, base_path: Path) -> dict[str, str]:
    metadata = metadata_from_prediction_path(path, base_path)
    split = "unknown"
    suffix = path.stem.removeprefix("predictions_")

    for candidate in ("main_val", "css_benchmark", "val", "test", "predict"):
        if suffix.startswith(candidate):
            split = candidate
            break

    return {**metadata, "split": split, "file": str(path)}


def transcript_key_columns(df: pd.DataFrame) -> list[str]:
    for columns in TRANSCRIPT_KEY_CANDIDATES:
        if all(column in df.columns for column in columns):
            return list(columns)
    return []


def duplicate_audit_row(
    *,
    df: pd.DataFrame,
    metadata: dict[str, str],
    source_counts: dict[str, int],
    key_columns: list[str],
) -> dict[str, Any]:
    if key_columns:
        duplicated_mask = df.duplicated(key_columns, keep=False)
        duplicated_rows = int(duplicated_mask.sum())
        duplicate_extra_rows = int(df.duplicated(key_columns, keep="first").sum())
        duplicate_keys = int(df.loc[duplicated_mask, key_columns].drop_duplicates().shape[0])
    else:
        duplicated_rows = 0
        duplicate_extra_rows = 0
        duplicate_keys = 0

    if "transcript_id" in df.columns:
        bare_key = ["transcript_id"]
        bare_duplicated_mask = df.duplicated(bare_key, keep=False)
        bare_duplicated_rows = int(bare_duplicated_mask.sum())
        bare_duplicate_extra_rows = int(df.duplicated(bare_key, keep="first").sum())
        bare_duplicate_keys = int(
            df.loc[bare_duplicated_mask, bare_key].drop_duplicates().shape[0]
        )
    else:
        bare_duplicated_rows = 0
        bare_duplicate_extra_rows = 0
        bare_duplicate_keys = 0

    return {
        "experiment": metadata["experiment"],
        "run": metadata["run"],
        "ablation_slug": metadata["ablation_slug"],
        "ablation_label": metadata["ablation_label"],
        "ablation_cagrad": metadata["ablation_cagrad"],
        "ablation_dataset_balanced_loss": metadata["ablation_dataset_balanced_loss"],
        "ablation_replica_objective": metadata["ablation_replica_objective"],
        "ablation_pcc_variant": metadata["ablation_pcc_variant"],
        "ablation_feature_preset": metadata["ablation_feature_preset"],
        "ablation_feature_tag": metadata["ablation_feature_tag"],
        "split": metadata["split"],
        "source_splits": ",".join(sorted(source_counts)),
        "key_columns": ",".join(key_columns),
        "n_rows_before_dedup": int(len(df)),
        "n_unique_transcripts": int(len(df) - duplicate_extra_rows),
        "duplicated_rows": duplicated_rows,
        "duplicate_keys": duplicate_keys,
        "duplicate_extra_rows": duplicate_extra_rows,
        "bare_transcript_duplicated_rows": bare_duplicated_rows,
        "bare_transcript_duplicate_keys": bare_duplicate_keys,
        "bare_transcript_duplicate_extra_rows": bare_duplicate_extra_rows,
        **{f"n_{split}_rows": int(source_counts.get(split, 0)) for split in COMBINED_VALIDATION_SOURCE_SPLITS},
    }


def duplicate_detail_rows(
    *,
    df: pd.DataFrame,
    metadata: dict[str, str],
    key_columns: list[str],
) -> pd.DataFrame:
    if not key_columns:
        return pd.DataFrame()

    duplicated = df[df.duplicated(key_columns, keep=False)].copy()
    if duplicated.empty:
        return pd.DataFrame()

    columns = key_columns.copy()
    if "_source_split" in duplicated.columns:
        columns.append("_source_split")
    if "length" in duplicated.columns:
        columns.append("length")

    details = duplicated[columns].sort_values(columns).reset_index(drop=True)
    details.insert(0, "run", metadata["run"])
    details.insert(0, "experiment", metadata["experiment"])
    details.insert(2, "ablation_slug", metadata["ablation_slug"])
    details.insert(2, "ablation_feature_preset", metadata["ablation_feature_preset"])
    details.insert(2, "ablation_feature_tag", metadata["ablation_feature_tag"])
    details.insert(2, "split", metadata["split"])
    return details


def concatenate_combined_validation(
    items: list[tuple[dict[str, str], Path]],
) -> tuple[pd.DataFrame, dict[str, str], list[str], pd.DataFrame, dict[str, Any]]:
    first_metadata = items[0][0]
    metadata = {
        **first_metadata,
        "split": COMBINED_VALIDATION_SPLIT,
        "file": ";".join(str(path) for _, path in items),
    }

    frames: list[pd.DataFrame] = []
    source_counts: dict[str, int] = {}
    for item_metadata, path in sorted(
        items,
        key=lambda item: COMBINED_VALIDATION_SPLIT_ORDER.get(item[0]["split"], 99),
    ):
        split = item_metadata["split"]
        df = pd.read_parquet(path).copy()
        df["_source_split"] = split
        df["_source_order"] = COMBINED_VALIDATION_SPLIT_ORDER.get(split, 99)
        frames.append(df)
        source_counts[split] = source_counts.get(split, 0) + int(len(df))

    combined = pd.concat(frames, ignore_index=True)
    key_columns = transcript_key_columns(combined)
    audit = duplicate_audit_row(
        df=combined,
        metadata=metadata,
        source_counts=source_counts,
        key_columns=key_columns,
    )
    duplicate_details = duplicate_detail_rows(
        df=combined,
        metadata=metadata,
        key_columns=key_columns,
    )
    if key_columns:
        combined = (
            combined.sort_values("_source_order", kind="stable")
            .drop_duplicates(key_columns, keep="first")
            .reset_index(drop=True)
        )
    combined = combined.drop(columns=["_source_order"])

    return combined, metadata, key_columns, duplicate_details, audit


def load_dataset_encoding(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return {int(value): str(name) for name, value in data.items()}


def dataset_label(dataset_id: Any, id_to_dataset: dict[int, str], fallback: str) -> str:
    try:
        dataset_int = int(dataset_id)
    except Exception:
        return fallback

    return id_to_dataset.get(dataset_int, f"dataset_{dataset_int}")


def compute_component_metrics(
    subset: pd.DataFrame,
    *,
    component: str,
    column: str,
) -> dict[str, Any]:
    if column not in subset.columns:
        return {
            "component": component,
            "resolved_column": None,
            **empty_metrics(),
        }

    pccs: list[float] = []
    lengths: list[int] = []

    for _, row in subset.iterrows():
        pred = sequence_or_none(row[column])
        target = sequence_or_none(row["target"])
        if pred is None or target is None:
            continue

        length = min(pred.size, target.size)
        pcc = safe_pearsonr(pred[:length], target[:length])
        if not np.isfinite(pcc):
            continue

        pccs.append(pcc)
        lengths.append(length)

    return {
        "component": component,
        "resolved_column": column,
        **fisher_weighted_pcc(pccs, lengths),
    }


def add_metrics_rows(
    rows: list[dict[str, Any]],
    *,
    df: pd.DataFrame,
    metadata: dict[str, str],
    id_to_dataset: dict[int, str],
    per_transcript_rows: list[dict[str, Any]] | None = None,
) -> None:
    if "target" not in df.columns:
        raise KeyError("Prediction file is missing required column: target")

    if "dataset_id" in df.columns:
        groups = df.groupby("dataset_id", sort=True)
    else:
        groups = [(metadata["experiment"], df)]

    for dataset_id, subset in groups:
        dataset = dataset_label(
            dataset_id,
            id_to_dataset,
            fallback=metadata["experiment"],
        )

        for component, column in COMPONENT_COLUMNS.items():
            rows.append(
                {
                    **metadata,
                    "dataset_id": dataset_id,
                    "dataset": dataset,
                    **compute_component_metrics(
                        subset,
                        component=component,
                        column=column,
                    ),
                }
            )

        if per_transcript_rows is not None:
            per_transcript_rows.extend(
                per_transcript_rank_rows(
                    subset,
                    metadata=metadata,
                    dataset_id=dataset_id,
                    dataset=dataset,
                )
            )


def _transcript_id_column(df: pd.DataFrame) -> str | None:
    for candidate in ("transcript_id", "ids"):
        if candidate in df.columns:
            return candidate
    return None


def per_transcript_rank_rows(
    subset: pd.DataFrame,
    *,
    metadata: dict[str, str],
    dataset_id: Any,
    dataset: str,
) -> list[dict[str, Any]]:
    """One row per (transcript, component) with per-transcript rank and Pearson PCC.

    Rank (Spearman) correlation is the primary quantity: Ribo-seq profiles are
    heavy-tailed, so Pearson is dominated by a few peaks while rank correlation
    reflects the whole ordering. These rows feed the min-dataset and paired
    rank-PCC summaries.
    """
    if "target" not in subset.columns:
        return []

    tid_col = _transcript_id_column(subset)
    run_type = run_type_for_experiment(
        experiment=metadata["experiment"],
        dataset=dataset,
        single_label=SINGLE_RUN_TYPE,
        mix_label=MIX_RUN_TYPE,
        other_label="other",
    )

    out: list[dict[str, Any]] = []
    for column in COMPONENT_COLUMNS.values():
        if column not in subset.columns:
            continue
        for pos, (_, row) in enumerate(subset.iterrows()):
            pred = sequence_or_none(row[column])
            target = sequence_or_none(row["target"])
            if pred is None or target is None:
                continue
            length = min(pred.size, target.size)
            if length < 4:
                continue
            rank_pcc = safe_spearmanr(pred[:length], target[:length])
            pearson_pcc = safe_pearsonr(pred[:length], target[:length])
            if not np.isfinite(rank_pcc):
                continue
            transcript_id = (
                str(row[tid_col]) if tid_col is not None else f"{dataset}__row{pos}"
            )
            out.append(
                {
                    "experiment": metadata["experiment"],
                    "run": metadata["run"],
                    "ablation_slug": metadata["ablation_slug"],
                    "ablation_label": metadata["ablation_label"],
                    "ablation_feature_preset": metadata["ablation_feature_preset"],
                    "ablation_feature_tag": metadata["ablation_feature_tag"],
                    "run_type": run_type,
                    "split": metadata["split"],
                    "dataset_id": dataset_id,
                    "dataset": dataset,
                    "transcript_id": transcript_id,
                    "component": {v: k for k, v in COMPONENT_COLUMNS.items()}[column],
                    "length": int(length),
                    "rank_pcc": float(rank_pcc),
                    "pearson_pcc": float(pearson_pcc)
                    if np.isfinite(pearson_pcc)
                    else np.nan,
                }
            )
    return out


def min_dataset_rank_summary(per_transcript_df: pd.DataFrame) -> pd.DataFrame:
    """Per-run worst-dataset rank-PCC and the spread across datasets.

    For each run/ablation the per-dataset rank-PCC is the median over that
    dataset's transcripts; the summary reports the minimum (worst dataset), mean,
    max and spread across datasets. In the multi-dataset mix this min is the key
    multi-task-balance metric: a setting that lifts the pooled score by
    sacrificing one dataset is penalized here.
    """
    if per_transcript_df.empty:
        return pd.DataFrame()

    group_cols = [
        "experiment",
        "run",
        "ablation_slug",
        "ablation_label",
        "run_type",
        "split",
        "component",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in per_transcript_df.groupby(group_cols, sort=True):
        per_dataset = (
            group.groupby("dataset")["rank_pcc"]
            .agg(["median", "size"])
            .reset_index()
        )
        per_dataset = per_dataset[per_dataset["size"] > 0]
        if per_dataset.empty:
            continue
        values = per_dataset["median"].to_numpy(dtype=np.float64)
        worst = per_dataset.iloc[int(np.argmin(values))]
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "n_datasets": int(len(per_dataset)),
                "n_transcripts": int(per_dataset["size"].sum()),
                "min_dataset_rank_pcc": float(np.min(values)),
                "min_dataset": str(worst["dataset"]),
                "mean_dataset_rank_pcc": float(np.mean(values)),
                "median_dataset_rank_pcc": float(np.median(values)),
                "max_dataset_rank_pcc": float(np.max(values)),
                "spread_rank_pcc": float(np.max(values) - np.min(values)),
                "std_dataset_rank_pcc": float(np.std(values, ddof=1))
                if values.size > 1
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def paired_rank_mix_vs_single(
    per_transcript_df: pd.DataFrame,
    *,
    min_pairs: int = 5,
) -> pd.DataFrame:
    """Paired per-transcript rank-PCC: each mix ablation vs each single-dataset run.

    Transcripts are matched by id within a shared dataset/split/component, so the
    comparison is paired (same transcripts scored under both settings). Reports
    median paired delta (mix - single), mix win rate and a two-sided Wilcoxon
    signed-rank p-value -- far more powerful than comparing marginal CIs.
    """
    if per_transcript_df.empty:
        return pd.DataFrame()

    single = per_transcript_df[per_transcript_df["run_type"] == SINGLE_RUN_TYPE]
    mix = per_transcript_df[per_transcript_df["run_type"] == MIX_RUN_TYPE]
    if single.empty or mix.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (split, component, dataset), single_group in single.groupby(
        ["split", "component", "dataset"], sort=True
    ):
        mix_group = mix[
            (mix["split"] == split)
            & (mix["component"] == component)
            & (mix["dataset"] == dataset)
        ]
        if mix_group.empty:
            continue

        for single_run, single_run_group in single_group.groupby("run", sort=True):
            single_map = single_run_group.groupby("transcript_id")["rank_pcc"].mean()
            for mix_run, mix_run_group in mix_group.groupby("run", sort=True):
                mix_map = mix_run_group.groupby("transcript_id")["rank_pcc"].mean()
                common = single_map.index.intersection(mix_map.index)
                if len(common) < min_pairs:
                    continue
                single_vals = single_map.loc[common].to_numpy(dtype=np.float64)
                mix_vals = mix_map.loc[common].to_numpy(dtype=np.float64)
                valid = np.isfinite(single_vals) & np.isfinite(mix_vals)
                single_vals = single_vals[valid]
                mix_vals = mix_vals[valid]
                if single_vals.size < min_pairs:
                    continue
                delta = mix_vals - single_vals
                rows.append(
                    {
                        "split": split,
                        "component": component,
                        "dataset": dataset,
                        "single_run": single_run,
                        "mix_run": mix_run,
                        "mix_ablation_slug": str(
                            mix_run_group["ablation_slug"].iloc[0]
                        ),
                        "mix_ablation_label": str(
                            mix_run_group["ablation_label"].iloc[0]
                        ),
                        "n_paired": int(delta.size),
                        "single_median_rank_pcc": float(np.median(single_vals)),
                        "mix_median_rank_pcc": float(np.median(mix_vals)),
                        "median_delta_rank_pcc": float(np.median(delta)),
                        "mean_delta_rank_pcc": float(np.mean(delta)),
                        "mix_win_rate": float(np.mean(delta > 0.0)),
                        "wilcoxon_p_two_sided": wilcoxon_signed_rank_p(delta),
                    }
                )
    return pd.DataFrame(rows)


def keep_mix_vs_single_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in metrics_df.iterrows():
        experiment = str(row["experiment"])
        dataset = str(row["dataset"])

        run_type = run_type_for_experiment(
            experiment=experiment,
            dataset=dataset,
            single_label=SINGLE_RUN_TYPE,
            mix_label=MIX_RUN_TYPE,
            other_label="other",
        )

        # Ignore malformed/unknown directory layouts instead of silently
        # treating them as a multi-dataset result.
        if run_type == "other":
            continue

        rows.append({**row.to_dict(), "run_type": run_type})

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(
        ["split", "component", "ablation_slug", "dataset", "run_type"],
        ascending=True,
    ).reset_index(drop=True)


def component_plot_data(
    metrics_df: pd.DataFrame,
    component: str,
    split: str,
    *,
    mix_ablation_slug: str | None = None,
) -> pd.DataFrame:
    """Rows to plot for one component/split.

    Every dataset with finite metrics is kept, regardless of whether it has a
    single-dataset run, a mix run, or both. This lets the script plot results
    independently of whether ``MIX_EXPERIMENT`` is present.
    """
    sub = metrics_df[
        (metrics_df["component"] == component)
        & (metrics_df["split"] == split)
        & (metrics_df["n_transcripts"].fillna(0).astype(int) > 0)
    ].copy()
    if mix_ablation_slug is not None:
        sub = sub[
            (sub["run_type"] == SINGLE_RUN_TYPE)
            | (
                (sub["run_type"] == MIX_RUN_TYPE)
                & (sub["ablation_slug"] == mix_ablation_slug)
            )
        ].copy()
    return sub


def present_run_types(sub: pd.DataFrame) -> list[str]:
    return [rt for rt in RUN_TYPE_ORDER if (sub["run_type"] == rt).any()]


def ordered_datasets(sub: pd.DataFrame) -> list[str]:
    pivot = sub.pivot_table(
        index="dataset",
        columns="run_type",
        values="pcc",
        aggfunc="first",
    )
    sort_col = SINGLE_RUN_TYPE if SINGLE_RUN_TYPE in pivot.columns else pivot.columns[0]
    return pivot.sort_values(sort_col, ascending=False).index.tolist()


def comparison_title(component: str, run_types: list[str]) -> str:
    if set(run_types) == set(RUN_TYPE_ORDER):
        descriptor = f"{SINGLE_RUN_TYPE} vs multi-dataset mix"
    elif run_types == [MIX_RUN_TYPE]:
        descriptor = "multi-dataset mix"
    else:
        descriptor = SINGLE_RUN_TYPE
    return f"PCC({component}, target): {descriptor}"


def ablation_short_label(row: pd.Series) -> str:
    replica = str(row.get("ablation_replica_objective", "unknown")).replace(
        "consensus_plus_replica",
        "cons+rep",
    )
    pcc = str(row.get("ablation_pcc_variant", "unknown")).replace(
        "raw_plus_varadj",
        "raw+var",
    )
    return (
        f"Features {row.get('ablation_feature_preset', 'unknown')}, "
        f"CG {row.get('ablation_cagrad', 'unknown')}, "
        f"DB {row.get('ablation_dataset_balanced_loss', 'unknown')}, "
        f"{replica}, PCC {pcc}"
    )


def mix_ablation_table(sub: pd.DataFrame) -> pd.DataFrame:
    mix_rows = sub[sub["run_type"] == MIX_RUN_TYPE].copy()
    if mix_rows.empty:
        return pd.DataFrame()
    cols = [
        "ablation_slug",
        "ablation_label",
        "ablation_cagrad",
        "ablation_dataset_balanced_loss",
        "ablation_replica_objective",
        "ablation_pcc_variant",
        "ablation_feature_preset",
        "ablation_feature_tag",
    ]
    return (
        mix_rows[cols]
        .drop_duplicates("ablation_slug")
        .sort_values("ablation_slug")
        .reset_index(drop=True)
    )


def pcc_xerr(row: pd.Series) -> np.ndarray:
    pcc = float(row["pcc"])
    return np.asarray(
        [[max(0.0, pcc - float(row["ci_lower"]))], [max(0.0, float(row["ci_upper"]) - pcc)]],
        dtype=np.float64,
    )


def plot_component_comparison(
    metrics_df: pd.DataFrame,
    *,
    component: str,
    split: str,
    out_dir: Path,
) -> Path | None:
    sub = component_plot_data(metrics_df, component, split)
    if sub.empty:
        print(f"[WARN] No rows for component={component}, split={split}.")
        return None

    if plt is None:
        return plot_component_comparison_with_pillow(
            sub,
            component=component,
            split=split,
            out_dir=out_dir,
        )

    runs = present_run_types(sub)
    datasets = ordered_datasets(sub)
    ablations = mix_ablation_table(sub)
    n_ablations = len(ablations)

    y = np.arange(len(datasets), dtype=np.float64)
    total_series = max(n_ablations + (1 if SINGLE_RUN_TYPE in runs else 0), 1)
    offsets = np.linspace(-0.36, 0.36, total_series) if total_series > 1 else np.asarray([0.0])
    single_offset_idx = 0
    mix_offset_start = 1 if SINGLE_RUN_TYPE in runs else 0

    fig, ax = plt.subplots(figsize=(13.5, max(6, 0.42 * len(datasets))))

    if SINGLE_RUN_TYPE in runs:
        single_sub = sub[sub["run_type"] == SINGLE_RUN_TYPE]
        single_label_used = False
        for idx, dataset in enumerate(datasets):
            match = single_sub[single_sub["dataset"] == dataset]
            if match.empty:
                continue
            row = match.iloc[0]
            ax.errorbar(
                float(row["pcc"]),
                y[idx] + offsets[single_offset_idx],
                xerr=pcc_xerr(row),
                fmt="s",
                markersize=4.2,
                color="#333333",
                ecolor="#777777",
                elinewidth=0.8,
                capsize=1.5,
                label=SINGLE_RUN_TYPE if not single_label_used else None,
                zorder=3,
            )
            single_label_used = True

    for ablation_idx, ablation_row in ablations.iterrows():
        slug = str(ablation_row["ablation_slug"])
        color = MIX_ABLATION_COLORS[ablation_idx % len(MIX_ABLATION_COLORS)]
        label = ablation_short_label(ablation_row)
        mix_sub = sub[
            (sub["run_type"] == MIX_RUN_TYPE)
            & (sub["ablation_slug"] == slug)
        ]
        label_used = False
        for dataset_idx, dataset in enumerate(datasets):
            match = mix_sub[mix_sub["dataset"] == dataset]
            if match.empty:
                continue
            row = match.iloc[0]
            ax.errorbar(
                float(row["pcc"]),
                y[dataset_idx] + offsets[mix_offset_start + ablation_idx],
                xerr=pcc_xerr(row),
                fmt="o",
                markersize=3.6,
                color=color,
                ecolor=color,
                alpha=0.9,
                elinewidth=0.65,
                capsize=1.0,
                label=label if not label_used else None,
                zorder=2,
            )
            label_used = True

    ax.set_yticks(y)
    ax.set_yticklabels(datasets, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Fisher-Z aggregated per-transcript PCC")
    ax.set_title(
        f"{comparison_title(component, runs)}\n"
        f"All {n_ablations} multi-dataset feature ablation(s)"
    )
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=7,
        title_fontsize=8,
        framealpha=0.85,
        borderpad=0.4,
        labelspacing=0.3,
        handlelength=1.2,
    )

    fig.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pcc_{split}_{component}_all_mix_ablations.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def pil_font(size: int, *, bold: bool = False) -> Any:
    if ImageFont is None:
        return None

    font_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = [
        f"/usr/share/fonts/truetype/dejavu/{font_name}",
        f"/usr/local/share/fonts/{font_name}",
    ]

    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass

    return ImageFont.load_default()


def plot_component_comparison_with_pillow(
    sub: pd.DataFrame,
    *,
    component: str,
    split: str,
    out_dir: Path,
) -> Path | None:
    if Image is None or ImageDraw is None:
        print("\n[WARN] Neither matplotlib nor Pillow is installed. Skipping plots.")
        return None

    runs = present_run_types(sub)
    datasets = ordered_datasets(sub)
    ablations = mix_ablation_table(sub)
    n_ablations = len(ablations)

    width = 1700
    row_height = max(28, 14 + 3 * max(n_ablations, 1))
    top = 86
    left = 230
    right = 470
    bottom = 78
    height = top + row_height * len(datasets) + bottom
    plot_width = width - left - right
    bg = "white"

    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    title_font = pil_font(16, bold=True)
    axis_font = pil_font(11)
    label_font = pil_font(10)
    legend_font = pil_font(10)

    pcc_min = float(np.nanmin(sub["ci_lower"].astype(float).to_numpy()))
    pcc_max = float(np.nanmax(sub["ci_upper"].astype(float).to_numpy()))
    x_min = min(0.0, np.floor((pcc_min - 0.03) * 10.0) / 10.0)
    x_max = min(1.0, max(0.7, np.ceil((pcc_max + 0.03) * 10.0) / 10.0))
    if x_max <= x_min:
        x_max = x_min + 1.0

    def x_pos(value: float) -> int:
        frac = (value - x_min) / (x_max - x_min)
        return int(round(left + frac * plot_width))

    title = (
        f"{comparison_title(component, runs)} - "
        f"all {n_ablations} multi-dataset feature ablation(s)"
    )
    draw.text((left, 18), title, fill="black", font=title_font)

    tick_start = int(np.ceil(x_min * 10.0))
    tick_end = int(np.floor(x_max * 10.0))
    axis_y = top + row_height * len(datasets) + 8
    for tick in range(tick_start, tick_end + 1):
        value = tick / 10.0
        x = x_pos(value)
        draw.line((x, top - 8, x, axis_y), fill="#E0E0E0", width=1)
        draw.text((x - 12, axis_y + 5), f"{value:.1f}", fill="black", font=axis_font)

    zero_x = x_pos(0.0)
    draw.line((zero_x, top - 8, zero_x, axis_y), fill="black", width=1)
    draw.line((left, axis_y, left + plot_width, axis_y), fill="black", width=1)
    draw.text(
        (left + plot_width // 2 - 140, height - 28),
        "Fisher-Z aggregated per-transcript PCC",
        fill="black",
        font=axis_font,
    )

    total_series = max(n_ablations + (1 if SINGLE_RUN_TYPE in runs else 0), 1)
    offsets = (
        np.linspace(-row_height * 0.35, row_height * 0.35, total_series)
        if total_series > 1
        else np.asarray([0.0])
    )
    single_offset_idx = 0
    mix_offset_start = 1 if SINGLE_RUN_TYPE in runs else 0

    for row_idx, dataset in enumerate(datasets):
        y_center = top + row_idx * row_height + row_height // 2
        draw.text((18, y_center - 7), dataset, fill="black", font=label_font)

        if SINGLE_RUN_TYPE in runs:
            match = sub[
                (sub["dataset"] == dataset)
                & (sub["run_type"] == SINGLE_RUN_TYPE)
            ]
            if not match.empty:
                row = match.iloc[0]
                pcc = float(row["pcc"])
                y = y_center + int(offsets[single_offset_idx])
                lo = x_pos(float(row["ci_lower"]))
                hi = x_pos(float(row["ci_upper"]))
                x = x_pos(pcc)
                draw.line((lo, y, hi, y), fill="#777777", width=1)
                draw.rectangle((x - 4, y - 4, x + 4, y + 4), fill="#333333", outline="black")

        for ablation_idx, ablation_row in ablations.iterrows():
            slug = str(ablation_row["ablation_slug"])
            color = MIX_ABLATION_COLORS[ablation_idx % len(MIX_ABLATION_COLORS)]
            match = sub[
                (sub["dataset"] == dataset)
                & (sub["run_type"] == MIX_RUN_TYPE)
                & (sub["ablation_slug"] == slug)
            ]
            if match.empty:
                continue
            row = match.iloc[0]
            pcc = float(row["pcc"])
            y = y_center + int(offsets[mix_offset_start + ablation_idx])
            lo = x_pos(float(row["ci_lower"]))
            hi = x_pos(float(row["ci_upper"]))
            x = x_pos(pcc)
            draw.line((lo, y, hi, y), fill=color, width=1)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline="black")

    legend_x = width - right + 22
    legend_y = 18
    draw.rectangle(
        (legend_x - 8, legend_y - 6, width - 20, legend_y + 22 * (n_ablations + 1) + 10),
        fill="white",
        outline="#BBBBBB",
        width=1,
    )
    legend_items = []
    if SINGLE_RUN_TYPE in runs:
        legend_items.append(("single dataset", "#333333", "square"))
    legend_items.extend(
        (
            ablation_short_label(row),
            MIX_ABLATION_COLORS[idx % len(MIX_ABLATION_COLORS)],
            "circle",
        )
        for idx, row in ablations.iterrows()
    )
    for idx, (label, color, marker) in enumerate(legend_items):
        y = legend_y + idx * 20
        if marker == "square":
            draw.rectangle((legend_x, y, legend_x + 12, y + 12), fill=color, outline="black")
        else:
            draw.ellipse((legend_x, y, legend_x + 12, y + 12), fill=color, outline="black")
        draw.text((legend_x + 20, y - 3), label, fill="black", font=legend_font)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pcc_{split}_{component}_all_mix_ablations.png"
    image.save(out_path)
    return out_path


def discover_configured_training_runs(results_root: Path) -> list[dict[str, Any]]:
    """Find result bundles created by the Slurm all/individual launchers."""
    runs: list[dict[str, Any]] = []
    if not results_root.is_dir():
        return runs

    for run_root in sorted(results_root.iterdir()):
        if not run_root.is_dir():
            continue
        match = CONFIGURED_RUN_PATTERN.match(run_root.name)
        result_dir = run_root / "results"
        if match is None or not result_dir.is_dir():
            continue
        prediction_files = discover_prediction_files(result_dir)
        if not prediction_files:
            print(f"[WARN] Skipping {run_root.name}: no prediction files found.")
            continue
        runs.append(
            {
                **match.groupdict(),
                "run_id": run_root.name,
                "root": run_root,
                "base_path": result_dir,
                "prediction_files": prediction_files,
            }
        )
    return runs


def configured_run_metrics(
    runs: list[dict[str, Any]],
    *,
    id_to_dataset: dict[int, str],
) -> pd.DataFrame:
    """Compute main-validation μ/L_bio PCC rows for each configured run."""
    rows: list[dict[str, Any]] = []
    for run in runs:
        run_rows: list[dict[str, Any]] = []
        for path in run["prediction_files"]:
            metadata = prediction_metadata(path, run["base_path"])
            # The requested comparison is against the main validation ground
            # truth. CSS benchmark files are intentionally kept out of it.
            if metadata["split"] != "main_val":
                continue
            try:
                df = pd.read_parquet(path)
                add_metrics_rows(
                    run_rows,
                    df=df,
                    metadata=metadata,
                    id_to_dataset=id_to_dataset,
                )
            except Exception as exc:
                print(f"[WARN] Failed to process {path}: {exc}")

        feature_preset = str(run["feature_preset"]).lower()
        for row in run_rows:
            row.update(
                {
                    "run_id": run["run_id"],
                    "training_mode": run["training_mode"],
                    "feature_preset": feature_preset,
                    "seed": run["seed"],
                    "job_id": run["job_id"],
                }
            )
        rows.extend(run_rows)

        if not run_rows:
            print(f"[WARN] No main-validation predictions found in {run['run_id']}.")

    return pd.DataFrame(rows)


def configured_summary(metrics_df: pd.DataFrame, *, component: str = "mu") -> pd.DataFrame:
    """Average duplicate jobs while retaining one dataset-level row per setting.

    ``component`` selects which resolved profile ("mu" or "L_bio") to summarize;
    the default keeps the original mu-only behavior for any existing caller.
    """
    if metrics_df.empty:
        return metrics_df
    subset = metrics_df[
        (metrics_df["split"] == "main_val")
        & (metrics_df["component"] == component)
        & (metrics_df["n_transcripts"].fillna(0).astype(int) > 0)
    ].copy()
    if subset.empty:
        return subset

    group_cols = ["training_mode", "feature_preset", "dataset"]
    summary = (
        subset.groupby(group_cols, as_index=False)
        .agg(
            pcc=("pcc", "mean"),
            ci_lower=("ci_lower", "mean"),
            ci_upper=("ci_upper", "mean"),
            n_transcripts=("n_transcripts", "sum"),
            n_runs=("run_id", "nunique"),
        )
    )
    return summary


def _configured_dataset_order(summary: pd.DataFrame) -> list[str]:
    ordering = (
        summary.groupby("dataset")["pcc"]
        .mean()
        .sort_values(ascending=True)
    )
    return ordering.index.astype(str).tolist()


def plot_component_all_vs_individual(
    summary: pd.DataFrame,
    *,
    component: str,
    feature_preset: str,
    out_dir: Path,
) -> Path | None:
    """Plot per-dataset PCC of one component for the joint vs single-dataset run.

    With ``component="L_bio"`` this is the collapse check: the orange (all-dataset)
    bar dropping below the blue (individual) bar for a dataset means the shared
    biological signal fits that dataset worse once it must serve every dataset.
    """
    axis_label = COMPONENT_AXIS_LABELS.get(component, component)
    if plt is None:
        print(f"[WARN] Matplotlib is unavailable; skipping {axis_label} PCC bar plots.")
        return None

    subset = summary[summary["feature_preset"] == feature_preset]
    if subset.empty:
        return None
    datasets = _configured_dataset_order(subset)
    x = np.arange(len(datasets), dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(14.0, 0.52 * len(datasets)), 8.0))
    colors = {"all": "#F58518", "individual": "#4C78A8"}
    labels = {"all": "all datasets", "individual": "individual dataset"}
    offsets = {"all": width / 2.0, "individual": -width / 2.0}
    available_modes = set(subset["training_mode"])

    for mode in ("all", "individual"):
        mode_df = subset[subset["training_mode"] == mode].set_index("dataset")
        aligned = mode_df.reindex(datasets)
        values = aligned["pcc"].to_numpy(dtype=float)
        ci_lower = aligned["ci_lower"].to_numpy(dtype=float)
        ci_upper = aligned["ci_upper"].to_numpy(dtype=float)
        valid = np.isfinite(values)
        if not valid.any():
            continue
        has_ci = valid & np.isfinite(ci_lower) & np.isfinite(ci_upper)
        lower_error = np.maximum(values - ci_lower, 0.0)
        upper_error = np.maximum(ci_upper - values, 0.0)
        yerr = np.vstack(
            (
                np.where(has_ci[valid], lower_error[valid], 0.0),
                np.where(has_ci[valid], upper_error[valid], 0.0),
            )
        )
        ax.bar(
            x[valid] + offsets[mode],
            values[valid],
            width=width,
            color=colors[mode],
            label=labels[mode],
            yerr=yerr,
            error_kw={"elinewidth": 0.9, "capsize": 2.0, "ecolor": "#333333"},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel(f"PCC({axis_label}, target)")
    ax.set_ylim(0.0, 1.0)
    if available_modes == {"all", "individual"}:
        comparison_label = "all-dataset vs individual models"
    elif available_modes == {"all"}:
        comparison_label = "all-dataset models (individual results unavailable)"
    else:
        comparison_label = "individual models (all-dataset results unavailable)"
    ax.set_title(
        f"{axis_label} PCC by dataset: {comparison_label} "
        f"({FEATURE_PRESET_LABELS.get(feature_preset, feature_preset)})"
    )
    ax.axvline(0.0, color="#555555", linewidth=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{component}_pcc_all_vs_individual_{feature_preset}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_component_feature_ablation(
    summary: pd.DataFrame,
    *,
    component: str,
    training_mode: str,
    out_dir: Path,
) -> Path | None:
    """Plot the feature-preset ablation for every dataset, for one component."""
    if plt is None:
        return None
    axis_label = COMPONENT_AXIS_LABELS.get(component, component)
    subset = summary[summary["training_mode"] == training_mode]
    presets = [preset for preset in FEATURE_PRESET_ORDER if preset in set(subset["feature_preset"])]
    if subset.empty or not presets:
        return None

    datasets = _configured_dataset_order(subset)
    x = np.arange(len(datasets), dtype=float)
    width = min(0.75 / len(presets), 0.22)
    fig, ax = plt.subplots(figsize=(max(14.0, 0.52 * len(datasets)), 8.0))
    palette = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2")
    for idx, preset in enumerate(presets):
        preset_df = subset[subset["feature_preset"] == preset].set_index("dataset")
        aligned = preset_df.reindex(datasets)
        values = aligned["pcc"].to_numpy(dtype=float)
        ci_lower = aligned["ci_lower"].to_numpy(dtype=float)
        ci_upper = aligned["ci_upper"].to_numpy(dtype=float)
        valid = np.isfinite(values)
        if not valid.any():
            continue
        has_ci = valid & np.isfinite(ci_lower) & np.isfinite(ci_upper)
        lower_error = np.maximum(values - ci_lower, 0.0)
        upper_error = np.maximum(ci_upper - values, 0.0)
        yerr = np.vstack(
            (
                np.where(has_ci[valid], lower_error[valid], 0.0),
                np.where(has_ci[valid], upper_error[valid], 0.0),
            )
        )
        offset = (idx - (len(presets) - 1) / 2.0) * width
        ax.bar(
            x[valid] + offset,
            values[valid],
            width=width,
            color=palette[idx % len(palette)],
            label=FEATURE_PRESET_LABELS.get(preset, preset),
            yerr=yerr,
            error_kw={"elinewidth": 0.9, "capsize": 2.0, "ecolor": "#333333"},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel(f"PCC({axis_label}, target)")
    ax.set_ylim(0.0, 1.0)
    mode_label = "all-dataset" if training_mode == "all" else "individual-dataset"
    ax.set_title(f"{axis_label} PCC feature ablation by dataset ({mode_label} models)")
    ax.axvline(0.0, color="#555555", linewidth=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{component}_pcc_feature_ablation_{training_mode}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def analyze_configured_training_runs(
    *,
    results_root: Path,
    dataset_encoding_path: Path,
) -> None:
    """Analyze the Slurm result bundles and write per-component PCC comparisons.

    Both the observation mean ``mu`` and the shared biological signal ``L_bio``
    are compared for joint (all-dataset) vs single-dataset (individual) training
    across every feature-preset ablation. Each component gets its own folder
    (``<component>_pcc_plots``) and summary CSV, so the L_bio plots make an
    eventual collapse of the shared branch under joint training directly visible.
    """
    runs = discover_configured_training_runs(results_root)
    if not runs:
        return
    id_to_dataset = load_dataset_encoding(dataset_encoding_path)
    metrics_df = configured_run_metrics(runs, id_to_dataset=id_to_dataset)
    if metrics_df.empty:
        raise RuntimeError("Configured runs were found, but no main-validation metrics were computed.")

    availability = pd.DataFrame(
        [
            {
                "run_id": run["run_id"],
                "training_mode": run["training_mode"],
                "feature_preset": run["feature_preset"],
                "seed": run["seed"],
                "job_id": run["job_id"],
                "prediction_file_count": len(run["prediction_files"]),
            }
            for run in runs
        ]
    )
    availability.to_csv(results_root / "mu_pcc_run_availability.csv", index=False)

    print(f"Configured runs: {len(runs)}")
    for component in CONFIGURED_COMPONENTS:
        summary = configured_summary(metrics_df, component=component)
        if summary.empty:
            print(f"[WARN] No {component} main-validation rows; skipping {component} plots.")
            continue
        output_csv = results_root / f"{component}_pcc_all_vs_individual_metrics.csv"
        summary.to_csv(output_csv, index=False)
        plot_dir = results_root / f"{component}_pcc_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saved {component}-PCC summary: {output_csv}")
        for preset in FEATURE_PRESET_ORDER:
            out_path = plot_component_all_vs_individual(
                summary,
                component=component,
                feature_preset=preset,
                out_dir=plot_dir,
            )
            if out_path is not None:
                print(f"Saved plot: {out_path}")
        for mode in ("all", "individual"):
            out_path = plot_component_feature_ablation(
                summary,
                component=component,
                training_mode=mode,
                out_dir=plot_dir,
            )
            if out_path is not None:
                print(f"Saved plot: {out_path}")


def resolve_base_path() -> Path:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    candidates = [
        Path.cwd() / "riboai_queueing",
        repo_root / "results" / "riboai_queueing",
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    return candidates[-1]


def main() -> None:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    results_root = repo_root / "results"
    configured_runs = discover_configured_training_runs(results_root)
    if configured_runs:
        analyze_configured_training_runs(
            results_root=results_root,
            dataset_encoding_path=repo_root / "Datasets" / "encodings" / "dataset_encoding.yaml",
        )
        return

    base_path = resolve_base_path()
    dataset_encoding_path = repo_root / "Datasets" / "encodings" / "dataset_encoding.yaml"

    prediction_files = discover_prediction_files(base_path)
    if len(prediction_files) == 0:
        raise RuntimeError(f"No prediction parquet files found under {base_path}.")

    id_to_dataset = load_dataset_encoding(dataset_encoding_path)
    rows: list[dict[str, Any]] = []
    per_transcript_rows: list[dict[str, Any]] = []
    combined_validation_files: dict[tuple[str, str], list[tuple[dict[str, str], Path]]] = {}
    standalone_files: list[tuple[dict[str, str], Path]] = []
    duplicate_audit_rows: list[dict[str, Any]] = []
    duplicate_detail_frames: list[pd.DataFrame] = []

    print(f"Base path: {base_path}")
    print(f"Prediction files: {len(prediction_files)}")
    print(f"Comparison baseline: {MIX_EXPERIMENT}")
    print(f"Components: {', '.join(COMPONENT_COLUMNS)}")

    for path in prediction_files:
        metadata = prediction_metadata(path, base_path)
        if metadata["split"] in COMBINED_VALIDATION_SOURCE_SPLITS:
            key = (metadata["experiment"], metadata["run"])
            combined_validation_files.setdefault(key, []).append((metadata, path))
        else:
            standalone_files.append((metadata, path))

    for metadata, path in standalone_files:
        print(f"Loading {path}")

        try:
            df = pd.read_parquet(path)
            add_metrics_rows(
                rows,
                df=df,
                metadata=metadata,
                id_to_dataset=id_to_dataset,
                per_transcript_rows=per_transcript_rows,
            )
        except Exception as exc:
            print(f"[WARN] Failed to process {path}: {exc}")

    for (experiment, run), items in sorted(combined_validation_files.items()):
        source_splits = ", ".join(metadata["split"] for metadata, _ in items)
        print(f"Loading combined validation for {experiment}/{run}: {source_splits}")

        try:
            (
                df,
                metadata,
                key_columns,
                duplicate_details,
                duplicate_audit,
            ) = concatenate_combined_validation(items)
            duplicate_audit_rows.append(duplicate_audit)
            if not duplicate_details.empty:
                duplicate_detail_frames.append(duplicate_details)
            if duplicate_audit["duplicate_extra_rows"] > 0:
                print(
                    "[WARN] Repeated transcript rows in combined validation "
                    f"for {experiment}/{run}: "
                    f"{duplicate_audit['duplicate_extra_rows']} extra rows across "
                    f"{duplicate_audit['duplicate_keys']} transcript keys. "
                    f"Keeping first by {COMBINED_VALIDATION_SOURCE_SPLITS}."
                )
            elif key_columns:
                print(
                    "  duplicate check: 0 repeated transcript keys "
                    f"using ({', '.join(key_columns)})"
                )
            add_metrics_rows(
                rows,
                df=df,
                metadata=metadata,
                id_to_dataset=id_to_dataset,
                per_transcript_rows=per_transcript_rows,
            )
        except Exception as exc:
            print(f"[WARN] Failed to process combined validation for {experiment}/{run}: {exc}")

    metrics_df = keep_mix_vs_single_rows(pd.DataFrame(rows))
    if metrics_df.empty:
        raise RuntimeError(
            "No recognized single-dataset or multi-dataset mix metrics were "
            "computed. Check the result-directory layout and prediction files."
        )

    available_run_types = present_run_types(metrics_df)
    if available_run_types == [MIX_RUN_TYPE]:
        print(
            "\n[INFO] No single-dataset training results found; continuing with "
            "multi-dataset mix ablation analysis only."
        )

    out_csv = base_path / "pcc_metrics_mu_rho_single_vs_mix.csv"
    metrics_df.to_csv(out_csv, index=False)
    print(f"\nSaved metrics table to: {out_csv}")

    if duplicate_audit_rows:
        duplicate_audit_df = pd.DataFrame(duplicate_audit_rows).sort_values(
            ["experiment", "run"],
            ascending=True,
        )
        duplicate_audit_path = base_path / "pcc_combined_validation_duplicate_audit.csv"
        duplicate_audit_df.to_csv(duplicate_audit_path, index=False)
        total_extra = int(duplicate_audit_df["duplicate_extra_rows"].sum())
        print(f"Saved duplicate audit to: {duplicate_audit_path}")
        print(f"Combined validation repeated transcript extra rows: {total_extra}")

        if duplicate_detail_frames:
            duplicate_detail_path = base_path / "pcc_combined_validation_repeated_transcripts.csv"
            pd.concat(duplicate_detail_frames, ignore_index=True).to_csv(
                duplicate_detail_path,
                index=False,
            )
            print(f"Saved repeated transcript details to: {duplicate_detail_path}")

    visible = metrics_df[
        metrics_df["n_transcripts"].fillna(0).astype(int) > 0
    ][
        [
            "split",
            "dataset",
            "ablation_slug",
            "ablation_label",
            "component",
            "run_type",
            "resolved_column",
            "pcc",
            "ci_lower",
            "ci_upper",
            "n_transcripts",
        ]
    ]
    print(f"\n=== {comparison_title('mu', available_run_types)} ===")
    print(visible.to_string(index=False))

    per_transcript_df = pd.DataFrame(per_transcript_rows)
    if not per_transcript_df.empty:
        # Parquet (not CSV): this table has one row per transcript x component x
        # run and its string columns repeat heavily, so dictionary-encoded parquet
        # is ~10-30x smaller than the equivalent CSV.
        per_tx_path = base_path / "pcc_per_transcript_rank.parquet"
        per_transcript_df.to_parquet(per_tx_path, index=False)
        print(f"\nSaved per-transcript rank/Pearson PCCs to: {per_tx_path}")

        min_dataset_df = min_dataset_rank_summary(per_transcript_df)
        if not min_dataset_df.empty:
            min_dataset_path = base_path / "pcc_min_dataset_rank_summary.csv"
            min_dataset_df.to_csv(min_dataset_path, index=False)
            print(f"Saved min-dataset rank summary to: {min_dataset_path}")
            visible_min = min_dataset_df[
                [
                    "experiment",
                    "run_type",
                    "ablation_label",
                    "split",
                    "component",
                    "n_datasets",
                    "min_dataset_rank_pcc",
                    "min_dataset",
                    "mean_dataset_rank_pcc",
                    "spread_rank_pcc",
                ]
            ].sort_values(
                ["component", "split", "min_dataset_rank_pcc"],
                ascending=[True, True, False],
            )
            print("\n=== Worst-dataset (min) rank-PCC per run ===")
            print(visible_min.to_string(index=False))

        paired_df = paired_rank_mix_vs_single(per_transcript_df)
        if not paired_df.empty:
            paired_path = base_path / "pcc_paired_rank_mix_vs_single.csv"
            paired_df.to_csv(paired_path, index=False)
            print(f"\nSaved paired mix-vs-single rank-PCC to: {paired_path}")
            visible_paired = paired_df[
                [
                    "split",
                    "component",
                    "dataset",
                    "mix_ablation_label",
                    "n_paired",
                    "single_median_rank_pcc",
                    "mix_median_rank_pcc",
                    "median_delta_rank_pcc",
                    "mix_win_rate",
                    "wilcoxon_p_two_sided",
                ]
            ].sort_values(["component", "split", "median_delta_rank_pcc"])
            print("\n=== Paired mix-vs-single rank-PCC (per transcript) ===")
            print(visible_paired.to_string(index=False))
        else:
            print(
                "\n[INFO] No paired mix-vs-single rank-PCC rows "
                "(need matching single and mix runs on shared transcripts)."
            )

    plot_dir = base_path / "../pcc_metric_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for old_plot in plot_dir.glob("pcc_*.png"):
        old_plot.unlink()

    for split in sorted(metrics_df["split"].dropna().unique()):
        for component in COMPONENT_COLUMNS:
            out_path = plot_component_comparison(
                metrics_df,
                component=component,
                split=str(split),
                out_dir=plot_dir,
            )
            if out_path is not None:
                print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
