from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from tqdm import tqdm


EPS = 1e-8
STOP_CODONS = {"TAA", "TAG", "TGA"}


# ============================================================
# Generic helpers
# ============================================================

def safe_array(x: Any, dtype=np.float64) -> np.ndarray:
    return np.asarray(x, dtype=dtype).reshape(-1)


def safe_median(x: Any) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else np.nan


def safe_mean(x: Any) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else np.nan


def robust_ratio(num: float, den: float, eps: float = EPS) -> float:
    if not np.isfinite(num) or not np.isfinite(den):
        return np.nan
    return float(num / max(abs(den), eps))


def safe_pcc(a: Any, b: Any) -> float:
    a = safe_array(a)
    b = safe_array(b)

    L = min(len(a), len(b))
    if L < 3:
        return np.nan

    a = a[:L]
    b = b[:L]

    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]

    if len(a) < 3:
        return np.nan

    if np.var(a) <= 1e-12 or np.var(b) <= 1e-12:
        return np.nan

    return float(pearsonr(a, b)[0])


def safe_spearman(a: Any, b: Any) -> float:
    a = safe_array(a)
    b = safe_array(b)

    L = min(len(a), len(b))
    if L < 3:
        return np.nan

    a = a[:L]
    b = b[:L]

    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]

    if len(a) < 3:
        return np.nan

    if np.var(a) <= 1e-12 or np.var(b) <= 1e-12:
        return np.nan

    r, _ = spearmanr(a, b)
    return float(r)


def sanitize_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "transcripts_id" in df.columns and "transcript_id" not in df.columns:
        df = df.rename(columns={"transcripts_id": "transcript_id"})

    if "y" in df.columns and "target" not in df.columns:
        df = df.rename(columns={"y": "target"})

    if "L" in df.columns and "L_queue" not in df.columns:
        df = df.rename(columns={"L": "L_queue"})

    return df


def parse_css_indices(css_raw: Any, L: int) -> np.ndarray:
    css_raw = np.asarray(css_raw)

    if css_raw.size == 0:
        return np.array([], dtype=int)

    if len(css_raw) >= L:
        return np.where(css_raw[:L] > 0)[0].astype(int)

    if np.issubdtype(css_raw.dtype, np.number):
        css_idx = css_raw[np.isfinite(css_raw)]
    else:
        css_idx = css_raw

    css_idx = np.asarray(css_idx, dtype=int)
    return css_idx[(css_idx >= 0) & (css_idx < L)]


def css_window_mask(css_idx: np.ndarray, L: int, css_window: int) -> np.ndarray:
    mask = np.zeros(L, dtype=bool)

    for idx in css_idx:
        lo = max(0, int(idx) - css_window)
        hi = min(L, int(idx) + css_window + 1)
        mask[lo:hi] = True

    return mask


def decode_ref_codons(ref_raw: Any, onehot2nt: dict[int, str]) -> list[str]:
    raw_nt_sequence = np.stack([np.stack(c) for c in ref_raw])  # [T, 3, 4]
    codons = []

    for codon in raw_nt_sequence:
        nts = [onehot2nt[int(np.argmax(n))] for n in codon]
        codons.append("".join(nts))

    return codons


def load_dataset_names(dataset_encoding_path: str | None) -> dict[int, str]:
    candidates: list[Path] = []

    if dataset_encoding_path is not None:
        candidates.append(Path(dataset_encoding_path))

    candidates.extend(
        [
            Path("../Datasets/encodings/dataset_encoding.yaml"),
            Path("./Datasets/encodings/dataset_encoding.yaml"),
            Path("../Datasets/encodings/datasets_encoding.yaml"),
            Path("./Datasets/encodings/datasets_encoding.yaml"),
        ]
    )

    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                name_to_id = yaml.safe_load(f)
            return {int(v): str(k) for k, v in name_to_id.items()}

    print("[WARNING] Could not find dataset encoding. Using dataset_<id> labels.")
    return {}


def load_onehot2nt(nt_encoding_path: str | None) -> dict[int, str] | None:
    candidates: list[Path] = []

    if nt_encoding_path is not None:
        candidates.append(Path(nt_encoding_path))

    candidates.extend(
        [
            Path("../Datasets/encodings/nt_encoding.yaml"),
            Path("./Datasets/encodings/nt_encoding.yaml"),
        ]
    )

    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                nt_encoding = yaml.safe_load(f)
            return {int(np.argmax(v)): str(k) for k, v in nt_encoding.items()}

    print("[WARNING] Could not find nt_encoding.yaml. Codon analyses will be skipped.")
    return None


def load_predictions(prediction_dir: str | Path, prediction_glob: str | None = None) -> pd.DataFrame:
    prediction_dir = Path(prediction_dir)

    if prediction_glob is None:
        files = sorted(glob.glob(str(prediction_dir / "comprehensive_predictions_rank*.parquet")))
    else:
        files = sorted(glob.glob(prediction_glob))

    if not files:
        raise FileNotFoundError(
            f"No prediction parquet files found. "
            f"prediction_dir={prediction_dir}, prediction_glob={prediction_glob}"
        )

    print(f"Loading {len(files)} parquet shard(s).")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = normalize_columns(df)

    required = {
        "dataset_id",
        "transcript_id",
        "length",
        "target",
        "L_queue",
        "total_scale",
        "b_offset",
        "css",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required prediction columns: {sorted(missing)}")

    df["dataset_id"] = df["dataset_id"].astype(int)
    df["transcript_id"] = df["transcript_id"].astype(str)

    before = len(df)
    df = df.drop_duplicates(subset=["dataset_id", "transcript_id"]).reset_index(drop=True)
    after = len(df)

    print(f"Dropped {before - after} duplicated DDP rows.")
    print(f"Loaded rows: {len(df)}")
    print(f"Datasets: {df['dataset_id'].nunique()}")
    print(f"Transcripts: {df['transcript_id'].nunique()}")

    return df


def load_sequence_refs(sequence_path: str | None) -> pd.DataFrame | None:
    candidates: list[Path] = []

    if sequence_path is not None:
        candidates.append(Path(sequence_path))

    candidates.extend(
        [
            Path("../Datasets/data/sequence/sequence_embeddings_with_css.parquet"),
            Path("./Datasets/data/sequence/sequence_embeddings_with_css.parquet"),
        ]
    )

    for path in candidates:
        if path.exists():
            seq_df = pd.read_parquet(path)
            seq_df = normalize_columns(seq_df)

            if "transcript_id" not in seq_df.columns or "ref" not in seq_df.columns:
                raise ValueError(f"Sequence file {path} must contain 'transcript_id' and 'ref'.")

            seq_df["transcript_id"] = seq_df["transcript_id"].astype(str)
            seq_df = seq_df[["transcript_id", "ref"]].drop_duplicates(subset=["transcript_id"])

            print(f"Loaded sequence references: {len(seq_df)} transcripts")
            return seq_df

    print("[WARNING] Could not find sequence parquet. Codon hierarchy analyses will be skipped.")
    return None


# ============================================================
# Shift-template residualization for b_offset
# ============================================================

def shift_template(L_queue: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        valid_positions, template

    Template:
        B_i(k) = log(L_{i+k} + eps) - log(L_i + eps)
    """
    L_queue = safe_array(L_queue)

    L = len(L_queue)
    idx = np.arange(L)
    j = idx + int(lag)

    valid = (j >= 0) & (j < L)
    if valid.sum() < 3:
        return np.array([], dtype=int), np.array([], dtype=np.float64)

    log_L = np.log(np.maximum(L_queue, EPS))
    template = log_L[j[valid]] - log_L[idx[valid]]

    return idx[valid], template


def b_template_pcc_for_lag(
    L_queue: np.ndarray,
    b_offset: np.ndarray,
    lag: int,
    gradient_quantile: float | None = 0.50,
) -> float:
    if lag == 0:
        return np.nan

    L_queue = safe_array(L_queue)
    b_offset = safe_array(b_offset)

    L = min(len(L_queue), len(b_offset))
    if L < 3:
        return np.nan

    L_queue = L_queue[:L]
    b_offset = b_offset[:L]

    valid_idx, template = shift_template(L_queue, lag)
    if len(valid_idx) < 3:
        return np.nan

    b = b_offset[valid_idx]

    finite = np.isfinite(template) & np.isfinite(b)
    template = template[finite]
    b = b[finite]

    if len(template) < 3:
        return np.nan

    if gradient_quantile is not None:
        threshold = np.quantile(np.abs(template), gradient_quantile)
        keep = np.abs(template) >= threshold
        template = template[keep]
        b = b[keep]

    if len(template) < 3:
        return np.nan

    return safe_pcc(b, template)


def compute_dataset_best_b_template_lag(
    df: pd.DataFrame,
    *,
    id_to_dataset: dict[int, str],
    lags: list[int],
    gradient_quantile: float | None,
) -> tuple[pd.DataFrame, dict[int, int]]:
    rows = []
    best_lags: dict[int, int] = {}

    for dataset_id, group in tqdm(df.groupby("dataset_id"), desc="Computing best b-template lag"):
        dataset_id = int(dataset_id)
        dataset_name = id_to_dataset.get(dataset_id, f"dataset_{dataset_id}")

        lag_to_r = {lag: [] for lag in lags if lag != 0}

        for _, row in group.iterrows():
            L = int(row["length"])
            L_queue = safe_array(row["L_queue"])[:L]
            b = safe_array(row["b_offset"])[:L]

            L_eff = min(len(L_queue), len(b))
            if L_eff < 10:
                continue

            L_queue = L_queue[:L_eff]
            b = b[:L_eff]

            for lag in lag_to_r:
                r = b_template_pcc_for_lag(
                    L_queue=L_queue,
                    b_offset=b,
                    lag=lag,
                    gradient_quantile=gradient_quantile,
                )
                if np.isfinite(r):
                    lag_to_r[lag].append(r)

        best_lag = 0
        best_median = -np.inf

        for lag, vals in lag_to_r.items():
            vals = np.asarray(vals, dtype=np.float64)
            vals = vals[np.isfinite(vals)]

            med = float(np.median(vals)) if vals.size else np.nan

            rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset": dataset_name,
                    "lag": int(lag),
                    "n_transcripts": int(vals.size),
                    "median_template_pcc": med,
                    "mean_template_pcc": float(np.mean(vals)) if vals.size else np.nan,
                }
            )

            if np.isfinite(med) and med > best_median:
                best_median = med
                best_lag = int(lag)

        best_lags[dataset_id] = best_lag

    return pd.DataFrame(rows), best_lags


def residualize_b_offset_against_shift_template(
    L_queue: np.ndarray,
    b_offset: np.ndarray,
    lag: int,
) -> np.ndarray:
    """
    Residualizes b_i against the best shift template:

        b_i = alpha + beta B_i(k) + residual_i

    Positions that cannot be matched to the shifted template retain NaN.
    """
    L_queue = safe_array(L_queue)
    b_offset = safe_array(b_offset)

    L = min(len(L_queue), len(b_offset))
    L_queue = L_queue[:L]
    b_offset = b_offset[:L]

    residual = np.full(L, np.nan, dtype=np.float64)

    if lag == 0:
        return b_offset - np.nanmean(b_offset)

    valid_idx, template = shift_template(L_queue, lag)
    if len(valid_idx) < 3:
        return residual

    b = b_offset[valid_idx]

    finite = np.isfinite(template) & np.isfinite(b)
    if finite.sum() < 3:
        return residual

    idx_fit = valid_idx[finite]
    x = template[finite]
    y = b[finite]

    if np.var(x) <= 1e-12:
        residual[idx_fit] = y - np.mean(y)
        return residual

    X = np.column_stack([np.ones_like(x), x])
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta_hat

    residual[idx_fit] = y - y_hat
    return residual


# ============================================================
# Main leakage computation
# ============================================================

def append_sampled(store: dict[int, list[np.ndarray]], dataset_id: int, values: np.ndarray, max_n: int) -> None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return

    if dataset_id not in store:
        store[dataset_id] = []

    store[dataset_id].append(values)

    total = sum(len(x) for x in store[dataset_id])
    if total > int(max_n * 1.25):
        all_vals = np.concatenate(store[dataset_id])
        if len(all_vals) > max_n:
            rng = np.random.default_rng(42 + int(dataset_id))
            keep = rng.choice(len(all_vals), size=max_n, replace=False)
            all_vals = all_vals[keep]
        store[dataset_id] = [all_vals]


def finalize_store(store: dict[int, list[np.ndarray]], dataset_id: int, max_n: int) -> np.ndarray:
    if dataset_id not in store or not store[dataset_id]:
        return np.array([], dtype=np.float64)

    arr = np.concatenate(store[dataset_id])
    arr = arr[np.isfinite(arr)]

    if len(arr) > max_n:
        rng = np.random.default_rng(42 + int(dataset_id))
        keep = rng.choice(len(arr), size=max_n, replace=False)
        arr = arr[keep]

    return arr


def compute_biology_leakage_metrics(
    df: pd.DataFrame,
    *,
    id_to_dataset: dict[int, str],
    best_b_template_lag: dict[int, int],
    css_window: int,
    max_positions_per_dataset: int,
) -> pd.DataFrame:
    """
    Computes per-dataset leakage metrics.

    Main quantities:
        |b| enrichment at CSS
        residual |b| enrichment at CSS
        raw target CSS enrichment
        corrected target CSS enrichment
        amount of CSS enrichment removed by b
        PCC(b, log L_queue)
        PCC(|b|, log L_queue)
        PCC(residual b, log L_queue)
    """

    stores: dict[str, dict[int, list[np.ndarray]]] = {
        "css_b": {},
        "bg_b": {},
        "css_abs_b": {},
        "bg_abs_b": {},
        "css_b_resid": {},
        "bg_b_resid": {},
        "css_abs_b_resid": {},
        "bg_abs_b_resid": {},
        "css_T_raw": {},
        "bg_T_raw": {},
        "css_T_corr": {},
        "bg_T_corr": {},
        "css_L": {},
        "bg_L": {},
    }

    transcript_corr_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing CSS leakage metrics"):
        dataset_id = int(row["dataset_id"])
        L = int(row["length"])

        y = safe_array(row["target"])[:L]
        L_queue = safe_array(row["L_queue"])[:L]
        total_scale = safe_array(row["total_scale"])[:L]
        b = safe_array(row["b_offset"])[:L]

        L_eff = min(len(y), len(L_queue), len(total_scale), len(b))
        if L_eff < 5:
            continue

        y = y[:L_eff]
        L_queue = L_queue[:L_eff]
        total_scale = total_scale[:L_eff]
        b = b[:L_eff]

        css_idx = parse_css_indices(row["css"], L_eff)
        css_mask = css_window_mask(css_idx, L_eff, css_window=css_window)
        bg_mask = ~css_mask

        exp_b = np.exp(np.clip(b, -20.0, 20.0))
        base_scale = total_scale / np.maximum(exp_b, EPS)

        T_raw = y / np.maximum(base_scale, EPS)
        T_corr = y / np.maximum(total_scale, EPS)

        best_lag = best_b_template_lag.get(dataset_id, 0)
        b_resid = residualize_b_offset_against_shift_template(
            L_queue=L_queue,
            b_offset=b,
            lag=best_lag,
        )

        log_L = np.log(np.maximum(L_queue, EPS))

        transcript_corr_rows.append(
            {
                "dataset_id": dataset_id,
                "pcc_b_logL": safe_pcc(b, log_L),
                "pcc_absb_logL": safe_pcc(np.abs(b), log_L),
                "pcc_bresid_logL": safe_pcc(b_resid, log_L),
                "pcc_abs_bresid_logL": safe_pcc(np.abs(b_resid), log_L),
            }
        )

        for key, arr, mask in [
            ("css_b", b, css_mask),
            ("bg_b", b, bg_mask),
            ("css_abs_b", np.abs(b), css_mask),
            ("bg_abs_b", np.abs(b), bg_mask),
            ("css_b_resid", b_resid, css_mask),
            ("bg_b_resid", b_resid, bg_mask),
            ("css_abs_b_resid", np.abs(b_resid), css_mask),
            ("bg_abs_b_resid", np.abs(b_resid), bg_mask),
            ("css_T_raw", T_raw, css_mask),
            ("bg_T_raw", T_raw, bg_mask),
            ("css_T_corr", T_corr, css_mask),
            ("bg_T_corr", T_corr, bg_mask),
            ("css_L", L_queue, css_mask),
            ("bg_L", L_queue, bg_mask),
        ]:
            append_sampled(stores[key], dataset_id, arr[mask], max_positions_per_dataset)

    corr_df = pd.DataFrame(transcript_corr_rows)

    rows = []

    for dataset_id in sorted(df["dataset_id"].unique()):
        dataset_id = int(dataset_id)
        dataset_name = id_to_dataset.get(dataset_id, f"dataset_{dataset_id}")

        vals = {
            key: finalize_store(stores[key], dataset_id, max_positions_per_dataset)
            for key in stores
        }

        med = {key: safe_median(val) for key, val in vals.items()}

        css_abs_b_enrichment = robust_ratio(med["css_abs_b"], med["bg_abs_b"])
        css_abs_b_resid_enrichment = robust_ratio(med["css_abs_b_resid"], med["bg_abs_b_resid"])

        css_signed_b_delta = med["css_b"] - med["bg_b"]
        css_signed_b_resid_delta = med["css_b_resid"] - med["bg_b_resid"]

        css_L_enrichment = robust_ratio(med["css_L"], med["bg_L"])
        css_raw_target_enrichment = robust_ratio(med["css_T_raw"], med["bg_T_raw"])
        css_corrected_target_enrichment = robust_ratio(med["css_T_corr"], med["bg_T_corr"])

        css_enrichment_removed_by_b = css_raw_target_enrichment - css_corrected_target_enrichment
        css_log_enrichment_removed_by_b = (
            np.log(max(css_raw_target_enrichment, EPS))
            - np.log(max(css_corrected_target_enrichment, EPS))
            if np.isfinite(css_raw_target_enrichment) and np.isfinite(css_corrected_target_enrichment)
            else np.nan
        )

        sub_corr = corr_df[corr_df["dataset_id"] == dataset_id]

        pcc_b_logL = safe_median(sub_corr["pcc_b_logL"]) if not sub_corr.empty else np.nan
        pcc_absb_logL = safe_median(sub_corr["pcc_absb_logL"]) if not sub_corr.empty else np.nan
        pcc_bresid_logL = safe_median(sub_corr["pcc_bresid_logL"]) if not sub_corr.empty else np.nan
        pcc_abs_bresid_logL = safe_median(sub_corr["pcc_abs_bresid_logL"]) if not sub_corr.empty else np.nan

        try:
            u_abs_b, p_abs_b = mannwhitneyu(
                vals["css_abs_b"],
                vals["bg_abs_b"],
                alternative="greater",
            )
            p_css_abs_b = float(p_abs_b)
        except Exception:
            p_css_abs_b = np.nan

        try:
            u_abs_b_resid, p_abs_b_resid = mannwhitneyu(
                vals["css_abs_b_resid"],
                vals["bg_abs_b_resid"],
                alternative="greater",
            )
            p_css_abs_b_resid = float(p_abs_b_resid)
        except Exception:
            p_css_abs_b_resid = np.nan

        leakage_score = (
            max(0.0, np.log(max(css_abs_b_enrichment, EPS)))
            + max(0.0, np.log(max(css_abs_b_resid_enrichment, EPS)))
            + max(0.0, css_log_enrichment_removed_by_b if np.isfinite(css_log_enrichment_removed_by_b) else 0.0)
            + abs(pcc_absb_logL if np.isfinite(pcc_absb_logL) else 0.0)
        )

        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset": dataset_name,
                "best_b_template_lag": int(best_b_template_lag.get(dataset_id, 0)),

                "n_css_b_positions": int(len(vals["css_b"])),
                "n_bg_b_positions": int(len(vals["bg_b"])),

                "median_css_b": med["css_b"],
                "median_bg_b": med["bg_b"],
                "css_signed_b_delta": css_signed_b_delta,

                "median_css_abs_b": med["css_abs_b"],
                "median_bg_abs_b": med["bg_abs_b"],
                "css_abs_b_enrichment": css_abs_b_enrichment,
                "p_css_abs_b_greater": p_css_abs_b,

                "median_css_b_resid": med["css_b_resid"],
                "median_bg_b_resid": med["bg_b_resid"],
                "css_signed_b_resid_delta": css_signed_b_resid_delta,

                "median_css_abs_b_resid": med["css_abs_b_resid"],
                "median_bg_abs_b_resid": med["bg_abs_b_resid"],
                "css_abs_b_resid_enrichment": css_abs_b_resid_enrichment,
                "p_css_abs_b_resid_greater": p_css_abs_b_resid,

                "css_L_enrichment": css_L_enrichment,
                "css_raw_target_enrichment": css_raw_target_enrichment,
                "css_corrected_target_enrichment": css_corrected_target_enrichment,
                "css_enrichment_removed_by_b": css_enrichment_removed_by_b,
                "css_log_enrichment_removed_by_b": css_log_enrichment_removed_by_b,

                "median_pcc_b_logL": pcc_b_logL,
                "median_pcc_absb_logL": pcc_absb_logL,
                "median_pcc_bresid_logL": pcc_bresid_logL,
                "median_pcc_abs_bresid_logL": pcc_abs_bresid_logL,

                "leakage_score": leakage_score,
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values("leakage_score", ascending=False).reset_index(drop=True)

    return summary_df


# ============================================================
# Codon hierarchy leakage
# ============================================================

def sample_dataset_arrays(
    codons: list[str],
    L_vals: list[float],
    b_vals: list[float],
    abs_b_vals: list[float],
    b_resid_vals: list[float],
    max_n: int,
    seed: int,
) -> tuple[list[str], list[float], list[float], list[float], list[float]]:
    n = len(codons)
    if n <= max_n:
        return codons, L_vals, b_vals, abs_b_vals, b_resid_vals

    rng = np.random.default_rng(seed)
    keep = rng.choice(n, size=max_n, replace=False)

    return (
        [codons[i] for i in keep],
        [L_vals[i] for i in keep],
        [b_vals[i] for i in keep],
        [abs_b_vals[i] for i in keep],
        [b_resid_vals[i] for i in keep],
    )


def compute_codon_leakage(
    merged_df: pd.DataFrame,
    *,
    onehot2nt: dict[int, str],
    id_to_dataset: dict[int, str],
    best_b_template_lag: dict[int, int],
    a_site_shift: int,
    exclude_terminal_codons: int,
    max_codon_positions_per_dataset: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each dataset and codon, computes median L_queue, b_offset, |b_offset|,
    residual b_offset, then correlates codon hierarchies.
    """

    dataset_stores: dict[int, dict[str, list[Any]]] = {}

    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Computing codon leakage"):
        dataset_id = int(row["dataset_id"])
        L = int(row["length"])

        try:
            codons = decode_ref_codons(row["ref"], onehot2nt)
        except Exception:
            continue

        L_queue = safe_array(row["L_queue"])[:L]
        b = safe_array(row["b_offset"])[:L]

        T = min(len(codons), len(L_queue), len(b))
        if T < 5:
            continue

        codons = np.asarray(codons[:T], dtype=object)
        L_queue = L_queue[:T]
        b = b[:T]

        best_lag = best_b_template_lag.get(dataset_id, 0)
        b_resid = residualize_b_offset_against_shift_template(L_queue, b, best_lag)

        pos = np.arange(T)
        codon_pos = pos + int(a_site_shift)

        valid = (codon_pos >= 0) & (codon_pos < T)

        if exclude_terminal_codons > 0:
            valid &= (
                (pos >= exclude_terminal_codons)
                & (pos < T - exclude_terminal_codons)
                & (codon_pos >= exclude_terminal_codons)
                & (codon_pos < T - exclude_terminal_codons)
            )

        if not np.any(valid):
            continue

        aligned_codons = codons[codon_pos[valid]]
        aligned_L = L_queue[valid]
        aligned_b = b[valid]
        aligned_b_resid = b_resid[valid]

        keep = np.array([c not in STOP_CODONS for c in aligned_codons], dtype=bool)

        aligned_codons = aligned_codons[keep]
        aligned_L = aligned_L[keep]
        aligned_b = aligned_b[keep]
        aligned_b_resid = aligned_b_resid[keep]

        finite = (
            np.isfinite(aligned_L)
            & np.isfinite(aligned_b)
            & np.isfinite(aligned_b_resid)
        )

        aligned_codons = aligned_codons[finite]
        aligned_L = aligned_L[finite]
        aligned_b = aligned_b[finite]
        aligned_b_resid = aligned_b_resid[finite]

        if len(aligned_codons) == 0:
            continue

        if dataset_id not in dataset_stores:
            dataset_stores[dataset_id] = {
                "codon": [],
                "L_queue": [],
                "b_offset": [],
                "abs_b_offset": [],
                "b_resid": [],
            }

        store = dataset_stores[dataset_id]
        store["codon"].extend(aligned_codons.tolist())
        store["L_queue"].extend(aligned_L.astype(float).tolist())
        store["b_offset"].extend(aligned_b.astype(float).tolist())
        store["abs_b_offset"].extend(np.abs(aligned_b).astype(float).tolist())
        store["b_resid"].extend(aligned_b_resid.astype(float).tolist())

        if len(store["codon"]) > int(max_codon_positions_per_dataset * 1.25):
            sampled = sample_dataset_arrays(
                store["codon"],
                store["L_queue"],
                store["b_offset"],
                store["abs_b_offset"],
                store["b_resid"],
                max_n=max_codon_positions_per_dataset,
                seed=42 + dataset_id,
            )

            store["codon"], store["L_queue"], store["b_offset"], store["abs_b_offset"], store["b_resid"] = sampled

    codon_summary_frames = []
    dataset_rows = []

    for dataset_id, store in dataset_stores.items():
        dataset_id = int(dataset_id)
        dataset_name = id_to_dataset.get(dataset_id, f"dataset_{dataset_id}")

        sampled = sample_dataset_arrays(
            store["codon"],
            store["L_queue"],
            store["b_offset"],
            store["abs_b_offset"],
            store["b_resid"],
            max_n=max_codon_positions_per_dataset,
            seed=42 + dataset_id,
        )

        codons, L_vals, b_vals, abs_b_vals, b_resid_vals = sampled

        flat = pd.DataFrame(
            {
                "dataset_id": dataset_id,
                "dataset": dataset_name,
                "codon": codons,
                "L_queue": L_vals,
                "b_offset": b_vals,
                "abs_b_offset": abs_b_vals,
                "b_resid": b_resid_vals,
            }
        )

        summary = (
            flat.groupby(["dataset_id", "dataset", "codon"], observed=True)
            .agg(
                median_L_queue=("L_queue", "median"),
                median_b_offset=("b_offset", "median"),
                median_abs_b_offset=("abs_b_offset", "median"),
                median_b_resid=("b_resid", "median"),
                count=("codon", "size"),
            )
            .reset_index()
        )

        codon_summary_frames.append(summary)

        if len(summary) >= 10:
            r_b = safe_spearman(summary["median_L_queue"], summary["median_b_offset"])
            r_abs_b = safe_spearman(summary["median_L_queue"], summary["median_abs_b_offset"])
            r_b_resid = safe_spearman(summary["median_L_queue"], summary["median_b_resid"])
        else:
            r_b = np.nan
            r_abs_b = np.nan
            r_b_resid = np.nan

        dataset_rows.append(
            {
                "dataset_id": dataset_id,
                "dataset": dataset_name,
                "n_codons": int(len(summary)),
                "spearman_codon_L_vs_b": r_b,
                "spearman_codon_L_vs_abs_b": r_abs_b,
                "spearman_codon_L_vs_b_resid": r_b_resid,
            }
        )

    if codon_summary_frames:
        codon_summary_df = pd.concat(codon_summary_frames, ignore_index=True)
    else:
        codon_summary_df = pd.DataFrame()

    codon_dataset_df = pd.DataFrame(dataset_rows)
    return codon_summary_df, codon_dataset_df


# ============================================================
# Plotting
# ============================================================

def plot_horizontal_bar(
    df: pd.DataFrame,
    *,
    value_col: str,
    title: str,
    xlabel: str,
    out_path: Path,
    color: str = "#4477AA",
    reference: float | None = None,
) -> None:
    plot_df = df.copy()
    plot_df = plot_df[np.isfinite(plot_df[value_col])].copy()

    if plot_df.empty:
        print(f"[plot] Skipping {value_col}: no finite values.")
        return

    plot_df = plot_df.sort_values(value_col, ascending=True)
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(10, max(6, 0.32 * len(plot_df))))

    ax.barh(
        y,
        plot_df[value_col].to_numpy(dtype=float),
        color=color,
        edgecolor="black",
        alpha=0.9,
    )

    if reference is not None:
        ax.axvline(reference, color="black", linestyle="--", linewidth=1.2)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["dataset"])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_css_target_enrichment(summary_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = summary_df.copy()
    plot_df = plot_df[
        np.isfinite(plot_df["css_raw_target_enrichment"])
        & np.isfinite(plot_df["css_corrected_target_enrichment"])
    ].copy()

    if plot_df.empty:
        return

    plot_df = plot_df.sort_values("css_enrichment_removed_by_b", ascending=True)

    y = np.arange(len(plot_df))
    h = 0.38

    fig, ax = plt.subplots(figsize=(11, max(6, 0.34 * len(plot_df))))

    ax.barh(
        y - h / 2,
        plot_df["css_raw_target_enrichment"],
        height=h,
        color="#D55E00",
        edgecolor="black",
        label=r"Raw target: $y/S$",
    )

    ax.barh(
        y + h / 2,
        plot_df["css_corrected_target_enrichment"],
        height=h,
        color="#0072B2",
        edgecolor="black",
        label=r"b-corrected target: $y/(S e^b)$",
    )

    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.0)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["dataset"])
    ax.set_xlabel("CSS / background median enrichment")
    ax.set_title(
        "Does b_offset remove CSS enrichment from the target?\n"
        r"$R_{raw}=\frac{median_{CSS}(y/S)}{median_{BG}(y/S)}$, "
        r"$R_{corr}=\frac{median_{CSS}(y/(S e^b))}{median_{BG}(y/(S e^b))}$"
    )
    ax.legend(loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_leakage_heatmap(summary_df: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        "css_abs_b_enrichment",
        "css_abs_b_resid_enrichment",
        "css_enrichment_removed_by_b",
        "median_pcc_absb_logL",
        "median_pcc_abs_bresid_logL",
        "spearman_codon_L_vs_abs_b",
        "spearman_codon_L_vs_b_resid",
        "leakage_score",
    ]

    available = [m for m in metrics if m in summary_df.columns]

    plot_df = summary_df[["dataset", *available]].copy()
    plot_df = plot_df.set_index("dataset")

    # Column-wise z-score so heterogeneous metrics can be seen together.
    z = plot_df.copy()
    for col in z.columns:
        x = z[col].to_numpy(dtype=float)
        mask = np.isfinite(x)
        if mask.sum() < 2 or np.nanstd(x[mask]) <= 1e-12:
            z[col] = np.nan
        else:
            z[col] = (x - np.nanmean(x[mask])) / np.nanstd(x[mask])

    data = z.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(14, max(6, 0.34 * len(z))))

    im = ax.imshow(np.ma.masked_invalid(data), aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)

    ax.set_xticks(np.arange(len(z.columns)))
    ax.set_xticklabels(z.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(z.index)))
    ax.set_yticklabels(z.index)

    ax.set_title(
        "Biology leakage summary heatmap\n"
        "Values are z-scored per metric; warmer means more suspicious relative to other datasets."
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Column-wise z-score")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def zscore_profile(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.isfinite(x)

    out = np.full_like(x, np.nan, dtype=np.float64)

    if m.sum() < 3:
        return out

    mu = np.nanmean(x[m])
    sd = np.nanstd(x[m])

    if sd <= 1e-12:
        return out

    out[m] = (x[m] - mu) / sd
    return out


def plot_css_metagene_for_top_datasets(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    best_b_template_lag: dict[int, int],
    out_dir: Path,
    top_n: int,
    window: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    top = summary_df.sort_values("leakage_score", ascending=False).head(top_n)

    for _, ds_row in top.iterrows():
        dataset_id = int(ds_row["dataset_id"])
        dataset_name = ds_row["dataset"]

        profiles_L = []
        profiles_b = []
        profiles_abs_b = []
        profiles_b_resid = []

        group = df[df["dataset_id"] == dataset_id]

        for _, row in group.iterrows():
            L = int(row["length"])

            css_idx = parse_css_indices(row["css"], L)
            if css_idx.size == 0:
                continue

            L_queue = safe_array(row["L_queue"])[:L]
            b = safe_array(row["b_offset"])[:L]

            T = min(len(L_queue), len(b), L)
            if T < 2 * window + 1:
                continue

            L_queue = L_queue[:T]
            b = b[:T]

            b_resid = residualize_b_offset_against_shift_template(
                L_queue=L_queue,
                b_offset=b,
                lag=best_b_template_lag.get(dataset_id, 0),
            )

            for idx in css_idx:
                idx = int(idx)
                if idx - window < 0 or idx + window >= T:
                    continue

                profiles_L.append(L_queue[idx - window: idx + window + 1])
                profiles_b.append(b[idx - window: idx + window + 1])
                profiles_abs_b.append(np.abs(b[idx - window: idx + window + 1]))
                profiles_b_resid.append(b_resid[idx - window: idx + window + 1])

        if not profiles_L:
            continue

        x = np.arange(-window, window + 1)

        mat_L = np.stack(profiles_L)
        mat_b = np.stack(profiles_b)
        mat_abs_b = np.stack(profiles_abs_b)
        mat_b_resid = np.stack(profiles_b_resid)

        med_L = zscore_profile(np.nanmedian(mat_L, axis=0))
        med_b = zscore_profile(np.nanmedian(mat_b, axis=0))
        med_abs_b = zscore_profile(np.nanmedian(mat_abs_b, axis=0))
        med_b_resid = zscore_profile(np.nanmedian(mat_b_resid, axis=0))

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(x, med_L, linewidth=2.2, label=r"$L_{queue}$ median, z-scored", color="#0072B2")
        ax.plot(x, med_b, linewidth=2.2, label=r"$b_{offset}$ median, z-scored", color="#D55E00")
        ax.plot(x, med_abs_b, linewidth=2.2, label=r"$|b_{offset}|$ median, z-scored", color="#CC79A7")
        ax.plot(x, med_b_resid, linewidth=2.2, label=r"$b_{resid}$ median, z-scored", color="#009E73")

        ax.axvline(0, color="black", linestyle="--", linewidth=1.2, label="CSS")
        ax.axhline(0, color="black", linestyle=":", linewidth=0.8)

        ax.set_title(
            f"{dataset_name}: CSS-centered metagene of L_queue and b_offset\n"
            f"Leakage score={ds_row['leakage_score']:.3f}, "
            f"|b| CSS enrichment={ds_row['css_abs_b_enrichment']:.2f}x"
        )
        ax.set_xlabel("Distance from CSS, codons")
        ax.set_ylabel("Z-scored median profile")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

        fig.tight_layout()
        fig.savefig(out_dir / f"{sanitize_filename(dataset_name)}_css_metagene.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_codon_scatter_for_top_datasets(
    codon_summary_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    out_dir: Path,
    top_n: int,
) -> None:
    if codon_summary_df.empty:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    top = summary_df.sort_values("leakage_score", ascending=False).head(top_n)

    for _, ds_row in top.iterrows():
        dataset_id = int(ds_row["dataset_id"])
        dataset_name = ds_row["dataset"]

        sub = codon_summary_df[codon_summary_df["dataset_id"] == dataset_id].copy()
        if len(sub) < 10:
            continue

        r_b = safe_spearman(sub["median_L_queue"], sub["median_b_offset"])
        r_abs = safe_spearman(sub["median_L_queue"], sub["median_abs_b_offset"])
        r_resid = safe_spearman(sub["median_L_queue"], sub["median_b_resid"])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        panels = [
            ("median_b_offset", r"$median(b_i \mid codon)$", r_b),
            ("median_abs_b_offset", r"$median(|b_i| \mid codon)$", r_abs),
            ("median_b_resid", r"$median(b^{resid}_i \mid codon)$", r_resid),
        ]

        for ax, (ycol, ylabel, r) in zip(axes, panels):
            ax.scatter(
                sub["median_L_queue"],
                sub[ycol],
                s=np.sqrt(sub["count"].to_numpy(dtype=float)) * 2.5,
                alpha=0.75,
                edgecolor="black",
                linewidth=0.4,
            )

            for _, rrow in sub.iterrows():
                ax.text(
                    rrow["median_L_queue"],
                    rrow[ycol],
                    str(rrow["codon"]),
                    fontsize=7,
                    alpha=0.75,
                )

            ax.set_xlabel(r"$median(L_{queue} \mid codon)$")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Spearman={r:.3f}" if np.isfinite(r) else "Spearman=NA")
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"{dataset_name}: codon hierarchy leakage check\n"
            "If high-L codons also have high b_offset, biology may be leaking into the correction head.",
            fontsize=13,
        )

        fig.tight_layout()
        fig.savefig(out_dir / f"{sanitize_filename(dataset_name)}_codon_leakage_scatter.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def make_all_plots(
    *,
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    codon_summary_df: pd.DataFrame,
    best_b_template_lag: dict[int, int],
    out_dir: Path,
    top_n_metagene: int,
    css_metagene_window: int,
) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_horizontal_bar(
        summary_df,
        value_col="leakage_score",
        title=(
            "Overall biology-leakage score for b_offset\n"
            "Higher means b_offset is more aligned with known biological signals."
        ),
        xlabel="Leakage score",
        out_path=plot_dir / "01_leakage_score.png",
        color="#882255",
    )

    plot_horizontal_bar(
        summary_df,
        value_col="css_abs_b_enrichment",
        title=(
            r"CSS enrichment of $|b_{offset}|$" + "\n"
            r"$R_{|b|,CSS} = median_{CSS}(|b_i|) / median_{BG}(|b_i|)$"
        ),
        xlabel=r"CSS / background enrichment of $|b_{offset}|$",
        out_path=plot_dir / "02_css_abs_b_enrichment.png",
        color="#CC6677",
        reference=1.0,
    )

    plot_horizontal_bar(
        summary_df,
        value_col="css_abs_b_resid_enrichment",
        title=(
            r"CSS enrichment of residual $|b_{offset}|$ after removing shift-template component" + "\n"
            r"$b_i = \alpha + \beta B_i(k_b^\star) + b_i^{resid}$"
        ),
        xlabel=r"CSS / background enrichment of $|b^{resid}|$",
        out_path=plot_dir / "03_css_abs_b_residual_enrichment.png",
        color="#AA4499",
        reference=1.0,
    )

    plot_css_target_enrichment(
        summary_df,
        out_path=plot_dir / "04_css_target_enrichment_raw_vs_corrected.png",
    )

    plot_horizontal_bar(
        summary_df,
        value_col="css_enrichment_removed_by_b",
        title=(
            "CSS enrichment removed by b_offset\n"
            r"$R_{raw} - R_{corr}$, where "
            r"$R_{raw}=CSS(y/S)/BG(y/S)$ and "
            r"$R_{corr}=CSS(y/(S e^b))/BG(y/(S e^b))$"
        ),
        xlabel=r"$R_{raw} - R_{corr}$",
        out_path=plot_dir / "05_css_enrichment_removed_by_b.png",
        color="#DDCC77",
        reference=0.0,
    )

    plot_horizontal_bar(
        summary_df,
        value_col="median_pcc_absb_logL",
        title=(
            r"Transcript-level association between $|b_{offset}|$ and biological queue signal" + "\n"
            r"$median_t\;PCC(|b_i|, \log(L_{queue,i}+\epsilon))$"
        ),
        xlabel=r"Median transcript PCC",
        out_path=plot_dir / "06_abs_b_vs_logL_pcc.png",
        color="#117733",
        reference=0.0,
    )

    if "spearman_codon_L_vs_abs_b" in summary_df.columns:
        plot_horizontal_bar(
            summary_df,
            value_col="spearman_codon_L_vs_abs_b",
            title=(
                "Codon hierarchy leakage\n"
                r"$Spearman_c(median(L_{queue}|c), median(|b_i||c))$"
            ),
            xlabel="Spearman correlation across codons",
            out_path=plot_dir / "07_codon_L_vs_abs_b_spearman.png",
            color="#332288",
            reference=0.0,
        )

    plot_leakage_heatmap(
        summary_df,
        out_path=plot_dir / "08_leakage_summary_heatmap.png",
    )

    plot_css_metagene_for_top_datasets(
        df,
        summary_df,
        best_b_template_lag=best_b_template_lag,
        out_dir=plot_dir / "css_metagenes_top_datasets",
        top_n=top_n_metagene,
        window=css_metagene_window,
    )

    if not codon_summary_df.empty:
        plot_codon_scatter_for_top_datasets(
            codon_summary_df,
            summary_df,
            out_dir=plot_dir / "codon_scatter_top_datasets",
            top_n=top_n_metagene,
        )


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prediction-dir",
        type=str,
        default="./riboai_queueing/33_datasets_mix_6e5e33",
        help="Directory containing comprehensive_predictions_rank*.parquet files.",
    )
    parser.add_argument(
        "--prediction-glob",
        type=str,
        default=None,
        help="Optional explicit glob for prediction parquet files. Overrides --prediction-dir.",
    )
    parser.add_argument(
        "--dataset-encoding",
        type=str,
        default="../Datasets/encodings/dataset_encoding.yaml",
        help="Path to dataset_encoding.yaml.",
    )
    parser.add_argument(
        "--sequence-path",
        type=str,
        default="../Datasets/data/sequence/sequence_embeddings_with_css.parquet",
        help="Path to sequence parquet with transcript_id and ref.",
    )
    parser.add_argument(
        "--nt-encoding",
        type=str,
        default="../Datasets/encodings/nt_encoding.yaml",
        help="Path to nt_encoding.yaml.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="b_offset_biology_leakage_diagnostics",
        help="Output directory.",
    )
    parser.add_argument(
        "--lags",
        type=int,
        nargs="+",
        default=[-3, -2, -1, 1, 2, 3],
        help="Candidate nonzero lags used to residualize shift-like b_offset.",
    )
    parser.add_argument(
        "--gradient-quantile",
        type=float,
        default=0.50,
        help="Quantile filter for b-template shift detection. Use -1 to disable.",
    )
    parser.add_argument(
        "--css-window",
        type=int,
        default=1,
        help="CSS tolerance window in codons. Default means CSS +/- 1 codon.",
    )
    parser.add_argument(
        "--css-metagene-window",
        type=int,
        default=30,
        help="Window size around CSS for metagene plots.",
    )
    parser.add_argument(
        "--max-positions-per-dataset",
        type=int,
        default=1_000_000,
        help="Maximum positions sampled per dataset for CSS/background summaries.",
    )
    parser.add_argument(
        "--max-codon-positions-per-dataset",
        type=int,
        default=300_000,
        help="Maximum codon positions sampled per dataset for codon hierarchy.",
    )
    parser.add_argument(
        "--a-site-shift",
        type=int,
        default=0,
        help="Codon alignment shift for codon hierarchy. metric[i] maps to codon[i + a_site_shift].",
    )
    parser.add_argument(
        "--exclude-terminal-codons",
        type=int,
        default=1,
        help="Number of terminal codons excluded from codon hierarchy.",
    )
    parser.add_argument(
        "--top-n-metagene",
        type=int,
        default=8,
        help="Number of most suspicious datasets for detailed CSS metagene and codon scatter plots.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gradient_quantile = None if args.gradient_quantile < 0 else float(args.gradient_quantile)

    id_to_dataset = load_dataset_names(args.dataset_encoding)

    df = load_predictions(
        prediction_dir=args.prediction_dir,
        prediction_glob=args.prediction_glob,
    )

    # --------------------------------------------------------
    # 1. Identify best shift-template lag per dataset
    # --------------------------------------------------------
    b_template_lag_df, best_b_template_lag = compute_dataset_best_b_template_lag(
        df,
        id_to_dataset=id_to_dataset,
        lags=list(args.lags),
        gradient_quantile=gradient_quantile,
    )

    b_template_lag_df.to_csv(out_dir / "b_template_lag_scan.csv", index=False)

    # --------------------------------------------------------
    # 2. CSS and target-enrichment leakage metrics
    # --------------------------------------------------------
    leakage_summary_df = compute_biology_leakage_metrics(
        df,
        id_to_dataset=id_to_dataset,
        best_b_template_lag=best_b_template_lag,
        css_window=int(args.css_window),
        max_positions_per_dataset=int(args.max_positions_per_dataset),
    )

    # --------------------------------------------------------
    # 3. Optional codon hierarchy leakage metrics
    # --------------------------------------------------------
    seq_df = load_sequence_refs(args.sequence_path)
    onehot2nt = load_onehot2nt(args.nt_encoding)

    codon_summary_df = pd.DataFrame()
    codon_dataset_df = pd.DataFrame()

    if seq_df is not None and onehot2nt is not None:
        print("Merging predictions with sequence references for codon leakage analyses.")
        merged_df = df.merge(seq_df, on="transcript_id", how="inner")

        codon_summary_df, codon_dataset_df = compute_codon_leakage(
            merged_df,
            onehot2nt=onehot2nt,
            id_to_dataset=id_to_dataset,
            best_b_template_lag=best_b_template_lag,
            a_site_shift=int(args.a_site_shift),
            exclude_terminal_codons=int(args.exclude_terminal_codons),
            max_codon_positions_per_dataset=int(args.max_codon_positions_per_dataset),
        )

        codon_summary_df.to_csv(out_dir / "codon_leakage_summary_by_dataset_codon.csv", index=False)
        codon_dataset_df.to_csv(out_dir / "codon_leakage_dataset_summary.csv", index=False)

        if not codon_dataset_df.empty:
            leakage_summary_df = leakage_summary_df.merge(
                codon_dataset_df,
                on=["dataset_id", "dataset"],
                how="left",
            )

            # Update leakage score with codon leakage terms.
            leakage_summary_df["leakage_score"] = (
                leakage_summary_df["leakage_score"].fillna(0.0)
                + leakage_summary_df["spearman_codon_L_vs_abs_b"].abs().fillna(0.0)
                + leakage_summary_df["spearman_codon_L_vs_b_resid"].abs().fillna(0.0)
            )

            leakage_summary_df = leakage_summary_df.sort_values(
                "leakage_score",
                ascending=False,
            ).reset_index(drop=True)

    leakage_summary_df.to_csv(out_dir / "b_offset_biology_leakage_summary.csv", index=False)

    # --------------------------------------------------------
    # 4. Plots
    # --------------------------------------------------------
    make_all_plots(
        df=df,
        summary_df=leakage_summary_df,
        codon_summary_df=codon_summary_df,
        best_b_template_lag=best_b_template_lag,
        out_dir=out_dir,
        top_n_metagene=int(args.top_n_metagene),
        css_metagene_window=int(args.css_metagene_window),
    )

    print("\nSaved outputs:")
    print(f"  Main summary:        {out_dir / 'b_offset_biology_leakage_summary.csv'}")
    print(f"  b-template lags:     {out_dir / 'b_template_lag_scan.csv'}")
    print(f"  Plots:               {out_dir / 'plots'}")

    print("\nTop suspicious datasets:")
    cols = [
        "dataset",
        "leakage_score",
        "best_b_template_lag",
        "css_abs_b_enrichment",
        "css_abs_b_resid_enrichment",
        "css_enrichment_removed_by_b",
        "median_pcc_absb_logL",
    ]

    extra_cols = [
        "spearman_codon_L_vs_abs_b",
        "spearman_codon_L_vs_b_resid",
    ]

    cols = [c for c in [*cols, *extra_cols] if c in leakage_summary_df.columns]
    print(leakage_summary_df[cols].head(15).to_string(index=False))

    print("\nInterpretation:")
    print("  High css_abs_b_enrichment means b_offset is active at known CSS.")
    print("  High css_abs_b_resid_enrichment means this remains true after removing shift-like b_offset.")
    print("  Positive css_enrichment_removed_by_b means b_offset reduces CSS enrichment from the target.")
    print("  High median_pcc_absb_logL means |b_offset| is active where L_queue is high.")
    print("  High codon Spearman means b_offset shares codon-level hierarchy with L_queue.")


if __name__ == "__main__":
    main()