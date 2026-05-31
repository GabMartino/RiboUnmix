#!/usr/bin/env python3
"""
Residual-case analysis for RiboAI queueing predictions saved by `predictions_to_parquet`.

This version is designed for your current prediction pipeline:

    main_predictions = trainer.predict(...)
    predictions_to_parquet(predictions=main_predictions, out_file=...)
    # -> predictions_main_val_<dataset_str>.parquet
    # -> predictions_css_benchmark_<dataset_str>.parquet

and your current `predict_step`, which returns fields such as:

    ids, dataset_id, lengths, mask, css, y, mu_obs,
    L_bio, L_obs, mu_L_bio, mu_L_obs,
    w_bio, w_obs, obs_bias_*, etc.

No argparse is used. Edit only the CONFIG block below.

The script classifies residuals into:

A_shift_offset
    The target profile matches prediction better after a small positional shift.

B_unexplained_positive_peaks_additive_candidate
    Strong positive target residual peaks occur where predicted/support signal is low
    and not near predicted peaks. This is the signature that would justify a sparse
    additive artifact branch.

C_smooth_broad_trend
    Residuals are broad/low-frequency, suggesting smooth positional bias rather than
    sparse additive peaks.

D_noise_low_signal_likelihood
    Residuals are weak/diffuse/low-signal, suggesting likelihood/noise/replicate
    uncertainty rather than a new deterministic branch.
"""

from __future__ import annotations

import ast
import json
import math
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import torch
except Exception:
    torch = None

try:
    import yaml
except Exception:
    yaml = None


# =============================================================================
# EDIT ONLY THIS BLOCK
# =============================================================================

CONFIG = {
    # Main validation parquet produced by your code:
    #   predictions_main_val_<dataset_str>.parquet
    #
    # Example:
    "pred": "results/riboai_queueing/grimson_2019_kutay_2021/NOPCGrad_random_dataset_per_transcript_SampleMeanLoss/predictions_main_val_grimson_2019_kutay_2021.parquet",

    # Output directory.
    "out_dir": "results/riboai_queueing/residual_case_analysis_main_val",

    # Optional. Used only to map dataset_id -> dataset name.
    "dataset_encoding": "Datasets/encodings/dataset_encoding.yaml",

    # Keys coming from predict_step / parquet columns.
    "target_key": "y",

    # Use mu_L_obs if exported; otherwise the script falls back to mu_obs.
    # Your current predict_step always exports mu_obs and exports mu_L_obs if present
    # in extras/important.
    "pred_key": "mu_L_obs",

    "fallback_pred_keys": ["mu_L_obs", "mu_obs", "mu", "pred"],

    # Supports used to classify residuals.
    "bio_support_key": "L_bio",
    "obs_support_key": "L_obs",

    "fallback_bio_support_keys": ["L_bio", "L_queue", "bio_q_base"],
    "fallback_obs_support_keys": ["L_obs", "L_queue_obs", "q", "bio_q", "mu_L_obs", "mu_obs"],

    # Metadata keys.
    # In your predict_step, the key is "ids", not "id".
    "id_key": "ids",
    "fallback_id_keys": ["ids", "id", "transcript_id", "transcript_ids"],

    "dataset_id_key": "dataset_id",
    "fallback_dataset_id_keys": ["dataset_id", "dataset_ids"],

    "length_key": "lengths",
    "fallback_length_keys": ["lengths", "length"],

    "mask_key": "mask",

    # For long-format parquet only. If your predictions_to_parquet writes one row
    # per transcript-dataset pair with list columns, this is ignored.
    "position_key": "position",

    # Residual classification parameters.
    "max_lag": 3,
    "peak_z": 2.0,
    "peak_tolerance": 2,
    "low_support_quantile": 0.25,
    "smooth_window": 101,

    # Case thresholds.
    "shift_gain_threshold": 0.03,
    "additive_score_threshold": 0.35,
    "smooth_frac_threshold": 0.35,

    # Numerical stability.
    "eps": 1.0e-12,

    # Plotting.
    "plot_top_n_per_case": 6,
    "make_plots": True,

    # Also save a small schema report to debug column names / row format.
    "write_schema_report": True,
}


# =============================================================================
# Generic utilities
# =============================================================================

def to_numpy(x: Any) -> Any:
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_numpy(v) for v in x]
    return x


def parse_maybe_serialized(x: Any) -> Any:
    """
    Parquet object/list columns usually come back as lists/np arrays.
    But some custom writers serialize arrays as strings. This handles both.
    """
    if x is None:
        return None

    if isinstance(x, float) and math.isnan(x):
        return None

    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()

    if isinstance(x, np.ndarray):
        return x

    if isinstance(x, (list, tuple)):
        return x

    if isinstance(x, bytes):
        try:
            x = x.decode("utf-8")
        except Exception:
            return x

    if isinstance(x, str):
        s = x.strip()
        if s in {"", "None", "none", "null", "NULL", "nan", "NaN"}:
            return None
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                return ast.literal_eval(s)
            except Exception:
                try:
                    return json.loads(s)
                except Exception:
                    return x
        return x

    return x


def as_1d_float(x: Any, valid_len: int | None = None) -> np.ndarray:
    x = parse_maybe_serialized(x)
    if x is None:
        arr = np.asarray([], dtype=float)
    else:
        arr = np.asarray(x, dtype=float).reshape(-1)

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    if valid_len is not None:
        arr = arr[:valid_len]

    return arr


def as_1d_bool(x: Any) -> np.ndarray:
    x = parse_maybe_serialized(x)
    if x is None:
        return np.asarray([], dtype=bool)

    arr = np.asarray(x)

    if arr.dtype == bool:
        return arr.reshape(-1)

    try:
        arr_f = arr.astype(float).reshape(-1)
        return arr_f > 0.5
    except Exception:
        return np.asarray([], dtype=bool)


def normalize_profile(x: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, None)
    s = float(x.sum())

    if s <= eps:
        if len(x) == 0:
            return x
        return np.ones_like(x, dtype=float) / float(len(x))

    return x / s


def pearson(x: np.ndarray, y: np.ndarray, eps: float = 1.0e-12) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.size < 2 or y.size < 2:
        return np.nan

    x = x - x.mean()
    y = y - y.mean()

    denom = math.sqrt(float((x * x).sum() * (y * y).sum()))

    if denom <= eps:
        return np.nan

    return float((x * y).sum() / denom)


def robust_z(x: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))

    if mad <= eps:
        sd = float(np.std(x))
        if sd <= eps:
            return np.zeros_like(x)
        return (x - float(np.mean(x))) / sd

    return 0.67448975 * (x - med) / mad


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if window <= 1 or x.size <= 2:
        return x.copy()

    window = int(min(window, x.size))

    if window % 2 == 0:
        window += 1

    if window > x.size:
        window = x.size if x.size % 2 == 1 else x.size - 1

    if window <= 1:
        return x.copy()

    pad = window // 2
    xp = np.pad(x, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)

    return np.convolve(xp, kernel, mode="valid")


def local_peaks_from_z(z: np.ndarray, threshold: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)

    if z.size == 0:
        return np.asarray([], dtype=int)

    left = np.empty_like(z)
    right = np.empty_like(z)

    left[0] = -np.inf
    left[1:] = z[:-1]

    right[-1] = -np.inf
    right[:-1] = z[1:]

    peak_mask = (z >= threshold) & (z >= left) & (z >= right)

    return np.flatnonzero(peak_mask).astype(int)


def any_near(index: int, candidates: np.ndarray, tol: int) -> bool:
    if candidates.size == 0:
        return False
    return bool(np.any(np.abs(candidates.astype(int) - int(index)) <= int(tol)))


def shifted_corrs(y: np.ndarray, pred: np.ndarray, max_lag: int) -> dict[int, float]:
    """
    corr(y_i, pred_{i+lag}) for lag in [-max_lag, max_lag].
    """
    out: dict[int, float] = {}
    L = len(y)

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            yy = y[-lag:]
            pp = pred[: L + lag]
        elif lag > 0:
            yy = y[: L - lag]
            pp = pred[lag:]
        else:
            yy = y
            pp = pred

        out[lag] = pearson(yy, pp)

    return out


def first_existing_key(columns: Iterable[str], preferred: Iterable[str]) -> str | None:
    colset = set(columns)
    for k in preferred:
        if k in colset:
            return k
    return None


def first_existing_value(row: pd.Series | dict[str, Any], preferred: Iterable[str]) -> Any:
    for k in preferred:
        if k in row and row[k] is not None:
            return row[k]
    return None


# =============================================================================
# Loading
# =============================================================================

def load_dataset_encoding(path: str | None) -> dict[int, str]:
    if path is None:
        return {}

    p = Path(path)
    if not p.exists():
        print(f"[WARN] dataset encoding not found: {p}")
        return {}

    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            print("[WARN] PyYAML unavailable. Dataset ids will not be decoded.")
            return {}
        data = yaml.safe_load(p.read_text())

    elif p.suffix.lower() == ".json":
        data = json.loads(p.read_text())

    else:
        print(f"[WARN] unsupported dataset encoding format: {p.suffix}")
        return {}

    return {int(v): str(k) for k, v in data.items()}


def load_prediction_object(path: str) -> Any:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Prediction file not found: {p}")

    suffix = p.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(p)

    if suffix == ".csv":
        return pd.read_csv(p)

    if suffix in {".pt", ".pth"}:
        if torch is None:
            raise RuntimeError("PyTorch required to load .pt/.pth predictions.")
        return torch.load(p, map_location="cpu")

    if suffix in {".pkl", ".pickle"}:
        with p.open("rb") as f:
            return pickle.load(f)

    if suffix == ".npz":
        return dict(np.load(p, allow_pickle=True))

    raise ValueError(f"Unsupported prediction file format: {suffix}")


def infer_row_valid_len(row: pd.Series, cfg: dict[str, Any], y_arr: np.ndarray) -> int:
    mask_key = first_existing_key(row.index, [cfg["mask_key"]])
    length_key = first_existing_key(row.index, cfg["fallback_length_keys"])

    if mask_key is not None:
        m = as_1d_bool(row[mask_key])
        if m.size > 0:
            return int(m.sum())

    if length_key is not None:
        try:
            return int(row[length_key])
        except Exception:
            arr = as_1d_float(row[length_key])
            if arr.size > 0:
                return int(arr.reshape(-1)[0])

    return int(len(y_arr))


def parquet_to_records(df: pd.DataFrame, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Handles two likely `predictions_to_parquet` layouts:

    1. Wide row format:
       one row per transcript-dataset sample, array/list columns:
       ids, dataset_id, lengths, mask, y, mu_obs, L_bio, L_obs, ...

    2. Long format:
       one row per position:
       ids/id/transcript_id, dataset_id, position, y, mu_obs, ...
    """
    columns = list(df.columns)

    target_key = first_existing_key(columns, [cfg["target_key"], "y", "target"])
    pred_key = first_existing_key(columns, [cfg["pred_key"]] + cfg["fallback_pred_keys"])

    if target_key is None:
        raise KeyError(f"Could not find target column. Tried {cfg['target_key']}, y, target. Columns={columns}")

    if pred_key is None:
        raise KeyError(f"Could not find prediction column. Tried {cfg['pred_key']} and fallbacks. Columns={columns}")

    id_key = first_existing_key(columns, [cfg["id_key"]] + cfg["fallback_id_keys"])
    dataset_id_key = first_existing_key(columns, [cfg["dataset_id_key"]] + cfg["fallback_dataset_id_keys"])
    position_key = cfg["position_key"] if cfg["position_key"] in columns else None

    # If position exists and y/pred are scalar columns, treat as long format.
    if position_key is not None:
        first_y = parse_maybe_serialized(df[target_key].iloc[0])
        y_is_scalar = np.asarray(first_y).ndim == 0

        if y_is_scalar:
            if id_key is None or dataset_id_key is None:
                raise KeyError("Long-format parquet requires id and dataset_id columns.")

            records = []
            bio_key = first_existing_key(columns, cfg["fallback_bio_support_keys"])
            obs_key = first_existing_key(columns, cfg["fallback_obs_support_keys"])

            for (tid, ds), g in df.groupby([id_key, dataset_id_key], sort=False):
                g = g.sort_values(position_key)
                rec = {
                    "id": tid,
                    "dataset_id": ds,
                    "length": int(len(g)),
                    "y": g[target_key].to_numpy(dtype=float),
                    "pred": g[pred_key].to_numpy(dtype=float),
                    "bio_support": g[bio_key].to_numpy(dtype=float) if bio_key is not None else None,
                    "obs_support": g[obs_key].to_numpy(dtype=float) if obs_key is not None else None,
                }
                records.append(rec)

            return records

    # Otherwise: wide row format with arrays/list columns.
    bio_key = first_existing_key(columns, [cfg["bio_support_key"]] + cfg["fallback_bio_support_keys"])
    obs_key = first_existing_key(columns, [cfg["obs_support_key"]] + cfg["fallback_obs_support_keys"])

    records = []

    for row_idx, row in df.iterrows():
        y_raw = as_1d_float(row[target_key])
        pred_raw = as_1d_float(row[pred_key])

        valid_len = infer_row_valid_len(row, cfg, y_raw)

        if valid_len <= 1:
            continue

        y = as_1d_float(row[target_key], valid_len)
        pred = as_1d_float(row[pred_key], valid_len)

        if len(y) <= 1 or len(pred) <= 1:
            continue

        tid = first_existing_value(row, [cfg["id_key"]] + cfg["fallback_id_keys"])
        ds = first_existing_value(row, [cfg["dataset_id_key"]] + cfg["fallback_dataset_id_keys"])

        bio_support = as_1d_float(row[bio_key], valid_len) if bio_key is not None and bio_key in row else None
        obs_support = as_1d_float(row[obs_key], valid_len) if obs_key is not None and obs_key in row else None

        rec = {
            "id": tid if tid is not None else f"row_{row_idx}",
            "dataset_id": ds if ds is not None else -1,
            "length": valid_len,
            "y": y,
            "pred": pred,
            "bio_support": bio_support,
            "obs_support": obs_support,
            "row_index": int(row_idx),
        }

        records.append(rec)

    return records


def list_batches_to_records(obj: Any, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Backward compatibility for direct trainer.predict output saved as .pt/.pkl.
    """
    obj = to_numpy(obj)

    if isinstance(obj, dict):
        for k in ["predictions", "outputs", "batches"]:
            if k in obj and isinstance(obj[k], list):
                obj = obj[k]
                break
        else:
            obj = [obj]

    if not isinstance(obj, list):
        raise ValueError(f"Expected list/dict prediction object, got {type(obj)!r}")

    records = []

    for batch_idx, batch in enumerate(obj):
        if not isinstance(batch, dict):
            continue

        y = first_existing_value(batch, [cfg["target_key"], "y", "target"])
        pred = first_existing_value(batch, [cfg["pred_key"]] + cfg["fallback_pred_keys"])

        if y is None or pred is None:
            continue

        y_arr = np.asarray(to_numpy(y))
        pred_arr = np.asarray(to_numpy(pred))

        if y_arr.ndim == 1:
            y_arr = y_arr[None, :]
        if pred_arr.ndim == 1:
            pred_arr = pred_arr[None, :]

        B = y_arr.shape[0]

        ids = first_existing_value(batch, [cfg["id_key"]] + cfg["fallback_id_keys"])
        ds_ids = first_existing_value(batch, [cfg["dataset_id_key"]] + cfg["fallback_dataset_id_keys"])
        lengths = first_existing_value(batch, [cfg["length_key"]] + cfg["fallback_length_keys"])
        masks = first_existing_value(batch, [cfg["mask_key"]])

        bio = first_existing_value(batch, [cfg["bio_support_key"]] + cfg["fallback_bio_support_keys"])
        obs = first_existing_value(batch, [cfg["obs_support_key"]] + cfg["fallback_obs_support_keys"])

        def item(x: Any, i: int, default: Any = None) -> Any:
            if x is None:
                return default
            x = to_numpy(x)
            if isinstance(x, (list, tuple)):
                return x[i] if i < len(x) else default
            arr = np.asarray(x, dtype=object if getattr(np.asarray(x), "dtype", None) == object else None)
            if arr.ndim == 0:
                return arr.item()
            if arr.shape[0] <= i:
                return default
            return arr[i]

        for i in range(B):
            if masks is not None:
                m = as_1d_bool(item(masks, i))
                valid_len = int(m.sum()) if m.size else y_arr.shape[1]
            elif lengths is not None:
                valid_len = int(np.asarray(item(lengths, i)).reshape(-1)[0])
            else:
                valid_len = y_arr.shape[1]

            if valid_len <= 1:
                continue

            records.append(
                {
                    "id": item(ids, i, f"batch{batch_idx}_sample{i}"),
                    "dataset_id": item(ds_ids, i, -1),
                    "length": valid_len,
                    "y": as_1d_float(y_arr[i], valid_len),
                    "pred": as_1d_float(pred_arr[i], valid_len),
                    "bio_support": as_1d_float(item(bio, i), valid_len) if bio is not None else None,
                    "obs_support": as_1d_float(item(obs, i), valid_len) if obs is not None else None,
                }
            )

    return records


def load_records(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], pd.DataFrame | None]:
    obj = load_prediction_object(cfg["pred"])

    if isinstance(obj, pd.DataFrame):
        return parquet_to_records(obj, cfg), obj

    return list_batches_to_records(obj, cfg), None


# =============================================================================
# Analysis
# =============================================================================

@dataclass
class ResidualCaseResult:
    sample_index: int
    transcript_id: str
    dataset_id: int | str
    dataset_name: str
    length: int

    pcc: float
    best_lag: int
    corr_lag0: float
    corr_best: float
    shift_gain: float
    case_A_shift_score: float

    positive_residual_peak_count: int
    unexplained_positive_peak_count: int
    unexplained_positive_peak_frac: float
    additive_residual_mass_frac: float
    case_B_additive_score: float

    lowfreq_var_frac: float
    broad_segment_count: int
    case_C_smooth_score: float

    residual_std: float
    robust_residual_peak_count: int
    low_signal_residual_mass_frac: float
    case_D_noise_score: float

    primary_case: str


def count_broad_segments(x: np.ndarray, threshold: float) -> int:
    mask = np.asarray(np.abs(x) >= threshold, dtype=bool)
    if mask.size == 0:
        return 0
    starts = mask & np.r_[True, ~mask[:-1]]
    return int(starts.sum())


def dataset_name(dataset_id: Any, mapping: dict[int, str]) -> str:
    try:
        return mapping.get(int(dataset_id), str(dataset_id))
    except Exception:
        return str(dataset_id)


def analyze_one(
    rec: dict[str, Any],
    *,
    sample_index: int,
    dataset_id_to_name: dict[int, str],
    cfg: dict[str, Any],
) -> tuple[ResidualCaseResult, dict[str, Any]]:
    eps = float(cfg["eps"])

    y_raw = as_1d_float(rec["y"], rec["length"])
    pred_raw = as_1d_float(rec["pred"], rec["length"])

    y = normalize_profile(y_raw, eps=eps)
    pred = normalize_profile(pred_raw, eps=eps)

    obs_support = rec.get("obs_support")
    bio_support = rec.get("bio_support")

    if obs_support is not None and len(obs_support) == len(y):
        support = normalize_profile(obs_support, eps=eps)
    elif bio_support is not None and len(bio_support) == len(y):
        support = normalize_profile(bio_support, eps=eps)
    else:
        support = pred

    resid = y - pred
    rz = robust_z(resid, eps=eps)

    # A: shift/offset.
    corrs = shifted_corrs(y, pred, max_lag=int(cfg["max_lag"]))
    corr0 = corrs.get(0, np.nan)
    valid_corrs = {k: v for k, v in corrs.items() if np.isfinite(v)}

    if valid_corrs:
        best_lag, corr_best = max(valid_corrs.items(), key=lambda kv: kv[1])
    else:
        best_lag, corr_best = 0, np.nan

    shift_gain = float(corr_best - corr0) if np.isfinite(corr_best) and np.isfinite(corr0) else 0.0
    case_A_shift_score = max(0.0, shift_gain) if int(best_lag) != 0 else 0.0

    # B: additive candidate.
    pos_resid_peaks = local_peaks_from_z(rz, threshold=float(cfg["peak_z"]))
    pred_z = robust_z(pred, eps=eps)
    pred_peaks = local_peaks_from_z(pred_z, threshold=float(cfg["peak_z"]))

    positive_support = support[support > eps]
    if positive_support.size > 0:
        low_support_thr = float(np.quantile(positive_support, float(cfg["low_support_quantile"])))
    else:
        low_support_thr = float(np.quantile(support, float(cfg["low_support_quantile"])))

    unexplained: list[int] = []
    additive_mass = 0.0
    total_positive_resid_mass = float(np.clip(resid, 0.0, None).sum())

    for p in pos_resid_peaks:
        p = int(p)
        far_from_pred_peak = not any_near(p, pred_peaks, int(cfg["peak_tolerance"]))
        low_support = bool(support[p] <= low_support_thr)

        if far_from_pred_peak and low_support:
            unexplained.append(p)
            additive_mass += float(max(resid[p], 0.0))

    pos_peak_count = int(len(pos_resid_peaks))
    unexplained_count = int(len(unexplained))
    unexplained_frac = unexplained_count / max(pos_peak_count, 1)
    additive_mass_frac = additive_mass / max(total_positive_resid_mass, eps)
    case_B_additive_score = 0.5 * unexplained_frac + 0.5 * additive_mass_frac

    # C: smooth/broad trend.
    smooth = moving_average(resid, int(cfg["smooth_window"]))
    resid_var = float(np.var(resid))
    smooth_var = float(np.var(smooth))
    lowfreq_var_frac = smooth_var / max(resid_var, eps)

    smooth_z = robust_z(smooth, eps=eps)
    broad_segment_count = count_broad_segments(smooth_z, threshold=1.5)
    case_C_smooth_score = lowfreq_var_frac

    # D: noise/low-signal.
    low_signal_mask = (y <= np.quantile(y, 0.25)) & (pred <= np.quantile(pred, 0.25))
    abs_resid = np.abs(resid)
    low_signal_resid_mass_frac = float(abs_resid[low_signal_mask].sum() / max(abs_resid.sum(), eps))
    robust_peak_count = int(len(local_peaks_from_z(np.abs(rz), threshold=float(cfg["peak_z"]))))
    residual_std = float(np.std(resid))

    structured_score = max(
        case_A_shift_score / max(float(cfg["shift_gain_threshold"]), eps),
        case_B_additive_score / max(float(cfg["additive_score_threshold"]), eps),
        case_C_smooth_score / max(float(cfg["smooth_frac_threshold"]), eps),
    )

    diffuse = 1.0 / (1.0 + robust_peak_count)
    case_D_noise_score = (
        max(0.0, min(1.0, 0.5 * low_signal_resid_mass_frac + 0.5 * diffuse))
        * max(0.0, 1.0 - min(1.0, structured_score))
    )

    # Priority order matters. Shift is tested first because shifted peaks can look
    # like additive peaks if you do not account for the lag.
    if case_A_shift_score >= float(cfg["shift_gain_threshold"]) and int(best_lag) != 0:
        primary_case = "A_shift_offset"
    elif case_B_additive_score >= float(cfg["additive_score_threshold"]) and unexplained_count > 0:
        primary_case = "B_unexplained_positive_peaks_additive_candidate"
    elif case_C_smooth_score >= float(cfg["smooth_frac_threshold"]) and broad_segment_count > 0:
        primary_case = "C_smooth_broad_trend"
    else:
        primary_case = "D_noise_low_signal_likelihood"

    result = ResidualCaseResult(
        sample_index=sample_index,
        transcript_id=str(rec.get("id", f"sample_{sample_index}")),
        dataset_id=rec.get("dataset_id", -1),
        dataset_name=dataset_name(rec.get("dataset_id", -1), dataset_id_to_name),
        length=int(rec["length"]),

        pcc=pearson(y, pred),
        best_lag=int(best_lag),
        corr_lag0=float(corr0) if np.isfinite(corr0) else np.nan,
        corr_best=float(corr_best) if np.isfinite(corr_best) else np.nan,
        shift_gain=float(shift_gain),
        case_A_shift_score=float(case_A_shift_score),

        positive_residual_peak_count=pos_peak_count,
        unexplained_positive_peak_count=unexplained_count,
        unexplained_positive_peak_frac=float(unexplained_frac),
        additive_residual_mass_frac=float(additive_mass_frac),
        case_B_additive_score=float(case_B_additive_score),

        lowfreq_var_frac=float(lowfreq_var_frac),
        broad_segment_count=int(broad_segment_count),
        case_C_smooth_score=float(case_C_smooth_score),

        residual_std=residual_std,
        robust_residual_peak_count=robust_peak_count,
        low_signal_residual_mass_frac=float(low_signal_resid_mass_frac),
        case_D_noise_score=float(case_D_noise_score),

        primary_case=primary_case,
    )

    arrays = {
        "y": y,
        "pred": pred,
        "support": support,
        "resid": resid,
        "rz": rz,
        "smooth_resid": smooth,
        "pos_resid_peaks": pos_resid_peaks,
        "pred_peaks": pred_peaks,
        "unexplained_peaks": np.asarray(unexplained, dtype=int),
        "corrs": corrs,
    }

    return result, arrays


# =============================================================================
# Plotting / summary
# =============================================================================

def plot_case(
    *,
    result: ResidualCaseResult,
    arrays: dict[str, Any],
    out_path: Path,
) -> None:
    y = arrays["y"]
    pred = arrays["pred"]
    support = arrays["support"]
    resid = arrays["resid"]
    rz = arrays["rz"]
    smooth = arrays["smooth_resid"]
    pos_peaks = arrays["pos_resid_peaks"]
    pred_peaks = arrays["pred_peaks"]
    unexplained = arrays["unexplained_peaks"]
    corrs = arrays["corrs"]

    x = np.arange(len(y))

    fig, axes = plt.subplots(4, 1, figsize=(18, 11), sharex=False)

    axes[0].plot(x, y, label="target normalized profile", linewidth=1.0)
    axes[0].plot(x, pred, label="pred normalized profile", linewidth=1.0)
    axes[0].set_title(
        f"{result.primary_case} | {result.dataset_name} | {result.transcript_id} | "
        f"PCC={result.pcc:.3f} | best_lag={result.best_lag} | shift_gain={result.shift_gain:.3f}"
    )
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x, resid, label="residual: target - pred", linewidth=0.9)
    axes[1].plot(x, smooth, label="smooth residual", linewidth=1.1)

    if pos_peaks.size:
        axes[1].scatter(pos_peaks, resid[pos_peaks], label="positive residual peaks", s=18)

    if unexplained.size:
        axes[1].scatter(
            unexplained,
            resid[unexplained],
            label="unexplained additive candidates",
            s=30,
            marker="x",
        )

    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(x, rz, label="robust z residual", linewidth=0.8)
    axes[2].plot(x, support, label="support/profile proxy", linewidth=0.8)

    if pred_peaks.size:
        axes[2].scatter(pred_peaks, support[pred_peaks], label="predicted peaks", s=15)

    axes[2].axhline(2.0, linestyle="--", linewidth=0.8)
    axes[2].axhline(-2.0, linestyle="--", linewidth=0.8)
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.25)

    lags = sorted(corrs.keys())
    vals = [corrs[k] for k in lags]

    axes[3].bar(lags, vals)
    axes[3].axvline(0, linestyle="--", linewidth=0.8)
    axes[3].set_xlabel("lag")
    axes[3].set_ylabel("corr(target, pred shifted)")
    axes[3].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    cases = [
        "A_shift_offset",
        "B_unexplained_positive_peaks_additive_candidate",
        "C_smooth_broad_trend",
        "D_noise_low_signal_likelihood",
    ]

    rows = []

    groups: list[tuple[str, pd.DataFrame]] = [("ALL", df)]
    groups.extend([(str(k), g) for k, g in df.groupby("dataset_name", dropna=False)])

    for name, g in groups:
        row = {
            "group": name,
            "n_samples": int(len(g)),
            "mean_pcc": float(g["pcc"].mean()),
            "median_pcc": float(g["pcc"].median()),
            "mean_shift_gain": float(g["shift_gain"].mean()),
            "mean_case_A_shift_score": float(g["case_A_shift_score"].mean()),
            "mean_case_B_additive_score": float(g["case_B_additive_score"].mean()),
            "mean_case_C_smooth_score": float(g["case_C_smooth_score"].mean()),
            "mean_case_D_noise_score": float(g["case_D_noise_score"].mean()),
            "mean_unexplained_positive_peak_frac": float(g["unexplained_positive_peak_frac"].mean()),
            "mean_additive_residual_mass_frac": float(g["additive_residual_mass_frac"].mean()),
            "mean_lowfreq_var_frac": float(g["lowfreq_var_frac"].mean()),
            "mean_low_signal_residual_mass_frac": float(g["low_signal_residual_mass_frac"].mean()),
        }

        vc_count = g["primary_case"].value_counts(normalize=False)
        vc_frac = g["primary_case"].value_counts(normalize=True)

        for c in cases:
            row[f"count_{c}"] = int(vc_count.get(c, 0))
            row[f"frac_{c}"] = float(vc_frac.get(c, 0.0))

        rows.append(row)

    return pd.DataFrame(rows)


def write_schema_report(df: pd.DataFrame | None, records: list[dict[str, Any]], out_dir: Path) -> None:
    if df is None:
        report = {
            "input_type": "non_dataframe_prediction_object",
            "n_records_loaded": len(records),
            "example_record_keys": list(records[0].keys()) if records else [],
        }
    else:
        report = {
            "input_type": "parquet_or_csv_dataframe",
            "n_rows_raw": int(len(df)),
            "columns": list(df.columns),
            "dtypes": {k: str(v) for k, v in df.dtypes.items()},
            "n_records_loaded": len(records),
            "example_record_keys": list(records[0].keys()) if records else [],
            "example_record_lengths": {
                k: (len(v) if isinstance(v, np.ndarray) else None)
                for k, v in records[0].items()
            } if records else {},
        }

    with (out_dir / "schema_report.json").open("w") as f:
        json.dump(report, f, indent=2)


def validate_config(cfg: dict[str, Any]) -> None:
    valid_plot = {
        "case_A_shift_score",
        "case_B_additive_score",
        "case_C_smooth_score",
        "case_D_noise_score",
        "residual_std",
        "pcc",
    }

    if int(cfg["max_lag"]) < 0:
        raise ValueError("CONFIG['max_lag'] must be >= 0.")

    if float(cfg["peak_z"]) <= 0:
        raise ValueError("CONFIG['peak_z'] must be > 0.")

    if not 0.0 <= float(cfg["low_support_quantile"]) <= 1.0:
        raise ValueError("CONFIG['low_support_quantile'] must be in [0, 1].")

    if "plot_by" in cfg and cfg["plot_by"] not in valid_plot:
        raise ValueError(f"Invalid plot_by={cfg['plot_by']!r}.")


def main() -> None:
    cfg = CONFIG
    validate_config(cfg)

    pred_path = Path(cfg["pred"])
    out_dir = Path(cfg["out_dir"])
    plot_dir = out_dir / "plots"

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    dataset_id_to_name = load_dataset_encoding(cfg.get("dataset_encoding"))

    records, raw_df = load_records(cfg)

    if len(records) == 0:
        raise RuntimeError(
            "No prediction records loaded. Check the parquet columns and CONFIG keys. "
            "A schema_report.json may have been written if possible."
        )

    if cfg.get("write_schema_report", True):
        write_schema_report(raw_df, records, out_dir)

    results: list[ResidualCaseResult] = []
    arrays_by_index: dict[int, dict[str, Any]] = {}

    for i, rec in enumerate(records):
        result, arrays = analyze_one(
            rec,
            sample_index=i,
            dataset_id_to_name=dataset_id_to_name,
            cfg=cfg,
        )
        results.append(result)
        arrays_by_index[i] = arrays

    df = pd.DataFrame([asdict(r) for r in results])
    summary = build_summary(df)

    per_sample_path = out_dir / "per_sample_residual_cases.csv"
    summary_path = out_dir / "summary_by_dataset.csv"
    config_path = out_dir / "analysis_config.json"

    df.to_csv(per_sample_path, index=False)
    summary.to_csv(summary_path, index=False)

    with config_path.open("w") as f:
        json.dump(cfg, f, indent=2)

    if bool(cfg["make_plots"]) and int(cfg["plot_top_n_per_case"]) > 0:
        n = int(cfg["plot_top_n_per_case"])

        case_score_map = {
            "A_shift_offset": "case_A_shift_score",
            "B_unexplained_positive_peaks_additive_candidate": "case_B_additive_score",
            "C_smooth_broad_trend": "case_C_smooth_score",
            "D_noise_low_signal_likelihood": "case_D_noise_score",
        }

        for case_name, score_col in case_score_map.items():
            sub = df[df["primary_case"] == case_name].copy()

            if sub.empty:
                continue

            sub = sub.sort_values(score_col, ascending=False).head(n)

            for _, row in sub.iterrows():
                idx = int(row["sample_index"])
                safe_case = str(row["primary_case"]).replace("/", "_")
                safe_ds = str(row["dataset_name"]).replace("/", "_")
                safe_id = str(row["transcript_id"]).replace("/", "_")
                out_path = plot_dir / f"{idx:06d}_{safe_ds}_{safe_case}_{safe_id}.png"

                plot_case(
                    result=results[idx],
                    arrays=arrays_by_index[idx],
                    out_path=out_path,
                )

    print(f"[OK] Prediction file: {pred_path}")
    print(f"[OK] Loaded samples: {len(records)}")
    print(f"[OK] Wrote: {per_sample_path}")
    print(f"[OK] Wrote: {summary_path}")
    print(f"[OK] Wrote: {config_path}")
    if cfg.get("write_schema_report", True):
        print(f"[OK] Wrote: {out_dir / 'schema_report.json'}")
    if bool(cfg["make_plots"]):
        print(f"[OK] Wrote plots to: {plot_dir}")

    print("\nSummary:")
    with pd.option_context("display.max_columns", 200, "display.width", 220):
        print(summary)


if __name__ == "__main__":
    main()