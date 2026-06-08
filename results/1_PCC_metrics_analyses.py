from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# ============================================================
# Configuration
# ============================================================

COMPONENT_ALIASES: dict[str, list[str]] = {
    "positive_mean": [
        "likelihood_positive_mean",
        "likelihood_unconditional_mean",
        "mu_positive",
        "mu_unconditional",
        "positive_mean",
        "unconditional_mean",
    ],
    "Mu": [
        "mu",
        "mu_L_obs",
        "mu_obs",
    ],
    "lambda_pre_dropout": [
        "lambda_pre_dropout",
        "mu_pre_dropout",
        "mu_obs",
    ],
    "rho_bio": [
        "rho_bio",
        "rho",
    ],
    "support": [
        "L_bio",
        "q_bio",
        "L_queue",
        "q",
    ],
}
CSS_COMPONENT_ALIASES: dict[str, list[str]] = {
    **COMPONENT_ALIASES,
    "target": ["target", "y"],
}

ZSCORE_THRESHOLDS = (1.0, 2.0, 3.0, 4.0, 5.0)
CSS_WINDOWS = (0, 1)
DEFAULT_CENSOR_THRESHOLD = 0.0
CALIBRATION_BIN_EDGES = np.linspace(0.0, 1.0, 11)

CSS_PLOT_COMPONENTS = [
    ("target", "Ground truth", "0.15"),
    ("positive_mean", "Log-normal mean", "tab:green"),
    ("Mu", "Raw mu", "tab:blue"),
    ("support", "Support", "tab:orange"),
]

TARGET_ALIASES = ["target", "y", "gt_profile", "ribo", "profile"]
DATASET_ID_ALIASES = ["dataset_id", "dataset_ids"]
ID_ALIASES = ["transcript_id", "ids", "id", "transcript_ids"]
PHI_ALIASES = ["phi", "dispersion"]


# ============================================================
# Robust per-transcript PCC + Fisher-Z aggregation
# ============================================================

def safe_pearsonr(x: Any, y: Any, eps: float = 1.0e-12) -> float:
    x_arr = sequence_or_none(x)
    y_arr = sequence_or_none(y)

    if x_arr is None or y_arr is None:
        return np.nan

    L = min(x_arr.size, y_arr.size)
    if L < 4:
        return np.nan

    x_arr = x_arr[:L]
    y_arr = y_arr[:L]

    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]

    if x_arr.size < 4:
        return np.nan

    if np.std(x_arr) <= eps or np.std(y_arr) <= eps:
        return np.nan

    x_arr = x_arr - x_arr.mean()
    y_arr = y_arr - y_arr.mean()

    denom = np.sqrt(np.sum(x_arr ** 2) * np.sum(y_arr ** 2))
    if denom <= eps:
        return np.nan

    return float(np.sum(x_arr * y_arr) / denom)


def fisher_weighted_pcc(pcc_s: list[float], n_s: list[int], alpha: float = 0.05) -> dict[str, float]:
    if len(pcc_s) == 0:
        return empty_metrics()

    pcc_array = np.asarray(pcc_s, dtype=np.float64)
    n_array = np.asarray(n_s, dtype=np.float64)

    r_clipped = np.clip(pcc_array, -0.9999, 0.9999)
    z_scores = np.arctanh(r_clipped)
    weights = np.maximum(n_array - 3.0, 1.0)

    z_mean = np.average(z_scores, weights=weights)
    se_z_mean = 1.0 / np.sqrt(np.sum(weights))
    z_critical = NormalDist().inv_cdf(1.0 - alpha / 2.0)

    z_ci_lower = z_mean - z_critical * se_z_mean
    z_ci_upper = z_mean + z_critical * se_z_mean

    return {
        "pcc": float(np.tanh(z_mean)),
        "ci_lower": float(np.tanh(z_ci_lower)),
        "ci_upper": float(np.tanh(z_ci_upper)),
        "n_transcripts": int(len(pcc_s)),
        "median_pcc": float(np.median(pcc_array)),
        "unweighted_mean_pcc": float(np.mean(pcc_array)),
    }


def empty_metrics() -> dict[str, float]:
    return {
        "pcc": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "n_transcripts": 0,
        "median_pcc": np.nan,
        "unweighted_mean_pcc": np.nan,
    }


# ============================================================
# Data normalization
# ============================================================

def first_existing(columns: set[str], candidates: list[str]) -> str | None:
    for key in candidates:
        if key in columns:
            return key
    return None


def sequence_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None

    if isinstance(value, float) and np.isnan(value):
        return None

    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value)
    else:
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

    if arr.size == 0:
        return None

    return arr


def zscore_profile(
    x: Any,
    *,
    valid_mask: np.ndarray | None = None,
    eps: float = 1.0e-8,
) -> np.ndarray | None:
    arr = sequence_or_none(x)
    if arr is None:
        return None

    finite = np.isfinite(arr)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        L = min(arr.size, valid_mask.size)
        arr = arr[:L]
        finite = finite[:L] & valid_mask[:L]

    vals = arr[finite]
    if vals.size < 4:
        return None

    sd = float(vals.std(ddof=0))
    if not np.isfinite(sd) or sd <= eps:
        return None

    z = np.full(arr.shape, np.nan, dtype=np.float64)
    z[finite] = (vals - float(vals.mean())) / sd
    return z


def css_positions_or_empty(css_value: Any, L: int) -> np.ndarray:
    arr = sequence_or_none(css_value)
    if arr is None or L <= 0:
        return np.asarray([], dtype=np.int64)

    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.asarray([], dtype=np.int64)

    # CSS may be stored either as a list of positions or as a 0/1 mask.
    if arr.size >= L:
        candidate_mask = arr[:L]
        if np.nanmin(candidate_mask) >= 0.0 and np.nanmax(candidate_mask) <= 1.0:
            return np.flatnonzero(candidate_mask > 0.0).astype(np.int64)

    positions = arr.astype(np.int64, copy=False)
    return np.unique(positions[(positions >= 0) & (positions < L)])


def css_mask_from_positions(css_idx: np.ndarray, L: int, window: int) -> np.ndarray:
    mask = np.zeros(L, dtype=bool)
    if css_idx.size == 0:
        return mask

    for idx in css_idx:
        lo = max(0, int(idx) - int(window))
        hi = min(L, int(idx) + int(window) + 1)
        mask[lo:hi] = True

    return mask


def css_site_hits_from_high_mask(
    *,
    css_idx: np.ndarray,
    high_mask: np.ndarray,
    window: int,
) -> int:
    n_hits = 0
    L = int(high_mask.size)

    for idx in css_idx:
        lo = max(0, int(idx) - int(window))
        hi = min(L, int(idx) + int(window) + 1)
        if np.any(high_mask[lo:hi]):
            n_hits += 1

    return n_hits


def css_site_count_in_valid_window(
    *,
    css_idx: np.ndarray,
    valid_mask: np.ndarray,
    window: int,
) -> int:
    n_valid = 0
    L = int(valid_mask.size)

    for idx in css_idx:
        lo = max(0, int(idx) - int(window))
        hi = min(L, int(idx) + int(window) + 1)
        if np.any(valid_mask[lo:hi]):
            n_valid += 1

    return n_valid


def normalize_prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    columns = set(df.columns)

    target_col = first_existing(columns, TARGET_ALIASES)
    if target_col is None:
        raise KeyError(f"No target column found. Tried {TARGET_ALIASES}.")

    if target_col != "target":
        df["target"] = df[target_col]

    dataset_id_col = first_existing(set(df.columns), DATASET_ID_ALIASES)
    if dataset_id_col is not None and dataset_id_col != "dataset_id":
        df["dataset_id"] = df[dataset_id_col]

    id_col = first_existing(set(df.columns), ID_ALIASES)
    if id_col is not None and id_col != "transcript_id":
        df["transcript_id"] = df[id_col]

    return df


def load_prediction_file(path: Path) -> pd.DataFrame:
    return normalize_prediction_frame(pd.read_parquet(path))


def discover_prediction_files(base_path: Path) -> list[Path]:
    patterns = [
        "predictions_*.parquet",
        "comprehensive_predictions_rank*.parquet",
    ]
    files: list[Path] = []

    for pattern in patterns:
        files.extend(base_path.rglob(pattern))

    return sorted(set(files))


def prediction_metadata(path: Path, base_path: Path) -> dict[str, str]:
    try:
        rel = path.relative_to(base_path)
    except ValueError:
        rel = path

    parts = rel.parts
    experiment = parts[0] if len(parts) >= 3 else path.parent.parent.name
    run = parts[1] if len(parts) >= 3 else path.parent.name

    split = "unknown"
    stem = path.stem
    if stem.startswith("predictions_"):
        suffix = stem.removeprefix("predictions_")
        for known in ("main_val", "css_benchmark", "val", "test", "predict"):
            if suffix.startswith(known):
                split = known
                break

    return {
        "experiment": experiment,
        "run": run,
        "split": split,
        "file": str(path),
    }


def load_dataset_encoding(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        dataset_encoding = yaml.safe_load(f) or {}

    return {int(v): str(k) for k, v in dataset_encoding.items()}


def load_loss_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    loss_cfg = config.get("loss", {})
    return dict(loss_cfg) if isinstance(loss_cfg, dict) else {}


def loss_censor_threshold(loss_cfg: dict[str, Any]) -> float:
    try:
        return float(loss_cfg.get("censor_threshold", DEFAULT_CENSOR_THRESHOLD))
    except (TypeError, ValueError):
        return DEFAULT_CENSOR_THRESHOLD


def normal_cdf_np(z: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf, otypes=[np.float64])
    return 0.5 * (1.0 + erf(z / math.sqrt(2.0)))


def broadcast_or_trim(arr: np.ndarray, L: int) -> np.ndarray | None:
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.size == 0 or L <= 0:
        return None

    if arr.size == 1:
        return np.full((L,), float(arr[0]), dtype=np.float64)

    if arr.size < L:
        return None

    return arr[:L]


def lognormal_params_np(
    *,
    mu: np.ndarray,
    phi: np.ndarray,
    loss_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    eps = float(loss_cfg.get("eps", 1.0e-8))
    mu_min = float(loss_cfg.get("mu_min", 1.0e-8))
    mu_max = float(loss_cfg.get("mu_max", 1.0e8))
    phi_min = float(loss_cfg.get("phi_min", 1.0e-4))
    phi_max = float(loss_cfg.get("phi_max", 10.0))
    log_sigma_min = float(loss_cfg.get("log_sigma_min", 0.05))
    log_sigma_max = float(loss_cfg.get("log_sigma_max", 2.0))
    mu_parameterization = str(
        loss_cfg.get("lognormal_mu_parameterization", "positive_mean")
    )

    mu_ref = np.clip(np.asarray(mu, dtype=np.float64), mu_min, mu_max)
    phi_t = np.asarray(phi, dtype=np.float64)
    if bool(loss_cfg.get("phi_input_is_log", False)):
        phi_t = np.exp(phi_t)
    phi_t = np.clip(phi_t, phi_min, phi_max)

    if mu_parameterization == "median":
        a = (1.0 / np.maximum(mu_ref, eps)) + np.maximum(phi_t, eps)
        x = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * a))
        log_sigma2 = np.log(np.maximum(x, 1.0 + eps))
        log_loc = np.log(np.maximum(mu_ref, eps))
    else:
        var_pos = np.maximum(mu_ref + phi_t * mu_ref**2, eps)
        log_sigma2 = np.log1p(var_pos / np.maximum(mu_ref**2, eps))
        log_sigma = np.sqrt(np.maximum(log_sigma2, eps))
        log_sigma = np.clip(log_sigma, log_sigma_min, log_sigma_max)
        log_loc = np.log(np.maximum(mu_ref, eps)) - 0.5 * log_sigma**2
        return log_loc, log_sigma

    log_sigma = np.sqrt(np.maximum(log_sigma2, eps))
    log_sigma = np.clip(log_sigma, log_sigma_min, log_sigma_max)
    return log_loc, log_sigma


def dataset_label(dataset_id: Any, id_to_dataset: dict[int, str], fallback: str) -> str:
    try:
        return id_to_dataset.get(int(dataset_id), f"dataset_{int(dataset_id)}")
    except Exception:
        return fallback


# ============================================================
# Metrics
# ============================================================

def component_metrics(
    df: pd.DataFrame,
    *,
    component: str,
    aliases: list[str],
) -> tuple[dict[str, float], str | None]:
    columns = set(df.columns)
    pred_col = first_existing(columns, aliases)

    if pred_col is None:
        return empty_metrics(), None

    pcc_s: list[float] = []
    n_s: list[int] = []

    for _, row in df.iterrows():
        pred = sequence_or_none(row[pred_col])
        target = sequence_or_none(row["target"])

        if pred is None or target is None:
            continue

        L = min(pred.size, target.size)
        r = safe_pearsonr(pred[:L], target[:L])

        if not np.isfinite(r):
            continue

        pcc_s.append(r)
        n_s.append(L)

    return fisher_weighted_pcc(pcc_s, n_s), pred_col


def add_metrics_rows(
    rows: list[dict[str, Any]],
    *,
    df: pd.DataFrame,
    metadata: dict[str, str],
    id_to_dataset: dict[int, str],
) -> None:
    if "dataset_id" in df.columns:
        grouped = df.groupby("dataset_id", sort=True)
    else:
        grouped = [(metadata["experiment"], df)]

    for dataset_id, subset in grouped:
        dataset = dataset_label(
            dataset_id,
            id_to_dataset,
            fallback=str(metadata["experiment"]),
        )

        for component, aliases in COMPONENT_ALIASES.items():
            metrics, resolved_col = component_metrics(
                subset,
                component=component,
                aliases=aliases,
            )

            rows.append(
                {
                    **metadata,
                    "dataset_id": dataset_id,
                    "dataset": dataset,
                    "component": component,
                    "resolved_column": resolved_col,
                    **metrics,
                }
            )


def censored_calibration_rows(
    *,
    df: pd.DataFrame,
    metadata: dict[str, str],
    id_to_dataset: dict[int, str],
    loss_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    columns = set(df.columns)
    mu_col = first_existing(columns, COMPONENT_ALIASES["Mu"])
    phi_col = first_existing(columns, PHI_ALIASES)
    if mu_col is None or phi_col is None:
        return []

    censor_threshold = loss_censor_threshold(loss_cfg)
    edges = np.asarray(CALIBRATION_BIN_EDGES, dtype=np.float64)
    n_bins = int(edges.size - 1)
    rows: list[dict[str, Any]] = []

    if "dataset_id" in df.columns:
        grouped = df.groupby("dataset_id", sort=True)
    else:
        grouped = [(metadata["experiment"], df)]

    for dataset_id, subset in grouped:
        dataset = dataset_label(
            dataset_id,
            id_to_dataset,
            fallback=str(metadata["experiment"]),
        )

        n_positions = np.zeros(n_bins, dtype=np.int64)
        observed_censored = np.zeros(n_bins, dtype=np.float64)
        predicted_censored = np.zeros(n_bins, dtype=np.float64)
        mu_sum = np.zeros(n_bins, dtype=np.float64)
        phi_sum = np.zeros(n_bins, dtype=np.float64)

        for _, row in subset.iterrows():
            target = sequence_or_none(row["target"])
            mu = sequence_or_none(row[mu_col])
            phi = sequence_or_none(row[phi_col])
            if target is None or mu is None or phi is None:
                continue

            L = min(target.size, mu.size)
            target = broadcast_or_trim(target, L)
            mu = broadcast_or_trim(mu, L)
            phi = broadcast_or_trim(phi, L)
            if target is None or mu is None or phi is None:
                continue

            valid = np.isfinite(target) & np.isfinite(mu) & np.isfinite(phi)
            if not np.any(valid):
                continue

            target = target[valid]
            mu = mu[valid]
            phi = phi[valid]

            log_loc, log_sigma = lognormal_params_np(
                mu=mu,
                phi=phi,
                loss_cfg=loss_cfg,
            )
            c = max(float(censor_threshold), float(loss_cfg.get("eps", 1.0e-8)))
            z_c = (math.log(c) - log_loc) / np.maximum(log_sigma, 1.0e-12)
            p_censored = np.clip(normal_cdf_np(z_c), 0.0, 1.0)
            y_censored = target <= float(censor_threshold)

            bin_idx = np.searchsorted(edges, p_censored, side="right") - 1
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)

            for bin_id in range(n_bins):
                m = bin_idx == bin_id
                if not np.any(m):
                    continue

                n = int(m.sum())
                n_positions[bin_id] += n
                observed_censored[bin_id] += float(y_censored[m].sum())
                predicted_censored[bin_id] += float(p_censored[m].sum())
                mu_sum[bin_id] += float(mu[m].sum())
                phi_sum[bin_id] += float(phi[m].sum())

        for bin_id in range(n_bins):
            n = int(n_positions[bin_id])
            if n == 0:
                continue

            observed_frac = observed_censored[bin_id] / n
            predicted_prob = predicted_censored[bin_id] / n
            error = observed_frac - predicted_prob

            rows.append(
                {
                    **metadata,
                    "dataset_id": dataset_id,
                    "dataset": dataset,
                    "component": "left_censored_lognormal",
                    "mu_column": mu_col,
                    "phi_column": phi_col,
                    "mu_parameterization": str(
                        loss_cfg.get("lognormal_mu_parameterization", "positive_mean")
                    ),
                    "censor_threshold": float(censor_threshold),
                    "prob_bin_lower": float(edges[bin_id]),
                    "prob_bin_upper": float(edges[bin_id + 1]),
                    "n_positions": n,
                    "observed_censored_frac": float(observed_frac),
                    "predicted_censored_prob": float(predicted_prob),
                    "calibration_error": float(error),
                    "abs_calibration_error": float(abs(error)),
                    "mean_mu": float(mu_sum[bin_id] / n),
                    "mean_phi": float(phi_sum[bin_id] / n),
                }
            )

    return rows


def censored_calibration_summary(calibration_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    group_cols = [
        "split",
        "dataset_id",
        "dataset",
        "experiment",
        "run",
        "run_label",
        "mu_column",
        "phi_column",
        "mu_parameterization",
        "censor_threshold",
    ]

    for keys, sub in calibration_df.groupby(group_cols, dropna=False, sort=True):
        n = sub["n_positions"].astype(float).to_numpy()
        if n.sum() <= 0.0:
            continue

        obs = sub["observed_censored_frac"].astype(float).to_numpy()
        pred = sub["predicted_censored_prob"].astype(float).to_numpy()
        err = obs - pred

        mean_observed = float(np.average(obs, weights=n))
        mean_predicted = float(np.average(pred, weights=n))
        bias = mean_observed - mean_predicted
        weighted_abs_error = float(np.average(np.abs(err), weights=n))
        weighted_rmse = float(np.sqrt(np.average(err**2, weights=n)))
        max_abs_error = float(np.nanmax(np.abs(err)))

        if weighted_abs_error < 0.02:
            quality = "excellent"
        elif weighted_abs_error < 0.05:
            quality = "good"
        elif weighted_abs_error < 0.10:
            quality = "moderate"
        else:
            quality = "poor"

        if bias > 0.02:
            direction = "underpredicts_censored_mass"
        elif bias < -0.02:
            direction = "overpredicts_censored_mass"
        else:
            direction = "balanced"

        rows.append(
            {
                **dict(zip(group_cols, keys, strict=True)),
                "n_positions": int(n.sum()),
                "mean_observed_censored_frac": mean_observed,
                "mean_predicted_censored_prob": mean_predicted,
                "calibration_bias": float(bias),
                "weighted_abs_calibration_error": weighted_abs_error,
                "weighted_rmse_calibration_error": weighted_rmse,
                "max_abs_calibration_error": max_abs_error,
                "calibration_quality": quality,
                "calibration_direction": direction,
            }
        )

    return pd.DataFrame(rows)


def css_zscore_rows(
    *,
    df: pd.DataFrame,
    metadata: dict[str, str],
    id_to_dataset: dict[int, str],
    censor_threshold: float,
) -> list[dict[str, Any]]:
    if "css" not in df.columns:
        return []

    rows: list[dict[str, Any]] = []

    if "dataset_id" in df.columns:
        grouped = df.groupby("dataset_id", sort=True)
    else:
        grouped = [(metadata["experiment"], df)]

    for dataset_id, subset in grouped:
        dataset = dataset_label(
            dataset_id,
            id_to_dataset,
            fallback=str(metadata["experiment"]),
        )

        for component, aliases in CSS_COMPONENT_ALIASES.items():
            pred_col = first_existing(set(subset.columns), aliases)
            if pred_col is None:
                continue

            counters = {
                (threshold, window): {
                    "n_profiles": 0,
                    "n_profiles_with_css": 0,
                    "n_profiles_with_high_z": 0,
                    "n_profiles_with_high_z_css_hit": 0,
                    "n_positions": 0,
                    "n_css_sites": 0,
                    "n_css_sites_hit": 0,
                    "n_css_window_positions": 0,
                    "n_high_z_positions": 0,
                    "n_high_z_css_positions": 0,
                    "n_censored_positions": 0,
                }
                for threshold in ZSCORE_THRESHOLDS
                for window in CSS_WINDOWS
            }

            for _, row in subset.iterrows():
                profile = sequence_or_none(row[pred_col])
                target = sequence_or_none(row["target"])
                if profile is None or target is None:
                    continue

                L = min(profile.size, target.size)
                profile = profile[:L]
                target = target[:L]
                zscore_support_mask = (
                    np.isfinite(target)
                    & (target > float(censor_threshold))
                )

                profile_z = zscore_profile(
                    profile,
                    valid_mask=zscore_support_mask,
                )
                if profile_z is None:
                    continue

                L = int(profile_z.size)
                zscore_valid_mask = np.isfinite(profile_z)
                n_valid_positions = int(zscore_valid_mask.sum())
                n_censored_positions = int((~zscore_valid_mask).sum())
                css_idx = css_positions_or_empty(row["css"], L)

                for threshold in ZSCORE_THRESHOLDS:
                    high_mask = np.isfinite(profile_z) & (profile_z > float(threshold))
                    n_high = int(high_mask.sum())

                    for window in CSS_WINDOWS:
                        css_window_mask = css_mask_from_positions(
                            css_idx,
                            L,
                            window=int(window),
                        )
                        css_window_eval_mask = css_window_mask & zscore_valid_mask
                        n_css_sites = css_site_count_in_valid_window(
                            css_idx=css_idx,
                            valid_mask=zscore_valid_mask,
                            window=int(window),
                        )
                        n_css_window_positions = int(css_window_eval_mask.sum())
                        n_hit_positions = int((high_mask & css_window_eval_mask).sum())
                        n_css_site_hits = css_site_hits_from_high_mask(
                            css_idx=css_idx,
                            high_mask=high_mask,
                            window=int(window),
                        )

                        c = counters[(threshold, window)]
                        c["n_profiles"] += 1
                        c["n_positions"] += n_valid_positions
                        c["n_css_sites"] += n_css_sites
                        c["n_css_sites_hit"] += n_css_site_hits
                        c["n_css_window_positions"] += n_css_window_positions
                        c["n_high_z_positions"] += n_high
                        c["n_high_z_css_positions"] += n_hit_positions
                        c["n_censored_positions"] += n_censored_positions

                        if n_css_sites > 0:
                            c["n_profiles_with_css"] += 1
                        if n_high > 0:
                            c["n_profiles_with_high_z"] += 1
                        if n_css_site_hits > 0:
                            c["n_profiles_with_high_z_css_hit"] += 1

            for (threshold, window), c in counters.items():
                n_high = c["n_high_z_positions"]
                n_hit = c["n_high_z_css_positions"]
                n_css_sites = c["n_css_sites"]
                n_css_site_hits = c["n_css_sites_hit"]
                n_css_window_positions = c["n_css_window_positions"]
                n_pos = c["n_positions"]
                n_profiles_high = c["n_profiles_with_high_z"]

                high_z_css_fraction = n_hit / n_high if n_high > 0 else np.nan
                css_recall = (
                    n_css_site_hits / n_css_sites
                    if n_css_sites > 0
                    else np.nan
                )
                background_css_rate = (
                    n_css_window_positions / n_pos
                    if n_pos > 0
                    else np.nan
                )
                profile_high_z_css_hit_fraction = (
                    c["n_profiles_with_high_z_css_hit"] / n_profiles_high
                    if n_profiles_high > 0
                    else np.nan
                )
                css_enrichment = (
                    high_z_css_fraction / background_css_rate
                    if (
                        np.isfinite(high_z_css_fraction)
                        and np.isfinite(background_css_rate)
                        and background_css_rate > 0.0
                    )
                    else np.nan
                )

                rows.append(
                    {
                        **metadata,
                        "dataset_id": dataset_id,
                        "dataset": dataset,
                        "component": component,
                        "resolved_column": pred_col,
                        "z_threshold": float(threshold),
                        "z_rule": (
                            f"{pred_col}_zscore > {float(threshold):g} "
                            f"on target > {float(censor_threshold):g}"
                        ),
                        "zscore_censor_threshold": float(censor_threshold),
                        "css_window": int(window),
                        "css_hit_definition": (
                            "exact_position"
                            if int(window) == 0
                            else f"+/-{int(window)} codons"
                        ),
                        "high_z_css_fraction": high_z_css_fraction,
                        "profile_high_z_css_hit_fraction": (
                            profile_high_z_css_hit_fraction
                        ),
                        "css_recall": css_recall,
                        "background_css_rate": background_css_rate,
                        "css_enrichment": css_enrichment,
                        "n_css_positions": n_css_sites,
                        **c,
                    }
                )

    return rows


# ============================================================
# Plotting
# ============================================================

def plot_component_comparison(
    res_df: pd.DataFrame,
    *,
    component: str,
    split: str,
    out_dir: Path,
) -> Path | None:
    if plt is None:
        return None

    sub = res_df[
        (res_df["component"] == component)
        & (res_df["split"] == split)
    ].copy()
    sub = sub.dropna(subset=["pcc"])

    if sub.empty:
        print(f"[WARN] No valid results for component={component}, split={split}.")
        return None

    run_labels = sorted(sub["run_label"].unique())
    datasets = sorted(sub["dataset"].unique())

    sort_values = []
    for dataset in datasets:
        best = sub.loc[sub["dataset"] == dataset, "pcc"].max()
        sort_values.append((dataset, best))

    datasets = [
        dataset
        for dataset, _ in sorted(
            sort_values,
            key=lambda item: np.inf if not np.isfinite(item[1]) else item[1],
        )
    ]

    y_indices = np.arange(len(datasets))
    bar_width = min(0.8 / max(len(run_labels), 1), 0.18)
    offsets = (
        np.arange(len(run_labels), dtype=np.float64)
        - (len(run_labels) - 1) / 2.0
    ) * bar_width

    fig, ax = plt.subplots(figsize=(14, max(5, 0.45 * len(datasets))))
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(run_labels), 1)))

    for offset, run_label, color in zip(offsets, run_labels, colors, strict=True):
        values = []
        err_lower = []
        err_upper = []

        for dataset in datasets:
            row = sub[
                (sub["dataset"] == dataset)
                & (sub["run_label"] == run_label)
            ]

            if row.empty:
                values.append(np.nan)
                err_lower.append(0.0)
                err_upper.append(0.0)
                continue

            r = row.sort_values("pcc", ascending=False).iloc[0]
            values.append(r["pcc"])
            err_lower.append(max(0.0, r["pcc"] - r["ci_lower"]))
            err_upper.append(max(0.0, r["ci_upper"] - r["pcc"]))

        ax.barh(
            y_indices + offset,
            np.asarray(values, dtype=np.float64),
            height=bar_width,
            xerr=np.asarray([err_lower, err_upper], dtype=np.float64),
            capsize=2,
            color=color,
            edgecolor="black",
            label=run_label,
        )

    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_yticks(y_indices)
    ax.set_yticklabels(datasets)
    ax.set_xlabel("Fisher-Z aggregated per-transcript PCC")
    ax.set_title(f"PCC({component}, target) by dataset ({split})")
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.9)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pcc_{split}_{component}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def plot_censored_calibration(
    calibration_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    split: str,
    dataset: str,
    run_label: str,
    out_dir: Path,
) -> Path | None:
    if plt is None:
        return None

    sub = calibration_df[
        (calibration_df["split"] == split)
        & (calibration_df["dataset"] == dataset)
        & (calibration_df["run_label"] == run_label)
    ].copy()
    sub = sub[sub["n_positions"].astype(int) > 0]
    if sub.empty:
        return None

    summary = summary_df[
        (summary_df["split"] == split)
        & (summary_df["dataset"] == dataset)
        & (summary_df["run_label"] == run_label)
    ]
    summary_row = summary.iloc[0] if not summary.empty else None

    sub = sub.sort_values("prob_bin_lower")
    x = sub["predicted_censored_prob"].astype(float).to_numpy()
    y = sub["observed_censored_frac"].astype(float).to_numpy()
    err = sub["calibration_error"].astype(float).to_numpy()
    n = sub["n_positions"].astype(float).to_numpy()
    centers = (
        sub["prob_bin_lower"].astype(float).to_numpy()
        + sub["prob_bin_upper"].astype(float).to_numpy()
    ) / 2.0
    sizes = 30.0 + 220.0 * np.sqrt(n / max(float(n.max()), 1.0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    axes[0].plot([0.0, 1.0], [0.0, 1.0], color="black", linewidth=1.0, linestyle="--")
    axes[0].scatter(
        x,
        y,
        s=sizes,
        color="tab:blue",
        edgecolor="black",
        alpha=0.8,
    )
    for xi, yi, ni in zip(x, y, n, strict=True):
        axes[0].annotate(
            str(int(ni)),
            (xi, yi),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
            alpha=0.8,
        )
    axes[0].set_xlim(-0.02, 1.02)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_xlabel("Mean predicted P(Y <= c)")
    axes[0].set_ylabel("Observed fraction Y <= c")
    axes[0].set_title("Calibration curve")
    axes[0].grid(True, alpha=0.3)

    colors = np.where(err >= 0.0, "tab:red", "tab:purple")
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].bar(
        centers,
        err,
        width=0.08,
        color=colors,
        edgecolor="black",
        alpha=0.85,
    )
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].set_xlabel("Predicted censored-probability bin")
    axes[1].set_ylabel("Observed - predicted")
    axes[1].set_title("Calibration error by bin")
    axes[1].grid(axis="y", alpha=0.3)

    if summary_row is not None:
        subtitle = (
            f"quality={summary_row['calibration_quality']} | "
            f"direction={summary_row['calibration_direction']} | "
            f"WACE={float(summary_row['weighted_abs_calibration_error']):.3f} | "
            f"bias={float(summary_row['calibration_bias']):+.3f}"
        )
    else:
        subtitle = "quality=unavailable"

    censor_threshold = float(sub["censor_threshold"].iloc[0])
    fig.suptitle(
        f"Censored calibration | {dataset} | {split} | {run_label}\n"
        f"c={censor_threshold:g} | {subtitle}",
        fontsize=11,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        out_dir
        / (
            f"censored_calibration_{sanitize_filename(split)}_"
            f"{sanitize_filename(dataset)}_"
            f"{sanitize_filename(run_label)}.png"
        )
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def sanitize_filename(value: Any) -> str:
    text = str(value)
    safe = []

    for char in text:
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")

    return "".join(safe).strip("_") or "unknown"


def plot_css_recall_dataset(
    css_df: pd.DataFrame,
    *,
    dataset: str,
    split: str,
    run_label: str,
    out_dir: Path,
) -> Path | None:
    if plt is None:
        return None

    sub = css_df[
        (css_df["dataset"] == dataset)
        & (css_df["split"] == split)
        & (css_df["run_label"] == run_label)
    ].copy()
    sub = sub.dropna(subset=["css_recall"])

    if sub.empty:
        print(
            f"[WARN] No valid CSS recall results for "
            f"dataset={dataset}, split={split}, run={run_label}."
        )
        return None

    thresholds = np.asarray(ZSCORE_THRESHOLDS, dtype=np.float64)
    window_labels = {
        0: "exact position",
        1: "+/-1 codon",
    }
    plot_components = [
        item
        for item in CSS_PLOT_COMPONENTS
        if item[0] in set(sub["component"])
    ]

    if len(plot_components) == 0:
        print(
            f"[WARN] No plottable CSS components for "
            f"dataset={dataset}, split={split}, run={run_label}."
        )
        return None

    fig, axes = plt.subplots(
        1,
        len(CSS_WINDOWS),
        figsize=(13, 5.5),
        sharey=True,
    )
    axes = np.atleast_1d(axes)

    x = np.arange(len(thresholds), dtype=np.float64)
    bar_width = min(0.8 / max(len(plot_components), 1), 0.24)
    offsets = (
        np.arange(len(plot_components), dtype=np.float64)
        - (len(plot_components) - 1) / 2.0
    ) * bar_width
    y_max = 0.0

    for ax, window in zip(axes, CSS_WINDOWS, strict=True):
        window = int(window)

        for offset, (component, label, color) in zip(
            offsets,
            plot_components,
            strict=True,
        ):
            curve = sub[
                (sub["component"] == component)
                & (sub["css_window"].astype(int) == window)
            ].copy()
            values = []

            for threshold in thresholds:
                row = curve[np.isclose(curve["z_threshold"], threshold)]
                if row.empty:
                    values.append(np.nan)
                    continue

                value = float(row.iloc[0]["css_recall"])
                values.append(value)
                if np.isfinite(value):
                    y_max = max(y_max, value)

            ax.bar(
                x + offset,
                np.asarray(values, dtype=np.float64),
                width=bar_width,
                color=color,
                edgecolor="black",
                linewidth=0.8,
                label=label,
            )

        ax.set_title(window_labels.get(window, f"+/-{window} codons"))
        ax.set_xticks(x)
        ax.set_xticklabels([f">{threshold:g}" for threshold in thresholds])
        ax.set_xlabel("Profile Z-score threshold")
        ax.grid(axis="y", linestyle="--", alpha=0.45)

    y_top = min(1.0, max(0.1, y_max * 1.18))
    for ax in axes:
        ax.set_ylim(0.0, y_top)

    axes[0].set_ylabel("CSS recall")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(plot_components),
        framealpha=0.9,
    )
    censor_threshold = None
    if "zscore_censor_threshold" in sub.columns:
        values = sub["zscore_censor_threshold"].dropna().unique()
        if len(values) > 0:
            censor_threshold = float(values[0])

    censor_suffix = (
        ""
        if censor_threshold is None
        else f" | z-score support: target > {censor_threshold:g}"
    )
    fig.suptitle(
        f"CSS recall: ground truth vs prediction | {dataset} | {split}"
        f"{censor_suffix}"
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        out_dir
        / (
            f"css_recall_{sanitize_filename(split)}_"
            f"{sanitize_filename(dataset)}_"
            f"{sanitize_filename(run_label)}.png"
        )
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


# ============================================================
# Main
# ============================================================

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
    base_path = resolve_base_path()
    dataset_encoding_path = repo_root / "Datasets" / "encodings" / "dataset_encoding.yaml"
    config_path = repo_root / "config" / "config_riboai_queuing_multidataset.yaml"

    prediction_files = discover_prediction_files(base_path)
    if len(prediction_files) == 0:
        raise RuntimeError(f"No prediction parquet files found under {base_path}.")

    id_to_dataset = load_dataset_encoding(dataset_encoding_path)
    loss_cfg = load_loss_config(config_path)
    censor_threshold = loss_censor_threshold(loss_cfg)
    rows: list[dict[str, Any]] = []
    css_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    css_df: pd.DataFrame | None = None

    print(f"Base path: {base_path}")
    print(f"Prediction files: {len(prediction_files)}")
    print(f"CSS z-score censor threshold: target > {censor_threshold:g}")

    for path in prediction_files:
        metadata = prediction_metadata(path, base_path)
        print(f"Loading {path}")

        try:
            df = load_prediction_file(path)
        except Exception as exc:
            print(f"[WARN] Failed to load {path}: {exc}")
            continue

        add_metrics_rows(
            rows,
            df=df,
            metadata=metadata,
            id_to_dataset=id_to_dataset,
        )
        calibration_rows.extend(
            censored_calibration_rows(
                df=df,
                metadata=metadata,
                id_to_dataset=id_to_dataset,
                loss_cfg=loss_cfg,
            )
        )
        css_rows.extend(
            css_zscore_rows(
                df=df,
                metadata=metadata,
                id_to_dataset=id_to_dataset,
                censor_threshold=censor_threshold,
            )
        )

    res_df = pd.DataFrame(rows)
    if res_df.empty:
        raise RuntimeError("No metrics were computed.")

    res_df["run_label"] = res_df["experiment"].astype(str) + " / " + res_df["run"].astype(str)
    res_df = res_df.sort_values(
        ["split", "component", "dataset", "run_label"],
        ascending=True,
    ).reset_index(drop=True)

    out_csv = base_path / "pcc_metrics_all_predictions.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved metrics table to: {out_csv}")

    visible = res_df[
        res_df["n_transcripts"].fillna(0).astype(int) > 0
    ][
        [
            "split",
            "dataset",
            "component",
            "run_label",
            "resolved_column",
            "pcc",
            "ci_lower",
            "ci_upper",
            "n_transcripts",
        ]
    ]
    print("\n=== Metrics with valid PCCs ===")
    print(visible.to_string(index=False))

    if len(calibration_rows) > 0:
        calibration_df = pd.DataFrame(calibration_rows)
        calibration_df["run_label"] = (
            calibration_df["experiment"].astype(str)
            + " / "
            + calibration_df["run"].astype(str)
        )
        calibration_df = calibration_df.sort_values(
            ["split", "dataset", "run_label", "prob_bin_lower"],
            ascending=True,
        ).reset_index(drop=True)

        calibration_csv = base_path / "censored_calibration_all_predictions.csv"
        calibration_df.to_csv(calibration_csv, index=False)
        print(f"\nSaved censored calibration table to: {calibration_csv}")

        calibration_summary_df = censored_calibration_summary(calibration_df)
        calibration_summary_csv = (
            base_path / "censored_calibration_summary_all_predictions.csv"
        )
        calibration_summary_df.to_csv(calibration_summary_csv, index=False)
        print(f"Saved censored calibration summary to: {calibration_summary_csv}")

        visible_calibration = calibration_df[
            calibration_df["n_positions"].fillna(0).astype(int) > 0
        ][
            [
                "split",
                "dataset",
                "run_label",
                "mu_column",
                "phi_column",
                "mu_parameterization",
                "censor_threshold",
                "prob_bin_lower",
                "prob_bin_upper",
                "n_positions",
                "observed_censored_frac",
                "predicted_censored_prob",
                "calibration_error",
                "abs_calibration_error",
                "mean_mu",
                "mean_phi",
            ]
        ]
        print("\n=== Censored mass calibration ===")
        print(visible_calibration.to_string(index=False))

        if not calibration_summary_df.empty:
            print("\n=== Censored calibration summary ===")
            print(
                calibration_summary_df[
                    [
                        "split",
                        "dataset",
                        "run_label",
                        "n_positions",
                        "mean_observed_censored_frac",
                        "mean_predicted_censored_prob",
                        "calibration_bias",
                        "weighted_abs_calibration_error",
                        "calibration_quality",
                        "calibration_direction",
                    ]
                ].to_string(index=False)
            )

        if plt is not None and not calibration_summary_df.empty:
            calibration_plot_dir = base_path / "censored_calibration_plots"
            calibration_plot_dir.mkdir(parents=True, exist_ok=True)
            for old_plot in calibration_plot_dir.glob("*.png"):
                old_plot.unlink()

            for (split, dataset, run_label), _ in calibration_df.groupby(
                ["split", "dataset", "run_label"],
                sort=True,
            ):
                out_path = plot_censored_calibration(
                    calibration_df,
                    calibration_summary_df,
                    split=str(split),
                    dataset=str(dataset),
                    run_label=str(run_label),
                    out_dir=calibration_plot_dir,
                )
                if out_path is not None:
                    print(f"Saved censored calibration plot: {out_path}")
    else:
        print("\n[WARN] No mu/phi columns found for censored calibration.")

    if len(css_rows) > 0:
        css_df = pd.DataFrame(css_rows)
        css_df["run_label"] = css_df["experiment"].astype(str) + " / " + css_df["run"].astype(str)
        css_df = css_df.sort_values(
            ["split", "component", "dataset", "run_label", "z_threshold"],
            ascending=True,
        ).reset_index(drop=True)

        css_csv = base_path / "css_zscore_presence_all_predictions.csv"
        css_df.to_csv(css_csv, index=False)
        print(f"\nSaved CSS Z-score table to: {css_csv}")

        visible_css = css_df[
            css_df["n_profiles"].fillna(0).astype(int) > 0
        ][
            [
                "split",
                "dataset",
                "component",
                "run_label",
                "resolved_column",
                "z_threshold",
                "z_rule",
                "zscore_censor_threshold",
                "css_window",
                "css_hit_definition",
                "high_z_css_fraction",
                "profile_high_z_css_hit_fraction",
                "css_recall",
                "background_css_rate",
                "css_enrichment",
                "n_profiles_with_high_z",
                "n_profiles_with_high_z_css_hit",
                "n_high_z_positions",
                "n_high_z_css_positions",
                "n_censored_positions",
                "n_css_positions",
                "n_css_sites",
                "n_css_sites_hit",
                "n_css_window_positions",
            ]
        ]
        print("\n=== CSS presence at high-Z positions ===")
        print(visible_css.to_string(index=False))
    else:
        print("\n[WARN] No CSS columns found in loaded predictions. Skipping CSS Z-score analysis.")

    if plt is None:
        print("\n[WARN] matplotlib is not installed. Skipping plots.")
        return

    plot_dir = base_path / "pcc_metric_plots"
    for split in sorted(res_df["split"].dropna().unique()):
        for component in COMPONENT_ALIASES:
            out_path = plot_component_comparison(
                res_df,
                component=component,
                split=str(split),
                out_dir=plot_dir,
            )
            if out_path is not None:
                print(f"Saved plot: {out_path}")

    if css_df is not None and not css_df.empty:
        recall_plot_dir = base_path / "css_recall_plots"
        recall_plot_dir.mkdir(parents=True, exist_ok=True)
        for old_plot in recall_plot_dir.glob("*.png"):
            old_plot.unlink()

        for (split, dataset, run_label), _ in css_df.groupby(
            ["split", "dataset", "run_label"],
            sort=True,
        ):
            out_path = plot_css_recall_dataset(
                css_df,
                dataset=str(dataset),
                split=str(split),
                run_label=str(run_label),
                out_dir=recall_plot_dir,
            )
            if out_path is not None:
                print(f"Saved CSS recall plot: {out_path}")


if __name__ == "__main__":
    main()
