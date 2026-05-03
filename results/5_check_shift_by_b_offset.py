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
from tqdm import tqdm


EPS = 1e-8

DIAG_BASE = "raw_base_target_vs_shifted_Lqueue"
DIAG_CORR = "b_corrected_target_vs_shifted_Lqueue"
DIAG_B = "b_offset_shift_template"

DIAG_LABELS = {
    DIAG_BASE: r"Raw target vs shifted $L_{queue}$: $PCC(L_{i+k}, y_i/S_i)$",
    DIAG_CORR: r"b-corrected target vs shifted $L_{queue}$: $PCC(L_{i+k}, y_i/total\_scale_i)$",
    DIAG_B: r"b-offset shift template: $PCC(b_i, \log L_{i+k}-\log L_i)$",
}

DIAG_SHORT = {
    DIAG_BASE: "Raw target",
    DIAG_CORR: "b-corrected target",
    DIAG_B: "b-offset template",
}


# ============================================================
# Basic helpers
# ============================================================

def safe_array(x: Any, dtype=np.float64) -> np.ndarray:
    return np.asarray(x, dtype=dtype).reshape(-1)


def safe_pcc(a: np.ndarray, b: np.ndarray) -> float:
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

    a = a - np.mean(a)
    b = b - np.mean(b)

    var_a = np.sum(a * a)
    var_b = np.sum(b * b)

    if var_a <= 1e-12 or var_b <= 1e-12:
        return np.nan

    return float(np.sum(a * b) / np.sqrt(var_a * var_b))


def center_finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).copy()
    mask = np.isfinite(x)
    if np.any(mask):
        x[mask] = x[mask] - np.mean(x[mask])
    return x


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


def load_dataset_names(dataset_encoding_path: str | None) -> dict[int, str]:
    candidates = []

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

    print("[WARNING] Could not find dataset encoding file. Using dataset_<id> labels.")
    return {}


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
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

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


# ============================================================
# Lag diagnostics
# ============================================================

def lagged_target_pcc(L_queue: np.ndarray, target: np.ndarray, lag: int) -> float:
    """
    Tests:

        target_i ≈ L_queue_{i + lag}

    Positive lag means target at coordinate i is best explained by
    intrinsic L_queue downstream at i + lag.
    """
    L_queue = safe_array(L_queue)
    target = safe_array(target)

    L = min(len(L_queue), len(target))
    if L < 3:
        return np.nan

    L_queue = L_queue[:L]
    target = target[:L]

    idx = np.arange(L)
    shifted_idx = idx + int(lag)

    valid = (shifted_idx >= 0) & (shifted_idx < L)
    if valid.sum() < 3:
        return np.nan

    return safe_pcc(L_queue[shifted_idx[valid]], target[idx[valid]])


def b_offset_shift_template_pcc(
    L_queue: np.ndarray,
    b_offset: np.ndarray,
    lag: int,
    gradient_quantile: float | None = 0.50,
) -> float:
    """
    If the dataset is shifted by k codons, then approximately:

        y_i / S_i ≈ L_{i+k}

    while the model uses:

        mu_i = L_i * S_i * exp(b_i)

    Therefore the ideal b-offset correction for lag k is:

        b_i ≈ log(L_{i+k} + eps) - log(L_i + eps)

    This diagnostic correlates learned b_i with that theoretical template.
    """
    if lag == 0:
        return np.nan

    L_queue = safe_array(L_queue)
    b_offset = safe_array(b_offset)

    L = min(len(L_queue), len(b_offset))
    if L < 3:
        return np.nan

    L_queue = L_queue[:L]
    b_offset = b_offset[:L]

    idx = np.arange(L)
    shifted_idx = idx + int(lag)

    valid = (shifted_idx >= 0) & (shifted_idx < L)
    if valid.sum() < 3:
        return np.nan

    log_L = np.log(np.maximum(L_queue, EPS))

    template = log_L[shifted_idx[valid]] - log_L[idx[valid]]
    b = b_offset[idx[valid]]

    finite = np.isfinite(template) & np.isfinite(b)
    template = template[finite]
    b = b[finite]

    if len(template) < 3:
        return np.nan

    # Flat regions are uninformative for shift detection.
    # Keep only positions where the theoretical shift template has enough magnitude.
    if gradient_quantile is not None:
        threshold = np.quantile(np.abs(template), gradient_quantile)
        informative = np.abs(template) >= threshold
        template = template[informative]
        b = b[informative]

    if len(template) < 3:
        return np.nan

    return safe_pcc(center_finite(b), center_finite(template))


def compute_shift_lag_curves(
    df: pd.DataFrame,
    *,
    id_to_dataset: dict[int, str],
    lags: list[int],
    gradient_quantile: float | None,
    min_length: int,
) -> pd.DataFrame:
    rows = []

    for dataset_id, group in tqdm(df.groupby("dataset_id"), desc="Computing dataset shift curves"):
        dataset_id = int(dataset_id)
        dataset_name = id_to_dataset.get(dataset_id, f"dataset_{dataset_id}")

        collectors = {
            DIAG_BASE: {lag: [] for lag in lags},
            DIAG_CORR: {lag: [] for lag in lags},
            DIAG_B: {lag: [] for lag in lags if lag != 0},
        }

        for _, row in group.iterrows():
            L = int(row["length"])
            if L < min_length:
                continue

            y = safe_array(row["target"])[:L]
            L_queue = safe_array(row["L_queue"])[:L]
            total_scale = safe_array(row["total_scale"])[:L]
            b_offset = safe_array(row["b_offset"])[:L]

            L_eff = min(len(y), len(L_queue), len(total_scale), len(b_offset))
            if L_eff < min_length:
                continue

            y = y[:L_eff]
            L_queue = L_queue[:L_eff]
            total_scale = total_scale[:L_eff]
            b_offset = b_offset[:L_eff]

            exp_b = np.exp(np.clip(b_offset, -20.0, 20.0))

            # total_scale = S * exp(b)
            # so base_scale recovers S.
            base_scale = total_scale / np.maximum(exp_b, EPS)

            target_base = y / np.maximum(base_scale, EPS)
            target_corrected = y / np.maximum(total_scale, EPS)

            for lag in lags:
                r_base = lagged_target_pcc(L_queue, target_base, lag)
                r_corr = lagged_target_pcc(L_queue, target_corrected, lag)

                if np.isfinite(r_base):
                    collectors[DIAG_BASE][lag].append(r_base)

                if np.isfinite(r_corr):
                    collectors[DIAG_CORR][lag].append(r_corr)

                if lag != 0:
                    r_b = b_offset_shift_template_pcc(
                        L_queue=L_queue,
                        b_offset=b_offset,
                        lag=lag,
                        gradient_quantile=gradient_quantile,
                    )

                    if np.isfinite(r_b):
                        collectors[DIAG_B][lag].append(r_b)

        for diagnostic, by_lag in collectors.items():
            for lag, vals in by_lag.items():
                vals = np.asarray(vals, dtype=np.float64)
                vals = vals[np.isfinite(vals)]

                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "dataset": dataset_name,
                        "diagnostic": diagnostic,
                        "diagnostic_label": DIAG_LABELS[diagnostic],
                        "lag": int(lag),
                        "n_transcripts": int(len(vals)),
                        "mean_pcc": float(np.mean(vals)) if len(vals) else np.nan,
                        "median_pcc": float(np.median(vals)) if len(vals) else np.nan,
                        "q25_pcc": float(np.percentile(vals, 25)) if len(vals) else np.nan,
                        "q75_pcc": float(np.percentile(vals, 75)) if len(vals) else np.nan,
                    }
                )

    return pd.DataFrame(rows)


# ============================================================
# Dataset-level shift calls
# ============================================================

def best_lag_record(sub: pd.DataFrame) -> dict[str, Any]:
    sub = sub[np.isfinite(sub["median_pcc"])].copy()
    if sub.empty:
        return {
            "best_lag": np.nan,
            "best_median_pcc": np.nan,
            "lag0_median_pcc": np.nan,
            "best_minus_lag0": np.nan,
            "second_best_median_pcc": np.nan,
            "best_minus_second": np.nan,
            "n_transcripts_at_best": 0,
        }

    sub = sub.sort_values("median_pcc", ascending=False).reset_index(drop=True)
    best = sub.iloc[0]

    if len(sub) > 1:
        second_best = float(sub.iloc[1]["median_pcc"])
    else:
        second_best = np.nan

    lag0 = sub[sub["lag"] == 0]
    lag0_pcc = float(lag0["median_pcc"].iloc[0]) if not lag0.empty else np.nan

    return {
        "best_lag": int(best["lag"]),
        "best_median_pcc": float(best["median_pcc"]),
        "lag0_median_pcc": lag0_pcc,
        "best_minus_lag0": (
            float(best["median_pcc"] - lag0_pcc)
            if np.isfinite(lag0_pcc)
            else np.nan
        ),
        "second_best_median_pcc": second_best,
        "best_minus_second": (
            float(best["median_pcc"] - second_best)
            if np.isfinite(second_best)
            else np.nan
        ),
        "n_transcripts_at_best": int(best["n_transcripts"]),
    }


def make_shift_summary(
    lag_df: pd.DataFrame,
    *,
    min_base_gain: float,
    min_b_template_pcc: float,
    min_b_template_specificity: float,
) -> pd.DataFrame:
    rows = []

    for (dataset_id, dataset), g in lag_df.groupby(["dataset_id", "dataset"], observed=True):
        dataset_id = int(dataset_id)

        base_rec = best_lag_record(g[g["diagnostic"] == DIAG_BASE])
        corr_rec = best_lag_record(g[g["diagnostic"] == DIAG_CORR])
        b_rec = best_lag_record(g[g["diagnostic"] == DIAG_B])

        base_best_lag = base_rec["best_lag"]
        corr_best_lag = corr_rec["best_lag"]
        b_best_lag = b_rec["best_lag"]

        base_gain = base_rec["best_minus_lag0"]
        corr_gain = corr_rec["best_minus_lag0"]
        b_median = b_rec["best_median_pcc"]
        b_specificity = b_rec["best_minus_second"]

        base_support = (
            np.isfinite(base_best_lag)
            and int(base_best_lag) != 0
            and np.isfinite(base_gain)
            and base_gain >= min_base_gain
        )

        b_support = (
            np.isfinite(b_best_lag)
            and int(b_best_lag) != 0
            and np.isfinite(b_median)
            and b_median >= min_b_template_pcc
            and np.isfinite(b_specificity)
            and b_specificity >= min_b_template_specificity
        )

        b_matches_base = (
            base_support
            and b_support
            and int(b_best_lag) == int(base_best_lag)
        )

        corrected_returns_to_zero = (
            np.isfinite(corr_best_lag)
            and (
                int(corr_best_lag) == 0
                or (
                    np.isfinite(corr_gain)
                    and corr_gain < min_base_gain
                )
            )
        )

        if base_support and b_matches_base and corrected_returns_to_zero:
            shift_call_strength = "strong"
            shift_call_lag = int(base_best_lag)
            interpretation = (
                "Raw target prefers nonzero lag; b_offset matches same shift template; "
                "corrected target returns near lag 0."
            )
        elif base_support and b_matches_base:
            shift_call_strength = "moderate"
            shift_call_lag = int(base_best_lag)
            interpretation = (
                "Raw target and b_offset template agree on nonzero lag, but corrected target "
                "does not clearly return to lag 0."
            )
        elif base_support:
            shift_call_strength = "weak"
            shift_call_lag = int(base_best_lag)
            interpretation = (
                "Raw target prefers a nonzero lag, but b_offset template does not strongly support it."
            )
        elif b_support:
            shift_call_strength = "weak"
            shift_call_lag = int(b_best_lag)
            interpretation = (
                "b_offset resembles a nonzero shift template, but raw target alignment is weak."
            )
        else:
            shift_call_strength = "none"
            shift_call_lag = 0
            interpretation = "No convincing nonzero shift signal."

        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset": dataset,

                "raw_target_best_lag": base_rec["best_lag"],
                "raw_target_best_median_pcc": base_rec["best_median_pcc"],
                "raw_target_lag0_median_pcc": base_rec["lag0_median_pcc"],
                "raw_target_shift_gain": base_rec["best_minus_lag0"],
                "raw_target_best_minus_second": base_rec["best_minus_second"],
                "raw_target_n_transcripts": base_rec["n_transcripts_at_best"],

                "corrected_target_best_lag": corr_rec["best_lag"],
                "corrected_target_best_median_pcc": corr_rec["best_median_pcc"],
                "corrected_target_lag0_median_pcc": corr_rec["lag0_median_pcc"],
                "corrected_target_shift_gain": corr_rec["best_minus_lag0"],
                "corrected_target_best_minus_second": corr_rec["best_minus_second"],
                "corrected_target_n_transcripts": corr_rec["n_transcripts_at_best"],

                "b_template_best_lag": b_rec["best_lag"],
                "b_template_best_median_pcc": b_rec["best_median_pcc"],
                "b_template_second_best_median_pcc": b_rec["second_best_median_pcc"],
                "b_template_specificity": b_rec["best_minus_second"],
                "b_template_n_transcripts": b_rec["n_transcripts_at_best"],

                "base_support": bool(base_support),
                "b_template_support": bool(b_support),
                "b_template_matches_raw_target": bool(b_matches_base),
                "corrected_returns_to_lag0": bool(corrected_returns_to_zero),

                "shift_call_lag": int(shift_call_lag),
                "shift_call_strength": shift_call_strength,
                "interpretation": interpretation,
            }
        )

    summary_df = pd.DataFrame(rows)

    strength_order = {"strong": 0, "moderate": 1, "weak": 2, "none": 3}
    summary_df["strength_order"] = summary_df["shift_call_strength"].map(strength_order).fillna(9)
    summary_df = summary_df.sort_values(
        ["strength_order", "shift_call_lag", "dataset"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    return summary_df


# ============================================================
# Plotting
# ============================================================

def plot_dataset_curves(
    lag_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    color_map = {
        DIAG_BASE: "#d95f02",
        DIAG_CORR: "#1f77b4",
        DIAG_B: "#7570b3",
    }

    for dataset, sub in lag_df.groupby("dataset", observed=True):
        summary_row = summary_df[summary_df["dataset"] == dataset]
        if summary_row.empty:
            continue
        summary_row = summary_row.iloc[0]

        fig, ax = plt.subplots(figsize=(10, 6))

        for diagnostic in [DIAG_BASE, DIAG_CORR, DIAG_B]:
            d = sub[sub["diagnostic"] == diagnostic].sort_values("lag")
            if d.empty:
                continue

            ax.plot(
                d["lag"],
                d["median_pcc"],
                marker="o",
                linewidth=2.0,
                color=color_map[diagnostic],
                label=DIAG_SHORT[diagnostic],
            )

            ax.fill_between(
                d["lag"].to_numpy(dtype=float),
                d["q25_pcc"].to_numpy(dtype=float),
                d["q75_pcc"].to_numpy(dtype=float),
                color=color_map[diagnostic],
                alpha=0.15,
            )

        ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.axhline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)

        shift_lag = int(summary_row["shift_call_lag"])
        strength = summary_row["shift_call_strength"]

        if strength != "none":
            ax.axvline(
                shift_lag,
                color="red",
                linestyle="-",
                linewidth=1.4,
                alpha=0.7,
                label=f"shift call: {shift_lag:+d}",
            )

        title = (
            f"{dataset}: shift diagnostic\n"
            f"Call: {shift_lag:+d} codon ({strength}) | "
            f"raw best={int(summary_row['raw_target_best_lag']):+d}, "
            f"corrected best={int(summary_row['corrected_target_best_lag']):+d}, "
            f"b-template best={int(summary_row['b_template_best_lag']):+d}"
            if np.isfinite(summary_row["b_template_best_lag"])
            else f"{dataset}: shift diagnostic"
        )

        ax.set_title(title, fontsize=12)
        ax.set_xlabel(
            "Candidate lag k\n"
            r"Positive k means target$_i$ behaves like $L_{queue,i+k}$"
        )
        ax.set_ylabel("Median transcript PCC")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", framealpha=0.9)

        fig.tight_layout()
        fig.savefig(out_dir / f"{sanitize_filename(dataset)}_shift_diagnostic.png", dpi=300)
        plt.close(fig)


def plot_lag_heatmap(
    lag_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    diagnostic: str,
    out_path: Path,
) -> None:
    sub = lag_df[lag_df["diagnostic"] == diagnostic].copy()
    if sub.empty:
        return

    ordered_datasets = summary_df["dataset"].tolist()

    pivot = sub.pivot_table(
        index="dataset",
        columns="lag",
        values="median_pcc",
        aggfunc="mean",
    )

    pivot = pivot.reindex(ordered_datasets)
    pivot = pivot.sort_index(axis=1)

    data = pivot.to_numpy(dtype=float)

    fig_h = max(6, 0.35 * len(pivot))
    fig_w = max(8, 0.75 * len(pivot.columns))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    masked = np.ma.masked_invalid(data)

    if diagnostic == DIAG_B:
        cmap = "coolwarm"
    else:
        cmap = "viridis"

    im = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{int(c):+d}" for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel(
        "Candidate lag k\n"
        r"Positive k means target$_i$ behaves like $L_{queue,i+k}$"
    )
    ax.set_ylabel("Dataset")
    ax.set_title(DIAG_LABELS[diagnostic])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Median transcript PCC")

    if data.size <= 300:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isfinite(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if val > np.nanmedian(data) else "black",
                    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_best_lag_summary(summary_df: pd.DataFrame, out_path: Path) -> None:
    cols = [
        "raw_target_best_lag",
        "corrected_target_best_lag",
        "b_template_best_lag",
        "shift_call_lag",
    ]

    labels = [
        "Raw target\nbest lag",
        "b-corrected target\nbest lag",
        "b-offset template\nbest lag",
        "Final shift\ncall",
    ]

    plot_df = summary_df[["dataset", "shift_call_strength", *cols]].copy()
    data = plot_df[cols].to_numpy(dtype=float)

    fig_h = max(6, 0.35 * len(plot_df))
    fig, ax = plt.subplots(figsize=(9, fig_h))

    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-3, vmax=3)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(plot_df["dataset"])

    ax.set_title(
        "Dataset-level shift calls\n"
        "Lag convention: +1 means target at i resembles L_queue at i+1"
    )

    for i in range(data.shape[0]):
        strength = str(plot_df.iloc[i]["shift_call_strength"])
        for j in range(data.shape[1]):
            val = data[i, j]
            text = f"{int(val):+d}"
            if j == len(cols) - 1:
                text = f"{int(val):+d}\n{strength}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Best lag / called lag")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_shift_evidence(summary_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = summary_df.copy()

    y = np.arange(len(plot_df))

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(18, max(6, 0.35 * len(plot_df))),
        sharey=True,
    )

    axes[0].barh(y, plot_df["raw_target_shift_gain"], color="#d95f02", edgecolor="black")
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Raw target nonzero-lag gain")
    axes[0].set_xlabel("Best lag PCC - lag0 PCC")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(plot_df["dataset"])
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].barh(y, plot_df["b_template_best_median_pcc"], color="#7570b3", edgecolor="black")
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("b-offset shift-template agreement")
    axes[1].set_xlabel("Best template median PCC")
    axes[1].grid(axis="x", alpha=0.3)

    axes[2].barh(y, plot_df["b_template_specificity"], color="#1b9e77", edgecolor="black")
    axes[2].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[2].set_title("b-offset template specificity")
    axes[2].set_xlabel("Best template PCC - second best")
    axes[2].grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Evidence for dataset coordinate shifts\n"
        "Stronger evidence requires raw target shift + b-offset template support + corrected target returning to lag 0",
        fontsize=13,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_all_plots(
    lag_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    out_dir: Path,
) -> None:
    plots_dir = out_dir / "plots"
    dataset_plots_dir = plots_dir / "per_dataset"
    heatmap_dir = plots_dir / "heatmaps"

    plots_dir.mkdir(parents=True, exist_ok=True)
    dataset_plots_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    plot_dataset_curves(
        lag_df=lag_df,
        summary_df=summary_df,
        out_dir=dataset_plots_dir,
    )

    for diagnostic in [DIAG_BASE, DIAG_CORR, DIAG_B]:
        plot_lag_heatmap(
            lag_df=lag_df,
            summary_df=summary_df,
            diagnostic=diagnostic,
            out_path=heatmap_dir / f"{diagnostic}.png",
        )

    plot_best_lag_summary(
        summary_df=summary_df,
        out_path=plots_dir / "best_lag_summary.png",
    )

    plot_shift_evidence(
        summary_df=summary_df,
        out_path=plots_dir / "shift_evidence_summary.png",
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
        "--out-dir",
        type=str,
        default="dataset_shift_diagnostics_from_b_offset",
        help="Output directory.",
    )
    parser.add_argument(
        "--lags",
        type=int,
        nargs="+",
        default=[-3, -2, -1, 0, 1, 2, 3],
        help="Candidate codon lags to scan.",
    )
    parser.add_argument(
        "--gradient-quantile",
        type=float,
        default=0.50,
        help=(
            "For b_offset template diagnostic, keep only positions where "
            "|log L_{i+k} - log L_i| is above this quantile. "
            "Use -1 to disable."
        ),
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=20,
        help="Minimum transcript length to include.",
    )
    parser.add_argument(
        "--min-base-gain",
        type=float,
        default=0.010,
        help="Minimum raw-target best-lag gain over lag0 needed for shift support.",
    )
    parser.add_argument(
        "--min-b-template-pcc",
        type=float,
        default=0.050,
        help="Minimum b_offset/template median PCC needed for b-template support.",
    )
    parser.add_argument(
        "--min-b-template-specificity",
        type=float,
        default=0.005,
        help="Minimum b-template best-minus-second PCC needed for b-template support.",
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

    lag_df = compute_shift_lag_curves(
        df,
        id_to_dataset=id_to_dataset,
        lags=list(args.lags),
        gradient_quantile=gradient_quantile,
        min_length=int(args.min_length),
    )

    summary_df = make_shift_summary(
        lag_df,
        min_base_gain=float(args.min_base_gain),
        min_b_template_pcc=float(args.min_b_template_pcc),
        min_b_template_specificity=float(args.min_b_template_specificity),
    )

    lag_csv = out_dir / "dataset_shift_lag_curves.csv"
    summary_csv = out_dir / "dataset_shift_summary.csv"

    lag_df.to_csv(lag_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    make_all_plots(
        lag_df=lag_df,
        summary_df=summary_df,
        out_dir=out_dir,
    )

    print("\nSaved:")
    print(f"  Lag curves: {lag_csv}")
    print(f"  Summary:    {summary_csv}")
    print(f"  Plots:      {out_dir / 'plots'}")

    print("\nShift calls:")
    cols = [
        "dataset",
        "shift_call_lag",
        "shift_call_strength",
        "raw_target_best_lag",
        "corrected_target_best_lag",
        "b_template_best_lag",
        "raw_target_shift_gain",
        "b_template_best_median_pcc",
        "b_template_specificity",
    ]
    print(summary_df[cols].to_string(index=False))

    print("\nInterpretation:")
    print("  +1 means target_i resembles L_queue_{i+1}.")
    print("  -1 means target_i resembles L_queue_{i-1}.")
    print("  Strong evidence requires:")
    print("    1. raw target prefers a nonzero lag,")
    print("    2. b_offset matches the same shift-template lag,")
    print("    3. b-corrected target returns near lag 0.")


if __name__ == "__main__":
    main()