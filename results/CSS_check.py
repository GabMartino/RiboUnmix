from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt


# ============================================================
# Loading utilities
# ============================================================

def resolve_run_folder(dataset_folder: Path, run_name: str) -> Path:
    aliases = {
        "NoPCGrad": ["NoPCGrad", "NOPCGrad", "no_pcgrad", "nopcgrad"],
        "NOPCGrad": ["NOPCGrad", "NoPCGrad", "no_pcgrad", "nopcgrad"],
        "PCGrad": ["PCGrad", "pcgrad"],
    }
    candidates = aliases.get(run_name, [run_name])
    for candidate in candidates:
        folder = dataset_folder / candidate
        if folder.is_dir():
            return folder
    return dataset_folder / candidates[0]


def load_prediction_folder(folder: Path) -> pd.DataFrame | None:
    parquet_files = []
    for pattern in ["predictions_*.parquet", "comprehensive_predictions_rank*.parquet", "*.parquet"]:
        parquet_files.extend(glob.glob(str(folder / pattern)))

    if not parquet_files:
        return None

    df = pd.concat([pd.read_parquet(f) for f in sorted(set(parquet_files))], ignore_index=True)
    if "target" not in df.columns and "y" in df.columns:
        df["target"] = df["y"]
    return df


def discover_individual_prediction_folders(base_path: Path, run_name: str, mixed_signature: str) -> dict[str, Path]:
    out = {}
    for dataset_dir in sorted(base_path.iterdir()):
        if dataset_dir.is_dir() and dataset_dir.name != mixed_signature:
            run_dir = resolve_run_folder(dataset_dir, run_name)
            if run_dir.is_dir() and load_prediction_folder(run_dir) is not None:
                out[dataset_dir.name] = run_dir
    return out


# ============================================================
# CSS utilities
# ============================================================

def normalize_css_positions(css_i, L: int) -> np.ndarray:
    if css_i is None: return np.asarray([], dtype=np.int64)
    arr = np.asarray(css_i).reshape(-1)
    if arr.size == 0: return np.asarray([], dtype=np.int64)

    if arr.dtype == np.bool_:
        pos = np.nonzero(arr[:L])[0] if arr.size >= L else np.nonzero(arr)[0]
        return np.unique(pos.astype(np.int64))

    if np.issubdtype(arr.dtype, np.number):
        arr = arr[np.isfinite(arr)].astype(np.int64)
        if arr.size == L and np.all((arr == 0) | (arr == 1)):
            pos = np.nonzero(arr.astype(bool))[0]
        else:
            pos = arr
        pos = pos[(pos >= 0) & (pos < L)]
        return np.unique(pos.astype(np.int64))

    return np.asarray([], dtype=np.int64)


def css_recall_for_profile(score, css, length: int | None = None, top_frac: float = 0.01, min_k: int = 10,
                           window: int = 3) -> dict:
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    L = min(int(length), int(score.size)) if length else int(score.size)

    empty_res = {"recall": np.nan, "n_css": 0, "n_hits": 0, "k": 0}
    if L < 4: return empty_res

    score = score[:L]
    valid_score = np.isfinite(score)
    if valid_score.sum() < 4: return empty_res

    score = np.where(valid_score, score, -np.inf)
    css_pos = normalize_css_positions(css, L=L)
    if css_pos.size == 0: return empty_res

    k = min(max(int(min_k), int(np.ceil(float(top_frac) * L))), L)
    if k <= 0: return {"recall": np.nan, "n_css": int(css_pos.size), "n_hits": 0, "k": 0}

    top_idx = np.sort(np.argpartition(-score, kth=k - 1)[:k])
    distances = np.abs(css_pos.reshape(-1, 1) - top_idx.reshape(1, -1))
    hit = distances.min(axis=1) <= int(window)

    return {
        "recall": float(hit.sum() / max(css_pos.size, 1)),
        "n_css": int(css_pos.size),
        "n_hits": int(hit.sum()),
        "k": int(k),
    }


def css_recall_metrics(df: pd.DataFrame, score_col: str, css_col: str = "css", length_col: str = "length",
                       top_frac: float = 0.01, min_k: int = 10, window: int = 3) -> dict:
    empty_res = {"macro_recall": np.nan, "median_recall": np.nan, "micro_recall": np.nan, "n_profiles_with_css": 0,
                 "n_css_sites": 0, "n_css_hits": 0, "mean_k": np.nan}
    if score_col not in df.columns or css_col not in df.columns: return empty_res

    recalls, k_values = [], []
    n_css_total, n_hits_total, n_profiles_with_css = 0, 0, 0

    for _, row in df.iterrows():
        length = int(row[length_col]) if length_col in df.columns and pd.notna(row[length_col]) else None
        out = css_recall_for_profile(row[score_col], row[css_col], length, top_frac, min_k, window)

        if out["n_css"] > 0:
            recalls.append(out["recall"])
            k_values.append(out["k"])
            n_css_total += out["n_css"]
            n_hits_total += out["n_hits"]
            n_profiles_with_css += 1

    if n_profiles_with_css == 0: return empty_res

    return {
        "macro_recall": float(np.nanmean(recalls)),
        "median_recall": float(np.nanmedian(recalls)),
        "micro_recall": float(n_hits_total / max(n_css_total, 1)),
        "n_profiles_with_css": int(n_profiles_with_css),
        "n_css_sites": int(n_css_total),
        "n_css_hits": int(n_hits_total),
        "mean_k": float(np.mean(k_values)),
    }


def add_recall_row(rows: list, dataset: str, run_type: str, df: pd.DataFrame, score_col: str, top_frac: float,
                   min_k: int, window: int):
    metrics = css_recall_metrics(df, score_col=score_col, top_frac=top_frac, min_k=min_k, window=window)
    rows.append({"dataset": dataset, "run_type": run_type, "score_col": score_col, "top_frac": top_frac, "min_k": min_k,
                 "window": window, **metrics})


# ============================================================
# Plotting
# ============================================================

def plot_css_recall_comparison(res_df: pd.DataFrame, metric_col: str, title: str) -> None:
    sub = res_df.dropna(subset=[metric_col]).copy()
    if sub.empty: return

    datasets = sorted(sub["dataset"].unique())
    sort_values = []

    for dataset in datasets:
        d_sub = sub[sub["dataset"] == dataset]
        vals = {rt: d_sub.loc[d_sub["run_type"] == rt, metric_col].values for rt in
                ["mixed_PCGrad", "mixed_NOPCGrad", "individual_NOPCGrad"]}

        # Sort logic: prioritize PCGrad, then NOPCGrad, then Individual
        val = next((v[0] for v in vals.values() if len(v) > 0 and np.isfinite(v[0])), np.nan)
        sort_values.append((dataset, val))

    sort_values.sort(key=lambda x: np.inf if not np.isfinite(x[1]) else x[1])
    datasets = [x[0] for x in sort_values]
    y = np.arange(len(datasets))
    bar_width = 0.24

    fig, ax = plt.subplots(figsize=(13, max(5, 0.50 * len(datasets))))
    plot_specs = [
        (-bar_width, "individual_NOPCGrad", "lightsteelblue", "Individual NOPCGrad"),
        (0.0, "mixed_NOPCGrad", "darkorange", "Mixed NOPCGrad"),
        (+bar_width, "mixed_PCGrad", "navy", "Mixed PCGrad"),
    ]

    for offset, run_type, color, label in plot_specs:
        values = [sub[(sub["dataset"] == ds) & (sub["run_type"] == run_type)][metric_col].values for ds in datasets]
        values = [v[0] if len(v) > 0 else np.nan for v in values]
        ax.barh(y + offset, values, height=bar_width, color=color, edgecolor="black", label=label)

    ax.set_yticks(y)
    ax.set_yticklabels(datasets)
    ax.set_xlabel(metric_col.replace("_", " "))
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    fig.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    base_path = Path("./riboai_queueing")
    mixed_signature = "green_2020_wu_2019"
    individual_run_name = "PCGrad"
    mixed_run_names = ["NOPCGrad", "PCGrad"]
    dataset_encoding_path = Path("../Datasets/encodings/dataset_encoding.yaml")

    # Analyze both biased (mu) and purely biological (L_queue) signals
    score_cols = ["mu", "L_queue"]
    top_frac, min_k, window = 0.01, 10, 3

    with dataset_encoding_path.open("r", encoding="utf-8") as f:
        id2dataset = {int(v): str(k) for k, v in yaml.safe_load(f).items()}

    rows = []

    # 1. Individual Runs
    individual_folders = discover_individual_prediction_folders(base_path, individual_run_name, mixed_signature)
    print("Processing Individual Runs...")
    for dataset_name, folder in individual_folders.items():
        df = load_prediction_folder(folder)
        if df is None or "css" not in df.columns: continue

        for col in score_cols:
            if col in df.columns:
                add_recall_row(rows, dataset_name, "individual_NOPCGrad", df, col, top_frac, min_k, window)

    # 2. Mixed Runs
    print("Processing Mixed Runs...")
    for mixed_run_name in mixed_run_names:
        mixed_folder = resolve_run_folder(base_path / mixed_signature, mixed_run_name)
        df_mix = load_prediction_folder(mixed_folder)

        if df_mix is None or "dataset_id" not in df_mix.columns or "css" not in df_mix.columns: continue
        run_type = f"mixed_{mixed_run_name}"

        for dataset_id in sorted(df_mix["dataset_id"].unique()):
            dataset_name = id2dataset.get(int(dataset_id), f"dataset_{int(dataset_id)}")
            subset = df_mix[df_mix["dataset_id"] == dataset_id]

            for col in score_cols:
                if col in subset.columns:
                    add_recall_row(rows, dataset_name, run_type, subset, col, top_frac, min_k, window)

    # 3. Save & Plot
    res_df = pd.DataFrame(rows).sort_values(["score_col", "dataset", "run_type"]).reset_index(drop=True)
    out_csv = base_path / f"css_recall_mu_vs_Lqueue_mixed_{mixed_signature}.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved combined CSS recall table to: {out_csv}")

    # Generate separate plots for mu and L_queue
    for col in score_cols:
        sub_df = res_df[res_df["score_col"] == col]
        if sub_df.empty: continue

        print(f"\nGenerating plots for: {col}")
        plot_css_recall_comparison(
            sub_df, metric_col="micro_recall",
            title=f"CSS Recall: {col} peaks (Micro)\nTop-k: max({min_k}, ceil({top_frac}×L)); hit window: ±{window}"
        )
        plot_css_recall_comparison(
            sub_df, metric_col="macro_recall",
            title=f"CSS Recall: {col} peaks (Macro)\nTop-k: max({min_k}, ceil({top_frac}×L)); hit window: ±{window}"
        )


if __name__ == "__main__":
    main()