from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from scipy.stats import norm


# ============================================================
# Robust per-transcript PCC + Fisher-Z aggregation
# ============================================================

def safe_pearsonr(x, y, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size < 4:
        return np.nan

    if np.std(x) <= eps or np.std(y) <= eps:
        return np.nan

    x = x - x.mean()
    y = y - y.mean()

    denom = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))

    if denom <= eps:
        return np.nan

    return float(np.sum(x * y) / denom)


def fisher_weighted_pcc(
    pred_list,
    target_list,
    alpha: float = 0.05,
) -> dict[str, float]:
    """
    Aggregates per-transcript PCC values using Fisher-Z weighting.

    Returns:
        mean_pcc:
            Fisher-Z weighted mean PCC.

        ci_lower, ci_upper:
            Approximate confidence interval.

        n_transcripts:
            Number of valid transcript-level PCC values.

        median_pcc:
            Median transcript-level PCC.

        unweighted_mean_pcc:
            Simple mean of transcript-level PCC values.
    """
    pcc_s = []
    n_s = []

    for pred, target in zip(pred_list, target_list):
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)
        target = np.asarray(target, dtype=np.float64).reshape(-1)

        L = min(pred.size, target.size)
        if L < 4:
            continue

        pred = pred[:L]
        target = target[:L]

        r = safe_pearsonr(pred, target)
        if not np.isfinite(r):
            continue

        pcc_s.append(r)
        n_s.append(L)

    if len(pcc_s) == 0:
        return {
            "mean_pcc": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_transcripts": 0,
            "median_pcc": np.nan,
            "unweighted_mean_pcc": np.nan,
        }

    pcc_array = np.asarray(pcc_s, dtype=np.float64)
    n_array = np.asarray(n_s, dtype=np.float64)

    r_clipped = np.clip(pcc_array, -0.9999, 0.9999)
    z_scores = np.arctanh(r_clipped)

    weights = np.maximum(n_array - 3.0, 1.0)

    z_mean = np.average(z_scores, weights=weights)

    se_z_mean = 1.0 / np.sqrt(np.sum(weights))
    z_critical = norm.ppf(1.0 - alpha / 2.0)

    z_ci_lower = z_mean - z_critical * se_z_mean
    z_ci_upper = z_mean + z_critical * se_z_mean

    return {
        "mean_pcc": float(np.tanh(z_mean)),
        "ci_lower": float(np.tanh(z_ci_lower)),
        "ci_upper": float(np.tanh(z_ci_upper)),
        "n_transcripts": int(len(pcc_s)),
        "median_pcc": float(np.median(pcc_array)),
        "unweighted_mean_pcc": float(np.mean(pcc_array)),
    }


def component_metrics(
    df: pd.DataFrame,
    component: str,
    target_col: str = "target",
) -> dict[str, float]:
    if component not in df.columns:
        return {
            "mean_pcc": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_transcripts": 0,
            "median_pcc": np.nan,
            "unweighted_mean_pcc": np.nan,
        }

    valid_df = df[[component, target_col]].dropna()

    if len(valid_df) == 0:
        return {
            "mean_pcc": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "n_transcripts": 0,
            "median_pcc": np.nan,
            "unweighted_mean_pcc": np.nan,
        }

    return fisher_weighted_pcc(
        pred_list=valid_df[component].values,
        target_list=valid_df[target_col].values,
    )


def load_prediction_folder(folder: Path) -> pd.DataFrame | None:
    parquet_files = sorted(glob.glob(str(folder / "comprehensive_predictions_rank*.parquet")))

    if not parquet_files:
        return None

    return pd.concat(
        [pd.read_parquet(f) for f in parquet_files],
        ignore_index=True,
    )


def add_metrics_row(
    rows: list[dict],
    *,
    dataset: str,
    run_type: str,
    df: pd.DataFrame,
    components: list[str],
) -> None:
    for component in components:
        metrics = component_metrics(df, component)

        rows.append(
            {
                "dataset": dataset,
                "run_type": run_type,
                "component": component,
                "pcc": metrics["mean_pcc"],
                "ci_lower": metrics["ci_lower"],
                "ci_upper": metrics["ci_upper"],
                "n_transcripts": metrics["n_transcripts"],
                "median_pcc": metrics["median_pcc"],
                "unweighted_mean_pcc": metrics["unweighted_mean_pcc"],
            }
        )


def plot_component_comparison(
    res_df: pd.DataFrame,
    *,
    component: str,
    title: str,
) -> None:
    sub = res_df[res_df["component"] == component].copy()
    sub = sub.dropna(subset=["pcc"])

    if sub.empty:
        print(f"[WARN] No valid results for component={component}. Skipping plot.")
        return

    pivot = sub.pivot(index="dataset", columns="run_type", values="pcc")
    sub = sub.sort_values("pcc", ascending=True)

    datasets = sorted(sub["dataset"].unique())

    # Sort by mixed performance if available; otherwise by individual.
    sort_values = []
    for d in datasets:
        d_sub = sub[sub["dataset"] == d]
        mix_val = d_sub.loc[d_sub["run_type"] == "mixed", "pcc"]
        indiv_val = d_sub.loc[d_sub["run_type"] == "individual", "pcc"]

        if len(mix_val) > 0 and np.isfinite(mix_val.iloc[0]):
            sort_values.append((d, mix_val.iloc[0]))
        elif len(indiv_val) > 0 and np.isfinite(indiv_val.iloc[0]):
            sort_values.append((d, indiv_val.iloc[0]))
        else:
            sort_values.append((d, np.nan))

    sort_values = sorted(sort_values, key=lambda x: np.inf if not np.isfinite(x[1]) else x[1])
    datasets = [x[0] for x in sort_values]

    y_indices = np.arange(len(datasets))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(datasets))))

    for offset, run_type, color, label in [
        (-bar_width / 2, "individual", "lightsteelblue", "Individual"),
        (+bar_width / 2, "mixed", "navy", "Mixed"),
    ]:
        values = []
        err_lower = []
        err_upper = []

        for dataset in datasets:
            row = sub[(sub["dataset"] == dataset) & (sub["run_type"] == run_type)]

            if len(row) == 0:
                values.append(np.nan)
                err_lower.append(0.0)
                err_upper.append(0.0)
                continue

            r = row.iloc[0]
            values.append(r["pcc"])
            err_lower.append(r["pcc"] - r["ci_lower"])
            err_upper.append(r["ci_upper"] - r["pcc"])

        values = np.asarray(values, dtype=np.float64)
        xerr = np.asarray([err_lower, err_upper], dtype=np.float64)

        ax.barh(
            y_indices + offset,
            values,
            height=bar_width,
            xerr=xerr,
            capsize=2,
            color=color,
            edgecolor="black",
            label=label,
        )

    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y_indices)
    ax.set_yticklabels(datasets)
    ax.set_xlabel("Fisher-Z aggregated per-transcript PCC")
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    fig.tight_layout()
    plt.show()


def main():
    base_path = Path("./riboai_queueing")

    mixed_folder_name = "eichhorn_2014_grimson_2019"
    dataset_encoding_path = Path("../Datasets/encodings/dataset_encoding.yaml")

    components = [
        "mu_obs",       # final predicted mean
        "mu_base",      # before additive residual
        "L_queue",      # raw biological branch
        "L_effective",  # shifted biological branch
    ]

    with dataset_encoding_path.open("r", encoding="utf-8") as f:
        dataset_encoding = yaml.safe_load(f)

    id2dataset = {int(v): str(k) for k, v in dataset_encoding.items()}

    rows = []

    # ------------------------------------------------------------
    # 1. Individual runs
    # ------------------------------------------------------------
    folders = [
        d for d in base_path.iterdir()
        if d.is_dir()
    ]

    for folder in folders:
        folder_name = folder.name

        if folder_name == mixed_folder_name:
            continue

        df = load_prediction_folder(folder)
        if df is None:
            continue

        # Assumption: individual run folder name is the dataset name.
        dataset_name = folder_name

        if "target" not in df.columns:
            print(f"[WARN] Missing target column in {folder}. Skipping.")
            continue

        add_metrics_row(
            rows,
            dataset=dataset_name,
            run_type="individual",
            df=df,
            components=components,
        )

    # ------------------------------------------------------------
    # 2. Mixed run, split by dataset_id
    # ------------------------------------------------------------
    mixed_folder = base_path / mixed_folder_name
    df_mix = load_prediction_folder(mixed_folder)

    if df_mix is None:
        raise FileNotFoundError(f"No prediction parquet files found in: {mixed_folder}")

    if "dataset_id" not in df_mix.columns:
        raise KeyError("Mixed dataframe does not contain dataset_id column.")

    for dataset_id in sorted(df_mix["dataset_id"].unique()):
        dataset_id = int(dataset_id)
        dataset_name = id2dataset.get(dataset_id, f"dataset_{dataset_id}")

        subset = df_mix[df_mix["dataset_id"] == dataset_id]

        add_metrics_row(
            rows,
            dataset=dataset_name,
            run_type="mixed",
            df=subset,
            components=components,
        )

    # ------------------------------------------------------------
    # 3. Save metrics table
    # ------------------------------------------------------------
    res_df = pd.DataFrame(rows)

    res_df = res_df.sort_values(
        ["component", "dataset", "run_type"],
        ascending=True,
    ).reset_index(drop=True)

    print(res_df)

    out_csv = base_path / f"metrics_individual_vs_mixed_{mixed_folder_name}.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved metrics table to: {out_csv}")

    # ------------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------------
    plot_component_comparison(
        res_df,
        component="mu_obs",
        title="Final prediction: mixed vs individual runs\nPCC(mu_obs, target)",
    )

    plot_component_comparison(
        res_df,
        component="mu_base",
        title="Base model before additive residual\nPCC(mu_base, target)",
    )

    plot_component_comparison(
        res_df,
        component="L_queue",
        title="Raw biological queueing branch\nPCC(L_queue, target)",
    )

    plot_component_comparison(
        res_df,
        component="L_effective",
        title="Shift-corrected biological branch\nPCC(L_effective, target)",
    )


if __name__ == "__main__":
    main()