from __future__ import annotations

import glob
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
        return empty_metrics()

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


def empty_metrics() -> dict[str, float]:
    return {
        "mean_pcc": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "n_transcripts": 0,
        "median_pcc": np.nan,
        "unweighted_mean_pcc": np.nan,
    }


def component_metrics(
    df: pd.DataFrame,
    component: str,
    target_col: str = "target",
) -> dict[str, float]:
    if component not in df.columns:
        return empty_metrics()

    if target_col not in df.columns:
        return empty_metrics()

    valid_df = df[[component, target_col]].dropna()

    if len(valid_df) == 0:
        return empty_metrics()

    return fisher_weighted_pcc(
        pred_list=valid_df[component].values,
        target_list=valid_df[target_col].values,
    )


def load_prediction_folder(folder: Path) -> pd.DataFrame | None:
    parquet_files = []

    patterns = [
        "predictions_*.parquet",
        "comprehensive_predictions_rank*.parquet",
        "*.parquet",
    ]

    for pattern in patterns:
        parquet_files.extend(glob.glob(str(folder / pattern)))

    parquet_files = sorted(set(parquet_files))

    if not parquet_files:
        return None

    df = pd.concat(
        [pd.read_parquet(f) for f in parquet_files],
        ignore_index=True,
    )

    if "target" not in df.columns and "y" in df.columns:
        df["target"] = df["y"]

    return df


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


def discover_individual_prediction_folders(
    *,
    base_path: Path,
    run_name: str,
    mixed_dataset_signature: str,
) -> dict[str, Path]:
    out = {}

    for dataset_dir in sorted(base_path.iterdir()):
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name

        if dataset_name == mixed_dataset_signature:
            continue

        run_dir = resolve_run_folder(dataset_dir, run_name)

        if not run_dir.is_dir():
            continue

        df = load_prediction_folder(run_dir)

        if df is None:
            continue

        out[dataset_name] = run_dir

    return out


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

    datasets = sorted(sub["dataset"].unique())

    sort_values = []

    for dataset in datasets:
        d_sub = sub[sub["dataset"] == dataset]

        mixed_pcgrad = d_sub.loc[d_sub["run_type"] == "mixed_PCGrad", "pcc"]
        mixed_nopcgrad = d_sub.loc[d_sub["run_type"] == "mixed_NOPCGrad", "pcc"]
        indiv_nopcgrad = d_sub.loc[d_sub["run_type"] == "individual_NOPCGrad", "pcc"]

        if len(mixed_pcgrad) > 0 and np.isfinite(mixed_pcgrad.iloc[0]):
            sort_values.append((dataset, mixed_pcgrad.iloc[0]))
        elif len(mixed_nopcgrad) > 0 and np.isfinite(mixed_nopcgrad.iloc[0]):
            sort_values.append((dataset, mixed_nopcgrad.iloc[0]))
        elif len(indiv_nopcgrad) > 0 and np.isfinite(indiv_nopcgrad.iloc[0]):
            sort_values.append((dataset, indiv_nopcgrad.iloc[0]))
        else:
            sort_values.append((dataset, np.nan))

    sort_values = sorted(
        sort_values,
        key=lambda x: np.inf if not np.isfinite(x[1]) else x[1],
    )

    datasets = [x[0] for x in sort_values]

    y_indices = np.arange(len(datasets))
    bar_width = 0.24

    fig, ax = plt.subplots(figsize=(13, max(5, 0.50 * len(datasets))))

    plot_specs = [
        (
            -bar_width,
            "individual_NOPCGrad",
            "lightsteelblue",
            "Individual NOPCGrad",
        ),
        (
            0.0,
            "mixed_NOPCGrad",
            "darkorange",
            "Mixed NOPCGrad",
        ),
        (
            +bar_width,
            "mixed_PCGrad",
            "navy",
            "Mixed PCGrad",
        ),
    ]

    for offset, run_type, color, label in plot_specs:
        values = []
        err_lower = []
        err_upper = []

        for dataset in datasets:
            row = sub[
                (sub["dataset"] == dataset)
                & (sub["run_type"] == run_type)
            ]

            if len(row) == 0:
                values.append(np.nan)
                err_lower.append(0.0)
                err_upper.append(0.0)
                continue

            r = row.iloc[0]

            values.append(r["pcc"])
            err_lower.append(max(0.0, r["pcc"] - r["ci_lower"]))
            err_upper.append(max(0.0, r["ci_upper"] - r["pcc"]))

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

    individual_run_name = "NOPCGrad"
    mixed_run_names = ["NOPCGrad", "PCGrad"]

    mixed_dataset_signature = "33_datasets_mix_6e5e33"
    dataset_encoding_path = Path("../Datasets/encodings/dataset_encoding.yaml")

    components = [
        "mu_obs",
        "mu_base",
        "L_queue",
        "L_effective",
    ]

    with dataset_encoding_path.open("r", encoding="utf-8") as f:
        dataset_encoding = yaml.safe_load(f)

    id2dataset = {
        int(v): str(k)
        for k, v in dataset_encoding.items()
    }

    rows = []

    # ------------------------------------------------------------
    # 1. Individual NOPCGrad single-dataset runs
    # ------------------------------------------------------------
    individual_folders = discover_individual_prediction_folders(
        base_path=base_path,
        run_name=individual_run_name,
        mixed_dataset_signature=mixed_dataset_signature,
    )

    print("\n=== Individual NOPCGrad prediction folders ===")

    for dataset_name, folder in individual_folders.items():
        print(f"{dataset_name}: {folder}")

        df = load_prediction_folder(folder)

        if df is None:
            continue

        if "target" not in df.columns:
            print(f"[WARN] Missing target/y column in {folder}. Skipping.")
            continue

        add_metrics_row(
            rows,
            dataset=dataset_name,
            run_type="individual_NOPCGrad",
            df=df,
            components=components,
        )

    # ------------------------------------------------------------
    # 2. Mixed multi-dataset runs: NOPCGrad and PCGrad
    # ------------------------------------------------------------
    print("\n=== Mixed multi-dataset runs ===")

    for mixed_run_name in mixed_run_names:
        mixed_folder = resolve_run_folder(
            base_path / mixed_dataset_signature,
            mixed_run_name,
        )

        df_mix = load_prediction_folder(mixed_folder)

        if df_mix is None:
            print(f"[WARN] No prediction parquet files found in: {mixed_folder}. Skipping.")
            continue

        if "dataset_id" not in df_mix.columns:
            raise KeyError(f"Mixed dataframe does not contain dataset_id column: {mixed_folder}")

        if "target" not in df_mix.columns:
            raise KeyError(f"Mixed dataframe does not contain target or y column: {mixed_folder}")

        print(f"{mixed_run_name}: {mixed_folder}")

        run_type = f"mixed_{mixed_run_name}"

        for dataset_id in sorted(df_mix["dataset_id"].unique()):
            dataset_id = int(dataset_id)
            dataset_name = id2dataset.get(dataset_id, f"dataset_{dataset_id}")

            subset = df_mix[df_mix["dataset_id"] == dataset_id]

            add_metrics_row(
                rows,
                dataset=dataset_name,
                run_type=run_type,
                df=subset,
                components=components,
            )

    # ------------------------------------------------------------
    # 3. Metrics table
    # ------------------------------------------------------------
    res_df = pd.DataFrame(rows)

    res_df = res_df.sort_values(
        ["component", "dataset", "run_type"],
        ascending=True,
    ).reset_index(drop=True)

    print("\n=== Metrics ===")
    print(res_df)

    out_csv = base_path / (
        f"metrics_individual_NOPCGrad_vs_mixed_NOPCGrad_PCGrad_"
        f"{mixed_dataset_signature}.csv"
    )

    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved metrics table to: {out_csv}")

    # ------------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------------
    plot_component_comparison(
        res_df,
        component="mu_obs",
        title=(
            "Final prediction\n"
            "Individual NOPCGrad vs Mixed NOPCGrad vs Mixed PCGrad\n"
            "PCC(mu_obs, target)"
        ),
    )

    plot_component_comparison(
        res_df,
        component="mu_base",
        title=(
            "Base model before additive residual\n"
            "Individual NOPCGrad vs Mixed NOPCGrad vs Mixed PCGrad\n"
            "PCC(mu_base, target)"
        ),
    )

    plot_component_comparison(
        res_df,
        component="L_queue",
        title=(
            "Raw biological queueing branch\n"
            "Individual NOPCGrad vs Mixed NOPCGrad vs Mixed PCGrad\n"
            "PCC(L_queue, target)"
        ),
    )

    plot_component_comparison(
        res_df,
        component="L_effective",
        title=(
            "Shift-corrected biological branch\n"
            "Individual NOPCGrad vs Mixed NOPCGrad vs Mixed PCGrad\n"
            "PCC(L_effective, target)"
        ),
    )


if __name__ == "__main__":
    main()