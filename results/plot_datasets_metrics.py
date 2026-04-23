import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from tqdm import tqdm
import yaml
import os
import hashlib
from pathlib import Path
import hydra
from omegaconf import DictConfig


def calculate_transcript_pcc(pred, target):
    """Safely calculate PCC for a single transcript."""
    if len(pred) < 2 or np.var(pred) <= 1e-12 or np.var(target) <= 1e-12:
        return np.nan
    return pearsonr(pred, target)[0]


def get_dataset_signature(datasets):
    """
    Mirrors the hashing logic in main.py to find the correct directory.
    Prevents 'File name too long' errors on Linux systems.
    """
    sorted_datasets = sorted(list(datasets))
    raw_dataset_str = "_".join(sorted_datasets)

    if len(raw_dataset_str) > 100:
        short_hash = hashlib.md5(raw_dataset_str.encode()).hexdigest()[:6]
        return f"{len(datasets)}_datasets_mix_{short_hash}"
    else:
        return raw_dataset_str


@hydra.main(version_base=None, config_path="../config", config_name="config_riboai_queuing_multidataset")
def main(cfg: DictConfig):
    print("\n=== DATASET PERFORMANCE DIAGNOSTICS (SYNCED) ===")

    # 1. Resolve Dataset List (Mirroring Trainer Logic)
    raw_datasets_cfg = cfg.experiment.dataset
    if isinstance(raw_datasets_cfg, str) and raw_datasets_cfg.lower() == "all":
        datasets = list(cfg.dataset_config.dataset_path.keys())
        print(f"[*] 'all' detected: processing {len(datasets)} datasets.")
    elif isinstance(raw_datasets_cfg, str):
        datasets = [raw_datasets_cfg]
    else:
        datasets = list(raw_datasets_cfg)

    # 2. Path Resolution with Signature Hashing
    dataset_str = get_dataset_signature(datasets)

    # We use cfg.paths.results as the base, following the training folder structure
    data_dir = Path(cfg.paths.results) / dataset_str

    # Robust Pathing: If run from inside a subfolder, try to find the data relative to root
    if not data_dir.exists():
        potential_data_dir = Path("..") / data_dir
        if potential_data_dir.exists():
            data_dir = potential_data_dir

    print(f"[*] Targeting results in: {data_dir.resolve()}")

    # 3. Load Dataset Mapping (with robust path check)
    dict_path = Path(cfg.paths.encodings.datasets)
    if not dict_path.exists():
        dict_path = Path("..") / dict_path
        if not dict_path.exists():
            raise FileNotFoundError(f"Could not find encoding file at {cfg.paths.encodings.datasets}")

    with open(dict_path, "r", encoding="utf-8") as f:
        name_to_idx = yaml.safe_load(f)
    idx_to_name = {v: k for k, v in name_to_idx.items()}

    # 4. Load Parquet Files
    parquet_files = glob.glob(str(data_dir / "comprehensive_predictions_rank*.parquet"))
    if not parquet_files:
        print(f"ERROR: No Parquet files found in {data_dir}")
        print("Check if you ran the prediction/inference step in main.py for this dataset combo.")
        return

    print(f"[*] Found {len(parquet_files)} rank files. Concatenating...")
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    print(f"[*] Total interactions loaded: {len(df)}")

    # 5. Compute Detailed Metrics
    metrics_records = []
    depth_records = []

    print("[*] Computing biophysical correlations...")
    for dataset_id, group in tqdm(df.groupby("dataset_id")):
        dataset_name = idx_to_name.get(dataset_id, f"Unknown_{dataset_id}")

        mu_pccs, rho_pccs, w_pccs, w_kls, read_depths = [], [], [], [], []

        for _, row in group.iterrows():
            mu_pred = row["mu"]
            rho_pred = row["rho"]
            w_pred = row["w_prob"]
            y_target = row["target"]
            total_scale = row["total_scale"]

            # Read depth for this transcript
            depth = np.sum(y_target)
            read_depths.append(depth)

            # Reconstruction for Rho and W (distribution targets)
            rho_target = y_target / np.maximum(total_scale, 1e-8)
            w_target = y_target / max(depth, 1e-6)

            mu_pccs.append(calculate_transcript_pcc(mu_pred, y_target))
            rho_pccs.append(calculate_transcript_pcc(rho_pred, rho_target))
            w_pccs.append(calculate_transcript_pcc(w_pred, w_target))

            # Exact KL matching PyTorch F.kl_div behavior
            eps = 1e-10
            wp = w_pred / np.maximum(np.sum(w_pred), eps)
            wt = w_target / np.maximum(np.sum(w_target), eps)

            # Only calculate log ratio where target is > 0 to avoid eps inflation on true zeros
            valid_mask = wt > 0
            kl_pointwise = np.zeros_like(wt)

            # KL(target || pred) = target * log(target / pred)
            kl_pointwise[valid_mask] = wt[valid_mask] * np.log(
                wt[valid_mask] / np.clip(wp[valid_mask], a_min=eps, a_max=None))

            w_kls.append(np.sum(kl_pointwise))

        # Aggregate averages
        mean_mu = np.nanmean(mu_pccs)
        mean_rho = np.nanmean(rho_pccs)
        mean_w = np.nanmean(w_pccs)
        mean_kl = np.nanmean(w_kls)
        mean_depth = np.mean(read_depths)

        # Record for plots
        metrics_records.append({"Dataset": dataset_name, "Metric": "Mu (Total Reads)", "PCC": mean_mu})
        metrics_records.append({"Dataset": dataset_name, "Metric": "Rho (Density)", "PCC": mean_rho})
        metrics_records.append({"Dataset": dataset_name, "Metric": "W_prob (Elongation)", "PCC": mean_w})

        depth_records.append({
            "Dataset": dataset_name,
            "Mean_Mu_PCC": mean_mu,
            "Mean_W_KL": mean_kl,
            "Mean_Read_Depth": mean_depth
        })

    df_metrics = pd.DataFrame(metrics_records)
    df_depth = pd.DataFrame(depth_records)

    # 6. Visualization
    output_plot_dir = Path("diagnostic_results") / dataset_str
    output_plot_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    # Barplot: Global Performance
    plt.figure(figsize=(18, 8))
    sns.barplot(data=df_metrics, x="Dataset", y="PCC", hue="Metric", palette="viridis")
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.title(f"Performance Metrics: {dataset_str}\n(Ranked by Dataset ID)", fontsize=16)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_plot_dir / "performance_summary.png", dpi=300)

    plt.figure(figsize=(10, 8))

    # THE FIX: Check if we actually have enough datasets to correlate
    if len(df_depth) >= 2:
        log_depth = np.log10(df_depth["Mean_Read_Depth"])
        r_val, p_val = pearsonr(log_depth, df_depth["Mean_Mu_PCC"])
        sns.regplot(x=df_depth["Mean_Read_Depth"], y=df_depth["Mean_Mu_PCC"],
                    scatter_kws={'s': 100, 'alpha': 0.6}, line_kws={'color': 'red', 'ls': '--'})
        title_str = f"Read Depth vs. Model Accuracy (r={r_val:.3f})"
    else:
        # Fallback for single-dataset runs
        sns.scatterplot(data=df_depth, x="Mean_Read_Depth", y="Mean_Mu_PCC", s=100, alpha=0.6, color="#3498db")
        title_str = "Read Depth vs. Model Accuracy (Single Dataset)"

    for i, row in df_depth.iterrows():
        plt.text(row["Mean_Read_Depth"], row["Mean_Mu_PCC"] + 0.01, row["Dataset"], fontsize=9, ha='center')

    plt.xscale('log')
    plt.title(title_str, fontweight='bold')
    plt.xlabel("Mean Transcript Read Depth (Log Scale)")
    plt.ylabel("Mean Mu PCC")
    plt.tight_layout()
    plt.savefig(output_plot_dir / "depth_vs_accuracy.png", dpi=300)

    print(f"\n[!] Success. All diagnostics and plots saved to: {output_plot_dir.resolve()}")


if __name__ == "__main__":
    main()