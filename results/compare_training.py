import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import yaml
from tqdm import tqdm


def calculate_median_pccs(df):
    """Calculates median PCC for both Mu (chemistry+biology) and W_prob (pure biology)."""
    mu_pccs = []
    w_pccs = []

    for _, row in df.iterrows():
        L = int(row["length"])
        y = row["target"][:L]
        mu = row["mu"][:L]
        w = row["w_prob"][:L]

        sum_y = np.sum(y)

        # Calculate empirical w_target (normalized reads)
        w_target = y / sum_y if sum_y > 0 else np.zeros_like(y)

        if len(y) > 2 and np.var(y) > 1e-12:
            # Mu PCC
            if np.var(mu) > 1e-12:
                p_mu = pearsonr(mu, y)[0]
                if not np.isnan(p_mu): mu_pccs.append(p_mu)

            # W_prob PCC
            if np.var(w) > 1e-12 and np.var(w_target) > 1e-12:
                p_w = pearsonr(w, w_target)[0]
                if not np.isnan(p_w): w_pccs.append(p_w)

    return {
        "Mu": np.median(mu_pccs) if mu_pccs else np.nan,
        "W": np.median(w_pccs) if w_pccs else np.nan
    }


def main():
    print("=== MULTI-TASK VS SINGLE-TASK LEARNING COMPARISON ===")

    base_dir = "../riboai_queueing"
    mix_folder_name = "32_datasets_mix_2aae07"
    mix_dir = os.path.join(base_dir, mix_folder_name)

    # 1. Load Dataset Name Mapping
    dict_path = "../../Datasets/encodings/dataset_encoding.yaml"
    if not os.path.exists(dict_path):
        dict_path = "../Datasets/encodings/dataset_encoding.yaml"
    with open(dict_path, "r", encoding="utf-8") as f:
        name_to_idx = yaml.safe_load(f)
    idx_to_name = {v: k for k, v in name_to_idx.items()}

    # 2. Process the Mixed Model
    print(f"\nLoading Mixed Model (Universal) predictions...")
    mix_files = glob.glob(os.path.join(mix_dir, "comprehensive_predictions_rank*.parquet"))
    print(mix_files)
    mix_df = pd.concat([pd.read_parquet(f) for f in mix_files], ignore_index=True)
    mix_df = mix_df.drop_duplicates(subset=["dataset_id", "transcripts_id"])

    mix_results = {}
    for ds_id, group in tqdm(mix_df.groupby("dataset_id"), desc="Mixed Model Metrics"):
        ds_name = idx_to_name.get(ds_id, f"Unknown_{ds_id}")
        mix_results[ds_name] = calculate_median_pccs(group)

    # 3. Process Individual Models
    print("\nScanning for Individually Trained Models...")
    indiv_folders = [d for d in os.listdir(base_dir)
                     if os.path.isdir(os.path.join(base_dir, d)) and d != mix_folder_name]

    indiv_results = {}
    for folder in tqdm(indiv_folders, desc="Individual Models"):
        folder_path = os.path.join(base_dir, folder)
        indiv_files = glob.glob(os.path.join(folder_path, "comprehensive_predictions_rank*.parquet"))
        print(indiv_files)
        if not indiv_files: continue

        indiv_df = pd.concat([pd.read_parquet(f) for f in indiv_files], ignore_index=True)
        indiv_df = indiv_df.drop_duplicates(subset=["dataset_id", "transcripts_id"])
        # Determine dataset name from the folder name
        ds_name = folder
        if ds_name in idx_to_name.values():  # Just in case folder is named exactly as dataset
            indiv_results[ds_name] = calculate_median_pccs(indiv_df)

    # 4. Combine Data
    print("\nMerging data for head-to-head comparison...")
    comparison_data = []

    for ds_name in indiv_results.keys():
        if ds_name in mix_results:
            # Add Individual
            comparison_data.append({
                "Dataset": ds_name, "PCC": indiv_results[ds_name]["Mu"],
                "Metric": "Mu (Predicted Reads)", "Architecture": "Isolated (Trained on 1 Dataset)"
            })
            comparison_data.append({
                "Dataset": ds_name, "PCC": indiv_results[ds_name]["W"],
                "Metric": "W_prob (Intrinsic Biology)", "Architecture": "Isolated (Trained on 1 Dataset)"
            })
            # Add Mixed
            comparison_data.append({
                "Dataset": ds_name, "PCC": mix_results[ds_name]["Mu"],
                "Metric": "Mu (Predicted Reads)", "Architecture": "Universal (Trained on 32 Datasets)"
            })
            comparison_data.append({
                "Dataset": ds_name, "PCC": mix_results[ds_name]["W"],
                "Metric": "W_prob (Intrinsic Biology)", "Architecture": "Universal (Trained on 32 Datasets)"
            })

    plot_df = pd.DataFrame(comparison_data)

    if len(plot_df) == 0:
        print("Warning: No matching individual models found to compare against the mixed model.")
        return

    # Sort order based on Universal Model's W_prob performance
    sort_mask = (plot_df["Architecture"] == "Universal (Trained on 32 Datasets)") & (
                plot_df["Metric"] == "W_prob (Intrinsic Biology)")
    sort_order = plot_df[sort_mask].sort_values("PCC", ascending=False)["Dataset"].tolist()

    # 5. Plotting (2-Panel Stacked Graph)
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

    # Panel 1: Mu
    sns.barplot(data=plot_df[plot_df["Metric"] == "Mu (Predicted Reads)"],
                x="Dataset", y="PCC", hue="Architecture", order=sort_order,
                palette=["#e74c3c", "#2ecc71"], ax=axes[0])
    axes[0].set_title("Architecture Comparison: Target vs Predicted Reads ($\mu$) [Chemistry + Biology]", fontsize=14,
                      fontweight="bold")
    axes[0].set_ylabel("Median PCC ($\mu$)", fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].legend(loc="upper right")

    # Panel 2: W_prob
    sns.barplot(data=plot_df[plot_df["Metric"] == "W_prob (Intrinsic Biology)"],
                x="Dataset", y="PCC", hue="Architecture", order=sort_order,
                palette=["#c0392b", "#27ae60"], ax=axes[1])
    axes[1].set_title("Architecture Comparison: Target vs Intrinsic Dwell Time ($w_{prob}$) [Pure Biology Only]",
                      fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Median PCC ($w_{prob}$)", fontsize=12)
    axes[1].set_xlabel("Laboratory Dataset", fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].legend(loc="upper right")

    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.tight_layout()

    out_path = "../biophysics_discoveries/architecture_generalization_comparison.png"
    os.makedirs("biophysics_discoveries", exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"\nSuccess! High-resolution dual-metric comparison saved to: {out_path}")


if __name__ == "__main__":
    main()