import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os
import yaml


def generate_scatter_plot(x, y, pcc, p_value, title, xlabel, ylabel, save_path):
    """Helper function to generate and save standardized scatter plots."""
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="ticks", context="talk")

    ax = sns.regplot(
        x=x,
        y=y,
        scatter_kws={'alpha': 0.15, 'color': '#2c3e50', 's': 20},
        line_kws={'color': '#e74c3c', 'linewidth': 3}
    )

    textstr = f'PCC = {pcc:.3f}\np < 0.001' if p_value < 0.001 else f'PCC = {pcc:.3f}\np = {p_value:.3f}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    plt.title(title, pad=15, fontweight='bold', fontsize=14)
    plt.xlabel(xlabel, labelpad=10, fontsize=12)
    plt.ylabel(ylabel, labelpad=10, fontsize=12)

    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    print("=== INITIATION THERMODYNAMICS (J) vs TE ===")

    # 1. Load Dataset Dictionary
    # Ensure this points to the exact file containing your dataset mapping
    dict_path = "../Datasets/encodings/dataset_encoding.yaml"
    print(f"Loading dataset mapping from: {dict_path}")
    with open(dict_path, "r", encoding="utf-8") as f:
        name_to_idx = yaml.safe_load(f)

    # Invert dictionary to map ID -> Name (e.g., 3 -> 'grimson_2019')
    idx_to_name = {v: k for k, v in name_to_idx.items()}

    # 2. Load Experimental TE Data (Global Median)
    te_path = "../Datasets/data/TE_ilr_residual.clr.median_across_datasets.csv"
    print(f"Loading Experimental TE data from: {te_path}")
    te_df = pd.read_csv(te_path)

    # 3. Load Distributed Parquet Predictions
    print("Loading distributed Parquet files...")
    parquet_files = glob.glob("riboai_queueing/32_datasets_mix_2aae07/comprehensive_predictions_rank*.parquet")
    if not parquet_files:
        raise FileNotFoundError("No comprehensive_predictions_rank*.parquet files found!")

    pred_df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

    # Standardize column names for the merge
    pred_df = pred_df.rename(columns={"transcripts_id": "transcript_id"})

    # Create output directory for the individual plots
    out_dir = "J_vs_TE_plots"
    os.makedirs(out_dir, exist_ok=True)

    # ==========================================
    # DATASET-SPECIFIC J SCATTER PLOTS
    # ==========================================
    print(f"\n--- Generating {len(idx_to_name)} Dataset-Specific Scatter Plots ---")

    dataset_metrics = []

    for dataset_id, group in pred_df.groupby("dataset_id"):

        # Look up the human-readable name using the inverted dictionary
        dataset_name = idx_to_name.get(dataset_id, f"Unknown_Dataset_{dataset_id}")

        merged_local = pd.merge(te_df, group, on="transcript_id", how="inner").dropna(subset=["TE_clr_median", "J"])

        if len(merged_local) < 2:
            print(f"[{dataset_name}] Not enough overlapping transcripts. Skipping.")
            continue

        te_array = merged_local["TE_clr_median"].values
        j_array = merged_local["J"].values

        pcc, p_value = pearsonr(te_array, j_array)
        dataset_metrics.append((dataset_name, pcc, len(merged_local)))

        print(f"Plotting {dataset_name} | PCC: {pcc:.4f}")

        generate_scatter_plot(
            x=te_array,
            y=j_array,
            pcc=pcc,
            p_value=p_value,
            title=f"Dataset: {dataset_name}\nInitiation Rate ($J$) vs. Global TE",
            xlabel="Experimental Global TE (clr_median)",
            ylabel=f"Predicted $J$ ({dataset_name})",
            save_path=f"{out_dir}/J_vs_TE_{dataset_name}.png"
        )

    # Print a summary report to detect bad disentanglement
    print("\n=== DATASET ROBUSTNESS REPORT ===")
    dataset_metrics.sort(key=lambda x: x[1], reverse=True)  # Sort by PCC descending
    for name, pcc, n in dataset_metrics:
        print(f"{name:<20} | PCC: {pcc:.4f} | n={n}")
    print("=================================")
    print(f"All plots successfully saved to the '{out_dir}/' directory.")


if __name__ == "__main__":
    main()