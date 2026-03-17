import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import yaml
import os


def calculate_transcript_pcc(pred, target):
    """Safely calculate PCC for a single transcript."""
    if len(pred) < 2 or np.var(pred) <= 1e-12 or np.var(target) <= 1e-12:
        return np.nan
    return pearsonr(pred, target)[0]


def main():
    print("=== TRANSCRIPT PROFILE VISUALIZATION (RANKED BY PCC) ===")

    # 1. Load Dataset Dictionary
    dict_path = "../Datasets/encodings/dataset_encoding.yaml"
    if not os.path.exists(dict_path):
        dict_path = "../Datasets/encodings/datasets_encoding.yaml"

    print(f"Loading dataset mapping from: {dict_path}")
    with open(dict_path, "r", encoding="utf-8") as f:
        name_to_idx = yaml.safe_load(f)

    idx_to_name = {v: k for k, v in name_to_idx.items()}

    print("Loading distributed Parquet files...")
    parquet_files = glob.glob("riboai_queueing/comprehensive_predictions_rank*.parquet")
    if not parquet_files:
        raise FileNotFoundError("No comprehensive_predictions_rank*.parquet files found!")

    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

    if "transcripts_id" in df.columns:
        df = df.rename(columns={"transcripts_id": "transcript_id"})

    # 2. Auto-Select the Best Transcript (Highest total reads across all datasets)
    print("Hunting for the most highly expressed transcript...")
    transcript_depths = df.groupby("transcript_id").apply(
        lambda g: np.sum([np.sum(t) for t in g["target"]])
    )
    best_transcript_id = transcript_depths.idxmax()

    print(f"Selected Transcript: {best_transcript_id} (Total Reads: {transcript_depths[best_transcript_id]:.0f})")

    t_df = df[df["transcript_id"] == best_transcript_id].copy()

    # --- Pre-calculate PCC to allow sorting ---
    print("Calculating local PCCs to rank the datasets...")
    pccs = []
    for _, row in t_df.iterrows():
        L = int(row["length"])
        y_target = row["target"][:L]
        mu_pred = row["mu"][:L]
        pccs.append(calculate_transcript_pcc(mu_pred, y_target))

    t_df["local_pcc"] = pccs

    # Sort descending by PCC (putting NaNs at the very bottom)
    t_df["sort_pcc"] = t_df["local_pcc"].fillna(-2.0)
    t_df = t_df.sort_values("sort_pcc", ascending=False)

    # 3. Setup the Stacked Plot
    num_datasets = len(t_df)
    fig, axes = plt.subplots(nrows=num_datasets + 1, ncols=1, figsize=(14, 2.5 * (num_datasets + 1)), sharex=True)

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    L = int(t_df.iloc[0]["length"])
    x_axis = np.arange(L)

    # Extract CSS array (it is intrinsic to the transcript, so we just take the first row)
    css_array = t_df.iloc[0]["css"][:L]
    # Handle both boolean masks and integer flags
    css_indices = np.where(css_array > 0)[0]
    has_css = len(css_indices) > 0

    if has_css:
        print(f"Identified {len(css_indices)} Conserved Stalling Sites on this transcript.")

    # ==========================================
    # TOP PANEL: The Universal Intrinsic w_prob
    # ==========================================
    w_prob_universal = t_df.iloc[0]["w_prob"][:L]

    ax_w = axes[0]
    ax_w.plot(x_axis, w_prob_universal, color='#27ae60', linewidth=2, label='Intrinsic Elongation Traffic ($w_{prob}$)')
    ax_w.fill_between(x_axis, 0, w_prob_universal, color='#2ecc71', alpha=0.3)

    # Plot CSS on the top track
    if has_css:
        for i, idx in enumerate(css_indices):
            label = "Conserved Stalling Site" if i == 0 else ""
            idx += 1
            ax_w.axvline(x=idx, color='#8e44ad', linestyle='--', linewidth=1.5, alpha=0.8, label=label)

    ax_w.set_title(f"Transcript: {best_transcript_id} | Universal Biomechanic Traffic Jam Probability",
                   fontweight='bold', fontsize=14, pad=10)
    ax_w.set_ylabel("$w_{prob}$", fontsize=12)
    ax_w.legend(loc="upper right")
    ax_w.grid(True, alpha=0.3)
    ax_w.set_xlim(0, L)

    # ==========================================
    # DATASET PANELS: Ranked by PCC
    # ==========================================
    for i, (_, row) in enumerate(t_df.iterrows()):
        ax = axes[i + 1]

        dataset_id = row["dataset_id"]
        dataset_name = idx_to_name.get(dataset_id, f"Unknown_{dataset_id}")

        y_target = row["target"][:L]
        mu_pred = row["mu"][:L]
        read_depth = np.sum(y_target)
        pcc = row["local_pcc"]

        pcc_str = f"{pcc:.3f}" if not np.isnan(pcc) else "N/A"

        # Ground Truth (Gray step-plot)
        ax.fill_between(x_axis, 0, y_target, step="mid", color='#95a5a6', alpha=0.5, label='True Reads ($y$)')
        ax.plot(x_axis, y_target, drawstyle="steps-mid", color='#7f8c8d', linewidth=1)

        # Model Prediction (Smooth red line)
        ax.plot(x_axis, mu_pred, color='#c0392b', linewidth=2, label='Predicted Reads ($\mu$)')

        # Plot CSS faintly on the dataset tracks to guide the eye
        if has_css:
            for idx in css_indices:
                idx += 1
                ax.axvline(x=idx, color='#8e44ad', linestyle='--', linewidth=1, alpha=0.4)

        # Highlight color logic based on the sorted ranking
        title_color = "black" if (not np.isnan(pcc) and pcc > 0.4) else "#c0392b"
        rank_str = f"Rank {i + 1}/{num_datasets}"

        ax.set_title(f"[{rank_str}] {dataset_name} | Depth: {read_depth:.0f} reads | PCC: {pcc_str}", fontsize=12,
                     color=title_color)
        ax.set_ylabel("Read Density", fontsize=10)

        if i == 0:
            ax.legend(loc="upper right")

        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Codon Position", fontsize=14, labelpad=10)

    plt.tight_layout()

    out_dir = "transcript_profiles"
    os.makedirs(out_dir, exist_ok=True)
    save_path = f"{out_dir}/profile_{best_transcript_id}_ranked.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nSuccess! High-resolution ranked genomic track saved to: {save_path}")


if __name__ == "__main__":
    main()