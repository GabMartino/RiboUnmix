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

    # --- CONFIGURATION ---
    top_n = 10  # Change this to plot more or fewer transcripts
    # ---------------------

    dict_path = "../Datasets/encodings/dataset_encoding.yaml"
    if not os.path.exists(dict_path):
        dict_path = "../Datasets/encodings/datasets_encoding.yaml"

    with open(dict_path, "r", encoding="utf-8") as f:
        name_to_idx = yaml.safe_load(f)

    idx_to_name = {v: k for k, v in name_to_idx.items()}

    print("Loading distributed Parquet files...")
    parquet_files = glob.glob("riboai_queueing/32_datasets_mix_2aae07/comprehensive_predictions_rank*.parquet")
    if not parquet_files:
        raise FileNotFoundError("No comprehensive_predictions_rank*.parquet files found!")

    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

    if "transcripts_id" in df.columns:
        df = df.rename(columns={"transcripts_id": "transcript_id"})

    print(f"Hunting for the top {top_n} most highly expressed transcripts...")
    transcript_depths = df.groupby("transcript_id").apply(
        lambda g: np.sum([np.sum(t) for t in g["target"]])
    )

    # Get the top N transcript IDs sorted by total read depth
    top_transcript_ids = transcript_depths.nlargest(top_n).index.tolist()

    out_dir = "transcript_profiles"
    os.makedirs(out_dir, exist_ok=True)

    # ==========================================
    # LOOP OVER TOP N TRANSCRIPTS
    # ==========================================
    for rank_idx, transcript_id in enumerate(top_transcript_ids, start=1):
        total_reads = transcript_depths[transcript_id]
        print(f"\n--- Plotting [{rank_idx}/{top_n}] Transcript: {transcript_id} (Reads: {total_reads:.0f}) ---")

        t_df = df[df["transcript_id"] == transcript_id].copy()
        L = int(t_df.iloc[0]["length"])
        w_prob_universal = t_df.iloc[0]["w_prob"][:L]

        pccs_mu = []
        pccs_w = []

        for _, row in t_df.iterrows():
            y_target = row["target"][:L]
            mu_pred = row["mu"][:L]

            sum_y = np.sum(y_target)
            w_target = y_target / sum_y if sum_y > 0 else np.zeros_like(y_target)

            pccs_mu.append(calculate_transcript_pcc(mu_pred, y_target))
            pccs_w.append(calculate_transcript_pcc(w_prob_universal, w_target))

        t_df["local_pcc_mu"] = pccs_mu
        t_df["local_pcc_w"] = pccs_w

        t_df["sort_pcc"] = t_df["local_pcc_mu"].fillna(-2.0)
        t_df = t_df.sort_values("sort_pcc", ascending=False)

        num_datasets = len(t_df)
        fig, axes = plt.subplots(nrows=num_datasets + 1, ncols=1, figsize=(14, 2.5 * (num_datasets + 1)), sharex=True)

        if not isinstance(axes, np.ndarray):
            axes = [axes]

        x_axis = np.arange(L)
        css_array = t_df.iloc[0]["css"][:L]
        css_indices = np.where(css_array > 0)[0]
        has_css = len(css_indices) > 0

        if has_css:
            print(f"  -> Identified {len(css_indices)} Conserved Stalling Sites.")

        # --- TOP PANEL: w_prob ---
        ax_w = axes[0]
        ax_w.plot(x_axis, w_prob_universal, color='#27ae60', linewidth=2, label='Predicted $w_{prob}$ (Universal)')
        ax_w.fill_between(x_axis, 0, w_prob_universal, color='#2ecc71', alpha=0.3)

        if has_css:
            for i, idx in enumerate(css_indices):
                label = "Conserved Stalling Site" if i == 0 else ""
                ax_w.axvline(x=idx + 1, color='#8e44ad', linestyle='--', linewidth=1.5, alpha=0.8, label=label)

        ax_w.set_title(f"Transcript: {transcript_id} | Universal Biomechanic Traffic Jam Probability",
                       fontweight='bold', fontsize=14, pad=10)
        ax_w.set_ylabel("$w_{prob}$", fontsize=12)
        ax_w.legend(loc="upper right")
        ax_w.grid(True, alpha=0.3)
        ax_w.set_xlim(0, L)

        # --- DATASET PANELS ---
        for i, (_, row) in enumerate(t_df.iterrows()):
            ax = axes[i + 1]

            dataset_id = row["dataset_id"]
            dataset_name = idx_to_name.get(dataset_id, f"Unknown_{dataset_id}")

            y_target = row["target"][:L]
            mu_pred = row["mu"][:L]
            read_depth = np.sum(y_target)

            w_target = y_target / read_depth if read_depth > 0 else np.zeros_like(y_target)

            pcc_mu = row["local_pcc_mu"]
            pcc_w = row["local_pcc_w"]

            pcc_mu_str = f"{pcc_mu:.3f}" if not np.isnan(pcc_mu) else "N/A"
            pcc_w_str = f"{pcc_w:.3f}" if not np.isnan(pcc_w) else "N/A"

            # LEFT AXIS: Raw Reads (Mu and y)
            ax.fill_between(x_axis, 0, y_target, step="mid", color='#bdc3c7', alpha=0.4, label='True Reads ($y$)')
            ax.plot(x_axis, y_target, drawstyle="steps-mid", color='#7f8c8d', linewidth=1)
            ax.plot(x_axis, mu_pred, color='#c0392b', linewidth=2, label='Predicted Reads ($\mu$)')
            ax.set_ylabel("Read Density", fontsize=10, color='#2c3e50')

            # RIGHT AXIS: Probabilities (w_prob)
            ax_w_twin = ax.twinx()
            ax_w_twin.plot(x_axis, w_target, color='#2980b9', alpha=0.5, linestyle=':', drawstyle="steps-mid",
                           label='Target $w_{prob}$')
            ax_w_twin.plot(x_axis, w_prob_universal, color='#2980b9', alpha=0.8, linewidth=1, label='Pred $w_{prob}$')
            ax_w_twin.set_ylabel("Probability", fontsize=10, color='#2980b9')
            ax_w_twin.tick_params(axis='y', labelcolor='#2980b9')

            if has_css:
                for idx in css_indices:
                    ax.axvline(x=idx + 1, color='#8e44ad', linestyle='--', linewidth=1, alpha=0.4)

            title_color = "black" if (not np.isnan(pcc_mu) and pcc_mu > 0.4) else "#c0392b"
            rank_str = f"Rank {i + 1}/{num_datasets}"

            ax.set_title(
                f"[{rank_str}] {dataset_name} | Depth: {read_depth:.0f} | PCC($\mu$): {pcc_mu_str} | PCC($w$): {pcc_w_str}",
                fontsize=12, color=title_color)

            if i == 0:
                lines_1, labels_1 = ax.get_legend_handles_labels()
                lines_2, labels_2 = ax_w_twin.get_legend_handles_labels()
                ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Codon Position", fontsize=14, labelpad=10)
        plt.tight_layout()

        save_path = f"{out_dir}/profile_rank{rank_idx}_{transcript_id}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  -> Saved: {save_path}")

        # CRITICAL: Destroy the figure to free up RAM before the next loop
        plt.close(fig)

    print("\nAll transcript profiles generated successfully.")


if __name__ == "__main__":
    main()