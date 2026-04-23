import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from tqdm import tqdm


def analyze_css_biophysics(df: pd.DataFrame, window_size: int = 30):
    print("Extracting Thermodynamic Drag with ±1 Codon Tolerance...")
    background_w = []
    css_w = []
    aligned_profiles = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing CSS"):
        L = row["length"]
        w_prob = np.array(row["w_prob"])
        css_raw = np.array(row["css"])

        # THE FIX: Bring back your original, correct logic
        if len(css_raw) == 0:
            true_css_indices = np.array([])
        elif len(css_raw) >= L:
            # It's a binary mask
            true_css_indices = np.where(css_raw[:L] > 0)[0]
        else:
            # It's a list of absolute codon coordinates!
            true_css_indices = css_raw[css_raw < L].astype(int)

        if len(true_css_indices) == 0:
            # If no CSS, the entire transcript is background
            background_w.extend(w_prob)
            continue

        is_css_mask = np.zeros(L, dtype=bool)

        for stall_idx in true_css_indices:
            start_idx = max(0, stall_idx - 1)
            end_idx = min(L, stall_idx + 2)

            peak_w = np.max(w_prob[start_idx:end_idx])
            css_w.append(peak_w)
            is_css_mask[start_idx:end_idx] = True

            if (stall_idx - window_size >= 0) and (stall_idx + window_size < L):
                windowed_w = w_prob[stall_idx - window_size: stall_idx + window_size + 1]
                aligned_profiles.append(windowed_w)

        background_w.extend(w_prob[~is_css_mask])

    if len(css_w) == 0:
        print("Warning: No valid CSS events found in this split.")
        return

    median_background = np.median(background_w)
    median_css = np.median(css_w)
    fold_change = median_css / median_background if median_background > 0 else 0
    stat, p_value = mannwhitneyu(css_w, background_w, alternative='greater')

    print("\n" + "=" * 50)
    print("=== CSS BIOPHYSICAL EFFECT SIZE ===")
    print(f"Total CSS Events:             {len(css_w)}")
    print(f"Background Median w_prob:     {median_background:.6f}")
    print(f"CSS Median w_prob:            {median_css:.6f}")
    print(f"Thermodynamic Drag (Fold ∆):  {fold_change:.2f}x")
    print(f"Mann-Whitney P-Value:         {p_value:.2e} (Note: Heavily influenced by large N)")
    print("=" * 50 + "\n")

    if len(aligned_profiles) > 0:
        css_matrix = np.stack(aligned_profiles)
        metagene_median = np.median(css_matrix, axis=0)
        p25 = np.percentile(css_matrix, 25, axis=0)
        p75 = np.percentile(css_matrix, 75, axis=0)
        x_axis = np.arange(-window_size, window_size + 1)

        plt.figure(figsize=(10, 5))
        plt.plot(x_axis, metagene_median, color='#d95f02', linewidth=2.5, label='Median w_prob')
        plt.fill_between(x_axis, p25, p75, color='#d95f02', alpha=0.2, label='IQR')
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.8, label='Annotated CSS')
        plt.title("Intrinsic Elongation Velocity (w_prob) around Conserved Stalling Sites")
        plt.xlabel("Distance from CSS (Codons)")
        plt.ylabel("Pure Dwell Time Probability")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("global_css_biophysics_proof.png", dpi=300)
        plt.close()  # Prevent memory leaks if running in a notebook or loop


def analyze_a_site_hierarchy(merged_df: pd.DataFrame):
    print("Mapping 61-Codon A-Site Hierarchy...")
    all_w_probs = []
    all_a_site_codons = []

    # Fast lookup array for decoding the argmax indices
    nt_chars = np.array(['A', 'C', 'T', 'G'])

    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
        ref_raw = row["ref"]
        w = np.array(row["w_prob"])
        L = len(w)

        if len(ref_raw) < L:
            continue

        # THE FIX: Restore your tensor decoder!
        raw_nt_sequence = np.stack([np.stack(c) for c in ref_raw])
        nt_indices = np.argmax(raw_nt_sequence, axis=-1)
        codon_chars = nt_chars[nt_indices]
        ref_codons = np.array(["".join(chars) for chars in codon_chars])

        w_aligned = w[:L - 1]
        a_site_codons = ref_codons[1:L]

        all_w_probs.append(w_aligned)
        all_a_site_codons.append(a_site_codons)

    df_flat = pd.DataFrame({"codon": np.concatenate(all_a_site_codons), "w_prob": np.concatenate(all_w_probs)})
    codon_medians = df_flat.groupby("codon")["w_prob"].median().sort_values(ascending=False)

    print("\n=== A-SITE DECODING DRAG ===")
    print("Top 5 Slowest Codons:")
    print(codon_medians.head(5))
    print("\nTop 5 Fastest Codons:")
    print(codon_medians.tail(5))


def analyze_initiation_thermodynamics(merged_df: pd.DataFrame):
    print("\n=== INITIATION THERMODYNAMICS (J) ===")
    j_stats = []
    grouped = merged_df.groupby("transcript_id")
    nt_chars = np.array(['A', 'C', 'T', 'G'])

    for transcript_id, group in tqdm(grouped, total=len(grouped)):
        mean_J = group["J"].mean()
        ref_raw = group.iloc[0]["ref"]
        L = len(ref_raw)

        if L < 50:
            continue

        # THE FIX: Restore tensor decoding here as well
        raw_nt_sequence = np.stack([np.stack(c) for c in ref_raw])
        nt_indices = np.argmax(raw_nt_sequence, axis=-1)
        codon_chars = nt_chars[nt_indices]
        ref_codons = np.array(["".join(chars) for chars in codon_chars])

        # Now we can safely join the string codons
        full_transcript_nts = "".join(ref_codons)
        first_150_nts = full_transcript_nts[:150]

        gc_count = first_150_nts.count('G') + first_150_nts.count('C')
        gc_content = (gc_count / len(first_150_nts)) * 100

        j_stats.append({
            "transcript_id": transcript_id,
            "J": mean_J,
            "length": L,
            "gc_content_5prime": gc_content
        })

    df_j = pd.DataFrame(j_stats)

    df_j["length_bin"] = pd.qcut(df_j["length"], q=4, labels=["Short", "Med-Short", "Med-Long", "Long"],
                                 duplicates='drop')
    print("\n--- J vs. Length ---")
    print(df_j.groupby("length_bin", observed=True)["J"].median())

    df_j["gc_bin"] = pd.qcut(df_j["gc_content_5prime"], q=4, labels=["Low GC", "Med-Low", "Med-High", "High GC"],
                             duplicates='drop')
    print("\n--- J vs. 5' GC Content ---")
    print(df_j.groupby("gc_bin", observed=True)["J"].median())

if __name__ == "__main__":

    # THE FIX: Do not hardcode the hash. If your hash changes, your code breaks.
    # Grab the most recently modified directory in the riboai_queueing folder.
    base_dir = "./riboai_queueing"

    try:
        subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        pred_dir = max(subdirs, key=os.path.getmtime)
        print(f"Auto-detected prediction directory: {pred_dir}")
    except ValueError:
        print("Error: Could not find any subdirectories in ./riboai_queueing")
        exit()

    parquet_files = glob.glob(os.path.join(pred_dir, "comprehensive_predictions_rank*.parquet"))

    if not parquet_files:
        print(f"Error: No predictions found in {pred_dir}. Did the Lightning Trainer finish?")
        exit()

    print(f"Aggregating {len(parquet_files)} distributed Parquet files...")
    df_list = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(df_list, ignore_index=True)

    initial_len = len(df)
    df = df.drop_duplicates(subset=["dataset_id", "transcripts_id"])
    print(f"Dropped {initial_len - len(df)} duplicate rows generated by DDP padding.")

    analyze_css_biophysics(df)

    seq_df = pd.read_parquet("../Datasets/data/sequence/sequence_embeddings_with_css.parquet")
    seq_df_light = seq_df[["transcript_id", "ref"]]

    print("\nMerging physical predictions with sequence references...")
    merged_df = pd.merge(seq_df_light, df, right_on="transcripts_id", left_on="transcript_id", how="inner")

    analyze_a_site_hierarchy(merged_df)
    analyze_initiation_thermodynamics(merged_df)