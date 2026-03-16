import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import seaborn as sns
from tqdm import tqdm


def analyze_css_biophysics(df: pd.DataFrame, window_size: int = 30):
    print("Extracting Thermodynamic Drag with ±1 Codon Tolerance...")
    background_w = []
    css_w = []
    aligned_profiles = []

    for _, row in df.iterrows():
        L = row["length"]
        w_prob = row["w_prob"]
        css = row["css"]

        if isinstance(css, (list, np.ndarray)):
            css_array = np.array(css)
            if len(css_array) >= L:
                true_css_indices = np.where(css_array[:L] > 0)[0]
            else:
                true_css_indices = css_array[css_array < L]
        else:
            continue

        if len(true_css_indices) == 0:
            # If no CSS, the entire transcript is background
            background_w.extend(w_prob)
            continue

        # Create a boolean mask to strictly exclude the CSS AND its flanks
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

        # THE FIX: Extract background inside the loop using the correct mask!
        background_w.extend(w_prob[~is_css_mask])

    # --- Statistical Proof ---
    if len(css_w) == 0:
        raise ValueError("No valid CSS events found.")

    median_background = np.median(background_w)
    median_css = np.median(css_w)
    fold_change = median_css / median_background if median_background > 0 else 0
    stat, p_value = mannwhitneyu(css_w, background_w, alternative='greater')

    print("\n" + "=" * 50)
    print("=== CSS BIOPHYSICAL EFFECT SIZE (±1 Tolerance) ===")
    print(f"Total CSS Events Analyzed:    {len(css_w)}")
    print(f"Background Median w_prob:     {median_background:.6f}")
    print(f"CSS Median w_prob:            {median_css:.6f}")
    print(f"Thermodynamic Drag (Fold ∆):  {fold_change:.2f}x")
    print(f"Mann-Whitney P-Value:         {p_value:.2e}")
    print("=" * 50 + "\n")

    # --- Metagene Generation ---
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
        plt.title("Intrinsic Elongation Velocity (w_prob) around Conserved Stalling Sites", fontsize=14)
        plt.xlabel("Distance from CSS (Codons)", fontsize=12)
        plt.ylabel("Pure Dwell Time Probability", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("global_css_biophysics_proof.png", dpi=300)


# =====================================================================
# THE FOUR PILLARS OF DISENTANGLEMENT
# =====================================================================
import pandas as pd
import numpy as np
from tqdm import tqdm


def analyze_a_site_hierarchy(merged_df: pd.DataFrame):
    """ Pillar 1: A-Site Decoding Hierarchy (Corrected for Speed, Physics, and One-Hot Tensors) """
    print("Mapping 61-Codon A-Site Hierarchy...")

    all_w_probs = []
    all_a_site_codons = []

    # Fast lookup array for decoding the argmax indices
    nt_chars = np.array(['A', 'C', 'T', 'G'])

    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
        ref = row["ref"]
        # Reconstruct the one-hot tensor: Shape [L, 3, 4]
        raw_nt_sequence = np.stack([np.stack(c) for c in ref])
        w = np.array(row["w_prob"])

        L = len(w)
        if len(raw_nt_sequence) < L:
            continue

        # --- THE TENSOR DECODER ---
        # 1. Find the index of the '1' in the one-hot vector. Shape becomes [L, 3]
        nt_indices = np.argmax(raw_nt_sequence, axis=-1)

        # 2. Map indices back to characters (e.g., 0 -> 'A'). Shape remains [L, 3]
        codon_chars = nt_chars[nt_indices]

        # 3. Join the 3 chars into string codons. Shape becomes [L] (e.g., ["ATG", "CGC", ...])
        decoded_codons = np.array(["".join(chars) for chars in codon_chars])
        # --------------------------

        # BIOPHYSICAL FIX: Shift by +1 to map P-site drag to A-site decoding
        w_aligned = w[:L - 1]
        a_site_codons = decoded_codons[1:L]

        all_w_probs.append(w_aligned)
        all_a_site_codons.append(a_site_codons)

    # COMPUTATIONAL FIX: Vectorize the aggregation
    flat_w = np.concatenate(all_w_probs)
    flat_codons = np.concatenate(all_a_site_codons)

    # Because we decoded the tensors into pure strings, this is now perfectly hashable
    df_flat = pd.DataFrame({"codon": flat_codons, "w_prob": flat_w})

    # Calculate median drag per codon and sort
    codon_medians = df_flat.groupby("codon")["w_prob"].median().sort_values(ascending=False)

    print("\n=== A-SITE DECODING DRAG ===")
    print("Top 5 Slowest Codons (Highest Drag):")
    print(codon_medians.head(5))
    print("\nTop 5 Fastest Codons (Lowest Drag):")
    print(codon_medians.tail(5))
    print("============================\n")


def analyze_electrostatic_drag(merged_df: pd.DataFrame):
    """ Pillar 2: Exit Tunnel Charge Drag (Corrected for One-Hot Tensors) """
    print("\nCalculating Electrostatic Exit Tunnel Drag...")

    # Arginine (R) and Lysine (K) codons
    pos_codons = set(['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG', 'AAA', 'AAG'])
    charge_w_probs = {0: [], 1: [], 2: [], 3: [], "4+": []}

    nt_chars = np.array(['A', 'C', 'T', 'G'])

    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
        ref = row["ref"]
        raw_nt_sequence = np.stack([np.stack(c) for c in ref])
        w_prob = np.array(row["w_prob"])
        L = len(w_prob)

        if len(raw_nt_sequence) < L:
            continue

        # --- THE TENSOR DECODER ---
        nt_indices = np.argmax(raw_nt_sequence, axis=-1)
        codon_chars = nt_chars[nt_indices]
        ref_codons = np.array(["".join(chars) for chars in codon_chars])
        # --------------------------

        # We need at least 25 codons to have a full exit tunnel upstream
        for i in range(25, L):
            # The peptide currently in the exit tunnel is roughly 10 to 25 codons UPSTREAM of the A-site
            tunnel_window = ref_codons[i - 25: i - 10]

            # Count the number of positively charged codons in the tunnel right now
            pos_charge_count = sum(1 for c in tunnel_window if c in pos_codons)

            key = pos_charge_count if pos_charge_count < 4 else "4+"
            # Map this upstream charge to the CURRENT dwell time in the A-site
            charge_w_probs[key].append(w_prob[i])

    print("\n=== ELECTROSTATIC EXIT TUNNEL DRAG ===")
    print("Median w_prob based on positive charges in exit tunnel:")
    for k in [0, 1, 2, 3, "4+"]:
        if len(charge_w_probs[k]) > 0:
            print(f"{k} charges: {np.median(charge_w_probs[k]):.6f} (n={len(charge_w_probs[k])})")
    print("======================================\n")


def analyze_rnase_bias(merged_df: pd.DataFrame):
    """ Pillar 3: Dataset Chemical Bias (Corrected for One-Hot Tensors) """
    print("\nExtracting Dataset-Specific RNase Biases from b_offset...")

    dataset_biases = {}
    nt_chars = np.array(['A', 'C', 'T', 'G'])

    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
        ds_id = row["dataset_id"]
        ref = row["ref"]
        raw_nt_sequence = np.stack([np.stack(c) for c in ref])
        b_offset = np.array(row["b_offset"])
        L = len(b_offset)

        if len(raw_nt_sequence) < L:
            continue

        # --- THE TENSOR DECODER ---
        nt_indices = np.argmax(raw_nt_sequence, axis=-1)
        codon_chars = nt_chars[nt_indices]
        ref_codons = np.array(["".join(chars) for chars in codon_chars])
        # --------------------------

        if ds_id not in dataset_biases:
            dataset_biases[ds_id] = {'A': [], 'C': [], 'T': [], 'G': []}

        # We look at how the +1 nucleotide (downstream cleavage context) affects the offset
        for i in range(L - 1):
            next_nt = ref_codons[i + 1][0]
            if next_nt in dataset_biases[ds_id]:
                dataset_biases[ds_id][next_nt].append(b_offset[i])

    print("\n=== RNASE DIGESTION BIAS (b_offset) ===")
    for ds_id, bases in dataset_biases.items():
        med_A = np.median(bases['A']) if bases['A'] else 0
        med_C = np.median(bases['C']) if bases['C'] else 0
        med_T = np.median(bases['T']) if bases['T'] else 0
        med_G = np.median(bases['G']) if bases['G'] else 0
        print(
            f"Dataset {ds_id} Median b_offset | A: {med_A:>6.3f}, C: {med_C:>6.3f}, T: {med_T:>6.3f}, G: {med_G:>6.3f}")
    print("=======================================\n")


def analyze_initiation_thermodynamics(merged_df: pd.DataFrame):
    """ Pillar 4: Initiation Rate (J) vs. Transcript Architecture """
    print("\n=== PILLAR 4: INITIATION THERMODYNAMICS (J) ===")

    j_stats = []
    nt_chars = np.array(['A', 'C', 'T', 'G'])

    # Group by transcript_id to get the intrinsic properties
    grouped = merged_df.groupby("transcript_id")

    for transcript_id, group in tqdm(grouped, total=len(grouped)):
        # J varies slightly by dataset due to conditioning, so we take the intrinsic mean
        mean_J = group["J"].mean()

        # All rows for a transcript have the same reference sequence
        ref = group.iloc[0]["ref"]
        raw_nt_sequence = np.stack([np.stack(c) for c in ref])
        L = len(raw_nt_sequence)

        if L < 50:
            continue

        # --- TENSOR DECODER ---
        nt_indices = np.argmax(raw_nt_sequence, axis=-1)
        codon_chars = nt_chars[nt_indices]

        # Flatten the first 50 codons (150 nucleotides) to check 5' folding proxy
        first_50_codons_nts = codon_chars[:50].flatten()

        # Calculate GC Content percentage
        gc_count = np.sum((first_50_codons_nts == 'G') | (first_50_codons_nts == 'C'))
        gc_content = (gc_count / len(first_50_codons_nts)) * 100

        j_stats.append({
            "transcript_id": transcript_id,
            "J": mean_J,
            "length": L,
            "gc_content_5prime": gc_content
        })

    df_j = pd.DataFrame(j_stats)

    # 1. Analyze by Length (Closed-loop kinetics)
    df_j["length_bin"] = pd.qcut(df_j["length"], q=4, labels=["Short", "Medium-Short", "Medium-Long", "Long"])
    length_medians = df_j.groupby("length_bin")["J"].median()

    print("\n--- J vs. Transcript Length (Closed-Loop Efficiency) ---")
    for bin_name, median_j in length_medians.items():
        print(f"{bin_name:>15} Transcripts: Median J = {median_j:.6f}")

    # 2. Analyze by 5' GC Content (Structural folding proxy)
    df_j["gc_bin"] = pd.qcut(df_j["gc_content_5prime"], q=4,
                             labels=["Low GC (Unstructured)", "Medium-Low", "Medium-High", "High GC (Rigid)"])
    gc_medians = df_j.groupby("gc_bin")["J"].median()

    print("\n--- J vs. 5' GC Content (Structural Rigidity Proxy) ---")
    for bin_name, median_j in gc_medians.items():
        print(f"{bin_name:>25}: Median J = {median_j:.6f}")
    print("==================================================\n")
if __name__ == "__main__":
    parquet_path = "./riboai_queueing/comprehensive_predictions.parquet"
    print(f"Loading master database: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    # 1. Run the CSS analysis with the fixed background logic
    analyze_css_biophysics(df)
    exit()

    # To run the pillars, you must merge the physical predictions with the sequence data
    # Ensure your extraction script and sequence dataframe align properly (e.g., via index or transcript_id).
    seq_df = pd.read_parquet("../Datasets/data/sequence/sequence_embeddings_with_css.parquet")
    print(seq_df.columns, df.columns)
    merged_df = pd.merge(seq_df, df, right_on="transcripts_id", left_on="transcript_id", how="inner")
    # analyze_a_site_hierarchy(merged_df)
    # analyze_electrostatic_drag(merged_df)
    #analyze_rnase_bias(merged_df)
    analyze_initiation_thermodynamics(merged_df)