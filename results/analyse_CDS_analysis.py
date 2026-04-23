import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Amino Acid mapping
CODON_TABLE = {
    'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M', 'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
    'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K', 'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
    'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L', 'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
    'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q', 'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
    'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V', 'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
    'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E', 'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
    'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S', 'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
    'TAC': 'Y', 'TAT': 'Y', 'TAA': '_', 'TAG': '_', 'TGC': 'C', 'TGT': 'C', 'TGA': '_', 'TGG': 'W',
}


def main():
    print("=== CDS DEEP PHYSICS EXTRACTION ===")

    # 1. Load Data
    out_dir = "cds_physics_discoveries"
    os.makedirs(out_dir, exist_ok=True)

    base_dir = "./riboai_queueing"
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    pred_dir = max(subdirs, key=os.path.getmtime)

    parquet_files = glob.glob(os.path.join(pred_dir, "comprehensive_predictions_rank*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    df = df.drop_duplicates(subset=["dataset_id", "transcripts_id"])

    print("\nMerging sequence tensors (Taking a 50,000 transcript sample to protect RAM)...")
    seq_df = pd.read_parquet("../Datasets/data/sequence/sequence_embeddings_with_css.parquet")[["transcript_id", "ref"]]
    merged_df = pd.merge(seq_df, df, right_on="transcripts_id", left_on="transcript_id", how="inner")

    if len(merged_df) > 50000:
        merged_df = merged_df.sample(50000, random_state=42)

    # 2. Data Structures for the 4 Pillars
    ramp_profiles = []

    proline_data = {'1P': [], '2P': [], '3P+': []}

    electrostatics_data = {0: [], 1: [], 2: [], 3: [], "4+": []}

    termination_context = {}

    nt_chars = np.array(['A', 'C', 'T', 'G'])

    # 3. Single-Pass Parser
    print("\nExtracting physical phenomena across the CDS...")
    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
        ref_raw = row["ref"]
        w = row["w_prob"]
        L = len(w)
        if len(ref_raw) < L: continue

        # Decode Tensors to Amino Acids
        raw_nt_sequence = np.stack([np.stack(c) for c in ref_raw])
        nt_indices = np.argmax(raw_nt_sequence, axis=-1)
        ref_codons = np.array(["".join(chars) for chars in nt_chars[nt_indices]])
        aa_seq = np.array([CODON_TABLE.get(c, 'X') for c in ref_codons])

        # --- PILLAR 1: The 5' Translational Ramp ---
        # Look at the first 100 codons
        ramp_len = min(L, 100)
        padded_ramp = np.full(100, np.nan)
        padded_ramp[:ramp_len] = w[:ramp_len]
        ramp_profiles.append(padded_ramp)

        # --- PILLAR 2 & 3: Proline Traps and Exit Tunnel Electrostatics ---
        for i in range(1, L):
            current_aa = aa_seq[i]
            current_w = w[i]

            # Poly-Proline Traps
            if current_aa == 'P':
                if i >= 2 and aa_seq[i - 1] == 'P' and aa_seq[i - 2] == 'P':
                    proline_data['3P+'].append(current_w)
                elif aa_seq[i - 1] == 'P':
                    proline_data['2P'].append(current_w)
                else:
                    proline_data['1P'].append(current_w)

            # Exit Tunnel Electrostatics (Look 15 to 30 AAs upstream)
            if i >= 30:
                tunnel_window = aa_seq[i - 30: i - 15]
                # Count positive charges (Arginine 'R' and Lysine 'K')
                pos_charges = np.sum((tunnel_window == 'R') | (tunnel_window == 'K'))
                key = pos_charges if pos_charges < 4 else "4+"
                electrostatics_data[key].append(current_w)

        # --- PILLAR 4: Termination P-Site Context ---
        # Find the Stop Codon ('_')
        stop_indices = np.where(aa_seq == '_')[0]
        if len(stop_indices) > 0:
            stop_idx = stop_indices[0]  # Take the first stop codon
            # Ensure there is a P-site amino acid
            if stop_idx >= 1:
                p_site_aa = aa_seq[stop_idx - 1]
                stop_w = w[stop_idx]
                if p_site_aa not in termination_context:
                    termination_context[p_site_aa] = []
                termination_context[p_site_aa].append(stop_w)

    # ==========================================
    # GENERATE PLOTS
    # ==========================================
    print("\nGenerating visual proofs...")

    # 1. Translational Ramp
    ramp_matrix = np.nan_to_num(np.stack(ramp_profiles), nan=np.nanmedian(np.concatenate(ramp_profiles)))
    ramp_median = np.median(ramp_matrix, axis=0)
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(100), ramp_median, color='#d35400', linewidth=2.5)
    plt.title("The 5' Translational Ramp (Global Bottleneck)")
    plt.xlabel("Codon Position (from Start)")
    plt.ylabel("Median $w_{prob}$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/1_translational_ramp.png", dpi=300)
    plt.close()

    # 2. Poly-Proline
    pp_df = pd.DataFrame([
                             {"Motif": "Single Proline (P)", "w_prob": w} for w in proline_data['1P']
                         ] + [
                             {"Motif": "Di-Proline (PP)", "w_prob": w} for w in proline_data['2P']
                         ] + [
                             {"Motif": "Tri-Proline (PPP+)", "w_prob": w} for w in proline_data['3P+']
                         ])
    plt.figure(figsize=(8, 6))
    sns.barplot(data=pp_df, x="Motif", y="w_prob", palette="viridis", capsize=0.1)
    plt.title("Poly-Proline Geometry Traps (A-Site & P-Site Clash)")
    plt.ylabel("Dwell Time ($w_{prob}$) on Final Proline")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/2_poly_proline.png", dpi=300)
    plt.close()

    # 3. Exit Tunnel Electrostatics
    elec_df = pd.DataFrame([
        {"Positive Charges in Tunnel": str(k), "w_prob": w}
        for k, vals in electrostatics_data.items() for w in vals
    ])
    plt.figure(figsize=(8, 6))
    sns.barplot(data=elec_df, x="Positive Charges in Tunnel", y="w_prob", palette="magma",
                order=["0", "1", "2", "3", "4+"])
    plt.title("Ribosome Exit Tunnel Drag (Electrostatic Friction)")
    plt.xlabel("Number of Arginine/Lysine (-30 to -15 codons upstream)")
    plt.ylabel("A-Site Dwell Time ($w_{prob}$)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/3_exit_tunnel_electrostatics.png", dpi=300)
    plt.close()

    # 4. Termination Context
    term_records = []
    for aa, w_list in termination_context.items():
        if len(w_list) > 50:  # Only plot if we have enough statistical power
            term_records.append({"P-Site Amino Acid": aa, "Median w_prob": np.median(w_list)})
    term_df = pd.DataFrame(term_records).sort_values("Median w_prob", ascending=False)

    plt.figure(figsize=(12, 5))
    sns.barplot(data=term_df, x="P-Site Amino Acid", y="Median w_prob", palette="coolwarm")
    plt.title("Termination Efficiency by P-Site Context")
    plt.xlabel("Amino Acid immediately preceding Stop Codon")
    plt.ylabel("Stop Codon Dwell Time ($w_{prob}$)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/4_termination_context.png", dpi=300)
    plt.close()

    print(f"Success! All 4 deep physics proofs saved to '{out_dir}/'")


if __name__ == "__main__":
    main()