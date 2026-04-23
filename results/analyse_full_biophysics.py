import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from scipy.stats import zscore
from tqdm import tqdm

# Amino Acid mapping for synonymous codon analysis
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


def load_dataset_names():
    dict_path = "../Datasets/encodings/dataset_encoding.yaml"
    if not os.path.exists(dict_path):
        dict_path = "../Datasets/encodings/datasets_encoding.yaml"
    with open(dict_path, "r", encoding="utf-8") as f:
        name_to_idx = yaml.safe_load(f)
    return {v: k for k, v in name_to_idx.items()}


def run_dataset_quality_profiling(df: pd.DataFrame, idx_to_name: dict, out_dir: str):
    print("\n[1/4] Profiling Dataset Quality (Zero-Inflation & Scale)...")

    # Calculate median dropout (pi) and scale across all transcripts for each dataset
    stats = []
    for ds_id, group in tqdm(df.groupby("dataset_id")):
        ds_name = idx_to_name.get(ds_id, f"DS_{ds_id}")

        # Flatten the arrays to get the global pi (dropout) and total_scale (depth)
        all_pi = np.concatenate(group["pi"].values)
        all_scale = np.concatenate(group["total_scale"].values)

        stats.append({
            "Dataset": ds_name,
            "Dropout_Probability_Pi": np.median(all_pi),
            "Median_Scale_Depth": np.median(all_scale)
        })

    ds_df = pd.DataFrame(stats).sort_values("Dropout_Probability_Pi")

    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=ds_df, x="Median_Scale_Depth", y="Dropout_Probability_Pi", s=100, color="#2980b9")

    # Annotate top/bottom datasets
    for i in range(len(ds_df)):
        if i < 3 or i > len(ds_df) - 4:
            plt.text(ds_df["Median_Scale_Depth"].iloc[i], ds_df["Dropout_Probability_Pi"].iloc[i],
                     ds_df["Dataset"].iloc[i], fontsize=9, alpha=0.7)

    plt.xscale("log")
    plt.title("Dataset Quality: Dropout Rate vs. Read Depth")
    plt.xlabel("Median Read Scale (Log)")
    plt.ylabel("Zero-Inflation Dropout Probability ($\pi$)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/1_dataset_quality.png", dpi=300)
    plt.close()


def run_rnase_bias_analysis(merged_df: pd.DataFrame, idx_to_name: dict, out_dir: str):
    print("\n[2/4] Extracting RNase Cleavage Bias (b_offset)...")
    nt_chars = np.array(['A', 'C', 'T', 'G'])

    bias_records = []

    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
        ds_id = row["dataset_id"]
        ref_raw = row["ref"]
        b_offset = row["b_offset"]
        L = len(b_offset)

        if len(ref_raw) < L: continue

        raw_nt_sequence = np.stack([np.stack(c) for c in ref_raw])
        nt_indices = np.argmax(raw_nt_sequence, axis=-1)
        codon_chars = nt_chars[nt_indices]
        ref_codons = np.array(["".join(chars) for chars in codon_chars])

        # Map the b_offset to the first nucleotide of the NEXT codon
        for i in range(L - 1):
            next_nt = ref_codons[i + 1][0]
            bias_records.append({"dataset_id": ds_id, "next_nt": next_nt, "b_offset": b_offset[i]})

    # Sample to save RAM before grouping
    b_df = pd.DataFrame(bias_records)
    if len(b_df) > 1000000: b_df = b_df.sample(1000000)

    heatmap_data = b_df.groupby(["dataset_id", "next_nt"])["b_offset"].median().unstack()
    heatmap_data.index = [idx_to_name.get(i, str(i)) for i in heatmap_data.index]

    plt.figure(figsize=(8, 10))
    sns.heatmap(heatmap_data, cmap="coolwarm", center=0, annot=True, fmt=".3f", cbar_kws={'label': 'Median b_offset'})
    plt.title("RNase Digestion Bias by Downstream Nucleotide")
    plt.ylabel("Laboratory Dataset")
    plt.xlabel("Downstream Nucleotide (+1 of next codon)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/2_rnase_bias_heatmap.png", dpi=300)
    plt.close()


def run_synonymous_codon_analysis(merged_df: pd.DataFrame, out_dir: str):
    print("\n[3/4] Mapping tRNA Pool via Synonymous Codon Drag...")
    nt_chars = np.array(['A', 'C', 'T', 'G'])

    # Your readable 3-letter encoding mapping
    AA_NAME_MAP = {
        'A': "Ala", 'C': "Cys", 'D': "Asp", 'E': "Glu", 'F': "Phe",
        'G': "Gly", 'H': "His", 'I': "Ile", 'K': "Lys", 'L': "Leu",
        'M': "Met", 'N': "Asn", 'P': "Pro", 'Q': "Gln", 'R': "Arg",
        'S': "Ser", 'T': "Thr", 'V': "Val", 'W': "Trp", 'Y': "Tyr",
        '_': "Stp"  # Catching the stop codons from our base CODON_TABLE
    }

    all_w = []
    all_codons = []

    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
        ref_raw = row["ref"]
        w = row["w_prob"]
        L = len(w)
        if len(ref_raw) < L: continue

        raw_nt_sequence = np.stack([np.stack(c) for c in ref_raw])
        nt_indices = np.argmax(raw_nt_sequence, axis=-1)
        ref_codons = np.array(["".join(chars) for chars in nt_chars[nt_indices]])

        all_w.extend(w[:L - 1])
        all_codons.extend(ref_codons[1:L])

    df_codon = pd.DataFrame({"codon": all_codons, "w_prob": all_w})

    # 1. Map to single letter first
    df_codon["AA_1letter"] = df_codon["codon"].map(CODON_TABLE)
    # 2. Map to readable 3-letter code
    df_codon["AA"] = df_codon["AA_1letter"].map(AA_NAME_MAP)

    # Filter out Stop codons to preserve the Y-axis resolution for tRNA wobble-pairs
    df_codon = df_codon[df_codon["AA"] != "Stp"]

    median_w = df_codon.groupby(["AA", "codon"])["w_prob"].median().reset_index()
    median_w = median_w.sort_values(["AA", "w_prob"], ascending=[True, False])

    plt.figure(figsize=(16, 6))
    sns.barplot(data=median_w, x="codon", y="w_prob", hue="AA", dodge=False, palette="tab20")
    plt.xticks(rotation=90, fontsize=10)
    plt.title("Synonymous Codon Drag (Inferred tRNA Abundance)")
    plt.ylabel("Median $w_{prob}$ (A-site Dwell Time)")
    plt.xlabel("Codon (Grouped by Amino Acid)")
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', ncol=2, title="Amino Acid")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/3_synonymous_codons_readable.png", dpi=300)
    plt.close()
def run_css_collision_metagene(df: pd.DataFrame, out_dir: str, window: int = 20):
    print("\n[4/4] Extracting Physical Traffic Collisions (Rho vs W_prob)...")

    aligned_w = []
    aligned_rho = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        L = row["length"]
        w = row["w_prob"]
        rho = row["rho"]
        css_raw = np.array(row["css"])

        if len(css_raw) == 0:
            continue
        elif len(css_raw) >= L:
            true_css = np.where(css_raw[:L] > 0)[0]
        else:
            true_css = css_raw[css_raw < L].astype(int)

        for stall_idx in true_css:
            if (stall_idx - window >= 0) and (stall_idx + window < L):
                aligned_w.append(w[stall_idx - window: stall_idx + window + 1])
                aligned_rho.append(rho[stall_idx - window: stall_idx + window + 1])

    if len(aligned_w) == 0: return

    mat_w = np.stack(aligned_w)
    mat_rho = np.stack(aligned_rho)

    # Z-score normalize to put them on the same scale for comparison
    mean_w = np.mean(mat_w, axis=0)
    mean_rho = np.mean(mat_rho, axis=0)

    norm_w = zscore(mean_w)
    norm_rho = zscore(mean_rho)
    x_axis = np.arange(-window, window + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, norm_w, color='#27ae60', linewidth=2.5, label='Intrinsic Speed Limit ($w_{prob}$)')
    plt.plot(x_axis, norm_rho, color='#2980b9', linewidth=2.5, linestyle='--',
             label='Physical Traffic Density ($\\rho$)')
    plt.axvline(x=0, color='black', linestyle=':', alpha=0.8, label='Conserved Stalling Site')

    plt.title("Ribosome Collision Dynamics around Stalling Sites")
    plt.xlabel("Distance from Stall (Codons)")
    plt.ylabel("Z-Scored Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/4_traffic_collisions.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    out_dir = "biophysics_discoveries"
    os.makedirs(out_dir, exist_ok=True)

    idx_to_name = load_dataset_names()

    # Find latest predictions
    base_dir = "./riboai_queueing"
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    pred_dir = max(subdirs, key=os.path.getmtime)

    parquet_files = glob.glob(os.path.join(pred_dir, "comprehensive_predictions_rank*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    df = df.drop_duplicates(subset=["dataset_id", "transcripts_id"])

    print(f"Loaded {len(df)} physical records.")

    # 1. Dataset Quality
    run_dataset_quality_profiling(df, idx_to_name, out_dir)

    # 2. Collision Dynamics
    run_css_collision_metagene(df, out_dir)

    # Merge for sequence-dependent metrics (Light merge to save RAM)
    print("\nMerging sequence tensors...")
    seq_df = pd.read_parquet("../Datasets/data/sequence/sequence_embeddings_with_css.parquet")[["transcript_id", "ref"]]
    merged_df = pd.merge(seq_df, df, right_on="transcripts_id", left_on="transcript_id", how="inner")

    # Take a 15% random sample of the merged df to prevent OOM when decoding millions of one-hot tensors
    if len(merged_df) > 50000:
        print("Sampling 50,000 transcript instances to protect RAM...")
        merged_df = merged_df.sample(50000, random_state=42)

    # 3. RNase Bias
    run_rnase_bias_analysis(merged_df, idx_to_name, out_dir)

    # 4. tRNA Pool
    run_synonymous_codon_analysis(merged_df, out_dir)

    print("\nAll Discoveries saved to the 'biophysics_discoveries' directory!")