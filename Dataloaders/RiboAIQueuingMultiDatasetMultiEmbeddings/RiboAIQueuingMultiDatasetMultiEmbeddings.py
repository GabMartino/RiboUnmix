import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from tqdm import tqdm


class RiboAIQueuingDatasetMultiDatasetMultiEmbeddings(Dataset):
    """
    Returns per-sample:
      (dataset_id, transcript_id, encoded_ref[T, F], ribo[T], css, sample_embeddings_dict)

    Collate returns:
      (ids_datasets_sorted, ids_sorted, seq_packed, prof_pad, lengths_sorted, mask_pad, css_sorted, batch_embeddings)
    """

    def __init__(
            self,
            nt_encoding: dict,
            codon_to_aa_encoding: dict,
            codon_encoding: dict,
            embeddings: list,
            aa_encoding: dict,
            datasets_encoding: dict,
            data: dict | None = None,
            lengths: np.ndarray | None = None,
            seed: int = 42
    ):
        super().__init__()

        self.data_records = data
        self.lengths = np.asarray(lengths)

        # Encodings / maps
        self.codon_map = codon_encoding
        self.aa_map = aa_encoding
        self.codon2aa_map = codon_to_aa_encoding
        self.nt_encoding = nt_encoding
        self.datasets_encoding = datasets_encoding
        self.idx_to_dataset = {v: k for k, v in self.datasets_encoding.items()}
        self.num_codons = 64
        self.n_codons = len(self.codon_map)
        self.n_aa = len(self.aa_map)
        self.onehot2nt = {np.argmax(v).item(): k for k, v in self.nt_encoding.items()}
        self.codon_idx_to_aa_idx = {v: self.aa_map[self.codon2aa_map[k]] for k, v in self.codon_map.items()}
        self.embeddings = embeddings

        # Dataset tracking
        self.num_datasets = len(self.data_records["ribo_profiles"].keys())
        self.datasets_names = list(self.data_records["ribo_profiles"].keys())

        self.seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        # --- Caches ---
        # 1. Main reference cache (Base 99 features)
        self._encoded_cache = [None] * len(self.data_records["ref"])

        # 2. Parallel embeddings cache
        self._embeddings_cache = {
            emb_name: [None] * len(self.data_records["ref"])
            for emb_name in self.embeddings
        }

    def __len__(self) -> int:
        return len(self.data_records["ref"]) * self.num_datasets

    def _extract_features(self, nucleotide_sequence_per_codon) -> np.ndarray:
        raw_nt_sequence = np.stack([np.stack(c) for c in nucleotide_sequence_per_codon])
        batch_size = raw_nt_sequence.shape[0]
        nt_sequence = raw_nt_sequence.reshape(batch_size, -1)

        codon_sequence = np.stack([np.stack([self.onehot2nt[np.argmax(n).item()] for n in c]) for c in raw_nt_sequence])
        codon_sequence = np.char.add(np.char.add(codon_sequence[:, 0], codon_sequence[:, 1]), codon_sequence[:, 2])
        codon_sequence = np.stack([self.codon_map[c] for c in codon_sequence])

        aa_sequence = np.stack([self.codon_idx_to_aa_idx[c] for c in codon_sequence])
        aa_sequence = np.eye(self.n_aa, dtype=np.int32)[aa_sequence]
        codon_sequence = np.eye(self.num_codons, dtype=np.int32)[codon_sequence]

        concatenated_sequence = np.concatenate([nt_sequence, codon_sequence, aa_sequence], axis=1)
        return concatenated_sequence

    def __getitem__(self, index: int):
        nT = len(self.data_records["ref"])
        nD = self.num_datasets
        if index < 0 or index >= nT * nD:
            raise IndexError(index)

        idx_dataset = index // nT
        idx_transcript = index % nT

        transcript_id = self.data_records["transcript_id"][idx_transcript]
        ref = self.data_records["ref"][idx_transcript]

        # --- 1. Fetch Main Reference ---
        encoded = self._encoded_cache[idx_transcript]
        if encoded is None:
            encoded = self._extract_features(ref).astype(np.float32, copy=False)
            self._encoded_cache[idx_transcript] = encoded

        # --- 2. Fetch Extra Embeddings ---
        sample_embeddings = {}
        for emb_name in self.embeddings:
            cached_emb = self._embeddings_cache[emb_name][idx_transcript]

            if cached_emb is None:
                # Raw list is length L. Elements are size 3 arrays (per NT in codon).
                raw_list = self.data_records[emb_name][idx_transcript]
                arr = np.stack(raw_list).astype(np.float32, copy=False)

                # Safeguard for 1D arrays
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)

                cached_emb = arr
                self._embeddings_cache[emb_name][idx_transcript] = cached_emb

            sample_embeddings[emb_name] = cached_emb

        # --- 3. Fetch Targets ---
        dataset_name = self.datasets_names[idx_dataset]
        ribo = self.data_records["ribo_profiles"][dataset_name][idx_transcript]
        css = self.data_records["css"][idx_transcript]
        real_idx_dataset = self.datasets_encoding[dataset_name]

        return real_idx_dataset, transcript_id, encoded, ribo, css, sample_embeddings

    def collate_fn(self, batch):
        idx_datasets, ids, sequences, profiles, css_s, sample_emb_dicts = zip(*batch)

        lengths = torch.tensor([s.shape[0] for s in sequences], dtype=torch.long)
        lengths_sorted, order = lengths.sort(descending=True)
        order = order.tolist()

        ids_datasets_sorted = torch.tensor([idx_datasets[i] for i in order], dtype=torch.long)
        ids_sorted = [ids[i] for i in order]
        seq_sorted = [torch.as_tensor(sequences[i], dtype=torch.float32) for i in order]
        prof_sorted = [torch.as_tensor(profiles[i], dtype=torch.float32) for i in order]
        css_sorted = [css_s[i] for i in order]

        seq_pad = pad_sequence(seq_sorted, batch_first=True, padding_value=0.0)
        prof_pad = pad_sequence(prof_sorted, batch_first=True, padding_value=0.0)

        Tmax = prof_pad.size(1)
        mask_pad = (torch.arange(Tmax).unsqueeze(0) < lengths_sorted.unsqueeze(1)).bool()

        seq_packed = pack_padded_sequence(seq_pad, lengths_sorted, batch_first=True, enforce_sorted=True)

        # --- THE EMBEDDINGS COLLATION ---
        batch_embeddings = {}
        if len(self.embeddings) > 0:
            for emb_name in self.embeddings:
                # Extract specific embedding, maintain sort order
                emb_sorted = [torch.as_tensor(sample_emb_dicts[i][emb_name], dtype=torch.float32) for i in order]
                # Pad to [Batch, Tmax, Features]
                emb_pad = pad_sequence(emb_sorted, batch_first=True, padding_value=0.0)
                batch_embeddings[emb_name] = emb_pad

        return ids_datasets_sorted, ids_sorted, seq_packed, prof_pad, lengths_sorted, mask_pad, css_sorted, batch_embeddings


# =========================================================================================
# TEST BLOCK
# =========================================================================================
def main():
    # 1. Load Encodings
    aa_encoding = yaml.safe_load(open("../../Datasets/encodings/aa_encoding.yaml"))
    codon2aa = yaml.safe_load(open("../../Datasets/encodings/codon2aa.yaml"))
    codon_encoding = yaml.safe_load(open("../../Datasets/encodings/codon_encoding.yaml"))
    nt_encoding = yaml.safe_load(open("../../Datasets/encodings/nt_encoding.yaml"))

    datasets_encoding = {"grimson_2019": 0, "martinez_2020": 1}  # Add others as needed

    # 2. Replicate the DataModule's Intersection Logic
    sequences_path = "../../Datasets/data/sequence/sequence_embeddings_with_css.parquet"
    datasets_paths = [
        "../../Datasets/data/raw_datasets_with_css/grimson_2019.parquet"
    ]

    print(f"Intersecting master sequences with {len(datasets_paths)} datasets...")

    seq_df = pd.read_parquet(sequences_path)
    if "transcript_id" in seq_df.columns:
        seq_df = seq_df.set_index("transcript_id")

    common_index = seq_df.index
    loaded_datasets: dict[str, pd.DataFrame] = {}

    for path in tqdm(datasets_paths, desc="Loading and Intersecting"):
        df = pd.read_parquet(path)
        dataset_name = path.split("/")[-1].split(".")[0]

        if "id" in df.columns:
            df = df.set_index("id")
        elif "transcript_id" in df.columns:
            df = df.set_index("transcript_id")

        common_index = common_index.intersection(df.index, sort=False)
        loaded_datasets[dataset_name] = df

    dataset_names = list(loaded_datasets.keys())

    print(f"Original sequences: {len(seq_df)}")
    print(f"Sequences surviving the Inner Join: {len(common_index)}")
    if len(common_index) == 0:
        raise RuntimeError("Empty intersection between master sequences and datasets.")

    # ---- 3) filter master to common transcripts ----
    seq_df_common = seq_df.loc[common_index]

    ref_arrays = seq_df_common["ref"].values
    aas_arrays = seq_df_common["aas"].values
    dom_arrays = seq_df_common["dom"].values
    exo_arrays = seq_df_common["exo"].values
    fra_arrays = seq_df_common["fra"].values
    gmp_arrays = seq_df_common["gmp"].values
    mod_arrays = seq_df_common["mod"].values
    tmp_arrays = seq_df_common["tmp"].values
    openen_arrays = seq_df_common["openen"].values
    tAI_profile_codon_arrays = seq_df_common["tAI_profile_codon"].values

    css_col = "conserved_stalling_sites" if "conserved_stalling_sites" in seq_df_common.columns else "css"
    css = seq_df_common[css_col].values

    lengths = np.array([len(x) for x in ref_arrays], dtype=np.int32)

    shared_data = {
        "transcript_id": common_index.values,
        "ref": ref_arrays,
        "aas": aas_arrays,
        "dom": dom_arrays,
        "exo": exo_arrays,
        "fra": fra_arrays,
        "gmp": gmp_arrays,
        "mod": mod_arrays,
        "tmp": tmp_arrays,
        "openen": openen_arrays,
        "tAI_profile_codon": tAI_profile_codon_arrays,
        "css": css,
        "ribo_profiles": {},
        "lengths": lengths,
        "datasets_names": dataset_names,
    }

    for dataset_name, df in loaded_datasets.items():
        aligned_df = df.loc[common_index]
        shared_data["ribo_profiles"][dataset_name] = [
            np.asarray(arr, dtype=np.float32) for arr in aligned_df["ribo"].values
        ]

    # 3. Instantiate the Dataset
    dataset = RiboAIQueuingDatasetMultiDatasetMultiEmbeddings(
        nt_encoding=nt_encoding,
        codon_to_aa_encoding=codon2aa,
        embeddings=['dom', 'exo', 'fra', 'gmp', 'tmp', 'openen', 'tAI_profile_codon'],
        codon_encoding=codon_encoding,
        aa_encoding=aa_encoding,
        datasets_encoding=datasets_encoding,
        data=shared_data,
        lengths=lengths
    )

    print(f"\nDataset instantiated successfully. Total virtual samples (T x D): {len(dataset)}")

    # 4. Test Dataloader and Collation
    dataloader = DataLoader(dataset=dataset, batch_size=3, collate_fn=dataset.collate_fn)
    batch = next(iter(dataloader))

    # UNPACK ALL 8 ITEMS
    ids_datasets_sorted, ids_sorted, seq_packed, prof_pad, lengths_sorted, mask_pad, css_sorted, batch_embeddings = batch

    print("\n=== BATCH COLLATION TEST ===")
    print(f"Dataset IDs: {ids_datasets_sorted}")
    print(f"Transcript IDs: {ids_sorted}")
    print(f"Packed Sequence Data shape: {seq_packed.data.shape}")
    print(f"Padded Profiles shape: {prof_pad.shape}")
    print(f"Sorted Lengths: {lengths_sorted}")
    print(f"Mask shape: {mask_pad.shape}")

    print("\n--- Fused Embedding Dictionaries ---")
    for k, v in batch_embeddings.items():
        print(f"  {k:>20}: {v.shape}")
    print("============================\n")


if __name__ == "__main__":
    main()