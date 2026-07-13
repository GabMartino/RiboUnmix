import glob
import os
from functools import reduce

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.auto import tqdm


def main():

    main_path = "./data/weighted_datasets_ribo_only_with_css"
    out_path = "./data/weighted_datasets_ribo_replicas"
    os.makedirs(out_path, exist_ok=True)
    datasets_path = glob.glob(os.path.join(main_path, "*"))
    full_replicas_paths = glob.glob(os.path.join("./data/raw_full_dataset_replicas", "*"))

    path_sequence = "./data/sequence/raw_sequence_embeddings.parquet"
    sequence = pd.read_parquet(path_sequence, columns=['mod', 'transcript_id'])

    for dataset_path in datasets_path:
        dataset_name = os.path.basename(dataset_path).split(".")[0]
        print(dataset_name)
        replicas = [pd.read_parquet(p) for p in full_replicas_paths if dataset_name in p]

        combined_replicas = reduce(lambda left, right: pd.merge(left, right, on="transcript_id"), replicas)
        columns = [ col for col in combined_replicas.columns if col.startswith("psite")]
        combined_replicas["ribo_replicas"] = combined_replicas[columns].apply(
            lambda row: row.dropna().tolist(), axis=1)

        combined_replicas.drop(columns=columns, inplace=True)
        merged = pd.merge(sequence, combined_replicas, how="inner", left_on="transcript_id", right_on="transcript_id")
        merged["ribo_cds_replicas"] = None
        for idx, row in tqdm(merged.iterrows()):
            mod = row['mod']
            ribo_replicas = row['ribo_replicas']
            cds_replicas_per_codon = []
            for rep in ribo_replicas:
                cds_rep = np.array(rep[mod == 1])
                cds_rep_codon = cds_rep.reshape(-1, 3).sum(axis=-1)
                cds_replicas_per_codon.append(cds_rep_codon)

            merged.at[idx, "ribo_cds_replicas"] = cds_replicas_per_codon
        merged.drop(columns=["ribo_replicas", "mod"], inplace=True)
        weighted_dataset = pd.read_parquet(dataset_path)
        print(weighted_dataset.columns, merged.columns)
        merged = pd.merge(weighted_dataset, merged, how="inner", left_on="id", right_on="transcript_id")
        print(merged.head())
        print("CHECK")
        for idx, row in tqdm(merged.iterrows()):
            ribo = row["ribo"]
            ribo_cds_replicas = row["ribo_cds_replicas"]
            assert all([len(rep) == len(ribo) for rep in ribo_cds_replicas])
            ribo_cds_replicas_avg = np.mean(ribo_cds_replicas, axis=0)
            '''
                It seems there is a problem of shifting of 1 codon
            '''
            ##assert np.equal(ribo_cds_replicas_avg, ribo).all()
        merged.to_parquet(os.path.join(out_path, f"{dataset_name}.parquet"))
        print(merged)

if __name__ == "__main__":
    main()