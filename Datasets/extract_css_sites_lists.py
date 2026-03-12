import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def main():

    base_path = "../Datasets/conserved_stalling_sites"
    css_data = pd.read_parquet(os.path.join(base_path, "stalling_sites.parquet"))
    sequence_data = pd.read_parquet(os.path.join("../Datasets/data/sequence", "sequence_embeddings.parquet"))
    css_data.rename(columns={'transcript_id': 'id'}, inplace=True)

    sequence_data["id"] = sequence_data["transcript_id"].str.split('.').str[0]
    merged_data = pd.merge(sequence_data, css_data, left_on="id", right_on="id", how="left")
    merged_data.drop("id", axis=1, inplace=True)

    validation_set_ids = merged_data.loc[merged_data['conserved_stalling_sites'].notna(), 'transcript_id'].tolist()
    training_set_ids = merged_data.loc[merged_data['conserved_stalling_sites'].isna(), 'transcript_id'].tolist()
    split = {
        "training_set": training_set_ids,
        "validation_set": validation_set_ids
    }
    with open(os.path.join("../Datasets/data/sequence", "css_split.json"), "w") as f:
        json.dump(split, f)

    merged_data.to_parquet(os.path.join("../Datasets/data/sequence", "sequence_embeddings_with_css.parquet"))


if __name__ == '__main__':
    main()