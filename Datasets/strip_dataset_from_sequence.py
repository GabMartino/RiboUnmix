import glob
import os

import pandas as pd


def main():


    base_path = "./data/raw_datasets_with_css"

    datasets_paths = os.listdir(base_path)
    base_path_out_path = "./data/raw_datasets_ribo_only_with_css"
    os.makedirs(base_path_out_path, exist_ok=True)
    for path in datasets_paths:
        print(path)
        data = pd.read_parquet(os.path.join(base_path, path))
        subset = data[["id", "transcript_id", "ribo"]]

        subset.to_parquet(os.path.join(base_path_out_path, path), index=False)
        del data


if __name__ == "__main__":
    main()