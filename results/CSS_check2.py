import os
import glob

import pandas as pd


def main():
    base_path = "riboai_queueing/green_2020_wu_2019"
    pcgrad_path = ["NOPCGrad", "PCGrad"]

    for p in pcgrad_path:
        path = os.path.join(base_path, p)
        data_path = glob.glob(os.path.join(path, "*.parquet"))[0]
        data = pd.read_parquet(data_path)

        print(data.columns)

if __name__ == "__main__":
    main()