import pandas as pd


def main():

    TE_data_path = "./Datasets/data/raw_datasets/TE_ilr_residual.clr.median_across_datasets.csv"

    TE_data = pd.read_csv(TE_data_path)



if __name__ == "__main__":
    main()