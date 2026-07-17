import pandas as pd


def main():


    data = pd.read_parquet("celegans_cds.parquet")
    print(len(data.index))
    data = pd.read_parquet("ecoli_cds.parquet")
    print(len(data.index))
    data = pd.read_parquet("human_cds.parquet")
    print(len(data.index))
    data = pd.read_parquet("yeast_cds.parquet")
    print(len(data.index))
if __name__ == "__main__":
    main()