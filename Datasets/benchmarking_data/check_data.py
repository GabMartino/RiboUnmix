import pandas as pd


def main():


    data = pd.read_parquet("ecoli_zhang_2016.parquet")
    print(data.head())
    print(data.columns)
    print(len(data.index))
if __name__ == "__main__":
    main()