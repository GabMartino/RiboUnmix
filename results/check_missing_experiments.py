import os

import yaml


def main():

    results_path = "./riboai_queueing"
    config_path_file = "../config/dataset_config/datasets_paths.yaml"

    config_file = yaml.safe_load(open(config_path_file))["dataset_path"]

    dataset_names = set(config_file.keys())

    dataset_name_results = [d for d in os.listdir(results_path) if "32" not in d]

    missing_results = dataset_names.difference(dataset_name_results)
    print(missing_results)

if __name__ == "__main__":
    main()