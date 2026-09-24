"""Print the active dataset names from a dataset-config YAML, one per line."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=Path("config/dataset_config/weighted_hek_riboseq_codon_replicas.yaml"),
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.manifest.read_text(encoding="utf-8")) or {}
    dataset_path = config.get("dataset_path")
    if not isinstance(dataset_path, dict) or not dataset_path:
        raise SystemExit(f"No non-empty dataset_path mapping in {args.manifest}")

    for dataset_name in sorted(map(str, dataset_path)):
        print(dataset_name)


if __name__ == "__main__":
    main()
