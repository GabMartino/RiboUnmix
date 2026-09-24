"""Select a deterministic nested subset for the dataset-quality ablation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml


DEFAULT_MANIFEST = Path(
    "config/dataset_config/weighted_hek_riboseq_codon_replicas.yaml"
)
DEFAULT_RANKING = Path("Datasets/data/HEK_riboseq_profile_quality_rank.tsv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("count", help="Number of datasets, or 'all'.")
    parser.add_argument(
        "strategy", choices=("rank_stratified", "top_quality")
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    return parser.parse_args()


def select_datasets(
    requested: str,
    strategy: str,
    manifest_path: Path,
    ranking_path: Path,
) -> list[str]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    active = list(manifest["dataset_path"])
    with ranking_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    rank_by_dataset = {
        str(row["dataset"]): float(row["quality_rank"]) for row in rows
    }
    missing = sorted(set(active) - set(rank_by_dataset))
    if missing:
        raise SystemExit(f"Active datasets missing from quality ranking: {missing}")

    ranked_active = sorted(active, key=lambda name: (rank_by_dataset[name], name))
    n = len(ranked_active) if requested.lower() == "all" else int(requested)
    if n < 2 or n > len(ranked_active):
        raise SystemExit(
            f"Dataset count must be in [2, {len(ranked_active)}], got {n}"
        )

    if strategy == "top_quality":
        ordered = ranked_active
    else:
        # Nested one-dimensional farthest-point ordering over global ranks.
        selected_indices = [0, len(ranked_active) - 1]
        remaining = set(range(1, len(ranked_active) - 1))
        while remaining:
            next_index = max(
                remaining,
                key=lambda index: (
                    min(abs(index - chosen) for chosen in selected_indices),
                    -index,
                ),
            )
            selected_indices.append(next_index)
            remaining.remove(next_index)
        ordered = [ranked_active[index] for index in selected_indices]

    return ordered[:n]


def main() -> None:
    args = parse_args()
    selected = select_datasets(
        args.count, args.strategy, args.manifest, args.ranking
    )
    print(len(selected))
    print("[" + ",".join(selected) + "]")


if __name__ == "__main__":
    main()
