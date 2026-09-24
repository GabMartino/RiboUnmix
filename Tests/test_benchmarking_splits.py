from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from main_ribounmix_benchmarking import (
    BENCHMARK_PREDICTION_CHECKPOINT_VARIANTS,
    find_benchmark_prediction_checkpoint,
    resolve_benchmark_prediction_checkpoint_variants,
    save_split_manifest,
    select_single_dataset,
    split_dataset_ids,
)


class BenchmarkingSplitTests(unittest.TestCase):
    def test_single_dataset_selection_rejects_combined_requests(self) -> None:
        available = ["human", "yeast"]
        self.assertEqual(select_single_dataset("human", available), "human")
        with self.assertRaisesRegex(ValueError, "one dataset"):
            select_single_dataset(["human", "yeast"], available)
        with self.assertRaisesRegex(ValueError, "not allowed"):
            select_single_dataset("all", available)

    def test_seeded_split_is_reproducible_disjoint_and_complete(self) -> None:
        ids = [f"t{index:03d}" for index in range(100)]
        first = split_dataset_ids(
            dataset_name="fixture",
            raw_ids=ids,
            train_frac=0.8,
            val_frac=0.1,
            test_frac=0.1,
            seed=42,
        )
        second = split_dataset_ids(
            dataset_name="fixture",
            raw_ids=ids,
            train_frac=0.8,
            val_frac=0.1,
            test_frac=0.1,
            seed=42,
        )
        self.assertEqual(first, second)
        train_ids, val_ids, test_ids, counts = first
        self.assertEqual(counts, {"total": 100, "train": 80, "validation": 10, "test": 10})
        self.assertEqual(set(train_ids) | set(val_ids) | set(test_ids), set(ids))
        self.assertFalse(set(train_ids) & set(val_ids))
        self.assertFalse(set(train_ids) & set(test_ids))
        self.assertFalse(set(val_ids) & set(test_ids))

    def test_manifest_exports_json_and_one_row_per_id_tsv(self) -> None:
        cfg = OmegaConf.create(
            {
                "experiment": {"seed": 42},
                "split": {"train_frac": 0.5, "val_frac": 0.25, "test_frac": 0.25},
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "split_manifest.json"
            save_split_manifest(
                out_file=output,
                dataset_name="fixture",
                train_ids=["t1", "t2"],
                val_ids=["t3"],
                test_ids=["t4"],
                counts={"total": 4, "train": 2, "validation": 1, "test": 1},
                cfg=cfg,
            )
            manifest = json.loads(output.read_text(encoding="utf-8"))
            table = pd.read_csv(output.with_suffix(".tsv"), sep="\t")
            self.assertEqual(manifest["train_ids"], ["t1", "t2"])
            self.assertEqual(manifest["split_seed"], 42)
            self.assertEqual(manifest["training_seed"], 42)
            self.assertEqual(
                table.to_dict("records"),
                [
                    {"transcript_id": "t1", "split": "train", "split_index": 0},
                    {"transcript_id": "t2", "split": "train", "split_index": 1},
                    {"transcript_id": "t3", "split": "validation", "split_index": 0},
                    {"transcript_id": "t4", "split": "test", "split_index": 0},
                ],
            )

    def test_split_seed_can_be_frozen_independently_of_training_seed(self) -> None:
        cfg = OmegaConf.create(
            {
                "experiment": {"seed": 44},
                "split": {"train_frac": 0.5, "val_frac": 0.25, "test_frac": 0.25},
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "split_manifest.json"
            save_split_manifest(
                out_file=output,
                dataset_name="fixture",
                train_ids=["t1", "t2"],
                val_ids=["t3"],
                test_ids=["t4"],
                counts={"total": 4, "train": 2, "validation": 1, "test": 1},
                cfg=cfg,
                split_seed=42,
            )
            manifest = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(manifest["seed"], 42)
            self.assertEqual(manifest["split_seed"], 42)
            self.assertEqual(manifest["training_seed"], 44)

    def test_benchmark_checkpoint_variants_include_common_nb_monitor(self) -> None:
        cfg = OmegaConf.create(
            {"prediction": {"checkpoint_variants": ["best_nb_nll", "best_pcc"]}}
        )
        self.assertEqual(
            resolve_benchmark_prediction_checkpoint_variants(cfg),
            ("best_nb_nll", "best_pcc"),
        )
        self.assertIn("best_nb_nll", BENCHMARK_PREDICTION_CHECKPOINT_VARIANTS)

    def test_nb_checkpoint_discovery_selects_smallest_common_metric(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worse = root / "nb-epoch=2-val_nb_nll_raw=1.2000.ckpt"
            better = root / "nb-epoch=5-val_nb_nll_raw=0.9000.ckpt"
            worse.touch()
            better.touch()
            self.assertEqual(
                find_benchmark_prediction_checkpoint(root, "best_nb_nll"),
                str(better),
            )


if __name__ == "__main__":
    unittest.main()
