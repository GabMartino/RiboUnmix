from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main_ribounmix_multidataset import (
    build_transcript_metadata,
    fixed_common_validation_split,
    get_split_universe_datasets,
    load_common_transcript_reliability,
    save_split_manifest,
)


class CommonWeightStratifiedSplitTests(unittest.TestCase):
    def test_all_master_universe_uses_active_dataset_config(self) -> None:
        cfg = OmegaConf.create(
            {
                "split": {"master_dataset_universe": "all"},
                "dataset_config": {
                    "dataset_path": {
                        "active_a": "active_a.parquet",
                        "active_b": "active_b.parquet",
                    }
                },
            }
        )
        resolved = get_split_universe_datasets(cfg, ["active_a"])
        self.assertEqual(resolved, ["active_a", "active_b"])

    def _write_fixture(self, root: Path) -> tuple[Path, list[Path]]:
        common = [f"t{i:02d}" for i in range(40)]
        all_ids = common + ["only_a", "only_b"]
        sequence_path = root / "sequences.parquet"
        pd.DataFrame(
            {
                "transcript_id": all_ids,
                # Alternate CSS-negative and CSS-positive candidates inside
                # every reliability quantile. CSS is a stratification signal,
                # not a separate held-out split.
                "css": [([] if index % 2 == 0 else [10]) for index in range(40)]
                + [[], []],
                "codons": [["ATG", "AAA", "TAA"] for _ in all_ids],
            }
        ).to_parquet(sequence_path, index=False)

        dataset_a = root / "dataset_a.parquet"
        dataset_b = root / "dataset_b.parquet"
        # Scores span low to high reliability. Dataset B is offset so the test
        # also checks the ordinary two-value median used for aggregation.
        weights_a = np.linspace(0.1, 2.0, len(common), dtype=np.float32)
        weights_b = weights_a + np.float32(0.2)
        pd.DataFrame(
            {
                "id": common + ["only_a"],
                "weight": np.concatenate([weights_a, np.asarray([1.0], np.float32)]),
            }
        ).to_parquet(dataset_a, index=False)
        pd.DataFrame(
            {
                "id": common + ["only_b"],
                "weight": np.concatenate([weights_b, np.asarray([1.0], np.float32)]),
            }
        ).to_parquet(dataset_b, index=False)

        return sequence_path, [dataset_a, dataset_b]

    def test_validation_is_common_weight_stratified_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sequence_path, dataset_paths = self._write_fixture(Path(tmp))

            kwargs = dict(
                sequences_path=sequence_path,
                split_universe_dataset_paths=dataset_paths,
                validation_frac=0.25,
                random_seed=17,
                validation_weight_bins=5,
            )
            with redirect_stdout(StringIO()):
                train, validation, metadata = fixed_common_validation_split(**kwargs)
                train_again, validation_again, _ = (
                    fixed_common_validation_split(**kwargs)
                )

            self.assertEqual(validation, validation_again)
            self.assertEqual(train, train_again)
            self.assertEqual(len(validation), 10)
            self.assertTrue(set(train).isdisjoint(validation))
            self.assertEqual(set(train) | set(validation), set(metadata))
            self.assertNotIn("only_a", validation)
            self.assertNotIn("only_b", validation)
            self.assertTrue(
                all(metadata[tid]["common_to_validation_datasets"] for tid in validation)
            )

            # Forty candidates form five equal rank bins. A ten-ID validation
            # target therefore takes two IDs from every reliability region.
            bin_counts = Counter(
                metadata[tid]["validation_reliability_bin"] for tid in validation
            )
            self.assertEqual(bin_counts, Counter({0: 2, 1: 2, 2: 2, 3: 2, 4: 2}))
            css_counts = Counter(metadata[tid]["css_bin"] for tid in validation)
            self.assertEqual(css_counts, Counter({"css_0": 5, "css_1": 5}))

            expected_score_t00 = (0.1 + 0.3) / 2.0
            self.assertAlmostEqual(
                metadata["t00"]["validation_reliability_score"],
                expected_score_t00,
                places=6,
            )

            manifest_path = Path(tmp) / "split_manifest.json"
            with redirect_stdout(StringIO()):
                save_split_manifest(
                    out_file=manifest_path,
                    experiment_datasets=["dataset_a", "dataset_b"],
                    split_universe_datasets=["dataset_a", "dataset_b"],
                    split_dataset_paths=dataset_paths,
                    training_dataset_paths=dataset_paths,
                    validation_weight_bins=5,
                    train_ids=train,
                    validation_ids=validation,
                    metadata=metadata,
                    seed=17,
                    validation_frac=0.25,
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            selection = manifest["validation_selection"]
            self.assertTrue(selection["all_validation_ids_are_common"])
            self.assertTrue(selection["fixed_across_experiment_dataset_subsets"])
            self.assertEqual(manifest["counts"]["common_validation_candidates"], 40)
            self.assertNotIn("css_benchmark_ids", manifest)
            self.assertEqual(
                manifest["validation_reliability_bin_counts"]["validation"],
                {f"qbin_{index:02d}": 2 for index in range(5)},
            )

    def test_max_cds_codons_excludes_complete_transcript_before_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sequence_path, dataset_paths = self._write_fixture(Path(tmp))
            sequences = pd.read_parquet(sequence_path)
            sequences.at[0, "codons"] = ["ATG"] * 6
            sequences.to_parquet(sequence_path, index=False)

            with redirect_stdout(StringIO()):
                metadata = build_transcript_metadata(
                    sequences_path=sequence_path,
                    datasets_paths=dataset_paths,
                    max_cds_codons=5,
                )

            self.assertNotIn("t00", metadata)
            self.assertIn("t01", metadata)
            self.assertEqual(metadata["t01"]["cds_codon_length"], 3)

    def test_zero_weight_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "bad_dataset.parquet"
            pd.DataFrame({"id": ["bad"], "weight": [0.0]}).to_parquet(
                dataset,
                index=False,
            )
            with self.assertRaisesRegex(ValueError, "strictly positive.*bad_dataset.*bad"):
                load_common_transcript_reliability(
                    datasets_paths=[dataset],
                    eligible_ids=["bad"],
                )


if __name__ == "__main__":
    unittest.main()
