from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from Dataloaders.RiboUnmixBenchmarking import available_ids_for_dataset
from Dataloaders.RiboUnmixBenchmarking.RiboUnmixBenchmarkingDataset import (
    RiboUnmixBenchmarkingDataset,
)
from Datasets.benchmarking_data.weight_benchmarking_datasets import (
    BenchmarkDatasetSpec,
    preprocess_benchmark_datasets,
)


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkingPreprocessingTests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> BenchmarkDatasetSpec:
        spec = BenchmarkDatasetSpec("fixture", "profiles.parquet", "cds.parquet")
        pd.DataFrame(
            {
                "id": ["zero", "low", "high"],
                "transcript_id": ["zero", "low", "high"],
                "gene_name": ["g0", "g1", "g2"],
                "ribo": [
                    np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
                    np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                    np.asarray([8.0, 8.0, 8.0], dtype=np.float32),
                ],
            }
        ).to_parquet(root / spec.ribo_filename, index=False)
        pd.DataFrame(
            {
                "id": ["zero", "low", "high"],
                "transcript_id": ["zero", "low", "high"],
                "gene_name": ["g0", "g1", "g2"],
                "cds_seq": [
                    np.asarray(["ATG", "GCT", "TAA"]),
                    np.asarray(["ATG", "GCT", "TAA"]),
                    np.asarray(["ATG", "GCT", "TAA"]),
                ],
            }
        ).to_parquet(root / spec.cds_filename, index=False)
        return spec

    def test_shared_pipeline_filters_zero_and_preserves_replica_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "weighted"
            spec = self._write_fixture(root)

            # Patch only the selected source registry for this isolated fixture.
            import Datasets.benchmarking_data.weight_benchmarking_datasets as module

            original_specs = module.DATASET_SPECS
            module.DATASET_SPECS = (spec,)
            try:
                summaries = preprocess_benchmark_datasets(
                    input_dir=root,
                    output_dir=output_dir,
                    codon_encoding_path=ROOT
                    / "Datasets/encodings/codon_encoding.yaml",
                )
            finally:
                module.DATASET_SPECS = original_specs

            weighted = pd.read_parquet(output_dir / "fixture.parquet")
            self.assertEqual(weighted["id"].tolist(), ["low", "high"])
            self.assertEqual(int(summaries[0]["removed_zero_rows"]), 1)
            self.assertAlmostEqual(float(weighted["weight"].median()), 1.0, places=6)
            self.assertGreater(float(weighted["weight"].max()), 1.0)
            for row in weighted.itertuples(index=False):
                replicas = np.stack(
                    [
                        np.asarray(replica, dtype=np.float32)
                        for replica in row.ribo_cds_replicas
                    ],
                    axis=0,
                )
                self.assertEqual(replicas.shape, (1, len(row.ribo)))
                np.testing.assert_array_equal(replicas[0], row.ribo)
                self.assertEqual(list(row.replica_ids), ["source_profile"])

            specs = {
                "fixture": {
                    "ribo_path": str(output_dir / "fixture.parquet"),
                    "cds_path": str(root / spec.cds_filename),
                }
            }
            self.assertEqual(available_ids_for_dataset(specs, "fixture"), ["low", "high"])

    def test_raw_unweighted_profile_is_rejected_by_loader_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = self._write_fixture(root)
            specs = {
                "fixture": {
                    "ribo_path": str(root / spec.ribo_filename),
                    "cds_path": str(root / spec.cds_filename),
                }
            }
            with self.assertRaisesRegex(KeyError, "not preprocessed"):
                available_ids_for_dataset(specs, "fixture")

    def test_invalid_codon_is_reported_with_dataset_and_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = self._write_fixture(root)
            cds = pd.read_parquet(root / spec.cds_filename)
            cds.at[1, "cds_seq"] = np.asarray(["ATG", "NNN", "TAA"])
            cds.to_parquet(root / spec.cds_filename, index=False)

            import Datasets.benchmarking_data.weight_benchmarking_datasets as module

            original_specs = module.DATASET_SPECS
            module.DATASET_SPECS = (spec,)
            try:
                with self.assertRaisesRegex(ValueError, "fixture.*low.*NNN"):
                    preprocess_benchmark_datasets(
                        input_dir=root,
                        output_dir=root / "weighted",
                        codon_encoding_path=ROOT
                        / "Datasets/encodings/codon_encoding.yaml",
                        dry_run=True,
                    )
            finally:
                module.DATASET_SPECS = original_specs

    def test_compact_codon_adapter_matches_current_collate_contract(self) -> None:
        def load_yaml(name: str):
            with (ROOT / "Datasets/encodings" / name).open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle)

        codon_encoding = load_yaml("codon_encoding.yaml")
        codon_ids = np.asarray(
            [codon_encoding["ATG"], codon_encoding["GCT"], codon_encoding["TAA"]],
            dtype=np.uint8,
        )
        refs = np.empty(1, dtype=object)
        refs[0] = codon_ids
        css = np.empty(1, dtype=object)
        css[0] = np.empty(0, dtype=np.int64)
        profile = np.asarray([1.0, 2.0, 1.0], dtype=np.float32)
        shared = {
            "transcript_id": np.asarray(["t1"]),
            "ref": refs,
            "css": css,
            "ribo_profiles": {"t1": {"fixture": profile}},
            "ribo_replicas": {"t1": {"fixture": profile[None, :]}},
            "sample_weights": {"t1": {"fixture": 1.25}},
            "lengths": np.asarray([3], dtype=np.int32),
            "datasets_names": ["fixture"],
            "sequence_features": {},
        }
        for precompute_features in (False, True):
            dataset = RiboUnmixBenchmarkingDataset(
                data=shared,
                lengths=shared["lengths"],
                nt_encoding=load_yaml("nt_encoding.yaml"),
                codon_to_aa_encoding=load_yaml("codon2aa.yaml"),
                codon_encoding=codon_encoding,
                aa_encoding=load_yaml("aa_encoding.yaml"),
                datasets_encoding={"fixture": 0},
                transcripts_ids=["t1"],
                precompute_features=precompute_features,
                additional_sequence_features={},
            )
            sample = dataset[0]
            self.assertEqual(sample[2].shape, (3, 0))
            np.testing.assert_array_equal(sample[3], codon_ids)
            batch = dataset.collate_fn([sample])
            self.assertEqual(tuple(batch[2].data.shape), (3, 97))
            self.assertEqual(batch[2].batch_sizes.tolist(), [1, 1, 1])
            self.assertAlmostEqual(float(batch[8][0]), 1.25)


if __name__ == "__main__":
    unittest.main()
