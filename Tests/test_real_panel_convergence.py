from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDatasetDataModule import (
    RiboUnmixMultiDatasetDataModule,
)
from run_real_independent_panel_convergence import (
    _inspect_weighted_dataset_sources,
    _resolved_panel_config,
    _training_command,
    parse_args,
)
from Utils.real_panel_convergence import (
    build_panel_stored_weight_manifest,
    build_common_transcript_split,
    deterministic_stratified_panel_assignment,
    fit_panel_reliability_manifest,
    panel_dictionary,
    summarize_dataset_quality,
)
from Utils.external_transcript_split import load_external_transcript_split
from Utils.reliability_references import (
    MANIFEST_VERSION,
    apply_dataset_reliability_reference,
    fit_dataset_reliability_reference,
    materialize_observed_pair_statistics,
)


class IndependentPanelAssignmentTests(unittest.TestCase):
    def _quality_table(self, number_of_datasets: int = 114) -> pd.DataFrame:
        index = np.arange(number_of_datasets, dtype=np.float64)
        return pd.DataFrame(
            {
                "dataset_name": [f"dataset_{value:03d}" for value in range(number_of_datasets)],
                "eligible": True,
                "median_read_density": np.exp(index / 18.0) + 0.1,
                "log1p_median_read_density": np.log1p(np.exp(index / 18.0) + 0.1),
                "median_positive_codon_coverage": 0.1 + 0.8 * index / max(number_of_datasets - 1, 1),
                "number_of_eligible_transcripts": 1000 + index.astype(int) * 7,
                "median_replica_PCC": np.where((index.astype(int) % 3) == 0, 0.4 + index / 300.0, np.nan),
            }
        )

    def test_four_panels_are_exact_disjoint_complete_and_deterministic(self) -> None:
        quality = self._quality_table()
        first, _ = deterministic_stratified_panel_assignment(
            quality, number_of_panels=4, seed=42
        )
        second, _ = deterministic_stratified_panel_assignment(
            quality, number_of_panels=4, seed=42
        )
        panels = panel_dictionary(first)
        self.assertEqual([len(panels[name]) for name in sorted(panels)], [29, 29, 28, 28])
        flattened = [dataset for values in panels.values() for dataset in values]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), set(quality["dataset_name"]))
        pd.testing.assert_frame_equal(
            first[["dataset_name", "panel"]], second[["dataset_name", "panel"]]
        )

    def test_source_families_are_atomic_as_well_as_datasets(self) -> None:
        quality = self._quality_table(number_of_datasets=28)
        sources = (
            ["large_source"] * 4
            + ["source_b"] * 3
            + ["source_c"] * 3
            + ["source_d"] * 2
            + ["source_e"] * 2
            + [f"singleton_{index:02d}" for index in range(14)]
        )
        quality["source_identifier"] = sources
        assignment, method = deterministic_stratified_panel_assignment(
            quality, number_of_panels=4, seed=42
        )
        self.assertEqual(
            assignment.groupby("panel").size().sort_index().tolist(),
            [7, 7, 7, 7],
        )
        self.assertTrue(
            bool(
                (
                    assignment.groupby("source_identifier")["panel"].nunique()
                    == 1
                ).all()
            )
        )
        self.assertTrue(method["source_groups_are_atomic"])
        self.assertEqual(method["number_of_source_groups"], 19)

    def test_train_only_reliability_is_the_default(self) -> None:
        self.assertEqual(
            parse_args([]).reliability_weight_mode,
            "train-only-snr",
        )

    def test_two_gpu_and_numerical_safety_defaults(self) -> None:
        args = parse_args([])
        self.assertEqual(args.gpus, "0,1")
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.max_pair_rows_per_forward, 512)
        self.assertEqual(args.max_padded_codon_tokens_per_forward, 256000)
        self.assertEqual(args.log_every_n_steps, 25)
        self.assertEqual(args.reference_chunk_size, 16)
        self.assertEqual(args.raw_log_gamma_bound, 8.0)

    def test_custom_sequence_path_reaches_resolved_config_and_hydra_command(self) -> None:
        args = parse_args([])
        sequence_path = Path("/tmp/custom_sequences.parquet")
        panel_directory = Path("/tmp/panel_01")
        resolved = _resolved_panel_config(
            base={},
            dataset_config={},
            run_name="test_panel",
            panel_name="panel_01",
            panel_datasets=["source_2020_a", "source_2021_b"],
            seed=42,
            split_manifest=Path("/tmp/split.json"),
            reliability_manifest=Path("/tmp/reliability.json"),
            sequences_path=sequence_path,
            panel_directory=panel_directory,
            args=args,
        )
        self.assertEqual(
            resolved["paths"]["sequences_path"], str(sequence_path)
        )
        self.assertEqual(
            resolved["model"]["gamma_centering"]["reference"]["chunk_size"],
            16,
        )
        self.assertEqual(resolved["data"]["batch_size"], 32)
        self.assertEqual(
            resolved["training"]["execution_microbatching"][
                "max_pair_rows_per_forward"
            ],
            512,
        )
        self.assertEqual(
            resolved["training"]["execution_microbatching"][
                "max_padded_codon_tokens_per_forward"
            ],
            256000,
        )
        self.assertEqual(
            resolved["model"]["dataset_bias_params"]["raw_log_gamma_bound"],
            8.0,
        )
        command = _training_command(
            args=args,
            run_name="test_panel",
            panel_name="panel_01",
            panel_datasets=["source_2020_a", "source_2021_b"],
            split_manifest=Path("/tmp/split.json"),
            reliability_manifest=Path("/tmp/reliability.json"),
            sequences_path=sequence_path,
            panel_directory=panel_directory,
        )
        self.assertIn(f"paths.sequences_path={sequence_path}", command)
        self.assertIn("data.batch_size=32", command)
        self.assertIn("model.gamma_centering.reference.chunk_size=16", command)
        self.assertIn(
            "training.execution_microbatching.max_pair_rows_per_forward=512",
            command,
        )
        self.assertIn(
            "training.execution_microbatching.max_padded_codon_tokens_per_forward=256000",
            command,
        )
        self.assertIn("model.dataset_bias_params.raw_log_gamma_bound=8.0", command)


class WeightedDatasetSourceContractTests(unittest.TestCase):
    def test_preflight_accepts_weighted_and_rejects_raw_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = np.asarray([1.0, 0.0], dtype=np.float32)
            zero_profile = np.asarray([0.0, 0.0], dtype=np.float32)
            weighted_path = root / "weighted.parquet"
            pd.DataFrame(
                {
                    "id": ["t1", "legacy_excluded"],
                    "ribo": [profile, zero_profile],
                    "ribo_cds_replicas": [[profile], [zero_profile]],
                    "weight": [1.0, 0.0],
                }
            ).to_parquet(weighted_path, index=False)
            valid = _inspect_weighted_dataset_sources(
                {"weighted": str(weighted_path)},
                dataset_config_path=root / "dataset.yaml",
            )
            self.assertEqual(valid["status"], "PASS")
            self.assertEqual(valid["passing_dataset_count"], 1)
            self.assertEqual(
                valid["datasets"][0]["schema_variant"],
                "compact_legacy_weighted",
            )
            self.assertEqual(
                valid["datasets"][0]["zero_weight_exclusion_row_count"], 1
            )

            raw_path = root / "raw.parquet"
            pd.DataFrame(
                {
                    "id": ["t1"],
                    "ribo": [profile],
                    "ribo_cds_replicas": [[profile]],
                }
            ).to_parquet(raw_path, index=False)
            invalid = _inspect_weighted_dataset_sources(
                {"raw": str(raw_path)},
                dataset_config_path=root / "dataset.yaml",
            )
            self.assertEqual(invalid["status"], "FAIL")
            self.assertEqual(
                invalid["failures"][0]["missing_required_columns"],
                ["weight"],
            )


class CommonPanelSplitTests(unittest.TestCase):
    def test_common_heldout_ids_and_panel_training_folds_have_zero_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common_ids = [f"common_{index:02d}" for index in range(40)]
            panel_only_ids = [f"panel_only_{index}" for index in range(1, 5)]
            all_ids = common_ids + panel_only_ids + [
                "legacy_zero_weight",
                "legacy_positive_weight_zero_profile",
            ]
            sequence_path = root / "sequences.parquet"
            pd.DataFrame(
                {
                    "transcript_id": all_ids,
                    "codons": [["ATG", "AAA", "TAA"] for _ in all_ids],
                    "css": [([] if index % 2 == 0 else [1]) for index in range(len(all_ids))],
                }
            ).to_parquet(sequence_path, index=False)

            mapping: dict[str, str] = {}
            panels: dict[str, list[str]] = {}
            for panel_index in range(1, 5):
                panel_name = f"panel_{panel_index:02d}"
                names = [f"p{panel_index}_a", f"p{panel_index}_b"]
                panels[panel_name] = names
                ids = common_ids + [
                    f"panel_only_{panel_index}",
                    "legacy_zero_weight",
                    "legacy_positive_weight_zero_profile",
                ]
                for dataset_offset, dataset_name in enumerate(names):
                    dataset_path = root / f"{dataset_name}.parquet"
                    pd.DataFrame(
                        {
                            "id": ids,
                            "weight": np.concatenate(
                                [
                                    np.linspace(
                                        0.5 + 0.01 * dataset_offset,
                                        1.5 + 0.01 * dataset_offset,
                                        len(ids) - 2,
                                        dtype=np.float32,
                                    ),
                                    np.asarray([0.0, 0.25], dtype=np.float32),
                                ]
                            ),
                            "ribo": [
                                (
                                    np.asarray([1.0, 0.0, 1.0], dtype=np.float32)
                                    if transcript_id
                                    not in {
                                        "legacy_zero_weight",
                                        "legacy_positive_weight_zero_profile",
                                    }
                                    else np.zeros(3, dtype=np.float32)
                                )
                                for transcript_id in ids
                            ],
                        }
                    ).to_parquet(dataset_path, index=False)
                    mapping[dataset_name] = str(dataset_path)

            manifest = build_common_transcript_split(
                experiment_name="test",
                dataset_mapping=mapping,
                panels=panels,
                sequences_path=sequence_path,
                seed=17,
                validation_fraction=0.10,
                test_fraction=0.10,
                reliability_bins=5,
                minimum_panel_support=2,
            )
            manifest_path = root / "common_split_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded_train, loaded_validation, loaded_test, _ = (
                load_external_transcript_split(
                    manifest_path,
                    panel_name="panel_01",
                    experiment_datasets=panels["panel_01"],
                )
            )
            self.assertEqual(
                loaded_train, manifest["panel_train_eligible_ids"]["panel_01"]
            )
            self.assertEqual(loaded_validation, manifest["common_validation_ids"])
            self.assertEqual(loaded_test, manifest["common_test_ids"])
            validation = set(manifest["common_validation_ids"])
            test = set(manifest["common_test_ids"])
            self.assertEqual(len(validation), 4)
            self.assertEqual(len(test), 4)
            self.assertTrue(validation.isdisjoint(test))
            self.assertTrue(validation.issubset(common_ids))
            self.assertTrue(test.issubset(common_ids))
            self.assertNotIn(
                "legacy_zero_weight", manifest["common_evaluation_ids"]
            )
            self.assertNotIn(
                "legacy_positive_weight_zero_profile",
                manifest["common_evaluation_ids"],
            )
            for panel_name, train_ids in manifest["panel_train_eligible_ids"].items():
                train = set(train_ids)
                self.assertTrue(train.isdisjoint(validation | test), panel_name)
                self.assertIn(
                    f"panel_only_{int(panel_name.removeprefix('panel_'))}", train
                )
                support = manifest["panel_support_statistics"][panel_name]["train_support"]
                self.assertGreaterEqual(support["minimum"], 2)


class TrainOnlyReliabilityReferenceTests(unittest.TestCase):
    def test_legacy_observed_statistics_are_materialized_without_changing_weights(self) -> None:
        frame = pd.DataFrame(
            {
                "id": ["t1", "t2"],
                "ribo": [
                    np.asarray([1.0, 0.0, 2.0], dtype=np.float32),
                    np.asarray([0.0, 3.0, 1.0], dtype=np.float32),
                ],
                "weight": [1.25, 0.75],
            }
        )
        result = materialize_observed_pair_statistics(
            frame, dataset_name="legacy_dataset"
        )
        np.testing.assert_allclose(result["read_density"], [1.0, 4.0 / 3.0])
        np.testing.assert_allclose(result["coverage"], [2.0 / 3.0, 2.0 / 3.0])
        np.testing.assert_allclose(result["weight"], frame["weight"])

    def test_production_datamodule_uses_compact_stored_weights_and_skips_zero(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript_ids = [
                "train",
                "validation",
                "test",
                "legacy_zero_weight",
                "legacy_positive_weight_zero_profile",
            ]
            sequence_path = root / "sequences.parquet"
            pd.DataFrame(
                {
                    "transcript_id": transcript_ids,
                    "codons": [["ATG", "AAA", "TAA"] for _ in transcript_ids],
                    "css": [[] for _ in transcript_ids],
                }
            ).to_parquet(sequence_path, index=False)
            profiles = [
                np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
                np.asarray([2.0, 1.0, 0.0], dtype=np.float32),
                np.asarray([0.0, 3.0, 1.0], dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
            ]
            dataset_path = root / "martinez_2019.parquet"
            pd.DataFrame(
                {
                    "id": transcript_ids,
                    "ribo": profiles,
                    "ribo_cds_replicas": [[profile] for profile in profiles],
                    "weight": [0.75, 1.25, 1.5, 0.0, 0.25],
                }
            ).to_parquet(dataset_path, index=False)
            datamodule = RiboUnmixMultiDatasetDataModule(
                sequences_path=str(sequence_path),
                datasets_paths=[str(dataset_path)],
                batch_size=1,
                split=(["train"], ["validation"], ["test"]),
                nt_encoding_path=str(project_root / "Datasets/encodings/nt_encoding.yaml"),
                codon_to_aa_encoding_path=str(project_root / "Datasets/encodings/codon2aa.yaml"),
                codon_encoding_path=str(project_root / "Datasets/encodings/codon_encoding.yaml"),
                aa_encoding_path=str(project_root / "Datasets/encodings/aa_encoding.yaml"),
                datasets_encoding_path=str(project_root / "Datasets/encodings/dataset_encoding.yaml"),
                num_workers=0,
                predict_num_workers=0,
                train_sampling_strategy="transcript_grouped_pairs",
                minimum_positive_datasets_per_transcript=1,
                dataset_quality_ranking_path=None,
                reliability_reference_manifest_path=None,
            )
            datamodule.setup("fit")
            self.assertEqual(datamodule.train_flat_transcript_ids.tolist(), ["train"])
            self.assertEqual(datamodule.val_flat_transcript_ids.tolist(), ["validation"])
            self.assertEqual(datamodule.predict_flat_transcript_ids.tolist(), ["test"])
            self.assertAlmostEqual(
                float(datamodule.train_dataset_obj.flat_sample_weights[0]), 0.75
            )

    def test_production_datamodule_rejects_unfiltered_schema(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript_ids = ["train", "validation", "test"]
            sequence_path = root / "sequences.parquet"
            pd.DataFrame(
                {
                    "transcript_id": transcript_ids,
                    "codons": [["ATG", "AAA", "TAA"] for _ in transcript_ids],
                    "css": [[] for _ in transcript_ids],
                }
            ).to_parquet(sequence_path, index=False)
            profiles = [
                np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
                np.asarray([2.0, 1.0, 0.0], dtype=np.float32),
                np.asarray([0.0, 3.0, 1.0], dtype=np.float32),
            ]
            dataset_path = root / "martinez_2019.parquet"
            profile_frame = pd.DataFrame(
                {
                    "id": transcript_ids,
                    "ribo": profiles,
                    "ribo_cds_replicas": [[profile] for profile in profiles],
                    "weight": np.ones(3, dtype=np.float32),
                }
            )
            profile_frame.to_parquet(dataset_path, index=False)
            reference = fit_dataset_reliability_reference(
                pd.DataFrame(
                    {
                        "id": transcript_ids,
                        "read_density": [2.0 / 3.0, 1.0, 4.0 / 3.0],
                        "coverage": [2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
                    }
                ),
                dataset_name="martinez_2019",
                training_transcript_ids=["train"],
            )
            reliability_path = root / "reliability.json"
            reliability_path.write_text(
                json.dumps(
                    {
                        "manifest_version": MANIFEST_VERSION,
                        "datasets": {"martinez_2019": reference},
                    }
                ),
                encoding="utf-8",
            )

            datamodule = RiboUnmixMultiDatasetDataModule(
                sequences_path=str(sequence_path),
                datasets_paths=[str(dataset_path)],
                batch_size=1,
                split=(["train"], ["validation"], ["test"]),
                nt_encoding_path=str(project_root / "Datasets/encodings/nt_encoding.yaml"),
                codon_to_aa_encoding_path=str(project_root / "Datasets/encodings/codon2aa.yaml"),
                codon_encoding_path=str(project_root / "Datasets/encodings/codon_encoding.yaml"),
                aa_encoding_path=str(project_root / "Datasets/encodings/aa_encoding.yaml"),
                datasets_encoding_path=str(project_root / "Datasets/encodings/dataset_encoding.yaml"),
                num_workers=0,
                predict_num_workers=0,
                train_sampling_strategy="transcript_grouped_pairs",
                minimum_positive_datasets_per_transcript=1,
                dataset_quality_ranking_path=None,
                reliability_reference_manifest_path=str(reliability_path),
            )
            with self.assertRaisesRegex(KeyError, "Raw profile artifacts are not accepted"):
                datamodule.setup("fit")

    def test_reliability_reference_rejects_missing_weighted_statistics(self) -> None:
        profile_only = pd.DataFrame(
            {
                "id": ["train_a", "train_b"],
                "ribo": [
                    np.asarray([1.0, 0.0], dtype=np.float32),
                    np.asarray([2.0, 2.0], dtype=np.float32),
                ],
            }
        )
        with self.assertRaisesRegex(KeyError, "raw profiles are not an accepted fallback"):
            fit_dataset_reliability_reference(
                profile_only,
                dataset_name="dataset",
                training_transcript_ids=["train_a", "train_b"],
            )

    def test_quality_accepts_compact_weighted_schema_and_preserves_stored_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = root / "legacy_weighted.parquet"
            profiles = [
                np.asarray([0.0, 2.0, 0.0, 2.0], dtype=np.float32),
                np.asarray([3.0, 3.0], dtype=np.float32),
                np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
                np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            ]
            pd.DataFrame(
                {
                    "id": [
                        "train_a",
                        "train_b",
                        "heldout_zero_weight",
                        "heldout_positive_weight_zero_profile",
                    ],
                    "ribo": profiles,
                    "ribo_cds_replicas": [[profile] for profile in profiles],
                    "weight": np.asarray(
                        [0.8, 1.2, 0.0, 0.25], dtype=np.float32
                    ),
                }
            ).to_parquet(dataset_path, index=False)

            summary = summarize_dataset_quality(
                dataset_name="legacy_weighted",
                dataset_path=dataset_path,
                eligible_sequence_ids={
                    "train_a",
                    "train_b",
                    "heldout_zero_weight",
                    "heldout_positive_weight_zero_profile",
                },
            )
            self.assertEqual(summary["number_of_eligible_transcripts"], 2)
            self.assertEqual(
                summary["zero_weight_exclusion_rows_in_weighted_artifact"], 1
            )
            self.assertEqual(
                summary["positive_weight_zero_information_rows_excluded"], 1
            )
            self.assertEqual(
                summary["weighted_artifact_schema_variant"],
                "compact_legacy_weighted",
            )
            stored_manifest = build_panel_stored_weight_manifest(
                experiment_name="test",
                panel_name="panel_01",
                panel_datasets=["legacy_weighted"],
                dataset_mapping={"legacy_weighted": str(dataset_path)},
                panel_training_ids=["train_a", "train_b"],
                validation_ids=[
                    "heldout_zero_weight",
                    "heldout_positive_weight_zero_profile",
                ],
                test_ids=[],
                source_split_manifest=root / "split.json",
            )
            self.assertEqual(stored_manifest["weight_mode"], "stored")
            self.assertEqual(
                stored_manifest["datasets"]["legacy_weighted"][
                    "zero_weight_exclusion_row_count"
                ],
                1,
            )
            self.assertEqual(
                stored_manifest["datasets"]["legacy_weighted"][
                    "positive_weight_zero_information_exclusion_row_count"
                ],
                1,
            )
            with self.assertRaisesRegex(KeyError, "raw/unfiltered"):
                fit_panel_reliability_manifest(
                    experiment_name="test",
                    panel_name="panel_01",
                    panel_datasets=["legacy_weighted"],
                    dataset_mapping={"legacy_weighted": str(dataset_path)},
                    panel_training_ids=["train_a", "train_b"],
                    validation_ids=["heldout_zero_weight"],
                    test_ids=[],
                    source_split_manifest=root / "split.json",
                )

    def test_heldout_rows_do_not_influence_fitted_references(self) -> None:
        base = pd.DataFrame(
            {
                "id": ["train_a", "train_b", "train_c", "validation", "test"],
                "read_density": [1.0, 4.0, 9.0, 16.0, 25.0],
                "coverage": [0.2, 0.4, 0.8, 0.5, 0.7],
            }
        )
        changed_heldout = base.copy()
        changed_heldout.loc[changed_heldout["id"] == "validation", "read_density"] = 1.0e9
        changed_heldout.loc[changed_heldout["id"] == "validation", "coverage"] = 0.99
        changed_heldout.loc[changed_heldout["id"] == "test", "read_density"] = 1.0e-6
        changed_heldout.loc[changed_heldout["id"] == "test", "coverage"] = 0.01
        training_ids = ["train_a", "train_b", "train_c"]
        first = fit_dataset_reliability_reference(
            base, dataset_name="dataset", training_transcript_ids=training_ids
        )
        second = fit_dataset_reliability_reference(
            changed_heldout,
            dataset_name="dataset",
            training_transcript_ids=training_ids,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["reference_split"], "training_only")
        self.assertEqual(first["training_reference_transcript_count"], 3)

        original_weights = apply_dataset_reliability_reference(
            base, dataset_name="dataset", reference=first
        )
        changed_weights = apply_dataset_reliability_reference(
            changed_heldout, dataset_name="dataset", reference=first
        )
        np.testing.assert_allclose(original_weights[:3], changed_weights[:3])
        self.assertFalse(np.allclose(original_weights[3:], changed_weights[3:]))


if __name__ == "__main__":
    unittest.main()
