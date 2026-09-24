from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDatasetDataModule import (
    RiboUnmixMultiDatasetDataModule,
)
from main_ribounmix_multidataset import (
    export_sequence_only_shared_profiles,
)
from analyses.analyze_real_exp8_stability import _compare_profiles, main as analysis_main
from Tests.test_gamma_centering import _center, _fixed_reference_helper
from Utils.external_transcript_split import load_external_transcript_split
from Utils.real_exp8_stability import (
    build_experiment_matrix,
    build_exp8_transcript_split,
    build_overlap_report,
    prepare_quality_pool,
    select_quality_matched_disjoint_pair,
    uniform_reference_weights,
)
from run_real_exp8_L_stability import (
    _checkpoint_filename_metadata,
    _completed_run,
    parse_args,
)
from Utils.real_panel_convergence import fit_panel_reliability_manifest


def toy_quality(number: int = 20) -> pd.DataFrame:
    index = np.arange(number, dtype=np.float64)
    return pd.DataFrame(
        {
            "dataset_name": [f"d{value:02d}" for value in range(number)],
            "source_identifier": [f"source_{value:02d}" for value in range(number)],
            "eligible": True,
            "median_read_density": np.exp(index / 10.0),
            "log1p_median_read_density": np.log1p(np.exp(index / 10.0)),
            "median_positive_codon_coverage": 0.1 + 0.8 * index / max(number - 1, 1),
            "number_of_eligible_transcripts": 1000 + 10 * index,
            "median_replica_PCC": 0.2 + 0.6 * index / max(number - 1, 1),
        }
    )


class SubsetDesignTests(unittest.TestCase):
    def test_two_gpu_and_numerical_safety_defaults(self) -> None:
        args = parse_args([])
        self.assertEqual(args.gpus, "0,1")
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.max_pair_rows_per_forward, 512)
        self.assertEqual(args.max_padded_codon_tokens_per_forward, 256000)
        self.assertEqual(args.log_every_n_steps, 25)
        self.assertEqual(args.reference_chunk_size, 16)
        self.assertEqual(args.raw_log_gamma_bound, 8.0)

    def test_deterministic_exact_atomic_disjoint_design(self) -> None:
        quality, families, metadata = prepare_quality_pool(toy_quality())
        first, _ = build_experiment_matrix(
            quality_pool=quality,
            family_to_datasets=families,
            dataset_sizes=[2, 5, 10, 20],
            disjoint_pairs=2,
            large_n_subsets=2,
            subset_seed=42,
            candidate_restarts=100,
        )
        second, _ = build_experiment_matrix(
            quality_pool=quality,
            family_to_datasets=families,
            dataset_sizes=[2, 5, 10, 20],
            disjoint_pairs=2,
            large_n_subsets=2,
            subset_seed=42,
            candidate_restarts=100,
        )
        self.assertEqual(first, second)
        self.assertFalse(metadata["legacy_scalar_quality_rank_used_for_subset_selection"])
        self.assertTrue(all(len(task["datasets"]) == task["N"] for task in first))
        overlap = build_overlap_report(first)
        designated = overlap.loc[overlap["is_designated_disjoint_pair"]]
        self.assertTrue(bool((designated["intersection_count"] == 0).all()))
        self.assertTrue(
            bool((designated["source_family_intersection_count"] == 0).all())
        )
        example = designated.iloc[0]
        self.assertEqual(example["union_count"], example["N_a"] + example["N_b"])
        self.assertEqual(example["jaccard"], 0.0)

    def test_multidataset_source_families_remain_atomic(self) -> None:
        table = toy_quality(12)
        table.loc[table.index[:3], "source_identifier"] = "family_three"
        table.loc[table.index[3:5], "source_identifier"] = "family_two"
        quality, families, metadata = prepare_quality_pool(table)
        pair = select_quality_matched_disjoint_pair(
            target_size=5,
            quality_pool=quality,
            family_to_datasets=families,
            z_columns=metadata["z_columns"],
            seed=11,
            restarts=200,
        )
        for side in ("A", "B"):
            selected = set(pair[side]["datasets"])
            for datasets in families.values():
                overlap = selected & set(datasets)
                self.assertIn(len(overlap), (0, len(datasets)))
        self.assertFalse(set(pair["A"]["source_families"]) & set(pair["B"]["source_families"]))

    def test_uniform_pi_is_exact(self) -> None:
        weights = uniform_reference_weights(["a", "b", "c"])
        self.assertEqual(set(weights), {"a", "b", "c"})
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=14)
        self.assertTrue(all(value == 1.0 / 3.0 for value in weights.values()))

    def test_legacy_rank_columns_cannot_change_subset_selection(self) -> None:
        ascending = toy_quality(20)
        ascending["quality_rank"] = np.arange(1, 21)
        descending = ascending.copy()
        descending["quality_rank"] = descending["quality_rank"].iloc[::-1].to_numpy()
        first_quality, first_families, first_metadata = prepare_quality_pool(ascending)
        second_quality, second_families, second_metadata = prepare_quality_pool(descending)
        first, _ = build_experiment_matrix(
            quality_pool=first_quality,
            family_to_datasets=first_families,
            dataset_sizes=[2, 5, 20],
            disjoint_pairs=1,
            large_n_subsets=1,
            subset_seed=42,
            candidate_restarts=100,
        )
        second, _ = build_experiment_matrix(
            quality_pool=second_quality,
            family_to_datasets=second_families,
            dataset_sizes=[2, 5, 20],
            disjoint_pairs=1,
            large_n_subsets=1,
            subset_seed=42,
            candidate_restarts=100,
        )
        self.assertEqual(first, second)
        self.assertFalse(
            first_metadata["legacy_scalar_quality_rank_used_for_subset_selection"]
        )
        self.assertFalse(
            second_metadata["legacy_scalar_quality_rank_used_for_subset_selection"]
        )

    def test_checkpoint_filename_metadata(self) -> None:
        metadata = _checkpoint_filename_metadata(
            Path("val-loss-epoch=17-val_loss=1.2345.ckpt")
        )
        self.assertEqual(metadata["epoch"], 17)
        self.assertAlmostEqual(metadata["validation_loss"], 1.2345)

    def test_resume_requires_both_prediction_and_best_loss_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = root / "L.parquet"
            profile.touch()
            (root / "subset_manifest.json").write_text(
                json.dumps({"design_hash": "abc"}), encoding="utf-8"
            )
            selected = {
                "checkpoint_variant": "best_val_loss",
                "checkpoint_path": str(root / "missing.ckpt"),
                "shared_profile_path": str(profile),
                "test_transcript_id_hash": "test-hash",
            }
            (root / "selected_checkpoint.json").write_text(
                json.dumps(selected), encoding="utf-8"
            )
            self.assertFalse(_completed_run(root, "abc", "test-hash"))
            checkpoint = root / "best.ckpt"
            checkpoint.touch()
            selected["checkpoint_path"] = str(checkpoint)
            (root / "selected_checkpoint.json").write_text(
                json.dumps(selected), encoding="utf-8"
            )
            self.assertTrue(_completed_run(root, "abc", "test-hash"))


class SplitAndReliabilityTests(unittest.TestCase):
    def test_common_test_never_leaks_and_validation_is_run_specific(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript_ids = [f"t{index:02d}" for index in range(30)]
            pd.DataFrame(
                {
                    "transcript_id": transcript_ids,
                    "codons": [["ATG", "AAA", "TAA"] for _ in transcript_ids],
                    "css": [[] for _ in transcript_ids],
                }
            ).to_parquet(root / "sequences.parquet", index=False)
            mapping = {}
            for dataset_index in range(4):
                name = f"d{dataset_index}"
                path = root / f"{name}.parquet"
                pd.DataFrame(
                    {
                        "id": transcript_ids,
                        "weight": np.linspace(0.5, 1.5, len(transcript_ids)),
                        "ribo": [np.asarray([1.0, 0.0, 2.0], dtype=np.float32)]
                        * len(transcript_ids),
                    }
                ).to_parquet(path, index=False)
                mapping[name] = str(path)
            tasks = [
                {
                    "run_id": "pair_A",
                    "datasets": ["d0", "d1"],
                    "N": 2,
                },
                {
                    "run_id": "pair_B",
                    "datasets": ["d2", "d3"],
                    "N": 2,
                },
            ]
            split = build_exp8_transcript_split(
                experiment_name="test",
                tasks=tasks,
                dataset_mapping=mapping,
                sequences_path=root / "sequences.parquet",
                subset_seed=42,
                validation_fraction=0.10,
                test_fraction=0.10,
                reliability_bins=3,
                maximum_cds_codons=None,
            )
            common_test = set(split["common_test_ids"])
            self.assertTrue(common_test)
            for task in tasks:
                run_id = task["run_id"]
                train = set(split["panel_train_eligible_ids"][run_id])
                validation = set(split["panel_validation_ids"][run_id])
                self.assertFalse(train & common_test)
                self.assertFalse(validation & common_test)
                self.assertFalse(train & validation)

            manifest_path = root / "split.json"
            manifest_path.write_text(json.dumps(split), encoding="utf-8")
            train, validation, test, _ = load_external_transcript_split(
                manifest_path,
                panel_name="pair_A",
                experiment_datasets=["d0", "d1"],
            )
            self.assertEqual(validation, split["panel_validation_ids"]["pair_A"])
            self.assertEqual(set(test), common_test)
            self.assertFalse(set(train) & set(test))

    def test_train_only_reliability_reference_excludes_heldout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "d.parquet"
            pd.DataFrame(
                {
                    "id": ["train1", "train2", "validation", "test"],
                    "read_density": [1.0, 3.0, 1000.0, 2000.0],
                    "coverage": [0.2, 0.8, 1.0, 1.0],
                    "weight": [1.0, 1.0, 1.0, 1.0],
                }
            ).to_parquet(path, index=False)
            manifest = fit_panel_reliability_manifest(
                experiment_name="test",
                panel_name="run",
                panel_datasets=["d"],
                dataset_mapping={"d": str(path)},
                panel_training_ids=["train1", "train2"],
                validation_ids=["validation"],
                test_ids=["test"],
                source_split_manifest=root / "split.json",
            )
            self.assertEqual(manifest["heldout_rows_used_for_fitting"], 0)
            self.assertEqual(
                manifest["datasets"]["d"]["depth_reference_tau"], 2.0
            )


class SharedProfileExportTests(unittest.TestCase):
    def test_sequence_only_prediction_loader_keeps_unobserved_test_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence_path = root / "sequences.parquet"
            pd.DataFrame(
                {
                    "transcript_id": ["train", "validation", "test_unobserved"],
                    "codons": [
                        ["ATG", "AAA", "TAA"],
                        ["ATG", "CCC", "TAA"],
                        ["ATG", "GGG", "TAA"],
                    ],
                    "css": [[], [], []],
                }
            ).to_parquet(sequence_path, index=False)
            dataset_paths = []
            for name in ("iwasaki_2014", "sauer_2019"):
                path = root / f"{name}.parquet"
                profiles = [
                    np.asarray([1.0, 0.0, 2.0], dtype=np.float32),
                    np.asarray([0.0, 1.0, 1.0], dtype=np.float32),
                ]
                pd.DataFrame(
                    {
                        "id": ["train", "validation"],
                        "ribo": profiles,
                        "ribo_cds_replicas": [[profile] for profile in profiles],
                        "weight": [1.0, 1.0],
                    }
                ).to_parquet(path, index=False)
                dataset_paths.append(str(path))
            datamodule = RiboUnmixMultiDatasetDataModule(
                sequences_path=str(sequence_path),
                datasets_paths=dataset_paths,
                batch_size=1,
                split=(["train"], ["validation"], ["test_unobserved"]),
                nt_encoding_path="Datasets/encodings/nt_encoding.yaml",
                codon_to_aa_encoding_path="Datasets/encodings/codon2aa.yaml",
                codon_encoding_path="Datasets/encodings/codon_encoding.yaml",
                aa_encoding_path="Datasets/encodings/aa_encoding.yaml",
                datasets_encoding_path="Datasets/encodings/dataset_encoding.yaml",
                num_workers=0,
                predict_num_workers=0,
                train_sampling_strategy="transcript_grouped_multidataset_pairs",
                minimum_positive_datasets_per_transcript=2,
                sequence_only_shared_profile_prediction=True,
            )
            datamodule.setup("fit")
            self.assertEqual(
                datamodule.predict_dataset_obj.flat_transcript_ids.tolist(),
                ["test_unobserved"],
            )
            self.assertEqual(len(datamodule.predict_dataset_obj), 1)
            self.assertEqual(len(datamodule.train_dataset_obj), 2)
            self.assertEqual(len(datamodule.val_dataset_obj), 2)

    def test_sequence_only_export_is_mean_one_and_ordered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction = root / "prediction.parquet"
            pd.DataFrame(
                {
                    "transcript_id": ["b", "a"],
                    "length": [3, 3],
                    "mask": [[True, True, True], [True, True, True]],
                    "L_bio": [
                        np.asarray([0.5, 1.0, 1.5], dtype=np.float32),
                        np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
                    ],
                }
            ).to_parquet(prediction, index=False)
            output = root / "L.parquet"
            count = export_sequence_only_shared_profiles(
                prediction_path=prediction,
                output_path=output,
                expected_transcript_ids=["a", "b"],
                run_id="run",
                dataset_count=2,
                subset_identifier="pair01_A",
            )
            exported = pd.read_parquet(output)
            self.assertEqual(count, 2)
            self.assertEqual(exported["transcript_id"].tolist(), ["a", "b"])
            self.assertTrue(np.allclose(exported["L_mean"], 1.0))

    def test_analysis_requires_identical_position_masks(self) -> None:
        left = {
            "t": {
                "values": np.asarray([1.0, 1.0, 1.0]),
                "mask": np.asarray([True, True, True]),
                "length": 3,
            }
        }
        right = {
            "t": {
                "values": np.asarray([1.0, 1.0, 1.0]),
                "mask": np.asarray([True, False, True]),
                "length": 2,
            }
        }
        with self.assertRaises(ValueError):
            _compare_profiles(left=left, right=right, transcript_ids=["t"])

    def test_fixed_reference_chunk_size_is_numerically_invariant(self) -> None:
        model = _fixed_reference_helper([0, 1], [1.0, 3.0])
        model.gamma_reference_chunk_size = 1
        first = _center(model, [0, 1, 2])["log_gamma"]
        model.gamma_reference_chunk_size = 2
        second = _center(model, [0, 1, 2])["log_gamma"]
        torch.testing.assert_close(first, second)

    def test_analysis_pipeline_writes_primary_and_secondary_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = [
                {
                    "run_id": "A",
                    "N": 2,
                    "kind": "designated_disjoint_pair",
                    "pair_id": "pair01",
                    "side": "A",
                    "training_seed": 42,
                    "datasets": ["d0", "d1"],
                    "source_families": ["s0", "s1"],
                },
                {
                    "run_id": "B",
                    "N": 2,
                    "kind": "designated_disjoint_pair",
                    "pair_id": "pair01",
                    "side": "B",
                    "training_seed": 42,
                    "datasets": ["d2", "d3"],
                    "source_families": ["s2", "s3"],
                },
                {
                    "run_id": "FULL",
                    "N": 4,
                    "kind": "full_collection",
                    "training_seed": 42,
                    "datasets": ["d0", "d1", "d2", "d3"],
                    "source_families": ["s0", "s1", "s2", "s3"],
                },
            ]
            (root / "experiment_manifest.json").write_text(
                json.dumps({"tasks": tasks}), encoding="utf-8"
            )
            transcript_ids = [f"t{index}" for index in range(5)]
            (root / "common_test_manifest.json").write_text(
                json.dumps({"common_test_ids": transcript_ids}), encoding="utf-8"
            )
            base = np.linspace(0.5, 1.5, 12)
            for task_index, task in enumerate(tasks):
                if task["kind"] == "designated_disjoint_pair":
                    directory = root / "N002" / f"pair01_{task['side']}"
                else:
                    directory = root / "N004" / "full"
                directory.mkdir(parents=True)
                rows = []
                for transcript_index, transcript_id in enumerate(transcript_ids):
                    perturbation = 0.015 * task_index * np.sin(
                        np.arange(12) + transcript_index
                    )
                    values = base + perturbation
                    values = values / values.mean()
                    rows.append(
                        {
                            "transcript_id": transcript_id,
                            "transcript_length": 12,
                            "L_t": values.astype(np.float32),
                            "valid_position_mask": np.ones(12, dtype=bool),
                        }
                    )
                profile_path = directory / "L.parquet"
                pd.DataFrame(rows).to_parquet(profile_path, index=False)
                (directory / "selected_checkpoint.json").write_text(
                    json.dumps(
                        {
                            "checkpoint_variant": "best_val_loss",
                            "shared_profile_path": str(profile_path),
                        }
                    ),
                    encoding="utf-8",
                )
            overlap_rows = []
            for left, right in ((tasks[0], tasks[1]), (tasks[0], tasks[2]), (tasks[1], tasks[2])):
                left_set, right_set = set(left["datasets"]), set(right["datasets"])
                intersection = left_set & right_set
                union = left_set | right_set
                designated = left["run_id"] == "A" and right["run_id"] == "B"
                overlap_rows.append(
                    {
                        "run_a": left["run_id"],
                        "run_b": right["run_id"],
                        "N_a": left["N"],
                        "N_b": right["N"],
                        "intersection_count": len(intersection),
                        "union_count": len(union),
                        "overlap_fraction_a": len(intersection) / len(left_set),
                        "overlap_fraction_b": len(intersection) / len(right_set),
                        "jaccard": len(intersection) / len(union),
                        "source_family_intersection_count": 0 if designated else 2,
                        "is_designated_disjoint_pair": designated,
                    }
                )
            pd.DataFrame(overlap_rows).to_csv(root / "overlap_report.csv", index=False)
            pd.DataFrame(
                [
                    {"run_id": "A", "N": 2, "quality_mismatch": 0.2},
                    {"run_id": "B", "N": 2, "quality_mismatch": 0.21},
                    {"run_id": "FULL", "N": 4, "quality_mismatch": 0.001},
                ]
            ).to_csv(root / "subset_quality_report.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "run_id": task["run_id"],
                        "N": task["N"],
                        "diversity_raw": np.nan,
                        "diversity_zscored": np.nan,
                        "fingerprint_table_supplied": False,
                    }
                    for task in tasks
                ]
            ).to_csv(root / "subset_diversity.csv", index=False)
            result = analysis_main(
                [
                    "--run-root",
                    str(root),
                    "--bootstrap-replicates",
                    "10",
                ]
            )
            self.assertEqual(result, 0)
            self.assertTrue(
                (root / "analysis/stability_disjoint_per_transcript.parquet").is_file()
            )
            self.assertTrue(
                (root / "analysis/convergence_to_full_summary.csv").is_file()
            )
            self.assertTrue((root / "analysis/same_N_all_pairs.csv").is_file())
            self.assertTrue(
                (root / "analysis/diversity_pair_stability_response.csv").is_file()
            )
            self.assertTrue((root / "analysis/figures/stability_vs_N.png").is_file())


if __name__ == "__main__":
    unittest.main()
