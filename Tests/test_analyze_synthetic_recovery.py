import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analyses.analyze_synthetic_recovery import (
    REPOSITORY_ROOT,
    _encoding_from_config,
    _find_prediction,
    cds_interior_mask,
    discover_run_directories,
    profile_recovery_metrics,
    synthetic_mass_conservation,
)
from analyses.analyze_synthetic_inter_artificial_bias import (
    build_parser as build_inter_bias_parser,
)
from analyses.analyze_synthetic_inter_shared_signal import (
    _resolve_matching_merge_column,
    annotate_selected_attempts,
    build_paired_ranking_comparison,
    build_parser as build_inter_signal_parser,
)
from analyses.compare_synthetic_gamma_results import build_matched_comparison
from analyses.analyze_synthetic_gamma_recovery import (
    gamma_vector_metrics,
    joint_log_gamma_gauge,
    strong_site_domain_diagnostics,
)


class SyntheticRecoveryMetricTests(unittest.TestCase):
    def test_inter_analyses_default_to_best_validation_loss(self) -> None:
        self.assertEqual(
            build_inter_signal_parser().parse_args([]).selection_metric,
            "val_loss",
        )
        self.assertEqual(
            build_inter_bias_parser().parse_args([]).selection_metric,
            "val_loss",
        )

    def test_retry_is_not_counted_as_an_independent_replicate(self) -> None:
        attempts = pd.DataFrame(
            [
                {
                    "run": "experiment_gammaequal_retry1_37",
                    "prediction_available": True,
                },
                {
                    "run": "experiment_gammaequal_37",
                    "prediction_available": False,
                },
            ]
        )
        annotated = annotate_selected_attempts(
            attempts, availability_column="prediction_available"
        )
        selected = annotated.loc[annotated["selected_attempt"], "run"].tolist()
        self.assertEqual(selected, ["experiment_gammaequal_retry1_37"])
        self.assertEqual(annotated["logical_run"].nunique(), 1)

    def test_outer_merge_accepts_equivalent_integer_and_float_metadata(self) -> None:
        merged = pd.DataFrame(
            {
                "run": ["complete", "setup_only"],
                "dataset_count_x": [30.0, np.nan],
                "dataset_count_y": [30, 21],
            }
        )
        _resolve_matching_merge_column(merged, "dataset_count")
        self.assertEqual(merged["dataset_count"].tolist(), [30.0, 21.0])
        self.assertNotIn("dataset_count_x", merged)
        self.assertNotIn("dataset_count_y", merged)

    def test_ranking_comparison_requires_a_matched_validation_panel(self) -> None:
        common = {
            "mass_condition": "mass_free",
            "depth": "cross_depth",
            "datasets": "A,B,C",
            "seed": 42,
            "prediction_checkpoint_variant": "best_val_loss",
            "selected_attempt": True,
        }
        runs = pd.DataFrame(
            [
                {
                    **common,
                    "run": "equal_3",
                    "dataset_count": 3,
                    "gamma_reference_weighting": "equal",
                    "artifact_available": True,
                    "validation_id_hash": "same",
                    "pcc": 0.8,
                    "rmse": 0.3,
                },
                {
                    **common,
                    "run": "quality_3",
                    "dataset_count": 3,
                    "gamma_reference_weighting": "quality_rank",
                    "artifact_available": True,
                    "validation_id_hash": "same",
                    "pcc": 0.9,
                    "rmse": 0.2,
                },
                {
                    **common,
                    "run": "equal_6",
                    "dataset_count": 6,
                    "gamma_reference_weighting": "equal",
                    "artifact_available": True,
                    "validation_id_hash": "first",
                    "pcc": 0.7,
                    "rmse": 0.4,
                },
                {
                    **common,
                    "run": "quality_6",
                    "dataset_count": 6,
                    "gamma_reference_weighting": "quality_rank",
                    "artifact_available": True,
                    "validation_id_hash": "different",
                    "pcc": 0.95,
                    "rmse": 0.1,
                },
                {
                    **common,
                    "run": "equal_9",
                    "dataset_count": 9,
                    "gamma_reference_weighting": "equal",
                    "artifact_available": True,
                    "validation_id_hash": "same",
                    "pcc": np.nan,
                    "rmse": 0.25,
                },
                {
                    **common,
                    "run": "quality_9",
                    "dataset_count": 9,
                    "gamma_reference_weighting": "quality_rank",
                    "artifact_available": True,
                    "validation_id_hash": "same",
                    "pcc": 0.0,
                    "rmse": 0.20,
                },
            ]
        )
        paired = build_paired_ranking_comparison(
            runs,
            metric_columns=("pcc", "rmse"),
            availability_column="artifact_available",
            checkpoint_column="prediction_checkpoint_variant",
        ).set_index("dataset_count")
        self.assertTrue(bool(paired.loc[3, "pair_comparable"]))
        self.assertTrue(bool(paired.loc[3, "pcc_pair_comparable"]))
        self.assertAlmostEqual(
            paired.loc[3, "pcc_quality_rank_minus_equal"], 0.1
        )
        self.assertAlmostEqual(
            paired.loc[3, "rmse_quality_rank_minus_equal"], -0.1
        )
        self.assertFalse(bool(paired.loc[6, "pair_comparable"]))
        self.assertFalse(bool(paired.loc[6, "pcc_pair_comparable"]))
        self.assertTrue(
            math.isnan(paired.loc[6, "pcc_quality_rank_minus_equal"])
        )
        self.assertTrue(bool(paired.loc[9, "pair_comparable"]))
        self.assertFalse(bool(paired.loc[9, "pcc_pair_comparable"]))
        self.assertTrue(bool(paired.loc[9, "rmse_pair_comparable"]))
        self.assertTrue(
            math.isnan(paired.loc[9, "pcc_quality_rank_minus_equal"])
        )
        self.assertAlmostEqual(
            paired.loc[9, "rmse_quality_rank_minus_equal"], -0.05
        )

    def test_cross_depth_encoding_is_reconstructed_from_resolved_universe(self) -> None:
        config = {
            "paths": {
                "encodings": {
                    "datasets": "/deleted/synthetic_cross_depth_dataset_encoding.yaml"
                }
            },
            "split": {
                "master_dataset_universe": [
                    "artificial_bias_3prime_aa_0p25_per_codon",
                    "artificial_bias_3prime_aa_2_per_codon",
                    "artificial_bias_3prime_aa_20_per_codon",
                    "artificial_bias_3prime_cc_0p25_per_codon",
                ]
            },
        }
        self.assertEqual(
            _encoding_from_config(config, REPOSITORY_ROOT),
            {
                0: "artificial_bias_3prime_aa_0p25_per_codon",
                1: "artificial_bias_3prime_aa_2_per_codon",
                2: "artificial_bias_3prime_aa_20_per_codon",
                3: "artificial_bias_3prime_cc_0p25_per_codon",
            },
        )

    def test_prediction_selection_distinguishes_checkpoint_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run = Path(temporary_directory)
            best_pcc = run / "predictions_main_val_best_pcc_data.parquet"
            best_loss = run / "predictions_main_val_best_val_loss_data.parquet"
            best_pcc.touch()
            best_loss.touch()
            self.assertEqual(_find_prediction(run, "best_pcc"), best_pcc)
            self.assertEqual(_find_prediction(run, "best_val_loss"), best_loss)

    def test_mass_conservation_comes_from_resolved_config(self) -> None:
        self.assertEqual(
            synthetic_mass_conservation(
                {"model": {"mass_conservation": False}},
                "run_without_massfree_tag",
            ),
            (False, "mass_free"),
        )
        self.assertEqual(
            synthetic_mass_conservation(
                {"model": {"mass_conservation": True}},
                "misleading_massfree_tag",
            ),
            (True, "mass_conserved"),
        )

    def test_massfree_run_name_is_a_legacy_fallback(self) -> None:
        self.assertEqual(
            synthetic_mass_conservation({}, "experiment_massfree_123"),
            (False, "mass_free"),
        )

    def test_run_discovery_uses_requested_prefix_and_immediate_children(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            expected = root / "riboai_synthetic_within_panel2"
            expected.mkdir()
            (root / "riboai_synthetic_inter_bias").mkdir()
            (root / "recovery_report").mkdir()
            nested = root / "past_results" / "riboai_synthetic_within_old"
            nested.mkdir(parents=True)
            self.assertEqual(
                discover_run_directories(root, "riboai_synthetic_within_"),
                [expected],
            )

    def test_run_discovery_rejects_empty_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "non-empty"):
                discover_run_directories(Path(temporary_directory), "")

    def test_scale_is_removed_before_shape_comparison(self) -> None:
        truth = np.array([0.5, 1.0, 1.5], dtype=np.float64)
        metrics = profile_recovery_metrics(20.0 * truth, truth)
        self.assertAlmostEqual(metrics.pcc, 1.0, places=12)
        self.assertAlmostEqual(metrics.mse, 0.0, places=12)
        self.assertAlmostEqual(metrics.rmse, 0.0, places=12)
        self.assertAlmostEqual(metrics.mae, 0.0, places=12)

    def test_error_metrics_have_expected_values(self) -> None:
        truth = np.array([0.5, 1.0, 1.5], dtype=np.float64)
        prediction = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        metrics = profile_recovery_metrics(prediction, truth)
        expected_mse = (0.5**2 + 0.0**2 + (-0.5) ** 2) / 3.0
        self.assertTrue(math.isnan(metrics.pcc))
        self.assertAlmostEqual(metrics.mse, expected_mse, places=12)
        self.assertAlmostEqual(metrics.rmse, math.sqrt(expected_mse), places=12)
        self.assertAlmostEqual(metrics.mae, 1.0 / 3.0, places=12)

    def test_invalid_negative_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "negative"):
            profile_recovery_metrics([1.0, -1.0], [1.0, 1.0])

    def test_interior_mask_excludes_exactly_five_codons_per_side(self) -> None:
        mask = cds_interior_mask(20)
        np.testing.assert_array_equal(
            np.flatnonzero(mask), np.arange(5, 15, dtype=np.int64)
        )
        self.assertFalse(bool(mask[:5].any()))
        self.assertFalse(bool(mask[-5:].any()))
        self.assertTrue(bool(mask[5:-5].all()))

    def test_interior_mask_handles_short_transcripts_without_leakage(self) -> None:
        for length in (0, 1, 9, 10):
            self.assertEqual(int(cds_interior_mask(length).sum()), 0)
        np.testing.assert_array_equal(np.flatnonzero(cds_interior_mask(11)), [5])

    def test_interior_profile_metrics_keep_historical_full_normalization(self) -> None:
        truth = np.arange(1.0, 13.0)
        prediction = truth.copy()
        prediction[:5] *= 10.0
        mask = cds_interior_mask(truth.size)
        metrics = profile_recovery_metrics(
            prediction, truth, position_mask=mask
        )
        pred_full_normalized = prediction / prediction.mean()
        truth_full_normalized = truth / truth.mean()
        expected = np.mean(
            (pred_full_normalized[mask] - truth_full_normalized[mask]) ** 2
        )
        self.assertAlmostEqual(metrics.mse, expected, places=12)

    def test_empty_interior_profile_domain_returns_nan_metrics(self) -> None:
        metrics = profile_recovery_metrics(
            np.ones(10), np.ones(10), position_mask=cds_interior_mask(10)
        )
        self.assertTrue(math.isnan(metrics.pcc))
        self.assertTrue(math.isnan(metrics.rmse))


class SyntheticGammaRecoveryTests(unittest.TestCase):
    def test_joint_gauge_satisfies_both_constraints(self) -> None:
        raw = np.array(
            [[0.0, 1.0, 0.2], [0.5, -0.2, 0.9], [-0.3, 0.4, 0.1]],
            dtype=np.float64,
        )
        weights = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        centered = joint_log_gamma_gauge(raw, weights)
        pi = weights / weights.sum()
        np.testing.assert_allclose((pi[:, None] * centered).sum(axis=0), 0.0, atol=1e-14)
        np.testing.assert_allclose(centered.mean(axis=1), 0.0, atol=1e-14)

    def test_joint_gauge_removes_unidentifiable_components(self) -> None:
        raw = np.array(
            [[0.0, 1.0, 0.2], [0.5, -0.2, 0.9], [-0.3, 0.4, 0.1]],
            dtype=np.float64,
        )
        dataset_constants = np.array([2.0, -1.0, 0.4])[:, None]
        shared_position_profile = np.array([0.3, -0.7, 1.2])[None, :]
        expected = joint_log_gamma_gauge(raw)
        actual = joint_log_gamma_gauge(
            raw + dataset_constants + shared_position_profile
        )
        np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_joint_gauge_uses_the_identical_interior_coordinates(self) -> None:
        raw = np.arange(36, dtype=np.float64).reshape(3, 12) / 10.0
        raw[1] = raw[1, ::-1]
        mask = cds_interior_mask(raw.shape[1])
        from_mask = joint_log_gamma_gauge(raw, [1.0, 2.0, 3.0], mask)
        from_slice = joint_log_gamma_gauge(raw[:, mask], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(from_mask, from_slice, atol=1e-14)
        self.assertEqual(from_mask.shape[1], 2)
        with self.assertRaisesRegex(ValueError, "position mask"):
            joint_log_gamma_gauge(raw, position_mask=np.ones(11, dtype=bool))

    def test_terminal_stop_exclusion_and_boundary_trim_share_physical_axis(self) -> None:
        # A length-13 saved profile already ends at modeled P-site 12; there is
        # no appended terminal stop coordinate for the analysis to remove.
        np.testing.assert_array_equal(
            np.flatnonzero(cds_interior_mask(13)), np.asarray([5, 6, 7])
        )

    def test_boundary_misses_are_excluded_but_reported(self) -> None:
        mask = cds_interior_mask(12)
        truth_full = np.zeros(12)
        predicted_full = np.zeros(12)
        truth_full[[0, 5]] = 1.2
        predicted_full[[0, 5]] = 0.2
        diagnostics = strong_site_domain_diagnostics(
            interior_true=truth_full[mask],
            interior_predicted=predicted_full[mask],
            full_true=truth_full,
            full_predicted=predicted_full,
            full_interior_mask=mask,
        )
        self.assertEqual(diagnostics["n_interior_strong_sites"], 1)
        self.assertEqual(diagnostics["n_boundary_strong_sites"], 1)
        self.assertAlmostEqual(
            diagnostics["interior_strong_site_miss_rate"], 1.0
        )
        self.assertAlmostEqual(
            diagnostics["boundary_strong_site_miss_rate"], 1.0
        )
        self.assertAlmostEqual(
            diagnostics["fraction_of_all_misses_at_boundary"], 0.5
        )

    def test_exact_gamma_recovery_metrics(self) -> None:
        truth = np.array([-0.5, 0.0, 0.5], dtype=np.float64)
        metrics = gamma_vector_metrics(truth, truth)
        self.assertAlmostEqual(metrics.pcc, 1.0, places=12)
        self.assertAlmostEqual(metrics.rmse, 0.0, places=12)
        self.assertAlmostEqual(metrics.mae, 0.0, places=12)
        self.assertAlmostEqual(metrics.calibration_slope, 1.0, places=12)
        self.assertAlmostEqual(metrics.mean_absolute_relative_error, 0.0, places=12)
        self.assertAlmostEqual(metrics.max_absolute_log_error, 0.0, places=12)
        self.assertAlmostEqual(metrics.fraction_within_1pct, 1.0, places=12)
        self.assertAlmostEqual(metrics.fraction_within_5pct, 1.0, places=12)
        self.assertAlmostEqual(metrics.fraction_within_10pct, 1.0, places=12)

    def test_root_comparison_matches_only_identical_validation_panels(self) -> None:
        common = {
            "checkpoint_variant": "best_pcc",
            "depth": "0p25_per_codon",
            "mass_condition": "mass_conserved",
            "dataset_count": 2,
            "datasets": "A,B",
        }
        combined = pd.DataFrame(
            [
                {
                    **common,
                    "result_set": "old3",
                    "run": "old",
                    "validation_id_hash": "same",
                    "mean_pair_log_gamma_pcc": 0.8,
                    "pooled_log_gamma_rmse": 0.3,
                },
                {
                    **common,
                    "result_set": "current",
                    "run": "new",
                    "validation_id_hash": "same",
                    "mean_pair_log_gamma_pcc": 0.9,
                    "pooled_log_gamma_rmse": 0.2,
                },
                {
                    **common,
                    "result_set": "current",
                    "run": "different_split",
                    "validation_id_hash": "different",
                    "mean_pair_log_gamma_pcc": 0.99,
                    "pooled_log_gamma_rmse": 0.01,
                },
            ]
        )
        matched = build_matched_comparison(
            combined,
            baseline_label="old3",
            candidate_label="current",
        )
        self.assertEqual(len(matched), 1)
        self.assertAlmostEqual(
            matched.iloc[0]["mean_pair_log_gamma_pcc_candidate_minus_baseline"],
            0.1,
        )
        self.assertAlmostEqual(
            matched.iloc[0]["pooled_log_gamma_rmse_improvement"],
            0.1,
        )


if __name__ == "__main__":
    unittest.main()
