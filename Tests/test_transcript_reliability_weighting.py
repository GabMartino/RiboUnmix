from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _load_weighting_module():
    path = ROOT / "Datasets" / "data" / "weight_hek_riboseq_codon_replicas.py"
    spec = importlib.util.spec_from_file_location("snr_weighting_test_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import weighting script from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


W = _load_weighting_module()


def _components(
    coverage: list[float],
    density: list[float],
    *,
    tau: float | None = None,
    mode: str = W.SNR_DEPTH_COVERAGE_MODE,
):
    index = pd.Index([f"t{i}" for i in range(len(coverage))])
    return W.calculate_transcript_weight_components(
        pd.Series(coverage, index=index, dtype="float64"),
        pd.Series(density, index=index, dtype="float64"),
        weighting_mode=mode,
        depth_reference_tau=tau,
        dataset_name="synthetic",
        transcript_ids=pd.Series(index, index=index),
    )


class DepthScoreTests(unittest.TestCase):
    def test_reference_point_is_one_half(self) -> None:
        result = _components([0.5], [4.0], tau=4.0)
        self.assertAlmostEqual(float(result.depth_snr_score.iloc[0]), 0.5, places=12)

    def test_depth_score_is_strictly_monotone(self) -> None:
        result = _components([0.5] * 5, [0.01, 0.1, 1.0, 10.0, 100.0], tau=1.0)
        self.assertTrue(bool((np.diff(result.depth_snr_score.to_numpy()) > 0.0).all()))

    def test_depth_score_saturates_below_one(self) -> None:
        score = float(_components([0.5], [1.0e20], tau=1.0).depth_snr_score.iloc[0])
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.99999)

    def test_low_depth_follows_square_root_ratio(self) -> None:
        scores = _components([0.5, 0.5], [0.01, 0.04], tau=1.0e6).depth_snr_score
        self.assertAlmostEqual(float(scores.iloc[1] / scores.iloc[0]), 2.0, places=3)

    def test_new_coverage_score_is_linear(self) -> None:
        result = _components([0.25], [1.0], tau=1.0)
        self.assertEqual(float(result.coverage_score.iloc[0]), 0.25)
        self.assertNotEqual(float(result.coverage_score.iloc[0]), 0.5)


class RawWeightTests(unittest.TestCase):
    def test_complete_raw_score_at_reference(self) -> None:
        raw = float(_components([0.25], [2.0], tau=2.0).raw_weights.iloc[0])
        self.assertAlmostEqual(raw, 0.425, places=12)

    def test_higher_depth_increases_raw_weight(self) -> None:
        raw = _components([0.5, 0.5], [0.1, 10.0], tau=1.0).raw_weights
        self.assertLess(float(raw.iloc[0]), float(raw.iloc[1]))

    def test_higher_coverage_increases_raw_weight(self) -> None:
        raw = _components([0.1, 0.9], [1.0, 1.0], tau=1.0).raw_weights
        self.assertLess(float(raw.iloc[0]), float(raw.iloc[1]))

    def test_depth_has_diminishing_returns(self) -> None:
        low = _components([0.5, 0.5], [0.1, 1.0], tau=1.0).raw_weights
        high = _components([0.5, 0.5], [10.0, 10.9], tau=1.0).raw_weights
        self.assertGreater(float(low.iloc[1] - low.iloc[0]), float(high.iloc[1] - high.iloc[0]))

    def test_new_mode_does_not_call_percentile_rank(self) -> None:
        with mock.patch.object(pd.Series, "rank", side_effect=AssertionError("rank called")):
            result = _components([0.25, 0.75], [0.5, 2.0], tau=1.0)
        self.assertTrue(bool(np.isfinite(result.raw_weights).all()))


class NormalizationAndCompatibilityTests(unittest.TestCase):
    def test_median_normalization_and_above_one(self) -> None:
        raw = pd.Series([0.4, 0.8, 1.0], dtype="float64")
        normalized, reference = W.normalize_transcript_weights_by_median(raw)
        self.assertEqual(reference, 0.8)
        np.testing.assert_allclose(normalized, [0.5, 1.0, 1.25])
        self.assertAlmostEqual(float(normalized.median()), 1.0, places=12)
        self.assertGreater(float(normalized.max()), 1.0)

    def test_external_tau_and_normalization_median_are_exact(self) -> None:
        components = _components([0.25], [4.0], tau=4.0)
        self.assertEqual(components.depth_reference_tau, 4.0)
        self.assertEqual(components.depth_reference_source, "external_reference")
        normalized, reference = W.normalize_transcript_weights_by_median(
            components.raw_weights,
            reference_median=0.25,
            dataset_name="synthetic",
        )
        self.assertEqual(reference, 0.25)
        self.assertAlmostEqual(float(normalized.iloc[0]), 1.7, places=12)

    def test_legacy_mode_reproduces_old_equation_exactly(self) -> None:
        coverage = np.asarray([0.04, 0.25, 1.0], dtype=np.float64)
        density = np.asarray([10.0, 1.0, 5.0], dtype=np.float64)
        result = _components(
            coverage.tolist(),
            density.tolist(),
            mode=W.LEGACY_COVERAGE_DENSITY_RANK_MODE,
        )
        expected_rank = pd.Series(np.log1p(density)).rank(pct=True).to_numpy()
        expected = 0.70 * np.sqrt(coverage) + 0.30 * expected_rank
        np.testing.assert_array_equal(result.raw_weights.to_numpy(), expected)
        self.assertTrue(bool(result.depth_snr_score.isna().all()))

        new = _components(coverage.tolist(), density.tolist()).raw_weights
        self.assertFalse(bool(np.allclose(new, result.raw_weights)))

    def test_invalid_inputs_report_dataset_and_transcript(self) -> None:
        coverage = pd.Series([0.5], index=[7])
        density = pd.Series([-1.0], index=[7])
        ids = pd.Series(["bad_transcript"], index=[7])
        with self.assertRaisesRegex(ValueError, "synthetic.*bad_transcript.*read_density"):
            W.calculate_transcript_weight_components(
                coverage,
                density,
                dataset_name="synthetic",
                transcript_ids=ids,
            )

    def test_parquet_columns_manifest_and_float32_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "synthetic.parquet"
            output_path = root / "weighted.parquet"
            pd.DataFrame(
                {
                    "id": ["zero", "low", "high"],
                    "ribo": [
                        np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
                        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                        np.asarray([10.0, 10.0, 10.0], dtype=np.float32),
                    ],
                    "ribo_cds_replicas": [
                        [np.asarray([0.0, 0.0, 0.0], dtype=np.float32)],
                        [np.asarray([1.0, 0.0, 0.0], dtype=np.float32)],
                        [np.asarray([10.0, 10.0, 10.0], dtype=np.float32)],
                    ],
                    "replica_ids": [["r1"], ["r1"], ["r1"]],
                }
            ).to_parquet(input_path, index=False)

            summary = W.add_weights(input_path, output_path)
            output = pd.read_parquet(output_path)
            self.assertEqual(output["id"].tolist(), ["low", "high"])
            self.assertGreater(float(output["weight"].max()), 1.0)
            expected_columns = {
                "coverage",
                "read_density",
                "depth_reference_tau",
                "depth_snr_score",
                "coverage_score",
                "raw_weight",
                "weight_raw",
                "weight",
            }
            self.assertTrue(expected_columns.issubset(output.columns))
            np.testing.assert_array_equal(output["raw_weight"], output["weight_raw"])
            for column in expected_columns:
                self.assertEqual(output[column].dtype, np.dtype("float32"))
            required_summary = {
                "weighting_mode",
                "depth_weight",
                "coverage_weight",
                "eligible_rows",
                "input_rows",
                "removed_zero_rows",
                "depth_reference_tau",
                "depth_reference_source",
                "raw_weight_median",
                "final_weight_median",
            }
            self.assertTrue(required_summary.issubset(summary))
            self.assertEqual(summary["weighting_mode"], W.SNR_DEPTH_COVERAGE_MODE)
            self.assertAlmostEqual(float(summary["final_weight_median"]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
