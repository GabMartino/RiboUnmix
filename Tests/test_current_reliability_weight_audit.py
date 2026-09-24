from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from Datasets.data.plot_current_reliability_weight_audit import (
    PROJECT_ROOT,
    WeightReference,
    _load_experiment_references,
    load_plot_data,
    normalized_reliability_weight,
)


class CurrentReliabilityWeightAuditTests(unittest.TestCase):
    def test_current_equation(self) -> None:
        reference = WeightReference(tau=4.0, normalization_median=0.5)
        density = np.asarray([1.0, 4.0], dtype=np.float64)
        coverage = np.asarray([0.25, 1.0], dtype=np.float64)
        expected_raw = 0.70 * np.asarray([1.0 / 3.0, 0.5]) + 0.30 * coverage
        actual = normalized_reliability_weight(density, coverage, reference)
        np.testing.assert_allclose(actual, expected_raw / 0.5, rtol=0.0, atol=1e-12)

    def test_artifact_points_match_current_formula(self) -> None:
        path = (
            PROJECT_ROOT
            / "Datasets/data/weighted_HEK_riboseq_codon_replicas/akichika_2019.parquet"
        )
        if not path.exists():
            self.skipTest("Local weighted HEK artifact is unavailable.")
        frame, reference, error = load_plot_data(path, "akichika_2019", None)
        self.assertEqual(reference.source, "stored_artifact")
        self.assertLessEqual(error, 2.0e-5)
        np.testing.assert_array_equal(
            frame["plot_weight"].to_numpy(), frame["weight"].to_numpy()
        )
        self.assertAlmostEqual(float(np.median(frame["plot_weight"])), 1.0, places=6)

    def test_completed_panel_manifests_have_unique_dataset_references(self) -> None:
        root = PROJECT_ROOT / "results/my_panels_a100_b32_20260906_114323"
        paths = sorted(root.glob("panel_*/reliability_reference_manifest.json"))
        if len(paths) != 4:
            self.skipTest("Completed four-panel reliability manifests are unavailable.")
        references = _load_experiment_references(paths)
        self.assertEqual(len(references), 114)
        self.assertTrue(
            all(ref.reference_split == "training_only" for ref in references.values())
        )


if __name__ == "__main__":
    unittest.main()
