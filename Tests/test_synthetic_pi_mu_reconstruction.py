"""Mathematical tests for the synthetic pi reconstruction analysis."""
import unittest

import numpy as np

from analyses.analyze_synthetic_pi_mu_reconstruction import profile_metrics


class TestSyntheticPiMuReconstruction(unittest.TestCase):
    def test_terminal_is_not_part_of_profile_metric(self):
        # evaluate_artifact removes the terminal; this unit tests the remaining
        # interior convention independently.
        target = np.arange(20, dtype=float)
        prediction = target.copy()
        prediction[:5] = 1000
        prediction[-5:] = -1000
        result = profile_metrics(prediction, target, trim=5)
        self.assertAlmostEqual(result["mu_pcc"], 1.0)
        self.assertAlmostEqual(result["mu_rmse"], 0.0)
        self.assertAlmostEqual(result["mu_relative_rmse"], 0.0)
        self.assertEqual(result["positions"], 10)

    def test_relative_rmse_removes_common_count_scale(self):
        target = np.arange(1, 21, dtype=float)
        prediction = target * 1.1
        first = profile_metrics(prediction, target, trim=5)
        second = profile_metrics(prediction * 20, target * 20, trim=5)
        self.assertAlmostEqual(first["mu_relative_rmse"], second["mu_relative_rmse"])
        self.assertAlmostEqual(second["mu_rmse"], 20 * first["mu_rmse"])

    def test_undefined_pcc_is_not_zero(self):
        result = profile_metrics(np.ones(20), np.arange(20), trim=5)
        self.assertTrue(np.isnan(result["mu_pcc"]))
        self.assertEqual(result["pcc_reason"], "near_constant")


if __name__ == "__main__":
    unittest.main()
