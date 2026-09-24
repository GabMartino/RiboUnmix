from __future__ import annotations

import unittest

import numpy as np
from scipy.stats import spearmanr

from analyses.compare_real_panel_weighting import metrics, paired_bootstrap, paired_rows


class PanelWeightingComparisonTests(unittest.TestCase):
    def test_metrics_flag_constants_and_reject_misalignment(self):
        self.assertFalse(metrics(np.ones(20), np.arange(20.0))["valid_PCC"])
        self.assertEqual(metrics(np.ones(1), np.ones(1))["reason"], "fewer_than_two_positions")
        with self.assertRaisesRegex(ValueError, "positions differ"):
            metrics(np.ones(2), np.ones(3))
        x = np.array([.5, 1, 1.5])
        self.assertAlmostEqual(metrics(x, 2 * x)["PCC"], 1)
        self.assertGreater(metrics(x, 2 * x)["RMSE"], 0)

    def test_spearman_uses_average_ties_and_flags_constants(self):
        x = np.array([1., 2., 2., 4., 5.])
        y = np.array([4., 1., 1., 5., 2.])
        self.assertAlmostEqual(metrics(x, y)["Spearman"], spearmanr(x, y).statistic)
        self.assertAlmostEqual(metrics(x, x ** 3)["Spearman"], 1.)
        self.assertTrue(np.isnan(metrics(x, np.ones_like(x))["Spearman"]))
        self.assertFalse(metrics(x, np.ones_like(x))["valid_Spearman"])

    def test_bootstrap_keeps_pairs_and_policies_in_same_transcript_cluster(self):
        x = np.arange(80.0).reshape(20, 4)
        point, ci, valid = paired_bootstrap(x, x + .25, 200, 42)
        self.assertAlmostEqual(point[2], .25)
        np.testing.assert_allclose(ci[:, 2], .25)
        self.assertTrue(valid.all())
        x[0, 1] = np.nan
        _, _, valid = paired_bootstrap(x, x + .25, 200, 42)
        self.assertEqual(valid.sum(), 19)

    def test_partial_run_uses_only_matched_pairs_and_does_not_normalize_interior(self):
        x = np.linspace(.25, 1.75, 80)
        y = x ** 2 / np.mean(x ** 2)
        def profile(values):
            return {"t1": dict(values=values, length=len(values))}
        profiles = {"equal": {"panel_01": profile(x), "panel_02": profile(y), "panel_03": profile(x)},
                    "ranked": {"panel_01": profile(x * .8), "panel_02": profile(y * 1.2)}}
        rows, sensitivity = paired_rows(profiles, ["t1"], ["panel_01", "panel_02"])
        self.assertEqual(len(rows), 4)
        self.assertEqual(set(rows.pair), {"panel_01__panel_02"})
        interior = rows[(rows.region == "interior_20") & (rows.policy == "equal")].iloc[0]
        self.assertEqual(interior.n_positions, 40)
        self.assertAlmostEqual(interior.RMSE, np.sqrt(np.mean((x[20:-20] - y[20:-20]) ** 2)))
        self.assertAlmostEqual(interior.Spearman, spearmanr(x[20:-20], y[20:-20]).statistic)
        self.assertGreater(sensitivity[sensitivity.panel == "panel_01"].RMSE.min(), 0)


if __name__ == "__main__":
    unittest.main()
