"""Checks of estimands and complete-case matching, not image pixel tests."""
import unittest

import numpy as np
import pandas as pd

from analyses.plot_synthetic_read_depth_effect import (
    COUNTS, complete_cohort, cohort_hash, mean_stats, paired_ratio_stats,
)


class TestDepthEffectPlot(unittest.TestCase):
    def test_ratio_is_ratio_of_means(self):
        draws = np.array([[0, 1], [0, 0], [1, 1]])
        result = paired_ratio_stats([2., 8.], [1., 2.], draws)
        self.assertAlmostEqual(result["ratio"], 10/3)
        self.assertNotAlmostEqual(result["ratio"], 3.)

    def test_bootstrap_preserves_pairing(self):
        draws = np.random.default_rng(1).integers(3, size=(200, 3))
        result = paired_ratio_stats([2., 4., 10.], [1., 2., 5.], draws)
        self.assertAlmostEqual(result["ratio_ci_low"], 2.)
        self.assertAlmostEqual(result["ratio_ci_high"], 2.)

    def test_mean_is_not_fisher_weighted(self):
        draws = np.array([[0, 1, 2], [1, 2, 0]])
        self.assertAlmostEqual(mean_stats([.1, .2, .9], draws)["mean"], .4)

    def test_cohort_is_fixed_across_n(self):
        frame = pd.DataFrame([dict(transcript_id=t, depth=d, n_datasets=n, metric=1.)
                              for t in ("a", "b") for d in ("low", "high") for n in COUNTS])
        frame.loc[(frame.transcript_id == "b") & (frame.n_datasets == 4), "metric"] = np.nan
        included, excluded = complete_cohort(frame, {"a", "b"}, ("low", "high"), ("metric",))
        self.assertEqual(included, ["a"])
        self.assertEqual(excluded, ["b"])

    def test_cohort_hash_is_order_invariant(self):
        self.assertEqual(cohort_hash(["a", "b"]), cohort_hash(["b", "a"]))


if __name__ == "__main__":
    unittest.main()
