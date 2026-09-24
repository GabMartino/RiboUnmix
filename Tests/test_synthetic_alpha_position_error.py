"""Relative-position coordinates, raw-replica means and error aggregation."""
import unittest
import numpy as np
import pandas as pd

from analyses.analyze_synthetic_alpha_position_error import (
    position_bins, error_profiles, correlation, aggregate,
)


class TestAlphaPositionError(unittest.TestCase):
    def test_bins_include_both_endpoints_without_dropping_codons(self):
        u, bins = position_bins(44, 20)
        self.assertEqual(u[0], 0.)
        self.assertEqual(u[-1], 1.)
        self.assertEqual(bins[0], 0)
        self.assertEqual(bins[-1], 19)
        self.assertEqual(len(np.unique(bins)), 20)
        self.assertEqual(np.bincount(bins).sum(), 44)

    def test_replica_specific_scales_do_not_use_consensus_mean_for_each_replica(self):
        replicas = np.array([[1., 2., 3.], [3., 6., 9.]])
        mu = replicas.mean(axis=0)
        scores = error_profiles(mu, mu, [.5, 1., 1.5], replicas, 1e-8, .1)
        for values in scores.values():
            np.testing.assert_allclose(values, 0.)
        self.assertGreater(np.mean((replicas-mu)**2), 0.)

    def test_standardized_error_uses_fixed_reference_variance(self):
        replicas = np.array([[0., 2.], [0., 2.]])
        scores = error_profiles([1., 1.], [0., 2.], [1., 1.], replicas, 1e-8, .1)
        np.testing.assert_allclose(scores["consensus_mse"], 1.)
        np.testing.assert_allclose(scores["replica_mse"], 1.)
        np.testing.assert_allclose(scores["reference_standardized_error"], 1/1.1)
        np.testing.assert_allclose(scores["consensus_reference_standardized_error"], 2/1.1)

    def test_wrong_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "arithmetic mean"):
            error_profiles([1., 1.], [9., 9.], [1., 1.], [[0., 2.], [0., 2.]], 1e-8, .1)

    def test_constant_alpha_correlation_is_flagged_not_zero(self):
        r, reason = correlation(np.ones(10)*.1, np.arange(10))
        self.assertTrue(np.isnan(r))
        self.assertEqual(reason, "nearly_constant_alpha")
        self.assertAlmostEqual(correlation(np.arange(10), np.arange(10))[0], 1.)

    def test_invalid_dataset_association_invalidates_transcript_mean(self):
        frame = pd.DataFrame([dict(run="r", depth="d", n_datasets=2, transcript_id="t", dataset=d,
                                  domain="full_sense", metric=m) for d, m in (("a", 1.), ("b", np.nan))])
        result = aggregate(frame, ["domain"], ["metric"])
        self.assertTrue(np.isnan(result.metric.iloc[0]))


if __name__ == "__main__":
    unittest.main()
