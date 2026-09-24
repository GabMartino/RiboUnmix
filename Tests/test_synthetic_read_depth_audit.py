"""Numerical and reference-weighting checks for the independent frozen audit."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "depth_audit", Path(__file__).resolve().parents[1] / "analyses/audit_synthetic_read_depth.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class TestDepthAudit(unittest.TestCase):
    def test_metrics_do_not_renormalize(self):
        pcc, rmse = audit.metrics(np.array([2., 4., 6.]), np.array([1., 2., 3.]), trim=0)
        self.assertAlmostEqual(pcc, 1.)
        self.assertAlmostEqual(rmse, np.sqrt(14/3))

    def test_constant_and_short_profiles(self):
        self.assertTrue(np.isnan(audit.metrics(np.ones(30), np.arange(30.))[0]))
        self.assertTrue(np.isnan(audit.metrics(np.arange(20.), np.arange(20.))[0]))

    def test_alignment_is_not_silently_truncated(self):
        with self.assertRaises(ValueError):
            audit.metrics(np.ones(30), np.ones(31))

    def test_balanced_depth_rank_has_same_bias_marginal(self):
        datasets = [f"artificial_bias_{bias}_{depth}"
                    for bias in ("3prime_aa", "3prime_cc") for depth in audit.DEPTHS]
        run = dict(datasets=datasets, weighting="equal")
        equal, weights = audit.bias_weights(run)
        self.assertAlmostEqual(weights.sum(), 1.)
        run.update(weighting="quality_rank", config={"model": {"gamma_centering": {
            "reference": {"quality_rank_power": 1.}}}})
        ranked, weights = audit.bias_weights(run)
        self.assertAlmostEqual(weights.sum(), 1.)
        self.assertEqual(equal, ranked)


if __name__ == "__main__":
    unittest.main()
