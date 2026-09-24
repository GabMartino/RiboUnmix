"""Dispersion scale, effective bounds, and equal-transcript aggregation."""
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from analyses.analyze_synthetic_alpha_depth_recovery import (
    METRICS, alpha_scores, aggregate_transcripts, summarize, plot, DEPTHS, COUNTS,
)


class TestAlphaRecovery(unittest.TestCase):
    def test_overview_contains_only_rmse_panel(self):
        summary = pd.DataFrame([dict(depth=d, n_datasets=n, metric="alpha_rmse", mean=.04,
                                     ci_low=.03, ci_high=.05) for d in DEPTHS for n in COUNTS])
        with tempfile.TemporaryDirectory() as directory, \
             patch("matplotlib.figure.Figure.savefig", autospec=True) as save:
            plot(summary, Path(directory), .1, 14.)
            fig = save.call_args_list[0].args[0]
            self.assertEqual(len(fig.axes), 1)
            self.assertIn("RMSE", fig.axes[0].get_ylabel())
            self.assertEqual(len(fig.axes[0].containers), 3)

    def test_exact_recovery_of_constant_alpha_needs_no_pcc(self):
        result = alpha_scores(np.full(20, np.log(.1)), .1, -5, 1)
        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["alpha_mean"], .1)
        self.assertAlmostEqual(result["alpha_rmse"], 0.)
        self.assertAlmostEqual(result["alpha_spatial_sd"], 0.)
        self.assertEqual(result["fraction_within_factor_two"], 1.)
        self.assertNotIn("pcc", result)

    def test_clamping_matches_likelihood_and_records_saturation(self):
        result = alpha_scores([-100., np.log(.1), 100.], .1, -5, 1)
        expected = np.array([np.exp(-5), .1, np.exp(1)])
        self.assertAlmostEqual(result["alpha_mean"], expected.mean())
        self.assertAlmostEqual(result["alpha_rmse"], np.sqrt(np.mean((expected-.1)**2)))
        self.assertAlmostEqual(result["fraction_at_floor"], 1/3)
        self.assertAlmostEqual(result["fraction_at_cap"], 1/3)

    def test_correct_mean_does_not_imply_correct_profile(self):
        result = alpha_scores(np.log([.05, .15]), .1, -5, 1)
        self.assertAlmostEqual(result["alpha_bias"], 0.)
        self.assertAlmostEqual(result["alpha_rmse"], .05)

    def test_nonfinite_values_are_flagged_not_dropped(self):
        result = alpha_scores([np.log(.1), np.nan], .1, -5, 1)
        self.assertFalse(result["valid"])
        self.assertTrue(np.isnan(result["alpha_rmse"]))

    def test_equal_dataset_and_transcript_aggregation(self):
        detail = []
        for depth in DEPTHS:
            for n in COUNTS:
                for tid, alpha in (("short", .1), ("long", .3)):
                    for d in range(n):
                        detail.append(dict(run=f"{depth}_{n}", depth=depth, n_datasets=n,
                                           transcript_id=tid, dataset=str(d),
                                           **alpha_scores(np.log([alpha]*(2 if tid == "short" else 200)), .1, -5, 1)))
        transcripts = aggregate_transcripts(pd.DataFrame(detail))
        result = summarize(transcripts, ["short", "long"], 100, 42)
        np.testing.assert_allclose(result.loc[result.metric == "alpha_mean", "mean"], .2)
        np.testing.assert_allclose(result.loc[result.metric == "alpha_rmse", "mean"], .1)
        # A mean of profile RMSEs is not the pooled RMSE sqrt((0^2+.2^2)/2).
        self.assertEqual(len(result[result.metric == "alpha_rmse"][["ci_low", "ci_high"]].drop_duplicates()), 1)


if __name__ == "__main__":
    unittest.main()
