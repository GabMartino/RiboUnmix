"""Physical scale, centering domain and transcript-cluster aggregation checks."""
import unittest

import numpy as np
import pandas as pd

from analyses.analyze_synthetic_gamma_depth_recovery import (
    METRICS, aggregate_transcripts, profile_scores, score_gamma_matrix,
)
from analyses.analyze_synthetic_gamma_recovery import joint_log_gamma_gauge


class TestGammaDepthRecovery(unittest.TestCase):
    def test_scores_use_multiplier_scale(self):
        result = profile_scores([1., 2., 4.], [1., 2., 3.])
        self.assertAlmostEqual(result["rmse"], 1/np.sqrt(3))
        self.assertAlmostEqual(result["pcc"], np.corrcoef([1., 2., 4.], [1., 2., 3.])[0, 1])

    def test_constant_reference_is_not_pcc_zero(self):
        result = profile_scores([1., 2., 4.], np.ones(3))
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "constant_reference")
        self.assertTrue(np.isnan(result["pcc"]))
        self.assertTrue(np.isfinite(result["rmse"]))

    def test_interior_gauge_precedes_scoring(self):
        rng = np.random.default_rng(6)
        truth = rng.normal(0, .3, (3, 40))
        prediction = truth.copy()
        prediction[0, :10] += 3  # errors outside the declared evaluation domain
        scores = score_gamma_matrix(prediction, truth, np.ones(3)/3)
        for row in scores:
            self.assertAlmostEqual(row["gamma_pcc"], 1.)
            self.assertAlmostEqual(row["gamma_rmse"], 0.)
            self.assertGreater(row["sense_gauge_gamma_rmse"], 0.)

    def test_error_is_exponentiated_after_gauge(self):
        rng = np.random.default_rng(1)
        truth = rng.normal(0, .3, (3, 40))
        prediction = truth + rng.normal(0, .1, truth.shape)
        scores = score_gamma_matrix(prediction, truth, np.ones(3)/3)
        a = np.exp(joint_log_gamma_gauge(prediction[:, 10:-10]))
        b = np.exp(joint_log_gamma_gauge(truth[:, 10:-10]))
        self.assertAlmostEqual(scores[0]["gamma_rmse"], np.sqrt(np.mean((a[0]-b[0])**2)))

    def test_dataset_means_preserve_transcript_clusters_and_invalidity(self):
        frame = pd.DataFrame([dict(run="run", depth="depth", n_datasets=2, transcript_id=t,
                                   dataset=d, pcc_valid=True, **{m: value for m in METRICS})
                              for t in ("a", "b") for d, value in (("one", .2), ("two", .8))])
        result = aggregate_transcripts(frame)
        np.testing.assert_allclose(result.gamma_rmse, .5)
        frame.loc[0, "gamma_pcc"] = np.nan
        frame.loc[0, "pcc_valid"] = False
        result = aggregate_transcripts(frame).set_index("transcript_id")
        self.assertTrue(np.isnan(result.loc["a", "gamma_pcc"]))
        self.assertEqual(result.loc["a", "n_valid_dataset_pcc"], 1)


if __name__ == "__main__":
    unittest.main()
