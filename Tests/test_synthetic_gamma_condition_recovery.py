"""Estimand tests for condition-resolved synthetic gamma recovery."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from analyses.analyze_synthetic_gamma_condition_recovery import (
    BIAS_ORDER,
    DEPTHS,
    load_condition_records,
    summarize,
)


class TestSyntheticGammaConditionRecovery(unittest.TestCase):
    def test_complete_grid_and_amplitude_slope(self):
        rows = []
        for depth in DEPTHS:
            for condition in BIAS_ORDER:
                for transcript in ("t1", "t2"):
                    rows.append(
                        {
                            "run": f"{depth}_10",
                            "depth": depth,
                            "n_datasets": 10,
                            "transcript_id": transcript,
                            "dataset": f"artificial_bias_{condition}",
                            "gamma_pcc": 0.8,
                            "gamma_rmse": 0.1,
                            "pcc_valid": True,
                            "predicted_gamma_variance": 4.0,
                            "reference_gamma_variance": 1.0,
                        }
                    )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gamma.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            frame, ids = load_condition_records(path)
        np.testing.assert_allclose(frame["amplitude_slope"], 1.6)
        result = summarize(frame, ids, repeats=100, seed=7)
        slope = result.loc[result["metric"] == "amplitude_slope"]
        np.testing.assert_allclose(slope["mean"], 1.6)
        self.assertTrue((slope["n_transcripts"] == 2).all())

    def test_missing_condition_is_rejected(self):
        rows = [
            {
                "run": "r",
                "depth": depth,
                "n_datasets": 10,
                "transcript_id": "t",
                "dataset": "artificial_bias_3prime_aa",
                "gamma_pcc": 0.9,
                "gamma_rmse": 0.1,
                "pcc_valid": True,
                "predicted_gamma_variance": 1.0,
                "reference_gamma_variance": 1.0,
            }
            for depth in DEPTHS
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gamma.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "bias conditions differ"):
                load_condition_records(path)


if __name__ == "__main__":
    unittest.main()
