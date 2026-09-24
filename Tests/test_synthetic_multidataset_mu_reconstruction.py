"""Focused tests for the multi-dataset reconstruction estimands."""
import unittest

import numpy as np
import pandas as pd

from analyses.analyze_synthetic_multidataset_mu_reconstruction import (
    METRICS,
    STANDARDIZED_METRIC,
    attach_standardized_error,
    fixed_cohorts,
    load_records,
)


class TestSyntheticMultidatasetMuReconstruction(unittest.TestCase):
    def test_standardized_error_is_square_rooted_and_key_matched(self):
        import tempfile
        from pathlib import Path

        frame = pd.DataFrame(
            [
                {
                    "run": "r",
                    "depth": "0p25_per_codon",
                    "n_datasets": 2,
                    "transcript_id": "t",
                }
            ]
        )
        source = frame.assign(
            domain="interior10",
            consensus_reference_standardized_error=2.25,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "standardized.csv"
            source.to_csv(path, index=False)
            result = attach_standardized_error(frame, path)
        self.assertAlmostEqual(result.loc[0, STANDARDIZED_METRIC], 1.5)

    def test_fixed_cohort_requires_every_n_and_metric(self):
        rows = []
        for transcript in ("complete_a", "complete_b", "missing"):
            for depth in ("0p25_per_codon", "2_per_codon", "20_per_codon"):
                for n in range(2, 11):
                    rows.append(
                        {
                            "transcript_id": transcript,
                            "depth": depth,
                            "n_datasets": n,
                            METRICS[0]: 0.8,
                            METRICS[1]: 1.0,
                            STANDARDIZED_METRIC: 1.0,
                        }
                    )
        frame = pd.DataFrame(rows)
        frame.loc[
            (frame["transcript_id"] == "missing")
            & (frame["depth"] == "2_per_codon")
            & (frame["n_datasets"] == 6),
            METRICS[0],
        ] = np.nan
        validation = {
            depth: {"complete_a", "complete_b", "missing"}
            for depth in ("0p25_per_codon", "2_per_codon", "20_per_codon")
        }
        primary, depth_cohorts, exclusions = fixed_cohorts(frame, validation)
        self.assertEqual(primary, ["complete_a", "complete_b"])
        self.assertEqual(
            depth_cohorts["0p25_per_codon"],
            ["complete_a", "complete_b", "missing"],
        )
        self.assertEqual(depth_cohorts["2_per_codon"], ["complete_a", "complete_b"])
        self.assertEqual(exclusions["common_three_depths"], ["missing"])

    def test_loader_does_not_replace_undefined_pcc_with_zero(self):
        # The full grid is required by load_records; exercise its completeness
        # rule directly on a temporary source table.
        import tempfile
        from pathlib import Path

        rows = []
        for depth in ("0p25_per_codon", "2_per_codon", "20_per_codon"):
            for n in range(2, 11):
                rows.append(
                    dict(
                        run=f"{depth}_{n}",
                        depth=depth,
                        n_datasets=n,
                        variant="best_pcc",
                        transcript_id="t",
                        sense_length=100,
                        mu_target_pcc_trim10=0.5,
                        mu_target_rmse_trim10=1.0,
                        mu_target_valid_datasets=n - (n == 4),
                    )
                )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            runs = [{"run": row["run"]} for row in rows]
            result = load_records(path, runs)
        affected = result[result["n_datasets"] == 4]
        self.assertTrue(affected[METRICS[0]].isna().all())
        self.assertTrue((affected[METRICS[1]] == 1.0).all())


if __name__ == "__main__":
    unittest.main()
