import unittest

import numpy as np
import pandas as pd

from analyses.analyze_synthetic_kinetic_alignment_matched import (
    DEPTH_ORDER,
    mask_digest,
    summarize,
)
from analyses.create_iclr_synthetic_compact_recovery import validate_kinetic


class SyntheticKineticAlignmentTests(unittest.TestCase):
    def _metrics(self) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        cohorts = {
            depth: [f"{depth}_t{index}" for index in range(5)]
            for depth in DEPTH_ORDER
        }
        rows = []
        for depth_index, depth in enumerate(DEPTH_ORDER):
            for n_datasets in range(2, 11):
                for transcript_index, transcript_id in enumerate(cohorts[depth]):
                    rows.append(
                        {
                            "run_id": f"{depth}_N{n_datasets}",
                            "depth": depth,
                            "n_datasets": n_datasets,
                            "reference_weighting": "equal",
                            "training_seed": 42,
                            "transcript_id": transcript_id,
                            "pcc_L_vs_K": (
                                0.60 + 0.01 * n_datasets + 0.001 * transcript_index
                                + 0.0001 * depth_index
                            ),
                            "pcc_defined": True,
                        }
                    )
        return pd.DataFrame(rows), cohorts

    def test_summary_preserves_fixed_cohort_and_expected_median(self):
        metrics, cohorts = self._metrics()
        summary = summarize(metrics, cohorts, replicates=50, seed=7)
        self.assertEqual(len(summary), 27)
        self.assertTrue(summary.n_transcripts.eq(5).all())
        self.assertTrue(summary.n_undefined.eq(0).all())
        row = summary.loc[
            summary.depth.eq(DEPTH_ORDER[0]) & summary.n_datasets.eq(2)
        ].iloc[0]
        self.assertAlmostEqual(row["median"], 0.622)
        self.assertLessEqual(row.bootstrap_ci_low, row["median"])
        self.assertGreaterEqual(row.bootstrap_ci_high, row["median"])

    def test_figure_validation_rejects_undefined_correlations(self):
        metrics, cohorts = self._metrics()
        summary = summarize(metrics, cohorts, replicates=20, seed=11)
        validate_kinetic(summary)
        summary.loc[0, "n_undefined"] = 1
        with self.assertRaises(ValueError):
            validate_kinetic(summary)

    def test_mask_digest_depends_on_exact_retained_positions(self):
        first = np.array([True, False, True, False], dtype=bool)
        second = np.array([True, False, False, True], dtype=bool)
        self.assertEqual(mask_digest(first), mask_digest(first.copy()))
        self.assertNotEqual(mask_digest(first), mask_digest(second))


if __name__ == "__main__":
    unittest.main()
