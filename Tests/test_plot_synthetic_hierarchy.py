from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from analyses.plot_synthetic_hierarchy import (
    dataframe_markdown_table,
    depth_adjusted_mean_counts,
    empirical_midrank_percentile,
    expected_per_unit_depth,
    summarize_cohort,
    validate_base_profile,
)


class SyntheticHierarchyFigureTests(unittest.TestCase):
    def test_occupancies_are_normalized_separately_and_terminal_is_excluded(self) -> None:
        base = validate_base_profile(
            "t",
            ["AAA", "CCC", "GGG", "TTT", "TAA"],
            np.array([0.5, 1.0, 1.5, 1.0]),
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([4.0, 3.0, 2.0, 1.0]),
            permitted_stops={"TAA", "TAG", "TGA"},
        )
        self.assertEqual(base.K.size, 4)
        self.assertEqual(base.codons, ["AAA", "CCC", "GGG", "TTT"])
        self.assertAlmostEqual(float(base.q1.mean()), 1.0)
        self.assertAlmostEqual(float(base.q2.mean()), 1.0)
        self.assertFalse(np.array_equal(base.q1, base.q2))

    def test_expected_profile_preserves_multiplier_amplitude(self) -> None:
        qbar = np.array([0.5, 1.0, 1.5])
        multiplier = np.array([1.0, 5.0, 1.0])
        expected = expected_per_unit_depth(qbar, multiplier)
        np.testing.assert_array_equal(expected, [0.5, 5.0, 1.5])
        self.assertNotAlmostEqual(float(expected.mean()), 1.0)

    def test_depth_adjustment_is_only_division_by_nominal_C(self) -> None:
        observed = depth_adjusted_mean_counts(
            np.array([0.0, 1.0, 3.0]),
            np.array([0.0, 3.0, 1.0]),
            0.25,
        )
        np.testing.assert_array_equal(observed, [0.0, 8.0, 8.0])
        self.assertNotAlmostEqual(float(observed.mean()), 1.0)

    def test_undefined_correlations_are_counted_not_coerced(self) -> None:
        frame = pd.DataFrame(
            {
                "comparison": ["a", "a", "a"],
                "pearson": [0.5, np.nan, 1.0],
                "rmse": [0.1, 0.2, 0.3],
            }
        )
        summary = summarize_cohort(frame, ["a"]).iloc[0]
        self.assertEqual(summary["n_total_transcripts"], 3)
        self.assertEqual(summary["n_valid_transcripts"], 2)
        self.assertEqual(summary["n_undefined_correlations"], 1)
        self.assertAlmostEqual(summary["median_pearson"], 0.75)
        self.assertEqual(summary["n_valid_rmse"], 3)

    def test_midrank_percentile_handles_ties(self) -> None:
        self.assertAlmostEqual(empirical_midrank_percentile([1.0, 2.0, 2.0, 3.0], 2.0), 50.0)

    def test_markdown_table_has_no_optional_dependency(self) -> None:
        rendered = dataframe_markdown_table(
            pd.DataFrame({"name": ["a|b"], "value": [0.125]}), float_digits=3
        )
        self.assertIn(r"a\|b", rendered)
        self.assertIn("0.125", rendered)

    def test_alignment_mismatch_fails_instead_of_truncating(self) -> None:
        with self.assertRaises(ValueError):
            validate_base_profile(
                "t",
                ["AAA", "CCC", "TAA"],
                np.ones(2),
                np.ones(3),
                np.ones(2),
                permitted_stops={"TAA", "TAG", "TGA"},
            )


if __name__ == "__main__":
    unittest.main()
