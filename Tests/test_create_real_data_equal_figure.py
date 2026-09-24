"""Focused tests for the uniform-reference Figure 1 estimands."""

from __future__ import annotations

import numpy as np
import pandas as pd
import unittest

from analyses.create_real_data_equal_figure import (
    Profile,
    build_figure,
    profile_agreement,
    require_uniform_pi,
)


class UniformReferenceFigureTests(unittest.TestCase):
    def test_uniform_pi_contract_is_exactly_the_requested_estimand(self) -> None:
        require_uniform_pi({"a": 0.5, "b": 0.5}, ["a", "b"])
        with self.assertRaisesRegex(ValueError, "uniform"):
            require_uniform_pi({"a": 0.75, "b": 0.25}, ["a", "b"])
        with self.assertRaisesRegex(ValueError, "membership"):
            require_uniform_pi({"a": 0.5, "c": 0.5}, ["a", "b"])

    def test_aligned_positions_and_undefined_pcc(self) -> None:
        increasing = Profile(np.array([0.5, 1.0, 1.5]), 3)
        decreasing = Profile(np.array([1.5, 1.0, 0.5]), 3)
        constant = Profile(np.ones(3), 3)
        valid = profile_agreement(increasing, decreasing)
        self.assertEqual(valid["status"], "valid")
        self.assertAlmostEqual(valid["PCC"], -1.0)
        self.assertEqual(
            profile_agreement(increasing, Profile(np.ones(4), 4))["status"],
            "misaligned_length",
        )
        undefined = profile_agreement(increasing, constant)
        self.assertEqual(undefined["status"], "constant_right")
        self.assertTrue(np.isnan(undefined["PCC"]))

    def test_figure_connects_only_pair_averages(self) -> None:
        pairs = ["P1\N{EN DASH}P2", "P1\N{EN DASH}P3", "P1\N{EN DASH}P4",
                 "P2\N{EN DASH}P3", "P2\N{EN DASH}P4", "P3\N{EN DASH}P4"]
        panel_a = pd.DataFrame(
            {
                "panel_pair": pairs,
                "p05_PCC": np.linspace(0.1, 0.2, 6),
                "p25_PCC": np.linspace(0.2, 0.3, 6),
                "median_PCC": np.linspace(0.3, 0.4, 6),
                "p75_PCC": np.linspace(0.4, 0.5, 6),
                "p95_PCC": np.linspace(0.5, 0.6, 6),
            }
        )
        pair_summary = pd.DataFrame(
            [
                {"N": N, "pair_id": pair, "R_N_p_mean_transcript_PCC": value}
                for N in (2, 5, 10, 20, 40)
                for pair, value in zip(
                    ("pair01", "pair02", "pair03"), (0.2, 0.4, 0.9), strict=True
                )
            ]
        )
        n_summary = pd.DataFrame(
            {"N": (2, 5, 10, 20, 40), "R_N_mean_over_pair_means": [0.5] * 5}
        )
        overlapping_summary = pd.DataFrame(
            {"N": [80], "mean_over_pair_means": [0.783333333333]}
        )
        adjacent = pd.DataFrame(
            {
                "plot_x_geometric_midpoint": np.sqrt(
                    np.array([2, 5, 10, 20, 40, 80])
                    * np.array([5, 10, 20, 40, 80, 114])
                ),
                "plot_x_larger_endpoint": [5, 10, 20, 40, 80, 114],
                "mean_over_model_pair_means": [0.7] * 6,
                "minimum_model_pair_mean_PCC": [0.6] * 6,
                "p25_model_pair_mean_PCC": [0.65] * 6,
                "p75_model_pair_mean_PCC": [0.75] * 6,
                "maximum_model_pair_mean_PCC": [0.8] * 6,
            }
        )
        figure = build_figure(
            panel_a,
            pair_summary,
            n_summary,
            overlapping_summary,
            adjacent,
            width=7.15,
        )
        axis_b = figure.axes[1]
        self.assertTrue(np.allclose(axis_b.lines[0].get_ydata(), 0.5))
        self.assertTrue(np.allclose(axis_b.lines[1].get_ydata(), 0.7))
        self.assertEqual(list(axis_b.lines[1].get_xdata()), [5, 10, 20, 40, 80, 114])
        self.assertEqual(axis_b.get_xscale(), "log")
        self.assertEqual(
            [tick.get_text() for tick in axis_b.get_xticklabels()],
            ["2", "5", "10", "20", "40", "80", "114"],
        )


if __name__ == "__main__":
    unittest.main()
