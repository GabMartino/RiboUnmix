"""Check metric mapping, four single-axis panels, and transcript aggregation."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analyses.plot_synthetic_depth_recovery_overview import (
    CONTROL_METRICS, COUNTS, DEPTHS, GAMMA_METRICS, PROXY_METRICS,
    plot_metric, plot_overview, summarize_shared,
)


def summary_fixture(metrics, values):
    return pd.DataFrame([dict(depth=d, n_datasets=n, metric=metric, mean=value,
                              ci_low=value-.0001, ci_high=value+.0001)
                         for d in DEPTHS for n in COUNTS
                         for metric, value in zip(metrics, values)])


class TestDepthRecoveryOverview(unittest.TestCase):
    def test_single_axes_use_correct_metric_and_marker_conventions(self):
        frame = summary_fixture(PROXY_METRICS, (.998, .04))
        fig, axes = plt.subplots(1, 2)
        try:
            for ax, metric in zip(axes, PROXY_METRICS):
                plot_metric(ax, frame, metric, 1.)
            self.assertEqual(len(fig.axes), 2)  # No secondary axes.
            self.assertEqual(axes[0].get_ylabel(), "Mean PCC")
            self.assertEqual(axes[1].get_ylabel(), "Mean RMSE")
            for container in axes[0].containers:
                np.testing.assert_allclose(np.asarray(container.lines[0].get_ydata(), dtype=float), .998)
                self.assertEqual(container.lines[0].get_linestyle(), "-")
            for container in axes[1].containers:
                np.testing.assert_allclose(np.asarray(container.lines[0].get_ydata(), dtype=float), .04)
                self.assertEqual(container.lines[0].get_linestyle(), "--")
                self.assertEqual(container.lines[0].get_markerfacecolor(), "white")
            self.assertEqual(axes[1].get_ylim()[0], 0.)
        finally:
            plt.close(fig)

    def test_four_panels_map_to_shared_pcc_rmse_then_gamma_pcc_rmse(self):
        shared = summary_fixture(PROXY_METRICS, (.998, .04))
        gamma = summary_fixture(GAMMA_METRICS, (.997, .03))
        with tempfile.TemporaryDirectory() as directory, \
             patch("matplotlib.figure.Figure.savefig", autospec=True) as save:
            plot_overview(shared, gamma, Path(directory), 12.8)
            fig = save.call_args_list[0].args[0]
            self.assertEqual(len(fig.axes), 4)
            self.assertEqual(save.call_count, 3)
            for ax, value, letter in zip(fig.axes, (.998, .04, .997, .03), "ABCD"):
                self.assertTrue(ax.get_title(loc="left").startswith(letter+"  "))
                self.assertEqual(ax.xaxis.label.get_fontsize(), 14.)
                self.assertEqual(ax.xaxis.label.get_fontweight(), "bold")
                self.assertEqual(len(ax.containers), 3)
                for container in ax.containers:
                    np.testing.assert_allclose(
                        np.asarray(container.lines[0].get_ydata(), dtype=float), value)

    def test_font_size_parameter_controls_labels_ticks_and_legend(self):
        shared = summary_fixture(PROXY_METRICS, (.998, .04))
        gamma = summary_fixture(GAMMA_METRICS, (.997, .03))
        with tempfile.TemporaryDirectory() as directory, \
             patch("matplotlib.figure.Figure.savefig", autospec=True) as save:
            typography = plot_overview(shared, gamma, Path(directory), 12.8, font_size=16.)
            fig = save.call_args_list[0].args[0]
            self.assertEqual(typography["font.size"], 16.)
            for ax in fig.axes:
                self.assertEqual(ax.xaxis.label.get_fontsize(), 16.)
                self.assertEqual(ax.get_xticklabels()[0].get_fontsize(), 16.)
            self.assertEqual(fig.legends[0].get_texts()[0].get_fontsize(), 16.)

    def test_means_are_per_transcript_not_fisher_or_pooled_rmse(self):
        rows = []
        for depth in DEPTHS:
            for n in COUNTS:
                for tid, pcc, rmse in (("a", .2, 1.), ("b", .9, 3.)):
                    rows.append(dict(depth=depth, n_datasets=n, transcript_id=tid,
                                     **{PROXY_METRICS[0]: pcc, PROXY_METRICS[1]: rmse,
                                        CONTROL_METRICS[0]: pcc, CONTROL_METRICS[1]: rmse}))
        result = summarize_shared(pd.DataFrame(rows), ["a", "b"], 100, 42)
        np.testing.assert_allclose(result.loc[result.metric == PROXY_METRICS[0], "mean"], .55)
        np.testing.assert_allclose(result.loc[result.metric == PROXY_METRICS[1], "mean"], 2.)
        intervals = result.loc[result.metric == PROXY_METRICS[0], ["ci_low", "ci_high"]]
        self.assertEqual(len(intervals.drop_duplicates()), 1)
        self.assertTrue(result.loc[result.metric.isin(PROXY_METRICS), "displayed"].all())
        self.assertFalse(result.loc[result.metric.isin(CONTROL_METRICS), "displayed"].any())


if __name__ == "__main__":
    unittest.main()
