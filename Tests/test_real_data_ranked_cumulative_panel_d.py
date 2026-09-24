"""Focused tests for the cumulative ranked panel-D estimand."""
import unittest

import numpy as np
import pandas as pd

from analyses.create_real_data_ranked_cumulative_panel_d import SERIES_ORDER, SUCCESSIVE_PAIRS, plot


class RankedCumulativePanelDTests(unittest.TestCase):
    def test_plot_uses_successive_pair_ticks_and_saved_intervals(self):
        rows = []
        for series_index, series in enumerate(SERIES_ORDER):
            for comparison_index, (N_a, N_b) in enumerate(SUCCESSIVE_PAIRS):
                value = .55 + .04*comparison_index + .12*series_index
                rows.append(dict(series=series, comparison_index=comparison_index,
                                 N_a=N_a, N_b=N_b, mean_PCC=value,
                                 ci_lower=value-.02, ci_upper=value+.02))
        summary = pd.DataFrame(rows)
        fig = plot(summary, 2.7, no_tex=True)
        axis = fig.axes[0]
        self.assertEqual(tuple(axis.get_xticks()), tuple(range(len(SUCCESSIVE_PAIRS))))
        self.assertEqual([tick.get_text() for tick in axis.get_xticklabels()],
                         [f"{a}:{b}" for a, b in SUCCESSIVE_PAIRS])
        for line, collection, series in zip(axis.lines, axis.collections, SERIES_ORDER):
            group = summary.loc[summary.series.eq(series)].sort_values("comparison_index")
            np.testing.assert_allclose(line.get_ydata(), group.mean_PCC)
            np.testing.assert_allclose(np.asarray(collection.get_segments())[:, :, 1],
                                       group[["ci_lower", "ci_upper"]])


if __name__ == "__main__":
    unittest.main()
