from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from analyses.create_real_data_ranked_panel_c import (
    INTERNAL_PAIRS,
    joint_paired_bootstrap,
)


class TestRankedPanelC(unittest.TestCase):
    def _records(self) -> pd.DataFrame:
        rows = []
        for transcript_index, transcript_id in enumerate(("t1", "t2", "t3", "t4")):
            for pair_index, pair in enumerate(INTERNAL_PAIRS):
                equal = 0.1 * transcript_index + 0.01 * pair_index
                ranked = equal + (0.02 if transcript_index < 3 else -0.05)
                rows.append(
                    {
                        "transcript_id": transcript_id,
                        "comparison_id": pair,
                        "PCC_equal": equal,
                        "PCC_ranked": ranked,
                    }
                )
        return pd.DataFrame(rows)

    def test_point_estimate_is_difference_of_medians(self) -> None:
        records = self._records()
        summary, _ = joint_paired_bootstrap(records, draws=20, seed=7)
        first = records.loc[records.comparison_id.eq(INTERNAL_PAIRS[0])]
        expected = np.median(first.PCC_ranked) - np.median(first.PCC_equal)
        self.assertAlmostEqual(summary.iloc[0].estimate, expected)

    def test_bootstrap_is_deterministic(self) -> None:
        records = self._records()
        left, left_digest = joint_paired_bootstrap(records, draws=40, seed=11)
        right, right_digest = joint_paired_bootstrap(records, draws=40, seed=11)
        pd.testing.assert_frame_equal(left, right)
        self.assertEqual(left_digest, right_digest)


if __name__ == "__main__":
    unittest.main()
