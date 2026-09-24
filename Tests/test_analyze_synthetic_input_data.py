from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from analyses.analyze_synthetic_input_data import (
    iter_grouped_profiles,
    mean_one,
    pearson,
    rmse_mean_one,
    sample_role,
)


class SyntheticInputAuditTests(unittest.TestCase):
    def test_pearson_preserves_undefined_constant_case(self) -> None:
        self.assertTrue(np.isnan(pearson(np.ones(5), np.arange(5))))
        self.assertAlmostEqual(pearson(np.arange(5), 3.0 * np.arange(5) + 2.0), 1.0)
        self.assertAlmostEqual(pearson(np.arange(7), -np.arange(7), trim=1), -1.0)

    def test_mean_one_and_rmse_use_amplitudes(self) -> None:
        observed = np.array([2.0, 4.0, 6.0])
        target = np.array([0.5, 1.0, 1.5])
        np.testing.assert_allclose(mean_one(observed), target)
        self.assertAlmostEqual(rmse_mean_one(observed, target), 0.0)
        self.assertIsNone(mean_one(np.zeros(3)))

    def test_sample_labels_do_not_treat_mean_as_replica(self) -> None:
        self.assertEqual(sample_role("gt_rep1"), "rep1")
        self.assertEqual(sample_role("bias_rep2"), "rep2")
        self.assertEqual(sample_role("bias_mean"), "mean")
        with self.assertRaises(ValueError):
            sample_role("rep3")

    def test_grouped_reader_handles_row_group_boundaries(self) -> None:
        rows = [
            {"sample": "x_rep1", "transcript_id": "a", "rib_profile": [1, 2]},
            {"sample": "x_rep2", "transcript_id": "a", "rib_profile": [3, 4]},
            {"sample": "x_mean", "transcript_id": "a", "rib_profile": [2, 3]},
            {"sample": "x_rep1", "transcript_id": "b", "rib_profile": [5, 6]},
            {"sample": "x_rep2", "transcript_id": "b", "rib_profile": [7, 8]},
            {"sample": "x_mean", "transcript_id": "b", "rib_profile": [6, 7]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2)
            grouped = list(iter_grouped_profiles(path, "rib_profile", batch_size=2))
        self.assertEqual([item[0] for item in grouped], ["a", "b"])
        np.testing.assert_array_equal(grouped[1][1]["rep2"], [7.0, 8.0])


if __name__ == "__main__":
    unittest.main()
