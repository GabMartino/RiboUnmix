from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from analyses.analyze_synthetic_tasep_occupancy_agreement import (
    calculate_matched_metrics,
    iter_occupancy_replicates,
    normalize_occupancy,
    occupancy_role,
)


class SyntheticTasepOccupancyAgreementTests(unittest.TestCase):
    def test_occupancy_reader_preserves_replicate_pairing_across_batches(self) -> None:
        rows = [
            {
                "sample": "replicate_1_mean_psite_occupancy",
                "transcript_id": "a",
                "rib_profile": [1.0, 2.0],
            },
            {
                "sample": "replicate_2_mean_psite_occupancy",
                "transcript_id": "a",
                "rib_profile": [2.0, 1.0],
            },
            {
                "sample": "replicate_1_mean_psite_occupancy",
                "transcript_id": "b",
                "rib_profile": [3.0, 4.0],
            },
            {
                "sample": "replicate_2_mean_psite_occupancy",
                "transcript_id": "b",
                "rib_profile": [4.0, 3.0],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "occupancy.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=1)
            observed = list(iter_occupancy_replicates(path, batch_size=1))
        self.assertEqual([item[0] for item in observed], ["a", "b"])
        np.testing.assert_array_equal(observed[1][1]["rep2"], [4.0, 3.0])

    def test_matching_uses_replicate_specific_q_and_qbar(self) -> None:
        raw_q1 = np.array([1.0, 2.0, 4.0, 8.0])
        raw_q2 = np.array([8.0, 5.0, 3.0, 1.0])
        q1 = raw_q1 / raw_q1.mean()
        q2 = raw_q2 / raw_q2.mean()
        sampled = np.vstack(
            [np.append(q1, 0.0), np.append(q2, 0.0)]
        )

        metrics = calculate_matched_metrics(
            sampled, {"rep1": raw_q1, "rep2": raw_q2}
        )

        self.assertAlmostEqual(metrics["rep1_tasep_pcc"], 1.0)
        self.assertAlmostEqual(metrics["rep2_tasep_pcc"], 1.0)
        self.assertAlmostEqual(metrics["consensus_tasep_pcc"], 1.0)
        self.assertAlmostEqual(metrics["consensus_tasep_rmse_mean1"], 0.0)

    def test_unknown_occupancy_label_fails(self) -> None:
        with self.assertRaises(ValueError):
            occupancy_role("mean_occupancy")

    def test_negative_occupancy_fails_before_normalization(self) -> None:
        with self.assertRaises(ValueError):
            normalize_occupancy([1.0, -0.1, 2.0])


if __name__ == "__main__":
    unittest.main()
