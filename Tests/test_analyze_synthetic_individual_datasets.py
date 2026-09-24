from __future__ import annotations

import numpy as np
import pandas as pd
import unittest

from analyses.analyze_synthetic_individual_datasets import (
    AVAILABLE_DATASET_ORDER,
    DATASET_ORDER,
    DEPTHS,
    DISPLAY_NAMES,
    box_statistics,
    centered_correlation_matrix,
    headline_mean_ci_summary,
    metric_summaries,
)


class SyntheticIndividualDatasetTests(unittest.TestCase):
    def test_unbiased_observation_condition_is_not_a_displayed_dataset(self) -> None:
        self.assertIn("artificial_ground_truth", AVAILABLE_DATASET_ORDER)
        self.assertNotIn("artificial_ground_truth", DATASET_ORDER)
        self.assertEqual(len(DATASET_ORDER), 10)

    def test_centered_correlation_matrix_recovers_known_relationships(self) -> None:
        x = np.arange(1.0, 7.0)
        profiles = np.vstack([x, 3.0 * x + 4.0, x[::-1], np.ones_like(x)])

        observed = centered_correlation_matrix(profiles)

        self.assertTrue(np.isclose(observed[0, 1], 1.0))
        self.assertTrue(np.isclose(observed[0, 2], -1.0))
        self.assertTrue(np.isnan(observed[0, 3]))
        self.assertTrue(np.allclose(observed, observed.T, equal_nan=True))

    def test_box_statistics_uses_requested_percentiles_and_ignores_nan(self) -> None:
        values = np.concatenate([np.arange(101, dtype=float), [np.nan]])

        observed = box_statistics(values, "test")

        self.assertEqual(observed["label"], "test")
        self.assertEqual(observed["whislo"], 5.0)
        self.assertEqual(observed["q1"], 25.0)
        self.assertEqual(observed["med"], 50.0)
        self.assertEqual(observed["q3"], 75.0)
        self.assertEqual(observed["whishi"], 95.0)

    def test_every_dataset_has_a_distinct_display_label(self) -> None:
        labels = [DISPLAY_NAMES[dataset] for dataset in DATASET_ORDER]

        self.assertEqual(len(labels), len(DATASET_ORDER))
        self.assertEqual(len(set(labels)), len(labels))

    def test_metric_summaries_keeps_datasets_and_representations_separate(self) -> None:
        rows = []
        for dataset_number, dataset in enumerate(DATASET_ORDER):
            for transcript_number in range(4):
                base = 0.1 * dataset_number + 0.01 * transcript_number
                rows.append(
                    {
                        "depth": "2_per_codon",
                        "nominal_reads_per_codon": 2.0,
                        "dataset": dataset,
                        "transcript_id": f"tx{transcript_number}",
                        "replicate_pcc": base,
                        "rep1_kinetic_pcc": base + 0.1,
                        "rep2_kinetic_pcc": base + 0.2,
                        "consensus_kinetic_pcc": base + 0.3,
                    }
                )

        replica, kinetic = metric_summaries(pd.DataFrame(rows))

        self.assertEqual(len(replica), len(DATASET_ORDER))
        self.assertEqual(len(kinetic), 3 * len(DATASET_ORDER))
        self.assertEqual(
            set(kinetic["representation"]),
            {"Replica 1", "Replica 2", "Arithmetic mean"},
        )
        first_consensus = kinetic.loc[
            (kinetic["dataset"] == DATASET_ORDER[0])
            & (kinetic["representation"] == "Arithmetic mean"),
            "median",
        ].item()
        self.assertTrue(np.isclose(first_consensus, 0.315))

    def test_headline_intervals_use_bias_units_and_dataset_jackknife(self) -> None:
        replica_rows = []
        kinetic_rows = []
        cross_rows = []
        for depth, nominal, _ in DEPTHS:
            for index, dataset in enumerate(DATASET_ORDER):
                value = 0.1 + 0.01 * index
                replica_rows.append(
                    {"depth": depth, "dataset": dataset, "median": value}
                )
                for representation, offset in (
                    ("Replica 1", 0.1),
                    ("Replica 2", 0.2),
                    ("Arithmetic mean", 0.3),
                ):
                    kinetic_rows.append(
                        {
                            "depth": depth,
                            "dataset": dataset,
                            "representation": representation,
                            "median": value + offset,
                        }
                    )
            for left, dataset_a in enumerate(DATASET_ORDER):
                for right in range(left + 1, len(DATASET_ORDER)):
                    cross_rows.append(
                        {
                            "depth": depth,
                            "dataset_a": dataset_a,
                            "dataset_b": DATASET_ORDER[right],
                            "median_pcc": 0.2 + 0.01 * (left + right),
                        }
                    )

        observed = headline_mean_ci_summary(
            pd.DataFrame(replica_rows),
            pd.DataFrame(kinetic_rows),
            pd.DataFrame(cross_rows),
        )

        self.assertEqual(len(observed), 4 * len(DEPTHS))
        self.assertTrue((observed["design_units"] == len(DATASET_ORDER)).all())
        cross = observed.loc[observed["metric"] == "cross_dataset_agreement"]
        self.assertTrue(cross["interval_method"].str.contains("jackknife").all())
        self.assertTrue((cross["ci95_half_width"] > 0.0).all())


if __name__ == "__main__":
    unittest.main()
