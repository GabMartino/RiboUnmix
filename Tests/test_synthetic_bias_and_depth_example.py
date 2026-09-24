"""Data identity, exact replica means, and plotting-scale checks."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from analyses.plot_synthetic_bias_and_depth_example import (
    DEPTHS, kinetic_bias_example, prepare_data, read_transcript, replica_profile, scale_counts,
    set_count_axis,
)


class TestSyntheticBiasAndDepthExample(unittest.TestCase):
    def test_linear_count_axis_retains_zero_and_half_counts(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        try:
            counts = np.array([0., .5, 1., 10., 100.])
            line, = ax.plot(counts)
            set_count_axis(ax, counts.max())
            self.assertEqual(ax.get_yscale(), "linear")
            transformed = ax.yaxis.get_transform().transform(counts)
            self.assertTrue(np.isfinite(transformed).all())
            self.assertTrue((np.diff(transformed) > 0).all())
            np.testing.assert_array_equal(line.get_ydata(), counts)
            self.assertIn(0., ax.get_yticks())
            self.assertEqual(ax.get_ylim()[0], 0.)
            self.assertGreater(ax.get_ylim()[1], counts.max())
        finally:
            plt.close(fig)

    def test_multiplier_is_one_plus_added_bias_without_renormalization(self):
        k = np.array([.5, 1., 1.5])
        product, bias, sites = kinetic_bias_example(k, np.array([0., 4., 0.]))
        np.testing.assert_array_equal(bias, [1., 5., 1.])
        np.testing.assert_array_equal(product, [.5, 5., 1.5])
        np.testing.assert_array_equal(sites, [False, True, False])
        self.assertNotAlmostEqual(product.mean(), 1.)

    def test_replica_mean_does_not_use_the_integerized_source_mean(self):
        rows = {"rep1": np.array([0., 1.]), "rep2": np.array([1., 0.]), "mean": np.array([1., 0.])}
        np.testing.assert_array_equal(replica_profile(rows, "mean"), [.5, .5])
        np.testing.assert_array_equal(replica_profile(rows, "rep1"), [0., 1.])

    def test_depth_scaling_preserves_zeros_and_does_not_force_mean_one(self):
        scaled = scale_counts(np.array([0., 1., 0.]), .25)
        np.testing.assert_array_equal(scaled, [0., 4., 0.])
        self.assertNotAlmostEqual(scaled.mean(), 1.)

    def test_panel_b_preserves_raw_count_scale_at_every_depth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(data_root=root, transcript="example", bias="3prime_cc",
                                   replicate="mean", left_mode="kinetics", depth_condition="unbiased")
            metadata = {"riboart.source_run_fingerprint": "test-run"}
            replies = [
                ({"kinetics_target": np.ones(3)},
                 {**metadata, "riboart.read_sampling_applied": "false"}),
                ({role: np.zeros(3) for role in ("rep1", "rep2", "mean")}, metadata),
            ]
            for _, depth in DEPTHS:
                replies.append((
                    {"rep1": np.array([0., 1., 3.]), "rep2": np.array([0., 2., 5.]),
                     "mean": np.array([0., 2., 4.])},
                    {**metadata, "riboart.counts_per_codon_unbiased_baseline": str(depth),
                     "riboart.observation_model": "negative_binomial_NB2",
                     "riboart.negative_binomial_dispersion_alpha": "0.1",
                     "riboart.sequence_bias_enabled": "false"}))
            with patch("analyses.plot_synthetic_bias_and_depth_example.read_transcript", side_effect=replies), \
                 patch("analyses.plot_synthetic_bias_and_depth_example.sha256", return_value="test-hash"):
                data = prepare_data(args)
            for slug, _ in DEPTHS:
                np.testing.assert_array_equal(data["depth_profiles"][slug], [0., 1.5, 4.])
            np.testing.assert_array_equal(data["left"][0], np.ones(3))

    def test_alignment_errors_are_not_silently_truncated(self):
        with self.assertRaises(ValueError):
            kinetic_bias_example(np.ones(3), np.zeros(2))

    def test_parquet_reader_selects_exact_transcript_and_sample_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"counts.parquet"
            table = pa.table({"transcript_id": ["a", "b", "b", "b", "c"],
                              "sample": ["mean", "gt_rep1", "gt_rep2", "mean", "mean"],
                              "rib_profile": [[9, 9], [0, 1], [1, 0], [1, 0], [9, 9]]})
            pq.write_table(table, path, row_group_size=2)
            profiles, _ = read_transcript(path, "b")
            self.assertEqual(set(profiles), {"rep1", "rep2", "mean"})
            np.testing.assert_array_equal(replica_profile(profiles, "mean"), [.5, .5])
            with self.assertRaisesRegex(ValueError, "absent"):
                read_transcript(path, "missing")


if __name__ == "__main__":
    unittest.main()
