import unittest

import numpy as np
from scipy.spatial.distance import pdist, squareform

from analyses.analyze_cumulative_gamma_geometry import classical_mds, log_distance_squared


class GammaGeometryTests(unittest.TestCase):
    def test_common_reference_curve_cancels(self):
        values = np.random.default_rng(7).normal(size=(5, 100))
        curve = np.linspace(-5, 3, 100)
        np.testing.assert_allclose(log_distance_squared(values), log_distance_squared(values + curve), atol=1e-12)

    def test_transcript_weights_do_not_depend_on_length(self):
        short = np.array([[0, 0], [1, 1]])
        long = np.array([[0] * 100, [3] * 100])
        aggregate = np.sqrt((log_distance_squared(short) + log_distance_squared(long)) / 2)
        self.assertAlmostEqual(aggregate[0, 1], np.sqrt(5))

    def test_reciprocal_corrections_are_symmetric(self):
        logs = np.log([[1, 1], [2, .5], [.5, 2]])
        distances = log_distance_squared(logs)
        self.assertAlmostEqual(distances[0, 1], distances[0, 2])
        self.assertAlmostEqual(distances[0, 1], np.log(2) ** 2)

    def test_mds_recovers_two_dimensional_distances(self):
        x = np.array([[0, 0], [1, 1], [2, 1], [-2, 3]])
        distance = squareform(pdist(x))
        coordinates, _, info = classical_mds(distance)
        np.testing.assert_allclose(pdist(coordinates), pdist(x), atol=1e-12)
        self.assertAlmostEqual(info["fraction_2d"], 1)
        self.assertLess(info["normalized_distance_stress"], 1e-12)

    def test_degenerate_and_nonfinite_profiles(self):
        np.testing.assert_array_equal(log_distance_squared(np.zeros((3, 10))), np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            classical_mds(np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            log_distance_squared([[0, np.nan], [1, 2]])

    def test_aggregate_metric_is_euclidean(self):
        rng = np.random.default_rng(10)
        blocks = [rng.normal(size=(6, n)) for n in (3, 15, 100)]
        distance = np.sqrt(np.mean([log_distance_squared(x) for x in blocks], axis=0))
        concatenated = np.concatenate([x / np.sqrt(3 * x.shape[1]) for x in blocks], axis=1)
        np.testing.assert_allclose(squareform(distance, checks=False), pdist(concatenated), atol=1e-12)
        _, eigenvalues, _ = classical_mds(distance)
        self.assertGreater(eigenvalues.min(), -1e-10)


if __name__ == "__main__":
    unittest.main()
