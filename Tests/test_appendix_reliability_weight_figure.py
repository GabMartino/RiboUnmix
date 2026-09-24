from __future__ import annotations

import unittest

import numpy as np

from Datasets.data.plot_current_reliability_weight_audit import WeightReference
from analyses.create_appendix_reliability_weight_figure import (
    equal_dataset_quantile,
    select_reference_quantile_datasets,
)


class AppendixReliabilityWeightFigureTests(unittest.TestCase):
    def test_selection_is_deterministic_and_quantile_based(self) -> None:
        references = {
            name: WeightReference(tau=tau, normalization_median=0.5)
            for name, tau in {"d4": 4.0, "d2": 2.0, "d1": 1.0, "d3": 3.0}.items()
        }
        first = select_reference_quantile_datasets(references, (0.0, 1.0))
        second = select_reference_quantile_datasets(references, (0.0, 1.0))
        self.assertEqual(first["dataset"].tolist(), ["d1", "d4"])
        self.assertEqual(first.to_dict(orient="records"), second.to_dict(orient="records"))

    def test_mixture_quantile_gives_each_dataset_equal_mass(self) -> None:
        many_low_values = np.zeros(100, dtype=np.float64)
        one_high_value = np.asarray([10.0], dtype=np.float64)
        self.assertAlmostEqual(
            equal_dataset_quantile([many_low_values, one_high_value], 0.49), 0.0
        )
        self.assertAlmostEqual(
            equal_dataset_quantile([many_low_values, one_high_value], 0.51), 10.0
        )


if __name__ == "__main__":
    unittest.main()
