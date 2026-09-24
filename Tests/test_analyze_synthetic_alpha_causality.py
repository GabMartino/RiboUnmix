from __future__ import annotations

import unittest
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from analyses.analyze_synthetic_alpha_causality import (
    _classification,
    _discover_latest_complete_group,
)


def _miss_rates(a: float, b: float, c: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "condition": [
                "A_standard",
                "B_beta1_decoupled",
                "C_fixed_true_alpha",
            ],
            "strong_site_miss_rate": [a, b, c],
        }
    )


class AlphaCausalityInterpretationTests(unittest.TestCase):
    def test_outcome_one(self) -> None:
        self.assertIn("Outcome 1", _classification(_miss_rates(0.10, 0.03, 0.02)))

    def test_outcome_two(self) -> None:
        self.assertIn("Outcome 2", _classification(_miss_rates(0.10, 0.09, 0.02)))

    def test_outcome_three(self) -> None:
        self.assertIn("Outcome 3", _classification(_miss_rates(0.10, 0.09, 0.11)))

    def test_outcome_four(self) -> None:
        self.assertIn("Outcome 4", _classification(_miss_rates(0.10, 0.02, 0.09)))

    def test_latest_complete_group_is_discovered_without_cli_arguments(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            old_prefix = "matched_old_"
            new_prefix = "matched_new_"
            for prefix in (old_prefix, new_prefix):
                for condition in (
                    "A_standard",
                    "B_beta1_decoupled",
                    "C_fixed_true_alpha",
                ):
                    (root / f"{prefix}{condition}").mkdir()
            # The incomplete newest-looking prefix must never be selected.
            (root / "incomplete_A_standard").mkdir()
            for condition in (
                "A_standard",
                "B_beta1_decoupled",
                "C_fixed_true_alpha",
            ):
                os.utime(root / f"{new_prefix}{condition}", (2_000_000_000, 2_000_000_000))

            selected, runs = _discover_latest_complete_group(root)
            self.assertEqual(selected, new_prefix)
            self.assertEqual(set(runs), {
                "A_standard",
                "B_beta1_decoupled",
                "C_fixed_true_alpha",
            })


if __name__ == "__main__":
    unittest.main()
