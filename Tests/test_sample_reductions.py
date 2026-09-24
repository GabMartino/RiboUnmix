from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Models.RiboUnmixLightningModule import (
    reduce_per_sample_quantity,
    resolve_sample_reduction_mode,
)


def reduce(
    values: torch.Tensor,
    weights: torch.Tensor,
    datasets: torch.Tensor,
    transcripts: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    return reduce_per_sample_quantity(
        values=values,
        sample_weights=weights,
        dataset_ids=datasets,
        transcript_group_ids=transcripts,
        mode=mode,
    )


class SampleReductionTests(unittest.TestCase):
    def test_complete_rectangle_unit_weights_all_modes_are_equal(self) -> None:
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        weights = torch.ones(6)
        datasets = torch.tensor([0, 1, 2, 0, 1, 2])
        transcripts = torch.tensor([0, 0, 0, 1, 1, 1])
        results = {
            mode: reduce(values, weights, datasets, transcripts, mode)
            for mode in (
                "global_weighted",
                "dataset_balanced",
                "transcript_balanced",
            )
        }
        for result in results.values():
            torch.testing.assert_close(result, torch.tensor(3.5))

    def test_variable_transcript_support_changes_outer_mass(self) -> None:
        values = torch.tensor([1.0, 1.0, 1.0, 1.0, 3.0, 3.0])
        weights = torch.ones(6)
        datasets = torch.tensor([0, 1, 2, 3, 0, 1])
        transcripts = torch.tensor([0, 0, 0, 0, 1, 1])
        torch.testing.assert_close(
            reduce(values, weights, datasets, transcripts, "global_weighted"),
            torch.tensor(5.0 / 3.0),
        )
        torch.testing.assert_close(
            reduce(values, weights, datasets, transcripts, "transcript_balanced"),
            torch.tensor(2.0),
        )

    def test_equal_support_with_variable_total_weights(self) -> None:
        values = torch.tensor([1.0, 1.0, 3.0, 3.0])
        weights = torch.tensor([1.5, 1.5, 0.5, 0.5])
        datasets = torch.tensor([0, 1, 0, 1])
        transcripts = torch.tensor([0, 0, 1, 1])
        torch.testing.assert_close(
            reduce(values, weights, datasets, transcripts, "global_weighted"),
            torch.tensor(1.5),
        )
        torch.testing.assert_close(
            reduce(values, weights, datasets, transcripts, "transcript_balanced"),
            torch.tensor(2.0),
        )

    def test_variable_dataset_sizes_are_dataset_balanced(self) -> None:
        values = torch.tensor([1.0, 1.0, 1.0, 1.0, 5.0])
        weights = torch.ones(5)
        datasets = torch.tensor([0, 0, 0, 0, 1])
        transcripts = torch.arange(5)
        torch.testing.assert_close(
            reduce(values, weights, datasets, transcripts, "global_weighted"),
            torch.tensor(1.8),
        )
        torch.testing.assert_close(
            reduce(values, weights, datasets, transcripts, "dataset_balanced"),
            torch.tensor(3.0),
        )

    def test_local_reliability_weights_remain_active(self) -> None:
        result = reduce(
            torch.tensor([1.0, 5.0]),
            torch.tensor([3.0, 1.0]),
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            "transcript_balanced",
        )
        torch.testing.assert_close(result, torch.tensor(2.0))

    def test_weights_above_one_are_preserved(self) -> None:
        values = torch.tensor([1.0, 5.0])
        weights = torch.tensor([1.25, 3.0])
        result = reduce(
            values,
            weights,
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            "transcript_balanced",
        )
        torch.testing.assert_close(result, (values * weights).sum() / weights.sum())
        self.assertEqual(float(weights[0]), 1.25)
        self.assertEqual(float(weights[1]), 3.0)

    def test_all_reducers_are_permutation_invariant(self) -> None:
        values = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])
        weights = torch.tensor([1.25, 0.75, 3.0, 2.0, 0.5])
        datasets = torch.tensor([0, 1, 0, 2, 1])
        transcripts = torch.tensor([0, 0, 1, 1, 1])
        order = list(range(values.numel()))
        random.Random(17).shuffle(order)
        permutation = torch.tensor(order)
        for mode in (
            "global_weighted",
            "dataset_balanced",
            "transcript_balanced",
        ):
            original = reduce(values, weights, datasets, transcripts, mode)
            permuted = reduce(
                values[permutation],
                weights[permutation],
                datasets[permutation],
                transcripts[permutation],
                mode,
            )
            torch.testing.assert_close(original, permuted)

    def test_transcript_balanced_autograd_matches_manual_gradient(self) -> None:
        values = torch.tensor([1.0, 5.0, 2.0], requires_grad=True)
        result = reduce(
            values,
            torch.tensor([3.0, 1.0, 2.0]),
            torch.tensor([0, 1, 0]),
            torch.tensor([0, 0, 1]),
            "transcript_balanced",
        )
        result.backward()
        torch.testing.assert_close(
            values.grad,
            torch.tensor([3.0 / 8.0, 1.0 / 8.0, 1.0 / 2.0]),
        )

    def test_mixed_precision_accumulates_in_float32(self) -> None:
        values = torch.tensor([1.0, 5.0], dtype=torch.float16, requires_grad=True)
        result = reduce(
            values,
            torch.tensor([1.25, 3.0], dtype=torch.float16),
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            "transcript_balanced",
        )
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(result)))
        result.backward()
        self.assertTrue(bool(torch.isfinite(values.grad).all()))

    def test_zero_weight_is_rejected_by_filtered_pipeline_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            reduce(
                torch.tensor([1.0, 2.0]),
                torch.tensor([1.0, 0.0]),
                torch.tensor([0, 1]),
                torch.tensor([0, 1]),
                "global_weighted",
            )

    def test_legacy_configuration_mapping_and_conflicts(self) -> None:
        self.assertEqual(
            resolve_sample_reduction_mode(
                SimpleNamespace(dataset_balanced_loss=False)
            ),
            "global_weighted",
        )
        self.assertEqual(
            resolve_sample_reduction_mode(
                SimpleNamespace(dataset_balanced_loss=True)
            ),
            "dataset_balanced",
        )
        self.assertEqual(
            resolve_sample_reduction_mode(SimpleNamespace()),
            "transcript_balanced",
        )
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            resolve_sample_reduction_mode(
                SimpleNamespace(
                    sample_reduction="transcript_balanced",
                    dataset_balanced_loss=True,
                )
            )


if __name__ == "__main__":
    unittest.main()
