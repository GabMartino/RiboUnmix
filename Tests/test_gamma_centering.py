from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Models.RiboUnmixModel.RiboUnmixModel import RiboUnmixModel


class _DatasetScore(nn.Module):
    """Deterministic stand-in for the dataset branch used by centering tests."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.offset = nn.Parameter(torch.arange(8, dtype=torch.float32) * 4.0)
        self.slope = nn.Parameter(torch.arange(8, dtype=torch.float32) + 1.0)

    def forward(
        self,
        *,
        dataset_ids: torch.Tensor,
        mask: torch.Tensor,
        codon_ids: torch.Tensor,
        position_features: torch.Tensor,
        sequence_features: torch.Tensor | None,
        compute_log_sigma: bool,
        embedding_center_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.calls += 1
        del codon_ids, position_features, sequence_features
        del compute_log_sigma, embedding_center_ids
        position = torch.arange(mask.shape[1], device=mask.device).float()
        score = self.offset[dataset_ids].reshape(-1, 1) + (
            self.slope[dataset_ids].reshape(-1, 1) * position.reshape(1, -1)
        )
        return {"gamma_raw": score * mask.float()}


def _fixed_reference_helper(
    reference_ids: list[int],
    reference_weights: list[float],
    *,
    minimum_datasets: int = 2,
) -> RiboUnmixModel:
    model = RiboUnmixModel.__new__(RiboUnmixModel)
    nn.Module.__init__(model)
    model.eps = 1.0e-8
    model.gamma_cross_dataset_centering_enabled = True
    model.gamma_dataset_constant_scale_gauge = "geometric_mean_one"
    model.gamma_reference_minimum_datasets = minimum_datasets
    model.gamma_reference_chunk_size = 32
    model.gamma_log_init = 0.0
    model.dataset_bias_model = _DatasetScore()
    model.register_buffer(
        "gamma_reference_dataset_ids",
        torch.tensor(reference_ids, dtype=torch.long),
    )
    model.register_buffer(
        "gamma_reference_weights",
        torch.tensor(reference_weights, dtype=torch.float32),
    )
    model.register_buffer(
        "gamma_selected_dataset_ids",
        torch.tensor(reference_ids, dtype=torch.long),
    )
    model.eval()
    return model


def _center(
    model: RiboUnmixModel,
    requested_dataset_ids: list[int],
) -> dict[str, torch.Tensor]:
    batch_size = len(requested_dataset_ids)
    positions = 3
    dataset_ids = torch.tensor(requested_dataset_ids, dtype=torch.long)
    mask = torch.ones(batch_size, positions, dtype=torch.bool)
    raw = model.dataset_bias_model(
        dataset_ids=dataset_ids,
        mask=mask,
        codon_ids=torch.zeros(batch_size, positions, dtype=torch.long),
        position_features=torch.zeros(batch_size, positions, 1),
        sequence_features=None,
        compute_log_sigma=False,
        embedding_center_ids=model.gamma_selected_dataset_ids,
    )["gamma_raw"]
    return model._center_log_gamma_fixed_reference(
        raw,
        mask_b=mask,
        sample_ids=["same_transcript"] * batch_size,
        id_datasets=dataset_ids,
        codon_ids=torch.zeros(batch_size, positions, dtype=torch.long),
        position_features=torch.zeros(batch_size, positions, 1),
        dataset_bias_sequence_features=None,
        biological_sequence_features=torch.zeros(batch_size, positions, 1),
    )


class FixedReferenceGammaCenteringTests(unittest.TestCase):
    def test_explicit_mode_activates_without_redundant_enabled_flag(self) -> None:
        model = RiboUnmixModel.__new__(RiboUnmixModel)
        nn.Module.__init__(model)
        model.eps = 1.0e-8
        model.selected_dataset_names = ("dataset_a", "dataset_b")
        model.selected_dataset_ids = (0, 1)
        model._configure_gamma_centering(
            {
                "gamma_centering": {
                    "mode": "fixed_reference",
                    "reference": {"weighting": "equal", "minimum_datasets": 2},
                }
            },
            reference_dataset_names=("dataset_a", "dataset_b"),
            reference_dataset_ids=(0, 1),
            reference_dataset_quality_weights=(1.0, 0.5),
        )

        self.assertEqual(model.gamma_centering_mode, "fixed_reference")
        self.assertTrue(model.gamma_cross_dataset_centering_enabled)

    def test_single_requested_dataset_uses_complete_reference_panel(self) -> None:
        # The complete reference panel is used even though only dataset 0 is
        # physically requested. The remaining shape has positional log mean 0.
        model = _fixed_reference_helper([0, 1], [1.0, 3.0])
        centered = _center(model, [0])

        torch.testing.assert_close(
            centered["log_gamma"],
            torch.tensor([[0.75, 0.0, -0.75]]),
        )
        self.assertLess(
            float(centered["positional_constraint_error"].max().detach()),
            1e-6,
        )
        self.assertTrue(bool(centered["applied"].all()))
        self.assertEqual(float(centered["reference_count"][0]), 2.0)
        self.assertTrue(bool(centered["all_requested_in_reference"].all()))

    def test_center_is_independent_of_requested_partner_rows(self) -> None:
        model = _fixed_reference_helper([0, 1], [1.0, 3.0])
        singleton = _center(model, [0])
        paired = _center(model, [0, 1])

        torch.testing.assert_close(singleton["log_gamma"][0], paired["log_gamma"][0])
        torch.testing.assert_close(
            paired["log_gamma"][1],
            torch.tensor([-0.25, 0.0, 0.25]),
        )

    def test_complete_eval_group_reuses_requested_reference_scores_only_in_eval(self) -> None:
        model = _fixed_reference_helper([0, 1], [1.0, 3.0])
        _center(model, [0, 1])
        self.assertEqual(model.dataset_bias_model.calls, 1)

        model.dataset_bias_model.calls = 0
        model.train()
        _center(model, [0, 1])
        self.assertEqual(model.dataset_bias_model.calls, 2)

    def test_joint_weighted_and_positional_gauges(self) -> None:
        model = _fixed_reference_helper([0, 1], [1.0, 3.0])
        centered = _center(model, [0, 1])
        log_gamma = centered["log_gamma"]
        pi = torch.tensor([0.25, 0.75]).reshape(-1, 1)

        self.assertLess(
            float((pi * log_gamma).sum(dim=0).abs().max().detach()),
            1e-6,
        )
        self.assertLess(float(log_gamma.mean(dim=1).abs().max().detach()), 1e-6)
        torch.testing.assert_close(
            torch.exp((pi * log_gamma).sum(dim=0)),
            torch.ones(3),
        )
        torch.testing.assert_close(
            torch.exp(log_gamma.mean(dim=1)),
            torch.ones(2),
        )

    def test_out_of_reference_request_has_positional_geometric_mean_one(self) -> None:
        model = _fixed_reference_helper([0, 1], [1.0, 3.0])
        reference = _center(model, [0, 1])
        outside = _center(model, [2])
        torch.testing.assert_close(
            outside["log_gamma"],
            torch.tensor([[-1.25, 0.0, 1.25]]),
        )
        self.assertLess(float(outside["log_gamma"].mean().abs().detach()), 1e-6)
        torch.testing.assert_close(
            outside["gamma_center"][0],
            reference["gamma_center"][0],
        )
        self.assertFalse(bool(outside["all_requested_in_reference"].any()))

    def test_row_order_and_reference_chunk_size_are_invariant(self) -> None:
        model = _fixed_reference_helper([0, 1], [1.0, 3.0])
        model.gamma_reference_chunk_size = 1
        first = _center(model, [0, 2, 1])["log_gamma"]
        model.gamma_reference_chunk_size = 2
        permuted = _center(model, [1, 0, 2])["log_gamma"]
        torch.testing.assert_close(first, permuted[[1, 2, 0]])

    def test_positional_gauge_leaves_mass_normalized_prediction_unchanged(self) -> None:
        model = _fixed_reference_helper([0, 1], [1.0, 3.0])
        centered = _center(model, [0, 1])
        raw = torch.stack((torch.tensor([0.0, 1.0, 2.0]), torch.tensor([4.0, 6.0, 8.0])))
        cross_only = raw - centered["gamma_center"]
        final = centered["log_gamma"]
        biological_shape = torch.tensor([[0.5, 1.0, 2.0], [0.5, 1.0, 2.0]])

        def normalized_mu(log_gamma: torch.Tensor) -> torch.Tensor:
            shape = torch.exp(log_gamma) * biological_shape
            return shape / shape.mean(dim=1, keepdim=True)

        torch.testing.assert_close(
            normalized_mu(cross_only),
            normalized_mu(final),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_arbitrary_dataset_transcript_constants_are_gauged_out(self) -> None:
        model = _fixed_reference_helper([0, 1], [1.0, 3.0])
        baseline = _center(model, [0, 1])["log_gamma"]
        with torch.no_grad():
            model.dataset_bias_model.offset[0].add_(7.0)
            model.dataset_bias_model.offset[1].sub_(2.0)
        shifted = _center(model, [0, 1])["log_gamma"]
        torch.testing.assert_close(shifted, baseline, rtol=1e-6, atol=1e-6)

        biological_shape = torch.tensor(
            [[0.5, 1.0, 2.0], [0.5, 1.0, 2.0]]
        )

        def normalized_mu(log_gamma: torch.Tensor) -> torch.Tensor:
            shape = torch.exp(log_gamma) * biological_shape
            return shape / shape.mean(dim=1, keepdim=True)

        torch.testing.assert_close(
            normalized_mu(shifted),
            normalized_mu(baseline),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_gradients_pass_through_requested_and_reference_evaluations(self) -> None:
        model = _fixed_reference_helper([0, 1], [1.0, 3.0])
        centered = _center(model, [0, 1, 2])
        loss = centered["log_gamma"].square().sum()
        loss.backward()
        slope_grad = model.dataset_bias_model.slope.grad
        self.assertIsNotNone(slope_grad)
        self.assertTrue(bool(torch.isfinite(slope_grad).all()))
        self.assertGreater(float(slope_grad.abs().sum()), 0.0)

    def test_reference_panel_below_minimum_skips_centering(self) -> None:
        model = _fixed_reference_helper([0], [1.0], minimum_datasets=2)
        centered = _center(model, [0])

        torch.testing.assert_close(
            centered["log_gamma"], torch.tensor([[0.0, 1.0, 2.0]])
        )
        self.assertFalse(bool(centered["applied"].any()))
        self.assertEqual(float(centered["reference_count"][0]), 1.0)


class BatchGroupedGammaCenteringTests(unittest.TestCase):
    def test_batch_grouped_mode_applies_both_gauges(self) -> None:
        model = RiboUnmixModel.__new__(RiboUnmixModel)
        nn.Module.__init__(model)
        model.eps = 1.0e-8
        model.gamma_cross_dataset_centering_enabled = True
        model.gamma_dataset_constant_scale_gauge = "geometric_mean_one"
        model.gamma_centering_weighting = "quality_rank"
        model.gamma_centering_quality_rank_power = 1.0
        raw = torch.tensor([[1.0, 2.0, 4.0], [3.0, 7.0, 8.0]])
        centered = model._center_log_gamma_across_transcripts(
            raw,
            mask_b=torch.ones_like(raw, dtype=torch.bool),
            sample_ids=["t", "t"],
            id_datasets=torch.tensor([0, 1]),
            dataset_quality_weights=torch.tensor([1.0, 3.0]),
        )
        log_gamma = centered["log_gamma"]
        pi = torch.tensor([0.25, 0.75]).reshape(-1, 1)
        self.assertLess(
            float((pi * log_gamma).sum(dim=0).abs().max().detach()),
            1e-6,
        )
        self.assertLess(float(log_gamma.mean(dim=1).abs().max().detach()), 1e-6)
        self.assertLess(
            float(centered["constraint_error"].max().detach()),
            1e-6,
        )
        self.assertLess(
            float(centered["positional_constraint_error"].max().detach()),
            1e-6,
        )


if __name__ == "__main__":
    unittest.main()
