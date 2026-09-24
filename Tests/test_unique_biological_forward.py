from __future__ import annotations

import copy
import unittest

import torch
from torch.nn.utils.rnn import pack_padded_sequence

from Models.RiboUnmixModel.SharedProfileModel import SharedProfileModel
from Models.RiboUnmixModel.RiboUnmixModel import RiboUnmixModel


def _packed(values: torch.Tensor, lengths: torch.Tensor):
    return pack_padded_sequence(
        values,
        lengths.cpu(),
        batch_first=True,
        enforce_sorted=True,
    )


class UniqueBiologicalForwardTests(unittest.TestCase):
    def _helper(self, biological_model: SharedProfileModel) -> RiboUnmixModel:
        helper = RiboUnmixModel.__new__(RiboUnmixModel)
        torch.nn.Module.__init__(helper)
        helper.biological_model = biological_model
        return helper

    def test_unique_gather_preserves_outputs_gradients_and_optimizer_update(self) -> None:
        torch.manual_seed(7)
        base = SharedProfileModel(
            {
                "input_size": 5,
                "hidden_size": 4,
                "num_layers": 2,
                "dropout": 0.0,
                "eps": 1.0e-8,
                "init_local_hazard_factor": 1.0,
                "init_local_hazard_weight_std": 1.0e-2,
            }
        )
        repeated_model = copy.deepcopy(base)
        unique_model = copy.deepcopy(base)
        repeated_model.train()
        unique_model.train()

        unique_values = torch.randn(2, 4, 5)
        unique_lengths = torch.tensor([4, 3], dtype=torch.long)
        unique_mask = torch.arange(4).unsqueeze(0) < unique_lengths.unsqueeze(1)
        group_index = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
        repeated_values = unique_values.index_select(0, group_index)
        repeated_lengths = unique_lengths.index_select(0, group_index)
        repeated_mask = unique_mask.index_select(0, group_index)

        repeated_helper = self._helper(repeated_model)
        unique_helper = self._helper(unique_model)
        repeated, _ = repeated_helper._forward_biological_unique_transcripts(
            x_packed=_packed(repeated_values, repeated_lengths),
            mask_b=repeated_mask,
            transcript_group_index=None,
        )
        gathered, gathered_inputs = unique_helper._forward_biological_unique_transcripts(
            x_packed=_packed(unique_values, unique_lengths),
            mask_b=repeated_mask,
            transcript_group_index=group_index,
        )

        torch.testing.assert_close(
            gathered_inputs,
            repeated_values * repeated_mask.unsqueeze(-1),
        )
        for name in ("w_raw", "w_norm", "J", "lambda_bio", "rho", "L_bio"):
            torch.testing.assert_close(gathered[name], repeated[name], rtol=1e-6, atol=1e-7)

        # Give every transcript--dataset row a different downstream coefficient.
        # The gather backward must sum all of these contributions at the unique
        # transcript output before applying the biological chain rule.
        coefficients = torch.tensor([0.3, 1.7, -0.2, 0.9, 2.1]).reshape(-1, 1)
        repeated_loss = (repeated["L_bio"] * coefficients).sum()
        unique_loss = (gathered["L_bio"] * coefficients).sum()
        torch.testing.assert_close(unique_loss, repeated_loss, rtol=1e-6, atol=1e-7)
        repeated_loss.backward()
        unique_loss.backward()

        for (name_a, parameter_a), (name_b, parameter_b) in zip(
            repeated_model.named_parameters(),
            unique_model.named_parameters(),
            strict=True,
        ):
            self.assertEqual(name_a, name_b)
            self.assertIsNotNone(parameter_a.grad, name_a)
            self.assertIsNotNone(parameter_b.grad, name_b)
            torch.testing.assert_close(
                parameter_b.grad,
                parameter_a.grad,
                rtol=2e-5,
                atol=2e-6,
                msg=lambda message, name=name_a: f"{name}: {message}",
            )

        repeated_optimizer = torch.optim.SGD(repeated_model.parameters(), lr=0.05)
        unique_optimizer = torch.optim.SGD(unique_model.parameters(), lr=0.05)
        repeated_optimizer.step()
        unique_optimizer.step()
        for parameter_a, parameter_b in zip(
            repeated_model.parameters(),
            unique_model.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(parameter_b, parameter_a, rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
