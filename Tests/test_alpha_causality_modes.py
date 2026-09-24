from __future__ import annotations

from types import SimpleNamespace
import math
import unittest

import torch
from torch.nn.utils.rnn import pack_padded_sequence

from Models.RiboUnmixModel.RiboUnmixModel import RiboUnmixModel
from Models.RiboUnmixLightningModule import (
    NegativeBinomialProfileLoss,
    RiboUnmixLightningModule,
)


class DecoupledNBGradientTests(unittest.TestCase):
    def _details(self, log_mu: torch.Tensor, log_alpha: torch.Tensor):
        return NegativeBinomialProfileLoss(
            sequence_reduction="mean",
            experiment_mode="decoupled_nb_mean_gradient",
            nb_mean_gradient_beta=1.0,
        )(
            mu_phys=log_mu.exp(),
            log_sigma=log_alpha,
            y_true=torch.tensor([[13.0]], dtype=torch.float32),
            mask=torch.tensor([[True]]),
            return_details=True,
        )

    def test_mean_and_alpha_losses_have_disjoint_gradients(self) -> None:
        log_mu = torch.tensor([[1.7]], requires_grad=True)
        log_alpha = torch.tensor([[0.4]], requires_grad=True)
        details = self._details(log_mu, log_alpha)

        details["mean_reweighted_nll_per_sample"].sum().backward(retain_graph=True)
        self.assertIsNotNone(log_mu.grad)
        self.assertGreater(float(log_mu.grad.abs().sum()), 0.0)
        self.assertIsNone(log_alpha.grad)

        log_mu.grad = None
        details["alpha_nll_per_sample"].sum().backward()
        self.assertIsNone(log_mu.grad)
        self.assertIsNotNone(log_alpha.grad)
        self.assertGreater(float(log_alpha.grad.abs().sum()), 0.0)

    def test_beta_one_mean_gradient_is_mu_minus_y(self) -> None:
        log_mu = torch.tensor([[1.7]], requires_grad=True)
        log_alpha = torch.tensor([[0.4]], requires_grad=True)
        details = self._details(log_mu, log_alpha)
        details["mean_reweighted_nll_per_sample"].sum().backward()
        expected = float(log_mu.detach().exp().item() - 13.0)
        self.assertAlmostEqual(float(log_mu.grad.item()), expected, places=4)

    def test_standard_mode_matches_ordinary_nb_gradient(self) -> None:
        y = torch.tensor([[1.0, 7.0]])
        mask = torch.tensor([[True, True]])
        log_mu_a = torch.tensor([[0.2, 1.1]], requires_grad=True)
        log_alpha_a = torch.tensor([[-1.0, -0.2]], requires_grad=True)
        loss_a = NegativeBinomialProfileLoss(
            sequence_reduction="mean",
            experiment_mode="standard_nb",
            nb_mean_gradient_beta=0.0,
        )(
            mu_phys=log_mu_a.exp(),
            log_sigma=log_alpha_a,
            y_true=y,
            mask=mask,
        ).sum()
        loss_a.backward()

        log_mu_b = log_mu_a.detach().clone().requires_grad_(True)
        log_alpha_b = log_alpha_a.detach().clone().requires_grad_(True)
        r = torch.exp(-log_alpha_b)
        ordinary = (
            torch.lgamma(r)
            - torch.lgamma(y + r)
            + torch.lgamma(y + 1.0)
            - r * torch.log(r)
            - y * log_mu_b
            + (r + y) * torch.log(r + log_mu_b.exp())
        ).mean()
        ordinary.backward()
        torch.testing.assert_close(loss_a.detach(), ordinary.detach())
        torch.testing.assert_close(log_mu_a.grad, log_mu_b.grad)
        torch.testing.assert_close(log_alpha_a.grad, log_alpha_b.grad)

    def test_modes_keep_shapes_masks_and_sequence_reduction(self) -> None:
        y = torch.tensor([[2.0, 1_000.0, 5.0]])
        mu = torch.tensor([[1.5, 1.0e6, 4.0]])
        mask = torch.tensor([[True, False, True]])
        modes = (
            ("standard_nb", 0.0, torch.tensor([[-1.0, 0.0, -0.5]])),
            ("decoupled_nb_mean_gradient", 1.0, torch.tensor([[-1.0, 0.0, -0.5]])),
            ("fixed_alpha", 0.0, torch.full((1, 3), math.log(0.1))),
        )
        outputs = []
        for mode, beta, log_alpha in modes:
            outputs.append(
                NegativeBinomialProfileLoss(
                    sequence_reduction="mean",
                    experiment_mode=mode,
                    nb_mean_gradient_beta=beta,
                )(
                    mu_phys=mu,
                    log_sigma=log_alpha,
                    y_true=y,
                    mask=mask,
                    return_details=True,
                )
            )
        for output in outputs:
            self.assertEqual(tuple(output["loss_per_sample"].shape), (1,))
            self.assertEqual(tuple(output["raw_nll_per_sample"].shape), (1,))
            self.assertTrue(bool(torch.isfinite(output["loss_per_sample"]).all()))


def _small_model(alpha_mode: str) -> RiboUnmixModel:
    return RiboUnmixModel(
        {
            "mass_conservation": False,
            "alpha_mode": alpha_mode,
            "fixed_alpha": 0.1,
            "init_gamma": 1.0,
            "gamma_centering": {"mode": "disabled"},
            "additional_sequence_features": {},
            "biological_params": {
                "input_size": 5,
                "hidden_size": 4,
                "num_layers": 1,
                "dropout": 0.0,
                "init_local_hazard_factor": 1.0,
                "init_local_hazard_weight_std": 1.0e-2,
            },
            "dataset_bias_params": {
                "position_features": ["rel_pos"],
                "position_scale": 100.0,
                "position_edge_tau": 10.0,
                "num_datasets": 2,
                "dataset_embeddings_size": 2,
                "num_codons": 8,
                "codon_embeddings_size": 3,
                "use_nucleotide_amino_acid_embeddings": False,
                "context_gru_hidden_size": 3,
                "context_gru_num_layers": 1,
                "context_gru_dropout": 0.0,
                "dataset_multiplicative_allocation_bias_submodule_params": {
                    "hidden_size": 4,
                    "dropout": 0.0,
                },
                "dataset_log_sigma_submodule_params": {
                    "hidden_size": 4,
                    "dropout": 0.0,
                },
            },
        }
    )


class FixedAlphaOracleTests(unittest.TestCase):
    def test_fixed_alpha_bypasses_head_and_is_point_one(self) -> None:
        model = _small_model("fixed")
        calls = []
        handle = model.dataset_bias_model.log_sigma_head.register_forward_hook(
            lambda *_args: calls.append(True)
        )
        features = torch.randn(2, 4, 5)
        lengths = torch.tensor([4, 3])
        packed = pack_padded_sequence(
            features,
            lengths,
            batch_first=True,
            enforce_sorted=True,
        )
        mask = torch.arange(4).unsqueeze(0) < lengths.unsqueeze(1)
        _, log_alpha, extras = model(
            x_packed=packed,
            codon_ids=torch.tensor([[1, 2, 3, 4], [2, 3, 4, 0]]),
            id_datasets=torch.tensor([0, 1]),
            mask=mask,
            target=torch.tensor([[1.0, 2.0, 0.0, 3.0], [2.0, 0.0, 1.0, 0.0]]),
            sample_ids=["t1", "t2"],
            transcript_group_index=torch.tensor([0, 1]),
        )
        handle.remove()

        self.assertEqual(calls, [])
        torch.testing.assert_close(
            log_alpha[mask].exp(),
            torch.full_like(log_alpha[mask], 0.1),
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        torch.testing.assert_close(
            extras["alpha"][mask],
            torch.full_like(extras["alpha"][mask], 0.1),
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.dataset_bias_model.log_sigma_head.parameters()
            )
        )

    def test_fixed_alpha_head_is_absent_from_optimizer(self) -> None:
        module = RiboUnmixLightningModule.__new__(
            RiboUnmixLightningModule
        )
        torch.nn.Module.__init__(module)
        module.model = _small_model("fixed")
        module.alpha_learning_rate_scale = 0.1
        module.config = SimpleNamespace(
            optim=SimpleNamespace(
                lr_biological=5.0e-4,
                lr_rest=1.0e-3,
                weight_decay_bio=1.0e-2,
                weight_decay_rest=1.0e-2,
                scheduler=SimpleNamespace(
                    monitor="val_loss",
                    mode="min",
                    factor=0.99,
                    patience=5,
                    min_lr=1.0e-6,
                ),
            )
        )
        optimizer = module.configure_optimizers()["optimizer"]
        self.assertNotIn("alpha", {group["name"] for group in optimizer.param_groups})
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        head_ids = {
            id(parameter)
            for parameter in module.model.dataset_bias_model.log_sigma_head.parameters()
        }
        self.assertTrue(optimizer_ids.isdisjoint(head_ids))


if __name__ == "__main__":
    unittest.main()
