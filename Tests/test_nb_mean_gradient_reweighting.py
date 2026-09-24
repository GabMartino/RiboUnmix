from __future__ import annotations

import unittest

import torch

from Models.RiboUnmixLightningModule import NegativeBinomialProfileLoss


def _ordinary_nb2_nll(
    y: torch.Tensor,
    mu: torch.Tensor,
    log_alpha: torch.Tensor,
) -> torch.Tensor:
    r = torch.exp(-log_alpha)
    return (
        torch.lgamma(r)
        - torch.lgamma(y + r)
        + torch.lgamma(y + 1.0)
        - r * torch.log(r)
        - y * torch.log(mu)
        + (r + y) * torch.log(r + mu)
    )


class NBMeanGradientReweightingTests(unittest.TestCase):
    def test_beta_zero_reproduces_original_values_and_gradients(self) -> None:
        y = torch.tensor([[0.0, 3.0, 7.0]], dtype=torch.float32)
        mask = torch.tensor([[True, True, True]])

        log_mu_new = torch.tensor(
            [[-0.2, 0.4, 1.0]], dtype=torch.float32, requires_grad=True
        )
        log_alpha_new = torch.tensor(
            [[-1.5, -0.7, 0.2]], dtype=torch.float32, requires_grad=True
        )
        loss = NegativeBinomialProfileLoss(
            sequence_reduction="mean",
            nb_mean_gradient_beta=0.0,
        )
        actual = loss(
            mu_phys=log_mu_new.exp(),
            log_sigma=log_alpha_new,
            y_true=y,
            mask=mask,
            return_per_sample=True,
        ).sum()
        actual.backward()

        log_mu_reference = log_mu_new.detach().clone().requires_grad_(True)
        log_alpha_reference = log_alpha_new.detach().clone().requires_grad_(True)
        expected = _ordinary_nb2_nll(
            y,
            log_mu_reference.exp(),
            log_alpha_reference,
        ).mean()
        expected.backward()

        torch.testing.assert_close(actual.detach(), expected.detach())
        torch.testing.assert_close(log_mu_new.grad, log_mu_reference.grad)
        torch.testing.assert_close(log_alpha_new.grad, log_alpha_reference.grad)

    def test_weight_is_detached_but_alpha_still_has_gradient(self) -> None:
        log_mu = torch.tensor([[1.2, 2.0]], requires_grad=True)
        log_alpha = torch.tensor([[-0.5, 0.3]], requires_grad=True)
        details = NegativeBinomialProfileLoss(
            sequence_reduction="mean",
            nb_mean_gradient_beta=0.5,
        )(
            mu_phys=log_mu.exp(),
            log_sigma=log_alpha,
            y_true=torch.tensor([[0.0, 25.0]]),
            mask=torch.tensor([[True, True]]),
            return_details=True,
        )

        self.assertFalse(details["mean_gradient_weight"].requires_grad)
        details["loss_per_sample"].sum().backward()
        self.assertIsNotNone(log_alpha.grad)
        self.assertTrue(bool(torch.isfinite(log_alpha.grad).all()))
        self.assertGreater(float(log_alpha.grad.abs().sum()), 0.0)

    def test_raw_validation_selection_keeps_ordinary_likelihood(self) -> None:
        details = NegativeBinomialProfileLoss(
            sequence_reduction="mean",
            nb_mean_gradient_beta=0.5,
        )(
            mu_phys=torch.tensor([[2.0, 8.0]]),
            log_sigma=torch.tensor([[-1.0, 0.5]]),
            y_true=torch.tensor([[1.0, 15.0]]),
            mask=torch.tensor([[True, True]]),
            apply_mean_gradient_reweighting=False,
            return_details=True,
        )
        torch.testing.assert_close(
            details["loss_per_sample"],
            details["raw_nll_per_sample"],
        )
        self.assertFalse(
            torch.allclose(
                details["raw_nll_per_sample"],
                details["mean_reweighted_nll_per_sample"],
            )
        )

    def test_higher_beta_increases_high_alpha_mu_mean_gradient(self) -> None:
        gradients = []
        for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
            log_mu = torch.tensor([[3.0]], requires_grad=True)
            loss = NegativeBinomialProfileLoss(
                sequence_reduction="mean",
                nb_mean_gradient_beta=beta,
            )(
                mu_phys=log_mu.exp(),
                log_sigma=torch.tensor([[0.5]]),
                y_true=torch.tensor([[1.0]]),
                mask=torch.tensor([[True]]),
            )
            loss.backward()
            gradients.append(float(log_mu.grad.abs().item()))

        self.assertTrue(
            all(right > left for left, right in zip(gradients, gradients[1:]))
        )

    def test_beta_one_cancels_log_mu_attenuation(self) -> None:
        log_mu = torch.tensor([[1.7]], dtype=torch.float64, requires_grad=True)
        mu = log_mu.exp()
        y = torch.tensor([[13.0]], dtype=torch.float64)
        loss = NegativeBinomialProfileLoss(
            sequence_reduction="mean",
            nb_mean_gradient_beta=1.0,
        )(
            # The active loss intentionally calculates in float32, matching
            # the project's mixed-precision-safe NB policy.
            mu_phys=mu.float(),
            log_sigma=torch.tensor([[0.4]], dtype=torch.float32),
            y_true=y.float(),
            mask=torch.tensor([[True]]),
        )
        loss.backward()

        expected = float(mu.detach().item() - y.item())
        self.assertAlmostEqual(float(log_mu.grad.item()), expected, places=4)

    def test_mask_and_sequence_reduction_are_unchanged(self) -> None:
        y = torch.tensor([[2.0, 1_000.0, 5.0]])
        mu = torch.tensor([[1.5, 1.0e6, 4.0]])
        log_alpha = torch.tensor([[-1.0, 1.0, -0.5]])
        mask = torch.tensor([[True, False, True]])
        loss = NegativeBinomialProfileLoss(
            sequence_reduction="mean",
            nb_mean_gradient_beta=0.5,
        )
        details = loss(
            mu_phys=mu,
            log_sigma=log_alpha,
            y_true=y,
            mask=mask,
            return_details=True,
        )

        expected_raw = _ordinary_nb2_nll(
            y[:, [0, 2]],
            mu[:, [0, 2]],
            log_alpha[:, [0, 2]],
        ).mean(dim=1)
        torch.testing.assert_close(details["raw_nll_per_sample"], expected_raw)

        expected_weight = (
            1.0 + log_alpha[:, [0, 2]].exp() * mu[:, [0, 2]]
        ).pow(0.5)
        expected_reweighted = (
            expected_weight
            * _ordinary_nb2_nll(
                y[:, [0, 2]],
                mu[:, [0, 2]],
                log_alpha[:, [0, 2]],
            )
        ).mean(dim=1)
        torch.testing.assert_close(
            details["mean_reweighted_nll_per_sample"],
            expected_reweighted,
        )


if __name__ == "__main__":
    unittest.main()
