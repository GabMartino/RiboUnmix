from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from Models.RiboUnmixModel.DatasetBiasSubmodel import DatasetBiasSubmodel
from Models.RiboUnmixLightningModule import (
    NegativeBinomialProfileLoss,
    RiboUnmixLightningModule,
)


def _small_bias_model() -> DatasetBiasSubmodel:
    model = DatasetBiasSubmodel(
        {
            "position_features": ["rel_pos"],
            "codon_embeddings_size": 3,
            "dataset_embeddings_size": 2,
            "num_datasets": 2,
            "num_codons": 8,
            "context_gru_hidden_size": 3,
            "context_gru_num_layers": 1,
            "dataset_multiplicative_allocation_bias_submodule_params": {
                "hidden_size": 4,
                "dropout": 0.0,
            },
            "dataset_log_sigma_submodule_params": {
                "hidden_size": 4,
                "dropout": 0.0,
            },
        }
    )
    # Both heads initialize their final weight to zero. Make it nonzero so this
    # test measures routing rather than the deliberately neutral initialization.
    with torch.no_grad():
        model.observation_bias_head.log_bias_head.weight.fill_(0.1)
        model.log_sigma_head.ff[-1].weight.fill_(0.1)
    return model


def _forward_bias(model: DatasetBiasSubmodel) -> dict[str, torch.Tensor]:
    return model(
        dataset_ids=torch.tensor([0, 1]),
        mask=torch.tensor([[True, True, True], [True, True, False]]),
        codon_ids=torch.tensor([[1, 2, 3], [3, 2, 0]]),
        position_features=torch.tensor(
            [[[0.0], [0.5], [1.0]], [[0.0], [1.0], [0.0]]]
        ),
    )


def _gradient_mass(module: nn.Module) -> float:
    return sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


class AlphaGradientRoutingTests(unittest.TestCase):
    def test_alpha_head_learns_without_upstream_context_gradients(self) -> None:
        model = _small_bias_model()
        output = _forward_bias(model)
        mask = torch.tensor([[True, True, True], [True, True, False]])
        nb2 = NegativeBinomialProfileLoss(sequence_reduction="mean")
        nb2(
            mu_phys=torch.tensor([[1.0, 2.0, 1.0], [2.0, 1.0, 1.0]]),
            log_sigma=output["log_sigma"],
            y_true=torch.tensor([[0.0, 5.0, 1.0], [4.0, 0.0, 0.0]]),
            mask=mask,
        ).backward()

        for name, parameter in model.log_sigma_head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertGreater(float(parameter.grad.detach().abs().sum()), 0.0, name)
        self.assertEqual(_gradient_mass(model.local_context_gru), 0.0)
        self.assertEqual(_gradient_mass(model.dataset_embedding), 0.0)
        self.assertEqual(_gradient_mass(model.codon_embedding), 0.0)

    def test_gamma_still_trains_the_shared_context_encoder(self) -> None:
        model = _small_bias_model()
        output = _forward_bias(model)
        output["gamma_raw"].sum().backward()

        self.assertGreater(_gradient_mass(model.observation_bias_head), 0.0)
        self.assertGreater(_gradient_mass(model.local_context_gru), 0.0)
        self.assertGreater(_gradient_mass(model.dataset_embedding), 0.0)
        self.assertGreater(_gradient_mass(model.codon_embedding), 0.0)

    def test_detach_does_not_change_alpha_forward_values(self) -> None:
        model = _small_bias_model()
        captured: dict[str, torch.Tensor] = {}

        def capture_input(_module, _args, kwargs) -> None:
            captured["x"] = kwargs["x"]
            captured["mask"] = kwargs["mask"]

        handle = model.log_sigma_head.register_forward_pre_hook(
            capture_input,
            with_kwargs=True,
        )
        output = _forward_bias(model)
        handle.remove()

        self.assertFalse(captured["x"].requires_grad)
        direct = model.log_sigma_head(
            x=captured["x"].clone().requires_grad_(True),
            mask=captured["mask"],
        )["log_sigma"]
        torch.testing.assert_close(output["log_sigma"], direct)


class _OptimizerTestModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.biological_model = nn.Linear(2, 2)
        self.dataset_bias_model = nn.Module()
        self.dataset_bias_model.context = nn.Linear(2, 2)
        self.dataset_bias_model.log_sigma_head = nn.Linear(2, 1)
        self.other = nn.Linear(2, 2)


def _optimizer_module(alpha_scale: float) -> RiboUnmixLightningModule:
    module = RiboUnmixLightningModule.__new__(
        RiboUnmixLightningModule
    )
    nn.Module.__init__(module)
    module.model = _OptimizerTestModel()
    module.alpha_learning_rate_scale = alpha_scale
    module.config = SimpleNamespace(
        optim=SimpleNamespace(
            lr_biological=5.0e-4,
            lr_rest=2.0e-3,
            weight_decay_bio=1.0e-2,
            weight_decay_rest=2.0e-2,
            scheduler=SimpleNamespace(
                monitor="val_loss",
                mode="min",
                factor=0.5,
                patience=1,
                min_lr=1.0e-6,
            ),
        )
    )
    return module


class AlphaOptimizerGroupTests(unittest.TestCase):
    def test_alpha_group_has_scaled_lr_and_no_duplicate_parameters(self) -> None:
        module = _optimizer_module(0.1)
        configured = module.configure_optimizers()
        optimizer = configured["optimizer"]
        groups = {group["name"]: group for group in optimizer.param_groups}

        self.assertAlmostEqual(groups["rest"]["lr"], 2.0e-3)
        self.assertAlmostEqual(groups["alpha"]["lr"], 2.0e-4)
        self.assertEqual(groups["alpha"]["weight_decay"], groups["rest"]["weight_decay"])

        grouped_ids = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
        self.assertEqual(
            set(grouped_ids),
            {id(parameter) for parameter in module.model.parameters()},
        )
        self.assertEqual(
            {id(parameter) for parameter in groups["alpha"]["params"]},
            {
                id(parameter)
                for parameter in module.model.dataset_bias_model.log_sigma_head.parameters()
            },
        )

        scheduler = configured["lr_scheduler"]["scheduler"]
        self.assertEqual(scheduler.min_lrs, [1.0e-6, 1.0e-6, 1.0e-7])
        for worsening_metric in range(30):
            scheduler.step(float(worsening_metric))
            self.assertAlmostEqual(
                groups["alpha"]["lr"],
                groups["rest"]["lr"] * 0.1,
            )

    def test_scale_one_restores_original_alpha_learning_rate(self) -> None:
        module = _optimizer_module(1.0)
        optimizer = module.configure_optimizers()["optimizer"]
        groups = {group["name"]: group for group in optimizer.param_groups}
        self.assertEqual(groups["alpha"]["lr"], groups["rest"]["lr"])


if __name__ == "__main__":
    unittest.main()
