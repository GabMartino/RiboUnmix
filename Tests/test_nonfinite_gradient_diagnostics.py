from types import SimpleNamespace
from unittest.mock import Mock

import lightning as pl
import unittest
import torch

from Models.RiboUnmixLightningModule import RiboUnmixLightningModule


def _module():
    module = RiboUnmixLightningModule.__new__(RiboUnmixLightningModule)
    pl.LightningModule.__init__(module)
    module.biological = torch.nn.Parameter(torch.tensor([1.0]))
    module.bias_branch = torch.nn.Parameter(torch.tensor([1.0]))
    module._gradient_batch_context = {
        "epoch": 1,
        "batch_idx": 2072,
        "ids": ["transcript_A", "transcript_A"],
        "dataset_ids": torch.tensor([0, 113]),
        "lengths": torch.tensor([512, 512]),
        "execution": {"logical_batch_index": 10, "execution_chunk_index": 2},
    }
    return module


class NonfiniteGradientDiagnosticsTests(unittest.TestCase):
    def test_failure_reports_actual_model_precision_not_a_saved_override(self):
        module = _module()
        encoder = torch.nn.Module()
        encoder.rnn = torch.nn.GRU(1, 1)
        encoder.precision = "float32"
        encoder.tbptt_window = 1024
        module.model = torch.nn.Module()
        module.model.dataset_bias_model = torch.nn.Module()
        module.model.dataset_bias_model.local_context_gru = encoder
        module._trainer = SimpleNamespace(
            precision_plugin=SimpleNamespace(scaler=None, precision="bf16-mixed")
        )
        module.bias_branch.grad = torch.full_like(module.bias_branch, float("inf"))
        for precision in ("float32", "inherit"):
            encoder.precision = precision
            with self.subTest(precision=precision), self.assertRaises(FloatingPointError) as error:
                module.on_after_backward()
            message = str(error.exception)
            self.assertIn("formulation=log-space-nb2-v1", message)
            self.assertIn("gru_compute_policy=cuda-amp-gru-fp32-v1", message)
            self.assertIn("trainer_precision=bf16-mixed", message)
            self.assertIn(f"bias_gru_precision={precision}", message)
            self.assertIn("bias_gru_tbptt_window=1024", message)
            # FP32 master weights alone do not tell us the GRU's AMP policy.
            self.assertIn("bias_gru_weight_dtype=torch.float32", message)
            self.assertIn("device=cpu", message)
            self.assertIn("torch=" + torch.__version__, message)
            self.assertTrue(torch.isinf(module.bias_branch.grad).all())

    def test_runtime_context_handles_a_module_without_a_bias_branch(self):
        message = _module()._numerical_runtime_context()
        self.assertIn("trainer_precision=unavailable", message)
        self.assertIn("bias_gru_precision=unavailable", message)

    def test_lightning_bf16_calls_guard_before_a_bad_update(self):
        class TinyModule(RiboUnmixLightningModule):
            def __init__(self):
                pl.LightningModule.__init__(self)
                self.layer = torch.nn.Linear(2, 1)
                self.forward_dtype = None

            def training_step(self, batch, batch_idx):
                prediction = self.layer(batch)
                self.forward_dtype = prediction.dtype
                return torch.nan_to_num(prediction / 0.0, posinf=0.0, neginf=0.0).sum()

            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=0.1)

            def on_fit_start(self):
                pass

            def on_train_epoch_start(self):
                pass

        module = TinyModule()
        initial = [p.detach().clone() for p in module.parameters()]
        trainer = pl.Trainer(
            accelerator="cpu", precision="bf16-mixed", max_epochs=1,
            logger=False, enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False, gradient_clip_val=1.0,
        )
        loader = torch.utils.data.DataLoader(torch.ones(2, 2), batch_size=2)
        with self.assertRaisesRegex(FloatingPointError, "after backward"):
            trainer.fit(module, train_dataloaders=loader)
        self.assertEqual(module.forward_dtype, torch.bfloat16)
        self.assertEqual(trainer.global_step, 0)
        for parameter, expected in zip(module.parameters(), initial):
            torch.testing.assert_close(parameter, expected)

    def test_nonfinite_branch_is_reported_before_clipping_can_contaminate_biology(self):
        module = _module()
        module.biological.grad = torch.ones_like(module.biological)
        module.bias_branch.grad = torch.full_like(module.bias_branch, float("nan"))
        module.optimizers = Mock(return_value=Mock())
        module.clip_gradients = Mock()
        module.execution_gradient_clip_val = 1.0
        module.execution_gradient_clip_algorithm = "norm"

        with self.assertRaises(FloatingPointError) as error:
            module._execution_optimizer_step()

        message = str(error.exception)
        assert "before gradient clipping" in message
        assert "['bias_branch']" in message
        assert "transcript_A" in message
        assert "batch_idx=2072" in message
        assert "execution_chunk_index" in message
        assert "dataset_ids=[0, 113]" in message
        module.clip_gradients.assert_not_called()
        module.optimizers.return_value.step.assert_not_called()
        assert torch.isfinite(module.biological.grad).all()

        # Reproduce the old order: the healthy parameter becomes NaN as well.
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        assert not torch.isfinite(module.biological.grad).all()


    def test_bf16_finite_loss_invalid_backward_is_caught_on_its_execution_chunk(self):
        module = _module()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            # Sanitizing a finite loss does not remove an invalid backward path.
            loss = module.biological.square().sum() + torch.nan_to_num(
                module.bias_branch / 0.0, posinf=0.0
            ).sum()
        assert torch.isfinite(loss)
        loss.backward()
        with self.assertRaisesRegex(FloatingPointError, "after backward.*before clipping"):
            module.on_after_backward()
        assert torch.isfinite(module.biological.grad).all()


    def test_finite_accumulated_gradients_are_unchanged_by_checks(self):
        module = _module()
        for _ in range(2):
            (module.biological.square() + module.bias_branch.square()).sum().backward()
            before = [p.grad.clone() for p in module.parameters()]
            module.on_after_backward()
            module.on_before_optimizer_step(None)
            for p, expected in zip(module.parameters(), before):
                torch.testing.assert_close(p.grad, expected)


    def test_scaled_fp16_gradients_wait_for_optimizer_hook(self):
        module = _module()
        module._trainer = SimpleNamespace(precision_plugin=SimpleNamespace(scaler=object()))
        module.bias_branch.grad = torch.full_like(module.bias_branch, float("inf"))
        module.on_after_backward()
        with self.assertRaisesRegex(FloatingPointError, "before the optimizer step"):
            module.on_before_optimizer_step(None)
