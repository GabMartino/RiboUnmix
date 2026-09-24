"""Numerical execution tests; no datasets, training runs, or GPUs required."""
from __future__ import annotations

import copy
import unittest

import torch

from Models.RiboUnmixModel.DatasetBiasSubmodel import (
    BiGRUContextEncoder,
    DatasetBiasSubmodel,
)
from Models.utils.stable_numerics import nb2_nll_from_log_mean
from Tests import test_dataset_bias_sequence_embeddings as embedding_tests
from Tests.test_gamma_centering import _fixed_reference_helper


def _bias_config(precision: str) -> dict:
    return embedding_tests.DatasetBiasSequenceEmbeddingTests._minimal_bias_config(
        context_gru_precision=precision,
        num_datasets=4,
        context_gru_hidden_size=4,
        context_gru_num_layers=2,
        use_nucleotide_amino_acid_embeddings=True,
        nucleotide_embeddings_size=2,
        amino_acid_embeddings_size=2,
        num_nucleotides=4,
        num_amino_acids=3,
        codon_nucleotide_ids=[[i % 4, (i + 1) % 4, (i + 2) % 4] for i in range(64)],
        codon_amino_acid_ids=[i % 3 for i in range(64)],
    )


class BiasGRUPrecisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_rejects_unknown_precision_and_true_low_precision_weights(self):
        with self.assertRaisesRegex(ValueError, "context_gru_precision"):
            DatasetBiasSubmodel(_bias_config("fp23"))
        encoder = BiGRUContextEncoder(2, 3, precision="float32").bfloat16()
        with self.assertRaisesRegex(ValueError, "FP32 model weights"):
            encoder(torch.zeros(1, 2, 5))

    def test_island_matches_explicit_fp32_forward_and_backward(self):
        base = BiGRUContextEncoder(3, 4, num_layers=2)
        island = copy.deepcopy(base)
        island.precision = "float32"
        mask = torch.arange(11)[None] < torch.tensor([7, 11])[:, None]
        x = torch.randn(2, 3, 11, requires_grad=True)
        x_island = x.detach().clone().requires_grad_(True)
        expected = base(x, mask, (7, 11))
        coefficients = torch.randn_like(expected)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = island(x_island, mask, (7, 11))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.dtype, torch.float32)
        (expected * coefficients).sum().backward()
        (actual * coefficients).sum().backward()
        torch.testing.assert_close(x_island.grad, x.grad, rtol=0, atol=0)
        for p, q in zip(base.parameters(), island.parameters(), strict=True):
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)

    def test_long_gru_can_overflow_in_bf16_with_finite_outputs_but_not_fp32(self):
        # Deliberately adversarial recurrent Jacobian, not a claim to reproduce
        # the unavailable epoch-9 CUDA state. Same 1463 positions as the report.
        # At h=0, its forward-direction derivative is 0.5 + 0.25*w_hh.
        # BF16 rounding changes repeated products enough to overflow even
        # though the log-space NB loss has a finite, small output derivative.
        base = BiGRUContextEncoder(1, 1)
        base.output_norm = torch.nn.Identity()
        with torch.no_grad():
            for parameter in base.parameters():
                parameter.zero_()
            base.rnn.weight_hh_l0[2, 0] = 2.258
        island = copy.deepcopy(base)
        island.precision = "float32"
        for model in (base, island):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                output = model(torch.zeros(1, 1, 1463))
                log_mean = output[0, 0, -1]
                # At eta=0, y=2, alpha=1: d(loss)/d(eta) = -1e-3.
                loss = 0.002 * nb2_nll_from_log_mean(
                    log_mean.new_tensor(2.0), log_mean, log_mean.new_tensor(0.0)
                )
            self.assertTrue(bool(output.isfinite().all()))
            self.assertTrue(bool(loss.isfinite()))
            loss.backward()
        self.assertFalse(bool(base.rnn.bias_ih_l0.grad.isfinite().all()))
        self.assertTrue(all(bool(p.grad.isfinite().all()) for p in island.parameters()))

    def test_embeddings_and_gru_remain_fp32_but_heads_still_use_autocast(self):
        model = DatasetBiasSubmodel(_bias_config("float32"))
        with torch.no_grad():
            model.observation_bias_head.log_bias_head.weight.fill_(0.2)
        seen = {}

        def capture_context(_module, args):
            seen["encoder_input_dtype"] = args[0].dtype
            seen["encoder_input"] = args[0].detach().clone()

        def capture_gru(_module, args):
            seen["gru_autocast"] = torch.is_autocast_enabled("cpu")
            seen["gru_input_dtype"] = args[0].data.dtype

        def capture_head(_module, _args, output):
            seen["head_dtype"] = output.dtype
            seen["head_autocast"] = torch.is_autocast_enabled("cpu")

        handles = [
            model.local_context_gru.register_forward_pre_hook(capture_context),
            model.local_context_gru.rnn.register_forward_pre_hook(capture_gru),
            model.observation_bias_head.shared[0].register_forward_hook(capture_head),
        ]
        try:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                out = model(
                    dataset_ids=torch.tensor([0, 1]),
                    codon_ids=torch.tensor([[1, 2, 3], [5, 4, 0]]),
                    mask=torch.tensor([[True, True, True], [True, True, False]]),
                    position_features=torch.zeros(2, 3, 1, dtype=torch.bfloat16),
                    cpu_lengths=(3, 2),
                )
            out["gamma_raw"].sum().backward()
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(seen["encoder_input_dtype"], torch.float32)
        self.assertEqual(seen["gru_input_dtype"], torch.float32)
        self.assertFalse(seen["gru_autocast"])
        self.assertTrue(seen["head_autocast"])
        self.assertEqual(seen["head_dtype"], torch.bfloat16)
        # A cast-to-BF16-then-back-to-FP32 "fix" must fail this check.
        actual_codon_embedding = seen["encoder_input"][0, 2:4, 0]
        torch.testing.assert_close(actual_codon_embedding, model.codon_embedding.weight[1])
        self.assertFalse(torch.equal(
            actual_codon_embedding, actual_codon_embedding.bfloat16().float()
        ))
        for name in ("dataset_embedding", "codon_embedding", "nucleotide_embedding", "amino_acid_embedding"):
            grad = getattr(model, name).weight.grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(bool(grad.isfinite().all()), name)
            self.assertGreater(float(grad.abs().sum()), 0.0, name)

    def test_old_state_dict_and_adam_state_load_without_parameter_changes(self):
        old = DatasetBiasSubmodel(_bias_config("inherit"))
        optimizer = torch.optim.Adam(old.parameters())
        sum(p.square().sum() for p in old.parameters()).backward()
        optimizer.step()
        fixed = DatasetBiasSubmodel(dict(_bias_config("float32"), context_gru_tbptt_window=1024))
        fixed.load_state_dict(old.state_dict(), strict=True)
        resumed = torch.optim.Adam(fixed.parameters())
        resumed.load_state_dict(optimizer.state_dict())
        self.assertEqual(list(fixed.state_dict()), list(old.state_dict()))
        self.assertEqual(fixed.local_context_gru.precision, "float32")
        self.assertEqual(fixed.local_context_gru.tbptt_window, 1024)
        for p, q in zip(old.parameters(), fixed.parameters(), strict=True):
            torch.testing.assert_close(q, p, rtol=0, atol=0)
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(resumed.state[q][key], optimizer.state[p][key])

    def test_reference_chunking_keeps_all_gradients_and_exact_center(self):
        self._check_reference_chunking(tbptt_window=0)

    def test_tbptt_reference_chunking_keeps_all_dataset_gradients_and_exact_center(self):
        self._check_reference_chunking(tbptt_window=3)

    def _check_reference_chunking(self, tbptt_window):
        # Dropout-free float32 execution for the invariance check. Stochastic
        # heads under AMP need not give bitwise equal values across chunk sizes.
        model = _fixed_reference_helper([0, 1, 2, 3], [1.0] * 4)
        model.dataset_bias_model = DatasetBiasSubmodel(
            dict(_bias_config("float32"), context_gru_tbptt_window=tbptt_window)
        )
        with torch.no_grad():
            model.dataset_bias_model.observation_bias_head.log_bias_head.weight.fill_(0.2)
        model.train()
        outputs, gradients = [], []
        for chunk_size in (1, 4):
            model.zero_grad(set_to_none=True)
            model.gamma_reference_chunk_size = chunk_size
            mask = torch.ones(1, 7, dtype=torch.bool)
            ids = torch.tensor([0])
            codons = torch.tensor([[0, 1, 2, 3, 4, 5, 6]])
            position = torch.linspace(0, 1, 7)[None, :, None]
            raw = model.dataset_bias_model(
                dataset_ids=ids, mask=mask, codon_ids=codons,
                position_features=position, compute_log_sigma=False,
                embedding_center_ids=model.gamma_selected_dataset_ids,
            )["gamma_raw"]
            result = model._center_log_gamma_fixed_reference(
                raw, mask_b=mask, sample_ids=["t"], id_datasets=ids,
                codon_ids=codons, position_features=position,
                dataset_bias_sequence_features=None, biological_sequence_features=None,
            )
            result["log_gamma"].square().sum().backward()
            outputs.append(result["log_gamma"].detach())
            gradients.append({k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None})
        torch.testing.assert_close(outputs[0], outputs[1], atol=2e-7, rtol=1e-5)
        self.assertEqual(gradients[0].keys(), gradients[1].keys())
        for key in gradients[0]:
            torch.testing.assert_close(gradients[0][key], gradients[1][key], atol=2e-7, rtol=1e-4)
        self.assertTrue(bool((gradients[1]["dataset_bias_model.dataset_embedding.weight"].abs().sum(1) > 0).all()))


if __name__ == "__main__":
    unittest.main()
