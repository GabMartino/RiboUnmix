from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn.utils.rnn import pack_padded_sequence

from Models.RiboUnmixModel.DatasetBiasSubmodel import BiGRUContextEncoder
from Models.utils.gru_failure_capture import capture_if_nonfinite
from replay_bias_gru_failure import replay


class GRUFailureCaptureTests(unittest.TestCase):
    def test_finite_hook_does_not_modify_gradients_or_create_files(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "not_created"
            x = torch.tensor([2.0], requires_grad=True)
            out = x.square()
            out.grad_fn.register_hook(lambda inputs, outputs: capture_if_nonfinite(
                destination, lambda: self.fail("No capture for a healthy operation"), inputs, outputs))
            out.backward()
            self.assertEqual(x.grad.item(), 4.0)
            self.assertFalse(destination.exists())

    def test_reports_internal_vs_upstream_failure_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            for incoming, outgoing, origin in [
                (torch.ones(1), torch.full((1,), float("inf")), "inside_fused_gru_backward"),
                (torch.full((1,), float("nan")), torch.ones(1), "upstream_adjoint_already_nonfinite"),
            ]:
                with self.assertRaisesRegex(FloatingPointError, origin):
                    capture_if_nonfinite(temp, lambda: {"format_version": 1}, (outgoing,), (incoming,))
            files = list(Path(temp).glob("gru_failure_*.pt"))
            self.assertEqual(len(files), 2)
            payloads = [torch.load(p, weights_only=False) for p in files]
            self.assertEqual({p["origin"] for p in payloads}, {
                "inside_fused_gru_backward", "upstream_adjoint_already_nonfinite"})

    def test_failed_capture_still_raises_the_numerical_error(self):
        with tempfile.TemporaryDirectory() as temp:
            bad_destination = Path(temp) / "not_a_directory"
            bad_destination.touch()
            with self.assertRaisesRegex(FloatingPointError, "Capture failed"):
                capture_if_nonfinite(bad_destination, lambda: {}, (torch.tensor(float("nan")),), (torch.ones(1),))

    def test_replay_matches_original_packed_input_and_parameter_vjp(self):
        torch.manual_seed(71)
        config = dict(input_size=3, hidden_size=2, num_layers=2, bias=True,
                      batch_first=True, dropout=0.0, bidirectional=True)
        rnn = torch.nn.GRU(**config)
        packed = pack_padded_sequence(torch.randn(2, 5, 3), [3, 5], batch_first=True, enforce_sorted=False)
        packed.data.requires_grad_(True)
        rng = torch.get_rng_state()
        output, hidden = rnn(packed)
        hidden_sorted = hidden.index_select(1, packed.sorted_indices)
        grad_outputs = [torch.randn_like(output.data), torch.randn_like(hidden_sorted), None, None, None]
        expected = torch.autograd.grad((output.data, hidden_sorted), (packed.data, *rnn.parameters()), grad_outputs[:2])
        payload = dict(
            format_version=1, rnn_config=config, rnn_state_dict=copy.deepcopy(rnn.state_dict()),
            input_data=packed.data.detach(), batch_sizes=packed.batch_sizes,
            sorted_indices=packed.sorted_indices, unsorted_indices=packed.unsorted_indices,
            grad_outputs=grad_outputs, autocast_enabled=False, autocast_dtype="torch.bfloat16",
            rng_cpu=rng, rng_cuda=rng, cudnn_enabled=False, cudnn_deterministic=False,
            cudnn_benchmark=False, cudnn_allow_tf32=True, matmul_allow_tf32=False,
        )
        original_weights = {k: v.clone() for k, v in payload["rnn_state_dict"].items()}
        result = replay(payload, device=torch.device("cpu"), mode="original", return_gradients=True)
        names = ["input_data", *dict(rnn.named_parameters())]
        self.assertEqual(result["nonfinite_gradients"], [])
        for name, gradient in zip(names, expected, strict=True):
            self.assertAlmostEqual(result["gradients"][name]["max_abs"], float(gradient.abs().max()), places=6)
            torch.testing.assert_close(result["gradient_tensors"][name], gradient, rtol=0, atol=0)
        for name, weight in original_weights.items():
            torch.testing.assert_close(payload["rnn_state_dict"][name], weight, rtol=0, atol=0)
        for mode in ("float32", "float64"):
            self.assertEqual(replay(payload, device=torch.device("cpu"), mode=mode)["nonfinite_gradients"], [])

    def test_capture_does_not_claim_to_support_native_cpu_gru(self):
        model = BiGRUContextEncoder(1, 1, failure_capture_dir="unused").train()
        with self.assertRaisesRegex(ValueError, "CUDA packed-sequence"):
            model(torch.zeros(1, 1, 4), torch.ones(1, 4, dtype=torch.bool), (4,))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA cuDNN node integration requires a GPU.")
    def test_cuda_hook_targets_fused_recurrence_and_captures_bad_upstream_adjoint(self):
        with tempfile.TemporaryDirectory() as temp:
            model = BiGRUContextEncoder(2, 2, failure_capture_dir=temp).cuda().train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(torch.randn(2, 2, 9, device="cuda"),
                            torch.ones(2, 9, device="cuda", dtype=torch.bool), (9, 9))
            out.register_hook(lambda grad: grad * float("nan"))
            with self.assertRaisesRegex(FloatingPointError, "upstream_adjoint_already_nonfinite"):
                out.sum().backward()
            payload = torch.load(next(Path(temp).glob("*.pt")), map_location="cpu", weights_only=False)
            self.assertIn("CudnnRnnBackward", payload["node_name"])
            self.assertEqual(payload["input_data"].shape, (18, 2))


if __name__ == "__main__":
    unittest.main()
