"""Actual recurrent dtype, full-BPTT gradients, and checkpoint invariants."""
import copy
import unittest

import torch
from torch.nn.utils.rnn import pack_padded_sequence

from Models.RiboUnmixModel.DatasetBiasSubmodel import BiGRUContextEncoder
from Models.RiboUnmixModel.SharedProfileModel import SharedProfileModel
from Models.utils.gru_tbptt import gru_tbptt
from Models.utils.stable_numerics import nb2_nll_from_log_mean, clip_grad_norm_stable


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA, not a CPU AMP substitute.")
class CUDAGRUPrecisionTests(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            self.skipTest("Native BF16 unavailable")
        torch.manual_seed(619)

    def test_legacy_inherit_matches_full_fp32_bptt_and_restores_amp(self):
        for amp_dtype in (torch.bfloat16, torch.float16):
            for packed in (False, True):
                with self.subTest(amp_dtype=amp_dtype, packed=packed):
                    encoder = BiGRUContextEncoder(3, 4, 2, precision="inherit").cuda()
                    reference = copy.deepcopy(encoder)
                    x = torch.randn(2, 3, 17, device="cuda", requires_grad=True)
                    xr = x.detach().clone().requires_grad_()
                    mask = torch.arange(17, device="cuda")[None] < torch.tensor([11, 17], device="cuda")[:, None]
                    args = (mask, (11, 17)) if packed else ()
                    expected = reference(xr, *args)
                    with torch.autocast("cuda", dtype=amp_dtype):
                        actual = encoder(x, *args)
                        self.assertTrue(torch.is_autocast_enabled("cuda"))
                    self.assertEqual(actual.dtype, torch.float32)
                    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
                    coefficient = torch.randn_like(actual)
                    (actual * coefficient).sum().backward()
                    (expected * coefficient).sum().backward()
                    torch.testing.assert_close(x.grad, xr.grad, rtol=1e-5, atol=2e-6)
                    for p, q in zip(encoder.parameters(), reference.parameters(), strict=True):
                        torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=3e-6)

    def test_biological_gru_is_fp32_but_head_is_bf16_in_train_and_eval(self):
        model = SharedProfileModel(dict(input_size=3, hidden_size=4, num_layers=2, dropout=0.)).cuda()
        seen = []
        handles = [
            model.rnn.register_forward_hook(lambda _m, _a, out: seen.append(("gru", out[0].data.dtype))),
            model.ff_local_hazard[0].register_forward_hook(lambda _m, _a, out: seen.append(("head", out.dtype))),
        ]
        try:
            for training in (True, False):
                model.train(training)
                x = torch.randn(2, 19, 3, device="cuda", requires_grad=True)
                mask = torch.arange(19, device="cuda")[None] < torch.tensor([19, 12], device="cuda")[:, None]
                packed = pack_padded_sequence(x, (19, 12), batch_first=True)
                with torch.autocast("cuda", dtype=torch.bfloat16), torch.set_grad_enabled(training):
                    out = model(packed, mask)
                self.assertEqual(seen[-2:], [("gru", torch.float32), ("head", torch.bfloat16)])
                self.assertEqual(out["h_n"].dtype, torch.float32)
                if training:
                    out["L_bio"].square().sum().backward()
                    self.assertTrue(x.grad.isfinite().all())
        finally:
            for handle in handles:
                handle.remove()

    def test_direct_tbptt_helper_is_protected_and_matches_fp32_gradients(self):
        rnn = torch.nn.GRU(3, 4, 2, batch_first=True, bidirectional=True).cuda()
        ref = copy.deepcopy(rnn)
        x = torch.randn(3, 39, 3, device="cuda", requires_grad=True)
        xr = x.detach().clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = gru_tbptt(rnn, x, (27, 39, 13), 16)
        expected = gru_tbptt(ref, xr, (27, 39, 13), 16)
        self.assertEqual(out.dtype, torch.float32)
        torch.testing.assert_close(out, expected)
        coefficients = torch.randn_like(out)
        (out * coefficients).sum().backward()
        (expected * coefficients).sum().backward()
        torch.testing.assert_close(x.grad, xr.grad)
        for p, q in zip(rnn.parameters(), ref.parameters(), strict=True):
            torch.testing.assert_close(p.grad, q.grad)

    def test_finite_loss_recurrent_overflow_counterexample_is_fixed_without_tbptt(self):
        # Constructed mechanism, not a claim to replay the cluster checkpoint.
        # Forward h=0, local dh/dh=1.05: FP16 overflows while FP32 is finite.
        encoder = BiGRUContextEncoder(1, 1, precision="inherit").cuda()
        encoder.output_norm = torch.nn.Identity()
        with torch.no_grad():
            for p in encoder.parameters():
                p.zero_()
            encoder.rnn.weight_hh_l0[2, 0] = 2.2
        legacy = copy.deepcopy(encoder.rnn)
        x = torch.zeros(1, 1, 1013, device="cuda")
        mask = torch.ones(1, 1013, device="cuda", dtype=torch.bool)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            old, _ = legacy(pack_padded_sequence(x.transpose(1, 2), (1013,), batch_first=True))
            fixed = encoder(x, mask, (1013,))
            def loss(eta):
                eta = eta.float()
                return .002 * nb2_nll_from_log_mean(eta.new_tensor(2), eta, eta.new_tensor(0))
            old_loss, fixed_loss = loss(old.data[-1, 0]), loss(fixed[0, 0, -1])
        self.assertEqual(encoder.tbptt_window, 0)
        self.assertEqual(fixed.dtype, torch.float32)
        torch.testing.assert_close(old_loss, fixed_loss, rtol=0, atol=0)
        old_loss.backward()
        fixed_loss.backward()
        if old.data.dtype == torch.float16:  # Do not require future PyTorch versions to retain the bug.
            self.assertFalse(all(p.grad.isfinite().all() for p in legacy.parameters()))
        self.assertTrue(all(p.grad.isfinite().all() for p in encoder.parameters()))
        self.assertGreater(float(encoder.rnn.bias_ih_l0.grad.abs().max()), 1e18)
        clip_grad_norm_stable(encoder.parameters(), max_norm=1.)
        opt = torch.optim.AdamW(encoder.parameters(), lr=1e-3)
        opt.step()
        self.assertTrue(all(p.isfinite().all() for p in encoder.parameters()))
        for state in opt.state.values():
            self.assertTrue(all(not torch.is_tensor(v) or v.isfinite().all() for v in state.values()))

    def test_true_low_precision_weights_are_rejected_before_cuda_recurrence(self):
        for dtype in (torch.float16, torch.bfloat16):
            encoder = BiGRUContextEncoder(2, 3).cuda().to(dtype)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with self.assertRaisesRegex(ValueError, "FP32 model weights"):
                    encoder(torch.zeros(1, 2, 5, device="cuda"))

    def test_protection_preserves_parameter_identity_state_and_optimizer_mapping(self):
        encoder = BiGRUContextEncoder(2, 3).cuda()
        identities = [id(p) for p in encoder.parameters()]
        state = copy.deepcopy(encoder.state_dict())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoder(torch.randn(2, 2, 19, device="cuda")).square().mean().backward()
        self.assertEqual(identities, [id(p) for p in encoder.parameters()])
        self.assertEqual(state.keys(), encoder.state_dict().keys())
        for key, value in state.items():
            torch.testing.assert_close(encoder.state_dict()[key], value, rtol=0, atol=0)
        optimizer = torch.optim.AdamW(encoder.parameters())
        optimizer.step()
        restored = BiGRUContextEncoder(2, 3, precision="inherit").cuda()
        restored.load_state_dict(encoder.state_dict(), strict=True)
        resumed = torch.optim.AdamW(restored.parameters())
        resumed.load_state_dict(optimizer.state_dict())
        for p, q in zip(encoder.parameters(), restored.parameters(), strict=True):
            for key in ("step", "exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(optimizer.state[p][key], resumed.state[q][key])


if __name__ == "__main__":
    unittest.main()
