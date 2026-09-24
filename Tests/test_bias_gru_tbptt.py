"""Scientific invariants of state-carrying, per-direction bias-GRU TBPTT."""
from __future__ import annotations

import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from Models.RiboUnmixModel.DatasetBiasSubmodel import BiGRUContextEncoder
from Models.utils.gru_tbptt import gru_tbptt
from Models.utils.stable_numerics import nb2_nll_from_log_mean, clip_grad_norm_stable


def _cell_reference(rnn, sequence, lengths, window):
    """Independent per-transcript recurrence; PyTorch's reset-after-linear GRU."""
    rows = []
    for row, length in zip(sequence, lengths, strict=True):
        x = row[:length]
        for layer in range(rnn.num_layers):
            directions = []
            for reverse in range(2 if rnn.bidirectional else 1):
                suffix = f"_l{layer}" + ("_reverse" if reverse else "")
                w_i, w_h, b_i, b_h = [getattr(rnn, n + suffix) for n in
                                      ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]
                h = x.new_zeros(rnn.hidden_size)
                steps = []
                for step, token in enumerate(x.flip(0) if reverse else x):
                    if step and step % window == 0:
                        h = h.detach()
                    ir, iz, inn = F.linear(token, w_i, b_i).chunk(3)
                    hr, hz, hn = F.linear(h, w_h, b_h).chunk(3)
                    r, z = (ir + hr).sigmoid(), (iz + hz).sigmoid()
                    n = (inn + r * hn).tanh()
                    h = (1 - z) * n + z * h
                    steps.append(h)
                direction = torch.stack(steps)
                directions.append(direction.flip(0) if reverse else direction)
            x = torch.cat(directions, dim=-1)
        rows.append(F.pad(x, (0, 0, 0, sequence.size(1) - length)))
    return torch.stack(rows)


class BiasGRUTBPTTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(310)

    def test_forward_and_truncated_gradients_match_independent_cell_reference(self):
        lengths = (5, 11, 7, 1)  # unsorted, different reverse boundary phases
        for bidirectional in (False, True):
            for window in (1, 3, 7, 20):
                with self.subTest(bidirectional=bidirectional, window=window):
                    rnn = nn.GRU(3, 4, 2, batch_first=True, bidirectional=bidirectional).double()
                    reference = copy.deepcopy(rnn)
                    x = torch.randn(4, 14, 3, dtype=torch.float64, requires_grad=True)
                    xr = x.detach().clone().requires_grad_()
                    out = gru_tbptt(rnn, x, lengths, window)
                    expected = _cell_reference(reference, xr, lengths, window)
                    packed, _ = rnn(pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False))
                    full, _ = pad_packed_sequence(packed, batch_first=True, total_length=14)
                    torch.testing.assert_close(out, full, atol=1e-12, rtol=1e-12)
                    torch.testing.assert_close(out, expected, atol=1e-12, rtol=1e-12)
                    coefficient = torch.randn_like(out)
                    (out * coefficient).sum().backward()
                    (expected * coefficient).sum().backward()
                    torch.testing.assert_close(x.grad, xr.grad, atol=1e-12, rtol=1e-11)
                    for p, q in zip(rnn.parameters(), reference.parameters(), strict=True):
                        torch.testing.assert_close(p.grad, q.grad, atol=1e-12, rtol=1e-11)
                    for row, length in enumerate(lengths):
                        self.assertEqual(x.grad[row, length:].abs().sum().item(), 0)

    def test_window_covering_cds_matches_full_bptt_gradients(self):
        rnn = nn.GRU(2, 3, 2, batch_first=True, bidirectional=True).double()
        full_rnn = copy.deepcopy(rnn)
        x = torch.randn(2, 9, 2, dtype=torch.float64, requires_grad=True)
        xf = x.detach().clone().requires_grad_()
        out = gru_tbptt(rnn, x, (9, 5), 9)
        packed, _ = full_rnn(pack_padded_sequence(xf, (9, 5), batch_first=True, enforce_sorted=False))
        full, _ = pad_packed_sequence(packed, batch_first=True, total_length=9)
        coefficient = torch.randn_like(out)
        (out * coefficient).sum().backward()
        (full * coefficient).sum().backward()
        torch.testing.assert_close(x.grad, xf.grad, atol=1e-12, rtol=1e-11)
        for p, q in zip(rnn.parameters(), full_rnn.parameters(), strict=True):
            torch.testing.assert_close(p.grad, q.grad, atol=1e-12, rtol=1e-11)

    def test_state_values_carry_but_temporal_derivatives_stop_in_both_directions(self):
        rnn = nn.GRU(1, 1, batch_first=True, bidirectional=True).double()
        for parameter in rnn.parameters():
            nn.init.constant_(parameter, 0.2)
        for channel, target, excluded in ((0, 7, slice(0, 6)), (1, 0, slice(2, 8))):
            x = torch.ones(1, 8, 1, dtype=torch.float64, requires_grad=True)
            out = gru_tbptt(rnn, x, (8,), 3)
            full, _ = rnn(x)
            torch.testing.assert_close(out, full, atol=1e-12, rtol=1e-12)
            out[0, target, channel].backward()
            self.assertEqual(x.grad[0, excluded].abs().sum().item(), 0)
            self.assertGreater(x.grad.abs().sum().item(), 0)
        # A reset at the final forward window would give a different value.
        reset, _ = rnn(x[:, 6:])
        self.assertGreater((out[0, 7, 0] - reset[0, 1, 0]).abs().item(), 1e-4)

    def test_eval_and_no_grad_use_original_gru_without_truncation(self):
        encoder = BiGRUContextEncoder(2, 3, 2, tbptt_window=3)
        x = torch.randn(2, 2, 11, requires_grad=True)
        mask = torch.arange(11)[None] < torch.tensor([11, 5])[:, None]
        calls = []
        handle = encoder.rnn.register_forward_hook(lambda *_: calls.append(True))
        try:
            train_out = encoder(x, mask)
            self.assertEqual(len(calls), 0)
            encoder.eval()
            eval_out = encoder(x, mask)
            self.assertEqual(len(calls), 1)
            encoder.train()
            with torch.no_grad():
                no_grad_out = encoder(x, mask)
            self.assertEqual(len(calls), 2)
            torch.testing.assert_close(train_out, eval_out, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(no_grad_out, eval_out, atol=0, rtol=0)
        finally:
            handle.remove()

    def test_rejects_incompatible_dropout_or_capture(self):
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            BiGRUContextEncoder(1, 2, tbptt_window=-1)
        with self.assertRaisesRegex(ValueError, "dropout=0"):
            BiGRUContextEncoder(1, 2, num_layers=2, dropout=0.1, tbptt_window=3)
        with self.assertRaisesRegex(ValueError, "failure capture"):
            BiGRUContextEncoder(1, 2, tbptt_window=3, failure_capture_dir="unused")

    def test_1024_window_prevents_controlled_bf16_temporal_overflow(self):
        # Controlled unstable recurrence, NOT a reproduction of the H100 run.
        # Forward h=0 is unchanged; only the long product of Jacobians fails.
        full = BiGRUContextEncoder(1, 1)
        full.output_norm = nn.Identity()
        with torch.no_grad():
            for parameter in full.parameters():
                parameter.zero_()
            full.rnn.weight_hh_l0[2, 0] = 2.258
        truncated = copy.deepcopy(full)
        truncated.tbptt_window = 1024
        outputs = []
        for model in (full, truncated):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                output = model(torch.zeros(1, 1, 1463))
                eta = output[0, 0, -1]
                loss = 0.002 * nb2_nll_from_log_mean(eta.new_tensor(2), eta, eta.new_tensor(0))
            self.assertTrue(loss.isfinite())
            outputs.append(output.detach())
            loss.backward()
        torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
        self.assertFalse(full.rnn.bias_ih_l0.grad.isfinite().all())
        self.assertTrue(all(p.grad.isfinite().all() for p in truncated.parameters()))
        clip_grad_norm_stable(truncated.parameters(), max_norm=1.0)
        self.assertTrue(all(p.grad.isfinite().all() for p in truncated.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_bf16_packed_window_backward(self):
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            self.skipTest("Native BF16 unavailable")
        model = BiGRUContextEncoder(8, 16, 2, tbptt_window=1024).cuda()
        x = torch.randn(3, 8, 2055, device="cuda", requires_grad=True)
        lengths = (2055, 1025, 17)
        mask = torch.arange(2055, device="cuda")[None] < torch.tensor(lengths, device="cuda")[:, None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, mask, lengths)
            loss = out.float().square().mean()
        loss.backward()
        self.assertTrue(x.grad.isfinite().all())
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
