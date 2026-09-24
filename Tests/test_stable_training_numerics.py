"""Value/gradient equivalence, extreme-value regression and AMP smoke tests."""
import math
import unittest

import torch
from torch.nn.utils.rnn import pack_padded_sequence

from Models.utils.stable_numerics import (
    log_softplus, masked_logmeanexp, masked_mean, nb2_nll_from_log_mean, nb_vst,
    clip_grad_norm_stable,
)
from Models.RiboUnmixLightningModule import NegativeBinomialProfileLoss, masked_pcc
from Tests.test_alpha_causality_modes import _small_model
from Tests.test_weight_and_loss_contracts import _make_loss_test_module


class StableTrainingNumericsTests(unittest.TestCase):
    def test_nb_matches_fp64_reference_values_and_both_gradients(self):
        torch.manual_seed(103)
        y = torch.rand(3, 7, dtype=torch.float64) * 20.0
        log_mu = torch.randn_like(y).requires_grad_()
        log_alpha = torch.randn_like(y).clamp(-5, 1).requires_grad_()
        r = (-log_alpha).exp()
        old = (torch.lgamma(r) - torch.lgamma(y+r) + torch.lgamma(y+1)
               -r*r.log() -y*log_mu +(r+y)*(r+log_mu.exp()).log())
        new = nb2_nll_from_log_mean(y, log_mu, log_alpha)
        torch.testing.assert_close(new, old, atol=1e-12, rtol=1e-12)
        for actual, expected in zip(torch.autograd.grad(new.sum(), (log_mu, log_alpha), retain_graph=True),
                                    torch.autograd.grad(old.sum(), (log_mu, log_alpha))):
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-11)
        self.assertTrue(torch.autograd.gradcheck(
            lambda m, a: nb2_nll_from_log_mean(y, m, a), (log_mu, log_alpha)))

    def test_log_mean_beyond_exp_range_has_finite_correct_gradient(self):
        log_mu = torch.tensor([[1000., 100., -1000.]], requires_grad=True)
        log_alpha = torch.tensor([[0., 1., -1.]], requires_grad=True)
        target = torch.tensor([[0., 25., 2.]])
        loss = NegativeBinomialProfileLoss(experiment_mode='standard_nb', nb_mean_gradient_beta=0)(
            mu_phys=None, log_mu_phys=log_mu, log_sigma=log_alpha,
            y_true=target, mask=torch.ones_like(target, dtype=torch.bool))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(log_mu.grad).all())
        self.assertTrue(torch.isfinite(log_alpha.grad).all())
        torch.testing.assert_close(log_mu.grad, torch.tensor([[1./3, math.exp(-1)/3, 0.]]))

    def test_large_count_gamma_ratio_avoids_cancellation(self):
        # r=1 is geometric: NLL at mu=y approaches 1+log(y).
        for dtype in (torch.float32, torch.float64):
            y = torch.tensor([[1e20]], dtype=dtype)
            log_mu = y.log().requires_grad_()
            alpha = torch.zeros_like(y, requires_grad=True)
            result = nb2_nll_from_log_mean(y, log_mu, alpha)
            torch.testing.assert_close(result, 1+y.log(), atol=1e-5, rtol=1e-6)
            result.sum().backward()
            self.assertTrue(torch.isfinite(alpha.grad).all())
        # Compare large-target asymptotics and alpha gradients against the
        # direct FP64 definition where it still has adequate significant bits.
        y = torch.tensor([[1e4, 1e5]], dtype=torch.float64)
        a = torch.tensor([[-2., 0.7]], dtype=torch.float64, requires_grad=True)
        m = y.log().requires_grad_()
        r = (-a).exp()
        old = (torch.lgamma(r)-torch.lgamma(y+r)+torch.lgamma(y+1)
               -r*r.log()-y*m+(r+y)*(r+m.exp()).log())
        new = nb2_nll_from_log_mean(y, m, a)
        torch.testing.assert_close(new, old, atol=1e-8, rtol=1e-8)
        self.assertTrue(torch.autograd.gradcheck(lambda aa: nb2_nll_from_log_mean(y, m, aa), (a,)))

    def test_pcc_preserves_regularization_values_and_gradients(self):
        x = torch.tensor([[1., 3., 2., 5.], [2., 2., 2., 2.]], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([[2., 1., 5., 6.], [0., 1., 0., 2.]], dtype=torch.float64)
        mask = torch.ones_like(x, dtype=torch.bool)
        xc, yc = x-x.mean(-1, keepdim=True), y-y.mean(-1, keepdim=True)
        expected = (xc*yc).mean(-1)/((xc.square().mean(-1)+1e-8)*(yc.square().mean(-1)+1e-8)).sqrt()
        actual = masked_pcc(x, y, mask)['pcc_per_sample']
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        ga = torch.autograd.grad(actual.sum(), x, retain_graph=True)[0]
        ge = torch.autograd.grad(expected.sum(), x)[0]
        torch.testing.assert_close(ga, ge, atol=1e-8, rtol=1e-10)

    def test_pcc_extreme_amplitudes_and_flat_profiles_keep_finite_backward(self):
        for amplitude in (1., 1e10, 1e20, 1e30):
            x = (torch.tensor([[1., 3., 2., 5.], [2., 2., 2., 2.]])*amplitude).requires_grad_()
            y = torch.tensor([[2., 1., 5., 6.], [0., 1., 0., 2.]])*amplitude
            result = masked_pcc(x, y, torch.ones_like(x, dtype=torch.bool))['pcc_per_sample']
            result.sum().backward()
            self.assertTrue(torch.isfinite(result).all(), amplitude)
            self.assertTrue(torch.isfinite(x.grad).all(), amplitude)
            self.assertEqual(float(result[1].detach()), 0.)

    def test_vst_matches_direct_formula_and_zero_gradient(self):
        x = torch.tensor([[0., 1., 10., 1e5]], dtype=torch.float64, requires_grad=True)
        alpha = torch.tensor([[0.01, 0.1, 1., 3.]], dtype=torch.float64, requires_grad=True)
        old = 2/(alpha+1e-8).sqrt()*torch.asinh((alpha*x+1e-8).sqrt())
        new = nb_vst(x, alpha, 1e-8)
        torch.testing.assert_close(new, old, atol=1e-12, rtol=1e-12)
        for ga, ge in zip(torch.autograd.grad(new.sum(), (x, alpha), retain_graph=True),
                          torch.autograd.grad(old.sum(), (x, alpha))):
            torch.testing.assert_close(ga, ge, atol=1e-10, rtol=1e-10)
        extreme = torch.tensor([[1e38]], requires_grad=True)
        v = nb_vst(extreme, torch.full_like(extreme, 20.), 1e-8)
        v.sum().backward()
        self.assertTrue(torch.isfinite(v).all())
        self.assertTrue(torch.isfinite(extreme.grad).all())

    def test_log_softplus_normalization_handles_underflow_and_padding(self):
        for value in (-1000., 1000.):
            logits = torch.full((2, 8), value, requires_grad=True)
            mask = torch.tensor([[True]*6+[False]*2, [False]*8])
            log_w = log_softplus(logits)
            log_norm = log_w-masked_logmeanexp(log_w, mask)
            norm = torch.where(mask, log_norm.exp(), 0.)
            torch.testing.assert_close(norm[0, :6].mean(), torch.tensor(1.), atol=5e-5, rtol=5e-5)
            norm.square().sum().backward()
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_masked_extremes_do_not_contaminate_loss_or_backward(self):
        x = torch.tensor([[1., float('nan'), float('inf')]], requires_grad=True)
        a = torch.tensor([[0., float('nan'), float('inf')]], requires_grad=True)
        loss = NegativeBinomialProfileLoss(experiment_mode='standard_nb', nb_mean_gradient_beta=0)(
            mu_phys=x, log_sigma=a, y_true=torch.tensor([[2., 1., 1.]]),
            mask=torch.tensor([[True, False, False]]))
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(a.grad).all())
        with self.assertRaisesRegex(FloatingPointError, 'NB mean'):
            NegativeBinomialProfileLoss()(mu_phys=x, log_sigma=a,
                y_true=torch.ones_like(x), mask=torch.ones_like(x, dtype=torch.bool))

    def test_masked_mean_does_not_overflow_its_sum(self):
        x = torch.full((1, 100), 1e37)
        self.assertFalse(torch.isfinite(x.sum()))
        torch.testing.assert_close(masked_mean(x, torch.ones_like(x, dtype=torch.bool)), x[:, :1])

    def test_global_gradient_clip_preserves_finite_extreme_gradient_direction(self):
        for magnitude in (0., 0.01, 1., 1e30):
            parameter = torch.nn.Parameter(torch.zeros(2))
            parameter.grad = torch.tensor([3., 4.]) * magnitude
            reference = torch.nn.Parameter(parameter.detach().double())
            reference.grad = parameter.grad.double()
            torch.nn.utils.clip_grad_norm_([reference], max_norm=1.)
            clip_grad_norm_stable([parameter], 1.)
            torch.testing.assert_close(parameter.grad, reference.grad.float())

    def _amp_steps(self, device, length, bias_precision='inherit', tbptt_window=0):
        torch.manual_seed(112)
        model = _small_model('learned').to(device).train()
        model.dataset_bias_model.local_context_gru.precision = bias_precision
        model.dataset_bias_model.local_context_gru.tbptt_window = tbptt_window
        with torch.no_grad():
            model.dataset_bias_model.observation_bias_head.log_bias_head.weight.fill_(0.02)
        module = _make_loss_test_module().to(device)
        module.loss_fn = NegativeBinomialProfileLoss(experiment_mode='standard_nb', nb_mean_gradient_beta=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        values = torch.randn(2, length, 5, device=device)
        lengths = torch.tensor([length, length-7])
        mask = torch.arange(length, device=device)[None] < lengths.to(device)[:, None]
        target = torch.poisson(torch.ones(2, length, device=device)*2) * mask
        ids = torch.tensor([0, 1], device=device)
        codons = torch.randint(0, 8, (2, length), device=device)
        observed = []
        handle = model.dataset_bias_model.observation_bias_head.log_bias_head.register_forward_hook(
            lambda _m, _a, out: observed.append(out.dtype))
        recurrent = {"biology": [], "bias": []}
        def record(name):
            def hook(_m, _a, out):
                recurrent[name].append((out[0].data.dtype, torch.is_autocast_enabled(device)))
            return hook
        rnn_handles = [
            model.biological_model.rnn.register_forward_hook(record("biology")),
            model.dataset_bias_model.local_context_gru.rnn.register_forward_hook(record("bias")),
        ]
        for _ in range(2):
            optimizer.zero_grad()
            packed = pack_padded_sequence(values, lengths, batch_first=True)
            with torch.autocast(device, dtype=torch.bfloat16):
                mu, log_sigma, extras = model(packed, codons, ids, mask, target,
                    sample_ids=['t1','t2'], transcript_group_index=ids)
                out = dict(mu=mu, log_sigma=log_sigma, extras=extras, target=target, mask=mask,
                           replica_profiles=target[:, None], replica_mask=torch.ones(2, 1, device=device, dtype=torch.bool))
                replica = module._compute_replica_loss_terms(out, optimize_with_reweighted_nb=True)
                consensus = module._compute_consensus_loss_terms(out, target, mask)
                loss = (replica['nll_per_sample'] + 0.5*consensus['raw_pcc_diag']['loss_per_sample']
                        + 0.5*consensus['nb_vst_pcc_diag']['loss_per_sample']).mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(model.dataset_bias_model.local_context_gru.rnn.weight_ih_l0.grad.abs().sum()), 0.)
            optimizer.step()
        handle.remove()
        for rnn_handle in rnn_handles:
            rnn_handle.remove()
        self.assertTrue(observed and all(dtype == torch.bfloat16 for dtype in observed))
        if device == 'cuda':
            self.assertTrue(recurrent['biology'])
            if not tbptt_window:
                self.assertTrue(recurrent['bias'])
            for branch in recurrent.values():
                self.assertTrue(all(dtype == torch.float32 and not amp for dtype, amp in branch))

    def test_cpu_bf16_model_and_all_active_losses_backward_and_update(self):
        self._amp_steps('cpu', 64)

    def test_cpu_long_sequence_amp_smoke(self):
        # Real failing lengths, but small synthetic model/data: not a replay
        # of the production CUDA kernel, weights, dropout RNG or minibatch.
        self._amp_steps('cpu', 3175)

    def test_cpu_5089_codons_fp32_bias_gru_with_bf16_heads_and_log_losses(self):
        # Length of the latest N040 report; synthetic data and initial weights,
        # not the unavailable epoch-20 CUDA failure state.
        self._amp_steps('cpu', 5089, bias_precision='float32')

    def test_cpu_5089_codons_bf16_tbptt_with_full_cds_losses_and_update(self):
        self._amp_steps('cpu', 5089, tbptt_window=1024)

    @unittest.skipUnless(torch.cuda.is_available(), 'Requires a CUDA GPU; CPU AMP is not a CUDA reproduction.')
    def test_cuda_bf16_long_sequence_backward_and_update(self):
        self.assertTrue(torch.cuda.is_bf16_supported(including_emulation=False))
        for length in (1807, 3175, 5089):
            for bias_precision in ('inherit', 'float32'):
                with self.subTest(length=length, bias_precision=bias_precision):
                    self._amp_steps('cuda', length, bias_precision=bias_precision)
            self._amp_steps('cuda', length, tbptt_window=1024)


if __name__ == '__main__':
    unittest.main()
