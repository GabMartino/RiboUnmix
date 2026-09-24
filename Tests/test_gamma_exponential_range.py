import math
import unittest

import torch

from Tests.test_gamma_centering import _fixed_reference_helper


class _ExtremeBoundedScores(torch.nn.Module):
    def __init__(self, sign):
        super().__init__()
        # Opposite nearly constant profiles with one exceptional position
        # approach the 4 * raw_bound limit after both centering adjustments.
        scores = torch.full((114, 256), 8.0)
        scores[:, 0] = -8.0
        scores[0] = -scores[0]
        self.scores = torch.nn.Parameter(sign * scores)

    def forward(self, *, dataset_ids, mask, **kwargs):
        del kwargs
        return {
            "gamma_raw": self.scores[dataset_ids].clamp(-8.0, 8.0).to(torch.bfloat16)
            * mask
        }


class GammaExponentialRangeTests(unittest.TestCase):
    def test_both_gauges_at_extreme_scores_keep_exp_and_backward_finite(self):
        for training in (False, True):
            for sign in (-1.0, 1.0):
                with self.subTest(training=training, sign=sign):
                    model = _fixed_reference_helper(list(range(114)), [1.0] * 114)
                    model.dataset_bias_model = _ExtremeBoundedScores(sign)
                    model.train(training)
                    ids = torch.tensor([0]) if training else torch.arange(114)
                    mask = torch.ones(len(ids), 256, dtype=torch.bool)
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        raw = model.dataset_bias_model(dataset_ids=ids, mask=mask)["gamma_raw"]
                        result = model._center_log_gamma_fixed_reference(
                            raw,
                            mask_b=mask,
                            sample_ids=["t"] * len(ids),
                            id_datasets=ids,
                            codon_ids=torch.zeros_like(mask, dtype=torch.long),
                            position_features=torch.zeros(len(ids), 256, 1),
                            dataset_bias_sequence_features=None,
                            biological_sequence_features=None,
                        )
                        log_gamma = result["log_gamma"]
                        gamma = log_gamma.exp()
                    self.assertEqual(log_gamma.dtype, torch.float32)
                    self.assertGreater(float(log_gamma.abs().max().detach()), 31.0)
                    self.assertLessEqual(float(log_gamma.abs().max().detach()), 32.0)
                    self.assertTrue(bool(torch.isfinite(gamma).all()))
                    self.assertTrue(bool((gamma > 0).all()))
                    gamma.sum().backward()
                    gradient = model.dataset_bias_model.scores.grad
                    self.assertIsNotNone(gradient)
                    self.assertTrue(bool(torch.isfinite(gradient).all()))

    def test_bf16_range_has_headroom_for_the_final_log_gamma_bound(self):
        self.assertGreater(math.log(torch.finfo(torch.bfloat16).max), 88.0)
        x = torch.tensor([-32.0, 32.0], dtype=torch.bfloat16, requires_grad=True)
        gamma = x.exp()
        gamma.sum().backward()
        self.assertTrue(bool(torch.isfinite(gamma).all()))
        self.assertTrue(bool((gamma > 0).all()))
        self.assertTrue(bool(torch.isfinite(x.grad).all()))
