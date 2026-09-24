#!/usr/bin/env python3
"""Exercise the real replica-NB helper with explicit synthetic counts; no training.

The two constant profiles deliberately differ only in amplitude. They illustrate
which discrepancies the current observation-derived replicate scales absorb;
they are not a simulation establishing statistical bias for real experiments.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize_scalar
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Models.RiboUnmixLightningModule import (
    NegativeBinomialProfileLoss, RiboUnmixLightningModule,
)


class CaptureNB(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = NegativeBinomialProfileLoss(log_alpha_min=-5., log_alpha_max=1.,
            experiment_mode="standard_nb", nb_mean_gradient_beta=0.)
        self.captured = {}

    def forward(self, **kwargs):
        self.captured = {key: value.detach().clone() for key, value in kwargs.items()
                         if torch.is_tensor(value)}
        return self.loss(**kwargs)


def main():
    module = RiboUnmixLightningModule.__new__(RiboUnmixLightningModule)
    nn.Module.__init__(module)
    module.loss_fn = CaptureNB()
    module.replica_nb_weight = 1.
    module.eps = 1e-8
    replicas = torch.tensor([[[10., 10., 10., 10.], [30., 30., 30., 30.]]])
    shape = torch.ones(1, 4, requires_grad=True)
    log_alpha = torch.full((1, 4), np.log(.2), requires_grad=True)
    out = dict(replica_profiles=replicas, replica_mask=torch.ones(1, 2, dtype=torch.bool),
               mask=torch.ones(1, 4, dtype=torch.bool), log_sigma=log_alpha,
               extras={"normalized_shape": shape})
    result = module._compute_replica_loss_terms(out, optimize_with_reweighted_nb=False)
    captured = module.loss_fn.captured
    mu = captured["mu_phys"]
    alpha = captured["log_sigma"].exp()
    torch.testing.assert_close(mu[:, 0], torch.tensor([10., 30.]))
    torch.testing.assert_close(alpha, torch.full((2, 4), .2))
    result["nll_per_sample"].sum().backward()
    assert log_alpha.grad is not None and torch.isfinite(log_alpha.grad).all()

    # Compare alpha fitting under the implemented means and one shared mean.
    # This is a one-dimensional diagnostic, not a neural model retraining.
    def best_alpha(means):
        def objective(value):
            return float(module.loss_fn.loss(mu_phys=means,
                log_sigma=torch.full_like(means, value), y_true=replicas.reshape(2, 4),
                mask=torch.ones(2, 4, dtype=torch.bool)).detach())
        fit = minimize_scalar(objective, bounds=(-5., 1.), method="bounded")
        candidates = [(float(fit.x), float(fit.fun)), (-5., objective(-5.)), (1., objective(1.))]
        value, nll = min(candidates, key=lambda item: item[1])
        return dict(alpha=float(np.exp(value)), mean_nll=nll)

    shared_mu = torch.full((2, 4), 20.)
    payload = dict(
        kind="explicit_synthetic_diagnostic",
        replica_profiles=replicas[0].tolist(),
        consensus_scale=20., replica_scales=replicas.mean(dim=2)[0].tolist(),
        actual_replica_mu=mu.tolist(), shared_alpha=alpha.tolist(),
        actual_replica_variance=(mu + alpha * mu.square()).tolist(),
        alternative_shared_mu=shared_mu.tolist(),
        alternative_shared_variance=(shared_mu + .2 * shared_mu.square()).tolist(),
        alpha_gradient=log_alpha.grad.tolist(),
        actual_means_alpha_fit=best_alpha(mu), shared_mean_alpha_fit=best_alpha(shared_mu),
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
