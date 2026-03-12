import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric


class CrossedPearsonCorrelation(nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x, y, mask):

        def cpcc(x, y):
            if x.numel() < 2 or y.numel() < 2:
                return torch.tensor(0.0, device=x.device, dtype=x.dtype, requires_grad=False)
            try:
                r = torch.corrcoef(torch.stack((x, y), dim=0))[0, 1]
                x_std, x_mean = torch.std_mean(x)
                y_std, y_mean = torch.std_mean(y)

                penalization_term = (y_mean - x_mean) ** 2
                denom_1 = (x_std.pow(2) + penalization_term + 1e-8).sqrt()
                denom_2 = (y_std.pow(2) + penalization_term + 1e-8).sqrt()
                result = (r * x_std * y_std - penalization_term) / (denom_1 * denom_2)
                return result

            except Exception as e:
                # Return a tensor instead of a float
                return torch.tensor(
                    0.0,
                    dtype=x.dtype,
                    device=x.device,
                    requires_grad=False  # Preserve gradient tracking
                )

        batch_cpcc_data = torch.stack([
            cpcc(p[m], t[m]) for p, t, m in zip(x, y, mask)
        ])

        return batch_cpcc_data.mean()


class PearsonCorrelation(nn.Module):
    def __init__(self, reduction = "none"):
        super().__init__()
        assert reduction in ["batch_mean", "sum", "none"]
        self.reduction = reduction

    @torch.no_grad()
    def forward(self, x, y, mask):

        def corr_coef(X):
            if X.shape[0] < 2:
                return -1
            try:
                r = torch.corrcoef(X)[0, 1]
            except:
                return 0.0
            if torch.isnan(r).any():
                return 0.0
            return r
        batch_pcc_data = torch.tensor([corr_coef(torch.stack((p[m],t[m]), dim=0)) for p, t, m in zip(x, y, mask)], device=x.device)
        if self.reduction == "sum":
            return batch_pcc_data.sum()
        elif self.reduction == "batch_mean":
            return batch_pcc_data.mean()
        else:
            return batch_pcc_data




class MSELossWithMask(nn.Module):
    def __init__(self, reduction='batch_mean'):
        super().__init__()
        assert reduction in ["batch_mean", "sum", "none"]
        self.reduction = reduction ## total_mean => mean over the non masked sample for all the batches
                                    ## batch_mean => first mean over the non masked samples of each batch, then mean the batches loss values

    def forward(self, pred, target, mask):
        assert pred.shape == target.shape == mask.shape
        def masked_mse_loss(p, t, m):
            assert p.shape == t.shape == m.shape
            size = sum(m) ## average over non zeros
            loss = torch.where(m, (p - t)**2, 0)
            return loss.sum() / size ## Average of the sequence
        loss = torch.vmap(masked_mse_loss, in_dims=(0, 0, 0))(pred, target, mask)
        if self.reduction == "batch_mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class HuberLossWithMask(nn.Module):
    def __init__(self, reduction='batch_mean'):
        super().__init__()
        assert reduction in ["batch_mean", "sum", "none"]
        self.reduction = reduction ## total_mean => mean over the non masked sample for all the batches
                                    ## batch_mean => first mean over the non masked samples of each batch, then mean the batches loss values

    def forward(self, pred, target, mask, delta):
        assert pred.shape == target.shape == mask.shape
        def masked_mse_loss(p, t, m):
            assert p.shape == t.shape == m.shape
            size = sum(m) ## average over non zeros
            loss = torch.where(m, F.huber_loss(p, t, delta=delta), 0)
            return loss.sum() / size ## Average of the sequence
        loss = torch.vmap(masked_mse_loss, in_dims=(0, 0, 0))(pred, target, mask)
        if self.reduction == "batch_mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class CrossEntropyLossWithMask(nn.Module):
    def __init__(self, reduction='total_mean', padding_idx = -1):
        super().__init__()
        assert reduction in ["batch_mean", "sum", "none"]
        self.reduction = reduction

        self.padding_idx = -1
    def forward(self, pred, target, mask):
        assert target.shape == mask.shape

        def masked_cross_entropy_loss(pred, target, mask):
            assert target.shape == mask.shape
            size = sum(mask) ## average over non zeros
            loss = torch.where(mask, F.cross_entropy(pred, target, reduction="none", ignore_index=self.padding_idx), 0)

            return loss.sum() / size

        loss = torch.vmap(masked_cross_entropy_loss, in_dims=(0, 0, 0))(pred, target, mask)
        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == "batch_mean":
            return loss.mean()
        else:
            return loss



class R2WithMask(nn.Module):
    """
    Computes masked R^2 per batch item, with reduction.
    By default returns a *loss* (1 - R^2). Set return_score=True to get R^2.

    Args:
        reduction: 'mean' | 'sum' | 'none'
        return_score: if True, return R^2; if False (default), return 1 - R^2
        eps: small constant to avoid div-by-zero
    """
    def __init__(self, reduction: str = 'mean', return_score: bool = True, eps: float = 1e-12):
        super().__init__()
        if reduction not in {'mean', 'sum', 'none'}:
            raise ValueError("reduction must be one of {'mean','sum','none'}")
        self.reduction = reduction
        self.return_score = return_score
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape or pred.shape != mask.shape:
            raise ValueError("pred, target, and mask must have the same shape")

        # Flatten per-sample, keep batch dimension
        B = pred.shape[0]
        p = pred.reshape(B, -1)
        t = target.reshape(B, -1)
        m = mask.reshape(B, -1).to(dtype=torch.bool)

        # Count valid elements per sample
        counts = m.sum(dim=1)  # (B,)
        has_any = counts > 0

        # Safe mean of target over masked region
        counts_safe = counts.clamp_min(1).to(t.dtype)              # avoid div-by-zero
        t_sum = (t * m).sum(dim=1)
        t_mean = t_sum / counts_safe                                # (B,)

        # Sum of squares (masked)
        ss_tot = ((t - t_mean.unsqueeze(1))**2 * m).sum(dim=1)      # variance of TARGET, not pred
        ss_res = ((p - t)**2 * m).sum(dim=1)

        # Handle degenerate cases:
        #  - No valid mask => define R^2 = 0 (neutral); you may prefer NaN.
        #  - Zero variance in target => sklearn convention: 1.0 if perfect fit, else 0.0.
        perfect_fit = ss_res <= self.eps
        nondeg = ss_tot > self.eps

        r2_raw = 1.0 - ss_res / (ss_tot + self.eps)
        r2_when_degenerate = torch.where(perfect_fit, torch.ones_like(r2_raw), torch.zeros_like(r2_raw))
        r2 = torch.where(has_any,
                         torch.where(nondeg, r2_raw, r2_when_degenerate),
                         torch.zeros_like(r2_raw))

        out = r2 if self.return_score else (1.0 - r2)

        if self.reduction == 'mean':
            return out.mean()
        elif self.reduction == 'sum':
            return out.sum()
        else:
            return out  # (B,)





class SpearmanRWithMask(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                    y: torch.Tensor,
                    mask: Optional[torch.Tensor] = None,
                    tol: float = 0.0
                ) -> torch.Tensor:
        """
        Batched Spearman's rank correlation (ρ) with masking and average-tie handling.

        Args:
            x, y: tensors of shape (B, N)
            mask: optional boolean tensor of shape (B, N); True = keep, False = ignore
                  If provided, elements where mask==False are dropped *per row*
                  (applies jointly to x and y).
            tol:  absolute tolerance for considering two values tied (default 0.0).

        Returns:
            rho: tensor of shape (B,) with Spearman ρ per batch row.
                 Rows with < 2 valid points or zero rank variance return NaN.
        """
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("x and y must be 2D tensors of shape (B, N).")
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape.")
        if mask is not None:
            if mask.shape != x.shape or mask.dtype != torch.bool:
                raise ValueError("mask must be bool with same shape as x/y.")

        B, N = x.shape
        rhos = x.new_full((B,), float("nan"))

        for b in range(B):
            if mask is not None:
                m = mask[b]
                # keep only entries where mask is True (and drop NaNs if present)
                xb = x[b][m]
                yb = y[b][m]
            else:
                xb = x[b]
                yb = y[b]

            # Optionally also drop NaNs (robustness)
            good = torch.isfinite(xb) & torch.isfinite(yb)
            xb = xb[good]
            yb = yb[good]

            n = xb.numel()
            if n < 2:
                rhos[b] = float("nan")
                continue

            # Rank with average ties
            rx = _rankdata_average_ties_1d(xb, tol=tol)
            ry = _rankdata_average_ties_1d(yb, tol=tol)

            # Pearson on ranks
            rx = rx - rx.mean()
            ry = ry - ry.mean()
            sx = torch.linalg.norm(rx)
            sy = torch.linalg.norm(ry)

            if sx == 0 or sy == 0:
                rhos[b] = float("nan")  # all-equal ranks -> undefined
                continue

            rhos[b] = (rx @ ry) / (sx * sy)

        return rhos




def _rankdata_average_ties_1d(v: torch.Tensor, tol: float = 0.0) -> torch.Tensor:
    """
    Average-rank a 1D tensor (ties get the average of their positions).
    Ranks are 1..m (float). Works on CPU/GPU. Equality tolerance via `tol`.
    """
    m = v.numel()
    if m == 0:
        return v.new_empty(0)

    # Sort values and remember original positions
    order = torch.argsort(v, stable=True)
    vs = v[order]
    # Positions 0..m-1 in the sorted order (we'll +1 at the end)
    pos = torch.arange(m, device=v.device, dtype=v.dtype)

    # Find tie groups (runs of equal values, with optional tolerance)
    if tol > 0:
        eq = torch.isclose(vs[1:], vs[:-1], rtol=0.0, atol=tol)
    else:
        eq = (vs[1:] == vs[:-1])

    # Start indices of each group
    starts = torch.cat([torch.tensor([0], device=v.device), torch.nonzero(~eq, as_tuple=False).flatten() + 1])
    # End indices (inclusive)
    ends = torch.cat([starts[1:] - 1, torch.tensor([m - 1], device=v.device)])

    # Average rank per group (in 0-based positions)
    # mean of [start..end] = (start + end)/2
    mean_pos = (starts.to(v.dtype) + ends.to(v.dtype)) / 2.0

    # Assign average positions to all members of each group
    avg_pos_sorted = pos.clone()
    for s, e, mp in zip(starts, ends, mean_pos):
        avg_pos_sorted[s:e+1] = mp

    # Map back to original order and convert to 1-based ranks
    ranks = torch.empty_like(avg_pos_sorted)
    ranks[order] = avg_pos_sorted + 1.0
    return ranks


import torch
import torch.nn as nn
import torch.nn.functional as F


class MM1QueueLikelihoodLoss(nn.Module):
    def __init__(self, eps=1e-6):
        """
        Calculates the Negative Log-Likelihood of an M/M/1 Queue (Geometric Distribution).
        """
        super().__init__()
        self.eps = eps

    def forward(self, rho_pred, y_true, mask):
        """
        Args:
            rho_pred: The predicted utilization (J * w_i) [Batch, SeqLen]
            y_true: The NORMALIZED queue size (Ribo-seq count / Transcript Abundance)
            mask: Binary mask for sequence length
        """
        # 1. Precision Quarantine
        y_true = y_true.to(torch.float32)
        rho_pred = rho_pred.to(torch.float32)

        # 2. Thermodynamic Bounds
        # M/M/1 queues explode to infinity if rho >= 1.0.
        # We strictly clamp rho to [eps, 0.999] to maintain queue stability and prevent log(0).
        rho_safe = torch.clamp(rho_pred, min=self.eps, max=1.0 - self.eps)

        # 3. Geometric Negative Log-Likelihood
        # NLL = - [ log(1 - rho) + y_true * log(rho) ]
        log_empty_prob = torch.log(1.0 - rho_safe)
        log_busy_prob = y_true * torch.log(rho_safe)

        nll = -(log_empty_prob + log_busy_prob)

        # 4. Mask and Average
        # We only calculate loss on valid codons
        masked_nll = nll * mask
        true_lengths = torch.clamp(torch.sum(mask, dim=1), min=1.0)

        # Average NLL across the transcript
        mean_nll = torch.sum(masked_nll, dim=1) / true_lengths

        # Return the batch average
        return torch.mean(mean_nll)


class ZeroInflatedExponentialLoss(nn.Module):
    def __init__(self, lambda_smooth=0.05, lambda_reg=0.1, warmup_epochs=1, eps=1e-6):
        """
        Calculates the Negative Log-Likelihood for a Zero-Inflated Exponential Distribution.
        Assumes the input data (y_true) is continuous strictly positive values or exact zeros.
        """
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.target_lambda_reg = lambda_reg
        self.warmup_epochs = warmup_epochs
        self.eps = eps

    def forward(self, rho, pi_dropout, w, y_true, mask, current_epoch):
        y_true = y_true.to(torch.float32)

        # 1. Precision & Stability
        pi_safe = torch.clamp(pi_dropout, min=self.eps, max=1.0 - self.eps)
        # rho cannot be 0, because lambda = 1/rho. We clamp it above zero.
        rho_safe = torch.clamp(rho, min=self.eps, max=1.0 - self.eps)

        # 2. ZERO-INFLATED EXPONENTIAL LOG-LIKELIHOOD
        is_zero = (y_true == 0.0).float()
        is_nonzero = 1.0 - is_zero

        # --- NLL for y == 0 ---
        # The only way to get EXACTLY 0.0 in a continuous exponential is if the dropout head fired.
        nll_zero = -torch.log(pi_safe) * is_zero

        # --- NLL for y > 0 ---
        # Likelihood L = (1 - pi) * (1 / rho) * exp(-y / rho)
        # NLL = - [ log(1 - pi) - log(rho) - (y / rho) ]
        # NLL = -log(1 - pi) + log(rho) + (y / rho)
        nll_nonzero = (
                              -torch.log(1.0 - pi_safe)
                              + torch.log(rho_safe)
                              + (y_true / rho_safe)
                      ) * is_nonzero

        # Combine NLLs
        nll_total = (nll_zero + nll_nonzero) * mask
        true_lengths = torch.clamp(torch.sum(mask, dim=1), min=1.0)

        # Batch Mean NLL
        loss_nll = torch.mean(torch.sum(nll_total, dim=1) / true_lengths)

        # 3. DROPOUT REGULARIZATION (Warmup Penalty)
        warmup_factor = min(1.0, current_epoch / max(1, self.warmup_epochs))
        current_lambda_reg = self.target_lambda_reg * warmup_factor
        loss_reg = torch.mean(pi_dropout * mask)

        # 4. WAIT-TIME SMOOTHNESS PENALTY (Total Variation)
        diff = torch.abs(w[:, 1:] - w[:, :-1])
        smooth_mask = mask[:, 1:] * mask[:, :-1]
        loss_smooth = torch.mean(torch.sum(diff * smooth_mask, dim=1) / true_lengths)

        return loss_nll + (current_lambda_reg * loss_reg) + (self.lambda_smooth * loss_smooth)
class TemporalShapeLoss(nn.Module):
    def __init__(self, alpha=0.5, reduction="mean"):
        """
        alpha: Weight for Magnitude vs Shape.
        reduction: 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, y_pred, y_true, mask=None):
        # 1. Magnitude Loss (Batch, Len)
        # Direct MSE on the normalized data
        mag_loss = (y_pred - y_true) ** 2

        # 2. Difference Loss (Batch, Len-1)
        diff_pred = y_pred[:, 1:] - y_pred[:, :-1]
        diff_true = y_true[:, 1:] - y_true[:, :-1]

        # Multiply by 10.0 to balance the scale
        diff_loss_raw = (diff_pred - diff_true) ** 2 * 10.0

        # 3. PAD Difference Loss to match (Batch, Len)
        # We add a column of zeros at the end.
        # So the shape error at index t is stored at index t.
        # The last pixel has no "next neighbor", so its shape error is 0.
        diff_loss = F.pad(diff_loss_raw, (0, 1), "constant", 0)

        # 4. Handle Masking for Diff Loss explicitly
        # If mask at t+1 is 0, then the diff at t is invalid.
        if mask is not None:
            # Mask for magnitude
            mask_float = mask.to(mag_loss.dtype)
            mag_loss = mag_loss * mask_float

            # Mask for diff (shifted logic)
            # Valid diff at t requires mask[t]=1 AND mask[t+1]=1
            mask_diff = mask_float[:, 1:] * mask_float[:, :-1]
            mask_diff = F.pad(mask_diff, (0, 1), "constant", 0)

            diff_loss = diff_loss * mask_diff

        # 5. Combine Element-wise (Batch, Len)
        # This preserves the batch dimension!
        loss_map = (self.alpha * mag_loss) + ((1 - self.alpha) * diff_loss)

        # 6. Reductions
        if self.reduction == "none":
            return loss_map  # Returns (Batch, Len)

        if mask is not None:
            # Global Mean Reduction (Sum / Total Valid Pixels)
            return loss_map.sum() / (mask.sum() + 1e-8)

        if self.reduction == "mean":
            return loss_map.mean()

        return loss_map.sum()


class RiboSeqLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        alpha: Weight between Poisson (Count) and Pearson (Shape).
               alpha=1.0 -> Pure Poisson.
               alpha=0.0 -> Pure Shape (Correlation).
        """
        super().__init__()
        self.alpha = alpha
        self.eps = 1e-8
        self.pcc = PearsonCorrelation()

    def forward(self, y_pred, y_true, mask):
        """
        y_pred: (Batch, Len) - Predicted rates (must be positive)
        y_true: (Batch, Len) - Raw Counts
        """

        # --- 1. Poisson Loss (NLL) ---
        # Poisson Loss = y_pred - y_true * log(y_pred)

        # Safety for log to avoid nan
        y_pred_safe = y_pred.clamp(min=self.eps)

        # Calculate element-wise loss
        poisson_loss_map = y_pred_safe - y_true * torch.log(y_pred_safe)

        if mask is not None:
            term1 = poisson_loss_map[mask].mean()
        else:
            term1 = poisson_loss_map

        # --- 2. Pearson Correlation Loss (Shape) ---
        # We want to maximize correlation, so we minimize (1 - r)

        if self.alpha < 1.0:

            r = self.pcc(y_pred, y_true, mask)

            # Loss = 1 - r
            pearson_loss = 1.0 - r

            # Reduction: SUM
            term2 = pearson_loss.mean()

        else:
            term2 = torch.tensor(0.0, device=y_pred.device)

        # --- 3. Combine ---
        return self.alpha * term1 + (1.0 - self.alpha) * term2



LOG_2PI = math.log(2.0 * math.pi)

import torch
import torch.nn as nn


class MaskedCosineSimilarity(Metric):
    # FIX: Increased eps to survive BF16 rounding
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.add_state("sum_cossim", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total_batches", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        # FIX: Force everything to FP32 to prevent BF16 math collapse
        preds = preds.to(torch.float32) * mask.to(torch.float32)
        target = target.to(torch.float32) * mask.to(torch.float32)

        dot_product = torch.sum(preds * target, dim=1)
        norm_preds = torch.sqrt(torch.sum(preds ** 2, dim=1))
        norm_target = torch.sqrt(torch.sum(target ** 2, dim=1))

        cossim = dot_product / (norm_preds * norm_target + self.eps)

        self.sum_cossim += torch.sum(cossim)
        self.total_batches += preds.shape[0]

    def compute(self):
        return self.sum_cossim / (self.total_batches + self.eps)


class PeakWeightedPoissonLoss(nn.Module):
    # FIX: Increased eps to 1e-4
    def __init__(self, peak_weight=100.0, signal_threshold=0.1, eps=1e-4):
        super().__init__()
        self.peak_weight = peak_weight
        self.threshold = signal_threshold
        self.eps = eps

    def forward(self, y_pred, y_true, mask):
        # Force FP32 inside the loss
        y_pred = y_pred.to(torch.float32)
        y_true = y_true.to(torch.float32)
        mask = mask.to(torch.float32)

        # The + self.eps now safely prevents log(0) -> NaN
        nll_elementwise = (y_pred + self.eps) - y_true * torch.log(y_pred + self.eps)

        weights = torch.where(y_true > self.threshold, self.peak_weight, 1.0)
        weighted_nll = nll_elementwise * weights * mask

        loss_per_transcript = torch.sum(weighted_nll, dim=1) / torch.clamp(torch.sum(mask, dim=1), min=1.0)

        return torch.mean(loss_per_transcript)
class ZeroInflatedWassersteinLoss(nn.Module):
    def __init__(self, lambda_shape=1.0, lambda_flux=1.0, lambda_reg=0.1,
                 lambda_smooth=0.05, warmup_epochs=1, epsilon=1e-8, norm_scale=1.3137):
        super().__init__()
        self.lambda_shape = lambda_shape
        self.lambda_flux = lambda_flux
        self.target_lambda_reg = lambda_reg
        self.lambda_smooth = lambda_smooth # New: Weight for wait-time smoothness
        self.warmup_epochs = warmup_epochs
        self.eps = epsilon
        self.norm_scale = norm_scale

    def forward(self, pred_norm, pi_dropout, w, y_true, mask, current_epoch):
        """
        Args:
            w: The predicted wait times [Batch, SeqLen]
        """
        y_true = y_true.to(torch.float32)

        # 1. THE MIXTURE EXPECTATION
        linear_rho = torch.expm1(pred_norm * self.norm_scale) * mask
        expected_linear = linear_rho * (1.0 - pi_dropout)
        expected_log = torch.log1p(expected_linear) / self.norm_scale

        # 2. BALANCED FLUX LOSS
        true_lengths = torch.clamp(torch.sum(mask, dim=1), min=1.0)
        mean_expected = torch.sum(expected_log * mask, dim=1) / true_lengths
        mean_true = torch.sum(y_true * mask, dim=1) / true_lengths
        loss_flux = torch.mean((mean_expected - mean_true) ** 2)

        # 3. WASSERSTEIN SHAPE LOSS
        linear_true = torch.expm1(y_true * self.norm_scale) * mask
        sum_exp = torch.sum(expected_linear, dim=1) + self.eps
        sum_true = torch.sum(linear_true, dim=1) + self.eps

        pdf_exp = expected_linear / sum_exp.unsqueeze(1)
        pdf_true = linear_true / sum_true.unsqueeze(1)

        cdf_exp = torch.cumsum(pdf_exp, dim=1)
        cdf_true = torch.cumsum(pdf_true, dim=1)
        loss_shape = torch.mean(torch.sum(torch.abs(cdf_exp - cdf_true) * mask, dim=1) / true_lengths)

        # 4. DYNAMIC DROPOUT REGULARIZATION
        warmup_factor = min(1.0, current_epoch / self.warmup_epochs)
        current_lambda_reg = self.target_lambda_reg * warmup_factor
        loss_reg = torch.mean(pi_dropout * mask)

        # 5. WAIT-TIME SMOOTHNESS PENALTY (Total Variation)
        # We penalize the difference between codon i and i+1.
        # This prevents the physics engine from 'hunting' high-frequency noise.
        # w shape: [Batch, SeqLen]
        diff = torch.abs(w[:, 1:] - w[:, :-1])
        # Mask the differences (only where both codons are valid)
        smooth_mask = mask[:, 1:] * mask[:, :-1]
        loss_smooth = torch.mean(torch.sum(diff * smooth_mask, dim=1) / true_lengths)

        # 6. COMPOSITE
        total_loss = (self.lambda_flux * loss_flux) + \
                     (self.lambda_shape * loss_shape) + \
                     (current_lambda_reg * loss_reg) + \
                     (self.lambda_smooth * loss_smooth)

        return total_loss
class WassersteinFluxLoss(nn.Module):
    def __init__(self, lambda_shape=1.0, lambda_flux=1.0, epsilon=1e-8, norm_scale=1.3137):
        super().__init__()
        self.lambda_shape = lambda_shape
        self.lambda_flux = lambda_flux
        self.eps = epsilon
        self.norm_scale = norm_scale # The scaling factor used in your model

    def forward(self, y_pred, y_true, mask):
        y_true = y_true.to(torch.float32)
        y_pred = y_pred * mask
        y_true = y_true * mask

        true_lengths = torch.clamp(torch.sum(mask, dim=1), min=1.0)

        # --- 1. THE FLUX LOSS (Safe in Log Space) ---
        # Because y_pred and y_true are ALREADY log1p transformed,
        # a simple MSE on their means perfectly balances the loss magnitude.
        mean_pred = torch.sum(y_pred, dim=1) / true_lengths
        mean_true = torch.sum(y_true, dim=1) / true_lengths
        loss_flux = torch.mean((mean_pred - mean_true) ** 2)

        # --- 2. THE WASSERSTEIN SHAPE LOSS (Must be in Linear Space) ---
        # We MUST undo the log1p transformation to calculate a valid physical PDF.
        linear_pred = torch.expm1(y_pred * self.norm_scale) * mask
        linear_true = torch.expm1(y_true * self.norm_scale) * mask

        sum_pred_lin = torch.sum(linear_pred, dim=1)
        sum_true_lin = torch.sum(linear_true, dim=1)

        # NOW we can safely create the true physical density
        pdf_pred = linear_pred / (sum_pred_lin.unsqueeze(1) + self.eps)
        pdf_true = linear_true / (sum_true_lin.unsqueeze(1) + self.eps)

        cdf_pred = torch.cumsum(pdf_pred, dim=1) * mask
        cdf_true = torch.cumsum(pdf_true, dim=1) * mask

        wasserstein_dist = torch.sum(torch.abs(cdf_pred - cdf_true) * mask, dim=1)
        normalized_wasserstein = wasserstein_dist / true_lengths

        loss_shape = torch.mean(normalized_wasserstein)

        # --- 3. THE COMPOSITE LOSS ---
        total_loss = (self.lambda_flux * loss_flux) + (self.lambda_shape * loss_shape)

        return total_loss


class NeuralPhysicsPoissonLoss(nn.Module):
    def __init__(self, lambda_shape=1.0, epsilon=1e-8):
        """
        Calculates a composite loss using Poisson NLL for magnitude and Pearson for shape.

        Args:
            lambda_shape: Weight given to the Pearson correlation (shape) loss.
            epsilon: Small value to prevent division by zero or log(0).
        """
        super().__init__()
        self.lambda_shape = lambda_shape
        self.eps = epsilon

    def forward(self, y_pred, y_true, mask=None):
        """
        Args:
            y_pred: Predicted rate/density profile (Batch, Sequence_Length). MUST NOT be log-scaled.
            y_true: Observed RAW Ribo-seq counts (Batch, Sequence_Length). MUST NOT be log-scaled.
            mask: Binary tensor (Batch, Sequence_Length) for padded sequences.
        """
        # 1. Per-Transcript Poisson NLL (Magnitude & Local Fit Loss)
        # The Poisson NLL formula (ignoring the log(y!) constant which drops from gradients):
        # Loss = y_pred - y_true * log(y_pred)

        # We add eps to y_pred to prevent log(0) explosion if the model predicts absolute zero
        nll_elementwise = y_pred - y_true * torch.log(y_pred + self.eps)

        if mask is not None:
            # Mask out the padding
            nll_masked = nll_elementwise * mask
            # Average over valid sequence length per transcript
            nll_per_transcript = torch.sum(nll_masked, dim=1) / torch.clamp(torch.sum(mask, dim=1), min=1.0)
        else:
            nll_per_transcript = torch.mean(nll_elementwise, dim=1)

        loss_magnitude = torch.mean(nll_per_transcript)

        # 2. Pearson Correlation (Shape Loss)
        if mask is not None:
            mean_pred = torch.sum(y_pred * mask, dim=1, keepdim=True) / torch.clamp(
                torch.sum(mask, dim=1, keepdim=True), min=1.0)
            mean_true = torch.sum(y_true * mask, dim=1, keepdim=True) / torch.clamp(
                torch.sum(mask, dim=1, keepdim=True), min=1.0)

            centered_pred = (y_pred - mean_pred) * mask
            centered_true = (y_true - mean_true) * mask
        else:
            mean_pred = torch.mean(y_pred, dim=1, keepdim=True)
            mean_true = torch.mean(y_true, dim=1, keepdim=True)

            centered_pred = y_pred - mean_pred
            centered_true = y_true - mean_true

        cov = torch.sum(centered_pred * centered_true, dim=1)
        var_pred = torch.sum(centered_pred ** 2, dim=1)
        var_true = torch.sum(centered_true ** 2, dim=1)

        std_pred = torch.sqrt(torch.clamp(var_pred, min=self.eps))
        std_true = torch.sqrt(torch.clamp(var_true, min=self.eps))

        pearson_r = cov / (std_pred * std_true + self.eps)

        # Maximize correlation -> minimize (1 - r)
        loss_shape = torch.mean(1.0 - pearson_r)

        # 3. Composite Loss
        total_loss = loss_magnitude + (self.lambda_shape * loss_shape)

        return total_loss, loss_magnitude, pearson_r.mean()
class NeuralPhysicsRiboLoss(nn.Module):
    def __init__(self, lambda_shape=1.0, epsilon=1e-8):
        """
        Args:
            lambda_shape: Weight given to the Pearson correlation (shape) loss.
            epsilon: Small value to prevent division by zero in variance calculations.
        """
        super().__init__()
        self.lambda_shape = lambda_shape
        self.eps = epsilon

    def forward(self, y_pred, y_true, mask=None):
        """
        Args:
            y_pred: Predicted density profile (Batch, Sequence_Length)
            y_true: Observed Ribo-seq counts (Batch, Sequence_Length)
            mask: Binary tensor (Batch, Sequence_Length) for padded sequences.
        """
        # 1. Per-Transcript MAE (Magnitude Loss)
        abs_error = torch.abs(y_pred - y_true)
        if mask is not None:
            # Average over valid sequence length, then average across batch
            mae_per_transcript = torch.sum(abs_error * mask, dim=1) / torch.clamp(torch.sum(mask, dim=1), min=1.0)
        else:
            mae_per_transcript = torch.mean(abs_error, dim=1)

        loss_magnitude = torch.mean(mae_per_transcript)

        # 2. Pearson Correlation (Shape Loss)
        if mask is not None:
            # Masked centering
            mean_pred = torch.sum(y_pred * mask, dim=1, keepdim=True) / torch.clamp(
                torch.sum(mask, dim=1, keepdim=True), min=1.0)
            mean_true = torch.sum(y_true * mask, dim=1, keepdim=True) / torch.clamp(
                torch.sum(mask, dim=1, keepdim=True), min=1.0)

            centered_pred = (y_pred - mean_pred) * mask
            centered_true = (y_true - mean_true) * mask
        else:
            mean_pred = torch.mean(y_pred, dim=1, keepdim=True)
            mean_true = torch.mean(y_true, dim=1, keepdim=True)

            centered_pred = y_pred - mean_pred
            centered_true = y_true - mean_true

        cov = torch.sum(centered_pred * centered_true, dim=1)
        var_pred = torch.sum(centered_pred ** 2, dim=1)
        var_true = torch.sum(centered_true ** 2, dim=1)

        #pearson_r = cov / (torch.sqrt(var_pred * var_true) + self.eps)
        std_pred = torch.sqrt(torch.clamp(var_pred, min=self.eps))
        std_true = torch.sqrt(torch.clamp(var_true, min=self.eps))

        pearson_r = cov / (std_pred * std_true + self.eps)
        # We want to maximize correlation, so minimize (1 - r)
        loss_shape = torch.mean(1.0 - pearson_r)

        # 3. Composite Loss
        total_loss = loss_magnitude + (self.lambda_shape * loss_shape)

        return total_loss, loss_magnitude, pearson_r.mean()



def hurdle_lognormal_nll(y, logit_zero, mu, log_sigma, mask=None, eps=1e-6, reduction="mean"):
    """
    Robust Hurdle Loss that prevents NaNs during backprop.
    """
    # 1. Clamp Sigma to prevent division by zero or explosion
    log_sigma = torch.clamp(log_sigma, min=-3.0, max=5.0)
    sigma = torch.exp(log_sigma)

    # 2. Probability of Zero
    pi = torch.sigmoid(logit_zero)
    pi = torch.clamp(pi, 1e-6, 1.0 - 1e-6)

    # 3. Safe Targets for LogNormal Branch
    # When y=0, we swap it with a dummy value (1.0) to prevent log(0) = -inf
    # The result of this branch will be discarded by torch.where anyway,
    # but we need the gradient calculation to remain finite.
    is_zero = (y <= eps)
    y_safe = torch.where(is_zero, torch.ones_like(y), y)

    # LogNormal Calculation on SAFE data
    log_y = torch.log(y_safe)
    log_f = -(
            log_y
            + log_sigma
            + 0.5 * math.log(2 * math.pi)
            + 0.5 * ((log_y - mu) / sigma) ** 2
    )

    # 4. Combine Branches
    # If y=0: Loss is -log(pi)
    # If y>0: Loss is -log(1-pi) - log_f
    loss_pixel = torch.where(
        is_zero,
        -torch.log(pi),
        -torch.log(1 - pi) - log_f
    )

    # 5. Apply Masking
    if mask is not None:
        loss_pixel = loss_pixel * mask
        if reduction == "mean":
            return loss_pixel.sum() / (mask.sum() + 1e-8)

    if reduction == "mean":
        return loss_pixel.mean()

    return loss_pixel.sum()


class TweedieLoss(nn.Module):
    def __init__(self, p=1.5, eps=1e-8, reduction='mean'):
        """
        Args:
            p (float): Tweedie power parameter. 1 < p < 2.
                       1.5 is standard for zero-inflated continuous data.
            eps (float): Stability epsilon.
        """
        super().__init__()
        assert 1 < p < 2, "p must be between 1 and 2 for Compound Poisson-Gamma"
        self.p = p
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_log_mu, target, mask=None):
        """
        Args:
            pred_log_mu: Log of the predicted mean. Shape (Batch, Len).
            target: True continuous values (>= 0). Shape (Batch, Len).
            mask: Boolean mask for padding.
        """
        # 1. Recover Mu (safely)
        mu = torch.exp(pred_log_mu)

        # 2. Calculate Term A (Target * Mu^(1-p) / (1-p))
        # We compute this in log-space for stability: log(mu^(1-p)) = (1-p)*log(mu)
        # term_a = target * exp((1-p) * log_mu) / (1-p)

        # Note: (1-p) is negative, so we divide by a negative.
        # This corresponds to the '-y * mu^(1-p) / (1-p)' part of the formula.

        term_a = target * torch.exp((1 - self.p) * pred_log_mu) / (1 - self.p)

        # 3. Calculate Term B (Mu^(2-p) / (2-p))
        term_b = torch.exp((2 - self.p) * pred_log_mu) / (2 - self.p)

        # 4. Loss = -Term A + Term B
        # The standard formulation is minimization of negative log-likelihood
        loss = -term_a + term_b

        # 5. Apply Masking
        if mask is not None:
            mask = mask.to(loss.dtype)
            loss = loss * mask

            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() + self.eps)
            elif self.reduction == 'sum':
                return loss.sum()

        if self.reduction == 'mean':
            return loss.mean()

        return loss


class CalibrationError(Metric):
    def __init__(self, n_bins=10, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.n_bins = n_bins
        # We accumulate all predictions and targets to calculate global calibration
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, pred_mu: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        """
        pred_mu: The predicted MEAN (exp(log_mu)), not the log values.
        target: The actual observed counts.
        mask: Optional boolean mask.
        """
        # Flatten everything
        p = pred_mu.flatten()
        t = target.flatten()

        if mask is not None:
            m = mask.flatten().bool()
            p = p[m]
            t = t[m]

        self.preds.append(p)
        self.targets.append(t)

    def compute(self):
        # Concatenate all batches
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)

        if len(preds) == 0:
            return torch.tensor(0.0)

        # 1. Sort by prediction (to group similar predicted means)
        sorted_indices = torch.argsort(preds)
        preds_sorted = preds[sorted_indices]
        targets_sorted = targets[sorted_indices]

        # 2. Binning
        bin_size = len(preds) // self.n_bins
        ece = 0.0
        total_samples = 0

        for i in range(self.n_bins):
            start = i * bin_size
            # The last bin takes any remainder
            end = (i + 1) * bin_size if i < self.n_bins - 1 else len(preds)

            if start >= end:
                break

            # 3. Compare Predicted Mean vs Observed Average
            bin_pred_mean = preds_sorted[start:end].mean()
            bin_target_mean = targets_sorted[start:end].mean()

            # The error is the absolute difference
            abs_diff = torch.abs(bin_pred_mean - bin_target_mean)

            # Weight by bin size
            current_samples = (end - start)
            ece += abs_diff * current_samples
            total_samples += current_samples

        # 4. Final Weighted Average
        return ece / total_samples


class PoissonPseudoR2(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_log_likelihood_model", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_log_likelihood_null", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("sum_y", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, pred_log_mu, target, mask=None):
        # Flatten and mask
        if mask is not None:
            mask = mask.bool()
            pred_log_mu = pred_log_mu[mask]
            target = target[mask]

        # 1. Model Log Likelihood (Poisson: y * log_mu - mu)
        mu = torch.exp(pred_log_mu)
        ll_model = target * pred_log_mu - mu

        # Accumulate
        self.sum_log_likelihood_model += ll_model.sum()
        self.sum_y += target.sum()
        self.count += target.numel()

    def compute(self):
        # 2. Null Model Log Likelihood
        # The best 'dumb' guess is the global average target
        global_mean = self.sum_y / self.count
        global_log_mean = torch.log(global_mean + 1e-8)

        # Null LL: y * log(mean) - mean
        # We can compute this analytically from sums
        ll_null = self.sum_y * global_log_mean - self.count * global_mean

        # 3. McFadden's Pseudo R2
        # 1 - (LL_Model / LL_Null) ??? No, LL are negative usually.
        # Deviance based R2: 1 - (D_model / D_null)

        # Simpler definition: 1 - (NLL_model / NLL_null)
        # NLL is -LL
        nll_model = -self.sum_log_likelihood_model
        nll_null = -ll_null

        return 1 - (nll_model / nll_null)


class HuberLossWithMask(nn.Module):
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        """
        Args:
            delta (float): Threshold where loss changes from Quadratic (MSE) to Linear (MAE).
                           For normalized data (0-1), delta=0.1 is usually recommended.
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            input: Predicted values. Shape (Batch, Len)
            target: Ground truth values. Shape (Batch, Len)
            mask: Boolean mask (Batch, Len). True = Valid, False = Padding/Ignore.
        """
        # 1. Compute element-wise Huber Loss
        # We enforce reduction='none' here so we can mask it first
        loss = F.huber_loss(input, target, delta=self.delta, reduction='none')

        # 2. Apply Mask
        if mask is not None:
            # Ensure mask is float for multiplication
            mask = mask.to(loss.dtype)
            loss = loss * mask

            if self.reduction == 'mean':
                # Divide ONLY by the number of valid elements
                # Clamp denominator to avoid division by zero
                return loss.sum() / (mask.sum() + 1e-8)

            elif self.reduction == 'sum':
                return loss.sum()

            else:  # reduction == 'none'
                return loss

        # 3. No Mask Case (Standard Behavior)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()

        return loss


class LogCoshLoss(nn.Module):
    def __init__(self, reduction = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true, mask=None):
        loss = torch.log(torch.cosh(y_pred - y_true))

        if mask is not None:
            loss = loss * mask

        if self.reduction == 'none':
            return loss  # Return (Batch, Len)

        # Standard reductions...
        if self.reduction == 'mean':
            return loss.sum() / (mask.sum() + 1e-8)


import math


class WingLoss(nn.Module):
    def __init__(self, w=10.0, epsilon=2.0):
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = w - w * math.log(1 + w / epsilon)

    def forward(self, y_pred, y_true, mask=None):
        diff = torch.abs(y_pred - y_true)

        # Two behaviors:
        # 1. If error is small (< w): Log-based (Sensitive to small details)
        # 2. If error is large (>= w): Linear (Stable like MAE)
        loss = torch.where(
            diff < self.w,
            self.w * torch.log(1 + diff / self.epsilon),
            diff - self.C
        )

        if mask is not None:
            mask = mask.to(loss.dtype)
            return (loss * mask).sum() / (mask.sum() + 1e-8)
        return loss.mean()


class TargetWeightedMSELoss(nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target, mask=None):
        # Standard MSE
        mse = (pred - target) ** 2

        # Weighting:
        # +1.0 ensures background (0 reads) still has some weight.
        # +target means a peak of 10 reads has 11x the weight of background.
        weights = 1.0 + target

        loss = mse * weights

        if mask is not None:
            loss = loss * mask

        if self.reduction == 'mean':
            return loss.mean()
        return loss


class CoralOrdinalLoss(nn.Module):
    def __init__(self, num_classes=3, ignore_index=-100):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.num_thresholds = num_classes - 1  # K-1

    @torch.no_grad()
    def predict_levels(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: (B,L,C) or (N,C)
        ordinal_logits = logits[..., :self.num_thresholds]  # (..., K-1)
        prob = torch.sigmoid(ordinal_logits)
        # number of thresholds passed -> class in {0..K-1}
        return (prob > 0.5).sum(dim=-1).long()
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (B, L, C) or (N, C)
        targets: (B, L) or (N,)
        returns: (B, L) or (N,) loss per item (no reduction)
        """
        # Ensure targets are integer class labels
        targets = targets.long()

        # Use only K-1 ordinal thresholds from logits
        # (If your head outputs exactly K-1, this is a no-op slice.)
        ordinal_logits = logits[..., :self.num_thresholds]

        # Build binary targets with broadcasting
        # targets[..., None] shape -> (..., 1)
        # levels shape -> (1, 1, ..., K-1) broadcastable to targets
        levels = torch.arange(self.num_thresholds, device=logits.device)
        levels = levels.view(*([1] * targets.ndim), self.num_thresholds)  # e.g. (1,1,K-1) for (B,L)

        if self.ignore_index is not None:
            valid = (targets != self.ignore_index)
            safe_targets = targets.clone()
            safe_targets[~valid] = 0
        else:
            valid = None
            safe_targets = targets

        binary_targets = (safe_targets.unsqueeze(-1) > levels).float()  # (..., K-1)

        bce = F.binary_cross_entropy_with_logits(
            ordinal_logits, binary_targets, reduction="none"
        )  # (..., K-1)

        loss = bce.sum(dim=-1)  # (...,)

        # Optional: zero-out ignore_index positions (mask will also kill them later)
        if valid is not None:
            loss = loss * valid.to(loss.dtype)

        return loss