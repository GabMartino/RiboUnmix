from __future__ import annotations

from typing import Any

import math

import lightning as pl
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from Models.utils.PCGrad_utils import (
    per_dataset_losses,
    pcgrad_combine,
    assign_flat_grads,
)
from Models.utils.advanced_metrics import (
    masked_mae,
    masked_auprc_peak_caller,
    masked_1d_wasserstein,
    physics_asymmetry_ratio,
)
from Models.utils.masked_pearson import MaskedPearsonCorrelation
from Models.utils.ribo_lightning_helpers import flatten_current_grads
from Models.utils.tweedie_deviance_loss import TweedieDevianceLoss


class RiboQueuingModelLightningModule(pl.LightningModule):
    def __init__(
        self,
        torch_model: nn.Module,
        *,
        config: Any,
        dataset_encoding: dict | None = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["torch_model", "config", "dataset_encoding"])

        self.model = torch_model
        self.config = config

        self.loss_fn = TweedieDevianceLoss(include_log_phi=True)

        enc = dataset_encoding or {}
        self.dataset_id_to_name = {int(v): str(k) for k, v in enc.items()}

        self.use_pcgrad = bool(self._cfg("optim.use_pcgrad", False))
        self.automatic_optimization = not self.use_pcgrad

        if self.use_pcgrad:
            print("Training is using PCGrad.")

        self.masked_pcc = MaskedPearsonCorrelation(
            eps=float(self._cfg("loss.eps", 1e-8)),
        )

        self._val_plot_logged_this_epoch = False

    # ============================================================
    # Config helper
    # ============================================================

    def _cfg(self, path: str, default: Any = None) -> Any:
        cur = self.config

        for key in path.split("."):
            if cur is None:
                return default

            if isinstance(cur, dict):
                if key not in cur:
                    return default
                cur = cur[key]
            else:
                if not hasattr(cur, key):
                    return default
                cur = getattr(cur, key)

        return cur

    # ============================================================
    # Loss helpers
    # ============================================================

    def _compute_raw_nll_per_sample(
        self,
        *,
        mu: torch.Tensor,
        p: torch.Tensor,
        phi: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.autocast(device_type=mu.device.type, enabled=False):
            return self.loss_fn(
                mu_phys=mu.float(),
                power=p.float(),
                phi=phi.float(),
                y_true=target.float(),
                mask=mask.bool(),
                return_per_sample=True,
                phi_reg_alpha= self._cfg("loss.phi_reg_alpha", 1),
            )

    def _soft_cap_loss_per_sample(
        self,
        loss_per_sample: torch.Tensor,
    ) -> torch.Tensor:
        cap = self._cfg("loss.nll_soft_cap", None)

        if cap is None:
            cap = self._cfg("loss.nll_soft_cap_value", None)

        if cap is None:
            return loss_per_sample

        cap = float(cap)

        if cap <= 0.0:
            return loss_per_sample

        return cap * torch.log1p(loss_per_sample / cap)

    def _dataset_balanced_scalar_loss(
        self,
        *,
        loss_per_sample: torch.Tensor,
        dataset_ids: torch.Tensor,
    ) -> torch.Tensor:
        dataset_losses = per_dataset_losses(
            loss_per_sample=loss_per_sample,
            dataset_ids=dataset_ids,
        )

        if len(dataset_losses) > 0:
            return torch.stack(dataset_losses).mean()

        return loss_per_sample.mean()

    def mass_matching_loss(
        self,
        *,
        mu: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        mask_f = mask.bool().to(dtype=mu.dtype)

        pred_mass = (mu.float() * mask_f.float()).sum(dim=1)
        target_mass = (target.float() * mask_f.float()).sum(dim=1).detach()

        pred_mass = pred_mass.clamp_min(eps)
        target_mass = target_mass.clamp_min(eps)

        return (torch.log(pred_mass) - torch.log(target_mass)).pow(2)

    def bias_log_regularization(
        self,
        *,
        extras: dict,
        mask: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        mask_b = mask.bool()
        regs = []

        exp_b = extras.get("exp_b")
        beta = extras.get("beta_per_position")

        if exp_b is not None:
            log_exp_b = torch.log(exp_b.float().clamp_min(eps))
            regs.append((log_exp_b[mask_b] ** 2).mean())

        if beta is not None:
            log_beta = torch.log(beta.float().clamp_min(eps))
            regs.append((log_beta[mask_b] ** 2).mean())

        if len(regs) == 0:
            return torch.zeros((), device=mask.device)

        return torch.stack(regs).sum()

    def _build_loss_terms(
        self,
        *,
        mu: torch.Tensor,
        p: torch.Tensor,
        phi: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        extras: dict,
    ) -> dict[str, torch.Tensor]:
        eps = float(self._cfg("loss.eps", 1e-8))

        raw_nll_per_sample = self._compute_raw_nll_per_sample(
            mu=mu,
            p=p,
            phi=phi,
            target=target,
            mask=mask,
        )

        nll_per_sample = self._soft_cap_loss_per_sample(raw_nll_per_sample)

        lambda_mass = float(self._cfg("loss.lambda_mass_match", 0.0))
        lambda_bias = float(self._cfg("loss.lambda_bias_log_l2", 0.0))

        mass_loss_per_sample = self.mass_matching_loss(
            mu=mu,
            target=target,
            mask=mask,
            eps=eps,
        )

        loss_per_sample = nll_per_sample + lambda_mass * mass_loss_per_sample

        bias_reg = self.bias_log_regularization(
            extras=extras,
            mask=mask,
            eps=eps,
        )

        extra_loss = lambda_bias * bias_reg

        return {
            "raw_nll_per_sample": raw_nll_per_sample,
            "nll_per_sample": nll_per_sample,
            "mass_loss_per_sample": mass_loss_per_sample,
            "loss_per_sample": loss_per_sample,
            "bias_reg": bias_reg,
            "extra_loss": extra_loss,
        }

    # ============================================================
    # Generic metric helpers
    # ============================================================

    def _normalize_profile(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        mask_f = mask.bool().to(dtype=x.dtype)
        x = x.float().clamp_min(0.0) * mask_f
        mass = x.sum(dim=1, keepdim=True).clamp_min(eps)
        return x / mass

    def _masked_mae_scalar(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_f = mask.bool().to(dtype=pred.dtype)
        err = (pred.float() - target.float()).abs() * mask_f.float()
        return err.sum() / mask_f.sum().clamp_min(1.0)

    def _log_basic_state(
        self,
        *,
        stage: str,
        mu: torch.Tensor,
        p: torch.Tensor,
        phi: torch.Tensor,
        extras: dict,
        mask: torch.Tensor,
        batch_size: int,
    ) -> None:
        mask_b = mask.bool()

        with torch.no_grad():
            self.log(
                f"{stage}_tweedie_p",
                p.detach().float().mean(),
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=batch_size,
            )

            self.log(
                f"{stage}_phi_mean",
                phi.detach().float()[mask_b].mean(),
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=batch_size,
            )

            self.log(
                f"{stage}_mu_mean",
                mu.detach().float()[mask_b].mean(),
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=batch_size,
            )

            exp_b = extras.get("exp_b")
            beta = extras.get("beta_per_position")

            if exp_b is not None:
                log_exp_b = torch.log(exp_b.detach().float().clamp_min(1e-8))
                self.log(
                    f"{stage}_exp_b_abs_log_mean",
                    log_exp_b[mask_b].abs().mean(),
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    batch_size=batch_size,
                )

            if beta is not None:
                log_beta = torch.log(beta.detach().float().clamp_min(1e-8))
                self.log(
                    f"{stage}_beta_abs_log_mean",
                    log_beta[mask_b].abs().mean(),
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    batch_size=batch_size,
                )

    def _log_nll_diagnostics(
        self,
        *,
        stage: str,
        raw_nll_per_sample: torch.Tensor,
        nll_per_sample: torch.Tensor,
        batch_size: int,
    ) -> None:
        with torch.no_grad():
            raw_nll = raw_nll_per_sample.detach().float()
            nll = nll_per_sample.detach().float()

            self.log(
                f"{stage}_raw_nll_mean",
                raw_nll.mean(),
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=batch_size,
            )

            self.log(
                f"{stage}_raw_nll_p99",
                torch.quantile(raw_nll, 0.99),
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=batch_size,
            )

            self.log(
                f"{stage}_nll_mean",
                nll.mean(),
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=batch_size,
            )

    # ============================================================
    # CSS recall helpers
    # ============================================================

    def _normalize_css_positions(
        self,
        css_i: Any,
        L: int,
    ) -> list[int]:
        if css_i is None:
            return []

        try:
            if torch.is_tensor(css_i):
                arr = css_i.detach().cpu().reshape(-1)
            else:
                arr = torch.as_tensor(css_i).detach().cpu().reshape(-1)
        except Exception:
            return []

        if arr.numel() == 0:
            return []

        if arr.dtype == torch.bool:
            arr_b = arr[:L]
            return [int(i) for i in torch.nonzero(arr_b, as_tuple=False).reshape(-1).tolist()]

        arr_f = arr.float()
        arr_f = arr_f[torch.isfinite(arr_f)]

        if arr_f.numel() == 0:
            return []

        if arr_f.numel() >= L:
            first_L = arr_f[:L]
            is_binary_mask = torch.all((first_L == 0.0) | (first_L == 1.0)).item()

            if is_binary_mask:
                return [
                    int(i)
                    for i in torch.nonzero(first_L.bool(), as_tuple=False).reshape(-1).tolist()
                ]

        pos = arr_f.long()
        pos = pos[(pos >= 0) & (pos < L)]

        return sorted(set(int(x) for x in pos.tolist()))

    def _peak_budget(
        self,
        L: int,
    ) -> int:
        top_frac = float(self._cfg("metrics.css_peak_top_frac", 0.02))
        min_peaks = int(self._cfg("metrics.css_peak_min_peaks", 1))
        max_peaks = int(self._cfg("metrics.css_peak_max_peaks", 50))

        k = int(math.ceil(float(L) * top_frac))
        k = max(k, min_peaks)
        k = min(k, max_peaks)
        k = min(k, L)

        return max(k, 0)

    def _topk_peak_positions_1d(
        self,
        signal_1d: torch.Tensor,
        *,
        k: int,
        nms_radius: int,
    ) -> list[int]:
        if signal_1d.numel() == 0 or k <= 0:
            return []

        x = signal_1d.detach().float().cpu()
        finite = torch.isfinite(x)

        if not finite.any():
            return []

        x = torch.where(finite, x, torch.full_like(x, -float("inf")))

        ordered = torch.argsort(x, descending=True).tolist()

        selected: list[int] = []

        for idx in ordered:
            if not torch.isfinite(x[idx]):
                continue

            if all(abs(idx - prev) > nms_radius for prev in selected):
                selected.append(int(idx))

            if len(selected) >= k:
                break

        return selected

    def _positions_hit_count(
        self,
        *,
        reference_positions: list[int],
        predicted_positions: list[int],
        tolerance: int,
    ) -> int:
        if len(reference_positions) == 0 or len(predicted_positions) == 0:
            return 0

        hits = 0

        for ref in reference_positions:
            if any(abs(ref - pred) <= tolerance for pred in predicted_positions):
                hits += 1

        return hits


    def _log_profile_diagnostics(
            self,
            *,
            stage: str,
            target: torch.Tensor,
            mask: torch.Tensor,
            css: list,
            dataset_ids: torch.Tensor,
            mu: torch.Tensor,
            L_queue: torch.Tensor,
            batch_size: int,
    ) -> None:
        dataset_ids = dataset_ids.detach()
        mask_b = mask.bool()
        mask_f = mask.float()

        # We only care about tracking the final observation (mu) and pure biology (L_queue)
        components_to_track = {"mu": mu.detach(), "L_queue": L_queue.detach()}

        with torch.no_grad():
            # 1. Global PCC
            for name, comp in components_to_track.items():
                pcc_per_sample = self.masked_pcc(pred=comp, target=target, mask=mask_b)
                self.log(
                    f"{stage}_{name}_pcc",
                    pcc_per_sample.mean(),
                    on_step=False, on_epoch=True, logger=True, batch_size=batch_size,
                )

            # 2. Core Observation Metrics (Only on mu)
            self.log(f"{stage}_mu_eval_MAE", masked_mae(components_to_track["mu"], target, mask_f),
                     on_epoch=True, logger=True, batch_size=batch_size)

            # 3. Per-Dataset PCC (Only track PCC per dataset to avoid massive metric bloat)
            unique_ids = torch.unique(dataset_ids).cpu().tolist()
            for ds_id in unique_ids:
                ds_id = int(ds_id)
                dataset_name = self.dataset_id_to_name.get(ds_id, f"dataset_{ds_id}")
                ds_mask = dataset_ids == ds_id
                ds_count = int(ds_mask.sum().item())

                if ds_count == 0: continue

                for name, comp in components_to_track.items():
                    pcc_per_sample = self.masked_pcc(pred=comp, target=target, mask=mask_b)
                    self.log(
                        f"{stage}_{name}_pcc_by_dataset/{dataset_name}",
                        pcc_per_sample[ds_mask].mean(),
                        on_epoch=True, logger=True, batch_size=ds_count,
                    )

    # ============================================================
    # CSS recall helpers (CLEANED)
    # ============================================================

    def _log_css_recall_diagnostics(
            self,
            *,
            stage: str,
            target: torch.Tensor,
            mask: torch.Tensor,
            css: list,
            dataset_ids: torch.Tensor,
            mu: torch.Tensor,
            L_queue: torch.Tensor,
            batch_size: int,
    ) -> None:
        if css is None: return

        tolerance = int(self._cfg("metrics.css_recall_tolerance", 1))
        nms_radius = int(self._cfg("metrics.css_peak_nms_radius", tolerance))
        device = target.device
        mask_b = mask.bool()

        # Track only target, mu, and L_queue
        recalls = {"target": [], "mu": [], "L_queue": []}
        components = {"mu": mu, "L_queue": L_queue}

        with torch.no_grad():
            B = int(target.shape[0])
            for i in range(B):
                mask_i = mask_b[i].cpu()
                L = int(mask_i.sum().item())
                if L < 2: continue

                css_i = css[i] if i < len(css) else None
                css_positions = self._normalize_css_positions(css_i, L)
                if not css_positions: continue

                k = self._peak_budget(L)
                y_valid = target[i].float().cpu()[mask_i]

                # Target Baseline
                y_peaks = self._topk_peak_positions_1d(y_valid, k=k, nms_radius=nms_radius)
                y_supported_css = [c for c in css_positions if any(abs(c - yp) <= tolerance for yp in y_peaks)]
                recalls["target"].append(float(len(y_supported_css)) / max(len(css_positions), 1))

                if not y_supported_css: continue

                # Model Predictions
                for name, comp in components.items():
                    comp_valid = comp[i].float().cpu()[mask_i]
                    pred_peaks = self._topk_peak_positions_1d(comp_valid, k=k, nms_radius=nms_radius)
                    hits = self._positions_hit_count(reference_positions=y_supported_css,
                                                     predicted_positions=pred_peaks, tolerance=tolerance)
                    recalls[name].append(float(hits) / max(len(y_supported_css), 1))

            # Log global averages only
            for name, vals in recalls.items():
                if vals:
                    self.log(f"{stage}_{name}_css_recall", torch.tensor(vals, device=device).mean(),
                             on_epoch=True, logger=True, batch_size=batch_size)

    # ============================================================
    # Shift diagnostics (CLEANED)
    # ============================================================

    def _log_shift_diagnostics(
            self,
            *,
            stage: str,
            dataset_ids: torch.Tensor,
            extras: dict,
    ) -> None:
        shift_weights_soft = extras.get("shift_weights_soft")
        if shift_weights_soft is None or not hasattr(self.model, "dataset_bias_model"): return

        shift_head = self.model.dataset_bias_model.dataset_shift_head
        shifts = torch.tensor(shift_head.shifts, device=shift_weights_soft.device, dtype=shift_weights_soft.dtype)

        expected_shift = (shift_weights_soft * shifts.reshape(1, -1)).sum(dim=1)

        # Removed the loop that logs every single shift weight value (massive bloat)
        for ds_id in torch.unique(dataset_ids).cpu().tolist():
            ds_mask = dataset_ids == int(ds_id)
            ds_count = int(ds_mask.sum().item())
            if ds_count == 0: continue

            dataset_name = self.dataset_id_to_name.get(int(ds_id), f"dataset_{int(ds_id)}")
            self.log(
                f"{stage}_expected_shift_by_dataset/{dataset_name}",
                expected_shift[ds_mask].mean(),
                on_epoch=True, logger=True, batch_size=ds_count,
            )

    def on_validation_epoch_start(self):
        self._val_plot_logged_this_epoch = False

    def _log_example_profile_plot(
        self,
        *,
        ids: list,
        dataset_ids: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        phi: torch.Tensor,
        p: torch.Tensor,
        mask: torch.Tensor,
        extras: dict,
        batch_idx: int,
    ) -> None:
        if self._val_plot_logged_this_epoch:
            return

        if batch_idx != 0:
            return

        if not getattr(self.trainer, "is_global_zero", True):
            return

        if self.logger is None:
            return

        experiment = getattr(self.logger, "experiment", None)

        if experiment is None:
            return

        sample_idx = int(self._cfg("predict.example_idx", 0))
        B = int(target.shape[0])
        sample_idx = max(0, min(sample_idx, B - 1))

        mask_i = mask[sample_idx].detach().bool().cpu()
        L = int(mask_i.sum().item())

        if L < 2:
            return

        def seq_to_np(x):
            if x is None:
                return None
            return x[sample_idx].detach().float().cpu()[mask_i].numpy()

        y_np = seq_to_np(target)
        mu_np = seq_to_np(mu)
        phi_np = seq_to_np(phi)

        L_queue_np = seq_to_np(extras.get("L_queue"))
        L_effective_np = seq_to_np(extras.get("L_effective"))
        L_shape_np = seq_to_np(extras.get("L_shape"))

        mu_base_np = seq_to_np(extras.get("mu_base"))
        corrected_shape_np = seq_to_np(extras.get("corrected_shape"))
        exp_b_np = seq_to_np(extras.get("exp_b"))
        beta_np = seq_to_np(extras.get("beta_per_position"))
        b_np = seq_to_np(extras.get("b"))

        x = torch.arange(L).cpu().numpy()

        dataset_id = int(dataset_ids[sample_idx].detach().cpu().item())
        dataset_name = self.dataset_id_to_name.get(dataset_id, f"dataset_{dataset_id}")
        transcript_id = ids[sample_idx]

        fig, axes = plt.subplots(
            4,
            1,
            figsize=(16, 11),
            sharex=True,
            gridspec_kw={"height_ratios": [1.4, 1.0, 1.0, 1.0]},
        )

        p_value = float(p.detach().float().mean().cpu())

        fig.suptitle(
            f"Validation profile diagnostic | dataset={dataset_name} | "
            f"transcript={transcript_id} | L={L} | Tweedie p={p_value:.4f}",
            fontsize=12,
        )

        axes[0].plot(x, y_np, label="target y", linewidth=1.0)
        axes[0].plot(x, mu_np, label="mu", linewidth=1.0)

        if mu_base_np is not None:
            axes[0].plot(x, mu_base_np, label="mu_base", linewidth=1.0)

        axes[0].set_title("Observed target vs predicted mean")
        axes[0].set_ylabel("profile")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right")

        if L_queue_np is not None:
            axes[1].plot(x, L_queue_np, label="L_queue", linewidth=1.0)

        if L_effective_np is not None:
            axes[1].plot(x, L_effective_np, label="L_effective", linewidth=1.0)

        if L_shape_np is not None:
            axes[1].plot(x, L_shape_np, label="L_shape", linewidth=1.0)

        axes[1].set_title("Biological branch")
        axes[1].set_ylabel("biology")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right")

        if corrected_shape_np is not None:
            axes[2].plot(x, corrected_shape_np, label="corrected_shape", linewidth=1.0)

        if exp_b_np is not None:
            ax2 = axes[2].twinx()
            ax2.plot(x, exp_b_np, label="exp_b", linewidth=0.8, linestyle="--")
            ax2.set_ylabel("exp_b")
            ax2.legend(loc="upper left")

        axes[2].set_title("Shape correction")
        axes[2].set_ylabel("shape")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="upper right")

        if phi_np is not None:
            axes[3].plot(x, phi_np, label="phi", linewidth=0.8)

        if beta_np is not None:
            ax3 = axes[3].twinx()
            ax3.plot(x, beta_np, label="beta", linewidth=0.8, linestyle="--")
            ax3.set_ylabel("beta")
            ax3.legend(loc="upper left")

        if b_np is not None:
            axes[3].plot(x, b_np, label="b", linewidth=0.8)

        axes[3].set_title("Dispersion and residual multipliers")
        axes[3].set_ylabel("value")
        axes[3].set_xlabel("codon position")
        axes[3].grid(True, alpha=0.3)
        axes[3].legend(loc="upper right")

        fig.tight_layout(rect=(0, 0, 1, 0.94))

        tag = "val/example_profile"

        if hasattr(experiment, "add_figure"):
            experiment.add_figure(tag, fig, global_step=self.global_step)
        elif hasattr(experiment, "log_figure"):
            experiment.log_figure(figure_name=tag, figure=fig, step=self.global_step)

        plt.close(fig)

        self._val_plot_logged_this_epoch = True

    # ============================================================
    # PCGrad
    # ============================================================

    def pcgrad_optimize(
            self,
            *,
            loss_per_sample: torch.Tensor,
            dataset_ids: torch.Tensor,
            extra_loss: torch.Tensor | None = None,
    ) -> torch.Tensor:
        opt = self.optimizers()
        opt.zero_grad(set_to_none=True)

        biological_model = self.model.biological_model
        biological_model_params = [
            p
            for p in biological_model.parameters()
            if p.requires_grad
        ]

        dataset_losses = per_dataset_losses(
            loss_per_sample=loss_per_sample,
            dataset_ids=dataset_ids,
        )

        if len(dataset_losses) > 0:
            scalar_loss = torch.stack(dataset_losses).mean()
        else:
            scalar_loss = loss_per_sample.mean()

        if extra_loss is not None:
            full_loss = scalar_loss + extra_loss
        else:
            full_loss = scalar_loss

        if len(dataset_losses) <= 1 or len(biological_model_params) == 0:
            self.manual_backward(full_loss)
            return scalar_loss.detach()

        max_datasets = int(
            self._cfg(
                "optim.pcgrad_max_datasets_per_step",
                len(dataset_losses),
            )
        )

        if len(dataset_losses) > max_datasets:
            perm = torch.randperm(len(dataset_losses), device=scalar_loss.device)
            selected = perm[:max_datasets].detach().cpu().tolist()
            dataset_losses_for_pcgrad = [dataset_losses[j] for j in selected]
        else:
            dataset_losses_for_pcgrad = dataset_losses

        raw_flat_dataset_grads = []
        unit_flat_dataset_grads = []

        for ds_loss in dataset_losses_for_pcgrad:
            opt.zero_grad(set_to_none=True)

            self.manual_backward(ds_loss, retain_graph=True)

            g_raw = flatten_current_grads(biological_model_params).detach()
            g_norm = g_raw.norm().clamp_min(1e-12)

            raw_flat_dataset_grads.append(g_raw)
            unit_flat_dataset_grads.append((g_raw / g_norm).detach())

        raw_grads = torch.stack(raw_flat_dataset_grads, dim=0)
        unit_grads = torch.stack(unit_flat_dataset_grads, dim=0)

        # ------------------------------------------------------------
        # PCGrad direction is computed on unit-normalized dataset grads.
        # This removes dataset dominance due only to gradient magnitude.
        # ------------------------------------------------------------
        pcgrad_flat = pcgrad_combine(unit_flat_dataset_grads).detach()

        mean_unit_grad = unit_grads.mean(dim=0)

        alpha = float(self._cfg("optim.pcgrad_alpha", 1.0))

        pcgrad_flat = (
                alpha * pcgrad_flat
                + (1.0 - alpha) * mean_unit_grad
        )

        # ------------------------------------------------------------
        # Rescale final direction to the average original dataset-grad norm.
        # ------------------------------------------------------------
        if bool(self._cfg("optim.pcgrad_rescale_to_mean_norm", True)):
            target_norm = raw_grads.norm(dim=1).mean().clamp_min(1e-12)
            pcgrad_norm = pcgrad_flat.norm().clamp_min(1e-12)
            pcgrad_flat = pcgrad_flat * (target_norm / pcgrad_norm)

        opt.zero_grad(set_to_none=True)

        # Normal gradients for all parameters.
        self.manual_backward(full_loss)

        # Replace only biological-model gradients with normalized-PCGrad result.
        assign_flat_grads(
            params=biological_model_params,
            flat_grad=pcgrad_flat,
        )
        del raw_flat_dataset_grads
        del unit_flat_dataset_grads
        del raw_grads
        del unit_grads
        del pcgrad_flat
        del mean_unit_grad
        return scalar_loss.detach()

    # ============================================================
    # Training / validation
    # ============================================================

    def _forward_batch(self, batch):
        (
            ids_datasets_sorted,
            ids_sorted,
            seq_packed,
            prof_pad,
            lengths_sorted,
            mask_pad,
            codon_ids_pad,
            css_sorted,
        ) = batch

        mu, p, phi, extras = self.model(
            seq_packed,
            codon_ids_pad,
            ids_datasets_sorted,
            prof_pad,
        )

        return {
            "dataset_ids": ids_datasets_sorted,
            "ids": ids_sorted,
            "seq_packed": seq_packed,
            "target": prof_pad,
            "lengths": lengths_sorted,
            "mask": mask_pad,
            "codon_ids": codon_ids_pad,
            "css": css_sorted,
            "mu": mu,
            "p": p,
            "phi": phi,
            "extras": extras,
        }

    def training_step(self, batch, batch_idx):
        out = self._forward_batch(batch)

        batch_size = int(out["target"].shape[0])

        loss_terms = self._build_loss_terms(
            mu=out["mu"],
            p=out["p"],
            phi=out["phi"],
            target=out["target"],
            mask=out["mask"],
            extras=out["extras"],
        )

        scalar_loss = self._dataset_balanced_scalar_loss(
            loss_per_sample=loss_terms["loss_per_sample"],
            dataset_ids=out["dataset_ids"],
        )

        loss = scalar_loss + loss_terms["extra_loss"]

        self.log(
            "train_loss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

        self.log(
            "train_raw_nll",
            loss_terms["raw_nll_per_sample"].mean().detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
        )

        self.log(
            "train_nll",
            loss_terms["nll_per_sample"].mean().detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
        )

        self.log(
            "train_mass_match_loss",
            loss_terms["mass_loss_per_sample"].mean().detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
        )

        self.log(
            "train_bias_log_l2",
            loss_terms["bias_reg"].detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
        )

        if batch_idx == 0:
            self._log_basic_state(
                stage="train",
                mu=out["mu"],
                p=out["p"],
                phi=out["phi"],
                extras=out["extras"],
                mask=out["mask"],
                batch_size=batch_size,
            )

            self._log_nll_diagnostics(
                stage="train",
                raw_nll_per_sample=loss_terms["raw_nll_per_sample"],
                nll_per_sample=loss_terms["nll_per_sample"],
                batch_size=batch_size,
            )

        if not self.use_pcgrad:
            return loss

        opt = self.optimizers()
        opt.zero_grad(set_to_none=True)

        pcgrad_loss = self.pcgrad_optimize(
            loss_per_sample=loss_terms["loss_per_sample"],
            dataset_ids=out["dataset_ids"],
            extra_loss=loss_terms["extra_loss"],
        )

        grad_clip_val = float(self._cfg("trainer.gradient_clip_val", 0.0))

        if grad_clip_val > 0.0:
            self.clip_gradients(
                opt,
                gradient_clip_val=grad_clip_val,
                gradient_clip_algorithm=str(self._cfg("trainer.gradient_clip_algorithm", "norm")),
            )

        opt.step()
        opt.zero_grad(set_to_none=True)

        return pcgrad_loss.detach()

    def validation_step(self, batch, batch_idx):
        out = self._forward_batch(batch)

        batch_size = int(out["target"].shape[0])

        loss_terms = self._build_loss_terms(
            mu=out["mu"],
            p=out["p"],
            phi=out["phi"],
            target=out["target"],
            mask=out["mask"],
            extras=out["extras"],
        )

        loss_sample_mean = loss_terms["loss_per_sample"].mean()

        loss_dataset_balanced = self._dataset_balanced_scalar_loss(
            loss_per_sample=loss_terms["loss_per_sample"],
            dataset_ids=out["dataset_ids"],
        )

        loss_dataset_balanced = loss_dataset_balanced + loss_terms["extra_loss"]

        components = {
            "mu": out["mu"],
            "mu_base": out["extras"].get("mu_base"),
            "L_queue": out["extras"].get("L_queue"),
            "L_effective": out["extras"].get("L_effective"),
            "L_shape": out["extras"].get("L_shape"),
            "base_shape": out["extras"].get("base_shape"),
            "corrected_shape": out["extras"].get("corrected_shape"),
        }

        self._log_basic_state(
            stage="val",
            mu=out["mu"],
            p=out["p"],
            phi=out["phi"],
            extras=out["extras"],
            mask=out["mask"],
            batch_size=batch_size,
        )

        self._log_nll_diagnostics(
            stage="val",
            raw_nll_per_sample=loss_terms["raw_nll_per_sample"],
            nll_per_sample=loss_terms["nll_per_sample"],
            batch_size=batch_size,
        )

        self._log_shift_diagnostics(
            stage="val",
            dataset_ids=out["dataset_ids"],
            extras=out["extras"],
        )

        # Replace the old dictionary and method calls with this:

        self._log_profile_diagnostics(
            stage="val",
            target=out["target"],
            mask=out["mask"],
            css=out["css"],
            dataset_ids=out["dataset_ids"],
            mu=out["mu"],
            L_queue=out["extras"].get("L_queue"),
            batch_size=batch_size,
        )

        self._log_css_recall_diagnostics(
            stage="val",
            target=out["target"],
            mask=out["mask"],
            css=out["css"],
            dataset_ids=out["dataset_ids"],
            mu=out["mu"],
            L_queue=out["extras"].get("L_queue"),
            batch_size=batch_size,
        )
        self._log_example_profile_plot(
            ids=out["ids"],
            dataset_ids=out["dataset_ids"],
            target=out["target"],
            mu=out["mu"],
            phi=out["phi"],
            p=out["p"],
            mask=out["mask"],
            extras=out["extras"],
            batch_idx=batch_idx,
        )

        self.log(
            "val_loss",
            loss_dataset_balanced.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

        self.log(
            "val_loss_sample_mean",
            loss_sample_mean.detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
        )

        self.log(
            "val_raw_nll",
            loss_terms["raw_nll_per_sample"].mean().detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
        )

        self.log(
            "val_nll",
            loss_terms["nll_per_sample"].mean().detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
        )

        self.log(
            "val_mass_match_loss",
            loss_terms["mass_loss_per_sample"].mean().detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
        )

        self.log(
            "val_bias_log_l2",
            loss_terms["bias_reg"].detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
        )

        return loss_dataset_balanced

    def on_validation_epoch_end(self):
        if not self.use_pcgrad:
            return

        sched = self.lr_schedulers()
        monitor = str(self._cfg("optim.scheduler.monitor", "val_loss"))

        if monitor not in self.trainer.callback_metrics:
            return

        metric = self.trainer.callback_metrics[monitor]
        sched.step(metric)

    # ============================================================
    # Prediction
    # ============================================================

    def predict_step(
        self,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Any]:
        out = self._forward_batch(batch)

        def to_cpu(x):
            if torch.is_tensor(x):
                return x.detach().cpu()
            if isinstance(x, (list, tuple)):
                return [to_cpu(v) for v in x]
            return x

        output = {
            "ids": to_cpu(out["ids"]),
            "dataset_id": to_cpu(out["dataset_ids"]),
            "lengths": to_cpu(out["lengths"]),
            "mask": to_cpu(out["mask"]),
            "css": to_cpu(out["css"]),
            "y": to_cpu(out["target"]),
            "mu_obs": to_cpu(out["mu"]),
            "phi": to_cpu(out["phi"]),
            "tweedie_p": to_cpu(out["p"]),
        }

        extras = out["extras"]

        if isinstance(extras, dict):
            for key, val in extras.items():
                if val is not None:
                    output[key] = to_cpu(val)

        return output

    # ============================================================
    # Optimizer
    # ============================================================

    def configure_optimizers(self):
        base_lr = float(self._cfg("optim.lr", 1e-3))
        bio_lr = float(self._cfg("optim.lr_biological", base_lr))
        rest_lr = float(self._cfg("optim.lr_rest", base_lr))

        weight_decay = float(self._cfg("optim.weight_decay", 1e-8))

        biological_params = []
        rest_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if name.startswith("biological_model."):
                biological_params.append(param)
            else:
                rest_params.append(param)

        param_groups = [
            {
                "params": biological_params,
                "lr": bio_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": rest_params,
                "lr": rest_lr,
                "weight_decay": weight_decay,
            },
        ]

        opt = torch.optim.AdamW(param_groups)

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=str(self._cfg("optim.scheduler.mode", "min")),
            factor=float(self._cfg("optim.scheduler.factor", 0.9)),
            patience=int(self._cfg("optim.scheduler.patience", 10)),
            min_lr=float(self._cfg("optim.scheduler.min_lr", 1e-6)),
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": str(self._cfg("optim.scheduler.monitor", "val_loss")),
                "interval": "epoch",
                "frequency": 1,
            },
        }