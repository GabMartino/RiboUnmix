from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import lightning as pl
import torch
import torch.nn as nn

from Models.utils.masked_pearson import MaskedPearsonCorrelation
from Models.utils.tweedie_deviance_loss import TweedieDevianceLoss


class RiboQueuingModelLightningModule(pl.LightningModule):
    def __init__(
        self,
        torch_model: nn.Module,
        *,
        config: Any,
        dataset_encoding: dict[str, int] | None = None,
    ):
        super().__init__()

        self.model = torch_model
        self.config = config
        self._val_plot_logged_this_epoch = False

        self.loss_fn = TweedieDevianceLoss(
            eps=float(self._cfg_get("loss.eps", 1e-8)),
            phi_min=float(self._cfg_get("loss.phi_min", 1e-4)),
            phi_max=float(self._cfg_get("loss.phi_max", 10.0)),
            censor_threshold=float(self._cfg_get("loss.censor_threshold", 0.0)),
            zero_censor_to_zero=bool(self._cfg_get("loss.zero_censor_to_zero", True)),
            include_log_phi=bool(self._cfg_get("loss.include_log_phi", True)),
        )

        self.masked_pcc = MaskedPearsonCorrelation(
            eps=float(self._cfg_get("loss.eps", 1e-8)),
        )

        if dataset_encoding is None:
            dataset_encoding = {}

        self.dataset_encoding = {str(k): int(v) for k, v in dataset_encoding.items()}
        self.dataset_id_to_name = {
            int(v): str(k)
            for k, v in self.dataset_encoding.items()
        }

        self.use_pcgrad = bool(self._cfg_get("optim.use_pcgrad", True))

        # PCGrad needs manual optimization because we manually rewrite gradients.
        self.automatic_optimization = not self.use_pcgrad

    # ============================================================
    # Config helper
    # ============================================================

    def _cfg_get(self, path: str, default: Any = None) -> Any:
        obj = self.config

        for part in path.split("."):
            try:
                obj = getattr(obj, part)
            except Exception:
                return default

        return obj

    # ============================================================
    # Epoch hooks
    # ============================================================

    def on_validation_epoch_start(self) -> None:
        self._val_plot_logged_this_epoch = False

    def on_validation_epoch_end(self) -> None:
        """
        Required only for manual optimization / PCGrad.

        In automatic optimization, Lightning steps the scheduler.
        In manual optimization, we step ReduceLROnPlateau ourselves.
        """
        if not self.use_pcgrad:
            return

        scheduler = self.lr_schedulers()

        if scheduler is None:
            return

        monitor = str(self._cfg_get("optim.scheduler.monitor", "val_loss_epoch"))
        metric = self.trainer.callback_metrics.get(monitor)

        if metric is None:
            return

        if isinstance(scheduler, list):
            for sched in scheduler:
                sched.step(metric)
        else:
            scheduler.step(metric)

    # ============================================================
    # CSS helpers
    # ============================================================

    @staticmethod
    def _get_css_item(css: Any, sample_idx: int) -> Any:
        if css is None:
            return None

        if isinstance(css, (list, tuple)):
            if sample_idx >= len(css):
                return None
            return css[sample_idx]

        if torch.is_tensor(css):
            if css.ndim == 0:
                return css
            if sample_idx >= css.shape[0]:
                return None
            return css[sample_idx]

        try:
            return css[sample_idx]
        except Exception:
            return None

    @staticmethod
    def _normalize_css_positions(
        css_i: Any,
        L: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Converts one sample's CSS annotation into valid integer positions.

        Supports:
          - list/array/tensor of positions
          - dense boolean mask of length L
          - dense 0/1 mask of length L
        """
        if css_i is None:
            return torch.empty(0, dtype=torch.long, device=device)

        try:
            if torch.is_tensor(css_i):
                arr = css_i.detach().cpu()
            else:
                arr = torch.as_tensor(css_i)
        except Exception:
            return torch.empty(0, dtype=torch.long, device=device)

        arr = arr.reshape(-1)

        if arr.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        if arr.dtype == torch.bool:
            if arr.numel() >= L:
                pos = torch.nonzero(arr[:L], as_tuple=False).reshape(-1)
            else:
                pos = torch.nonzero(arr, as_tuple=False).reshape(-1)
        else:
            if torch.is_floating_point(arr):
                arr = arr[torch.isfinite(arr)]

            if arr.numel() == 0:
                return torch.empty(0, dtype=torch.long, device=device)

            arr_long = arr.to(torch.long)

            if arr_long.numel() == L and torch.all((arr_long == 0) | (arr_long == 1)):
                pos = torch.nonzero(arr_long.bool(), as_tuple=False).reshape(-1)
            else:
                pos = arr_long.reshape(-1)

        pos = pos.to(dtype=torch.long)
        pos = pos[(pos >= 0) & (pos < int(L))]

        if pos.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        pos = torch.unique(pos, sorted=True)
        return pos.to(device=device)

    @staticmethod
    def _css_window_mask(
        css_pos: torch.Tensor,
        L: int,
        window: int,
        device: torch.device,
    ) -> torch.Tensor:
        css_mask = torch.zeros(int(L), dtype=torch.bool, device=device)

        if css_pos.numel() == 0:
            return css_mask

        window = max(0, int(window))

        for p in css_pos.detach().cpu().tolist():
            left = max(0, int(p) - window)
            right = min(int(L), int(p) + window + 1)
            css_mask[left:right] = True

        return css_mask

    def _css_rank_recall_enrichment_for_score(
        self,
        *,
        score: torch.Tensor,
        mask_b: torch.Tensor,
        css: Any,
        top_frac: float,
        min_k: int,
        window: int,
        eps: float = 1e-8,
    ) -> tuple[dict[str, torch.Tensor], int]:
        device = score.device

        rank_percentiles = []
        recalls = []
        enrichments = []

        B = int(score.shape[0])

        for i in range(B):
            L = int(mask_b[i].sum().detach().cpu().item())

            if L < 3:
                continue

            score_i = score[i, :L].detach().float()
            css_i = self._get_css_item(css, i)

            css_pos = self._normalize_css_positions(
                css_i=css_i,
                L=L,
                device=device,
            )

            if css_pos.numel() == 0:
                continue

            css_win = self._css_window_mask(
                css_pos=css_pos,
                L=L,
                window=window,
                device=device,
            )

            non_css_win = ~css_win

            if css_win.sum() == 0 or non_css_win.sum() == 0:
                continue

            css_scores = score_i[css_pos]

            percentiles = []
            for s in css_scores:
                percentiles.append((score_i <= s).float().mean())

            rank_percentiles.append(torch.stack(percentiles).mean())

            k = max(int(min_k), int(math.ceil(float(top_frac) * L)))
            k = min(k, L)

            if k > 0:
                top_idx = torch.topk(score_i, k=k, largest=True).indices

                distances = (
                    css_pos.reshape(-1, 1)
                    - top_idx.reshape(1, -1)
                ).abs()

                hit = distances.min(dim=1).values <= int(window)
                recalls.append(hit.float().mean())

            css_mean = score_i[css_win].mean()
            bg_mean = score_i[non_css_win].mean().clamp_min(eps)

            enrichments.append(css_mean / bg_mean)

        if not rank_percentiles:
            return {}, 0

        out = {
            "css_rank_percentile": torch.stack(rank_percentiles).mean(),
            "css_recall_topk_window": (
                torch.stack(recalls).mean()
                if recalls
                else torch.zeros((), device=device)
            ),
            "css_enrichment": torch.stack(enrichments).mean(),
        }

        return out, len(rank_percentiles)

    def _css_delta_for_values(
        self,
        *,
        values: torch.Tensor,
        mask_b: torch.Tensor,
        css: Any,
        window: int,
    ) -> tuple[torch.Tensor | None, int]:
        """
        Computes:

            mean(values at CSS ± window) - mean(values elsewhere)

        Useful for b and phi.
        """
        device = values.device
        deltas = []

        B = int(values.shape[0])

        for i in range(B):
            L = int(mask_b[i].sum().detach().cpu().item())

            if L < 3:
                continue

            values_i = values[i, :L].detach().float()
            css_i = self._get_css_item(css, i)

            css_pos = self._normalize_css_positions(
                css_i=css_i,
                L=L,
                device=device,
            )

            if css_pos.numel() == 0:
                continue

            css_win = self._css_window_mask(
                css_pos=css_pos,
                L=L,
                window=window,
                device=device,
            )

            non_css_win = ~css_win

            if css_win.sum() == 0 or non_css_win.sum() == 0:
                continue

            deltas.append(values_i[css_win].mean() - values_i[non_css_win].mean())

        if not deltas:
            return None, 0

        return torch.stack(deltas).mean(), len(deltas)

    def _compute_css_diagnostics(
        self,
        *,
        L_queue: torch.Tensor,
        mask_b: torch.Tensor,
        css: Any,
        L_effective: torch.Tensor | None = None,
        mu_base: torch.Tensor | None = None,
        additive_bg: torch.Tensor | None = None,
        b_offset: torch.Tensor | None = None,
        phi: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], int]:
        top_frac = float(self._cfg_get("metrics.css_top_frac", 0.01))
        min_k = int(self._cfg_get("metrics.css_min_k", 10))
        window = int(self._cfg_get("metrics.css_window", 3))
        eps = float(self._cfg_get("loss.eps", 1e-8))

        logs: dict[str, torch.Tensor] = {}

        L_metrics, css_count = self._css_rank_recall_enrichment_for_score(
            score=L_queue,
            mask_b=mask_b,
            css=css,
            top_frac=top_frac,
            min_k=min_k,
            window=window,
            eps=eps,
        )

        if css_count == 0:
            return logs, 0

        for name, value in L_metrics.items():
            logs[f"css_L_queue_{name}"] = value

        if L_effective is not None:
            L_eff_metrics, _ = self._css_rank_recall_enrichment_for_score(
                score=L_effective,
                mask_b=mask_b,
                css=css,
                top_frac=top_frac,
                min_k=min_k,
                window=window,
                eps=eps,
            )

            for name, value in L_eff_metrics.items():
                logs[f"css_L_effective_{name}"] = value

        if mu_base is not None:
            mu_base_metrics, _ = self._css_rank_recall_enrichment_for_score(
                score=mu_base,
                mask_b=mask_b,
                css=css,
                top_frac=top_frac,
                min_k=min_k,
                window=window,
                eps=eps,
            )

            for name, value in mu_base_metrics.items():
                logs[f"css_mu_base_{name}"] = value

        if additive_bg is not None:
            A_metrics, _ = self._css_rank_recall_enrichment_for_score(
                score=additive_bg,
                mask_b=mask_b,
                css=css,
                top_frac=top_frac,
                min_k=min_k,
                window=window,
                eps=eps,
            )

            for name, value in A_metrics.items():
                logs[f"css_additive_bg_{name}"] = value

        if b_offset is not None:
            b_delta, b_count = self._css_delta_for_values(
                values=b_offset,
                mask_b=mask_b,
                css=css,
                window=window,
            )

            if b_delta is not None and b_count > 0:
                logs["css_b_delta"] = b_delta

        if phi is not None:
            phi_delta, phi_count = self._css_delta_for_values(
                values=phi,
                mask_b=mask_b,
                css=css,
                window=window,
            )

            if phi_delta is not None and phi_count > 0:
                logs["css_phi_delta"] = phi_delta

        return logs, css_count

    # ============================================================
    # PCGrad helpers
    # ============================================================

    def _pcgrad_target_parameters(self) -> list[torch.nn.Parameter]:
        """
        By default, apply PCGrad only to the shared biological branch.

        Dataset-specific nuisance heads should remain dataset-specific.
        """
        biology_only = bool(self._cfg_get("optim.pcgrad_biology_only", True))

        if biology_only:
            biological_model = getattr(self.model, "biological_model", None)

            if biological_model is None:
                raise AttributeError(
                    "optim.pcgrad_biology_only=True, but self.model.biological_model "
                    "does not exist."
                )

            params = [
                p for p in biological_model.parameters()
                if p.requires_grad
            ]
        else:
            params = [
                p for p in self.model.parameters()
                if p.requires_grad
            ]

        return params

    @staticmethod
    def _flatten_current_grads(
        params: list[torch.nn.Parameter],
    ) -> torch.Tensor:
        flats = []

        for p in params:
            if p.grad is None:
                flats.append(torch.zeros_like(p).reshape(-1))
            else:
                flats.append(p.grad.detach().clone().reshape(-1))

        if not flats:
            return torch.empty(0)

        return torch.cat(flats, dim=0)

    @staticmethod
    def _assign_flat_grads(
        params: list[torch.nn.Parameter],
        flat_grad: torch.Tensor,
    ) -> None:
        offset = 0

        for p in params:
            n = p.numel()
            g = flat_grad[offset:offset + n].view_as(p)
            offset += n

            if p.grad is None:
                p.grad = g.detach().clone()
            else:
                p.grad.detach().copy_(g)

    @staticmethod
    def _pcgrad_combine(
        flat_grads: list[torch.Tensor],
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        PCGrad projection.

        If two dataset gradients conflict:

            dot(g_i, g_j) < 0

        remove the conflicting component.
        """
        if len(flat_grads) == 0:
            raise ValueError("No gradients passed to PCGrad.")

        if len(flat_grads) == 1:
            return flat_grads[0]

        projected = []

        for i, g_i_original in enumerate(flat_grads):
            g_i = g_i_original.clone()

            order = torch.randperm(len(flat_grads), device=g_i.device)

            for j_tensor in order:
                j = int(j_tensor.item())

                if j == i:
                    continue

                g_j = flat_grads[j]

                dot = torch.dot(g_i, g_j)
                denom = torch.dot(g_j, g_j).clamp_min(eps)

                if dot < 0:
                    g_i = g_i - (dot / denom) * g_j

            projected.append(g_i)

        return torch.stack(projected, dim=0).mean(dim=0)

    @staticmethod
    def _pcgrad_pairwise_stats(
        flat_grads: list[torch.Tensor],
        eps: float = 1e-12,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if len(flat_grads) < 2:
            return None, None

        cosines = []
        conflicts = []

        for i in range(len(flat_grads)):
            for j in range(i + 1, len(flat_grads)):
                g_i = flat_grads[i]
                g_j = flat_grads[j]

                denom = (g_i.norm() * g_j.norm()).clamp_min(eps)
                cosine = torch.dot(g_i, g_j) / denom

                cosines.append(cosine)
                conflicts.append((cosine < 0).float())

        return torch.stack(cosines).mean(), torch.stack(conflicts).mean()

    @staticmethod
    def _per_dataset_losses(
        loss_per_sample: torch.Tensor,
        dataset_ids: torch.Tensor,
    ) -> list[torch.Tensor]:
        dataset_ids = dataset_ids.reshape(-1)
        loss_per_sample = loss_per_sample.reshape(-1)

        losses = []

        for dataset_id in torch.unique(dataset_ids.detach()):
            ds_mask = dataset_ids == dataset_id

            if torch.any(ds_mask):
                losses.append(loss_per_sample[ds_mask].mean())

        return losses

    # ============================================================
    # Plotting
    # ============================================================

    def _log_validation_profile_plot(
        self,
        *,
        y: torch.Tensor,
        mu: torch.Tensor,
        phi: torch.Tensor,
        tweedie_p: torch.Tensor,
        L_queue: torch.Tensor,
        mask_b: torch.Tensor,
        mu_pcc_per_sample: torch.Tensor,
        L_queue_pcc_per_sample: torch.Tensor,
        batch_idx: int,
        sample_idx: int = 0,
        tag: str = "val/profile_diagnostic",
        mu_base: torch.Tensor | None = None,
        additive_bg: torch.Tensor | None = None,
        additive_rel: torch.Tensor | None = None,
        css: Any | None = None,
    ) -> None:
        if self._val_plot_logged_this_epoch:
            return

        if batch_idx != 0:
            return

        if not getattr(self.trainer, "is_global_zero", True):
            return

        experiment = (
            getattr(self.logger, "experiment", None)
            if self.logger is not None
            else None
        )

        if experiment is None:
            return

        B = y.shape[0]

        if B == 0:
            return

        sample_idx = max(0, min(int(sample_idx), B - 1))

        with torch.no_grad():
            valid = mask_b[sample_idx].detach().bool().cpu()
            L = int(valid.sum().item())

            if L < 2:
                return

            y_i = y[sample_idx].detach().float().cpu()[valid]
            mu_i = mu[sample_idx].detach().float().cpu()[valid]
            phi_i = phi[sample_idx].detach().float().cpu()[valid]
            L_queue_i = L_queue[sample_idx].detach().float().cpu()[valid]

            p_scalar = (
                tweedie_p.detach()
                .float()
                .reshape(-1)
                .mean()
                .cpu()
                .clamp(1.0001, 1.9999)
            )

            tweedie_var_i = phi_i.clamp_min(1e-8) * torch.pow(
                mu_i.clamp_min(1e-8),
                p_scalar,
            )

            mu_pcc_i = float(mu_pcc_per_sample[sample_idx].detach().float().cpu())
            L_queue_pcc_i = float(
                L_queue_pcc_per_sample[sample_idx].detach().float().cpu()
            )
            p_value = float(p_scalar)

            x = torch.arange(y_i.numel()).numpy()

            y_np = y_i.numpy()
            mu_np = mu_i.numpy()
            L_queue_np = L_queue_i.numpy()
            var_np = tweedie_var_i.numpy()
            phi_np = phi_i.numpy()

            if mu_base is not None:
                mu_base_np = mu_base[sample_idx].detach().float().cpu()[valid].numpy()
            else:
                mu_base_np = None

            if additive_bg is not None:
                additive_bg_np = additive_bg[sample_idx].detach().float().cpu()[valid].numpy()
            else:
                additive_bg_np = None

            if additive_rel is not None:
                additive_rel_np = additive_rel[sample_idx].detach().float().cpu()[valid].numpy()
            else:
                additive_rel_np = None

            css_i = self._get_css_item(css, sample_idx)
            css_pos = self._normalize_css_positions(
                css_i=css_i,
                L=L,
                device=torch.device("cpu"),
            )

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(16, 9),
            sharex=True,
            gridspec_kw={"height_ratios": [1.4, 1.0, 1.0]},
        )

        fig.suptitle(
            f"Validation profile diagnostic | sample={sample_idx} | "
            f"PCC(mu, y)={mu_pcc_i:.4f} | "
            f"PCC(L_queue, y)={L_queue_pcc_i:.4f} | "
            f"Tweedie p={p_value:.4f}",
            fontsize=12,
        )

        axes[0].plot(x, y_np, label="target y", linewidth=1.2)
        axes[0].plot(x, mu_np, label="mu = mu_base + additive", linewidth=1.2)

        if mu_base_np is not None:
            axes[0].plot(x, mu_base_np, label="mu_base", linewidth=1.0)

        if additive_bg_np is not None:
            axes[0].plot(x, additive_bg_np, label="additive_bg", linewidth=1.0)

        axes[0].set_ylabel("profile")
        axes[0].set_title("Target profile vs predicted mean")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(x, L_queue_np, label="L_queue", linewidth=1.2)
        axes[1].set_ylabel("L_queue")
        axes[1].set_title("Biological queueing prediction")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(
            x,
            var_np,
            label=r"Tweedie variance $\phi\mu^p$",
            linewidth=1.2,
        )
        axes[2].plot(
            x,
            phi_np,
            label=r"$\phi$",
            linewidth=1.0,
            linestyle=":",
        )

        if additive_rel_np is not None:
            ax2 = axes[2].twinx()
            ax2.plot(
                x,
                additive_rel_np,
                label="additive_rel = A/S",
                linewidth=1.0,
                linestyle="--",
            )
            ax2.set_ylabel("additive_rel")
            ax2.legend(loc="upper left")

        axes[2].set_ylabel("variance / phi")
        axes[2].set_xlabel("codon position")
        axes[2].set_title("Tweedie variance, phi, and additive_rel diagnostic")
        axes[2].grid(True, alpha=0.3)

        for ax in axes:
            for j, css_position in enumerate(css_pos.detach().cpu().tolist()):
                ax.axvline(
                    int(css_position),
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.35,
                    label="CSS" if j == 0 else None,
                )
            ax.legend(loc="upper right")

        fig.tight_layout(rect=(0, 0, 1, 0.93))

        if hasattr(experiment, "add_figure"):
            experiment.add_figure(tag, fig, global_step=self.global_step)
        elif hasattr(experiment, "log_figure"):
            experiment.log_figure(figure_name=tag, figure=fig, step=self.global_step)

        plt.close(fig)
        self._val_plot_logged_this_epoch = True

    # ============================================================
    # Dataset logging helpers
    # ============================================================

    def _log_dataset_loss_metrics(
        self,
        *,
        stage: str,
        dataset_ids: torch.Tensor,
        nll_per_sample: torch.Tensor,
        loss_per_sample: torch.Tensor,
    ) -> None:
        dataset_ids = dataset_ids.detach()

        for dataset_id in torch.unique(dataset_ids).detach().cpu().tolist():
            dataset_id = int(dataset_id)
            dataset_name = self.dataset_id_to_name.get(dataset_id, f"dataset_{dataset_id}")

            ds_mask = dataset_ids == dataset_id
            ds_count = int(ds_mask.sum().detach().cpu().item())

            if ds_count == 0:
                continue

            self.log(
                f"{stage}_nll_by_dataset/{dataset_name}",
                nll_per_sample[ds_mask].mean().detach(),
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=ds_count,
            )

            self.log(
                f"{stage}_loss_by_dataset/{dataset_name}",
                loss_per_sample[ds_mask].mean().detach(),
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=ds_count,
            )

    def _log_dataset_component_pcc_metrics(
        self,
        *,
        stage: str,
        dataset_ids: torch.Tensor,
        component_pccs: dict[str, torch.Tensor],
    ) -> None:
        dataset_ids = dataset_ids.detach()

        for dataset_id in torch.unique(dataset_ids).detach().cpu().tolist():
            dataset_id = int(dataset_id)
            dataset_name = self.dataset_id_to_name.get(dataset_id, f"dataset_{dataset_id}")

            ds_mask = dataset_ids == dataset_id
            ds_count = int(ds_mask.sum().detach().cpu().item())

            if ds_count == 0:
                continue

            for component_name, pcc_per_sample in component_pccs.items():
                self.log(
                    f"{stage}_{component_name}_pcc_by_dataset/{dataset_name}",
                    pcc_per_sample[ds_mask].mean().detach(),
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    batch_size=ds_count,
                )

    def _log_dataset_phi_metrics(
        self,
        *,
        stage: str,
        dataset_ids: torch.Tensor,
        phi: torch.Tensor,
        mask_b: torch.Tensor,
    ) -> None:
        dataset_ids = dataset_ids.detach()
        mask_f = mask_b.float()
        phi_detached = phi.detach().float()

        for dataset_id in torch.unique(dataset_ids).detach().cpu().tolist():
            dataset_id = int(dataset_id)
            dataset_name = self.dataset_id_to_name.get(dataset_id, f"dataset_{dataset_id}")

            ds_mask = dataset_ids == dataset_id
            ds_count = int(ds_mask.sum().detach().cpu().item())

            if ds_count == 0:
                continue

            ds_phi = phi_detached[ds_mask]
            ds_mask_f = mask_f[ds_mask]

            ds_phi_mean = (
                (ds_phi * ds_mask_f).sum()
                / ds_mask_f.sum().clamp_min(1.0)
            )

            self.log(
                f"{stage}_phi_mean_by_dataset/{dataset_name}",
                ds_phi_mean.detach(),
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=ds_count,
            )

    # ============================================================
    # Shift logging
    # ============================================================

    def _get_shift_values(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        shift_head = getattr(
            getattr(self.model, "dataset_bias_model", None),
            "dataset_shift_head",
            None,
        )

        if shift_head is None:
            return None

        shifts = getattr(shift_head, "shifts", None)

        if shifts is None:
            return None

        return torch.tensor(
            list(shifts),
            device=device,
            dtype=dtype,
        )

    def _log_dataset_shift_metrics(
        self,
        *,
        stage: str,
        dataset_ids: torch.Tensor,
        shift_weights_used: torch.Tensor | None,
        shift_weights_soft: torch.Tensor | None,
    ) -> None:
        weights_ref = shift_weights_soft if shift_weights_soft is not None else shift_weights_used

        if weights_ref is None:
            return

        dataset_ids = dataset_ids.detach()

        shifts = self._get_shift_values(
            device=weights_ref.device,
            dtype=weights_ref.dtype,
        )

        if shifts is None:
            return

        def _expected_shift(weights: torch.Tensor) -> torch.Tensor:
            return (weights * shifts.reshape(1, -1)).sum(dim=1)

        expected_used = (
            _expected_shift(shift_weights_used.detach())
            if shift_weights_used is not None
            else None
        )

        expected_soft = (
            _expected_shift(shift_weights_soft.detach())
            if shift_weights_soft is not None
            else None
        )

        max_prob_soft = (
            shift_weights_soft.detach().max(dim=1).values
            if shift_weights_soft is not None
            else None
        )

        argmax_shift_soft = None

        if shift_weights_soft is not None:
            argmax_idx = shift_weights_soft.detach().argmax(dim=1)
            argmax_shift_soft = shifts[argmax_idx]

        for dataset_id in torch.unique(dataset_ids).detach().cpu().tolist():
            dataset_id = int(dataset_id)
            dataset_name = self.dataset_id_to_name.get(dataset_id, f"dataset_{dataset_id}")

            ds_mask = dataset_ids == dataset_id
            ds_count = int(ds_mask.sum().detach().cpu().item())

            if ds_count == 0:
                continue

            if expected_used is not None:
                self.log(
                    f"{stage}_shift_expected_used_by_dataset/{dataset_name}",
                    expected_used[ds_mask].mean().detach(),
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    batch_size=ds_count,
                )

            if expected_soft is not None:
                self.log(
                    f"{stage}_shift_expected_soft_by_dataset/{dataset_name}",
                    expected_soft[ds_mask].mean().detach(),
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    batch_size=ds_count,
                )

            if argmax_shift_soft is not None:
                self.log(
                    f"{stage}_shift_argmax_soft_by_dataset/{dataset_name}",
                    argmax_shift_soft[ds_mask].float().mean().detach(),
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    batch_size=ds_count,
                )

            if max_prob_soft is not None:
                self.log(
                    f"{stage}_shift_max_prob_soft_by_dataset/{dataset_name}",
                    max_prob_soft[ds_mask].mean().detach(),
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    batch_size=ds_count,
                )

            if shift_weights_soft is not None:
                for j, shift_value in enumerate(shifts.detach().cpu().tolist()):
                    shift_int = int(shift_value)

                    self.log(
                        f"{stage}_shift_prob_soft_k={shift_int:+d}_by_dataset/{dataset_name}",
                        shift_weights_soft[ds_mask, j].mean().detach(),
                        on_step=False,
                        on_epoch=True,
                        logger=True,
                        batch_size=ds_count,
                    )

    # ============================================================
    # Forward/loss helper
    # ============================================================

    def _compute_loss_info(
        self,
        batch: Any,
    ) -> dict[str, Any]:
        ids_datasets_sorted, ids, packed_sequence, profiles_target, lengths, mask, css = batch

        y = profiles_target.to(torch.float32)
        mask_b = mask.bool()
        mask_f = mask_b.float()

        mu, tweedie_p, phi, extras = self.model(
            packed_sequence,
            ids_datasets_sorted,
            y,
        )

        nll_per_sample = self.loss_fn(
            mu_phys=mu,
            power=tweedie_p,
            phi=phi,
            y_true=y,
            mask=mask_b,
            return_per_sample=True,
        )

        loss_per_sample = nll_per_sample

        additive_rel = extras[12] if len(extras) > 12 else None

        lambda_additive_l1 = float(self._cfg_get("loss.lambda_additive_l1", 0.0))

        additive_penalty = torch.zeros(
            (),
            device=y.device,
            dtype=loss_per_sample.dtype,
        )

        additive_rel_mean = torch.zeros(
            (),
            device=y.device,
            dtype=loss_per_sample.dtype,
        )

        if additive_rel is not None:
            additive_rel_float = additive_rel.float()

            additive_penalty_per_sample = (
                additive_rel_float * mask_f
            ).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)

            additive_penalty = additive_penalty_per_sample.mean()

            additive_rel_mean = (
                additive_rel_float * mask_f
            ).sum() / mask_f.sum().clamp_min(1.0)

            if lambda_additive_l1 > 0.0:
                loss_per_sample = (
                    loss_per_sample
                    + lambda_additive_l1 * additive_penalty_per_sample
                )

        dataset_losses = self._per_dataset_losses(
            loss_per_sample=loss_per_sample,
            dataset_ids=ids_datasets_sorted,
        )

        loss_sample_mean = loss_per_sample.mean()

        if len(dataset_losses) > 0:
            loss_dataset_balanced = torch.stack(dataset_losses).mean()
        else:
            loss_dataset_balanced = loss_sample_mean

        use_dataset_balanced_loss = bool(
            self._cfg_get("optim.use_dataset_balanced_loss", False)
        )

        loss_train_objective = (
            loss_dataset_balanced
            if use_dataset_balanced_loss
            else loss_sample_mean
        )

        return {
            "ids_datasets_sorted": ids_datasets_sorted,
            "ids": ids,
            "packed_sequence": packed_sequence,
            "profiles_target": profiles_target,
            "lengths": lengths,
            "mask": mask,
            "css": css,
            "y": y,
            "mask_b": mask_b,
            "mask_f": mask_f,
            "mu": mu,
            "tweedie_p": tweedie_p,
            "phi": phi,
            "extras": extras,
            "nll_per_sample": nll_per_sample,
            "loss_per_sample": loss_per_sample,
            "dataset_losses": dataset_losses,
            "loss_sample_mean": loss_sample_mean,
            "loss_dataset_balanced": loss_dataset_balanced,
            "loss_train_objective": loss_train_objective,
            "additive_penalty": additive_penalty,
            "additive_rel_mean": additive_rel_mean,
        }

    # ============================================================
    # Logging from computed info
    # ============================================================

    def _log_step_info(
        self,
        *,
        info: dict[str, Any],
        stage: str,
        batch_idx: int,
    ) -> None:
        ids_datasets_sorted = info["ids_datasets_sorted"]
        y = info["y"]
        mask_b = info["mask_b"]
        mask_f = info["mask_f"]
        css = info["css"]

        mu = info["mu"]
        tweedie_p = info["tweedie_p"]
        phi = info["phi"]
        extras = info["extras"]

        batch_size = int(y.shape[0])

        L_queue = extras[2]
        L_effective = extras[5] if len(extras) > 5 else None
        b_offset = extras[8] if len(extras) > 8 else None
        mu_base = extras[10] if len(extras) > 10 else None
        additive_bg = extras[11] if len(extras) > 11 else None
        additive_rel = extras[12] if len(extras) > 12 else None
        shift_weights_used = extras[14] if len(extras) > 14 else None
        shift_weights_soft = extras[15] if len(extras) > 15 else None

        with torch.no_grad():
            mu_detached = mu.detach()
            L_queue_detached = L_queue.detach()
            phi_detached = phi.detach()

            mu_pcc_per_sample = self.masked_pcc(
                pred=mu_detached,
                target=y,
                mask=mask_b,
            )

            L_queue_pcc_per_sample = self.masked_pcc(
                pred=L_queue_detached,
                target=y,
                mask=mask_b,
            )

            component_pccs = {
                "mu": mu_pcc_per_sample,
                "L_queue": L_queue_pcc_per_sample,
            }

            if L_effective is not None:
                L_effective_pcc_per_sample = self.masked_pcc(
                    pred=L_effective.detach(),
                    target=y,
                    mask=mask_b,
                )
                component_pccs["L_effective"] = L_effective_pcc_per_sample

            if mu_base is not None:
                mu_base_pcc_per_sample = self.masked_pcc(
                    pred=mu_base.detach(),
                    target=y,
                    mask=mask_b,
                )
                component_pccs["mu_base"] = mu_base_pcc_per_sample

            mu_pcc = mu_pcc_per_sample.mean()
            L_queue_pcc = L_queue_pcc_per_sample.mean()

            phi_mean = (
                (phi_detached.float() * mask_f).sum()
                / mask_f.sum().clamp_min(1.0)
            )

            css_logs: dict[str, torch.Tensor] = {}
            css_count = 0

            if stage == "val":
                css_logs, css_count = self._compute_css_diagnostics(
                    L_queue=L_queue_detached,
                    L_effective=(
                        L_effective.detach()
                        if L_effective is not None
                        else None
                    ),
                    mu_base=(
                        mu_base.detach()
                        if mu_base is not None
                        else None
                    ),
                    mask_b=mask_b,
                    css=css,
                    additive_bg=(
                        additive_bg.detach()
                        if additive_bg is not None
                        else None
                    ),
                    b_offset=(
                        b_offset.detach()
                        if b_offset is not None
                        else None
                    ),
                    phi=phi_detached,
                )

        if stage == "val":
            sample_idx = int(
                getattr(
                    getattr(self.config, "predict", object()),
                    "example_idx",
                    0,
                )
            )

            self._log_validation_profile_plot(
                y=y,
                mu=mu,
                phi=phi,
                tweedie_p=tweedie_p,
                L_queue=L_queue.detach(),
                mask_b=mask_b,
                mu_pcc_per_sample=mu_pcc_per_sample,
                L_queue_pcc_per_sample=L_queue_pcc_per_sample,
                batch_idx=batch_idx,
                sample_idx=sample_idx,
                tag="val/profile_diagnostic",
                mu_base=mu_base,
                additive_bg=additive_bg,
                additive_rel=additive_rel,
                css=css,
            )

        loss_for_log = (
            info["loss_train_objective"]
            if stage == "train"
            else info["loss_sample_mean"]
        )

        # ------------------------------------------------------------
        # Global logs
        # ------------------------------------------------------------
        self.log(
            f"{stage}_loss",
            loss_for_log.detach(),
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

        self.log(
            f"{stage}_loss_sample_mean",
            info["loss_sample_mean"].detach(),
            on_step=(stage == "train"),
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
        )

        self.log(
            f"{stage}_loss_dataset_balanced",
            info["loss_dataset_balanced"].detach(),
            on_step=(stage == "train"),
            on_epoch=True,
            logger=True,
            prog_bar=(stage == "val"),
            batch_size=batch_size,
        )

        self.log(
            f"{stage}_nll",
            info["nll_per_sample"].mean().detach(),
            on_step=(stage == "train"),
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
        )

        self.log(
            f"{stage}_p",
            tweedie_p.detach().reshape(-1).mean(),
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
        )

        self.log(
            f"{stage}_phi_mean",
            phi_mean.detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
        )

        self.log(
            f"{stage}_mu_pcc",
            mu_pcc.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=(stage == "val"),
            logger=True,
            batch_size=batch_size,
        )

        self.log(
            f"{stage}_L_queue_pcc",
            L_queue_pcc.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=(stage == "val"),
            logger=True,
            batch_size=batch_size,
        )

        if additive_rel is not None:
            self.log(
                f"{stage}_additive_rel_mean",
                info["additive_rel_mean"].detach(),
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=batch_size,
            )

            lambda_additive_l1 = float(self._cfg_get("loss.lambda_additive_l1", 0.0))
            if lambda_additive_l1 > 0.0:
                self.log(
                    f"{stage}_additive_l1_penalty",
                    info["additive_penalty"].detach(),
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    batch_size=batch_size,
                )

        # ------------------------------------------------------------
        # Dataset-specific logs
        # ------------------------------------------------------------
        self._log_dataset_loss_metrics(
            stage=stage,
            dataset_ids=ids_datasets_sorted,
            nll_per_sample=info["nll_per_sample"],
            loss_per_sample=info["loss_per_sample"],
        )

        self._log_dataset_component_pcc_metrics(
            stage=stage,
            dataset_ids=ids_datasets_sorted,
            component_pccs=component_pccs,
        )

        self._log_dataset_phi_metrics(
            stage=stage,
            dataset_ids=ids_datasets_sorted,
            phi=phi,
            mask_b=mask_b,
        )

        self._log_dataset_shift_metrics(
            stage=stage,
            dataset_ids=ids_datasets_sorted,
            shift_weights_used=shift_weights_used,
            shift_weights_soft=shift_weights_soft,
        )

        # ------------------------------------------------------------
        # Validation summary logs
        # ------------------------------------------------------------
        if stage == "val":
            self.log(
                "val_loss_epoch",
                info["loss_sample_mean"].detach(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
            )

            self.log(
                "val_loss_dataset_balanced_epoch",
                info["loss_dataset_balanced"].detach(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
            )

            self.log(
                "val_mu_pcc_epoch",
                mu_pcc.detach(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
            )

            self.log(
                "val_L_queue_pcc_epoch",
                L_queue_pcc.detach(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
            )

            if css_count > 0:
                for name, value in css_logs.items():
                    self.log(
                        f"val_{name}",
                        value.detach(),
                        on_step=False,
                        on_epoch=True,
                        logger=True,
                        prog_bar=(
                            name in {
                                "css_L_queue_css_rank_percentile",
                                "css_L_queue_css_recall_topk_window",
                            }
                        ),
                        batch_size=css_count,
                    )

    # ============================================================
    # Shared automatic step
    # ============================================================

    def _shared_step(
        self,
        batch: Any,
        stage: str,
        batch_idx: int,
    ) -> torch.Tensor:
        info = self._compute_loss_info(batch)

        loss = (
            info["loss_train_objective"]
            if stage == "train"
            else info["loss_sample_mean"]
        )

        self._log_step_info(
            info=info,
            stage=stage,
            batch_idx=batch_idx,
        )

        return loss

    # ============================================================
    # Lightning API
    # ============================================================

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        if not self.use_pcgrad:
            return self._shared_step(batch, stage="train", batch_idx=batch_idx)

        opt = self.optimizers()
        opt.zero_grad(set_to_none=True)

        info = self._compute_loss_info(batch)

        loss = info["loss_train_objective"]
        dataset_losses = info["dataset_losses"]
        batch_size = int(info["y"].shape[0])

        pcgrad_params = self._pcgrad_target_parameters()

        if len(dataset_losses) <= 1 or len(pcgrad_params) == 0:
            self.manual_backward(loss)
        else:
            flat_dataset_grads = []

            # One biological gradient per dataset present in the batch.
            for ds_loss in dataset_losses:
                opt.zero_grad(set_to_none=True)

                self.manual_backward(
                    ds_loss,
                    retain_graph=True,
                )

                flat_g = self._flatten_current_grads(pcgrad_params)
                flat_dataset_grads.append(flat_g)

            pcgrad_flat = self._pcgrad_combine(flat_dataset_grads)

            cosine_mean, conflict_frac = self._pcgrad_pairwise_stats(
                flat_dataset_grads
            )

            # Normal backward for all parameters.
            # Then overwrite biological_model gradients with PCGrad gradients.
            opt.zero_grad(set_to_none=True)
            self.manual_backward(loss)

            self._assign_flat_grads(
                params=pcgrad_params,
                flat_grad=pcgrad_flat,
            )

            if cosine_mean is not None:
                self.log(
                    "train_pcgrad_bio_grad_cosine_mean",
                    cosine_mean.detach(),
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                    batch_size=batch_size,
                )

            if conflict_frac is not None:
                self.log(
                    "train_pcgrad_bio_grad_conflict_frac",
                    conflict_frac.detach(),
                    on_step=True,
                    on_epoch=True,
                    logger=True,
                    batch_size=batch_size,
                )

        gradient_clip_val = float(self._cfg_get("trainer.gradient_clip_val", 0.0))
        gradient_clip_algorithm = str(
            self._cfg_get("trainer.gradient_clip_algorithm", "norm")
        )

        if gradient_clip_val > 0.0:
            self.clip_gradients(
                opt,
                gradient_clip_val=gradient_clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm,
            )

        opt.step()
        opt.zero_grad(set_to_none=True)

        self._log_step_info(
            info=info,
            stage="train",
            batch_idx=batch_idx,
        )

        return loss.detach()

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val", batch_idx=batch_idx)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        ids_datasets_sorted, ids, packed_sequence, profiles_target, lengths, mask, css = batch

        def to_cpu(x):
            if torch.is_tensor(x):
                return x.detach().cpu()
            return x

        y = profiles_target.to(torch.float32)

        mu, tweedie_p, phi, extras = self.model(
            packed_sequence,
            ids_datasets_sorted,
            y,
        )

        rho_diag = extras[0] if len(extras) > 0 else None
        w_prob = extras[1] if len(extras) > 1 else None
        L_queue = extras[2] if len(extras) > 2 else None
        J = extras[3] if len(extras) > 3 else None
        S_mean = extras[4] if len(extras) > 4 else None
        L_effective = extras[5] if len(extras) > 5 else None
        b_offset = extras[8] if len(extras) > 8 else None
        multiplier = extras[9] if len(extras) > 9 else None
        mu_base = extras[10] if len(extras) > 10 else None
        additive_bg = extras[11] if len(extras) > 11 else None
        additive_rel = extras[12] if len(extras) > 12 else None
        codon_ids = extras[13] if len(extras) > 13 else None
        shift_weights_used = extras[14] if len(extras) > 14 else None
        shift_weights_soft = extras[15] if len(extras) > 15 else None

        return {
            "ids": ids if isinstance(ids, list) else to_cpu(ids),
            "dataset_id": to_cpu(ids_datasets_sorted),
            "lengths": to_cpu(lengths),
            "mask": to_cpu(mask),
            "rho": to_cpu(rho_diag),
            "w_prob": to_cpu(w_prob),
            "L_queue": to_cpu(L_queue),
            "L_effective": to_cpu(L_effective),
            "J": to_cpu(J),
            "S_mean": to_cpu(S_mean),
            "mu_obs": to_cpu(mu),
            "mu_total": to_cpu(mu),
            "mu_base": to_cpu(mu_base),
            "phi": to_cpu(phi),
            "tweedie_p": to_cpu(tweedie_p),
            "b_offset": to_cpu(b_offset),
            "multiplier": to_cpu(multiplier),
            "additive_bg": to_cpu(additive_bg),
            "additive_rel": to_cpu(additive_rel),
            "codon_ids": to_cpu(codon_ids),
            "shift_weights": to_cpu(shift_weights_used),
            "shift_weights_soft": to_cpu(shift_weights_soft),
            "css": to_cpu(css),
            "y": to_cpu(profiles_target),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.optim.lr),
            weight_decay=float(self.config.optim.weight_decay),
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.config.optim.scheduler.mode,
            factor=float(self.config.optim.scheduler.factor),
            patience=int(self.config.optim.scheduler.patience),
            min_lr=float(self.config.optim.scheduler.min_lr),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": self.config.optim.scheduler.monitor,
            },
        }