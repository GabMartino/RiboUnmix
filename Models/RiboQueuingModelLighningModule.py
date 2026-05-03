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
            eps=float(getattr(self.config.loss, "eps", 1e-8)),
            phi_min=float(getattr(self.config.loss, "phi_min", 1e-4)),
            phi_max=float(getattr(self.config.loss, "phi_max", 10.0)),
            censor_threshold=float(getattr(self.config.loss, "censor_threshold", 0.0)),
            zero_censor_to_zero=bool(getattr(self.config.loss, "zero_censor_to_zero", True)),
            include_log_phi=bool(getattr(self.config.loss, "include_log_phi", True)),
        )

        self.masked_pcc = MaskedPearsonCorrelation(
            eps=float(getattr(self.config.loss, "eps", 1e-8)),
        )

        if dataset_encoding is None:
            dataset_encoding = {}

        self.dataset_encoding = {str(k): int(v) for k, v in dataset_encoding.items()}
        self.dataset_id_to_name = {
            int(v): str(k)
            for k, v in self.dataset_encoding.items()
        }

    def on_validation_epoch_start(self) -> None:
        self._val_plot_logged_this_epoch = False

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

            if valid.sum().item() < 2:
                return

            y_i = y[sample_idx].detach().float().cpu()[valid]
            mu_i = mu[sample_idx].detach().float().cpu()[valid]
            phi_i = phi[sample_idx].detach().float().cpu()[valid]
            L_queue_i = L_queue[sample_idx].detach().float().cpu()[valid]

            p_scalar = (
                tweedie_p.detach()
                .float()
                .reshape(())
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
            if mu_base is not None:
                mu_base_i = mu_base[sample_idx].detach().float().cpu()[valid]
                mu_base_np = mu_base_i.numpy()
            else:
                mu_base_np = None

            if additive_bg is not None:
                additive_bg_i = additive_bg[sample_idx].detach().float().cpu()[valid]
                additive_bg_np = additive_bg_i.numpy()
            else:
                additive_bg_np = None

            if additive_rel is not None:
                additive_rel_i = additive_rel[sample_idx].detach().float().cpu()[valid]
                additive_rel_np = additive_rel_i.numpy()
            else:
                additive_rel_np = None
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
        axes[0].legend(loc="upper right")

        axes[1].plot(x, L_queue_np, label="L_queue", linewidth=1.2)
        axes[1].set_ylabel("L_queue")
        axes[1].set_title("Biological queueing prediction")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right")

        axes[2].plot(
            x,
            var_np,
            label=r"Tweedie variance $\phi\mu^p$",
            linewidth=1.2,
        )
        if additive_rel_np is not None:
            ax2 = axes[2].twinx()
            ax2.plot(x, additive_rel_np, label="additive_rel = A/S", linewidth=1.0, linestyle="--")
            ax2.set_ylabel("additive_rel")
            ax2.legend(loc="upper left")
        axes[2].set_ylabel("variance")
        axes[2].set_xlabel("codon position")
        axes[2].set_title("Tweedie variance and additive_rel diagnostic")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="upper right")

        fig.tight_layout(rect=(0, 0, 1, 0.93))

        if hasattr(experiment, "add_figure"):
            experiment.add_figure(tag, fig, global_step=self.global_step)
        elif hasattr(experiment, "log_figure"):
            experiment.log_figure(figure_name=tag, figure=fig, step=self.global_step)

        plt.close(fig)
        self._val_plot_logged_this_epoch = True

    def _log_dataset_pcc_metrics(
        self,
        *,
        stage: str,
        dataset_ids: torch.Tensor,
        mu_pcc_per_sample: torch.Tensor,
        L_queue_pcc_per_sample: torch.Tensor,
    ) -> None:
        dataset_ids = dataset_ids.detach()

        for dataset_id in torch.unique(dataset_ids).detach().cpu().tolist():
            dataset_id = int(dataset_id)
            dataset_name = self.dataset_id_to_name.get(
                dataset_id,
                f"dataset_{dataset_id}",
            )

            ds_mask = dataset_ids == dataset_id
            ds_count = int(ds_mask.sum().detach().cpu().item())

            if ds_count == 0:
                continue

            ds_mu_pcc = mu_pcc_per_sample[ds_mask].mean()
            ds_L_queue_pcc = L_queue_pcc_per_sample[ds_mask].mean()

            self.log(
                f"{stage}_mu_pcc_by_dataset/{dataset_name}",
                ds_mu_pcc.detach(),
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=ds_count,
            )

            self.log(
                f"{stage}_L_queue_pcc_by_dataset/{dataset_name}",
                ds_L_queue_pcc.detach(),
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=ds_count,
            )
    @staticmethod
    def _normalize_css_positions(
        css_i: Any,
        L: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Converts one sample's CSS annotation into a 1D LongTensor of valid positions.

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

            # Dense 0/1 CSS mask.
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
        """
        Builds a boolean mask covering CSS positions ± window codons.
        """
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
        """
        Computes CSS diagnostics for a positive score such as L_queue or additive_bg.

        Returns:
          css_rank_percentile:
              Mean percentile rank of exact CSS positions.
              Higher is better. 0.90 means CSS are around top 10%.

          css_recall_topk_window:
              Fraction of CSS positions hit by top-k score positions,
              allowing ±window codons.

          css_enrichment:
              mean(score at CSS ± window) / mean(score elsewhere)
        """
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
            css_i = css[i] if isinstance(css, (list, tuple)) else None

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

            # --------------------------------------------------------
            # Rank percentile at exact CSS positions.
            # --------------------------------------------------------
            css_scores = score_i[css_pos]

            # Percentile = fraction of positions with score <= CSS score.
            # Higher is better.
            percentiles = []
            for s in css_scores:
                percentiles.append((score_i <= s).float().mean())

            rank_percentiles.append(torch.stack(percentiles).mean())

            # --------------------------------------------------------
            # Recall@top-k, allowing positional window.
            # --------------------------------------------------------
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

            # --------------------------------------------------------
            # CSS enrichment.
            # --------------------------------------------------------
            css_mean = score_i[css_win].mean()
            bg_mean = score_i[non_css_win].mean().clamp_min(eps)

            enrichments.append(css_mean / bg_mean)

        if not rank_percentiles:
            return {}, 0

        out = {
            "css_rank_percentile": torch.stack(rank_percentiles).mean(),
            "css_recall_topk_window": torch.stack(recalls).mean() if recalls else torch.zeros((), device=device),
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

        Useful for b_offset, where a ratio is less meaningful because b can be negative.
        """
        device = values.device
        deltas = []

        B = int(values.shape[0])

        for i in range(B):
            L = int(mask_b[i].sum().detach().cpu().item())

            if L < 3:
                continue

            values_i = values[i, :L].detach().float()
            css_i = css[i] if isinstance(css, (list, tuple)) else None

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
        additive_bg: torch.Tensor | None = None,
        b_offset: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], int]:
        """
        Computes validation CSS diagnostics.

        Main biological interpretation:

          high css_L_queue_*:
              CSS are captured by biological queueing branch.

          high css_additive_bg_enrichment:
              additive branch may be stealing CSS signal.

          high positive css_b_delta:
              multiplicative technical bias may be explaining CSS peaks.
        """
        top_frac = float(getattr(self.config.metrics, "css_top_frac", 0.01))
        min_k = int(getattr(self.config.metrics, "css_min_k", 10))
        window = int(getattr(self.config.metrics, "css_window", 3))
        eps = float(getattr(self.config.loss, "eps", 1e-8))

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

        return logs, css_count
    def _shared_step(
        self,
        batch: Any,
        stage: str,
        batch_idx: int,
    ) -> torch.Tensor:
        ids_datasets_sorted, ids, packed_sequence, profiles_target, lengths, mask, css = batch

        batch_size = int(profiles_target.shape[0])

        y = profiles_target.to(torch.float32)
        mask_b = mask.bool()
        mask_f = mask_b.float()

        mu, tweedie_p, phi, extras = self.model(
            packed_sequence,
            ids_datasets_sorted,
            y,
        )

        # Current expected extras layout:
        # extras[2]  = L_queue
        # extras[8]  = b
        # extras[10] = mu_base
        # extras[11] = additive_bg
        # extras[12] = additive_rel
        L_queue = extras[2]

        b_offset = extras[8] if len(extras) > 8 else None
        mu_base = extras[10] if len(extras) > 10 else None
        additive_bg = extras[11] if len(extras) > 11 else None
        additive_rel = extras[12] if len(extras) > 12 else None
        shift_weights_used = extras[14] if len(extras) > 14 else None
        shift_weights_soft = extras[15] if len(extras) > 15 else None
        nll_per_sample = self.loss_fn(
            mu_phys=mu,
            power=tweedie_p,
            phi=phi,
            y_true=y,
            mask=mask_b,
            return_per_sample=True,
        )

        loss_per_sample = nll_per_sample

        lambda_additive_l1 = float(
            getattr(self.config.loss, "lambda_additive_l1", 0.0)
        )

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

        loss = loss_per_sample.mean()

        with torch.no_grad():
            mu_detached = mu.detach()
            L_queue_detached = L_queue.detach()

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

            mu_pcc = mu_pcc_per_sample.mean()
            L_queue_pcc = L_queue_pcc_per_sample.mean()

            css_logs: dict[str, torch.Tensor] = {}
            css_count = 0

            if stage == "val":
                css_logs, css_count = self._compute_css_diagnostics(
                    L_queue=L_queue_detached,
                    mask_b=mask_b,
                    css=css,
                    additive_bg=additive_bg.detach() if additive_bg is not None else None,
                    b_offset=b_offset.detach() if b_offset is not None else None,
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
                L_queue=L_queue_detached,
                mask_b=mask_b,
                mu_pcc_per_sample=mu_pcc_per_sample,
                L_queue_pcc_per_sample=L_queue_pcc_per_sample,
                batch_idx=batch_idx,
                sample_idx=sample_idx,
                tag="val/profile_diagnostic",
                mu_base=mu_base,
                additive_bg=additive_bg,
                additive_rel=additive_rel,
            )

        self.log(
            f"{stage}_loss",
            loss.detach(),
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

        self.log(
            f"{stage}_nll",
            nll_per_sample.mean().detach(),
            on_step=(stage == "train"),
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
        )

        self.log(
            f"{stage}_p",
            tweedie_p.detach().reshape(()),
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=batch_size,
        )

        self.log(
            f"{stage}_mu_pcc",
            mu_pcc,
            on_step=False,
            on_epoch=True,
            prog_bar=(stage == "val"),
            logger=True,
            batch_size=batch_size,
        )

        self.log(
            f"{stage}_L_queue_pcc",
            L_queue_pcc,
            on_step=False,
            on_epoch=True,
            prog_bar=(stage == "val"),
            logger=True,
            batch_size=batch_size,
        )

        if len(extras) > 12:
            self.log(
                f"{stage}_additive_rel_mean",
                additive_rel_mean.detach(),
                on_step=False,
                on_epoch=True,
                logger=True,
                batch_size=batch_size,
            )

            if lambda_additive_l1 > 0.0:
                self.log(
                    f"{stage}_additive_l1_penalty",
                    additive_penalty.detach(),
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    batch_size=batch_size,
                )

        self._log_dataset_pcc_metrics(
            stage=stage,
            dataset_ids=ids_datasets_sorted,
            mu_pcc_per_sample=mu_pcc_per_sample,
            L_queue_pcc_per_sample=L_queue_pcc_per_sample,
        )
        self._log_dataset_shift_metrics(
            stage=stage,
            dataset_ids=ids_datasets_sorted,
            shift_weights_used=shift_weights_used,
            shift_weights_soft=shift_weights_soft,
        )

        if stage == "val":
            self.log(
                "val_loss_epoch",
                loss.detach(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
            )

            self.log(
                "val_mu_pcc_epoch",
                mu_pcc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
            )

            self.log(
                "val_L_queue_pcc_epoch",
                L_queue_pcc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=batch_size,
            )
        if stage == "val" and css_count > 0:
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

        return loss
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
        """
        Logs dataset-specific shift diagnostics.

        Expected logs:
          {stage}_shift_expected_used_by_dataset/<dataset>
          {stage}_shift_expected_soft_by_dataset/<dataset>
          {stage}_shift_max_prob_soft_by_dataset/<dataset>
          {stage}_shift_prob_soft_k=-1_by_dataset/<dataset>
          {stage}_shift_prob_soft_k=+0_by_dataset/<dataset>
          {stage}_shift_prob_soft_k=+1_by_dataset/<dataset>
        """
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
            dataset_name = self.dataset_id_to_name.get(
                dataset_id,
                f"dataset_{dataset_id}",
            )

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
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train", batch_idx=batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val", batch_idx=batch_idx)

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