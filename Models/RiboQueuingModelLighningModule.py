from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import lightning as pl
import torch
import torch.nn as nn

from Models.utils.masked_pearson import MaskedPearsonCorrelation
from Models.utils.ribo_lightning_helpers import (
    assign_flat_grads,
    cfg_get,
    compute_css_diagnostics,
    flatten_current_grads,
    get_pcgrad_target_parameters,
    get_shift_values,
    make_validation_profile_figure,
    pcgrad_combine,
    pcgrad_pairwise_stats,
    per_dataset_losses,
    unpack_batch,
)
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
            eps=float(cfg_get(self.config, "loss.eps", 1e-8)),
            phi_min=float(cfg_get(self.config, "loss.phi_min", 1e-4)),
            phi_max=float(cfg_get(self.config, "loss.phi_max", 10.0)),
            censor_threshold=float(cfg_get(self.config, "loss.censor_threshold", 0.0)),
            zero_censor_to_zero=bool(
                cfg_get(self.config, "loss.zero_censor_to_zero", True)
            ),
            include_log_phi=bool(cfg_get(self.config, "loss.include_log_phi", True)),
        )

        self.masked_pcc = MaskedPearsonCorrelation(
            eps=float(cfg_get(self.config, "loss.eps", 1e-8)),
        )

        if dataset_encoding is None:
            dataset_encoding = {}

        self.dataset_encoding = {str(k): int(v) for k, v in dataset_encoding.items()}
        self.dataset_id_to_name = {
            int(v): str(k)
            for k, v in self.dataset_encoding.items()
        }

        self.use_pcgrad = bool(cfg_get(self.config, "optim.use_pcgrad", True))

        # PCGrad requires manual optimization because gradients are rewritten.
        self.automatic_optimization = not self.use_pcgrad

    # ============================================================
    # Epoch hooks
    # ============================================================

    def on_validation_epoch_start(self) -> None:
        self._val_plot_logged_this_epoch = False

    def on_validation_epoch_end(self) -> None:
        if not self.use_pcgrad:
            return

        scheduler = self.lr_schedulers()

        if scheduler is None:
            return

        monitor = str(cfg_get(self.config, "optim.scheduler.monitor", "val_loss_epoch"))
        metric = self.trainer.callback_metrics.get(monitor)

        if metric is None:
            return

        if isinstance(scheduler, list):
            for sched in scheduler:
                sched.step(metric)
        else:
            scheduler.step(metric)

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

        fig = make_validation_profile_figure(
            y=y,
            mu=mu,
            phi=phi,
            tweedie_p=tweedie_p,
            L_queue=L_queue,
            mask_b=mask_b,
            mu_pcc_per_sample=mu_pcc_per_sample,
            L_queue_pcc_per_sample=L_queue_pcc_per_sample,
            sample_idx=sample_idx,
            mu_base=mu_base,
            additive_bg=additive_bg,
            additive_rel=additive_rel,
            css=css,
        )

        if fig is None:
            return

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

        shifts = get_shift_values(
            self.model,
            device=weights_ref.device,
            dtype=weights_ref.dtype,
        )

        if shifts is None:
            return

        def expected_shift(weights: torch.Tensor) -> torch.Tensor:
            return (weights * shifts.reshape(1, -1)).sum(dim=1)

        expected_used = (
            expected_shift(shift_weights_used.detach())
            if shift_weights_used is not None
            else None
        )

        expected_soft = (
            expected_shift(shift_weights_soft.detach())
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
        batch_data = unpack_batch(batch)

        ids_datasets_sorted = batch_data["ids_datasets_sorted"]
        ids = batch_data["ids"]
        packed_sequence = batch_data["packed_sequence"]
        profiles_target = batch_data["profiles_target"]
        lengths = batch_data["lengths"]
        mask = batch_data["mask"]
        codon_ids = batch_data["codon_ids"]
        css = batch_data["css"]

        y = profiles_target.to(torch.float32)
        mask_b = mask.bool()
        mask_f = mask_b.float()

        mu, tweedie_p, phi, extras = self.model(
            packed_sequence,
            codon_ids,
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

        lambda_additive_l1 = float(
            cfg_get(self.config, "loss.lambda_additive_l1", 0.0)
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

        dataset_losses = per_dataset_losses(
            loss_per_sample=loss_per_sample,
            dataset_ids=ids_datasets_sorted,
        )

        loss_sample_mean = loss_per_sample.mean()

        if len(dataset_losses) > 0:
            loss_dataset_balanced = torch.stack(dataset_losses).mean()
        else:
            loss_dataset_balanced = loss_sample_mean

        use_dataset_balanced_loss = bool(
            cfg_get(self.config, "optim.use_dataset_balanced_loss", False)
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
            "codon_ids": codon_ids,
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
                component_pccs["L_effective"] = self.masked_pcc(
                    pred=L_effective.detach(),
                    target=y,
                    mask=mask_b,
                )

            if mu_base is not None:
                component_pccs["mu_base"] = self.masked_pcc(
                    pred=mu_base.detach(),
                    target=y,
                    mask=mask_b,
                )

            mu_pcc = mu_pcc_per_sample.mean()
            L_queue_pcc = L_queue_pcc_per_sample.mean()

            phi_mean = (
                (phi_detached.float() * mask_f).sum()
                / mask_f.sum().clamp_min(1.0)
            )

            css_logs: dict[str, torch.Tensor] = {}
            css_count = 0

            if stage == "val":
                css_logs, css_count = compute_css_diagnostics(
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
                    top_frac=float(cfg_get(self.config, "metrics.css_top_frac", 0.01)),
                    min_k=int(cfg_get(self.config, "metrics.css_min_k", 10)),
                    window=int(cfg_get(self.config, "metrics.css_window", 3)),
                    eps=float(cfg_get(self.config, "loss.eps", 1e-8)),
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

            lambda_additive_l1 = float(
                cfg_get(self.config, "loss.lambda_additive_l1", 0.0)
            )

            if lambda_additive_l1 > 0.0:
                self.log(
                    f"{stage}_additive_l1_penalty",
                    info["additive_penalty"].detach(),
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                    batch_size=batch_size,
                )

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

        pcgrad_every_n_steps = int(
            cfg_get(self.config, "optim.pcgrad_every_n_steps", 1)
        )

        use_pcgrad_this_step = (
            pcgrad_every_n_steps > 0
            and (self.global_step % pcgrad_every_n_steps == 0)
        )

        pcgrad_params = get_pcgrad_target_parameters(
            self.model,
            biology_only=bool(cfg_get(self.config, "optim.pcgrad_biology_only", True)),
        )

        if (
            not use_pcgrad_this_step
            or len(dataset_losses) <= 1
            or len(pcgrad_params) == 0
        ):
            self.manual_backward(loss)

        else:
            flat_dataset_grads = []

            for ds_loss in dataset_losses:
                opt.zero_grad(set_to_none=True)

                self.manual_backward(
                    ds_loss,
                    retain_graph=True,
                )

                flat_g = flatten_current_grads(pcgrad_params)
                flat_dataset_grads.append(flat_g)

            mean_flat_grad = torch.stack(flat_dataset_grads, dim=0).mean(dim=0)
            pcgrad_flat = pcgrad_combine(flat_dataset_grads)

            pcgrad_alpha = float(cfg_get(self.config, "optim.pcgrad_alpha", 1.0))
            pcgrad_alpha = max(0.0, min(pcgrad_alpha, 1.0))

            if pcgrad_alpha < 1.0:
                pcgrad_flat = (
                    pcgrad_alpha * pcgrad_flat
                    + (1.0 - pcgrad_alpha) * mean_flat_grad
                )

            mean_grad_norm = mean_flat_grad.norm().clamp_min(1e-12)
            pcgrad_norm = pcgrad_flat.norm()
            pcgrad_norm_ratio = pcgrad_norm / mean_grad_norm

            if bool(cfg_get(self.config, "optim.pcgrad_rescale_to_mean_norm", False)):
                pcgrad_flat = pcgrad_flat * (
                    mean_grad_norm / pcgrad_flat.norm().clamp_min(1e-12)
                )

            cosine_mean, conflict_frac = pcgrad_pairwise_stats(flat_dataset_grads)

            opt.zero_grad(set_to_none=True)
            self.manual_backward(loss)

            assign_flat_grads(
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

            self.log(
                "train_pcgrad_bio_norm_ratio",
                pcgrad_norm_ratio.detach(),
                on_step=True,
                on_epoch=True,
                logger=True,
                batch_size=batch_size,
            )

            self.log(
                "train_pcgrad_bio_norm",
                pcgrad_norm.detach(),
                on_step=True,
                on_epoch=True,
                logger=True,
                batch_size=batch_size,
            )

            self.log(
                "train_pcgrad_bio_mean_grad_norm",
                mean_grad_norm.detach(),
                on_step=True,
                on_epoch=True,
                logger=True,
                batch_size=batch_size,
            )

        gradient_clip_val = float(cfg_get(self.config, "trainer.gradient_clip_val", 0.0))
        gradient_clip_algorithm = str(
            cfg_get(self.config, "trainer.gradient_clip_algorithm", "norm")
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
        batch_data = unpack_batch(batch)

        ids_datasets_sorted = batch_data["ids_datasets_sorted"]
        ids = batch_data["ids"]
        packed_sequence = batch_data["packed_sequence"]
        profiles_target = batch_data["profiles_target"]
        lengths = batch_data["lengths"]
        mask = batch_data["mask"]
        codon_ids = batch_data["codon_ids"]
        css = batch_data["css"]

        y = profiles_target.to(torch.float32)
        mask_b = mask.bool()
        mask_f = mask_b.float()

        mu, tweedie_p, phi, extras = self.model(
            packed_sequence,
            codon_ids,
            ids_datasets_sorted,
            y,
        )

        def to_cpu(x):
            if torch.is_tensor(x):
                return x.detach().cpu()
            if isinstance(x, list):
                return [to_cpu(v) for v in x]
            if isinstance(x, tuple):
                return tuple(to_cpu(v) for v in x)
            return x

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
        codon_ids = extras[13] if len(extras) > 13 else codon_ids_from_batch
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