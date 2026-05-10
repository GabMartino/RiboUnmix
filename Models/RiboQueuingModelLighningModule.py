from __future__ import annotations

from typing import Any

import lightning as pl
import torch
import torch.nn as nn

from Models.utils.PCGrad_utils import per_dataset_losses, pcgrad_combine, assign_flat_grads
from Models.utils.advanced_metrics import masked_mae, masked_auprc_peak_caller, masked_1d_wasserstein, \
    physics_asymmetry_ratio
from Models.utils.masked_pearson import MaskedPearsonCorrelation
from Models.utils.ribo_lightning_helpers import flatten_current_grads
from Models.utils.tweedie_deviance_loss import TweedieDevianceLoss


class RiboQueuingModelLightningModule(pl.LightningModule):
    def __init__(self, torch_model: nn.Module, *, config: Any, dataset_encoding: dict = None):
        super().__init__()
        self.save_hyperparameters(ignore=["torch_model", "config", "dataset_encoding"])
        self.model = torch_model
        self.config = config
        self._val_plot_logged_this_epoch = False

        self.loss_fn = TweedieDevianceLoss(include_log_phi=True)

        enc = dataset_encoding or {}
        self.dataset_id_to_name = {int(v): str(k) for k, v in enc.items()}
        self.use_pcgrad = self.config.optim.use_pcgrad
        if self.use_pcgrad:
            print("Training is using PCGrad.")
        self.automatic_optimization = not self.use_pcgrad

        self.masked_pcc = MaskedPearsonCorrelation(
            eps=float(self.config.loss.eps),
        )

    def _log_profile_diagnostics(
            self,
            *,
            stage: str,
            target: torch.Tensor,
            mask: torch.Tensor,
            css: list,
            dataset_ids: torch.Tensor,
            components: dict[str, torch.Tensor | None],
            batch_size: int,
    ) -> None:
        dataset_ids = dataset_ids.detach()
        mask_b = mask.bool()
        mask_f = mask.float()

        with torch.no_grad():
            # ==========================================
            # 1. GLOBAL METRICS
            # ==========================================
            for component_name, value in components.items():
                if value is None: continue
                pcc_per_sample = self.masked_pcc(pred=value.detach(), target=target, mask=mask_b)

                self.log(
                    f"{stage}_{component_name}_pcc",
                    pcc_per_sample.mean().detach(),
                    on_step=False, on_epoch=True,
                    prog_bar=(stage == "val" and component_name in {"mu", "L_queue", "mu_base"}),
                    logger=True, batch_size=batch_size,
                )

            eval_tensor = components.get("L_queue") if components.get("L_queue") is not None else components.get("mu")

            if eval_tensor is not None:
                eval_tensor = eval_tensor.detach()
                global_mae = masked_mae(eval_tensor, target, mask_f)
                global_emd = masked_1d_wasserstein(eval_tensor, target, mask_f)
                global_auprc = masked_auprc_peak_caller(eval_tensor, target, mask_f)
                global_phys_ratio = physics_asymmetry_ratio(eval_tensor, css)

                self.log(f"{stage}_eval_MAE", global_mae, on_epoch=True, logger=True, batch_size=batch_size)
                self.log(f"{stage}_eval_EMD", global_emd, on_epoch=True, logger=True, batch_size=batch_size)
                self.log(f"{stage}_eval_AUPRC", global_auprc, on_epoch=True, logger=True, batch_size=batch_size)
                self.log(f"{stage}_eval_Physics_Ratio", global_phys_ratio, on_epoch=True, logger=True,
                         batch_size=batch_size)

            # ==========================================
            # 2. PER-DATASET METRICS
            # ==========================================
            unique_ids = torch.unique(dataset_ids).detach().cpu().tolist()

            for ds_id in unique_ids:
                ds_id = int(ds_id)
                dataset_name = self.dataset_id_to_name.get(ds_id, f"dataset_{ds_id}")
                ds_mask = (dataset_ids == ds_id)
                ds_count = int(ds_mask.sum().detach().cpu().item())

                if ds_count == 0: continue
                component_pccs = {}
                for component_name, value in components.items():
                    if value is None:
                        continue

                    pcc_per_sample = self.masked_pcc(
                        pred=value.detach(),
                        target=target,
                        mask=mask_b,
                    )

                    component_pccs[component_name] = pcc_per_sample
                    for component_name, pcc_per_sample in component_pccs.items():
                        self.log(
                            f"{stage}_{component_name}_pcc_by_dataset/{dataset_name}",
                            pcc_per_sample[ds_mask].mean().detach(),
                            on_step=False,
                            on_epoch=True,
                            logger=True,
                            batch_size=ds_count,
                        )

                # B. Per-Dataset Advanced Metrics
                if eval_tensor is not None:
                    # Isolate tensors for this specific dataset
                    ds_eval = eval_tensor[ds_mask]
                    ds_target = target[ds_mask]
                    ds_mask_f = mask_f[ds_mask]

                    # Isolate CSS list for this dataset
                    ds_css = [css[i] for i, m in enumerate(ds_mask.tolist()) if m]

                    ds_mae = masked_mae(ds_eval, ds_target, ds_mask_f)
                    ds_emd = masked_1d_wasserstein(ds_eval, ds_target, ds_mask_f)
                    ds_auprc = masked_auprc_peak_caller(ds_eval, ds_target, ds_mask_f)
                    ds_phys_ratio = physics_asymmetry_ratio(ds_eval, ds_css)

                    self.log(f"{stage}_eval_MAE_by_dataset/{dataset_name}", ds_mae, on_epoch=True, logger=True,
                             batch_size=ds_count)
                    self.log(f"{stage}_eval_EMD_by_dataset/{dataset_name}", ds_emd, on_epoch=True, logger=True,
                             batch_size=ds_count)
                    self.log(f"{stage}_eval_AUPRC_by_dataset/{dataset_name}", ds_auprc, on_epoch=True, logger=True,
                             batch_size=ds_count)
                    self.log(f"{stage}_eval_Physics_Ratio_by_dataset/{dataset_name}", ds_phys_ratio, on_epoch=True,
                             logger=True, batch_size=ds_count)



    def pcgrad_optimize(self, loss_per_sample, dataset_ids):
        opt = self.optimizers()
        opt.zero_grad(set_to_none=True)

        biological_model = self.model.biological_model
        biological_model_params = [
            p for p in biological_model.parameters()
            if p.requires_grad
        ]

        dataset_losses = per_dataset_losses(
            loss_per_sample=loss_per_sample,
            dataset_ids=dataset_ids,
        )

        if len(dataset_losses) > 0:
            loss = torch.stack(dataset_losses).mean()
        else:
            loss = loss_per_sample.mean()

        if len(dataset_losses) <= 1 or len(biological_model_params) == 0:
            self.manual_backward(loss)
            return loss.detach()

        max_datasets = int(self.config.optim.pcgrad_max_datasets_per_step)

        if len(dataset_losses) > max_datasets:
            perm = torch.randperm(len(dataset_losses), device=loss.device)
            selected = perm[:max_datasets].detach().cpu().tolist()
            dataset_losses_for_pcgrad = [dataset_losses[j] for j in selected]
        else:
            dataset_losses_for_pcgrad = dataset_losses

        flat_dataset_grads = []

        for ds_loss in dataset_losses_for_pcgrad:
            opt.zero_grad(set_to_none=True)
            self.manual_backward(ds_loss, retain_graph=True)
            flat_dataset_grads.append(
                flatten_current_grads(biological_model_params)
            )

        pcgrad_flat = pcgrad_combine(flat_dataset_grads)
        mean_flat_grad = torch.stack(flat_dataset_grads, dim=0).mean(dim=0)

        pcgrad_alpha = float(self.config.optim.pcgrad_alpha)

        pcgrad_flat = (
                pcgrad_alpha * pcgrad_flat
                + (1.0 - pcgrad_alpha) * mean_flat_grad
        )

        if bool(self.config.optim.pcgrad_rescale_to_mean_norm):
            mean_norm = mean_flat_grad.norm().clamp_min(1e-12)
            pcgrad_norm = pcgrad_flat.norm().clamp_min(1e-12)
            pcgrad_flat = pcgrad_flat * (mean_norm / pcgrad_norm)

        opt.zero_grad(set_to_none=True)

        # Normal gradients for all parameters.
        self.manual_backward(loss)

        # Replace only biological gradients.
        assign_flat_grads(
            params=biological_model_params,
            flat_grad=pcgrad_flat,
        )

        return loss.detach()

    def training_step(self, batch, batch_idx):
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

        nll_per_sample = self.loss_fn(
            mu_phys=mu,
            power=p,
            phi=phi,
            y_true=prof_pad,
            mask=mask_pad.bool(),
            return_per_sample=True,
        )

        loss_per_sample = nll_per_sample

        dataset_losses = per_dataset_losses(
            loss_per_sample=loss_per_sample,
            dataset_ids=ids_datasets_sorted,
        )
        if batch_idx == 0:
            self._log_profile_diagnostics(  # Updated name
                stage="train",  # (or "val" in validation_step)
                target=prof_pad,
                mask=mask_pad,
                css=css_sorted,  # Added CSS for physics ratio
                dataset_ids=ids_datasets_sorted,
                components={
                    "mu": mu,
                    "L_queue": extras.get("L_queue"),
                    "L_effective": extras.get("L_effective"),
                    "mu_base": extras.get("mu_base"),
                    "additive_bias": extras.get("additive_bias"),
                },
                batch_size=int(prof_pad.shape[0]),
            )
        if len(dataset_losses) > 0:
            loss = torch.stack(dataset_losses).mean()
        else:
            loss = loss_per_sample.mean()

        # ------------------------------------------------------------
        # Automatic optimization path: Lightning handles backward + step.
        # ------------------------------------------------------------
        if not self.use_pcgrad:
            self.log(
                "train_loss",
                loss.detach(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=int(prof_pad.shape[0]),
            )
            return loss

        # ------------------------------------------------------------
        # Manual optimization path: PCGrad handles backward, then step.
        # ------------------------------------------------------------
        opt = self.optimizers()
        opt.zero_grad(set_to_none=True)

        loss = self.pcgrad_optimize(
            loss_per_sample=loss_per_sample,
            dataset_ids=ids_datasets_sorted,
        )

        if float(self.config.trainer.gradient_clip_val) > 0.0:
            self.clip_gradients(
                opt,
                gradient_clip_val=float(self.config.trainer.gradient_clip_val),
                gradient_clip_algorithm=self.config.trainer.gradient_clip_algorithm,
            )

        opt.step()
        opt.zero_grad(set_to_none=True)

        self.log(
            "train_loss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=int(prof_pad.shape[0]),
        )

        return loss.detach()

    def validation_step(self, batch, batch_idx):
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

        nll_per_sample = self.loss_fn(
            mu_phys=mu,
            power=p,
            phi=phi,
            y_true=prof_pad,
            mask=mask_pad.bool(),
            return_per_sample=True,
        )

        loss_per_sample = nll_per_sample

        loss_sample_mean = loss_per_sample.mean()

        dataset_losses = per_dataset_losses(
            loss_per_sample=loss_per_sample,
            dataset_ids=ids_datasets_sorted,
        )

        if len(dataset_losses) > 0:
            loss_dataset_balanced = torch.stack(dataset_losses).mean()
        else:
            loss_dataset_balanced = loss_sample_mean

        self._log_profile_diagnostics(  # Updated name
            stage="val",  # (or "val" in validation_step)
            target=prof_pad,
            mask=mask_pad,
            css=css_sorted,  # Added CSS for physics ratio
            dataset_ids=ids_datasets_sorted,
            components={
                "mu": mu,
                "L_queue": extras.get("L_queue"),
                "L_effective": extras.get("L_effective"),
                "mu_base": extras.get("mu_base"),
                "additive_bias": extras.get("additive_bias"),
            },
            batch_size=int(prof_pad.shape[0]),
        )
        # Main monitor metric.
        self.log(
            "val_loss",
            loss_dataset_balanced,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=int(prof_pad.shape[0]),
        )

        # Useful diagnostic: raw sample-weighted validation loss.
        self.log(
            "val_loss_sample_mean",
            loss_sample_mean,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=int(prof_pad.shape[0]),
        )

        return loss_dataset_balanced

    def on_validation_epoch_end(self):
        if not self.use_pcgrad:
            return

        sched = self.lr_schedulers()
        metric = self.trainer.callback_metrics[self.config.optim.scheduler.monitor]
        sched.step(metric)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> dict[str, Any]:
        # 1. Unpack the batch exactly as you do in training_step
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

        # 2. Run the forward pass
        mu, p, phi, extras = self.model(
            seq_packed,
            codon_ids_pad,
            ids_datasets_sorted,
            prof_pad,
        )

        # 3. CPU Mover Helper
        # Crucial for predict_step to prevent OOM errors when collecting predictions
        def to_cpu(x):
            if torch.is_tensor(x):
                return x.detach().cpu()
            if isinstance(x, (list, tuple)):
                return [to_cpu(v) for v in x]
            return x

        # 4. Build the core output dictionary
        output = {
            "ids": to_cpu(ids_sorted),
            "dataset_id": to_cpu(ids_datasets_sorted),
            "lengths": to_cpu(lengths_sorted),
            "mask": to_cpu(mask_pad),
            "css": to_cpu(css_sorted),
            "y": to_cpu(prof_pad),
            "mu_obs": to_cpu(mu),
            "phi": to_cpu(phi),
            "tweedie_p": to_cpu(p),
        }

        # 5. Safely unpack all biological internals from the 'extras' dictionary
        if isinstance(extras, dict):
            for key, val in extras.items():
                if val is not None:
                    output[key] = to_cpu(val)

        return output


    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.optim.lr),
            weight_decay=float(self.config.optim.weight_decay),
        )

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=str(self.config.optim.scheduler.mode),
            factor=float(self.config.optim.scheduler.factor),
            patience=int(self.config.optim.scheduler.patience),
            min_lr=float(self.config.optim.scheduler.min_lr),
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": str(self.config.optim.scheduler.monitor),
                "interval": "epoch",
                "frequency": 1,
            },
        }