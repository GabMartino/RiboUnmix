from __future__ import annotations

from typing import Any

import lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from Dataloaders.RiboAIQueuingMultiDataset.RiboAIQueuingDatamoduleMultiDataset import open_file
from Models.utils.targets import mu_total_from_median_lognormal
from Models.utils.zi_lognormal_loss import ScaledZeroInflatedLogNormalLoss
from Models.utils.log_plot import log_plot_validation


def kl_w_target_vs_w_prob(
    w_prob: torch.Tensor,
    w_target: torch.Tensor,
    mask_b: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    mask_f = mask_b.float()
    wp = w_prob.float() * mask_f
    wt = w_target.float() * mask_f

    wp = wp / wp.sum(dim=1, keepdim=True).clamp_min(eps)
    wt = wt / wt.sum(dim=1, keepdim=True).clamp_min(eps)

    log_wp = wp.clamp_min(eps).log()
    kl_pointwise = F.kl_div(log_wp, wt, reduction="none", log_target=False)
    kl = (kl_pointwise * mask_f).sum(dim=1)
    return kl


class RiboQueuingModelLightningModule(pl.LightningModule):
    _IDX_LOSS = 0
    _IDX_L_PCC = 1
    _IDX_MU_PCC = 2
    _IDX_W_KL = 3
    _IDX_ALPHA = 4
    _NUM_METRICS = 5

    def __init__(self, torch_model: nn.Module, *, config: Any):
        super().__init__()
        self.model = torch_model
        self.config = config

        self.loss_fn = ScaledZeroInflatedLogNormalLoss(
            censor_threshold=self.config.metrics.censor_threshold
        )

        self._val_plot_logged_this_epoch = False

        dataset_encoding = open_file(self.config.paths.encodings.datasets)
        self.dataset_encoding = {k: int(v) for k, v in dataset_encoding.items()}
        self.idx_to_dataset_enc = {v: k for k, v in self.dataset_encoding.items()}

        self.used_datasets_names = list(self.config.experiment.dataset)
        self.num_used_datasets = len(self.used_datasets_names)

        self.used_dataset_ids: list[int] = []
        self.dataset_name_to_used_idx: dict[str, int] = {}
        self.dataset_id_to_used_idx: dict[int, int] = {}

        for idx, dataset_name in enumerate(self.used_datasets_names):
            if dataset_name not in self.dataset_encoding:
                raise ValueError(
                    f"Dataset '{dataset_name}' not found in dataset encoding file."
                )
            dataset_id = int(self.dataset_encoding[dataset_name])
            self.used_dataset_ids.append(dataset_id)
            self.dataset_name_to_used_idx[dataset_name] = idx
            self.dataset_id_to_used_idx[dataset_id] = idx

        self.register_buffer(
            "_train_metric_sums",
            torch.zeros(self._NUM_METRICS, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_train_metric_count",
            torch.zeros((), dtype=torch.float64),
            persistent=False,
        )

        self.register_buffer(
            "_val_metric_sums",
            torch.zeros(self._NUM_METRICS, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_val_metric_count",
            torch.zeros((), dtype=torch.float64),
            persistent=False,
        )

        self.register_buffer(
            "_val_dataset_metric_sums",
            torch.zeros(self.num_used_datasets, self._NUM_METRICS, dtype=torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "_val_dataset_metric_counts",
            torch.zeros(self.num_used_datasets, dtype=torch.float64),
            persistent=False,
        )

    @staticmethod
    def _per_sample_pcc(
        a_t: torch.Tensor,
        b_t: torch.Tensor,
        mask_b: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        a_t = a_t.float()
        b_t = b_t.float()
        m = mask_b.bool()

        a_t = a_t * m
        b_t = b_t * m

        n = m.sum(dim=1).clamp_min(1)

        mean_a = a_t.sum(dim=1) / n
        mean_b = b_t.sum(dim=1) / n

        a_centered = (a_t - mean_a.unsqueeze(1)) * m
        b_centered = (b_t - mean_b.unsqueeze(1)) * m

        cov = (a_centered * b_centered).sum(dim=1)
        var_a = (a_centered ** 2).sum(dim=1)
        var_b = (b_centered ** 2).sum(dim=1)

        denom = torch.sqrt(var_a * var_b).clamp_min(eps)
        pcc = cov / denom

        invalid = (m.sum(dim=1) < 2) | (var_a <= 1e-12) | (var_b <= 1e-12)
        pcc = torch.where(invalid, torch.zeros_like(pcc), pcc)
        return pcc

    @staticmethod
    def _mean_from_sums_and_count(
        sums: torch.Tensor,
        count: torch.Tensor,
    ) -> torch.Tensor:
        denom = count.clamp_min(1.0)
        means = sums / denom
        means = torch.where(count > 0, means, torch.zeros_like(means))
        return means

    def _is_distributed(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    def _all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._is_distributed():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def _reset_train_accumulators(self) -> None:
        self._train_metric_sums.zero_()
        self._train_metric_count.zero_()

    def _reset_val_accumulators(self) -> None:
        self._val_metric_sums.zero_()
        self._val_metric_count.zero_()
        self._val_dataset_metric_sums.zero_()
        self._val_dataset_metric_counts.zero_()

    def _accumulate_global_metrics(
        self,
        stage: str,
        loss_per_sample: torch.Tensor,
        l_pcc_per_sample: torch.Tensor,
        mu_pcc_per_sample: torch.Tensor,
        w_kl_per_sample: torch.Tensor,
        alpha_per_sample: torch.Tensor,
    ) -> None:
        metric_vector = torch.stack(
            [
                loss_per_sample.detach().double().sum(),
                l_pcc_per_sample.detach().double().sum(),
                mu_pcc_per_sample.detach().double().sum(),
                w_kl_per_sample.detach().double().sum(),
                alpha_per_sample.detach().double().sum(),
            ],
            dim=0,
        )

        sample_count = torch.tensor(
            float(loss_per_sample.numel()),
            device=metric_vector.device,
            dtype=torch.float64,
        )

        if stage == "train":
            self._train_metric_sums += metric_vector
            self._train_metric_count += sample_count
        else:
            self._val_metric_sums += metric_vector
            self._val_metric_count += sample_count

    def _accumulate_val_dataset_metrics(
        self,
        ids_datasets_sorted: torch.Tensor,
        loss_per_sample: torch.Tensor,
        l_pcc_per_sample: torch.Tensor,
        mu_pcc_per_sample: torch.Tensor,
        w_kl_per_sample: torch.Tensor,
        alpha_per_sample: torch.Tensor,
    ) -> None:
        for used_idx, dataset_id in enumerate(self.used_dataset_ids):
            ds_mask = ids_datasets_sorted == dataset_id
            if not torch.any(ds_mask):
                continue

            ds_count = ds_mask.sum().to(dtype=torch.float64)
            self._val_dataset_metric_counts[used_idx] += ds_count

            self._val_dataset_metric_sums[used_idx, self._IDX_LOSS] += (
                loss_per_sample[ds_mask].detach().double().sum()
            )
            self._val_dataset_metric_sums[used_idx, self._IDX_L_PCC] += (
                l_pcc_per_sample[ds_mask].detach().double().sum()
            )
            self._val_dataset_metric_sums[used_idx, self._IDX_MU_PCC] += (
                mu_pcc_per_sample[ds_mask].detach().double().sum()
            )
            self._val_dataset_metric_sums[used_idx, self._IDX_W_KL] += (
                w_kl_per_sample[ds_mask].detach().double().sum()
            )
            self._val_dataset_metric_sums[used_idx, self._IDX_ALPHA] += (
                alpha_per_sample[ds_mask].detach().double().sum()
            )

    def _compute_synced_train_means(self) -> torch.Tensor:
        sums = self._train_metric_sums.clone()
        count = self._train_metric_count.clone()

        self._all_reduce_sum(sums)
        self._all_reduce_sum(count)

        return self._mean_from_sums_and_count(sums, count)

    def _compute_synced_val_means(self) -> torch.Tensor:
        sums = self._val_metric_sums.clone()
        count = self._val_metric_count.clone()

        self._all_reduce_sum(sums)
        self._all_reduce_sum(count)

        return self._mean_from_sums_and_count(sums, count)

    def _compute_synced_val_dataset_means(self) -> torch.Tensor:
        sums = self._val_dataset_metric_sums.clone()
        counts = self._val_dataset_metric_counts.clone()

        if self.num_used_datasets > 0:
            self._all_reduce_sum(sums)
            self._all_reduce_sum(counts)

        denom = counts.unsqueeze(1).clamp_min(1.0)
        means = sums / denom
        means = torch.where(counts.unsqueeze(1) > 0, means, torch.zeros_like(means))
        return means

    def on_train_epoch_start(self) -> None:
        self._reset_train_accumulators()

        start_temp = float(self.config.model.w_temperature)
        min_temp = 0.5
        progress = self.current_epoch / max(1, int(self.config.trainer.max_epochs))
        current_temp = start_temp - (start_temp - min_temp) * progress
        self.model.w_temperature = max(current_temp, min_temp)

        self.log(
            "train_w_temperature",
            float(self.model.w_temperature),
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )

    def on_validation_epoch_start(self) -> None:
        self._reset_val_accumulators()
        self._val_plot_logged_this_epoch = False

    def _shared_step(self, batch: Any, stage: str, batch_idx: int) -> torch.Tensor:
        ids_datasets_sorted, ids, packed_sequence, profiles_target, lengths, mask, css = batch
        batch_size = int(profiles_target.shape[0])

        y = profiles_target.to(torch.float32)
        mask_b = mask.bool()
        mask_f = mask_b.float()
        eps = 1e-8

        mu_obs, pi, sigma, extras = self.model(packed_sequence, ids_datasets_sorted, y)
        (rho_diag, w_prob, L_queue, J, S_mean, total_scale, b, alpha, log_sigma) = extras

        loss_per_sample = self.loss_fn(
            mu_obs,
            pi,
            sigma,
            y,
            mask_b,
            return_per_sample=True,
        )
        loss = loss_per_sample.mean()

        with torch.no_grad():
            mu_total = mu_total_from_median_lognormal(
                mu_obs.clamp_min(eps),
                pi,
                sigma,
            )

            L_target = (y / S_mean.clamp_min(eps)) * mask_f
            alpha_mean_per_sample = (alpha * mask_f).sum(dim=1) / lengths.clamp_min(1)

            l_pcc_per_sample = self._per_sample_pcc(L_queue.detach(), L_target, mask_b)
            mu_pcc_per_sample = self._per_sample_pcc(mu_total.detach(), y, mask_b)

            y_shape = y * mask_f
            w_target = y_shape / y_shape.sum(dim=1, keepdim=True).clamp_min(eps)
            w_kl_per_sample = kl_w_target_vs_w_prob(w_prob, w_target, mask_b)

        self._accumulate_global_metrics(
            stage=stage,
            loss_per_sample=loss_per_sample,
            l_pcc_per_sample=l_pcc_per_sample,
            mu_pcc_per_sample=mu_pcc_per_sample,
            w_kl_per_sample=w_kl_per_sample,
            alpha_per_sample=alpha_mean_per_sample,
        )

        if stage == "train":
            self.log(
                "train_loss",
                loss.detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
                sync_dist=False,
                batch_size=batch_size,
            )
            self.log(
                "train_l_pcc",
                l_pcc_per_sample.mean().detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
                sync_dist=False,
                batch_size=batch_size,
            )
            self.log(
                "train_alpha",
                alpha_mean_per_sample.mean().detach(),
                on_step=True,
                on_epoch=False,
                logger=True,
                sync_dist=False,
                batch_size=batch_size,
            )
        else:
            self._accumulate_val_dataset_metrics(
                ids_datasets_sorted=ids_datasets_sorted,
                loss_per_sample=loss_per_sample,
                l_pcc_per_sample=l_pcc_per_sample,
                mu_pcc_per_sample=mu_pcc_per_sample,
                w_kl_per_sample=w_kl_per_sample,
                alpha_per_sample=alpha_mean_per_sample,
            )

            if (
                batch_idx == 0
                and not self._val_plot_logged_this_epoch
                and getattr(self.trainer, "is_global_zero", True)
            ):
                exp = getattr(self.logger, "experiment", None) if self.logger is not None else None

                log_plot_validation(
                    profiles_target.detach().cpu(),
                    mu_obs.detach().cpu(),
                    mu_total.detach().cpu(),
                    pi.detach().cpu(),
                    w_prob.detach().cpu(),
                    sigma.detach().cpu(),
                    css=css,
                    lengths=lengths,
                    sample=0,
                    experiment=exp,
                    step=self.global_step,
                    tag="val/profile_diag",
                )
                self._val_plot_logged_this_epoch = True

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train", batch_idx=batch_idx)

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val", batch_idx=batch_idx)

    def on_train_epoch_end(self) -> None:
        train_means = self._compute_synced_train_means()

        self.log(
            "train_loss_epoch",
            train_means[self._IDX_LOSS].float(),
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
        self.log(
            "train_l_pcc_epoch",
            train_means[self._IDX_L_PCC].float(),
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
        self.log(
            "train_mu_pcc_epoch",
            train_means[self._IDX_MU_PCC].float(),
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
        self.log(
            "train_kl_w_epoch",
            train_means[self._IDX_W_KL].float(),
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
        self.log(
            "train_alpha_epoch",
            train_means[self._IDX_ALPHA].float(),
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )

        self._reset_train_accumulators()

    def on_validation_epoch_end(self) -> None:
        val_means = self._compute_synced_val_means()
        val_dataset_means = self._compute_synced_val_dataset_means()

        val_loss = val_means[self._IDX_LOSS].float()
        val_l_pcc = val_means[self._IDX_L_PCC].float()
        val_mu_pcc = val_means[self._IDX_MU_PCC].float()
        val_kl_w = val_means[self._IDX_W_KL].float()
        val_alpha = val_means[self._IDX_ALPHA].float()

        physics_score = (1.0 - val_l_pcc) + val_kl_w + val_alpha

        self.log("val_loss_epoch", val_loss, on_step=False, on_epoch=True, sync_dist=False)
        self.log("val_l_pcc_epoch", val_l_pcc, on_step=False, on_epoch=True, sync_dist=False)
        self.log("val_mu_pcc_epoch", val_mu_pcc, on_step=False, on_epoch=True, sync_dist=False)
        self.log("val_kl_w_epoch", val_kl_w, on_step=False, on_epoch=True, sync_dist=False)
        self.log("val_alpha_epoch", val_alpha, on_step=False, on_epoch=True, sync_dist=False)
        self.log(
            "val_physics_score",
            physics_score,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
            prog_bar=True,
        )

        for used_idx, dataset_name in enumerate(self.used_datasets_names):
            ds_means = val_dataset_means[used_idx]

            self.log(
                f"val_loss_epoch/{dataset_name}",
                ds_means[self._IDX_LOSS].float(),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
            self.log(
                f"val_l_pcc_epoch/{dataset_name}",
                ds_means[self._IDX_L_PCC].float(),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
            self.log(
                f"val_mu_pcc_epoch/{dataset_name}",
                ds_means[self._IDX_MU_PCC].float(),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
            self.log(
                f"val_kl_w_epoch/{dataset_name}",
                ds_means[self._IDX_W_KL].float(),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
            self.log(
                f"val_alpha_epoch/{dataset_name}",
                ds_means[self._IDX_ALPHA].float(),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

        self._reset_val_accumulators()
        self._val_plot_logged_this_epoch = False

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        id_datasets, ids, x_packed, profiles_target, lengths, mask, css = batch
        eps = 1e-8

        mu_obs, pi, sigma, extras = self.model(x_packed, id_datasets, profiles_target)
        (rho_diag, w_prob, L_queue, J, S_mean, total_scale, b, alpha, log_sigma) = extras

        mu_total = mu_total_from_median_lognormal(
            mu_obs.clamp_min(eps),
            pi,
            sigma,
        )

        return {
            "ids": ids.detach().cpu(),
            "dataset_id": id_datasets.detach().cpu(),
            "lengths": lengths.detach().cpu(),
            "mask": mask.detach().cpu(),
            "rho": rho_diag.detach().cpu(),
            "w_prob": w_prob.detach().cpu(),
            "L_queue": L_queue.detach().cpu(),
            "J": J.detach().cpu(),
            "S_mean": S_mean.detach().cpu(),
            "total_scale": total_scale.detach().cpu(),
            "mu_obs": mu_obs.detach().cpu(),
            "mu_total": mu_total.detach().cpu(),
            "sigma": sigma.detach().cpu(),
            "alpha": alpha.detach().cpu(),
            "b_offset": b.detach().cpu(),
            "log_sigma": log_sigma.detach().cpu(),
            "pi": pi.detach().cpu(),
            "css": css,
            "y": profiles_target.detach().cpu(),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.config.optim.scheduler.mode,
            factor=self.config.optim.scheduler.factor,
            patience=self.config.optim.scheduler.patience,
            min_lr=self.config.optim.scheduler.min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": self.config.optim.scheduler.monitor,
            },
        }