from __future__ import annotations

from typing import Any

import lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from Dataloaders.RiboAIQueuingMultiDataset.RiboAIQueuingDatamoduleMultiDataset import open_file
from Models.utils.targets import mu_total_from_median_lognormal
from Models.utils.zi_lognormal_loss import ScaledZeroInflatedLogNormalLoss
from Models.utils.log_plot import log_plot_validation
from Utils.utils import PearsonCorrelation
import torch.nn.functional as F

def kl_w_target_vs_w_prob(
    w_prob: torch.Tensor,
    w_target: torch.Tensor,
    mask_b: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Returns per-sample KL(w_target || w_prob), shape [B].
    Uses PyTorch native F.kl_div to safely handle true zeros in the target.
    """
    mask_f = mask_b.float()

    # 1. Apply mask to isolate the biological sequence from padding
    wp = w_prob.float() * mask_f
    wt = w_target.float() * mask_f

    # 2. Normalize strictly per sequence so the valid positions sum to exactly 1.0
    # The denominator clamp prevents division by zero if a sequence is entirely masked/empty
    wp = wp / wp.sum(dim=1, keepdim=True).clamp_min(eps)
    wt = wt / wt.sum(dim=1, keepdim=True).clamp_min(eps)

    # 3. Convert prediction to log-space (Required by F.kl_div)
    # We MUST clamp wp here to prevent log(0) from creating -inf
    log_wp = wp.clamp_min(eps).log()

    # 4. Pointwise KL Divergence
    # By not clamping wt, we allow PyTorch to evaluate 0 * log(0 / pred) as 0 natively.
    kl_pointwise = F.kl_div(log_wp, wt, reduction='none', log_target=False)

    # 5. Apply the mask again to silence any numerical noise in the padded regions, then sum
    kl = (kl_pointwise * mask_f).sum(dim=1)  # [B]

    return kl


def masked_variance(tensor: torch.Tensor, mask_b: torch.Tensor) -> torch.Tensor:
    """
    Computes the variance of a tensor along dim=1, ignoring padded regions.
    Returns shape [B].
    """
    m = mask_b.float()
    n = m.sum(dim=1).clamp_min(2)  # Need at least 2 valid points for variance

    # 1. Masked Mean
    mean = (tensor * m).sum(dim=1) / n

    # 2. Masked Centering
    centered = (tensor - mean.unsqueeze(1)) * m

    # 3. Bessel's Correction (n-1) for unbiased sample variance
    var = (centered ** 2).sum(dim=1) / (n - 1)

    # Silence sequences that were too short
    invalid = m.sum(dim=1) < 2
    return torch.where(invalid, torch.zeros_like(var), var)



class RiboQueuingModelMultiEmbeddingsLightningModule(pl.LightningModule):
    def __init__(self, torch_model: nn.Module, *, config: Any):
        super().__init__()
        self.model = torch_model
        self.config = config

        self.loss_fn = ScaledZeroInflatedLogNormalLoss(
            censor_threshold=self.config.loss.censor_threshold
        )

        self._sigma_is_frozen = False
        self.pcc = PearsonCorrelation("batch_mean")

        self.train_loss_epoch = torchmetrics.MeanMetric()
        self.val_loss_epoch = torchmetrics.MeanMetric()

        self.rho_train_epoch = torchmetrics.MeanMetric()
        self.rho_val_epoch = torchmetrics.MeanMetric()

        self.mu_train_epoch = torchmetrics.MeanMetric()
        self.mu_val_epoch = torchmetrics.MeanMetric()

        self.w_train_epoch = torchmetrics.MeanMetric()
        self.w_val_epoch = torchmetrics.MeanMetric()

        self.kl_w_train_epoch = torchmetrics.MeanMetric()
        self.kl_w_val_epoch = torchmetrics.MeanMetric()

        self._val_plot_logged_this_epoch = False




        self.dataset_encoding = open_file(self.config.paths.encodings.datasets)
        self.idx_to_dataset_enc = {v: k for k, v in self.dataset_encoding.items()}

        self.used_datasets_names = list(self.config.experiment.dataset)
        self.num_used_datasets = len(self.used_datasets_names)

        self.val_loss_per_dataset = nn.ModuleList(
            [torchmetrics.MeanMetric() for _ in range(self.num_used_datasets)]
        )
        self.val_mu_pcc_per_dataset = nn.ModuleList(
            [torchmetrics.MeanMetric() for _ in range(self.num_used_datasets)]
        )
        self.val_rho_pcc_per_dataset = nn.ModuleList(
            [torchmetrics.MeanMetric() for _ in range(self.num_used_datasets)]
        )
        self.val_w_pcc_per_dataset = nn.ModuleList(
            [torchmetrics.MeanMetric() for _ in range(self.num_used_datasets)]
        )
        self.val_w_kl_per_dataset = nn.ModuleList(
            [torchmetrics.MeanMetric() for _ in range(self.num_used_datasets)]
        )

    def _set_sigma_frozen(self, frozen: bool) -> None:
        for p in self.model.ff_delta_sigma.parameters():
            p.requires_grad = not frozen
        self.model.ff_delta_sigma.eval() if frozen else self.model.ff_delta_sigma.train()
        self._sigma_is_frozen = frozen



    def on_validation_epoch_start(self) -> None:
        self._val_plot_logged_this_epoch = False

    def on_train_epoch_start(self) -> None:
        # Existing sigma freeze logic
        should_freeze = self.current_epoch < self.config.sigma.freeze_epochs
        if should_freeze != self._sigma_is_frozen:
            self._set_sigma_frozen(should_freeze)

        # NEW: Dynamic Temperature Annealing
        # Starts at initial w_temperature (e.g., 2.0) and linearly decays to 0.5
        start_temp = self.config.model.w_temperature
        min_temp = 0.5
        progress = self.current_epoch / max(1, self.config.trainer.max_epochs)

        # Calculate current decayed temperature
        current_temp = start_temp - (start_temp - min_temp) * progress

        # Update the model's physical temperature parameter
        self.model.w_temperature = max(current_temp, min_temp)
        self.log("train_w_temperature", self.model.w_temperature, on_step=False, on_epoch=True , sync_dist=True)

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        if self.current_epoch < self.config.sigma.freeze_epochs:
            self.model.ff_delta_sigma.eval()

    def _shared_step(self, batch: Any, stage: str, batch_idx: int) -> torch.Tensor:
        ids_datasets_sorted, ids, packed_sequence, profiles_target, lengths, mask, css, batch_embeddings = batch
        batch_size = profiles_target.shape[0]

        y = profiles_target.to(torch.float32)
        mask_b = mask.bool()
        mask_f = mask_b.float()
        eps = 1e-8

        mu_obs, pi, sigma, extras = self.model(packed_sequence, ids_datasets_sorted, y, batch_embeddings)
        rho, w_prob, J, transcript_scale_S, log_transcript_scale, total_scale, a, b, log_sigma = extras

        loss_per_sample = self.loss_fn(mu_obs, pi, sigma, y, mask_b, return_per_sample=True)
        loss = loss_per_sample.mean()

        with torch.no_grad():
            mu_total = mu_total_from_median_lognormal(mu_obs.clamp_min(eps), pi, sigma)
            a_var_per_sample = masked_variance(a.detach(), mask_b)
            b_var_per_sample = masked_variance(b.detach(), mask_b)
            scale_var_per_sample = masked_variance(total_scale.detach(), mask_b)

            # rho target: divide by the total scale
            rho_t = (y / total_scale.clamp_min(eps)) * mask_f

            # w target: total_scale is multiplicative, so it cancels under normalization
            y_shape = y * mask_f
            den = y_shape.sum(dim=1, keepdim=True).clamp_min(1e-6)
            w_target = y_shape / den

            def per_sample_pcc(a_t: torch.Tensor, b_t: torch.Tensor, mask_bool: torch.Tensor) -> torch.Tensor:
                a_t = a_t.float()
                b_t = b_t.float()
                m = mask_bool.bool()

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

                denom = torch.sqrt(var_a * var_b).clamp_min(1e-8)
                pcc = cov / denom

                invalid = (m.sum(dim=1) < 2) | (var_a <= 1e-12) | (var_b <= 1e-12)
                pcc = torch.where(invalid, torch.zeros_like(pcc), pcc)
                return pcc

            mu_pcc_per_sample = per_sample_pcc(mu_total.detach(), y, mask_b)
            rho_pcc_per_sample = per_sample_pcc(rho.detach(), rho_t, mask_b)
            w_pcc_per_sample = per_sample_pcc(w_prob.detach(), w_target, mask_b)

            mu_pcc = mu_pcc_per_sample.mean()
            rho_pcc = rho_pcc_per_sample.mean()
            w_pcc = w_pcc_per_sample.mean()

            w_kl = kl_w_target_vs_w_prob(w_prob, w_target, mask_b)
            w_kl_mean = w_kl.mean()

            if stage == "train":
                self.log("train_loss", loss, on_step=True, on_epoch=False, batch_size=batch_size)
                self.log("train_loss_nll", loss, on_step=True, on_epoch=False, batch_size=batch_size)

                self.log("train_rho_pcc", rho_pcc, on_step=True, on_epoch=False, batch_size=batch_size)
                self.log("train_mu_pcc", mu_pcc, on_step=True, on_epoch=False, batch_size=batch_size)
                self.log("train_w_pcc", w_pcc, on_step=True, on_epoch=False, batch_size=batch_size)
                self.log("train_kl_div_w", w_kl_mean, on_step=True, on_epoch=False, batch_size=batch_size)

                sigma_valid = sigma[mask_b]
                log_sigma_valid = log_sigma[mask_b]
                pi_valid = pi[mask_b]
                mu_valid = mu_obs[mask_b]
                total_scale_valid = total_scale[mask_b]
                a_valid = a[mask_b]
                b_valid = b[mask_b]

                if sigma_valid.numel() > 0:
                    self.log("train_sigma_mean", sigma_valid.mean(), on_step=True, on_epoch=False,
                             batch_size=batch_size)
                    self.log("train_sigma_min", sigma_valid.min(), on_step=True, on_epoch=False, batch_size=batch_size)
                    self.log("train_log_sigma_mean", log_sigma_valid.mean(), on_step=True, on_epoch=False,
                             batch_size=batch_size)

                if pi_valid.numel() > 0:
                    self.log("train_pi_mean", pi_valid.mean(), on_step=True, on_epoch=False, batch_size=batch_size)

                if mu_valid.numel() > 0:
                    self.log("train_mu_obs_mean", mu_valid.mean(), on_step=True, on_epoch=False, batch_size=batch_size)

                if total_scale_valid.numel() > 0:
                    self.log("train_total_scale_mean", total_scale_valid.mean(), on_step=True, on_epoch=False,
                             batch_size=batch_size)

                if a_valid.numel() > 0:
                    self.log("train_a_mean", a_valid.mean(), on_step=True, on_epoch=False, batch_size=batch_size)

                if b_valid.numel() > 0:
                    self.log("train_b_mean", b_valid.mean(), on_step=True, on_epoch=False, batch_size=batch_size)

                self.log("train_log_transcript_offset_mean", log_transcript_scale.mean(), on_step=True, on_epoch=False,
                         batch_size=batch_size)
                self.log("train_J_mean", J.mean(), on_step=True, on_epoch=False, batch_size=batch_size)

                self.train_loss_epoch.update(loss.detach())
                self.rho_train_epoch.update(rho_pcc.detach())
                self.mu_train_epoch.update(mu_pcc.detach())
                self.w_train_epoch.update(w_pcc.detach())
                self.kl_w_train_epoch.update(w_kl_mean.detach())

            else:
                self.log("val_diag_a_variance", a_var_per_sample.mean(),
                         on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log("val_diag_b_variance", b_var_per_sample.mean(),
                         on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.log("val_diag_scale_variance", scale_var_per_sample.mean(),
                         on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)
                self.val_loss_epoch.update(loss.detach())
                self.rho_val_epoch.update(rho_pcc.detach())
                self.mu_val_epoch.update(mu_pcc.detach())
                self.w_val_epoch.update(w_pcc.detach())
                self.kl_w_val_epoch.update(w_kl_mean.detach())

                for d in ids_datasets_sorted.unique():
                    '''
                        These indeces are based on the encoding
                    '''
                    d_int = int(d.item())
                    dataset_name = self.idx_to_dataset_enc[d_int]
                    '''
                        Report on theindeces of the used datasets
                    '''
                    d_int = self.used_datasets_names.index(dataset_name)
                    ds_mask = ids_datasets_sorted == d

                    if ds_mask.any():
                        self.val_loss_per_dataset[d_int].update(loss_per_sample[ds_mask].mean().detach())
                        self.val_mu_pcc_per_dataset[d_int].update(mu_pcc_per_sample[ds_mask].mean().detach())
                        self.val_rho_pcc_per_dataset[d_int].update(rho_pcc_per_sample[ds_mask].mean().detach())
                        self.val_w_pcc_per_dataset[d_int].update(w_pcc_per_sample[ds_mask].mean().detach())
                        self.val_w_kl_per_dataset[d_int].update(w_kl[ds_mask].mean().detach())

                if (not self._val_plot_logged_this_epoch) and (batch_idx == 0):
                    exp = getattr(self.logger, "experiment", None) if self.logger is not None else None
                    log_plot_validation(
                        profiles_target.detach().cpu(),
                        mu_phys=mu_obs.detach().cpu(),
                        mu_total=mu_total.detach().cpu(),
                        pi=pi.detach().cpu(),
                        w_prob=w_prob.detach().cpu(),
                        sigma=sigma.detach().cpu(),
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

    def on_validation_epoch_end(self):
        # Pass the Metric OBJECT directly to self.log.
        # REMOVED: .compute(), sync_dist=True, and .reset()
        self.log("val_loss_epoch", self.val_loss_epoch, on_step=False, on_epoch=True)
        self.log("val_rho_pcc_epoch", self.rho_val_epoch, on_step=False, on_epoch=True)
        self.log("val_mu_pcc_epoch", self.mu_val_epoch, on_step=False, on_epoch=True)
        self.log("val_w_pcc_epoch", self.w_val_epoch, on_step=False, on_epoch=True)
        self.log("val_kl_w_epoch", self.kl_w_val_epoch, on_step=False, on_epoch=True)

        for d in range(self.num_used_datasets):
            name = self.used_datasets_names[d]

            # Pass the Metric OBJECT directly to self.log.
            self.log(f"val_loss_epoch/{name}", self.val_loss_per_dataset[d], on_step=False, on_epoch=True)
            self.log(f"val_mu_pcc_epoch/{name}", self.val_mu_pcc_per_dataset[d], on_step=False, on_epoch=True)
            self.log(f"val_rho_pcc_epoch/{name}", self.val_rho_pcc_per_dataset[d], on_step=False, on_epoch=True)
            self.log(f"val_w_pcc_epoch/{name}", self.val_w_pcc_per_dataset[d], on_step=False, on_epoch=True)
            self.log(f"val_w_kl_epoch/{name}", self.val_w_kl_per_dataset[d], on_step=False, on_epoch=True)

        self._val_plot_logged_this_epoch = False

    def on_train_epoch_end(self) -> None:
        # Pass the Metric OBJECT directly to self.log.
        # REMOVED: .compute(), sync_dist=True, and .reset()
        self.log("train_loss_epoch", self.train_loss_epoch, on_step=False, on_epoch=True)
        self.log("train_rho_pcc_epoch", self.rho_train_epoch, on_step=False, on_epoch=True)
        self.log("train_mu_pcc_epoch", self.mu_train_epoch, on_step=False, on_epoch=True)
        self.log("train_w_pcc_epoch", self.w_train_epoch, on_step=False, on_epoch=True)
        self.log("train_kl_w_epoch", self.kl_w_train_epoch, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # The *extra_items catches transcript_ids, lengths, or anything else the collate_fn yields
        id_datasets, ids, x_packed, profiles_target, lengths, mask, css, batch_embeddings = batch

        # Forward pass
        mu, pi, sigma, (rho, w_prob, J, transcript_scale_S, log_transcript_scale, total_scale, a, b,
                        log_sigma) = self.model(x_packed, id_datasets, profiles_target, batch_embeddings)

        # Extract lengths from packed sequence
        from torch.nn.utils.rnn import pad_packed_sequence
        _, lengths = pad_packed_sequence(x_packed, batch_first=True)

        # Return the complete physical state
        return {
            "ids": ids,
            "dataset_id": id_datasets.detach().cpu(),
            "lengths": lengths.detach().cpu(),
            "J": J.detach().cpu(),
            "w_prob": w_prob.detach().cpu(),
            "rho": rho.detach().cpu(),
            "mu": mu.detach().cpu(),
            "sigma": sigma.detach().cpu(),
            "b_offset": b.detach().cpu(),
            "pi": pi.detach().cpu(),
            "css": css
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay,
        )

        # Force the model to explore, then smoothly settle, completely ignoring val_loss spikes
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.trainer.max_epochs,
            eta_min=self.config.optim.scheduler.min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }