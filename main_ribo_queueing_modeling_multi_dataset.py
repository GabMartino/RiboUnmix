from __future__ import annotations

import os
import pathlib
from pathlib import Path
import hashlib
from typing import Any

import hydra
import lightning as pl
import numpy as np
import pandas as pd
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from Dataloaders.RiboAIQueuingMultiDataset.RiboAIQueuingDatamoduleMultiDataset import (
    RiboAIQueuingDatamoduleMultiDataset,
)
from Models.RiboQueuingModel import RiboQueuingModel
from Models.RiboQueuingModelLighningModule import RiboQueuingModelLightningModule
from Utils.checkpoints import find_checkpoint
from Utils.splits import conserved_stalling_sites_aware_split


safe_globals = [np.dtype]

try:
    safe_globals.append(np._core.multiarray.scalar)
except AttributeError:
    safe_globals.append(np.core.multiarray.scalar)

try:
    safe_globals.append(np.dtypes.StrDType)
except AttributeError:
    pass

torch.serialization.add_safe_globals(safe_globals)

def open_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML file to contain a dictionary, got {type(data).__name__}: {path}")

    return data

@hydra.main(
    version_base=None,
    config_path="config",
    config_name="config_riboai_queuing_multidataset",
)
def main(cfg: DictConfig):
    # -------------------------
    # Seeds
    # -------------------------
    seed = int(cfg.experiment.seed)
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)

    def cfg_get(path: str, default=None):
        value = OmegaConf.select(cfg, path)
        return default if value is None else value

    # -------------------------
    # Dataset split & "all" logic
    # -------------------------
    raw_datasets_cfg = cfg.experiment.dataset

    if isinstance(raw_datasets_cfg, str) and raw_datasets_cfg.lower() == "all":
        datasets = list(cfg.dataset_config.dataset_path.keys())
        print(f"Command 'all' detected. Loading all {len(datasets)} datasets from config.")

        OmegaConf.set_struct(cfg, False)
        cfg.experiment.dataset = datasets
        OmegaConf.set_struct(cfg, True)

    elif isinstance(raw_datasets_cfg, str):
        datasets = [raw_datasets_cfg]
    else:
        datasets = list(raw_datasets_cfg)

    split_size = float(cfg.experiment.split_size)

    datasets_paths = []
    for dataset in datasets:
        if dataset not in cfg.dataset_config.dataset_path:
            raise KeyError(f"Dataset '{dataset}' not found in dataset_config.dataset_path.")
        datasets_paths.append(cfg.dataset_config.dataset_path[dataset])

    train_fold, val_fold = conserved_stalling_sites_aware_split(
        cfg.paths.css_split,
        split_size=split_size,
        random_seed=seed,
    )

    # -------------------------
    # OS-safe folder naming
    # -------------------------
    sorted_datasets = sorted(datasets)
    raw_dataset_str = "_".join(sorted_datasets)

    if len(raw_dataset_str) > 100:
        short_hash = hashlib.md5(raw_dataset_str.encode()).hexdigest()[:6]
        dataset_str = f"{len(datasets)}_datasets_mix_{short_hash}"
    else:
        dataset_str = raw_dataset_str

    print(f"Tracking experiment under dataset signature: {dataset_str}")

    # -------------------------
    # Paths
    # -------------------------
    paths_logs = str(Path(cfg.paths.logs) / dataset_str)
    paths_results = Path(cfg.paths.results) / dataset_str
    paths_checkpoints = Path(cfg.paths.checkpoints) / dataset_str

    # -------------------------
    # Model
    # -------------------------
    dataset_encoding = open_file(cfg.paths.encodings.datasets)
    num_datasets_from_encoding = max(int(v) for v in dataset_encoding.values()) + 1
    num_datasets = max(int(cfg.model.num_datasets), num_datasets_from_encoding)

    torch_model = RiboQueuingModel(
        input_size=int(cfg.model.input_size),
        hidden_size=int(cfg.model.hidden_dims),
        num_layers=int(cfg.model.num_layers),
        dropout=float(cfg.model.dropout),
        num_datasets=int(cfg.model.num_datasets),

        eps=float(cfg_get("model.eps", 1e-8)),
        mu_max=float(cfg_get("model.mu_max", 1e8)),

        codon_feature_start=int(cfg.model.codon_feature_start),
        num_codons=int(cfg.model.num_codons),

        dataset_emb_dim=int(cfg.model.dataset_emb_dim),
        codon_emb_dim=int(cfg.model.codon_emb_dim),
        bias_hidden_dim=int(cfg.model.bias_hidden_dim),
        b_clip=float(cfg.model.b_clip),

        additive_dataset_emb_dim=int(cfg_get("model.additive_dataset_emb_dim", 16)),
        additive_codon_emb_dim=int(cfg_get("model.additive_codon_emb_dim", 8)),
        additive_hidden_dim=int(cfg_get("model.additive_hidden_dim", 32)),
        additive_init_bias=float(cfg_get("model.additive_init_bias", -8.0)),

        phi_min=float(cfg_get("model.phi_min", cfg_get("loss.phi_min", 0.05))),
        phi_max=float(cfg_get("model.phi_max", cfg_get("loss.phi_max", 5.0))),
        init_phi=float(cfg_get("model.init_phi", 1.0)),
        phi_dataset_emb_dim=int(cfg_get("model.phi_dataset_emb_dim", 16)),
        phi_codon_emb_dim=int(cfg_get("model.phi_codon_emb_dim", 8)),
        phi_hidden_dim=int(cfg_get("model.phi_hidden_dim", 32)),
    )

    lit_model = RiboQueuingModelLightningModule(torch_model,
                                                config=cfg,
                                                dataset_encoding = open_file(cfg.paths.encodings.datasets))

    # -------------------------
    # Optional restore
    # -------------------------
    from_ckpt = bool(cfg.experiment.from_checkpoint)

    if from_ckpt:
        ckpt_path = find_checkpoint(str(paths_checkpoints), prefer="latest")
        if ckpt_path is None:
            raise FileNotFoundError(
                f"from_checkpoint=True but no checkpoint found under: {paths_checkpoints}"
            )

        lit_model = lit_model.__class__.load_from_checkpoint(
            checkpoint_path=str(ckpt_path),
            torch_model=torch_model,
            config=cfg,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
        )

    # -------------------------
    # Datamodule
    # -------------------------
    datamodule = RiboAIQueuingDatamoduleMultiDataset(
        sequences_path=cfg.paths.sequences_path,
        datasets_paths=datasets_paths,
        batch_size=int(cfg.data.batch_size),
        split=(train_fold, val_fold),
        split_p=split_size,
        num_workers=int(cfg.data.num_workers),
        seed=seed,
        nt_encoding_path=cfg.paths.encodings.nt,
        codon_encoding_path=cfg.paths.encodings.codon,
        codon_to_aa_encoding_path=cfg.paths.encodings.codon_to_aa,
        aa_encoding_path=cfg.paths.encodings.aa,
        datasets_encoding_path=cfg.paths.encodings.datasets,
        balanced_train_sampling=bool(cfg_get("data.balanced_train_sampling", False)),
        dataset_balance_gamma=float(cfg_get("data.dataset_balance_gamma", 1.0)),
        train_samples_per_epoch=cfg_get("data.train_samples_per_epoch", None),
    )

    # -------------------------
    # Logger
    # -------------------------
    logger_dir = paths_logs
    tb_logger = TensorBoardLogger(save_dir=logger_dir, name="")

    exp_name = Path(tb_logger.log_dir or tb_logger.save_dir).name

    # -------------------------
    # Callbacks
    # -------------------------
    ckpt_dir = os.path.join(str(paths_checkpoints), exp_name)
    pathlib.Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    monitor = str(cfg.optim.scheduler.monitor)
    metric_mode = str(cfg_get("optim.scheduler.mode", "max" if "pcc" in monitor else "min"))

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="{epoch}-{val_loss_epoch:.4f}",
        save_top_k=1,
        mode=metric_mode,
        save_last=True,
        monitor=monitor,
        save_weights_only=True,
    )

    early_pat = int(cfg.callbacks.early_stopping_patience)
    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=early_pat,
        mode=metric_mode,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # -------------------------
    # Trainer
    # -------------------------
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=int(cfg.trainer.accumulate_grad_batches),
        max_epochs=int(cfg.trainer.max_epochs),
        logger=tb_logger,
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        gradient_clip_val=float(cfg.trainer.gradient_clip_val),
        gradient_clip_algorithm=cfg.trainer.gradient_clip_algorithm,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    # -------------------------
    # Train / predict
    # -------------------------
    do_train = bool(cfg.experiment.train)
    do_predict = bool(cfg.experiment.predict)

    if do_train:
        trainer.fit(lit_model, datamodule=datamodule)

    if do_predict:
        out_dir = Path(paths_results)
        out_dir.mkdir(parents=True, exist_ok=True)

        ckpt_to_use = checkpoint_callback.best_model_path or None

        if not ckpt_to_use:
            ckpt_found = find_checkpoint(str(ckpt_dir), prefer="best")
            ckpt_to_use = str(ckpt_found) if ckpt_found is not None else None

        current_rank = trainer.global_rank

        print(f"[Rank {current_rank}] Running comprehensive physical inference extraction...")
        preds = trainer.predict(lit_model, datamodule=datamodule, ckpt_path=ckpt_to_use)

        if not preds:
            print(f"[Rank {current_rank}] No predictions to process. Exiting safely.")
            return

        print(f"[Rank {current_rank}] Slicing padding and compiling master Parquet database...")

        master_rows = []

        def _to_numpy(x):
            if x is None:
                return None
            if torch.is_tensor(x):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        def _to_python_scalar(x):
            if torch.is_tensor(x):
                if x.ndim == 0:
                    return x.item()
                return x.detach().cpu().tolist()
            if isinstance(x, np.generic):
                return x.item()
            return x

        def _normalize_css(css_i, L: int):
            css_arr = _to_numpy(css_i).reshape(-1)

            if np.issubdtype(css_arr.dtype, np.number):
                css_arr = css_arr[np.isfinite(css_arr)]
                css_arr = css_arr.astype(np.int64, copy=False)
                css_arr = css_arr[(css_arr >= 0) & (css_arr < L)]

            return css_arr

        def _batch_get(batch: dict, key: str, fallback_key: str | None = None, default=None):
            if key in batch and batch[key] is not None:
                return batch[key]
            if fallback_key is not None and fallback_key in batch and batch[fallback_key] is not None:
                return batch[fallback_key]
            return default

        def _slice_array(arr, i: int, L: int, dtype=np.float32):
            if arr is None:
                return None
            return arr[i, :L].astype(dtype, copy=False)

        def _slice_optional(row_data: dict, name: str, arr, i: int, L: int, dtype=np.float32):
            if arr is not None:
                row_data[name] = _slice_array(arr, i, L, dtype=dtype)

        def _scalar_optional(row_data: dict, name: str, arr, i: int):
            if arr is not None:
                row_data[name] = float(np.asarray(arr[i]).reshape(-1)[0])

        for batch in preds:
            lengths = _to_numpy(batch["lengths"]).astype(np.int64, copy=False)
            transcript_ids = batch["ids"]
            dataset_ids = _to_numpy(batch["dataset_id"]).astype(np.int64, copy=False)

            J_vals = _to_numpy(batch["J"])
            w_probs = _to_numpy(batch["w_prob"])
            rhos = _to_numpy(batch["rho"])

            L_queue = _to_numpy(batch["L_queue"])
            L_effective = _to_numpy(_batch_get(batch, "L_effective", default=None))

            S_mean = _to_numpy(batch["S_mean"])
            total_scale = _to_numpy(batch["total_scale"])

            mu_obs = _to_numpy(batch["mu_obs"])
            mu_total = _to_numpy(_batch_get(batch, "mu_total", default=batch["mu_obs"]))

            # Hurdle-Gamma naming.
            phi = _to_numpy(_batch_get(batch, "phi", fallback_key="alpha", default=None))
            log_phi = _to_numpy(_batch_get(batch, "log_phi", fallback_key="log_sigma", default=None))

            b_offset = _to_numpy(batch["b_offset"])
            pi = _to_numpy(batch["pi"])
            masks = _to_numpy(batch["mask"])
            y_vals = _to_numpy(batch["y"])

            # Optional decomposition diagnostics from new model.
            shift_weights = _to_numpy(_batch_get(batch, "shift_weights", default=None))
            additive_bg = _to_numpy(_batch_get(batch, "additive_bg", default=None))
            bg_fraction = _to_numpy(_batch_get(batch, "bg_fraction", default=None))
            bg_q = _to_numpy(_batch_get(batch, "bg_q", default=None))
            p_bio = _to_numpy(_batch_get(batch, "p_bio", default=None))
            mu_bio = _to_numpy(_batch_get(batch, "mu_bio", default=None))
            M_y = _to_numpy(_batch_get(batch, "M_y", default=None))

            css_batch = batch["css"]

            batch_size = len(lengths)

            for i in range(batch_size):
                L = int(lengths[i])
                transcript_id = _to_python_scalar(transcript_ids[i])
                css_i = _normalize_css(css_batch[i], L)

                row_data = {
                    "dataset_id": int(dataset_ids[i]),
                    "transcript_id": transcript_id,
                    "length": L,

                    "J": float(np.asarray(J_vals[i]).reshape(-1)[0]),

                    "w_prob": w_probs[i, :L].astype(np.float32, copy=False),
                    "rho": rhos[i, :L].astype(np.float32, copy=False),

                    "L_queue": L_queue[i, :L].astype(np.float32, copy=False),
                    "S_mean": (
                        S_mean[i].astype(np.float32, copy=False)
                        if np.ndim(S_mean[i]) > 0
                        else np.float32(S_mean[i])
                    ),
                    "total_scale": total_scale[i, :L].astype(np.float32, copy=False),

                    "mu_obs": mu_obs[i, :L].astype(np.float32, copy=False),
                    "mu_total": mu_total[i, :L].astype(np.float32, copy=False),

                    # New Hurdle-Gamma names.
                    "phi": phi[i, :L].astype(np.float32, copy=False) if phi is not None else None,
                    "log_phi": (
                        log_phi[i, :L].astype(np.float32, copy=False)
                        if log_phi is not None
                        else None
                    ),

                    # Backward-compatible aliases for old analysis scripts.
                    "alpha": phi[i, :L].astype(np.float32, copy=False) if phi is not None else None,
                    "sigma": phi[i, :L].astype(np.float32, copy=False) if phi is not None else None,
                    "log_sigma": (
                        log_phi[i, :L].astype(np.float32, copy=False)
                        if log_phi is not None
                        else None
                    ),

                    "b_offset": b_offset[i, :L].astype(np.float32, copy=False),
                    "pi": pi[i, :L].astype(np.float32, copy=False),
                    "mask": masks[i, :L].astype(np.bool_, copy=False),
                    "css": css_i,
                    "target": y_vals[i, :L].astype(np.float32, copy=False),
                }

                _slice_optional(row_data, "L_effective", L_effective, i, L)
                _slice_optional(row_data, "additive_bg", additive_bg, i, L)
                _slice_optional(row_data, "bg_q", bg_q, i, L)
                _slice_optional(row_data, "p_bio", p_bio, i, L)
                _slice_optional(row_data, "mu_bio", mu_bio, i, L)

                _scalar_optional(row_data, "bg_fraction", bg_fraction, i)
                _scalar_optional(row_data, "M_y", M_y, i)

                if shift_weights is not None:
                    row_data["shift_weights"] = np.asarray(
                        shift_weights[i],
                        dtype=np.float32,
                    )

                master_rows.append(row_data)

        print(f"[Rank {current_rank}] Aggregated {len(master_rows)} dataset-transcript interactions.")

        df_results = pd.DataFrame(master_rows)

        parquet_path = out_dir / f"comprehensive_predictions_rank{current_rank}.parquet"
        df_results.to_parquet(parquet_path, engine="pyarrow")

        print("==================================================")
        print(f"[Rank {current_rank}] Full Physical State Saved: {parquet_path}")
        print("==================================================")

        trainer.strategy.barrier()


if __name__ == "__main__":
    main()