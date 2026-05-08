from __future__ import annotations

import hashlib
import os
import pathlib
from pathlib import Path
from typing import Any

import hydra
import lightning as pl
import numpy as np
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

    return {} if data is None else data


def get_datasets(cfg: DictConfig) -> list[str]:
    raw = cfg.experiment.dataset

    if isinstance(raw, str) and raw.lower() == "all":
        datasets = list(cfg.dataset_config.dataset_path.keys())

        OmegaConf.set_struct(cfg, False)
        cfg.experiment.dataset = datasets
        OmegaConf.set_struct(cfg, True)

        print(f"Command 'all' detected. Loading all {len(datasets)} datasets.")
        return datasets

    if isinstance(raw, str):
        return [raw]

    return list(raw)


def make_dataset_signature(datasets: list[str]) -> str:
    raw = "_".join(sorted(datasets))

    if len(raw) <= 100:
        return raw

    short_hash = hashlib.md5(raw.encode()).hexdigest()[:6]
    return f"{len(datasets)}_datasets_mix_{short_hash}"


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="config_riboai_queuing_multidataset",
)
def main(cfg: DictConfig) -> None:
    seed = int(cfg.experiment.seed)
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)

    datasets = get_datasets(cfg)
    split_size = float(cfg.experiment.split_size)

    datasets_paths = [
        cfg.dataset_config.dataset_path[dataset]
        for dataset in datasets
    ]

    train_fold, val_fold = conserved_stalling_sites_aware_split(
        cfg.paths.css_split,
        split_size=split_size,
        random_seed=seed,
    )

    dataset_str = make_dataset_signature(datasets)
    print(f"Tracking experiment under dataset signature: {dataset_str}")

    paths_logs = str(Path(cfg.paths.logs) / dataset_str)
    paths_results = Path(cfg.paths.results) / dataset_str
    paths_checkpoints = Path(cfg.paths.checkpoints) / dataset_str

    dataset_encoding = open_file(cfg.paths.encodings.datasets)

    torch_model = RiboQueuingModel(
        model_configs=cfg.model,
        eps=float(cfg.model.get("eps", 1e-8)),
        mu_max=float(cfg.model.get("mu_max", 1e8)),
    )

    lit_model = RiboQueuingModelLightningModule(
        torch_model,
        config=cfg,
        dataset_encoding=dataset_encoding,
    )

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
        balanced_train_sampling=bool(cfg.data.balanced_train_sampling),
        dataset_balance_gamma=float(cfg.data.dataset_balance_gamma),
        train_samples_per_epoch=cfg.data.train_samples_per_epoch,
    )

    tb_logger = TensorBoardLogger(
        save_dir=paths_logs,
        name="",
    )

    exp_name = Path(tb_logger.log_dir or tb_logger.save_dir).name
    ckpt_dir = Path(paths_checkpoints) / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    monitor = str(cfg.optim.scheduler.monitor)
    metric_mode = str(cfg.optim.scheduler.mode)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="{epoch}-{val_loss:.4f}",
        save_top_k=1,
        save_last=True,
        save_weights_only=True,
        monitor=monitor,
        mode=metric_mode,
    )

    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=int(cfg.callbacks.early_stopping_patience),
        mode=metric_mode,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        max_epochs=int(cfg.trainer.max_epochs),
        logger=tb_logger,
        log_every_n_steps=int(cfg.trainer.log_every_n_steps),
        callbacks=[
            checkpoint_callback,
            early_stopping,
            lr_monitor,
        ],
    )

    if bool(cfg.experiment.from_checkpoint):
        ckpt_path = find_checkpoint(str(paths_checkpoints), prefer="latest")

        if ckpt_path is None:
            raise FileNotFoundError(
                f"from_checkpoint=True but no checkpoint found under: {paths_checkpoints}"
            )

        trainer.fit(
            lit_model,
            datamodule=datamodule,
            ckpt_path=str(ckpt_path),
        )

    elif bool(cfg.experiment.train):
        trainer.fit(
            lit_model,
            datamodule=datamodule,
        )

    if bool(cfg.experiment.predict):
        out_dir = Path(paths_results)
        out_dir.mkdir(parents=True, exist_ok=True)

        ckpt_to_use = checkpoint_callback.best_model_path or None

        if not ckpt_to_use:
            ckpt_found = find_checkpoint(str(ckpt_dir), prefer="best")
            ckpt_to_use = str(ckpt_found) if ckpt_found is not None else None

        print(f"Prediction requested. Checkpoint selected: {ckpt_to_use}")


if __name__ == "__main__":
    main()