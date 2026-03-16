from __future__ import annotations

import os
import pathlib
from pathlib import Path

import hydra
import lightning as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig

from Dataloaders.RiboAIQueuing.RiboAIQueuingDatamodule import RiboAIQueuingDatamodule
from Models.RiboQueuingModel import RiboQueuingModel
from Models.RiboQueuingModelLighningModule import RiboQueuingModelLightningModule
from Utils.checkpoints import find_checkpoint
from Utils.prediction_io import flatten_predictions, plot_example_profile, save_predictions_parquet
from Utils.splits import conserved_stalling_sites_aware_split


@hydra.main(version_base=None, config_path="config", config_name="config_riboai_queuing")
def main(cfg: DictConfig):

    # -------------------------
    # seeds
    # -------------------------
    seed = cfg.experiment.seed
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)

    # -------------------------
    # dataset split
    # -------------------------
    dataset = cfg.experiment.dataset
    split_size = cfg.experiment.split_size

    dataset_path = cfg.dataset_config.dataset_path[dataset]
    train_fold, val_fold = conserved_stalling_sites_aware_split(dataset_path, split_size=split_size, random_seed=seed)

    # -------------------------
    # paths
    # -------------------------
    paths_ckpt = cfg.paths.checkpoints
    paths_logs = cfg.paths.logs
    paths_results = cfg.paths.results

    # -------------------------
    # model
    # -------------------------
    torch_model = RiboQueuingModel(
        input_size=cfg.model.input_size,
        hidden_size=cfg.model.hidden_dims,
        dropout=cfg.model.dropout,
        num_layers=cfg.model.num_layers,
        w_temperature=cfg.model.w_temperature,
        rho_eps=cfg.model.rho_eps,
        sigma_min=cfg.model.sigma_min,
        sigma_max=cfg.model.sigma_max,
    )

    lit_model = RiboQueuingModelLightningModule(torch_model, config=cfg)

    # optional restore
    from_ckpt = cfg.experiment.from_checkpoint
    if from_ckpt:
        ckpt_root = paths_ckpt
        ckpt_path = find_checkpoint(ckpt_root, prefer="latest")
        if ckpt_path is None:
            raise FileNotFoundError(f"from_checkpoint=True but no checkpoint found under: {ckpt_root}")

        lit_model = lit_model.__class__.load_from_checkpoint(
            checkpoint_path=str(ckpt_path),
            torch_model=torch_model,
            config=cfg,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
        )

    # -------------------------
    # datamodule
    # -------------------------
    datamodule = RiboAIQueuingDatamodule(
        dataset_path=dataset_path,
        batch_size=cfg.data.batch_size,
        split=[train_fold, val_fold],
        split_p=split_size,
        num_workers=cfg.data.num_workers,
        nt_encoding_path=cfg.paths.encodings.nt,
        codon_encoding_path=cfg.paths.encodings.codon,
        codon_to_aa_encoding_path=cfg.paths.encodings.codon_to_aa,
        aa_encoding_path=cfg.paths.encodings.aa,
    )

    # -------------------------
    # logger
    # -------------------------
    logger_dir = paths_logs
    tb_logger = TensorBoardLogger(save_dir=str(logger_dir), name="")

    exp_name = Path(tb_logger.log_dir or tb_logger.save_dir).name

    # -------------------------
    # callbacks
    # -------------------------
    ckpt_dir = os.path.join(paths_ckpt, exp_name)
    pathlib.Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    monitor = cfg.optim.scheduler.monitor

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="{epoch}-{val_loss_epoch:.4f}",
        save_top_k=1,
        mode="min",
        save_last=True,
        monitor=monitor,
        save_weights_only=True,
    )

    early_pat = cfg.callbacks.early_stopping_patience
    early_stopping = EarlyStopping(monitor=monitor, patience=early_pat, mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # -------------------------
    # trainer
    # -------------------------
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        max_epochs=cfg.trainer.max_epochs,
        logger=tb_logger,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        gradient_clip_algorithm=cfg.trainer.gradient_clip_algorithm,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
    )

    # -------------------------
    # train / predict
    # -------------------------
    do_train = cfg.experiment.train
    do_predict = cfg.experiment.predict

    if do_train:
        trainer.fit(lit_model, datamodule=datamodule)

    if do_predict:
        out_dir = paths_results
        out_dir.mkdir(parents=True, exist_ok=True)

        ckpt_to_use = checkpoint_callback.best_model_path or None
        if not ckpt_to_use:
            ckpt_found = find_checkpoint(ckpt_dir, prefer="best")
            ckpt_to_use = str(ckpt_found) if ckpt_found is not None else None

        preds = trainer.predict(lit_model, datamodule=datamodule, ckpt_path=ckpt_to_use)
        rows = flatten_predictions(preds)
        if len(rows) == 0:
            raise RuntimeError("trainer.predict returned no dict rows. Check predict_step output.")

        parquet_path = save_predictions_parquet(rows, out_dir / "results.parquet")

        df = pd.DataFrame(rows)
        example_idx = cfg.predict.example_idx
        fig_path = plot_example_profile(df, out_dir=out_dir, example_idx=example_idx)

        print("Saved:", parquet_path)
        print("Saved plot:", fig_path)


if __name__ == "__main__":
    main()
