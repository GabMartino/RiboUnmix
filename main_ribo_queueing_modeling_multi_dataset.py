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

from Dataloaders.RiboAIQueuingMultiDataset.RiboAIQueuingDatamoduleMultiDataset import \
    RiboAIQueuingDatamoduleMultiDataset
from Models.RiboQueuingModel import RiboQueuingModel
from Models.RiboQueuingModelLighningModule2 import RiboQueuingModelLightningModule
from Utils.checkpoints import find_checkpoint
from Utils.prediction_io import flatten_predictions, plot_example_profile, save_predictions_parquet
from Utils.splits import conserved_stalling_sites_aware_split


@hydra.main(version_base=None, config_path="config", config_name="config_riboai_queuing_multidataset")
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
    datasets = cfg.experiment.dataset
    split_size = cfg.experiment.split_size
    datasets_paths = []
    for dataset in datasets:
        dataset_path = cfg.dataset_config.dataset_path[dataset]
        datasets_paths.append(dataset_path)
        train_fold, val_fold = conserved_stalling_sites_aware_split(cfg.paths.css_split, split_size=split_size, random_seed=seed)
    # -------------------------
    # paths
    # -------------------------
    paths_logs = cfg.paths.logs
    paths_results = cfg.paths.results

    # -------------------------
    # model
    # -------------------------
    torch_model = RiboQueuingModel(
        input_size=cfg.model.input_size,
        hidden_size=cfg.model.hidden_dims,
        dropout=cfg.model.dropout,
        num_datasets=len(cfg.experiment.dataset),
        num_layers=cfg.model.num_layers,
        w_temperature=cfg.model.w_temperature,
        rho_eps=cfg.model.rho_eps,
    )

    lit_model = RiboQueuingModelLightningModule(torch_model, config=cfg)

    # optional restore
    from_ckpt = cfg.experiment.from_checkpoint
    if from_ckpt:
        ckpt_path = find_checkpoint(cfg.paths.checkpoints, prefer="latest")
        if ckpt_path is None:
            raise FileNotFoundError(f"from_checkpoint=True but no checkpoint found under: {cfg.paths.checkpoints}")

        lit_model = lit_model.__class__.load_from_checkpoint(
            checkpoint_path=str(ckpt_path),
            torch_model=torch_model,
            config=cfg,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
        )

    # -------------------------
    # datamodule
    # -------------------------
    datamodule = RiboAIQueuingDatamoduleMultiDataset(
        sequences_path=cfg.paths.sequences_path,
        datasets_paths=datasets_paths,
        batch_size=cfg.data.batch_size,
        split=(train_fold, val_fold),
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
    ckpt_dir = os.path.join(cfg.paths.checkpoints, exp_name)
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
        #gradient_clip_val=cfg.trainer.gradient_clip_val,
        #gradient_clip_algorithm=cfg.trainer.gradient_clip_algorithm,
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
        import matplotlib.pyplot as plt
        import numpy as np

        out_dir = paths_results
        out_dir.mkdir(parents=True, exist_ok=True)

        ckpt_to_use = checkpoint_callback.best_model_path or None
        if not ckpt_to_use:
            ckpt_found = find_checkpoint(ckpt_dir, prefer="best")
            ckpt_to_use = str(ckpt_found) if ckpt_found is not None else None

        # 1. Run the targeted inference
        print("Running biological inference extraction...")
        preds = trainer.predict(lit_model, datamodule=datamodule, ckpt_path=ckpt_to_use)

        if len(preds) == 0:
            raise RuntimeError("trainer.predict returned empty results.")

        # 2. Extract and Align the Start Codons
        window_size = 50
        start_codon_profiles = []

        print("Aligning transcripts at the Start Codon...")
        for batch in preds:
            w_probs = batch["w_prob"]  # [B, T]
            lengths = batch["lengths"]  # [B]

            for i in range(len(lengths)):
                L = int(lengths[i].item())
                # Only use transcripts long enough to fit the window
                if L >= window_size:
                    # Slice the first 50 codons (un-padded pure biology)
                    w_start = w_probs[i, :window_size].numpy()
                    start_codon_profiles.append(w_start)

        # 3. Calculate the Global Biological Traffic Jam
        start_codon_matrix = np.stack(start_codon_profiles)
        # We use median to prevent a few crazy outliers from skewing the biological consensus
        metagene_profile = np.median(start_codon_matrix, axis=0)

        # 4. Generate the Proof
        print(f"Aggregated {len(start_codon_profiles)} transcripts. Generating Metagene plot...")
        plt.figure(figsize=(12, 5))
        plt.plot(metagene_profile, color='indigo', linewidth=2.5)

        # Highlight the Start Codon (Index 0)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Start Codon')

        plt.title("w_prob Metagene Alignment (Pure Elongation Velocity)", fontsize=14)
        plt.xlabel("Codon Position (0 = Start Codon)", fontsize=12)
        plt.ylabel("Median w_prob (Predicted Dwell Time)", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        metagene_path = out_dir / "metagene_start_codon_proof.png"
        plt.savefig(metagene_path, dpi=300)
        plt.close()

        print("==================================================")
        print(f"Biological Validation Saved: {metagene_path}")
        print("==================================================")


if __name__ == "__main__":
    main()
