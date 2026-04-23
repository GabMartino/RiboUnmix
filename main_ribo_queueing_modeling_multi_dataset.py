from __future__ import annotations

import os
import pathlib
from pathlib import Path
import hashlib  # Added for OS-safe hashing

import hydra
import lightning as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from Dataloaders.RiboAIQueuingMultiDataset.RiboAIQueuingDatamoduleMultiDataset import \
    RiboAIQueuingDatamoduleMultiDataset
from Models.RiboQueuingModel import RiboQueuingModel
from Models.RiboQueuingModelLighningModule import RiboQueuingModelLightningModule
from Utils.checkpoints import find_checkpoint
from Utils.prediction_io import flatten_predictions, plot_example_profile, save_predictions_parquet
from Utils.splits import conserved_stalling_sites_aware_split

safe_globals = [np.dtype]

# Safely add multiarray scalar
try:
    safe_globals.append(np._core.multiarray.scalar)
except AttributeError:
    safe_globals.append(np.core.multiarray.scalar)
# Safely add String DType
try:
    safe_globals.append(np.dtypes.StrDType)
except AttributeError:
    pass

# Apply the whitelista
torch.serialization.add_safe_globals(safe_globals)


@hydra.main(version_base=None, config_path="config", config_name="config_riboai_queuing_multidataset")
def main(cfg: DictConfig):
    # -------------------------
    # seeds
    # -------------------------
    seed = cfg.experiment.seed
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)

    # -------------------------
    # dataset split & "all" logic
    # -------------------------
    raw_datasets_cfg = cfg.experiment.dataset

    # --- THE NEW "ALL" LOGIC ---
    if isinstance(raw_datasets_cfg, str) and raw_datasets_cfg.lower() == "all":
        # Extract all dataset names dynamically from the dataset_config dictionary
        datasets = list(cfg.dataset_config.dataset_path.keys())
        print(f"Command 'all' detected. Loading all {len(datasets)} datasets from config.")
    elif isinstance(raw_datasets_cfg, str):
        # Catch case where a single dataset was provided as a string instead of a list
        datasets = [raw_datasets_cfg]
    else:
        # It's already a list
        datasets = list(raw_datasets_cfg)

    split_size = cfg.experiment.split_size
    datasets_paths = []
    for dataset in datasets:
        dataset_path = cfg.dataset_config.dataset_path[dataset]
        datasets_paths.append(dataset_path)

    train_fold, val_fold = conserved_stalling_sites_aware_split(cfg.paths.css_split, split_size=split_size,
                                                                random_seed=seed)

    # --- OS-SAFE FOLDER NAMING LOGIC ---
    sorted_datasets = sorted(datasets)
    raw_dataset_str = "_".join(sorted_datasets)

    if len(raw_dataset_str) > 100:
        short_hash = hashlib.md5(raw_dataset_str.encode()).hexdigest()[:6]
        dataset_str = f"{len(datasets)}_datasets_mix_{short_hash}"
    else:
        dataset_str = raw_dataset_str

    print(f"Tracking experiment under dataset signature: {dataset_str}")

    # -------------------------
    # paths (Dynamically routed by dataset_str)
    # -------------------------
    paths_logs = str(Path(cfg.paths.logs) / dataset_str)
    paths_results = Path(cfg.paths.results) / dataset_str
    paths_checkpoints = Path(cfg.paths.checkpoints) / dataset_str

    # -------------------------
    # model
    # -------------------------
    torch_model = RiboQueuingModel(
        input_size=cfg.model.input_size,
        hidden_size=cfg.model.hidden_dims,
        dropout=cfg.model.dropout,
        num_datasets=cfg.model.num_datasets,
        num_layers=cfg.model.num_layers,
        w_temperature=cfg.model.w_temperature,
        rho_eps=cfg.model.rho_eps,
        log_sigma_min=cfg.model.log_sigma_min,
        log_sigma_max=cfg.model.log_sigma_max,
        S_quantile=cfg.metrics.S_quantile,
        S_eps=cfg.metrics.S_eps,
        S_use_nonzero_only=cfg.metrics.S_use_nonzero_only,
        S_use_censor_threshold=cfg.metrics.S_use_censor_threshold,
        censor_threshold=cfg.metrics.censor_threshold,
    )

    if raw_datasets_cfg == "all":
        datasets = list(cfg.dataset_config.dataset_path.keys())
        OmegaConf.set_struct(cfg, False)
        cfg.experiment.dataset = datasets
        OmegaConf.set_struct(cfg, True)
    lit_model = RiboQueuingModelLightningModule(torch_model, config=cfg)

    # optional restore
    from_ckpt = cfg.experiment.from_checkpoint
    if from_ckpt:
        ckpt_path = find_checkpoint(str(paths_checkpoints), prefer="latest")
        if ckpt_path is None:
            raise FileNotFoundError(f"from_checkpoint=True but no checkpoint found under: {paths_checkpoints}")

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
        datasets_encoding_path=cfg.paths.encodings.datasets
    )

    # -------------------------
    # logger
    # -------------------------
    logger_dir = paths_logs
    tb_logger = TensorBoardLogger(save_dir=logger_dir, name="")

    exp_name = Path(tb_logger.log_dir or tb_logger.save_dir).name

    # -------------------------
    # callbacks
    # -------------------------
    ckpt_dir = os.path.join(str(paths_checkpoints), exp_name)
    pathlib.Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    monitor = cfg.optim.scheduler.monitor
    metric_mode = "max" if "pcc" in monitor else "min"

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="{epoch}-{val_loss_epoch:.4f}",
        save_top_k=1,
        mode=metric_mode,
        save_last=True,
        monitor=monitor,
        save_weights_only=True,
    )

    early_pat = cfg.callbacks.early_stopping_patience
    early_stopping = EarlyStopping(monitor=monitor, patience=early_pat, mode=metric_mode)
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
        import numpy as np
        import pandas as pd

        out_dir = Path(paths_results)
        out_dir.mkdir(parents=True, exist_ok=True)

        ckpt_to_use = checkpoint_callback.best_model_path or None
        if not ckpt_to_use:
            ckpt_found = find_checkpoint(str(ckpt_dir), prefer="best")
            ckpt_to_use = str(ckpt_found) if ckpt_found is not None else None

        # Fetch the global rank of the current GPU
        current_rank = trainer.global_rank

        print(f"[Rank {current_rank}] Running comprehensive physical inference extraction...")
        preds = trainer.predict(lit_model, datamodule=datamodule, ckpt_path=ckpt_to_use)

        if not preds or len(preds) == 0:
            print(f"[Rank {current_rank}] No predictions to process. Exiting safely.")
            return

        print(f"[Rank {current_rank}] Slicing padding and compiling master Parquet database...")
        master_rows = []

        for batch in preds:
            lengths = batch["lengths"].numpy()
            transcripts_ids = batch["ids"]
            dataset_ids = batch["dataset_id"].numpy()
            J_vals = batch["J"].numpy()

            w_probs = batch["w_prob"].numpy()
            rhos = batch["rho"].numpy()
            mus = batch["mu"].numpy()
            sigmas = batch["sigma"].numpy()
            b_offsets = batch["b_offset"].numpy()
            pis = batch["pi"].numpy()

            css_batch = batch["css"]
            y_vals = batch["y"].numpy()
            total_scales = batch["total_scale"].numpy()

            for i in range(len(lengths)):
                L = int(lengths[i])

                css_i = css_batch[i]
                if torch.is_tensor(css_i):
                    css_i = css_i.numpy()
                else:
                    css_i = np.array(css_i)

                row_data = {
                    "dataset_id": int(dataset_ids[i]),
                    "length": L,
                    "transcripts_id": transcripts_ids[i],
                    "J": float(J_vals[i].item()),
                    "w_prob": w_probs[i, :L].astype(np.float32),
                    "rho": rhos[i, :L].astype(np.float32),
                    "mu": mus[i, :L].astype(np.float32),
                    "sigma": sigmas[i, :L].astype(np.float32),
                    "b_offset": b_offsets[i, :L].astype(np.float32),
                    "pi": pis[i, :L].astype(np.float32),
                    "css": css_i[:L] if len(css_i) >= L else css_i,
                    "target": y_vals[i, :L].astype(np.float32),
                    "total_scale": total_scales[i, :L].astype(np.float32)
                }
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