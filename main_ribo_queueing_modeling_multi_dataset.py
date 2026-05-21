from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import hydra
import lightning as pl
import numpy as np
import pandas as pd
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.base import ContainerMetadata

try:
    from omegaconf.nodes import AnyNode
except ImportError:
    AnyNode = None

from Dataloaders.RiboAIQueuingMultiDataset.RiboAIQueuingDatamoduleMultiDataset import (
    RiboAIQueuingDatamoduleMultiDataset,
)
from Models.RiboQueuingModel import RiboQueuingModel
from Models.RiboQueuingModelLighningModule import RiboQueuingModelLightningModule
from Utils.checkpoints import find_checkpoint
from Utils.splits import conserved_stalling_sites_aware_split


safe_globals = [
    np.dtype,
    DictConfig,
    ListConfig,
    ContainerMetadata,
    Any,
]

if AnyNode is not None:
    safe_globals.append(AnyNode)

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


def make_run_tag(cfg: DictConfig) -> str:
    return "PCGrad" if bool(cfg.optim.use_pcgrad) else "NOPCGrad"


def load_weights_only(
    lit_model: pl.LightningModule,
    ckpt_path: str | Path,
) -> None:
    """
    Loads only model weights.

    This is correct for checkpoints saved with:
        save_weights_only=True

    It intentionally does not restore optimizer/scheduler state.
    """
    ckpt_path = Path(ckpt_path)

    ckpt = torch.load(
        str(ckpt_path),
        map_location="cpu",
        weights_only=False,
    )

    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    incompatible = lit_model.load_state_dict(
        state_dict,
        strict=False,
    )

    print(f"Loaded weights from: {ckpt_path}")

    if len(incompatible.missing_keys) > 0:
        print(f"Missing keys: {len(incompatible.missing_keys)}")

    if len(incompatible.unexpected_keys) > 0:
        print(f"Unexpected keys: {len(incompatible.unexpected_keys)}")


def choose_checkpoint(
    *,
    checkpoint_callback: ModelCheckpoint,
    ckpt_dir: Path,
    run_checkpoint_root: Path,
    prefer: str = "best",
) -> str | None:
    if checkpoint_callback.best_model_path:
        return checkpoint_callback.best_model_path

    ckpt_found = find_checkpoint(str(ckpt_dir), prefer=prefer)

    if ckpt_found is not None:
        return str(ckpt_found)

    ckpt_found = find_checkpoint(str(run_checkpoint_root), prefer=prefer)

    if ckpt_found is not None:
        return str(ckpt_found)

    return None


def predictions_to_parquet(
    *,
    predictions: list[dict[str, Any]],
    out_file: Path,
) -> None:
    rows = []

    sequence_keys = {
        "y",
        "target",
        "mu",
        "mu_obs",
        "mu_total",
        "mu_base",
        "L_queue",
        "L_effective",
        "additive_bias",
        "additive_bg",
        "beta_per_position",
        "phi",
        "rho",
        "rho_diag",
        "w_prob",
        "exp_b",
        "b",
        "mask",
        "codon_ids",
    }

    def to_numpy(x):
        if x is None:
            return None

        if torch.is_tensor(x):
            return x.detach().cpu().numpy()

        return x

    def to_python_list(x):
        x = to_numpy(x)

        if x is None:
            return None

        if isinstance(x, np.ndarray):
            return x.tolist()

        if torch.is_tensor(x):
            return x.detach().cpu().tolist()

        return x

    def get_batch_item(x, i: int):
        x = to_numpy(x)

        if x is None:
            return None

        if isinstance(x, (list, tuple)):
            return x[i]

        arr = np.asarray(x)

        if arr.ndim == 0:
            return arr.item()

        return arr[i]

    def slice_sequence(x, key: str, i: int, valid_len: int):
        x = to_numpy(x)

        if x is None:
            return None

        arr = np.asarray(x)

        if arr.ndim == 0:
            return arr.item()

        if arr.ndim == 1:
            item = arr[i]
            return item.item() if np.ndim(item) == 0 else np.asarray(item).tolist()

        sliced = arr[i, :valid_len]

        if key == "mask":
            return sliced.astype(np.bool_, copy=False).tolist()

        if key == "codon_ids":
            return sliced.astype(np.int64, copy=False).tolist()

        return sliced.astype(np.float32, copy=False).tolist()

    for batch in predictions:
        batch_size = len(batch["ids"])

        for i in range(batch_size):
            transcript_id = str(batch["ids"][i])
            dataset_id = int(get_batch_item(batch["dataset_id"], i))
            valid_len = int(get_batch_item(batch["lengths"], i))

            row = {
                "transcript_id": transcript_id,
                "dataset_id": dataset_id,
                "length": valid_len,
                "css": to_python_list(batch["css"][i]) if "css" in batch else None,
            }

            for key, val in batch.items():
                if key in {"ids", "dataset_id", "lengths", "css"}:
                    continue

                val_np = to_numpy(val)

                if val_np is None:
                    continue

                if key in sequence_keys:
                    row[key] = slice_sequence(val_np, key, i, valid_len)
                    continue

                arr = np.asarray(val_np)

                if arr.ndim == 0:
                    row[key] = arr.item()

                elif arr.ndim == 1:
                    if arr.shape[0] == batch_size:
                        item = arr[i]
                        row[key] = item.item() if np.ndim(item) == 0 else np.asarray(item).tolist()
                    else:
                        row[key] = arr.astype(np.float32, copy=False).tolist()

                else:
                    item = np.asarray(arr[i])

                    if item.size == 1:
                        row[key] = item.reshape(-1)[0].item()
                    else:
                        row[key] = item.astype(np.float32, copy=False).tolist()

            rows.append(row)

    df_predictions = pd.DataFrame(rows)

    print(f"Saving {len(df_predictions)} biological profiles to {out_file}...")

    df_predictions.to_parquet(
        out_file,
        engine="pyarrow",
        index=False,
    )


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
    run_tag = make_run_tag(cfg)

    print(f"Tracking dataset signature: {dataset_str}")
    print(f"Run tag: {run_tag}")

    # Final layout:
    #
    # logs/riboai_queueing/<dataset_str>/<run_tag>/version_x
    # checkpoints/riboai_queueing/<dataset_str>/<run_tag>/version_x
    # results/riboai_queueing/<dataset_str>/<run_tag>/predictions_<dataset_str>.parquet
    paths_logs = Path(cfg.paths.logs) / dataset_str / run_tag
    paths_results = Path(cfg.paths.results) / dataset_str / run_tag
    paths_checkpoints = Path(cfg.paths.checkpoints) / dataset_str / run_tag

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
        dataset_aware_batching=bool(cfg.data.dataset_aware_batching),
        datasets_per_batch=int(cfg.data.datasets_per_batch),
    )

    tb_logger = TensorBoardLogger(
        save_dir=str(paths_logs),
        name="",
    )

    exp_name = Path(tb_logger.log_dir or tb_logger.save_dir).name

    ckpt_dir = paths_checkpoints / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    monitor = str(cfg.optim.scheduler.monitor)
    metric_mode = str(cfg.optim.scheduler.mode)

    filename = "{epoch}-{" + monitor + ":.4f}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=filename,
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

    trainer_kwargs = {
        "accelerator": cfg.trainer.accelerator,
        "devices": cfg.trainer.devices,
        "precision": cfg.trainer.precision,
        "max_epochs": int(cfg.trainer.max_epochs),
        "logger": tb_logger,
        "log_every_n_steps": int(cfg.trainer.log_every_n_steps),
        "callbacks": [
            checkpoint_callback,
            early_stopping,
            lr_monitor,
        ],
    }

    # Automatic optimization can use Lightning clipping.
    # PCGrad/manual optimization should clip inside the LightningModule.
    if not bool(cfg.optim.use_pcgrad):
        trainer_kwargs["gradient_clip_val"] = float(cfg.trainer.gradient_clip_val)
        trainer_kwargs["gradient_clip_algorithm"] = str(cfg.trainer.gradient_clip_algorithm)

    trainer = pl.Trainer(**trainer_kwargs)

    do_train = bool(cfg.experiment.train)
    do_predict = bool(cfg.experiment.predict)
    from_checkpoint = bool(cfg.experiment.from_checkpoint)

    selected_ckpt = None

    if from_checkpoint:
        prefer = "latest" if do_train else "best"

        selected_ckpt = choose_checkpoint(
            checkpoint_callback=checkpoint_callback,
            ckpt_dir=ckpt_dir,
            run_checkpoint_root=paths_checkpoints,
            prefer=prefer,
        )

        if selected_ckpt is None:
            raise FileNotFoundError(
                f"from_checkpoint=True but no checkpoint found under: {paths_checkpoints}"
            )

        print(f"Checkpoint selected from disk: {selected_ckpt}")

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    if do_train:
        if selected_ckpt is not None:
            print("Loading checkpoint weights before training.")
            load_weights_only(
                lit_model=lit_model,
                ckpt_path=selected_ckpt,
            )

        trainer.fit(
            lit_model,
            datamodule=datamodule,
        )

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------
    if do_predict:
        paths_results.mkdir(parents=True, exist_ok=True)

        ckpt_to_use = checkpoint_callback.best_model_path or selected_ckpt

        if ckpt_to_use is None:
            ckpt_to_use = choose_checkpoint(
                checkpoint_callback=checkpoint_callback,
                ckpt_dir=ckpt_dir,
                run_checkpoint_root=paths_checkpoints,
                prefer="best",
            )

        if ckpt_to_use is not None:
            print(f"Loading checkpoint weights for prediction: {ckpt_to_use}")
            load_weights_only(
                lit_model=lit_model,
                ckpt_path=ckpt_to_use,
            )
        else:
            print("No checkpoint found. Predicting with current model weights.")

        predictions = trainer.predict(
            model=lit_model,
            datamodule=datamodule,
            ckpt_path=None,
        )

        if predictions is None or len(predictions) == 0:
            print("No predictions were returned.")
            return

        print("Processing and trimming padded predictions...")

        out_file = paths_results / f"predictions_{dataset_str}.parquet"

        predictions_to_parquet(
            predictions=predictions,
            out_file=out_file,
        )

        print(f"Prediction complete: {out_file}")


if __name__ == "__main__":
    main()