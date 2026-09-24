from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

import hydra
import lightning as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, ListConfig, OmegaConf

from Dataloaders.RiboUnmixBenchmarking import (
    RiboUnmixBenchmarkingDataModule,
    available_ids_for_dataset,
)
from Models.RiboUnmixModel import RiboUnmixModel
from Models.RiboUnmixLightningModule import RiboUnmixLightningModule
from main_ribounmix_multidataset import (
    cfg_bool,
    cfg_get,
    choose_checkpoint,
    env_global_rank,
    find_prediction_checkpoint,
    load_weights_only,
    make_run_tag,
    open_file,
    sanitize_metric_name_for_filename,
    save_predictions_for_trainer,
    shared_logger_version_from_environment,
)


BENCHMARK_PREDICTION_CHECKPOINT_VARIANTS = (
    "best_val_loss",
    "best_pcc",
    "best_nb_nll",
)


def resolve_benchmark_prediction_checkpoint_variants(
    cfg: DictConfig,
) -> tuple[str, ...]:
    """Resolve benchmark checkpoint exports, including the common NB2 monitor."""
    raw = cfg_get(
        cfg,
        "prediction.checkpoint_variants",
        list(BENCHMARK_PREDICTION_CHECKPOINT_VARIANTS),
    )
    variants = (raw,) if isinstance(raw, str) else tuple(map(str, raw))
    if not variants:
        raise ValueError("prediction.checkpoint_variants must not be empty.")
    unknown = sorted(set(variants) - set(BENCHMARK_PREDICTION_CHECKPOINT_VARIANTS))
    if unknown:
        raise ValueError(
            "Benchmark prediction checkpoint variants must be drawn from "
            f"{BENCHMARK_PREDICTION_CHECKPOINT_VARIANTS}; got {unknown}."
        )
    if len(set(variants)) != len(variants):
        raise ValueError("prediction.checkpoint_variants contains duplicates.")
    return tuple(map(str, variants))


def find_benchmark_prediction_checkpoint(
    checkpoint_root: str | Path,
    variant: str,
) -> str | None:
    """Find a benchmark checkpoint without confusing objective-specific files."""
    if variant != "best_nb_nll":
        return find_prediction_checkpoint(checkpoint_root, variant)

    root = Path(checkpoint_root)
    if not root.exists():
        return None
    pattern = re.compile(
        r"val_nb_nll_raw=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    )
    scored: list[tuple[float, Path]] = []
    for path in root.rglob("nb-*.ckpt"):
        match = pattern.search(path.name)
        if path.is_file() and match is not None:
            scored.append((float(match.group(1)), path))
    return str(min(scored, key=lambda item: item[0])[1]) if scored else None


def select_single_dataset(raw: Any, available: Sequence[str]) -> str:
    if not isinstance(raw, str):
        raise ValueError(
            "Benchmarking experiment.dataset must be one dataset name, not a list. "
            f"Received: {raw}"
        )
    if raw.lower() == "all":
        raise ValueError(
            "experiment.dataset='all' is not allowed for benchmarking. "
            "Select exactly one dataset."
        )

    dataset_name = raw
    if dataset_name not in available:
        raise KeyError(
            f"Unknown benchmarking dataset {dataset_name!r}. "
            f"Available: {list(available)}"
        )
    return dataset_name


def split_dataset_ids(
    *,
    dataset_name: str,
    raw_ids: Sequence[str],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[str], list[str], list[str], dict[str, int]]:
    fractions = np.asarray([train_frac, val_frac, test_frac], dtype=np.float64)
    if np.any(fractions < 0.0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("split.train_frac + val_frac + test_frac must equal 1.")
    if train_frac <= 0.0 or val_frac <= 0.0:
        raise ValueError("Benchmarking train_frac and val_frac must be positive.")

    rng = np.random.default_rng(int(seed))
    ids = np.asarray(list(map(str, raw_ids)), dtype=object)
    if len(ids) < 3:
        raise ValueError(
            f"Benchmarking dataset {dataset_name!r} needs at least 3 aligned rows."
        )
    ids = ids[rng.permutation(len(ids))]

    n_val = max(1, int(round(len(ids) * val_frac)))
    n_test = max(1, int(round(len(ids) * test_frac))) if test_frac > 0.0 else 0
    while n_val + n_test >= len(ids):
        if n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            raise ValueError(f"Cannot form non-empty splits for {dataset_name}.")

    n_train = len(ids) - n_val - n_test
    train_ids = list(map(str, ids[:n_train]))
    val_ids = list(map(str, ids[n_train : n_train + n_val]))
    test_ids = list(map(str, ids[n_train + n_val :]))
    counts = {
        "total": len(ids),
        "train": len(train_ids),
        "validation": len(val_ids),
        "test": len(test_ids),
    }

    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Generated benchmarking splits overlap.")

    return train_ids, val_ids, test_ids, counts


def save_split_manifest(
    *,
    out_file: Path,
    dataset_name: str,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
    counts: dict[str, int],
    cfg: DictConfig,
    split_seed: int | None = None,
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    training_seed = int(cfg.experiment.seed)
    resolved_split_seed = training_seed if split_seed is None else int(split_seed)
    manifest = {
        # ``seed`` is retained for backward-compatible readers and denotes the
        # split RNG seed, as it did before training/split seeds were separable.
        "seed": resolved_split_seed,
        "split_seed": resolved_split_seed,
        "training_seed": training_seed,
        "dataset": str(dataset_name),
        "fractions": {
            "train": float(cfg.split.train_frac),
            "validation": float(cfg.split.val_frac),
            "test": float(cfg.split.test_frac),
        },
        "counts": counts,
        "train_ids": list(map(str, train_ids)),
        "validation_ids": list(map(str, val_ids)),
        "test_ids": list(map(str, test_ids)),
    }
    with out_file.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    split_rows = []
    for split_name, split_ids in (
        ("train", train_ids),
        ("validation", val_ids),
        ("test", test_ids),
    ):
        split_rows.extend(
            {
                "transcript_id": str(transcript_id),
                "split": split_name,
                "split_index": int(split_index),
            }
            for split_index, transcript_id in enumerate(split_ids)
        )
    tsv_file = out_file.with_suffix(".tsv")
    pd.DataFrame(split_rows).to_csv(tsv_file, sep="\t", index=False)
    print(f"Saved benchmarking split manifests: {out_file} and {tsv_file}")


def make_datamodule(
    *,
    cfg: DictConfig,
    dataset_specs: dict[str, dict[str, str]],
    dataset_name: str,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
) -> RiboUnmixBenchmarkingDataModule:
    feature_cfg = OmegaConf.to_container(
        cfg_get(cfg, "model.additional_sequence_features", {}),
        resolve=True,
    )
    return RiboUnmixBenchmarkingDataModule(
        dataset_specs=dataset_specs,
        dataset_name=dataset_name,
        batch_size=int(cfg.data.batch_size),
        split=(list(train_ids), list(val_ids)),
        predict_ids=list(test_ids),
        num_workers=int(cfg.data.num_workers),
        predict_num_workers=int(cfg_get(cfg, "data.predict_num_workers", 0)),
        seed=int(cfg.experiment.seed),
        nt_encoding_path=str(cfg.paths.encodings.nt),
        codon_encoding_path=str(cfg.paths.encodings.codon),
        codon_to_aa_encoding_path=str(cfg.paths.encodings.codon_to_aa),
        aa_encoding_path=str(cfg.paths.encodings.aa),
        datasets_encoding_path=str(cfg.paths.encodings.datasets),
        train_sampling_strategy=str(
            cfg_get(cfg, "data.train_sampling_strategy", "transcript_grouped_pairs")
        ),
        pin_memory=cfg_bool(cfg, "data.pin_memory", True),
        prefetch_factor=cfg_get(cfg, "data.prefetch_factor", 2),
        multiprocessing_context=cfg_get(
            cfg, "data.multiprocessing_context", "spawn"
        ),
        precompute_features=cfg_bool(cfg, "data.precompute_features", False),
        additional_sequence_features=dict(feature_cfg or {}),
    )


_CONFIG_DIR = str((Path(__file__).resolve().parent / "config"))


@hydra.main(
    version_base=None,
    config_path=_CONFIG_DIR,
    config_name="config_ribounmix_benchmarking",
)
def main(cfg: DictConfig) -> None:
    seed = int(cfg.experiment.seed)
    raw_split_seed = cfg_get(cfg, "split.seed", None)
    split_seed = seed if raw_split_seed is None else int(raw_split_seed)
    pl.seed_everything(seed, workers=True)
    torch.manual_seed(seed)

    dataset_specs = OmegaConf.to_container(
        cfg.benchmarking.datasets,
        resolve=True,
    )
    dataset_specs = {
        str(name): dict(spec) for name, spec in dict(dataset_specs).items()
    }
    dataset_name = select_single_dataset(
        cfg.experiment.dataset,
        list(dataset_specs),
    )
    available_ids = available_ids_for_dataset(dataset_specs, dataset_name)
    train_ids, val_ids, test_ids, split_counts = split_dataset_ids(
        dataset_name=dataset_name,
        raw_ids=available_ids,
        train_frac=float(cfg.split.train_frac),
        val_frac=float(cfg.split.val_frac),
        test_frac=float(cfg.split.test_frac),
        seed=split_seed,
    )

    print("\n=== Benchmarking dataset ===")
    print(
        f"{dataset_name:25s} total={split_counts['total']:6d} "
        f"train={split_counts['train']:6d} "
        f"val={split_counts['validation']:5d} test={split_counts['test']:5d}"
    )
    print(
        "Single-dataset benchmark: mu/profile recovery is evaluable, but the "
        "cross-dataset L_bio/gamma decomposition is not identifiable from this "
        "run alone. batch_grouped gamma centering uses its singleton fallback."
    )

    dataset_encoding = open_file(cfg.paths.encodings.datasets)
    if dataset_name not in dataset_encoding:
        raise KeyError(
            f"Benchmarking dataset {dataset_name!r} is missing from dataset encoding."
        )
    num_dataset_embeddings = int(cfg.model.dataset_bias_params.num_datasets)
    dataset_id = int(dataset_encoding[dataset_name])
    if not 0 <= dataset_id < num_dataset_embeddings:
        raise ValueError(
            "Benchmark dataset IDs must fit model.dataset_bias_params.num_datasets: "
            f"{dataset_name}={dataset_id}"
        )

    dataset_signature = dataset_name
    run_tag = make_run_tag(cfg)
    paths_logs = Path(cfg.paths.logs) / dataset_signature / run_tag
    paths_results = Path(cfg.paths.results) / dataset_signature / run_tag
    paths_checkpoints = Path(cfg.paths.checkpoints) / dataset_signature / run_tag

    if env_global_rank() == 0:
        save_split_manifest(
            out_file=paths_results / "split_manifest.json",
            dataset_name=dataset_name,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            counts=split_counts,
            cfg=cfg,
            split_seed=split_seed,
        )
        # Keep a short, run-tag-independent copy so transcript IDs for all four
        # independent datasets are easy to find before inspecting run folders.
        save_split_manifest(
            out_file=(
                Path(cfg.paths.results)
                / "split_manifests"
                / f"{dataset_name}_splitseed{split_seed}.json"
            ),
            dataset_name=dataset_name,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            counts=split_counts,
            cfg=cfg,
            split_seed=split_seed,
        )

    torch_model = RiboUnmixModel(
        model_configs=cfg.model,
        eps=float(cfg.model.get("eps", 1e-8)),
        nt_encoding=open_file(cfg.paths.encodings.nt),
        codon_to_aa_encoding=open_file(cfg.paths.encodings.codon_to_aa),
        codon_encoding=open_file(cfg.paths.encodings.codon),
        aa_encoding=open_file(cfg.paths.encodings.aa),
    )
    lit_model = RiboUnmixLightningModule(
        torch_model,
        config=cfg,
        dataset_encoding=dataset_encoding,
    )
    datamodule = make_datamodule(
        cfg=cfg,
        dataset_specs=dataset_specs,
        dataset_name=dataset_name,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
    )

    tb_logger = TensorBoardLogger(
        save_dir=str(paths_logs),
        name="",
        version=shared_logger_version_from_environment(),
    )
    tb_log_dir = Path(tb_logger.log_dir)
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    if env_global_rank() == 0:
        OmegaConf.save(cfg, tb_log_dir / "config.yaml")

    ckpt_dir = paths_checkpoints / tb_log_dir.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    monitor = str(cfg.optim.scheduler.monitor)
    metric_mode = str(cfg.optim.scheduler.mode)
    safe_monitor = sanitize_metric_name_for_filename(monitor)
    filename = "{epoch}-{" + monitor + ":.4f}"
    if "/" in monitor:
        filename = "{epoch}-" + safe_monitor + "-{" + monitor + ":.4f}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=filename,
        save_top_k=1,
        save_last=True,
        save_weights_only=True,
        monitor=monitor,
        mode=metric_mode,
    )
    pcc_checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="pcc-{epoch}-{val_mu_pcc:.4f}",
        save_top_k=1,
        save_last=False,
        save_weights_only=True,
        monitor="val_mu_pcc",
        mode="max",
    )
    nb_checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="nb-{epoch}-{val_nb_nll_raw:.4f}",
        save_top_k=1,
        save_last=False,
        save_weights_only=True,
        monitor="val_nb_nll_raw",
        mode="min",
    )
    callbacks = [
        checkpoint_callback,
        pcc_checkpoint_callback,
        nb_checkpoint_callback,
        EarlyStopping(
            monitor=monitor,
            patience=int(cfg.callbacks.early_stopping_patience),
            mode=metric_mode,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    devices_cfg = cfg.trainer.devices
    if isinstance(devices_cfg, (list, tuple, ListConfig)):
        n_devices = len(devices_cfg)
    else:
        try:
            n_devices = int(devices_cfg)
        except (TypeError, ValueError):
            n_devices = 1

    trainer_kwargs = {
        "accelerator": cfg.trainer.accelerator,
        "devices": devices_cfg,
        "precision": cfg.trainer.precision,
        "max_epochs": int(cfg.trainer.max_epochs),
        "logger": tb_logger,
        "log_every_n_steps": int(cfg.trainer.log_every_n_steps),
        "accumulate_grad_batches": int(
            cfg_get(cfg, "trainer.accumulate_grad_batches", 1)
        ),
        "callbacks": callbacks,
        "use_distributed_sampler": cfg_bool(
            cfg, "trainer.use_distributed_sampler", False
        ),
        "gradient_clip_val": float(
            cfg_get(cfg, "trainer.gradient_clip_val", 0.0)
        ),
        "gradient_clip_algorithm": str(
            cfg_get(cfg, "trainer.gradient_clip_algorithm", "norm")
        ),
    }
    if n_devices > 1:
        trainer_kwargs["strategy"] = "ddp_find_unused_parameters_true"
    trainer = pl.Trainer(**trainer_kwargs)

    do_train = cfg_bool(cfg, "experiment.train", True)
    do_predict = cfg_bool(cfg, "experiment.predict", True)
    from_checkpoint = cfg_bool(cfg, "experiment.from_checkpoint", False)
    selected_ckpt = None

    if from_checkpoint:
        selected_ckpt = choose_checkpoint(
            checkpoint_callback=checkpoint_callback,
            ckpt_dir=ckpt_dir,
            run_checkpoint_root=paths_checkpoints,
            prefer="latest" if do_train else "best",
        )
        if selected_ckpt is None:
            raise FileNotFoundError(
                f"from_checkpoint=True but no checkpoint exists under {paths_checkpoints}."
            )
        load_weights_only(lit_model=lit_model, ckpt_path=selected_ckpt)

    if do_train:
        trainer.fit(lit_model, datamodule=datamodule)

    if do_predict:
        paths_results.mkdir(parents=True, exist_ok=True)
        callback_paths = {
            "best_val_loss": checkpoint_callback.best_model_path,
            "best_pcc": pcc_checkpoint_callback.best_model_path,
            "best_nb_nll": nb_checkpoint_callback.best_model_path,
        }
        prediction_manifest: dict[str, dict[str, Any]] = {}
        for variant in resolve_benchmark_prediction_checkpoint_variants(cfg):
            ckpt_to_use = callback_paths[variant] or find_benchmark_prediction_checkpoint(
                paths_checkpoints,
                variant,
            )
            if ckpt_to_use is None:
                raise FileNotFoundError(
                    f"Prediction requested {variant!r}, but no matching checkpoint "
                    f"was found below {paths_checkpoints}."
                )
            print(f"Loading {variant} checkpoint for benchmark prediction: {ckpt_to_use}")
            load_weights_only(lit_model=lit_model, ckpt_path=ckpt_to_use)
            predictions = trainer.predict(
                model=lit_model,
                datamodule=datamodule,
                ckpt_path=None,
            )
            out_file = paths_results / (
                f"predictions_test_{variant}_{dataset_signature}.parquet"
            )
            prediction_rows = save_predictions_for_trainer(
                predictions=predictions,
                out_file=out_file,
                trainer=trainer,
            )
            prediction_manifest[variant] = {
                "checkpoint_path": str(ckpt_to_use),
                "output_path": str(out_file),
                "prediction_rows": int(prediction_rows),
            }
            if trainer.is_global_zero:
                print(
                    f"Saved {prediction_rows} {variant} benchmarking test "
                    f"predictions to {out_file}"
                )

        if trainer.is_global_zero:
            (paths_results / "prediction_checkpoint_manifest.json").write_text(
                json.dumps(prediction_manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
