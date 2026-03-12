RiboAI Queueing — Training and Inference

Overview

This repository implements a queue-inspired, physics‑aware neural model for ribosome profiling signals (Ribo‑seq). It uses a bidirectional GRU backbone coupled with identifiable factors for per‑position dropout, occupancy (rho), position weights (w), and a global flux (J). Training and evaluation are orchestrated with PyTorch Lightning and Hydra for configuration.

Key features

- PyTorch + Lightning training loop with TensorBoard logging and checkpointing
- Hydra configuration system with dataset registry in YAML
- Custom DataModule and Dataset for Parquet‑backed inputs and sequence encodings
- Support for forward/backward sequence direction and optional curriculum sampling
- Automatic checkpoint discovery/resume helpers


Stack and Tooling

- Language: Python
- Core libraries:
  - torch (PyTorch)
  - lightning (PyTorch Lightning, imported as `import lightning as pl`)
  - numpy, pandas
  - matplotlib (diagnostics/plots)
  - hydra-core, omegaconf
  - torchmetrics
  - pyyaml
  - pyarrow (required by pandas for Parquet IO)
- Logging/vis: TensorBoard via Lightning’s `TensorBoardLogger`
- Configuration: Hydra YAML under `config/`
- Hardware: NVIDIA GPU required (Lightning `accelerator="gpu"`, `precision="bf16-mixed"`); CPU is not configured by default.

Package manager and versions

No `requirements.txt` or `pyproject.toml` are present. Minimal dependencies (names only):

```
python
torch
lightning
torchmetrics
hydra-core
omegaconf
numpy
pandas
matplotlib
pyyaml
pyarrow
tensorboard  # for viewing logs
```

TODO: Add an environment file (`requirements.txt` or `environment.yml`) with exact versions tested on your system.


Project Structure

- `main_ribo_queueing_modeling.py` — Entry point for training/prediction with Hydra config.
- `config/`
  - `config_riboai_queuing.yaml` — Main configuration (paths, training flags, hyperparameters).
  - `dataset_config/preprocessed_datasets.yaml` — Dataset registry: dataset names → Parquet paths (and aggregated variants).
- `Dataloaders/`
  - `RiboAIQueuing/RiboAIQueuingDatamodule.py` — Lightning `DataModule` (Parquet loading, batching, curriculum, splits).
  - `RiboAIQueuing/RiboAIQueuingDataset.py` — Dataset that builds features from sequence and encodings.
- `Models/`
  - `RiboQueuingModel/RiboQueuingModel.py` — BiGRU‑based model producing `pi_i`, `rho`, `w_prob`, `J`, and `sigma_i`.
  - `RiboQueuingModelLighningModule.py` — LightningModule (losses, optimizer/scheduler, training/validation/predict).
- `Datasets/`
  - `encodings/*.yaml` — Nucleotide/codon/amino‑acid encodings.
  - `data/` — Parquet datasets; also `.json` files with pre‑computed train/val bucket splits per dataset.
- `Utils/utils.py` — Collection of masked metrics and loss functions.
- `outputs/`, `checkpoints/` — Output and checkpoint directories (created at runtime).
- `analyse_TE_correlation.py` — WIP analysis script (placeholder at present).


Requirements

- Python 3.x (exact version not pinned; tested Python version is not specified)
- CUDA‑capable NVIDIA GPU for training (Lightning is set to `accelerator="gpu"` and `precision="bf16-mixed"`)
- System packages for Parquet (via `pyarrow`)

TODO: Specify the exact Python and CUDA versions known to work (e.g., Python 3.10, CUDA 12.x), and the matching PyTorch build.


Data Expectations

The DataModule expects Parquet files with at least the following fields (per row = per transcript):

- `sequence`: list of codon strings or integer indices
- `ref`: numeric features per position (array per position)
- `ribo`: observed ribosome occupancy/profile per position (float array)
- `classes`: integer class labels per position (e.g., 0/1/2); padding index used during batching
- `transcript_id`, `total_reads` (metadata)

Splits: For each selected dataset `<name>.parquet`, a companion JSON `<name>.json` is used to define `training_set` and `validation_set` bucket indices. The main script converts the configured Parquet path to a JSON path automatically (same base name) and performs a stalling‑site‑aware split.

Configured dataset registry: see `config/dataset_config/preprocessed_datasets.yaml` for available dataset keys (e.g., `kutay_2021`, `weber_2020`, etc.) and their Parquet locations under `Datasets/data/preprocessed_datasets_with_css/`.


Configuration (Hydra)

Default config: `config/config_riboai_queuing.yaml`

Key options (non‑exhaustive):

- Paths:
  - `checkpoints_path`: default `./checkpoints/riboai_queueing`
  - `log_dir`: default `./logs/riboai_queueing`
  - `results_dir`: default `./results/riboai_queueing`
  - Encodings: `Datasets/encodings/{nt_encoding.yaml,codon2aa.yaml,codon_encoding.yaml,aa_encoding.yaml}`
- Workflow flags:
  - `from_checkpoint` (bool): load latest checkpoint for the selected dataset/direction if available
  - `train` (bool), `predict` (bool)
  - `dataset`: key from `dataset_config.preprocessed_datasets`
- Training:
  - `seed`, `split_size` (float; fraction for train)
  - `batch_size`, `num_workers`, `max_epochs`, `early_stopping_patience`
  - Model hparams: `hidden_dims`, `num_layers`, `dropout`, `sigma_min`, `sigma_max`
  - Directionality: `bidirectional` (bool); if `False`, set `direction: forward|backward`
- Hardware:
  - `device`: list of GPU indices, e.g., `[0]`

You can override any value from the command line using Hydra syntax, e.g., `batch_size=128 dataset=weber_2020`.


Entry Points and Scripts

1) Training / Prediction

Main entry point: `main_ribo_queueing_modeling.py`

- Train from scratch (example):

```
python main_ribo_queueing_modeling.py \
  dataset=kutay_2021 \
  train=True predict=False from_checkpoint=False \
  batch_size=64 hidden_dims=256 num_layers=3 dropout=0.5 \
  device=[0]
```

- Resume/finetune from the latest checkpoint for the current dataset/direction:

```
python main_ribo_queueing_modeling.py dataset=kutay_2021 from_checkpoint=True train=True predict=False
```

- Run prediction on validation split with the latest/best checkpoint:

```
python main_ribo_queueing_modeling.py dataset=kutay_2021 from_checkpoint=True train=False predict=True
```

Notes

- Checkpoints and logs are organized under: `{checkpoints_path}/{dataset_config._name_}/{dataset}/{bidirectional_or_direction}/`.
- When `bidirectional: False`, an extra `direction` subfolder (`forward` or `backward`) is used.
- The code automatically discovers the “latest” checkpoint if `from_checkpoint=True`.
- TensorBoard logs: see `log_dir` with the same hierarchy. Launch with `tensorboard --logdir=logs/riboai_queueing`.

2) Analysis (work in progress)

- `analyse_TE_correlation.py` — Placeholder that loads `Datasets/data/raw_datasets/TE_ilr_residual.clr.median_across_datasets.csv`. Extend as needed.


Setup and Installation

1) Create and activate an environment (examples)

- pip + venv (example):

```
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install torch lightning torchmetrics hydra-core omegaconf numpy pandas matplotlib pyyaml pyarrow tensorboard
```

- conda (example):

```
conda create -n riboai python=3.10 -y
conda activate riboai
pip install torch lightning torchmetrics hydra-core omegaconf numpy pandas matplotlib pyyaml pyarrow tensorboard
```

2) Data

- Place Parquet datasets under `Datasets/data/preprocessed_datasets_with_css/` matching the paths in `config/dataset_config/preprocessed_datasets.yaml`.
- Ensure a corresponding JSON split file exists for each selected dataset (same base name, `.json` extension) with `{"training_set": [...], "validation_set": [...]}`.
- Verify the encoding YAMLs exist under `Datasets/encodings/`.

3) Run training

```
python main_ribo_queueing_modeling.py dataset=<one_of_registered_keys> train=True predict=False device=[0]
```


Environment Variables

No required environment variables are used by the code at this time. Paths and options are provided via Hydra configuration and CLI overrides.

TODO: Document any environment variables if introduced later (e.g., for data roots or experiment naming).


Testing

No test suite is present in the repository.

TODO: Add unit tests for:
- Dataset feature assembly and collate behavior
- Losses/metrics in `Utils/utils.py`
- Model forward pass shapes and value ranges
- End‑to‑end training step smoke test on a tiny synthetic dataset


License

No license file is present.

TODO: Add a `LICENSE` file (e.g., MIT, BSD‑3‑Clause, Apache‑2.0) and mention it here.


Troubleshooting

- GPU/Precision: The Trainer is hard‑coded to `accelerator="gpu"` and `precision="bf16-mixed"`. Ensure your GPU and drivers support bfloat16. If you need CPU or different precision, you will have to modify `main_ribo_queueing_modeling.py` accordingly.
- Parquet errors: Install `pyarrow` and ensure Parquet file schema matches expected columns.
- Checkpoint not found: Verify the directory layout under `checkpoints_path` and that `from_checkpoint=True` matches an existing run.


Citation

If you use this code in a publication, please cite the repository and the related RiboMIMO/queuing‑based modeling work. TODO: Add formal citation once available.
