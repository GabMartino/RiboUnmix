# RiboAI Queueing

## Overview

This repository implements a queue-inspired, physics-aware neural model for
ribosome profiling signals (Ribo-seq). The current model separates a
dataset-invariant biological profile (`L_bio`) from dataset-specific
multiplicative (`gamma`) corrections. Gamma is the
exponential of its centered log-score, so it is strictly positive on valid
positions and remains unbounded above. The dataset head uses
dataset, codon-context, and position features; it does not consume `L_bio`.
The recommended fixed-reference gamma centering uses one checkpointed dataset
panel in every stage, while batch-grouped centering remains available for old
experiments.

Training, validation, prediction, checkpointing, and logging use PyTorch
Lightning. Hydra controls the model, data, split, optimizer, and runtime
configuration.

## Model

For transcript `t`, dataset `d`, and codon position `i`, the active mean model is

```text
shape[d,t,i] = mean_normalize(gamma[d,t,i] * L_bio[t,i])
mu[d,t,i] = S[d,t] * shape[d,t,i]
```

- `S[d,t]` is the valid-position mean of the observed target profile.
- `L_bio` is positive and normalized to mean one over valid positions.
- `gamma` is positive and dataset-conditioned.
- `rho_bio = L_bio / (1 + L_bio)` is the bounded occupancy representation.
- The explicit active objective is consensus hybrid PCC plus raw-replica NB2.
- NB2 uses learned
  per-position log-dispersion, exposed as `log_sigma` for compatibility.

Because `S[d,t]` is computed from the target, the current implementation models
profile shape and relative allocation. It is not a target-free predictor of
absolute transcript abundance.

Further model notes, including gradients, diagnostics, invariances, and
limitations, are in
[Docs/queueing_model.md](Docs/queueing_model.md). A longer historical
identifiability analysis is in
[`memory documents/modeling_mathematical_analysis_and_identifiability.md`](memory%20documents/modeling_mathematical_analysis_and_identifiability.md).

## Repository Layout

```text
main_ribo_queueing_modeling_multi_dataset.py   Hydra training entry point
config/                                       Runtime and dataset configuration
Dataloaders/RiboAIQueuingMultiDataset/         Parquet loading and batching
Models/RiboQueuingModel/                       Biological and dataset branches
Models/RiboQueuingModelLighningModule.py       Losses, metrics, and prediction IO
Datasets/                                      Encodings and local data assets
results/                                       Analysis scripts and generated results
tests/                                         Model invariance and forward checks
memory documents/                              Maintained modeling notes
```

## Setup

The checked environment uses Python 3.12. Create an isolated environment and
install the declared dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Training defaults to GPU execution with mixed bfloat16 precision. Select CPU or
a different precision through Hydra overrides when required by the local
hardware.

## Data Configuration

The default configuration is
[`config/config_riboai_queuing_multidataset.yaml`](config/config_riboai_queuing_multidataset.yaml).
It selects `weighted_hek_riboseq_codon_replicas` and expects:

- Sequence features at
  `Datasets/data/sequence/MANE.selection.sequence_embeddings_with_css.parquet`.
- Dataset-specific profiles registered under `config/dataset_config/`.
- Encoding maps under `Datasets/encodings/`.

The sequence parquet's CSS annotations guide joint validation stratification;
there is no separate CSS benchmark split or CSS holdout percentage. Validation
uses one deterministic transcript panel common to the configured master dataset
universe, and every other available transcript is assigned to training.

Each sample represents a transcript-dataset pair. The dataset implementation
precomputes sequence features and codon IDs, caches contiguous ribosome profiles,
and can carry every biological replica as a padded `[replica, position]` tensor.

## Training And Prediction

Run the two-dataset default experiment:

```bash
python main_ribo_queueing_modeling_multi_dataset.py
```

Run one registered dataset:

```bash
python main_ribo_queueing_modeling_multi_dataset.py \
  experiment.dataset="['kutay_2021']" \
  split.master_dataset_universe="['kutay_2021']" \
  'trainer.devices=[0]'
```

Run prediction from a checkpoint without training:

```bash
python main_ribo_queueing_modeling_multi_dataset.py \
  experiment.from_checkpoint=true \
  experiment.train=false \
  experiment.predict=true
```

Runtime artifacts are written below `checkpoints/riboai_queueing`,
`logs/riboai_queueing`, and `results/riboai_queueing` unless `paths.*` is
overridden. Inspect logs with:

```bash
tensorboard --logdir logs/riboai_queueing
```

The Slurm launchers use the same nested Hydra keys as the main configuration.
`run_parallel.sh` schedules the 30 active datasets across two visible GPUs.

## Analysis

Correlate exported model quantities with the supplied translation-efficiency
measure. A directory input is searched recursively for `predictions*.parquet`:

```bash
python analyse_TE_correlation.py path/to/predictions.parquet \
  --output results/te_correlations.csv
```

Render per-transcript profile plots from an explicit prediction file:

```bash
python results/visualize_profiles.py path/to/predictions.parquet \
  --limit 10 \
  --output-dir results/profile_plots
```

With no positional path, `visualize_profiles.py` uses the most recently modified
run under the selected experiment. Other scripts in `results/` aggregate model
metrics, compare single-dataset and mixed runs, and inspect learned biological
signals.

## Verification

The focused model checks are plain Python executables:

```bash
python tests/test_simple_queue_model.py
python tests/test_gamma_centering.py
```

They cover forward constraints, masking, target scaling, gamma centering, and
the main invariance properties. Hydra configuration can be checked without a
training launch using `hydra.compose` from Python.

## Current Identifiability Boundary

Fixed-reference centering defines a deterministic cross-dataset gamma gauge,
including for singleton requested inference. This algebraically fixes the
weighted common gamma mode over the configured reference panel, but it does not
prove that `L_bio` is purely biological or that gamma is purely technical:
sequence-correlated technical effects and genuine dataset-specific biology are
not labeled separately by the data. These conditions are detailed in
`Docs/model_mathematics.html`.

No license file is currently included.
