# RiboAI Queueing

## Overview

This repository implements a queue-inspired, physics-aware neural model for
ribosome profiling signals (Ribo-seq). The current model separates a
dataset-invariant biological profile (`L_bio`) from dataset-specific
multiplicative (`gamma`) and additive corrections. Gamma factorizes into an
exponential amplitude and a sparse entmax support gate, so it is nonnegative,
can contain exact zeros, and remains unbounded above. The dataset head uses
dataset, codon-context, and position features; it does not consume `L_bio`.
Reliability-weighted gamma centering removes a multiplicative reference
ambiguity when matched cross-dataset transcript observations are present.

Training, validation, prediction, checkpointing, and logging use PyTorch
Lightning. Hydra controls the model, data, split, optimizer, and runtime
configuration.

## Model

For transcript `t`, dataset `d`, and codon position `i`, the active mean model is

```text
mu[d,t,i] = S[d,t] * (gamma[d,t,i] * L_bio[t,i] + additive_bias[d,t,i])
```

- `S[d,t]` is the valid-position mean of the observed target profile.
- `L_bio` is positive and normalized to mean one over valid positions.
- `gamma` is nonnegative and dataset-conditioned, with exact entmax zeros.
- `additive_bias` is nonnegative.
- `rho_bio = L_bio / (1 + L_bio)` is the bounded occupancy representation.
- The observation loss uses an NB2 parameterization with learned
  per-position log-dispersion, exposed as `log_sigma` for compatibility.

Because `S[d,t]` is computed from the target, the current implementation models
profile shape and relative allocation. It is not a target-free predictor of
absolute transcript abundance.

The complete current model, including entmax support, gradients, diagnostics,
invariances, and limitations, is in
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
It selects `weighted_datasets_paths_replicas` and expects:

- Sequence features at
  `Datasets/data/sequence/sequence_embeddings_with_css.parquet`.
- Dataset-specific profiles registered under `config/dataset_config/`.
- Encoding maps under `Datasets/encodings/`.
- The CSS split at `Datasets/data/sequence/css_split.json`.

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

Enable CAGrad for a matched multi-dataset run:

```bash
python main_ribo_queueing_modeling_multi_dataset.py \
  cagrad.enabled=true \
  experiment.dataset="['kutay_2021','grimson_2019','sako_2020']" \
  split.master_dataset_universe="['kutay_2021','grimson_2019','sako_2020']"
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

Gamma centering needs repeated transcript observations across datasets in the
same centering scope. Singleton transcripts and disconnected overlap components
do not share a data-driven multiplicative reference. The unrestricted additive
branch also permits decompositions that produce the same mean profile. Practical
next steps include hard reference-dataset anchoring, connected-component gauges,
low-dimensional additive nuisance models, and explicit cross-dataset consensus
losses. These options and their mathematical conditions are detailed in the
modeling analysis document linked above.

No license file is currently included.
