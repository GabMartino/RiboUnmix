# Synthetic training and validation smoke test

This directory is a compact integration example for the production RiboUnmix
pipeline. It contains 64 transcripts from two simulated bias conditions at a
nominal depth of two reads per codon. The example runs the real data module,
shared and dataset-specific branches, raw-replicate NB2 objective, consensus
shape terms, fixed-reference gamma centering, validation checkpoint selection,
and prediction export.

It is deliberately **not** an article-scale experiment. The network is reduced
to 15.3k trainable parameters and is trained for only two epochs so that a
reviewer can execute it on a CPU. Its metrics must not be cited as evidence for
the scientific performance of the full model.

## Install

Use Python 3.12 and install a platform-appropriate PyTorch build, followed by
the project dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

## Verify the committed checkpoint and validation export

```bash
python examples/synthetic_smoke/run.py verify
```

The verifier checks SHA-256 hashes before loading the checkpoint, asserts that
all learned tensors and exported quantities are finite, checks positivity,
mean-one shared profiles, duplicate-row equality of the shared output, both
gamma gauges, and recomputes validation PCC and RMSE from the frozen export.

## Replay validation inference from the committed checkpoint

```bash
python examples/synthetic_smoke/run.py replay
```

This invokes the production inference path and writes a fresh export below
`outputs/synthetic_smoke/replay/`. It does not train or modify the committed
reference files.

## Train from a fresh initialization and validate

```bash
python examples/synthetic_smoke/run.py train
```

Outputs are written below `outputs/synthetic_smoke/`, which is ignored by Git.
The production split code deterministically selects 48 training and 16
validation transcripts with seed 42. Exact floating-point values need not be
bitwise identical across PyTorch versions or hardware, but the run must finish
with finite gradients, a `best_val_loss` checkpoint, and 32 validation rows.

## Provenance and limitations

- `data_manifest.json` records the complete source-file hashes, selection rule,
  selected transcript IDs, and hashes of the three published Parquet files.
- The two biases are `artificial_bias_5prime_aa` and
  `artificial_bias_3prime_cc`; their data were not selected using model output.
- The sequence table contains only transcript ID, codons, and conserved-site
  metadata needed by this run. Full project data remain excluded.
- The validation fold is used for checkpoint selection and evaluation. It is a
  smoke-test validation set, not an independent test set.
- The committed checkpoint is a reproducibility fixture, not a pretrained model
  intended for transfer to real experiments.
