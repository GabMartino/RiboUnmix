# Benchmark preprocessing

The training config uses the four files under `weighted/`, not the raw profile
parquets in this directory. Regenerate them from the repository root with:

```bash
.venv/bin/python Datasets/benchmarking_data/weight_benchmarking_datasets.py --overwrite
```

The adapter checks unique and matching profile/CDS IDs, one-dimensional aligned
profiles and CDS arrays, and codons supported by `codon_encoding.yaml`. Each
source exposes one profile, so that observed profile is recorded explicitly as
one replica (`replica_ids=["source_profile"]`). It then calls the same weighting
implementation used for the HEK datasets.

The default reliability calculation is:

```text
tau_d       = median positive read density in dataset d
depth_score = sqrt(density) / (sqrt(density) + sqrt(tau_d))
raw_weight  = 0.70 * depth_score + 0.30 * coverage
weight      = raw_weight / median(raw_weight)
```

Zero-information profiles are physically removed. Positive normalized weights
may exceed one and are not clipped. Exact statistics and source mappings are in
`weighted/weight_manifest.tsv` and `weighted/preprocessing_manifest.json`.

The provided CDS arrays already align one-to-one with the profile positions.
Their terminal stop codons are therefore preserved rather than trimmed.

Training is deliberately single-dataset. Use the local launcher to schedule
the four independent runs across one or more GPUs:

```bash
GPU_DEVICES=0,1,2,3 ./run_benchmarking_datasets_local.sh
```

The launcher resumes safely when invoked again with the same `RUN_ID`. By
default it skips a dataset only when both `best_val_loss` and `best_pcc`
prediction parquets are readable and their final prediction manifest contains
matching positive row counts. Partial runs are scheduled again. Use
`SKIP_COMPLETED=false` to force all selected datasets to run again.

Each run writes both `split_manifest.json` and `split_manifest.tsv`. Convenient
run-independent copies are written below
`results/riboai_queueing_benchmarking/split_manifests/` for the default paths.
