# N=10 gamma-unity controls across sequencing depth

## Scientific question

These three controls ask whether the correction-enabled system is preferable to
a shared-only multi-dataset model when the ten injected bias families are held
fixed and only nominal sequencing depth changes:

\[
C\in\{0.25,2,20\}\ \text{reads/codon},\qquad N=10,
\]

\[
\mu_{t,d,i}^{(r)}=S_{t,d}^{(r)}L_{t,i},\qquad
\gamma_{t,d,i}\equiv1.
\]

This is a fresh-training ablation. It is not post-hoc removal of gamma from a
trained full model. Raw replicas, transcript--dataset reliability weights,
equal gamma-reference metadata, the shared encoder, learned dispersion head,
loss, optimizer, grouped sampling, precision, and checkpoint rule are inherited
from the exact archived N=10 full-model run at each depth.

## Frozen source runs

| Array index | Depth | Archived comparator |
|---:|---:|---|
| 0 | 0.25 | `riboai_synthetic_within_0p25_per_codon_panel10_gammaequal_seed42_massfree_bf16_test_20260828_232959_8` |
| 1 | 2 | `riboai_synthetic_within_2_per_codon_panel10_gammaequal_seed42_massfree_bf16_test_20260828_232959_17` |
| 2 | 20 | `riboai_synthetic_within_20_per_codon_panel10_gammaequal_seed42_massfree_bf16_test_20260828_232959_26` |

All three panels contain the same ten artificial bias families in the same
order. Each archived split contains 17,284 training and 1,920 validation
transcripts. The controls remain validation-based historical comparisons; they
do not create an independent test cohort or independent count realization.

The historical validation identities differ by depth: pairwise intersections
contain 184--225 transcripts and the three-way intersection contains only 27.
This is acceptable for the matched ablation because each unity run is compared
with its own depth-specific archived learned-gamma run. It does mean that
transcript-paired inference across depths is not supported. Cross-depth plots
must show separate within-depth effects and disclose the different cohorts.

## Files

- Design: `config/experiment_designs/synthetic_gamma_unity_n10_depths.yaml`
- Preparation/training entry point: `run_synthetic_gamma_unity.py`
- Leonardo launcher: `run_synthetic_gamma_unity_n10_depths.slurm`
- Direct two-GPU launcher: `run_synthetic_gamma_unity_n10_depths_2gpu.sh`
- CPU-only setup audit: `Tests/audit_synthetic_gamma_unity_setup.py`
- Default local preparation root:
  `results/synthetic_gamma_unity_N10_depths_seed42`

## Local preparation and audit

Preparation is non-training by default:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python run_synthetic_gamma_unity.py \
  --design config/experiment_designs/synthetic_gamma_unity_n10_depths.yaml \
  --output-root results/synthetic_gamma_unity_N10_depths_seed42 \
  --arm unity --dry-run
```

Then verify all depth-specific historical splits, model initialization, exact
unit gamma, gradient paths, input hashes, and Hydra resolution without fitting:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python Tests/audit_synthetic_gamma_unity_setup.py \
  --output-root results/synthetic_gamma_unity_N10_depths_seed42
```

## Direct two-GPU execution with `nohup`

The direct launcher assigns one sequential queue to each visible GPU. With the
three-task design, GPU 0 runs task 0 and then task 2, while GPU 1 runs task 1.
The archived configurations are frozen to training seed 42 and to zero data
loader workers. The corresponding environment variables are assertions: the
launcher rejects other values instead of silently changing the matched setup.

```bash
nohup env \
  VENV_PATH="$PWD/.venv" \
  GPU_DEVICES=0,1 \
  TRAINING_SEEDS=42 \
  NUM_WORKERS=0 \
  PREDICT_NUM_WORKERS=0 \
  OUTPUT_ROOT="$PWD/results/synthetic_gamma_unity_N10_depths_seed42" \
  bash run_synthetic_gamma_unity_n10_depths_2gpu.sh \
  > synthetic_gamma_unity_n10_depths_2gpu.nohup.log 2>&1 &

echo $! > synthetic_gamma_unity_n10_depths_2gpu.pid
disown
```

Run the same command once with `DRY_RUN=1` and a fresh output root when testing
the launcher on a new machine. A contemporaneous learned-gamma arm must use
`ARM=learned` and a distinct output root.

## Leonardo submission

```bash
sbatch run_synthetic_gamma_unity_n10_depths.slurm
```

The array indices are fixed as 0 = 0.25, 1 = 2, and 2 = 20 reads/codon.
Training is opt-in and each task refuses to overwrite an existing attempt.

For contemporaneous learned-gamma comparators, use a separate output root:

```bash
ARM=learned \
OUTPUT_ROOT=/leonardo_work/EUHPC_D35_089/${USER}/riboai_runs/synthetic_gamma_learned_N10_depths_seed42 \
sbatch run_synthetic_gamma_unity_n10_depths.slurm
```

## Interpretation limits

This matrix isolates the fitted-system effect of enabling dataset-specific mean
corrections across the three existing depths. It does not resolve the original
single-seed, validation-selection, common-count-randomness, or fixed-bias-panel
limitations. The learned alpha head retains detached dataset-context inputs;
removing the gamma mean path also removes the context encoder's training
gradient. Consequently this is a whole-system shared-only ablation, not a
dispersion-controlled causal intervention.

After training, comparisons must be paired within depth on the exact common
validation transcripts and positions. Primary endpoints should include shared
profile agreement with latent occupancy, observation reconstruction, and a
check that every exported unity correction equals one. Constant gamma PCC is
undefined and must not be replaced by zero.
