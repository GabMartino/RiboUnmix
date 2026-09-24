# Synthetic shared-only controls: gamma = 1

## Question and matrix

Does explicitly learning dataset-specific positional corrections improve the
shared sequence profile relative to pooling the same observations with no such
correction? This is an ablation of the fitted system, not a post-hoc removal of
gamma and not a diversity-versus-total-read-count experiment.

Two new fits are configured at nominal depth **C=2 reads/codon**, seed 42:

| Array index | N | Bias conditions |
|---|---:|---|
| 0 | 2 | 3′-AA, 3′-CC |
| 1 | 10 | The exact ten-condition panel of the archived full-model run |

The exact comparator runs are recorded in
`config/experiment_designs/synthetic_gamma_unity.yaml`. The launcher clones
their saved resolved configs, **not today's synthetic defaults**. It removes
only the cached runtime accumulation preview and relocates input/output paths.
The deliberate scientific change is `model.mean_correction: unity`:

\[
\mu_{t,d,i}^{(r)}=S_{t,d}^{(r)}L_{t,i},\qquad\gamma_{t,d,i}=1.
\]

The existing model implements this intervention during training and inference.
No weights are loaded from a trained full model. The learned dispersion head,
shared encoder, standard NB objective, PCC coefficients, reliability weights,
optimizer, batch size, accumulation target, execution chunks, early stopping
and precision are inherited. Best-validation-loss is the only exported
checkpoint; saving PCC-best checkpoints is disabled without changing selection.

Both archived runs use 17,284 train and 1,920 validation transcripts. Native
split construction is retained and **IDs and their order must exactly match**
the saved manifest before training. The original 11-dataset split universe
(including the unbiased data for split construction only) is preserved; only
the selected 2 or 10 biased datasets train the models. A changed split fails
instead of silently substituting a new cohort. These remain validation-based
experiments; no independent test set is invented.

## Important dispersion caveat

In the current production model, the alpha head receives detached features
from the dataset-context encoder. When gamma is fixed to one, the encoder no
longer receives a gradient from the mean and does not receive alpha gradients
either. The alpha head itself still learns, using that fixed context. Thus the
architecture and alpha-head optimization rule are retained, but the learned
dispersion features are **not held identical** between arms. Do not describe
this as a fully dispersion-controlled causal gamma intervention. Reconnecting
alpha gradients only in this control would introduce another method change.

Historical code may also differ from current production code. To remove this
code-version confound, the same launcher supports **two optional fresh learned-
gamma comparators** with `--arm learned` and a separate output root. They are
not launched or prepared automatically. For a strict current-code comparison,
run both arms under the same checkout/environment, giving four fits in total.

## Commands

Prepare and inspect locally (no GPU training; also the no-argument default):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python run_synthetic_gamma_unity.py --dry-run
```

On Leonardo, prepare there rather than copying machine-specific generated
configs. The two original run folders and synthetic inputs must be available
under the project. The generated output tree has separate N002/unity and
N010/unity directories, configs, historical split copies, input/code hashes,
and an experiment manifest. The two jobs run in parallel, one GPU each:

```bash
sbatch --account=euhpc_d35_089 --array=0-1%2 \
  --export=ALL,OUTPUT_ROOT=/leonardo_work/EUHPC_D35_089/synthetic_gamma_unity_C2_seed42 \
  run_synthetic_gamma_unity.slurm
```

Adjust `PROJECT_DIR`, `VENV_PATH`, account/QoS or memory to the local allocation
if necessary. Slurm's GPU visibility is preserved; no physical device ID is
overwritten. The script requests one GPU and 64 GB host RAM per task. Historical
BF16 and execution limits are preserved, not tuned for this arm.

Optional contemporaneous full-model reruns:

```bash
sbatch --account=euhpc_d35_089 --array=0-1%2 \
  --export=ALL,ARM=learned,OUTPUT_ROOT=/leonardo_work/EUHPC_D35_089/synthetic_gamma_learned_C2_seed42 \
  run_synthetic_gamma_unity.slurm
```

Local single-GPU training, only when explicitly desired:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python run_synthetic_gamma_unity.py --run --task-index 0
```

Each task writes `train.log` and `execution_status.json`; predictions retain the
production checkpoint manifest and model export format. An existing attempt
is never silently overwritten or restarted. This launcher does not implement
automatic walltime resume: inspect an interrupted task's saved checkpoints
before arranging a full-state resume. A process exit of zero is recorded as
`process_finished`, not claimed as validated scientific recovery.

## Evaluation after training

Primary endpoints: transcript-level PCC and RMSE between the exported shared
`L_bio` and the saved two-trajectory consensus `q_bar`. Use the same transcript
identities and exact modeled sense-codon positions as the full-model analysis,
including its boundary exclusion. Retain original profile amplitudes after
masking; do not re-normalize the cropped arrays. Use paired transcript-level
differences and transcript bootstrap intervals conditional on these fitted
models. Undefined PCCs remain missing; compare only matched valid pairs.

Secondary endpoints: observation reconstruction and learned alpha summaries.
Verify gamma=1 in the new exports. H is not the shared-only model's prescribed
target, so L-versus-H must not replace L-versus-q_bar for the ablation.

Higher recovery for the full model supports the usefulness of the complete
correction-enabled system in this synthetic setting. Similar recovery would
suggest that simple pooling already captures much of the benefit. Neither
outcome repairs the shared-NB2-randomness or single-ordering limitations of the
original simulator experiment.

## Setup verification (no training)

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python Tests/audit_synthetic_gamma_unity_setup.py
```

On 21 September 2026 the full-data split audit reproduced both archived folds
exactly (including ordering). Production CPU initialization probes confirmed
identical starting parameters, gamma exactly one, finite gradients in the shared
encoder and alpha head, and no gradient in the detached dataset-context encoder
for the unity arm. Hydra resolved both generated configs. Parent-process peak
RSS was 0.93 GiB; no training epochs or GPU probes were run.

Results and provenance are saved under the prepared output root in
`setup_audit.json`, `computation_audit.json` and `dry_run.log`. The computation
manifest is a validated legacy-v1 record; full current input/code hashes are
also in `experiment_manifest.json`. These checks do not prove equivalence to
historical source code or confirm full-run GPU stability.

Ten targeted unit tests passed: eight configuration/split tests plus the existing
initialization and unity-gradient tests in `Tests/test_reference_campaign.py`.
Python compilation, shell syntax and launcher `--help` checks passed.
