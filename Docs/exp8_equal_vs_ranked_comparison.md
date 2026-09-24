# Audit of the equal-weight and cumulative ranked Exp8 results

Audited 10 September 2026 from the locally copied result artifacts. These two
runs **are not the same experiment with only pi changed**. The ranked run has
higher successive-size profile agreement, but selection and overlap differ
substantially. These results cannot attribute that difference to pi alone.

## Project and experiment provenance

The documentation starting points were
[project onboarding](project_model_onboarding.html),
[the model configuration explanation](current_configured_model.md),
[original Exp8 design](real_exp8_shared_profile_stability.html), and
[cumulative ranked design](real_exp8_cumulative_quality_rank.md).
Saved task configurations take precedence over current base configuration and
dated documentation. In particular, the current base model's mass conservation
and extra sequence features do not describe these saved experiments.

RiboUnmix learns a dataset-independent, sequence-only positive
profile L, normalized to mean one over the coding sequence, and a
dataset-conditioned multiplicative profile gamma. These experiments use
mu = S * L * gamma, where S is an observed mean count scale, with mass
conservation disabled. Fixed-reference centering imposes a weighted zero mean
of log gamma across selected datasets; pi defines this reference. It is distinct
from transcript/dataset reliability weights w_dt in the training objective.
Comparing L from different fitted models measures representation agreement,
not prediction of total abundance or recovery of known biological truth.

| Property | Equal-weight results | Ranked results |
|---|---|---|
| Result root | `results/my_exp8_a100_b32_20260906_114340` | `results/real_exp8_L_stability_quality_rank/cumulative_qrankp1.0_seed42` |
| Launcher | `run_real_exp8_L_stability.py` / `.slurm` | `run_real_exp8_L_stability_quality_rank.py` / `.slurm` |
| Manifest experiment name | `real_exp8_L_stability` | `real_exp8_L_stability_cumulative_quality_rank` |
| Dataset counts | 2, 5, 10, 20, 40, 80, 114 | Same |
| Planned fits | 34 | 7 |
| Small panels | Three designated disjoint A/B pairs at each N through 40 | One nested top-N quality prefix at each N |
| Large panels | Three N=80 subsets; one N=114 full model | One model at each N |
| Selection | Match the full pool's continuous QC distribution | Ascending frozen global quality rank |
| Publication/source families | Atomic; designated A/B pairs also source-disjoint | Exact prefixes can split families |
| Gamma weights | pi_d = 1/N | pi_d = q_d / sum(q), q_d = (115-r_d+1)/115 |
| Optimization seed | 42 | 42; independent initialization at every N |
| Held-out transcripts | 1,771 | Exactly the same 1,771 |

The ranked launcher reuses the original launcher's production configuration
builder and overrides reference weighting and ranking provenance. Inspection
of **all 41 saved resolved configurations** confirms matching core architecture,
loss, and original execution settings: learned alpha, no mass conservation,
standard NB with beta=0, replica-NB/raw-PCC/VST-PCC coefficients 1/0.5/0.5,
transcript-balanced loss, grouped sampling, at least two observed datasets per
eligible training transcript, logical batch quota 32, bf16-mixed precision,
raw log-gamma bound 8, and disabled additional sequence features. The ranked
configuration additionally records `trainer.detect_anomaly=false`.
Each task's dataset membership, reference, external split panel and reliability
manifest change as expected from the different design.

Training/validation eligibility is task-specific, despite sharing test IDs.
For example, equal N002/pair01_A has 10,305 eligible training and 1,145 validation
transcripts; ranked N002/trainseed42 has 11,568 and 1,285 respectively. Reliability
references are fitted on each task's training transcripts only.
The common test hash is
`af519c94d063c21c3c0cfa546f668785dbed86cafe0ff5c959ff8fb85ed953f0`.

No equal-weight subset at N<114 matches the ranked subset at the same N.
Only the planned full N=114 dataset memberships match; neither full model has
a verified prediction export locally. At N=2, equal pair01_A uses `soto_2021`
and `weber_2024` with pi=(0.5, 0.5); ranked uses `patel_2020` and `calviello_2016`
with pi=(0.5021834, 0.4978166). Thus the ranked N=2 weights are almost uniform,
while membership is entirely different for this example pair.

These are copied cluster artifacts. Both experiment manifests lack a recorded
commit hash, so identical historical source code cannot be established.
Some equal-weight fits have resume histories. In particular, the two completed
N=40 tasks record full-state resumes with the dataset-bias GRU precision
overridden to float32 and zero data workers. Original resolved configurations
are therefore records of intended launch settings, not proof of identical
runtime trajectories. N=40 is only in the equal-only companion figure/table.

## Availability and what the requested figure measures

The requested old figure is generated by `analyses/analyze_real_exp8_partial.py`.
At each N it pools per-transcript PCC from the designated same-N disjoint A/B
pairs, plotting arithmetic mean +/- sample SD. There is no corresponding ranked
curve for that statistic: the ranked design fits only one model per N/seed.
Treating a ranked N-to-next-N comparison as another disjoint same-N comparison
would give the two curves different meanings.

The new script therefore uses the same observable for both curves: per-transcript
full-CDS PCC between models at N and the next **planned** dataset count.
Within each experiment, all available model pairs sharing the training seed
enter the statistic. No pair is selected according to its PCC, no unavailable N
is bridged, and no smaller fit substitutes for the absent N=114 reference.
The styling follows the requested figure: serif/LaTeX typography, connected
means, and SD bars. Its x-axis explicitly labels the N-to-next-N transitions.

| N | Equal verified / planned | Ranked verified / planned |
|---|---:|---:|
| 2 | 6 / 6 | 1 / 1 |
| 5 | 6 / 6 | 1 / 1 |
| 10 | 6 / 6 | 1 / 1 |
| 20 | 4 / 6 | 1 / 1 |
| 40 | 2 / 6 | 0 / 1 |
| 80 | 0 / 3 | 0 / 1 |
| 114 | 0 / 1 | 0 / 1 |

All 28 present exports passed checks for test identity, runtime best_val_loss
sequence-only provenance, run ID/N, reference membership/pi, mean-one profiles,
finite nonnegative values, full-CDS masks and pairwise position alignment.
No PCC values were undefined in the computed comparisons.

| Transition | Equal mean PCC +/- SD | Ranked mean PCC +/- SD | Equal / ranked model pairs | Mean dataset Jaccard, equal / ranked |
|---|---:|---:|---:|---:|
| 2 to 5 | 0.5251 +/- 0.1513 | 0.8403 +/- 0.0438 | 36 / 1 | 0.0185 / 0.4000 |
| 5 to 10 | 0.7215 +/- 0.0986 | 0.8977 +/- 0.0581 | 36 / 1 | 0.0324 / 0.5000 |
| 10 to 20 | 0.7381 +/- 0.1319 | 0.8816 +/- 0.0592 | 24 / 1 | 0.0749 / 0.5000 |

The ranked design shows higher agreement at every jointly available transition.
It also retains all datasets from the smaller panel, whereas equal-weight
cross-size panels overlap little. Dataset selection, source overlap, eligibility
and reference definition confound a weighting-only interpretation. SD measures
dispersion across pooled transcript/model-pair values; observations share models
and transcripts, and the ranked curve has one model pair per point. These are
not independent replicates or confidence intervals.

Equal 20-to-40 agreement (0.7190 +/- 0.1483, eight model pairs) is retained in the
source table but excluded from the joint figure because ranked N=40 is absent.
The updated original disjoint plot also includes the newly available N=40 pair
(mean PCC 0.6980). Its N=2,5,10,20 results reproduce the old plot's statistics.

## Reproduce and inspect

From the repository root:

```bash
MPLCONFIGDIR=/tmp/exp8-comparison-mpl .venv/bin/python analyses/compare_real_exp8_weighting.py
```

Use `--equal-root`, `--ranked-root` and `--output-dir` to change paths. `--no-tex`
uses Matplotlib fonts when LaTeX is unavailable. Source experiments are read
only; output defaults to `../results/archive/exp8_equal_vs_ranked_comparison`.

- `stability_equal_vs_ranked.png` and `.pdf`: both experiments, common transitions.
- `stability_equal_disjoint_updated.png` and `.pdf`: original metric, updated exports.
- `successive_N_summary.csv` and `successive_N_per_transcript.parquet`: plotted
  and one-sided comparisons, counts, dataset/source overlap and raw PCCs.
- `equal_disjoint_summary.csv` and `equal_disjoint_per_transcript.parquet`:
  original metric source values.
- `dataset_membership_and_pi.csv`: actual planned membership and weights for all tasks.
- `config_differences_from_equal_N002_pair01_A.csv`: exhaustive saved configuration differences.
- `export_audit.csv`: per-task availability, verified file paths/hashes and checkpoint provenance.
- `analysis_manifest.json` and `figure_caption.txt`: reproducibility metadata and interpretation.

To isolate pi, the missing experiment is a matched re-training: reuse the exact
original disjoint subsets, train/validation/test IDs, reliability references,
initialization seeds and execution settings, changing only reference weighting
to the frozen rank rule. Both arms could then be compared using the original
same-N disjoint statistic. Alternatively, train equal pi on the exact ranked
prefixes to isolate weighting within the cumulative design. Reweighting saved
L predictions cannot replace either re-training experiment. No new training
was launched for this analysis.

Validation: the script completed on the real exports and generated 189,497
successive-size transcript/model-pair PCCs. All original plotted disjoint means
and SDs were reproduced to relative tolerance 1e-12. Twelve focused comparison
and prediction-provenance tests passed, and the generated main figure was
visually inspected. `git diff --check` passed.
