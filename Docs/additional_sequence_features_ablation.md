# Ablation study for optional sequence features

## Objective

The optional features are dataset-independent sequence annotations, but the
model can expose them to the shared biological branch, the dataset-bias branch,
or both. The ablation must distinguish two questions:

1. Do the annotations contain useful profile-shape information beyond the
   nucleotide, codon, and amino-acid one-hot inputs?
2. If they are useful, should they explain the shared biological load
   $$L_{\mathrm{bio}}$$, dataset-conditioned adaptation $$\gamma$$, or both?

The `fra` column is excluded from every experiment because it is exactly
`[1,0,0]` at every position in the current parquet. The tAI feature uses zero
for the undefined stop-codon values stored as `-1000` or NaN.

## Stage 1: information and routing

Run these four experiments first with identical splits, optimization settings,
and seeds:

| Preset | Biological branch | Dataset-bias branch | Question |
|---|---|---|---|
| `baseline` | Basic 97 one-hot channels | Codon and dataset embeddings | Reference model |
| `biological` | All five useful optional features | No optional features | Do features improve the shared signal? |
| `dataset_bias` | No optional features | All five useful optional features | Do features mainly explain measurement adaptation? |
| `both` | All five useful optional features | All five useful optional features | Are feature-by-dataset interactions additionally useful? |

Use at least three seeds for the finalists. Select using representative
validation $$\mathrm{PCC}(\mu,y)$$, but report the following together:

- `val_mu_pcc` and `val_mu_pcc/<dataset>`;
- `val_loss`, `val_nll`, and their per-dataset values;
- `val_pcc_raw_loss/<dataset>` and `val_pcc_nb_vst_loss/<dataset>`;
- gamma exact-zero fraction and zero precision/recall;
- CSS-stratified validation subgroups as a secondary biological-peak analysis.

Routing channels also changes the first-layer GRU parameter count. With the
current hidden sizes, 13 biological channels add 19,968 parameters, 13 bias
channels add 9,984, and `both` adds 29,952 in total. These additions are small
relative to the whole model but should be reported as a capacity caveat; compare
validation behavior across several seeds rather than treating a single small
gain as feature evidence.

The routing comparison should be performed on the joint multi-dataset model.
In a single-dataset run, a sequence-conditioned bias branch can absorb shared
biology without any cross-dataset evidence that identifies it as bias.

## Stage 2: feature contribution

If `biological` beats `baseline`, run the five biological-only feature presets:

| Preset | Enabled feature | Data character |
|---|---|---|
| `openen_bio` | `openen` | Continuous, approximately standardized |
| `tai_bio` | cleaned `tAI_profile_codon` | Codon-level value in approximately $$[0.0995,1]$$ |
| `tmp_bio` | `tmp` | Three bounded continuous/ordinal channels |
| `gmp_bio` | `gmp` | Three bounded channels, strongly concentrated at 1 |
| `exo_bio` | `exo` | Three sparse binary channels |
| `core_bio` | `openen` and cleaned tAI | Compact high-priority subset |

The priority order is `core_bio`, `openen_bio`, `tai_bio`, `tmp_bio`, then
`gmp_bio` and `exo_bio`. This ordering follows the observed information content:
`fra` is constant, `exo` is approximately 98.3% all-zero by codon, and `gmp` is
approximately 89.7% equal to 1 per channel.

After identifying the best biological subset, route only that subset through
`dataset_bias` and `both`. This avoids interpreting a routing effect that is
actually caused by one weak or redundant feature.

## Interpretation

- `biological > baseline`, with no gain from `both`: keep features only in
  $$L_{\mathrm{bio}}$$. This gives the cleanest decomposition.
- `dataset_bias > biological`: the annotations mainly predict dataset-specific
  measurement effects, but inspect whether gamma is replacing biological
  structure.
- `both > biological`: feature-by-dataset interactions add useful information;
  verify that $$L_{\mathrm{bio}}$$ remains stable and gamma centering remains
  well behaved.
- Better NLL without better raw PCC indicates calibration improvement rather
  than better peak ordering.
- Better NB-VST PCC without better raw PCC indicates improvement concentrated
  in low and medium counts rather than extreme peaks.

## Slurm usage

The launchers accept `FEATURE_PRESET`. For example:

```bash
sbatch --export=ALL,FEATURE_PRESET=baseline run_all_datasets.slurm
sbatch --export=ALL,FEATURE_PRESET=biological run_all_datasets.slurm
sbatch --export=ALL,FEATURE_PRESET=dataset_bias run_all_datasets.slurm
sbatch --export=ALL,FEATURE_PRESET=both run_all_datasets.slurm
sbatch --export=ALL,FEATURE_PRESET=core_bio run_all_datasets.slurm
```

The resolved feature routing is also encoded automatically in the internal
checkpoint, TensorBoard, and results directory tag, so manual Hydra routing
overrides remain traceable even when they do not correspond to a launcher
preset.

The tag begins with a readable inferred preset and then records the exact
routing. Examples are `FeatPresetBaseline_SeqFeatBase`,
`FeatPresetBiological_SeqFeatBio-...`, `FeatPresetDatasetBias_SeqFeatBias-...`,
`FeatPresetBoth_SeqFeatBoth-...`, and
`FeatPresetCoreBio_SeqFeatBio-open-taiFill0p0`. A route combination that does
not match a launcher preset is labeled `FeatPresetCustom` while retaining its
complete route list.
