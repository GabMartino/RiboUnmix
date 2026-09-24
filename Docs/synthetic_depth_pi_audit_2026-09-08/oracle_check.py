"""No-training gauge diagnostic using 128 deterministic, nonrandom transcripts.

Run from repository root. Outputs are illustrations, not held-out model scores.
Whole-profile mean-one normalization precedes five-codon boundary trimming.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / 'Datasets/Synthetic_data'
BIAS = ['3prime_aa', '3prime_cc', '3prime_gg', '3prime_uu',
        '5prime_aa', '5prime_cc', '5prime_gg', '5prime_uu',
        'gc_fraction_gt_0p7', 'au_fraction_gt_0p7']
truth = next(pq.ParquetFile(DATA / 'artificial_ground_truth_kinetics_target_mean_one.parquet')
             .iter_batches(batch_size=128)).to_pandas()
ids = truth.transcript_id.tolist()
profiles = []
for bias in BIAS:
    frame = pq.read_table(
        DATA / f'bias_profile/artificial_bias_{bias}_compendium_added_bias_only.parquet',
        filters=[('transcript_id', 'in', ids), ('sample', '=', f'{bias}_rep1')]
    ).to_pandas().set_index('transcript_id')
    profiles.append([np.log1p(np.asarray(frame.loc[t, 'added_bias'], dtype=float)) for t in ids])

rows = []
max_cancel_error = 0.
for strategy in ['family_complete', 'latin0', 'latin1', 'latin2']:
    if strategy == 'family_complete':
        order = [(b, d) for b in range(10) for d in range(3)]
    else:
        rotation = int(strategy[-1])
        order = [(b, (b + p + rotation) % 3) for p in range(3) for b in range(10)]
    for n in range(3, 31, 3):
        chosen = order[:n]
        estimates = {}
        for policy in ['equal', 'quality_rank']:
            weights = np.array([1 if policy == 'equal' else d + 1 for b, d in chosen], dtype=float)
            weights /= weights.sum()
            errors, correlations, estimates[policy] = [], [], []
            for ti, k_raw in enumerate(truth.rib_profile):
                k = np.asarray(k_raw, dtype=float)
                k = k / k.mean()
                logs = np.stack([profiles[b][ti] for b, d in chosen])
                assert logs.shape[1] == len(k)
                center = weights @ logs
                l = k * np.exp(center)
                l /= l.mean()
                estimates[policy].append(l)
                correlations.append(np.corrcoef(l[5:-5], k[5:-5])[0, 1])
                errors.append(np.mean((l[5:-5] - k[5:-5]) ** 2))
            rows.append(dict(strategy=strategy, dataset_count=n, policy=policy,
                             transcripts=len(ids), oracle_pcc=np.mean(correlations),
                             oracle_rmse=np.sqrt(np.mean(errors))))
        if strategy == 'family_complete' or n == 30:
            max_cancel_error = max(max_cancel_error, max(
                np.max(np.abs(a - b)) for a, b in zip(estimates['equal'], estimates['quality_rank'])))

pd.DataFrame(rows).to_csv(OUT / 'oracle_gauge_results.tsv', sep='\t', index=False)
pd.DataFrame({'transcript_id': ids}).to_csv(OUT / 'oracle_transcript_ids.tsv', sep='\t', index=False)
print('Maximum cancellation error in normalized oracle profile:', max_cancel_error)
assert max_cancel_error < 1e-12
print(pd.DataFrame(rows).query('dataset_count in [3, 9, 30]').to_string(index=False))

# Separate finite-count precision from shared-factor identifiability: remove
# the known physical bias from the arithmetic replica mean. No model fitting.
depth_rows = []
for depth in ['0p25_per_codon', '2_per_codon', '20_per_codon']:
    counts = pq.read_table(
        DATA / depth / f'artificial_bias_3prime_aa_psite_counts_{depth}.parquet',
        filters=[('transcript_id', 'in', ids),
                 ('sample', 'in', ['3prime_aa_rep1', '3prime_aa_rep2'])]
    ).to_pandas().set_index(['transcript_id', 'sample'])
    pcc, mse = [], []
    for ti, (t, k_raw) in enumerate(zip(ids, truth.rib_profile)):
        observed = sum(np.asarray(counts.loc[(t, f'3prime_aa_rep{r}'), 'rib_profile'], dtype=float)
                       for r in [1, 2]) / 2
        corrected = observed / np.exp(profiles[0][ti])
        k = np.asarray(k_raw, dtype=float)
        k = k / k.mean()
        if corrected.mean() <= 0:
            continue
        corrected /= corrected.mean()
        pcc.append(np.corrcoef(corrected[5:-5], k[5:-5])[0, 1])
        mse.append(np.mean((corrected[5:-5] - k[5:-5]) ** 2))
    depth_rows.append(dict(depth=depth, transcripts=len(pcc),
                           corrected_observation_pcc=np.mean(pcc),
                           corrected_observation_rmse=np.sqrt(np.mean(mse))))
pd.DataFrame(depth_rows).to_csv(OUT / 'oracle_bias_corrected_depth_results.tsv', sep='\t', index=False)
print(pd.DataFrame(depth_rows).to_string(index=False))
