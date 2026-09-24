#!/usr/bin/env python3
"""Build an offline HTML explanation from saved Exp8 design/QC manifests."""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import shlex
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Utils.real_exp8_stability import prepare_quality_pool, quality_mismatch


def read_json(path):
    return json.loads(path.read_text())


def esc(value):
    return html.escape(str(value), quote=True)


def table(headers, rows):
    return ('<div class="table-scroll"><table><thead><tr>' +
            ''.join(f'<th scope="col">{esc(h)}</th>' for h in headers) +
            '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(
                f'<td>{esc(cell)}</td>' for cell in row) + '</tr>' for row in rows) +
            '</tbody></table></div>')


def svg(figure, label):
    stream = io.StringIO()
    figure.savefig(stream, format='svg', bbox_inches='tight')
    plt.close(figure)
    body = stream.getvalue()
    body = body[body.index('<svg '):]
    return body.replace('<svg ', f'<svg role="img" aria-label="{esc(label)}" ', 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-root', type=Path,
        default=ROOT / 'results/my_exp8_a100_b32_20260906_114340')
    parser.add_argument('--ranked-root', type=Path,
        default=ROOT / 'results/real_exp8_L_stability_quality_rank_10components/cumulative_qrank10components_p1.0_seed42')
    parser.add_argument('--output', type=Path,
        help='Default: <ranked-root>/analysis_setup/exp8_cumulative_quality_rank_setup.html')
    args = parser.parse_args(argv)
    old_root, new_root = args.previous_root.resolve(), args.ranked_root.resolve()
    output = (args.output or new_root / 'analysis_setup/exp8_cumulative_quality_rank_setup.html').resolve()
    old, new = [read_json(p / 'experiment_manifest.json') for p in (old_root, new_root)]
    if new['experiment_design'] != 'cumulative_top_quality':
        raise ValueError('The new experiment must be the cumulative top-quality design.')
    old_quality, new_quality = [pd.read_csv(p / 'subset_quality_report.csv') for p in (old_root, new_root)]
    order = pd.read_csv(new_root / 'cumulative_dataset_order.csv')
    # Copied runs retain the cluster's absolute path in the saved manifest.
    ranking_path = new_root / Path(new.get('frozen_ranking_table', 'frozen_quality_ranking.tsv')).name
    ranking = pd.read_csv(ranking_path, sep='\t')
    if hashlib.sha256(ranking_path.read_bytes()).hexdigest() != new['ranking_sha256']:
        raise ValueError('Frozen ranking hash does not match the manifest.')
    # rank_component_count is metadata, not an eleventh QC component.
    component_columns = sorted(c for c in ranking.columns
                               if c.startswith('rank_') and c != 'rank_component_count')
    ranking_source = Path(new.get('ranking_source', ranking_path.name)).name
    pool, families, meta = prepare_quality_pool(pd.read_csv(new_root / 'dataset_quality_table.csv'))
    old_pool, _, old_meta = prepare_quality_pool(pd.read_csv(old_root / 'dataset_quality_table.csv'))
    # Verify that the overlay uses the same observed QC data and standardization.
    assert meta['quality_columns'] == old_meta['quality_columns']
    columns = meta['quality_columns']
    np.testing.assert_allclose(pool.set_index('dataset_name')[columns].sort_index(),
                               old_pool.set_index('dataset_name')[columns].sort_index(), rtol=1e-10, atol=1e-12)
    for manifest, report in ((old, old_quality), (new, new_quality)):
        for task in manifest['tasks']:
            actual = quality_mismatch(task['datasets'], pool, z_columns=meta['z_columns'])
            saved = report.loc[report.run_id == task['run_id'], 'quality_mismatch'].item()
            np.testing.assert_allclose(actual, saved, rtol=1e-9, atol=1e-12)
    seeds = sorted({t['training_seed'] for t in new['tasks']})
    seed = seeds[0]
    tasks = sorted([t for t in new['tasks'] if t['training_seed'] == seed], key=lambda t: t['N'])
    sizes = [t['N'] for t in tasks]
    if set(sizes) != set(old_quality.N):
        raise ValueError('Reports must use the same N grid.')
    references = {}
    for task in tasks:
        assert task['datasets'] == order.dataset.tolist()[:task['N']]
        references[task['N']] = read_json(new_root / task['directory'] / 'subset_manifest.json')['fixed_gamma_reference']
        np.testing.assert_allclose(sum(references[task['N']]['pi'].values()), 1.)
    current = new_quality.set_index('run_id').loc[[t['run_id'] for t in tasks]]
    common_old, common_new = [read_json(p / 'common_test_manifest.json')['common_test_ids'] for p in (old_root, new_root)]
    same_test = set(common_old) == set(common_new)

    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'svg.hashsalt': 'exp8-setup'}):
        fig, ax = plt.subplots(figsize=(10, 3.8), layout='constrained')
        x = np.arange(len(sizes))
        boxes = [old_quality.loc[old_quality.N == n, 'quality_mismatch'].values for n in sizes]
        ax.boxplot(boxes, positions=x, widths=.32, patch_artist=True,
                   boxprops={'facecolor': '#dbeafe', 'edgecolor': '#2563eb'},
                   medianprops={'color': '#1d4ed8'}, whiskerprops={'color': '#2563eb'},
                   capprops={'color': '#2563eb'}, flierprops={'markeredgecolor': '#2563eb'})
        ax.plot([], [], color='#2563eb', linewidth=6, label='Previous: quality-matched subsets (boxes)')
        ax.plot(x, current.quality_mismatch, 'o-', color='#c2410c',
                label='New: one top-quality prefix per N')
        ax.set(xticks=x, xticklabels=sizes, xlabel='Number of datasets N (equally spaced categories)',
               ylabel='QC mismatch to the full collection')
        ax.set_ylim(bottom=-.035)
        ax.grid(axis='y', alpha=.2)
        ax.legend(frameon=False, fontsize=9, loc='upper right')
        mismatch_svg = svg(fig, 'Actual QC mismatch: previous subset distributions versus new cumulative prefixes')
        fig, ax = plt.subplots(figsize=(10, 3.3), layout='constrained')
        for i, n in enumerate(sizes):
            previous = sizes[i-1] if i else 0
            ax.barh(i, len(order), color='#edf1f6', height=.62)
            if previous:
                ax.barh(i, previous, color='#93c5fd', height=.62)
            ax.barh(i, n-previous, left=previous, color='#0f766e', height=.62)
            ax.text(n+.8, i, f'+{n-previous}', va='center', fontsize=9)
        ax.set(yticks=range(len(sizes)), yticklabels=[f'N = {n}' for n in sizes],
               xlabel='Position in the quality-sorted ACTIVE dataset list', xlim=(0, len(order)+12))
        ax.invert_yaxis()
        ax.spines[['top', 'right']].set_visible(False)
        membership_svg = svg(fig, 'Nested top-N membership: previously included datasets and newly added datasets')

    overview, details = [], []
    for i, task in enumerate(tasks):
        n, names = task['N'], task['datasets']
        previous = sizes[i-1] if i else 0
        ref = references[n]
        selected = set(names)
        split_families = [source for source, members in families.items()
                          if selected & set(members) and not set(members) <= selected]
        old_values = old_quality.loc[old_quality.N == n, 'quality_mismatch']
        mismatch = float(current.loc[task['run_id'], 'quality_mismatch'])
        overview.append([n, f'1–{n}', previous, n-previous, len(old_values),
                         f'{old_values.median():.4f}', f'{mismatch:.4f}',
                         f"{ref['effective_reference_dataset_count']:.2f}", len(split_families)])
        member_rows = []
        for position, name in enumerate(names, 1):
            source = order.loc[order.dataset == name, 'source_identifier'].item()
            member_rows.append([position, name, f"{ref['global_rank'][name]:g}",
                'Kept' if position <= previous else 'Added', source, f"{ref['pi'][name]:.8f}"])
        details.append(f'<details><summary>N = {n}: keep {previous}, add {n-previous} '
            f'— full membership and weights</summary><p>Partial source families: '
            f'{esc(", ".join(split_families) or "none")}. '
            'Weights below are gamma-reference π, not loss weights.</p>' + table(
            ['Active position', 'Dataset', 'Frozen global rank', 'Membership', 'Source family', 'π within this N'],
            member_rows) + '</details>')

    top = order.dataset.tolist()
    original_image = old_root / 'analysis_partial/subset_quality_balance.png'
    original = ''
    if original_image.exists():
        encoded = base64.b64encode(original_image.read_bytes()).decode()
        original = ('<details><summary>View your original quality-balance PNG</summary>'
            f'<img alt="Original previous-Exp8 subset quality mismatch boxplot" src="data:image/png;base64,{encoded}"></details>')
    inactive = ranking.loc[~ranking.dataset.isin(order.dataset), ['dataset', 'quality_rank']]
    inactive_text = ', '.join(f'{row.dataset} (rank {row.quality_rank})' for row in inactive.itertuples())
    p = float(new['quality_rank_power'])
    max_rank = float(ranking.quality_rank.max())
    provenance = {'previous_root': str(old_root), 'ranked_root': str(new_root),
        'ranking_sha256': new['ranking_sha256'], 'seed_shown': seed,
        'frozen_ranking_table': str(ranking_path), 'ranking_source': ranking_source,
        'ranking_component_count': len(component_columns),
        'ranking_component_columns': component_columns,
        'same_heldout_ids': same_test, 'heldout_count': len(common_new),
        'tasks': tasks, 'gamma_references': references,
        'new_quality_mismatch': dict(zip(map(str, sizes), current.quality_mismatch.tolist()))}
    document = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exp8 — how cumulative quality-ranked subsets are constructed</title>
<style>
:root{font:16px/1.6 system-ui,-apple-system,Segoe UI,sans-serif;color:#172536;background:#f3f6fa}
*{box-sizing:border-box}body{margin:0}main{max-width:1120px;margin:auto;padding:40px 28px 64px}
h1{font-size:clamp(1.9rem,4vw,2.8rem);line-height:1.15;letter-spacing:-.035em;max-width:900px}
h2{font-size:1.5rem;margin:0 0 16px;line-height:1.3}h3{font-size:1.08rem}p{margin:12px 0}
.eyebrow{color:#0f766e;font-size:.8rem;font-weight:750;letter-spacing:.13em;text-transform:uppercase}
.lead{font-size:1.13rem;max-width:900px}.muted,figcaption{color:#526275;font-size:.9rem}
section{background:white;border:1px solid #dbe3ec;border-radius:14px;padding:26px;margin-top:24px}
.callout{background:#e9f5f2;border-left:4px solid #0f766e;padding:14px 18px;border-radius:6px;margin:18px 0}
.warning{background:#fff5e9;border-color:#c2410c}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}
.tag{display:inline-block;padding:4px 10px;border-radius:5px;background:#e7eef7;font-size:.82rem}
figure{margin:20px 0}svg,img{width:100%;height:auto;display:block}figcaption{margin-top:10px}
.table-scroll{overflow-x:auto;margin:16px 0}table{border-collapse:collapse;width:100%;font-size:.86rem}
th{background:#edf2f7;text-align:left;font-weight:650}th,td{padding:10px 12px;border-bottom:1px solid #dfe6ee}
td{font-variant-numeric:tabular-nums}tr:nth-child(even) td{background:#f8fafc}summary{cursor:pointer;font-weight:650;padding:14px 0}
details{border-top:1px solid #dbe3ec;margin-top:10px}code{font-size:.88em;overflow-wrap:anywhere}
pre{background:#172536;color:#f0f5fa;padding:16px;border-radius:8px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:.85rem}
.formula{font-family:Georgia,serif;font-size:1.13rem;background:#f3f6fa;padding:16px}
li{margin:8px 0}.source{font-size:.85rem;overflow-wrap:anywhere}a{color:#175e9c}
@media(max-width:700px){main{padding:24px 14px}section{padding:18px}.grid{grid-template-columns:1fr}}
@media print{body{background:white}main{max-width:none;padding:0}section{break-inside:avoid;border:0;padding:12px 0}details{break-inside:auto}pre{background:#f3f6fa;color:#172536}}
</style></head><body><main>
'''
    document += f'''<div class="eyebrow">Experiment 8 · setup audit</div>
<h1>From quality-matched subsets<br>to cumulative top-quality datasets</h1>
<p class="lead">The previous experiment tried to make small subsets <strong>representative of the full collection</strong>.
The new experiment instead starts with the highest-ranked datasets and progressively adds the next ones.</p>
<span class="tag">{len(order)} active datasets</span> <span class="tag">{len(tasks)} sizes · seed {seed} shown</span>
<span class="tag">Saved design/QC audit — not training results</span>
<div class="callout"><strong>“N datasets” is a model's training collection, not an optimizer mini-batch.</strong>
At each N, all selected datasets form the collection available throughout training. The training batch setting remains a separate parameter.</div>

<section><h2>1. What changed?</h2><div class="grid"><div><h3>Previous: quality-matched</h3>
<p>For N = 2, 5, 10, 20 and 40: three A/B pairs, i.e. six subsets at each size.
The two sides of each designated pair share neither datasets nor source families.
Different pairs can overlap. At N = 80 there are three large subsets; at N = 114 there is one full collection.</p>
<p>Subset selection sought similar observed QC statistics to the full pool, keeping source families intact.
It did <strong>not</strong> select the top N from the scalar ranking. The saved design has {len(old['tasks'])} tasks.</p>
<p>Gamma-reference weights: equal within each subset, π = 1/N.</p></div>
<div><h3>New: one nested chain</h3><p>For each training seed, there is exactly <strong>one subset per N</strong>:
2 → 5 → 10 → 20 → 40 → 80 → 114. No A/B pairs and no alternative random subsets at a given N.</p>
<p>The active datasets are sorted by frozen global quality rank (smaller is better), then dataset name for ties.
Take the first N entries. Every smaller subset is contained in every larger one.</p>
<p>Gamma-reference weights: quality ranked, with power p = {p:g}. The default has {len(new['tasks'])} tasks.</p></div></div>
<div class="callout warning">This changes <strong>both membership and gamma weighting</strong>. It is not a weighting-only version of the old subset experiment.</div></section>

<section><h2>2. How the datasets accumulate</h2>
<p><strong>N = 2:</strong> {esc(', '.join(top[:2]))}.<br>
<strong>N = 5:</strong> keep those two; add {esc(', '.join(top[2:5]))}.<br>
<strong>N = 10:</strong> keep those five; add {esc(', '.join(top[5:10]))}.</p>
<figure>{membership_svg}<figcaption>Blue: datasets already included at the previous size. Green: newly added datasets.
Gray: not yet included. These positions refer to the active list, not re-assigned global ranks.</figcaption></figure>
<p>The ranking table has {len(ranking)} rows, but only {len(order)} datasets are active in the configured collection.
Not active: {esc(inactive_text)}. This is a configuration exclusion, not an exclusion made by the ranked launcher.
Global ranks remain unchanged, so active position and global rank can differ near the end of the list.</p>
{table(['N', 'Active positions', 'Kept', 'Added', 'Old subsets', 'Old median mismatch', 'New mismatch', 'Effective π count', 'Partial families'], overview)}
<p class="muted">Effective π count = 1 / Σπ²; it describes reference-weight concentration, not independent sample size.
Partial families count source groups represented by only some of their active datasets.
The near-zero full-pool mismatch is floating-point round-off, displayed as 0.0000.</p></section>

<section><h2>3. What your quality-balance plot means now</h2>
<p>The y-axis in your old image is a <strong>distance from the full collection's QC distribution</strong>.
Lower means “more similar to the full pool,” not “higher quality.” Each old box summarizes the mismatch values of the subsets at that N,
not transcripts and not prediction PCC.</p>
<figure>{mismatch_svg}<figcaption>Recomputed and verified against the saved source tables; no values were extracted from the PNG.
The previous boxes show quartiles and the median, with whiskers to the most extreme values within 1.5 IQR and separate outliers.
The new design has one subset per N, so it is shown as points joined by a line—not a boxplot or uncertainty band.</figcaption></figure>
<div class="callout warning"><strong>The larger orange mismatch is expected.</strong> The new small subsets deliberately favor top-ranked datasets,
whereas the old subsets were chosen to resemble the entire pool. This does not establish worse dataset quality or worse model predictions.</div>
<p>For example, N = 2 changes from an old median mismatch of {old_quality.loc[old_quality.N == 2, 'quality_mismatch'].median():.3f}
to {float(current.loc[tasks[0]['run_id'], 'quality_mismatch']):.3f}. The new curve need not decrease at every step.
Both designs reach zero (up to numerical precision) when all active datasets are included.</p>
<details><summary>Exact QC mismatch calculation</summary>
<p>The four observed QC variables are log(1 + median read density), median positive-codon coverage,
number of eligible transcripts, and median replica PCC. They are standardized across the full active pool.
Missing replica-PCC values are median-imputed under the shared helper's rules; the saved reports use all four variables.</p>
<p>For each standardized variable, collect the subset-minus-full difference in its mean and half the differences
in its 25th, 50th and 75th percentiles. The mismatch is the root mean square of those 16 numbers.
The half factor is applied before squaring. This diagnostic uses observed weighted-artifact replica data; it is not the ranking score itself.</p>
<p>The old and new QC source tables were checked to agree numerically before making this comparison.
The new launcher saves this score for auditing, but does <strong>not</strong> optimize it when choosing subsets.</p></details>{original}</section>

<section><h2>4. Ranking, reference weights and training</h2>
<p>Saved ranking source: <code>{esc(ranking_source)}</code>. The hash-verified frozen table contains
{len(component_columns)} component-rank columns: {esc(', '.join(component_columns))}.</p>
<p>The six-component <code>HEK_riboseq_profile_quality_rank.tsv</code> and ten-component
<code>HEK_riboseq_profile_quality_rank_components.tsv</code> define different rankings.
The latter also includes mapping, contamination and replicate agreement. More components do not by themselves establish a better ranking.
This report uses the saved run's frozen table, not the current configuration defaults.</p>
<div class="formula">q<sub>d</sub> = (R − r<sub>d</sub> + 1) / R &nbsp;;&nbsp;
π<sub>d,N</sub> = q<sub>d</sub><sup>p</sup> / Σ<sub>j ∈ top-N</sub> q<sub>j</sub><sup>p</sup><br>
Here R = {max_rank:g} (the full frozen ranking table), p = {p:g}.</div>
<p>Ranks and q stay fixed as N grows; π is normalized within the current subset and therefore changes with N.
At N = 2 the weights are {references[2]['pi'][top[0]]:.6f} for {esc(top[0])} and
{references[2]['pi'][top[1]]:.6f} for {esc(top[1])}. They are close because ranks 1 and 2 have similar q values.</p>
<ul><li>π weights the fixed gamma-centering reference; it is <strong>not</strong> a new direct rank multiplier on the training loss.</li>
<li>Transcript–dataset reliability weights w<sub>dt</sub> remain separate; their reference parameters are fitted using training transcripts only.</li>
<li>Each N starts from its own initialization. The N = 5 model does not resume the N = 2 checkpoint.</li>
<li>The inspected old and new designs {'have exactly the same' if same_test else 'do not have the same'} held-out transcript IDs.
The new set contains {len(common_new):,} transcripts. Training/validation eligibility is task-specific; held-out IDs never enter reliability fitting.</li>
<li>More training seeds repeat the same dataset memberships with new model initializations; they do not generate new subset selections.</li></ul></section>

<section><h2>5. Exact membership and π for every size</h2>
<p>Expand a size to see every actual dataset, its original global rank, source group, and normalized gamma-reference weight.
“Kept” refers to membership from the previous size, not reused model parameters.</p>{''.join(details)}</section>

<section><h2>6. What this experiment can establish</h2>
<p>It asks: <strong>How stable are inferred shared profiles as we expand a quality-prioritized dataset collection under ranked gamma centering?</strong></p>
<p>It does not isolate the effect of weighting alone. Nested subsets, shared transcripts, and reused models in comparisons make the resulting estimates dependent.
Exact top-N boundaries may split source families; this is not the old source-disjoint reproducibility design.
PCC measures agreement of inferred profiles, not biological accuracy. The full ranked model is an empirical comparison reference, not ground truth.</p>
<div class="callout">If the intended question is instead “What changes when only gamma weighting changes?”, the controlled design would keep the old dataset memberships,
splits and training settings and change only π. That is a different experiment; this report has not changed the current launcher.</div></section>

<section><h2>7. Reproduce and inspect</h2><p>Launch the new experiment on the cluster:</p>
<pre>sbatch run_real_exp8_L_stability_quality_rank.slurm</pre>
<p>Regenerate this offline report from its saved source tables, from the repository root:</p>
<pre>{esc(shlex.join(['.venv/bin/python', 'analyses/create_exp8_quality_rank_setup_report.py', '--previous-root', str(old_root), '--ranked-root', str(new_root), '--output', str(output)]))}</pre>
<p class="source"><strong>Previous source:</strong> {esc(old_root)}<br>
<strong>Ranked design source:</strong> {esc(new_root)}<br>
<strong>Rank SHA-256:</strong> <code>{esc(new['ranking_sha256'])}</code></p>
<p class="muted">Sources: experiment manifests, subset manifests, cumulative_dataset_order.csv, dataset_quality_table.csv,
subset_quality_report.csv, {esc(ranking_path.name)} and common_test_manifest.json.
All diagrams and membership tables are embedded; no internet, JavaScript or external fonts are required.</p></section>
'''
    document += '<script type="application/json" id="source-data">' + json.dumps(provenance).replace('<', '\\u003c') + '</script>'
    document += '</main></body></html>'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding='utf-8')
    print(f'Wrote {output} ({output.stat().st_size:,} bytes)')
    print(f'Verified {len(old["tasks"])} previous + {len(new["tasks"])} new mismatch values; same test IDs: {same_test}.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
