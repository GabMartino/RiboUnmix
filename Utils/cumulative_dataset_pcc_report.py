"""Per-dataset views of selected-checkpoint observed-profile mu PCC."""
from __future__ import annotations

import html
import json

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


def dataset_mu_grid(observed, weights, sizes, arms):
    """Keep dataset membership, unavailable models and missing metrics distinct.

    Mu can be reported without an L_bio tag: the paired diagnostic's `matched`
    flag is deliberately not used to decide whether a mu PCC is available.
    """
    keys = ['dataset_id', 'N', 'arm']
    metadata = weights[['dataset_id', 'global_rank']].drop_duplicates('dataset_id').sort_values('global_rank')
    index = pd.MultiIndex.from_product([metadata.dataset_id, sizes, arms], names=keys)
    membership = weights[keys].assign(in_subset=True)
    columns = keys + ['mu_pcc', 'status', 'selected_epoch']
    grid = membership.merge(observed[columns], on=keys, how='left', validate='one_to_one')
    grid = grid.set_index(keys).reindex(index).reset_index().merge(metadata, on='dataset_id', how='left')
    in_subset = grid.in_subset.eq(True)
    selected = grid.status.eq('validated_predictions') & np.isfinite(grid.selected_epoch)
    finite = np.isfinite(grid.mu_pcc)
    grid['cell_status'] = np.select(
        [~in_subset, ~selected, ~finite],
        ['not_in_subset', 'model_unavailable', 'mu_not_logged'], default='available')
    grid['mu_pcc'] = grid.mu_pcc.where(grid.cell_status == 'available')
    return grid[['dataset_id', 'global_rank', 'N', 'arm', 'mu_pcc', 'cell_status', 'selected_epoch']]


def dataset_spread_sentence(grid):
    equal = grid[(grid.arm == 'equal') & grid.mu_pcc.notna()]
    counts = equal.groupby('N').size()
    sizes = counts[counts >= 2].index
    if not len(sizes):
        return 'At least two datasets with selected-checkpoint μ PCC are needed to describe between-dataset spread.'
    n = max(sizes)
    values = equal[equal.N == n]
    low, high = values.loc[values.mu_pcc.idxmin()], values.loc[values.mu_pcc.idxmax()]
    return (f'At N={int(n)} with equal reference, μ PCC ranges from {low.mu_pcc:.3f} '
            f'({low.dataset_id}) to {high.mu_pcc:.3f} ({high.dataset_id}) across {len(values)} datasets, '
            f'whereas their mean is {values.mu_pcc.mean():.3f}; the mean conceals substantial differences in observed-profile fit.')


def _dataset_caption(frame, sizes, arms, cohort_note=None):
    available = frame[frame.mu_pcc.notna()]
    if available.empty:
        return 'No selected-checkpoint μ PCC is available for this dataset in the current snapshot.'
    if available.N.nunique() == 1:
        n = int(available.N.iloc[0])
        return (f'Only N={n} is available: μ PCC ranges from {available.mu_pcc.min():.3f} '
                f'to {available.mu_pcc.max():.3f} across {len(available)} policies, so an across-N trend cannot yet be assessed.')
    arm = next(arm for arm in arms if (available.arm == arm).any())
    values = available[available.arm == arm].sort_values('N')
    first, last = values.iloc[0], values.iloc[-1]
    return (f'For {arm}, μ PCC is {first.mu_pcc:.3f} at N={int(first.N)} and '
            f'{last.mu_pcc:.3f} at N={int(last.N)}; '
            + (cohort_note or 'validation transcript coverage changes across N.'))


def _dataset_table(frame, sizes, arms):
    labels = {'not_in_subset': 'Not included', 'model_unavailable': 'Pending', 'mu_not_logged': 'Not logged'}
    table = pd.DataFrame(index=pd.Index(sizes, name='N'), columns=arms, dtype=object)
    for row in frame.itertuples():
        table.loc[row.N, row.arm] = f'{row.mu_pcc:.4f}' if row.cell_status == 'available' else labels[row.cell_status]
    return '<div class="table">' + table.reset_index().to_html(index=False, border=0) + '</div>'


def _draw_dataset(ax, frame, sizes, arms, colors, limits, *, small=False, evaluation_label='Validation'):
    x = np.arange(len(sizes))
    entry = int(frame.loc[frame.cell_status != 'not_in_subset', 'N'].min())
    if sizes.index(entry) > 0:
        ax.axvspan(-.3, sizes.index(entry) - .5, color='#f0f2f5', zorder=0)
    for arm in arms:
        values = frame[frame.arm == arm].set_index('N').mu_pcc.reindex(sizes)
        ax.plot(x, values, 'o--' if 'reverse' in arm else 'o-', color=colors[arm],
                label=arm, markersize=3 if small else 5, linewidth=1.2 if small else 1.7)
    name = frame.dataset_id.iloc[0]
    rank = int(frame.global_rank.iloc[0])
    ax.set(title=f'#{rank}  {name}\nEnters at N={entry}', ylim=limits, xlim=(-.3, len(sizes) - .7),
           xticks=x, xticklabels=sizes, xlabel='Number of training datasets N', ylabel=f'{evaluation_label} PCC(μ, observed)')
    ax.tick_params(labelsize=8 if small else 10)
    ax.title.set_fontsize(9 if small else 12)
    ax.xaxis.label.set_size(8 if small else 10)
    ax.yaxis.label.set_size(8 if small else 10)
    ax.grid(alpha=.2)


def write_dataset_mu_report(observed, weights, sizes, arms, colors, out, *, evaluation_label='Validation', cohort_note=None):
    """Write all individual curves, overview pages and an HTML dataset selector."""
    grid = dataset_mu_grid(observed, weights, sizes, arms)
    grid.to_csv(out / 'dataset_mu_pcc_grid.csv', index=False)
    names = grid.dataset_id.drop_duplicates().tolist()
    available_names = grid.loc[grid.mu_pcc.notna(), 'dataset_id'].drop_duplicates().tolist()
    finite = grid.mu_pcc.dropna()
    limits = (min(0, np.floor(finite.min() * 10) / 10), max(.1, np.ceil(finite.max() * 10) / 10)) if len(finite) else (0, 1)
    views, options, pages = {}, [], []
    with plt.rc_context({'svg.fonttype': 'none', 'pdf.fonttype': 42}):
        for number, name in enumerate(names, start=1):
            frame = grid[grid.dataset_id == name]
            rank = int(frame.global_rank.iloc[0])
            available = frame.mu_pcc.notna().any()
            image = f'dataset_mu_pcc_{number:03d}.svg' if available else ''
            if available:
                fig, ax = plt.subplots(figsize=(10, 4.1))
                _draw_dataset(ax, frame, sizes, arms, colors, limits, evaluation_label=evaluation_label)
                ax.legend(loc='upper center', bbox_to_anchor=(.5, -.22), ncol=len(arms), frameon=False, fontsize=9)
                fig.tight_layout()
                fig.savefig(out / image, bbox_inches='tight')
                plt.close(fig)
            views[name] = dict(image=image, caption=_dataset_caption(frame, sizes, arms, cohort_note),
                               table=_dataset_table(frame, sizes, arms))
            label = f'#{rank} · {name}' + ('' if available else ' · awaiting results')
            options.append(f'<option value="{html.escape(name, quote=True)}">{html.escape(label)}</option>')

        if available_names:
            with PdfPages(out / 'dataset_mu_pcc_overview.pdf') as pdf:
                for start in range(0, len(available_names), 12):
                    batch = available_names[start:start + 12]
                    rows = int(np.ceil(len(batch) / 3))
                    fig, axes = plt.subplots(rows, 3, figsize=(13, 2.65 * rows + 1.1), squeeze=False)
                    for ax, name in zip(axes.flat, batch):
                        _draw_dataset(ax, grid[grid.dataset_id == name], sizes, arms, colors, limits, small=True, evaluation_label=evaluation_label)
                    for ax in list(axes.flat)[len(batch):]:
                        ax.axis('off')
                    handles, labels = axes[0, 0].get_legend_handles_labels()
                    fig.legend(handles, labels, loc='lower center', ncol=len(arms), frameon=False, fontsize=10)
                    fig.suptitle('One curve panel per dataset · identical PCC scale\n'
                                 'Gray = dataset not yet included; gaps = unavailable selected-checkpoint metrics', fontsize=12)
                    fig.tight_layout(rect=(0, .045, 1, .94))
                    stem = f'dataset_mu_pcc_overview_{start // 12 + 1:02d}'
                    fig.savefig(out / f'{stem}.svg', bbox_inches='tight')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                    values = grid[grid.dataset_id.isin(batch) & grid.mu_pcc.notna()]
                    n = int(values.N.max())
                    latest = values[values.N == n]
                    caption = (f'At N={n}, these datasets span μ PCC {latest.mu_pcc.min():.3f}–{latest.mu_pcc.max():.3f} '
                               'across available policies, with a shared axis making differences between datasets visible.')
                    pages.append(f'<img loading="lazy" src="{stem}.svg" alt="Individual dataset μ PCC curves, '
                                 f'overview page {start // 12 + 1}"><p>{caption}</p>')

    first = available_names[0] if available_names else names[0]
    initial = views[first]
    options_html = ''.join(options).replace(f'value="{html.escape(first, quote=True)}"',
                                            f'value="{html.escape(first, quote=True)}" selected', 1)
    img = (f'<img id="dataset-mu-image" src="{initial["image"]}" alt="Per-dataset μ PCC curves">'
           if initial['image'] else '<img id="dataset-mu-image" hidden alt="Per-dataset μ PCC curves">')
    pdf_link = '<a href="dataset_mu_pcc_overview.pdf">All dataset panels as a PDF</a> · ' if pages else ''
    cohort_text = (cohort_note if cohort_note else
        'Across N, even this individual-dataset view still uses changing validation transcript cohorts and is not a fixed held-out test.')
    section = f'''<section id="dataset-mu-pcc">
<h2>μ PCC for each individual dataset</h2>
<p class="note">{html.escape(dataset_spread_sentence(grid))}</p>
<p><b>{len(available_names)}/{len(names)} datasets have selected-checkpoint μ PCC in this snapshot.</b>
Each point averages transcript–dataset pairs within this one dataset; it never averages different datasets together.
Datasets are ordered by their frozen global quality rank, with one color per reference policy and the same PCC axis in every panel.</p>
<div style="display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin:18px 0">
<label for="dataset-mu-select"><b>Dataset</b></label>
<button type="button" id="dataset-mu-previous" aria-label="Previous dataset">←</button>
<select id="dataset-mu-select" style="font:inherit;max-width:100%;padding:7px">{options_html}</select>
<button type="button" id="dataset-mu-next" aria-label="Next dataset">→</button>
</div>
{img}<p id="dataset-mu-caption">{html.escape(initial['caption'])}</p>
<div id="dataset-mu-table">{initial['table']}</div>
<p>Gray plot regions and “Not included” cells precede a dataset's entry into the cumulative collection;
“Pending” means its model has no validated selected export, and “Not logged” means the selected model has no finite μ PCC tag.
These states are never represented as zero, and lines never bridge missing sizes.</p>
<p><b>Compare policies within the same dataset and N.</b> This exposes dataset-specific gains and losses that can cancel in a mean.
Differences between datasets also reflect their noise, coverage and biological or technical composition;
a lower PCC alone does not show that ranking failed, and quality rank does not imply a monotonic ordering of PCC.
{html.escape(cohort_text)}</p>
<h3>All datasets with available values</h3>
{''.join(pages) if pages else '<p>No completed dataset curves are available yet.</p>'}
<p>{pdf_link}<a href="dataset_mu_pcc_grid.csv">Values, ranks and missing-data states</a> ·
<a href="observed_fit_per_dataset.csv">Selected epochs and metric provenance</a></p>
</section>'''
    script = '''<script>
(() => {
  const views = __VIEWS__;
  const select = document.getElementById('dataset-mu-select');
  function showDataset() {
    const view = views[select.value];
    const image = document.getElementById('dataset-mu-image');
    image.hidden = !view.image;
    if (view.image) image.src = view.image; else image.removeAttribute('src');
    image.alt = __EVALUATION_LABEL__ + ' mu PCC for ' + select.value + ' across dataset counts and reference policies';
    document.getElementById('dataset-mu-caption').textContent = view.caption;
    document.getElementById('dataset-mu-table').innerHTML = view.table;
  }
  select.addEventListener('change', showDataset);
  for (const [id, delta] of [['dataset-mu-previous', -1], ['dataset-mu-next', 1]]) {
    document.getElementById(id).addEventListener('click', () => {
      select.selectedIndex = (select.selectedIndex + delta + select.length) % select.length;
      showDataset();
    });
  }
})();
</script>'''.replace('__VIEWS__', json.dumps(views, ensure_ascii=False).replace('<', '\\u003c')).replace('__EVALUATION_LABEL__', json.dumps(evaluation_label))
    return section + script
