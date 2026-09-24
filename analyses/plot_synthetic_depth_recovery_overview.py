#!/usr/bin/env python3
"""Two-by-two read-depth overview with one recovery metric per panel.

A/B: mean transcript PCC/RMSE(L, H), respectively.
C/D: gamma PCC/RMSE on the direct multiplier scale, respectively.
H = mean_one(K * geometric_mean(injected_biases)) is a diagnostic proxy,
NOT the biological target K or an unnoised TASEP occupancy oracle.

Shared metrics reuse audited scalars; gamma exports are streamed and cached
as scalar metrics. No training or inference.
The older recovery overview and its outputs are deliberately left unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Utils.publication_plot_style import publication_rc
from analyses.plot_synthetic_read_depth_effect import (
    COUNTS, DEPTHS, cohort_hash, complete_cohort, file_hash,
    mean_stats, verified_runs,
)
from analyses.analyze_synthetic_gamma_depth_recovery import (
    PLOTTED_METRICS as GAMMA_METRICS, load_or_analyze_gamma, write_gamma_summaries,
)

AUDIT_DIR = ROOT / "analyses/artifacts/synthetic/read_depth"
STEM = "synthetic_recovery_overview_depth_effect"
PROXY_METRICS = ("pcc_Kg_trim10", "rmse_Kg_trim10")
CONTROL_METRICS = ("pcc_K_trim10", "rmse_K_trim10")
STYLES = {
    DEPTHS[0]: dict(color="#0072B2", marker="o", label="0.25 reads/codon"),
    DEPTHS[1]: dict(color="#E69F00", marker="s", label="2 reads/codon"),
    DEPTHS[2]: dict(color="#009E73", marker="^", label="20 reads/codon"),
}
FIGURE_ASPECT = 1.55


def load_shared_metrics(path, runs, candidates):
    """Stream a scalar CSV, retaining only the common validation intersection."""
    columns = ["run", "depth", "n_datasets", "variant", "transcript_id",
               "sense_length", *PROXY_METRICS, *CONTROL_METRICS]
    names = {run["run"] for run in runs}
    parts = []
    with pd.read_csv(path, usecols=columns, chunksize=5000) as reader:
        for chunk in reader:
            keep = ((chunk.variant == "best_pcc") & chunk.run.isin(names)
                    & chunk.transcript_id.isin(candidates))
            parts.append(chunk.loc[keep].copy())
    frame = pd.concat(parts, ignore_index=True)
    if frame.duplicated(["transcript_id", "depth", "n_datasets"]).any():
        raise ValueError("Shared metrics contain duplicate transcript/model rows.")
    return frame


def summarize_shared(frame, ids, repeats, seed):
    """Equal-transcript means; paired transcript bootstrap across all conditions."""
    if not ids:
        raise ValueError("There are no common validation transcripts with valid proxy metrics.")
    draws = np.random.default_rng(seed).integers(len(ids), size=(repeats, len(ids)))
    lookup = frame.set_index(["depth", "n_datasets", "transcript_id"]).sort_index()
    rows = []
    for depth in DEPTHS:
        for n in COUNTS:
            cell = lookup.loc[(depth, n)].reindex(ids)
            for metric in (*PROXY_METRICS, *CONTROL_METRICS):
                rows.append(dict(depth=depth, n_datasets=n, metric=metric,
                                 displayed=metric in PROXY_METRICS,
                                 n_transcripts=len(ids), cohort_hash=cohort_hash(ids),
                                 **mean_stats(cell[metric], draws)))
    return pd.DataFrame(rows)


def plot_metric(ax, summary, metric, scale):
    """One y-axis and one metric; equal-transcript means with bootstrap intervals."""
    is_rmse = "rmse" in metric
    for depth in DEPTHS:
        cell = summary.loc[(summary.depth == depth) & (summary.metric == metric)]
        cell = cell.set_index("n_datasets").loc[list(COUNTS)]
        ax.errorbar(COUNTS, cell["mean"],
                    yerr=[cell["mean"]-cell.ci_low, cell.ci_high-cell["mean"]],
                    **STYLES[depth], linestyle="--" if is_rmse else "-",
                    linewidth=2.7*scale, markersize=6.7*scale,
                    markerfacecolor="white" if is_rmse else STYLES[depth]["color"],
                    markeredgecolor=STYLES[depth]["color"] if is_rmse else "white",
                    markeredgewidth=(1.6 if is_rmse else .9)*scale,
                    capsize=2.6*scale, elinewidth=1.35*scale, zorder=3)
    ax.set(xlim=(1.6, 10.4), xticks=list(COUNTS),
           xlabel="Number of training datasets",
           ylabel="Mean RMSE" if is_rmse else "Mean PCC")
    cell = summary.loc[summary.metric == metric]
    if is_rmse:
        ax.set_ylim(0, np.ceil(cell.ci_high.max()*1.10*100)/100)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    else:
        lower = np.floor((cell.ci_low.min()-.0005)*1000)/1000
        ax.set_ylim(lower, 1.0005)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    # Keep ticks representable by the fixed decimal formatter (no .0015/.025 steps).
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax.grid(axis="y")
    ax.set_axisbelow(True)


def plot_overview(summary, gamma_summary, out, width, font_size=14.):
    style = publication_rc()
    scale = width / 12.8
    style.update({"font.size": font_size, "axes.labelsize": font_size,
                  "axes.titlesize": font_size+1, "xtick.labelsize": font_size,
                  "ytick.labelsize": font_size, "legend.fontsize": font_size,
                  "axes.labelpad": 3, "font.weight": "bold",
                  "axes.labelweight": "bold", "axes.titleweight": "bold"})
    if style["text.usetex"]:
        # Also bold TeX-rendered ticks and mathematical symbols.
        style["text.latex.preamble"] += (
            r"\renewcommand{\seriesdefault}{\bfdefault}\AtBeginDocument{\boldmath}")
    for key in ("font.size", "axes.labelsize", "axes.titlesize", "xtick.labelsize",
                "ytick.labelsize", "legend.fontsize", "axes.labelpad",
                "axes.linewidth", "xtick.major.width", "ytick.major.width", "grid.linewidth"):
        style[key] *= scale
    for key in ("xtick.major.size", "ytick.major.size", "xtick.major.pad", "ytick.major.pad"):
        style[key] = matplotlib.rcParamsDefault[key] * scale
    with matplotlib.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(width, width / FIGURE_ASPECT))
        fig.subplots_adjust(left=.085, right=.975, bottom=.09, top=.88,
                            wspace=.28, hspace=.40)
        panels = (
            (summary, PROXY_METRICS[0], r"A  Shared $L_t$ vs proxy $H_t^{(N)}$: PCC"),
            (summary, PROXY_METRICS[1], r"B  Shared $L_t$ vs proxy $H_t^{(N)}$: RMSE"),
            (gamma_summary, GAMMA_METRICS[0],
             r"C  $\widetilde\gamma_{dt}$ vs $b_{dt}^{\mathrm{ref}}$: PCC"),
            (gamma_summary, GAMMA_METRICS[1],
             r"D  $\widetilde\gamma_{dt}$ vs $b_{dt}^{\mathrm{ref}}$: RMSE"),
        )
        for ax, (source, metric, title) in zip(axes.flat, panels):
            plot_metric(ax, source, metric, scale)
            ax.set_title(title, loc="left", pad=10*scale)
        handles = [Line2D([], [], **STYLES[depth], linewidth=2.7*scale, markersize=6.7*scale)
                   for depth in DEPTHS]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .985),
                   ncol=3, columnspacing=2, handletextpad=.5, handlelength=2.1)
        fig.savefig(out / f"{STEM}.pdf")
        fig.savefig(out / f"{STEM}.png", dpi=600)
        fig.savefig(out / f"{STEM}.svg")
        plt.close(fig)
    return {key: style[key] for key in ("text.usetex", "font.family", "font.size", "font.weight")}


def write_documentation(out, summary, gamma_summary, n, excluded, repeats, command, typography):
    caption = rf"""\textbf{{Read depth, shared-profile agreement and gamma recovery.}}
Models use seed 42, frozen best-validation-PCC checkpoints, and the original
transcript--dataset reliability weights. For the cumulative experiments,
$\pi_d=1/N$, $N=2,\ldots,10$, in the fixed bias order $3'$-AA, CC, GG, UU;
$5'$-AA, CC, GG, UU; GC-rich; AU-rich.
Define $G_{{ti}}^{{(N)}}=\prod_{{d=1}}^N b_{{dti}}^{{1/N}}$ and
$H_{{ti}}^{{(N)}}=K_{{ti}}G_{{ti}}^{{(N)}}/
[\ell_t^{{-1}}\sum_{{j=1}}^{{\ell_t}}K_{{tj}}G_{{tj}}^{{(N)}}]$.
The same $H_t^{{(N)}}$ is used at all three depths for a given transcript and $N$;
it is a kinetic-plus-reference-bias diagnostic proxy, not the unbiased kinetic
target $K_t$ or a noiseless TASEP occupancy oracle.
Because $H_t^{{(N)}}$ changes with $N$, compare depths at fixed $N$; slopes across
$N$ are not a fixed-target sample-complexity estimate.
\textbf{{(A,B)}} Arithmetic means of transcript-level PCC$(L_t,H_t^{{(N)}})$ and
RMSE$(L_t,H_t^{{(N)}})$, respectively, shown on separate axes.
The proxy is not the biological target $K_t$.
\textbf{{(C,D)}} PCC and RMSE, respectively, for recovery of the reference-centered injected bias. Let
$\mathcal I_t=\{{11,\ldots,\ell_t-10\}}$ and define the two-way operator
$[\mathcal C_{{\pi,\mathcal I_t}}(a)]_{{di}}
=a_{{di}}-\sum_e\pi_ea_{{ei}}-\bar a_{{d,\mathcal I_t}}
+\sum_e\pi_e\bar a_{{e,\mathcal I_t}}$, with positional means taken on $\mathcal I_t$.
Both frozen learned and known log profiles are re-centered on this same interior:
$\widetilde\gamma=\exp[\mathcal C_{{\pi,\mathcal I_t}}(\log\gamma)]$,
$b^{{\mathrm{{ref}}}}=\exp[\mathcal C_{{\pi,\mathcal I_t}}(\log b)]$.
PCC and RMSE are scored on the direct multiplier scale, not the log scale,
then averaged equally over datasets within transcript and equally over transcripts.
This evaluates interior-relative gamma shape and magnitude under the specified
centering, not the absolute calibration of the unmodified native gamma exports.
Saved model outputs and parameters are never changed. The native full-domain gauge
is checked, and a full-sense-gauge-then-trim sensitivity analysis is also supplied.
All four panels use the same {n} transcripts held out at all three depths, fixed across
every $N$, with the first and last 10 sense codons excluded. The single terminal
boundary entry is removed from the saved $L_t$ without renormalizing the retained
values; $H_t$ is normalized before trimming. Error bars are pointwise 95\%
percentile intervals from {repeats:,} paired transcript-bootstrap resamples,
conditional on these trained models. The left column (A,C) shows PCC with solid
lines and filled markers (higher is better); the right column (B,D) shows RMSE
with dashed lines and open markers (lower is better). Each panel has one y-axis.
Both PCC axes are magnified and both RMSE axes start at zero; y-axis ranges differ
between the shared-profile and gamma rows. No single-dataset count-prediction
results are included. No undefined PCCs were replaced by zero in the displayed
data. $N=1$ is excluded because
reference centering is bypassed there. All results are validation results, not
independent test or multi-seed results. Better agreement with $H_t$ is not evidence
of better recovery of $K_t$, whose matched scores are also supplied in the source table.
"""
    (out / "caption.tex").write_text(caption)
    endpoint = summary.loc[summary.n_datasets.isin([2, 10]) & summary.displayed]
    table = endpoint.pivot(index=["n_datasets", "depth"], columns="metric", values="mean")
    gamma_endpoints = gamma_summary.loc[gamma_summary.n_datasets.isin([2, 10]) & gamma_summary.displayed]
    gamma_table = gamma_endpoints.pivot(index=["n_datasets", "depth"], columns="metric", values="mean")
    amplitude = pd.read_csv(out / "gamma_recovery/gamma_amplitude_by_dataset.csv")
    cc = amplitude.loc[(amplitude.depth == "2_per_codon") & (amplitude.n_datasets == 10)
                       & (amplitude.dataset == "artificial_bias_5prime_cc")].iloc[0]
    control = gamma_summary.loc[gamma_summary.metric.isin(
        ["gamma_rmse", "sense_gauge_gamma_rmse"])].pivot(
            index=["n_datasets", "depth"], columns="metric", values="mean")
    gauge_delta = control.sense_gauge_gamma_rmse-control.gamma_rmse
    (out / "README.md").write_text(f"""# Read-depth overview: shared-profile and gamma recovery

![Read-depth overview]({STEM}.png)

The 2-by-2 layout separates outcomes and metrics: A is PCC(L,H), B is RMSE(L,H),
C is PCC(gamma,b_ref), and D is RMSE(gamma,b_ref). The gamma vectors are aligned
to the same interior log-centering convention as detailed below. Blue/orange/green
depth colors and circle/square/triangle markers are shared across all four panels.
Solid/filled curves show PCC; dashed/open curves show RMSE. Each panel has one
y-axis, with no secondary axes or single-dataset count-prediction results.
The original downloaded figure is not overwritten. PDF and SVG are vector exports;
PNG is 600 dpi. The canvas has a 1.55 width/height ratio,
at an editable 12.8-inch default width (`--width-in`).
All text, including labels, ticks, legends and mathematical notation, is bold.
Labels and ticks in this export are {typography['font.size']:g} pt before manuscript
resizing. Use `--font-size 16` to increase text (default 14 pt at 12.8-inch width);
titles are one point larger at that width. Typography scales with `--width-in`.

## What is being tested?

For each N, the diagnostic target is H = mean-one(K times the geometric mean of
the N injected bias multipliers). K is the programmed kinetic target before
traffic and count sampling. H is fixed across depths, so its reference noise does
not increase at low depth. This is a post-hoc diagnostic, not a target used for
fitting or choosing reference weights. H still contains injected bias and is not
the simulator's unavailable noiseless occupancy q. Do not call this biological
ground-truth recovery or a pure estimation-error/irreducible-error decomposition.
H changes when the bias panel changes with N. Depth comparisons therefore apply
within a fixed N; slopes across N are not a fixed-target sample-complexity curve.

There are {n} common validation transcripts, with {len(excluded)} additional
complete-case exclusions. The cohort is fixed across 27 cumulative models
(three depths, all N=2,...,10). No codon pooling, smoothing, Fisher aggregation,
observed-zero filtering, or post-trim normalization of L is applied. For each transcript,
RMSE is sqrt(mean((L-H)^2)) on the retained codons; plotted values average these
transcript RMSEs, not their squared errors. PCC is also averaged across transcripts.
The same bootstrap transcript indices are used across all depths, N, and metrics.

The 0.25-depth models agree less closely with H, while 2 and 20 reads/codon can be
much closer to each other. The plots do not force a large separation between every
depth. The original K metrics are preserved as unplotted controls in
`shared_profile_summary.csv` (displayed=False), allowing the reader to check that
changing the target does not establish a deterioration in K recovery.

Endpoint means from the exact plotted cohort:

```text
{table.to_string(float_format=lambda x: f'{x:.6f}')}
```

## Gamma recovery: panels C and D

The known bias is b = 1 + added_bias, from the simulator's depth-independent bias
annotations. Replica 1, replica 2, and mean annotations are verified to agree.
The new analysis uses the **same best-PCC checkpoints, matched transcripts, and
10-codon interior as A/B**, not the old best-loss / trim-5 / log-scale summaries.

Gamma has two log-centering constraints: a reference-weighted mean of zero at
each position, and a positional mean of zero for each dataset. Bias annotations
omit the terminal boundary, whereas the native model uses sense CDS plus that
boundary. Therefore both log(gamma) and log(b) are re-centered on the same trimmed
sense interior using the repository's documented two-way operator. Exponentiating
these centered profiles gives the two multiplier vectors compared in C/D. This is
an evaluation-only alignment of relative gamma profiles, not a change to the
model or evidence for unmodified gamma's absolute count calibration.

For each transcript and dataset we compute direct-scale PCC and RMSE, then average
datasets within transcript before averaging transcripts. Bootstrap resampling
clusters all dataset rows of a transcript together and is paired across depths/N.
Undefined dataset PCCs are recorded, never replaced by zero or silently omitted
from a panel mean. All four panels use a common complete-case cohort across all conditions.

Endpoint gamma means:

```text
{gamma_table.to_string(float_format=lambda x: f'{x:.6f}')}
```

The depth ordering is not uniformly monotonic. At N=2, gamma PCC and RMSE both
improve from 0.25 to 2 to 20 reads/codon. At N=10, the 2-depth model has better
PCC but worse RMSE than the 0.25-depth model. PCC measures centered shape, whereas
RMSE also penalizes multiplier-amplitude errors; high PCC alone is not evidence
of accurate bias magnitudes. Compare depths within each N because the reference
panel and its centered bias target change with N.

One concrete example is 5-prime CC at N=10 and 2 reads/codon: mean gamma PCC is
{cc.gamma_pcc:.6f}, but RMSE is {cc.gamma_rmse:.6f}. The mean within-transcript
predicted/reference variance ratio is {cc.variance_ratio:.3f}, and the mean
linear calibration slope is {cc.calibration_slope:.3f}. This is consistent with
compressed multiplier variation despite highly correlated spatial patterns.
The slope is computed diagnostically as PCC times the predicted/reference
standard-deviation ratio; it does not modify the model. It does not establish
why this particular training run differs from another depth's run.

Using a full-sense gauge before trimming instead of gauging on the interior
changes mean RMSE by {gauge_delta.mean():.6f} on average across the 27 conditions
(range {gauge_delta.min():.6f} to {gauge_delta.max():.6f}). This sensitivity check
does not evaluate unobserved terminal bias or restore an absolute gamma target.

`gamma_recovery/gamma_per_transcript_dataset.csv` retains every dataset-level score,
variance, validity flag and reason. `gamma_per_transcript.csv` contains the
within-transcript dataset means. `gamma_summary.csv` includes primary multiplier
metrics, log-scale controls, and a sensitivity check that gauges on the full sense
CDS before trimming. `gamma_by_dataset_summary.csv` resolves individual biases;
`gamma_depth_differences.csv` contains paired high-minus-low contrasts and pointwise
95% intervals. `gamma_amplitude_by_dataset.csv` reports the variance ratios and
calibration slopes. Native gauge-constraint residuals and input provenance are also saved.
The plotting script automatically creates or verifies this compact gamma cache;
use `--recompute-gamma` to rebuild it from prediction exports.

## Scope and limitations

The former single-dataset panel has been removed. This workflow no longer reads
its predictions, summary files, or manifests. Any `single_dataset_*.csv` files
remaining in an existing output directory are legacy artifacts, not sources for
this four-panel figure.

All models use training seed 42. The same validation split selects the checkpoint
and supplies these scores. There is no independent test set or replication over
training seeds; the simulator reuses traffic trajectories across conditions. The
small three-depth intersection and slightly different training identities limit
causal/general claims. The larger paired low/high comparison remains in
`../depth_effect_figure/`, but is not mixed into these three-depth curves.

## Reproduce

From the repository root:

```bash
{command}
```

To rebuild the upstream audit, see `analyses/audit_synthetic_read_depth.py --help`.
`caption.tex` gives the complete mathematical definition and aggregation details.
The [MathJax audit report](../read_depth_audit.html) explains the underlying model,
the proxy's motivation, and why agreement with K can change little.
`provenance.json` records input hashes, checkpoint/configuration identities,
cohort hashes, exclusions and bootstrap settings. All source tables contain
scalar transcript metrics, never a codon-level pandas table.
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=AUDIT_DIR / "depth_recovery_overview")
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--width-in", type=float, default=12.8)
    parser.add_argument("--font-size", type=float, default=18.,
                        help="Base text size in points at 12.8-inch width (default: 18; bold).")
    parser.add_argument("--recompute-gamma", action="store_true",
                        help="Recompute gamma metrics from the frozen exports instead of using a verified cache.")
    args = parser.parse_args()
    if args.bootstrap_repeats < 2 or not all(
            np.isfinite(v) and v > 0 for v in (args.width_in, args.font_size)):
        parser.error("Require at least two bootstrap draws and finite positive width/font size.")
    runs, validation = verified_runs(args.audit_dir)
    candidates = set.intersection(*validation.values())
    frame = load_shared_metrics(args.audit_dir / "recovery_per_transcript.csv", runs, candidates)
    ids, excluded = complete_cohort(frame, candidates, DEPTHS, PROXY_METRICS)
    shared_excluded = set(excluded)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    gamma_detail, gamma_transcripts, gamma_provenance = load_or_analyze_gamma(
        runs, set(ids), out / "gamma_recovery", args.recompute_gamma)
    ids, gamma_excluded = complete_cohort(gamma_transcripts, set(ids), DEPTHS, GAMMA_METRICS)
    excluded = sorted(set(excluded) | set(gamma_excluded))
    gamma_lengths = gamma_detail.groupby("transcript_id").sense_length.first()
    shared_lengths = frame.groupby("transcript_id").sense_length.first()
    if not gamma_lengths.equals(shared_lengths.reindex(gamma_lengths.index)):
        raise ValueError("Shared and gamma recovery have different sense-CDS alignment.")
    frame = frame.loc[frame.transcript_id.isin(ids)].sort_values(
        ["depth", "n_datasets", "transcript_id"])
    summary = summarize_shared(frame, ids, args.bootstrap_repeats, args.bootstrap_seed)
    gamma_summary = write_gamma_summaries(gamma_detail, gamma_transcripts, ids, out / "gamma_recovery",
                                          args.bootstrap_repeats, args.bootstrap_seed)
    frame.to_csv(out / "shared_profile_per_transcript.csv", index=False)
    summary.to_csv(out / "shared_profile_summary.csv", index=False)
    pd.DataFrame({"transcript_id": ids}).to_csv(out / "matched_transcript_ids.csv", index=False)
    pd.DataFrame({"transcript_id": excluded,
                  "reason": ["Missing or nonfinite shared metric" if tid in shared_excluded
                             else "Missing or nonfinite gamma metric" for tid in excluded]}
                 ).to_csv(out / "exclusions.csv", index=False)
    typography = plot_overview(summary, gamma_summary, out, args.width_in, args.font_size)
    command = f"RIBOUNMIX_PLOT_TEX={int(typography['text.usetex'])} " + shlex.join([
        str(Path(sys.executable).relative_to(ROOT)) if Path(sys.executable).is_relative_to(ROOT)
        else sys.executable, str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]])
    write_documentation(out, summary, gamma_summary, len(ids), excluded,
                        args.bootstrap_repeats, command, typography)
    sources = [args.audit_dir / name for name in ("provenance.json", "recovery_per_transcript.csv")]
    sources += [Path(__file__), ROOT / "analyses/plot_synthetic_read_depth_effect.py",
                ROOT / "analyses/audit_synthetic_read_depth.py", ROOT / "Utils/publication_plot_style.py",
                ROOT / "analyses/analyze_synthetic_gamma_depth_recovery.py",
                out / "gamma_recovery/provenance.json", out / "gamma_recovery/gamma_summary.csv"]
    provenance = dict(
        command=command, sources={str(p.resolve()): file_hash(p) for p in sources},
        training_seed=42, checkpoint_variant="best_pcc", gamma_reference="uniform, pi=1/N",
        training_reliability_weights="unchanged", inference_or_training_performed=False,
        proxy="H=mean_one_sense_CDS(K*exp(mean_d(log(b_d)))); b=1+added_bias",
        prediction_normalization="saved L unchanged; remove terminal boundary only",
        gamma_analysis=gamma_provenance,
        plotted_panels=dict(A="shared L vs H: PCC", B="shared L vs H: RMSE",
                            C="interior-recentered gamma vs injected reference bias: PCC",
                            D="interior-recentered gamma vs injected reference bias: RMSE"),
        boundary_trim_codons=10, statistic="arithmetic mean of transcript PCCs or transcript RMSEs",
        bootstrap=dict(repeats=args.bootstrap_repeats, seed=args.bootstrap_seed,
                       paired_across_depths_N_and_metrics=True, interval="pointwise percentile 95%",
                       uncertainty="validation transcript resampling, conditional on fitted models"),
        cohort=dict(n_candidates=len(candidates), n=len(ids), hash=cohort_hash(ids), excluded=excluded),
        runs=runs, typography=typography, single_dataset_results_included=False,
        figure_width_inches=args.width_in, aspect_ratio=FIGURE_ASPECT,
        limitations=["H is not K or a q-based oracle", "validation, not independent test",
                     "only 27 transcripts in the three-depth split intersection",
                     "one training seed; depth-specific training identities"])
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (out / "commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n")
    print(f"A-D: {len(ids)} matched validation transcripts, {len(runs)} frozen models; "
          f"{len(excluded)} complete-case exclusions.")
    print(f"Figure: {out / (STEM + '.pdf')}")


if __name__ == "__main__":
    main()
