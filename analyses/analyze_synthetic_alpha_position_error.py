#!/usr/bin/env python3
"""Frozen alpha, relative-CDS position and observed-count error diagnostics.

Keep only the matched validation profiles for one run and fixed-bin/scalar
summaries. No codon-level DataFrame, smoothing, training or checkpoint loading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Utils.publication_plot_style import publication_rc
from analyses.audit_synthetic_read_depth import rows, array_hash
from analyses.analyze_synthetic_recovery import _encoding_from_config
from analyses.analyze_synthetic_alpha_depth_recovery import verify_simulator, TRIM
from analyses.plot_synthetic_depth_recovery_overview import AUDIT_DIR, STYLES
from analyses.plot_synthetic_read_depth_effect import (
    DEPTHS, COUNTS, verified_runs, cohort_hash, file_hash, mean_stats,
)

KEYS = ["run", "depth", "n_datasets", "transcript_id"]
PROFILE_METRICS = ("alpha_mean", "consensus_mse", "replica_mse", "reference_standardized_error",
                   "consensus_reference_standardized_error")
ASSOCIATIONS = ("alpha_consensus_error_pcc", "alpha_standardized_consensus_error_pcc",
                "alpha_replica_error_pcc", "alpha_standardized_replica_error_pcc")
ASSOCIATION_TARGETS = ("consensus_mse", "consensus_reference_standardized_error",
                       "replica_mse", "reference_standardized_error")
STEM = "synthetic_alpha_position_error"


def position_bins(length, bins):
    if length < bins or bins < 2:
        raise ValueError("Require at least two bins and one position per bin.")
    u = np.arange(length)/(length-1)
    return u, np.minimum((u*bins).astype(int), bins-1)


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return np.nan, "short_or_nonfinite"
    dx, dy = x-x.mean(), y-y.mean()
    for a, delta, name in ((x, dx, "alpha"), (y, dy, "error")):
        if np.linalg.norm(delta) <= 1e-10*np.sqrt(len(a))*max(1., abs(a.mean())):
            return np.nan, f"nearly_constant_{name}"
    return float(np.clip(dx@dy/(np.linalg.norm(dx)*np.linalg.norm(dy)), -1, 1)), "ok"


def error_profiles(mu, target, shape, replicas, eps, alpha_truth):
    """Reproduce likelihood replica means, including its observed full-domain scale."""
    mu, target, shape, replicas = map(lambda x: np.asarray(x, dtype=float), (mu, target, shape, replicas))
    if (replicas.ndim != 2 or replicas.shape[1:] != mu.shape or target.shape != mu.shape
            or shape.shape != mu.shape or not all(np.isfinite(a).all() for a in (mu, target, shape, replicas))
            or any((a < 0).any() for a in (mu, target, shape, replicas))):
        raise ValueError("Invalid count, mean, or shape arrays.")
    if not np.allclose(target, replicas.mean(axis=0), rtol=1e-6, atol=1e-6):
        raise ValueError("Exported target is not the arithmetic mean of the supplied raw replicas.")
    scales = np.maximum(replicas.mean(axis=1), eps)
    mu_replicas = scales[:, None]*shape[None, :]
    if not np.allclose(mu_replicas.mean(axis=0), mu, rtol=2e-5, atol=2e-6):
        raise ValueError("Reconstructed replica means disagree with the exported consensus mu.")
    mu_replicas = np.maximum(mu_replicas, eps)
    squared = (mu_replicas-replicas)**2
    variance = mu_replicas+alpha_truth*mu_replicas**2
    consensus_squared = (mu-target)**2
    return dict(consensus_mse=consensus_squared, replica_mse=squared.mean(axis=0),
                reference_standardized_error=(squared/variance).mean(axis=0),
                consensus_reference_standardized_error=consensus_squared/(variance.sum(axis=0)/len(replicas)**2))


def load_replicas(run, ids, lengths):
    """Only selected validation replicas, one experiment at a time."""
    selected, digest, sources = {}, hashlib.sha256(), []
    for relative in run["training_paths"]:
        path = ROOT / relative
        dataset = path.stem
        sources.append(dict(path=str(path.resolve()), size_bytes=path.stat().st_size,
                            mtime_ns=path.stat().st_mtime_ns))
        with pq.ParquetFile(path) as reader:
            for batch in reader.iter_batches(columns=["id", "ribo_cds_replicas"], batch_size=32, use_threads=False):
                # Never materialize unselected nested replica arrays in Python.
                for i, tid in enumerate(batch.column("id").to_pylist()):
                    if tid not in ids:
                        continue
                    key = tid, dataset
                    values = np.asarray(batch.column("ribo_cds_replicas")[i].as_py(), dtype=float)
                    if (key in selected or values.shape != (2, lengths[tid]+1)
                            or not np.isfinite(values).all() or (values < 0).any() or (values[:, -1] != 0).any()):
                        raise ValueError(f"Invalid/duplicate raw replicas: {key}")
                    selected[key] = values
                    digest.update((tid+dataset+array_hash(values)).encode())
    if set(selected) != {(tid, d) for tid in ids for d in run["datasets"]}:
        raise ValueError(f"Missing observed raw replicas in {run['run']}")
    return selected, dict(selected_replicas_sha256=digest.hexdigest(), observed_sources=sources)


def evaluate_run(run, ids, lengths, truth, bins):
    config = yaml.load((ROOT/run["config_path"]).read_text(), Loader=yaml.CSafeLoader)
    loss = config["loss"]
    if config["model"].get("alpha_mode") != "learned" or loss["experiment_mode"] != "standard_nb":
        raise ValueError("Need the frozen learned-alpha standard-NB configuration.")
    lower, upper = loss["nb_log_alpha_min"], loss["nb_log_alpha_max"]
    replicas, check = load_replicas(run, ids, lengths)
    encoding = _encoding_from_config(config, ROOT)
    positions, profiles, seen, digest = [], [], set(), hashlib.sha256()
    columns = ["transcript_id", "dataset_id", "length", "mask", "log_sigma", "mu", "target", "normalized_shape"]
    for row in rows(ROOT/run["prediction_path"], columns, batch_size=16):
        tid = row["transcript_id"]
        if tid not in ids:
            continue
        key = tid, encoding[int(row["dataset_id"])]
        length = lengths[tid]+1
        mask = np.asarray(row["mask"], dtype=bool)
        if (key in seen or key not in replicas or int(row["length"]) != length
                or len(mask) < length or not mask[:length].all() or mask[length:].any()):
            raise ValueError(f"Invalid identity/alignment: {key}")
        seen.add(key)
        arrays = {name: np.asarray(row[name][:length], dtype=float)
                  for name in ("log_sigma", "mu", "target", "normalized_shape")}
        if any(len(a) != length or not np.isfinite(a).all() for a in arrays.values()):
            raise ValueError(f"Nonfinite or truncated predictions: {key}")
        for name, a in arrays.items():
            digest.update((tid+key[1]+name+array_hash(a)).encode())
        errors = error_profiles(arrays["mu"], arrays["target"], arrays["normalized_shape"],
                                replicas.pop(key), loss["eps"], truth)
        alpha = np.exp(np.clip(arrays["log_sigma"], lower, upper))[:-1]
        values = {"alpha_mean": alpha, **{name: a[:-1] for name, a in errors.items()}}
        identity = dict(run=run["run"], depth=run["depth"], n_datasets=run["n_datasets"],
                        transcript_id=tid, dataset=key[1])
        _, labels = position_bins(lengths[tid], bins)
        for b in range(bins):
            take = labels == b
            positions.append({**identity, "position_bin": b, "n_positions": int(take.sum()),
                              **{name: float(a[take].mean()) for name, a in values.items()}})
        for domain, take in (("full_sense", slice(None)), ("interior10", slice(TRIM, -TRIM))):
            record = {**identity, "domain": domain, "n_positions": len(alpha[take]),
                      **{name: float(a[take].mean()) for name, a in values.items()}}
            for metric, target in zip(ASSOCIATIONS, ASSOCIATION_TARGETS):
                value, reason = correlation(alpha[take], values[target][take])
                record[metric], record[metric+"_reason"] = value, reason
            profiles.append(record)
    if replicas:
        raise ValueError(f"Missing predictions: {list(replicas)[:5]}")
    check.update(run=run["run"], selected_predictions_sha256=digest.hexdigest(),
                 log_alpha_bounds=[lower, upper], eps=loss["eps"], targets_and_replica_means_verified=len(seen))
    return positions, profiles, check


def aggregate(frame, extra, metrics):
    records = []
    for identity, group in frame.groupby(KEYS+extra, sort=True):
        if len(group) != identity[2] or group.dataset.nunique() != identity[2]:
            raise ValueError("Incomplete dataset panel in diagnostic aggregation.")
        records.append(dict(zip(KEYS+extra, identity)) | {
            name: float(group[name].mean()) if np.isfinite(group[name]).all() else np.nan
            for name in metrics})
    return pd.DataFrame(records)


def summarize(frame, ids, group_keys, metrics, draws):
    records = []
    for identity, group in frame.groupby(group_keys, sort=True):
        cell = group.set_index("transcript_id").reindex(ids)
        for metric in metrics:
            records.append(dict(zip(group_keys, identity)) | dict(metric=metric, n_transcripts=len(ids),
                           cohort_hash=cohort_hash(ids), **mean_stats(cell[metric], draws)))
    return pd.DataFrame(records)


def plot_position(summary, out, bins, font_size):
    style = publication_rc()
    style.update({"font.size": font_size, "axes.labelsize": font_size,
                  "xtick.labelsize": font_size, "ytick.labelsize": font_size,
                  "legend.fontsize": font_size, "axes.titlesize": font_size+1,
                  "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
                  "axes.linewidth": 1.2, "xtick.major.width": 1.2,
                  "ytick.major.width": 1.2, "grid.linewidth": 0.75})
    if style["text.usetex"]:
        style["text.latex.preamble"] += r"\renewcommand{\seriesdefault}{\bfdefault}\AtBeginDocument{\boldmath}"
    with matplotlib.rc_context(style):
        fig, axes = plt.subplots(2, 3, figsize=(12.8, 6.7), sharex=True, sharey="row")
        fig.subplots_adjust(left=.08, right=.98, bottom=.10, top=.86, hspace=.30, wspace=.28)
        for col, depth in enumerate(DEPTHS):
            for row, metric in enumerate(
                ("alpha_mean", "consensus_reference_standardized_error")
            ):
                ax = axes[row, col]
                for n in (2, 10):
                    cell = summary.loc[(summary.depth == depth) & (summary.n_datasets == n)
                                       & (summary.metric == metric)].sort_values("position_bin")
                    x = (cell.position_bin+.5)/bins
                    ax.plot(x, cell["mean"], color=STYLES[depth]["color"], linewidth=3.0,
                            linestyle="--" if n == 2 else "-", alpha=.65 if n == 2 else 1.)
                    ax.fill_between(x, cell.ci_low, cell.ci_high, color=STYLES[depth]["color"], alpha=.09)
                ax.set(xlim=(0, 1), xticks=[0, .25, .5, .75, 1])
                ax.grid(axis="y")
                ax.set_axisbelow(True)
                if row == 0:
                    ax.axhline(.1, linestyle=":", color="#666666", linewidth=1.6)
                    limits = summary.loc[(summary.metric == metric) & summary.n_datasets.isin([2, 10])]
                    ax.set_ylim(min(.09, limits.ci_low.min()*.95), limits.ci_high.max()*1.05)
                    ax.set_title(f"{chr(65+col)}  {STYLES[depth]['label']}", loc="left")
                else:
                    ax.set_ylim(bottom=0)
                    ax.axhline(1.0, linestyle=":", color="#666666", linewidth=1.6)
                    ax.set_xlabel("Relative CDS position")
                    ax.set_title(f"{chr(68+col)}  Std. residual", loc="left")
                if col == 0:
                    ax.set_ylabel(
                        r"Mean $\widehat\alpha$"
                        if row == 0
                        else "Std. squared residual",
                        fontsize=17,
                        labelpad=9,
                    )
        fig.legend(handles=[Line2D([], [], color="#333333", linestyle=ls, linewidth=3.0, label=f"N={n}")
                            for n, ls in ((2, "--"), (10, "-"))], loc="upper center", ncol=2,
                   bbox_to_anchor=(.5, .985))
        for suffix in ("pdf", "png", "svg"):
            fig.savefig(out/f"{STEM}.{suffix}", dpi=600)
        plt.close(fig)
    return {key: style[key] for key in ("text.usetex", "font.size", "font.weight")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT_DIR)
    parser.add_argument("--alpha-dir", type=Path, default=AUDIT_DIR/"alpha_recovery")
    parser.add_argument("--position-bins", type=int, default=20)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--font-size", type=float, default=18.)
    args = parser.parse_args()
    if args.bootstrap_repeats < 2 or args.position_bins < 2 or not np.isfinite(args.font_size) or args.font_size <= 0:
        parser.error("Require positive font size, at least two bins and two bootstrap draws.")
    runs, validation = verified_runs(args.audit_dir)
    ids_path = args.alpha_dir/"matched_transcript_ids.csv"
    ids = sorted(pd.read_csv(ids_path).transcript_id)
    if not ids or not set(ids).issubset(set.intersection(*validation.values())):
        raise ValueError("Cohort is not validation-only in every run.")
    lengths_path = args.audit_dir/"depth_recovery_overview/shared_profile_per_transcript.csv"
    length_table = pd.read_csv(lengths_path, usecols=["transcript_id", "sense_length"])
    lengths = length_table.groupby("transcript_id").sense_length.first().to_dict()
    if args.position_bins > min(lengths[t] for t in ids):
        parser.error("More bins than codons in the shortest selected transcript.")
    truth, _ = verify_simulator()
    position_rows, profile_rows, checks = [], [], []
    for run in runs:
        positions, profiles, check = evaluate_run(run, set(ids), lengths, truth, args.position_bins)
        position_rows.extend(positions)
        profile_rows.extend(profiles)
        checks.append(check)
        print(f"position/error {run['depth']} N={run['n_datasets']}: {len(profiles)//2} verified profiles", flush=True)
    position_detail, profile_detail = pd.DataFrame(position_rows), pd.DataFrame(profile_rows)
    del position_rows, profile_rows
    position_transcripts = aggregate(position_detail, ["position_bin"], PROFILE_METRICS)
    profile_transcripts = aggregate(profile_detail, ["domain"], (*PROFILE_METRICS, *ASSOCIATIONS))
    draws = np.random.default_rng(args.bootstrap_seed).integers(len(ids), size=(args.bootstrap_repeats, len(ids)))
    position_summary = summarize(position_transcripts, ids, ["depth", "n_datasets", "position_bin"], PROFILE_METRICS, draws)
    profile_summary = summarize(profile_transcripts, ids, ["depth", "n_datasets", "domain"], PROFILE_METRICS, draws)
    # All association summaries use one valid transcript intersection, never varying per cell.
    invalid = profile_transcripts.loc[~np.isfinite(profile_transcripts[list(ASSOCIATIONS)]).all(axis=1), "transcript_id"]
    association_ids = sorted(set(ids)-set(invalid))
    association_summary = pd.DataFrame()
    if association_ids:
        a_draws = np.random.default_rng(args.bootstrap_seed).integers(len(association_ids),
                    size=(args.bootstrap_repeats, len(association_ids)))
        association_summary = summarize(profile_transcripts, association_ids,
            ["depth", "n_datasets", "domain"], ASSOCIATIONS, a_draws)
    out = args.alpha_dir/"position_error"
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in (("position_per_transcript_dataset", position_detail), ("profile_per_transcript_dataset", profile_detail),
                        ("position_per_transcript", position_transcripts), ("profile_per_transcript", profile_transcripts),
                        ("position_summary", position_summary), ("profile_summary", profile_summary),
                        ("association_summary", association_summary)):
        frame.to_csv(out/f"{name}.csv", index=False)
    typography = plot_position(position_summary, out, args.position_bins, args.font_size)
    command = f"RIBOUNMIX_PLOT_TEX={int(typography['text.usetex'])} "+shlex.join(
        [sys.executable, str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]])
    sources = [Path(__file__), ids_path, lengths_path, ROOT/"Models/RiboUnmixLightningModule.py",
               ROOT/"analyses/analyze_synthetic_alpha_depth_recovery.py", ROOT/"analyses/audit_synthetic_read_depth.py",
               ROOT/"analyses/plot_synthetic_read_depth_effect.py", ROOT/"analyses/analyze_synthetic_recovery.py"]
    provenance = dict(command=command, runs=runs, checks=checks, source_sha256={str(p): file_hash(p) for p in sources},
        cohort=dict(n=len(ids), hash=cohort_hash(ids)), association_cohort=dict(n=len(association_ids), ids=association_ids),
        association_excluded_ids=sorted(set(ids)-set(association_ids)), bins=args.position_bins,
        relative_position="zero-based i/(sense_length-1); equal-width [left,right), final bin includes u=1",
        plotted_N=[2,10], evaluated_N=list(COUNTS), plot_domain="full sense CDS; terminal export entry excluded",
        sensitivity_domain="exclude first/last 10 sense codons", raw_counts=True, smoothing=False,
        standardized_error="mean_r[(mu_replica_r - Y_r)^2/(mu_replica_r+0.1*mu_replica_r^2)]",
        standardized_consensus_error="(mu - mean_r Y_r)^2 / (sum_r[mu_r + 0.1*mu_r^2]/R^2)",
        replica_mean="max(mean_i(Y_r on full model domain including terminal),eps)*exported normalized_shape",
        fitting_or_inference=False, typography=typography,
        max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        bootstrap=dict(repeats=args.bootstrap_repeats, seed=args.bootstrap_seed, cluster="transcript", paired=True))
    (out/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    (out/"commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n"+command+"\n")
    write_report(out, position_summary, profile_summary, association_summary, ids, association_ids, args, command)
    print(f"Figure: {out/(STEM+'.pdf')}; {len(ids)} profile transcripts; {len(association_ids)} association transcripts")


def write_report(out, position_summary, profile_summary, associations, ids, association_ids, args, command):
    endpoint = profile_summary.loc[(profile_summary.n_datasets == 10) & (profile_summary.domain == "interior10")]
    table = endpoint.pivot(index="depth", columns="metric", values="mean")
    association_table = (associations.loc[(associations.n_datasets == 10) & (associations.domain == "interior10")]
                         .pivot(index="depth", columns="metric", values="mean") if len(associations) else pd.DataFrame())
    association_findings = []
    if len(associations):
        for metric in ASSOCIATIONS:
            cell = associations.loc[(associations.n_datasets == 10) & (associations.domain == "interior10")
                                    & (associations.metric == metric)]
            containing_zero = int(((cell.ci_low <= 0) & (cell.ci_high >= 0)).sum())
            association_findings.append(f"* {metric}: means {cell['mean'].min():.4f} to {cell['mean'].max():.4f}; "
                                        f"{containing_zero}/3 bootstrap intervals include zero.")
    a = position_summary.loc[(position_summary.n_datasets == 10) & (position_summary.metric == "alpha_mean")]
    middle_bins = [b for b in range(args.position_bins) if .4 <= (b+.5)/args.position_bins < .6]
    regions = []
    for depth in DEPTHS:
        cell = a.loc[a.depth == depth].set_index("position_bin")
        regions.append(dict(depth=depth, first_bin=cell.loc[0, "mean"],
                            middle_40_to_60_percent=cell.loc[middle_bins, "mean"].mean() if middle_bins else np.nan,
                            last_bin=cell.loc[args.position_bins-1, "mean"]))
    regions = pd.DataFrame(regions).set_index("depth")
    regions.to_csv(out/"alpha_position_regions_N10.csv")
    (out/"README.md").write_text(f"""# Alpha, relative CDS position and observed-count error

![Position and error]({STEM}.png)

The top row shows mean effective alpha across relative CDS position; the bottom
row shows the consensus squared residual divided by its variance under the
simulator NB2 model. Columns are the three read depths. N=2 and N=10 are
shown as preselected endpoints; all N=2,...,10 are in the source tables. Mean-alpha
and standardized-residual axes are shared across depths. The dotted references
are simulator alpha 0.1 and standardized squared residual one. Shading is
pointwise 95% transcript-bootstrap
uncertainty, conditional on these fitted models, not variability across seeds.

## Exact definitions

Use full sense CDS, excluding the artificial terminal export entry. Relative
position is u=i/(n_t-1), zero based. Use {args.position_bins} fixed equal-width bins,
including u=1 in the last bin, with no smoothing, interpolation or zero filtering.
First average codons within each transcript/dataset/bin, then datasets equally
within transcript, then transcripts equally. Long transcripts therefore do not
dominate the plots. All bins and conditions use the same {len(ids)} validation
transcripts. The bootstrap uses {args.bootstrap_repeats} paired transcript draws,
seed {args.bootstrap_seed}, across depths, panel sizes, bins and metrics.

* Effective alpha: exp(clip(log_sigma, configured loss bounds)).
* Consensus squared error: (mu_i - mean_r Y_ri)^2.
* Replica squared error: mean_r (mu_ri - Y_ri)^2.
* Replica means: mu_ri = max(mean_j Y_rj, eps) * saved normalized_shape_i,
  using all mask-valid model positions including the terminal entry to reconstruct
  the original likelihood scale. Evaluate only sense positions afterwards.
* Reference-standardized squared error:
  mean_r [(mu_ri-Y_ri)^2/(mu_ri + 0.1*mu_ri^2)]. The denominator uses the fixed
  simulator alpha, NOT learned alpha; an association with alpha is therefore not
  manufactured by putting that same alpha in the denominator.
* Reference-standardized CONSENSUS squared error:
  (mu_i - mean_r Y_ri)^2 / [sum_r(mu_ri + 0.1*mu_ri^2)/R^2], with R=2.
  This uses the variance of the replica average, not the variance of one replica.
  It provides a normalization comparison without changing the error target.

Every target is checked against its processed raw-replica arithmetic mean. The
average reconstructed replica means must match the saved consensus mu. Profile
summary tables contain both full-sense and 10-codon-trimmed values; do not confuse
the full-sense positional plots with the trimmed alpha-recovery overview.

## Does alpha track prediction error?

Within EACH transcript/dataset, compute PCC across positions between alpha and
(a) consensus squared error, (b) standardized consensus squared error,
(c) replica squared error, and (d) standardized replica squared error.
Average dataset PCCs within transcript, then transcript PCCs; do not pool codons.
Undefined correlations receive explicit reason codes, never zero substitution.
All association summaries use one complete-case intersection across N, depths,
metrics and full/interior domains: {len(association_ids)} transcripts.
These correlations are well-defined when BOTH profiles vary, unlike a PCC
between alpha and the constant 0.1 reference. They are descriptive associations,
not evidence that prediction errors causally drive alpha. Position, sequence,
mean level and training dynamics may confound them.

## Actual results

At N=10 the start/end bins show elevated alpha relative to the central 40--60%
region. The elevation is not confined to the terminal export entry, which was
excluded. The N=2 start pattern differs, especially at the lowest depth, so this
is not a uniform boundary rule for every fitted model.

The scored observations are sense-CDS P-site positions, not explicit UTR count
positions. There is nevertheless a concrete boundary-context mismatch in the
historical simulator: the programmed 30-nt RPF features for the first and last
five sense codons can depend on flanking UTR sequence, whereas the fitted model
receives CDS sequence only. Elevated edge residuals and alpha are therefore
consistent with unavailable UTR context. This is a mechanistic explanation, not
a completed causal ablation; attributing every edge effect to UTR context would
require regenerating matched data with boundary features removed or giving the
model the same flanking sequence.

```text
{regions.to_string(float_format=lambda x: f'{x:.6f}')}
```

N=10 interior association results, retaining each error target before and after
variance scaling:

{chr(10).join(association_findings)}

A weak local association does not exclude global dispersion inflation from
mean-model mismatch. Raw and standardized associations should be compared within
the same error target; consensus and replica errors are separate diagnostics.

N=10, trimmed-interior means:

```text
{table.to_string(float_format=lambda x: f'{x:.6f}')}
```

N=10, trimmed within-profile associations (mean PCC):

```text
{association_table.to_string(float_format=lambda x: f'{x:.6f}')}
```

## Interpretation limits

Raw MSE, retained in the source tables, includes observation noise and grows with
the count scale. It is NOT a clean estimator-error metric, and larger raw MSE at
greater depth does not imply worse biological recovery. The plotted standardized diagnostic adjusts for the NB2
mean/variance scale, but is not an oracle: mu is fitted, the per-replica scale
uses observed counts, and the two simulator traffic profiles are not identical.
A value of one is a heuristic calibration reference, not an exact null expectation
under this fitted, plug-in construction. Two replicas cannot establish a precise
per-position dispersion independently. This remains one training seed, 27 small-
intersection validation sequences and validation-selected checkpoints.

## Reproduce

```bash
{command}
```

`position_summary.csv` contains all positional means and intervals, including
replica MSE and standardized error. `association_summary.csv` contains correlation
intervals; the per-transcript/dataset tables retain validity reasons and counts.
No training, diagnostic fitting, smoothing or modification of pi/w_dt is performed.
""")
    (out/"caption.tex").write_text(r"""\textbf{Positional structure of learned dispersion and scale-adjusted count error.}
\textbf{(A--C)} Mean effective NB2 dispersion and \textbf{(D--F)} consensus squared
residual divided by $\sum_r(\mu_{dtri}+0.1\mu_{dtri}^2)/R^2$, the variance of the
two-replicate mean under the simulator NB2 model, across relative sense-CDS
position $u=i/(n_t-1)$.
Columns show 0.25, 2 and 20 expected reads/codon; dashed and solid curves show
the preselected $N=2$ and $N=10$ cumulative panels. The dotted alpha reference is
the simulator value 0.1; the dotted standardized-residual reference is one.
Fixed equal-width bins cover the full sense CDS, with
the terminal export entry excluded; no smoothing or observed-zero filtering is
applied. Bin means are averaged equally over datasets within transcript and
then transcripts. Both rows share their ranges across depths. The first/last five
sense codons can use programmed 30-nt RPF contexts extending into UTR sequence,
which is absent from the CDS-only model input; the displayed edge elevation is
consistent with this context mismatch but is not a causal ablation. Raw count MSE
is retained only in the numeric source tables.
"""+f"All curves use the same {len(ids)} validation transcripts and {args.position_bins} bins. "
        +f"Shading indicates pointwise 95\\% intervals from {args.bootstrap_repeats:,} paired transcript-bootstrap "
        +"resamples, conditional on the frozen models.\n")


if __name__ == "__main__":
    main()
