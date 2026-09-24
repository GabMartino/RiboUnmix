#!/usr/bin/env python3
"""Audit NB2 dispersion recovery from frozen exports, without model loading.

Same cumulative uniform-reference runs and validation cohort as the L/gamma
overview. Stream small Arrow batches; retain only transcript/dataset scalars.
PCC against a constant alpha is undefined and is deliberately not computed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
from pathlib import Path
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Utils.publication_plot_style import publication_rc
from analyses.audit_synthetic_read_depth import rows, array_hash
from analyses.analyze_synthetic_recovery import _encoding_from_config
from analyses.plot_synthetic_read_depth_effect import (
    BIAS_ORDER, COUNTS, DEPTHS, cohort_hash, complete_cohort, file_hash, mean_stats, verified_runs,
)
from analyses.plot_synthetic_depth_recovery_overview import AUDIT_DIR, STYLES

TRIM = 10
STEM = "synthetic_alpha_depth_recovery"
METRICS = ("alpha_mean", "alpha_bias", "alpha_rmse", "alpha_mae", "alpha_spatial_sd",
           "mean_log2_ratio", "mean_absolute_log2_ratio", "fraction_within_factor_two",
           "fraction_at_floor", "fraction_at_cap")


def alpha_scores(log_sigma, truth, lower, upper):
    """Score the effective likelihood dispersion, not the unclamped head output."""
    log = np.asarray(log_sigma, dtype=np.float64)
    if not np.isfinite(truth) or truth <= 0 or not np.isfinite([lower, upper]).all() or lower > upper:
        raise ValueError("Invalid truth or log-alpha bounds.")
    if log.ndim != 1 or not len(log) or not np.isfinite(log).all():
        return dict(valid=False, reason="empty_or_nonfinite_log_sigma",
                    **{metric: np.nan for metric in METRICS})
    alpha = np.exp(np.clip(log, lower, upper))
    error = alpha-truth
    ratio = np.log2(alpha/truth)
    return dict(valid=True, reason="ok", alpha_mean=float(alpha.mean()),
                alpha_bias=float(error.mean()), alpha_rmse=float(np.sqrt(np.mean(error**2))),
                alpha_mae=float(np.mean(np.abs(error))), alpha_spatial_sd=float(alpha.std()),
                mean_log2_ratio=float(ratio.mean()), mean_absolute_log2_ratio=float(np.abs(ratio).mean()),
                fraction_within_factor_two=float(np.mean(np.abs(ratio) <= 1+1e-12)),
                fraction_at_floor=float(np.mean(log <= lower)),
                fraction_at_cap=float(np.mean(log >= upper)))


def verify_simulator():
    """Read the actual per-condition simulator metadata, not an assumed target."""
    sources, alphas = {}, set()
    for depth in DEPTHS:
        for bias in BIAS_ORDER:
            path = ROOT / "Datasets/Synthetic_data" / depth / f"artificial_bias_{bias}_psite_counts_{depth}.parquet"
            with pq.ParquetFile(path) as reader:
                meta = {k.decode(): v.decode() for k, v in reader.schema_arrow.metadata.items()
                        if k.startswith(b"riboart.")}
            if meta["riboart.observation_model"] != "negative_binomial_NB2":
                raise ValueError(f"Not NB2: {path}")
            alphas.add(float(meta["riboart.negative_binomial_dispersion_alpha"]))
            sources[str(path.relative_to(ROOT))] = dict(metadata=meta, size_bytes=path.stat().st_size)
    if alphas != {0.1}:
        raise ValueError(f"Expected one generative dispersion of 0.1, found {alphas}")
    return alphas.pop(), sources


def evaluate_run(run, ids, lengths, truth):
    config = yaml.load((ROOT / run["config_path"]).read_text(), Loader=yaml.CSafeLoader)
    loss = config["loss"]
    if (config["model"].get("alpha_mode") != "learned" or loss["experiment_mode"] != "standard_nb"
            or loss["replica_nb_weight"] <= 0 or loss["nb_mean_gradient_beta"] != 0):
        raise ValueError(f"Not a standard raw-replica learned-alpha run: {run['run']}")
    lower, upper = float(loss["nb_log_alpha_min"]), float(loss["nb_log_alpha_max"])
    if not lower < np.log(truth) < upper:
        raise ValueError("Simulator alpha is outside the learnable interval.")
    encoding = _encoding_from_config(config, ROOT)
    path = ROOT / run["prediction_path"]
    columns = ["transcript_id", "dataset_id", "length", "mask", "log_sigma"]
    with pq.ParquetFile(path) as reader:
        missing = set(columns)-set(reader.schema_arrow.names)
        if missing:
            raise ValueError(f"Missing saved alpha inputs {missing}: {path}")
    records, seen, digest = [], set(), hashlib.sha256()
    for row in rows(path, columns, batch_size=16):
        tid = row["transcript_id"]
        if tid not in ids:
            continue
        dataset = encoding[int(row["dataset_id"])]
        key = tid, dataset
        length = lengths[tid]+1  # Model export includes one terminal boundary.
        if key in seen or dataset not in run["datasets"] or int(row["length"]) != length:
            raise ValueError(f"Duplicate, wrong dataset, or misaligned alpha: {run['run']}/{key}")
        mask = np.asarray(row["mask"], dtype=bool)
        log = np.asarray(row["log_sigma"][:length], dtype=np.float64)
        if len(log) != length or len(mask) < length or not mask[:length].all() or mask[length:].any():
            raise ValueError(f"Invalid export mask or missing alpha positions: {key}")
        seen.add(key)
        digest.update((tid+dataset+array_hash(log)).encode())
        score = alpha_scores(log[TRIM:lengths[tid]-TRIM], truth, lower, upper)
        records.append(dict(run=run["run"], depth=run["depth"], n_datasets=run["n_datasets"],
                            transcript_id=tid, dataset=dataset, sense_length=lengths[tid],
                            n_positions=lengths[tid]-2*TRIM, **score))
    expected = {(tid, dataset) for tid in ids for dataset in run["datasets"]}
    if seen != expected:
        raise ValueError(f"Missing alpha exports in {run['run']}: {sorted(expected-seen)[:5]}")
    return records, dict(run=run["run"], log_alpha_bounds=[lower, upper],
                         effective_alpha_bounds=[float(np.exp(lower)), float(np.exp(upper))],
                         selected_log_sigma_sha256=digest.hexdigest(),
                         prediction_size_bytes=path.stat().st_size,
                         prediction_mtime_ns=path.stat().st_mtime_ns)


def aggregate_transcripts(detail):
    """Equal datasets within each transcript; never silently omit invalid rows."""
    result = []
    keys = ["run", "depth", "n_datasets", "transcript_id"]
    for identity, group in detail.groupby(keys, sort=True):
        if len(group) != identity[2] or group.dataset.nunique() != identity[2]:
            raise ValueError("Incomplete or duplicate transcript-dataset alpha panel.")
        result.append(dict(zip(keys, identity)) | {
            metric: float(group[metric].mean()) if np.isfinite(group[metric]).all() else np.nan
            for metric in METRICS})
    return pd.DataFrame(result)


def summarize(frame, ids, repeats, seed):
    draws = np.random.default_rng(seed).integers(len(ids), size=(repeats, len(ids)))
    lookup = frame.set_index(["depth", "n_datasets", "transcript_id"]).sort_index()
    return pd.DataFrame([
        dict(depth=depth, n_datasets=n, metric=metric, n_transcripts=len(ids),
             cohort_hash=cohort_hash(ids), **mean_stats(lookup.loc[(depth, n)].reindex(ids)[metric], draws))
        for depth in DEPTHS for n in COUNTS for metric in METRICS])


def plot(summary, out, truth, font_size):
    style = publication_rc()
    style.update({"font.size": font_size, "axes.labelsize": font_size,
                  "xtick.labelsize": font_size, "ytick.labelsize": font_size,
                  "legend.fontsize": font_size, "axes.titlesize": font_size+1,
                  "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold"})
    if style["text.usetex"]:
        style["text.latex.preamble"] += r"\renewcommand{\seriesdefault}{\bfdefault}\AtBeginDocument{\boldmath}"
    with matplotlib.rc_context(style):
        fig, ax = plt.subplots(figsize=(7.2, 4.5), layout="constrained")
        for depth in DEPTHS:
            cell = summary.loc[(summary.depth == depth) & (summary.metric == "alpha_rmse")].sort_values("n_datasets")
            ax.errorbar(cell.n_datasets, cell["mean"],
                        yerr=[cell["mean"]-cell.ci_low, cell.ci_high-cell["mean"]],
                        **STYLES[depth], linewidth=1.8, markersize=5, capsize=2, elinewidth=.8)
        ax.axhline(0, color="#666666", linestyle="--", linewidth=1.1)
        ax.set(xlabel="Number of training datasets",
               ylabel=rf"Mean RMSE$(\widehat\alpha_{{dt}},{truth:g})$",
               title="Dispersion recovery error", xticks=list(COUNTS), xlim=(1.6, 10.4))
        ax.set_ylim(bottom=0)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.legend(loc="lower left")
        for suffix in ("pdf", "png", "svg"):
            fig.savefig(out / f"{STEM}.{suffix}", dpi=600)
        plt.close(fig)
    return {key: style[key] for key in ("text.usetex", "font.size", "font.weight")}


def write_report(out, summary, ids, excluded, command, repeats):
    endpoints = summary.loc[summary.n_datasets.isin([2, 10])].pivot(
        index=["n_datasets", "depth"], columns="metric", values="mean")
    endpoints.to_csv(out / "alpha_endpoint_summary.csv")
    mean_cells = summary.loc[summary.metric == "alpha_mean"]
    rmse_cells = summary.loc[summary.metric == "alpha_rmse"].pivot(
        index="n_datasets", columns="depth", values="mean")
    high_best = int((rmse_cells.idxmin(axis=1) == "20_per_codon").sum())
    max_clipped = summary.loc[summary.metric.isin(["fraction_at_floor", "fraction_at_cap"]), "mean"].max()
    findings = (f"Mean learned alpha ranges from {mean_cells['mean'].min():.4f} to "
                f"{mean_cells['mean'].max():.4f} across the 27 conditions, above 0.1. "
                f"The 20-reads/codon models have the lowest alpha RMSE at {high_best}/{len(COUNTS)} "
                f"panel sizes. The 0.25-versus-2 ordering is not uniformly monotonic. "
                f"The largest mean floor/cap fraction is {max_clipped:.6f}. "
                "The true dispersion is therefore not recovered exactly, even at high depth.")
    (out / "caption.tex").write_text(r"""\textbf{Recovery of synthetic NB2 dispersion.}
Mean profile RMSE against
the simulator value $\alpha_0=0.1$, for cumulative uniform-reference models at
0.25, 2 and 20 expected reads/codon. Effective dispersion is
$\widehat\alpha=\exp[\operatorname{clip}(\texttt{log\_sigma},\ell_{\min},\ell_{\max})]$,
using the loss's configured log-dispersion bounds; the dashed reference is zero.
Each transcript--dataset pair is scored on the sense-CDS interior after excluding
10 codons at each end, then scores are averaged equally across datasets within
transcript and equally across transcripts. No alpha normalization or smoothing
is performed. PCC is undefined because the reference alpha is constant.
"""+f"The same {len(ids)} validation transcripts are used at every depth and panel size. "
        +f"Error bars are pointwise 95\\% intervals from {repeats:,} paired transcript-bootstrap resamples, "
        +"conditional on the frozen best-validation-PCC models (training seed 42). "
        +"Learned dispersion can absorb mean-model mismatch and is not necessarily a pure estimate "
        +"of the simulator's conditional count noise.\n")
    (out / "README.md").write_text(f"""# Synthetic alpha recovery

![Alpha recovery]({STEM}.png)

Only alpha RMSE is plotted. Mean alpha and the other diagnostics remain in the
source tables. Relative-position and count-error diagnostics are provided by
`analyses/analyze_synthetic_alpha_position_error.py` in `position_error/`.

## Definition and scope

Simulator metadata in all 30 biased count files confirms NB2: Var(Y | mu) = mu + 0.1 mu^2.
The likelihood is evaluated on the two **raw replicas**, not their arithmetic mean.
The benchmark is therefore 0.1, not 0.05. Exported `log_sigma` denotes log alpha,
not log standard deviation. We exponentiate log_sigma after clipping to the
resolved likelihood bounds, recorded per run in provenance, exactly as the loss
does. Fractions at either bound are saved. PCC to a constant reference is
mathematically undefined.

Same 27 cumulative models as the L/gamma overview, uniform pi=1/N, training seed 42,
N=2,...,10, best-validation-PCC exports. Reliability weights and neural parameters
are not changed. Primary cohort: {len(ids)} matched validation transcripts;
{len(excluded)} additional complete-case exclusions. The cohort is fixed across every cell.
Remove the terminal export entry and exclude the first/last 10 sense codons.

For each transcript/dataset pair, compute mean alpha, signed error, RMSE, MAE,
spatial standard deviation, mean log2(alpha/0.1), mean absolute log2 error,
fraction within [0.05,0.2], and floor/cap fractions. First average datasets equally
within each transcript, then transcripts equally. The plotted RMSE is a mean of
profile RMSEs, NOT sqrt of a pooled squared error, nor the error of the mean alpha.
All intervals use {repeats:,} paired transcript bootstrap draws, seed 42 by default;
dataset/codon rows are never resampled as independent observations.

## Actual endpoint results

{findings}

```text
{endpoints[['alpha_mean','alpha_rmse','alpha_bias','fraction_within_factor_two','fraction_at_floor','fraction_at_cap']].to_string(float_format=lambda x: f'{x:.6f}')}
```

## Interpretation limits

The synthetic conditional alpha is known, but the fitted mean is not the simulator
mean: two distinct traffic trajectories share one predicted shape, scaled by each
replica's observed mean count. Alpha may absorb residual trajectory variation,
gamma/shared-profile errors, and effects of plug-in scale estimation. A deviation
from 0.1 is therefore not proof of a faulty optimizer or dispersion head. At low mu,
the alpha-dependent variance term is small relative to Poisson variance: the ratio
is alpha*mu. Dispersion can be weakly constrained in sparse counts. These are
possible explanations, not causal conclusions from this observational comparison.
The validation cohort selected the checkpoint; there is no independent test or
multi-seed replication. The 27-transcript intersection limits generalization.

The separate 2x2 L/gamma recovery figure is unchanged. This analysis adds no PCC
against constant alpha and makes no assumption that greater depth must improve recovery.

## Reproduce

```bash
{command}
```

Use `--font-size 16` to enlarge the bold LaTeX text. Scalar source tables,
`provenance.json`, exclusions and the caption accompany the PDF/PNG/SVG.
Streaming uses Arrow batches of 16 profiles and retains only scalar results;
no checkpoints or complete codon-level table are loaded.
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT_DIR)
    parser.add_argument("--overview-dir", type=Path, default=AUDIT_DIR / "depth_recovery_overview")
    parser.add_argument("--output-dir", type=Path, default=AUDIT_DIR / "alpha_recovery")
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--font-size", type=float, default=14.)
    args = parser.parse_args()
    if args.bootstrap_repeats < 2 or not np.isfinite(args.font_size) or args.font_size <= 0:
        parser.error("Require at least two bootstrap draws and a finite positive font size.")
    runs, validation = verified_runs(args.audit_dir)
    ids_path = args.overview_dir / "matched_transcript_ids.csv"
    ids = set(pd.read_csv(ids_path).transcript_id)
    if not ids or not ids.issubset(set.intersection(*validation.values())):
        raise ValueError("The overview cohort is not held out in every selected model.")
    lengths_path = args.overview_dir / "shared_profile_per_transcript.csv"
    lengths_frame = pd.read_csv(lengths_path, usecols=["transcript_id", "sense_length"])
    if (lengths_frame.groupby("transcript_id").sense_length.nunique() != 1).any():
        raise ValueError("Inconsistent sense-CDS lengths in the audited source.")
    lengths = lengths_frame.groupby("transcript_id").sense_length.first().to_dict()
    truth, simulator = verify_simulator()
    records, checks = [], []
    for run in runs:
        current, check = evaluate_run(run, ids, lengths, truth)
        records.extend(current)
        checks.append(check)
        print(f"alpha {run['depth']}, N={run['n_datasets']}: {len(current)} scalar profiles", flush=True)
    detail = pd.DataFrame(records)
    transcripts = aggregate_transcripts(detail)
    matched, excluded = complete_cohort(transcripts, ids, DEPTHS, METRICS)
    if not matched:
        raise ValueError("No complete alpha evaluation cohort remains.")
    summary = summarize(transcripts, matched, args.bootstrap_repeats, args.bootstrap_seed)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "alpha_per_transcript_dataset.csv", index=False)
    transcripts.to_csv(out / "alpha_per_transcript.csv", index=False)
    summary.to_csv(out / "alpha_summary.csv", index=False)
    pd.DataFrame({"transcript_id": matched}).to_csv(out / "matched_transcript_ids.csv", index=False)
    pd.DataFrame({"transcript_id": excluded, "reason": ["nonfinite_alpha_metric"]*len(excluded)}).to_csv(
        out / "exclusions.csv", index=False)
    typography = plot(summary, out, truth, args.font_size)
    command = f"RIBOUNMIX_PLOT_TEX={int(typography['text.usetex'])} " + shlex.join(
        [sys.executable, str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]])
    write_report(out, summary, matched, excluded, command, args.bootstrap_repeats)
    sources = [Path(__file__), ids_path, lengths_path, args.audit_dir / "provenance.json",
               ROOT / "Models/RiboUnmixLightningModule.py",
               ROOT / "Models/RiboUnmixModel/submodels/DatasetLogSigmaHead.py",
               ROOT / "analyses/audit_synthetic_read_depth.py",
               ROOT / "analyses/analyze_synthetic_recovery.py",
               ROOT / "Utils/publication_plot_style.py",
               ROOT / "analyses/plot_synthetic_read_depth_effect.py"]
    provenance = dict(command=command, alpha_truth=truth, simulator_files=simulator, runs=runs, checks=checks,
        cohort=dict(n=len(matched), hash=cohort_hash(matched), excluded=excluded),
        checkpoint="best_pcc", seed=42, boundary_trim_codons=TRIM,
        alpha_definition="exp(clip(log_sigma, resolved loss.nb_log_alpha_min, resolved loss.nb_log_alpha_max))",
        pcc_valid=False, pcc_reason="constant_reference", typography=typography,
        fitting_or_inference=False, max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        source_sha256={str(p.resolve()): file_hash(p) for p in sources},
        bootstrap=dict(repeats=args.bootstrap_repeats, seed=args.bootstrap_seed, paired=True,
                       cluster="transcript", interval="pointwise percentile 95%"))
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    (out / "commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n"+command+"\n")
    print(f"Completed: {len(matched)} matched transcripts; {len(excluded)} exclusions. Figure: {out / (STEM+'.pdf')}")
    print(summary.loc[summary.n_datasets.isin([2,10]) & summary.metric.isin(["alpha_mean","alpha_rmse"])].to_string(index=False))


if __name__ == "__main__":
    main()
