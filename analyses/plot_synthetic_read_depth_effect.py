#!/usr/bin/env python3
"""Show depth-dependent estimation error without relabeling it as K recovery.

Consumes the independently audited, frozen best-PCC prediction metrics. No
inference, training, count-level data frame, or change to reference weights.
The main figure uses a common three-depth cohort and a larger paired low/high
cohort, each fixed across all nine centered cumulative panel sizes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Utils.publication_plot_style import publication_rc

DEPTHS = ("0p25_per_codon", "2_per_codon", "20_per_codon")
DEPTH_VALUES = (.25, 2., 20.)
COUNTS = tuple(range(2, 11))
ENDPOINTS = (2, 10)
METRICS = ("rmse_Kg_trim10", "pcc_K_trim10", "mu_target_pcc_trim10")
BIAS_ORDER = ("3prime_aa", "3prime_cc", "3prime_gg", "3prime_uu",
              "5prime_aa", "5prime_cc", "5prime_gg", "5prime_uu",
              "gc_fraction_gt_0p7", "au_fraction_gt_0p7")
STYLES = {2: dict(color="#0072B2", marker="o", linestyle="-"),
          10: dict(color="#009E73", marker="s", linestyle="--")}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cohort_hash(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def verified_runs(audit_dir):
    """Check saved run identity, uniform references and the actual split manifests."""
    source = json.loads((audit_dir / "provenance.json").read_text())
    checks = {c["run"]: c for c in source["checks"] if c["variant"] == "best_pcc"}
    selected, validation = [], {}
    for run in source["runs"]:
        depth, datasets = run["depth"], run["datasets"]
        if depth not in DEPTHS or len(datasets) not in COUNTS:
            continue
        config_path = ROOT / run["config_path"]
        if file_hash(config_path) != run["config_sha256"]:
            raise ValueError(f"Configuration changed since the source audit: {config_path}")
        config = yaml.load(config_path.read_text(), Loader=yaml.CSafeLoader)
        ref = config["model"]["gamma_centering"]["reference"]
        assert config["experiment"]["seed"] == run["seed"] == 42
        assert config["experiment"]["dataset"] == datasets
        assert datasets == ["artificial_bias_"+b for b in BIAS_ORDER[:len(datasets)]]
        assert config["model"]["gamma_centering"]["mode"] == "fixed_reference"
        assert ref["weighting"] == run["weighting"] == "equal"
        assert ref["dataset_names"] is None and ref["minimum_datasets"] <= len(datasets)
        split_path = ROOT / run["split_path"]
        split = json.loads(split_path.read_text())
        ids = set(split["validation_ids"])
        assert not ids & set(split["train_ids"])
        assert cohort_hash(ids) == run["validation_hash"]
        assert all(f"/{depth}/" in p for p in split["training_dataset_paths"])
        if depth in validation and validation[depth] != ids:
            raise ValueError(f"Validation cohort varies across panel size at {depth}")
        validation[depth] = ids
        check = checks[run["run"]]
        assert check["duplicate_L_max_abs_difference"] <= 1e-5
        assert np.allclose(list(check["actual_reference_weights"].values()), 1/len(datasets))
        assert (ROOT / check["prediction_path"]).is_file()
        selected.append({**run, "n_datasets": len(datasets),
                         "split_sha256": file_hash(split_path),
                         "prediction_path": check["prediction_path"],
                         "checkpoint": check["checkpoint"],
                         "duplicate_L_max_abs_difference": check["duplicate_L_max_abs_difference"]})
    conditions = [(r["depth"], r["n_datasets"]) for r in selected]
    expected = {(d, n) for d in DEPTHS for n in COUNTS}
    if len(conditions) != len(expected) or set(conditions) != expected:
        raise ValueError("A complete, unique 3-depth by 9-panel grid is required.")
    return selected, validation


def load_metrics(path, runs, ids):
    columns = ["run", "depth", "n_datasets", "variant", "transcript_id", "sense_length",
               "mu_target_valid_datasets", *METRICS]
    parts = []
    names = {r["run"] for r in runs}
    for chunk in pd.read_csv(path, usecols=columns, chunksize=5000):
        keep = ((chunk.variant == "best_pcc") & chunk.run.isin(names)
                & chunk.transcript_id.isin(ids))
        parts.append(chunk.loc[keep].copy())
    frame = pd.concat(parts, ignore_index=True)
    if frame.duplicated(["transcript_id", "depth", "n_datasets"]).any():
        raise ValueError("Expected one scalar row per transcript and model, not dataset rows.")
    # A count-fit mean must include the same complete dataset panel, not a
    # variable subset of rows with defined PCC.
    frame.loc[frame.mu_target_valid_datasets != frame.n_datasets, "mu_target_pcc_trim10"] = np.nan
    return frame


def complete_cohort(frame, candidates, depths, metrics):
    """Use one complete-case cohort for every N and requested metric."""
    ids = sorted(candidates)
    index = pd.MultiIndex.from_product([ids, depths, COUNTS],
                                      names=["transcript_id", "depth", "n_datasets"])
    grid = frame.set_index(list(index.names)).reindex(index)[list(metrics)]
    valid = np.isfinite(grid.to_numpy()).reshape(len(ids), -1).all(axis=1)
    return [tid for tid, good in zip(ids, valid) if good], [tid for tid, good in zip(ids, valid) if not good]


def mean_stats(values, draws):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite metric reached the bootstrap.")
    boot = values[draws].mean(axis=1)
    return dict(mean=float(values.mean()), ci_low=float(np.quantile(boot, .025)),
                ci_high=float(np.quantile(boot, .975)))


def paired_ratio_stats(low, high, draws):
    """Ratio of mean errors, not mean of per-transcript ratios."""
    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high <= 0):
        raise ValueError("The paired error ratio needs finite errors and a positive denominator.")
    low_boot, high_boot = low[draws].mean(axis=1), high[draws].mean(axis=1)
    ratio = low_boot/high_boot
    difference = low_boot-high_boot
    return dict(mean_low=float(low.mean()), mean_high=float(high.mean()),
                ratio=float(low.mean()/high.mean()),
                ratio_ci_low=float(np.quantile(ratio, .025)),
                ratio_ci_high=float(np.quantile(ratio, .975)),
                difference=float(low.mean()-high.mean()),
                difference_ci_low=float(np.quantile(difference, .025)),
                difference_ci_high=float(np.quantile(difference, .975)))


def summarize(frame, common, paired, repeats, seed):
    lookup = frame.set_index(["depth", "n_datasets", "transcript_id"])

    def values(depth, n, ids, metric):
        return lookup.loc[(depth, n), metric].reindex(ids).to_numpy()

    rng = np.random.default_rng(seed)
    # Reuse each cohort's resampled transcript IDs across all N, depths and
    # metrics to preserve model-pair and within-transcript dependence.
    common_draws = rng.integers(len(common), size=(repeats, len(common)))
    common_summary = []
    for n in COUNTS:
        for depth in DEPTHS:
            for metric in METRICS:
                common_summary.append(dict(n_datasets=n, depth=depth, metric=metric,
                                           n_transcripts=len(common), cohort_hash=cohort_hash(common),
                                           **mean_stats(values(depth, n, common, metric), common_draws)))
    paired_draws = rng.integers(len(paired), size=(repeats, len(paired)))
    ratios = []
    for n in COUNTS:
        low = values(DEPTHS[0], n, paired, "rmse_Kg_trim10")
        high = values(DEPTHS[-1], n, paired, "rmse_Kg_trim10")
        ratios.append(dict(n_datasets=n, n_transcripts=len(paired),
                           cohort_hash=cohort_hash(paired),
                           **paired_ratio_stats(low, high, paired_draws)))
    return pd.DataFrame(common_summary), pd.DataFrame(ratios)


def depth_curve(ax, summary, metric):
    for n in ENDPOINTS:
        data = (summary.loc[(summary.n_datasets == n) & (summary.metric == metric)]
                .set_index("depth").loc[list(DEPTHS)])
        ax.errorbar(DEPTH_VALUES, data["mean"],
                    yerr=[data["mean"]-data.ci_low, data.ci_high-data["mean"]],
                    **STYLES[n], markersize=5, linewidth=1.7, capsize=3,
                    elinewidth=.9, label=rf"$N={n}$")
    ax.set_xscale("log")
    ax.set_xticks(DEPTH_VALUES, ["0.25", "2", "20"])
    ax.set_xlim(.18, 28)
    ax.minorticks_off()
    ax.set_xlabel("Expected reads per codon")


def plot(out, summary, ratios, common_n, paired_n):
    style = publication_rc()
    with matplotlib.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.15), layout="constrained")
        depth_curve(axes[0], summary, "rmse_Kg_trim10")
        axes[0].set(title=rf"Reference-proxy error ($n={common_n}$)",
                    ylabel=r"Mean RMSE$(L_t,H_t)$", ylim=(0, .105))
        axes[0].legend(loc="lower left")
        axes[1].axhline(1, color="#777777", linestyle="--", linewidth=1)
        axes[1].errorbar(ratios.n_datasets, ratios.ratio,
                         yerr=[ratios.ratio-ratios.ratio_ci_low,
                               ratios.ratio_ci_high-ratios.ratio],
                         color="#D55E00", marker="o", linewidth=1.7, markersize=4,
                         capsize=3, elinewidth=.9)
        axes[1].set(title=rf"Low-depth error penalty ($n={paired_n}$)",
                    ylabel="Mean RMSE ratio: 0.25 / 20", xlabel="Number of bias datasets",
                    xticks=list(COUNTS), xlim=(1.6, 10.4),
                    ylim=(.85, max(2.1, ratios.ratio_ci_high.max()*1.08)))
        axes[1].text(2, 1.04, "Equal error", color="#666666", va="bottom")
        depth_curve(axes[2], summary, "pcc_K_trim10")
        axes[2].set(title=rf"Kinetic recovery ($n={common_n}$)",
                    ylabel=r"Mean PCC$(L_t,K_t)$", ylim=(.73, 1.0))
        axes[2].legend(loc="center right")
        for letter, ax in zip("ABC", axes):
            ax.grid(axis="y")
            ax.text(-.17, 1.07, letter, transform=ax.transAxes, fontweight="bold")
        for suffix, dpi in (("pdf", 600), ("png", 600)):
            fig.savefig(out / f"read_depth_effect.{suffix}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        # Deliberately separate this noise-affected observation score from the
        # main model-versus-noiseless-proxy error figure.
        fig, ax = plt.subplots(figsize=(5.5, 4.1), layout="constrained")
        depth_curve(ax, summary, "mu_target_pcc_trim10")
        ax.set(title=rf"Count prediction: noise-affected control ($n={common_n}$)",
               ylabel=r"Mean PCC$(\mu_{dt},\overline{{Y}}_{dt})$", ylim=(.4, 1.0))
        ax.grid(axis="y")
        ax.legend(loc="lower right")
        fig.savefig(out / "count_prediction_noise_control.pdf", bbox_inches="tight")
        fig.savefig(out / "count_prediction_noise_control.png", dpi=600, bbox_inches="tight")
        plt.close(fig)
    return {"text.usetex": style["text.usetex"], "font.size": style["font.size"],
            "font.family": style["font.family"], "font.serif": style["font.serif"]}


def write_caption(out, common_n, paired_n, ratios, repeats):
    caption = rf"""\textbf{{Read depth affects estimation error even when kinetic recovery changes little.}}
All models use uniform reference weights, the same within-depth cumulative bias order,
and training seed 42. Frozen best-validation-PCC checkpoints are evaluated on validation
transcripts, excluding the first and last 10 sense codons. The diagnostic proxy is
$H_t=\operatorname{{mean\!\!\!-\!one}}(K_t G_t)$, where
$G_{{ti}}=\prod_{{d=1}}^N b_{{dti}}^{{1/N}}$.
Known kinetic and bias profiles enter only this post-hoc evaluation, not training or weight selection.
\textbf{{(A)}} Mean transcript-level RMSE against $H_t$ for $N=2$ and $N=10$, using the same
{common_n} transcripts held out at all three depths; lower is better.
\textbf{{(B)}} Ratio of mean RMSE at 0.25 versus 20 reads/codon on the same larger
{paired_n}-transcript low/high validation intersection, fixed across every $N=2,\ldots,10$.
Values above one indicate greater error at low depth. This is a ratio of means, not a mean
of individual ratios. \textbf{{(C)}} Mean PCC against the original kinetic target $K_t$
on the same {common_n} transcripts as A; higher is better. Error bars are pointwise 95\%
percentile intervals from {repeats:,} transcript-cluster bootstrap draws; resampling is paired
across models and panel sizes within each cohort. Intervals condition on the trained models
and do not capture training-seed or independent count-simulation uncertainty.
$H_t$ is a kinetic-plus-reference-bias proxy, not the unnoised TASEP profile or an oracle
for the mass-free model. Better agreement with $H_t$ must not be relabeled better recovery
of $K_t$. Single-dataset models are excluded because gamma centering is bypassed at $N=1$.
"""
    (out / "caption.tex").write_text(caption)
    r2, r10 = (ratios.set_index("n_datasets").loc[n] for n in ENDPOINTS)
    (out / "README.md").write_text(f"""# What deteriorates at low read depth?

![Matched read-depth comparison](read_depth_effect.png)

The main quantity is model error relative to the fixed **kinetic-plus-reference-bias
proxy**, H = mean-one(K times the geometric mean of the injected bias multipliers).
This is not the original biological-recovery target K, nor a simulated traffic oracle.
Lower proxy RMSE is a model-versus-noiseless-reference comparison, not merely attenuation
against noisy count observations.

- A and C use all {common_n} transcripts shared by the three validation cohorts. The same
  cohort is fixed across every dataset count. N=2 and N=10 are the first and last centered
  cumulative experiments; they were the endpoints examined before plotting.
- B uses the larger {paired_n}-transcript 0.25/20 intersection and displays **all nine**
  centered panel sizes. No panel size was selected by the size of its depth effect.
- At N=2, low-depth mean RMSE is **{r2.ratio:.2f} times** high-depth RMSE
  (95% paired transcript-bootstrap interval {r2.ratio_ci_low:.2f}–{r2.ratio_ci_high:.2f}).
- At N=10, the ratio is **{r10.ratio:.2f}**
  ({r10.ratio_ci_low:.2f}–{r10.ratio_ci_high:.2f}).
- Across all nine N, ratios range from {ratios.ratio.min():.2f} to {ratios.ratio.max():.2f}.
  Full ratios, paired differences, and pointwise intervals are in `paired_error_ratios.csv`.

This supports a deterioration in **agreement with the specified diagnostic proxy** at
low depth. It does not establish worse K recovery, a multi-seed causal effect, or an exact
statistical estimation-error decomposition. All results are validation, not independent test
results. There is one training seed; the two simulator traffic trajectories are reused
across depths and bias conditions. The total training cohort also differs slightly because
depth-specific validation identities differ.

`count_prediction_noise_control.pdf` is a separate observation-level control. It plots
model mu versus the actual arithmetic mean of the two count replicas, averaging datasets
within each transcript and then transcripts equally. Its depth effect includes noise in
the evaluation target, so it is not evidence by itself of poorer biological inference.

## Reproduction

Run from the repository root:

```bash
RIBOUNMIX_PLOT_TEX=1 .venv/bin/python analyses/plot_synthetic_read_depth_effect.py
.venv/bin/python -m unittest Tests/test_synthetic_read_depth_effect_plot.py
```

The required `read_depth_audit` inputs are generated by
`analyses/audit_synthetic_read_depth.py`. The figure script rechecks saved configurations,
split identities, reference weights, and source-audit deduplication records. It changes
no training parameters, predictions, reference weights, or transcript reliability weights.
See `caption.tex`, `provenance.json`, and the scalar source CSVs for exact definitions.
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path,
                        default=ROOT / "analyses/artifacts/synthetic/read_depth")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_repeats < 100:
        parser.error("Use at least 100 bootstrap draws.")
    audit_dir = args.audit_dir.resolve()
    out = (args.output_dir or audit_dir / "depth_effect_figure").resolve()
    runs, validation = verified_runs(audit_dir)
    common_candidates = set.intersection(*(validation[d] for d in DEPTHS))
    paired_candidates = validation[DEPTHS[0]] & validation[DEPTHS[-1]]
    source_path = audit_dir / "recovery_per_transcript.csv"
    frame = load_metrics(source_path, runs, common_candidates | paired_candidates)
    common, common_excluded = complete_cohort(frame, common_candidates, DEPTHS, METRICS)
    paired, paired_excluded = complete_cohort(frame, paired_candidates,
                                              (DEPTHS[0], DEPTHS[-1]), ("rmse_Kg_trim10",))
    if len(common) < 2 or len(paired) < 2:
        raise ValueError("Insufficient complete matched validation transcripts.")
    summary, ratios = summarize(frame, common, paired, args.bootstrap_repeats, args.bootstrap_seed)
    out.mkdir(parents=True, exist_ok=True)
    frame["in_three_depth_cohort"] = frame.transcript_id.isin(common)
    frame["in_low_high_cohort"] = frame.transcript_id.isin(paired) & frame.depth.isin([DEPTHS[0], DEPTHS[-1]])
    frame.to_csv(out / "per_transcript.csv", index=False)
    summary.to_csv(out / "matched_depth_summary.csv", index=False)
    ratios.to_csv(out / "paired_error_ratios.csv", index=False)
    cohort_rows = [dict(cohort=name, transcript_id=tid) for name, ids in
                   (("three_depth", common), ("low_high", paired)) for tid in ids]
    pd.DataFrame(cohort_rows).to_csv(out / "cohort_ids.csv", index=False)
    typography = plot(out, summary, ratios, len(common), len(paired))
    command = " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
    provenance = dict(command=command, source_audit=str(audit_dir),
                      source_sha256={name: file_hash(audit_dir / name) for name in
                                     ("provenance.json", "recovery_per_transcript.csv")},
                      checkpoint_variant="best_pcc", training_seed=42,
                      boundary_trim_sense_codons=10, reference_weighting="equal",
                      reference_proxy="H=mean_one(K * product_d b_d**(1/N)); not unnoised q",
                      error_scale="unchanged saved L; H mean-one on full sense CDS; no interior renormalization",
                      statistic="mean of transcript RMSEs; mean of transcript PCCs; ratio of mean RMSEs",
                      bootstrap=dict(repeats=args.bootstrap_repeats, seed=args.bootstrap_seed,
                                     unit="transcript", paired=True, interval="pointwise percentile 95%",
                                     training_seed_uncertainty_included=False),
                      cohorts={name: dict(n=len(ids), hash=cohort_hash(ids), ids=ids)
                               for name, ids in (("three_depth", common), ("low_high", paired))},
                      excluded=dict(three_depth=common_excluded, low_high=paired_excluded,
                                    reason="missing/nonfinite required metric in at least one model"),
                      typography=typography, runs=runs)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    write_caption(out, len(common), len(paired), ratios, args.bootstrap_repeats)
    print(f"Matched cohorts: {len(common)} at all depths; {len(paired)} in low/high comparison.")
    print(ratios[["n_datasets", "mean_low", "mean_high", "ratio", "ratio_ci_low", "ratio_ci_high"]].to_string(index=False))
    print(out / "read_depth_effect.pdf")


if __name__ == "__main__":
    main()
