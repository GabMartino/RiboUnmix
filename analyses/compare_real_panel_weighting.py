#!/usr/bin/env python3
"""Matched equal-vs-quality-ranked real-panel reproducibility, not ground-truth recovery.

Uses frozen best-validation-loss exports only. Raw dataset-duplicated exports
are compacted in small Arrow batches by the shared panel analyzer. No model,
training data, or checkpoint is loaded. Partial runs use identical panel pairs
in both policies; re-running automatically includes newly completed panels.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import shlex
import sys

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Utils.publication_plot_style import latex_paper_style
from Utils.reliability_references import transcript_id_hash
from analyses.analyze_real_panel_convergence import _extract_panel_profiles, _locate_panel_prediction
from run_real_independent_panel_convergence_quality_rank import (
    _assert_only_gamma_reference_changed, _flatten_config,
)

DEFAULT_EQUAL = ROOT / "results/my_panels_a100_b32_20260906_114323"
DEFAULT_RANKED = ROOT / "results/my_panels_qrank_a100_b32_20260908_103510"
COLORS = {"equal": "#0072B2", "ranked": "#D55E00"}
LABELS = {"equal": r"Equal $\pi$", "ranked": r"Quality-ranked $\pi$"}
REGIONS = {"full_cds": "Full CDS", "interior_20": "CDS interior"}
METRICS = ("PCC", "Spearman", "RMSE")


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_ranking(ranked, strategy, explicit=None):
    """Resolve the recorded ranking by content, including relocated run copies."""
    candidates = ([Path(explicit)] if explicit is not None else [
        Path(strategy['ranking_table']),
        ranked.parent / 'frozen_quality_rank_10components.tsv',
        ROOT / 'Datasets/data/HEK_riboseq_profile_quality_rank.tsv',
        ROOT / 'Datasets/data/HEK_riboseq_profile_quality_rank_components.tsv',
    ])
    for path in candidates:
        if path.is_file() and sha256(path) == strategy['ranking_table_sha256']:
            return path
    observed = {str(path): sha256(path) if path.is_file() else 'missing' for path in candidates}
    raise ValueError(
        f"No ranking file matches the frozen SHA256={strategy['ranking_table_sha256']}; "
        f"checked={observed}. Supply --ranking-table with the frozen input."
    )


def audit_design(equal, ranked, ranking_table=None):
    roots = {"equal": equal, "ranked": ranked}
    manifests = {k: read_json(r / "panel_manifest.json") for k, r in roots.items()}
    panels = manifests["equal"]["panels"]
    if panels != manifests["ranked"]["panels"]:
        raise ValueError("Panel memberships differ; refusing a weighting-only comparison.")
    if manifests["equal"]["panel_source_families"] != manifests["ranked"]["panel_source_families"]:
        raise ValueError("Source-family panel membership differs.")
    splits = {k: read_json(r / "common_split_manifest.json") for k, r in roots.items()}
    for fold in ("common_test_ids", "common_validation_ids", "panel_train_eligible_ids"):
        if splits["equal"][fold] != splits["ranked"][fold]:
            raise ValueError(f"Unequal transcript cohorts: {fold}")
    ids = sorted(splits["equal"]["common_test_ids"])
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Empty or duplicated common test IDs")
    ranking = resolve_ranking(ranked, manifests['ranked']['gamma_reference_strategy'], ranking_table)
    ranks = pd.read_csv(ranking, sep="\t").set_index("dataset").quality_rank
    rank_max = float(ranks.max())
    weights_rows, differences, provenance = [], [], []
    for panel, datasets in sorted(panels.items()):
        configurations = {k: yaml.safe_load((r / panel / "resolved_config.yaml").read_text())
                          for k, r in roots.items()}
        _assert_only_gamma_reference_changed(reference_root=equal, panel_name=panel,
                                             resolved_config=configurations["ranked"])
        flat = {k: _flatten_config(c) for k, c in configurations.items()}
        for key in sorted(flat["equal"].keys() | flat["ranked"].keys()):
            if flat["equal"].get(key) != flat["ranked"].get(key):
                differences.append(dict(panel=panel, key=key,
                                        equal=json.dumps(flat["equal"].get(key)),
                                        ranked=json.dumps(flat["ranked"].get(key))))
        reliability = {k: read_json(r / panel / "reliability_reference_manifest.json")
                       for k, r in roots.items()}
        ignored = {"created_at_utc", "source_split_manifest", "experiment_name"}
        if ({k: v for k, v in reliability["equal"].items() if k not in ignored}
                != {k: v for k, v in reliability["ranked"].items() if k not in ignored}):
            raise ValueError(f"{panel}: local transcript reliability weights differ.")
        panel_splits = {k: read_json(r / panel / "split_manifest.json") for k, r in roots.items()}
        for fold in ("train_ids", "validation_ids", "test_ids"):
            if set(panel_splits["equal"][fold]) != set(panel_splits["ranked"][fold]):
                raise ValueError(f"{panel}: unequal {fold}")
        heldout = set(ids) | set(splits["equal"]["common_validation_ids"])
        if set(panel_splits["equal"]["train_ids"]) & heldout:
            raise ValueError(f"{panel}: held-out leakage in training manifest")
        for policy, root in roots.items():
            run = read_json(root / panel / "run_manifest.json")
            ref = run["fixed_gamma_reference"]
            if run["selected_datasets"] != datasets or ref["dataset_names"] != datasets:
                raise ValueError(f"{panel}/{policy}: dataset identity mismatch")
            pi = np.array([ref["pi"][d] for d in datasets])
            expected_pi = np.ones(len(datasets))
            if policy == "ranked":
                power = configurations[policy]["model"]["gamma_centering"]["reference"]["quality_rank_power"]
                if power != 1.0:
                    raise ValueError("This prespecified sensitivity uses power=1, without tuning.")
                expected_pi = ((rank_max + 1 - ranks.loc[datasets].to_numpy(float)) / rank_max) ** power
            expected_pi /= expected_pi.sum()
            np.testing.assert_allclose(pi, expected_pi, atol=1e-12, rtol=1e-10)
            weights_rows.extend(dict(policy=policy, panel=panel, dataset=d,
                                     global_quality_rank=float(ranks.loc[d]), pi=float(p))
                                for d, p in zip(datasets, pi))
            provenance.append(dict(policy=policy, panel=panel, n_datasets=len(datasets),
                                   effective_reference_count=float(1 / np.sum(pi ** 2)), pi_sum=float(pi.sum()),
                                   train_count=len(panel_splits[policy]["train_ids"]),
                                   train_id_hash=transcript_id_hash(panel_splits[policy]["train_ids"]),
                                   config_sha256=sha256(root / panel / "resolved_config.yaml"),
                                   reliability_sha256=sha256(root / panel / "reliability_reference_manifest.json")))
    return ids, sorted(panels), pd.DataFrame(weights_rows), pd.DataFrame(differences), provenance


def collect(root, panels, ids, policy):
    profiles, availability = {}, []
    for panel in panels:
        directory = root / panel
        artifact, source, detail = _locate_panel_prediction(directory)
        row = dict(policy=policy, panel=panel, available=artifact is not None, detail=detail)
        if artifact is not None:
            print(f"Reading {policy}/{panel}: {artifact.name}", flush=True)
            runtime_paths = list((directory / "predictions").rglob("prediction_checkpoint_manifest.json"))
            if len(runtime_paths) != 1:
                raise ValueError(f"{policy}/{panel}: ambiguous runtime manifest")
            runtime = read_json(runtime_paths[0])["best_val_loss"]
            if (runtime["split_name"] != "test" or runtime["transcript_count"] != len(ids)
                    or runtime["transcript_id_hash"] != transcript_id_hash(ids)):
                raise ValueError(f"{policy}/{panel}: runtime test cohort mismatch")
            gamma = read_json(runtime_paths[0].parent / "gamma_reference_manifest.json")
            run = read_json(directory / "run_manifest.json")
            names = gamma["reference_dataset_names"]
            if (gamma["centering_mode"] != "fixed_reference"
                    or gamma["weighting"] != ("equal" if policy == "equal" else "quality_rank")
                    or set(names) != set(run["selected_datasets"])):
                raise ValueError(f"{policy}/{panel}: runtime gamma-reference mismatch")
            np.testing.assert_allclose(gamma["reference_pi"],
                [run["fixed_gamma_reference"]["pi"][d] for d in names], rtol=1e-6, atol=1e-9)
            profiles[panel], checks = _extract_panel_profiles(
                panel_name=panel, run_identifier=root.name, prediction_path=artifact,
                expected_ids=set(ids), mean_one_tolerance=1e-4)
            compact = next((directory / "predictions").rglob("common_test_L_profiles.parquet"))
            row.update(profile_path=str(compact), profile_sha256=sha256(compact),
                       source_manifest=str(source), source_manifest_sha256=sha256(source),
                       checkpoint=runtime["checkpoint_path"],
                       max_mean_one_deviation=float(checks.absolute_mean_one_deviation.max()))
        availability.append(row)
    return profiles, availability


def metrics(x, y):
    if x.shape != y.shape:
        raise ValueError("Profile positions differ; refusing truncation or interpolation")
    if x.size < 2:
        return dict(PCC=np.nan, Spearman=np.nan, RMSE=np.nan, variance_a=np.nan, variance_b=np.nan,
                    valid_PCC=False, valid_Spearman=False, reason="fewer_than_two_positions")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite frozen profile")
    vx, vy = float(x.var()), float(y.var())
    # Diagnostic rule fixed independently of agreement; means stay unmodified.
    near = min(vx, vy) <= 1e-12
    xc, yc = x - x.mean(), y - y.mean()
    pcc = np.nan if near else float(np.clip(np.dot(xc, yc) / (np.linalg.norm(xc) * np.linalg.norm(yc)), -1, 1))
    spearman = np.nan
    if not near:
        # Rank only the already matched positions; average ranks preserve ties.
        xr, yr = rankdata(x, method="average"), rankdata(y, method="average")
        xr, yr = xr - xr.mean(), yr - yr.mean()
        spearman = float(np.clip(np.dot(xr, yr) / (np.linalg.norm(xr) * np.linalg.norm(yr)), -1, 1))
    return dict(PCC=pcc, Spearman=spearman, RMSE=float(np.sqrt(np.mean((x - y) ** 2))),
                variance_a=vx, variance_b=vy, valid_PCC=not near, valid_Spearman=not near,
                reason="nearly_constant_profile" if near else "ok")


def paired_rows(profiles, ids, common_panels):
    rows, sensitivity = [], []
    for tid in ids:
        lengths = {p[panel][tid]["length"] for p in profiles.values() for panel in common_panels}
        if len(lengths) != 1:
            raise ValueError(f"{tid}: inconsistent CDS lengths")
        for region, section in (("full_cds", slice(None)), ("interior_20", slice(20, -20))):
            for left, right in itertools.combinations(common_panels, 2):
                pair = f"{left}__{right}"
                for policy, panel_profiles in profiles.items():
                    x, y = (panel_profiles[p][tid]["values"][section] for p in (left, right))
                    rows.append(dict(policy=policy, region=region, pair=pair, panel_a=left, panel_b=right,
                                     transcript_id=tid, n_positions=len(x), **metrics(x, y)))
            for panel in common_panels:
                x, y = (profiles[p][panel][tid]["values"][section] for p in ("equal", "ranked"))
                sensitivity.append(dict(panel=panel, transcript_id=tid, region=region,
                                        n_positions=len(x), **metrics(x, y)))
    return pd.DataFrame(rows), pd.DataFrame(sensitivity)


def paired_bootstrap(equal, ranked, n_boot, seed):
    """Paired transcript-cluster bootstrap of medians, retaining all panel pairs.

    Arrays have shape (transcripts, pairs). A sampled transcript brings all its
    model pairs and both weighting policies. No independent pair/position resampling.
    """
    if equal.shape != ranked.shape or equal.ndim != 2:
        raise ValueError("Expected matched transcript-by-pair matrices")
    valid = np.isfinite(equal).all(axis=1) & np.isfinite(ranked).all(axis=1)
    a, b = equal[valid], ranked[valid]
    if not len(a):
        raise ValueError("No matched valid transcripts")
    point = np.array([np.median(a), np.median(b), np.median(b) - np.median(a)])
    rng = np.random.default_rng(seed)
    samples = np.empty((n_boot, 3))
    for start in range(0, n_boot, 32):
        size = min(32, n_boot - start)
        indices = rng.integers(0, len(a), size=(size, len(a)))
        ea = np.median(a[indices], axis=(1, 2))
        rb = np.median(b[indices], axis=(1, 2))
        samples[start:start + size] = np.column_stack([ea, rb, rb - ea])
    return point, np.percentile(samples, [2.5, 97.5], axis=0), valid


def summarize(frame, ids, n_boot, seed):
    rows = []
    all_pairs = sorted(frame.pair.unique())
    for region in REGIONS:
        for group, pairs in [(p, [p]) for p in all_pairs] + [("matched_all_pairs", all_pairs)]:
            g = frame[(frame.region == region) & frame.pair.isin(pairs)]
            for metric in METRICS:
                matrices = [g[g.policy == p].pivot(index="transcript_id", columns="pair", values=metric)
                            .reindex(index=ids, columns=pairs).to_numpy(float) for p in ("equal", "ranked")]
                point, ci, valid = paired_bootstrap(*matrices, n_boot, seed)
                for i, policy in enumerate(("equal", "ranked", "ranked_minus_equal")):
                    rows.append(dict(region=region, group=group, metric=metric, policy=policy,
                                     estimate=point[i], ci_low=ci[0, i], ci_high=ci[1, i],
                                     n_transcripts=int(valid.sum()), n_excluded=int((~valid).sum()),
                                     matched_transcript_hash=transcript_id_hash(np.array(ids)[valid]),
                                     n_pairs=len(pairs), statistic="median" if i < 2 else "difference_of_medians",
                                     bootstrap_replicates=n_boot))
    return pd.DataFrame(rows)


def save_figure(fig, stem):
    for extension in ("pdf", "png"):
        fig.savefig(stem.with_suffix("." + extension), dpi=300, bbox_inches="tight")
    plt.close(fig)


def zoom_to_intervals(axis, rows):
    """Explicitly zoom a dot/interval plot; no distributions are cropped."""
    low, high = float(rows.ci_low.min()), float(rows.ci_high.max())
    pad = max(.015, (high - low) * .45)
    axis.set_ylim(low - pad, high + pad)


@latex_paper_style
def plot_comparison(summary, out, n_panels, n_planned):
    g = summary[summary.group == "matched_all_pairs"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5))
    for ax, metric, letter in zip(axes, ("PCC", "RMSE"), ("A", "B")):
        for i, (policy, marker) in enumerate((("equal", "o"), ("ranked", "s"))):
            q = g[(g.policy == policy) & (g.metric == metric)].set_index("region").loc[list(REGIONS)]
            xx = np.arange(2) + (-.09 if i == 0 else .09)
            ax.errorbar(xx, q.estimate, yerr=[q.estimate - q.ci_low, q.ci_high - q.estimate],
                        fmt=marker, color=COLORS[policy], ms=7, capsize=5, lw=1.7, label=LABELS[policy])
        ax.set_xticks(range(2), REGIONS.values())
        ax.set_xlim(-.5, 1.5)
        ax.set_ylabel("Median inter-panel " + metric)
        ax.set_title(f"{letter}   " + ("Profile-shape agreement" if metric == "PCC" else "Profile-value disagreement"), loc="left")
        zoom_to_intervals(ax, g[(g.metric == metric) & g.policy.isin(COLORS)])
        ax.grid(axis="y")
    axes[0].legend(loc="lower left")
    n = int(g.n_transcripts.min())
    prefix = "Partial: " if n_panels < n_planned else ""
    fig.suptitle(f"{prefix}{n_panels}/{n_planned} matched panels; {n:,} held-out transcripts", y=1.01, fontsize=13)
    fig.text(.5, .01, "Same panel pairs and transcripts; 95\\% paired transcript-bootstrap intervals.\n"
             "Vertical axes zoomed to median/CI range. Interior excludes 20 codons at each end.",
             ha="center", va="bottom", fontsize=12)
    fig.subplots_adjust(left=.08, right=.99, bottom=.24, top=.89, wspace=.28)
    save_figure(fig, out / "equal_vs_ranked_panel_reproducibility")


@latex_paper_style
def plot_pairs(summary, out):
    g = summary[(summary.group != "matched_all_pairs") & (summary.metric == "PCC")]
    pairs = sorted(g.group.unique())
    fig, axes = plt.subplots(1, 2, figsize=(max(9.5, len(pairs) * 1.7), 4.5))
    for ax, region in zip(axes, REGIONS):
        for i, policy in enumerate(("equal", "ranked")):
            q = g[(g.region == region) & (g.policy == policy)].set_index("group").loc[pairs]
            ax.errorbar(np.arange(len(pairs)) + (-.1 if i == 0 else .1), q.estimate,
                        yerr=[q.estimate - q.ci_low, q.ci_high - q.estimate], fmt="o" if i == 0 else "s",
                        color=COLORS[policy], capsize=4, ms=6, label=LABELS[policy])
        labels = ["--".join(str(int(p.split("_")[-1])) for p in pair.split("__")) for pair in pairs]
        ax.set_xticks(range(len(pairs)), labels)
        ax.set_xlim(-.5, len(pairs) - .5)
        zoom_to_intervals(ax, g[g.policy.isin(COLORS)])
        ax.set_ylabel("Median inter-panel PCC")
        ax.set_xlabel("Matched panel pair")
        ax.set_title(REGIONS[region])
        ax.grid(axis="y")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    save_figure(fig, out / "equal_vs_ranked_each_panel_pair")


@latex_paper_style
def plot_paired_effects(summary, out, n_panels, n_planned):
    g = summary[(summary.group == "matched_all_pairs") & (summary.policy == "ranked_minus_equal")]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), sharey=True)
    for ax, metric in zip(axes, ("PCC", "RMSE")):
        q = g[g.metric == metric].set_index("region").loc[list(REGIONS)]
        ax.axvline(0, color="#777777", linestyle="--", lw=1)
        for j, (_, row) in enumerate(q.iterrows()):
            ax.errorbar(row.estimate, j, xerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]],
                        fmt="o", color="#75528C", ms=7, capsize=5, lw=1.7)
        lo, hi = min(0, float(q.ci_low.min())), max(0, float(q.ci_high.max()))
        pad = max(.005, (hi - lo) * .22)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_xlabel(r"$\Delta$ median " + metric + " (ranked minus equal)")
        ax.set_title("Positive = higher shape agreement" if metric == "PCC" else "Positive = greater profile disagreement")
        ax.set_yticks(range(2), REGIONS.values())
        ax.set_ylim(1.5, -.5)
        ax.grid(axis="x")
    label = "Partial comparison" if n_panels < n_planned else "Matched comparison"
    fig.suptitle(f"{label}: {n_panels}/{n_planned} panels, paired weighting differences", y=1.01, fontsize=13)
    fig.text(.5, .01, "95\\% paired transcript-bootstrap intervals; zero denotes no change.\n"
             "Inter-panel reproducibility is not biological ground-truth accuracy.", ha="center", fontsize=12)
    fig.subplots_adjust(left=.12, right=.99, top=.84, bottom=.27, wspace=.24)
    save_figure(fig, out / "paired_weighting_effects")


@latex_paper_style
def plot_sensitivity(sensitivity, out):
    panels = sorted(sensitivity.panel.unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, region in zip(axes, REGIONS):
        values = [sensitivity.loc[(sensitivity.region == region) & (sensitivity.panel == p), "PCC"].dropna().to_numpy()
                  for p in panels]
        ax.boxplot(values, positions=range(len(panels)), widths=.45, showfliers=False,
                   patch_artist=True, boxprops=dict(facecolor="#E8E0F0", edgecolor="#75528C"),
                   medianprops=dict(color="#75528C", linewidth=2))
        ax.set_xticks(range(len(panels)), ["Panel " + str(int(p.split("_")[-1])) for p in panels])
        ax.set_ylim(-1, 1.04)
        ax.set_ylabel(r"Same-panel PCC: equal vs. ranked $\pi$")
        ax.set_title(REGIONS[region])
        ax.grid(axis="y")
    fig.text(.5, .01, "Same datasets, different reference weights. Boxes: median/IQR; whiskers: 1.5 IQR.\n"
             "This measures sensitivity to weighting, not recovery of a known biological profile.", ha="center", fontsize=12)
    fig.subplots_adjust(left=.08, right=.99, top=.89, bottom=.25, wspace=.27)
    save_figure(fig, out / "same_panel_weighting_sensitivity")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equal-root", type=Path, default=DEFAULT_EQUAL)
    parser.add_argument("--ranked-root", type=Path, default=DEFAULT_RANKED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analyses/artifacts/real_data/panels_equal_vs_ranked",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260910)
    parser.add_argument("--require-all-panels", action="store_true")
    parser.add_argument("--ranking-table", type=Path, default=None,
                        help="Optional relocated ranking file; must match the saved SHA256.")
    args = parser.parse_args(argv)
    if args.bootstrap_replicates < 100:
        parser.error("Use at least 100 bootstrap replicates (paper default 5000).")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ids, planned, weights, differences, audit = audit_design(args.equal_root, args.ranked_root, args.ranking_table)
    weights.to_csv(out / "reference_weights.csv", index=False)
    differences.to_csv(out / "configuration_differences.csv", index=False)
    profiles, availability = {}, []
    for policy, root in (("equal", args.equal_root), ("ranked", args.ranked_root)):
        profiles[policy], rows = collect(root, planned, ids, policy)
        availability.extend(rows)
    pd.DataFrame(availability).to_csv(out / "panel_availability.csv", index=False)
    common = sorted(set(profiles["equal"]) & set(profiles["ranked"]))
    if len(common) < 2 or (args.require_all_panels and common != planned):
        raise ValueError(f"Insufficient matched completed panels: {common}; planned {planned}")
    print(f"Matched panels: {common}; {len(ids)} common held-out transcripts", flush=True)
    values, sensitivity = paired_rows(profiles, ids, common)
    del profiles
    values.to_csv(out / "matched_inter_panel_metrics.csv", index=False)
    values.to_parquet(out / "matched_inter_panel_metrics.parquet", index=False)
    sensitivity.to_csv(out / "same_panel_weighting_metrics.csv", index=False)
    summary = summarize(values, ids, args.bootstrap_replicates, args.bootstrap_seed)
    summary.to_csv(out / "matched_bootstrap_summary.csv", index=False)
    sensitivity.groupby(["region", "panel"])[list(METRICS)].agg(["count", "median", "mean"]).to_csv(
        out / "same_panel_weighting_summary.csv")
    values.groupby(["region", "policy", "reason"]).size().rename("rows").to_csv(out / "validity_counts.csv")
    plot_comparison(summary, out, len(common), len(planned))
    plot_pairs(summary, out)
    plot_paired_effects(summary, out, len(common), len(planned))
    plot_sensitivity(sensitivity, out)
    command_args = [str(Path(sys.executable).absolute()), str(Path(__file__).resolve()),
                         "--equal-root", str(args.equal_root.resolve()), "--ranked-root", str(args.ranked_root.resolve()),
                         "--output-dir", str(out), "--bootstrap-replicates", str(args.bootstrap_replicates),
                         "--bootstrap-seed", str(args.bootstrap_seed)]
    if args.require_all_panels:
        command_args.append("--require-all-panels")
    if args.ranking_table is not None:
        command_args.extend(['--ranking-table', str(args.ranking_table.resolve())])
    command = shlex.join(command_args)
    (out / "reproduce.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n")
    provenance = dict(equal_root=str(args.equal_root.resolve()), ranked_root=str(args.ranked_root.resolve()),
                      planned_panels=planned, matched_panels=common, complete=common == planned,
                      common_test_count=len(ids), common_test_id_hash=transcript_id_hash(ids),
                      design_audit=audit, availability=availability, bootstrap_replicates=args.bootstrap_replicates,
                      bootstrap_seed=args.bootstrap_seed, cluster_unit="transcript, with all panel pairs and both policies retained",
                      statistic="median of transcript/pair values; paired difference of policy medians",
                      validity="Variance <= 1e-12 invalidates PCC and Spearman; matched complete-case transcript cohorts per statistic",
                      spearman_tie_convention="Average ranks at matched positions, separately within each CDS region",
                      profile_transformations="none; CDS interior uses original profile values without renormalization",
                      ground_truth_available=False, font="LaTeX + lmodern; minimum 12 pt",
                      caveat="Saved design/configuration match except reference weights. Original git commits unavailable; identical training source cannot be verified.",
                      peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2,
                      command=command, script_sha256=sha256(Path(__file__)))
    (out / "comparison_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    main_summary = summary[summary.group == "matched_all_pairs"]
    report = ["# Equal versus quality-ranked panel reproducibility", "",
              f"Completed matched panels: {', '.join(common)} ({len(common)}/{len(planned)}); common test transcripts: {len(ids):,}.", "",
              "Only identical available panel pairs are compared. Missing ranked panels are not replaced with equal-weight panels.", "",
              "| Region | Metric | Equal median [95% CI] | Ranked median [95% CI] | Ranked minus equal [95% CI] |", "|---|---|---|---|---|"]
    for region in REGIONS:
        for metric in METRICS:
            g = main_summary[(main_summary.region == region) & (main_summary.metric == metric)].set_index("policy")
            cells = [f"{g.loc[p, 'estimate']:.4f} [{g.loc[p, 'ci_low']:.4f}, {g.loc[p, 'ci_high']:.4f}]"
                     for p in ("equal", "ranked", "ranked_minus_equal")]
            report.append("| " + " | ".join([REGIONS[region], metric, *cells]) + " |")
    report += ["", f"The paired bootstrap resamples transcripts with all their panel pairs and both policies together ({args.bootstrap_replicates:,} replicates; seed {args.bootstrap_seed}). It does not quantify variation over training seeds or alternative panel partitions.", "",
               "PCC measures linear profile agreement, Spearman uses average ranks for tied positions, and RMSE measures disagreement between predictions, not error against biological ground truth. These measure reproducibility, not biological accuracy. The interior excludes 20 codons per end without renormalizing; all three metrics and their paired bootstrap intervals are in the source tables.", "",
               "The saved panel memberships, train/validation/test IDs, training hyperparameters and local reliability references match. Reference pi differs; its entries are normalized within each panel. Original source-code identity is not verifiable because the original git commits were not recorded.", "",
               "Figures: [median agreement](equal_vs_ranked_panel_reproducibility.pdf), [paired changes](paired_weighting_effects.pdf), [each matched pair](equal_vs_ranked_each_panel_pair.pdf), [same-panel sensitivity](same_panel_weighting_sensitivity.pdf). PNG copies and CSV/Parquet source tables are also provided. All labels use real LaTeX/Latin Modern, at least 12 pt.", "",
               ("Re-run `bash analyses/run_real_panel_weighting_analysis.sh` after downloading newly finished panels. Do not treat this partial comparison as the final four-panel result."
                if common != planned else "All four matched panels are available; six panel pairs are included."), ""]
    (out / "README.md").write_text("\n".join(report))
    print(main_summary[["region", "metric", "policy", "estimate", "ci_low", "ci_high", "n_transcripts"]].to_string(index=False))
    print(f"Figures and tables: {out}; peak RSS {provenance['peak_rss_gb']:.2f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
