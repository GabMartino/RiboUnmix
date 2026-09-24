#!/usr/bin/env python3
"""Analyze best-first versus worst-first cumulative dataset selection."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_cumulative_selection_direction import DEFAULT_OUTPUT
from run_cumulative_stability import object_sha256, sha256, write_json
from Utils.quality_selection_experiments import CUMULATIVE_DESIGN
from analyses.analyze_rank_balanced_reference_directionality import relocate
from analyses.analyze_real_exp8_reference_directionality import collect, pair_record
from analyses.paths import artifact_directory


COLORS = {("best_first", "equal"): "#267eab", ("worst_first", "equal"): "#d17820",
          ("best_first", "quality_p3"): "#006d50", ("worst_first", "quality_p3"): "#9a519b"}
STYLES = {"equal": "-", "quality_p3": "--"}


def _table(frame):
    return frame.to_html(index=False, border=0, na_rep="—", float_format=lambda x: f"{x:.4f}")


def _task_profile(profiles, task):
    return profiles.get((task["training_seed"], task["arm"], task["panel_id"]))


def metric_rows(profiles, tasks, ids):
    task_by = {(t["selection_direction"], t["reference_policy"], int(t["N"])): t for t in tasks}
    anchor_task = task_by[("best_first", "equal", 2)]
    anchor = _task_profile(profiles, anchor_task)
    anchor_rows, same_n_rows, adjacent_rows = [], [], []
    if anchor is not None:
        for task in tasks:
            current = _task_profile(profiles, task)
            if current is None:
                continue
            directions = (["best_first", "worst_first"]
                          if task["selection_direction"] == "shared_full"
                          else [task["selection_direction"]])
            for direction in directions:
                for tid in ids:
                    anchor_rows.append(dict(
                        N=task["N"], selection_direction=direction,
                        source_direction=task["selection_direction"],
                        reference_policy=task["reference_policy"], transcript_id=tid,
                        **pair_record(anchor[tid], current[tid], "full_cds")))
    sizes = sorted({int(t["N"]) for t in tasks})
    for n in sizes:
        if n == max(sizes):
            continue
        for policy in ("equal", "quality_p3"):
            best_task = task_by.get(("best_first", policy, n))
            worst_task = task_by.get(("worst_first", policy, n))
            if best_task is None or worst_task is None:
                continue
            best, worst = _task_profile(profiles, best_task), _task_profile(profiles, worst_task)
            if best is None or worst is None:
                continue
            for tid in ids:
                same_n_rows.append(dict(N=n, reference_policy=policy, transcript_id=tid,
                    **pair_record(best[tid], worst[tid], "full_cds")))
    full_n = max(sizes)
    for direction in ("best_first", "worst_first"):
        for policy in ("equal", "quality_p3"):
            path = [task_by.get((direction, policy, n)) for n in sizes if n < full_n]
            path.append(task_by.get(("shared_full", policy, full_n)))
            path = [task for task in path if task is not None]
            for left_task, right_task in zip(path[:-1], path[1:]):
                left, right = _task_profile(profiles, left_task), _task_profile(profiles, right_task)
                if left is None or right is None:
                    continue
                for tid in ids:
                    adjacent_rows.append(dict(selection_direction=direction, reference_policy=policy,
                        N_a=left_task["N"], N_b=right_task["N"], transcript_id=tid,
                        **pair_record(left[tid], right[tid], "full_cds")))
    columns = ["PCC", "RMSE", "variance_a", "variance_b", "reason"]
    return (pd.DataFrame(anchor_rows, columns=["N", "selection_direction", "source_direction",
                "reference_policy", "transcript_id", *columns]),
            pd.DataFrame(same_n_rows, columns=["N", "reference_policy", "transcript_id", *columns]),
            pd.DataFrame(adjacent_rows, columns=["selection_direction", "reference_policy", "N_a", "N_b",
                "transcript_id", *columns]))


def within_path_anchor_rows(profiles, tasks, ids):
    """Compare each path with its own policy-matched N=2 solution.

    These diagnostics remain estimable when one selection direction is only
    partially available.  They are deliberately separate from the primary
    common best-N=2/equal anchor used for the matched experiment.
    """
    task_by = {(t["selection_direction"], t["reference_policy"], int(t["N"])): t
               for t in tasks}
    sizes = sorted({int(t["N"]) for t in tasks})
    full_n = max(sizes)
    rows = []
    for direction in ("best_first", "worst_first"):
        for policy in ("equal", "quality_p3"):
            anchor_task = task_by.get((direction, policy, 2))
            anchor = _task_profile(profiles, anchor_task) if anchor_task is not None else None
            if anchor is None:
                continue
            path = [task_by.get((direction, policy, n)) for n in sizes if n < full_n]
            path.append(task_by.get(("shared_full", policy, full_n)))
            for task in (task for task in path if task is not None):
                current = _task_profile(profiles, task)
                if current is None:
                    continue
                for tid in ids:
                    rows.append(dict(
                        selection_direction=direction,
                        reference_policy=policy,
                        anchor_N=2,
                        N=task["N"],
                        transcript_id=tid,
                        **pair_record(anchor[tid], current[tid], "full_cds")))
    columns = ["PCC", "RMSE", "variance_a", "variance_b", "reason"]
    return pd.DataFrame(rows, columns=["selection_direction", "reference_policy", "anchor_N", "N",
                                       "transcript_id", *columns])


def summarize(frame, group_columns):
    rows = []
    for keys, group in frame.groupby(group_columns, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        common = dict(zip(group_columns, keys))
        for metric in ("PCC", "RMSE"):
            values = group[metric].to_numpy(float)
            values = values[np.isfinite(values)]
            rows.append(dict(**common, metric=metric, n_valid=len(values),
                             mean=float(values.mean()) if len(values) else np.nan,
                             median=float(np.median(values)) if len(values) else np.nan))
    return pd.DataFrame(rows, columns=[*group_columns, "metric", "n_valid", "mean", "median"])


def plot_results(anchor_summary, direct_summary, membership, sizes, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for direction in ("best_first", "worst_first"):
        for policy in ("equal", "quality_p3"):
            color = COLORS[direction, policy]
            label = f"{direction}, {policy}"
            selected = anchor_summary[(anchor_summary.selection_direction == direction)
                                      & (anchor_summary.reference_policy == policy)]
            for ax, metric in zip(axes[0], ("PCC", "RMSE")):
                values = selected[selected.metric == metric].set_index("N").reindex(sizes)["mean"]
                ax.plot(sizes, values, marker="o", linestyle=STYLES[policy], color=color, label=label)
    axes[0, 0].set(title="Agreement with the same best-N=2 equal anchor", ylabel="Mean transcript PCC")
    axes[0, 1].set(title="Distance from the same best-N=2 equal anchor", ylabel="Mean transcript RMSE")

    for policy, color in (("equal", "#334155"), ("quality_p3", "#006d50")):
        selected = direct_summary[(direct_summary.reference_policy == policy)
                                  & (direct_summary.metric == "PCC")].set_index("N")
        axes[1, 0].plot(sizes, selected.reindex(sizes)["mean"], "o-", color=color, label=policy)
    axes[1, 0].set(title="Direct best-N versus worst-N agreement", ylabel="Mean transcript PCC")
    overlap = []
    for n in sizes:
        groups = membership[membership.N == n].groupby("selection_direction").dataset_id.apply(set)
        if {"best_first", "worst_first"}.issubset(groups.index):
            a, b = groups["best_first"], groups["worst_first"]
            overlap.append((n, len(a & b) / len(a | b)))
        else:
            overlap.append((n, 1.0))
    axes[1, 1].plot([x for x, _ in overlap], [y for _, y in overlap], "o-", color="#64748b")
    axes[1, 1].set(title="Dataset-set overlap", ylabel="Jaccard(best-N, worst-N)", ylim=(-.03, 1.05))
    for ax in axes.flat:
        ax.set(xlabel="Number of selected datasets N", xticks=sizes)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "selection_direction_results.svg")
    fig.savefig(out / "selection_direction_results.pdf")
    plt.close(fig)


def _adaptive_pcc_axis(ax, values):
    values = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(values):
        return
    lower, upper = float(values.min()), float(values.max())
    span = max(upper - lower, 0.12)
    ax.set_ylim(max(-1.0, lower - 0.12 * span), min(1.01, upper + 0.12 * span))


def _pcc_distribution(frame, group_columns):
    selected = frame[frame["reason"].eq("ok") & np.isfinite(frame["PCC"])].copy()
    if selected.empty:
        return pd.DataFrame(columns=[*group_columns, "mean", "q10", "q90"])
    return (selected.groupby(group_columns, as_index=False)["PCC"]
            .agg(mean="mean", q10=lambda x: x.quantile(.10), q90=lambda x: x.quantile(.90)))


def plot_worst_first(adjacent, within_anchor, concentration, sizes, out):
    """Plot currently estimable worst-first results without requiring best-first runs."""
    adjacent_pcc = _pcc_distribution(
        adjacent[adjacent["selection_direction"].eq("worst_first")],
        ["reference_policy", "N_a", "N_b"])
    anchor_pcc = _pcc_distribution(
        within_anchor[within_anchor["selection_direction"].eq("worst_first")],
        ["reference_policy", "N"])
    worst_concentration = concentration[
        concentration["selection_direction"].eq("worst_first")].copy()
    if adjacent_pcc.empty and anchor_pcc.empty:
        return False

    labels = {"equal": "Equal reference", "quality_p3": r"Quality weighted, $p=3$"}
    colors = {"equal": COLORS[("worst_first", "equal")],
              "quality_p3": COLORS[("worst_first", "quality_p3")]}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))

    for policy in ("equal", "quality_p3"):
        group = adjacent_pcc[adjacent_pcc["reference_policy"].eq(policy)].sort_values("N_b")
        if not group.empty:
            yerr = np.vstack([group["mean"] - group["q10"], group["q90"] - group["mean"]])
            axes[0].errorbar(group["N_b"], group["mean"], yerr=yerr, marker="o", capsize=3,
                             lw=2, color=colors[policy], label=labels[policy])
        group = anchor_pcc[anchor_pcc["reference_policy"].eq(policy)].sort_values("N")
        if not group.empty:
            yerr = np.vstack([group["mean"] - group["q10"], group["q90"] - group["mean"]])
            axes[1].errorbar(group["N"], group["mean"], yerr=yerr, marker="o", capsize=3,
                             lw=2, color=colors[policy], label=labels[policy])
        group = worst_concentration[worst_concentration["reference_policy"].eq(policy)].sort_values("N")
        if not group.empty:
            axes[2].plot(group["N"], group["N_ref"] / group["N"], "o-", lw=2,
                         color=colors[policy], label=labels[policy])

    axes[0].set_title("A  Adjacent worst-first stability", fontweight="bold", loc="left")
    axes[0].set_ylabel(r"Mean transcript PCC of $\mathbf{L}_t$", fontweight="bold")
    axes[0].set_xlabel(r"New collection size $N_b$", fontweight="bold")
    axes[1].set_title(r"B  Agreement with own $N=2$ anchor", fontweight="bold", loc="left")
    axes[1].set_ylabel(r"Mean transcript PCC of $\mathbf{L}_t$", fontweight="bold")
    axes[1].set_xlabel("Number of selected datasets $N$", fontweight="bold")
    axes[2].set_title("C  Gamma-reference concentration", fontweight="bold", loc="left")
    axes[2].set_ylabel(r"Effective fraction $N_{\mathrm{eff}}/N$", fontweight="bold")
    axes[2].set_xlabel("Number of selected datasets $N$", fontweight="bold")

    _adaptive_pcc_axis(axes[0], adjacent_pcc["mean"])
    _adaptive_pcc_axis(axes[1], anchor_pcc["mean"])
    axes[2].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(sizes, [str(n) for n in sizes])
        ax.grid(alpha=.22)
        ax.tick_params(axis="both", labelsize=10)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
    axes[0].legend(frameon=False, fontsize=10)
    fig.suptitle("Worst-first path: progressively better-ranked datasets are added",
                 fontsize=14, fontweight="bold")
    fig.text(.5, .015, "Points are transcript means; bars span the 10th–90th transcript percentiles.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .93))
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(out / f"worst_first_partial_results.{suffix}", dpi=300)
    plt.close(fig)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    root = args.experiment_root.expanduser().resolve()
    manifest = json.loads((root / "experiment_manifest.json").read_text())
    if manifest["experiment_design"] != CUMULATIVE_DESIGN or object_sha256(manifest["tasks"]) != manifest["tasks_sha256"]:
        raise ValueError("Expected an unchanged cumulative dataset-selection experiment.")
    recorded = Path(manifest["output_root"])
    for original, expected in manifest["frozen_file_sha256"].items():
        if sha256(relocate(original, root, recorded)) != expected:
            raise ValueError(f"Changed frozen input: {original}")
    tasks, seed = manifest["tasks"], manifest["training_seeds"][0]
    ids = manifest["source_folds"][tasks[0]["panel_id"]]["test_ids"]
    if any(fold["test_ids"] != ids or fold["train_ids"] != manifest["source_folds"][tasks[0]["panel_id"]]["train_ids"]
           or fold["validation_ids"] != manifest["source_folds"][tasks[0]["panel_id"]]["validation_ids"]
           for fold in manifest["source_folds"].values()):
        raise ValueError("Transcript folds are not identical across collections.")
    configs = {(t["training_seed"], t["arm"], t["panel_id"]):
               yaml.safe_load(relocate(t["config_path"], root, recorded).read_text()) for t in tasks}
    weights = pd.read_csv(root / "reference_weights.csv")
    profiles, availability = collect(root, manifest, configs, ids, weights)
    metadata = pd.DataFrame(tasks)[["run_id", "selection_direction", "reference_policy", "panel_id"]]
    availability = availability.merge(metadata, left_on="task_id", right_on="run_id", how="left")
    anchor, direct, adjacent = metric_rows(profiles, tasks, ids)
    within_anchor = within_path_anchor_rows(profiles, tasks, ids)
    anchor_summary = summarize(anchor, ["N", "selection_direction", "reference_policy"])
    direct_summary = summarize(direct, ["N", "reference_policy"])
    adjacent_summary = summarize(adjacent, ["selection_direction", "reference_policy", "N_a", "N_b"])
    within_anchor_summary = summarize(
        within_anchor, ["selection_direction", "reference_policy", "anchor_N", "N"])
    out = artifact_directory("real_data", root)
    out.mkdir(exist_ok=True)
    for name, frame in (("availability", availability), ("anchor_metrics", anchor),
                        ("anchor_summary", anchor_summary), ("best_vs_worst_metrics", direct),
                        ("best_vs_worst_summary", direct_summary), ("adjacent_metrics", adjacent),
                        ("adjacent_summary", adjacent_summary),
                        ("within_path_anchor_metrics", within_anchor),
                        ("within_path_anchor_summary", within_anchor_summary)):
        frame.to_csv(out / f"{name}.csv", index=False)
    membership = pd.read_csv(root / "inputs/collection_membership.csv")
    concentration = pd.read_csv(root / "reference_concentration.csv")
    sizes = list(manifest["sizes"])
    plot_results(anchor_summary, direct_summary, membership, sizes, out)
    has_worst_figure = plot_worst_first(
        adjacent, within_anchor, concentration, sizes, out)
    complete = int(availability.status.eq("validated_predictions").sum())
    worst_adjacent_pcc = adjacent_summary[
        adjacent_summary["selection_direction"].eq("worst_first")
        & adjacent_summary["metric"].eq("PCC")]
    worst_anchor_pcc = within_anchor_summary[
        within_anchor_summary["selection_direction"].eq("worst_first")
        & within_anchor_summary["metric"].eq("PCC")]
    worst_figure_html = ('''
<h2>Available worst-first trajectory</h2>
<img src="worst_first_partial_results.svg" alt="Worst-first cumulative profile stability and reference concentration">
<p>Increasing <i>N</i> along this path adds progressively better-ranked datasets. Lower stability under q<sup>3</sup> is therefore a directional response: the gamma reference moves toward the newly added, better datasets. It is not evidence that ranking should preserve a solution defined initially by the worst datasets. Error bars show transcript heterogeneity, not uncertainty over retraining.</p>
<h3>Adjacent PCC</h3>''' + _table(worst_adjacent_pcc) + '''
<h3>Agreement with each policy's own worst-first <i>N</i>=2 anchor</h3>''' + _table(worst_anchor_pcc)
        if has_worst_figure else
        '<h2>Available worst-first trajectory</h2><p>No validated adjacent worst-first exports are available yet.</p>')
    primary_figure_html = ('''
<h2>Primary matched best-first versus worst-first analysis</h2>
<img src="selection_direction_results.svg" alt="Best-first and worst-first cumulative selection results">'''
        if not anchor_summary.empty or not direct_summary.empty else
        '''<h2>Primary matched best-first versus worst-first analysis</h2>
<p>The matched primary plot is withheld because the best-first <i>N</i>=2 anchor and matched best-versus-worst exports are not yet available.</p>''')
    body = f'''<!doctype html><html><head><meta charset="utf-8"><title>Dataset-selection directionality</title>
<style>body{{max-width:1100px;margin:35px auto;padding:0 24px;font:16px/1.6 system-ui;color:#203448}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:6px 9px;border-bottom:1px solid #ddd}}img{{max-width:100%}}.note{{padding:14px;background:#f3f7fb;border-left:4px solid #267eab}}</style></head><body>
<h1>Best-first versus worst-first cumulative dataset selection</h1>
<p class="note">{complete}/{len(tasks)} models currently have validated exports. All comparisons use the same {len(ids):,} complete-cohort test transcripts and the same CDS coordinates.</p>
{worst_figure_html}
{primary_figure_html}
<h2>How to interpret the curves</h2>
<p>The primary equal-reference contrast changes dataset membership directly. A lower best-<i>N</i> versus worst-<i>N</i> PCC is evidence that dataset quality composition changes <b>L<sub>t</sub></b>. The common best-<i>N</i>=2 anchor shows whether adding worse datasets moves the best-first path away and whether adding better datasets moves the worst-first path toward it.</p>
<p>Best and worst collections are disjoint through N=40, overlap at N=80, and are identical at N=114. The two paths must converge at the full collection, so a monotonic separation through N=114 is neither expected nor possible.</p>
<p>The q³ curves test a separate gamma-reference intervention within each selected collection. They must not be described as reversed dataset selection.</p>
<h2>Anchor summary</h2>{_table(anchor_summary)}
<h2>Direct best-N versus worst-N summary</h2>{_table(direct_summary)}
<h2>Availability</h2>{_table(availability[["N", "selection_direction", "reference_policy", "status"]])}
<p><a href="../design_report.html">Frozen design</a> · <a href="anchor_metrics.csv">Per-transcript anchor metrics</a> · <a href="best_vs_worst_metrics.csv">Per-transcript direct contrasts</a></p>
</body></html>'''
    (out / "analysis_report.html").write_text(body)
    write_json(out / "analysis_manifest.json", dict(experiment_design=CUMULATIVE_DESIGN,
        validated_models=complete, planned_models=len(tasks), common_test_transcripts=len(ids),
        analysis_code_sha256=sha256(Path(__file__))))
    print(f"{complete}/{len(tasks)} validated exports; report: {out / 'analysis_report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
