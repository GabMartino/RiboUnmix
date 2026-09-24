#!/usr/bin/env python3
"""Analyze four contiguous dataset-quality strata on a common transcript cohort."""
from __future__ import annotations

import argparse
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

from run_four_quality_strata import DEFAULT_OUTPUT
from run_cumulative_stability import object_sha256, sha256, write_json
from Utils.quality_selection_experiments import FOUR_PANEL_DESIGN
from analyses.analyze_rank_balanced_reference_directionality import relocate
from analyses.analyze_real_exp8_reference_directionality import collect, pair_record
from analyses.paths import artifact_directory


PANELS = ("q1_best", "q2_upper_middle", "q3_lower_middle", "q4_worst")
POLICIES = ("equal", "quality_p3")
COLORS = {"equal": "#267eab", "quality_p3": "#006d50"}


def _table(frame):
    return frame.to_html(index=False, border=0, na_rep="—", float_format=lambda x: f"{x:.4f}")


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    root = args.experiment_root.expanduser().resolve()
    manifest = json.loads((root / "experiment_manifest.json").read_text())
    if manifest["experiment_design"] != FOUR_PANEL_DESIGN or object_sha256(manifest["tasks"]) != manifest["tasks_sha256"]:
        raise ValueError("Expected an unchanged four-quality-strata experiment.")
    recorded = Path(manifest["output_root"])
    for original, expected in manifest["frozen_file_sha256"].items():
        if sha256(relocate(original, root, recorded)) != expected:
            raise ValueError(f"Changed frozen input: {original}")
    tasks = manifest["tasks"]
    ids = manifest["source_folds"][PANELS[0]]["test_ids"]
    first_fold = manifest["source_folds"][PANELS[0]]
    if any(fold["train_ids"] != first_fold["train_ids"] or
           fold["validation_ids"] != first_fold["validation_ids"] or
           fold["test_ids"] != ids for fold in manifest["source_folds"].values()):
        raise ValueError("Transcript folds are not identical across quality strata.")
    configs = {(t["training_seed"], t["arm"], t["panel_id"]):
               yaml.safe_load(relocate(t["config_path"], root, recorded).read_text()) for t in tasks}
    weights = pd.read_csv(root / "reference_weights.csv")
    profiles, availability = collect(root, manifest, configs, ids, weights)
    task_by = {(t["panel_id"], t["reference_policy"]): t for t in tasks}
    rows = []
    common_anchor_task = task_by[(PANELS[0], "equal")]
    common_anchor = profiles.get((common_anchor_task["training_seed"], common_anchor_task["arm"], PANELS[0]))
    if common_anchor is not None:
        for panel in PANELS:
            for policy in POLICIES:
                task = task_by[(panel, policy)]
                current = profiles.get((task["training_seed"], task["arm"], panel))
                if current is None:
                    continue
                for tid in ids:
                    rows.append(dict(panel_id=panel, quality_stratum=task["quality_stratum"],
                        reference_policy=policy, transcript_id=tid,
                        **pair_record(common_anchor[tid], current[tid], "full_cds")))
    rows = pd.DataFrame(rows, columns=["panel_id", "quality_stratum", "reference_policy", "transcript_id",
                                      "PCC", "RMSE", "variance_a", "variance_b", "reason"])
    summary = summarize(rows, ["panel_id", "quality_stratum", "reference_policy"])
    out = artifact_directory("real_data", root)
    out.mkdir(exist_ok=True)
    rows.to_csv(out / "quality_stratum_metrics.csv", index=False)
    summary.to_csv(out / "quality_stratum_summary.csv", index=False)
    metadata = pd.DataFrame(tasks)[["run_id", "quality_stratum", "reference_policy"]]
    availability = availability.merge(metadata, left_on="task_id", right_on="run_id", how="left")
    availability.to_csv(out / "availability.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for policy in POLICIES:
        selected = summary[summary.reference_policy == policy]
        for ax, metric in zip(axes, ("PCC", "RMSE")):
            values = selected[selected.metric == metric].set_index("quality_stratum").reindex(range(1, 5))["mean"]
            ax.plot(range(1, 5), values, "o-" if policy == "equal" else "o--",
                    color=COLORS[policy], label=policy)
    axes[0].set(title="Agreement with the same q1-best equal anchor", ylabel="Mean transcript PCC")
    axes[1].set(title="Distance from the same q1-best equal anchor", ylabel="Mean transcript RMSE")
    for ax in axes:
        ax.set(xlabel="Dataset quality stratum", xticks=range(1, 5),
               xticklabels=["Q1 best", "Q2", "Q3", "Q4 worst"])
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / "quality_strata_results.svg")
    fig.savefig(out / "quality_strata_results.pdf")
    plt.close(fig)

    complete = int(availability.status.eq("validated_predictions").sum())
    body = f'''<!doctype html><html><head><meta charset="utf-8"><title>Four quality strata</title>
<style>body{{max-width:1050px;margin:35px auto;padding:0 24px;font:16px/1.6 system-ui;color:#203448}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:6px 9px;border-bottom:1px solid #ddd}}img{{max-width:100%}}.note{{padding:14px;background:#f3f7fb;border-left:4px solid #267eab}}</style></head><body>
<h1>Shared-profile agreement from the best to worst dataset-quality stratum</h1>
<p class="note">{complete}/{len(tasks)} models currently have validated exports. Every panel uses the same {len(ids):,} train/validation/test cohort partition.</p>
<img src="quality_strata_results.svg" alt="Agreement across dataset quality strata">
<p>The primary equal-reference curve changes actual dataset membership from ranks 1–29 to the worst 28 configured ranks. A falling PCC and rising RMSE would directly show that the QC-ranked composition changes L_bio. The q³ curve asks whether quality weighting within each stratum changes that result.</p>
<p>These four strata may split related datasets from one study. They are controlled quality bands, not four independent biological replications, and agreement with Q1 does not establish that Q1 is ground truth.</p>
<h2>Summary</h2>{_table(summary)}
<h2>Availability</h2>{_table(availability[["panel_id", "quality_stratum", "reference_policy", "status"]])}
<p><a href="../design_report.html">Frozen design</a> · <a href="quality_stratum_metrics.csv">Per-transcript metrics</a></p>
</body></html>'''
    (out / "analysis_report.html").write_text(body)
    write_json(out / "analysis_manifest.json", dict(experiment_design=FOUR_PANEL_DESIGN,
        validated_models=complete, planned_models=len(tasks), common_test_transcripts=len(ids),
        analysis_code_sha256=sha256(Path(__file__))))
    print(f"{complete}/{len(tasks)} validated exports; report: {out / 'analysis_report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
