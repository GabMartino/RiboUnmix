#!/usr/bin/env python3
"""Train four fixed panels under directional quality-score gamma references.

The primary design uses the existing source-disjoint, quality-balanced panel
assignment.  Panel membership is held fixed while the gamma reference is
uniform, best-oriented, or worst-oriented using the aggregate
``quality_rank_score`` with powers 1, 3 and 5.  An optional ``quality_strata``
design applies the same seven policies to four contiguous QC-rank strata.

All panels share one complete-case train/validation/test transcript split and
dataset reliability references fitted once on the common training fold.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

from run_cumulative_stability import (
    ROOT,
    make_config,
    object_sha256,
    run_task,
    sha256,
    write_json,
)
from run_four_panel_stability import load_panels
from Utils.cumulative_fixed_cohort import build_fixed_cumulative_split
from Utils.quality_selection_experiments import (
    directional_quality_score_weights,
    four_quality_collections,
)
from Utils.real_panel_convergence import (
    fit_panel_reliability_manifest,
    infer_source_identifier,
)


BALANCED_DESIGN = "four_panel_quality_score_directional_balanced_v1"
STRATA_DESIGN = "four_panel_quality_score_directional_strata_v1"
DEFAULT_BALANCED_OUTPUT = ROOT / "results/four_panel_quality_score_directional_balanced_seed42"
DEFAULT_STRATA_OUTPUT = ROOT / "results/four_panel_quality_score_directional_strata_seed42"
DEFAULT_PANELS = ROOT / "config/experiment_designs/panels_equal_seed42_20260906_114323.json"

# Orientation names deliberately match the cumulative score experiment.  In
# this fixed-membership experiment they describe reference orientation, not a
# change in dataset selection order.
REFERENCE_VARIANTS = (
    ("equal", "equal", 0),
    ("best_first", "score_p1", 1),
    ("worst_first", "score_p1", 1),
    ("best_first", "score_p3", 3),
    ("worst_first", "score_p3", 3),
    ("best_first", "score_p5", 5),
    ("worst_first", "score_p5", 5),
)
TASK_COUNT = 4 * len(REFERENCE_VARIANTS)


def _validate_score_table(mapping, ranking):
    required = {"dataset", "quality_rank", "quality_rank_score"}
    if not required <= set(ranking):
        raise ValueError(f"Ranking table lacks columns: {sorted(required - set(ranking))}")
    if ranking.dataset.duplicated().any():
        raise ValueError("Ranking table contains duplicate dataset rows.")
    indexed = ranking.set_index("dataset")
    missing = sorted(set(mapping) - set(indexed.index))
    if missing:
        raise ValueError(f"Configured datasets missing from the ranking: {missing}")
    ranks = indexed.quality_rank.astype(float)
    scores = indexed.quality_rank_score.astype(float)
    configured_ranks = ranks.loc[list(mapping)].to_numpy(float)
    configured_scores = scores.loc[list(mapping)].to_numpy(float)
    if (
        not np.isfinite(configured_ranks).all()
        or (configured_ranks <= 0).any()
        or len(np.unique(configured_ranks)) != len(configured_ranks)
    ):
        raise ValueError("Configured quality ranks must be finite, positive and unique.")
    if not np.isfinite(configured_scores).all() or (configured_scores <= 0).any():
        raise ValueError("Configured quality_rank_score values must be finite and positive.")
    return ranks, scores


def build_panel_collections(panel_design, panels_path, mapping, ranking):
    """Return four exhaustive collections and a saved membership table."""
    ranks, scores = _validate_score_table(mapping, ranking)
    ordered_names = sorted(mapping, key=lambda name: (float(ranks[name]), name))
    if len(ordered_names) != 114:
        raise ValueError(f"Expected the active 114-dataset universe, found {len(ordered_names)}.")

    if panel_design == "balanced":
        panels, assignment, _ = load_panels(panels_path, mapping, ranking)
        collections = [
            dict(
                collection_id=panel,
                panel_id=panel,
                N=len(names),
                panel_design=panel_design,
                datasets=list(names),
            )
            for panel, names in panels.items()
        ]
        source_lookup = assignment.set_index("dataset_name").source_identifier.to_dict()
    elif panel_design == "quality_strata":
        collections = four_quality_collections(ordered_names)
        for collection in collections:
            collection["panel_design"] = panel_design
        source_lookup = {name: infer_source_identifier(name) for name in ordered_names}
    else:
        raise ValueError(f"Unknown panel design: {panel_design!r}")

    sizes = sorted(collection["N"] for collection in collections)
    if len(collections) != 4 or sizes != [28, 28, 29, 29]:
        raise ValueError(f"Expected four panels of sizes 29,29,28,28; found {sizes}.")
    flattened = [name for collection in collections for name in collection["datasets"]]
    if len(flattened) != 114 or set(flattened) != set(mapping) or len(set(flattened)) != 114:
        raise ValueError("Panels must be a disjoint exhaustive partition of the 114 datasets.")

    membership = pd.DataFrame([
        dict(
            panel_id=collection["collection_id"],
            panel_design=panel_design,
            dataset_id=name,
            global_rank=float(ranks[name]),
            quality_rank_score=float(scores[name]),
            source_family=source_lookup[name],
        )
        for collection in collections
        for name in collection["datasets"]
    ])
    return collections, membership, ranks, scores


def _weight_formula(orientation, power):
    if orientation == "equal":
        return "raw_d=1; pi_d=1/N"
    if orientation == "best_first":
        return "raw_d=(global_min_score/quality_rank_score_d)^p; pi_d=raw_d/sum(raw)"
    return "raw_d=(quality_rank_score_d/global_max_score)^p; pi_d=raw_d/sum(raw)"


def _design_report(root, panel_design, collections, membership, concentration, folds, seed):
    rows = []
    for collection in collections:
        panel = collection["collection_id"]
        group = membership[membership.panel_id.eq(panel)]
        rows.append(dict(
            panel=panel,
            datasets=len(group),
            source_families=group.source_family.nunique(),
            minimum_rank=int(group.global_rank.min()),
            maximum_rank=int(group.global_rank.max()),
            mean_rank=float(group.global_rank.mean()),
            mean_quality_rank_score=float(group.quality_rank_score.mean()),
            training_transcripts=len(folds[panel]["train_ids"]),
        ))
    panel_table = pd.DataFrame(rows).to_html(index=False, border=0, float_format=lambda x: f"{x:.2f}")
    geometry = concentration.copy()
    geometry["effective_fraction"] = geometry.N_ref / geometry.N
    geometry_table = geometry[[
        "collection_id", "arm", "N_ref", "effective_fraction",
        "weighted_mean_rank", "weighted_mean_quality_rank_score",
    ]].to_html(index=False, border=0, float_format=lambda x: f"{x:.4f}")
    first_fold = folds[collections[0]["collection_id"]]
    strata_warning = (
        "The quality-strata option deliberately changes dataset composition and can split related source families; "
        "it is a secondary sensitivity experiment rather than the clean cross-panel reproducibility test."
        if panel_design == "quality_strata" else
        "The balanced panels are source-disjoint and have similar rank composition, so cross-panel comparisons primarily test reproducibility across dataset collections."
    )
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Four-panel quality-score directionality</title>
<style>body{{max-width:1080px;margin:35px auto;padding:0 24px;font:16px/1.65 system-ui;color:#203448}}table{{width:100%;border-collapse:collapse;font-size:13px}}td,th{{padding:7px 9px;border-bottom:1px solid #dce5eb;text-align:right}}td:first-child,th:first-child{{text-align:left}}.note{{padding:14px 17px;background:#edf6fa;border-left:4px solid #267eab}}.caution{{background:#fff7ed;border-color:#d17820}}pre{{padding:14px;background:#173146;color:#eef6fa;overflow:auto}}code{{background:#edf1f3;padding:2px 4px}}</style></head><body>
<h1>Four fixed panels with directional quality-score references</h1>
<p class="note"><b>Frozen design:</b> {panel_design}; four panels × seven policies × seed {seed} = {TASK_COUNT} fresh fits. Every fit uses the same {len(first_fold['train_ids']):,} training, {len(first_fold['validation_ids']):,} validation and {len(first_fold['test_ids']):,} test transcript IDs.</p>
<h2>Intervention</h2>
<p>Dataset membership is fixed within each panel. Let <i>s</i><sub>d</sub> be <code>quality_rank_score</code>, where lower is better. The seven references are equal, best-oriented &pi;<sub>d</sub>&prop;<i>s</i><sub>d</sub><sup>&minus;p</sup>, and worst-oriented &pi;<sub>d</sub>&prop;<i>s</i><sub>d</sub><sup>p</sup>, for p∈{{1,3,5}}. The min/max factors stored in the raw weights cancel after normalization.</p>
<p><b>Prespecified interpretation:</b> p=1 is the primary score intervention; p=3 and p=5 are increasingly concentrated mechanism stress tests. In particular, a p=5 result should always be read together with N<sub>eff</sub>/N and the maximum dataset mass.</p>
<p>{strata_warning}</p>
<p class="note caution"><b>Interpretation:</b> “worst-oriented” does not select worse datasets first. It assigns more gamma-reference mass to the worse-scoring datasets already inside the same fixed panel. Moreover, equal p does not guarantee equal N<sub>eff</sub>/N across panels or orientations; use the table below when comparing effects.</p>
<h2>Panel composition</h2>{panel_table}
<h2>Reference geometry</h2>{geometry_table}
<h2>Primary analysis after training</h2>
<p>Compare transcript-level PCC of <b>L<sub>t</sub></b> across all six panel pairs under the same policy. Equal weighting is the reproducibility baseline. Best- and worst-oriented policies test whether the gamma-reference gauge changes that reproducibility. Same-panel policy comparisons quantify how much the reference itself moves <b>L<sub>t</sub></b>.</p>
<p>High cross-panel PCC demonstrates reproducibility, not biological truth. A best-oriented advantage over both equal and worst-oriented references would support the QC direction specifically; similar gains for both orientations would instead implicate concentration.</p>
<h2>Leonardo submission</h2><pre>bash submit_four_panel_quality_score_directionality_leonardo.sh {panel_design}</pre>
<p>The submission wrapper runs one preparation job and then a dependent 28-task GPU array. Completed outputs are validated and skipped on resubmission; interrupted tasks resume only from compatible full-state checkpoints.</p>
<h2>Analysis</h2><pre>python analyses/analyze_four_panel_quality_score_directionality.py \\
  --experiment-root {root}</pre>
<p>The read-only analyzer accepts partial results, retains explicit gaps, and compares every policy on the same held-out transcript IDs and CDS coordinates.</p>
</body></html>'''
    (root / "design_report.html").write_text(document)


def prepare(args):
    root = args.output_root
    design = BALANCED_DESIGN if args.panel_design == "balanced" else STRATA_DESIGN
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        request = dict(
            design=design,
            panel_design=args.panel_design,
            seed=args.seed,
            config_sha256=sha256(args.config),
            dataset_yaml_sha256=sha256(args.datasets),
            ranking_sha256=sha256(args.ranking),
            panels_sha256=(sha256(args.panels) if args.panel_design == "balanced" else None),
            split_protocol="fixed_complete_transcripts",
            reference_weight_source="quality_rank_score_directional",
        )
        manifest_path = root / "experiment_manifest.json"
        if manifest_path.exists():
            plan = json.loads(manifest_path.read_text())
            if (
                plan.get("output_root") != str(root)
                or plan.get("setup_request") != request
                or plan.get("experiment_design") != design
            ):
                raise ValueError("Output directory contains another setup; choose a fresh --output-root.")
            return plan

        base = yaml.safe_load(args.config.read_text())
        mapping = yaml.safe_load(args.datasets.read_text())["dataset_path"]
        mapping = {name: str((ROOT / Path(path)).resolve()) for name, path in mapping.items()}
        base["paths"]["sequences_path"] = str((ROOT / Path(base["paths"]["sequences_path"])).resolve())
        base["paths"]["encodings"] = {
            name: str((ROOT / Path(path)).resolve())
            for name, path in base["paths"]["encodings"].items()
        }
        ranking = pd.read_csv(args.ranking, sep="\t")
        collections, membership, ranks, scores = build_panel_collections(
            args.panel_design, args.panels, mapping, ranking
        )
        panels = {collection["collection_id"]: collection["datasets"] for collection in collections}

        inputs, configs = root / "inputs", root / "configs"
        inputs.mkdir(exist_ok=True)
        configs.mkdir(exist_ok=True)
        shutil.copy2(args.ranking, inputs / "ranking.tsv")
        membership.to_csv(inputs / "panel_assignment.csv", index=False)

        split_tasks = [
            dict(run_id=collection["collection_id"], datasets=collection["datasets"])
            for collection in collections
        ]
        print("Building one complete 114-dataset transcript cohort shared by all panels...", flush=True)
        split = build_fixed_cumulative_split(
            experiment_name=design,
            tasks=split_tasks,
            dataset_mapping=mapping,
            sequences_path=base["paths"]["sequences_path"],
            subset_seed=args.seed,
            validation_fraction=.1,
            test_fraction=.1,
            reliability_bins=10,
            maximum_cds_codons=base["data"].get("max_cds_codons"),
        )
        write_json(inputs / "split.json", split)
        folds = {
            collection["collection_id"]: dict(
                source_panel=collection["collection_id"],
                train_ids=split["panel_train_eligible_ids"][collection["collection_id"]],
                validation_ids=split["panel_validation_ids"][collection["collection_id"]],
                test_ids=split["common_test_ids"],
            )
            for collection in collections
        }
        first_fold = folds[collections[0]["collection_id"]]
        if any(
            fold["train_ids"] != first_fold["train_ids"]
            or fold["validation_ids"] != first_fold["validation_ids"]
            or fold["test_ids"] != first_fold["test_ids"]
            for fold in folds.values()
        ):
            raise AssertionError("Complete-cohort builder returned panel-dependent transcript folds.")

        ordered_names = sorted(mapping, key=lambda name: (float(ranks[name]), name))
        print("Fitting each dataset reliability reference once on the common training fold...", flush=True)
        complete_reference = fit_panel_reliability_manifest(
            experiment_name=design,
            panel_name="complete_114_reference",
            panel_datasets=ordered_names,
            dataset_mapping=mapping,
            panel_training_ids=first_fold["train_ids"],
            validation_ids=first_fold["validation_ids"],
            test_ids=first_fold["test_ids"],
            source_split_manifest=inputs / "split.json",
        )
        write_json(inputs / "reliability_complete_114.json", complete_reference)
        for collection in collections:
            panel, selected = collection["collection_id"], collection["datasets"]
            reference = copy.deepcopy(complete_reference)
            reference["panel_name"] = panel
            reference["datasets"] = {
                name: complete_reference["datasets"][name] for name in selected
            }
            reference["derived_from_complete_114_reference"] = True
            reference["complete_reference_dataset_count"] = len(ordered_names)
            write_json(inputs / f"reliability_{panel}.json", reference)

        score_minimum = float(scores.loc[ordered_names].min())
        score_maximum = float(scores.loc[ordered_names].max())
        source_lookup = membership.set_index("dataset_id").source_family
        tasks, weight_rows = [], []
        for collection in collections:
            panel, selected = collection["collection_id"], collection["datasets"]
            score_values = scores.loc[selected].to_numpy(float)
            for orientation, policy, power in REFERENCE_VARIANTS:
                if orientation == "equal":
                    raw = np.ones(len(selected), dtype=float)
                    base_orientation = np.ones(len(selected), dtype=float)
                    arm = "equal"
                else:
                    raw = directional_quality_score_weights(
                        score_values,
                        power,
                        orientation,
                        global_minimum=score_minimum,
                        global_maximum=score_maximum,
                    )
                    base_orientation = directional_quality_score_weights(
                        score_values,
                        1,
                        orientation,
                        global_minimum=score_minimum,
                        global_maximum=score_maximum,
                    )
                    arm = f"{orientation}_{policy}"
                pi = raw / raw.sum()
                task = dict(
                    array_index=len(tasks),
                    panel_id=panel,
                    N=len(selected),
                    arm=arm,
                    reference_policy=policy,
                    reference_orientation=orientation,
                    power=power,
                    reverse=False,
                    training_seed=args.seed,
                    datasets=selected,
                    source_panel=panel,
                    panel_design=args.panel_design,
                    experiment_design=design,
                    reference_weight_source="quality_rank_score_directional",
                    run_id=f"four_{panel}_{arm}_seed{args.seed}",
                    directory=f"runs/seed{args.seed}/{arm}/{panel}",
                )
                cfg = make_config(base, mapping, task, root, dict(zip(selected, map(float, raw))))
                cfg["orchestrator"].update(
                    panel_design=args.panel_design,
                    reference_weight_source="quality_rank_score_directional",
                    reference_weight_orientation=orientation,
                    reference_weight_formula=_weight_formula(orientation, power),
                    reference_weight_power=power,
                    global_minimum_quality_rank_score=score_minimum,
                    global_maximum_quality_rank_score=score_maximum,
                )
                cfg["orchestrator"].pop("task_contract_sha256")
                cfg["orchestrator"]["task_contract_sha256"] = object_sha256(cfg)
                path = configs / f"{task['run_id']}.yaml"
                path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                task.update(
                    config_path=str(path),
                    config_sha256=sha256(path),
                    task_contract_sha256=cfg["orchestrator"]["task_contract_sha256"],
                )
                directory = root / task["directory"]
                directory.mkdir(parents=True, exist_ok=True)
                task["command"] = [
                    sys.executable,
                    "-u",
                    str(ROOT / "main_ribounmix_multidataset.py"),
                    f"--config-path={configs}",
                    f"--config-name={path.stem}",
                    f"hydra.run.dir={directory / 'hydra'}",
                    "hydra.job.chdir=false",
                ]
                tasks.append(task)
                weight_rows.extend(dict(
                    panel_id=panel,
                    collection_id=panel,
                    N=len(selected),
                    arm=arm,
                    reference_policy=policy,
                    reference_orientation=orientation,
                    power=power,
                    dataset_id=name,
                    global_rank=float(ranks[name]),
                    quality_rank_score=float(score),
                    directional_score_base=float(base),
                    assigned_q=float(value),
                    raw_reference_weight=float(value),
                    pi=float(weight),
                    source_family=source_lookup[name],
                ) for name, score, base, value, weight in zip(
                    selected, score_values, base_orientation, raw, pi
                ))

        if len(tasks) != TASK_COUNT:
            raise AssertionError(f"Expected {TASK_COUNT} tasks, created {len(tasks)}.")
        weights = pd.DataFrame(weight_rows)
        weights.to_csv(root / "reference_weights.csv", index=False)
        concentration = pd.DataFrame([
            dict(
                collection_id=panel,
                N=int(group.N.iloc[0]),
                arm=arm,
                reference_policy=group.reference_policy.iloc[0],
                reference_orientation=group.reference_orientation.iloc[0],
                power=int(group.power.iloc[0]),
                N_ref=float(1.0 / np.square(group.pi).sum()),
                weighted_mean_rank=float(np.dot(group.pi, group.global_rank)),
                weighted_mean_quality_rank_score=float(
                    np.dot(group.pi, group.quality_rank_score)
                ),
                max_reference_mass=float(group.pi.max()),
            )
            for (panel, arm), group in weights.groupby(["panel_id", "arm"], sort=False)
        ])
        concentration.to_csv(root / "reference_concentration.csv", index=False)
        weights.groupby(["panel_id", "arm", "source_family"], as_index=False).pi.sum().to_csv(
            root / "source_family_reference_mass.csv", index=False
        )
        pd.DataFrame(tasks).drop(columns=["command", "datasets"]).to_csv(
            root / "task_matrix.csv", index=False
        )

        plan = dict(
            experiment_design=design,
            output_root=str(root),
            setup_request=request,
            training_seeds=[args.seed],
            panel_design=args.panel_design,
            panels=panels,
            collections=collections,
            source_folds=folds,
            tasks=tasks,
            tasks_sha256=object_sha256(tasks),
            primary_intervention="gamma-reference orientation from quality_rank_score within fixed panel membership",
            reference_weight_source="quality_rank_score_directional",
            reference_weight_formula=(
                "best: raw_d=(global_min_score/score_d)^p; "
                "worst: raw_d=(score_d/global_max_score)^p; pi_d=raw_d/sum(raw)"
            ),
            reference_policies={
                "equal": "uniform gamma reference",
                "best_first_score_p1": "best-oriented quality_rank_score reference, p=1",
                "worst_first_score_p1": "worst-oriented quality_rank_score reference, p=1",
                "best_first_score_p3": "best-oriented quality_rank_score reference, p=3",
                "worst_first_score_p3": "worst-oriented quality_rank_score reference, p=3",
                "best_first_score_p5": "best-oriented quality_rank_score reference, p=5",
                "worst_first_score_p5": "worst-oriented quality_rank_score reference, p=5",
            },
            cohort_limitation=split.get("cohort_limitation"),
            interpretation_limits=[
                "Worst-oriented changes reference mass inside a fixed panel; it does not select the worst datasets first.",
                "Equal score powers are not guaranteed to match effective concentration across panels or orientations.",
                "Cross-panel agreement measures reproducibility, not biological correctness.",
                "One seed does not estimate optimization variability.",
            ],
            frozen_file_sha256={
                str(path): sha256(path)
                for path in [*inputs.iterdir(), *configs.iterdir(), root / "reference_weights.csv"]
            },
        )
        write_json(manifest_path, plan)
        _design_report(
            root, args.panel_design, collections, membership, concentration, folds, args.seed
        )
        print(
            f"Ready: {len(tasks)} tasks; {len(first_fold['test_ids'])} common test transcripts. {root}",
            flush=True,
        )
        return plan


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "config/config_ribounmix_multidataset.yaml",
    )
    parser.add_argument(
        "--datasets", type=Path,
        default=ROOT / "config/dataset_config/weighted_hek_riboseq_codon_replicas.yaml",
    )
    parser.add_argument(
        "--ranking", type=Path,
        default=ROOT / "Datasets/data/HEK_riboseq_profile_quality_rank_components.tsv",
    )
    parser.add_argument("--panels", type=Path, default=DEFAULT_PANELS)
    parser.add_argument("--panel-design", choices=("balanced", "quality_strata"), default="balanced")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--task-index", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pair-rows-per-forward", type=positive_int)
    parser.add_argument("--max-padded-codon-tokens-per-forward", type=positive_int)
    parser.add_argument("--require-resume-checkpoint", action="store_true")
    args = parser.parse_args(argv)
    if args.output_root is None:
        args.output_root = (
            DEFAULT_BALANCED_OUTPUT
            if args.panel_design == "balanced"
            else DEFAULT_STRATA_OUTPUT
        )
    for key in ("config", "datasets", "ranking", "panels", "output_root"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.task_index is not None and not 0 <= args.task_index < TASK_COUNT:
        parser.error(f"--task-index must be in 0..{TASK_COUNT - 1}")
    plan = prepare(args)
    if len(plan["tasks"]) != TASK_COUNT:
        raise ValueError(f"Expected {TASK_COUNT} frozen tasks, found {len(plan['tasks'])}.")
    return 0 if args.prepare_only else run_task(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
