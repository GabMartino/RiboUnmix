"""Preparation shared by dataset-membership directionality experiments.

These experiments change which datasets are selected.  This is deliberately
different from the historical ``reverse_p*`` controls, which retained the same
dataset collection and only permuted gamma-reference weights within it.
"""
from __future__ import annotations

import copy
import fcntl
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

from run_cumulative_stability import make_config, object_sha256, sha256, write_json
from Utils.cumulative_fixed_cohort import build_fixed_cumulative_split
from Utils.real_panel_convergence import fit_panel_reliability_manifest, infer_source_identifier


CUMULATIVE_DESIGN = "cumulative_dataset_selection_direction_v1"
CUMULATIVE_SCORE_DESIGN = "cumulative_dataset_selection_direction_quality_score_v2_directional"
FOUR_PANEL_DESIGN = "four_quality_strata_v1"
CUMULATIVE_SIZES = (2, 5, 10, 20, 40, 80, 114)
REFERENCE_POLICIES = (("equal", 0), ("quality_p3", 3))
SCORE_REFERENCE_POLICIES = (
    ("equal", 0),
    ("score_p1", 1),
    ("score_p3", 3),
    ("score_p5", 5),
)


def inverse_quality_score_weights(scores, power: int, *, global_minimum=None):
    """Return raw positive weights proportional to ``quality_rank_score**(-power)``.

    ``quality_rank_score`` is a sum of component ranks, so lower values denote
    better profiles.  Scaling by the global minimum keeps the best raw weight
    at one without changing the normalized reference weights.
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("Quality scores must be a non-empty one-dimensional array.")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Quality scores must be finite and strictly positive.")
    if not isinstance(power, (int, np.integer)) or power < 0:
        raise ValueError("Quality-score power must be a non-negative integer.")
    if power == 0:
        return np.ones_like(values)
    minimum = float(values.min() if global_minimum is None else global_minimum)
    if not np.isfinite(minimum) or minimum <= 0 or minimum > values.min():
        raise ValueError("Global minimum must be finite, positive, and no larger than selected scores.")
    return np.power(minimum / values, power)


def directional_quality_score_weights(
        scores, power: int, direction: str, *, global_minimum, global_maximum):
    """Construct score-derived weights favoring the requested quality extreme."""
    values = np.asarray(scores, dtype=float)
    if direction == "best_first":
        return inverse_quality_score_weights(
            values, power, global_minimum=global_minimum)
    if direction != "worst_first":
        raise ValueError("Directional score weighting requires best_first or worst_first.")
    if values.ndim != 1 or not len(values):
        raise ValueError("Quality scores must be a non-empty one-dimensional array.")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Quality scores must be finite and strictly positive.")
    if not isinstance(power, (int, np.integer)) or power < 0:
        raise ValueError("Quality-score power must be a non-negative integer.")
    if power == 0:
        return np.ones_like(values)
    maximum = float(global_maximum)
    if not np.isfinite(maximum) or maximum < values.max():
        raise ValueError("Global maximum must be finite and no smaller than selected scores.")
    return np.power(values / maximum, power)


def _ranked_universe(mapping: dict[str, str], ranking: pd.DataFrame):
    required = {"dataset", "quality_rank"}
    if not required.issubset(ranking.columns):
        raise ValueError(f"Ranking table must contain {sorted(required)}.")
    ranks = ranking.set_index("dataset").quality_rank.astype(float)
    if ranks.index.has_duplicates or not np.isfinite(ranks).all() or (ranks <= 0).any():
        raise ValueError("Quality ranks must be unique, finite and positive.")
    missing = sorted(set(mapping) - set(ranks.index))
    if missing:
        raise ValueError(f"Configured datasets missing from ranking: {missing}")
    names = sorted(mapping, key=lambda name: (float(ranks[name]), name))
    if len(names) != 114:
        raise ValueError(f"Expected the configured 114-dataset universe, found {len(names)}.")
    return names, ranks


def cumulative_collections(names: list[str], sizes=CUMULATIVE_SIZES):
    """Nested best-first and worst-first collections, with one shared full set."""
    if list(sizes) != sorted(set(sizes)) or min(sizes) < 2 or max(sizes) != len(names):
        raise ValueError("Cumulative sizes must be unique, increasing and end at the full universe.")
    collections = []
    for n in sizes:
        if n == len(names):
            collections.append(dict(collection_id=f"full_N{n:03d}", N=n,
                                    selection_direction="shared_full", datasets=list(names)))
            continue
        collections.append(dict(collection_id=f"best_first_N{n:03d}", N=n,
                                selection_direction="best_first", datasets=list(names[:n])))
        # Select from the descending cumulative path, then restore ascending rank
        # order inside the model so ordering itself is not an extra intervention.
        collections.append(dict(collection_id=f"worst_first_N{n:03d}", N=n,
                                selection_direction="worst_first", datasets=list(names[-n:])))
    return collections


def reference_variants(collection: dict, design: str):
    """Return ``(direction, policy, power)`` fits for one collection."""
    direction = collection["selection_direction"]
    if design != CUMULATIVE_SCORE_DESIGN:
        return [(direction, policy, power) for policy, power in REFERENCE_POLICIES]
    if direction != "shared_full":
        return [(direction, policy, power) for policy, power in SCORE_REFERENCE_POLICIES]
    # Full membership is identical along both paths.  Uniform pi is therefore
    # one shared fit, whereas non-uniform score orientation remains distinct.
    variants = [("shared_full", "equal", 0)]
    variants.extend((orientation, policy, power)
                    for orientation in ("best_first", "worst_first")
                    for policy, power in SCORE_REFERENCE_POLICIES if power > 0)
    return variants


def four_quality_collections(names: list[str]):
    """Four disjoint contiguous global-rank strata of sizes 29, 29, 28 and 28."""
    labels = ("q1_best", "q2_upper_middle", "q3_lower_middle", "q4_worst")
    groups = [list(group) for group in np.array_split(np.asarray(names, dtype=object), 4)]
    collections = [dict(collection_id=label, panel_id=label, N=len(group),
                        quality_stratum=index + 1, selection_direction="quality_stratum",
                        datasets=group)
                   for index, (label, group) in enumerate(zip(labels, groups))]
    flat = [name for collection in collections for name in collection["datasets"]]
    if flat != names or [len(group) for group in groups] != [29, 29, 28, 28]:
        raise AssertionError("Quality strata must be an ordered, exhaustive 29/29/28/28 partition.")
    return collections


def _design_report(root: Path, design: str, collections: list[dict], weights: pd.DataFrame,
                   seed: int, task_count: int):
    rows = []
    for collection in collections:
        ranks = weights[(weights.collection_id == collection["collection_id"])
                        & (weights.reference_policy == "equal")].global_rank
        rows.append(dict(collection_id=collection["collection_id"], N=collection["N"],
                         selection_direction=collection["selection_direction"],
                         minimum_rank=int(ranks.min()), maximum_rank=int(ranks.max())))
    table = pd.DataFrame(rows).to_html(index=False, border=0)
    cumulative = design in {CUMULATIVE_DESIGN, CUMULATIVE_SCORE_DESIGN}
    score_weighted = design == CUMULATIVE_SCORE_DESIGN
    question = ("Do best-first and worst-first dataset collections learn different shared profiles?"
                if cumulative else
                "Does the learned shared profile change across contiguous dataset-quality strata?")
    primary = ("At each N, compare best-N with worst-N under equal gamma-reference weights; also track both paths against the single best-N=2 anchor."
               if cumulative else
               "Compare each equal-reference quality stratum with q1_best on identical held-out transcripts.")
    caveat = (("Best-N and worst-N memberships are disjoint through N=40, overlap at N=80, and become identical at N=114. At N=114 the equal-reference fit is shared, but best- and worst-oriented score weights remain different interventions and need not converge."
               if score_weighted else
               "Best-N and worst-N collections are disjoint through N=40, overlap at N=80, and become the same collection at N=114; convergence at the full set is therefore a design property.")
              if cumulative else
              "Quality strata deliberately split the ranked dataset universe and may split related source families; panel pairs are descriptive quality contrasts, not independent replicates.")
    launcher = (("run_cumulative_selection_direction_quality_score_univie.slurm"
                 if score_weighted else "run_cumulative_selection_direction_univie.slurm") if cumulative
                else "run_four_quality_strata_univie.slurm")
    analysis = (None if score_weighted else
                ("analyses/analyze_cumulative_selection_direction.py" if cumulative
                 else "analyses/analyze_four_quality_strata.py"))
    run_commands = f"sbatch {launcher}"
    if analysis is not None:
        run_commands += f"\npython {analysis}"
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>{question}</title>
<style>body{{max-width:1050px;margin:35px auto;padding:0 24px;font:16px/1.6 system-ui;color:#203448}}table{{border-collapse:collapse}}td,th{{padding:7px 11px;border-bottom:1px solid #ddd}}.note{{padding:14px;background:#f3f7fb;border-left:4px solid #267eab}}code,pre{{background:#f3f3f3}}pre{{padding:13px;overflow:auto}}</style></head><body>
<h1>{question}</h1>
<p class="note">{task_count} fresh fits, seed {seed}. Training, validation and test transcript identities are fixed across every collection using the complete 114-dataset observation cohort.</p>
<h2>Primary estimand</h2><p>{primary}</p>
<p>The primary <b>equal</b> condition isolates dataset membership. {'The directional score conditions use <code>quality_rank_score</code> directly: best-oriented π<sub>d</sub>∝<i>s</i><sub>d</sub><sup>−p</sup> and worst-oriented π<sub>d</sub>∝<i>s</i><sub>d</sub><sup>p</sup>, for <i>p</i>∈{{1,3,5}}. Lower aggregate rank scores denote better profiles.' if score_weighted else 'The secondary <b>quality_p3</b> condition applies the prespecified q³ gamma-reference weights inside each selected collection. No arm reverses weights.'} Dataset selection and reference weighting are distinct interventions.</p>
<p>{caveat}</p>
<h2>Collections</h2>{table}
<h2>Interpretation limits</h2><p>The top-two profile is an anchor rather than biological truth. Stability can reveal sensitivity to dataset quality but cannot establish biological accuracy without an external endpoint or known-signal corruption experiment.</p>
<h2>Run</h2><pre>{run_commands}</pre>
</body></html>'''
    (root / "design_report.html").write_text(html)


def prepare_quality_selection_experiment(*, root: Path, project_root: Path, config: Path,
                                         datasets: Path, ranking_path: Path, seed: int,
                                         design: str):
    """Create a frozen, self-contained experiment plan and return its manifest."""
    if design not in {CUMULATIVE_DESIGN, CUMULATIVE_SCORE_DESIGN, FOUR_PANEL_DESIGN}:
        raise ValueError(f"Unknown design {design!r}.")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        request = dict(design=design, seed=seed, config_sha256=sha256(config),
                       dataset_yaml_sha256=sha256(datasets), ranking_sha256=sha256(ranking_path),
                       split_protocol="fixed_complete_transcripts")
        manifest_path = root / "experiment_manifest.json"
        if manifest_path.exists():
            plan = json.loads(manifest_path.read_text())
            if (plan.get("output_root") != str(root) or plan.get("setup_request") != request
                    or plan.get("experiment_design") != design):
                raise ValueError("Output directory contains another experiment; choose a fresh root.")
            return plan

        base = yaml.safe_load(config.read_text())
        mapping = yaml.safe_load(datasets.read_text())["dataset_path"]
        mapping = {name: str((project_root / Path(path)).resolve()) for name, path in mapping.items()}
        base["paths"]["sequences_path"] = str((project_root / Path(base["paths"]["sequences_path"])).resolve())
        base["paths"]["encodings"] = {
            name: str((project_root / Path(path)).resolve())
            for name, path in base["paths"]["encodings"].items()}
        ranking = pd.read_csv(ranking_path, sep="\t")
        names, ranks = _ranked_universe(mapping, ranking)
        scores = None
        if design == CUMULATIVE_SCORE_DESIGN:
            if "quality_rank_score" not in ranking.columns:
                raise ValueError("Score-weighted design requires a quality_rank_score column.")
            scores = ranking.set_index("dataset").quality_rank_score.astype(float)
            selected_scores = scores.loc[names].to_numpy(float)
            if not np.isfinite(selected_scores).all() or (selected_scores <= 0).any():
                raise ValueError("Configured quality_rank_score values must be finite and positive.")
        collections = (cumulative_collections(names) if design in {
                           CUMULATIVE_DESIGN, CUMULATIVE_SCORE_DESIGN}
                       else four_quality_collections(names))

        inputs, configs = root / "inputs", root / "configs"
        inputs.mkdir(exist_ok=True)
        configs.mkdir(exist_ok=True)
        shutil.copy2(ranking_path, inputs / "ranking.tsv")
        membership = pd.DataFrame([
            dict(collection_id=c["collection_id"], selection_direction=c["selection_direction"],
                 N=c["N"], dataset_id=name, global_rank=float(ranks[name]),
                 **({"quality_rank_score": float(scores[name])} if scores is not None else {}),
                 source_family=infer_source_identifier(name))
            for c in collections for name in c["datasets"]])
        membership.to_csv(inputs / "collection_membership.csv", index=False)

        split_tasks = [dict(run_id=c["collection_id"], datasets=c["datasets"]) for c in collections]
        print("Building one complete transcript cohort shared by all collections...", flush=True)
        split = build_fixed_cumulative_split(
            experiment_name=design, tasks=split_tasks, dataset_mapping=mapping,
            sequences_path=base["paths"]["sequences_path"], subset_seed=seed,
            validation_fraction=.1, test_fraction=.1, reliability_bins=10,
            maximum_cds_codons=base["data"].get("max_cds_codons"))
        write_json(inputs / "split.json", split)

        folds = {}
        for collection in collections:
            collection_id = collection["collection_id"]
            folds[collection_id] = dict(
                source_panel=collection_id,
                train_ids=split["panel_train_eligible_ids"][collection_id],
                validation_ids=split["panel_validation_ids"][collection_id],
                test_ids=split["common_test_ids"])
        first_fold = folds[collections[0]["collection_id"]]
        if any(fold["train_ids"] != first_fold["train_ids"] or
               fold["validation_ids"] != first_fold["validation_ids"] or
               fold["test_ids"] != first_fold["test_ids"] for fold in folds.values()):
            raise AssertionError("The complete-cohort builder returned collection-dependent folds.")
        print("Fitting every dataset reliability reference once on the shared training IDs...", flush=True)
        complete_reference = fit_panel_reliability_manifest(
            experiment_name=design, panel_name="complete_114_reference", panel_datasets=names,
            dataset_mapping=mapping, panel_training_ids=first_fold["train_ids"],
            validation_ids=first_fold["validation_ids"], test_ids=first_fold["test_ids"],
            source_split_manifest=inputs / "split.json")
        write_json(inputs / "reliability_complete_114.json", complete_reference)
        for collection in collections:
            collection_id, selected = collection["collection_id"], collection["datasets"]
            reference = copy.deepcopy(complete_reference)
            reference["panel_name"] = collection_id
            reference["datasets"] = {name: complete_reference["datasets"][name] for name in selected}
            reference["derived_from_complete_114_reference"] = True
            reference["complete_reference_dataset_count"] = len(names)
            write_json(inputs / f"reliability_{collection_id}.json", reference)

        tasks, weight_rows = [], []
        maximum_rank = float(ranks.max())
        score_minimum = float(scores.loc[names].min()) if scores is not None else None
        score_maximum = float(scores.loc[names].max()) if scores is not None else None
        policies = SCORE_REFERENCE_POLICIES if design == CUMULATIVE_SCORE_DESIGN else REFERENCE_POLICIES
        for collection in collections:
            selected = collection["datasets"]
            q = (maximum_rank - ranks.loc[selected].to_numpy(float) + 1.0) / maximum_rank
            score_values = (scores.loc[selected].to_numpy(float) if scores is not None else None)
            score_quality = (score_minimum / score_values if score_values is not None else None)
            variants = reference_variants(collection, design)
            for task_direction, policy, power in variants:
                if design == CUMULATIVE_SCORE_DESIGN:
                    if task_direction == "shared_full":
                        raw = np.ones(len(selected), dtype=float)
                        base_quality = np.ones(len(selected), dtype=float)
                        weight_formula = "raw_d=1; pi_d=1/N"
                    else:
                        raw = directional_quality_score_weights(
                            score_values, power, task_direction,
                            global_minimum=score_minimum, global_maximum=score_maximum)
                        base_quality = directional_quality_score_weights(
                            score_values, 1, task_direction,
                            global_minimum=score_minimum, global_maximum=score_maximum)
                        weight_formula = (
                            "raw_d=(global_min_score/quality_rank_score_d)^p; pi_d=raw_d/sum(raw)"
                            if task_direction == "best_first" else
                            "raw_d=(quality_rank_score_d/global_max_score)^p; pi_d=raw_d/sum(raw)")
                else:
                    raw = np.ones(len(selected), dtype=float) if policy == "equal" else q ** power
                    base_quality = q
                    weight_formula = "q_d=(R-global_rank_d+1)/R; raw_d=q_d^p; pi_d=raw_d/sum(raw)"
                pi = raw / raw.sum()
                arm = (f'{task_direction}_{policy}'
                       if design in {CUMULATIVE_DESIGN, CUMULATIVE_SCORE_DESIGN}
                       else policy)
                task = dict(array_index=len(tasks), panel_id=collection["collection_id"], N=collection["N"],
                            arm=arm, reference_policy=policy, power=power, reverse=False,
                            selection_direction=task_direction, training_seed=seed,
                            datasets=selected, source_panel=collection["collection_id"],
                            experiment_design=design,
                            reference_weight_source=("quality_rank_score_directional"
                                                     if design == CUMULATIVE_SCORE_DESIGN
                                                     else "global_quality_rank"),
                            reference_weight_orientation=task_direction,
                            run_id=(f'{collection["collection_id"]}_{task_direction}_{policy}_seed{seed}'
                                    if design == CUMULATIVE_SCORE_DESIGN else
                                    f'{collection["collection_id"]}_{policy}_seed{seed}'),
                            directory=(f'runs/seed{seed}/{arm}/{collection["collection_id"]}'
                                       if design == CUMULATIVE_SCORE_DESIGN else
                                       f'runs/seed{seed}/{policy}/{collection["collection_id"]}'))
                if "quality_stratum" in collection:
                    task["quality_stratum"] = collection["quality_stratum"]
                cfg = make_config(base, mapping, task, root, dict(zip(selected, map(float, raw))))
                if design == CUMULATIVE_SCORE_DESIGN:
                    cfg["orchestrator"].update(
                        reference_weight_source="quality_rank_score_directional",
                        reference_weight_orientation=task_direction,
                        reference_weight_formula=weight_formula,
                        reference_weight_power=power,
                        global_minimum_quality_rank_score=score_minimum,
                        global_maximum_quality_rank_score=score_maximum,
                    )
                    cfg["orchestrator"].pop("task_contract_sha256")
                    cfg["orchestrator"]["task_contract_sha256"] = object_sha256(cfg)
                path = configs / f'{task["run_id"]}.yaml'
                path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                task.update(config_path=str(path), config_sha256=sha256(path),
                            task_contract_sha256=cfg["orchestrator"]["task_contract_sha256"])
                directory = root / task["directory"]
                directory.mkdir(parents=True, exist_ok=True)
                task["command"] = [sys.executable, "-u",
                    str(project_root / "main_ribounmix_multidataset.py"),
                    f"--config-path={configs}", f"--config-name={path.stem}",
                    f"hydra.run.dir={directory / 'hydra'}", "hydra.job.chdir=false"]
                tasks.append(task)
                weight_rows.extend(dict(
                    collection_id=collection["collection_id"], N=collection["N"], arm=arm,
                    reference_policy=policy, selection_direction=task_direction,
                    dataset_id=name, global_rank=float(ranks[name]), rank_q=float(rank_quality),
                    original_q=float(original),
                    **({"quality_rank_score": float(score),
                        "score_quality": float(quality),
                        "directional_score_base": float(original)}
                       if score_values is not None else {}),
                    assigned_q=float(value), raw_reference_weight=float(value), pi=float(weight),
                    source_family=infer_source_identifier(name))
                    for name, rank_quality, original, score, quality, value, weight in zip(
                        selected, q, base_quality,
                        score_values if score_values is not None else np.full(len(selected), np.nan),
                        score_quality if score_quality is not None else np.full(len(selected), np.nan),
                        raw, pi))

        weights = pd.DataFrame(weight_rows)
        weights.to_csv(root / "reference_weights.csv", index=False)
        concentration = pd.DataFrame([
            dict(collection_id=collection_id, N=int(group.N.iloc[0]), arm=arm,
                 reference_policy=group.reference_policy.iloc[0],
                 selection_direction=group.selection_direction.iloc[0],
                 N_ref=float(1.0 / np.square(group.pi).sum()),
                 weighted_mean_rank=float(np.dot(group.pi, group.global_rank)),
                 **({"weighted_mean_quality_rank_score": float(
                        np.dot(group.pi, group.quality_rank_score))}
                    if "quality_rank_score" in group else {}))
            for (collection_id, arm), group in weights.groupby(["collection_id", "arm"], sort=False)])
        concentration.to_csv(root / "reference_concentration.csv", index=False)
        weights.groupby(["collection_id", "arm", "source_family"], as_index=False).pi.sum().to_csv(
            root / "source_family_reference_mass.csv", index=False)
        pd.DataFrame(tasks).drop(columns=["command", "datasets"]).to_csv(root / "task_matrix.csv", index=False)

        plan = dict(experiment_design=design, output_root=str(root), setup_request=request,
                    training_seeds=[seed], sizes=(list(CUMULATIVE_SIZES)
                                                  if design in {CUMULATIVE_DESIGN,
                                                                CUMULATIVE_SCORE_DESIGN}
                                                  else None),
                    collections=collections, source_folds=folds, tasks=tasks,
                    tasks_sha256=object_sha256(tasks),
                    primary_intervention=(
                        "dataset membership and gamma-reference orientation selected by global quality rank"
                        if design == CUMULATIVE_SCORE_DESIGN else
                        "dataset membership selected by global quality rank"),
                    reference_weight_source=("quality_rank_score_directional"
                                             if design == CUMULATIVE_SCORE_DESIGN
                                             else "global_quality_rank"),
                    reference_weight_formula=(
                        "best: raw_d=(global_min_score/score_d)^p; worst: raw_d=(score_d/global_max_score)^p; pi_d=raw_d/sum(raw)"
                        if design == CUMULATIVE_SCORE_DESIGN else
                        "q_d=(R-global_rank_d+1)/R; raw_d=q_d^p; pi_d=raw_d/sum(raw)"),
                    reference_policies=(
                        {policy: ("uniform gamma reference" if power == 0 else
                                  f"directional quality_rank_score gamma reference, p={power}")
                         for policy, power in policies}),
                    frozen_file_sha256={str(path): sha256(path) for path in [
                        *inputs.iterdir(), *configs.iterdir(), root / "reference_weights.csv"]})
        write_json(manifest_path, plan)
        _design_report(root, design, collections, weights, seed, len(tasks))
        print(f"Ready: {len(tasks)} tasks and {len(split['common_test_ids'])} common test transcripts. {root}", flush=True)
        return plan
