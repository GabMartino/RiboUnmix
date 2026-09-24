#!/usr/bin/env python3
"""Figure 2 C/D: matched ten-component reference-weight effects, never retraining.

Inspect manifests before reading predictions. Missing/incompatible comparisons
produce an actionable report and exit 2, not a figure with invented effects.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import html
import itertools
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyses import create_real_data_equal_figure as uniform
from Utils.reliability_references import transcript_id_hash
from run_real_exp8_L_stability_quality_rank import inspect_ranking_components
from run_real_independent_panel_convergence_quality_rank import _flatten_config

N_VALUES = uniform.N_VALUES
PAIR_IDS = uniform.PAIR_IDS
PANEL_NAMES = uniform.PANEL_NAMES
PAIR_ORDER = tuple(f"P{a}–P{b}" for a, b in itertools.combinations(range(1, 5), 2))
DRAW_COUNT, BOOTSTRAP_SEED, TRAINING_SEED = 5000, 20260910, 42
ORANGE, LIGHT_ORANGE = "#D55E00", "#E9B995"
RANKING = ROOT / "Datasets/data/HEK_riboseq_profile_quality_rank_components.tsv"
MANUSCRIPT_RANKING_SHA256 = "5811cadf68c56740e83b232b2990299d527630326205db9cba8bc7024d3cf1f8"
read_json, sha256 = uniform.read_json, uniform.file_sha256
PAIRED_COLUMNS = ["experiment", "N", "comparison_id", "training_seed", "transcript_id",
                  "PCC_equal", "PCC_ranked", "status_equal", "status_ranked", "alignment_status",
                  "n_positions", "included", "exclusion_reason", "equal_run_a", "equal_run_b",
                  "ranked_run_a", "ranked_run_b", "source_artifacts"]
SUMMARY_COLUMNS = ["experiment", "N", "comparison_id", "training_seed", "n_transcripts",
                   "equal_statistic", "ranked_statistic", "estimate", "ci_lower", "ci_upper",
                   "statistic", "bootstrap_draws", "bootstrap_seed"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_table(path, rows, columns=None):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if frame.empty and columns is not None:
        frame = pd.DataFrame(columns=columns)
    frame.to_csv(path, index=False, float_format="%.17g")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-equal-root", type=Path, default=uniform.DEFAULT_PANEL_ROOT)
    parser.add_argument("--panel-ranked-root", type=Path,
                        help="Explicit ten-component matched four-panel root; otherwise search metadata only.")
    parser.add_argument("--stability-equal-root", type=Path, default=uniform.DEFAULT_STABILITY_ROOT)
    parser.add_argument("--stability-ranked-root", type=Path,
                        help="Representative matched A/B experiment, NEVER the cumulative top-quality chain.")
    parser.add_argument("--ranking-table", type=Path, default=RANKING)
    parser.add_argument("--scan-root", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--reference-figure", type=Path, default=ROOT / "figures/real_data_equal.pdf")
    parser.add_argument("--no-tex", action="store_true", help="Explicit built-in serif rendering, recorded in report.")
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    return args


def load_ranking(path, required_sha256=MANUSCRIPT_RANKING_SHA256):
    components = inspect_ranking_components(path, expected_count=10)
    if required_sha256 is not None and sha256(path) != required_sha256:
        raise ValueError("Ranking bytes do not match the frozen manuscript ten-component table. "
                         "A different/incomplete ranking universe cannot be substituted.")
    table = pd.read_csv(path, sep="\t")
    if table.dataset.isna().any() or table.dataset.duplicated().any():
        raise ValueError("Ranking dataset IDs must be complete and unique; no fuzzy aliases.")
    ranks = pd.to_numeric(table.quality_rank, errors="raise")
    if not np.isfinite(ranks).all() or (ranks < 1).any():
        raise ValueError("Missing/invalid global ranks; no imputation.")
    # Exactly the production conversion, evaluated on the COMPLETE input table.
    from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDatasetDataModule import load_dataset_quality_ranking
    rank_by_name, q = load_dataset_quality_ranking(str(path), dataset_column="dataset", rank_column="quality_rank")
    R = float(ranks.max())
    expected = (R - ranks.to_numpy(float) + 1) / R
    np.testing.assert_allclose([q[d] for d in table.dataset], expected, rtol=1e-13, atol=0)
    return rank_by_name, q, dict(path=str(path), sha256=sha256(path), R=R, rows=len(table),
        components=components["columns"], component_count=10, direction="1 = best",
        formula="q_d=(R-r_d+1)/R; pi_d=q_d/sum_selected(q)", power=1,
        global_ranks_recomputed=False, transcript_scope="not verified",
        historical_freeze_time="not verified; each accepted training manifest must record this checksum")


@dataclass(frozen=True)
class Run:
    experiment: str
    collection: str
    N: int
    directory: Path
    root: Path
    datasets: tuple[str, ...]
    sources: tuple[str, ...]
    test_ids: tuple[str, ...]
    run_id: str
    seed: int = TRAINING_SEED
    pair_id: str = ""
    side: str = ""


def read_runs(root, experiment):
    """Reuse frozen membership and designated-pair identities, never construct subsets."""
    if experiment == "C":
        manifest = read_json(root / "panel_manifest.json")
        panels = manifest["panels"]
        if tuple(len(panels.get(p, [])) for p in PANEL_NAMES) != uniform.PANEL_SIZES:
            raise ValueError("Expected panels 29,29,28,28 with canonical panel_01..04 IDs.")
        uniform.assert_disjoint(panels, "Dataset")
        sources = manifest["panel_source_families"]
        uniform.assert_disjoint(sources, "Source family")
        common = read_json(root / "common_split_manifest.json")
        ids = tuple(map(str, common["common_test_ids"]))
        if manifest.get("random_seed") != TRAINING_SEED:
            raise ValueError("Four-panel training seed is not the manuscript seed 42.")
        assignment = pd.read_csv(root/"panel_assignment.csv")
        if assignment.dataset_name.duplicated().any() or set(assignment.dataset_name) != set(itertools.chain.from_iterable(panels.values())):
            raise ValueError("Frozen panel assignment does not account for every dataset exactly once.")
        for p in PANEL_NAMES:
            assigned = assignment[assignment.panel == p]
            if set(assigned.dataset_name) != set(panels[p]) or set(assigned.source_identifier) != set(sources[p]):
                raise ValueError(f"{p}: frozen dataset/source-family assignment differs from the manifest.")
        runs = [Run(experiment, p, len(panels[p]), root/p, root, tuple(panels[p]), tuple(sources[p]), ids,
                    read_json(root/p/"run_manifest.json")["experiment_name"])
                for p in PANEL_NAMES]
    else:
        manifest = read_json(root / "experiment_manifest.json")
        if "cumulative" in str(manifest.get("experiment_name", "")) or manifest.get("experiment_design") == "cumulative_top_quality":
            raise ValueError("Cumulative top-quality chain is NOT the representative A/B experiment.")
        if manifest.get("sampling_mode") != "quality_matched":
            raise ValueError("Representative subset sampling_mode=quality_matched is not established.")
        if manifest.get("training_seeds") != [TRAINING_SEED]:
            raise ValueError("Do not mix training seeds; expected exactly [42].")
        common = read_json(root / "common_test_manifest.json")
        ids = tuple(map(str, common["common_test_ids"]))
        tasks = [t for t in manifest["tasks"] if t.get("kind") == "designated_disjoint_pair" and t["N"] in N_VALUES]
        keys = {(t["N"], t["pair_id"], t["side"]) for t in tasks}
        expected = set(itertools.product(N_VALUES, PAIR_IDS, ("A", "B")))
        if keys != expected or len(tasks) != len(expected):
            raise ValueError("All 30 unique designated subset tasks are required at N=2,5,10,20,40.")
        runs = []
        for task in sorted(tasks, key=lambda t: (t["N"], t["pair_id"], t["side"])):
            directory = uniform._task_directory(root, task)
            run = Run(experiment, f"N{task['N']:03}_{task['pair_id']}_{task['side']}", int(task["N"]),
                directory, root, tuple(task["datasets"]), tuple(task["source_families"]), ids,
                task["run_id"], int(task["training_seed"]), task["pair_id"], task["side"])
            if len(run.datasets) != run.N or run.seed != TRAINING_SEED:
                raise ValueError(f"{run.collection}: N/seed mismatch.")
            families = manifest["source_family_mapping"]
            if set(run.datasets) != {d for s in run.sources for d in families[s]}:
                raise ValueError(f"{run.collection}: dataset/source-family mapping mismatch or split family.")
            runs.append(run)
        for N, pair in itertools.product(N_VALUES, PAIR_IDS):
            a, b = [r for r in runs if r.N == N and r.pair_id == pair]
            uniform.assert_disjoint({"A": a.datasets, "B": b.datasets}, "Dataset")
            uniform.assert_disjoint({"A": a.sources, "B": b.sources}, "Source family")
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Common test IDs are empty or duplicated.")
    for run in runs:
        if len(run.datasets) != len(set(run.datasets)):
            raise ValueError(f"{run.collection}: repeated dataset IDs.")
    return runs, manifest


def ranking_hash(manifest):
    strategy = manifest.get("gamma_reference_strategy", {})
    return strategy.get("ranking_table_sha256", manifest.get("ranking_sha256"))


def discover_manifests(scan_root):
    """Metadata inventory only; do not read old performance or prediction tables."""
    skip = {"checkpoints", "logs", "predictions", "code_snapshot", "_archive", ".cache", ".scratch"}
    found = []
    for directory, folders, files in os.walk(scan_root):
        folders[:] = [name for name in folders if name not in skip and not name.startswith(
            ("analysis", "riboai_synthetic", "synthetic_", "riboai_benchmarking", "superseded_"))]
        for name in ("panel_manifest.json", "experiment_manifest.json"):
            if name in files:
                path = Path(directory)/name
                data = read_json(path)
                experiment = data.get("experiment_name", "")
                if "panel" in experiment or "exp8" in experiment or "panels" in data:
                    found.append((path, data))
    return found


def candidate_inventory(found, ranking_digest):
    known_rankings = {}
    for path in (RANKING, ROOT/"Datasets/data/HEK_riboseq_profile_quality_rank.tsv"):
        if path.is_file():
            known_rankings[sha256(path)] = (str(path), inspect_ranking_components(path)["count"])
    rows = []
    for path, data in found:
        cumulative = "cumulative" in str(data.get("experiment_name", "")) or data.get("experiment_design") == "cumulative_top_quality"
        digest = ranking_hash(data)
        reasons = []
        if cumulative:
            reasons.append("cumulative_top_quality_not_representative_pairs")
        if digest != ranking_digest:
            reasons.append("not_the_frozen_ten_component_ranking" if digest else "uniform_or_unverified_ranking")
        rows.append(dict(manifest=str(path), manifest_sha256=sha256(path), experiment=data.get("experiment_name"),
            ranking_sha256=digest, candidate_for_ten_component_match=not reasons,
            identified_ranking_path=known_rankings.get(digest, (None, None))[0],
            actual_component_count=known_rankings.get(digest, (None, None))[1],
            reason="; ".join(reasons) or "metadata candidate; full matching still required",
            recorded_git=json.dumps(data.get("git")),
            checkpoint_files=len(list(path.parent.glob("**/*.ckpt"))) if "code_snapshot" not in path.parts else 0))
    return rows


def choose_ranked_root(experiment, equal, explicit, found, digest):
    if explicit:
        candidates = [explicit]
    else:
        filename = "panel_manifest.json" if experiment == "C" else "experiment_manifest.json"
        candidates = [p.parent for p, m in found if p.name == filename and ranking_hash(m) == digest
                      and "cumulative" not in str(m.get("experiment_name", ""))]
    accepted, reasons = [], []
    for path in candidates:
        try:
            runs, manifest = read_runs(path, experiment)
            if ranking_hash(manifest) != digest:
                raise ValueError("Training ranking checksum is not the frozen ten-component checksum.")
            if [(r.collection, r.datasets, r.sources, r.seed, sorted(r.test_ids)) for r in runs] != [
                (r.collection, r.datasets, r.sources, r.seed, sorted(r.test_ids)) for r in equal]:
                raise ValueError("Membership/order, source assignment, seed or held-out cohort differs.")
            accepted.append(runs)
        except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
            reasons.append(f"{path}: {exc}")
    if len(accepted) > 1:
        raise ValueError("Multiple metadata-matched ranked runs. Select an explicit root; no newest/best-run selection.")
    return (accepted[0] if accepted else None), reasons


def declaration(run):
    name = "run_manifest.json" if run.experiment == "C" else "subset_manifest.json"
    path = run.directory/name
    data = read_json(path)
    selected = tuple(data.get("selected_datasets", data.get("datasets", [])))
    if selected != run.datasets or data.get("checkpoint_selection") != "best_val_loss":
        raise ValueError(f"{path}: membership/checkpoint-selection mismatch.")
    if tuple(data.get("selected_source_families", data.get("source_families", []))) != run.sources:
        raise ValueError(f"{path}: task source families differ from the experiment manifest.")
    return data


def split_ids(run):
    path = run.directory/"split_manifest.json"
    data = read_json(path)
    result = {"train": data["train_ids"], "validation": data["validation_ids"],
              "test": data.get("test_ids", data.get("common_test_ids"))}
    for fold, ids in result.items():
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"{path}: empty/duplicate {fold} IDs.")
    uniform.assert_disjoint(result, "Transcript split")
    if set(result["test"]) != set(run.test_ids):
        raise ValueError(f"{path}: task and common held-out IDs differ.")
    return result


def reliability_values(run, split):
    data = read_json(run.directory/"reliability_reference_manifest.json")
    if (data.get("reference_split") != "training_only" or data.get("heldout_rows_used_for_fitting") != 0
        or data.get("panel_training_transcript_id_hash") != transcript_id_hash(split["train"])
        or set(data["datasets"]) != set(run.datasets)):
        raise ValueError(f"{run.directory}: train-only w_dt provenance is invalid.")
    # Only location/creation labels are excluded. All numerical references and
    # training-ID fingerprints remain in the exact equality check.
    ignored = {"created_at_utc", "source_split_manifest", "experiment_name", "panel_name", "source_dataset_path"}
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k not in ignored}
        return value
    return clean(data)


def config_for(run):
    cfg = yaml.safe_load((run.directory/"resolved_config.yaml").read_text())
    if tuple(cfg["experiment"]["dataset"]) != run.datasets or cfg["experiment"]["seed"] != run.seed:
        raise ValueError(f"{run.directory}: resolved dataset/seed mismatch.")
    if cfg["model"]["mass_conservation"] is not False or cfg["model"]["gamma_centering"]["mode"] != "fixed_reference":
        raise ValueError(f"{run.directory}: not the production mass-free fixed-reference model.")
    if cfg["loss"]["sample_reduction"] != "transcript_balanced" or cfg["data"]["train_sampling_strategy"] != "transcript_grouped_multidataset_pairs":
        raise ValueError(f"{run.directory}: grouped sampling/reduction mismatch.")
    if cfg["prediction"]["checkpoint_variants"] != ["best_val_loss"]:
        raise ValueError(f"{run.directory}: checkpoint selection must be best_val_loss only.")
    return cfg


def compare_configs(equal, ranked):
    a, b = _flatten_config(equal), _flatten_config(ranked)
    allowed = {"name", "data.reliability_reference_manifest", "data.dataset_quality_ranking.path",
               "model.gamma_centering.reference.weighting", "model.gamma_centering.reference.quality_rank_power",
               "split.external_manifest", "experiment.from_checkpoint", "experiment.resume_training_state",
               "experiment.resume_checkpoint_path", "experiment.train", "experiment.predict",
               "prediction.sequence_only_shared_profile"}
    differences = []
    for key in sorted(a.keys() | b.keys()):
        if a.get(key) == b.get(key):
            continue
        permitted = key in allowed or key.startswith(("orchestrator.", "hydra.", "paths.", "dataset_config.dataset_path."))
        differences.append(dict(key=key, equal=json.dumps(a.get(key)), ranked=json.dumps(b.get(key)),
            permitted=permitted, note="paths/data artifacts checked separately" if key.startswith(("paths.", "dataset_config.")) else ""))
    forbidden = [r["key"] for r in differences if not r["permitted"]]
    if forbidden:
        raise ValueError("Unmatched training configuration: " + ", ".join(forbidden))
    return differences


def execution_evidence(run):
    """Use existing training-launcher evidence, not today's Git state or guesses."""
    for parent in (run.root, run.root.parent):
        path = parent/"matched_experiment.json"
        if path.is_file():
            payload = read_json(path)
            identity = payload.get("execution_identity", {})
            if not identity.get("sources") or not identity.get("packages"):
                raise ValueError(f"Incomplete source/environment evidence: {path}")
            # These immutable files were verified by the matched launcher on
            # each training attempt. Check the relevant task evidence again.
            frozen = payload.get("frozen_files", {})
            for name in ("launch_command.sh", "run_manifest.json", "split_manifest.json", "reliability_reference_manifest.json"):
                local = run.directory/name
                if name == "run_manifest.json" and run.experiment == "D":
                    local = run.directory/"subset_manifest.json"
                key = str(local.relative_to(parent))
                if key not in frozen or sha256(local) != frozen[key]:
                    raise ValueError(f"Missing/changed training-linked frozen input: {local}")
            from resume_real_experiment_from_checkpoints import _read_launch_command, _command_override
            launch = _read_launch_command(run.directory/"launch_command.sh")
            if _command_override(launch, "experiment.from_checkpoint") != "false" or _command_override(launch, "experiment.train") != "true":
                raise ValueError("Original frozen launch does not establish fresh training; changing pi in an existing model is not accepted.")
            return identity, str(path)
    raise ValueError(f"{run.directory}: matching historical training code/environment is NOT VERIFIED; "
                     "no training-linked source snapshot was found. Matching YAML or a current Git hash is insufficient.")


def local_input(value):
    path = Path(value)
    if path.is_file():
        return path.resolve()
    if not path.is_absolute() and (ROOT/path).is_file():
        return (ROOT/path).resolve()
    # Recorded dataset paths may have a cluster prefix. Resolve only the same
    # repository-relative suffix, then compare file CONTENT, never rewrite inputs.
    if "Datasets" in path.parts:
        candidate = ROOT/Path(*path.parts[path.parts.index("Datasets"):])
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot verify configured data artifact: {value}")


@lru_cache(maxsize=None)
def input_digest(path):
    return sha256(path)


def matched_data(equal, ranked, datasets):
    def files(cfg):
        selected = {d: cfg["dataset_config"]["dataset_path"][d] for d in datasets}
        extra = _flatten_config({k: v for k, v in cfg["paths"].items() if k not in ("results", "checkpoints", "logs")})
        return {**selected, **extra}
    left, right = files(equal), files(ranked)
    if left.keys() != right.keys():
        raise ValueError("Data/sequence/encoding artifact definitions differ.")
    rows = []
    for key in left:
        a, b = local_input(left[key]), local_input(right[key])
        if input_digest(a) != input_digest(b):
            raise ValueError(f"Equal/ranked input bytes differ for {key}.")
        rows.append(dict(artifact=key, equal_path=str(a), ranked_path=str(b), sha256=input_digest(a)))
    return rows


def require_weights(run, policy, cfg, q):
    reference = declaration(run)["fixed_gamma_reference"]
    expected_kind = "equal" if policy == "equal" else "quality_rank"
    model_reference = cfg["model"]["gamma_centering"]["reference"]
    if reference.get("weighting") != expected_kind or model_reference["weighting"] != expected_kind:
        raise ValueError(f"{run.directory}: declared/training reference policy mismatch.")
    if policy == "ranked" and model_reference["quality_rank_power"] != 1:
        raise ValueError("Ranked reference exponent is not the prespecified power one.")
    if model_reference.get("dataset_names") is not None and set(model_reference["dataset_names"]) != set(run.datasets):
        raise ValueError("Configured gamma reference does not use the full selected collection.")
    raw = np.array([q[d] if policy == "ranked" else 1. for d in run.datasets])
    if not np.isfinite(raw).all() or np.any(raw <= 0):
        raise ValueError("All reference scores must be positive and finite.")
    pi = raw/raw.sum()
    if set(reference["pi"]) != set(run.datasets):
        raise ValueError("Reference universe is not the complete selected collection.")
    np.testing.assert_allclose([reference["pi"][d] for d in run.datasets], pi, rtol=1e-10, atol=1e-12)
    return raw, pi


def prediction_evidence(run, policy, q):
    manifest = uniform.require_one(run.directory/"predictions", "prediction_checkpoint_manifest.json")
    runtime = read_json(manifest)
    if set(runtime) != {"best_val_loss"}:
        raise ValueError(f"{manifest}: only best_val_loss exports are accepted.")
    runtime = runtime["best_val_loss"]
    if (runtime["split_name"] != "test" or runtime["transcript_count"] != len(run.test_ids)
        or runtime["transcript_id_hash"] != transcript_id_hash(run.test_ids)):
        raise ValueError(f"{manifest}: frozen held-out cohort mismatch.")
    raw_path = uniform.require_one(run.directory/"predictions", Path(runtime["output_path"]).name)
    gamma_path = manifest.parent/"gamma_reference_manifest.json"
    gamma = read_json(gamma_path)
    names = gamma["reference_dataset_names"]
    expected = np.array([q[d] if policy == "ranked" else 1. for d in names])
    if gamma["centering_mode"] != "fixed_reference" or gamma["weighting"] != ("equal" if policy == "equal" else "quality_rank"):
        raise ValueError("Export runtime reference policy does not match training.")
    if set(names) != set(run.datasets) or len(names) != len(run.datasets):
        raise ValueError("Export uses a truncated/different reference universe.")
    np.testing.assert_allclose(gamma["reference_pi"], expected/expected.sum(), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(gamma["reference_raw_weights"], expected, rtol=1e-6, atol=1e-9)
    if policy == "ranked" and gamma["quality_rank_power"] != 1:
        raise ValueError("Runtime reference exponent is not one.")
    checkpoint = uniform.require_one(run.directory/"checkpoints", Path(runtime["checkpoint_path"]).name)
    if checkpoint.name == "last.ckpt":
        raise ValueError("last.ckpt is not the selected best_val_loss checkpoint.")
    import torch
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    selectors = [value for value in saved.get("callbacks", {}).values()
                 if isinstance(value, dict) and value.get("monitor") == "val_loss" and "best_model_path" in value]
    if len(selectors) != 1 or Path(selectors[0]["best_model_path"]).name != checkpoint.name:
        raise ValueError("Checkpoint callback does not identify this file as best_val_loss.")
    state = saved["state_dict"]
    extra = state["model._extra_state"]
    selected = extra["gamma_reference_dataset_names"]
    if (set(selected) != set(run.datasets) or len(selected) != len(run.datasets)
        or extra["gamma_centering_weighting"] != gamma["weighting"]
        or extra["gamma_centering_mode"] != "fixed_reference"):
        raise ValueError("Saved CHECKPOINT was trained with a different gamma reference.")
    expected_checkpoint = np.array([q[d] if policy == "ranked" else 1. for d in selected])
    # Production checkpoints store RAW q**power (ones for equal), not pi.
    # The centering operator divides by their sum during the forward pass.
    checkpoint_raw = state["model.gamma_reference_weights"].numpy().astype(float)
    np.testing.assert_allclose(checkpoint_raw, expected_checkpoint, rtol=2e-6, atol=1e-9)
    np.testing.assert_allclose(checkpoint_raw/checkpoint_raw.sum(),
        expected_checkpoint/expected_checkpoint.sum(), rtol=2e-6, atol=1e-9)
    if policy == "ranked" and extra["gamma_centering_quality_rank_power"] != 1:
        raise ValueError("Checkpoint rank exponent differs.")
    shapes = {k: list(v.shape) for k, v in state.items() if isinstance(v, torch.Tensor)}
    dataset_ids = dict(zip(selected, state["model.gamma_reference_dataset_ids"].tolist(), strict=True))
    checkpoint_training = {"epoch": saved.get("epoch"), "global_step": saved.get("global_step"),
        "hyper_parameters": saved.get("hyper_parameters", {}), "alpha_mode": extra.get("alpha_mode"),
        "raw_log_gamma_bound": extra.get("raw_log_gamma_bound")}
    del saved, state
    return dict(run_id=run.run_id, raw_path=str(raw_path), raw_sha256=sha256(raw_path),
        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        prediction_manifest=str(manifest), prediction_manifest_sha256=sha256(manifest),
        gamma_manifest=str(gamma_path), gamma_manifest_sha256=sha256(gamma_path),
        dataset_ids=dataset_ids, parameter_shapes=shapes, checkpoint_training=checkpoint_training,
        checkpoint_reference_buffer="raw q**power; normalized by the production centering operator")


def paired_run_audit(equal, ranked, q):
    a, b = config_for(equal), config_for(ranked)
    differences = compare_configs(a, b)
    sa, sb = split_ids(equal), split_ids(ranked)
    if sa != sb:
        raise ValueError("Ordered train/validation/test lists differ between policies.")
    if reliability_values(equal, sa) != reliability_values(ranked, sb):
        raise ValueError("Numerical train-fitted w_dt references differ between policies.")
    ea, pa = execution_evidence(equal)
    eb, pb = execution_evidence(ranked)
    if ea != eb:
        raise ValueError("Training source/environment fingerprints differ.")
    artifacts = matched_data(a, b, equal.datasets)
    weights = []
    for policy, run, cfg in (("equal", equal, a), ("ranked", ranked, b)):
        raw, pi = require_weights(run, policy, cfg, q)
        weights.extend(dict(experiment=run.experiment, collection=run.collection, N=run.N, policy=policy,
            dataset_id=d, raw_reference_weight=float(w), pi=float(p)) for d, w, p in zip(run.datasets, raw, pi))
    export_a, export_b = prediction_evidence(equal, "equal", q), prediction_evidence(ranked, "ranked", q)
    if export_a["parameter_shapes"] != export_b["parameter_shapes"] or export_a["dataset_ids"] != export_b["dataset_ids"]:
        raise ValueError("Checkpoint architecture/dataset encoding differs.")
    for policy, export, cfg in (("equal", export_a, a), ("ranked", export_b, b)):
        training = export["checkpoint_training"]
        if training["alpha_mode"] != cfg["model"]["alpha_mode"]:
            raise ValueError(f"{policy}: checkpoint alpha treatment differs from the resolved configuration.")
        hyper = training["hyper_parameters"]
        for key in ("experiment_mode", "sample_reduction", "nb_mean_gradient_beta"):
            if hyper.get(key) != cfg["loss"].get(key):
                raise ValueError(f"{policy}: checkpoint loss {key} differs from resolved configuration.")
    sequence = next(row for row in artifacts if row["artifact"] == "sequences_path")
    export_a["sequence_path"] = sequence["equal_path"]
    export_b["sequence_path"] = sequence["ranked_path"]
    return dict(equal=export_a, ranked=export_b, weights=weights, differences=differences,
                data_artifacts=artifacts, source_evidence=[pa, pb])


@lru_cache(maxsize=None)
def sequence_lengths(path):
    lengths = {}
    with pq.ParquetFile(path) as reader:
        column = "codons" if "codons" in reader.schema_arrow.names else "ref"
        for batch in reader.iter_batches(batch_size=128, columns=["transcript_id", column]):
            for row in batch.to_pylist():
                tid = str(row["transcript_id"])
                if tid in lengths:
                    raise ValueError(f"Duplicate transcript ID in frozen sequence artifact: {tid}")
                lengths[tid] = len(row[column])
    return lengths


def read_profiles(evidence, expected_ids):
    """Stream original predictions, enforcing full-CDS masks and duplicate equality."""
    path = Path(evidence["raw_path"])
    expected = set(expected_ids)
    lengths = sequence_lengths(evidence["sequence_path"])
    result, invalid = {}, {}
    required = ["transcript_id", "length", "mask", "L_bio"]
    with pq.ParquetFile(path) as reader:
        if not set(required) <= set(reader.schema_arrow.names):
            raise ValueError(f"Raw shared-profile columns missing: {path}")
        for batch in reader.iter_batches(batch_size=16, columns=required, use_threads=False):
            for row in batch.to_pylist():
                tid = str(row["transcript_id"])
                if tid not in expected:
                    raise ValueError(f"Unexpected held-out transcript {tid} in {path}.")
                x, mask = np.asarray(row["L_bio"], float), np.asarray(row["mask"], bool)
                length = int(row["length"])
                if length != lengths.get(tid):
                    invalid[tid] = "length_differs_from_frozen_complete_CDS"
                    continue
                if x.ndim != 1 or x.shape != mask.shape or length < 2 or int(mask.sum()) != length or not np.all(mask[:length]) or np.any(mask[length:]):
                    invalid[tid] = "not_complete_CDS_alignment"
                    continue
                values = x[mask]  # remove padding only; never observations/zeros
                if tid in result:
                    if not np.array_equal(result[tid].values, values, equal_nan=True):
                        invalid[tid] = "shared_output_differs_across_dataset_rows"
                    continue
                result[tid] = uniform.Profile(values, length)
                if not np.isfinite(values).all():
                    invalid[tid] = "nonfinite_profile"
                elif np.any(values <= 0):
                    invalid[tid] = "nonpositive_profile"
                elif abs(float(values.mean()) - 1) > uniform.MEAN_ONE_TOLERANCE:
                    invalid[tid] = "not_mean_one_no_rescaling"
    return result, invalid


def comparison_records(experiment, equal, ranked, evidence):
    eq = {r.collection: r for r in equal}
    rk = {r.collection: r for r in ranked}
    if experiment == "C":
        comparisons = [(label, 0, a, b) for label, (a, b) in zip(PAIR_ORDER, itertools.combinations(PANEL_NAMES, 2))]
    else:
        comparisons = [(p, N, f"N{N:03}_{p}_A", f"N{N:03}_{p}_B") for N, p in itertools.product(N_VALUES, PAIR_IDS)]
    rows, cache = [], {}
    for label, N, a, b in comparisons:
        for policy in ("equal", "ranked"):
            for collection in (a, b):
                key = policy, collection
                if key not in cache:
                    cache[key] = read_profiles(evidence[collection][policy], eq[collection].test_ids)
        sources = [evidence[c][p]["raw_path"] for p in ("equal", "ranked") for c in (a, b)]
        for tid in sorted(eq[a].test_ids):
            metrics, shapes = {}, []
            for policy in ("equal", "ranked"):
                left, li = cache[(policy, a)]
                right, ri = cache[(policy, b)]
                reason = li.get(tid) or ri.get(tid)
                metrics[policy] = (dict(PCC=np.nan, status=reason, transcript_length=np.nan) if reason else
                    uniform.profile_agreement(left.get(tid), right.get(tid)))
                shapes.extend([p[tid].length for p in (left, right) if tid in p])
            aligned = len(shapes) == 4 and len(set(shapes)) == 1
            valid = aligned and all(metrics[p]["status"] == "valid" for p in metrics)
            reasons = [f"{p}:{metrics[p]['status']}" for p in metrics if metrics[p]["status"] != "valid"]
            if not aligned:
                reasons.append("cross_policy_positions_not_aligned")
            rows.append(dict(experiment=experiment, N=N if experiment == "D" else np.nan,
                comparison_id=label, training_seed=TRAINING_SEED, transcript_id=tid,
                PCC_equal=metrics["equal"]["PCC"], PCC_ranked=metrics["ranked"]["PCC"],
                status_equal=metrics["equal"]["status"], status_ranked=metrics["ranked"]["status"],
                alignment_status="identical_full_CDS" if aligned else "misaligned_or_missing", n_positions=shapes[0] if aligned else 0,
                included=valid, exclusion_reason=";".join(reasons), equal_run_a=eq[a].run_id,
                equal_run_b=eq[b].run_id, ranked_run_a=rk[a].run_id, ranked_run_b=rk[b].run_id,
                source_artifacts=json.dumps(sources)))
        if experiment == "D":
            cache.clear()  # four original arrays at a time, independent of the N-series size
    frame = pd.DataFrame(rows, columns=PAIRED_COLUMNS)
    if experiment == "D":
        common = frame.groupby("transcript_id").included.all()
        removed = frame.included & ~frame.transcript_id.map(common)
        frame.loc[removed, "exclusion_reason"] = "excluded_by_common_cohort_across_all_N_and_pairs"
        frame.loc[removed, "included"] = False
    return frame


def bootstrap_effects(equal, ranked, statistic, draws=DRAW_COUNT, seed=BOOTSTRAP_SEED):
    """Resample transcript indices once per draw across EVERY comparison and policy."""
    equal, ranked = np.asarray(equal, float), np.asarray(ranked, float)
    if equal.ndim != 2 or ranked.shape != equal.shape or not len(equal):
        raise ValueError("Expected nonempty aligned transcript-by-comparison matrices.")
    valid = np.isfinite(equal) & np.isfinite(ranked)
    a, b = np.where(valid, equal, np.nan), np.where(valid, ranked, np.nan)
    if not valid.any(axis=0).all():
        raise ValueError("A comparison has no matched finite transcripts.")
    if statistic == "D" and (a.shape[1] != 15 or not valid.all()):
        raise ValueError("D requires a common complete cohort across all 15 designated comparisons.")
    def reduce(x):
        return np.nanmedian(x, axis=0) if statistic == "C" else x.mean(axis=0).reshape(5, 3).mean(axis=1)
    if statistic not in ("C", "D"):
        raise ValueError("Unknown statistic.")
    ae, be = reduce(a), reduce(b)
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, len(ae)))
    index_digest = hashlib.sha256()
    for i in range(draws):
        indices = rng.integers(0, len(a), len(a))
        index_digest.update(indices.astype("<i8").tobytes())
        samples[i] = reduce(b[indices]) - reduce(a[indices])
    if not np.isfinite(samples).all():
        raise ValueError("A bootstrap draw has no finite matched cases; no draws were discarded or replaced.")
    lower, upper = np.quantile(samples, [.025, .975], axis=0, method="linear")
    return ae, be, lower, upper, index_digest.hexdigest()


def summarize_records(frame, experiment):
    keys = ["experiment", "N", "comparison_id", "training_seed", "transcript_id"]
    if frame.duplicated(keys).any() or set(frame.training_seed) != {TRAINING_SEED}:
        raise ValueError("Duplicated paired records or mixed training seeds.")
    columns = ["comparison_id"] if experiment == "C" else ["N", "comparison_id"]
    order = list(PAIR_ORDER) if experiment == "C" else list(itertools.product(N_VALUES, PAIR_IDS))
    included = frame[frame.included].copy()
    matrices = [included.pivot(index="transcript_id", columns=columns, values=f"PCC_{policy}").reindex(columns=order).sort_index()
                for policy in ("equal", "ranked")]
    if not matrices[0].index.equals(matrices[1].index):
        raise ValueError("Unequal policy cohorts.")
    ae, be, low, high, digest = bootstrap_effects(*(m.to_numpy() for m in matrices), statistic=experiment)
    rows = []
    for index, label in enumerate(PAIR_ORDER if experiment == "C" else N_VALUES):
        count = int(matrices[0].iloc[:, index].notna().sum()) if experiment == "C" else len(matrices[0])
        rows.append(dict(experiment=experiment, N=np.nan if experiment == "C" else label,
            comparison_id=label if experiment == "C" else "mean_over_three_designated_pairs", training_seed=TRAINING_SEED,
            n_transcripts=count, equal_statistic=ae[index], ranked_statistic=be[index],
            estimate=be[index]-ae[index], ci_lower=low[index], ci_upper=high[index],
            statistic="difference_of_transcript_medians" if experiment == "C" else "difference_of_mean_of_three_pair_means",
            bootstrap_draws=DRAW_COUNT, bootstrap_seed=BOOTSTRAP_SEED))
    pair_rows = []
    if experiment == "D":
        for (N, pair), group in included.groupby(["N", "comparison_id"], sort=True):
            a, b = group.PCC_equal.mean(), group.PCC_ranked.mean()
            pair_rows.append(dict(N=N, comparison_id=pair, training_seed=TRAINING_SEED,
                n_transcripts=len(group), equal_mean_PCC=a, ranked_mean_PCC=b, estimate=b-a))
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS), pd.DataFrame(pair_rows), dict(
        index_stream_sha256=digest, transcript_order_sha256=transcript_id_hash(matrices[0].index),
        resampling_unit="transcript with both policies and all associated comparisons",
        interval="2.5th and 97.5th percentiles; NumPy linear quantiles", draws=DRAW_COUNT, seed=BOOTSTRAP_SEED)


def figure_geometry(reference):
    info = dict(reference_pdf=str(reference), reference_available=reference.is_file(),
                style_source=str(Path(uniform.__file__)), style_source_sha256=sha256(Path(uniform.__file__)),
                width_inches=7.15, height_inches=2.65, panel_pair_order=list(PAIR_ORDER),
                N_ticks=list(N_VALUES), N_scale="log", N_limits=[1.65, 48.0])
    if reference.is_file():
        result = subprocess.run(["pdfinfo", str(reference)], capture_output=True, text=True, check=True)
        match = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", result.stdout)
        if not match:
            raise ValueError("Cannot read reference PDF page dimensions.")
        info.update(reference_sha256=sha256(reference), width_inches=float(match[1])/72,
                    height_inches=float(match[2])/72)
    return info


def padded_limits(values):
    values = np.r_[np.asarray(values, float).ravel(), 0.]
    lo, hi = float(values.min()), float(values.max())
    padding = max((hi-lo)*.12, .005)
    return lo-padding, hi+padding


def build_figure(c, d, pairs, geometry, no_tex=False):
    c = c.set_index("comparison_id").loc[list(PAIR_ORDER)].reset_index()
    d = d.set_index("N").loc[list(N_VALUES)].reset_index()
    if not np.isfinite(c[["estimate", "ci_lower", "ci_upper"]]).all().all() or not np.isfinite(d[["estimate", "ci_lower", "ci_upper"]]).all().all():
        raise ValueError("Every plotted effect and interval must be finite.")
    style = dict(uniform.FIGURE_RC)
    if no_tex:
        style.update({"text.usetex": False, "text.latex.preamble": "", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "cm"})
    with matplotlib.rc_context(style):
        fig, (axc, axd) = plt.subplots(1, 2, figsize=(geometry["width_inches"], geometry["height_inches"]),
            gridspec_kw={"width_ratios": (1, 1.08)}, layout="constrained")
        y = np.arange(5, -1, -1)
        axc.axvline(0, color=".45", ls="--", lw=.8, zorder=0)
        axc.hlines(y, c.ci_lower, c.ci_upper, color=ORANGE, lw=1.1)
        axc.scatter(c.estimate, y, s=30, color=ORANGE, edgecolor="white", linewidth=.5, zorder=3)
        axc.set_yticks(y, [p.replace("–", "--") if not no_tex else p for p in PAIR_ORDER])
        axc.set_ylim(-.65, 5.65)
        axc.set_xlim(*padded_limits(c[["estimate", "ci_lower", "ci_upper"]].to_numpy()))
        axc.set_xlabel("Change in median PCC (ranked − equal)")
        axc.set_title(r"\textbf{C}\quad Ranking effect on reproducibility" if not no_tex else "C  Ranking effect on reproducibility", loc="left")
        axc.grid(axis="x")
        axc.set_axisbelow(True)
        axd.axhline(0, color=".45", ls="--", lw=.8, zorder=0)
        axd.scatter(pairs.N, pairs.estimate, s=16, color=LIGHT_ORANGE, edgecolor="white", linewidth=.4, zorder=2)
        axd.plot(d.N, d.estimate, color=ORANGE, lw=1, zorder=3)
        axd.vlines(d.N, d.ci_lower, d.ci_upper, color=ORANGE, lw=1.1, zorder=3)
        axd.scatter(d.N, d.estimate, s=35, color=ORANGE, edgecolor="white", linewidth=.55, zorder=4)
        axd.set_xscale("log")
        axd.set_xlim(1.65, 48)
        axd.set_xticks(N_VALUES, [str(n) for n in N_VALUES])
        axd.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        axd.set_ylim(*padded_limits(np.r_[d[["estimate", "ci_lower", "ci_upper"]].to_numpy().ravel(), pairs.estimate]))
        axd.set_xlabel("Number of datasets per model")
        axd.set_ylabel("Change in mean PCC\n(ranked − equal)")
        axd.set_title(r"\textbf{D}\quad Ranking effect on stability" if not no_tex else "D  Ranking effect on stability", loc="left")
        axd.grid(axis="both")
        axd.set_axisbelow(True)
    return fig


def caption():
    return r"""% Render only after all comparisons pass matching and provenance checks.
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/real_data_ranking_effect.pdf}
\caption{\textbf{Effect of quality-ranked reference weights on shared-profile agreement.}
All contrasts use fresh or provenance-matched fitted models with identical dataset memberships,
transcript splits, training seed (42), model/training configuration and local reliability weights
$w_{dt}$, selecting best-validation-loss checkpoints separately under each policy.
The frozen ten-component QC table defines $q_d=(R-r_d+1)/R$, with rank 1 best and $R$
the maximum rank in the complete table; ranked reference weights are
$\pi_d=q_d/\sum_{j\in\mathcal D}q_j$, versus $1/|\mathcal D|$ for uniform weights.
\textbf{C}, for each of six pairs of source-disjoint panels (29,29,28,28 datasets),
the point is the median full-CDS transcript-level PCC under ranked weights minus the
median under uniform weights, recomputed on matched finite cases. This is a difference
of medians, not a median of paired differences.
\textbf{D}, at each $N=2,5,10,20,40$, each policy is summarized by the mean of the
three designated source-disjoint A/B pair means across transcripts; large connected points
show ranked minus uniform, and smaller light points show the three pair-level changes.
The primary D cohort is fixed across both policies, every $N$, and all designated pairs.
N=80, N=114 and cumulative top-quality collections are not used. All profiles are native,
unsmoothed full-CDS outputs with identical aligned positions and no post-hoc rescaling.
Bars are 95\% percentile intervals from 5,000 paired transcript-cluster bootstrap draws
(seed 20260910): each transcript carries both policies and all associated comparisons,
using the same resample across all six C pairs or all D counts and pairs.
Intervals are conditional on the fitted models and selected collections; comparisons
sharing transcripts or models are dependent. Positive changes indicate improved agreement,
not demonstrated biological accuracy; QC rank is not biological ground truth.
Exact cohorts, exclusions and their differences from Figure 1 are supplied in the source tables.}
\label{fig:real-data-ranking-effect}
\end{figure*}
"""


def baseline_inventory(run, q, rank, ranking_provenance):
    """Describe required matches without reading any incompatible model outputs."""
    cfg, split = config_for(run), split_ids(run)
    reliability_values(run, split)
    require_weights(run, "equal", cfg, q)
    missing = sorted(set(run.datasets) - rank.keys())
    if missing:
        raise ValueError(f"Missing global ranks (no aliases/imputation): {missing}")
    directory = run.directory
    record = dict(experiment=run.experiment, collection=run.collection, N=run.N,
        pair_id=run.pair_id, side=run.side, training_seed=run.seed, equal_run_id=run.run_id,
        equal_directory=str(directory), ordered_datasets=json.dumps(run.datasets),
        source_families=json.dumps(run.sources), checkpoint_selection="best_val_loss",
        resolved_config_sha256=sha256(directory/"resolved_config.yaml"),
        split_manifest_sha256=sha256(directory/"split_manifest.json"),
        reliability_manifest_sha256=sha256(directory/"reliability_reference_manifest.json"),
        train_count=len(split["train"]), validation_count=len(split["validation"]), test_count=len(split["test"]),
        train_id_hash=transcript_id_hash(split["train"]), validation_id_hash=transcript_id_hash(split["validation"]),
        test_id_hash=transcript_id_hash(split["test"]),
        prediction_manifests=len(list((directory/"predictions").rglob("prediction_checkpoint_manifest.json"))),
        checkpoint_files=len(list((directory/"checkpoints").rglob("*.ckpt"))))
    try:
        _, record["training_code_evidence"] = execution_evidence(run)
        record["training_code_status"] = "training_linked_snapshot_available"
    except ValueError as exc:
        record["training_code_status"] = "unverified"
        record["training_code_evidence"] = str(exc)
    raw = np.array([q[d] for d in run.datasets])
    weights = [dict(experiment=run.experiment, collection=run.collection, N=run.N,
        dataset_order=i, dataset_id=d, global_rank=rank[d], raw_reference_score_q=float(w),
        pi_equal=1/run.N, pi_ranked=float(w/raw.sum()), R=ranking_provenance["R"],
        ranking_sha256=ranking_provenance["sha256"],
        status="required_design_weights_NOT_evidence_of_ranked_training")
        for i, (d, w) in enumerate(zip(run.datasets, raw))]
    return record, weights


def figure_one_cohorts(reference, paired):
    """Use Figure 1 per-transcript tables for cohort accounting, never subtraction."""
    source = reference.parent/"real_data_equal_source"
    provenance, differences = {}, []
    for experiment, name, comparison in (("C", "panel_a_per_transcript.csv", "panel_pair"),
                                          ("D", "panel_b_per_transcript.csv", "pair_id")):
        path = source/name
        if not path.is_file():
            provenance[experiment] = dict(status="Figure 1 cohort not verified; source table unavailable", path=str(path))
            continue
        table = pd.read_csv(path)
        if experiment == "D":
            table = table[table.N.isin(N_VALUES)]
        table = table[table.profile_domain == "full_CDS"]
        provenance[experiment] = dict(path=str(path), sha256=sha256(path), rows=len(table),
            transcripts=table.transcript_id.nunique(), valid_rows=int((table.status == "valid").sum()),
            transcript_id_hash=transcript_id_hash(table.transcript_id.unique()),
            use="cohort accounting only; no Figure 1 summary is subtracted")
        new = paired.get(experiment)
        if new is None or new.empty:
            provenance[experiment]["cohort_difference"] = "not computable: no valid matched ranked records"
            continue
        for keys, group in table.groupby([comparison] if experiment == "C" else ["N", comparison]):
            label = keys[0] if experiment == "C" else keys[1]
            N = None if experiment == "C" else int(keys[0])
            subset = new[new.comparison_id == label]
            if N is not None:
                subset = subset[subset.N == N]
            old_ids = set(group.loc[group.status == "valid", "transcript_id"])
            new_ids = set(subset.loc[subset.included, "transcript_id"])
            for tid in sorted(old_ids | new_ids):
                differences.append(dict(experiment=experiment, N=N, comparison_id=label, transcript_id=tid,
                    figure_1_included=tid in old_ids, figure_2_included=tid in new_ids,
                    status="same" if (tid in old_ids) == (tid in new_ids) else "cohort_changed"))
            provenance[experiment].setdefault("comparisons", []).append(dict(N=N, comparison_id=label,
                figure_1_count=len(old_ids), figure_2_count=len(new_ids), removed=len(old_ids-new_ids), added=len(new_ids-old_ids)))
    return provenance, differences


def verify_plot(fig, c, d, pairs):
    """Check plotted coordinates and interval endpoints against the saved tables."""
    a, b = fig.axes
    c = c.set_index("comparison_id").loc[list(PAIR_ORDER)]
    d = d.set_index("N").loc[list(N_VALUES)]
    np.testing.assert_allclose(a.collections[1].get_offsets()[:, 0], c.estimate)
    segments = np.asarray(a.collections[0].get_segments())
    np.testing.assert_allclose(segments[:, :, 0], c[["ci_lower", "ci_upper"]])
    np.testing.assert_allclose(b.collections[0].get_offsets(), pairs[["N", "estimate"]])
    np.testing.assert_allclose(b.collections[2].get_offsets(), np.c_[N_VALUES, d.estimate])
    np.testing.assert_allclose(b.lines[1].get_ydata(), d.estimate)
    np.testing.assert_allclose(np.asarray(b.collections[1].get_segments())[:, :, 1], d[["ci_lower", "ci_upper"]])
    for limits, values in ((a.get_xlim(), c[["estimate", "ci_lower", "ci_upper"]].to_numpy()),
                           (b.get_ylim(), np.r_[d[["estimate", "ci_lower", "ci_upper"]].to_numpy().ravel(), pairs.estimate])):
        if not limits[0] <= np.min(values) <= np.max(values) <= limits[1]:
            raise ValueError("A plotted effect/interval is outside its axis.")
    if len(a.lines) != 1 or b.get_xscale() != "log" or tuple(b.get_xticks()) != N_VALUES:
        raise ValueError("Categorical connection or N-axis contract changed.")


def write_report(source, manifest, availability, missing, candidates):
    ranking = manifest.get("ranking", {})
    lines = ["# Figure 2: matched ten-component ranking effects", "",
        f"Status: **{manifest['status']}**.", "",
        "No new training or checkpoint alteration was performed. Incompatible predictions were not loaded.", "",
        "## Scientific matching", "",
        "C requires four source-disjoint 29/29/28/28 panels. D requires the original representative "
        "source-disjoint pairs pair01–pair03, sides A/B, at N=2,5,10,20,40. All are seed 42 and best_val_loss.", "",
        "Membership and ordering, source families, ordered splits, eligibility/training configuration, numerical "
        "train-only w_dt references, data/sequence bytes, and training-linked code/environment must match. "
        "Today's code or matching YAML alone does not establish historical implementation equivalence.", "",
        "## Frozen ranking", "",
        f"File: `{ranking.get('path', 'unavailable')}`.", "",
        f"SHA-256: `{ranking.get('sha256', 'unverified')}`; complete-table R={ranking.get('R', 'unverified')}, "
        f"rows={ranking.get('rows', 'unverified')}; rank 1 is best, power=1.", "",
        "Components: " + ", ".join(ranking.get("components", [])) + ".", "",
        "q=(R-r+1)/R; ranked pi=q/sum(q within the selected collection); equal pi=1/N. "
        "The complete table remains fixed, including global datasets not selected into a particular collection. "
        "`required_ranked_reference_weights.csv` is a design specification, NOT evidence that a ranked model "
        "was trained. Only passed checkpoint and runtime checks populate `verified_reference_weights.csv`.", "",
        "QC transcript scope and the historical ranking freeze date are **not verified**. "
        "Verified training-only local reliability fitting does not establish train-only global QC ranks.", "",
        "## Availability and missing matches", ""]
    for experiment in ("C", "D"):
        rows = [r for r in availability if r["experiment"] == experiment]
        absent = [r for r in missing if r["experiment"] == experiment]
        lines.append(f"- {experiment}: {len(rows)} equal task declarations checked; {len(absent)} missing/incompatible ranked matches.")
    lines.extend(["", "The exact required task IDs, ordered datasets, split hashes, w_dt hashes and failure reasons "
        "are in `missing_runs.csv`. Baseline prediction availability is not proof of matched training-code provenance.", ""])
    for experiment in ("C", "D"):
        labels = [r["collection"] for r in missing if r["experiment"] == experiment]
        if labels:
            lines.append(f"{experiment}: " + ", ".join(f"`{label}`" for label in labels) + ".")
    lines.extend(["", "## Candidate rejections", ""])
    for row in candidates:
        component = row.get("actual_component_count")
        component_note = f" Recorded checksum identifies the {component}-component table." if component else ""
        lines.append(f"- `{row['manifest']}`: {row['reason']}.{component_note}")
    lines.extend(["", "## Cohorts and estimands", "",
        "C is a difference of medians, computed separately per panel pair on its matched finite cohort. "
        "D is a difference between policy-specific averages of the three pair means, using one complete "
        "cohort across every N/pair and both policies. No old figure summary is subtracted. "
        "Source arrays retain their full-CDS amplitudes; masks remove padding only. Missing, invalid, "
        "misaligned, constant and near-constant profiles remain excluded with reasons, never zero-imputed.", "",
        "The near-constant criterion is SD <= 1e-8 * max(1, |mean|), reused from Figure 1. "
        "Nonpositive/nonfinite or non-mean-one shared outputs are reported, not repaired.", ""])
    for experiment, cohort in manifest.get("figure_1_cohorts", {}).items():
        lines.append(f"- {experiment} Figure 1 cohort: " + json.dumps(cohort, ensure_ascii=False) + ".")
    lines.extend(["", "Exactly 5,000 transcript-cluster resamples use NumPy default_rng seed 20260910, "
        "with both policies and every comparison attached to each sampled transcript. The resample is shared "
        "across the six C pairs or all 15 D comparisons. 95% intervals are linear-interpolated 2.5th/97.5th "
        "percentiles, conditional on fitted models and selected collections. Neither codons nor panel-pair "
        "rows are independent bootstrap units. Agreement is not biological accuracy.", "",
        "## Rendering and provenance limitations", "",
        "Reference geometry and typography: " + json.dumps(manifest.get("figure_geometry", {}), ensure_ascii=False) + ".", "",
        "The current Figure 1 includes additional N=80/114 convergence ticks. This figure uses the same "
        "logarithmic scale and retained 2/5/10/20/40 ticks, excluding convergence and those two N values as requested.", "",
        "Figure 1 dimensions/style were inspected, not digitized. Native source arrays are required for every effect. "
        "README descriptions are subordinate to the resolved mass-free production configuration. The only "
        "permitted scientific configuration difference is the reference policy/ranking input; all differences "
        "are recorded before evaluating predictions.", "",
        "## Blockers", ""])
    lines.extend(f"- {failure}" for failure in manifest.get("failures", []))
    if manifest["status"] != "complete":
        lines.extend(["", "**No article PDF or PNG was created.** Empty paired/summary CSVs contain headers only, "
            "not dummy effects or placeholder confidence intervals. No inference-only repair is justified for "
            "a six-component model or a cumulative-subset checkpoint. Download the compatible original exports, "
            "selected checkpoints and training-linked provenance if they exist elsewhere. Otherwise genuinely "
            "matched trainings require separate authorization; unverified old equal controls cannot automatically be reused."])
    lines.extend(["", "## Exact regeneration command", "", "```bash", manifest["regeneration_command"], "```", ""])
    text = "\n".join(lines)
    (source/"MATCHING_PROVENANCE_EXCLUSIONS.md").write_text(text)
    # No Markdown dependency; keep the machine-readable tables next to a
    # readable, escaped HTML report rather than embedding raw artifact content.
    blocks = []
    for block in text.split("\n\n"):
        escaped = html.escape(block)
        if block.startswith("# "):
            blocks.append("<h1>" + escaped[2:] + "</h1>")
        elif block.startswith("## "):
            blocks.append("<h2>" + escaped[3:] + "</h2>")
        else:
            blocks.append("<p style='white-space:pre-wrap'>" + escaped + "</p>")
    (source/"matching_report.html").write_text("<!doctype html><meta charset='utf-8'><title>Figure 2 matching report</title>"
        "<style>body{max-width:1000px;margin:3em auto;padding:0 2em;font:16px/1.55 system-ui;color:#222}"
        "p{overflow-wrap:anywhere}h2{margin-top:2em}</style>" + "\n".join(blocks))


def main(argv=None):
    args = parse_args(argv)
    output = args.output_dir
    source = output/"real_data_ranking_effect_source"
    source.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve())]
    for key, value in vars(args).items():
        if isinstance(value, Path):
            command.extend(["--"+key.replace("_", "-"), str(value)])
    if args.no_tex:
        command.append("--no-tex")
    manifest = dict(status="blocked_missing_matched_runs", created_at_utc=datetime.now(timezone.utc).isoformat(),
        script=str(Path(__file__).resolve()), script_sha256=sha256(Path(__file__).resolve()),
        training_seed=TRAINING_SEED, bootstrap_draws=DRAW_COUNT, bootstrap_seed=BOOTSTRAP_SEED,
        regeneration_command=shlex.join(command), failures=[], new_training_launched=False,
        inference_launched=False, software=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__,
                                              matplotlib=matplotlib.__version__))
    availability, missing, required_weights, verified_weights, candidates = [], [], [], [], []
    audits, paired, summaries, pair_summary = {}, {}, {}, pd.DataFrame()
    errors = (ValueError, KeyError, FileNotFoundError, RuntimeError, AssertionError)
    try:
        rank, q, manifest["ranking"] = load_ranking(args.ranking_table)
        manifest["figure_geometry"] = figure_geometry(args.reference_figure)
        manifest["figure_geometry"]["usetex"] = not args.no_tex
        found = discover_manifests(args.scan_root)
        candidates = candidate_inventory(found, manifest["ranking"]["sha256"])
        for experiment, equal_root, ranked_root in (
                ("C", args.panel_equal_root, args.panel_ranked_root),
                ("D", args.stability_equal_root, args.stability_ranked_root)):
            try:
                equal, _ = read_runs(equal_root, experiment)
                baseline = {}
                for run in equal:
                    record, required = baseline_inventory(run, q, rank, manifest["ranking"])
                    baseline[run.collection] = record
                    availability.append(record)
                    required_weights.extend(required)
                ranked, reasons = choose_ranked_root(experiment, equal, ranked_root, found, manifest["ranking"]["sha256"])
                manifest.setdefault("candidate_matching_rejections", {})[experiment] = reasons
                if ranked is None:
                    reason = "No ten-component ranked experiment with the same ordered collections, splits and seed was found."
                    missing.extend({**baseline[r.collection], "required_policy": "ten_component_quality_rank_power_one",
                        "reason": reason, "ranking_sha256": manifest["ranking"]["sha256"],
                        "required_artifacts": "training-linked source/environment; resolved_config.yaml; launch_command.sh; "
                        "run/subset_manifest.json; split_manifest.json; reliability_reference_manifest.json; "
                        "selected best_val_loss checkpoint; prediction_checkpoint_manifest.json; gamma_reference_manifest.json; original L_bio parquet"}
                        for r in equal)
                    manifest["failures"].append(f"{experiment}: {reason} {len(equal)} ranked counterparts are missing locally.")
                    continue
                checks = {}
                for a, b in zip(equal, ranked, strict=True):
                    try:
                        checks[a.collection] = paired_run_audit(a, b, q)
                        verified_weights.extend(checks[a.collection]["weights"])
                    except errors as exc:
                        missing.append({**baseline[a.collection], "required_policy": "ten_component_quality_rank_power_one",
                            "ranked_directory": str(b.directory), "reason": f"{type(exc).__name__}: {exc}"})
                audits[experiment] = checks
                if len(checks) != len(equal):
                    manifest["failures"].append(f"{experiment}: {len(equal)-len(checks)} equal/ranked model matches failed; no partial family is plotted.")
                    continue
                paired[experiment] = comparison_records(experiment, equal, ranked, checks)
                # Bootstrap from a round-trip of the saved paired records, not an
                # unrelated in-memory/precomputed Figure 1 summary.
                path = source/f"panel_{experiment.lower()}_paired_per_transcript.csv"
                write_table(path, paired[experiment])
                paired[experiment] = pd.read_csv(path, float_precision="round_trip")
                summary, pairs, boot = summarize_records(paired[experiment], experiment)
                summaries[experiment] = summary
                manifest.setdefault("bootstrap", {})[experiment] = boot
                if experiment == "D":
                    pair_summary = pairs
            except errors as exc:
                manifest["failures"].append(f"{experiment}: {type(exc).__name__}: {exc}")
    except errors as exc:
        manifest["failures"].append(f"{type(exc).__name__}: {exc}")

    manifest["figure_1_cohorts"], cohort_rows = figure_one_cohorts(args.reference_figure, paired)
    for experiment in ("C", "D"):
        write_table(source/f"panel_{experiment.lower()}_paired_per_transcript.csv", paired.get(experiment, []), PAIRED_COLUMNS)
        write_table(source/f"panel_{experiment.lower()}_summary.csv", summaries.get(experiment, []), SUMMARY_COLUMNS)
    write_table(source/"panel_d_pair_summary.csv", pair_summary,
                ["N", "comparison_id", "training_seed", "n_transcripts", "equal_mean_PCC", "ranked_mean_PCC", "estimate"])
    write_table(source/"equal_run_availability.csv", availability, ["experiment", "collection", "equal_directory"])
    write_table(source/"missing_runs.csv", missing, ["experiment", "collection", "reason"])
    write_table(source/"candidate_inventory.csv", candidates, ["manifest", "reason"])
    write_table(source/"required_ranked_reference_weights.csv", required_weights,
                ["experiment", "collection", "dataset_id", "global_rank", "raw_reference_score_q", "pi_equal", "pi_ranked", "status"])
    write_table(source/"verified_reference_weights.csv", verified_weights,
                ["experiment", "collection", "N", "policy", "dataset_id", "raw_reference_weight", "pi"])
    exclusions = pd.concat([f[~f.included] for f in paired.values()], ignore_index=True) if paired else []
    write_table(source/"exclusions.csv", exclusions, PAIRED_COLUMNS)
    write_table(source/"figure_1_cohort_comparison.csv", cohort_rows,
                ["experiment", "N", "comparison_id", "transcript_id", "figure_1_included", "figure_2_included", "status"])
    write_json(source/"matched_run_checks.json", audits)
    (output/"real_data_ranking_effect.tex").write_text(caption())
    (source/"regenerate.sh").write_text("#!/bin/bash\nset -euo pipefail\n" + shlex.join(command) + "\n")
    if set(summaries) == {"C", "D"} and not manifest["failures"]:
        c, d = (pd.read_csv(source/f"panel_{exp}_summary.csv", float_precision="round_trip") for exp in ("c", "d"))
        pairs = pd.read_csv(source/"panel_d_pair_summary.csv", float_precision="round_trip")
        fig = build_figure(c, d, pairs, manifest["figure_geometry"], args.no_tex)
        verify_plot(fig, c, d, pairs)
        # Save inside the same rc_context: TeX settings also affect deferred draw.
        with matplotlib.rc_context(dict(uniform.FIGURE_RC, **({"text.usetex": False} if args.no_tex else {}))):
            fig.savefig(output/"real_data_ranking_effect.pdf")
            fig.savefig(output/"real_data_ranking_effect.png", dpi=600)
        plt.close(fig)
        manifest.update(status="complete", plotted_coordinates_verified_from_saved_tables=True, png_dpi=600,
            output_pdf_sha256=sha256(output/"real_data_ranking_effect.pdf"),
            output_png_sha256=sha256(output/"real_data_ranking_effect.png"))
    manifest["missing_ranked_matches"] = {exp: sum(row["experiment"] == exp for row in missing) for exp in ("C", "D")}
    write_report(source, manifest, availability, missing, candidates)
    manifest["source_table_sha256"] = {p.name: sha256(p) for p in sorted(source.glob("*.csv"))}
    write_json(source/"figure_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "missing_ranked_matches": manifest["missing_ranked_matches"],
        "report": str(source/"MATCHING_PROVENANCE_EXCLUSIONS.md"), "failures": manifest["failures"]}, indent=2))
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
