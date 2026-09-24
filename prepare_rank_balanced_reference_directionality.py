#!/usr/bin/env python3
"""Prepare or run a matched three-arm reference-direction experiment.

The four-panel partition must already have been prepared by
``analyses/audit_four_panel_reference_quality.py --mode prepare-rank-balanced``. This
entrypoint does not search for another partition.  It expands the frozen
templates into three otherwise matched arms:

``equal``
    Uniform fixed-reference weights.
``ranked``
    The production global-rank mapping q=(R-r+1)/R, normalized in panel.
``reverse``
    The same ranked q multiset assigned in reverse rank order within panel.

Reversing the assignment, rather than defining a second nonlinear weight
formula, preserves the ranked arm's exact concentration and isolates whether
the direction of the dataset-quality assignment matters.  Preparation is
CPU-only.  Training requires both ``--task-index`` and
``--authorize-training`` and delegates to the frozen production entrypoint.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from hydra import compose, initialize_config_dir

from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDatasetDataModule import (
    load_dataset_quality_ranking,
)
from Utils.campaign_training import initialization_audit
from Utils.panel_reference_preparation import assert_training_contract
from run_real_exp8_L_stability_quality_rank import inspect_ranking_components
from run_real_independent_panel_convergence_quality_rank import _flatten_config


ROOT = Path(__file__).resolve().parent
ARMS = ("equal", "ranked", "reverse")
DOCUMENTED_RANKING_SHA256 = (
    "5811cadf68c56740e83b232b2990299d527630326205db9cba8bc7024d3cf1f8"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_training_seeds(text: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(token.strip()) for token in text.split(",") if token.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("training seeds must be comma-separated integers") from exc
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("training seeds must be nonempty and distinct")
    return seeds


def normalized(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all() or (array <= 0).any():
        raise ValueError("Reference weights must be a nonempty, finite, positive vector.")
    return array / array.sum(dtype=np.float64)


def reverse_weight_assignment(
    datasets: Iterable[str], ranks: dict[str, float], quality: dict[str, float]
) -> tuple[dict[str, float], dict[str, str]]:
    """Assign the forward q multiset in reverse global-rank order.

    Dataset-ID tie breaking is explicit and deterministic.  The authoritative
    ranks themselves are never edited or re-ranked.
    """
    names = tuple(map(str, datasets))
    if len(names) != len(set(names)):
        raise ValueError("Panel dataset IDs must be unique.")
    missing = sorted(set(names) - set(ranks) | (set(names) - set(quality)))
    if missing:
        raise ValueError(f"Datasets missing from the complete ranking: {missing}")
    ordered = sorted(names, key=lambda name: (float(ranks[name]), name))
    donors = tuple(reversed(ordered))
    assigned = {recipient: float(quality[donor]) for recipient, donor in zip(ordered, donors)}
    donor_by_recipient = dict(zip(ordered, donors))
    forward = normalized(quality[name] for name in names)
    reverse = normalized(assigned[name] for name in names)
    np.testing.assert_allclose(
        np.sort(forward), np.sort(reverse), rtol=0.0, atol=1e-15,
        err_msg="Reverse assignment changed the ranked weight multiset.",
    )
    return assigned, donor_by_recipient


def _reference_policy(
    arm: str,
    datasets: list[str],
    ranks: dict[str, float],
    quality: dict[str, float],
) -> tuple[dict, list[dict]]:
    reverse, donors = reverse_weight_assignment(datasets, ranks, quality)
    if arm == "equal":
        raw = {name: 1.0 for name in datasets}
        implementation = dict(weighting="equal", quality_rank_power=0.0)
    elif arm == "ranked":
        raw = {name: float(quality[name]) for name in datasets}
        donors = {name: name for name in datasets}
        implementation = dict(weighting="quality_rank", quality_rank_power=1.0)
    elif arm == "reverse":
        raw = reverse
        implementation = dict(
            weighting="explicit", quality_rank_power=1.0, explicit_weights=raw
        )
    else:
        raise ValueError(f"Unknown arm: {arm}")
    pi_values = normalized(raw[name] for name in datasets)
    pi = dict(zip(datasets, map(float, pi_values)))
    rows = []
    for name in datasets:
        donor = donors.get(name)
        rows.append(
            dict(
                arm=arm,
                dataset_id=name,
                global_rank=float(ranks[name]),
                original_q=float(quality[name]),
                assigned_q=float(raw[name]),
                assigned_from_dataset=donor if arm != "equal" else None,
                assigned_from_global_rank=(float(ranks[donor]) if donor is not None else np.nan),
                pi=pi[name],
            )
        )
    return implementation, rows


def _assert_arm_match(equal: dict, other: dict) -> list[dict]:
    """Require the scientific configurations to differ only in reference arm."""
    flat_equal, flat_other = _flatten_config(equal), _flatten_config(other)
    exact = {
        "name",
        "model.gamma_centering.reference.weighting",
        "model.gamma_centering.reference.quality_rank_power",
    }
    prefixes = (
        "model.gamma_centering.reference.explicit_weights.",
        "paths.checkpoints",
        "paths.logs",
        "paths.results",
        "orchestrator.",
    )
    differences = []
    for key in sorted(set(flat_equal) | set(flat_other)):
        before, after = flat_equal.get(key), flat_other.get(key)
        if before == after:
            continue
        if key not in exact and not key.startswith(prefixes):
            raise ValueError(f"Unapproved paired-arm configuration difference: {key}")
        differences.append(dict(key=key, equal=before, other=after))
    return differences


def _load_base(
    prepared_root: Path, expected_ranking_sha256: str | None = None
) -> dict:
    required = [
        prepared_root / "rerun_plan.json",
        prepared_root / "partition_search.json",
        prepared_root / "frozen_global_ranking.tsv",
        prepared_root / "partition_and_split_manifests/panel_assignment.csv",
        prepared_root / "partition_and_split_manifests/common_split_manifest.json",
        prepared_root / "heldout_support_report.csv",
        prepared_root / "code_snapshot/main_ribounmix_multidataset.py",
        prepared_root / "frozen_execution.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete rank-balanced preparation: {missing}")
    plan = read_json(prepared_root / "rerun_plan.json")
    if plan.get("status") != "prepared_not_launched" or plan.get("training_launched"):
        raise ValueError(
            f"Base preparation is incomplete: {prepared_root}; status={plan.get('status')!r}; "
            f"last error={plan.get('reason', 'not recorded')}. Finish the saved preparation with "
            f"analyses/audit_four_panel_reference_quality.py --resume-preparation {prepared_root} --gpus inherit."
        )
    ranking_path = prepared_root / "frozen_global_ranking.tsv"
    observed_sha = sha256(ranking_path)
    declared_sha = plan.get("frozen_ranking_sha256")
    if not declared_sha:
        raise ValueError("Base preparation did not record its frozen ranking checksum.")
    if observed_sha != declared_sha:
        raise ValueError(
            "The ranking copied into the prepared partition changed after it was "
            f"frozen: observed SHA256={observed_sha}, recorded={declared_sha}."
        )
    original_sha = plan.get("original_ranking_sha256")
    if original_sha is not None and original_sha != declared_sha:
        raise ValueError(
            "The completed base preparation records different source and frozen "
            "ranking bytes; this experiment requires the canonical unmodified table."
        )
    if expected_ranking_sha256 is not None and observed_sha != expected_ranking_sha256:
        raise ValueError(
            "The explicitly requested ranking checksum differs from the completed "
            f"base preparation: observed={observed_sha}, requested={expected_ranking_sha256}."
        )
    rank_metadata = inspect_ranking_components(ranking_path, expected_count=10)
    rank_by_name, quality_by_name = load_dataset_quality_ranking(str(ranking_path))
    ranking_frame = pd.read_csv(ranking_path, sep="\t")
    rank_universe = float(pd.to_numeric(ranking_frame.quality_rank, errors="raise").max())
    if len(ranking_frame) != 115 or rank_universe != 115.0:
        raise ValueError(
            "The ten-component reference universe must contain 115 rows with R=115; "
            f"found rows={len(ranking_frame)}, R={rank_universe}."
        )

    assignment = pd.read_csv(required[3])
    expected_columns = {"dataset_id", "panel_id", "source_family"}
    if not expected_columns <= set(assignment):
        raise ValueError(f"Panel assignment lacks columns {sorted(expected_columns - set(assignment))}.")
    if assignment.dataset_id.duplicated().any() or len(assignment) != 114:
        raise ValueError("Expected 114 unique retained datasets in the proposed partition.")
    sizes = assignment.groupby("panel_id", sort=True).size().to_dict()
    if sizes != {"panel_01": 29, "panel_02": 29, "panel_03": 28, "panel_04": 28}:
        raise ValueError(f"Unexpected panel capacities: {sizes}")
    crossings = assignment.groupby("source_family").panel_id.nunique()
    if (crossings > 1).any():
        raise ValueError("A source family crosses proposed panels.")
    if assignment.source_family.nunique() != 85:
        raise ValueError(
            f"Expected 85 source families, found {assignment.source_family.nunique()}."
        )
    missing_rank = sorted(set(assignment.dataset_id) - set(rank_by_name))
    if missing_rank:
        raise ValueError(f"Retained datasets missing global ranks: {missing_rank}")

    support = pd.read_csv(prepared_root / "heldout_support_report.csv")
    blocked = support[support.blocks_preparation.astype(bool)]
    if len(blocked):
        raise ValueError(
            f"Proposed partition has {len(blocked)} blocking validation-support rows."
        )
    split = read_json(required[4])
    if len(split.get("common_validation_ids", [])) != 1593 or len(split.get("common_test_ids", [])) != 1593:
        raise ValueError("Frozen validation/test transcript counts differ from 1,593 each.")
    return dict(
        plan=plan,
        assignment=assignment,
        split=split,
        ranking_path=ranking_path,
        ranking_sha256=observed_sha,
        rank_metadata=rank_metadata,
        rank_universe=rank_universe,
        ranks=rank_by_name,
        quality=quality_by_name,
        support=support,
    )


def _preparation_identity(output_root: Path) -> dict:
    manifest = read_json(output_root / "experiment_manifest.json")
    for raw, expected in manifest["frozen_file_sha256"].items():
        path = Path(raw)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Frozen experiment input changed: {path}")
    matrix = pd.read_csv(output_root / "task_matrix.csv")
    if len(matrix) != manifest["number_of_tasks"]:
        raise ValueError("Task matrix length differs from the frozen manifest.")
    tasks = read_json(output_root / "tasks.json")
    if object_sha256(tasks) != manifest["tasks_sha256"]:
        raise ValueError("Prepared task commands changed after freezing.")
    return manifest


def prepare(args: argparse.Namespace) -> dict:
    root = args.output_root
    root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root.parent / f".{root.name}.prepare.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "experiment_manifest.json").is_file():
            manifest = _preparation_identity(root)
            requested = list(args.training_seeds)
            if requested != manifest["training_seeds"]:
                raise ValueError(
                    f"Prepared seeds are {manifest['training_seeds']}, requested {requested}; "
                    "use a new output root for a different frozen experiment."
                )
            return manifest
        if root.exists() and any(root.iterdir()):
            raise ValueError(
                f"Refusing to overwrite incomplete/nonempty experiment root: {root}"
            )

        base = _load_base(args.prepared_root, args.expected_ranking_sha256)
        root.mkdir(parents=True)
        frozen = root / "frozen_inputs"
        frozen.mkdir()
        copied = {
            "ranking": (base["ranking_path"], frozen / "HEK_riboseq_profile_quality_rank_components.tsv"),
            "panel_assignment": (
                args.prepared_root / "partition_and_split_manifests/panel_assignment.csv",
                frozen / "panel_assignment.csv",
            ),
            "common_split": (
                args.prepared_root / "partition_and_split_manifests/common_split_manifest.json",
                frozen / "common_split_manifest.json",
            ),
            "design_config": (args.prepared_root / "design_config.json", frozen / "design_config.json"),
            "partition_search": (
                args.prepared_root / "partition_search.json",
                frozen / "partition_search.json",
            ),
        }
        for source, destination in copied.values():
            destination.write_bytes(source.read_bytes())
        for panel in sorted(base["assignment"].panel_id.unique()):
            source = args.prepared_root / f"partition_and_split_manifests/{panel}_reliability_reference_manifest.json"
            destination = frozen / f"{panel}_reliability_reference_manifest.json"
            destination.write_bytes(source.read_bytes())

        config_dir = root / "resolved_configs"
        config_dir.mkdir()
        rows, tasks, differences, initialization_rows = [], [], [], []
        panels = sorted(base["assignment"].panel_id.unique())
        source_lookup = base["assignment"].set_index("dataset_id").source_family.to_dict()
        for seed in args.training_seeds:
            for arm in ARMS:
                for panel in panels:
                    template_path = args.prepared_root / f"resolved_configs/{panel}_equal.yaml"
                    template = yaml.safe_load(template_path.read_text())
                    datasets = list(map(str, template["experiment"]["dataset"]))
                    expected = set(base["assignment"].loc[base["assignment"].panel_id == panel, "dataset_id"])
                    if set(datasets) != expected or len(datasets) != len(expected):
                        raise ValueError(f"{panel}: template dataset order/membership mismatch.")
                    implementation, weight_rows = _reference_policy(
                        arm, datasets, base["ranks"], base["quality"]
                    )
                    for record in weight_rows:
                        record.update(panel_id=panel, source_family=source_lookup[record["dataset_id"]])
                    rows.extend(weight_rows)

                    task_id = f"rankbal10_{panel}_{arm}_seed{seed}"
                    task_root = root / "runs" / f"seed{seed}" / arm / panel
                    task_root.mkdir(parents=True)
                    cfg = copy.deepcopy(template)
                    cfg["name"] = task_id
                    cfg["experiment"].update(
                        seed=int(seed),
                        from_checkpoint=False,
                        resume_training_state=False,
                        resume_checkpoint_path=None,
                        allow_weights_only_resume=False,
                        train=True,
                        predict=True,
                    )
                    cfg["prediction"].update(
                        checkpoint_variants=["best_val_loss"],
                        sequence_only_shared_profile=True,
                    )
                    cfg["split"].update(
                        external_manifest=str(frozen / "common_split_manifest.json"),
                        external_panel_name=panel,
                    )
                    cfg["data"]["reliability_reference_manifest"] = str(
                        frozen / f"{panel}_reliability_reference_manifest.json"
                    )
                    cfg["data"]["dataset_quality_ranking"].update(
                        path=str(frozen / "HEK_riboseq_profile_quality_rank_components.tsv"),
                        dataset_column="dataset",
                        rank_column="quality_rank",
                        strict=True,
                    )
                    cfg["model"]["mean_correction"] = "learned"
                    reference = cfg["model"]["gamma_centering"]["reference"]
                    reference.pop("explicit_weights", None)
                    reference.update(dataset_names=None, **implementation)
                    # Same safe numerical execution in every arm: mixed BF16
                    # outside the two recurrent kernels, full (untruncated) BPTT.
                    cfg["trainer"]["precision"] = "bf16-mixed"
                    cfg["model"]["dataset_bias_params"].update(
                        context_gru_precision="float32", context_gru_tbptt_window=0
                    )
                    cfg["trainer"]["devices"] = [0]
                    cfg["paths"].update(
                        checkpoints=str(task_root / "checkpoints"),
                        logs=str(task_root / "logs"),
                        results=str(task_root / "predictions"),
                    )
                    cfg["orchestrator"] = dict(
                        experiment="rank_balanced_qrank10_directionality",
                        panel=panel,
                        arm=arm,
                        training_seed=int(seed),
                        partition_assignment_sha256=sha256(frozen / "panel_assignment.csv"),
                        reference_definition_sha256=object_sha256(weight_rows),
                    )
                    assert_training_contract(cfg)
                    config_path = config_dir / f"{task_id}.yaml"
                    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
                        compose(config_name=config_path.stem)

                    check = initialization_audit(cfg)
                    expected_reference_buffer = hashlib.sha256(
                        np.asarray(
                            [row["assigned_q"] for row in weight_rows],
                            dtype=np.float32,
                        ).tobytes()
                    ).hexdigest()
                    if check["reference_buffer_sha256"] != expected_reference_buffer:
                        raise ValueError(
                            f"{task_id}: production model reference buffer differs "
                            "from the frozen weight table."
                        )
                    check["expected_reference_buffer_sha256"] = expected_reference_buffer
                    # The production entrypoint independently recomputes this
                    # hash before training.  This turns the preparation audit
                    # into an execution-time matched-initialization guard.
                    cfg["orchestrator"]["initialization_sha256"] = check[
                        "parameter_sha256"
                    ]
                    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
                        compose(config_name=config_path.stem)
                    initialization_rows.append(dict(task_id=task_id, panel_id=panel, arm=arm, training_seed=seed, **check))
                    run_manifest = dict(
                        task_id=task_id,
                        selected_datasets=datasets,
                        training_seed=int(seed),
                        checkpoint_selection="best_val_loss",
                        fixed_gamma_reference=dict(
                            arm=arm,
                            implementation_weighting=implementation["weighting"],
                            dataset_names=datasets,
                            pi={row["dataset_id"]: row["pi"] for row in weight_rows},
                            definition=(
                                "uniform"
                                if arm == "equal"
                                else "q=(R-r+1)/R, normalized within panel"
                                if arm == "ranked"
                                else "forward q multiset assigned in reverse global-rank order, normalized within panel"
                            ),
                            global_rank_universe=base["rank_universe"],
                            ranking_sha256=base["ranking_sha256"],
                        ),
                        split_sha256=sha256(frozen / "common_split_manifest.json"),
                        reliability_sha256=sha256(
                            frozen / f"{panel}_reliability_reference_manifest.json"
                        ),
                        config_sha256=sha256(config_path),
                    )
                    write_json(task_root / "run_manifest.json", run_manifest)
                    command = [
                        sys.executable,
                        "-u",
                        str(args.prepared_root / "code_snapshot/main_ribounmix_multidataset.py"),
                        f"--config-path={config_dir}",
                        f"--config-name={config_path.stem}",
                        f"hydra.run.dir={task_root / 'hydra'}",
                        "hydra.job.chdir=false",
                    ]
                    tasks.append(
                        dict(
                            array_index=len(tasks),
                            task_id=task_id,
                            panel_id=panel,
                            arm=arm,
                            training_seed=int(seed),
                            milestone="seed42_core" if seed == 42 else "seed_repetition",
                            task_root=str(task_root),
                            config_path=str(config_path),
                            config_sha256=sha256(config_path),
                            command=command,
                        )
                    )

            seed_tasks = [task for task in tasks if task["training_seed"] == seed]
            by_panel_arm = {(task["panel_id"], task["arm"]): task for task in seed_tasks}
            for panel in panels:
                configs = {
                    arm: yaml.safe_load(Path(by_panel_arm[(panel, arm)]["config_path"]).read_text())
                    for arm in ARMS
                }
                initial = {
                    row["arm"]: row["parameter_sha256"]
                    for row in initialization_rows
                    if row["panel_id"] == panel and int(row["training_seed"]) == seed
                }
                if len(set(initial.values())) != 1:
                    raise ValueError(f"{panel}/seed{seed}: trainable initialization differs by arm.")
                for arm in ("ranked", "reverse"):
                    differences.extend(
                        dict(panel_id=panel, training_seed=seed, compared_arm=arm, **row)
                        for row in _assert_arm_match(configs["equal"], configs[arm])
                    )

        weights = pd.DataFrame(rows).drop_duplicates(["panel_id", "arm", "dataset_id"])
        if len(weights) != 114 * len(ARMS):
            raise ValueError("Reference-weight table is incomplete or duplicated.")
        summaries = []
        for (panel, arm), frame in weights.groupby(["panel_id", "arm"], sort=True):
            pi = frame.pi.to_numpy(float)
            rank = frame.global_rank.to_numpy(float)
            source_mass = frame.groupby("source_family", sort=True).pi.sum()
            largest_source_mass = float(source_mass.max())
            largest_sources = ";".join(
                source_mass.index[
                    np.isclose(
                        source_mass,
                        largest_source_mass,
                        rtol=1e-12,
                        atol=1e-15,
                    )
                ].tolist()
            )
            summaries.append(
                dict(
                    panel_id=panel,
                    arm=arm,
                    n_datasets=len(frame),
                    n_source_families=len(source_mass),
                    pi_sum=float(pi.sum()),
                    effective_reference_count=float(1.0 / np.square(pi).sum()),
                    maximum_dataset_weight=float(pi.max()),
                    maximum_source_family_mass=largest_source_mass,
                    largest_source_families=largest_sources,
                    pearson_global_rank_vs_pi=(
                        float(np.corrcoef(rank, pi)[0, 1]) if np.std(pi) > 0 else np.nan
                    ),
                )
            )
        concentration = pd.DataFrame(summaries)
        for panel in panels:
            ranked = weights[(weights.panel_id == panel) & (weights.arm == "ranked")].pi
            reverse = weights[(weights.panel_id == panel) & (weights.arm == "reverse")].pi
            np.testing.assert_allclose(
                np.sort(ranked), np.sort(reverse), rtol=0.0, atol=1e-15
            )
            pair = concentration[concentration.panel_id == panel].set_index("arm")
            np.testing.assert_allclose(
                pair.loc["ranked", ["effective_reference_count", "maximum_dataset_weight"]],
                pair.loc["reverse", ["effective_reference_count", "maximum_dataset_weight"]],
                rtol=0.0,
                atol=1e-14,
            )
        weights.to_csv(root / "reference_weights.csv", index=False)
        (
            weights.groupby(["panel_id", "arm", "source_family"], as_index=False)
            .pi.sum()
            .rename(columns={"pi": "source_family_reference_mass"})
            .to_csv(root / "source_family_reference_mass.csv", index=False)
        )
        concentration.to_csv(root / "reference_concentration.csv", index=False)
        pd.DataFrame(tasks).drop(columns="command").to_csv(root / "task_matrix.csv", index=False)
        write_json(root / "tasks.json", tasks)
        pd.DataFrame(differences).to_csv(root / "approved_arm_differences.csv", index=False)
        pd.DataFrame(initialization_rows).to_csv(
            root / "initialization_and_gradient_audit.csv", index=False
        )

        search = read_json(frozen / "partition_search.json")
        frozen_files = {
            str(path): sha256(path)
            for path in [
                *frozen.iterdir(),
                *config_dir.glob("*.yaml"),
                root / "reference_weights.csv",
                root / "source_family_reference_mass.csv",
                root / "reference_concentration.csv",
                root / "task_matrix.csv",
                root / "tasks.json",
                root / "approved_arm_differences.csv",
                root / "initialization_and_gradient_audit.csv",
                Path(__file__).resolve(),
                args.prepared_root / "frozen_execution.json",
                args.prepared_root / "code_snapshot/main_ribounmix_multidataset.py",
            ]
            if path.is_file()
        }
        manifest = dict(
            schema_version=1,
            status="prepared_not_launched",
            scientific_design="one frozen rank-balanced partition; equal, ranked, and reverse-ranked fixed gamma references",
            training_launched=False,
            prepared_root=str(args.prepared_root),
            output_root=str(root),
            ranking=dict(
                source=str(base["ranking_path"]),
                frozen_copy=str(frozen / "HEK_riboseq_profile_quality_rank_components.tsv"),
                sha256=base["ranking_sha256"],
                documented_repository_sha256=DOCUMENTED_RANKING_SHA256,
                matches_documented_repository_copy=(
                    base["ranking_sha256"] == DOCUMENTED_RANKING_SHA256
                ),
                component_count=base["rank_metadata"]["count"],
                component_columns=base["rank_metadata"]["columns"],
                rank_direction="1 = best",
                rank_universe=base["rank_universe"],
                retained_datasets=114,
            ),
            partition=dict(
                capacities=[29, 29, 28, 28],
                source_families=85,
                source_atomic=True,
                objective_before=search["original_objective"],
                objective_after=search["selected"]["objective"],
                selected_search_seed=search["selected"]["seed"],
                performance_inputs_used=False,
                validation_support_failures=0,
                nonblocking_test_support_rows=int(
                    ((base["support"].fold == "test") & ~base["support"].supported).sum()
                ),
            ),
            arms={
                "equal": "pi_d=1/N_panel",
                "ranked": "q_d=(R-r_d+1)/R; pi_d=q_d/sum_panel(q)",
                "reverse": "same q/pi multiset as ranked, assigned to dataset identities in reverse global-rank order",
            },
            training_seeds=list(args.training_seeds),
            number_of_tasks=len(tasks),
            seed42_core_tasks=sum(task["training_seed"] == 42 for task in tasks),
            tasks_sha256=object_sha256(tasks),
            numerical_execution=dict(
                trainer_precision="bf16-mixed",
                context_gru_precision="float32",
                biological_gru_precision="float32 (production implementation)",
                context_gru_tbptt_window=0,
                note="Identical across arms; full BPTT, no gradient truncation.",
            ),
            matching_contract=dict(
                same_panel_membership=True,
                same_dataset_order=True,
                same_transcript_split=True,
                same_training_only_reliability_reference=True,
                same_architecture_objective_optimizer_scheduler=True,
                same_trainable_initialization_within_panel_and_seed=True,
                sole_scientific_difference="fixed gamma-reference weights",
                checkpoint_selection="best_val_loss",
            ),
            interpretation=(
                "Ranked-minus-equal tests sensitivity to unequal QC-directed reference weights. "
                "Ranked-minus-reverse tests whether assignment direction matters at identical "
                "dataset-weight concentration. Dataset-level reversal need not preserve source-family "
                "reference mass, which is saved as a diagnostic. Neither contrast establishes "
                "biological accuracy."
            ),
            analysis_contract=dict(
                primary="full-CDS per-transcript cross-panel PCC on the frozen common test set",
                comparisons=["ranked-minus-equal", "ranked-minus-reverse", "reverse-minus-equal"],
                uncertainty="paired transcript-cluster bootstrap; all six panel pairs and arms carried together",
                optimization_variability="report each training seed before pooling",
            ),
            software=dict(python=platform.python_version()),
            frozen_file_sha256=frozen_files,
        )
        write_json(root / "experiment_manifest.json", manifest)
        return manifest


def run_task(args: argparse.Namespace, manifest: dict) -> int:
    if not args.authorize_training:
        raise ValueError("Training requires the explicit --authorize-training flag.")
    tasks = read_json(args.output_root / "tasks.json")
    if args.task_index < 0 or args.task_index >= len(tasks):
        raise ValueError(f"Task index must be in 0..{len(tasks) - 1}.")
    task = tasks[args.task_index]
    if args.dry_run:
        print(json.dumps(task, indent=2))
        return 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible:
        raise ValueError("No scheduler-assigned CUDA device is visible.")
    import torch

    if torch.cuda.device_count() != 1:
        raise ValueError(
            f"Each task requires exactly one visible GPU; found {torch.cuda.device_count()}."
        )
    task_root = Path(task["task_root"])
    with (task_root / ".training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Task is already running: {task['task_id']}") from exc
        write_json(
            task_root / "execution_environment.json",
            dict(
                task_id=task["task_id"],
                hostname=platform.node(),
                python=sys.executable,
                cuda_visible_devices=visible,
                visible_gpu=torch.cuda.get_device_name(0),
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                slurm_array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"),
                slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
                experiment_manifest_sha256=sha256(args.output_root / "experiment_manifest.json"),
                config_sha256=task["config_sha256"],
            ),
        )
        command = list(task["command"])
        command[0] = sys.executable
        return subprocess.run(
            command,
            cwd=args.prepared_root / "code_snapshot",
            check=False,
        ).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=ROOT
        / "results/four_panel_rank_balanced_qrank10_directionality_seed42/partition_audit_qrank10_v2/global_rank_balanced_partition_v2",
        help="Completed ten-component prepare-rank-balanced directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT
        / "results/four_panel_rank_balanced_qrank10_directionality_seed42/equal_ranked_reverse_qrank10",
    )
    parser.add_argument(
        "--training-seeds",
        type=parse_training_seeds,
        default=parse_training_seeds("42,43,44"),
        help="Comma-separated prespecified training seeds (default: 42,43,44).",
    )
    parser.add_argument(
        "--expected-ranking-sha256",
        default=None,
        help=(
            "Optional externally pinned checksum. By default the immutable checksum "
            "recorded by the completed base preparation is authoritative."
        ),
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--list-tasks", action="store_true")
    action.add_argument("--task-index", type=int)
    parser.add_argument("--authorize-training", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.prepared_root = args.prepared_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    try:
        if args.prepare_only:
            manifest = prepare(args)
            print(
                f"Prepared {manifest['number_of_tasks']} fresh trainings at {args.output_root}. "
                "No training launched."
            )
            return 0
        manifest = _preparation_identity(args.output_root)
        if list(args.training_seeds) != manifest["training_seeds"]:
            raise ValueError(
                "--training-seeds must match the already frozen experiment manifest."
            )
        if args.list_tasks:
            frame = pd.read_csv(args.output_root / "task_matrix.csv")
            print(frame.to_string(index=False))
            return 0
        return run_task(args, manifest)
    except (ValueError, KeyError, FileNotFoundError, RuntimeError, AssertionError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
