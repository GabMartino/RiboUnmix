#!/usr/bin/env python3
"""Create the article's uniform-reference real-data Figure 1.

Panel A reports transcript-level agreement between four source-family-disjoint
29/29/28/28-dataset models. Panel B reports stability between the three
pre-designated source-family-disjoint A/B subset pairs at N=2,5,10,20,40,
separately marks overlapping same-N stability at N=80, and summarizes agreement
across adjacent dataset sizes through the N=80 versus N=114 transition.
Only frozen best-validation-loss outputs with uniform gamma-reference weights
are accepted. No model inference or training is performed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import shlex
import sys
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Utils.publication_plot_style import LATEX_PAPER_RC
from Utils.reliability_references import transcript_id_hash
from analyses.analyze_real_exp8_stability import _task_directory


DEFAULT_PANEL_ROOT = ROOT / "results/my_panels_a100_b32_20260906_114323"
DEFAULT_STABILITY_ROOT = ROOT / "results/my_exp8_a100_b32_20260906_114340"
DEFAULT_OUTPUT = ROOT / "figures"
PANEL_NAMES = tuple(f"panel_{index:02d}" for index in range(1, 5))
PANEL_SIZES = (29, 29, 28, 28)
N_VALUES = (2, 5, 10, 20, 40)
PAIR_IDS = ("pair01", "pair02", "pair03")
MANUSCRIPT_SEED = 42
NEAR_CONSTANT_RELATIVE_STD = 1.0e-8
MEAN_ONE_TOLERANCE = 1.0e-4

FIGURE_RC = dict(LATEX_PAPER_RC)
FIGURE_RC.update(
    {
        "font.size": 8.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "legend.title_fontsize": 8.0,
    }
)


@dataclass(frozen=True)
class FrozenRun:
    """Validated metadata and artifacts for one frozen fitted model."""

    identifier: str
    directory: Path
    datasets: tuple[str, ...]
    source_families: tuple[str, ...]
    seed: int
    N: int
    compact_path: Path
    raw_path: Path
    gamma_manifest_path: Path
    checkpoint_manifest_path: Path
    reliability_manifest_path: Path
    pair_id: str = ""
    side: str = ""
    kind: str = ""


@dataclass(frozen=True)
class Profile:
    """One unmodified complete-CDS shared profile."""

    values: np.ndarray
    length: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, default=DEFAULT_PANEL_ROOT)
    parser.add_argument("--stability-root", type=Path, default=DEFAULT_STABILITY_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=MANUSCRIPT_SEED)
    parser.add_argument("--figure-width", type=float, default=7.15)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--mean-one-tolerance", type=float, default=MEAN_ONE_TOLERANCE)
    parser.add_argument(
        "--raw-batch-size",
        type=int,
        default=16,
        help="Rows per streaming raw-export audit batch (default: 16).",
    )
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_one(directory: Path, pattern: str) -> Path:
    candidates = sorted(directory.rglob(pattern))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one {pattern!r} below {directory}; found {len(candidates)}."
        )
    return candidates[0].resolve()


def require_uniform_pi(pi_by_dataset: Mapping[str, Any], datasets: Sequence[str]) -> None:
    if set(pi_by_dataset) != set(datasets):
        raise ValueError("Gamma-reference dataset membership does not match the fitted panel.")
    pi = np.asarray([pi_by_dataset[name] for name in datasets], dtype=np.float64)
    expected = np.full(len(datasets), 1.0 / len(datasets), dtype=np.float64)
    if np.any(pi <= 0.0) or not np.isclose(pi.sum(), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("Gamma-reference weights are not positive and normalized.")
    if not np.allclose(pi, expected, rtol=1e-10, atol=1e-12):
        raise ValueError("The run does not use uniform gamma-reference weights pi_d=1/N.")


def validate_gamma_manifest(path: Path, datasets: Sequence[str]) -> None:
    gamma = read_json(path)
    if gamma.get("centering_mode") != "fixed_reference" or gamma.get("weighting") != "equal":
        raise ValueError(f"{path} is not a fixed, equal-weight gamma reference.")
    names = list(map(str, gamma.get("reference_dataset_names", [])))
    if set(names) != set(datasets) or len(names) != len(datasets):
        raise ValueError(f"{path} has the wrong gamma-reference membership.")
    pi = dict(zip(names, gamma.get("reference_pi", []), strict=True))
    require_uniform_pi(pi, list(datasets))


def validate_reliability_manifest(
    directory: Path,
    declaration: Mapping[str, Any],
    declared_hash: str | None = None,
) -> Path:
    if declaration.get("symbol") != "w_dt":
        raise ValueError(f"{directory}: reliability declaration is not w_dt.")
    separate = declaration.get("separate_from_gamma_pi", declaration.get("separate_from_pi"))
    if separate is not True:
        raise ValueError(f"{directory}: w_dt is not declared separate from gamma pi.")
    fit_split = declaration.get("fitting_split", declaration.get("fit_split"))
    if fit_split not in {"training_only", "training"}:
        raise ValueError(f"{directory}: w_dt was not fitted on training transcripts only.")
    path = directory / "reliability_reference_manifest.json"
    reliability = read_json(path)
    if reliability.get("heldout_rows_used_for_fitting") != 0:
        raise ValueError(f"{path}: held-out rows contributed to w_dt fitting.")
    if reliability.get("reference_split") not in {"training", "training_only"}:
        raise ValueError(f"{path}: reliability reference is not training-only.")
    if declared_hash is not None and canonical_hash(reliability) != declared_hash:
        raise ValueError(f"{path}: canonical reliability-manifest hash mismatch.")
    return path.resolve()


def assert_disjoint(groups: Mapping[str, Iterable[str]], label: str) -> None:
    for (name_a, values_a), (name_b, values_b) in itertools.combinations(groups.items(), 2):
        overlap = set(values_a) & set(values_b)
        if overlap:
            raise ValueError(f"{label} overlap between {name_a} and {name_b}: {sorted(overlap)[:5]}")


def validate_split(
    *,
    path: Path,
    expected_test_ids: set[str],
    panel_style: bool,
) -> None:
    split = read_json(path)
    test_key = "test_ids" if panel_style else "common_test_ids"
    test_ids = set(map(str, split.get(test_key, [])))
    if test_ids != expected_test_ids:
        raise ValueError(f"{path}: held-out test transcript identities differ from the cohort.")
    train_ids = set(map(str, split.get("train_ids", [])))
    validation_ids = set(map(str, split.get("validation_ids", [])))
    if test_ids & train_ids or test_ids & validation_ids or train_ids & validation_ids:
        raise ValueError(f"{path}: train/validation/test transcript leakage detected.")
    hashes = split.get("fold_id_hashes", split.get("fold_hashes", {}))
    if hashes.get("test") != transcript_id_hash(test_ids):
        raise ValueError(f"{path}: test transcript hash mismatch.")


def validate_panel_experiment(root: Path, seed: int) -> tuple[list[FrozenRun], list[str], dict[str, Any]]:
    panel_manifest_path = root / "panel_manifest.json"
    panel_manifest = read_json(panel_manifest_path)
    common_path = root / "common_split_manifest.json"
    common = read_json(common_path)
    if panel_manifest.get("experiment_name") != "independent_dataset_panel_convergence":
        raise ValueError("Panel A root is not the independent-panel convergence experiment.")
    if int(panel_manifest.get("random_seed", -1)) != seed or int(common.get("random_seed", -1)) != seed:
        raise ValueError("Panel A does not use the requested manuscript seed consistently.")
    panels = panel_manifest.get("panels", {})
    sizes = tuple(len(panels.get(name, [])) for name in PANEL_NAMES)
    if sizes != PANEL_SIZES or tuple(panel_manifest.get("panel_sizes", {}).get(name) for name in PANEL_NAMES) != PANEL_SIZES:
        raise ValueError(f"Panel A sizes are {sizes}, not {PANEL_SIZES}.")
    sources = panel_manifest.get("panel_source_families", {})
    assert_disjoint({name: panels[name] for name in PANEL_NAMES}, "Dataset identity")
    assert_disjoint({name: sources[name] for name in PANEL_NAMES}, "Source-family identity")
    test_ids = list(map(str, common.get("common_test_ids", [])))
    if not test_ids or len(set(test_ids)) != len(test_ids):
        raise ValueError("Panel A common test cohort is empty or duplicated.")
    test_set = set(test_ids)
    test_hash = transcript_id_hash(test_set)

    runs: list[FrozenRun] = []
    for panel, expected_N in zip(PANEL_NAMES, PANEL_SIZES, strict=True):
        directory = root / panel
        run_manifest = read_json(directory / "run_manifest.json")
        selected = tuple(map(str, run_manifest.get("selected_datasets", [])))
        selected_sources = tuple(map(str, run_manifest.get("selected_source_families", [])))
        if run_manifest.get("panel_name") != panel or selected != tuple(panels[panel]):
            raise ValueError(f"{panel}: run membership differs from panel_manifest.json.")
        if selected_sources != tuple(sources[panel]):
            raise ValueError(f"{panel}: source-family membership differs from panel_manifest.json.")
        if int(run_manifest.get("random_seed", -1)) != seed:
            raise ValueError(f"{panel}: training seed differs from manuscript seed {seed}.")
        reference = run_manifest.get("fixed_gamma_reference", {})
        if reference.get("weighting") != "equal":
            raise ValueError(f"{panel}: run manifest does not declare equal gamma weights.")
        require_uniform_pi(reference.get("pi", {}), selected)
        reliability_path = validate_reliability_manifest(
            directory, run_manifest.get("reliability_weight", {})
        )
        validate_split(path=directory / "split_manifest.json", expected_test_ids=test_set, panel_style=True)

        scientific_path = directory / "scientific_checkpoint_manifest.json"
        scientific = read_json(scientific_path)
        if (
            scientific.get("panel_name") != panel
            or scientific.get("checkpoint_variant") != "best_val_loss"
            or scientific.get("prediction_split") != "common_test"
            or scientific.get("test_transcript_id_hash") != test_hash
            or tuple(scientific.get("selected_datasets", [])) != selected
        ):
            raise ValueError(f"{scientific_path}: scientific checkpoint contract mismatch.")
        compact = require_one(directory / "predictions", "common_test_L_profiles.parquet")
        raw = require_one(directory / "predictions", "predictions_main_test_best_val_loss_*.parquet")
        if Path(str(scientific["prediction_path"])).name != raw.name:
            raise ValueError(f"{scientific_path}: raw prediction identity mismatch.")
        gamma_path = require_one(directory / "predictions", "gamma_reference_manifest.json")
        validate_gamma_manifest(gamma_path, selected)
        runs.append(
            FrozenRun(
                identifier=panel, directory=directory.resolve(), datasets=selected,
                source_families=selected_sources, seed=seed, N=expected_N,
                compact_path=compact, raw_path=raw, gamma_manifest_path=gamma_path,
                checkpoint_manifest_path=scientific_path.resolve(),
                reliability_manifest_path=reliability_path,
                kind="source_disjoint_panel",
            )
        )
    provenance = {
        "experiment": "four source-family-disjoint panels",
        "manifest": str(panel_manifest_path.resolve()),
        "manifest_sha256": file_sha256(panel_manifest_path),
        "common_test_manifest": str(common_path.resolve()),
        "common_test_count": len(test_ids),
        "common_test_hash": test_hash,
        "panel_sizes": list(PANEL_SIZES),
        "training_seed": seed,
    }
    return runs, test_ids, provenance


def validate_stability_experiment(
    root: Path, seed: int
) -> tuple[list[FrozenRun], list[FrozenRun], list[str], dict[str, Any]]:
    experiment_path = root / "experiment_manifest.json"
    experiment = read_json(experiment_path)
    if (
        experiment.get("experiment_name") != "real_exp8_L_stability"
        or experiment.get("sampling_mode") != "quality_matched"
        or experiment.get("gamma_centering_mode") != "fixed_reference"
        or experiment.get("gamma_pi_strategy") != "uniform within each selected subset"
    ):
        raise ValueError("Panel B root is not the original quality-matched equal-reference Exp8.")
    if list(map(int, experiment.get("training_seeds", []))) != [seed]:
        raise ValueError("Panel B does not contain exactly the requested manuscript seed.")
    common_path = root / "common_test_manifest.json"
    common = read_json(common_path)
    test_ids = list(map(str, common.get("common_test_ids", [])))
    if not test_ids or len(set(test_ids)) != len(test_ids):
        raise ValueError("Panel B common test cohort is empty or duplicated.")
    test_set = set(test_ids)
    test_hash = transcript_id_hash(test_set)
    if common.get("transcript_id_hash") != test_hash:
        raise ValueError("Panel B common test transcript hash mismatch.")

    selected_tasks = [
        task for task in experiment.get("tasks", [])
        if task.get("kind") == "designated_disjoint_pair"
        and int(task.get("N", -1)) in N_VALUES
        and int(task.get("training_seed", -1)) == seed
    ]
    if len(selected_tasks) != len(N_VALUES) * len(PAIR_IDS) * 2:
        raise ValueError("Panel B does not have all 30 requested designated A/B tasks.")
    runs: list[FrozenRun] = []
    for N in N_VALUES:
        tasks_at_N = [task for task in selected_tasks if int(task["N"]) == N]
        for pair_id in PAIR_IDS:
            pair_tasks = [task for task in tasks_at_N if str(task.get("pair_id")) == pair_id]
            sides = {str(task.get("side")): task for task in pair_tasks}
            if set(sides) != {"A", "B"} or len(pair_tasks) != 2:
                raise ValueError(f"Panel B is missing the designated {pair_id} A/B pair at N={N}.")
            if set(sides["A"]["datasets"]) & set(sides["B"]["datasets"]):
                raise ValueError(f"Panel B {pair_id} at N={N} overlaps in datasets.")
            if set(sides["A"]["source_families"]) & set(sides["B"]["source_families"]):
                raise ValueError(f"Panel B {pair_id} at N={N} overlaps in source families.")
            for side in ("A", "B"):
                task = sides[side]
                directory = _task_directory(root, task)
                subset = read_json(directory / "subset_manifest.json")
                datasets = tuple(map(str, task["datasets"]))
                source_families = tuple(map(str, task["source_families"]))
                if (
                    subset.get("run_id") != task["run_id"]
                    or int(subset.get("N", -1)) != N
                    or int(subset.get("training_seed", -1)) != seed
                    or tuple(subset.get("datasets", [])) != datasets
                    or tuple(subset.get("source_families", [])) != source_families
                    or subset.get("kind") != "designated_disjoint_pair"
                    or subset.get("pair_id") != pair_id
                    or subset.get("side") != side
                    or subset.get("common_test_hash") != test_hash
                ):
                    raise ValueError(f"{directory}: task and subset manifests disagree.")
                reference = subset.get("fixed_gamma_reference", {})
                if reference.get("weighting") != "equal":
                    raise ValueError(f"{directory}: subset manifest does not declare equal weights.")
                require_uniform_pi(reference.get("pi", {}), datasets)
                reliability_path = validate_reliability_manifest(
                    directory,
                    subset.get("reliability_weight", {}),
                    str(subset.get("reliability_reference_hash")),
                )
                validate_split(path=directory / "split_manifest.json", expected_test_ids=test_set, panel_style=False)

                runtime_path = require_one(directory / "predictions", "prediction_checkpoint_manifest.json")
                runtime_container = read_json(runtime_path)
                if set(runtime_container) != {"best_val_loss"}:
                    raise ValueError(f"{runtime_path}: expected only the best_val_loss prediction.")
                runtime = runtime_container["best_val_loss"]
                if (
                    runtime.get("sequence_only_shared_profile_prediction") is not True
                    or runtime.get("split_name") != "test"
                    or int(runtime.get("transcript_count", -1)) != len(test_ids)
                    or runtime.get("transcript_id_hash") != test_hash
                ):
                    raise ValueError(f"{runtime_path}: frozen prediction contract mismatch.")
                # Runs completed by the original launcher predate selected_checkpoint.json.
                # In those runs the immutable runtime manifest is the authoritative record
                # that the export came from the best-validation-loss checkpoint. Resumed
                # runs additionally carry a selection manifest, which must agree exactly.
                selected_path = directory / "selected_checkpoint.json"
                checkpoint_manifest_path = runtime_path
                if selected_path.is_file():
                    selected = read_json(selected_path)
                    if (
                        selected.get("run_id") != task["run_id"]
                        or int(selected.get("N", -1)) != N
                        or selected.get("checkpoint_variant") != "best_val_loss"
                        or selected.get("test_transcript_id_hash") != test_hash
                        or runtime.get("checkpoint_path") != selected.get("checkpoint_path")
                    ):
                        raise ValueError(f"{selected_path}: checkpoint selection contract mismatch.")
                    checkpoint_manifest_path = selected_path
                compact = require_one(directory / "predictions", "common_test_L_profiles.parquet")
                raw = require_one(directory / "predictions", "predictions_main_test_best_val_loss_*.parquet")
                if Path(str(runtime["shared_profile_output_path"])).name != compact.name:
                    raise ValueError(f"{runtime_path}: compact profile identity mismatch.")
                if Path(str(runtime["output_path"])).name != raw.name:
                    raise ValueError(f"{runtime_path}: raw prediction identity mismatch.")
                gamma_path = require_one(directory / "predictions", "gamma_reference_manifest.json")
                validate_gamma_manifest(gamma_path, datasets)
                runs.append(
                    FrozenRun(
                        identifier=str(task["run_id"]), directory=directory.resolve(),
                        datasets=datasets, source_families=source_families, seed=seed, N=N,
                        compact_path=compact, raw_path=raw, gamma_manifest_path=gamma_path,
                        checkpoint_manifest_path=checkpoint_manifest_path.resolve(),
                        reliability_manifest_path=reliability_path,
                        pair_id=pair_id, side=side, kind="designated_disjoint_pair",
                    )
                )
    auxiliary_runs = validate_auxiliary_stability_runs(
        root=root,
        tasks=experiment.get("tasks", []),
        expected_test_ids=test_ids,
        test_hash=test_hash,
        seed=seed,
    )
    provenance = {
        "experiment": "quality-matched same-size and adjacent-size stability",
        "manifest": str(experiment_path.resolve()),
        "manifest_sha256": file_sha256(experiment_path),
        "common_test_manifest": str(common_path.resolve()),
        "common_test_count": len(test_ids),
        "common_test_hash": test_hash,
        "N_values": list(N_VALUES),
        "designated_pairs_per_N": len(PAIR_IDS),
        "training_seed": seed,
        "overlapping_same_N": 80,
        "adjacent_size_transitions": [[2, 5], [5, 10], [10, 20], [20, 40], [40, 80], [80, 114]],
        "full_collection_reference_N": 114,
    }
    return runs, auxiliary_runs, test_ids, provenance


def validate_auxiliary_stability_runs(
    *,
    root: Path,
    tasks: Sequence[Mapping[str, Any]],
    expected_test_ids: Sequence[str],
    test_hash: str,
    seed: int,
) -> list[FrozenRun]:
    """Validate the three N=80 subsets and the single N=114 reference model."""
    selected_tasks = [
        task
        for task in tasks
        if task.get("kind") in {"large_N_subset", "full_collection"}
        and int(task.get("training_seed", -1)) == seed
    ]
    kinds = [str(task.get("kind")) for task in selected_tasks]
    if kinds.count("large_N_subset") != 3 or kinds.count("full_collection") != 1:
        raise ValueError("Expected exactly three N=80 subsets and one N=114 full model.")
    expected_set = set(map(str, expected_test_ids))
    runs: list[FrozenRun] = []
    for task in selected_tasks:
        kind = str(task["kind"])
        N = int(task["N"])
        if (kind == "large_N_subset" and N != 80) or (kind == "full_collection" and N != 114):
            raise ValueError(f"Unexpected auxiliary task design: {task['run_id']} at N={N}.")
        directory = _task_directory(root, task)
        subset = read_json(directory / "subset_manifest.json")
        datasets = tuple(map(str, task["datasets"]))
        source_families = tuple(map(str, task["source_families"]))
        if (
            subset.get("run_id") != task["run_id"]
            or int(subset.get("N", -1)) != N
            or int(subset.get("training_seed", -1)) != seed
            or tuple(subset.get("datasets", [])) != datasets
            or tuple(subset.get("source_families", [])) != source_families
            or subset.get("kind") != kind
            or subset.get("common_test_hash") != test_hash
        ):
            raise ValueError(f"{directory}: auxiliary task and subset manifests disagree.")
        reference = subset.get("fixed_gamma_reference", {})
        if reference.get("weighting") != "equal":
            raise ValueError(f"{directory}: subset manifest does not declare equal weights.")
        require_uniform_pi(reference.get("pi", {}), datasets)
        reliability_path = validate_reliability_manifest(
            directory,
            subset.get("reliability_weight", {}),
            str(subset.get("reliability_reference_hash")),
        )
        validate_split(
            path=directory / "split_manifest.json",
            expected_test_ids=expected_set,
            panel_style=False,
        )
        runtime_path = require_one(directory / "predictions", "prediction_checkpoint_manifest.json")
        runtime_container = read_json(runtime_path)
        if set(runtime_container) != {"best_val_loss"}:
            raise ValueError(f"{runtime_path}: expected only the best_val_loss prediction.")
        runtime = runtime_container["best_val_loss"]
        if (
            runtime.get("sequence_only_shared_profile_prediction") is not True
            or runtime.get("split_name") != "test"
            or int(runtime.get("transcript_count", -1)) != len(expected_test_ids)
            or runtime.get("transcript_id_hash") != test_hash
        ):
            raise ValueError(f"{runtime_path}: frozen prediction contract mismatch.")
        selected_path = directory / "selected_checkpoint.json"
        checkpoint_manifest_path = runtime_path
        if selected_path.is_file():
            selected = read_json(selected_path)
            if (
                selected.get("run_id") != task["run_id"]
                or int(selected.get("N", -1)) != N
                or selected.get("checkpoint_variant") != "best_val_loss"
                or selected.get("test_transcript_id_hash") != test_hash
                or runtime.get("checkpoint_path") != selected.get("checkpoint_path")
            ):
                raise ValueError(f"{selected_path}: checkpoint selection contract mismatch.")
            checkpoint_manifest_path = selected_path
        compact = require_one(directory / "predictions", "common_test_L_profiles.parquet")
        raw = require_one(directory / "predictions", "predictions_main_test_best_val_loss_*.parquet")
        if Path(str(runtime["shared_profile_output_path"])).name != compact.name:
            raise ValueError(f"{runtime_path}: compact profile identity mismatch.")
        if Path(str(runtime["output_path"])).name != raw.name:
            raise ValueError(f"{runtime_path}: raw prediction identity mismatch.")
        gamma_path = require_one(directory / "predictions", "gamma_reference_manifest.json")
        validate_gamma_manifest(gamma_path, datasets)
        runs.append(
            FrozenRun(
                identifier=str(task["run_id"]),
                directory=directory.resolve(),
                datasets=datasets,
                source_families=source_families,
                seed=seed,
                N=N,
                compact_path=compact,
                raw_path=raw,
                gamma_manifest_path=gamma_path,
                checkpoint_manifest_path=checkpoint_manifest_path.resolve(),
                reliability_manifest_path=reliability_path,
                kind=kind,
            )
        )
    return sorted(runs, key=lambda run: (run.N, run.identifier))


def load_compact_profiles(
    run: FrozenRun,
    expected_ids: Sequence[str],
    mean_one_tolerance: float,
) -> tuple[dict[str, Profile], dict[str, Any]]:
    schema = set(pq.read_schema(run.compact_path).names)
    required = {"transcript_id", "transcript_length", "L_t"}
    if not required <= schema:
        raise KeyError(f"{run.compact_path} lacks {sorted(required - schema)}.")
    columns = sorted(required | ({"valid_position_mask"} if "valid_position_mask" in schema else set()))
    frame = pd.read_parquet(run.compact_path, columns=columns)
    frame["transcript_id"] = frame["transcript_id"].astype(str)
    if frame["transcript_id"].duplicated().any():
        raise ValueError(f"{run.identifier}: compact export contains duplicate transcripts.")
    expected = set(map(str, expected_ids))
    observed = set(frame["transcript_id"])
    missing, extra = expected - observed, observed - expected
    profiles: dict[str, Profile] = {}
    maximum_deviation = 0.0
    for row in frame.itertuples(index=False):
        values = np.asarray(row.L_t, dtype=np.float64)
        length = int(row.transcript_length)
        if "valid_position_mask" in columns:
            mask = np.asarray(row.valid_position_mask, dtype=bool)
            if mask.shape != values.shape or int(mask.sum()) != length or not np.all(mask):
                raise ValueError(f"{run.identifier}/{row.transcript_id}: compact mask is not the full CDS.")
            values = values[mask]
        if values.ndim != 1 or len(values) != length or length < 2:
            raise ValueError(f"{run.identifier}/{row.transcript_id}: invalid compact profile length.")
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError(f"{run.identifier}/{row.transcript_id}: invalid compact profile values.")
        deviation = abs(float(values.mean()) - 1.0)
        if deviation > mean_one_tolerance:
            raise ValueError(
                f"{run.identifier}/{row.transcript_id}: mean-one deviation {deviation:.3g} exceeds tolerance."
            )
        maximum_deviation = max(maximum_deviation, deviation)
        profiles[str(row.transcript_id)] = Profile(values=values, length=length)
    audit = {
        "run_id": run.identifier,
        "compact_path": str(run.compact_path),
        "compact_sha256": file_sha256(run.compact_path),
        "expected_transcripts": len(expected),
        "observed_transcripts": len(observed),
        "missing_transcripts": len(missing),
        "extra_transcripts": len(extra),
        "maximum_abs_mean_one_deviation": maximum_deviation,
    }
    return profiles, audit


def audit_raw_export(
    run: FrozenRun,
    compact: Mapping[str, Profile],
    expected_ids: Sequence[str],
    batch_size: int,
    expect_repeated_rows: bool,
) -> dict[str, Any]:
    """Stream raw rows and verify shared profiles before any deduplication."""
    required = {"transcript_id", "dataset_id", "length", "mask", "L_bio"}
    available = set(pq.read_schema(run.raw_path).names)
    if not required <= available:
        raise KeyError(f"{run.raw_path} lacks raw shared-profile columns {sorted(required - available)}.")
    expected = set(map(str, expected_ids))
    first_values: dict[str, np.ndarray] = {}
    first_masks: dict[str, np.ndarray] = {}
    row_counts: dict[str, int] = {}
    dataset_ids: dict[str, set[int]] = {}
    repeated_rows_checked = 0
    with pq.ParquetFile(run.raw_path) as reader:
        raw_rows = int(reader.metadata.num_rows)
        for batch in reader.iter_batches(
            batch_size=batch_size,
            columns=["transcript_id", "dataset_id", "length", "mask", "L_bio"],
            use_threads=False,
        ):
            for row in batch.to_pylist():
                transcript_id = str(row["transcript_id"])
                if transcript_id not in expected:
                    raise ValueError(f"{run.identifier}: unexpected raw transcript {transcript_id}.")
                mask = np.asarray(row["mask"], dtype=bool)
                full = np.asarray(row["L_bio"], dtype=np.float64)
                length = int(row["length"])
                if mask.ndim != 1 or full.shape != mask.shape or int(mask.sum()) != length:
                    raise ValueError(f"{run.identifier}/{transcript_id}: invalid raw profile alignment.")
                values = full[mask]
                if not np.isfinite(values).all() or np.any(values <= 0.0):
                    raise ValueError(f"{run.identifier}/{transcript_id}: invalid raw shared profile.")
                if transcript_id in first_values:
                    repeated_rows_checked += 1
                    if not np.array_equal(first_masks[transcript_id], mask) or not np.array_equal(
                        first_values[transcript_id], values
                    ):
                        raise ValueError(
                            f"{run.identifier}/{transcript_id}: L_t differs across repeated dataset rows."
                        )
                else:
                    first_values[transcript_id] = values.copy()
                    first_masks[transcript_id] = mask.copy()
                row_counts[transcript_id] = row_counts.get(transcript_id, 0) + 1
                dataset_ids.setdefault(transcript_id, set()).add(int(row["dataset_id"]))
                compact_row = compact.get(transcript_id)
                if compact_row is None or compact_row.length != length or not np.array_equal(
                    values.astype(np.float32), compact_row.values.astype(np.float32)
                ):
                    raise ValueError(
                        f"{run.identifier}/{transcript_id}: compact L_t is not the exact float32 materialization of raw L_bio."
                    )
            del batch
    observed = set(first_values)
    missing, extra = expected - observed, observed - expected
    counts = np.asarray(list(row_counts.values()), dtype=int)
    if expect_repeated_rows and (not len(counts) or int(counts.min()) < 2):
        raise ValueError(f"{run.identifier}: expected repeated dataset rows for every test transcript.")
    if not expect_repeated_rows and (not len(counts) or int(counts.min()) != 1 or int(counts.max()) != 1):
        raise ValueError(f"{run.identifier}: sequence-only export unexpectedly contains repeated rows.")
    return {
        "run_id": run.identifier,
        "raw_path": str(run.raw_path),
        "raw_size_bytes": run.raw_path.stat().st_size,
        "raw_rows": raw_rows,
        "unique_transcripts": len(observed),
        "missing_transcripts": len(missing),
        "extra_transcripts": len(extra),
        "minimum_rows_per_transcript": int(counts.min()) if len(counts) else 0,
        "maximum_rows_per_transcript": int(counts.max()) if len(counts) else 0,
        "minimum_distinct_datasets_per_transcript": min(map(len, dataset_ids.values())) if dataset_ids else 0,
        "maximum_distinct_datasets_per_transcript": max(map(len, dataset_ids.values())) if dataset_ids else 0,
        "repeated_rows_checked": repeated_rows_checked,
        "all_repeated_profiles_bitwise_identical": True,
        "compact_matches_raw_float32": True,
        "raw_export_mode": "repeated_dataset_rows" if expect_repeated_rows else "sequence_only_one_row_per_transcript",
    }


def profile_agreement(
    left: Profile | None,
    right: Profile | None,
) -> dict[str, Any]:
    if left is None or right is None:
        missing = "both" if left is None and right is None else ("left" if left is None else "right")
        return {"PCC": np.nan, "status": f"missing_{missing}", "transcript_length": np.nan}
    if left.length != right.length or left.values.shape != right.values.shape:
        return {"PCC": np.nan, "status": "misaligned_length", "transcript_length": left.length}
    if not np.isfinite(left.values).all() or not np.isfinite(right.values).all():
        return {"PCC": np.nan, "status": "nonfinite_profile", "transcript_length": left.length}
    left_std = float(np.std(left.values, dtype=np.float64))
    right_std = float(np.std(right.values, dtype=np.float64))
    left_constant = float(np.ptp(left.values)) == 0.0
    right_constant = float(np.ptp(right.values)) == 0.0
    left_near = left_std <= NEAR_CONSTANT_RELATIVE_STD * max(1.0, abs(float(left.values.mean())))
    right_near = right_std <= NEAR_CONSTANT_RELATIVE_STD * max(1.0, abs(float(right.values.mean())))
    if left_constant or right_constant:
        side = "both" if left_constant and right_constant else ("left" if left_constant else "right")
        return {"PCC": np.nan, "status": f"constant_{side}", "transcript_length": left.length}
    if left_near or right_near:
        side = "both" if left_near and right_near else ("left" if left_near else "right")
        return {"PCC": np.nan, "status": f"near_constant_{side}", "transcript_length": left.length}
    left_centered = left.values - left.values.mean()
    right_centered = right.values - right.values.mean()
    denominator = math.sqrt(float(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered)))
    return {
        "PCC": float(np.dot(left_centered, right_centered) / denominator),
        "status": "valid",
        "transcript_length": left.length,
    }


def calculate_panel_a(
    runs: Sequence[FrozenRun],
    profiles: Mapping[str, Mapping[str, Profile]],
    transcript_ids: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for left, right in itertools.combinations(runs, 2):
        pair = f"P{int(left.identifier[-2:])}\N{EN DASH}P{int(right.identifier[-2:])}"
        for transcript_id in transcript_ids:
            rows.append(
                {
                    "transcript_id": transcript_id,
                    "panel_a": left.identifier,
                    "panel_b": right.identifier,
                    "panel_pair": pair,
                    "profile_domain": "full_CDS",
                    **profile_agreement(
                        profiles[left.identifier].get(transcript_id),
                        profiles[right.identifier].get(transcript_id),
                    ),
                }
            )
    per_transcript = pd.DataFrame(rows)
    summaries = []
    for pair, group in per_transcript.groupby("panel_pair", sort=False):
        valid = group.loc[group.status == "valid", "PCC"].to_numpy(dtype=np.float64)
        quantiles = np.quantile(valid, [0.05, 0.25, 0.50, 0.75, 0.95])
        summaries.append(
            {
                "panel_pair": pair,
                "cohort_transcripts": len(transcript_ids),
                "valid_PCC": len(valid),
                "undefined_PCC": len(group) - len(valid),
                "mean_PCC": float(valid.mean()),
                "p05_PCC": float(quantiles[0]),
                "p25_PCC": float(quantiles[1]),
                "median_PCC": float(quantiles[2]),
                "p75_PCC": float(quantiles[3]),
                "p95_PCC": float(quantiles[4]),
            }
        )
    return per_transcript, pd.DataFrame(summaries)


def calculate_panel_b(
    runs: Sequence[FrozenRun],
    transcript_ids: Sequence[str],
    mean_one_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    rows, compact_audits = [], []
    lookup = {(run.N, run.pair_id, run.side): run for run in runs}
    for N in N_VALUES:
        for pair_id in PAIR_IDS:
            left, right = lookup[(N, pair_id, "A")], lookup[(N, pair_id, "B")]
            left_profiles, left_audit = load_compact_profiles(left, transcript_ids, mean_one_tolerance)
            right_profiles, right_audit = load_compact_profiles(right, transcript_ids, mean_one_tolerance)
            compact_audits.extend([left_audit, right_audit])
            for transcript_id in transcript_ids:
                rows.append(
                    {
                        "transcript_id": transcript_id,
                        "N": N,
                        "pair_id": pair_id,
                        "run_a": left.identifier,
                        "run_b": right.identifier,
                        "profile_domain": "full_CDS",
                        **profile_agreement(
                            left_profiles.get(transcript_id), right_profiles.get(transcript_id)
                        ),
                    }
                )
            del left_profiles, right_profiles
    per_transcript = pd.DataFrame(rows)
    pair_rows = []
    for (N, pair_id, run_a, run_b), group in per_transcript.groupby(
        ["N", "pair_id", "run_a", "run_b"], sort=True
    ):
        valid = group.loc[group.status == "valid", "PCC"].to_numpy(dtype=np.float64)
        pair_rows.append(
            {
                "N": int(N), "pair_id": pair_id, "run_a": run_a, "run_b": run_b,
                "cohort_transcripts": len(transcript_ids), "valid_PCC": len(valid),
                "undefined_PCC": len(group) - len(valid),
                "R_N_p_mean_transcript_PCC": float(valid.mean()),
                "median_transcript_PCC": float(np.median(valid)),
            }
        )
    pair_summary = pd.DataFrame(pair_rows)
    N_rows = []
    for N, group in pair_summary.groupby("N", sort=True):
        estimates = group["R_N_p_mean_transcript_PCC"].to_numpy(dtype=np.float64)
        if len(estimates) != len(PAIR_IDS):
            raise RuntimeError(f"N={N}: expected three designated pair estimates, found {len(estimates)}.")
        N_rows.append(
            {
                "N": int(N),
                "number_of_designated_pairs": len(estimates),
                "R_N_mean_over_pair_means": float(estimates.mean()),
                "minimum_pair_mean_PCC": float(estimates.min()),
                "maximum_pair_mean_PCC": float(estimates.max()),
            }
        )
    return per_transcript, pair_summary, pd.DataFrame(N_rows), compact_audits


def calculate_overlapping_N80_stability(
    runs: Sequence[FrozenRun],
    transcript_ids: Sequence[str],
    mean_one_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Describe same-N agreement among the three overlapping N=80 subsets."""
    if len(runs) != 3 or any(run.N != 80 for run in runs):
        raise ValueError("N=80 stability requires exactly the three large-N subset runs.")
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    profiles_by_run: dict[str, dict[str, Profile]] = {}
    for run in sorted(runs, key=lambda item: item.identifier):
        profiles, audit = load_compact_profiles(run, transcript_ids, mean_one_tolerance)
        profiles_by_run[run.identifier] = profiles
        audits.append(audit)
    overlap_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for left, right in itertools.combinations(sorted(runs, key=lambda item: item.identifier), 2):
        datasets_left, datasets_right = set(left.datasets), set(right.datasets)
        sources_left, sources_right = set(left.source_families), set(right.source_families)
        overlap_by_pair[(left.identifier, right.identifier)] = {
            "dataset_intersection": len(datasets_left & datasets_right),
            "dataset_jaccard": len(datasets_left & datasets_right) / len(datasets_left | datasets_right),
            "source_family_intersection": len(sources_left & sources_right),
            "source_family_jaccard": len(sources_left & sources_right) / len(sources_left | sources_right),
        }
        for transcript_id in transcript_ids:
            rows.append(
                {
                    "transcript_id": transcript_id,
                    "N": 80,
                    "run_a": left.identifier,
                    "run_b": right.identifier,
                    "comparison_kind": "overlapping_same_N",
                    "profile_domain": "full_CDS",
                    **profile_agreement(
                        profiles_by_run[left.identifier].get(transcript_id),
                        profiles_by_run[right.identifier].get(transcript_id),
                    ),
                }
            )
    per_transcript = pd.DataFrame(rows)
    pair_rows: list[dict[str, Any]] = []
    for (run_a, run_b), group in per_transcript.groupby(["run_a", "run_b"], sort=True):
        valid = group.loc[group.status == "valid", "PCC"].to_numpy(dtype=np.float64)
        pair_rows.append(
            {
                "N": 80,
                "run_a": run_a,
                "run_b": run_b,
                "cohort_transcripts": len(transcript_ids),
                "valid_PCC": len(valid),
                "undefined_PCC": len(group) - len(valid),
                "mean_transcript_PCC": float(valid.mean()),
                "median_transcript_PCC": float(np.median(valid)),
                **overlap_by_pair[(run_a, run_b)],
            }
        )
    pair_summary = pd.DataFrame(pair_rows)
    estimates = pair_summary.mean_transcript_PCC.to_numpy(dtype=np.float64)
    summary = pd.DataFrame(
        [
            {
                "N": 80,
                "number_of_overlapping_subset_pairs": len(estimates),
                "mean_over_pair_means": float(estimates.mean()),
                "minimum_pair_mean_PCC": float(estimates.min()),
                "maximum_pair_mean_PCC": float(estimates.max()),
            }
        ]
    )
    return per_transcript, pair_summary, summary, audits


def calculate_adjacent_size_agreement(
    runs: Sequence[FrozenRun],
    transcript_ids: Sequence[str],
    mean_one_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Average all available run-pair agreements across adjacent dataset sizes.

    Runs are not nested. The resulting curve is therefore a descriptive
    cross-size agreement summary, not a within-construction trajectory.
    """
    transitions = tuple(zip((2, 5, 10, 20, 40, 80), (5, 10, 20, 40, 80, 114)))
    expected_counts = {2: 6, 5: 6, 10: 6, 20: 6, 40: 6, 80: 3, 114: 1}
    groups = {
        N: sorted((run for run in runs if run.N == N), key=lambda run: run.identifier)
        for N in expected_counts
    }
    for N, expected in expected_counts.items():
        if len(groups[N]) != expected:
            raise ValueError(f"N={N}: expected {expected} runs, found {len(groups[N])}.")

    rows: list[dict[str, Any]] = []
    overlap_by_pair: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    audits: list[dict[str, Any]] = []

    def load_group(group: Sequence[FrozenRun]) -> dict[str, dict[str, Profile]]:
        loaded: dict[str, dict[str, Profile]] = {}
        for run in group:
            profiles, audit = load_compact_profiles(run, transcript_ids, mean_one_tolerance)
            loaded[run.identifier] = profiles
            audits.append(audit)
        return loaded

    left_N = transitions[0][0]
    left_profiles = load_group(groups[left_N])
    for N_from, N_to in transitions:
        if N_from != left_N:
            raise RuntimeError("Adjacent-size transition order is inconsistent.")
        right_profiles = load_group(groups[N_to])
        for left, right in itertools.product(groups[N_from], groups[N_to]):
            datasets_left, datasets_right = set(left.datasets), set(right.datasets)
            sources_left, sources_right = set(left.source_families), set(right.source_families)
            key = (N_from, N_to, left.identifier, right.identifier)
            overlap_by_pair[key] = {
                "dataset_intersection": len(datasets_left & datasets_right),
                "dataset_jaccard": len(datasets_left & datasets_right) / len(datasets_left | datasets_right),
                "source_family_intersection": len(sources_left & sources_right),
                "source_family_jaccard": len(sources_left & sources_right) / len(sources_left | sources_right),
            }
            for transcript_id in transcript_ids:
                rows.append(
                    {
                        "transcript_id": transcript_id,
                        "N_from": N_from,
                        "N_to": N_to,
                        "transition": f"{N_from}\N{EN DASH}{N_to}",
                        "run_from": left.identifier,
                        "run_to": right.identifier,
                        "profile_domain": "full_CDS",
                        **profile_agreement(
                            left_profiles[left.identifier].get(transcript_id),
                            right_profiles[right.identifier].get(transcript_id),
                        ),
                    }
                )
        del left_profiles
        left_profiles = right_profiles
        left_N = N_to
    del left_profiles
    per_transcript = pd.DataFrame(rows)

    pair_rows: list[dict[str, Any]] = []
    group_columns = ["N_from", "N_to", "transition", "run_from", "run_to"]
    for keys, group in per_transcript.groupby(group_columns, sort=True):
        N_from, N_to, transition, run_from, run_to = keys
        valid = group.loc[group.status == "valid", "PCC"].to_numpy(dtype=np.float64)
        pair_rows.append(
            {
                "N_from": int(N_from),
                "N_to": int(N_to),
                "transition": transition,
                "run_from": run_from,
                "run_to": run_to,
                "cohort_transcripts": len(transcript_ids),
                "valid_PCC": len(valid),
                "undefined_PCC": len(group) - len(valid),
                "mean_transcript_PCC": float(valid.mean()),
                "median_transcript_PCC": float(np.median(valid)),
                **overlap_by_pair[(int(N_from), int(N_to), run_from, run_to)],
            }
        )
    pair_summary = pd.DataFrame(pair_rows)

    transition_rows: list[dict[str, Any]] = []
    for (N_from, N_to, transition), group in pair_summary.groupby(
        ["N_from", "N_to", "transition"], sort=True
    ):
        estimates = group.mean_transcript_PCC.to_numpy(dtype=np.float64)
        quantiles = np.quantile(estimates, [0.25, 0.75])
        transition_rows.append(
            {
                "N_from": int(N_from),
                "N_to": int(N_to),
                "transition": transition,
                "plot_x_geometric_midpoint": math.sqrt(float(N_from) * float(N_to)),
                "plot_x_larger_endpoint": int(N_to),
                "number_of_model_pairs": len(estimates),
                "mean_over_model_pair_means": float(estimates.mean()),
                "minimum_model_pair_mean_PCC": float(estimates.min()),
                "p25_model_pair_mean_PCC": float(quantiles[0]),
                "p75_model_pair_mean_PCC": float(quantiles[1]),
                "maximum_model_pair_mean_PCC": float(estimates.max()),
                "mean_dataset_jaccard": float(group.dataset_jaccard.mean()),
                "minimum_dataset_jaccard": float(group.dataset_jaccard.min()),
                "maximum_dataset_jaccard": float(group.dataset_jaccard.max()),
            }
        )
    return per_transcript, pair_summary, pd.DataFrame(transition_rows), audits


def build_figure(
    panel_a_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    N_summary: pd.DataFrame,
    overlapping_N80_summary: pd.DataFrame,
    adjacent_summary: pd.DataFrame,
    width: float,
):
    pair_order = [f"P{a}\N{EN DASH}P{b}" for a, b in itertools.combinations(range(1, 5), 2)]
    ordered_a = panel_a_summary.set_index("panel_pair").loc[pair_order].reset_index()
    with matplotlib.rc_context(FIGURE_RC):
        figure, (axis_a, axis_b) = plt.subplots(
            1, 2, figsize=(width, 2.65), gridspec_kw={"width_ratios": (1.0, 1.08)},
            layout="constrained",
        )
        y = np.arange(len(pair_order) - 1, -1, -1)
        axis_a.hlines(y, ordered_a.p05_PCC, ordered_a.p95_PCC, color="#9AA6B2", linewidth=1.0, zorder=1)
        axis_a.hlines(y, ordered_a.p25_PCC, ordered_a.p75_PCC, color="#4C78A8", linewidth=5.0, zorder=2)
        axis_a.scatter(ordered_a.median_PCC, y, s=25, color="#0B5A8C", edgecolor="white", linewidth=.5, zorder=3)
        low = max(-1.0, float(ordered_a.p05_PCC.min()) - 0.035)
        high = min(1.0, float(ordered_a.p95_PCC.max()) + 0.025)
        axis_a.set_xlim(low, high)
        axis_a.set_ylim(-0.65, len(pair_order) - 0.35)
        axis_a.set_yticks(y, [pair.replace("\N{EN DASH}", "--") for pair in pair_order])
        axis_a.set_xlabel("Cross-panel PCC")
        axis_a.set_title(r"\textbf{A}\quad Reproducibility", loc="left")
        axis_a.grid(axis="x")
        axis_a.set_axisbelow(True)

        for _, group in pair_summary.groupby("N", sort=True):
            axis_b.scatter(
                group.N, group.R_N_p_mean_transcript_PCC,
                s=18, color="#A7B2BC", edgecolor="white", linewidth=.45, zorder=2,
            )
        axis_b.plot(
            N_summary.N, N_summary.R_N_mean_over_pair_means,
            color="#0B5A8C", linewidth=1.2, zorder=3,
            label="Disjoint A/B stability",
        )
        axis_b.scatter(
            N_summary.N, N_summary.R_N_mean_over_pair_means,
            s=35, color="#0B5A8C", edgecolor="white", linewidth=.55, zorder=4,
        )
        axis_b.scatter(
            overlapping_N80_summary.N,
            overlapping_N80_summary.mean_over_pair_means,
            s=39,
            marker="s",
            facecolor="white",
            edgecolor="#0B5A8C",
            linewidth=1.25,
            zorder=4,
            label=r"Overlapping same-$N$ ($N=80$)",
        )
        transition_x = adjacent_summary.plot_x_larger_endpoint.to_numpy(dtype=float)
        axis_b.plot(
            transition_x,
            adjacent_summary.mean_over_model_pair_means,
            color="#66727C",
            linestyle="--",
            marker="D",
            markersize=4.0,
            markeredgecolor="white",
            markeredgewidth=.45,
            linewidth=1.1,
            zorder=3,
            label="Adjacent-size agreement",
        )
        axis_b.set_xscale("log")
        axis_b.set_xlim(1.65, 132.0)
        displayed_values = np.concatenate(
            [
                pair_summary.R_N_p_mean_transcript_PCC.to_numpy(dtype=float),
                overlapping_N80_summary.mean_over_pair_means.to_numpy(dtype=float),
                adjacent_summary.mean_over_model_pair_means.to_numpy(dtype=float),
            ]
        )
        axis_b.set_ylim(
            max(-1.0, float(displayed_values.min()) - .035),
            min(1.0, float(displayed_values.max()) + .04),
        )
        displayed_N = (*N_VALUES, 80, 114)
        axis_b.set_xticks(displayed_N, [str(value) for value in displayed_N])
        axis_b.get_xaxis().set_minor_locator(matplotlib.ticker.NullLocator())
        axis_b.set_xlabel("Number of datasets per model")
        axis_b.set_ylabel("Mean transcript-level PCC")
        axis_b.set_title(r"\textbf{B}\quad Stability across dataset scales", loc="left")
        axis_b.grid(axis="both")
        axis_b.set_axisbelow(True)
        axis_b.legend(loc="lower right", frameon=False, handlelength=2.1)
        return figure


def latex_caption(panel_count: int, stability_count: int) -> str:
    return rf"""% Requires \usepackage{{graphicx}}
\begin{{figure*}}[t]
  \centering
  \includegraphics[width=\textwidth]{{figures/real_data_equal.pdf}}
  \caption{{\textbf{{Reproducibility and stability across dataset scales under uniform reference weights.}}
  All models use fixed-reference centering with $\pi_d=1/N$ while retaining the fitted transcript--dataset reliability weights $w_{{dt}}$. \textbf{{A}}, full-CDS Pearson correlation of the frozen, mean-one shared profiles $L_t$ for {panel_count:,} held-out transcripts across the six pairs of four source-family-disjoint panels. Points are medians across transcripts; thick and thin intervals are the 25th--75th and 5th--95th percentiles, respectively, and describe transcript heterogeneity rather than uncertainty in the median. \textbf{{B}}, complementary stability summaries on a separate fixed cohort of {stability_count:,} held-out transcripts. Filled blue circles report source-family-disjoint same-$N$ stability for three pre-designated A/B subset pairs at $N=2,5,10,20,40$: small points are individual pair means and large connected points are their unweighted means. The disconnected open square at $N=80$ is the mean of the three pairwise comparisons among the three large-$N$ subsets; these subsets overlap and this marker is therefore not a continuation of the disjoint series. Gray diamonds report adjacent-size agreement for $2$--$5$, $5$--$10$, $10$--$20$, $20$--$40$, $40$--$80$, and $80$--$114$, plotted at the larger endpoint of each transition (thus the diamond at 114 represents $80$--$114$). Each diamond is the unweighted mean of all available cross-size model-pair means (36, 36, 36, 36, 18, and 3 pairs, respectively); model-pair spreads and overlap diagnostics are provided in the source tables. The subset constructions are not nested and their dataset overlap increases with $N$, so the gray curve measures agreement across available model scales rather than a causal incremental effect of adding datasets. These metrics quantify reproducibility, not biological accuracy.}}
  \label{{fig:real-data-uniform-reference}}
\end{{figure*}}
"""


def write_report(
    path: Path,
    panel_provenance: Mapping[str, Any],
    stability_provenance: Mapping[str, Any],
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    overlapping_N80: pd.DataFrame,
    adjacent_size: pd.DataFrame,
    raw_audit: pd.DataFrame,
) -> None:
    def status_counts(frame: pd.DataFrame) -> str:
        counts = frame.status.value_counts().sort_index()
        return ", ".join(f"{name}={int(value)}" for name, value in counts.items())

    repeated = raw_audit.loc[raw_audit.raw_export_mode == "repeated_dataset_rows"]
    sequence_only = raw_audit.loc[raw_audit.raw_export_mode == "sequence_only_one_row_per_transcript"]
    path.write_text(
        "# Figure 1 provenance and exclusions\n\n"
        f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n"
        "## Inputs\n\n"
        f"- Panel A: `{panel_provenance['manifest']}`; seed {panel_provenance['training_seed']}; "
        f"{panel_provenance['common_test_count']:,} held-out transcripts; hash "
        f"`{panel_provenance['common_test_hash']}`.\n"
        f"- Panel B: `{stability_provenance['manifest']}`; seed {stability_provenance['training_seed']}; "
        f"{stability_provenance['common_test_count']:,} held-out transcripts; hash "
        f"`{stability_provenance['common_test_hash']}`.\n"
        "- Every accepted runtime gamma manifest uses fixed-reference equal weighting with exactly "
        "$\\pi_d=1/N$. Reliability manifests remain training-only and explicitly separate $w_{dt}$ from $\\pi$.\n"
        "- Panel A memberships are mutually disjoint at dataset and inferred source-family levels. "
        "Within each Panel B designated A/B pair, both dataset and source-family intersections are zero. "
        "Different designated pairs are separate design replicates and may overlap.\n\n"
        "- The additional three N=80 models are overlapping large-N subsets and the N=114 model is the "
        "single full-collection model. They enter the separately encoded N=80 and adjacent-size summaries.\n\n"
        "## Raw-export verification\n\n"
        f"The four Panel A raw exports contained repeated dataset rows. Exactly {int(repeated.repeated_rows_checked.sum()):,} "
        "repeated rows were checked before deduplication; masks and shared profiles were bitwise identical within "
        "every transcript, and their float32 materializations exactly matched the compact exports. "
        f"The {len(sequence_only)} Panel B exports were sequence-only, with one row per transcript, so no repeated-row "
        "deduplication was required; each raw profile also exactly matched its compact export.\n\n"
        "## Metric validity and exclusions\n\n"
        f"- Panel A row statuses: {status_counts(panel_a)}.\n"
        f"- Panel B row statuses: {status_counts(panel_b)}.\n"
        f"- Overlapping N=80 row statuses: {status_counts(overlapping_N80)}.\n"
        f"- Adjacent-size row statuses: {status_counts(adjacent_size)}.\n"
        f"- A profile is flagged near-constant when its full-CDS standard deviation is at most "
        f"{NEAR_CONSTANT_RELATIVE_STD:g} times max(1, |mean|). Undefined PCCs remain missing and are never set to zero.\n"
        "- No smoothing, observed-zero filtering, cross-transcript codon pooling, post-hoc normalization, or image digitization was used.\n"
        "- No N=80 disjoint-stability value and no N=114 same-size value are reported. N=80 is shown as an "
        "overlapping-subset summary; N=114 enters only the 80--114 adjacent-size comparison.\n\n"
        "## Interpretation boundary\n\n"
        "Panel A intervals are transcript-distribution intervals, not confidence intervals. Panel B has only three "
        "designated pair estimates per N; accordingly no confidence ribbon is drawn. The connected blue curve is a "
        "descriptive mean over those three pair means. The N=80 open square is disconnected because its subsets "
        "overlap. The gray transition curve averages all available cross-size model-pair means; its model-pair range "
        "does not represent independent uncertainty, and increasing overlap is a confounder. Agreement among frozen "
        "models is reproducibility, not proof of biological correctness.\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seed != MANUSCRIPT_SEED:
        raise ValueError(f"This manuscript workflow is fixed to the documented seed {MANUSCRIPT_SEED}.")
    if not np.isfinite(args.figure_width) or args.figure_width <= 0.0:
        raise ValueError("--figure-width must be finite and positive.")
    if args.dpi != 600:
        raise ValueError("The article preview is fixed at 600 dpi.")
    if args.raw_batch_size <= 0:
        raise ValueError("--raw-batch-size must be positive.")
    if not np.isfinite(args.mean_one_tolerance) or args.mean_one_tolerance < 0.0:
        raise ValueError("--mean-one-tolerance must be finite and non-negative.")

    panel_root = args.panel_root.expanduser().resolve()
    stability_root = args.stability_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = output / "real_data_equal_source"
    source.mkdir(parents=True, exist_ok=True)
    for obsolete_name in (
        "panel_b_convergence_per_transcript.csv",
        "panel_b_convergence_run_summary.csv",
        "panel_b_convergence_N_summary.csv",
        "panel_b_full_reference_marker.csv",
    ):
        (source / obsolete_name).unlink(missing_ok=True)

    panel_runs, panel_ids, panel_provenance = validate_panel_experiment(panel_root, args.seed)
    stability_runs, auxiliary_runs, stability_ids, stability_provenance = validate_stability_experiment(
        stability_root, args.seed
    )

    compact_audits: list[dict[str, Any]] = []
    raw_audits: list[dict[str, Any]] = []
    panel_profiles: dict[str, dict[str, Profile]] = {}
    for run in panel_runs:
        profiles, compact_audit = load_compact_profiles(
            run, panel_ids, args.mean_one_tolerance
        )
        compact_audits.append(compact_audit)
        raw_audits.append(
            audit_raw_export(
                run, profiles, panel_ids, args.raw_batch_size, expect_repeated_rows=True
            )
        )
        panel_profiles[run.identifier] = profiles
        print(f"Validated raw repeated profiles: {run.identifier}", flush=True)
    panel_a_values, panel_a_summary = calculate_panel_a(panel_runs, panel_profiles, panel_ids)
    del panel_profiles

    panel_b_values, panel_b_pair_summary, panel_b_N_summary, panel_b_compact_audits = calculate_panel_b(
        stability_runs, stability_ids, args.mean_one_tolerance
    )
    compact_audits.extend(panel_b_compact_audits)
    N80_runs = [run for run in auxiliary_runs if run.kind == "large_N_subset"]
    full_runs = [run for run in auxiliary_runs if run.kind == "full_collection"]
    if len(N80_runs) != 3 or len(full_runs) != 1:
        raise RuntimeError("Validated auxiliary run set is incomplete.")
    N80_values, N80_pair_summary, N80_summary, N80_audits = (
        calculate_overlapping_N80_stability(
            N80_runs,
            stability_ids,
            args.mean_one_tolerance,
        )
    )
    adjacent_values, adjacent_pair_summary, adjacent_summary, adjacent_audits = (
        calculate_adjacent_size_agreement(
            [*stability_runs, *auxiliary_runs],
            stability_ids,
            args.mean_one_tolerance,
        )
    )
    compact_audits.extend([*N80_audits, *adjacent_audits])
    for run in [*stability_runs, *auxiliary_runs]:
        profiles, _ = load_compact_profiles(run, stability_ids, args.mean_one_tolerance)
        raw_audits.append(
            audit_raw_export(
                run, profiles, stability_ids, args.raw_batch_size, expect_repeated_rows=False
            )
        )
        del profiles
        print(f"Validated sequence-only raw profile: {run.identifier}", flush=True)

    raw_audit = pd.DataFrame(raw_audits)
    compact_audit = pd.DataFrame(compact_audits).drop_duplicates("run_id")
    if (
        panel_a_values.status.ne("valid").any()
        or panel_b_values.status.ne("valid").any()
        or N80_values.status.ne("valid").any()
        or adjacent_values.status.ne("valid").any()
    ):
        # Tables and report still identify exact exclusions, but an incomplete primary figure is not exported.
        panel_a_values.to_csv(source / "panel_a_per_transcript.csv", index=False)
        panel_b_values.to_csv(source / "panel_b_per_transcript.csv", index=False)
        N80_values.to_csv(source / "panel_b_N80_per_transcript.csv", index=False)
        adjacent_values.to_csv(source / "panel_b_adjacent_size_per_transcript.csv", index=False)
        raw_audit.to_csv(source / "raw_repeat_audit.csv", index=False)
        raise RuntimeError(
            "Undefined or invalid profile comparisons remain. Source tables were written, but the final figure was withheld."
        )

    panel_a_values.to_csv(source / "panel_a_per_transcript.csv", index=False)
    panel_a_summary.to_csv(source / "panel_a_summary.csv", index=False)
    panel_b_values.to_csv(source / "panel_b_per_transcript.csv", index=False)
    panel_b_pair_summary.to_csv(source / "panel_b_pair_summary.csv", index=False)
    panel_b_N_summary.to_csv(source / "panel_b_N_summary.csv", index=False)
    N80_values.to_csv(source / "panel_b_N80_per_transcript.csv", index=False)
    N80_pair_summary.to_csv(source / "panel_b_N80_pair_summary.csv", index=False)
    N80_summary.to_csv(source / "panel_b_N80_summary.csv", index=False)
    adjacent_values.to_csv(source / "panel_b_adjacent_size_per_transcript.csv", index=False)
    adjacent_pair_summary.to_csv(source / "panel_b_adjacent_size_pair_summary.csv", index=False)
    adjacent_summary.to_csv(source / "panel_b_adjacent_size_summary.csv", index=False)
    raw_audit.to_csv(source / "raw_repeat_audit.csv", index=False)
    compact_audit.to_csv(source / "compact_profile_audit.csv", index=False)

    run_rows = []
    for experiment, runs in (
        ("panel_A", panel_runs),
        ("panel_B_stability", stability_runs),
        ("panel_B_additional_scales", auxiliary_runs),
    ):
        for run in runs:
            run_rows.append(
                {
                    "figure_panel": experiment, "run_id": run.identifier, "N": run.N,
                    "training_seed": run.seed, "pair_id": run.pair_id, "side": run.side,
                    "experiment_kind": run.kind,
                    "number_of_datasets": len(run.datasets),
                    "number_of_source_families": len(run.source_families),
                    "compact_path": str(run.compact_path), "raw_path": str(run.raw_path),
                    "gamma_manifest": str(run.gamma_manifest_path),
                    "checkpoint_manifest": str(run.checkpoint_manifest_path),
                    "reliability_manifest": str(run.reliability_manifest_path),
                    "gamma_reference_weighting": "equal", "pi_value": 1.0 / run.N,
                }
            )
    pd.DataFrame(run_rows).to_csv(source / "run_provenance.csv", index=False)
    (source / "cohort_provenance.json").write_text(
        json.dumps({"panel_A": panel_provenance, "panel_B": stability_provenance}, indent=2) + "\n"
    )

    figure = build_figure(
        panel_a_summary,
        panel_b_pair_summary,
        panel_b_N_summary,
        N80_summary,
        adjacent_summary,
        args.figure_width,
    )
    with matplotlib.rc_context(FIGURE_RC):
        figure.savefig(output / "real_data_equal.pdf", bbox_inches="tight")
        figure.savefig(output / "real_data_equal.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    caption = latex_caption(len(panel_ids), len(stability_ids))
    (output / "real_data_equal.tex").write_text(caption, encoding="utf-8")
    write_report(
        source / "PROVENANCE_AND_EXCLUSIONS.md",
        panel_provenance, stability_provenance,
        panel_a_values, panel_b_values, N80_values, adjacent_values, raw_audit,
    )

    command = shlex.join(
        [
            # Keep the virtual-environment interpreter path; resolving the
            # symlink would incorrectly record the dependency-free system Python.
            str(Path(sys.executable)), str(Path(__file__).resolve()),
            "--panel-root", str(panel_root), "--stability-root", str(stability_root),
            "--output-dir", str(output), "--seed", str(args.seed),
            "--figure-width", str(args.figure_width), "--dpi", str(args.dpi),
            "--mean-one-tolerance", str(args.mean_one_tolerance),
            "--raw-batch-size", str(args.raw_batch_size),
        ]
    )
    regenerate = source / "regenerate.sh"
    regenerate.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n")
    regenerate.chmod(0o755)
    (source / "figure_manifest.json").write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "script": str(Path(__file__).resolve()),
                "script_sha256": file_sha256(Path(__file__)),
                "training_seed": args.seed,
                "panel_A_test_count": len(panel_ids),
                "panel_A_test_hash": transcript_id_hash(panel_ids),
                "panel_B_test_count": len(stability_ids),
                "panel_B_test_hash": transcript_id_hash(stability_ids),
                "panel_B_disjoint_stability_N": list(N_VALUES),
                "panel_B_overlapping_same_N": 80,
                "panel_B_adjacent_size_transitions": [
                    [2, 5], [5, 10], [10, 20], [20, 40], [40, 80], [80, 114]
                ],
                "adjacent_size_models_are_not_nested": True,
                "uniform_gamma_reference": True,
                "reliability_weights_preserved": True,
                "raw_repeated_profile_audit": "passed",
                "output_pdf": str((output / "real_data_equal.pdf").resolve()),
                "output_png": str((output / "real_data_equal.png").resolve()),
                "png_dpi": args.dpi,
                "regeneration_command": command,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {output / 'real_data_equal.pdf'}", flush=True)
    print(panel_a_summary.to_string(index=False), flush=True)
    print(panel_b_pair_summary.to_string(index=False), flush=True)
    print(panel_b_N_summary.to_string(index=False), flush=True)
    print(N80_pair_summary.to_string(index=False), flush=True)
    print(N80_summary.to_string(index=False), flush=True)
    print(adjacent_pair_summary.to_string(index=False), flush=True)
    print(adjacent_summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
