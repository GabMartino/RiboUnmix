#!/usr/bin/env python3
"""Audit and analyse the directional quality-score cumulative experiment.

The analysis is deliberately read-only.  It validates the frozen task design,
the realised gamma reference, the common sequence-only test cohort and the
compact shared-profile exports before computing transcript-level comparisons.
Missing planned sizes remain missing; the script never bridges over them.

The primary estimand compares every fitted profile in a selection direction
with that direction's *equal-reference* N=2 profile.  Policy-specific N=2
anchors, adjacent-size comparisons and same-N policy sensitivity are reported
as diagnostics rather than substituted into the primary estimand.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyses.paths import artifact_directory

DEFAULT_ROOT = (
    PROJECT_ROOT
    / "results/cumulative_selection_direction_quality_score_directional_seed42"
)
DESIGN = "cumulative_dataset_selection_direction_quality_score_v2_directional"
DIRECTIONS = ("best_first", "worst_first")
POLICIES = ("equal", "score_p1", "score_p3", "score_p5")
POLICY_LABELS = {
    "equal": "Equal reference",
    "score_p1": r"Score weighted, $p=1$",
    "score_p3": r"Score weighted, $p=3$",
    "score_p5": r"Score weighted, $p=5$",
}
POLICY_HTML = {
    "equal": "Equal reference",
    "score_p1": "Score weighted, <i>p</i>=1",
    "score_p3": "Score weighted, <i>p</i>=3",
    "score_p5": "Score weighted, <i>p</i>=5",
}
POLICY_COLORS = {
    "equal": "#4B5563",
    "score_p1": "#0072B2",
    "score_p3": "#D55E00",
    "score_p5": "#7A3E9D",
}
DIRECTION_LABELS = {"best_first": "Best-first", "worst_first": "Worst-first"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def transcript_id_hash(values) -> str:
    payload = "\n".join(sorted(set(map(str, values)))).encode()
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_json(path: Path, value) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def relocate(path, root: Path, recorded_root: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        try:
            return root / path.relative_to(recorded_root)
        except ValueError as error:
            raise ValueError(
                f"Recorded path is outside the frozen experiment root: {path}"
            ) from error
    return root / path


def flatten_config(value, prefix=""):
    flat = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_config(item, child))
    else:
        flat[prefix] = value
    return flat


def verify_runtime_config(prepared, actual, state) -> list[str]:
    """Allow only recorded resume and execution-microbatching changes."""
    left, right = flatten_config(prepared), flatten_config(actual)
    changes = sorted(key for key in left.keys() | right.keys() if left.get(key) != right.get(key))
    if not changes:
        return []
    allowed_resume = {
        "experiment.from_checkpoint",
        "experiment.resume_training_state",
        "experiment.resume_checkpoint_path",
    }
    recorded_overrides = {}
    for attempt in state.get("attempts", []):
        recorded_overrides.update(attempt.get("runtime_overrides", {}))
    allowed_execution = set(recorded_overrides)
    if set(changes) - allowed_resume - allowed_execution:
        raise ValueError(f"Unapproved runtime configuration changes: {changes}")
    for key in set(changes) & allowed_execution:
        if right.get(key) != recorded_overrides[key]:
            raise ValueError(f"Runtime value for {key} differs from the recorded override.")
    resume_changes = set(changes) & allowed_resume
    if resume_changes:
        checkpoints = [
            attempt.get("resume_checkpoint", {}).get("path")
            if isinstance(attempt.get("resume_checkpoint"), dict)
            else attempt.get("resume_checkpoint")
            for attempt in state.get("attempts", [])
        ]
        if (
            not right.get("experiment.from_checkpoint")
            or not right.get("experiment.resume_training_state")
            or right.get("experiment.allow_weights_only_resume")
            or right.get("experiment.resume_checkpoint_path") not in checkpoints
        ):
            raise ValueError("Runtime resume fields do not match a recorded full-state resume.")
    return changes


def audit_design(root: Path):
    manifest_path = root / "experiment_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("experiment_design") != DESIGN:
        raise ValueError(f"Expected {DESIGN!r}, found {manifest.get('experiment_design')!r}.")
    tasks = manifest["tasks"]
    if object_sha256(tasks) != manifest["tasks_sha256"]:
        raise ValueError("Frozen task matrix checksum mismatch.")
    if manifest.get("training_seeds") != [42]:
        raise ValueError("This report expects the declared single seed 42 experiment.")
    sizes = list(map(int, manifest["sizes"]))
    if sizes != [2, 5, 10, 20, 40, 80, 114]:
        raise ValueError(f"Unexpected cumulative sizes: {sizes}")

    recorded_root = Path(manifest["output_root"])
    audit_rows = []
    for original, expected in manifest["frozen_file_sha256"].items():
        local = relocate(original, root, recorded_root)
        if not local.is_file():
            raise FileNotFoundError(f"Frozen input is unavailable: {local}")
        observed = sha256(local)
        if observed != expected:
            raise ValueError(f"Frozen input checksum mismatch: {local}")
        audit_rows.append(
            dict(category="frozen_input", local_path=str(local), sha256=observed)
        )

    folds = manifest["source_folds"]
    first = next(iter(folds.values()))
    for field in ("train_ids", "validation_ids", "test_ids"):
        if any(fold[field] != first[field] for fold in folds.values()):
            raise ValueError(f"{field} is not fixed across cumulative collections.")
        if len(first[field]) != len(set(map(str, first[field]))):
            raise ValueError(f"Duplicate transcript IDs in {field}.")
    train_ids = list(map(str, first["train_ids"]))
    validation_ids = list(map(str, first["validation_ids"]))
    test_ids = list(map(str, first["test_ids"]))
    if set(train_ids) & set(validation_ids) or set(train_ids) & set(test_ids) or set(validation_ids) & set(test_ids):
        raise ValueError("Train, validation and test transcript folds overlap.")
    split = read_json(root / "inputs/split.json")
    expected_hash = split["fold_id_hashes"]["test"]
    if transcript_id_hash(test_ids) != expected_hash:
        raise ValueError("Common test transcript hash differs from the split manifest.")

    weights = pd.read_csv(root / "reference_weights.csv")
    required = {
        "collection_id", "N", "arm", "reference_policy", "selection_direction",
        "dataset_id", "global_rank", "quality_rank_score", "assigned_q", "pi",
    }
    if not required <= set(weights):
        raise ValueError(f"Reference weights lack columns: {sorted(required - set(weights))}")
    if weights.duplicated(["collection_id", "arm", "dataset_id"]).any():
        raise ValueError("Duplicate rows in reference_weights.csv.")

    configs = {}
    seen_task_ids = set()
    for task in tasks:
        if task["run_id"] in seen_task_ids:
            raise ValueError(f"Duplicate task ID: {task['run_id']}")
        seen_task_ids.add(task["run_id"])
        config_path = relocate(task["config_path"], root, recorded_root)
        if sha256(config_path) != task["config_sha256"]:
            raise ValueError(f"Prepared configuration checksum mismatch: {config_path}")
        cfg = yaml.safe_load(config_path.read_text())
        if cfg["experiment"]["dataset"] != task["datasets"]:
            raise ValueError(f"{task['run_id']}: dataset order differs from the task.")
        if int(cfg["experiment"]["seed"]) != int(task["training_seed"]):
            raise ValueError(f"{task['run_id']}: training seed differs from the task.")
        if cfg["split"]["external_panel_name"] != task["source_panel"]:
            raise ValueError(f"{task['run_id']}: split panel differs from the task.")
        selected = weights[
            (weights.collection_id == task["panel_id"]) & (weights.arm == task["arm"])
        ].set_index("dataset_id")
        if set(selected.index) != set(task["datasets"]):
            raise ValueError(f"{task['run_id']}: reference-weight membership differs.")
        selected = selected.loc[task["datasets"]]
        reference = cfg["model"]["gamma_centering"]["reference"]
        explicit = reference.get("explicit_weights", {})
        expected_explicit = dict(zip(task["datasets"], selected.assigned_q.astype(float)))
        if reference.get("weighting") != "explicit" or set(explicit) != set(expected_explicit):
            raise ValueError(f"{task['run_id']}: prepared gamma reference is incomplete.")
        np.testing.assert_allclose(
            [explicit[name] for name in task["datasets"]],
            [expected_explicit[name] for name in task["datasets"]],
            rtol=1e-12, atol=1e-14,
        )
        configs[task["run_id"]] = cfg

    # Recompute every concentration diagnostic from pi instead of trusting the CSV.
    concentration_rows = []
    for (collection_id, arm), group in weights.groupby(["collection_id", "arm"], sort=False):
        pi = group.pi.to_numpy(float)
        if not np.isfinite(pi).all() or (pi <= 0).any() or not np.isclose(pi.sum(), 1.0, atol=1e-10):
            raise ValueError(f"Invalid normalized reference for {collection_id}/{arm}.")
        first_row = group.iloc[0]
        concentration_rows.append(dict(
            collection_id=collection_id,
            N=int(first_row.N),
            arm=arm,
            reference_policy=first_row.reference_policy,
            selection_direction=first_row.selection_direction,
            N_ref=float(1.0 / np.square(pi).sum()),
            effective_fraction=float(1.0 / np.square(pi).sum() / int(first_row.N)),
            weighted_mean_rank=float(np.dot(pi, group.global_rank)),
            weighted_mean_quality_rank_score=float(np.dot(pi, group.quality_rank_score)),
            max_reference_mass=float(pi.max()),
        ))
    concentration = pd.DataFrame(concentration_rows)
    saved = pd.read_csv(root / "reference_concentration.csv")
    merged = concentration.merge(
        saved,
        on=["collection_id", "N", "arm", "reference_policy", "selection_direction"],
        suffixes=("", "_saved"),
        validate="one_to_one",
    )
    for field in ("N_ref", "weighted_mean_rank", "weighted_mean_quality_rank_score"):
        np.testing.assert_allclose(merged[field], merged[f"{field}_saved"], rtol=1e-11, atol=1e-12)
    return manifest, configs, weights, concentration, test_ids, audit_rows


def locate_prediction_manifest(task_dir: Path) -> Path | None:
    paths = sorted(task_dir.rglob("prediction_checkpoint_manifest.json"))
    if not paths:
        return None
    if len(paths) != 1:
        raise ValueError(f"Expected one prediction manifest in {task_dir}, found {len(paths)}.")
    return paths[0]


def read_validated_export(
    root: Path,
    recorded_root: Path,
    task: dict,
    cfg: dict,
    weights: pd.DataFrame,
    test_ids: list[str],
):
    task_dir = root / task["directory"]
    state_path = task_dir / "execution_status.json"
    if not state_path.is_file():
        return None, dict(status="planned_not_available", detail="No local execution record")
    state = read_json(state_path)
    recorded_status = state.get("status", "unknown")
    if recorded_status != "completed":
        status = "recorded_failure" if recorded_status == "failed" else "incomplete"
        return None, dict(status=status, recorded_status=recorded_status,
                          detail=state.get("reason", f"Execution status: {recorded_status}"))
    if state.get("task_id") != task["run_id"] or state.get("config_sha256") != task["config_sha256"]:
        raise ValueError("Execution record does not match the frozen task identity.")

    runtime_cfg_path = task_dir / "hydra/.hydra/config.yaml"
    if not runtime_cfg_path.is_file():
        raise FileNotFoundError("Completed task has no downloaded Hydra configuration.")
    runtime_cfg = yaml.safe_load(runtime_cfg_path.read_text())
    runtime_changes = verify_runtime_config(cfg, runtime_cfg, state)

    prediction_manifest_path = locate_prediction_manifest(task_dir)
    if prediction_manifest_path is None:
        raise FileNotFoundError("Completed task has no prediction checkpoint manifest.")
    prediction_manifest = read_json(prediction_manifest_path)
    if set(prediction_manifest) != {"best_val_loss"}:
        raise ValueError("Expected exactly the best-validation-loss prediction export.")
    runtime = prediction_manifest["best_val_loss"]
    expected_test_hash = transcript_id_hash(test_ids)
    if (
        runtime.get("split_name") != "test"
        or runtime.get("transcript_count") != len(test_ids)
        or runtime.get("transcript_id_hash") != expected_test_hash
        or not runtime.get("sequence_only_shared_profile_prediction")
        or runtime.get("observation_dependent_dummy_outputs_are_scientific") is not False
    ):
        raise ValueError("Prediction export is not the frozen sequence-only common test cohort.")

    compact = relocate(runtime["shared_profile_output_path"], root, recorded_root)
    if not compact.is_file():
        candidate = prediction_manifest_path.parent / "common_test_L_profiles.parquet"
        if not candidate.is_file():
            raise FileNotFoundError("Compact shared-profile export is missing.")
        compact = candidate
    compact_hash = sha256(compact)
    outputs = state.get("outputs", {})
    if outputs.get("prediction_sha256") != compact_hash:
        raise ValueError("Compact prediction checksum differs from execution_status.json.")
    if outputs.get("test_transcript_id_hash") != expected_test_hash or outputs.get("n_test") != len(test_ids):
        raise ValueError("Execution output cohort differs from the frozen test cohort.")

    gamma_path = prediction_manifest_path.parent / "gamma_reference_manifest.json"
    gamma = read_json(gamma_path)
    selected = weights[
        (weights.collection_id == task["panel_id"]) & (weights.arm == task["arm"])
    ].set_index("dataset_id").loc[task["datasets"]]
    if (
        gamma.get("centering_mode") != "fixed_reference"
        or gamma.get("reference_dataset_names") != task["datasets"]
        or gamma.get("selected_dataset_names") != task["datasets"]
        or gamma.get("weighting") != "explicit"
        or gamma.get("pi_is_gamma_reference_only") is not True
    ):
        raise ValueError("Realised gamma reference differs from the frozen selected collection.")
    np.testing.assert_allclose(gamma["reference_pi"], selected.pi, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(
        gamma["reference_raw_weights"], selected.assigned_q, rtol=1e-6, atol=1e-9
    )

    frame = pd.read_parquet(compact)
    required = {"transcript_id", "transcript_length", "L_t", "valid_position_mask", "run_id", "N"}
    if not required <= set(frame):
        raise ValueError(f"Compact profile lacks columns: {sorted(required - set(frame))}")
    if frame.transcript_id.duplicated().any() or set(map(str, frame.transcript_id)) != set(test_ids):
        raise ValueError("Compact profile does not contain exactly the common test transcripts.")
    if frame.run_id.ne(task["run_id"]).any() or frame.N.ne(int(task["N"])).any():
        raise ValueError("Compact profile task identity differs from the frozen task.")

    profiles = {}
    mean_errors, variances = [], []
    for row in frame.itertuples(index=False):
        transcript = str(row.transcript_id)
        values = np.asarray(row.L_t, dtype=np.float64)
        mask = np.asarray(row.valid_position_mask, dtype=bool)
        if values.ndim != 1 or mask.shape != values.shape or len(values) != int(row.transcript_length):
            raise ValueError(f"{transcript}: compact profile length/mask mismatch.")
        if not mask.all():
            raise ValueError(f"{transcript}: compact profile does not span the full valid CDS.")
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{transcript}: nonfinite or nonpositive shared profile.")
        error = abs(float(values.mean()) - 1.0)
        if error > 1e-4:
            raise ValueError(f"{transcript}: shared profile is not mean one ({error:.3g}).")
        profiles[transcript] = dict(values=values, length=len(values))
        mean_errors.append(error)
        variances.append(float(values.var()))

    # Validate the exact codon coordinate sequence and the compact/raw profile match.
    raw = relocate(runtime["output_path"], root, recorded_root)
    if not raw.is_file():
        candidate = prediction_manifest_path.parent / Path(runtime["output_path"]).name
        if not candidate.is_file():
            raise FileNotFoundError("Raw sequence-only prediction export is missing.")
        raw = candidate
    seen = set()
    columns = ["transcript_id", "length", "mask", "codon_ids", "L_bio"]
    for batch in pq.ParquetFile(raw).iter_batches(batch_size=32, use_threads=False, columns=columns):
        for row in batch.to_pylist():
            transcript = str(row["transcript_id"])
            if transcript not in profiles or transcript in seen:
                raise ValueError(f"Unexpected or duplicate raw transcript: {transcript}")
            mask = np.asarray(row["mask"], dtype=bool)
            values = np.asarray(row["L_bio"], dtype=np.float64)
            codons = np.asarray(row["codon_ids"], dtype="<i8")
            if values.shape != mask.shape or codons.shape != mask.shape or int(mask.sum()) != int(row["length"]):
                raise ValueError(f"{transcript}: raw coordinate/length mismatch.")
            np.testing.assert_allclose(
                values[mask], profiles[transcript]["values"], rtol=2e-5, atol=2e-6
            )
            coordinate_hash = hashlib.sha256(
                codons[mask].tobytes() + np.flatnonzero(mask).astype("<i8").tobytes()
            ).hexdigest()
            profiles[transcript]["coordinate_hash"] = coordinate_hash
            seen.add(transcript)
    if seen != set(test_ids):
        raise ValueError("Raw prediction and compact test cohorts differ.")

    epoch_match = re.search(r"epoch=(\d+)", Path(runtime["checkpoint_path"]).name)
    return profiles, dict(
        status="validated_predictions",
        recorded_status=recorded_status,
        prediction_path=str(compact),
        prediction_sha256=compact_hash,
        raw_prediction_path=str(raw),
        prediction_manifest=str(prediction_manifest_path),
        gamma_manifest=str(gamma_path),
        runtime_config=str(runtime_cfg_path),
        runtime_config_changes=";".join(runtime_changes),
        selected_epoch=int(epoch_match.group(1)) if epoch_match else np.nan,
        validation_loss=outputs.get("validation_loss", np.nan),
        n_test_transcripts=len(test_ids),
        max_mean_one_error=float(max(mean_errors)),
        median_profile_variance=float(np.median(variances)),
        min_profile_variance=float(np.min(variances)),
        near_constant_profiles=int(np.sum(np.asarray(variances) <= 1e-12)),
        detail="Frozen best-validation-loss export validated",
    )


def collect_exports(root, manifest, configs, weights, test_ids):
    recorded_root = Path(manifest["output_root"])
    profiles, availability, profile_audit = {}, [], []
    for task in manifest["tasks"]:
        base = dict(
            array_index=int(task["array_index"]),
            run_id=task["run_id"],
            N=int(task["N"]),
            selection_direction=task["selection_direction"],
            reference_policy=task["reference_policy"],
            arm=task["arm"],
            panel_id=task["panel_id"],
            task_directory=str(root / task["directory"]),
        )
        try:
            values, info = read_validated_export(
                root, recorded_root, task, configs[task["run_id"]], weights, test_ids
            )
            availability.append(dict(**base, **info))
            if values is not None:
                profiles[task["run_id"]] = values
                profile_audit.append({**base, **{
                    key: info[key] for key in (
                        "n_test_transcripts", "max_mean_one_error", "median_profile_variance",
                        "min_profile_variance", "near_constant_profiles", "selected_epoch",
                        "validation_loss", "prediction_sha256",
                    )
                }})
        except (ValueError, KeyError, AssertionError, FileNotFoundError, RuntimeError, OSError) as error:
            availability.append(dict(
                **base,
                status="invalid_artifacts",
                detail=f"{type(error).__name__}: {error}",
            ))
    if profiles:
        for transcript in test_ids:
            coordinate_hashes = {
                values[transcript]["coordinate_hash"] for values in profiles.values()
            }
            if len(coordinate_hashes) != 1:
                raise ValueError(
                    f"{transcript}: codon coordinates differ across validated fits."
                )
    return profiles, pd.DataFrame(availability), pd.DataFrame(profile_audit)


def task_lookup(manifest, profiles):
    """Map a scientific (direction, policy, N) slot to a validated profile."""
    lookup = {}
    for task in manifest["tasks"]:
        if task["run_id"] not in profiles:
            continue
        if task["selection_direction"] == "shared_full":
            for direction in DIRECTIONS:
                lookup[direction, task["reference_policy"], int(task["N"])] = profiles[task["run_id"]]
        else:
            lookup[
                task["selection_direction"], task["reference_policy"], int(task["N"])
            ] = profiles[task["run_id"]]
    return lookup


def pcc_record(left, right):
    if left["length"] != right["length"] or left["coordinate_hash"] != right["coordinate_hash"]:
        return dict(PCC=np.nan, n_positions=0, reason="coordinate_or_length_mismatch")
    x, y = left["values"], right["values"]
    if len(x) < 2 or np.var(x) <= 1e-12 or np.var(y) <= 1e-12:
        return dict(PCC=np.nan, n_positions=len(x), reason="constant_or_too_short")
    value = float(np.corrcoef(x, y)[0, 1])
    return dict(PCC=value, n_positions=len(x), reason="ok" if np.isfinite(value) else "nonfinite")


def compute_metrics(lookup, ids, sizes):
    anchor_rows, own_rows, adjacent_rows, sensitivity_rows, direct_rows = [], [], [], [], []
    for direction in DIRECTIONS:
        common_anchor = lookup.get((direction, "equal", 2))
        for policy in POLICIES:
            own_anchor = lookup.get((direction, policy, 2))
            for n in sizes:
                current = lookup.get((direction, policy, n))
                if current is None:
                    continue
                if common_anchor is not None:
                    for transcript in ids:
                        anchor_rows.append(dict(
                            selection_direction=direction, reference_policy=policy, anchor_policy="equal",
                            anchor_N=2, N=n, transcript_id=transcript,
                            **pcc_record(common_anchor[transcript], current[transcript]),
                        ))
                if own_anchor is not None:
                    for transcript in ids:
                        own_rows.append(dict(
                            selection_direction=direction, reference_policy=policy, anchor_policy=policy,
                            anchor_N=2, N=n, transcript_id=transcript,
                            **pcc_record(own_anchor[transcript], current[transcript]),
                        ))
                equal = lookup.get((direction, "equal", n))
                if policy != "equal" and equal is not None:
                    for transcript in ids:
                        sensitivity_rows.append(dict(
                            selection_direction=direction, reference_policy=policy, N=n,
                            transcript_id=transcript,
                            **pcc_record(equal[transcript], current[transcript]),
                        ))
            for n_a, n_b in zip(sizes[:-1], sizes[1:]):
                left = lookup.get((direction, policy, n_a))
                right = lookup.get((direction, policy, n_b))
                if left is None or right is None:
                    continue
                for transcript in ids:
                    adjacent_rows.append(dict(
                        selection_direction=direction, reference_policy=policy,
                        N_a=n_a, N_b=n_b, transcript_id=transcript,
                        **pcc_record(left[transcript], right[transcript]),
                    ))
    for policy in POLICIES:
        for n in sizes:
            best = lookup.get(("best_first", policy, n))
            worst = lookup.get(("worst_first", policy, n))
            if best is None or worst is None:
                continue
            for transcript in ids:
                direct_rows.append(dict(
                    reference_policy=policy, N=n, transcript_id=transcript,
                    **pcc_record(best[transcript], worst[transcript]),
                ))
    columns = ["PCC", "n_positions", "reason"]
    return (
        pd.DataFrame(anchor_rows, columns=["selection_direction", "reference_policy", "anchor_policy", "anchor_N", "N", "transcript_id", *columns]),
        pd.DataFrame(own_rows, columns=["selection_direction", "reference_policy", "anchor_policy", "anchor_N", "N", "transcript_id", *columns]),
        pd.DataFrame(adjacent_rows, columns=["selection_direction", "reference_policy", "N_a", "N_b", "transcript_id", *columns]),
        pd.DataFrame(sensitivity_rows, columns=["selection_direction", "reference_policy", "N", "transcript_id", *columns]),
        pd.DataFrame(direct_rows, columns=["reference_policy", "N", "transcript_id", *columns]),
    )


def bootstrap_mean_interval(values, replicates, seed):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    results = np.empty(replicates, dtype=float)
    batch = 250
    for start in range(0, replicates, batch):
        stop = min(replicates, start + batch)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        results[start:stop] = values[indices].mean(axis=1)
    return tuple(np.quantile(results, [0.025, 0.975]))


def summarize_pcc(frame, groups, replicates, bootstrap_seed):
    rows = []
    if frame.empty:
        return pd.DataFrame(columns=[*groups, "n_valid", "n_excluded", "mean", "median", "q10", "q90", "ci_low", "ci_high"])
    for keys, group in frame.groupby(groups, sort=False, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        values = group.loc[group.reason.eq("ok"), "PCC"].to_numpy(float)
        values = values[np.isfinite(values)]
        salt = "|".join(map(str, keys))
        local_seed = (bootstrap_seed + int(hashlib.sha256(salt.encode()).hexdigest()[:8], 16)) % (2**32)
        low, high = bootstrap_mean_interval(values, replicates, local_seed)
        rows.append(dict(
            **dict(zip(groups, keys)),
            n_valid=len(values), n_excluded=len(group) - len(values),
            mean=float(np.mean(values)) if len(values) else np.nan,
            median=float(np.median(values)) if len(values) else np.nan,
            q10=float(np.quantile(values, .10)) if len(values) else np.nan,
            q90=float(np.quantile(values, .90)) if len(values) else np.nan,
            ci_low=low, ci_high=high,
        ))
    return pd.DataFrame(rows)


def ranking_effects(anchor_metrics, replicates, bootstrap_seed):
    if anchor_metrics.empty:
        return pd.DataFrame(), pd.DataFrame()
    equal = anchor_metrics[anchor_metrics.reference_policy.eq("equal")][
        ["selection_direction", "N", "transcript_id", "PCC"]
    ].rename(columns={"PCC": "PCC_equal"})
    score = anchor_metrics[~anchor_metrics.reference_policy.eq("equal")][
        ["selection_direction", "reference_policy", "N", "transcript_id", "PCC"]
    ]
    paired = score.merge(equal, on=["selection_direction", "N", "transcript_id"], validate="many_to_one")
    paired["delta_PCC_vs_equal"] = paired.PCC - paired.PCC_equal
    rows = []
    for keys, group in paired.groupby(["selection_direction", "reference_policy", "N"], sort=False):
        values = group.delta_PCC_vs_equal.to_numpy(float)
        values = values[np.isfinite(values)]
        salt = "effect|" + "|".join(map(str, keys))
        local_seed = (bootstrap_seed + int(hashlib.sha256(salt.encode()).hexdigest()[:8], 16)) % (2**32)
        low, high = bootstrap_mean_interval(values, replicates, local_seed)
        rows.append(dict(
            selection_direction=keys[0], reference_policy=keys[1], N=int(keys[2]),
            n_valid=len(values), mean_delta=float(values.mean()) if len(values) else np.nan,
            median_delta=float(np.median(values)) if len(values) else np.nan,
            fraction_positive=float(np.mean(values > 0)) if len(values) else np.nan,
            ci_low=low, ci_high=high,
        ))
    return paired, pd.DataFrame(rows)


def save_figure(fig, out: Path, stem: str):
    fig.savefig(out / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def style_axis(ax):
    ax.grid(True, color="#CBD5E1", alpha=.55, linewidth=.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def set_size_axis(ax, sizes):
    positions = np.arange(len(sizes))
    ax.set_xticks(positions, [str(n) for n in sizes])
    ax.set_xlim(-.25, len(sizes) - .75)
    ax.set_xlabel("Number of selected datasets $N$", fontweight="bold")
    return dict(zip(sizes, positions))


def adaptive_pcc_axis(ax, values, *, include_one=False, include_zero=False):
    values = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(values):
        return
    if include_one:
        values = np.r_[values, 1.0]
    if include_zero:
        values = np.r_[values, 0.0]
    low, high = float(values.min()), float(values.max())
    span = max(high - low, .08)
    margin = max(.018, .10 * span)
    ax.set_ylim(max(-1.0, low - margin), min(1.015, high + margin))


def geometry_for_plot(concentration, sizes):
    rows = []
    for row in concentration.to_dict("records"):
        if row["selection_direction"] == "shared_full":
            for direction in DIRECTIONS:
                copied = dict(row)
                copied["selection_direction"] = direction
                rows.append(copied)
        else:
            rows.append(row)
    return pd.DataFrame(rows)


def plot_reference_geometry(concentration, sizes, out):
    geometry = geometry_for_plot(concentration, sizes)
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.4), sharex=True)
    xmap = dict(zip(sizes, np.arange(len(sizes))))
    for column, direction in enumerate(DIRECTIONS):
        selected = geometry[geometry.selection_direction.eq(direction)]
        for policy in POLICIES:
            group = selected[selected.reference_policy.eq(policy)].sort_values("N")
            if group.empty:
                continue
            x = [xmap[int(n)] for n in group.N]
            axes[0, column].plot(x, group.effective_fraction, "o-", lw=2.2,
                                 color=POLICY_COLORS[policy], label=POLICY_LABELS[policy])
            axes[1, column].plot(x, group.weighted_mean_rank, "o-", lw=2.2,
                                 color=POLICY_COLORS[policy])
        axes[0, column].set_title(
            f"{'A' if column == 0 else 'B'}  {DIRECTION_LABELS[direction]} concentration",
            loc="left", fontweight="bold", fontsize=13,
        )
        axes[1, column].set_title(
            f"{'C' if column == 0 else 'D'}  {DIRECTION_LABELS[direction]} rank location",
            loc="left", fontweight="bold", fontsize=13,
        )
        axes[0, column].set_ylim(0, 1.04)
        axes[1, column].set_ylim(0, 116)
        set_size_axis(axes[1, column], sizes)
        for ax in axes[:, column]:
            ax.set_xticks(np.arange(len(sizes)), [str(n) for n in sizes])
            style_axis(ax)
    axes[0, 0].set_ylabel(r"Effective fraction $N_{\rm eff}/N$", fontweight="bold")
    axes[1, 0].set_ylabel("Reference-weighted global QC rank", fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=10, ncol=2, loc="lower left")
    fig.suptitle(
        "The same score exponent does not match reference concentration across directions",
        fontsize=15, fontweight="bold", y=.995,
    )
    fig.tight_layout(rect=(0, 0, 1, .97))
    save_figure(fig, out, "reference_geometry")


def plot_partial_results(anchor_summary, direct_summary, sizes, out):
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
    xmap = dict(zip(sizes, np.arange(len(sizes))))
    all_values = [[], [], []]
    for column, direction in enumerate(DIRECTIONS):
        selected = anchor_summary[anchor_summary.selection_direction.eq(direction)]
        for policy in POLICIES:
            group = selected[selected.reference_policy.eq(policy)].sort_values("N")
            if group.empty:
                continue
            x = np.asarray([xmap[int(n)] for n in group.N])
            y = group["mean"].to_numpy(float)
            axes[column].plot(x, y, "o-", lw=2.2, ms=6, color=POLICY_COLORS[policy],
                              label=POLICY_LABELS[policy])
            axes[column].fill_between(x, group.ci_low, group.ci_high,
                                      color=POLICY_COLORS[policy], alpha=.13, linewidth=0)
            all_values[column].extend(group.ci_low.tolist() + group.ci_high.tolist())
        axes[column].set_title(
            f"{'A' if column == 0 else 'B'}  {DIRECTION_LABELS[direction]} vs equal $N=2$",
            loc="left", fontweight="bold", fontsize=13,
        )
        axes[column].set_ylabel(r"Mean transcript PCC of $\mathbf{L}_{\mathbf{t}}$", fontweight="bold")
        adaptive_pcc_axis(axes[column], all_values[column], include_one=True)

    for policy in POLICIES:
        group = direct_summary[direct_summary.reference_policy.eq(policy)].sort_values("N")
        if group.empty:
            continue
        x = np.asarray([xmap[int(n)] for n in group.N])
        axes[2].plot(x, group["mean"], "o-", lw=2.2, ms=6, color=POLICY_COLORS[policy],
                     label=POLICY_LABELS[policy])
        axes[2].fill_between(x, group.ci_low, group.ci_high,
                            color=POLICY_COLORS[policy], alpha=.13, linewidth=0)
        all_values[2].extend(group.ci_low.tolist() + group.ci_high.tolist())
    axes[2].set_title("C  Direct best-first vs worst-first", loc="left", fontweight="bold", fontsize=13)
    axes[2].set_ylabel(r"Mean transcript PCC of $\mathbf{L}_{\mathbf{t}}$", fontweight="bold")
    adaptive_pcc_axis(axes[2], all_values[2], include_zero=True)

    for index, ax in enumerate(axes):
        set_size_axis(ax, sizes)
        if index == 2:
            ax.axvspan(xmap[80] - .45, len(sizes) - .75, color="#F1F5F9", zorder=-5)
            ax.text(
                (xmap[80] + xmap[114]) / 2, .035,
                "forced membership overlap",
                transform=ax.get_xaxis_transform(), ha="center",
                color="#64748B", fontweight="bold", fontsize=8.5,
            )
        style_axis(ax)
    axes[0].legend(frameon=False, fontsize=10, loc="lower left")
    fig.suptitle(
        "Shared-profile evidence on one fixed held-out transcript cohort",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.text(
        .5, -.01,
        "Bands are 95% transcript-bootstrap intervals for the mean; they do not represent retraining or seed uncertainty.",
        ha="center", fontsize=9.5,
    )
    fig.tight_layout(rect=(0, .035, 1, .98))
    save_figure(fig, out, "partial_profile_results")


def plot_adjacent_stability(adjacent_summary, sizes, out):
    """Plot only prespecified consecutive-size comparisons; never bridge gaps."""
    transitions = list(zip(sizes[:-1], sizes[1:]))
    xmap = {transition: index for index, transition in enumerate(transitions)}
    labels = [f"{left}→{right}" for left, right in transitions]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7), sharex=True)
    for column, direction in enumerate(DIRECTIONS):
        selected = adjacent_summary[
            adjacent_summary.selection_direction.eq(direction)
        ]
        plotted_values = []
        for policy in POLICIES:
            group = selected[selected.reference_policy.eq(policy)].sort_values("N_b")
            if group.empty:
                continue
            x = np.asarray([
                xmap[int(row.N_a), int(row.N_b)] for row in group.itertuples()
            ])
            axes[column].plot(
                x, group["mean"], "o-", lw=2.2, ms=6,
                color=POLICY_COLORS[policy], label=POLICY_LABELS[policy],
            )
            axes[column].fill_between(
                x, group.ci_low, group.ci_high,
                color=POLICY_COLORS[policy], alpha=.13, linewidth=0,
            )
            plotted_values.extend(group.ci_low.tolist() + group.ci_high.tolist())
        axes[column].set_title(
            f"{'A' if column == 0 else 'B'}  {DIRECTION_LABELS[direction]} adjacent stability",
            loc="left", fontweight="bold", fontsize=13,
        )
        axes[column].set_ylabel(
            r"Mean transcript PCC of $\mathbf{L}_{\mathbf{t}}$",
            fontweight="bold",
        )
        axes[column].set_xlabel("Consecutive cumulative collections", fontweight="bold")
        axes[column].set_xticks(np.arange(len(transitions)), labels, rotation=20)
        axes[column].set_xlim(-.25, len(transitions) - .75)
        adaptive_pcc_axis(axes[column], plotted_values, include_one=True)
        style_axis(axes[column])
    axes[0].legend(frameon=False, fontsize=10, loc="upper right")
    fig.suptitle(
        "Agreement of the shared profile between consecutive cumulative sizes",
        fontsize=15, fontweight="bold", y=1.01,
    )
    fig.text(
        .5, -.005,
        "Only completed adjacent endpoints are joined; unavailable policy endpoints remain explicit gaps.",
        ha="center", fontsize=9.5,
    )
    fig.tight_layout(rect=(0, .04, 1, .97))
    save_figure(fig, out, "adjacent_stability")


def plot_policy_diagnostics(effect_summary, sensitivity_summary, sizes, out):
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.5), sharex=True)
    xmap = dict(zip(sizes, np.arange(len(sizes))))
    for column, direction in enumerate(DIRECTIONS):
        effect = effect_summary[effect_summary.selection_direction.eq(direction)]
        sensitivity = sensitivity_summary[sensitivity_summary.selection_direction.eq(direction)]
        effect_values, sensitivity_values = [], []
        for policy in POLICIES[1:]:
            group = effect[effect.reference_policy.eq(policy)].sort_values("N")
            if not group.empty:
                x = np.asarray([xmap[int(n)] for n in group.N])
                axes[0, column].plot(x, group.mean_delta, "o-", lw=2.2,
                                     color=POLICY_COLORS[policy], label=POLICY_LABELS[policy])
                axes[0, column].fill_between(x, group.ci_low, group.ci_high,
                                             color=POLICY_COLORS[policy], alpha=.13, linewidth=0)
                effect_values.extend(group.ci_low.tolist() + group.ci_high.tolist())
            group = sensitivity[sensitivity.reference_policy.eq(policy)].sort_values("N")
            if not group.empty:
                x = np.asarray([xmap[int(n)] for n in group.N])
                axes[1, column].plot(x, group["mean"], "o-", lw=2.2,
                                     color=POLICY_COLORS[policy])
                axes[1, column].fill_between(x, group.ci_low, group.ci_high,
                                             color=POLICY_COLORS[policy], alpha=.13, linewidth=0)
                sensitivity_values.extend(group.ci_low.tolist() + group.ci_high.tolist())
        axes[0, column].axhline(0, color="#475569", lw=1.1)
        axes[0, column].set_title(
            f"{'A' if column == 0 else 'B'}  {DIRECTION_LABELS[direction]} anchor gain",
            loc="left", fontweight="bold", fontsize=13,
        )
        axes[1, column].set_title(
            f"{'C' if column == 0 else 'D'}  {DIRECTION_LABELS[direction]} policy sensitivity",
            loc="left", fontweight="bold", fontsize=13,
        )
        axes[0, column].set_ylabel(r"Mean $\Delta$PCC vs equal reference", fontweight="bold")
        axes[1, column].set_ylabel(r"Mean PCC(score, equal) of $\mathbf{L}_{\mathbf{t}}$", fontweight="bold")
        if effect_values:
            low, high = min(effect_values + [0]), max(effect_values + [0])
            span = max(high - low, .04)
            axes[0, column].set_ylim(low - .12 * span, high + .12 * span)
        adaptive_pcc_axis(axes[1, column], sensitivity_values, include_one=True)
        for ax in axes[:, column]:
            set_size_axis(ax, sizes)
            if 20 in xmap:
                ax.axvspan(xmap[20] - .45, len(sizes) - .75, color="#F1F5F9", zorder=-5)
            style_axis(ax)
    axes[0, 0].legend(frameon=False, fontsize=10)
    fig.suptitle(
        "A stronger reference preserves each path's own extreme while changing the fitted solution",
        fontsize=15, fontweight="bold", y=.995,
    )
    fig.tight_layout(rect=(0, 0, 1, .97))
    save_figure(fig, out, "policy_diagnostics")


def html_table(frame, columns=None, rename=None, precision=4):
    view = frame.copy()
    if columns is not None:
        view = view[columns]
    if rename:
        view = view.rename(columns=rename)
    return '<div class="table-wrap">' + view.to_html(
        index=False, border=0, escape=True, na_rep="—",
        float_format=lambda value: f"{value:.{precision}f}",
        classes="data-table",
    ) + "</div>"


def availability_table(availability, sizes):
    symbol = {
        "validated_predictions": "complete",
        "planned_not_available": "pending",
        "incomplete": "partial",
        "recorded_failure": "failed",
        "invalid_artifacts": "invalid",
    }
    table = availability.copy()
    table["status_label"] = table.status.map(symbol).fillna(table.status)
    pivot = table.pivot_table(
        index=["selection_direction", "reference_policy"], columns="N",
        values="status_label", aggfunc="first",
    ).reindex(columns=sizes).reset_index()
    pivot["selection_direction"] = pivot.selection_direction.map(
        {**DIRECTION_LABELS, "shared_full": "Shared full set"}
    )
    pivot["reference_policy"] = pivot.reference_policy.map(
        {key: re.sub(r"\$|\\mathbf|[{}]", "", value) for key, value in POLICY_LABELS.items()}
    )
    return pivot


def value_at(frame, **filters):
    selected = frame
    for key, value in filters.items():
        selected = selected[selected[key].eq(value)]
    return float(selected.iloc[0]["mean"]) if len(selected) else np.nan


def render_report(
    root, out, manifest, availability, profile_audit, concentration,
    anchor_summary, own_summary, adjacent_summary, sensitivity_summary,
    direct_summary, effect_summary, test_ids, sizes, replicates,
):
    complete = int(availability.status.eq("validated_predictions").sum())
    available_sizes = sorted(availability.loc[availability.status.eq("validated_predictions"), "N"].unique())
    invalid = int(availability.status.eq("invalid_artifacts").sum())
    geometry = geometry_for_plot(concentration, sizes)

    def fraction(direction, policy, n):
        selected = geometry[
            geometry.selection_direction.eq(direction)
            & geometry.reference_policy.eq(policy)
            & geometry.N.eq(n)
        ]
        return float(selected.iloc[0].effective_fraction) if len(selected) else np.nan

    common_policy_sizes = sorted(
        set.intersection(*(
            set(anchor_summary.loc[
                anchor_summary.selection_direction.eq(direction)
                & anchor_summary.reference_policy.eq(policy), "N"
            ].astype(int))
            for direction in DIRECTIONS for policy in POLICIES
        ))
    )
    reporting_n = max(common_policy_sizes)
    best_equal_report = value_at(anchor_summary, selection_direction="best_first", reference_policy="equal", N=reporting_n)
    best_p5_report = value_at(anchor_summary, selection_direction="best_first", reference_policy="score_p5", N=reporting_n)
    worst_equal_report = value_at(anchor_summary, selection_direction="worst_first", reference_policy="equal", N=reporting_n)
    worst_p5_report = value_at(anchor_summary, selection_direction="worst_first", reference_policy="score_p5", N=reporting_n)
    direct_equal_2 = value_at(direct_summary, reference_policy="equal", N=2)
    direct_equal_40 = value_at(direct_summary, reference_policy="equal", N=40)
    direct_equal_80 = value_at(direct_summary, reference_policy="equal", N=80)
    worst_p1_sens_2 = value_at(sensitivity_summary, selection_direction="worst_first", reference_policy="score_p1", N=2)
    adjacent_best_equal_25 = value_at(
        adjacent_summary, selection_direction="best_first", reference_policy="equal", N_a=2, N_b=5
    )
    adjacent_best_equal_510 = value_at(
        adjacent_summary, selection_direction="best_first", reference_policy="equal", N_a=5, N_b=10
    )
    adjacent_worst_equal_25 = value_at(
        adjacent_summary, selection_direction="worst_first", reference_policy="equal", N_a=2, N_b=5
    )
    adjacent_worst_equal_510 = value_at(
        adjacent_summary, selection_direction="worst_first", reference_policy="equal", N_a=5, N_b=10
    )

    availability_html = html_table(availability_table(availability, sizes), precision=3)
    core = anchor_summary[
        anchor_summary.N.isin(available_sizes)
    ].copy()
    core["direction"] = core.selection_direction.map(DIRECTION_LABELS)
    core["policy"] = core.reference_policy.map({k: POLICY_HTML[k].replace("<i>", "").replace("</i>", "") for k in POLICIES})
    core_table = html_table(
        core,
        ["direction", "policy", "N", "n_valid", "mean", "median", "q10", "q90"],
        {"direction": "Direction", "policy": "Reference", "n_valid": "Transcripts",
         "mean": "Mean PCC", "median": "Median", "q10": "10th pct.", "q90": "90th pct."},
    )
    concentration_display = geometry[
        geometry.N.isin([2, 10, 20, 114])
    ][["selection_direction", "reference_policy", "N", "effective_fraction", "weighted_mean_rank"]].copy()
    concentration_display["selection_direction"] = concentration_display.selection_direction.map(DIRECTION_LABELS)
    concentration_display["reference_policy"] = concentration_display.reference_policy.map(
        {k: POLICY_HTML[k].replace("<i>", "").replace("</i>", "") for k in POLICIES}
    )
    concentration_table = html_table(
        concentration_display,
        rename={"selection_direction": "Direction", "reference_policy": "Reference",
                "effective_fraction": "N_eff / N", "weighted_mean_rank": "Weighted QC rank"},
    )

    pending = len(manifest["tasks"]) - complete
    status_note = (
        f"{complete}/{len(manifest['tasks'])} planned fits are locally complete and validated. "
        f"The available model evidence covers N={', '.join(map(str, available_sizes))}. "
        f"All policies are available through N={reporting_n}; {pending} full-collection "
        "directional fits at N=114 remain incomplete."
    )
    if invalid:
        status_note += f" {invalid} downloaded fit(s) failed artifact validation and were excluded."

    style = """
    :root{--ink:#172B3A;--muted:#526777;--blue:#0072B2;--orange:#D55E00;--line:#D9E2E8;--paper:#fff;--wash:#F4F8FB}
    *{box-sizing:border-box} body{margin:0;background:#EEF3F6;color:var(--ink);font:16px/1.62 Inter,system-ui,-apple-system,Segoe UI,sans-serif}
    main{max-width:1180px;margin:30px auto;background:var(--paper);padding:42px 54px 64px;box-shadow:0 8px 35px #17344918}
    h1{font-size:2.15rem;line-height:1.15;margin:0 0 10px;letter-spacing:-.025em}h2{font-size:1.45rem;margin:42px 0 12px;border-bottom:2px solid var(--line);padding-bottom:7px}h3{font-size:1.08rem;margin:24px 0 6px}
    p{margin:9px 0 14px}.lede{font-size:1.12rem;color:var(--muted);max-width:940px}.status{padding:15px 18px;background:#EAF5FB;border-left:5px solid var(--blue);margin:24px 0}.warning{padding:15px 18px;background:#FFF5E8;border-left:5px solid var(--orange);margin:18px 0}
    .cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:20px 0}.card{background:var(--wash);border:1px solid var(--line);padding:16px;border-radius:8px}.card b{display:block;font-size:1.35rem;color:#0B5D82;margin-bottom:4px}.card span{font-size:.93rem;color:var(--muted)}
    figure{margin:24px 0 34px}figure img{display:block;width:100%;height:auto;border:1px solid #E5EBEF;background:white}figcaption{color:var(--muted);font-size:.94rem;margin-top:9px}
    .table-wrap{overflow:auto;border:1px solid var(--line);border-radius:7px;margin:12px 0 24px}.data-table{border-collapse:collapse;width:100%;font-size:.86rem}.data-table th{background:#EDF4F7;position:sticky;top:0}.data-table td,.data-table th{padding:7px 9px;border-bottom:1px solid #E4EAEE;text-align:right;white-space:nowrap}.data-table td:first-child,.data-table th:first-child{text-align:left}
    code{background:#EDF2F5;padding:2px 5px;border-radius:4px}a{color:#006A9E}.formula{font-family:Georgia,serif;font-size:1.07rem;text-align:center;background:var(--wash);padding:13px;margin:16px 0}.small{font-size:.9rem;color:var(--muted)}ul{padding-left:23px}li{margin:7px 0}.downloads a{margin-right:15px}
    @media(max-width:760px){main{margin:0;padding:28px 20px}.cards{grid-template-columns:1fr}h1{font-size:1.75rem}}
    """

    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Directional quality-score cumulative experiment</title><style>{style}</style></head><body><main>
<h1>Directional quality-score gamma reference</h1>
<p class="lede">A partial, transcript-matched analysis of how dataset-selection direction and the revised <code>quality_rank_score</code> reference change the recovered shared profile <b>L<sub>t</sub></b>.</p>
<div class="status"><b>Current snapshot.</b> {status_note} Every reported comparison uses the same {len(test_ids):,} held-out transcripts and exact CDS coordinates.</div>

<div class="cards">
  <div class="card"><b>{best_equal_report:.3f} → {best_p5_report:.3f}</b><span>Best-first mean PCC to its equal N=2 anchor at N={reporting_n}, equal versus p=5.</span></div>
  <div class="card"><b>{worst_equal_report:.3f} → {worst_p5_report:.3f}</b><span>Worst-first mean PCC to its equal N=2 anchor at N={reporting_n}, equal versus p=5.</span></div>
  <div class="card"><b>{direct_equal_2:.3f} → {direct_equal_40:.3f}</b><span>Equal-reference best-versus-worst concordance from N=2 to N=40, while memberships remain disjoint.</span></div>
</div>

<h2>What changed in this experiment?</h2>
<p>Let <i>s</i><sub>d</sub> be the aggregate <code>quality_rank_score</code>, where smaller values indicate better measured quality. The revised intervention orients the gamma reference toward the extreme that defines each cumulative path:</p>
<div class="formula">best-first: &pi;<sub>d</sub> &prop; <i>s</i><sub>d</sub><sup>&minus;<i>p</i></sup> &nbsp;&nbsp;&nbsp; worst-first: &pi;<sub>d</sub> &prop; <i>s</i><sub>d</sub><sup><i>p</i></sup>, &nbsp; <i>p</i>&isin;{{1,3,5}}.</div>
<p>The global minimum or maximum used in the implementation cancels after normalization and therefore does not change &pi;. This version asks whether an oriented reference preserves the profile defined by the corresponding quality extreme. In particular, the worst-first score arms intentionally favor the worst-ranked datasets; they are an adverse directional control, not a candidate quality-improving reference.</p>
<div class="warning"><b>Important design confound.</b> Equal exponents are not concentration matched across directions. At N=114, p=3 gives N<sub>eff</sub>/N={fraction('best_first','score_p3',114):.3f} best-first versus {fraction('worst_first','score_p3',114):.3f} worst-first; for p=5 the fractions are {fraction('best_first','score_p5',114):.3f} and {fraction('worst_first','score_p5',114):.3f}. A direct best-versus-worst comparison at the same p therefore changes both orientation and intervention strength.</div>

<figure><img src="reference_geometry.svg" alt="Reference concentration and weighted rank across cumulative sizes">
<figcaption><b>Figure 1 | Planned reference geometry.</b> N<sub>eff</sub>=1/&Sigma;<sub>d</sub>&pi;<sub>d</sub><sup>2</sup>. The best-first score distribution is much more heterogeneous, so powers concentrate its reference sooner. These curves are fixed by design and are available even where model training is unfinished.</figcaption></figure>

<h2>What do the completed fits show?</h2>
<p>The primary comparison fixes one equal-reference N=2 anchor inside each selection direction. This makes all policy curves answer the same question within a path. The best-first and worst-first anchors are deliberately different datasets and must not be interpreted as biological ground truth.</p>
<figure><img src="partial_profile_results.svg" alt="Partial anchor stability and direct best versus worst concordance">
<figcaption><b>Figure 2 | Shared-profile stability and selection-direction concordance.</b> Stronger score weighting preserves both the best-defined and the worst-defined path more closely as datasets are added. The symmetry demonstrates control of the gamma-reference gauge; it does not establish that the best-first profile is biologically more accurate. Best-N and worst-N memberships are disjoint through N=40, share 46 datasets at N=80, and are identical at N=114.</figcaption></figure>
<p>At N={reporting_n}, p=5 raises agreement with the equal N=2 anchor by {best_p5_report-best_equal_report:+.3f} on the best-first path and {worst_p5_report-worst_equal_report:+.3f} on the worst-first path. Meanwhile, the equal-reference extremes increase from {direct_equal_2:.3f} at N=2 to {direct_equal_40:.3f} at the largest disjoint size N=40 and {direct_equal_80:.3f} at N=80. The first increase is consistent with a common component becoming more visible as independent extreme collections grow; the N=80 comparison is additionally helped by 46 shared datasets. None of these anchors is biological ground truth.</p>

<h2>How much does L<sub>t</sub> change at each addition?</h2>
<figure><img src="adjacent_stability.svg" alt="PCC of the shared profile between consecutive cumulative collection sizes">
<figcaption><b>Figure 3 | Consecutive-size concordance of the shared profile.</b> Each point compares exactly the same 714 transcripts between N=2 and N=5, then N=5 and N=10, under the same selection direction and reference policy. Future transitions are shown as pending and are never replaced by a non-adjacent comparison.</figcaption></figure>
<p>Under equal weighting, adjacent PCC rises from {adjacent_best_equal_25:.3f} for 2→5 to {adjacent_best_equal_510:.3f} for 5→10 on the best-first path, and from {adjacent_worst_equal_25:.3f} to {adjacent_worst_equal_510:.3f} on the worst-first path. Stronger directional references further increase adjacent agreement. This local smoothness does not contradict the decline relative to N=2: several comparatively small transitions can accumulate into substantial anchor drift, so the adjacent and fixed-anchor plots answer complementary questions.</p>

<h2>Does the ranking stabilize or merely move the profile?</h2>
<figure><img src="policy_diagnostics.svg" alt="Anchor gains and same-size policy sensitivity">
<figcaption><b>Figure 4 | Benefit relative to the equal anchor and cost in policy sensitivity.</b> The top row is the transcript-paired change in anchor PCC relative to equal weighting. The bottom row compares score-weighted and equal profiles trained on exactly the same datasets. A high anchor gain accompanied by lower same-N agreement means that the reference changes the selected decomposition rather than adding independent evidence for correctness.</figcaption></figure>
<p>The worst-first p=1 fit at N=2 has PCC {worst_p1_sens_2:.3f} with its equal counterpart even though its reference is almost uniform (N<sub>eff</sub>/N={fraction('worst_first','score_p1',2):.3f}). This disproportionate response could reflect sensitivity of the learned decomposition or of non-convex training to a small gauge change. With one seed, the current experiment cannot distinguish those explanations.</p>

<h2>What can and cannot be claimed now?</h2>
<ul>
  <li><b>Supported:</b> dataset selection changes the learned shared profile; the best-two and worst-two solutions are nearly unrelated under equal weighting.</li>
  <li><b>Supported conditionally on seed 42:</b> increasing p makes each directional trajectory retain more of its own extreme-anchor solution through N={reporting_n}.</li>
  <li><b>Not supported:</b> that score weighting recovers a truer biological <b>L<sub>t</sub></b>. The worst-oriented reference stabilizes the worst anchor too, and N=2 is not a truth target.</li>
  <li><b>Partially unresolved:</b> at N=114 only equal and best-oriented p=1 are complete; the five stronger or opposite-direction full-collection fits remain unavailable.</li>
  <li><b>Not estimated:</b> training-seed uncertainty. Transcript bootstrap intervals quantify heterogeneity over this held-out cohort only.</li>
</ul>
<p>A stronger directional conclusion requires concentration-matched controls. One clean implementation is &pi;<sub>d</sub>&prop;exp(&plusmn;&beta;z<sub>d</sub>) with a separate &beta; in each direction calibrated to the same target N<sub>eff</sub>/N at every N. This separates <i>which</i> datasets receive mass from <i>how concentrated</i> the mass is. Because <code>quality_rank_score</code> is a sum of component ranks, treating score ratios as cardinal distances is also an assumption; percentile or standardized-component scores would make that assumption easier to defend.</p>

<h2>Completion and exact numerical results</h2>
<h3>Planned task availability</h3>{availability_html}
<h3>Primary equal-N=2-anchor summary</h3>{core_table}
<h3>Selected reference diagnostics</h3>{concentration_table}
<p class="small">PCC is computed transcript by transcript over identical full-CDS coordinates and then averaged. Undefined PCCs are excluded and counted; all currently plotted groups have their exact valid count in the CSV summaries. No unavailable endpoint is replaced by another size.</p>

<h2>Audit and downloads</h2>
<p>The analysis verified the task checksum, all frozen input and configuration checksums, the single fixed train/validation/test partition, runtime configurations, realised gamma-reference names and weights, prediction hashes, common test ID hash, mean-one profile constraint, and raw codon-coordinate alignment. Checkpoint tensors were not reloaded; the report analyses the frozen best-validation-loss prediction exports.</p>
<p class="downloads"><a href="availability.csv">Availability</a><a href="reference_geometry.csv">Reference geometry</a><a href="common_equal_anchor_metrics.csv">Per-transcript anchor PCC</a><a href="common_equal_anchor_summary.csv">Anchor summary</a><a href="adjacent_metrics.csv">Per-transcript adjacent PCC</a><a href="adjacent_summary.csv">Adjacent summary</a><a href="policy_sensitivity_summary.csv">Policy sensitivity</a><a href="best_vs_worst_summary.csv">Best vs worst</a><a href="analysis_manifest.json">Analysis manifest</a></p>
<p class="small">Generated {datetime.now(timezone.utc).isoformat()} with {replicates:,} deterministic transcript-bootstrap replicates per reported mean.</p>
</main></body></html>'''
    (out / "analysis_report.html").write_text(document)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260918)
    args = parser.parse_args(argv)
    root = args.experiment_root.expanduser().resolve()
    if args.bootstrap_replicates < 200:
        parser.error("--bootstrap-replicates must be at least 200")

    manifest, configs, weights, concentration, test_ids, audit_rows = audit_design(root)
    profiles, availability, profile_audit = collect_exports(
        root, manifest, configs, weights, test_ids
    )
    lookup = task_lookup(manifest, profiles)
    sizes = list(map(int, manifest["sizes"]))
    anchor, own, adjacent, sensitivity, direct = compute_metrics(lookup, test_ids, sizes)
    anchor_summary = summarize_pcc(
        anchor, ["selection_direction", "reference_policy", "anchor_policy", "anchor_N", "N"],
        args.bootstrap_replicates, args.bootstrap_seed,
    )
    own_summary = summarize_pcc(
        own, ["selection_direction", "reference_policy", "anchor_policy", "anchor_N", "N"],
        args.bootstrap_replicates, args.bootstrap_seed,
    )
    adjacent_summary = summarize_pcc(
        adjacent, ["selection_direction", "reference_policy", "N_a", "N_b"],
        args.bootstrap_replicates, args.bootstrap_seed,
    )
    sensitivity_summary = summarize_pcc(
        sensitivity, ["selection_direction", "reference_policy", "N"],
        args.bootstrap_replicates, args.bootstrap_seed,
    )
    direct_summary = summarize_pcc(
        direct, ["reference_policy", "N"],
        args.bootstrap_replicates, args.bootstrap_seed,
    )
    effect_metrics, effect_summary = ranking_effects(
        anchor, args.bootstrap_replicates, args.bootstrap_seed
    )

    out = artifact_directory("real_data", root)
    out.mkdir(exist_ok=True)
    outputs = {
        "availability": availability,
        "profile_audit": profile_audit,
        "artifact_audit": pd.DataFrame(audit_rows),
        "reference_geometry": concentration,
        "common_equal_anchor_metrics": anchor,
        "common_equal_anchor_summary": anchor_summary,
        "own_policy_anchor_metrics": own,
        "own_policy_anchor_summary": own_summary,
        "adjacent_metrics": adjacent,
        "adjacent_summary": adjacent_summary,
        "policy_sensitivity_metrics": sensitivity,
        "policy_sensitivity_summary": sensitivity_summary,
        "best_vs_worst_metrics": direct,
        "best_vs_worst_summary": direct_summary,
        "ranking_effect_metrics": effect_metrics,
        "ranking_effect_summary": effect_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(out / f"{name}.csv", index=False)

    plot_reference_geometry(concentration, sizes, out)
    if not anchor_summary.empty and not direct_summary.empty:
        plot_partial_results(anchor_summary, direct_summary, sizes, out)
    if not adjacent_summary.empty:
        plot_adjacent_stability(adjacent_summary, sizes, out)
    if not effect_summary.empty and not sensitivity_summary.empty:
        plot_policy_diagnostics(effect_summary, sensitivity_summary, sizes, out)
    render_report(
        root, out, manifest, availability, profile_audit, concentration,
        anchor_summary, own_summary, adjacent_summary, sensitivity_summary,
        direct_summary, effect_summary, test_ids, sizes,
        args.bootstrap_replicates,
    )

    artifact_hashes = {
        path.name: sha256(path)
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != "analysis_manifest.json"
    }
    analysis_manifest = dict(
        experiment_design=DESIGN,
        experiment_manifest_sha256=sha256(root / "experiment_manifest.json"),
        analysis_code=str(Path(__file__).resolve()),
        analysis_code_sha256=sha256(Path(__file__).resolve()),
        generated_utc=datetime.now(timezone.utc).isoformat(),
        planned_models=len(manifest["tasks"]),
        validated_models=int(availability.status.eq("validated_predictions").sum()),
        invalid_models=int(availability.status.eq("invalid_artifacts").sum()),
        common_test_transcripts=len(test_ids),
        common_test_transcript_hash=transcript_id_hash(test_ids),
        completed_sizes=sorted(map(int, availability.loc[
            availability.status.eq("validated_predictions"), "N"
        ].unique())),
        bootstrap=dict(
            unit="transcript", replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
            scope="conditional on the completed seed-42 fits; not training-seed uncertainty",
        ),
        primary_estimand="mean per-transcript PCC with the direction-specific equal-reference N=2 profile",
        outputs=artifact_hashes,
    )
    write_json(out / "analysis_manifest.json", analysis_manifest)
    print(
        f"Validated {analysis_manifest['validated_models']}/{analysis_manifest['planned_models']} fits; "
        f"report: {out / 'analysis_report.html'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
