#!/usr/bin/env python3
"""Verify the public smoke checkpoint and its frozen validation export."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


EXAMPLE_ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_close(name: str, observed: float, expected: float, tolerance: float) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(
            f"{name} differs: observed={observed:.12g}, expected={expected:.12g}, "
            f"absolute tolerance={tolerance:.3g}."
        )


def _verify_hashes(manifest: dict[str, Any]) -> None:
    for relative_path, metadata in manifest["files"].items():
        path = EXAMPLE_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing reference artifact: {path}")
        observed_size = path.stat().st_size
        if observed_size != int(metadata["bytes"]):
            raise AssertionError(
                f"Size mismatch for {relative_path}: {observed_size} != "
                f"{metadata['bytes']}."
            )
        observed_hash = sha256(path)
        if observed_hash != metadata["sha256"]:
            raise AssertionError(
                f"SHA-256 mismatch for {relative_path}: {observed_hash} != "
                f"{metadata['sha256']}."
            )


def _prediction_statistics(prediction_path: Path) -> dict[str, float | int]:
    frame = pd.read_parquet(prediction_path)
    required = {
        "transcript_id",
        "dataset_id",
        "mask",
        "target",
        "mu",
        "L_bio",
        "gamma",
        "log_gamma",
        "log_sigma",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Prediction export is missing columns: {missing}")

    pcc_values: list[float] = []
    rmse_values: list[float] = []
    maximum_l_mean_error = 0.0
    maximum_gamma_position_gauge_error = 0.0
    minimum_alpha = math.inf
    maximum_alpha = -math.inf

    for row in frame.itertuples(index=False):
        mask = np.asarray(row.mask, dtype=bool)
        target = np.asarray(row.target, dtype=np.float64)[mask]
        mean = np.asarray(row.mu, dtype=np.float64)[mask]
        shared = np.asarray(row.L_bio, dtype=np.float64)[mask]
        log_gamma = np.asarray(row.log_gamma, dtype=np.float64)[mask]
        alpha = np.exp(np.asarray(row.log_sigma, dtype=np.float64)[mask])

        arrays = (target, mean, shared, log_gamma, alpha)
        if any(array.ndim != 1 or array.size == 0 for array in arrays):
            raise AssertionError("Every scored prediction vector must be non-empty and 1D.")
        if any(not np.isfinite(array).all() for array in arrays):
            raise AssertionError("Prediction export contains non-finite scored values.")
        if np.any(mean <= 0.0) or np.any(shared <= 0.0) or np.any(alpha <= 0.0):
            raise AssertionError("Model means, shared profiles, and alpha must be positive.")

        if np.std(target) > 0.0 and np.std(mean) > 0.0:
            pcc_values.append(float(np.corrcoef(target, mean)[0, 1]))
        rmse_values.append(float(np.sqrt(np.mean(np.square(target - mean)))))
        maximum_l_mean_error = max(
            maximum_l_mean_error, abs(float(np.mean(shared)) - 1.0)
        )
        maximum_gamma_position_gauge_error = max(
            maximum_gamma_position_gauge_error, abs(float(np.mean(log_gamma)))
        )
        minimum_alpha = min(minimum_alpha, float(np.min(alpha)))
        maximum_alpha = max(maximum_alpha, float(np.max(alpha)))

    maximum_shared_duplicate_error = 0.0
    maximum_reference_constraint_error = 0.0
    for transcript_id, group in frame.groupby("transcript_id", sort=True):
        if len(group) != 2:
            raise AssertionError(
                f"Expected two dataset rows for {transcript_id}, found {len(group)}."
            )
        shared_profiles = [np.asarray(value, dtype=np.float64) for value in group.L_bio]
        maximum_shared_duplicate_error = max(
            maximum_shared_duplicate_error,
            float(np.max(np.abs(shared_profiles[0] - shared_profiles[1]))),
        )
        masks = [np.asarray(value, dtype=bool) for value in group["mask"]]
        log_gammas = [np.asarray(value, dtype=np.float64) for value in group.log_gamma]
        if not np.array_equal(masks[0], masks[1]):
            raise AssertionError(f"Dataset masks differ for transcript {transcript_id}.")
        valid = masks[0]
        equal_reference_log_mean = 0.5 * (
            log_gammas[0][valid] + log_gammas[1][valid]
        )
        maximum_reference_constraint_error = max(
            maximum_reference_constraint_error,
            float(np.max(np.abs(equal_reference_log_mean))),
        )

    return {
        "prediction_rows": int(len(frame)),
        "validation_transcripts": int(frame.transcript_id.nunique()),
        "defined_row_pcc": int(len(pcc_values)),
        "mean_row_pcc": float(np.mean(pcc_values)),
        "median_row_pcc": float(np.median(pcc_values)),
        "mean_row_rmse": float(np.mean(rmse_values)),
        "maximum_shared_mean_one_error": maximum_l_mean_error,
        "maximum_gamma_position_gauge_error": maximum_gamma_position_gauge_error,
        "maximum_shared_duplicate_error": maximum_shared_duplicate_error,
        "maximum_equal_reference_constraint_error": maximum_reference_constraint_error,
        "minimum_alpha": minimum_alpha,
        "maximum_alpha": maximum_alpha,
    }


def verify(tolerance: float) -> dict[str, Any]:
    manifest_path = EXAMPLE_ROOT / "reference_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _verify_hashes(manifest)

    checkpoint_path = EXAMPLE_ROOT / manifest["checkpoint_path"]
    # Hash verification happens before deserialization. This checkpoint is a
    # trusted artifact committed with the repository, not an arbitrary input.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", {})
    if not state_dict:
        raise AssertionError("Checkpoint has no model state_dict.")
    nonfinite_parameters = [
        name
        for name, value in state_dict.items()
        if torch.is_tensor(value)
        and (value.is_floating_point() or value.is_complex())
        and not bool(torch.isfinite(value).all())
    ]
    if nonfinite_parameters:
        raise AssertionError(
            f"Checkpoint contains non-finite parameters: {nonfinite_parameters[:5]}"
        )

    prediction_path = EXAMPLE_ROOT / manifest["prediction_path"]
    statistics = _prediction_statistics(prediction_path)
    for name, expected in manifest["expected_validation"].items():
        observed = statistics[name]
        if isinstance(expected, int):
            if observed != expected:
                raise AssertionError(f"{name} differs: {observed} != {expected}.")
        else:
            _assert_close(name, float(observed), float(expected), tolerance)

    if int(checkpoint.get("epoch", -1)) != int(manifest["checkpoint_epoch"]):
        raise AssertionError("Checkpoint epoch does not match the reference manifest.")
    if int(checkpoint.get("global_step", -1)) != int(manifest["checkpoint_global_step"]):
        raise AssertionError("Checkpoint global step does not match the reference manifest.")

    checkpoint_tensor_entries = sum(
        int(torch.is_tensor(value)) for value in state_dict.values()
    )
    return {
        "status": "ok",
        "checkpoint_state_entries": len(state_dict),
        "checkpoint_tensor_entries": checkpoint_tensor_entries,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_global_step": int(checkpoint["global_step"]),
        **statistics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--absolute-tolerance",
        type=float,
        default=1.0e-9,
        help="Tolerance for metrics recomputed from the frozen Parquet export.",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    report = verify(args.absolute_tolerance)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
