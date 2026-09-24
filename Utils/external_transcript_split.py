"""Validation for externally designed transcript-level train/validation/test folds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from Utils.reliability_references import transcript_id_hash


MANIFEST_VERSION = 1


def assert_expected_transcript_split(
    path: str | Path,
    *,
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
    experiment_datasets: Sequence[str],
) -> None:
    """Verify a regenerated legacy train/validation split without changing it.

    Used by matched ablations of historical synthetic runs, which have no
    independent test fold. Require the saved order too: sampler inputs should
    not silently change while transcript membership appears unchanged.
    """
    expected = json.loads(Path(path).read_text(encoding="utf-8"))
    if sorted(expected["experiment_datasets"]) != sorted(experiment_datasets):
        raise ValueError("Expected split belongs to different experiment datasets.")
    for field, actual in (("train_ids", train_ids), ("validation_ids", validation_ids)):
        saved = _unique_ids(expected[field], field=field)
        if saved != list(map(str, actual)):
            raise ValueError(
                f"Regenerated {field} differs from the historical split (IDs or order). "
                "Refusing an unmatched ablation; check data and split-code provenance."
            )
    if set(expected["train_ids"]) & set(expected["validation_ids"]):
        raise ValueError("Expected training and validation folds overlap.")


def _unique_ids(payload: Any, *, field: str) -> list[str]:
    if not isinstance(payload, list):
        raise TypeError(f"External split field {field!r} must be a JSON list.")
    values = list(map(str, payload))
    if len(values) != len(set(values)):
        raise ValueError(f"External split field {field!r} contains duplicate IDs.")
    if not values:
        raise ValueError(f"External split field {field!r} is empty.")
    return values


def load_external_transcript_split(
    path: str | Path,
    *,
    panel_name: str,
    experiment_datasets: Sequence[str],
) -> tuple[list[str], list[str], list[str], dict[str, Any]]:
    """Load one panel's folds and enforce dataset/fold identity contracts."""
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(payload.get("manifest_version", -1)) != MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported external split manifest version in {manifest_path}."
        )

    panels = payload.get("panels")
    train_by_panel = payload.get("panel_train_eligible_ids")
    if not isinstance(panels, dict) or panel_name not in panels:
        raise KeyError(
            f"External split manifest has no dataset definition for {panel_name!r}."
        )
    if not isinstance(train_by_panel, dict) or panel_name not in train_by_panel:
        raise KeyError(
            f"External split manifest has no training fold for {panel_name!r}."
        )

    expected_datasets = list(map(str, panels[panel_name]))
    actual_datasets = list(map(str, experiment_datasets))
    if set(expected_datasets) != set(actual_datasets) or len(expected_datasets) != len(
        actual_datasets
    ):
        raise ValueError(
            f"Experiment datasets do not equal external panel {panel_name}: "
            f"expected={expected_datasets}, actual={actual_datasets}."
        )

    train_ids = _unique_ids(train_by_panel[panel_name], field=f"{panel_name}.train")
    validation_by_panel = payload.get("panel_validation_ids")
    if validation_by_panel is not None:
        if not isinstance(validation_by_panel, dict) or panel_name not in validation_by_panel:
            raise KeyError(
                "External split manifest declares panel_validation_ids but has no "
                f"validation fold for {panel_name!r}."
            )
        validation_ids = _unique_ids(
            validation_by_panel[panel_name],
            field=f"{panel_name}.validation",
        )
    else:
        # Backwards-compatible Experiment-1 behavior: one validation fold is
        # shared by every independent panel.
        validation_ids = _unique_ids(
            payload.get("common_validation_ids"), field="common_validation_ids"
        )
    test_ids = _unique_ids(payload.get("common_test_ids"), field="common_test_ids")

    folds = {
        "train": set(train_ids),
        "validation": set(validation_ids),
        "test": set(test_ids),
    }
    overlaps = {
        "train_validation": sorted(folds["train"] & folds["validation"]),
        "train_test": sorted(folds["train"] & folds["test"]),
        "validation_test": sorted(folds["validation"] & folds["test"]),
    }
    nonempty_overlaps = {key: value for key, value in overlaps.items() if value}
    if nonempty_overlaps:
        raise ValueError(
            "External transcript folds overlap: "
            + ", ".join(
                f"{key}={values[:5]}" for key, values in nonempty_overlaps.items()
            )
        )

    expected_hashes = payload.get("fold_id_hashes", {})
    for fold_name, ids in (("validation", validation_ids), ("test", test_ids)):
        if fold_name == "validation" and validation_by_panel is not None:
            panel_hashes = expected_hashes.get("validation_by_panel", {})
            expected = (
                panel_hashes.get(panel_name)
                if isinstance(panel_hashes, dict)
                else None
            )
        else:
            expected = expected_hashes.get(fold_name)
        observed = transcript_id_hash(ids)
        if expected is not None and str(expected) != observed:
            raise ValueError(
                f"External {fold_name} ID hash does not match its manifest."
            )

    return train_ids, validation_ids, test_ids, payload
