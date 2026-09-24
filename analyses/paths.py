"""Canonical locations for derived analysis artifacts.

Experiment directories are immutable inputs. Analyses should write beside
other analyses, not back into a checkpoint/prediction tree.
"""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "analyses" / "artifacts"

_SOURCE_ALIASES = {
    "benchmark_20260829_194806": "mu_fit",
    "cumulative_selection_direction_quality_score_directional_seed42": (
        "cumulative_selection_quality_score"
    ),
    "cumulative_selection_direction_seed42": "cumulative_selection_direction",
    "cumulative_stability_fixed_cohort_seed42": "cumulative_stability_fixed_cohort",
    "cumulative_stability_seed42": "cumulative_stability",
    "four_panel_stability_seed42": "four_panel_stability",
    "four_quality_strata_seed42": "four_quality_strata",
}


def artifact_directory(
    domain: str,
    source_root: str | Path,
    *parts: str,
) -> Path:
    """Return the canonical artifact path associated with an experiment root."""

    if domain not in {"synthetic", "real_data", "benchmarking"}:
        raise ValueError(f"Unknown analysis domain: {domain!r}")
    source_name = Path(source_root).expanduser().resolve().name
    directory = ARTIFACT_ROOT / domain / _SOURCE_ALIASES.get(source_name, source_name)
    return directory.joinpath(*parts)
