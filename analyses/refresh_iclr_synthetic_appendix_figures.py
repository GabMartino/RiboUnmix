#!/usr/bin/env python3
"""Refresh two appendix figures using one saved transcript and audited summaries.

Run from the repository root:
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    .venv/bin/python analyses/refresh_iclr_synthetic_appendix_figures.py

Outputs are staged for visual inspection; this script never copies them into
the manuscript or loads a trained model. The cohort statistics are reused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(key, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from analyses.plot_synthetic_hierarchy import (
    load_base_profile, load_bias, load_config, load_counts,
    pearson, plot_profile_layers, resolve_path, sha256,
)
from analyses.analyze_synthetic_observation_layers import plot_observation_agreement


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "analyses/artifacts/manuscript_structure_revision_20260921/figures",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    hierarchy_path = ROOT / "analyses/configs/synthetic_hierarchy_figures.yaml"
    observation_path = ROOT / "analyses/configs/synthetic_observation_layers.yaml"
    config = load_config(hierarchy_path)
    data = config["data"]
    example = config["examples"]["profile_layers"]
    base, metadata = load_base_profile(
        example["transcript_id"],
        kinetic_path=resolve_path(data["kinetic_profile"]),
        occupancy_path=resolve_path(data["occupancy_replicates"]),
        sequence_path=resolve_path(data["sequence_annotations"]),
        permitted_stops=set(config["validation"]["permitted_terminal_codons"]),
    )
    condition = example["condition"]
    depth = next(d for d in config["depths"] if d["slug"] == example["depth_slug"])
    bias_path = resolve_path(data["bias_annotation_template"].format(condition=condition))
    count_path = resolve_path(data["count_template"].format(
        condition=condition, depth_slug=depth["slug"],
    ))
    bias = load_bias(bias_path, base.transcript_id, condition)
    counts = load_counts(count_path, base.transcript_id, condition,
                         depth["slug"], float(depth["value"]))
    mean_deviations = {
        name: abs(float(values.mean()) - 1)
        for name, values in (("K", base.K), ("q1", base.q1),
                             ("q2", base.q2), ("qbar", base.qbar))
    }
    assert max(mean_deviations.values()) <= config["validation"]["mean_one_absolute_tolerance"]
    assert bias.multiplier.shape == counts.rep1.shape == base.K.shape
    assert np.array_equal(counts.rep1, np.floor(counts.rep1))
    assert np.array_equal(counts.rep2, np.floor(counts.rep2))
    fingerprints = {
        m["riboart.source_run_fingerprint"]
        for m in (metadata["kinetic"], bias.metadata, counts.metadata)
    }
    assert len(fingerprints) == 1

    # Check against the existing example audit before changing its appearance.
    audit_path = ROOT / "analyses/artifacts/synthetic/hierarchy/synthetic_K_q_representative_examples.csv"
    audit = pd.read_csv(audit_path)
    reference = audit.loc[
        audit.transcript_id.eq(base.transcript_id) & audit.observation_effect.eq(condition)
    ].iloc[0]
    for name, value in (
        ("q1_vs_q2_pearson", pearson(base.q1, base.q2)),
        ("qbar_vs_K_pearson", pearson(base.qbar, base.K)),
    ):
        np.testing.assert_allclose(value, reference[name], rtol=0, atol=1e-12)
    profile_source = pd.DataFrame({
        "psite_index": base.positions, "K": base.K,
        "q1": base.q1, "q2": base.q2, "qbar": base.qbar,
        "bias_multiplier": bias.multiplier, "affected": bias.affected,
        "expected_per_depth": base.qbar * bias.multiplier,
        "Y1_per_depth": counts.rep1 / depth["value"],
        "Y2_per_depth": counts.rep2 / depth["value"],
    })
    profile_source.to_csv(output / "profile_layers_source.csv", index=False)
    paths = plot_profile_layers(
        base, bias, counts,
        label=next(c["label"] for c in config["conditions"] if c["key"] == condition),
        colors=config["colors"], plot_config=config["plot"],
        output_directory=output, stem=config["outputs"]["profile_layers_stem"],
    )
    observation = yaml.safe_load(observation_path.read_text())
    summary_path = resolve_path(observation["outputs"]["results_directory"]) / "observation_layer_agreement_summary.csv"
    summary = pd.read_csv(summary_path)
    paths.extend(plot_observation_agreement(
        summary, config=observation, output_directory=output,
    ))
    summary.to_csv(output / "observation_layer_source.csv", index=False)
    manifest = {
        "purpose": "Layout and notation revision; existing numerical comparisons retained",
        "transcript_id": base.transcript_id,
        "condition": condition,
        "nominal_depth": depth["value"],
        "positions": len(base.positions),
        "coordinates": "Saved zero-based sense-codon indices; terminal entry excluded",
        "mean_one_max_deviations": mean_deviations,
        "example_PCC_matches_existing_audit_atol": 1e-12,
        "profile_values_sha256": hashlib.sha256(
            profile_source.to_csv(index=False).encode()
        ).hexdigest(),
        "inputs": [
            {"path": str(p.relative_to(ROOT)), "sha256": sha256(p)}
            for p in (hierarchy_path, observation_path, summary_path, audit_path)
        ],
        "outputs": [{"path": str(p), "sha256": sha256(p)} for p in paths],
    }
    (output / "render_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"PASS: {len(base.positions)} aligned sense codons; existing example PCCs reproduced.")
    print(f"PASS: reused {len(summary)} rows of audited observation summaries.")
    print(f"Maximum positional-mean deviation: {max(mean_deviations.values()):.3g}")
    print(f"Staged figures: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
