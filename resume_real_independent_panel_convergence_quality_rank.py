#!/usr/bin/env python3
"""Resume the saved ranked-panel design without recreating panels or splits.

Uses the existing full-state resume/checkpoint selection path. Unstarted panels
start from their saved resolved configuration; completed exports are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text())


def preflight(root: Path) -> dict:
    # This exact import failed before panels 3/4 could initialize their models.
    from Utils.transcript_batch_metadata import ValidatedTranscriptMetadata
    from Utils.reliability_references import transcript_id_hash
    from analyses.analyze_real_panel_convergence import _locate_panel_prediction

    assert ValidatedTranscriptMetadata is not None
    manifest = read_json(root / "panel_manifest.json")
    strategy = manifest["gamma_reference_strategy"]
    if strategy["weighting"] != "quality_rank" or strategy["quality_rank_power"] != 1.0:
        raise ValueError("This continuation is for the saved quality_rank, power=1 design.")
    ranking = ROOT / "Datasets/data/HEK_riboseq_profile_quality_rank.tsv"
    digest = hashlib.sha256(ranking.read_bytes()).hexdigest()
    if digest != strategy["ranking_table_sha256"]:
        raise ValueError(f"Frozen ranking changed: {ranking}; expected {strategy['ranking_table_sha256']}, got {digest}")
    table = pd.read_csv(ranking, sep="\t").set_index("dataset")
    rank_max = float(table.quality_rank.max())
    common = read_json(root / "common_split_manifest.json")
    test_hash = transcript_id_hash(common["common_test_ids"])
    rows = []
    for panel, datasets in sorted(manifest["panels"].items()):
        directory = root / panel
        run = read_json(directory / "run_manifest.json")
        config = yaml.safe_load((directory / "resolved_config.yaml").read_text())
        split = read_json(directory / "split_manifest.json")
        if datasets != run["selected_datasets"] or datasets != config["experiment"]["dataset"]:
            raise ValueError(f"{panel}: inconsistent dataset membership")
        if (transcript_id_hash(split["test_ids"]) != test_hash
                or run["common_test_id_hash"] != test_hash
                or set(split["train_ids"]) & (set(common["common_test_ids"]) | set(common["common_validation_ids"]))):
            raise ValueError(f"{panel}: inconsistent or leaking transcript split")
        ref = config["model"]["gamma_centering"]["reference"]
        if ref["weighting"] != "quality_rank" or ref["quality_rank_power"] != 1.0:
            raise ValueError(f"{panel}: saved configuration is not the requested weighting")
        pi = np.array([run["fixed_gamma_reference"]["pi"][d] for d in datasets])
        weights = (rank_max + 1 - table.loc[datasets, "quality_rank"].to_numpy(float)) / rank_max
        np.testing.assert_allclose(pi, weights / weights.sum(), rtol=1e-10, atol=1e-12)
        artifact, _, reason = _locate_panel_prediction(directory)
        if artifact is not None:
            # Do not let a truncated download silently count as a complete run.
            with pq.ParquetFile(artifact) as reader:
                if reader.metadata.num_rows == 0:
                    raise ValueError(f"{panel}: empty completed prediction {artifact}")
        rows.append(dict(panel=panel, completed_prediction=artifact is not None,
                         prediction=str(artifact) if artifact else None, reason=reason,
                         checkpoints_found=len(list((directory / "checkpoints").rglob("*.ckpt"))),
                         pi_sum=float(pi.sum()), config_sha256=hashlib.sha256(
                             (directory / "resolved_config.yaml").read_bytes()).hexdigest()))
    code_files = ["Utils/transcript_batch_metadata.py", "resume_real_experiment_from_checkpoints.py",
                  "main_ribounmix_multidataset.py", "Models/RiboUnmixModel/DatasetBiasSubmodel.py",
                  "Dataloaders/RiboUnmixMultiDataset/RiboUnmixMultiDataset.py"]
    return dict(run_root=str(root), ranking_table=str(ranking), ranking_sha256=digest,
                common_test_count=len(common["common_test_ids"]), common_test_hash=test_hash,
                panels=rows, current_source_sha256={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
                                                  for f in code_files},
                code_version_caveat="Original runs did not record a git commit; saved settings do not verify identical source code.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = args.run_root.resolve()
    audit = preflight(root)
    print(json.dumps(audit, indent=2), flush=True)
    (root / "quality_rank_resume_preflight.json").write_text(json.dumps(audit, indent=2) + "\n")
    from resume_real_experiment_from_checkpoints import main as resume
    command = ["--run-root", str(root), "--gpus", args.gpus,
               "--throughput-profile", "unchanged", "--use-saved-resolved-config"]
    if args.dry_run:
        command.append("--dry-run")
    return resume(command)


if __name__ == "__main__":
    sys.exit(main())
