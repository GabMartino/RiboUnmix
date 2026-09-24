#!/usr/bin/env python3
"""CPU-only setup audit against saved data/configs; no training or checkpoints.

Run after run_synthetic_gamma_unity.py --dry-run. Uses the production split and
constructor to check historical IDs and gamma/alpha gradient paths, and asks
Hydra to resolve the generated commands with --cfg job (never trainer.fit).
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from run_synthetic_gamma_unity import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/synthetic_gamma_unity_C2_seed42")
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    root = args.output_root.resolve()
    plan_path = root / "experiment_manifest.json"
    plan = json.loads(plan_path.read_text())
    from main_ribounmix_multidataset import fixed_common_validation_split
    from Utils.external_transcript_split import assert_expected_transcript_split
    from Utils.campaign_training import initialization_audit

    configurations = [
        yaml.safe_load(Path(task["config"]).read_text()) for task in plan["tasks"]
    ]
    checks = []
    validation_sets = {}
    for task, cfg in zip(plan["tasks"], configurations):
        # Regenerate from each depth-specific universe; never assume that
        # differently sampled count files imply the same historical split.
        train, validation, _ = fixed_common_validation_split(
            sequences_path=cfg["paths"]["sequences_path"],
            split_universe_dataset_paths=list(
                cfg["dataset_config"]["dataset_path"].values()
            ),
            validation_frac=cfg["split"]["validation_frac"],
            random_seed=cfg["experiment"]["seed"],
            validation_weight_bins=cfg["split"]["validation_weight_bins"],
            max_cds_codons=cfg["data"]["max_cds_codons"],
        )
        assert_expected_transcript_split(cfg["split"]["expected_manifest"], train_ids=train,
            validation_ids=validation, experiment_datasets=cfg["experiment"]["dataset"])
        full_cfg = copy.deepcopy(cfg)
        full_cfg["model"]["mean_correction"] = "learned"
        control_cfg = copy.deepcopy(cfg)
        control_cfg["model"]["mean_correction"] = "unity"
        full, control = initialization_audit(full_cfg), initialization_audit(control_cfg)
        assert full["parameter_sha256"] == control["parameter_sha256"]
        result = subprocess.run(task["command"] + ["--cfg", "job"], cwd=ROOT,
                                capture_output=True, text=True, check=True)
        resolved = yaml.safe_load(result.stdout)
        assert resolved["model"]["mean_correction"] == cfg["model"]["mean_correction"]
        assert resolved["prediction"]["checkpoint_variants"] == ["best_val_loss"]
        validation_hash = hashlib.sha256(
            "\n".join(sorted(map(str, validation))).encode("utf-8")
        ).hexdigest()
        validation_sets[task["task_id"]] = set(map(str, validation))
        checks.append(dict(task=task["task_id"], N=task["N"],
                           depth_reads_per_codon=task.get("depth_reads_per_codon"),
                           dataset_config=cfg["dataset_config"]["_name_"],
                           train_ids_exact=True, validation_ids_exact=True,
                           validation_transcript_id_hash=validation_hash,
                           train_count=len(train), validation_count=len(validation),
                           full_initialization=full, unity_initialization=control,
                           hydra_config_resolved=True))
        print(f"PASS {task['task_id']}: historical split, identical initial parameters, gamma=1, alpha/shared gradients, Hydra", flush=True)
    # Check frozen inputs and code again; input equality to historical original
    # bytes is not asserted, since the historical run did not record these hashes.
    for path, digest in plan["input_sha256"].items():
        assert sha256(path) == digest, path
    for path, digest in plan["code_sha256"].items():
        assert sha256(ROOT / path) == digest, path
    pairwise_validation_overlap = []
    task_ids = list(validation_sets)
    for index, task_a in enumerate(task_ids):
        for task_b in task_ids[index + 1:]:
            pairwise_validation_overlap.append(dict(
                task_a=task_a,
                task_b=task_b,
                intersection=len(validation_sets[task_a] & validation_sets[task_b]),
            ))
    common_validation = (
        len(set.intersection(*validation_sets.values())) if validation_sets else 0
    )
    summary = dict(checks=checks,
                   cohort_alignment=dict(
                       common_validation_intersection=common_validation,
                       pairwise_validation_overlap=pairwise_validation_overlap,
                       interpretation=(
                           "Full-versus-unity comparisons are matched within depth. "
                           "Historical validation cohorts differ across depths, so "
                           "cross-depth transcript-paired inference is not supported."
                       ),
                   ),
                   max_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                   scope="CPU initialization/backward probes and exact data split; no GPU training",
                   gamma_tolerance="Exact torch.equal(gamma, ones)",
                   numerical_tolerance="Finite FP32 gradients; initialization hashes match byte-for-byte")
    audit_path = root / "setup_audit.json"
    write_json(audit_path, summary)
    manifest = dict(schema_version=1, claim_id="synthetic_gamma_unity_setup_only",
        repository=dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), dirty=True),
        command=[sys.executable, str(Path(__file__).relative_to(ROOT)), "--output-root", str(root)],
        environment=dict(software=[dict(name=p, version=importlib.metadata.version(p)) for p in ["torch", "lightning", "hydra-core", "numpy", "pyarrow"]],
                         hardware=f"CPU-only {platform.machine()}; numerical-library threads set to one by command environment"),
        mathematics=dict(assertion_tested="Shared-only configs preserve source training settings, reproduce historical folds and enforce gamma=1 with trainable alpha/shared heads",
            coefficient_domain="FP32 CPU probes; exact transcript identities and SHA-256 configuration checks",
            conventions="Existing production model, target-derived scale, mean-one L; no new split or simulator data",
            inputs=[dict(path=str(plan_path.relative_to(ROOT)), sha256=sha256(plan_path)),
                    dict(path=str(Path(__file__).relative_to(ROOT)), sha256=sha256(__file__))],
            bounds=dict(
                N=sorted({int(task["N"]) for task in plan["tasks"]}),
                depth=sorted({
                    float(task["depth_reads_per_codon"])
                    for task in plan["tasks"]
                    if task.get("depth_reads_per_codon") is not None
                }),
                training_seed=sorted({
                    int(yaml.safe_load(Path(task["config"]).read_text())["experiment"]["seed"])
                    for task in plan["tasks"]
                }),
                probe_positions=8,
                probe_transcripts=1,
                trained_epochs=0,
            ),
            non_claims=["No trained recovery metrics or BF16 GPU stability verified", "Historical implementation/data-byte identity not established", "Detached alpha context is not held identical after training"]),
        randomness=dict(used=True, generator="Production Lightning/PyTorch seed initialization; NumPy split sampling", seed=42),
        run=dict(started_at=started, runtime_seconds=time.monotonic()-start, exit_status=0),
        outputs=[dict(path=str(audit_path.relative_to(ROOT)), sha256=sha256(audit_path))],
        checks=["Every depth-specific historical train/validation list matched exactly", "Paired initial model parameters byte-identical", "Unity gamma exactly one; finite shared/alpha gradients and no context gradient", "Hydra configs resolved without launching training", "Prepared input/code hashes unchanged", "Cross-depth validation overlap recorded explicitly"],
        result="implementation and finite assertion verified in the stated range",
        residual_risks=plan["limitations"] + [
            "Historical validation identities differ across depth; compare learned versus unity within depth.",
            "CPU-only check; cluster environment and GPU memory not tested",
        ])
    write_json(root / "computation_audit.json", manifest)
    print(f"Setup audit saved; peak parent-process RSS {summary['max_rss_gb']:.2f} GiB. No training launched.")


if __name__ == "__main__":
    main()
