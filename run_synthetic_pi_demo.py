#!/usr/bin/env python3
"""Run one small cross-depth π demonstration.

The panel deliberately assigns one depth to each different bias family, so a
depth-only ranking cannot cancel inside complete bias triplets.  The Slurm
wrapper runs this script nine times: three policies (equal, depth-ranked,
depth-reversed) at three seeds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import shlex
import subprocess
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parent
PANEL = (
    ("artificial_bias_3prime_aa", "0p25_per_codon"),
    ("artificial_bias_3prime_cc", "2_per_codon"),
    ("artificial_bias_3prime_gg", "20_per_codon"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _measured_panel_quality(
    dataset_names: list[str],
    sources: list[Path],
) -> dict[str, dict[str, float]]:
    """Summarize observable profile quality without using latent ground truth."""
    summaries: dict[str, dict[str, float]] = {}
    for dataset_name, source in zip(dataset_names, sources, strict=True):
        try:
            frame = pd.read_parquet(source, columns=["read_density", "coverage"])
        except Exception as error:
            raise RuntimeError(
                f"Could not read measured quality columns from {source}."
            ) from error
        density = pd.to_numeric(frame["read_density"], errors="coerce").dropna()
        coverage = pd.to_numeric(frame["coverage"], errors="coerce").dropna()
        if density.empty or coverage.empty:
            raise ValueError(
                f"Measured quality columns are empty or non-numeric in {source}."
            )
        summaries[dataset_name] = {
            "median_read_density": float(density.median()),
            "median_coverage": float(coverage.median()),
        }
    return summaries


def _policy_ranks(
    policy: str,
    measured_quality: dict[str, dict[str, float]],
) -> dict[str, int]:
    """Rank by observed median density, then optionally invert that ranking."""
    ordered = sorted(
        measured_quality,
        key=lambda name: (
            -measured_quality[name]["median_read_density"],
            name,
        ),
    )
    natural = {name: rank for rank, name in enumerate(ordered, start=1)}
    if policy == "reversed":
        count = len(natural)
        return {name: count + 1 - rank for name, rank in natural.items()}
    return natural


def _write_launcher_inputs(
    *,
    output_dir: Path,
    policy: str,
    seed: int,
) -> tuple[list[str], list[Path], Path, Path]:
    """Persist and validate the exact dataset-identity and pi specification.

    The multi-dataset loader defines an active dataset's identity from the
    Parquet filename stem.  Hydra mapping keys do not rename a Parquet file,
    so the encoding and quality table must use these same canonical names.
    """
    dataset_names = [bias for bias, _ in PANEL]
    if len(dataset_names) != len(set(dataset_names)):
        raise ValueError(f"The pi-demo panel contains duplicate datasets: {dataset_names}")

    sources: list[Path] = []
    for dataset_name, depth in PANEL:
        source = (
            ROOT
            / "Datasets/data/weighted_synthetic"
            / depth
            / f"{dataset_name}.parquet"
        ).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stem != dataset_name:
            raise ValueError(
                "Dataset identity must equal the Parquet filename stem: "
                f"configured={dataset_name!r}, stem={source.stem!r}, path={source}."
            )
        sources.append(source)

    measured_quality = _measured_panel_quality(dataset_names, sources)
    ranks_by_dataset = _policy_ranks(policy, measured_quality)
    ranks = [float(ranks_by_dataset[dataset_name]) for dataset_name in dataset_names]
    maximum_rank = max(ranks)
    quality_weights = [
        (maximum_rank - rank + 1.0) / maximum_rank for rank in ranks
    ]
    reference_raw_weights = (
        [1.0] * len(PANEL) if policy == "equal" else quality_weights
    )
    reference_weight_sum = sum(reference_raw_weights)
    reference_pi = [
        weight / reference_weight_sum for weight in reference_raw_weights
    ]

    inputs_dir = output_dir / "launcher_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    encoding = inputs_dir / "datasets.yaml"
    quality = inputs_dir / "quality.tsv"
    encoding.write_text(
        "\n".join(f"{name}: {index}" for index, name in enumerate(dataset_names))
        + "\n",
        encoding="utf-8",
    )
    quality.write_text(
        "dataset\tquality_rank\n"
        + "\n".join(
            f"{dataset_name}\t{ranks_by_dataset[dataset_name]}"
            for dataset_name in dataset_names
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "policy": policy,
        "seed": int(seed),
        "dataset_identity_rule": "canonical dataset name equals Parquet filename stem",
        "natural_quality_metric": (
            "descending median read_density computed from each measured input parquet"
        ),
        "policy_rank_transform": (
            "exact inversion of the measured ranking"
            if policy == "reversed"
            else "measured ranking unchanged"
        ),
        "gamma_centering_mode": "fixed_reference",
        "gamma_reference_weighting": (
            "equal" if policy == "equal" else "quality_rank"
        ),
        "gamma_reference_quality_rank_power": 0.0 if policy == "equal" else 1.0,
        "panel": [
            {
                "dataset": dataset_name,
                "read_depth": depth,
                "source_parquet": str(source),
                **measured_quality[dataset_name],
                "quality_rank": int(rank),
                "rank_derived_quality_weight": float(quality_weight),
                "expected_reference_raw_weight": float(reference_weight),
                "expected_reference_pi": float(pi),
            }
            for (dataset_name, depth), source, rank, quality_weight, reference_weight, pi
            in zip(
                PANEL,
                sources,
                ranks,
                quality_weights,
                reference_raw_weights,
                reference_pi,
                strict=True,
            )
        ],
        "encoding_path": str(encoding),
        "quality_table_path": str(quality),
    }
    (output_dir / "pi_demo_design_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return dataset_names, sources, encoding, quality


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("equal", "quality", "reversed"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/synthetic_pi_demo"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and persist the design, print the command, but do not train.",
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    label = f"pi_demo_{args.policy}_seed{args.seed}"
    out = args.output.expanduser().resolve() / label
    out.mkdir(parents=True, exist_ok=True)
    dataset_names, sources, encoding, quality = _write_launcher_inputs(
        output_dir=out,
        policy=args.policy,
        seed=args.seed,
    )
    dataset_overrides = [
        f"dataset_config.dataset_path.{dataset_name}={source}"
        for dataset_name, source in zip(dataset_names, sources, strict=True)
    ]
    command = [
        sys.executable,
        "main_ribounmix_synthetic.py",
        "dataset_config=synthetic_datasets_paths_20_per_codon",
        f"experiment.dataset=[{','.join(dataset_names)}]",
        f"split.master_dataset_universe=[{','.join(dataset_names)}]",
        "split.validation_weight_bins=2",
        "model.gamma_centering.mode=fixed_reference",
        "model.gamma_centering.reference.dataset_names="
        f"[{','.join(dataset_names)}]",
        "model.gamma_centering.reference.weighting="
        f"{'equal' if args.policy == 'equal' else 'quality_rank'}",
        "model.gamma_centering.reference.quality_rank_power="
        f"{0.0 if args.policy == 'equal' else 1.0}",
        "model.dataset_bias_params.num_datasets=3",
        f"experiment.seed={args.seed}",
        "synthetic_ground_truth.observed_path=null",
        "model.mass_conservation=false",
        "model.alpha_mode=learned",
        "loss.experiment_mode=standard_nb",
        "loss.nb_mean_gradient_beta=0.0",
        "experiment.from_checkpoint=false",
        "experiment.train=true",
        "experiment.predict=true",
        "trainer.devices=1",
        "trainer.use_distributed_sampler=false",
        "data.train_sampling_strategy=transcript_grouped_multidataset_pairs",
        "training.grouped_optimizer_batch.enabled=true",
        "training.grouped_optimizer_batch.auto_accumulate_grad_batches=true",
        "training.grouped_optimizer_batch.target_unique_transcripts_per_optimizer_step=32",
        "data.num_workers=0",
        "data.predict_num_workers=0",
        f"paths.encodings.datasets={encoding}",
        f"data.dataset_quality_ranking.path={quality}",
        "data.dataset_quality_ranking.strict=true",
        f"paths.checkpoints={out / 'checkpoints'}",
        f"paths.logs={out / 'logs'}",
        f"paths.results={out / 'results'}",
        f"hydra.run.dir={out / 'hydra'}",
        *dataset_overrides,
        *args.overrides,
    ]
    rendered_command = shlex.join(map(str, command))
    (out / "launch_command.sh").write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(ROOT))}\n{rendered_command}\n",
        encoding="utf-8",
    )
    print("Running:", rendered_command, flush=True)

    status_path = out / "launcher_status.json"
    if args.dry_run:
        status_path.write_text(
            json.dumps(
                {"status": "dry_run", "checked_at_utc": _utc_now()},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return

    status_path.write_text(
        json.dumps(
            {"status": "running", "started_at_utc": _utc_now()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    except subprocess.CalledProcessError as error:
        status_path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "failed_at_utc": _utc_now(),
                    "return_code": int(error.returncode),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    status_path.write_text(
        json.dumps(
            {"status": "complete", "completed_at_utc": _utc_now()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
