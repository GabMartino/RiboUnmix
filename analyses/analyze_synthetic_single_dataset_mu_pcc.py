#!/usr/bin/env python3
"""Compare single-dataset validation mu PCC across bias types and read depths.

This analysis targets the 30-run matrix produced by
``run_synthetic_bias_read_depth_single_dataset_local.sh``: ten artificial-bias
datasets, each trained independently at 0.25, 2, and 20 reads per codon.  It
discovers exactly one run for every bias/depth cell, reads the requested
validation-prediction artifact, and creates one grouped bar plot.

The launcher produces a common validation panel across the ten biases within
each read depth, but the reliability-stratified panel is not common across
read depths.  This script hard-checks the within-depth identity, retains all
validation transcripts for the bias comparison, and writes the cross-depth
overlap explicitly rather than silently reducing the analysis to the very
small three-depth intersection.

Metric definition
-----------------
For every validation transcript, Pearson correlation is computed between the
predicted ``mu`` profile and the observed ``target`` profile at positions where
``mask`` is true and both arrays are finite.  A transcript with negligible
target or prediction variance contributes 0.0.  The bar height is the
unweighted arithmetic mean over validation transcripts.  This matches the raw
``val_mu_pcc`` diagnostic in ``RiboUnmixLightningModule``: there is no
Fisher-z aggregation, reliability weighting, or edge trimming.

By default the script uses the explicit ``best_pcc`` prediction artifacts,
matching the existing synthetic-analysis convention.  ``best_val_loss`` can
be selected explicitly to avoid selecting the checkpoint on the displayed
metric.

Example
-------
::

    .venv/bin/python analyses/analyze_synthetic_single_dataset_mu_pcc.py \
        --run-id single_20260830_205110
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "riboai_synthetic_experiments"
DEFAULT_RUN_ID = "single_20260830_205110"
CHECKPOINT_VARIANTS = ("best_pcc", "best_val_loss")


@dataclass(frozen=True)
class DepthSpec:
    slug: str
    label: str
    color: str


@dataclass(frozen=True)
class BiasSpec:
    slug: str
    label: str


DEPTHS: tuple[DepthSpec, ...] = (
    DepthSpec("0p25", "0.25 reads/codon", "#4C78A8"),
    DepthSpec("2", "2 reads/codon", "#F58518"),
    DepthSpec("20", "20 reads/codon", "#54A24B"),
)

BIAS_TYPES: tuple[BiasSpec, ...] = (
    BiasSpec("artificial_bias_3prime_aa", "3′ AA"),
    BiasSpec("artificial_bias_3prime_cc", "3′ CC"),
    BiasSpec("artificial_bias_3prime_gg", "3′ GG"),
    BiasSpec("artificial_bias_3prime_uu", "3′ UU"),
    BiasSpec("artificial_bias_5prime_aa", "5′ AA"),
    BiasSpec("artificial_bias_5prime_cc", "5′ CC"),
    BiasSpec("artificial_bias_5prime_gg", "5′ GG"),
    BiasSpec("artificial_bias_5prime_uu", "5′ UU"),
    BiasSpec("artificial_bias_gc_fraction_gt_0p7", "GC fraction\n> 0.7"),
    BiasSpec("artificial_bias_au_fraction_gt_0p7", "AU fraction\n> 0.7"),
)


@dataclass(frozen=True)
class RunArtifact:
    depth: str
    depth_label: str
    bias: str
    bias_label: str
    task_index: int
    run_directory: str
    prediction_path: str
    checkpoint_path: str


def _run_prefix(
    *, depth: DepthSpec, bias: BiasSpec, seed: int, run_id: str
) -> str:
    return (
        f"riboai_synthetic_within_{depth.slug}_per_codon_single_{bias.slug}"
        f"_gammaequal_seed{seed}_massfree_{run_id}_"
    )


def discover_run_directory(
    *,
    results_root: Path,
    depth: DepthSpec,
    bias: BiasSpec,
    seed: int,
    run_id: str,
    expected_task_index: int,
) -> Path:
    prefix = _run_prefix(depth=depth, bias=bias, seed=seed, run_id=run_id)
    candidates: list[Path] = []
    for path in results_root.iterdir():
        if not path.is_dir() or not path.name.startswith(prefix):
            continue
        suffix = path.name[len(prefix) :]
        if suffix.isdigit():
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"Missing run for depth={depth.slug!r}, bias={bias.slug!r}; "
            f"expected directory prefix {prefix!r} below {results_root}."
        )
    if len(candidates) != 1:
        names = sorted(path.name for path in candidates)
        raise ValueError(
            f"Expected exactly one run for depth={depth.slug!r}, "
            f"bias={bias.slug!r}, found {names}."
        )

    selected = candidates[0]
    task_index = int(selected.name[len(prefix) :])
    if task_index != expected_task_index:
        raise ValueError(
            f"Unexpected task index for {selected.name}: got {task_index}, "
            f"expected {expected_task_index} from the launcher matrix."
        )
    return selected


def resolve_prediction_artifact(
    *, run_directory: Path, bias: BiasSpec, checkpoint_variant: str
) -> tuple[Path, str]:
    pattern = f"predictions_main_val_{checkpoint_variant}_{bias.slug}.parquet"
    predictions = sorted(
        path
        for path in run_directory.rglob(pattern)
        if path.is_file() and path.stat().st_size > 0
    )
    if len(predictions) != 1:
        raise FileNotFoundError(
            f"Expected exactly one non-empty {pattern!r} below {run_directory}; "
            f"found {[str(path) for path in predictions]}."
        )

    manifests = sorted(run_directory.rglob("prediction_checkpoint_manifest.json"))
    if len(manifests) != 1:
        raise FileNotFoundError(
            "Expected exactly one prediction_checkpoint_manifest.json below "
            f"{run_directory}; found {[str(path) for path in manifests]}."
        )
    try:
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON manifest: {manifests[0]}") from exc
    entry = manifest.get(checkpoint_variant)
    if not isinstance(entry, dict):
        raise KeyError(
            f"{manifests[0]} has no {checkpoint_variant!r} checkpoint entry."
        )
    recorded_output = entry.get("output_path")
    if recorded_output and Path(str(recorded_output)).name != predictions[0].name:
        raise ValueError(
            f"Manifest output basename {Path(str(recorded_output)).name!r} does "
            f"not match discovered prediction {predictions[0].name!r}."
        )
    return predictions[0], str(entry.get("checkpoint_path", ""))


def _as_array(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=dtype)
    return np.asarray(value, dtype=dtype).reshape(-1)


def transcript_mu_pcc(
    *,
    target_value: Any,
    mu_value: Any,
    mask_value: Any,
    pcc_prediction_floor: float,
    eps: float = 1.0e-8,
) -> tuple[float, bool, int]:
    """Return model-compatible raw mu PCC, validity, and valid-position count."""
    target = _as_array(target_value, dtype=np.float64)
    mu = _as_array(mu_value, dtype=np.float64)
    mask = _as_array(mask_value, dtype=np.bool_)
    length = min(target.size, mu.size, mask.size)
    if length == 0:
        return 0.0, False, 0

    target = target[:length]
    mu = mu[:length]
    valid_positions = mask[:length] & np.isfinite(target) & np.isfinite(mu)
    if pcc_prediction_floor > 0.0:
        mu = np.where(mu >= pcc_prediction_floor, mu, 0.0)
        valid_positions &= np.isfinite(mu)

    n_valid_positions = int(valid_positions.sum())
    if n_valid_positions == 0:
        return 0.0, False, 0

    target_valid = target[valid_positions]
    mu_valid = mu[valid_positions]
    target_centered = target_valid - target_valid.mean()
    mu_centered = mu_valid - mu_valid.mean()
    target_var = float(np.dot(target_centered, target_centered))
    mu_var = float(np.dot(mu_centered, mu_centered))
    if target_var <= eps or mu_var <= eps:
        return 0.0, False, n_valid_positions

    correlation = float(
        np.dot(target_centered, mu_centered)
        / np.sqrt(target_var * mu_var + eps * eps)
    )
    if not np.isfinite(correlation):
        return 0.0, False, n_valid_positions
    return correlation, True, n_valid_positions


def analyze_prediction(
    *,
    artifact: RunArtifact,
    pcc_prediction_floor: float,
) -> tuple[list[dict[str, Any]], set[str]]:
    prediction_path = Path(artifact.prediction_path)
    parquet = pq.ParquetFile(prediction_path)
    required = {"transcript_id", "target", "mu", "mask"}
    missing = sorted(required - set(parquet.schema_arrow.names))
    if missing:
        raise KeyError(f"{prediction_path} is missing columns {missing}.")

    columns_to_read = ["transcript_id", "target", "mu", "mask"]
    if "length" in parquet.schema_arrow.names:
        columns_to_read.append("length")

    rows: list[dict[str, Any]] = []
    transcript_ids: set[str] = set()
    for batch in parquet.iter_batches(columns=columns_to_read, batch_size=64):
        values = batch.to_pydict()
        lengths = values.get("length", [None] * batch.num_rows)
        for transcript_id, target, mu, mask, profile_length in zip(
            values["transcript_id"],
            values["target"],
            values["mu"],
            values["mask"],
            lengths,
            strict=True,
        ):
            transcript_id = str(transcript_id)
            if transcript_id in transcript_ids:
                raise ValueError(
                    f"Duplicate transcript {transcript_id!r} in {prediction_path}."
                )
            transcript_ids.add(transcript_id)
            pcc, valid, n_valid_positions = transcript_mu_pcc(
                target_value=target,
                mu_value=mu,
                mask_value=mask,
                pcc_prediction_floor=pcc_prediction_floor,
            )
            rows.append(
                {
                    "transcript_id": transcript_id,
                    "bias": artifact.bias,
                    "bias_label": artifact.bias_label.replace("\n", " "),
                    "read_depth": artifact.depth,
                    "read_depth_label": artifact.depth_label,
                    "mu_pcc": pcc,
                    "pcc_variance_valid": valid,
                    "n_valid_positions": n_valid_positions,
                    "profile_length": (
                        int(profile_length)
                        if profile_length is not None
                        else n_valid_positions
                    ),
                    "task_index": artifact.task_index,
                    "prediction_path": artifact.prediction_path,
                }
            )
    if not rows:
        raise ValueError(f"No prediction rows found in {prediction_path}.")
    return rows, transcript_ids


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float]:
    if replicates <= 0:
        return float("nan"), float("nan")
    if values.size == 0:
        return float("nan"), float("nan")

    means = np.empty(replicates, dtype=np.float64)
    chunk_size = 256
    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        indices = rng.integers(
            0,
            values.size,
            size=(stop - start, values.size),
            endpoint=False,
        )
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_metrics(
    frame: pd.DataFrame,
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(bootstrap_seed)
    rows: list[dict[str, Any]] = []
    for bias in BIAS_TYPES:
        for depth in DEPTHS:
            group = frame.loc[
                (frame["bias"] == bias.slug)
                & (frame["read_depth"] == depth.slug)
            ]
            if group.empty:
                raise ValueError(
                    f"No transcript metrics for bias={bias.slug}, depth={depth.slug}."
                )
            values = group["mu_pcc"].to_numpy(dtype=np.float64)
            ci_low, ci_high = bootstrap_mean_interval(
                values,
                rng=rng,
                replicates=bootstrap_replicates,
            )
            rows.append(
                {
                    "bias": bias.slug,
                    "bias_label": bias.label.replace("\n", " "),
                    "read_depth": depth.slug,
                    "read_depth_label": depth.label,
                    "mean_mu_pcc": float(values.mean()),
                    "median_mu_pcc": float(np.median(values)),
                    "std_mu_pcc": float(values.std(ddof=1)),
                    "q25_mu_pcc": float(np.quantile(values, 0.25)),
                    "q75_mu_pcc": float(np.quantile(values, 0.75)),
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "n_transcripts": int(values.size),
                    "n_valid_pcc_transcripts": int(
                        group["pcc_variance_valid"].sum()
                    ),
                    "n_zero_variance_transcripts": int(
                        (~group["pcc_variance_valid"]).sum()
                    ),
                    "task_index": int(group["task_index"].iloc[0]),
                    "prediction_path": str(group["prediction_path"].iloc[0]),
                }
            )
    return pd.DataFrame(rows)


def plot_grouped_bars(
    *,
    summary: pd.DataFrame,
    output_png: Path,
    checkpoint_variant: str,
    n_transcripts_by_depth: dict[str, int],
    show_confidence_intervals: bool,
) -> None:
    matplotlib_cache = Path(tempfile.gettempdir()) / "riboai_matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(BIAS_TYPES), dtype=np.float64)
    width = 0.25
    offsets = np.asarray((-width, 0.0, width))
    fig, ax = plt.subplots(figsize=(14.5, 6.7))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.17, top=0.77)

    for depth_index, depth in enumerate(DEPTHS):
        depth_rows = summary.loc[summary["read_depth"] == depth.slug].set_index(
            "bias"
        )
        depth_rows = depth_rows.loc[[bias.slug for bias in BIAS_TYPES]]
        scores = depth_rows["mean_mu_pcc"].to_numpy(dtype=np.float64)
        yerr = None
        if show_confidence_intervals:
            lower = depth_rows["bootstrap_ci95_low"].to_numpy(dtype=np.float64)
            upper = depth_rows["bootstrap_ci95_high"].to_numpy(dtype=np.float64)
            yerr = np.vstack((scores - lower, upper - scores))
        bars = ax.bar(
            x + offsets[depth_index],
            scores,
            width=width,
            color=depth.color,
            edgecolor="white",
            linewidth=0.7,
            label=depth.label,
            yerr=yerr,
            error_kw={"elinewidth": 0.8, "capsize": 2.0, "capthick": 0.8},
            zorder=3,
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.2f}" for value in scores],
            padding=3,
            fontsize=7.5,
        )

    ax.set_xticks(x, [bias.label for bias in BIAS_TYPES], fontsize=10)
    ax.set_xlim(-0.65, len(BIAS_TYPES) - 0.35)
    ax.set_ylim(0.0, 1.02)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_ylabel("Mean validation PCC(predicted μ, observed profile)", fontsize=11)
    checkpoint_label = checkpoint_variant.replace("_", " ")
    panel_sizes = sorted(set(n_transcripts_by_depth.values()))
    if len(panel_sizes) == 1:
        transcript_text = f"{panel_sizes[0]:,} transcripts/depth"
    else:
        transcript_text = ", ".join(
            f"{depth}={count:,}" for depth, count in n_transcripts_by_depth.items()
        )
    fig.suptitle(
        "Single-dataset μ agreement across synthetic bias types\n"
        f"checkpoint: {checkpoint_label}; {transcript_text}; "
        "biases matched within depth",
        fontsize=13,
        y=0.975,
    )
    ax.grid(axis="y", alpha=0.25, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Observed read depth",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncols=3,
        frameon=False,
    )
    if show_confidence_intervals:
        fig.text(
            0.995,
            0.005,
            "Error bars: 95% transcript-bootstrap CI",
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.35",
        )

    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_png.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory containing the 30 single-dataset experiment directories.",
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help=(
            "Launcher RUN_ID shared by the 30 runs "
            f"(default: {DEFAULT_RUN_ID})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed encoded in the run directory names (default: 42).",
    )
    parser.add_argument(
        "--checkpoint-variant",
        choices=CHECKPOINT_VARIANTS,
        default="best_pcc",
        help="Prediction checkpoint variant to analyze (default: best_pcc).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: analyses/artifacts/synthetic/single_dataset_mu/"
            "<run-id>/<checkpoint-variant>."
        ),
    )
    parser.add_argument(
        "--pcc-prediction-floor",
        type=float,
        default=0.0,
        help="Optional prediction-only μ floor used for PCC (default: 0.0).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Seed for transcript bootstrap confidence intervals (default: 42).",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=2_000,
        help="Number of transcript bootstrap replicates; 0 disables CIs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.results_root.is_dir():
        raise FileNotFoundError(f"Results root does not exist: {args.results_root}")
    if args.pcc_prediction_floor < 0.0:
        raise ValueError("--pcc-prediction-floor must be non-negative.")
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap-replicates must be non-negative.")

    output_dir = args.output_dir or (
        REPO_ROOT
        / "analyses"
        / "artifacts"
        / "synthetic"
        / "single_dataset_mu"
        / args.run_id
        / args.checkpoint_variant
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[RunArtifact] = []
    transcript_rows: list[dict[str, Any]] = []
    validation_ids_by_depth: dict[str, set[str]] = {}
    for depth_index, depth in enumerate(DEPTHS):
        for bias_index, bias in enumerate(BIAS_TYPES):
            expected_task_index = depth_index * len(BIAS_TYPES) + bias_index
            run_directory = discover_run_directory(
                results_root=args.results_root,
                depth=depth,
                bias=bias,
                seed=args.seed,
                run_id=args.run_id,
                expected_task_index=expected_task_index,
            )
            prediction_path, checkpoint_path = resolve_prediction_artifact(
                run_directory=run_directory,
                bias=bias,
                checkpoint_variant=args.checkpoint_variant,
            )
            artifact = RunArtifact(
                depth=depth.slug,
                depth_label=depth.label,
                bias=bias.slug,
                bias_label=bias.label.replace("\n", " "),
                task_index=expected_task_index,
                run_directory=str(run_directory.resolve()),
                prediction_path=str(prediction_path.resolve()),
                checkpoint_path=checkpoint_path,
            )
            artifacts.append(artifact)
            rows, transcript_ids = analyze_prediction(
                artifact=artifact,
                pcc_prediction_floor=args.pcc_prediction_floor,
            )
            depth_reference_ids = validation_ids_by_depth.get(depth.slug)
            if depth_reference_ids is None:
                validation_ids_by_depth[depth.slug] = transcript_ids
            elif transcript_ids != depth_reference_ids:
                missing = sorted(depth_reference_ids - transcript_ids)[:10]
                extra = sorted(transcript_ids - depth_reference_ids)[:10]
                raise AssertionError(
                    "Validation transcript IDs differ across biases at the same "
                    f"read depth {depth.slug}: "
                    f"{run_directory}; missing examples={missing}; extra examples={extra}."
                )
            transcript_rows.extend(rows)
            print(
                f"task={expected_task_index:02d} depth={depth.slug:>5s} "
                f"bias={bias.slug:42s} transcripts={len(rows):,}"
            )

    if not validation_ids_by_depth:
        raise RuntimeError("No single-dataset predictions were analyzed.")
    expected_conditions = len(DEPTHS) * len(BIAS_TYPES)
    if len(artifacts) != expected_conditions:
        raise AssertionError(
            f"Expected {expected_conditions} artifacts, resolved {len(artifacts)}."
        )

    transcript_frame = pd.DataFrame(transcript_rows)
    condition_counts = transcript_frame.groupby(
        ["bias", "read_depth"], sort=False
    )["transcript_id"].nunique()
    for (bias, depth), count in condition_counts.items():
        expected_count = len(validation_ids_by_depth[str(depth)])
        if int(count) != expected_count:
            raise AssertionError(
                f"Condition bias={bias}, depth={depth} has {count} transcripts; "
                f"expected the depth-panel size {expected_count}."
            )

    overlap_rows: list[dict[str, Any]] = []
    for depth_index, depth_a in enumerate(DEPTHS):
        ids_a = validation_ids_by_depth[depth_a.slug]
        for depth_b in DEPTHS[depth_index + 1 :]:
            ids_b = validation_ids_by_depth[depth_b.slug]
            intersection = ids_a & ids_b
            union = ids_a | ids_b
            overlap_rows.append(
                {
                    "read_depth_a": depth_a.slug,
                    "read_depth_b": depth_b.slug,
                    "n_a": len(ids_a),
                    "n_b": len(ids_b),
                    "n_intersection": len(intersection),
                    "n_union": len(union),
                    "jaccard": len(intersection) / len(union),
                }
            )
    all_depth_intersection = set.intersection(*validation_ids_by_depth.values())
    overlap_frame = pd.DataFrame(overlap_rows)

    summary = summarize_metrics(
        transcript_frame,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    transcript_path = output_dir / "single_dataset_mu_pcc_per_transcript.csv"
    summary_path = output_dir / "single_dataset_mu_pcc_summary.csv"
    overlap_path = output_dir / "validation_transcript_overlap_by_depth.csv"
    figure_path = output_dir / "single_dataset_mu_pcc_by_bias_and_depth.png"
    transcript_frame.to_csv(transcript_path, index=False)
    summary.to_csv(summary_path, index=False)
    overlap_frame.to_csv(overlap_path, index=False)
    plot_grouped_bars(
        summary=summary,
        output_png=figure_path,
        checkpoint_variant=args.checkpoint_variant,
        n_transcripts_by_depth={
            depth: len(ids) for depth, ids in validation_ids_by_depth.items()
        },
        show_confidence_intervals=args.bootstrap_replicates > 0,
    )

    manifest = {
        "analysis": "synthetic_single_dataset_mu_pcc_by_bias_and_depth",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_results_root": str(args.results_root.resolve()),
        "run_id": args.run_id,
        "training_seed": args.seed,
        "checkpoint_variant": args.checkpoint_variant,
        "pcc_prediction_floor": args.pcc_prediction_floor,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "metric_definition": (
            "Per-transcript Pearson PCC(predicted mu, observed target) over mask-valid "
            "finite positions; degenerate profiles contribute 0; bars are unweighted "
            "arithmetic means over each depth-specific validation transcript panel."
        ),
        "n_conditions": len(artifacts),
        "validation_panel_contract": (
            "Validation IDs are identical across all ten biases within a read depth. "
            "They differ across read depths because the split is reliability-stratified; "
            "cross-depth bars therefore use depth-specific panels."
        ),
        "n_validation_transcripts_by_depth": {
            depth: len(ids) for depth, ids in validation_ids_by_depth.items()
        },
        "validation_transcript_ids_by_depth": {
            depth: sorted(ids) for depth, ids in validation_ids_by_depth.items()
        },
        "n_all_depth_intersection": len(all_depth_intersection),
        "validation_overlap_by_depth": overlap_rows,
        "artifacts": [asdict(artifact) for artifact in artifacts],
        "outputs": {
            "per_transcript_csv": str(transcript_path.resolve()),
            "summary_csv": str(summary_path.resolve()),
            "validation_overlap_csv": str(overlap_path.resolve()),
            "figure_png": str(figure_path.resolve()),
            "figure_svg": str(figure_path.with_suffix(".svg").resolve()),
        },
    }
    manifest_path = output_dir / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    display = summary.pivot(
        index="bias_label", columns="read_depth_label", values="mean_mu_pcc"
    ).loc[[bias.label.replace("\n", " ") for bias in BIAS_TYPES]]
    print("\nMean validation mu PCC")
    print(display.to_string(float_format=lambda value: f"{value:.4f}"))
    print("\nValidation transcripts by read depth")
    for depth in DEPTHS:
        print(f"  {depth.label:18s} {len(validation_ids_by_depth[depth.slug]):,}")
    print(f"  all-depth intersection {len(all_depth_intersection):,}")
    print("Cross-depth validation overlap")
    print(overlap_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Summary: {summary_path}")
    print(f"Per-transcript values: {transcript_path}")
    print(f"Validation overlap: {overlap_path}")
    print(f"Figure: {figure_path}")
    print(f"Figure (SVG): {figure_path.with_suffix('.svg')}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
