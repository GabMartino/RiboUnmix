#!/usr/bin/env python3
"""Compare sampled synthetic profiles with their matched TASEP occupancies.

The two exported occupancy trajectories are replicate-specific.  This analysis
therefore compares sampled replica r with q_t^(r), and compares the arithmetic
sampled consensus with qbar_t = (q_t^(1) + q_t^(2)) / 2.  It streams one
transcript at a time and never materializes position-level profiles globally.
The programmed kinetic target K_t remains a separate target and is not
overwritten by this analysis.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Iterator

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyses.analyze_synthetic_input_data import (  # noqa: E402
    SortedCursor,
    iter_rows,
    mean_one,
    pearson,
    rmse_mean_one,
    sha256,
)
from analyses.analyze_synthetic_individual_datasets import (  # noqa: E402
    AVAILABLE_DATASET_ORDER,
    DATASET_ORDER,
    DEPTHS,
    PLOT_DISPLAY_NAMES,
    _mean_t_interval,
    _style_boxplot,
    box_statistics,
    configure_iclr_typography,
    save_figure,
)

WEIGHTED_ROOT = ROOT / "Datasets" / "data" / "weighted_synthetic"
OCCUPANCY_PATH = (
    ROOT
    / "Datasets"
    / "Synthetic_data"
    / "artificial_ground_truth_tasep_occupancy_replicates.parquet"
)
DEFAULT_OUTPUT = ROOT / "analyses" / "artifacts" / "synthetic" / "individual_dataset"

METRIC_SCHEMA = pa.schema(
    [
        ("depth", pa.string()),
        ("nominal_reads_per_codon", pa.float64()),
        ("dataset", pa.string()),
        ("is_biased", pa.bool_()),
        ("transcript_id", pa.string()),
        ("sense_length", pa.int32()),
        ("occupancy_replica_pcc", pa.float64()),
        ("rep1_tasep_pcc", pa.float64()),
        ("rep2_tasep_pcc", pa.float64()),
        ("consensus_tasep_pcc", pa.float64()),
        ("consensus_tasep_pcc_interior10", pa.float64()),
        ("rep1_tasep_rmse_mean1", pa.float64()),
        ("rep2_tasep_rmse_mean1", pa.float64()),
        ("consensus_tasep_rmse_mean1", pa.float64()),
    ]
)

REPRESENTATIONS = (
    ("rep1_tasep_pcc", "Replica 1", "#3676B8"),
    ("rep2_tasep_pcc", "Replica 2", "#E6862F"),
    ("consensus_tasep_pcc", "Arithmetic mean", "#3A923A"),
)


def occupancy_role(sample: str) -> str:
    mapping = {
        "replicate_1_mean_psite_occupancy": "rep1",
        "replicate_2_mean_psite_occupancy": "rep2",
    }
    try:
        return mapping[sample]
    except KeyError as exc:
        raise ValueError(f"Unrecognized TASEP occupancy sample: {sample!r}") from exc


def iter_occupancy_replicates(
    path: Path, *, batch_size: int = 128
) -> Iterator[tuple[str, dict[str, np.ndarray]]]:
    """Yield the two raw occupancy profiles for each sorted transcript."""
    current_id: str | None = None
    current: dict[str, np.ndarray] = {}
    for row in iter_rows(
        path, ["sample", "transcript_id", "rib_profile"], batch_size=batch_size
    ):
        transcript_id = str(row["transcript_id"])
        if current_id is not None and transcript_id != current_id:
            if set(current) != {"rep1", "rep2"}:
                raise ValueError(
                    f"{path}: transcript {current_id!r} has roles {sorted(current)}"
                )
            yield current_id, current
            current = {}
        current_id = transcript_id
        role = occupancy_role(str(row["sample"]))
        if role in current:
            raise ValueError(f"{path}: duplicate {role} for {transcript_id}")
        current[role] = np.asarray(row["rib_profile"], dtype=np.float64)
    if current_id is not None:
        if set(current) != {"rep1", "rep2"}:
            raise ValueError(
                f"{path}: transcript {current_id!r} has roles {sorted(current)}"
            )
        yield current_id, current


def iter_weighted_profiles(path: Path, *, batch_size: int) -> Iterator[tuple[str, Any]]:
    for row in iter_rows(
        path, ["id", "ribo_cds_replicas"], batch_size=batch_size
    ):
        yield str(row["id"]), row["ribo_cds_replicas"]


def normalize_occupancy(values: Any) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    if raw.ndim != 1 or raw.size < 3 or not np.isfinite(raw).all():
        raise ValueError("TASEP occupancy must be a finite one-dimensional profile")
    if np.any(raw < 0.0):
        raise ValueError("TASEP occupancy cannot be negative")
    normalized = mean_one(raw)
    if normalized is None:
        raise ValueError("TASEP occupancy must have a positive mean")
    return normalized


def calculate_matched_metrics(
    replicas_with_stop: Any,
    occupancy: dict[str, np.ndarray],
) -> dict[str, float | int]:
    """Calculate shape metrics after exact replicate and coordinate matching."""
    q1 = normalize_occupancy(occupancy["rep1"])
    q2 = normalize_occupancy(occupancy["rep2"])
    return calculate_metrics_from_mean_one_targets(replicas_with_stop, q1, q2)


def calculate_metrics_from_mean_one_targets(
    replicas_with_stop: Any,
    q1: np.ndarray,
    q2: np.ndarray,
) -> dict[str, float | int]:
    """Core metric calculation for prevalidated, mean-one occupancy targets."""
    sampled = np.asarray(replicas_with_stop, dtype=np.float64)
    if sampled.ndim != 2 or sampled.shape[0] != 2:
        raise ValueError("Expected exactly two sampled replicas")
    if sampled.shape[1] != q1.size + 1 or q1.shape != q2.shape:
        raise ValueError(
            f"Sample/occupancy length mismatch: sampled={sampled.shape}, "
            f"q1={q1.shape}, q2={q2.shape}"
        )
    if np.any(sampled[:, -1] != 0.0):
        raise ValueError("The appended terminal boundary must be zero")
    rep1, rep2 = sampled[:, :-1]
    consensus = 0.5 * (rep1 + rep2)
    qbar = 0.5 * (q1 + q2)
    if not (
        math.isclose(float(q1.mean()), 1.0, abs_tol=1e-12, rel_tol=0.0)
        and math.isclose(float(q2.mean()), 1.0, abs_tol=1e-12, rel_tol=0.0)
        and math.isclose(float(qbar.mean()), 1.0, abs_tol=1e-12, rel_tol=0.0)
    ):
        raise ValueError("Normalized TASEP targets are not mean-one")
    return {
        "sense_length": int(q1.size),
        "occupancy_replica_pcc": pearson(q1, q2),
        "rep1_tasep_pcc": pearson(rep1, q1),
        "rep2_tasep_pcc": pearson(rep2, q2),
        "consensus_tasep_pcc": pearson(consensus, qbar),
        "consensus_tasep_pcc_interior10": pearson(consensus, qbar, trim=10),
        "rep1_tasep_rmse_mean1": rmse_mean_one(rep1, q1),
        "rep2_tasep_rmse_mean1": rmse_mean_one(rep2, q2),
        "consensus_tasep_rmse_mean1": rmse_mean_one(consensus, qbar),
    }


def _flush(writer: pq.ParquetWriter, records: list[dict[str, Any]]) -> int:
    if not records:
        return 0
    writer.write_table(pa.Table.from_pylist(records, schema=METRIC_SCHEMA))
    count = len(records)
    records.clear()
    return count


def stream_metrics(
    output_path: Path,
    *,
    occupancy_path: Path,
    batch_size: int,
    max_transcripts: int | None,
) -> tuple[int, dict[str, list[str]]]:
    """Stream all datasets by depth while reading the occupancy file once/depth."""
    total_rows = 0
    skipped_by_depth: dict[str, list[str]] = {}
    with pq.ParquetWriter(
        output_path, METRIC_SCHEMA, compression="zstd", use_dictionary=True
    ) as writer:
        for depth, nominal, _ in DEPTHS:
            paths = {
                dataset: WEIGHTED_ROOT / depth / f"{dataset}.parquet"
                for dataset in AVAILABLE_DATASET_ORDER
            }
            missing = [str(path) for path in paths.values() if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing weighted inputs: {missing}")
            anchor = AVAILABLE_DATASET_ORDER[0]
            anchor_iterator = iter_weighted_profiles(paths[anchor], batch_size=batch_size)
            cursors = {
                dataset: SortedCursor(
                    iter_weighted_profiles(path, batch_size=batch_size), str(path)
                )
                for dataset, path in paths.items()
                if dataset != anchor
            }
            occupancy_cursor = SortedCursor(
                iter_occupancy_replicates(occupancy_path, batch_size=batch_size),
                str(occupancy_path),
            )
            buffer: list[dict[str, Any]] = []
            transcript_count = 0
            for transcript_id, anchor_profile in anchor_iterator:
                if max_transcripts is not None and transcript_count >= max_transcripts:
                    break
                profiles = {anchor: anchor_profile}
                profiles.update(
                    {
                        dataset: cursor.get(transcript_id)
                        for dataset, cursor in cursors.items()
                    }
                )
                occupancy = occupancy_cursor.get(transcript_id)
                q1 = normalize_occupancy(occupancy["rep1"])
                q2 = normalize_occupancy(occupancy["rep2"])
                for dataset in AVAILABLE_DATASET_ORDER:
                    metrics = calculate_metrics_from_mean_one_targets(
                        profiles[dataset], q1, q2
                    )
                    buffer.append(
                        {
                            "depth": depth,
                            "nominal_reads_per_codon": float(nominal),
                            "dataset": dataset,
                            "is_biased": dataset != "artificial_ground_truth",
                            "transcript_id": transcript_id,
                            **metrics,
                        }
                    )
                    if len(buffer) >= 1000:
                        total_rows += _flush(writer, buffer)
                transcript_count += 1
            total_rows += _flush(writer, buffer)
            skipped_by_depth[depth] = list(occupancy_cursor.skipped_ids)
            print(
                f"[{depth}] transcripts={transcript_count:,}; "
                f"rows={transcript_count * len(AVAILABLE_DATASET_ORDER):,}; "
                f"occupancy IDs skipped before matches={len(occupancy_cursor.skipped_ids)}",
                flush=True,
            )
    return total_rows, skipped_by_depth


def summarize_metrics(metrics_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(metrics_path)
    frame = frame.loc[frame["dataset"].isin(DATASET_ORDER)]
    rows: list[dict[str, Any]] = []
    for (depth, nominal, dataset), group in frame.groupby(
        ["depth", "nominal_reads_per_codon", "dataset"], sort=False
    ):
        for column, label, _ in REPRESENTATIONS:
            values = group[column].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            rows.append(
                {
                    "depth": depth,
                    "nominal_reads_per_codon": nominal,
                    "dataset": dataset,
                    "representation": label,
                    "n_valid": int(values.size),
                    "p05": float(np.quantile(values, 0.05)),
                    "q25": float(np.quantile(values, 0.25)),
                    "median": float(np.quantile(values, 0.50)),
                    "q75": float(np.quantile(values, 0.75)),
                    "p95": float(np.quantile(values, 0.95)),
                }
            )
    return pd.DataFrame.from_records(rows)


def summarize_occupancy_replicate_agreement(metrics_path: Path) -> pd.DataFrame:
    """Summarize q1--q2 trajectory agreement once per transcript.

    The same two occupancy trajectories are reused for all datasets and
    depths, so counting repeated copies would artificially multiply n.
    """
    depth = DEPTHS[0][0]
    dataset = AVAILABLE_DATASET_ORDER[0]
    frame = pd.read_parquet(
        metrics_path,
        columns=["depth", "dataset", "transcript_id", "occupancy_replica_pcc"],
        filters=[("depth", "=", depth), ("dataset", "=", dataset)],
    )
    if frame["transcript_id"].duplicated().any():
        raise ValueError("Occupancy replicate summary contains duplicate transcripts")
    values = frame["occupancy_replica_pcc"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Undefined q1--q2 occupancy correlations")
    return pd.DataFrame.from_records(
        [
            {
                "transcripts": int(values.size),
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values, ddof=1)),
                "p05": float(np.quantile(values, 0.05)),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.quantile(values, 0.50)),
                "q75": float(np.quantile(values, 0.75)),
                "p95": float(np.quantile(values, 0.95)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
        ]
    )


def plot_occupancy_boxes(summary_source: pd.DataFrame, figure_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(
        summary_source.attrs["metrics_path"],
        columns=["depth", "dataset", *(item[0] for item in REPRESENTATIONS)],
    )
    raw = raw.loc[raw["dataset"].isin(DATASET_ORDER)]
    figure, axes = plt.subplots(3, 1, figsize=(22.0, 19.0), constrained_layout=True)
    base_positions = np.arange(1, len(DATASET_ORDER) + 1, dtype=np.float64)
    offsets = (-0.25, 0.0, 0.25)
    for axis, (depth, _, title) in zip(axes, DEPTHS):
        subset = raw.loc[raw["depth"] == depth]
        for (column, label, color), offset in zip(REPRESENTATIONS, offsets):
            stats = [
                box_statistics(
                    subset.loc[subset["dataset"] == dataset, column], label
                )
                for dataset in DATASET_ORDER
            ]
            result = axis.bxp(
                stats,
                positions=base_positions + offset,
                widths=0.22,
                showfliers=False,
                patch_artist=True,
                manage_ticks=False,
            )
            _style_boxplot(result, color, alpha=0.8)
            result["boxes"][0].set_label(label)
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        axis.set_ylim(-0.25, 1.02)
        axis.set_xlim(0.45, len(DATASET_ORDER) + 0.55)
        axis.set_xticks(
            base_positions, [PLOT_DISPLAY_NAMES[d] for d in DATASET_ORDER]
        )
        axis.tick_params(axis="x", labelrotation=32)
        axis.set_ylabel("Transcript-level PCC")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
        axis.legend(frameon=False, ncol=3, loc="lower left")
    figure.suptitle(
        r"Sampled profiles versus matched pre-bias TASEP occupancies $q_t$",
        fontsize=24,
    )
    stem = figure_dir / "tasep_occupancy_agreement_boxplots_by_dataset_and_depth"
    return save_figure(figure, stem)


def target_headline_summary(
    target_summary: pd.DataFrame,
    individual_audit_dir: Path,
) -> pd.DataFrame:
    replica = pd.read_csv(individual_audit_dir / "replica_agreement_summary.csv")
    cross = pd.read_csv(individual_audit_dir / "cross_dataset_correlation_summary.csv")
    rows: list[dict[str, Any]] = []
    for depth, nominal, _ in DEPTHS:
        target = target_summary.loc[target_summary["depth"] == depth].pivot(
            index="dataset", columns="representation", values="median"
        )
        condition_metrics = {
            "replica_agreement": replica.loc[
                replica["depth"] == depth
            ].set_index("dataset")["median"],
            "single_replica_vs_matching_q": target[["Replica 1", "Replica 2"]].mean(
                axis=1
            ),
            "replica_mean_vs_qbar": target["Arithmetic mean"],
        }
        for metric, values in condition_metrics.items():
            estimate, half_width, low, high = _mean_t_interval(values)
            rows.append(
                {
                    "depth": depth,
                    "nominal_reads_per_codon": nominal,
                    "metric": metric,
                    "estimate": estimate,
                    "ci95_half_width": half_width,
                    "ci95_low": low,
                    "ci95_high": high,
                    "design_units": len(values),
                    "interval_method": "two-sided Student-t CI across bias conditions",
                }
            )
        pairs = cross.loc[cross["depth"] == depth]
        datasets = sorted(set(pairs["dataset_a"]) | set(pairs["dataset_b"]))
        estimate = float(pairs["median_pcc"].mean())
        leave_one_out = np.asarray(
            [
                pairs.loc[
                    (pairs["dataset_a"] != dataset)
                    & (pairs["dataset_b"] != dataset),
                    "median_pcc",
                ].mean()
                for dataset in datasets
            ],
            dtype=np.float64,
        )
        n_units = len(datasets)
        standard_error = float(
            np.sqrt(
                (n_units - 1)
                / n_units
                * np.sum((leave_one_out - leave_one_out.mean()) ** 2)
            )
        )
        half_width = float(student_t.ppf(0.975, n_units - 1) * standard_error)
        rows.append(
            {
                "depth": depth,
                "nominal_reads_per_codon": nominal,
                "metric": "cross_dataset_agreement",
                "estimate": estimate,
                "ci95_half_width": half_width,
                "ci95_low": estimate - half_width,
                "ci95_high": estimate + half_width,
                "design_units": n_units,
                "interval_method": (
                    "two-sided t interval from leave-one-dataset-out jackknife SE"
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--occupancy-path", type=Path, default=OCCUPANCY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reference-audit-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory containing the existing replica and cross-dataset summaries.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-transcripts", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_transcripts is not None and args.max_transcripts <= 0:
        parser.error("--max-transcripts must be positive")
    occupancy_path = args.occupancy_path.resolve()
    if not occupancy_path.is_file():
        raise FileNotFoundError(occupancy_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "tasep_occupancy_per_transcript.parquet"
    generated = [
        metrics_path,
        output_dir / "tasep_occupancy_agreement_summary.csv",
        output_dir / "headline_tasep_mean_ci_summary.csv",
        output_dir / "tasep_occupancy_replica_agreement_summary.csv",
        output_dir / "tasep_occupancy_provenance.json",
    ]
    if any(path.exists() for path in generated) and not args.overwrite:
        raise FileExistsError("TASEP analysis outputs exist; use --overwrite")
    if metrics_path.exists():
        metrics_path.unlink()
    total_rows, skipped = stream_metrics(
        metrics_path,
        occupancy_path=occupancy_path,
        batch_size=args.batch_size,
        max_transcripts=args.max_transcripts,
    )
    summary = summarize_metrics(metrics_path)
    summary.attrs["metrics_path"] = str(metrics_path)
    summary_path = output_dir / "tasep_occupancy_agreement_summary.csv"
    summary.to_csv(summary_path, index=False)
    occupancy_replica_summary = summarize_occupancy_replicate_agreement(metrics_path)
    occupancy_replica_summary.to_csv(
        output_dir / "tasep_occupancy_replica_agreement_summary.csv", index=False
    )
    headline = target_headline_summary(summary, args.reference_audit_dir.resolve())
    headline_path = output_dir / "headline_tasep_mean_ci_summary.csv"
    headline.to_csv(headline_path, index=False)
    configure_iclr_typography()
    figure_paths = plot_occupancy_boxes(summary, output_dir / "figures")
    provenance = {
        "analysis": "synthetic_sampled_profiles_vs_matched_tasep_occupancy",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python": platform.python_version(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "occupancy_path": str(occupancy_path),
        "occupancy_sha256": sha256(occupancy_path),
        "occupancy_metadata": {
            key.decode(): value.decode()
            for key, value in (pq.ParquetFile(occupancy_path).schema_arrow.metadata or {}).items()
        },
        "target_definition": {
            "replica": "sampled Y^(r) versus matching mean_one(raw occupancy O^(r))",
            "consensus": "0.5*(Y1+Y2) versus 0.5*(q1+q2)",
            "terminal": "exactly one appended terminal boundary removed from sampled profiles",
            "kinetic_target_policy": "K_t results retained separately; not overwritten",
        },
        "rows": total_rows,
        "full_run": args.max_transcripts is None,
        "max_transcripts": args.max_transcripts,
        "occupancy_ids_skipped_before_weighted_matches": skipped,
        "occupancy_replicate_agreement": occupancy_replica_summary.iloc[0].to_dict(),
        "figures": [str(path) for path in figure_paths],
    }
    (output_dir / "tasep_occupancy_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {metrics_path} ({total_rows:,} rows)", flush=True)
    print(headline.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
