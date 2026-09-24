#!/usr/bin/env python3
"""Recompute the cross-organism benchmark on a common normalized endpoint.

The analysis uses archived position-level test predictions only.  It removes
the requested number of codons from each CDS end and reports transcript-level
PCC, SCC, and normalized RMSE on positions with a positive observed count.
For RMSE, both profiles are divided by the observed target mean over the
complete trimmed window, including zero-count positions, before the positive
position mask is applied.  No model is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata


ORGANISMS = (
    "celegans_stein_2021",
    "ecoli_zhang_2016",
    "human_iwasaki_2014",
    "yeast_stein_2021",
)

RIBOMIMO_PROTOCOLS = {
    "ribomimo": "Own preprocessing",
    "unweighted": "RiboUnmix unweighted",
    "weighted": "RiboUnmix weighted",
}

EXTERNAL_SUMMARY_PATH = "four_model_canonical_test_positive_median_ci95.tsv"
EXTERNAL_PROVENANCE_PATH = "four_model_canonical_training_checkpoint_provenance.tsv"

EXTERNAL_SUMMARY_SPECS = {
    "iXnos": {
        "training_to_protocol": {
            "readme_filter__keep_zeros_full_canonical_test": "Own preprocessing",
            "unweighted": "RiboUnmix unweighted",
            "weighted": "RiboUnmix weighted",
        },
    },
    "RiboExp": {
        "training_to_protocol": {
            "paper_top500__mask_zeros_full_canonical_test": "Own preprocessing",
            "unweighted": "RiboUnmix unweighted",
            "weighted": "RiboUnmix weighted",
        },
    },
    "Riboformer": {
        "training_to_protocol": {
            "canonical_native_filtered_unweighted": "Own preprocessing",
            "unweighted": "RiboUnmix unweighted",
            "weighted": "RiboUnmix weighted",
        },
    },
    "Seq2Ribo": {
        "training_to_protocol": {
            "paper_filtered_native_train_validation": "Own preprocessing",
            "unweighted": "RiboUnmix unweighted",
            "weighted": "RiboUnmix weighted",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: inferred from this script).",
    )
    parser.add_argument(
        "--trim-codons",
        type=int,
        default=5,
        help="Codons removed from each end before metric calculation (default: 5).",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10_000,
        help="Transcript bootstrap resamples for percentile intervals.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/cross_organism_benchmark_reanalysis).",
    )
    return parser.parse_args()


def one_match(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one match for {pattern!r} below {root}, found {len(matches)}"
        )
    return matches[0]


def prediction_paths(project_root: Path) -> list[dict[str, str | Path]]:
    mimo_root = (
        project_root
        / "results/ribomimo_benchmarking_experiments/fixed_test_three_protocols/results"
    )
    ribounmix_root = (
        project_root
        / "results/riboai_benchmarking_experiments/benchmark_20260829_194806/results"
    )
    rows: list[dict[str, str | Path]] = []
    for directory, protocol in RIBOMIMO_PROTOCOLS.items():
        for organism in ORGANISMS:
            path = one_match(
                mimo_root / directory / organism / "bidirectional",
                f"predictions_test_best_pcc_{organism}.parquet",
            )
            rows.append(
                {
                    "architecture": "RiboMIMO",
                    "protocol": protocol,
                    "organism": organism,
                    "checkpoint_rule": "best_validation_PCC",
                    "format": "ribomimo",
                    "path": path,
                }
            )
    for organism in ORGANISMS:
        path = one_match(
            ribounmix_root / organism,
            f"*/predictions_test_best_val_loss_{organism}.parquet",
        )
        rows.append(
            {
                "architecture": "RiboUnmix",
                "protocol": "Weighted",
                "organism": organism,
                "checkpoint_rule": "best_validation_loss",
                "format": "ribounmix",
                "path": path,
            }
        )
    return rows


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return np.nan
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return np.nan
    return pearson(rankdata(x, method="average"), rankdata(y, method="average"))


def profile_digest(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_prediction_rows(path: Path, columns: Iterable[str]) -> Iterable[dict]:
    parquet = pq.ParquetFile(path)
    try:
        for batch in parquet.iter_batches(batch_size=64, columns=list(columns)):
            yield from batch.to_pylist()
    finally:
        del parquet


def evaluate_file(spec: dict[str, str | Path], trim_codons: int) -> tuple[pd.DataFrame, dict]:
    path = Path(spec["path"])
    file_format = str(spec["format"])
    if file_format == "ribomimo":
        columns = ("transcript_id", "target", "raw_target", "mu", "normalization_factor")
    elif file_format == "ribounmix":
        columns = ("transcript_id", "target", "mu", "normalized_shape", "scale_dt")
    else:
        raise ValueError(f"Unsupported prediction format: {file_format}")

    results: list[dict] = []
    max_conversion_deviation = 0.0
    for row in iter_prediction_rows(path, columns):
        transcript_id = str(row["transcript_id"])
        if file_format == "ribomimo":
            observed = np.asarray(row["raw_target"], dtype=np.float64)
            normalized_target = np.asarray(row["target"], dtype=np.float64)
            normalization_factor = float(row["normalization_factor"])
            predicted = np.asarray(row["mu"], dtype=np.float64) * normalization_factor
            deviation = np.max(
                np.abs(normalized_target * normalization_factor - observed), initial=0.0
            )
        else:
            observed = np.asarray(row["target"], dtype=np.float64)
            predicted = np.asarray(row["mu"], dtype=np.float64)
            normalized_shape = np.asarray(row["normalized_shape"], dtype=np.float64)
            scale = float(row["scale_dt"])
            deviation = np.max(np.abs(normalized_shape * scale - predicted), initial=0.0)
        max_conversion_deviation = max(max_conversion_deviation, float(deviation))

        if observed.shape != predicted.shape or observed.ndim != 1:
            raise ValueError(
                f"Shape mismatch for {transcript_id} in {path}: "
                f"observed={observed.shape}, predicted={predicted.shape}"
            )
        n_total = int(observed.size)
        if n_total <= 2 * trim_codons:
            results.append(
                {
                    "transcript_id": transcript_id,
                    "n_total_positions": n_total,
                    "n_positions_after_trim": 0,
                    "n_positive_positions": 0,
                    "n_rmse_positions": 0,
                    "pcc": np.nan,
                    "scc": np.nan,
                    "rmse_normalized": np.nan,
                    "rmse_normalization_mean": np.nan,
                    "validity": "too_short_after_trim",
                    "target_digest": profile_digest(observed),
                }
            )
            continue

        interior = slice(trim_codons, n_total - trim_codons)
        observed_eval = observed[interior]
        predicted_eval = predicted[interior]
        valid = (
            np.isfinite(observed_eval)
            & np.isfinite(predicted_eval)
            & (observed_eval > 0.0)
        )
        y = observed_eval[valid]
        y_hat = predicted_eval[valid]
        pcc = pearson(y_hat, y)
        scc = spearman(y_hat, y)
        finite_target = observed_eval[np.isfinite(observed_eval)]
        normalization_mean = (
            float(np.mean(finite_target)) if finite_target.size else np.nan
        )
        rmse = (
            float(
                np.sqrt(
                    np.mean(
                        np.square(
                            y_hat / normalization_mean - y / normalization_mean
                        )
                    )
                )
            )
            if y.size
            and np.isfinite(normalization_mean)
            and normalization_mean > 0.0
            else np.nan
        )

        if y.size == 0:
            validity = "no_positive_finite_positions"
        elif y.size < 2:
            validity = "fewer_than_two_positive_positions"
        elif np.ptp(y) == 0.0:
            validity = "constant_target"
        elif np.ptp(y_hat) == 0.0:
            validity = "constant_prediction"
        else:
            validity = "valid"

        results.append(
            {
                "transcript_id": transcript_id,
                "n_total_positions": n_total,
                "n_positions_after_trim": int(observed_eval.size),
                "n_positive_positions": int(y.size),
                "n_rmse_positions": int(y.size),
                "pcc": pcc,
                "scc": scc,
                "rmse_normalized": rmse,
                "rmse_normalization_mean": normalization_mean,
                "validity": validity,
                "target_digest": profile_digest(observed),
            }
        )

    frame = pd.DataFrame(results)
    for key in ("architecture", "protocol", "organism", "checkpoint_rule"):
        frame.insert(len(frame.columns) - 1, key, str(spec[key]))
    audit = {
        "architecture": spec["architecture"],
        "protocol": spec["protocol"],
        "organism": spec["organism"],
        "prediction_path": str(path.resolve()),
        "n_rows": int(len(frame)),
        "n_too_short": int((frame["validity"] == "too_short_after_trim").sum()),
        "n_valid_pcc": int(frame["pcc"].notna().sum()),
        "n_valid_scc": int(frame["scc"].notna().sum()),
        "n_valid_rmse_normalized": int(frame["rmse_normalized"].notna().sum()),
        "max_native_scale_conversion_deviation": max_conversion_deviation,
    }
    return frame, audit


def bootstrap_median_interval(
    values: np.ndarray,
    n_resamples: int,
    seed: int,
    context: str,
    batch_size: int = 256,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    context_seed = zlib.crc32(context.encode("utf-8"))
    rng = np.random.default_rng(np.random.SeedSequence([seed, context_seed]))
    medians = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, batch_size):
        stop = min(start + batch_size, n_resamples)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        medians[start:stop] = np.median(values[indices], axis=1)
    lower, upper = np.percentile(medians, (2.5, 97.5))
    return float(lower), float(upper)


def summarize(
    metrics: pd.DataFrame,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    summaries: list[dict] = []
    group_columns = ["architecture", "protocol", "organism", "checkpoint_rule"]
    for keys, group in metrics.groupby(group_columns, sort=False):
        metadata = dict(zip(group_columns, keys, strict=True))
        for metric in ("pcc", "scc", "rmse_normalized"):
            values = group[metric].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            context = "|".join((*map(str, keys), metric))
            lower, upper = bootstrap_median_interval(
                finite, n_resamples=n_resamples, seed=seed, context=context
            )
            summaries.append(
                {
                    **metadata,
                    "metric": metric,
                    "n_total_transcripts": int(len(group)),
                    "n_valid_transcripts": int(finite.size),
                    "median": float(np.median(finite)) if finite.size else np.nan,
                    "ci_2p5": lower,
                    "ci_97p5": upper,
                    "ci_95_half_width": (
                        float((upper - lower) / 2.0)
                        if np.isfinite(lower) and np.isfinite(upper)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(summaries)


def load_external_summary(
    path: Path,
    provenance_path: Path,
    architecture: str,
    training_to_protocol: dict[str, str],
    trim_codons: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Validate and translate a supplied aggregate benchmark summary.

    Position-level exports for these external models are not stored in this
    repository, so this function checks the endpoint metadata before import.
    All imported rows use the full canonical test definition; native settings
    may nevertheless have fewer saved or metric-valid transcript profiles.
    """
    source = pd.read_csv(path, sep="\t")
    required = {
        "dataset",
        "model",
        "training",
        "split",
        "scope",
        "aggregation",
        "test_cohort",
        "n_test_transcripts",
        "n_saved_test_transcripts",
        "correlation_region",
        "rmse_region",
        "rmse_units",
        "rmse_additional_normalization",
        "rmse_n_positive_positions",
        "n_positive_positions",
        "n_bootstrap",
        "bootstrap_seed",
        "bootstrap_unit",
        "ci_method",
    }
    for metric in ("pcc", "scc", "rmse"):
        required.update(
            {
                f"{metric}_n_transcripts",
                f"{metric}_median",
                f"{metric}_ci95_half_width",
            }
        )
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(
            f"{architecture} summary is missing required columns: {missing}"
        )

    import_mask = source["model"].eq(architecture) & source["training"].isin(
        training_to_protocol
    )
    imported = source.loc[import_mask].copy()
    expected_keys = {
        (dataset, training)
        for dataset in ORGANISMS
        for training in training_to_protocol
    }
    observed_keys = set(zip(imported["dataset"], imported["training"], strict=True))
    if observed_keys != expected_keys or len(imported) != len(expected_keys):
        raise ValueError(
            f"{architecture} rows do not form exactly one dataset-by-training grid: "
            f"expected={sorted(expected_keys)}, observed={sorted(observed_keys)}"
        )

    expected_region = f"CDS[{trim_codons}:-{trim_codons}]"
    scalar_expectations = {
        "split": "test",
        "test_cohort": "full_canonical_test",
        "scope": "observed_gt_0",
        "aggregation": "median_across_transcripts",
        "correlation_region": expected_region,
        "rmse_region": expected_region,
        "rmse_units": "trimmed_observed_mean_one",
        "rmse_additional_normalization": (
            "both_arrays_divided_by_trimmed_target_mean_including_zeros"
        ),
        "n_bootstrap": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_unit": "transcript",
        "ci_method": "percentile",
    }
    for column, expected in scalar_expectations.items():
        values = set(imported[column].tolist())
        if values != {expected}:
            raise ValueError(
                f"{architecture} endpoint mismatch for {column}: expected {expected!r}, "
                f"observed {sorted(values, key=str)!r}"
            )
    if np.any(
        imported["rmse_n_positive_positions"].to_numpy()
        > imported["n_positive_positions"].to_numpy()
    ):
        raise ValueError(
            f"{architecture} RMSE position count exceeds available positives"
        )
    if np.any(
        imported["n_saved_test_transcripts"].to_numpy()
        > imported["n_test_transcripts"].to_numpy()
    ):
        raise ValueError(
            f"{architecture} saved test count exceeds the canonical test count"
        )
    canonical = imported[imported["training"].isin(("unweighted", "weighted"))]
    native_training = next(
        training
        for training, protocol in training_to_protocol.items()
        if protocol == "Own preprocessing"
    )
    native = imported[imported["training"].eq(native_training)]
    provenance_source = pd.read_csv(provenance_path, sep="\t")
    provenance_required = {
        "dataset",
        "model",
        "training",
        "n_training_transcripts",
        "n_validation_transcripts",
        "selected_epoch",
        "epoch_numbering",
        "epochs_trained",
        "checkpoint_criterion",
        "criterion_metric",
        "criterion_direction",
        "selected_checkpoint",
        "source_run",
    }
    provenance_missing = sorted(provenance_required.difference(provenance_source.columns))
    if provenance_missing:
        raise ValueError(
            f"{architecture} provenance is missing required columns: "
            f"{provenance_missing}"
        )
    provenance = provenance_source.loc[
        provenance_source["model"].eq(architecture)
        & provenance_source["training"].isin(training_to_protocol)
    ].copy()
    provenance_keys = set(zip(provenance["dataset"], provenance["training"], strict=True))
    if provenance_keys != expected_keys or len(provenance) != len(expected_keys):
        raise ValueError(
            f"{architecture} provenance does not form the expected grid: "
            f"expected={sorted(expected_keys)}, observed={sorted(provenance_keys)}"
        )
    provenance_by_key = {
        (str(row.dataset), str(row.training)): row
        for row in provenance.itertuples(index=False)
    }

    rows: list[dict] = []
    metric_names = {"pcc": "pcc", "scc": "scc", "rmse": "rmse_normalized"}
    for row in imported.itertuples(index=False):
        provenance_row = provenance_by_key[(str(row.dataset), str(row.training))]
        for source_metric, metric in metric_names.items():
            rows.append(
                {
                    "architecture": architecture,
                    "protocol": training_to_protocol[str(row.training)],
                    "organism": str(row.dataset),
                    "checkpoint_rule": str(provenance_row.checkpoint_criterion),
                    "metric": metric,
                    "n_total_transcripts": int(row.n_test_transcripts),
                    "n_valid_transcripts": int(
                        getattr(row, f"{source_metric}_n_transcripts")
                    ),
                    "median": float(getattr(row, f"{source_metric}_median")),
                    # The supplied TSV stores only the interval half-width.
                    "ci_2p5": np.nan,
                    "ci_97p5": np.nan,
                    "ci_95_half_width": float(
                        getattr(row, f"{source_metric}_ci95_half_width")
                    ),
                    "summary_source": "external_consolidated_canonical_tsv",
                }
            )
    audit = {
        "path": str(path.resolve()),
        "sha256": file_digest(path),
        "provenance_path": str(provenance_path.resolve()),
        "provenance_sha256": file_digest(provenance_path),
        "n_source_rows": int(len(source)),
        "n_provenance_source_rows": int(len(provenance_source)),
        "n_rows_imported": int(len(imported)),
        "n_canonical_rows_imported": int(len(canonical)),
        "n_native_rows_imported": int(len(native)),
        "n_unrecognized_rows_excluded": int((~import_mask).sum()),
        "native_cohort_sizes": {
            str(row.dataset): int(row.n_test_transcripts)
            for row in native.itertuples(index=False)
        },
        "native_saved_test_transcripts": {
            str(row.dataset): int(row.n_saved_test_transcripts)
            for row in native.itertuples(index=False)
        },
        "native_rmse_valid_transcripts": {
            str(row.dataset): int(row.rmse_n_transcripts)
            for row in native.itertuples(index=False)
        },
        "held_out_status": {
            str(row.dataset) + "/" + str(row.training): (
                None if pd.isna(row.test_is_held_out) else bool(row.test_is_held_out)
            )
            for row in imported.itertuples(index=False)
        },
        "validated_metadata": scalar_expectations,
        "position_level_revalidation": (
            "not possible from the aggregate TSV; endpoint metadata and the "
            "separate checkpoint-provenance grid were audited"
        ),
    }
    return pd.DataFrame(rows), audit


def validate_common_targets(metrics: pd.DataFrame) -> dict[str, dict[str, int]]:
    audit: dict[str, dict[str, int]] = {}
    for organism, group in metrics.groupby("organism", sort=False):
        reference = group[
            (group["architecture"] == "RiboMIMO")
            & (group["protocol"] == "RiboUnmix weighted")
        ][["transcript_id", "target_digest"]].set_index("transcript_id")
        if reference.index.has_duplicates:
            raise ValueError(f"Duplicate reference transcript IDs for {organism}")
        mismatches = 0
        compared_rows = 0
        for (architecture, protocol), candidate in group.groupby(
            ["architecture", "protocol"], sort=False
        ):
            candidate = candidate[["transcript_id", "target_digest"]].set_index("transcript_id")
            if set(candidate.index) != set(reference.index):
                raise ValueError(
                    f"Test transcript IDs differ for {organism}, {architecture}, {protocol}"
                )
            aligned = candidate.loc[reference.index]
            mismatches += int((aligned["target_digest"] != reference["target_digest"]).sum())
            compared_rows += int(len(aligned))
        if mismatches:
            raise ValueError(
                f"Observed test profiles differ across methods for {organism}: "
                f"{mismatches}/{compared_rows} profile hashes mismatch"
            )
        audit[str(organism)] = {
            "n_transcripts": int(len(reference)),
            "n_compared_profile_rows": compared_rows,
            "n_target_profile_mismatches": mismatches,
        }
    return audit


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else project_root / "results/cross_organism_benchmark_reanalysis"
    )
    if args.trim_codons < 0:
        raise ValueError("--trim-codons must be non-negative")
    if args.bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    file_audits: list[dict] = []
    specs = prediction_paths(project_root)
    for spec in specs:
        print(
            f"Evaluating {spec['architecture']} / {spec['protocol']} / "
            f"{spec['organism']}"
        )
        frame, audit = evaluate_file(spec, trim_codons=args.trim_codons)
        frames.append(frame)
        file_audits.append(audit)
    metrics = pd.concat(frames, ignore_index=True)
    target_audit = validate_common_targets(metrics)
    summary = summarize(
        metrics,
        n_resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )

    expected_test_sizes = (
        summary.groupby("organism", sort=False)["n_total_transcripts"].first().to_dict()
    )
    external_frames: list[pd.DataFrame] = []
    external_audits: dict[str, dict] = {}
    external_path = project_root / "results" / EXTERNAL_SUMMARY_PATH
    provenance_path = project_root / "results" / EXTERNAL_PROVENANCE_PATH
    for architecture, external_spec in EXTERNAL_SUMMARY_SPECS.items():
        external_summary, external_audit = load_external_summary(
            external_path,
            provenance_path,
            architecture=architecture,
            training_to_protocol=dict(external_spec["training_to_protocol"]),
            trim_codons=args.trim_codons,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
        for organism, observed in (
            external_summary.groupby("organism", sort=False)[
                "n_total_transcripts"
            ].first().items()
        ):
            expected = int(expected_test_sizes[organism])
            if int(observed) != expected:
                raise ValueError(
                    f"{architecture} canonical test-cohort size differs for "
                    f"{organism}: external={int(observed)}, common={expected}"
                )
        external_frames.append(external_summary)
        external_audits[architecture] = external_audit
    summary.insert(len(summary.columns), "summary_source", "recomputed_from_profiles")
    combined_summary = pd.concat([*external_frames, summary], ignore_index=True)

    endpoint = f"trim{args.trim_codons}_normalized_rmse"
    metrics_path = output_dir / f"per_transcript_metrics_{endpoint}.csv.gz"
    summary_path = output_dir / f"summary_{endpoint}.csv"
    combined_summary_path = output_dir / f"combined_summary_{endpoint}.csv"
    audit_path = output_dir / "analysis_manifest.json"
    public_columns = [column for column in metrics.columns if column != "target_digest"]
    metrics[public_columns].to_csv(metrics_path, index=False, compression="gzip")
    summary.to_csv(summary_path, index=False)
    combined_summary.to_csv(combined_summary_path, index=False)

    manifest = {
        "analysis": "cross_organism_benchmark_trimmed_positive_position_metrics",
        "project_root": str(project_root),
        "trim_codons_per_end": args.trim_codons,
        "evaluation_mask": (
            "finite prediction and finite observed count with observed count > 0"
        ),
        "rmse_definition": (
            "sqrt(mean(((prediction_raw / observed_full_window_mean) - "
            "(observed_raw / observed_full_window_mean))^2)) over positive-count "
            "positions; observed_full_window_mean includes zero-count positions"
        ),
        "prediction_rescaling": {
            "RiboMIMO": "saved mu multiplied by saved normalization_factor",
            "RiboUnmix": "saved mu already has the same units as saved target",
        },
        "rmse_common_target_scale_normalization_applied": True,
        "prediction_independent_normalization_applied": False,
        "rmse_normalization_scope": (
            "complete five-codon-trimmed target window, including zeros"
        ),
        "bootstrap": {
            "unit": "transcript",
            "statistic": "median transcript-level metric",
            "method": "nonparametric percentile",
            "resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "reported_interval": "[2.5th percentile, 97.5th percentile]",
            "table_display": (
                "median +/- (97.5th percentile - 2.5th percentile) / 2"
            ),
        },
        "common_target_validation": target_audit,
        "prediction_file_audits": file_audits,
        "external_summary_audits": external_audits,
        "external_summary_limitation": (
            "iXnos, RiboExp, Riboformer, and Seq2Ribo rows were imported from the "
            "consolidated aggregate TSV after endpoint and checkpoint-provenance "
            "validation. Their position-level exports are not stored locally, so "
            "transcript identities and target-profile hashes could not be "
            "independently compared with the RiboMIMO/RiboUnmix files. Held-out "
            "split overlap is explicitly verified in the supplied file only for "
            "Riboformer; blank fields for other architectures remain unverified."
        ),
        "outputs": {
            "per_transcript_metrics": str(metrics_path.resolve()),
            "summary": str(summary_path.resolve()),
            "combined_summary": str(combined_summary_path.resolve()),
        },
    }
    audit_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    display = combined_summary.pivot_table(
        index=["architecture", "protocol", "organism"],
        columns="metric",
        values=["median", "ci_2p5", "ci_97p5"],
        aggfunc="first",
    )
    print("\n", display.to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"\nWrote {metrics_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {combined_summary_path}")
    print(f"Wrote {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
