#!/usr/bin/env python3
"""Audit separate-depth gamma recovery on the main shared-profile cohorts.

Only saved best-validation-loss predictions are read. Reuse the reference-target
audit's run identities, equal weights, and exact ten-codon-interior masks, and
the existing gamma audit's two-way log gauge. Shape correlation, centered
log-RMSE, and calibration are evaluated together. Prediction groups are
discarded when complete; only scalar results survive each run. No model is
loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses.analyze_synthetic_gamma_recovery import joint_log_gamma_gauge, _pcc
from analyses.analyze_synthetic_reference_target_audit import (
    DEPTH_ORDER, base_bias_name, discover_runs, load_biases, sha256,
)

DEFAULT_AUDIT = ROOT / "analyses/artifacts/synthetic/manuscript_revision/reference_target_best_val_loss"


def iter_transcripts(run, cohort):
    """Yield complete groups even when equal-length transcripts are interleaved."""
    seen = set()
    pending = {}
    columns = ["transcript_id", "dataset_id", "mask", "log_gamma"]
    with pq.ParquetFile(run.prediction_path) as reader:
        for batch in reader.iter_batches(columns=columns, batch_size=32, use_threads=False):
            for row in batch.to_pylist():
                tid = str(row["transcript_id"])
                if tid not in cohort:
                    continue
                if tid in seen:
                    raise ValueError(f"Duplicate completed transcript: {run.run_id}/{tid}")
                rows = pending.setdefault(tid, {})
                dataset = run.dataset_id_to_name[int(row["dataset_id"])]
                if dataset in rows:
                    raise ValueError(f"Duplicate prediction: {run.run_id}/{tid}/{dataset}")
                rows[dataset] = row
                if len(rows) == run.n_datasets:
                    seen.add(tid)
                    yield tid, pending.pop(tid)
    if pending or seen != cohort:
        raise ValueError(f"Incomplete prediction cohort for {run.run_id}: {len(seen)} / {len(cohort)}")


def evaluate(run, cohort, masks, biases, weights):
    pi = np.array([weights[dataset] for dataset in run.datasets], dtype=float)
    if not np.allclose(pi, 1 / run.n_datasets, rtol=0, atol=1e-12):
        raise ValueError(f"Nonuniform reference in {run.run_id}")
    scalars, transcript_scalars = [], []
    residual_max = 0.0
    mask_index = masks.set_index("transcript_id")
    for tid, rows in iter_transcripts(run, cohort):
        if set(rows) != set(run.datasets):
            raise ValueError(f"Incomplete dataset group: {run.run_id}/{tid}")
        audit = mask_index.loc[tid]
        sense_length = int(audit.sense_length)
        truth, predicted, common_take = [], [], None
        for dataset in run.datasets:
            row = rows[dataset]
            log_gamma = np.asarray(row["log_gamma"], dtype=np.float64)
            mask = np.asarray(row["mask"], dtype=bool)
            bias = biases[base_bias_name(dataset)][tid]
            if (log_gamma.shape != (int(audit.model_length),)
                    or mask.shape != log_gamma.shape or bias.shape != (sense_length,)):
                raise ValueError(f"Length disagreement with shared-profile audit: {run.run_id}/{tid}/{dataset}")
            # The saved sense positions are followed by at most one terminal
            # entry; no interpolation, truncation to a minimum, or reordering.
            if log_gamma.size != sense_length + int(audit.terminal_entry_removed):
                raise ValueError(f"Terminal convention differs for {run.run_id}/{tid}")
            positions = np.arange(sense_length)
            take = mask[:sense_length] & (positions >= 10) & (positions < sense_length - 10)
            mask_hash = hashlib.sha256(np.packbits(take).tobytes()).hexdigest()
            if mask_hash != audit.mask_sha256 or int(take.sum()) != audit.evaluated_positions:
                raise ValueError(f"Mask disagreement with panels A/B: {run.run_id}/{tid}/{dataset}")
            if common_take is not None and not np.array_equal(take, common_take):
                raise ValueError(f"Different dataset masks: {run.run_id}/{tid}")
            common_take = take
            predicted.append(log_gamma[:sense_length][take])
            truth.append(np.log(bias[take]))
        truth = joint_log_gamma_gauge(np.stack(truth), pi)
        predicted = joint_log_gamma_gauge(np.stack(predicted), pi)
        for values in (truth, predicted):
            residual_max = max(residual_max, float(np.abs(pi @ values).max()),
                               float(np.abs(values.mean(axis=1)).max()))
        pccs = []
        squared_errors = []
        for dataset, observed, target in zip(run.datasets, predicted, truth):
            # Exact constant targets are undefined; small cancellation residuals
            # from two-way centering also cannot define a meaningful PCC.
            defined = min(np.ptp(observed), np.ptp(target)) > 1e-12
            pcc = _pcc(observed, target) if defined else np.nan
            pccs.append(pcc)
            squared_errors.append(float(np.mean((observed-target)**2)))
            scalars.append({
                "run_id": run.run_id, "depth": run.depth, "n_datasets": run.n_datasets,
                "transcript_id": tid, "dataset": dataset,
                "log_gamma_pcc": pcc,
                "log_gamma_rmse": float(np.sqrt(squared_errors[-1])),
                "evaluated_positions": int(common_take.sum()), "mask_sha256": audit.mask_sha256,
                "valid": defined, "reason": "" if defined else "constant_profile",
            })
        # Do not silently vary the number of datasets contributing to a mean.
        complete = bool(np.isfinite(pccs).all())
        target_sum_squares = float(np.sum(truth**2))
        transcript_scalars.append({
            "run_id": run.run_id, "depth": run.depth, "n_datasets": run.n_datasets,
            "transcript_id": tid, "valid_dataset_pccs": int(np.isfinite(pccs).sum()),
            "mean_dataset_pcc": float(np.mean(pccs)) if complete else np.nan,
            # Every dataset uses the same retained positions, so this is the
            # RMSE over the complete dataset-by-position correction array.
            "joint_log_gamma_rmse": float(np.sqrt(np.mean(squared_errors))),
            "joint_calibration_slope": (
                float(np.sum(predicted * truth) / target_sum_squares)
                if target_sum_squares > 1e-24 else np.nan
            ),
        })
        if len(transcript_scalars) == 1 and complete:
            np.testing.assert_allclose(pccs, [np.corrcoef(x, y)[0, 1]
                for x, y in zip(predicted, truth)], rtol=0, atol=1e-12)
    if residual_max > 1e-12:
        raise ValueError(f"Two-way log gauge failed: max residual {residual_max}")
    return pd.DataFrame(scalars), pd.DataFrame(transcript_scalars), residual_max


def summarize(transcripts, bootstraps, seed):
    """Resample whole transcripts jointly across N and all reported metrics."""
    summaries = []
    for offset, depth in enumerate(DEPTH_ORDER):
        selected = transcripts.loc[transcripts.depth.eq(depth)]
        pcc = selected.pivot(
            index="transcript_id", columns="n_datasets", values="mean_dataset_pcc"
        ).sort_index()
        rmse = selected.pivot(
            index="transcript_id", columns="n_datasets", values="joint_log_gamma_rmse"
        ).reindex_like(pcc)
        slope = selected.pivot(
            index="transcript_id", columns="n_datasets", values="joint_calibration_slope"
        ).reindex_like(pcc)
        if pcc.columns.tolist() != list(range(2, 11)):
            raise ValueError(f"Incomplete size series: {depth}")
        # Correlation has the strictest eligibility rule. Use its fixed cohort
        # for all metrics so PCC and RMSE changes cannot be driven by different
        # transcripts. RMSE itself remains defined for constant targets.
        valid = pcc.notna().all(axis=1)
        if not rmse.loc[valid].notna().all().all() or not slope.loc[valid].notna().all().all():
            raise ValueError(f"RMSE/calibration is undefined on the fixed PCC cohort: {depth}")
        pcc_values = pcc.loc[valid].to_numpy()
        rmse_values = rmse.loc[valid].to_numpy()
        slope_values = slope.loc[valid].to_numpy()
        if not len(pcc_values):
            raise ValueError(f"Empty common gamma cohort: {depth}")
        rng = np.random.default_rng(seed + offset)
        pcc_estimates = np.empty((bootstraps, pcc_values.shape[1]))
        rmse_estimates = np.empty_like(pcc_estimates)
        slope_estimates = np.empty_like(pcc_estimates)
        for start in range(0, bootstraps, 50):
            stop = min(start+50, bootstraps)
            indices = rng.integers(
                0, len(pcc_values), size=(stop-start, len(pcc_values))
            )
            pcc_estimates[start:stop] = np.median(pcc_values[indices], axis=1)
            rmse_estimates[start:stop] = np.median(rmse_values[indices], axis=1)
            slope_estimates[start:stop] = np.median(slope_values[indices], axis=1)
        pcc_low, pcc_high = np.quantile(pcc_estimates, [0.025, 0.975], axis=0)
        rmse_low, rmse_high = np.quantile(rmse_estimates, [0.025, 0.975], axis=0)
        slope_low, slope_high = np.quantile(slope_estimates, [0.025, 0.975], axis=0)
        for col, n in enumerate(pcc.columns):
            pcc_median = float(np.median(pcc_values[:, col]))
            summaries.append({
                "run_id": selected.loc[selected.n_datasets.eq(n), "run_id"].iloc[0],
                "depth": depth, "n_datasets": n, "reference_weighting": "equal",
                "n_transcripts": len(pcc_values), "n_cohort_transcripts": len(pcc),
                "excluded_transcripts": int((~valid).sum()),
                # Keep the original names as explicit backward-compatible PCC
                # aliases for existing downstream audits.
                "median": pcc_median,
                "bootstrap_ci_low": pcc_low[col],
                "bootstrap_ci_high": pcc_high[col],
                "pcc_median": pcc_median,
                "pcc_bootstrap_ci_low": pcc_low[col],
                "pcc_bootstrap_ci_high": pcc_high[col],
                "log_rmse_median": float(np.median(rmse_values[:, col])),
                "log_rmse_bootstrap_ci_low": rmse_low[col],
                "log_rmse_bootstrap_ci_high": rmse_high[col],
                "calibration_slope_median": float(np.median(slope_values[:, col])),
                "calibration_slope_bootstrap_ci_low": slope_low[col],
                "calibration_slope_bootstrap_ci_high": slope_high[col],
                "bootstrap_seed": seed + offset, "bootstrap_resamples": bootstraps,
                "checkpoint_variant": "best_val_loss", "boundary_trim_codons": 10,
            })
    return pd.DataFrame(summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analyses/artifacts/synthetic/manuscript_revision/gamma_matched_depths_best_val_loss")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    started = time.monotonic()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs = {name: args.reference_audit / f"{name}.csv" for name in (
        "run_manifest", "matched_transcript_ids", "evaluation_masks", "panel_reference_weights")}
    frames = {name: pd.read_csv(path) for name, path in inputs.items()}
    manifest = frames["run_manifest"].query('family == "within_depth"').set_index("run_id")
    runs = discover_runs(ROOT / "results/riboai_synthetic_experiments", seed=42,
                         checkpoint_variant="best_val_loss", include_cross=False)
    if {run.run_id for run in runs} != set(manifest.index) or len(runs) != 27:
        raise ValueError("The 27 runs do not match panels A/B")
    masks = frames["evaluation_masks"]
    # The existing audit deduplicates identical masks across runs, so run_id
    # records the first occurrence, not the only run to which the mask applies.
    if masks.transcript_id.duplicated().any() or not masks.boundary_trim_codons.eq(10).all():
        raise ValueError("Expected the audited ten-codon-interior domain")
    cohorts = frames["matched_transcript_ids"]
    all_ids = set(cohorts.loc[cohorts.cohort_key.isin(manifest.analysis_cohort_key), "transcript_id"])
    print(f"Reading ten saved bias-annotation files for {len(all_ids)} distinct transcript IDs.", flush=True)
    biases, bias_provenance = load_biases(ROOT / "Datasets/Synthetic_data/bias_profile", all_ids)
    transcript_tables, run_provenance = [], []
    pair_writer = None
    try:
        for run in runs:
            saved = manifest.loc[run.run_id]
            if (run.prediction_path != (ROOT / saved.prediction_path)
                    or sha256(run.config_path) != saved.config_sha256
                    or sha256(run.checkpoint_manifest_path) != saved.checkpoint_manifest_sha256
                    or sha256(run.split_path) != saved.split_sha256):
                raise ValueError(f"Input provenance differs from panels A/B: {run.run_id}")
            config = yaml.safe_load(run.config_path.read_text())
            centering = config["model"]["gamma_centering"]
            expected_centering = {
                "mode": "fixed_reference",
                "dataset_constant_scale_gauge": "geometric_mean_one",
            }
            if (any(centering[key] != value for key, value in expected_centering.items())
                    or centering["reference"]["weighting"] != "equal"
                    or float(centering["reference"]["quality_rank_power"]) != 0.0):
                raise ValueError(f"Unexpected gamma-centering configuration: {run.run_id}")
            cohort = set(cohorts.loc[cohorts.cohort_key.eq(saved.analysis_cohort_key), "transcript_id"])
            if not cohort <= run.validation_ids or cohort & run.train_ids:
                raise ValueError(f"Cohort is not validation-only: {run.run_id}")
            pi_rows = frames["panel_reference_weights"].loc[frames["panel_reference_weights"].run_id.eq(run.run_id)]
            pi = dict(zip(pi_rows.dataset, pi_rows.pi))
            pairs, transcripts, residual = evaluate(run, cohort, masks, biases, pi)
            table = pa.Table.from_pandas(pairs, preserve_index=False)
            if pair_writer is None:
                pair_writer = pq.ParquetWriter(out / "gamma_per_transcript_dataset.parquet", table.schema, compression="zstd")
            pair_writer.write_table(table, row_group_size=1000)
            transcript_tables.append(transcripts)
            run_provenance.append({"run_id": run.run_id,
                "prediction_path": saved.prediction_path, "prediction_sha256": sha256(run.prediction_path),
                "matched_transcripts": len(cohort), "undefined_dataset_pccs": int((~pairs.valid).sum()),
                "max_gauge_residual": residual})
            print(f"{run.depth} N={run.n_datasets}: {len(cohort)} transcripts, "
                  f"{len(pairs)} pairs, {(~pairs.valid).sum()} undefined, gauge residual={residual:.2e}", flush=True)
    finally:
        if pair_writer is not None:
            pair_writer.close()
    transcript_table = pd.concat(transcript_tables, ignore_index=True)
    transcript_table.to_csv(out / "gamma_per_transcript.csv.gz", index=False)
    summary = summarize(transcript_table, args.bootstrap_resamples, args.seed)
    summary.to_csv(out / "gamma_summary.csv", index=False)
    (out / "provenance.json").write_text(json.dumps({
        "inputs": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in inputs.items()},
        "bias_annotations": bias_provenance, "runs": run_provenance,
        "checkpoint_variant": "best_val_loss", "models_loaded_or_trained": False,
        "mask": "exact A/B mask hash; appended terminal entry removed; 10 sense codons excluded at each end",
        "gauge": "existing joint_log_gamma_gauge applied to log b and saved log gamma on the same interior",
        "aggregation": (
            "mean dataset PCC and joint dataset-by-position log-RMSE within transcript, "
            "then medians over the same fixed PCC-eligible cohort"
        ),
        "bootstrap": {"resamples": args.bootstrap_resamples, "base_seed": args.seed,
                      "generator": "NumPy default_rng PCG64", "unit": "whole transcript; common indices across N within depth"},
        "non_claims": ["No independent test evaluation", "No independent training replicates", "Depth cohorts differ"],
        "tolerance": {"gauge_residual": 1e-12, "constant_profile_range": 1e-12},
        "elapsed_seconds": time.monotonic()-started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }, indent=2) + "\n")
    print(summary[[
        "depth", "n_datasets", "n_transcripts", "pcc_median",
        "log_rmse_median", "calibration_slope_median",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
