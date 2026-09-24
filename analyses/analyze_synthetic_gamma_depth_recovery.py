#!/usr/bin/env python3
"""Matched-depth gamma recovery from frozen best-PCC prediction exports.

Use the repository's two-way log-centering operator on the identical interior
of truth and predictions, then score the exponentiated multipliers. No neural
fitting, inference, changed reference weights, or codon-level pandas table.
Only selected validation profiles are retained, one experiment at a time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyses.audit_synthetic_read_depth import array_hash, rows
from analyses.analyze_synthetic_gamma_recovery import joint_log_gamma_gauge
from analyses.analyze_synthetic_recovery import _encoding_from_config
from analyses.plot_synthetic_read_depth_effect import (
    BIAS_ORDER, COUNTS, DEPTHS, cohort_hash, complete_cohort, file_hash,
    mean_stats, verified_runs,
)

BIAS_ROOT = ROOT / "Datasets/Synthetic_data/bias_profile"
TRIM = 10
METRICS = ("gamma_pcc", "gamma_rmse", "log_gamma_pcc", "log_gamma_rmse",
           "sense_gauge_gamma_pcc", "sense_gauge_gamma_rmse")
PLOTTED_METRICS = METRICS[:2]


def profile_scores(prediction, reference):
    """Direct-scale PCC/RMSE, with explicit degeneracy rather than PCC zero."""
    x, y = np.asarray(prediction, dtype=float), np.asarray(reference, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("Gamma vectors must have the same one-dimensional shape.")
    if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return dict(pcc=np.nan, rmse=np.nan, valid=False, reason="short_or_nonfinite",
                    prediction_variance=np.nan, reference_variance=np.nan)
    dx, dy = x-x.mean(), y-y.mean()
    nx, ny = np.linalg.norm(dx), np.linalg.norm(dy)
    valid = nx * ny > 1e-12
    reason = "ok" if valid else ("constant_reference" if ny <= nx else "constant_prediction")
    return dict(pcc=float(np.clip(dx @ dy / (nx*ny), -1, 1)) if valid else np.nan,
                rmse=float(np.sqrt(np.mean((x-y)**2))), valid=bool(valid), reason=reason,
                prediction_variance=float(np.mean(dx**2)), reference_variance=float(np.mean(dy**2)))


def score_gamma_matrix(log_prediction, log_bias, weights):
    """Align the gauge domain before scoring; retain a full-sense-gauge control."""
    predicted, truth = np.asarray(log_prediction), np.asarray(log_bias)
    if predicted.shape != truth.shape or predicted.ndim != 2 or predicted.shape[1] < 22:
        raise ValueError("Need aligned dataset-by-sense-codon matrices with at least two interior codons.")
    interior = np.zeros(predicted.shape[1], dtype=bool)
    interior[TRIM:-TRIM] = True
    p = joint_log_gamma_gauge(predicted, weights, interior)
    q = joint_log_gamma_gauge(truth, weights, interior)
    p_full = joint_log_gamma_gauge(predicted, weights)[:, interior]
    q_full = joint_log_gamma_gauge(truth, weights)[:, interior]
    result = []
    for d in range(len(weights)):
        direct = profile_scores(np.exp(p[d]), np.exp(q[d]))
        log = profile_scores(p[d], q[d])
        full = profile_scores(np.exp(p_full[d]), np.exp(q_full[d]))
        result.append(dict(
            gamma_pcc=direct["pcc"], gamma_rmse=direct["rmse"],
            log_gamma_pcc=log["pcc"], log_gamma_rmse=log["rmse"],
            sense_gauge_gamma_pcc=full["pcc"], sense_gauge_gamma_rmse=full["rmse"],
            pcc_valid=direct["valid"], reason=direct["reason"],
            predicted_gamma_variance=direct["prediction_variance"],
            reference_gamma_variance=direct["reference_variance"],
            prediction_interior_log_shift=float((p[d]-predicted[d, interior]).mean()),
            reference_interior_log_shift=float((q[d]-truth[d, interior]).mean())))
    return result


def load_log_biases(ids):
    """Retain the selected truth only; verify replica and mean annotations agree."""
    profiles, lengths = {}, {}
    for bias in BIAS_ORDER:
        name = f"artificial_bias_{bias}"
        path = BIAS_ROOT / f"{name}_compendium_added_bias_only.parquet"
        selected, samples = {}, {}
        for row in rows(path, ["transcript_id", "sample", "added_bias"], batch_size=32):
            tid = row["transcript_id"]
            if tid not in ids:
                continue
            a = np.asarray(row["added_bias"], dtype=float)
            if not np.isfinite(a).all() or np.any(a < 0):
                raise ValueError(f"Invalid bias multiplier annotation: {name}/{tid}")
            value = np.log1p(a)
            suffix = row["sample"].removeprefix(bias + "_")
            if suffix in samples.setdefault(tid, set()):
                raise ValueError(f"Duplicate bias annotation: {name}/{tid}/{suffix}")
            samples[tid].add(suffix)
            if tid in selected and not np.allclose(selected[tid], value, rtol=0, atol=1e-12):
                raise ValueError(f"Bias differs across replicas: {name}/{tid}")
            selected[tid] = value
            if tid in lengths and lengths[tid] != len(value):
                raise ValueError(f"Bias annotations are misaligned: {name}/{tid}")
            lengths[tid] = len(value)
        if set(selected) != set(ids) or any(s != {"rep1", "rep2", "mean"} for s in samples.values()):
            raise ValueError(f"Missing selected bias annotations: {path}")
        profiles[name] = selected
    return profiles, lengths


def evaluate_gamma_run(run, ids, truth, lengths):
    config = yaml.load((ROOT / run["config_path"]).read_text(), Loader=yaml.CSafeLoader)
    if config["model"]["gamma_centering"].get("dataset_constant_scale_gauge") != "geometric_mean_one":
        raise ValueError("This analysis requires the checkpoint's two-way geometric-mean-one convention.")
    id_to_name = _encoding_from_config(config, ROOT)
    name_to_id = {name: did for did, name in id_to_name.items()}
    datasets = run["datasets"]
    n = len(datasets)
    expected_dids = {name_to_id[d] for d in datasets}
    selected, exp_difference, exp_relative_difference = {}, 0.0, 0.0
    columns = ["transcript_id", "dataset_id", "length", "mask", "gamma", "log_gamma",
               "gamma_centering_applied", "gamma_centering_reliability",
               "gamma_reference_dataset_ids", "gamma_reference_weighting"]
    for row in rows(ROOT / run["prediction_path"], columns, batch_size=32):
        tid = row["transcript_id"]
        if tid not in ids:
            continue
        did = int(row["dataset_id"])
        model_length = lengths[tid] + 1
        if (did not in expected_dids or int(row["length"]) != model_length
                or set(row["gamma_reference_dataset_ids"]) != expected_dids
                or row["gamma_reference_weighting"] != "equal"):
            raise ValueError(f"Gamma identity, reference, or length mismatch: {run['run']}/{tid}")
        if not np.all(row["mask"][:model_length]) or np.any(row["mask"][model_length:]):
            raise ValueError(f"Unexpected gamma sequence mask: {tid}")
        if (not np.all(row["gamma_centering_applied"][:model_length])
                or not np.allclose(row["gamma_centering_reliability"][:model_length], 1/n)):
            raise ValueError(f"Uniform gamma centering was not applied: {tid}")
        log = np.asarray(row["log_gamma"][:model_length], dtype=float)
        gamma = row["gamma"][:model_length]
        if log.size != model_length or len(gamma) != model_length or not np.isfinite(log).all():
            raise ValueError(f"Incomplete or nonfinite gamma profile: {tid}")
        if not np.allclose(np.exp(log), gamma, rtol=2e-6, atol=1e-7):
            raise ValueError(f"Saved gamma and exp(log_gamma) disagree: {tid}")
        exp_difference = max(exp_difference, float(np.max(np.abs(np.exp(log)-gamma))))
        exp_relative_difference = max(exp_relative_difference,
            float(np.max(np.abs(np.exp(log)-gamma)/np.exp(log))))
        key = tid, id_to_name[did]
        if key in selected:
            raise ValueError(f"Duplicate gamma prediction: {key}")
        selected[key] = log.copy()
    expected = {(tid, d) for tid in ids for d in datasets}
    if set(selected) != expected:
        raise ValueError(f"Missing frozen gamma profiles: {sorted(expected-set(selected))[:5]}")
    records, diagnostics = [], []
    digest = hashlib.sha256()
    weights = np.full(n, 1/n)
    for tid in sorted(ids):
        full = np.stack([selected.pop((tid, d)) for d in datasets])
        cross = float(np.max(np.abs(weights @ full)))
        positional = float(np.max(np.abs(full.mean(axis=1))))
        if max(cross, positional) > 5e-5:
            raise ValueError(f"Saved gamma violates its native two-way constraint: {tid}")
        digest.update((tid + array_hash(full)).encode())
        log_truth = np.stack([truth[d][tid] for d in datasets])
        scores = score_gamma_matrix(full[:, :-1], log_truth, weights)
        for d, score in zip(datasets, scores):
            records.append(dict(run=run["run"], depth=run["depth"], n_datasets=n,
                                transcript_id=tid, dataset=d, sense_length=lengths[tid],
                                n_positions=lengths[tid]-2*TRIM, **score))
        diagnostics.append(dict(run=run["run"], transcript_id=tid,
                                native_cross_dataset_constraint_max=cross,
                                native_positional_constraint_max=positional))
    return records, diagnostics, dict(run=run["run"],
                                     selected_log_gamma_sha256=digest.hexdigest(),
                                     gamma_vs_exp_log_gamma_max_abs=exp_difference,
                                     gamma_vs_exp_log_gamma_max_relative=exp_relative_difference)


def aggregate_transcripts(frame):
    """First average datasets within transcript; never silently drop invalid PCCs."""
    result = []
    keys = ["run", "depth", "n_datasets", "transcript_id"]
    for identity, group in frame.groupby(keys, sort=True):
        if len(group) != identity[2] or group.dataset.nunique() != identity[2]:
            raise ValueError("Incomplete transcript-by-reference-panel gamma metrics.")
        result.append(dict(zip(keys, identity)) | dict(
            n_valid_dataset_pcc=int(group.pcc_valid.sum()),
            **{metric: float(group[metric].mean()) if np.isfinite(group[metric]).all() else np.nan
               for metric in METRICS}))
    return pd.DataFrame(result)


def load_or_analyze_gamma(runs, ids, out, force=False):
    """Reuse compact scalars only when inputs, cohort and analysis code agree."""
    out.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), ROOT / "analyses/analyze_synthetic_gamma_recovery.py",
               ROOT / "analyses/audit_synthetic_read_depth.py",
               ROOT / "analyses/analyze_synthetic_recovery.py",
               ROOT / "Datasets/encodings/synthetic_dataset_encoding.yaml"]
    sources += [BIAS_ROOT / f"artificial_bias_{b}_compendium_added_bias_only.parquet" for b in BIAS_ORDER]
    signature = dict(cohort_hash=cohort_hash(ids), n_transcripts=len(ids), boundary_trim_codons=TRIM,
                     input_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in sources},
                     predictions={r["run"]: dict(path=r["prediction_path"],
                         size_bytes=(ROOT/r["prediction_path"]).stat().st_size,
                         mtime_ns=(ROOT/r["prediction_path"]).stat().st_mtime_ns,
                         config_sha256=r["config_sha256"]) for r in runs})
    manifest_path = out / "provenance.json"
    if manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text())
        if manifest["signature"] == signature and all(
                (out/name).is_file() and file_hash(out/name) == value
                for name, value in manifest["output_sha256"].items()):
            print("Reusing verified compact gamma metrics.", flush=True)
            return (pd.read_csv(out / "gamma_per_transcript_dataset.csv"),
                    pd.read_csv(out / "gamma_per_transcript.csv"), manifest)
    truth, lengths = load_log_biases(ids)
    records, diagnostics, checks = [], [], []
    for run in runs:
        current, diag, check = evaluate_gamma_run(run, ids, truth, lengths)
        records.extend(current)
        diagnostics.extend(diag)
        checks.append(check)
        print(f"gamma {run['depth']}, N={len(run['datasets'])}: {len(current)} scalar rows", flush=True)
    detailed = pd.DataFrame(records)
    transcripts = aggregate_transcripts(detailed)
    outputs = dict(gamma_per_transcript_dataset=detailed, gamma_per_transcript=transcripts,
                   gamma_native_gauge_diagnostics=pd.DataFrame(diagnostics))
    for name, frame in outputs.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    manifest = dict(signature=signature, checks=checks, checkpoint_variant="best_pcc", seed=42,
                    reference_weights="pi_d=1/N, preserved", training_or_inference=False,
                    primary="exp(two-way-gauge(log_gamma)) vs exp(two-way-gauge(log(1+added_bias)))",
                    gauge_domain="sense interior: zero-based 10 <= i < sense_length-10",
                    sensitivity="gauge on full sense CDS, then score the same trimmed interior",
                    aggregation="equal dataset means within each transcript, then equal transcript means",
                    model_export_domain="sense CDS plus terminal; native gauge checked on this full domain",
                    undefined_pcc_rows=int((~detailed.pcc_valid).sum()),
                    output_sha256={f"{name}.csv": file_hash(out/f"{name}.csv") for name in outputs})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return detailed, transcripts, manifest


def write_gamma_summaries(detailed, transcripts, ids, out, repeats, seed):
    if not ids or repeats < 2:
        raise ValueError("Gamma summaries require a nonempty matched cohort and at least two bootstrap draws.")
    draws = np.random.default_rng(seed).integers(len(ids), size=(repeats, len(ids)))
    lookup = transcripts.set_index(["depth", "n_datasets", "transcript_id"]).sort_index()
    summary, datasets, differences = [], [], []
    for n in COUNTS:
        for depth in DEPTHS:
            cell = lookup.loc[(depth, n)].reindex(ids)
            for metric in METRICS:
                summary.append(dict(depth=depth, n_datasets=n, metric=metric,
                                    displayed=metric in PLOTTED_METRICS, n_transcripts=len(ids),
                                    cohort_hash=cohort_hash(ids), **mean_stats(cell[metric], draws)))
            sub = detailed.loc[(detailed.depth == depth) & (detailed.n_datasets == n)]
            for dataset, group in sub.groupby("dataset"):
                for metric in PLOTTED_METRICS:
                    values = group.set_index("transcript_id")[metric].reindex(ids)
                    datasets.append(dict(depth=depth, n_datasets=n, dataset=dataset, metric=metric,
                                         n_transcripts=len(ids), **mean_stats(values, draws)))
        for i, low in enumerate(DEPTHS):
            for high in DEPTHS[i+1:]:
                for metric in PLOTTED_METRICS:
                    delta = lookup.loc[(high, n), metric].reindex(ids)-lookup.loc[(low, n), metric].reindex(ids)
                    differences.append(dict(n_datasets=n, depth_low=low, depth_high=high, metric=metric,
                                            n_transcripts=len(ids), direction="high_depth_minus_low_depth",
                                            **mean_stats(delta, draws)))
    result = pd.DataFrame(summary)
    result.to_csv(out / "gamma_summary.csv", index=False)
    pd.DataFrame(datasets).to_csv(out / "gamma_by_dataset_summary.csv", index=False)
    pd.DataFrame(differences).to_csv(out / "gamma_depth_differences.csv", index=False)
    amplitude = detailed.loc[detailed.transcript_id.isin(ids)].copy()
    amplitude["variance_ratio"] = amplitude.predicted_gamma_variance/amplitude.reference_gamma_variance
    # The ordinary least-squares slope with an intercept equals r * sigma_x/sigma_y.
    # This is a diagnostic of saved outputs, not fitting or modifying the neural model.
    amplitude["calibration_slope"] = amplitude.gamma_pcc*np.sqrt(amplitude.variance_ratio)
    amplitude.groupby(["depth", "n_datasets", "dataset"])[
        ["gamma_pcc", "gamma_rmse", "variance_ratio", "calibration_slope"]].mean().reset_index().to_csv(
            out / "gamma_amplitude_by_dataset.csv", index=False)
    (out / "summary_provenance.json").write_text(json.dumps(dict(
        cohort_ids=ids, cohort_hash=cohort_hash(ids), bootstrap_repeats=repeats, bootstrap_seed=seed,
        intervals="pointwise paired transcript-cluster percentile 95%, conditional on frozen models",
        metric_scale="multiplier, not log; log and full-sense-gauge controls also supplied"), indent=2)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=ROOT/"analyses/artifacts/synthetic/read_depth")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    runs, validation = verified_runs(args.audit_dir)
    candidates = set.intersection(*validation.values())
    out = args.output_dir or args.audit_dir/"depth_recovery_overview/gamma_recovery"
    detailed, transcripts, _ = load_or_analyze_gamma(runs, candidates, out, args.recompute)
    ids, excluded = complete_cohort(transcripts, candidates, DEPTHS, PLOTTED_METRICS)
    write_gamma_summaries(detailed, transcripts, ids, out, args.bootstrap_repeats, args.bootstrap_seed)
    print(f"Gamma recovery complete: {len(ids)} matched transcripts; excluded: {excluded}")


if __name__ == "__main__":
    main()
