#!/usr/bin/env python3
"""Audit frozen synthetic recovery, count depth, and the reference-gauge effect.

No training, inference, checkpoint loading, or changes to existing analyses.
Read only requested Parquet columns in small batches. Biological recovery is
scored once per transcript, after verifying repeated dataset rows agree.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEPTHS = ("0p25_per_codon", "2_per_codon", "20_per_codon")


def id_hash(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def array_hash(array):
    return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes()).hexdigest()


def rows(path, columns, batch_size=64):
    """List-column views avoid constructing Python objects for every codon."""
    with pq.ParquetFile(path) as reader:
        for batch in reader.iter_batches(batch_size=batch_size, columns=columns,
                                        use_threads=False):
            data = {}
            for name in columns:
                col = batch.column(batch.schema.get_field_index(name))
                if pa.types.is_list(col.type) and not pa.types.is_list(col.type.value_type):
                    data[name] = (col.offsets.to_numpy(),
                                  col.values.to_numpy(zero_copy_only=False))
                else:
                    data[name] = col.to_pylist()
            for i in range(batch.num_rows):
                yield {name: value[1][value[0][i]:value[0][i+1]]
                       if isinstance(value, tuple) else value[i]
                       for name, value in data.items()}


def metrics(a, b, trim=10):
    if len(a) != len(b):
        raise ValueError(f"Unaligned profiles: {len(a)} != {len(b)}")
    sl = slice(trim, -trim if trim else None)
    x, y = np.asarray(a[sl], dtype=float), np.asarray(b[sl], dtype=float)
    if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return np.nan, np.nan
    dx, dy = x - x.mean(), y - y.mean()
    norm = np.linalg.norm(dx) * np.linalg.norm(dy)
    r = np.dot(dx, dy) / norm if norm > 1e-12 else np.nan
    return float(r), float(np.sqrt(np.mean((x-y)**2)))


def write_csv(path, data):
    if not data:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def inventory(root):
    runs, missing = [], []
    for directory in sorted(root.glob("riboai_synthetic_*")):
        configs = list(directory.rglob("config.yaml"))
        preds = list(directory.rglob("predictions_main_val_best_pcc_*.parquet"))
        if not preds:
            missing.append({"run": directory.name, "reason": "no best_pcc prediction export"})
            continue
        if len(configs) != 1 or len(preds) != 1:
            raise ValueError(f"Ambiguous artifacts: {directory}")
        config = yaml.load(configs[0].read_text(), Loader=yaml.CSafeLoader)
        split_path, = directory.rglob("split_manifest*.json")
        split = json.loads(split_path.read_text())
        checkpoint_path, = directory.rglob("prediction_checkpoint_manifest.json")
        checkpoints = json.loads(checkpoint_path.read_text())
        depth = next((d for d in DEPTHS if f"within_{d}_" in directory.name), "mixed")
        datasets = config["experiment"]["dataset"]
        ref = config["model"]["gamma_centering"]["reference"]
        validation = set(split["validation_ids"])
        training = set(split["train_ids"])
        assert not training & validation
        assert config["experiment"]["seed"] == 42
        assert config["model"]["gamma_centering"]["mode"] == "fixed_reference"
        assert all(x != "artificial_ground_truth" for x in datasets)
        if depth != "mixed":
            assert all(f"/{depth}/" in p for p in split["training_dataset_paths"])
            assert ref["weighting"] == "equal"
        # Resolved configs also embed large transcript records; retain only the
        # model/experiment settings, not a second copy of every split manifest.
        config = {k: config[k] for k in ("experiment", "model", "loss", "synthetic_ground_truth")}
        runs.append(dict(run=directory.name, directory=directory, config=config,
                         config_path=configs[0], split_path=split_path,
                         depth=depth, datasets=datasets, n_datasets=len(datasets),
                         weighting=ref["weighting"], validation=validation,
                         training=training, checkpoints=checkpoints,
                         training_paths=split["training_dataset_paths"]))
    runs.sort(key=lambda r: (r["depth"] == "mixed", r["depth"], r["n_datasets"], r["run"]))
    return runs, missing


def reference_profiles(requested):
    source = ROOT / "Datasets/Synthetic_data"
    kinetic_path = source / "artificial_ground_truth_kinetics_target_mean_one.parquet"
    kinetic = {r["transcript_id"]: r["rib_profile"].copy()
               for r in rows(kinetic_path, ["transcript_id", "rib_profile"])
               if r["transcript_id"] in requested}
    assert set(kinetic) == requested
    assert max(abs(x.mean()-1) for x in kinetic.values()) < 1e-10
    biases = {}
    for path in sorted((source / "bias_profile").glob("*.parquet")):
        name = path.name.removesuffix("_compendium_added_bias_only.parquet")
        profiles = {}
        for r in rows(path, ["sample", "transcript_id", "added_bias"]):
            tid = r["transcript_id"]
            if tid in requested and r["sample"].endswith("_rep1"):
                profiles[tid] = np.log1p(r["added_bias"])
        assert set(profiles) == requested
        biases[name] = profiles
    return kinetic, biases


def bias_weights(run):
    """Weights from the recorded equal/depth-ranked construction, checked below."""
    names = run["datasets"]
    weights = np.ones(len(names), dtype=float)
    if run["weighting"] == "quality_rank":
        assert run["config"]["model"]["gamma_centering"]["reference"]["quality_rank_power"] == 1
        weights = np.array([next(i+1 for i, d in enumerate(DEPTHS)
                                 if name.endswith("_"+d)) for name in names], dtype=float)
    weights /= weights.sum()
    answer = {}
    for name, weight in zip(names, weights):
        base = re.sub(r"_(0p25|2|20)_per_codon$", "", name)
        answer[base] = answer.get(base, 0.) + weight
    return answer, weights


def audit_counts(out, kinetic, cohorts):
    """Compare processed replicas with raw source draws and quantify actual depth."""
    summaries, observation_metrics, observed, target_hashes = [], [], {}, {}
    for depth in DEPTHS:
        observed[depth] = {}
        target_hashes[depth] = {}
        for path in sorted((ROOT / "Datasets/data/weighted_synthetic" / depth).glob("*.parquet")):
            raw_path = ROOT / "Datasets/Synthetic_data" / depth / f"{path.stem}_psite_counts_{depth}.parquet"
            raw_hashes = {}
            target_hashes[depth][path.stem] = {}
            for r in rows(raw_path, ["sample", "transcript_id", "rib_profile"]):
                for rep in ("rep1", "rep2"):
                    if r["sample"] == rep or r["sample"].endswith("_"+rep):
                        raw_hashes[(r["transcript_id"], rep)] = array_hash(r["rib_profile"])
            with pq.ParquetFile(raw_path) as f:
                metadata = {k.decode(): v.decode() for k, v in f.schema_arrow.metadata.items()
                            if k != b"ARROW:schema"}
            n = positions = zero = consensus_zero = mismatches = training_positions = 0
            counts = training_counts = consensus_discrepancy = 0.
            weights = []
            for r in rows(path, ["id", "ribo", "ribo_cds_replicas", "weight"]):
                tid = r["id"]
                reps = np.asarray(r["ribo_cds_replicas"], dtype=float)
                assert reps.shape[0] == 2 and np.all(reps[:, -1] == 0)
                reps = reps[:, :-1]  # preprocessing appends the terminal boundary
                for index, rep in enumerate(("rep1", "rep2")):
                    mismatches += array_hash(reps[index]) != raw_hashes[(tid, rep)]
                target = reps.mean(axis=0)  # actual dataloader consensus, NOT integerized mean
                if tid in kinetic:
                    target_hashes[depth][path.stem][tid] = array_hash(target)
                n += 1
                positions += target.size
                counts += reps.sum()
                zero += np.count_nonzero(reps == 0)
                consensus_zero += np.count_nonzero(target == 0)
                consensus_discrepancy += np.abs(r["ribo"][:-1]-target).sum()
                weights.append(r["weight"])
                if tid in cohorts[depth]["train"]:
                    training_positions += target.size
                    training_counts += reps.sum()
                if path.stem == "artificial_ground_truth" and tid in kinetic:
                    normalized = target / target.mean()
                    observed[depth][tid] = normalized
                    pcc, rmse = metrics(normalized, kinetic[tid])
                    observation_metrics.append(dict(depth=depth, transcript_id=tid,
                                                    pcc_K=pcc, rmse_K=rmse))
            summaries.append(dict(depth=depth, dataset=path.stem, transcripts=n,
                                  sense_positions=positions, reads_both_replicates=counts,
                                  reads_per_codon_per_replicate=counts/(2*positions),
                                  replicate_zero_fraction=zero/(2*positions),
                                  consensus_zero_fraction=consensus_zero/positions,
                                  training_positions=training_positions,
                                  training_reads_both_replicates=training_counts,
                                  raw_replica_mismatches=mismatches,
                                  stored_consensus_absdiff=consensus_discrepancy,
                                  weight_p05=float(np.quantile(weights, .05)),
                                  weight_median=float(np.median(weights)),
                                  weight_p95=float(np.quantile(weights, .95)),
                                  source_metadata=json.dumps(metadata, sort_keys=True)))
            print(f"counts {depth}/{path.stem}: {counts/(2*positions):.4f} reads/codon, "
                  f"{mismatches} mismatches", flush=True)
    write_csv(out / "input_count_audit.csv", summaries)
    write_csv(out / "unbiased_observation_per_transcript.csv", observation_metrics)
    return observed, target_hashes


def evaluate_run(run, variant, kinetic, biases, observed, common, target_hashes=None):
    path, = run["directory"].rglob(f"predictions_main_val_{variant}_*.parquet")
    names = ["transcript_id", "dataset_id", "length", "mask", "L_bio",
             "gamma_centering_reliability", "gamma_reference_dataset_count",
             "mu", "target"]
    records, profiles, seen_datasets, reference_weights = {}, {}, {}, {}
    nrows, duplicate_max, saved_mean_error = 0, 0., 0.
    gauge_weights, expected_weights = bias_weights(run)
    encoding = yaml.load((ROOT / "Datasets/encodings/synthetic_dataset_encoding.yaml").read_text(),
                         Loader=yaml.CSafeLoader)
    dataset_names = {index: name for name, index in encoding.items()}
    targets_checked = 0
    for row in rows(path, names):
        tid, did = row["transcript_id"], row["dataset_id"]
        assert tid in run["validation"] and tid not in run["training"]
        k = kinetic[tid]
        if target_hashes is not None and run["depth"] != "mixed":
            assert array_hash(row["target"][:len(k)]) == target_hashes[run["depth"]][dataset_names[did]][tid]
            targets_checked += 1
        length = int(row["length"])
        assert length == len(k) + 1
        assert np.all(row["mask"][:length]) and not np.any(row["mask"][length:])
        prediction = row["L_bio"][:length]
        saved_mean_error = max(saved_mean_error, abs(prediction.mean()-1))
        assert np.isfinite(prediction).all() and np.all(prediction > 0)
        p = prediction[:-1]
        reference_weights[did] = float(row["gamma_centering_reliability"][0])
        nrows += 1
        if tid in records:
            duplicate_max = max(duplicate_max, float(np.max(np.abs(profiles[tid]-p))))
            assert did not in seen_datasets[tid]
        else:
            profiles[tid] = p.copy()
            seen_datasets[tid] = set()
            log_g = sum(w * biases[name][tid] for name, w in gauge_weights.items())
            gauge = k * np.exp(log_g)
            gauge /= gauge.mean()
            pcc, rmse = metrics(p, k)
            pcc_g, rmse_g = metrics(p, gauge)
            pcc_gk, rmse_gk = metrics(gauge, k)
            record = dict(run=run["run"], depth=run["depth"], n_datasets=run["n_datasets"],
                          weighting=run["weighting"], variant=variant, transcript_id=tid,
                          common_three_depths=tid in common, sense_length=len(k),
                          pcc_K_trim10=pcc, rmse_K_trim10=rmse,
                          rmse_K_sense_mean1=metrics(p/p.mean(), k)[1],
                          pcc_K_trim5=metrics(p, k, 5)[0], pcc_K_full=metrics(p, k, 0)[0],
                          pcc_Kg_trim10=pcc_g, rmse_Kg_trim10=rmse_g,
                          pcc_Kg_vs_K_trim10=pcc_gk, rmse_Kg_vs_K_trim10=rmse_gk,
                          pcc_L_unbiased=np.nan, pcc_K_unbiased=np.nan,
                          mu_target_pcc_trim10=0., mu_target_valid_datasets=0,
                          mu_target_rmse_trim10=0.)
            if run["depth"] != "mixed":
                obs = observed[run["depth"]][tid]
                record["pcc_L_unbiased"] = metrics(p, obs)[0]
                record["pcc_K_unbiased"] = metrics(k, obs)[0]
            records[tid] = record
        seen_datasets[tid].add(did)
        mu_pcc, mu_rmse = metrics(row["mu"][:len(k)], row["target"][:len(k)])
        if np.isfinite(mu_pcc):
            records[tid]["mu_target_pcc_trim10"] += mu_pcc
            records[tid]["mu_target_valid_datasets"] += 1
        records[tid]["mu_target_rmse_trim10"] += mu_rmse
    assert set(records) == run["validation"]
    assert duplicate_max < 1e-5
    assert all(len(s) == run["n_datasets"] for s in seen_datasets.values())
    if run["n_datasets"] > 1:
        assert np.allclose(sorted(reference_weights.values()), sorted(expected_weights))
    for record in records.values():
        nvalid = record["mu_target_valid_datasets"]
        record["mu_target_pcc_trim10"] = record["mu_target_pcc_trim10"]/nvalid if nvalid else np.nan
        record["mu_target_rmse_trim10"] /= run["n_datasets"]
    check = dict(run=run["run"], variant=variant, prediction_path=str(path.relative_to(ROOT)),
                 rows=nrows, transcripts=len(records), duplicate_L_max_abs_difference=duplicate_max,
                 prediction_targets_verified_against_input=targets_checked,
                 max_saved_mean_one_error=saved_mean_error,
                 validation_hash=id_hash(records),
                 prediction_L_sha256=id_hash([tid+":"+array_hash(profiles[tid]) for tid in sorted(profiles)]),
                 actual_reference_weights=reference_weights,
                 effective_bias_family_reference_weights=gauge_weights,
                 checkpoint=run["checkpoints"][variant])
    return list(records.values()), check


def summaries(out, common):
    import pandas as pd
    frame = pd.read_csv(out / "recovery_per_transcript.csv")
    excluded = frame.loc[frame.pcc_K_trim10.isna(),
                         ["run", "depth", "variant", "transcript_id", "sense_length"]].copy()
    excluded["reason"] = np.where(excluded.sense_length < 22,
                                   "fewer_than_two_positions_after_trim10", "near_constant_profile")
    excluded.to_csv(out / "recovery_exclusions.csv", index=False)
    # Keep both observed-profile reconstruction endpoints.  The previous
    # prefix list included mu PCC but accidentally omitted mu RMSE because the
    # latter begins with ``mu_target_`` rather than ``rmse_``.
    value_cols = [c for c in frame if c.startswith(("pcc_", "rmse_", "mu_target_"))]
    result = []
    for cohort, selected in [("own_validation", frame),
                              ("common_three_depths", frame.loc[frame.transcript_id.isin(common)])]:
        for keys, group in selected.groupby(["run", "depth", "n_datasets", "weighting", "variant"], sort=False):
            row = dict(zip(["run", "depth", "n_datasets", "weighting", "variant"], keys))
            row.update(cohort=cohort, transcripts=len(group), id_hash=id_hash(group.transcript_id))
            for col in value_cols:
                row[col+"_mean"] = group[col].mean()
            row["pcc_K_trim10_median"] = group.pcc_K_trim10.median()
            finite = group.pcc_K_trim10.notna()
            z = np.arctanh(np.clip(group.loc[finite, "pcc_K_trim10"], -1+1e-12, 1-1e-12))
            weights = np.maximum(group.loc[finite, "sense_length"]-20-3, 1)
            row["pcc_K_trim10_fisher_length_weighted"] = np.tanh(np.average(z, weights=weights))
            row["undefined_pcc_K"] = int((~finite).sum())
            result.append(row)
    write_csv(out / "recovery_summary.csv", result)
    # Pairwise intersections give more held-out transcripts than the three-way
    # intersection. CIs resample transcripts, not training runs (only seed 42).
    pairs = []
    rng = np.random.default_rng(42)
    panels = frame.loc[(frame.variant == "best_pcc") & (frame.depth != "mixed") & (frame.n_datasets >= 2)]
    for n, group in panels.groupby("n_datasets"):
        for i, low in enumerate(DEPTHS):
            for high in DEPTHS[i+1:]:
                a = group.loc[group.depth == low].set_index("transcript_id")
                b = group.loc[group.depth == high].set_index("transcript_id")
                ids = sorted(a.index.intersection(b.index))
                for metric in ("pcc_K_trim10", "rmse_K_trim10", "pcc_Kg_trim10", "rmse_Kg_trim10"):
                    delta = (b.loc[ids, metric]-a.loc[ids, metric]).dropna().to_numpy()
                    boot = delta[rng.integers(0, len(delta), size=(3000, len(delta)))].mean(axis=1)
                    pairs.append(dict(n_datasets=int(n), low_depth=low, high_depth=high,
                                      metric=metric, transcripts=len(delta),
                                      difference_high_minus_low=delta.mean(),
                                      ci025=np.quantile(boot,.025), ci975=np.quantile(boot,.975),
                                      intersection_hash=id_hash(ids)))
    write_csv(out / "paired_depth_differences.csv", pairs)
    decomposition = []
    for n in (2, 10):
        group = panels.loc[panels.n_datasets == n].copy()
        low = group.loc[group.depth == DEPTHS[0], "transcript_id"]
        high = group.loc[group.depth == DEPTHS[-1], "transcript_id"]
        paired_ids = set(low) & set(high)
        selections = [("own_validation", group),
                      ("common_three_depths", group.loc[group.transcript_id.isin(common)]),
                      ("paired_low_high_validation", group.loc[group.transcript_id.isin(paired_ids)
                                                               & group.depth.isin([DEPTHS[0], DEPTHS[-1]])])]
        for cohort, selected in selections:
            for depth, values in selected.groupby("depth"):
                mse_k = values.rmse_K_trim10**2
                mse_h = values.rmse_Kg_trim10**2
                mse_hk = values.rmse_Kg_vs_K_trim10**2
                # Exact vector identity, not an assumed orthogonal decomposition:
                # L-K = (H-K) + (L-H). Cross term = 2 <H-K, L-H>.
                decomposition.append(dict(n_datasets=n, depth=depth, cohort=cohort,
                                          valid_transcripts=int(mse_k.notna().sum()),
                                          mean_mse_L_K=mse_k.mean(), mean_mse_L_H=mse_h.mean(),
                                          mean_mse_H_K=mse_hk.mean(),
                                          mean_cross_term=(mse_k-mse_h-mse_hk).mean()))
    write_csv(out / "error_decomposition.csv", decomposition)


def centering_activation_audit(out):
    """Record the N=1 centering bypass directly from saved output flags."""
    provenance = json.loads((out / "provenance.json").read_text())
    activation = []
    for check in provenance["checks"]:
        if check["variant"] != "best_pcc":
            continue
        nrows = applied_rows = total = applied = 0
        for row in rows(ROOT / check["prediction_path"],
                        ["length", "gamma_centering_applied", "gamma_reference_dataset_count"]):
            flag = row["gamma_centering_applied"][:int(row["length"])]
            nrows += 1
            applied_rows += bool(np.all(flag))
            total += len(flag)
            applied += np.count_nonzero(flag)
        activation.append(dict(run=check["run"], reference_datasets=int(row["gamma_reference_dataset_count"]),
                               rows=nrows, fully_centered_rows=applied_rows,
                               valid_positions=total, centered_positions=applied))
    write_csv(out / "centering_activation.csv", activation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results/riboai_synthetic_experiments")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analyses/artifacts/synthetic/read_depth")
    parser.add_argument("--max-runs", type=int, help="Development smoke test only; not a complete audit")
    parser.add_argument("--skip-count-audit", action="store_true", help="Reuse previously audited observed profiles")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    runs, missing = inventory(args.results_root)
    cohorts = {}
    for depth in DEPTHS:
        matches = [r for r in runs if r["depth"] == depth]
        validation = matches[0]["validation"]
        assert all(r["validation"] == validation for r in matches)
        cohorts[depth] = {"validation": validation, "train": matches[0]["training"]}
    common = set.intersection(*(c["validation"] for c in cohorts.values()))
    print(f"{len(runs)} completed runs; common three-depth validation = {len(common)}", flush=True)
    requested = set.union(*(r["validation"] for r in runs))
    kinetic, biases = reference_profiles(requested)
    if not args.skip_count_audit:
        observed, target_hashes = audit_counts(out, kinetic, cohorts)
    else:
        observed = {}
        target_hashes = None
        for depth in DEPTHS:
            path = ROOT / "Datasets/data/weighted_synthetic" / depth / "artificial_ground_truth.parquet"
            observed[depth] = {}
            for r in rows(path, ["id", "ribo_cds_replicas"]):
                if r["id"] in requested:
                    mean = np.mean(r["ribo_cds_replicas"], axis=0)[:-1]
                    observed[depth][r["id"]] = mean/mean.mean()
    provenance = dict(command=" ".join(sys.argv), missing=missing,
                      common_three_depths=dict(n=len(common), hash=id_hash(common), ids=sorted(common)),
                      cohorts={d: {name: dict(n=len(ids), hash=id_hash(ids), ids=sorted(ids))
                                   for name, ids in c.items()} for d, c in cohorts.items()},
                      definition=dict(checkpoint="best_pcc; best_val_loss sensitivity for N=2,10",
                                      target="deterministic kinetics_target, NOT unbiased sampled counts",
                                      terminal="remove exactly one appended terminal position; no length truncation",
                                      primary="arithmetic mean of transcript PCCs and transcript direct-scale RMSEs; trim10",
                                      normalization="primary uses saved L unchanged; sense-mean-one RMSE also supplied",
                                      gauge_proxy="normalize_mean_one(K * exp(sum_d pi_d log b_d)); NOT a q-based oracle or strict ceiling",
                                      uncertainty="paired transcript bootstrap, 3000 draws, seed42; excludes training-seed uncertainty"),
                      runs=[], checks=[])
    for run in runs:
        provenance["runs"].append(dict(run=run["run"], depth=run["depth"], datasets=run["datasets"],
                                       config_path=str(run["config_path"].relative_to(ROOT)),
                                       config_sha256=hashlib.sha256(run["config_path"].read_bytes()).hexdigest(),
                                       split_path=str(run["split_path"].relative_to(ROOT)),
                                       validation_n=len(run["validation"]), validation_hash=id_hash(run["validation"]),
                                       training_paths=run["training_paths"], seed=42,
                                       weighting=run["weighting"]))
    with (out / "recovery_per_transcript.csv").open("w", newline="") as handle:
        writer = None
        for run in runs[:args.max_runs]:
            variants = ["best_pcc"]
            if run["depth"] != "mixed" and run["n_datasets"] in (2, 10):
                variants.append("best_val_loss")
            for variant in variants:
                records, check = evaluate_run(run, variant, kinetic, biases, observed, common, target_hashes)
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                    writer.writeheader()
                writer.writerows(records)
                handle.flush()
                provenance["checks"].append(check)
                print(f"evaluated {run['run']} {variant}: "
                      f"PCC(K)={np.nanmean([r['pcc_K_trim10'] for r in records]):.6f}", flush=True)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    summaries(out, common)
    centering_activation_audit(out)


if __name__ == "__main__":
    main()
