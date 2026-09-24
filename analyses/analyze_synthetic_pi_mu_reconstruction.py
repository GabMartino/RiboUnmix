#!/usr/bin/env python3
"""Test whether synthetic reference policy changes observed-profile reconstruction.

The equal, measured-depth-ranked and deliberately reversed policies use the
same three datasets within each training seed.  This analysis reads their
frozen best-validation-loss exports and compares ``mu`` with the arithmetic
replicate consensus.  It complements, but does not replace, the existing
latent-``L`` recovery analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Utils.publication_plot_style import publication_rc
from analyses.analyze_synthetic_pi_demo import (
    DEFAULT_RESULTS_ROOT,
    DEFAULT_TRUTH_PATH,
    POLICY_COLORS,
    POLICY_ORDER,
    RunArtifact,
    inspect_runs,
    load_truth_subset,
)


PRIMARY_VARIANT = "best_val_loss"
TRIM = 5
RAW_METRICS = ("mu_pcc", "mu_rmse")
RELATIVE_RMSE = "mu_relative_rmse"
METRICS = (*RAW_METRICS, RELATIVE_RMSE)
PLOT_METRICS = ("mu_pcc", RELATIVE_RMSE)
POLICY_LABELS = {"reversed": "Reversed", "equal": "Equal", "quality": "Depth-ranked"}
POLICY_TICK_LABELS = {"reversed": "Reverse", "equal": "Equal", "quality": "Depth-\nranked"}
DEPTH_BY_DATASET_ID = {0: "0.25", 1: "2", 2: "20"}
STEM = "synthetic_pi_mu_reconstruction"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def profile_metrics(mu: np.ndarray, target: np.ndarray, trim: int = TRIM) -> dict:
    """PCC, raw RMSE and scale-free relative RMSE on aligned positions."""
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if mu.shape != target.shape:
        raise ValueError(f"Unaligned mu and target: {mu.shape} != {target.shape}")
    stop = len(mu) - trim
    if trim < 0 or stop - trim < 2:
        return {"positions": max(stop - trim, 0), "mu_pcc": np.nan, "mu_rmse": np.nan,
                RELATIVE_RMSE: np.nan,
                "pcc_reason": "insufficient_positions"}
    x, y = mu[trim:stop], target[trim:stop]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return {"positions": len(x), "mu_pcc": np.nan, "mu_rmse": np.nan,
                RELATIVE_RMSE: np.nan,
                "pcc_reason": "nonfinite"}
    dx, dy = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(dx) * np.linalg.norm(dy)
    pcc = float(np.clip(dx @ dy / denominator, -1.0, 1.0)) if denominator > 1e-12 else np.nan
    reason = "ok" if np.isfinite(pcc) else "near_constant"
    rmse = float(np.sqrt(np.mean((x - y) ** 2)))
    target_rms = float(np.sqrt(np.mean(y**2)))
    relative_rmse = rmse / target_rms if target_rms > 1e-12 else np.nan
    if not np.isfinite(relative_rmse):
        reason = "near_zero_target_scale" if reason == "ok" else reason
    return {
        "positions": len(x),
        "mu_pcc": pcc,
        "mu_rmse": rmse,
        RELATIVE_RMSE: relative_rmse,
        "target_rms": target_rms,
        "pcc_reason": reason,
    }


def evaluate_artifact(
    artifact: RunArtifact,
    truth_lengths: dict[str, int],
) -> tuple[list[dict], dict[tuple[int, str, int], str]]:
    """Stream one prediction export and retain scalar profile diagnostics."""
    parquet = pq.ParquetFile(artifact.prediction_path)
    required = {"transcript_id", "dataset_id", "length", "mask", "mu", "target"}
    missing = required.difference(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"{artifact.prediction_path} lacks {sorted(missing)}")
    rows: list[dict] = []
    target_hashes: dict[tuple[int, str, int], str] = {}
    seen: set[tuple[str, int]] = set()
    for batch in parquet.iter_batches(batch_size=24, columns=sorted(required), use_threads=False):
        for row in batch.to_pylist():
            transcript_id = str(row["transcript_id"])
            dataset_id = int(row["dataset_id"])
            key = (transcript_id, dataset_id)
            if key in seen:
                raise ValueError(f"Duplicate prediction row in {artifact.run_name}: {key}")
            seen.add(key)
            if transcript_id not in truth_lengths:
                raise ValueError(f"Unexpected transcript in {artifact.run_name}: {transcript_id}")
            sense_length = truth_lengths[transcript_id]
            declared_length = int(row["length"])
            mu = np.asarray(row["mu"], dtype=np.float64).reshape(-1)
            target = np.asarray(row["target"], dtype=np.float64).reshape(-1)
            mask = np.asarray(row["mask"], dtype=bool).reshape(-1)
            if declared_length != sense_length + 1:
                raise ValueError(
                    f"Expected one appended terminal entry for {key}: "
                    f"declared={declared_length}, sense={sense_length}"
                )
            if min(len(mu), len(target), len(mask)) < declared_length:
                raise ValueError(f"Truncated prediction arrays for {key}")
            if not mask[:declared_length].all() or mask[declared_length:].any():
                raise ValueError(f"Invalid saved mask for {key}")
            # The terminal export entry is not part of the latent P-site scope.
            mu, target = mu[:sense_length], target[:sense_length]
            metrics = profile_metrics(mu, target)
            rows.append(
                {
                    "run_name": artifact.run_name,
                    "policy": artifact.policy,
                    "seed": artifact.seed,
                    "checkpoint_variant": artifact.variant,
                    "transcript_id": transcript_id,
                    "dataset_id": dataset_id,
                    "nominal_reads_per_codon": DEPTH_BY_DATASET_ID.get(dataset_id, "unknown"),
                    "sense_length": sense_length,
                    **metrics,
                }
            )
            digest = hashlib.sha256(np.asarray(target, dtype="<f8").tobytes()).hexdigest()
            target_hashes[(artifact.seed, transcript_id, dataset_id)] = digest
    close = getattr(parquet, "close", None)
    if callable(close):
        close()
    expected = {
        (transcript_id, dataset_id)
        for transcript_id in artifact.validation_ids
        for dataset_id in range(artifact.expected_dataset_count)
    }
    if seen != expected:
        raise ValueError(
            f"Prediction identity mismatch for {artifact.run_name}: "
            f"missing={len(expected-seen)}, unexpected={len(seen-expected)}"
        )
    return rows, target_hashes


def analyze_predictions(
    artifacts: list[RunArtifact],
    truth_lengths: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    target_owners: dict[tuple[int, str, int], tuple[str, str]] = {}
    integrity = []
    for artifact in artifacts:
        artifact_rows, hashes = evaluate_artifact(artifact, truth_lengths)
        rows.extend(artifact_rows)
        for key, digest in hashes.items():
            previous = target_owners.get(key)
            if previous is not None and previous[1] != digest:
                raise ValueError(
                    f"Observed target differs across policies for seed/transcript/dataset {key}: "
                    f"{previous[0]} versus {artifact.policy}"
                )
            target_owners[key] = (artifact.policy, digest)
        integrity.append(
            {
                "run_name": artifact.run_name,
                "policy": artifact.policy,
                "seed": artifact.seed,
                "prediction_rows": len(artifact_rows),
                "transcripts": len({row["transcript_id"] for row in artifact_rows}),
                "datasets": len({row["dataset_id"] for row in artifact_rows}),
                "undefined_pcc_rows": sum(not np.isfinite(row["mu_pcc"]) for row in artifact_rows),
                "target_hashes_matched_across_policies": True,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(integrity)


def aggregate_transcripts(dataset_metrics: pd.DataFrame, expected_datasets: int = 3) -> pd.DataFrame:
    """Equal-dataset macro averages within each transcript."""
    records = []
    keys = ["run_name", "policy", "seed", "checkpoint_variant", "transcript_id"]
    for identity, group in dataset_metrics.groupby(keys, sort=True):
        if len(group) != expected_datasets or group["dataset_id"].nunique() != expected_datasets:
            raise ValueError(f"Incomplete dataset panel for {identity}")
        pcc = group["mu_pcc"].to_numpy(float)
        rmse = group["mu_rmse"].to_numpy(float)
        relative_rmse = group[RELATIVE_RMSE].to_numpy(float)
        records.append(
            dict(
                zip(keys, identity),
                dataset_count=expected_datasets,
                valid_pcc_datasets=int(np.isfinite(pcc).sum()),
                mu_pcc=float(pcc.mean()) if np.isfinite(pcc).all() else np.nan,
                mu_rmse=float(rmse.mean()) if np.isfinite(rmse).all() else np.nan,
                mu_relative_rmse=(
                    float(relative_rmse.mean()) if np.isfinite(relative_rmse).all() else np.nan
                ),
            )
        )
    return pd.DataFrame(records)


def summarize_runs(
    transcript_metrics: pd.DataFrame,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    records = []
    rng = np.random.default_rng(seed)
    for identity, group in transcript_metrics.groupby(
        ["run_name", "policy", "seed", "checkpoint_variant"], sort=True
    ):
        group = group.sort_values("transcript_id")
        if group[list(METRICS)].isna().any().any():
            raise ValueError(f"Undefined primary metric in {identity}")
        draws = rng.integers(len(group), size=(repeats, len(group)))
        record = dict(zip(["run_name", "policy", "seed", "checkpoint_variant"], identity))
        record["n_transcripts"] = len(group)
        for metric in METRICS:
            values = group[metric].to_numpy(float)
            bootstrap = values[draws].mean(axis=1)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_ci_low"] = float(np.quantile(bootstrap, 0.025))
            record[f"{metric}_ci_high"] = float(np.quantile(bootstrap, 0.975))
        records.append(record)
    return pd.DataFrame(records).sort_values(["seed", "policy"], ignore_index=True)


def paired_policy_effects(
    transcript_metrics: pd.DataFrame,
    repeats: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Within-seed paired policy effects; no transcript-policy rows are independent."""
    comparisons = (("reversed", "equal"), ("equal", "quality"), ("reversed", "quality"))
    records = []
    rng = np.random.default_rng(bootstrap_seed)
    for seed, seed_frame in transcript_metrics.groupby("seed", sort=True):
        policy_ids = {
            policy: sorted(seed_frame.loc[seed_frame["policy"] == policy, "transcript_id"])
            for policy in POLICY_ORDER
        }
        if any(policy_ids[policy] != policy_ids[POLICY_ORDER[0]] for policy in POLICY_ORDER[1:]):
            raise ValueError(f"Policy cohorts differ within seed {seed}")
        shared_ids = policy_ids[POLICY_ORDER[0]]
        # Reuse sampled transcript identities across every policy contrast and
        # both metrics so their dependence is preserved in downstream contrasts.
        draws = rng.integers(len(shared_ids), size=(repeats, len(shared_ids)))
        for left, right in comparisons:
            a = seed_frame[seed_frame["policy"] == left].set_index("transcript_id")
            b = seed_frame[seed_frame["policy"] == right].set_index("transcript_id")
            ids = shared_ids
            for metric in METRICS:
                delta = b.loc[ids, metric].to_numpy(float) - a.loc[ids, metric].to_numpy(float)
                boot = delta[draws].mean(axis=1)
                records.append(
                    {
                        "seed": int(seed),
                        "left_policy": left,
                        "right_policy": right,
                        "metric": metric,
                        "contrast": f"{right} minus {left}",
                        "n_transcripts": len(ids),
                        "estimate": float(delta.mean()),
                        "ci_low": float(np.quantile(boot, 0.025)),
                        "ci_high": float(np.quantile(boot, 0.975)),
                    }
                )
    return pd.DataFrame(records)


def dataset_summary(dataset_metrics: pd.DataFrame) -> pd.DataFrame:
    records = []
    for identity, group in dataset_metrics.groupby(
        ["policy", "seed", "dataset_id", "nominal_reads_per_codon"], sort=True
    ):
        record = dict(zip(["policy", "seed", "dataset_id", "nominal_reads_per_codon"], identity))
        record.update(
            n_transcripts=len(group),
            valid_pcc=int(group["mu_pcc"].notna().sum()),
            mean_mu_pcc=float(group["mu_pcc"].mean()),
            mean_mu_rmse=float(group["mu_rmse"].mean()),
            mean_mu_relative_rmse=float(group[RELATIVE_RMSE].mean()),
        )
        records.append(record)
    return pd.DataFrame(records)


def plot(run_summary: pd.DataFrame, output: Path, width: float) -> dict:
    style = publication_rc()
    style.update(
        {
            "font.size": 12.0,
            "font.weight": "bold",
            "axes.labelsize": 12.0,
            "axes.labelweight": "bold",
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 11.0,
            "axes.titlesize": 13.0,
            "axes.titleweight": "bold",
            "axes.linewidth": 1.1,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
            "grid.linewidth": 0.7,
        }
    )
    if style["text.usetex"]:
        style["text.latex.preamble"] += (
            r"\renewcommand{\seriesdefault}{\bfdefault}"
            r"\AtBeginDocument{\boldmath}"
        )
    policies = list(POLICY_ORDER)
    x = np.arange(len(policies))
    seeds = sorted(run_summary["seed"].unique())
    with matplotlib.rc_context(style):
        figure, axes = plt.subplots(1, 2, figsize=(width, width / 2.5))
        figure.subplots_adjust(left=0.145, right=0.985, bottom=0.20, top=0.82, wspace=0.39)
        for axis, metric, title in zip(
            axes,
            PLOT_METRICS,
            ("A  Observed-profile correlation", "B  Scale-adjusted error"),
        ):
            mean_column = f"{metric}_mean"
            for seed_index, training_seed in enumerate(seeds):
                cell = run_summary[run_summary["seed"] == training_seed].set_index("policy").loc[policies]
                axis.plot(
                    x,
                    cell[mean_column],
                    color="#A0A0A0",
                    linewidth=2.0,
                    alpha=0.75,
                    zorder=1,
                )
                offset = (seed_index - (len(seeds) - 1) / 2) * 0.045
                for policy_index, policy in enumerate(policies):
                    row = cell.loc[policy]
                    axis.errorbar(
                        policy_index + offset,
                        row[mean_column],
                        yerr=[[row[mean_column] - row[f"{metric}_ci_low"]],
                              [row[f"{metric}_ci_high"] - row[mean_column]]],
                        color=POLICY_COLORS[policy],
                        marker=("o", "s", "^")[seed_index],
                        markersize=6.2,
                        markeredgecolor="white",
                        markeredgewidth=0.8,
                        capsize=2.5,
                        elinewidth=1.2,
                        linewidth=0,
                        zorder=3,
                    )
            axis.set_title(title, loc="left", pad=7)
            axis.set_xticks(x, [POLICY_TICK_LABELS[policy] for policy in policies])
            axis.set_xlim(-0.35, len(policies) - 0.65)
            axis.grid(axis="y")
            axis.set_axisbelow(True)
        axes[0].set_ylabel(r"Mean PCC$(\mu_{dt},\overline{Y}_{dt})$")
        axes[1].set_ylabel("Relative RMSE")
        handles = [
            plt.Line2D([], [], color="#666666", marker=marker, linewidth=0,
                       markersize=6.0, label=f"Seed {training_seed}")
            for marker, training_seed in zip(("o", "s", "^"), seeds)
        ]
        figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0.995),
                      ncol=len(handles), handletextpad=0.25, columnspacing=1.3)
        figure.savefig(output / f"{STEM}.pdf", bbox_inches="tight")
        figure.savefig(output / f"{STEM}.svg", bbox_inches="tight")
        figure.savefig(output / f"{STEM}.png", dpi=600, bbox_inches="tight")
        plt.close(figure)
    return {
        "text.usetex": style["text.usetex"],
        "font.family": style["font.family"],
        "font.size": style["font.size"],
        "axes.titlesize": style["axes.titlesize"],
        "figure_width_inches": width,
    }


def write_caption(output: Path, repeats: int, seed: int) -> None:
    (output / "caption.tex").write_text(
        rf"""\textbf{{Reference policy and observed-profile reconstruction in the synthetic depth panel.}}
The same three biased datasets at 0.25, 2, and 20 reads per codon are fitted under
reversed $(1/2,1/3,1/6)$, equal $(1/3,1/3,1/3)$, and measured-depth-ranked
$(1/6,1/3,1/2)$ fixed-reference weights, listed in increasing-depth order. Points show
training seeds 42--44 and gray lines connect matched seeds. The best-validation-loss
checkpoint is used. PCC and relative RMSE compare $\boldsymbol\mu_{{dt}}$ with the arithmetic
replicate consensus $\overline{{\mathbf Y}}_{{dt}}$ after removing the appended terminal
entry and excluding five codons at each end. Relative RMSE divides each profile's raw
RMSE by $\sqrt{{|\mathcal I_t|^{{-1}}\sum_i\overline Y_{{dti}}^2}}$, preventing the
20-read dataset from dominating only because of its count units. Raw RMSE remains in the
source tables. Metrics are computed within each
transcript--dataset profile, averaged equally across the three datasets within transcript,
and then equally across transcripts. Error bars are pointwise 95\% percentile intervals
from {repeats:,} transcript-bootstrap draws (seed {seed}), conditional on each fitted model.
The same observed targets and validation transcripts are used by all three policies within
a training seed. These validation-reconstruction metrics assess fit to sampled observations,
not latent kinetic recovery or biological accuracy. The weights $\pi_d$ constrain the
$L/\gamma$ reference decomposition and are not observation weights in the loss.
"""
    )


def write_report(output: Path, summary: pd.DataFrame, effects: pd.DataFrame) -> None:
    lines = [
        "# Reference policy and observed-profile reconstruction",
        "",
        "The fitted means are evaluated against exactly matched observed consensus profiles within each seed.",
        "These results should be read together with the existing latent-L recovery analysis.",
        "",
        "## Run means",
        "",
        "| Seed | Policy | Mean mu PCC | Mean relative RMSE | Raw mean RMSE |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {int(row.seed)} | {POLICY_LABELS[row.policy]} | "
            f"{row.mu_pcc_mean:.4f} | {row.mu_relative_rmse_mean:.4f} | "
            f"{row.mu_rmse_mean:.4f} |"
        )
    lines.extend(["", "## Matched policy effects", ""])
    for comparison in (("reversed", "equal"), ("equal", "quality"), ("reversed", "quality")):
        left, right = comparison
        selected = effects[(effects["left_policy"] == left) & (effects["right_policy"] == right)]
        lines.append(f"### {POLICY_LABELS[left]} to {POLICY_LABELS[right]}")
        lines.append("")
        for row in selected.itertuples(index=False):
            direction = "right minus left"
            lines.append(
                f"- Seed {row.seed}, {row.metric}: {row.estimate:+.5f} "
                f"[{row.ci_low:+.5f}, {row.ci_high:+.5f}] ({direction})."
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation contract",
            "",
            "- Similar mu reconstruction with different latent-L recovery supports a change in decomposition/gauge, not a claim that all learned components are identical.",
            "- A reconstruction difference can reflect optimization under the constraint, but pi still does not directly weight likelihood rows.",
            "- The plotted relative RMSE divides by each target profile's RMS amplitude, so the deeply sequenced dataset cannot dominate merely through its count scale.",
            "- Raw RMSE remains in `run_summary.csv`, `paired_policy_effects.csv`, and `dataset_summary.csv` for audit, but it is not used for cross-depth visual interpretation.",
            "- The same validation data enter checkpoint selection and evaluation, so this is not an untouched test estimate.",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--truth-path", type=Path, default=DEFAULT_TRUTH_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=5_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--figure-width", type=float, default=6.6)
    args = parser.parse_args()
    if args.bootstrap_repeats < 100:
        parser.error("Use at least 100 bootstrap draws.")

    results_root = args.results_root.resolve()
    output = (
        args.output_dir
        or ROOT / "analyses" / "artifacts" / "synthetic" / "pi_demo" / "mu_reconstruction"
    ).resolve()
    availability, artifacts, run_provenance = inspect_runs(results_root, [PRIMARY_VARIANT])
    artifacts = [artifact for artifact in artifacts if artifact.variant == PRIMARY_VARIANT]
    expected = {(policy, seed) for policy in POLICY_ORDER for seed in (42, 43, 44)}
    actual = {(artifact.policy, artifact.seed) for artifact in artifacts}
    if actual != expected:
        raise ValueError(f"Incomplete policy-by-seed grid: missing={sorted(expected-actual)}")
    for seed in (42, 43, 44):
        cohorts = {artifact.validation_hash for artifact in artifacts if artifact.seed == seed}
        if len(cohorts) != 1:
            raise ValueError(f"Validation cohort differs across policies at seed {seed}")

    requested_ids = set().union(*(set(artifact.validation_ids) for artifact in artifacts))
    truth = load_truth_subset(args.truth_path.resolve(), requested_ids)
    truth_lengths = {transcript_id: len(profile) for transcript_id, profile in truth.items()}
    dataset_metrics, integrity = analyze_predictions(artifacts, truth_lengths)
    transcript_metrics = aggregate_transcripts(dataset_metrics)
    summary = summarize_runs(transcript_metrics, args.bootstrap_repeats, args.bootstrap_seed)
    effects = paired_policy_effects(
        transcript_metrics,
        args.bootstrap_repeats,
        args.bootstrap_seed,
    )
    by_dataset = dataset_summary(dataset_metrics)

    output.mkdir(parents=True, exist_ok=True)
    dataset_metrics.to_parquet(output / "per_transcript_dataset.parquet", index=False)
    transcript_metrics.to_csv(output / "per_transcript.csv", index=False)
    summary.to_csv(output / "run_summary.csv", index=False)
    effects.to_csv(output / "paired_policy_effects.csv", index=False)
    by_dataset.to_csv(output / "dataset_summary.csv", index=False)
    integrity.to_csv(output / "prediction_integrity.csv", index=False)
    availability.to_csv(output / "run_availability.csv", index=False)
    typography = plot(summary, output, args.figure_width)
    write_caption(output, args.bootstrap_repeats, args.bootstrap_seed)
    write_report(output, summary, effects)

    command = " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv])
    provenance = {
        "command": command,
        "results_root": str(results_root),
        "truth_path": str(args.truth_path.resolve()),
        "truth_sha256": sha256_file(args.truth_path.resolve()),
        "checkpoint_variant": PRIMARY_VARIANT,
        "boundary_trim_codons": TRIM,
        "terminal_entry_removed": True,
        "aggregation": "within transcript-dataset, equal datasets within transcript, equal transcripts",
        "relative_rmse": {
            "definition": "RMSE(mu,Ybar) / sqrt(mean_i(Ybar_i^2)) within each transcript-dataset interior",
            "purpose": "dimensionless comparison across count scales; not an NB2 calibration statistic",
            "raw_rmse_retained": True,
        },
        "bootstrap": {
            "repeats": args.bootstrap_repeats,
            "seed": args.bootstrap_seed,
            "unit": "transcript",
            "interval": "pointwise percentile 95%",
        },
        "target_identity_checked_across_policies_within_seed": True,
        "run_provenance": run_provenance,
        "typography": typography,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        summary[
            ["seed", "policy", "n_transcripts", "mu_pcc_mean", "mu_relative_rmse_mean", "mu_rmse_mean"]
        ].to_string(index=False)
    )
    print(output / f"{STEM}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
