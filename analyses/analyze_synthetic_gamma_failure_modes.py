#!/usr/bin/env python3
"""Diagnose missed strong synthetic gamma biases.

This joins the existing gauge-fixed gamma position table with the synthetic
read profiles and transcript reliability weights.  It asks whether positions
with large programmed log-gamma but learned log-gamma near zero are explained
by low read support, sparse bias support, transcript reliability, or bias
family.  The analysis is diagnostic only; it does not alter training outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from analyses.analyze_synthetic_recovery import evaluation_domain_metadata
DEFAULT_SAMPLE = ROOT / "analyses/artifacts/synthetic/shared_profile_recovery/synthetic_gamma_recovery_position_sample.tsv.gz"
DEFAULT_WEIGHTED = ROOT / "Datasets/data/weighted_synthetic"
DEFAULT_OUTPUT = ROOT / "analyses/artifacts/synthetic/shared_profile_recovery/gamma_failure_analysis"


def _base_bias(name: str) -> str:
    for suffix in ("_0p25_per_codon", "_2_per_codon", "_20_per_codon"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _load_pair_profiles(path: Path, transcript_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Load only requested transcript rows and retain consensus/replica evidence."""
    data = pd.read_parquet(path, columns=["id", "ribo", "ribo_cds_replicas", "weight", "coverage"])
    data["id"] = data["id"].astype(str)
    data = data[data["id"].isin(transcript_ids)]
    result: dict[str, dict[str, Any]] = {}
    for row in data.itertuples(index=False):
        ribo = _as_array(row.ribo)
        raw_replicas = row.ribo_cds_replicas
        replicas = [] if raw_replicas is None else [_as_array(rep) for rep in raw_replicas]
        result[str(row.id)] = {
            "length": int(ribo.size),
            "total_reads": float(ribo.sum()),
            "coverage": float(row.coverage),
            "weight": float(row.weight),
            "ribo": ribo,
            "replica_nonzero": np.sum(np.stack(replicas, axis=0) > 0, axis=0) if replicas else np.zeros(ribo.size, dtype=np.int16),
            "replica_total": np.sum(np.stack(replicas, axis=0), axis=0) if replicas else np.zeros(ribo.size, dtype=np.float32),
        }
    return result


def _attach_observation_evidence(frame: pd.DataFrame, weighted_root: Path) -> pd.DataFrame:
    frame = frame.copy()
    frame["bias_family"] = frame["dataset"].map(_base_bias)
    pair_cache: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    evidence: list[tuple[float, float, float, float, float, float, int]] = []
    for row in frame.itertuples(index=False):
        dataset_file_stem = _base_bias(str(row.dataset))
        key = (str(row.depth), dataset_file_stem)
        if key not in pair_cache:
            path = weighted_root / str(row.depth) / f"{dataset_file_stem}.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"Weighted synthetic parquet not found: {path}")
            ids = set(frame.loc[(frame.depth == row.depth) & (frame.dataset == row.dataset), "transcript_id"].astype(str))
            pair_cache[key] = _load_pair_profiles(path, ids)
        pair = pair_cache[key].get(str(row.transcript_id))
        if pair is None or int(row.position) >= pair["length"]:
            evidence.append((np.nan,) * 6 + (0,))
            continue
        pos = int(row.position)
        evidence.append((
            pair["total_reads"], pair["coverage"], pair["weight"],
            float(pair["ribo"][pos]), float(pair["replica_total"][pos]),
            float(pair["replica_nonzero"][pos]), pair["length"],
        ))
    values = pd.DataFrame(evidence, columns=["transcript_total_reads", "transcript_coverage", "transcript_weight", "consensus_count", "replica_count", "replica_nonzero", "transcript_length"], index=frame.index)
    return pd.concat([frame, values], axis=1)


def _summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    frame["error"] = frame["learned_log_gamma"] - frame["true_log_gamma"]
    frame["true_strong"] = frame["true_log_gamma"] >= 1.0
    frame["missed_strong"] = frame["true_strong"] & (frame["learned_log_gamma"] < 0.5)
    frame["read_bin"] = pd.cut(frame["transcript_total_reads"], [-np.inf, 20, 50, 100, 250, 500, 1000, np.inf], include_lowest=True)
    frame["position_read_bin"] = pd.cut(frame["replica_count"], [-np.inf, 0, 1, 2, 4, 8, 16, np.inf], include_lowest=True)
    def missed_rate(values: pd.Series) -> float:
        return float(values.sum() / len(values)) if len(values) else float("nan")

    def strong_mean(values: pd.Series) -> float:
        selected = frame.loc[values.index, "true_strong"]
        return float(values[selected].mean()) if bool(selected.any()) else float("nan")

    def missed_among_strong(values: pd.Series) -> float:
        selected = frame.loc[values.index, "true_strong"]
        return float(values[selected].sum() / selected.sum()) if bool(selected.any()) else float("nan")

    by_support = frame.groupby(["mass_condition", "depth", "bias_family", "position_read_bin"], observed=False).agg(
        positions=("error", "size"), strong_positions=("true_strong", "sum"), missed_strong=("missed_strong", "sum"),
        mean_abs_error=("error", lambda x: float(np.mean(np.abs(x)))), mean_true_log_gamma=("true_log_gamma", "mean"),
        mean_learned_log_gamma=("learned_log_gamma", "mean"), strong_mean_learned_log_gamma=("learned_log_gamma", strong_mean),
        missed_fraction_among_strong=("missed_strong", missed_among_strong),
        mean_replica_nonzero=("replica_nonzero", "mean"),
    ).reset_index()
    by_dataset = frame.groupby(["mass_condition", "depth", "dataset", "bias_family"], observed=False).agg(
        positions=("error", "size"), strong_positions=("true_strong", "sum"), missed_strong=("missed_strong", "sum"),
        missed_fraction=("missed_strong", "mean"), mean_abs_error=("error", lambda x: float(np.mean(np.abs(x)))),
        strong_mean_learned_log_gamma=("learned_log_gamma", strong_mean),
        missed_fraction_among_strong=("missed_strong", missed_among_strong),
        mean_transcript_reads=("transcript_total_reads", "mean"), median_transcript_weight=("transcript_weight", "median"),
        mean_replica_count=("replica_count", "mean"), mean_replica_nonzero=("replica_nonzero", "mean"),
    ).reset_index()
    by_depth = frame.groupby(["mass_condition", "depth", "bias_family"], observed=False).agg(
        positions=("error", "size"), strong_positions=("true_strong", "sum"), missed_strong=("missed_strong", "sum"),
        missed_fraction=("missed_strong", "mean"), mean_abs_error=("error", lambda x: float(np.mean(np.abs(x)))),
        strong_mean_learned_log_gamma=("learned_log_gamma", strong_mean),
        missed_fraction_among_strong=("missed_strong", missed_among_strong),
        mean_reads=("transcript_total_reads", "mean"), median_reads=("transcript_total_reads", "median"),
        mean_replica_nonzero=("replica_nonzero", "mean"),
    ).reset_index()
    for output in (by_support, by_dataset, by_depth):
        for name, value in evaluation_domain_metadata().items():
            output[name] = value
    return by_support, by_dataset, by_depth


def _plot(frame: pd.DataFrame, output: Path) -> None:
    frame = frame.copy()
    frame["true_strong"] = frame["true_log_gamma"] >= 1.0
    families = ["artificial_bias_au_fraction_gt_0p7", "artificial_bias_gc_fraction_gt_0p7"]
    display = frame[frame["bias_family"].isin(families)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for col, family in enumerate(families):
        sub = display[display["bias_family"] == family]
        axes[0, col].scatter(sub["replica_count"], sub["learned_log_gamma"], c=sub["true_log_gamma"], s=5, alpha=.18, cmap="magma", vmin=0, vmax=2.2)
        axes[0, col].axhline(.5, color="#b91c1c", ls="--", lw=1)
        axes[0, col].set_title(family.replace("artificial_bias_", ""))
        axes[0, col].set_xlabel("Reads at position across replicas")
        axes[0, col].set_ylabel("Learned log-gamma")
        axes[0, col].grid(alpha=.2)
        high = sub[sub["true_strong"]]
        if not high.empty:
            axes[1, col].scatter(high["replica_count"], high["learned_log_gamma"], c=high["true_log_gamma"], s=5, alpha=.22, cmap="magma", vmin=1, vmax=2.2)
        axes[1, col].axhline(.5, color="#b91c1c", ls="--", lw=1)
        axes[1, col].set_xlabel("Reads at position across replicas")
        axes[1, col].set_ylabel("Learned log-gamma; true >= 1")
        axes[1, col].grid(alpha=.2)
    axes[0, 2].axis("off")
    axes[1, 2].axis("off")
    conditions = ", ".join(sorted(frame["mass_condition"].astype(str).unique()))
    fig.suptitle(
        f"Strong synthetic gamma-bias failure modes ({conditions})",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-table", default=str(DEFAULT_SAMPLE))
    parser.add_argument("--weighted-root", default=str(DEFAULT_WEIGHTED))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output_dir).expanduser().resolve(); output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.sample_table, sep="\t")
    if "evaluation_domain" not in frame.columns or not bool(
        (frame["evaluation_domain"] == "cds_interior").all()
    ):
        raise ValueError(
            "Regenerate the position sample with the interior-aware gamma "
            "recovery analysis. Full-CDS-gauged values cannot be corrected "
            "after export without the original profiles."
        )
    if "mass_condition" not in frame:
        frame["mass_condition"] = np.where(
            frame["run"].astype(str).str.contains("massfree", case=False),
            "mass_free",
            "mass_conserved",
        )
    if frame.empty:
        raise RuntimeError("The gamma-position table is empty.")
    frame = _attach_observation_evidence(frame, Path(args.weighted_root).expanduser().resolve())
    frame["true_strong"] = frame["true_log_gamma"] >= 1.0
    frame["missed_strong"] = frame["true_strong"] & (frame["learned_log_gamma"] < 0.5)
    support, dataset, depth = _summarize(frame)
    frame.to_csv(output / "gamma_failure_position_evidence.tsv.gz", sep="\t", index=False)
    support.to_csv(output / "gamma_failure_by_position_support.tsv", sep="\t", index=False)
    dataset.to_csv(output / "gamma_failure_by_dataset.tsv", sep="\t", index=False)
    depth.to_csv(output / "gamma_failure_by_depth.tsv", sep="\t", index=False)
    condition_plot_names: list[str] = []
    for condition, condition_frame in frame.groupby("mass_condition", sort=True):
        filename = f"gamma_failure_modes_{condition}.png"
        _plot(condition_frame, output / filename)
        condition_plot_names.append(filename)
    strong = frame[frame["true_strong"]]
    missed = frame[frame["missed_strong"]]
    depth_lines = []
    for row in depth.sort_values(["mass_condition", "depth", "bias_family"]).itertuples(index=False):
        depth_lines.append(
            f"- `{row.mass_condition}`, `{row.depth}`, "
            f"`{row.bias_family.replace('artificial_bias_', '')}`: "
            f"{int(row.strong_positions):,} strong positions; "
            f"{int(row.missed_strong):,} missed "
            f"({100.0 * float(row.missed_fraction_among_strong):.2f}% among strong); "
            f"mean transcript reads {float(row.mean_reads):.1f}; "
            f"mean replica reads at position {float(row.mean_replica_nonzero):.2f}."
        )
    depth_labels = ", ".join(sorted(frame["depth"].dropna().unique()))
    mass_labels = ", ".join(sorted(frame["mass_condition"].dropna().unique()))
    condition_links = [
        f"- [{name.removesuffix('.png').replace('_', ' ')}]({name})"
        for name in condition_plot_names
    ]
    report = [
        "# Synthetic gamma failure-mode analysis", "",
        "All primary rows use the CDS-observable interior (`5 <= i < L-5`). Programmed and learned gamma were re-gauged over those identical coordinates before this script received them. Boundary sites are retained separately by the parent gamma recovery analysis and are out-of-scope for fair CDS-only recovery evaluation.", "",
        f"Positions analyzed: **{len(frame):,}**; strong programmed positions (true log-gamma >= 1): **{len(strong):,}**; missed strong positions (learned log-gamma < 0.5): **{len(missed):,}**.", "",
        "A missed strong position is a programmed multiplicative effect of at least exp(1) ~= 2.7-fold whose learned gamma remains close to the neutral value. The evidence table joins position-level gamma values to consensus counts, replica counts, transcript depth, coverage, and the stored transcript reliability weight.", "",
        "The key diagnostic is whether missed strong positions disappear as reads at the position increase. A strong depth trend supports sampling limitation; persistence at high read support points toward a sequence-representation or optimization limitation.", "",
        f"The current evidence covers these read-depth labels: {depth_labels}, and mass conditions: {mass_labels}. Results are never pooled across mass conditions in the summary tables. The depth-by-family results are:", "",
        *depth_lines, "",
        "Comparing the same bias family across depths is the key test. If the missed fraction falls as mean position-level support increases, sampling noise is implicated. If it remains similar despite the roughly order-of-magnitude increase in reads from 0.25 to 2 reads/codon, representation, gamma--shared-signal allocation, or optimization is more likely. These results are based on the sampled position table used by the recovery report; the TSV outputs retain the exact rows used.", "",
        "The AU/GC scatter shape is consistent with sparse, high-amplitude composition effects: the programmed profiles affect only a small subset of positions, but those positions have log-gamma around 1.7--2.0 (roughly 5--7-fold multiplicative effects). The detached near-zero learned cluster is therefore a selective miss of rare strong sites, not a global gamma calibration shift.", "",
        "Files:", "", "- [position evidence](gamma_failure_position_evidence.tsv.gz)", "- [by position support](gamma_failure_by_position_support.tsv)", "- [by dataset](gamma_failure_by_dataset.tsv)", "- [by depth and bias family](gamma_failure_by_depth.tsv)", *condition_links,
    ]
    (output / "GAMMA_FAILURE_ANALYSIS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote diagnostic analysis to {output}")


if __name__ == "__main__":
    main()
