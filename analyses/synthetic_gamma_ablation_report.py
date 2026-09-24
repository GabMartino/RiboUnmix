"""Combine synthetic recovery and biological-quality outputs into one scorecard.

The historical output directory is named ``gamma_ablation_analysis``. The
report itself only calls the selected results an ablation when more than one
gamma-centering configuration is actually present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from gamma_ablation.common import DEFAULT_OUTPUT_ROOT
except ModuleNotFoundError:  # pragma: no cover - depends on invocation form
    from analyses.gamma_ablation.common import DEFAULT_OUTPUT_ROOT


CONDITION_KEYS = [
    "run_id",
    "strategy",
    "training_scope",
    "depth",
    "mass_condition",
    "n_datasets",
    "quality_rank_power",
    "gamma_weighting",
    "feature_preset",
    "seed",
]

DEPTH_ORDER = {
    "0p25_per_codon": 0,
    "2_per_codon": 1,
    "20_per_codon": 2,
    "cross_depth": 3,
}


def _sort_by_depth(frame: pd.DataFrame, remaining: list[str]) -> pd.DataFrame:
    output = frame.copy()
    output["__depth_order"] = output["depth"].map(DEPTH_ORDER).fillna(99)
    return output.sort_values(
        ["__depth_order", *remaining],
        na_position="last",
    ).drop(
        columns="__depth_order"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "report")
    return parser.parse_args()


def recovery_scorecard(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=CONDITION_KEYS)
    if frame.empty or "split" not in frame.columns:
        return pd.DataFrame(columns=CONDITION_KEYS)
    frame = frame[frame["split"] == "main_val"].copy()
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict("records"):
        scope = str(record["comparison_scope"])
        component = str(record["component"])
        if scope == "all_included":
            name = f"{component}_vs_observed_pcc_dataset_macro"
        elif scope == "shared_latent_truth_interior" and component == "L_bio":
            name = "L_bio_vs_latent_truth_pcc_interior"
        elif scope in {"shared_consensus_equal", "shared_consensus_equal_interior"} and component == "L_bio":
            name = "L_bio_vs_equal_consensus_pcc"
        elif scope in {"shared_consensus_dataset_quality", "shared_consensus_dataset_quality_interior"} and component == "L_bio":
            name = "L_bio_vs_quality_consensus_pcc"
        else:
            continue
        rows.append(
            {
                **{key: record.get(key) for key in CONDITION_KEYS},
                "measure": name,
                "value": record["pearson_dataset_macro_mean"],
                "reference_kind": record.get("reference_kind", ""),
            }
        )
    tidy = pd.DataFrame(rows)
    if tidy.empty:
        return pd.DataFrame(columns=CONDITION_KEYS)
    values = tidy.pivot_table(index=CONDITION_KEYS, columns="measure", values="value", aggfunc="first").reset_index()
    kinds = tidy.groupby(CONDITION_KEYS, dropna=False)["reference_kind"].first().reset_index()
    return values.merge(kinds, on=CONDITION_KEYS, how="left")


def biology_scorecard(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=CONDITION_KEYS)
    if frame.empty or "signal" not in frame.columns:
        return pd.DataFrame(columns=CONDITION_KEYS)
    frame = frame[frame["signal"] == "L_bio"].copy()
    rows = []
    for record in frame.to_dict("records"):
        metric = str(record["metric"])
        if metric == "css_recall":
            if float(record["z_threshold"]) != 3.0 or int(record["margin_codons"]) not in (0, 1):
                continue
            name = f"css_recall_z3_margin{int(record['margin_codons'])}"
        elif metric in {"motif_P", "motif_PP", "motif_PPP", "start_ramp", "stop_ramp"}:
            name = f"{metric}_mean_log2_enrichment"
        else:
            continue
        rows.append(
            {
                **{key: record.get(key) for key in CONDITION_KEYS},
                "measure": name,
                "value": record["value"],
            }
        )
    tidy = pd.DataFrame(rows)
    if tidy.empty:
        return pd.DataFrame(columns=CONDITION_KEYS)
    return tidy.pivot_table(index=CONDITION_KEYS, columns="measure", values="value", aggfunc="first").reset_index()


def best_lines(scorecard: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = []
    for column in columns:
        if column not in scorecard or not np.isfinite(scorecard[column]).any():
            continue
        row = scorecard.loc[scorecard[column].idxmax()]
        lines.append(
            f"- Highest `{column}`: {row[column]:.4f} at "
            f"depth={row['depth']}, mass={row['mass_condition']}, "
            f"strategy={row['strategy']}, N={int(row['n_datasets'])}, "
            f"p={row['quality_rank_power']:g}, preset={row['feature_preset']}."
        )
    return lines


def power_overview(scorecard: pd.DataFrame) -> list[str]:
    columns = {
        "L_bio_vs_latent_truth_pcc_interior": "L_bio latent PCC (interior)",
        "L_bio_vs_equal_consensus_pcc": "L_bio consensus PCC",
        "mu_vs_observed_pcc_dataset_macro": "mu observed PCC",
        "css_recall_z3_margin0": "CSS exact",
        "css_recall_z3_margin1": "CSS +/-1",
    }
    available = [column for column in columns if column in scorecard]
    if scorecard.empty or not available:
        return ["No comparable recovery/biological metrics were available for the selected runs."]
    # Keep dataset count explicit. Averaging over N would make depths with
    # incomplete downloads (for example N=2..6) look directly comparable to
    # depths with N=2..10 even though their panel ranges differ.
    table = scorecard.groupby(
        [
            "depth",
            "mass_condition",
            "strategy",
            "quality_rank_power",
            "n_datasets",
        ],
        as_index=False,
    )[available].mean()
    table = _sort_by_depth(
        table,
        ["mass_condition", "strategy", "quality_rank_power", "n_datasets"],
    )
    lines = [
        "| Read depth | N datasets | Mass mode | Strategy | p | "
        + " | ".join(columns[column] for column in available)
        + " |",
        "|---|---:|---|---|---:|" + "---:|" * len(available),
    ]
    for record in table.to_dict("records"):
        values = " | ".join(f"{record[column]:.4f}" for column in available)
        lines.append(
            f"| {record['depth']} | {int(record['n_datasets'])} | "
            f"{str(record['mass_condition']).replace('_', ' ')} | "
            f"{str(record['strategy']).replace('_', ' ')} | "
            f"{record['quality_rank_power']:g} | {values} |"
        )
    return lines


def structure_findings(scorecard: pd.DataFrame, biological: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if scorecard.empty:
        return ["No condition rows were available for structural diagnostics."]
    # A run may export only main-validation predictions.  In that case CSS or
    # motif columns are legitimately absent; report the available views rather
    # than failing while indexing an optional metric.
    for column in (
        "L_bio_vs_equal_consensus_pcc",
        "css_recall_z3_margin1",
        "css_recall_z3_margin0",
        "start_ramp_mean_log2_enrichment",
        "stop_ramp_mean_log2_enrichment",
    ):
        if column not in scorecard:
            scorecard[column] = np.nan
    collapsed = scorecard[
        (scorecard["L_bio_vs_equal_consensus_pcc"] < 0.05)
        & (scorecard["css_recall_z3_margin1"] < 0.01)
    ]
    if not collapsed.empty:
        labels = [
            f"{row.depth}/{row.mass_condition}/{row.strategy}/"
            f"N={int(row.n_datasets)}/p={row.quality_rank_power:g}"
            for row in collapsed.itertuples(index=False)
        ]
        lines.append(
            "- Near-flat/shared-branch collapse flag (consensus PCC < 0.05 and CSS +/-1 recall < 0.01): "
            + ", ".join(labels)
            + "."
        )

    meaningful = float(np.log2(1.05))
    for (depth, mass_condition, strategy), group in _sort_by_depth(
        scorecard,
        ["mass_condition", "strategy", "n_datasets"],
    ).groupby(
        ["depth", "mass_condition", "strategy"], dropna=False, sort=False
    ):
        label = (
            f"{depth}, {str(mass_condition).replace('_', ' ')}, "
            f"{strategy.replace('_', ' ')}"
        )
        counts = []
        for motif in ("P", "PP", "PPP"):
            column = f"motif_{motif}_mean_log2_enrichment"
            if column not in group:
                counts.append(f"{motif} unavailable")
            else:
                counts.append(f"{motif} {int((group[column] > meaningful).sum())}/{len(group)}")
        lines.append(
            f"- {label}: conditions with >5% motif enrichment were " + ", ".join(counts) + "."
        )
        lines.append(
            f"- {label}: positive 5' ramp in {int((group['start_ramp_mean_log2_enrichment'] > 0).sum())}/{len(group)} conditions; "
            f"positive stop-proximal ramp in {int((group['stop_ramp_mean_log2_enrichment'] > 0).sum())}/{len(group)}."
        )

    if biological.empty or "metric" not in biological.columns:
        return lines
    ramp_rows = biological[
        biological["metric"].isin(["start_ramp", "stop_ramp"])
    ]
    if not ramp_rows.empty:
        means = (
            ramp_rows.groupby(
                ["depth", "mass_condition", "strategy", "signal", "metric"],
                as_index=False,
            )["value"]
            .mean()
            .pivot(
                index=["depth", "mass_condition", "strategy", "signal"],
                columns="metric",
                values="value",
            )
            .reset_index()
        )
        lines.extend(
            [
                "",
                "Mean terminal log2 enrichments across conditions (the selected dataset-specific anchor for target/mu/gamma; de-duplicated shared profiles for L_bio):",
                "",
                "| Read depth | Mass mode | Strategy | Signal | 5' | Stop |",
                "|---|---|---|---|---:|---:|",
            ]
        )
        means = _sort_by_depth(means, ["mass_condition", "strategy", "signal"])
        for row in means.itertuples(index=False):
            lines.append(
                f"| {row.depth} | {str(row.mass_condition).replace('_', ' ')} | "
                f"{str(row.strategy).replace('_', ' ')} | {row.signal} | "
                f"{row.start_ramp:.4f} | {row.stop_ramp:.4f} |"
            )
    return lines


def write_report(
    scorecard: pd.DataFrame,
    biological: pd.DataFrame,
    path: Path,
    recovery_settings: dict[str, Any] | None = None,
) -> None:
    latent = bool(
        "L_bio_vs_latent_truth_pcc_interior" in scorecard
        and np.isfinite(scorecard["L_bio_vs_latent_truth_pcc_interior"]).any()
    )
    gamma_settings = (
        scorecard[["gamma_weighting", "quality_rank_power"]]
        .drop_duplicates()
        .sort_values(["gamma_weighting", "quality_rank_power"])
    )
    is_gamma_ablation = len(gamma_settings) > 1
    lines = [
        (
            "# Gamma-centering ablation report"
            if is_gamma_ablation
            else "# Synthetic recovery and biological-quality report"
        ),
        "",
        f"Conditions in scorecard: {len(scorecard)}.",
        "",
        "## Interpretation guardrails",
        "",
    ]
    if not is_gamma_ablation and not gamma_settings.empty:
        setting = gamma_settings.iloc[0]
        lines.append(
            "Despite the historical `gamma_ablation_analysis` directory name, "
            "this result is **not a gamma-centering ablation**: it contains only "
            f"`weighting={setting['gamma_weighting']}` with "
            f"`quality_rank_power={setting['quality_rank_power']:g}`. The directory "
            "name is retained only for compatibility with the numbered analysis scripts."
        )
    if latent:
        trim = (recovery_settings or {}).get("boundary_trim_codons", 5)
        lines.append(
            "The primary shared-profile result compares `L_bio` directly with the "
            "deterministic synthetic latent `K` (`rib_profile`), once per transcript "
            f"on the interior after excluding {trim} codons at each CDS end."
        )
        lines.append(
            "Observed-target consensus and individual-target values are secondary "
            "diagnostics: they retain sampling noise and programmed dataset-specific bias."
        )
    else:
        lines.append(
            "The current bundles do not contain latent `L_bio_true` or `mu_true`; recovery columns therefore measure agreement with held-out observed `target` profiles."
        )
    dataset_filter = (recovery_settings or {}).get("per_dataset_filter")
    if dataset_filter:
        lines.append(
            "Per-dataset recovery and dataset-specific biological signals in this generated result use the common anchor: "
            + ", ".join(map(str, dataset_filter))
            + ". Shared L_bio consensus uses every dataset included in each run."
        )
    lines.extend(
        [
            "CSS recall is site sensitivity at a fixed within-transcript z threshold; it is not precision or PR-AUC.",
            "P/PP/PPP values mark all residues participating in each motif and compare their mean profile with the remaining valid codons.",
            "Positive start/stop values mean higher signal in codons 0-49 than in codons 100-199 from the corresponding terminus.",
            "A terminal pattern is more consistent with a technical effect when it is stronger or more dataset-variable in `target`/`mu`/`gamma` than in shared `L_bio`; the terminal meta-profile outputs support that comparison.",
            "Read-depth conditions are never pooled: every table row and every figure is partitioned by read depth and mass mode.",
            "",
            "## Configuration overview by read depth and exact dataset count",
            "",
        ]
    )
    lines.extend(power_overview(scorecard))
    lines.extend(
        [
            "",
            "## Structural diagnostics",
            "",
        ]
    )
    lines.extend(structure_findings(scorecard, biological))
    lines.extend(
        [
            "",
            "## Metric-wise maxima (not a composite ranking)",
            "",
        ]
    )
    lines.extend(
        best_lines(
            scorecard,
            [
                "L_bio_vs_latent_truth_pcc_interior",
                "L_bio_vs_equal_consensus_pcc",
                "mu_vs_observed_pcc_dataset_macro",
                "css_recall_z3_margin0",
                "css_recall_z3_margin1",
                "motif_P_mean_log2_enrichment",
                "motif_PP_mean_log2_enrichment",
                "motif_PPP_mean_log2_enrichment",
            ],
        )
    )
    lines.extend(
        [
            "",
            "No single composite score is produced: prediction agreement, CSS sensitivity, motif enrichment, and terminal ramps answer different questions and should be inspected together.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    recovery_path = args.analysis_root / "recovery" / "recovery_by_condition.csv"
    biology_path = args.analysis_root / "biological_quality" / "biological_quality_by_condition.csv"
    missing = [str(path) for path in (recovery_path, biology_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run scripts 13 and 14 first. Missing: " + ", ".join(missing)
        )
    recovery = recovery_scorecard(recovery_path)
    biology = biology_scorecard(biology_path)
    scorecard = recovery.merge(biology, on=CONDITION_KEYS, how="outer")
    scorecard = _sort_by_depth(
        scorecard,
        [
            "mass_condition",
            "strategy",
            "feature_preset",
            "seed",
            "n_datasets",
            "quality_rank_power",
        ],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scorecard.to_csv(args.output_dir / "condition_scorecard.csv", index=False)
    settings_path = args.analysis_root / "recovery" / "analysis_settings.json"
    recovery_settings = (
        json.loads(settings_path.read_text(encoding="utf-8"))
        if settings_path.exists()
        else None
    )
    write_report(
        scorecard,
        pd.read_csv(biology_path),
        args.output_dir / "REPORT.md",
        recovery_settings,
    )
    print(f"Saved combined report to {args.output_dir}")


if __name__ == "__main__":
    main()
