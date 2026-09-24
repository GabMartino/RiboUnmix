#!/usr/bin/env python3
"""Render the independent synthetic-depth audit; never recompute or fit a model."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys
from string import Template

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Utils.publication_plot_style import publication_rc

DEPTHS = ["0p25_per_codon", "2_per_codon", "20_per_codon"]
LABELS = dict(zip(DEPTHS, ["0.25", "2", "20"]))
COLORS = dict(zip(DEPTHS, ["#D55E00", "#0072B2", "#009E73"]))


def table(frame, digits=4):
    display = frame.rename(columns=COLUMN_LABELS).copy()
    if "depth" in frame:
        display["Depth"] = frame.depth.astype(str).replace(LABELS).to_numpy()
    if "metric" in frame:
        display["Metric"] = frame.metric.replace(METRIC_LABELS).to_numpy()
    markup = display.to_html(index=False, border=0, na_rep="undefined",
                             float_format=lambda x: f"{x:.{digits}f}")
    return '<div class="table-scroll">'+markup+"</div>"


def figures(out, frame):
    own = frame.loc[(frame.cohort == "own_validation") & (frame.variant == "best_pcc")
                    & frame.depth.isin(DEPTHS) & (frame.n_datasets >= 2)]
    common = frame.loc[(frame.cohort == "common_three_depths") & (frame.variant == "best_pcc")
                       & frame.depth.isin(DEPTHS) & (frame.n_datasets == 2)]
    with matplotlib.rc_context(publication_rc()):
        fig, axes = plt.subplots(1, 3, figsize=(13.3, 4.2), layout="constrained")
        for depth in DEPTHS:
            group = own.loc[own.depth == depth].sort_values("n_datasets")
            axes[0].plot(group.n_datasets, group.pcc_K_trim10_mean, "o-",
                         color=COLORS[depth], label=LABELS[depth]+" reads/codon", markersize=4)
            axes[1].plot(group.n_datasets, group.rmse_Kg_trim10_mean, "o-",
                         color=COLORS[depth], markersize=4)
        axes[0].set(xlabel="Number of bias datasets", ylabel=r"Mean PCC$(L_t,K_t)$",
                    title="Kinetic-target recovery")
        axes[1].set(xlabel="Number of bias datasets", ylabel=r"Mean RMSE$(L_t,H_t)$",
                    title="Error relative to bias-reference proxy")
        axes[0].legend(loc="lower right")
        for ax in axes[:2]:
            ax.set_xticks([2, 4, 6, 8, 10])
        group = common.set_index("depth").loc[DEPTHS]
        axes[2].plot([.25, 2, 20], group.pcc_K_trim10_mean, "o-", color="#0072B2",
                     label=r"Learned $L_t$ versus $K_t$")
        axes[2].plot([.25, 2, 20], group.pcc_K_unbiased_mean, "s--", color="#777777",
                     label=r"Unbiased counts versus $K_t$")
        axes[2].set(xscale="log", xlabel="Expected reads per codon", ylabel="Mean PCC",
                    title=f"Matched validation, $N=2$ ($n={int(group.transcripts.iloc[0])}$)")
        axes[2].set_xticks([.25, 2, 20], ["0.25", "2", "20"])
        axes[2].legend(loc="lower right")
        for letter, ax in zip("ABC", axes):
            ax.text(-.14, 1.06, letter, transform=ax.transAxes, fontweight="bold")
            ax.grid(axis="y")
        fig.savefig(out / "read_depth_diagnostic.pdf", bbox_inches="tight")
        fig.savefig(out / "read_depth_diagnostic.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


COLUMN_LABELS = {
    "depth": "Depth", "n_datasets": r"\(N\)", "transcripts": "Transcripts",
    "valid_transcripts": "Scored transcripts", "weighting": "Reference weights",
    "pcc_K_trim10_mean": r"\(\bar r(L,K)\)",
    "rmse_K_trim10_mean": r"\(\bar E(L,K)\)",
    "pcc_Kg_trim10_mean": r"\(\bar r(L,H)\)",
    "rmse_Kg_trim10_mean": r"\(\bar E(L,H)\)",
    "pcc_Kg_vs_K_trim10_mean": r"\(\bar r(H,K)\)",
    "ratio": r"\(R_N\)", "ratio_ci_low": "95% lower", "ratio_ci_high": "95% upper",
    "mean_low": r"\(\bar E_{\rm LH}^{(0.25,N)}(H)\)",
    "mean_high": r"\(\bar E_{\rm LH}^{(20,N)}(H)\)",
    "n_transcripts": "Matched transcripts", "metric": "Metric",
    "difference_high_minus_low": r"\(\Delta\): high minus low",
    "ci025": "95% lower", "ci975": "95% upper",
    "mean_mse_L_K": r"\(\overline{D(L,K)}\)",
    "mean_mse_L_H": r"\(\overline{D(L,H)}\)",
    "mean_mse_H_K": r"\(\overline{D(H,K)}\)",
    "mean_cross_term": r"\(\bar\chi\)",
    "training_positions": "Training positions across datasets",
    "training_reads_both_replicates": "Training reads across both replicas",
    "reads_per_codon_per_replicate": "Actual reads/codon/replica",
    "replicate_zero_fraction": "Zero fraction: replica",
    "consensus_zero_fraction": "Zero fraction: replica mean",
    "variant": "Checkpoint", "pcc_K_full_mean": r"\(\bar r(L,K)\), full",
    "pcc_K_trim5_mean": r"\(\bar r(L,K)\), trim5",
    "pcc_K_trim10_fisher_length_weighted": "PCC: length-weighted Fisher, trim10",
    "rmse_K_sense_mean1_mean": "K RMSE: sense-CDS renormalized",
    "train_n": r"\(|\mathcal T_C|\)", "validation_n": r"\(|\mathcal V_C|\)",
    "scored_n": "Valid trim10 profiles", "bias": "Single bias",
    "0p25_per_codon": r"\(C=0.25\)", "2_per_codon": r"\(C=2\)", "20_per_codon": r"\(C=20\)",
}
METRIC_LABELS = {
    "pcc_K_trim10": r"\(\bar r(L,K)\)", "rmse_K_trim10": r"\(\bar E(L,K)\)",
    "pcc_Kg_trim10": r"\(\bar r(L,H)\)", "rmse_Kg_trim10": r"\(\bar E(L,H)\)",
}


def render_report(out, *, regenerate_plots=False):
    """Render definitions and saved numerical results; never alter the analysis."""
    provenance = json.loads((out / "provenance.json").read_text())
    frame = pd.read_csv(out / "recovery_summary.csv")
    frame["valid_transcripts"] = frame.transcripts-frame.undefined_pcc_K
    frame["depth"] = pd.Categorical(frame.depth, categories=[*DEPTHS, "mixed"], ordered=True)
    counts = pd.read_csv(out / "input_count_audit.csv")
    paired = pd.read_csv(out / "paired_depth_differences.csv")
    decomposition = pd.read_csv(out / "error_decomposition.csv")
    activation = pd.read_csv(out / "centering_activation.csv")
    figure_dir = out / "depth_effect_figure"
    if not (figure_dir / "provenance.json").is_file():
        raise FileNotFoundError(
            "This report includes the matched-depth figure. First run "
            "analyses/plot_synthetic_read_depth_effect.py with this audit directory."
        )
    figure = json.loads((figure_dir / "provenance.json").read_text())
    ratios = pd.read_csv(figure_dir / "paired_error_ratios.csv")
    if figure["excluded"]["three_depth"] or figure["excluded"]["low_high"]:
        raise ValueError("Matched exclusions changed; update the report's cohort explanation.")
    if regenerate_plots:
        figures(out, frame)

    primary = frame.loc[(frame.variant == "best_pcc") & frame.depth.isin(DEPTHS)]
    own = primary.loc[primary.cohort == "own_validation"]
    selected_columns = ["n_datasets", "depth", "valid_transcripts", "pcc_K_trim10_mean",
                        "rmse_K_trim10_mean", "pcc_Kg_trim10_mean", "rmse_Kg_trim10_mean",
                        "pcc_Kg_vs_K_trim10_mean"]
    endpoints = own.loc[own.n_datasets.isin([2, 10]), selected_columns]
    matched = primary.loc[primary.n_datasets.isin([2, 10])
                          & (primary.cohort == "common_three_depths"), selected_columns]
    count_control = counts.loc[counts.dataset == "artificial_ground_truth",
                               ["depth", "transcripts", "reads_per_codon_per_replicate",
                                "replicate_zero_fraction", "consensus_zero_fraction"]]
    contrast = paired.loc[paired.n_datasets.isin([2, 10]) & (paired.low_depth == DEPTHS[0])
                          & (paired.high_depth == DEPTHS[-1]),
                          ["n_datasets", "metric", "transcripts", "difference_high_minus_low",
                           "ci025", "ci975"]]
    mixed = frame.loc[(frame.depth == "mixed") & (frame.variant == "best_pcc")
                      & (frame.cohort == "own_validation"),
                      ["n_datasets", "weighting", "transcripts",
                       "pcc_K_trim10_mean", "rmse_K_trim10_mean"]]
    sensitivity = frame.loc[frame.n_datasets.isin([2, 10]) & frame.depth.isin(DEPTHS)
                            & (frame.cohort == "own_validation"),
                            ["n_datasets", "depth", "variant", "pcc_K_full_mean",
                             "pcc_K_trim5_mean", "pcc_K_trim10_mean",
                             "pcc_K_trim10_fisher_length_weighted", "rmse_K_trim10_mean",
                             "rmse_K_sense_mean1_mean"]]
    n2counts = counts.loc[counts.dataset.isin(["artificial_bias_3prime_aa", "artificial_bias_3prime_cc"])]
    train = n2counts.groupby("depth", as_index=False)[
        ["training_positions", "training_reads_both_replicates"]].sum()
    single = own.loc[own.n_datasets == 1].copy()
    single["bias"] = single.run.str.extract("single_artificial_bias_(.*?)_gammaequal")[0]
    single_table = single.pivot(index="bias", columns="depth",
                                values="pcc_K_trim10_mean").reindex(columns=DEPTHS).reset_index()
    cumulative = own.loc[own.n_datasets >= 2].pivot(
        index="n_datasets", columns="depth",
        values="pcc_K_trim10_mean").reindex(columns=DEPTHS).reset_index()
    cohort_rows = [
        dict(depth=d, train_n=provenance["cohorts"][d]["train"]["n"],
             validation_n=provenance["cohorts"][d]["validation"]["n"],
             scored_n=int(own.loc[own.depth == d, "valid_transcripts"].min()))
        for d in DEPTHS
    ]
    checks = provenance["checks"]
    paired_decomposition = decomposition.loc[
        decomposition.cohort == "paired_low_high_validation"].drop(columns="cohort")
    ratio_columns = ["n_datasets", "n_transcripts", "mean_low", "mean_high",
                     "ratio", "ratio_ci_low", "ratio_ci_high"]
    r2, r10 = (ratios.set_index("n_datasets").loc[n] for n in (2, 10))
    missing = "".join(
        f"<li><code>{html.escape(r['run'])}</code>: {html.escape(r['reason'])}; "
        "the completed retry is included.</li>" for r in provenance["missing"]
    )
    values = {
        "run_count": len(provenance["runs"]),
        "common_n": figure["cohorts"]["three_depth"]["n"],
        "paired_n": figure["cohorts"]["low_high"]["n"],
        "common_hash": figure["cohorts"]["three_depth"]["hash"],
        "paired_hash": figure["cohorts"]["low_high"]["hash"],
        "figure_bootstrap": f"{figure['bootstrap']['repeats']:,}",
        "bootstrap_seed": figure["bootstrap"]["seed"],
        "ratio_min": f"{ratios.ratio.min():.2f}", "ratio_max": f"{ratios.ratio.max():.2f}",
        "ratio2": f"{r2.ratio:.2f}", "ratio2_low": f"{r2.ratio_ci_low:.2f}",
        "ratio2_high": f"{r2.ratio_ci_high:.2f}",
        "ratio10": f"{r10.ratio:.2f}", "ratio10_low": f"{r10.ratio_ci_low:.2f}",
        "ratio10_high": f"{r10.ratio_ci_high:.2f}",
        "target_checks": f"{sum(c['prediction_targets_verified_against_input'] for c in checks):,}",
        "replica_mismatches": int(counts.raw_replica_mismatches.sum()),
        "duplicate_error": f"{max(c['duplicate_L_max_abs_difference'] for c in checks):.3g}",
        "unique_hashes": len({c["prediction_L_sha256"] for c in checks if c["variant"] == "best_pcc"}),
        "uncentered_single_count": int(((activation.reference_datasets == 1)
                                       & (activation.centered_positions == 0)).sum()),
        "missing_runs": missing,
        "cohort_table": table(pd.DataFrame(cohort_rows), digits=0),
        "ratio_table": table(ratios[ratio_columns]),
        "matched_table": table(matched.sort_values(["n_datasets", "depth"])),
        "endpoint_table": table(endpoints.sort_values(["n_datasets", "depth"])),
        "cumulative_table": table(cumulative),
        "contrast_table": table(contrast, 5),
        "decomposition_table": table(paired_decomposition, 6),
        "training_table": table(train, 0),
        "mixed_table": table(mixed.sort_values(["n_datasets", "weighting"])),
        "count_table": table(count_control),
        "single_table": table(single_table),
        "sensitivity_table": table(sensitivity.sort_values(["n_datasets", "depth", "variant"])),
    }
    template = Path(__file__).with_name("templates") / "synthetic_read_depth_audit.html"
    return Template(template.read_text()).substitute(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path,
                        default=ROOT / "analyses/artifacts/synthetic/read_depth")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Update only the HTML; leave all tables and images unchanged.")
    args = parser.parse_args()
    out = args.audit_dir.resolve()
    report = render_report(out, regenerate_plots=not args.skip_plots)
    path = out / "read_depth_audit.html"
    path.write_text(report)
    print(path)


if __name__ == "__main__":
    main()
