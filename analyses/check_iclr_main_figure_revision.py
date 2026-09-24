#!/usr/bin/env python3
"""Verify the active main-text figures against their frozen scalar sources."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REVISION = ROOT / "analyses/artifacts/manuscript_main_figures_20260924"
GAMMA_REVISION = ROOT / "analyses/artifacts/manuscript_main_figures_20260923"
PREVIOUS_REVISION = ROOT / "analyses/artifacts/manuscript_main_figures_20260922"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    gamma_dir = GAMMA_REVISION / "gamma_pcc_rmse"
    manifest = json.loads((gamma_dir / "manifest.json").read_text())
    assert manifest["run"]["status"] == "completed"
    assert all(x["sha256"] == x["sha256_after"] for x in manifest["input_artifacts"])
    # Shared-profile estimates and uncertainty are reused, not recomputed.
    source = pd.read_csv(ROOT / "analyses/artifacts/synthetic/manuscript_revision/reference_target_best_val_loss/aggregate_summary.csv")
    source = source.loc[source.family.eq("within_depth") & source.reference_weighting.eq("equal")
                        & source.metric.isin(["pearson", "clr_rmse"])
                        & source.comparison.eq("analysis3_L_vs_Q")]
    plotted = pd.read_csv(REVISION / "synthetic/synthetic_occupancy_recovery_compact_source.csv")
    plotted_q = plotted.loc[plotted.source_quantity.eq("L_vs_qbar"), source.columns]
    pd.testing.assert_frame_equal(
        source.reset_index(drop=True), plotted_q.reset_index(drop=True), check_dtype=False
    )
    kinetic_path = REVISION / "kinetic_alignment/L_vs_K_summary.csv"
    kinetic = pd.read_csv(kinetic_path)
    plotted_k = plotted.loc[plotted.source_quantity.eq("L_vs_K"), kinetic.columns]
    pd.testing.assert_frame_equal(
        kinetic.reset_index(drop=True), plotted_k.reset_index(drop=True), check_dtype=False
    )
    assert len(plotted) == 81
    assert plotted_q.groupby("metric").size().to_dict() == {"clr_rmse": 27, "pearson": 27}
    assert len(plotted_k) == 27 and plotted_k.n_undefined.eq(0).all()
    kinetic_transcripts = pd.read_parquet(REVISION / "kinetic_alignment/L_vs_K_per_transcript.parquet")
    assert kinetic_transcripts.pcc_defined.all()
    for (depth, n_datasets), rows in kinetic_transcripts.groupby(["depth", "n_datasets"]):
        reported = kinetic.loc[
            kinetic.depth.eq(depth) & kinetic.n_datasets.eq(n_datasets)
        ].iloc[0]
        assert len(rows) == reported.n_transcripts
        np.testing.assert_allclose(np.median(rows.pcc_L_vs_K), reported["median"], rtol=0, atol=1e-15)
    kinetic_provenance = json.loads(
        (REVISION / "kinetic_alignment/provenance.json").read_text()
    )
    assert kinetic_provenance["method"]["checkpoint_variant"] == "best validation loss"
    assert kinetic_provenance["method"]["boundary_trim_codons_each_end"] == 10
    assert kinetic_provenance["undefined_pcc"] == 0
    gamma = pd.read_csv(gamma_dir / "gamma_summary.csv")
    assert set(gamma.run_id) == set(plotted.run_id)
    assert gamma.checkpoint_variant.eq("best_val_loss").all()
    assert gamma.boundary_trim_codons.eq(10).all()
    pd.testing.assert_frame_equal(
        gamma,
        pd.read_csv(REVISION / "synthetic/synthetic_gamma_recovery_source.csv"),
        check_exact=True,
    )
    transcripts = pd.read_csv(gamma_dir / "gamma_per_transcript.csv.gz")
    pairs = pd.read_parquet(
        gamma_dir / "gamma_per_transcript_dataset.parquet",
        columns=["run_id", "transcript_id", "log_gamma_rmse"],
    )
    replay_rmse = (
        pairs.assign(squared_error=lambda x: x.log_gamma_rmse**2)
        .groupby(["run_id", "transcript_id"], as_index=False).squared_error.mean()
    )
    replay_rmse["replayed_joint_log_gamma_rmse"] = np.sqrt(replay_rmse.squared_error)
    replay_rmse = replay_rmse.drop(columns="squared_error")
    replay = transcripts.merge(
        replay_rmse, on=["run_id", "transcript_id"], validate="one_to_one"
    )
    np.testing.assert_allclose(
        replay.joint_log_gamma_rmse,
        replay.replayed_joint_log_gamma_rmse,
        rtol=0,
        atol=1e-15,
    )
    excluded = []
    for depth, rows in transcripts.groupby("depth"):
        pcc = rows.pivot(index="transcript_id", columns="n_datasets", values="mean_dataset_pcc")
        rmse = rows.pivot(index="transcript_id", columns="n_datasets", values="joint_log_gamma_rmse").reindex_like(pcc)
        slope = rows.pivot(index="transcript_id", columns="n_datasets", values="joint_calibration_slope").reindex_like(pcc)
        complete = pcc.notna().all(axis=1)
        assert rmse.loc[complete].notna().all().all()
        assert slope.loc[complete].notna().all().all()
        curve = gamma.loc[gamma.depth.eq(depth)].sort_values("n_datasets")
        np.testing.assert_allclose(pcc.loc[complete].median().to_numpy(), curve["pcc_median"], rtol=0, atol=1e-15)
        np.testing.assert_allclose(rmse.loc[complete].median().to_numpy(), curve["log_rmse_median"], rtol=0, atol=1e-15)
        np.testing.assert_allclose(slope.loc[complete].median().to_numpy(), curve["calibration_slope_median"], rtol=0, atol=1e-15)
        np.testing.assert_allclose(curve["median"], curve["pcc_median"], rtol=0, atol=0)
        np.testing.assert_allclose(curve["bootstrap_ci_low"], curve["pcc_bootstrap_ci_low"], rtol=0, atol=0)
        np.testing.assert_allclose(curve["bootstrap_ci_high"], curve["pcc_bootstrap_ci_high"], rtol=0, atol=0)
        assert curve.n_transcripts.eq(int(complete.sum())).all()
        for tid in pcc.index[~complete]:
            excluded.append({"depth": depth, "transcript_id": tid, "reason": "constant true correction at N=2"})
    pd.DataFrame(excluded).to_csv(REVISION / "gamma_cohort_exclusions.csv", index=False)
    previous_gamma = pd.read_csv(PREVIOUS_REVISION / "gamma_verified/gamma_summary.csv")
    previous_gamma = previous_gamma.sort_values(["depth", "n_datasets"]).reset_index(drop=True)
    current_gamma = gamma.sort_values(["depth", "n_datasets"]).reset_index(drop=True)
    np.testing.assert_allclose(previous_gamma["median"], current_gamma["pcc_median"], rtol=0, atol=0)
    np.testing.assert_allclose(previous_gamma["bootstrap_ci_low"], current_gamma["pcc_bootstrap_ci_low"], rtol=0, atol=0)
    np.testing.assert_allclose(previous_gamma["bootstrap_ci_high"], current_gamma["pcc_bootstrap_ci_high"], rtol=0, atol=0)
    original_panels = pd.read_csv(ROOT / "figures/assets_5_real_datasets_4_panels/main_text_four_panel_equal_source.csv")
    original_panels = original_panels.loc[original_panels.arm.eq("equal"), ["pair_label", "transcript_id", "PCC"]]
    panels = pd.read_csv(PREVIOUS_REVISION / "real/real_data_panel_reproducibility_per_transcript_source.csv")
    pd.testing.assert_frame_equal(original_panels.reset_index(drop=True), panels)
    original_anchor = pd.read_csv(ROOT / "analyses/artifacts/real_data/cumulative_selection_quality_score/own_policy_anchor_summary.csv")
    original_anchor = original_anchor.loc[original_anchor.selection_direction.eq("best_first")]
    anchor = pd.read_csv(PREVIOUS_REVISION / "real/real_data_best_first_own_anchor_source.csv")
    pd.testing.assert_frame_equal(original_anchor.reset_index(drop=True), anchor)
    assert len(anchor) == 28 and anchor.anchor_policy.eq(anchor.reference_policy).all()
    np.testing.assert_allclose(anchor.loc[anchor.N.eq(2), "mean"], 1, rtol=0, atol=1e-12)
    abstract = ROOT / "ICLR_draft/sections/abstract.tex"
    assert sha(abstract) == sha(PREVIOUS_REVISION / "before/ICLR_draft/sections/abstract.tex")
    log = (ROOT / "ICLR_draft/main.log").read_text()
    assert not re.search(r"Overfull|undefined references|multiply defined|Citation .* undefined|Reference .* undefined", log)
    aux = (ROOT / "ICLR_draft/main.aux").read_text()
    conclusion_page = int(re.search(r"newlabel\{sec:conclusion\}\{\{[^}]+\}\{(\d+)\}", aux)[1])
    assert conclusion_page <= 9
    result = {
        "passed": True, "compact_figure_source_rows": len(plotted),
        "shared_profile_source_rows": len(plotted_q),
        "kinetic_alignment_source_rows": len(plotted_k),
        "panel_a_rows_unchanged": len(panels), "own_policy_anchor_rows": len(anchor),
        "gamma_runs": len(gamma), "gamma_run_transcripts": len(transcripts),
        "fixed_gamma_cohort_exclusions": len(excluded),
        "gamma_pcc_unchanged_from_previous_audit": True,
        "gamma_rmse_replayed_from_dataset_scalars": True,
        "abstract_unchanged": True,
        "conclusion_page": conclusion_page,
        "checks": ["L-qbar PCC and RMSE scalar estimates preserved",
                   "L-K PCC uses the exact best-loss cohorts and ten-codon masks",
                   "own-policy N=2 anchors",
                   "constant targets excluded on a fixed within-depth cohort",
                   "gamma PCC exactly matches prior audit", "joint gamma RMSE replayed from dataset scalars",
                   "no unresolved references or overfull boxes"],
        "figure_hashes": {str(path.relative_to(ROOT)): sha(path) for path in [
            ROOT / "ICLR_draft/main.pdf",
            ROOT / "ICLR_draft/figures/main/04_synthetic/synthetic_occupancy_recovery_compact.pdf",
            ROOT / "ICLR_draft/figures/main/05_real_data/real_data_stability.pdf"]},
    }
    (REVISION / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
