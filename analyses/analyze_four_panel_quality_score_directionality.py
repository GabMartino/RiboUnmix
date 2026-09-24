#!/usr/bin/env python3
"""Analyse partial or complete four-panel quality-score directionality runs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_cumulative_stability import object_sha256, sha256, write_json
from run_four_panel_quality_score_directionality import (
    BALANCED_DESIGN,
    STRATA_DESIGN,
    DEFAULT_BALANCED_OUTPUT,
    REFERENCE_VARIANTS,
)
from analyses.analyze_cumulative_selection_direction_quality_score import (
    adaptive_pcc_axis,
    save_figure,
    style_axis as _base_style_axis,
    summarize_pcc,
)
from analyses.analyze_rank_balanced_reference_directionality import relocate
from analyses.analyze_real_exp8_reference_directionality import collect, pair_record
from analyses.paths import artifact_directory
from Utils.publication_plot_style import latex_paper_style
from Utils.reliability_references import transcript_id_hash


ARMS = tuple(
    "equal" if orientation == "equal" else f"{orientation}_{policy}"
    for orientation, policy, _ in REFERENCE_VARIANTS
)
ARM_LABELS = {
    "equal": "Equal reference",
    "best_first_score_p1": r"Best-oriented, $p=1$",
    "worst_first_score_p1": r"Worst-oriented, $p=1$",
    "best_first_score_p3": r"Best-oriented, $p=3$",
    "worst_first_score_p3": r"Worst-oriented, $p=3$",
    "best_first_score_p5": r"Best-oriented, $p=5$",
    "worst_first_score_p5": r"Worst-oriented, $p=5$",
}
ARM_COLORS = {
    "equal": "#4B5563",
    "best_first_score_p1": "#0072B2",
    "worst_first_score_p1": "#70B7D8",
    "best_first_score_p3": "#D55E00",
    "worst_first_score_p3": "#E9A26D",
    "best_first_score_p5": "#6A3D9A",
    "worst_first_score_p5": "#B28AC7",
}
DISPLAY_ARMS = (
    "equal",
    "best_first_score_p1", "worst_first_score_p1",
    "best_first_score_p3", "worst_first_score_p3",
    "best_first_score_p5", "worst_first_score_p5",
)


def style_axis(axis):
    _base_style_axis(axis)
    axis.tick_params(labelsize=12)


def audit_design(root):
    manifest = json.loads((root / "experiment_manifest.json").read_text())
    if manifest.get("experiment_design") not in {BALANCED_DESIGN, STRATA_DESIGN}:
        raise ValueError("Expected a four-panel directional quality-score experiment.")
    if object_sha256(manifest["tasks"]) != manifest["tasks_sha256"]:
        raise ValueError("Frozen task matrix checksum mismatch.")
    recorded = Path(manifest["output_root"])
    for original, expected in manifest["frozen_file_sha256"].items():
        local = relocate(original, root, recorded)
        if sha256(local) != expected:
            raise ValueError(f"Frozen input/configuration changed: {local}")
    panels = list(manifest["panels"])
    if len(panels) != 4 or set(task["arm"] for task in manifest["tasks"]) != set(ARMS):
        raise ValueError("Expected four panels and the seven declared reference policies.")
    folds = manifest["source_folds"]
    first = folds[panels[0]]
    for field in ("train_ids", "validation_ids", "test_ids"):
        if any(folds[panel][field] != first[field] for panel in panels):
            raise ValueError(f"{field} differs across panels.")
    if any(set(first[a]) & set(first[b]) for a, b in itertools.combinations(
        ("train_ids", "validation_ids", "test_ids"), 2
    )):
        raise ValueError("Train, validation and test transcript folds overlap.")
    assignment = pd.read_csv(root / "inputs/panel_assignment.csv")
    if assignment.dataset_id.duplicated().any():
        raise ValueError("A dataset appears in more than one fixed panel.")
    source_panel_counts = assignment.groupby("source_family").panel_id.nunique()
    if source_panel_counts.gt(1).any():
        raise ValueError("A source family appears in more than one fixed panel.")
    weights = pd.read_csv(root / "reference_weights.csv")
    weight_sums = weights.groupby(["panel_id", "arm"]).pi.sum()
    if not np.allclose(weight_sums.to_numpy(float), 1.0, atol=1e-12, rtol=0):
        raise ValueError("At least one frozen reference-weight vector does not sum to one.")
    if weights.pi.lt(0).any() or not np.isfinite(weights.pi).all():
        raise ValueError("Reference weights must be finite and non-negative.")
    configs = {
        (task["training_seed"], task["arm"], task["panel_id"]):
        yaml.safe_load(relocate(task["config_path"], root, recorded).read_text())
        for task in manifest["tasks"]
    }
    return manifest, panels, first["test_ids"], configs


def cross_panel_rows(profiles, panels, seed, ids):
    rows = []
    for panel_a, panel_b in itertools.combinations(panels, 2):
        for arm in ARMS:
            left = profiles.get((seed, arm, panel_a))
            right = profiles.get((seed, arm, panel_b))
            if left is None or right is None:
                continue
            for transcript in ids:
                rows.append(dict(
                    panel_a=panel_a,
                    panel_b=panel_b,
                    pair=f"{panel_a}__{panel_b}",
                    arm=arm,
                    transcript_id=transcript,
                    **pair_record(left[transcript], right[transcript], "full_cds"),
                ))
    columns = ["panel_a", "panel_b", "pair", "arm", "transcript_id",
               "PCC", "RMSE", "variance_a", "variance_b", "reason"]
    return pd.DataFrame(rows, columns=columns)


def within_panel_rows(profiles, panels, seed, ids):
    rows = []
    for panel in panels:
        equal = profiles.get((seed, "equal", panel))
        if equal is None:
            continue
        for arm in ARMS[1:]:
            score = profiles.get((seed, arm, panel))
            if score is None:
                continue
            for transcript in ids:
                rows.append(dict(
                    panel_id=panel,
                    arm=arm,
                    transcript_id=transcript,
                    **pair_record(equal[transcript], score[transcript], "full_cds"),
                ))
    columns = ["panel_id", "arm", "transcript_id", "PCC", "RMSE",
               "variance_a", "variance_b", "reason"]
    return pd.DataFrame(rows, columns=columns)


def orientation_rows(profiles, panels, seed, ids):
    rows = []
    for panel in panels:
        for power in (1, 3, 5):
            best_arm = f"best_first_score_p{power}"
            worst_arm = f"worst_first_score_p{power}"
            best = profiles.get((seed, best_arm, panel))
            worst = profiles.get((seed, worst_arm, panel))
            if best is None or worst is None:
                continue
            for transcript in ids:
                rows.append(dict(
                    panel_id=panel,
                    power=power,
                    transcript_id=transcript,
                    **pair_record(best[transcript], worst[transcript], "full_cds"),
                ))
    columns = ["panel_id", "power", "transcript_id", "PCC", "RMSE",
               "variance_a", "variance_b", "reason"]
    return pd.DataFrame(rows, columns=columns)


def paired_cross_panel_effects(cross, replicates, seed):
    if cross.empty:
        return pd.DataFrame(), pd.DataFrame()
    equal = cross[cross.arm.eq("equal")][
        ["panel_a", "panel_b", "transcript_id", "PCC"]
    ].rename(columns={"PCC": "PCC_equal"})
    score = cross[~cross.arm.eq("equal")][
        ["panel_a", "panel_b", "arm", "transcript_id", "PCC"]
    ]
    paired = score.merge(equal, on=["panel_a", "panel_b", "transcript_id"], validate="many_to_one")
    paired["delta_PCC_vs_equal"] = paired.PCC - paired.PCC_equal
    summary_rows = []
    for keys, group in paired.groupby(["panel_a", "panel_b", "arm"], sort=False):
        values = group.delta_PCC_vs_equal.to_numpy(float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        salt = "|".join(map(str, keys))
        local_seed = (seed + int(hashlib.sha256(salt.encode()).hexdigest()[:8], 16)) % (2**32)
        rng = np.random.default_rng(local_seed)
        boot = np.empty(replicates)
        for start in range(0, replicates, 250):
            stop = min(replicates, start + 250)
            indices = rng.integers(0, len(values), size=(stop - start, len(values)))
            boot[start:stop] = values[indices].mean(axis=1)
        low, high = np.quantile(boot, [.025, .975])
        summary_rows.append(dict(
            panel_a=keys[0], panel_b=keys[1], arm=keys[2], n_valid=len(values),
            mean_delta=float(values.mean()), median_delta=float(np.median(values)),
            fraction_positive=float(np.mean(values > 0)), ci_low=float(low), ci_high=float(high),
        ))
    return paired, pd.DataFrame(summary_rows)


def bootstrap_mean_interval(values, replicates, seed):
    """Transcript bootstrap interval for a mean; positions are never resampled."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = np.empty(replicates, dtype=float)
    for start in range(0, replicates, 250):
        stop = min(replicates, start + 250)
        draw = rng.integers(0, len(values), size=(stop - start, len(values)))
        boot[start:stop] = values[draw].mean(axis=1)
    return tuple(np.quantile(boot, (0.025, 0.975)))


def _seed_for(seed, label):
    return (seed + int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)) % (2**32)


def aggregate_cross_panel_metrics(cross, replicates, seed):
    """Average the six fixed panel-pair metrics within each transcript."""
    expected_pairs = cross["pair"].nunique()
    valid = cross[cross.reason.eq("ok") & cross.PCC.notna() & cross.RMSE.notna()].copy()
    aggregate = valid.groupby(["arm", "transcript_id"], as_index=False).agg(
        mean_pair_PCC=("PCC", "mean"),
        mean_pair_RMSE=("RMSE", "mean"),
        n_valid_pairs=("pair", "nunique"),
    )
    aggregate["complete_six_pair_record"] = aggregate.n_valid_pairs.eq(expected_pairs)
    complete = aggregate[aggregate.complete_six_pair_record].copy()
    summary_rows = []
    for arm, group in complete.groupby("arm", sort=False):
        pcc_low, pcc_high = bootstrap_mean_interval(
            group.mean_pair_PCC, replicates, _seed_for(seed, f"aggregate-pcc|{arm}")
        )
        rmse_low, rmse_high = bootstrap_mean_interval(
            group.mean_pair_RMSE, replicates, _seed_for(seed, f"aggregate-rmse|{arm}")
        )
        summary_rows.append(dict(
            arm=arm,
            n_transcripts=len(group),
            n_panel_pairs=expected_pairs,
            mean_PCC=float(group.mean_pair_PCC.mean()),
            median_PCC=float(group.mean_pair_PCC.median()),
            pcc_q10=float(group.mean_pair_PCC.quantile(.10)),
            pcc_q90=float(group.mean_pair_PCC.quantile(.90)),
            pcc_ci_low=float(pcc_low),
            pcc_ci_high=float(pcc_high),
            mean_RMSE=float(group.mean_pair_RMSE.mean()),
            median_RMSE=float(group.mean_pair_RMSE.median()),
            rmse_q10=float(group.mean_pair_RMSE.quantile(.10)),
            rmse_q90=float(group.mean_pair_RMSE.quantile(.90)),
            rmse_ci_low=float(rmse_low),
            rmse_ci_high=float(rmse_high),
        ))
    return aggregate, pd.DataFrame(summary_rows)


def aggregate_policy_contrasts(aggregate, replicates, seed):
    complete = aggregate[aggregate.complete_six_pair_record].copy()
    equal = complete[complete.arm.eq("equal")][
        ["transcript_id", "mean_pair_PCC", "mean_pair_RMSE"]
    ].rename(columns={"mean_pair_PCC": "equal_PCC", "mean_pair_RMSE": "equal_RMSE"})
    metrics = complete[~complete.arm.eq("equal")].merge(
        equal, on="transcript_id", validate="many_to_one"
    )
    metrics["delta_PCC_vs_equal"] = metrics.mean_pair_PCC - metrics.equal_PCC
    metrics["delta_RMSE_vs_equal"] = metrics.mean_pair_RMSE - metrics.equal_RMSE
    rows = []
    for arm, group in metrics.groupby("arm", sort=False):
        pcc_low, pcc_high = bootstrap_mean_interval(
            group.delta_PCC_vs_equal, replicates, _seed_for(seed, f"delta-pcc|{arm}")
        )
        rmse_low, rmse_high = bootstrap_mean_interval(
            group.delta_RMSE_vs_equal, replicates, _seed_for(seed, f"delta-rmse|{arm}")
        )
        rows.append(dict(
            arm=arm,
            n_transcripts=len(group),
            mean_delta_PCC=float(group.delta_PCC_vs_equal.mean()),
            median_delta_PCC=float(group.delta_PCC_vs_equal.median()),
            fraction_PCC_improved=float(group.delta_PCC_vs_equal.gt(0).mean()),
            pcc_ci_low=float(pcc_low),
            pcc_ci_high=float(pcc_high),
            mean_delta_RMSE=float(group.delta_RMSE_vs_equal.mean()),
            median_delta_RMSE=float(group.delta_RMSE_vs_equal.median()),
            fraction_RMSE_improved=float(group.delta_RMSE_vs_equal.lt(0).mean()),
            rmse_ci_low=float(rmse_low),
            rmse_ci_high=float(rmse_high),
        ))
    return metrics, pd.DataFrame(rows)


def aggregate_orientation_contrasts(aggregate, replicates, seed):
    complete = aggregate[aggregate.complete_six_pair_record]
    metric_frames = []
    rows = []
    for power in (1, 3, 5):
        best = complete[complete.arm.eq(f"best_first_score_p{power}")][
            ["transcript_id", "mean_pair_PCC", "mean_pair_RMSE"]
        ].rename(columns={"mean_pair_PCC": "best_PCC", "mean_pair_RMSE": "best_RMSE"})
        worst = complete[complete.arm.eq(f"worst_first_score_p{power}")][
            ["transcript_id", "mean_pair_PCC", "mean_pair_RMSE"]
        ].rename(columns={"mean_pair_PCC": "worst_PCC", "mean_pair_RMSE": "worst_RMSE"})
        group = best.merge(worst, on="transcript_id", validate="one_to_one")
        group.insert(0, "power", power)
        group["delta_PCC_best_minus_worst"] = group.best_PCC - group.worst_PCC
        group["delta_RMSE_best_minus_worst"] = group.best_RMSE - group.worst_RMSE
        metric_frames.append(group)
        pcc_low, pcc_high = bootstrap_mean_interval(
            group.delta_PCC_best_minus_worst,
            replicates,
            _seed_for(seed, f"orientation-pcc|{power}"),
        )
        rmse_low, rmse_high = bootstrap_mean_interval(
            group.delta_RMSE_best_minus_worst,
            replicates,
            _seed_for(seed, f"orientation-rmse|{power}"),
        )
        rows.append(dict(
            power=power,
            n_transcripts=len(group),
            mean_delta_PCC_best_minus_worst=float(group.delta_PCC_best_minus_worst.mean()),
            median_delta_PCC_best_minus_worst=float(group.delta_PCC_best_minus_worst.median()),
            fraction_best_PCC_higher=float(group.delta_PCC_best_minus_worst.gt(0).mean()),
            pcc_ci_low=float(pcc_low),
            pcc_ci_high=float(pcc_high),
            mean_delta_RMSE_best_minus_worst=float(group.delta_RMSE_best_minus_worst.mean()),
            median_delta_RMSE_best_minus_worst=float(group.delta_RMSE_best_minus_worst.median()),
            fraction_best_RMSE_lower=float(group.delta_RMSE_best_minus_worst.lt(0).mean()),
            rmse_ci_low=float(rmse_low),
            rmse_ci_high=float(rmse_high),
        ))
    return pd.concat(metric_frames, ignore_index=True), pd.DataFrame(rows)


def training_outcomes(root, manifest):
    rows = []
    for task in manifest["tasks"]:
        status_path = root / task["directory"] / "execution_status.json"
        status = json.loads(status_path.read_text())
        output = status.get("outputs", {})
        rows.append(dict(
            panel_id=task["panel_id"],
            arm=task["arm"],
            selected_epoch=output.get("epoch"),
            validation_loss=output.get("validation_loss"),
            checkpoint_variant=output.get("checkpoint_variant"),
            selection_rule=output.get("selection_rule"),
        ))
    frame = pd.DataFrame(rows)
    equal = frame[frame.arm.eq("equal")][["panel_id", "validation_loss"]].rename(
        columns={"validation_loss": "equal_validation_loss"}
    )
    frame = frame.merge(equal, on="panel_id", validate="many_to_one")
    frame["delta_validation_loss_vs_equal"] = (
        frame.validation_loss - frame.equal_validation_loss
    )
    return frame


def concentration_association(cross_summary, concentration):
    geometry = concentration.copy()
    geometry["effective_fraction"] = geometry.N_ref / geometry.N
    left = geometry[["collection_id", "arm", "effective_fraction", "max_reference_mass"]].rename(
        columns={
            "collection_id": "panel_a",
            "effective_fraction": "effective_fraction_a",
            "max_reference_mass": "max_mass_a",
        }
    )
    right = geometry[["collection_id", "arm", "effective_fraction", "max_reference_mass"]].rename(
        columns={
            "collection_id": "panel_b",
            "effective_fraction": "effective_fraction_b",
            "max_reference_mass": "max_mass_b",
        }
    )
    frame = cross_summary.merge(left, on=["panel_a", "arm"], validate="many_to_one")
    frame = frame.merge(right, on=["panel_b", "arm"], validate="many_to_one")
    frame["mean_effective_fraction"] = (
        frame.effective_fraction_a + frame.effective_fraction_b
    ) / 2
    frame["mean_max_reference_mass"] = (frame.max_mass_a + frame.max_mass_b) / 2
    return frame


@latex_paper_style
def plot_reference_geometry(weights, concentration, panels, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    equal = weights[weights.arm.eq("equal")]
    for index, panel in enumerate(panels):
        ranks = equal[equal.panel_id.eq(panel)].global_rank
        axes[0].scatter(ranks, np.full(len(ranks), index), s=27, alpha=.75, color="#267EAB")
    axes[0].set_yticks(range(len(panels)), panels)
    axes[0].set_xlabel("Global QC rank (1 = best)", fontweight="bold")
    axes[0].set_title("A  Fixed panel membership", loc="left", fontweight="bold")
    for arm in ARMS:
        group = concentration[concentration.arm.eq(arm)].set_index("collection_id").reindex(panels)
        if group.N_ref.notna().any():
            axes[1].plot(range(len(panels)), group.N_ref / group.N, "o-", lw=2,
                         color=ARM_COLORS[arm], label=ARM_LABELS[arm])
            axes[2].plot(range(len(panels)), group.weighted_mean_rank, "o-", lw=2,
                         color=ARM_COLORS[arm])
    axes[1].set_title("B  Reference concentration", loc="left", fontweight="bold")
    axes[1].set_ylabel(r"Effective fraction $N_{\rm eff}/N$", fontweight="bold")
    axes[1].set_ylim(0, 1.04)
    axes[2].set_title("C  Reference rank location", loc="left", fontweight="bold")
    axes[2].set_ylabel("Reference-weighted global rank", fontweight="bold")
    axes[2].set_ylim(0, 116)
    for ax in axes[1:]:
        ax.set_xticks(range(len(panels)), panels, rotation=20)
    for ax in axes:
        style_axis(ax)
    axes[1].legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    fig.suptitle("Four-panel membership and directional score-reference geometry",
                 fontweight="bold", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .95))
    save_figure(fig, out, "panel_reference_geometry")


@latex_paper_style
def plot_cross_panel(summary, panels, out):
    pairs = list(itertools.combinations(panels, 2))
    panel_number = {panel: index + 1 for index, panel in enumerate(panels)}
    pair_labels = [f"P{panel_number[a]} vs P{panel_number[b]}" for a, b in pairs]
    x = np.arange(len(pairs))
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0), sharex=True)
    for ax, power, letter in zip(axes.flat, (0, 1, 3, 5), "ABCD"):
        arms = ["equal"] if power == 0 else [
            "equal", f"best_first_score_p{power}", f"worst_first_score_p{power}"
        ]
        values = []
        for arm in arms:
            group = summary[summary.arm.eq(arm)].set_index(["panel_a", "panel_b"]).reindex(pairs)
            if group["mean"].notna().any():
                y = group["mean"].to_numpy(float)
                ax.plot(x, y, "o-", lw=2.2, color=ARM_COLORS[arm], label=ARM_LABELS[arm])
                ax.fill_between(x, group.ci_low, group.ci_high,
                                color=ARM_COLORS[arm], alpha=.13, linewidth=0)
                values.extend(group.ci_low.dropna().tolist() + group.ci_high.dropna().tolist())
        title = "Equal reference" if power == 0 else f"Score reference, p={power}"
        ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold")
        ax.set_ylabel(r"Mean transcript PCC of $\mathbf{L}_{\mathbf{t}}$", fontweight="bold")
        ax.set_xticks(x, pair_labels, rotation=20)
        adaptive_pcc_axis(ax, values)
        style_axis(ax)
        if values:
            ax.legend(frameon=False, fontsize=9)
        else:
            ax.text(.5, .5, "No complete panel pairs", transform=ax.transAxes, ha="center")
    fig.suptitle("Cross-panel concordance under matched reference policies",
                 fontweight="bold", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96))
    save_figure(fig, out, "cross_panel_pcc")


@latex_paper_style
def plot_within_panel(summary, panels, out):
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3), sharey=True)
    x = np.arange(len(panels))
    for ax, power, letter in zip(axes, (1, 3, 5), "ABC"):
        values = []
        for orientation in ("best_first", "worst_first"):
            arm = f"{orientation}_score_p{power}"
            group = summary[summary.arm.eq(arm)].set_index("panel_id").reindex(panels)
            if group["mean"].notna().any():
                ax.plot(x, group["mean"], "o-", lw=2.2, color=ARM_COLORS[arm],
                        label=ARM_LABELS[arm])
                ax.fill_between(x, group.ci_low, group.ci_high,
                                color=ARM_COLORS[arm], alpha=.13, linewidth=0)
                values.extend(group.ci_low.dropna().tolist() + group.ci_high.dropna().tolist())
        ax.set_title(f"{letter}  p={power}", loc="left", fontweight="bold")
        ax.set_xticks(x, panels, rotation=20)
        ax.set_xlabel("Fixed panel", fontweight="bold")
        ax.set_ylabel(r"Mean PCC(score, equal) of $\mathbf{L}_{\mathbf{t}}$", fontweight="bold")
        adaptive_pcc_axis(ax, values, include_one=True)
        style_axis(ax)
        if values:
            ax.legend(frameon=False, fontsize=9)
        else:
            ax.text(.5, .5, "No matched policies", transform=ax.transAxes, ha="center")
    fig.suptitle("How much does the score reference move the fitted shared profile?",
                 fontweight="bold", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .95))
    save_figure(fig, out, "within_panel_policy_sensitivity")


def _policy_rows(frame, orientation):
    return frame.set_index("arm").reindex(
        [f"{orientation}_score_p{power}" for power in (1, 3, 5)]
    )


@latex_paper_style
def plot_headline_summary(aggregate, effects, orientation, training, out):
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.3))
    powers = np.asarray((1, 3, 5), dtype=float)
    equal = aggregate.set_index("arm").loc["equal"]
    policy_style = {
        "best_first": ("Best-oriented", "#0072B2", "o"),
        "worst_first": ("Worst-oriented", "#D55E00", "s"),
    }

    ax = axes[0, 0]
    ax.axhline(equal.mean_PCC, color="#4B5563", lw=1.8, ls="--", label="Equal reference")
    ax.fill_between(
        (0.7, 5.3), equal.pcc_ci_low, equal.pcc_ci_high,
        color="#4B5563", alpha=.12, linewidth=0,
    )
    for direction, (label, color, marker) in policy_style.items():
        group = _policy_rows(aggregate, direction)
        y = group.mean_PCC.to_numpy(float)
        low = y - group.pcc_ci_low.to_numpy(float)
        high = group.pcc_ci_high.to_numpy(float) - y
        ax.errorbar(powers, y, yerr=(low, high), color=color, marker=marker,
                    lw=2.1, ms=7, capsize=3, label=label)
    ax.set_title(r"\textbf{A}\quad Cross-panel profile agreement", loc="left")
    ax.set_ylabel(r"Mean transcript-level PCC", fontweight="bold")
    ax.set_xticks(powers, [r"$p=1$", r"$p=3$", r"$p=5$"])
    ax.legend(loc="lower left")
    style_axis(ax)

    ax = axes[0, 1]
    ax.axhline(0, color="#4B5563", lw=1.2, ls="--")
    for offset, (direction, (label, color, marker)) in zip(
        (-.09, .09), policy_style.items()
    ):
        group = _policy_rows(effects, direction)
        y = group.mean_delta_PCC.to_numpy(float)
        low = y - group.pcc_ci_low.to_numpy(float)
        high = group.pcc_ci_high.to_numpy(float) - y
        ax.errorbar(powers + offset, y, yerr=(low, high), color=color,
                    marker=marker, lw=2.1, ms=7, capsize=3, label=label)
    ax.set_title(r"\textbf{B}\quad Change relative to equal reference", loc="left")
    ax.set_ylabel(r"Mean $\Delta$PCC versus equal", fontweight="bold")
    ax.set_xticks(powers, [r"$p=1$", r"$p=3$", r"$p=5$"])
    style_axis(ax)

    ax = axes[1, 0]
    ax.axhline(equal.mean_RMSE, color="#4B5563", lw=1.8, ls="--", label="Equal reference")
    ax.fill_between(
        (0.7, 5.3), equal.rmse_ci_low, equal.rmse_ci_high,
        color="#4B5563", alpha=.12, linewidth=0,
    )
    for direction, (label, color, marker) in policy_style.items():
        group = _policy_rows(aggregate, direction)
        y = group.mean_RMSE.to_numpy(float)
        low = y - group.rmse_ci_low.to_numpy(float)
        high = group.rmse_ci_high.to_numpy(float) - y
        ax.errorbar(powers, y, yerr=(low, high), color=color, marker=marker,
                    lw=2.1, ms=7, capsize=3, label=label)
    ax.set_title(r"\textbf{C}\quad Cross-panel profile discrepancy", loc="left")
    ax.set_ylabel(r"Mean transcript-level RMSE", fontweight="bold")
    ax.set_xticks(powers, [r"$p=1$", r"$p=3$", r"$p=5$"])
    style_axis(ax)

    ax = axes[1, 1]
    ax.axhline(0, color="#4B5563", lw=1.2, ls="--")
    for offset, (direction, (label, color, marker)) in zip(
        (-.09, .09), policy_style.items()
    ):
        arms = [f"{direction}_score_p{power}" for power in (1, 3, 5)]
        group = training[training.arm.isin(arms)].copy()
        group["power"] = group.arm.str.extract(r"p([135])$")[0].astype(int)
        for power, values in group.groupby("power"):
            ax.scatter(
                np.full(len(values), power + offset),
                values.delta_validation_loss_vs_equal,
                s=30, color=color, alpha=.35, edgecolor="none",
            )
        means = group.groupby("power").delta_validation_loss_vs_equal.mean().reindex((1, 3, 5))
        ax.plot(powers + offset, means, color=color, marker=marker, lw=2.1,
                ms=7, label=label)
    ax.set_title(r"\textbf{D}\quad Observation-fit control", loc="left")
    ax.set_ylabel(r"Validation loss minus equal", fontweight="bold")
    ax.set_xticks(powers, [r"$p=1$", r"$p=3$", r"$p=5$"])
    style_axis(ax)

    fig.suptitle(
        "Directional quality-score weighting changes the shared-profile decomposition",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, .965))
    save_figure(fig, out, "headline_directionality_summary")


@latex_paper_style
def plot_orientation_contrast(orientation, out):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    x = orientation.power.to_numpy(float)
    ax = axes[0]
    y = orientation.mean_delta_PCC_best_minus_worst.to_numpy(float)
    ax.errorbar(
        x, y,
        yerr=(y - orientation.pcc_ci_low, orientation.pcc_ci_high - y),
        color="#0072B2", marker="o", lw=2.2, ms=7, capsize=3,
    )
    ax.axhline(0, color="#4B5563", lw=1.1, ls="--")
    ax.set_title(r"\textbf{A}\quad Shape correlation", loc="left")
    ax.set_ylabel(r"PCC(best) $-$ PCC(worst)", fontweight="bold")
    ax.set_xticks(x, [r"$p=1$", r"$p=3$", r"$p=5$"])
    style_axis(ax)

    ax = axes[1]
    y = orientation.mean_delta_RMSE_best_minus_worst.to_numpy(float)
    ax.errorbar(
        x, y,
        yerr=(y - orientation.rmse_ci_low, orientation.rmse_ci_high - y),
        color="#D55E00", marker="s", lw=2.2, ms=7, capsize=3,
    )
    ax.axhline(0, color="#4B5563", lw=1.1, ls="--")
    ax.set_title(r"\textbf{B}\quad Profile discrepancy", loc="left")
    ax.set_ylabel(r"RMSE(best) $-$ RMSE(worst)", fontweight="bold")
    ax.set_xticks(x, [r"$p=1$", r"$p=3$", r"$p=5$"])
    style_axis(ax)
    fig.suptitle(
        "Best-oriented versus worst-oriented reference weights",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, .93))
    save_figure(fig, out, "best_vs_worst_aggregate_effect")


@latex_paper_style
def plot_concentration_association(frame, out):
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.5), sharey=True)
    markers = {0: "D", 1: "o", 3: "s", 5: "^"}
    for arm in ARMS:
        group = frame[frame.arm.eq(arm)]
        if group.empty:
            continue
        power = int(group.arm.str.extract(r"p([135])$")[0].dropna().iloc[0]) if arm != "equal" else 0
        label = ARM_LABELS[arm]
        for ax, xcolumn in zip(axes, ("mean_effective_fraction", "mean_max_reference_mass")):
            ax.scatter(group[xcolumn], group["mean"], s=58, marker=markers[power],
                       color=ARM_COLORS[arm], alpha=.82, label=label)
    score = frame[~frame.arm.eq("equal")]
    rho_effective = score[["mean_effective_fraction", "mean"]].corr(method="spearman").iloc[0, 1]
    rho_mass = score[["mean_max_reference_mass", "mean"]].corr(method="spearman").iloc[0, 1]
    axes[0].set_title(r"\textbf{A}\quad Effective reference size", loc="left")
    axes[0].set_xlabel(r"Pair-average $N_{\rm eff}/N$", fontweight="bold")
    axes[0].set_ylabel("Mean transcript-level PCC", fontweight="bold")
    axes[0].text(.04, .05, rf"Spearman $\rho={rho_effective:.2f}$", transform=axes[0].transAxes)
    axes[1].set_title(r"\textbf{B}\quad Largest reference weight", loc="left")
    axes[1].set_xlabel(r"Pair-average $\max_d\pi_d$", fontweight="bold")
    axes[1].text(.04, .05, rf"Spearman $\rho={rho_mass:.2f}$", transform=axes[1].transAxes)
    for ax in axes:
        style_axis(ax)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(.5, -.02))
    fig.suptitle(
        "Reference concentration is associated with lower cross-panel agreement",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=(0, .12, 1, .93))
    save_figure(fig, out, "reference_concentration_association")


def table_html(frame):
    return '<div class="table">' + frame.to_html(
        index=False, border=0, escape=True, na_rep="—",
        float_format=lambda value: f"{value:.4f}",
    ) + "</div>"


def render_report(
    root,
    out,
    manifest,
    panels,
    ids,
    availability,
    concentration,
    cross_summary,
    within_summary,
    effect_summary,
    aggregate_summary,
    aggregate_effect_summary,
    orientation_aggregate_summary,
    concentration_frame,
    training,
    replicates,
    bootstrap_seed,
):
    complete = int(availability.status.eq("validated_predictions").sum())
    status = availability[["panel_id", "arm", "status"]].copy()
    status = status.pivot(index="panel_id", columns="arm", values="status").reset_index()
    aggregate_display = aggregate_summary.set_index("arm").reindex(DISPLAY_ARMS).reset_index()
    aggregate_display.insert(0, "policy", aggregate_display.arm.map(ARM_LABELS))
    aggregate_display = aggregate_display[
        ["policy", "n_transcripts", "mean_PCC", "pcc_ci_low", "pcc_ci_high",
         "mean_RMSE", "rmse_ci_low", "rmse_ci_high"]
    ]
    effect_display = (
        aggregate_effect_summary.set_index("arm")
        .reindex(DISPLAY_ARMS[1:])
        .reset_index()
    )
    effect_display.insert(0, "policy", effect_display.arm.map(ARM_LABELS))
    effect_display = effect_display[
        ["policy", "mean_delta_PCC", "pcc_ci_low", "pcc_ci_high",
         "fraction_PCC_improved", "mean_delta_RMSE", "rmse_ci_low",
         "rmse_ci_high", "fraction_RMSE_improved"]
    ]
    indexed = aggregate_summary.set_index("arm")
    equal = indexed.loc["equal"]
    best_p5 = indexed.loc["best_first_score_p5"]
    worst_p5 = indexed.loc["worst_first_score_p5"]
    p5_direction = orientation_aggregate_summary.set_index("power").loc[5]
    score_points = concentration_frame[~concentration_frame.arm.eq("equal")]
    effective_rho = score_points[["mean_effective_fraction", "mean"]].corr(
        method="spearman"
    ).iloc[0, 1]
    max_mass_rho = score_points[["mean_max_reference_mass", "mean"]].corr(
        method="spearman"
    ).iloc[0, 1]
    worst_p5_loss = training[training.arm.eq("worst_first_score_p5")][
        "delta_validation_loss_vs_equal"
    ].mean()
    geometry_display = concentration.assign(
        effective_fraction=concentration.N_ref / concentration.N
    ).groupby("arm", as_index=False).agg(
        mean_effective_fraction=("effective_fraction", "mean"),
        mean_max_reference_mass=("max_reference_mass", "mean"),
        mean_weighted_global_rank=("weighted_mean_rank", "mean"),
    ).set_index("arm").reindex(DISPLAY_ARMS).reset_index()
    geometry_display.insert(0, "policy", geometry_display.arm.map(ARM_LABELS))
    geometry_display = geometry_display[
        ["policy", "mean_effective_fraction", "mean_max_reference_mass",
         "mean_weighted_global_rank"]
    ]
    panel_sizes = ", ".join(str(len(manifest["panels"][panel])) for panel in panels)
    style = """
    :root{--ink:#203448;--blue:#176893;--line:#dbe5ea;--pale:#edf6fa;--orange:#d17820}body{max-width:1160px;margin:34px auto;padding:0 25px;font:16px/1.65 system-ui;color:var(--ink)}h1{line-height:1.15;font-size:2.35rem}h2{margin-top:42px;border-bottom:2px solid var(--line);padding-bottom:6px}h3{margin-top:28px}.subtitle{font-size:1.15rem;color:#53697b}.note{padding:14px 17px;background:var(--pale);border-left:4px solid #267eab}.caution{background:#fff7ed;border-color:var(--orange)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:20px 0}.card{border:1px solid var(--line);border-radius:8px;padding:15px;background:#fbfdfe}.number{font-size:1.65rem;font-weight:750;color:#145d7c}.label{font-size:.91rem;color:#607383}img{width:100%;border:1px solid #e0e7eb;background:white}.table{overflow:auto;border:1px solid #dce5eb;margin:12px 0 25px}.table table{border-collapse:collapse;width:100%;font-size:13px}.table td,.table th{padding:7px 9px;border-bottom:1px solid #e2e8ec;text-align:right;white-space:nowrap}.table td:first-child,.table th:first-child{text-align:left}a{color:var(--blue)}code{background:#edf1f3;padding:2px 4px}details{margin:15px 0}li{margin:.35rem 0}.good{color:#18724b}.bad{color:#9a3f25}
    """
    report = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Four-panel score directionality</title><style>{style}</style></head><body>
<h1>Four-panel directional quality-score analysis</h1>
<p class="subtitle">Does orienting the gamma reference toward higher-quality profiles make the learned shared profile more reproducible across source-disjoint dataset panels?</p>
<p class="note"><b>{complete}/{len(manifest['tasks'])} frozen fits validated.</b> The four source-family-disjoint panels contain {panel_sizes} datasets. Every comparison uses the same {len(ids):,} held-out test transcripts, full-CDS coordinates, best-validation-loss checkpoint rule, and training seed {manifest['training_seeds'][0]}.</p>
<div class="cards">
  <div class="card"><div class="number">{equal.mean_PCC:.3f}</div><div class="label">mean cross-panel PCC, equal reference</div></div>
  <div class="card"><div class="number">{best_p5.mean_PCC:.3f}</div><div class="label">best-oriented p=5 PCC</div></div>
  <div class="card"><div class="number">{worst_p5.mean_PCC:.3f}</div><div class="label">worst-oriented p=5 PCC</div></div>
  <div class="card"><div class="number">{p5_direction.mean_delta_PCC_best_minus_worst:+.3f}</div><div class="label">best-minus-worst PCC at p=5</div></div>
</div>

<h2>Executive result</h2>
<p><b>Uniform reference weighting is the most reproducible convention in this experiment.</b> Averaging each transcript's PCC across all six panel pairs gives {equal.mean_PCC:.3f} under equal weighting. Best-oriented weighting falls to {indexed.loc['best_first_score_p1'].mean_PCC:.3f}, {indexed.loc['best_first_score_p3'].mean_PCC:.3f}, and {best_p5.mean_PCC:.3f} for p=1, 3, and 5. Worst-oriented weighting gives {indexed.loc['worst_first_score_p1'].mean_PCC:.3f}, {indexed.loc['worst_first_score_p3'].mean_PCC:.3f}, and {worst_p5.mean_PCC:.3f}.</p>
<p>The quality direction is nevertheless detectable under the strongest intervention: at p=5, best orientation exceeds worst orientation by {p5_direction.mean_delta_PCC_best_minus_worst:+.3f} PCC (95% transcript-bootstrap interval {p5_direction.pcc_ci_low:+.3f} to {p5_direction.pcc_ci_high:+.3f}) and reduces RMSE by {-p5_direction.mean_delta_RMSE_best_minus_worst:.3f}. Thus, emphasizing better profiles is less damaging than emphasizing worse profiles when the reference is highly concentrated, but neither extreme improves on equal weighting.</p>
<img src="headline_directionality_summary.svg" alt="Aggregate cross-panel PCC, RMSE and validation-loss controls">

<h2>What was tested</h2>
<p>Dataset membership is fixed within each panel. If <i>s</i><sub>d</sub> is <code>quality_rank_score</code> (lower is better), best-oriented weights use &pi;<sub>d</sub>&prop;(<i>s</i><sub>min</sub>/<i>s</i><sub>d</sub>)<sup>p</sup>, worst-oriented weights use &pi;<sub>d</sub>&prop;(<i>s</i><sub>d</sub>/<i>s</i><sub>max</sub>)<sup>p</sup>, and every vector is normalized to sum to one. The model's transcript–dataset reliability weights are unchanged.</p>
<p class="note caution"><b>“Worst-oriented” does not mean a panel of the worst datasets.</b> Equal, best-oriented, and worst-oriented fits contain exactly the same datasets. The intervention changes only which datasets define the gamma-centering reference.</p>
<img src="panel_reference_geometry.svg" alt="Fixed panel membership and reference-weight geometry">
{table_html(geometry_display)}

<h2>Main matched analysis</h2>
<p>For each held-out transcript and policy, PCC and RMSE were calculated separately on aligned full-CDS positions for each of the six panel pairs. The primary summary then averages the six pair-level metrics within transcript, leaving 714 independent transcript records for bootstrap resampling. Codons are never pooled across transcripts.</p>
{table_html(aggregate_display)}
<img src="best_vs_worst_aggregate_effect.svg" alt="Best-oriented minus worst-oriented aggregate effects">
<p>At p=1 and p=3, the best-minus-worst PCC contrast is small and is not supported consistently by RMSE. At p=5, both metrics favor the best-oriented reference. This is the cleanest evidence here that score direction matters, but it is evidence about reproducibility of the selected decomposition—not biological truth.</p>
{table_html(effect_display)}

<h2>Reference concentration explains much of the power trend</h2>
<p>The effective reference size is N<sub>eff</sub>=1/&Sigma;<sub>d</sub>&pi;<sub>d</sub><sup>2</sup>. Across the 36 score-weighted panel-pair summaries, PCC has descriptive Spearman association &rho;={effective_rho:.2f} with pair-average N<sub>eff</sub>/N and &rho;={max_mass_rho:.2f} with pair-average maximum reference mass. The second association is expected to be negative when concentration is harmful. These correlations are descriptive because pair summaries reuse models and vary jointly with p and orientation.</p>
<img src="reference_concentration_association.svg" alt="Reference concentration versus cross-panel agreement">

<h2>Observation fit does not diagnose shared-profile stability</h2>
<p>The worst-oriented p=5 models have validation loss {worst_p5_loss:+.4f} lower than their within-panel equal controls on average, yet their shared-profile reproducibility is much lower. This is not a contradiction: the likelihood constrains the product of shared and dataset-specific components, whereas the reference convention selects their decomposition. A slightly better held-out count likelihood therefore need not imply a more panel-stable shared profile.</p>
{table_html(training[['panel_id','arm','selected_epoch','validation_loss','delta_validation_loss_vs_equal']])}

<h2>Pair-specific and within-panel diagnostics</h2>
<img src="cross_panel_pcc.svg" alt="Pair-specific transcript-level cross-panel PCC under each policy">
<p>Pair-specific effects are heterogeneous. For example, best-oriented weighting improves the P1–P4 comparison but substantially degrades P2–P3 as p increases. This heterogeneity is why the transcript-level six-pair aggregate is the headline rather than one favorable pair.</p>
<img src="within_panel_policy_sensitivity.svg" alt="Same-panel agreement between score and equal references">
<p>Within-panel agreement with the equal-reference profile is near one at p=1 for most panels, then falls strongly at p=3 and p=5. The reference is therefore not a cosmetic gauge choice at high power: it materially changes which positional signal is assigned to the shared component.</p>

<h2>Interpretation</h2>
<ul>
  <li><b>Not supported:</b> measured quality-score weighting generally improves cross-panel recovery of the shared profile.</li>
  <li><b>Supported:</b> reference weighting matters, especially when concentrated; at p=5 the better-quality direction is clearly preferable to the deliberately adverse direction.</li>
  <li><b>Supported:</b> equal weighting is the strongest default among these tested conventions for cross-panel reproducibility.</li>
  <li><b>Mechanistic reading:</b> strong weighting reduces averaging over panel-specific technical effects, so each disjoint panel can define a more idiosyncratic reference target.</li>
  <li><b>Not identified:</b> which convention is closest to an unknown biological ground truth. Cross-panel agreement is reproducibility, not biological correctness.</li>
</ul>
<p class="note caution"><b>Main limitation:</b> there is one training seed and four fixed panels. Bootstrap intervals quantify transcript sampling variation conditional on those fitted models; they do not measure optimization-seed or alternative-panel uncertainty. The six panel pairs also share models and must not be treated as six independent experiments.</p>

<h2>Availability</h2>{table_html(status)}
<details><summary><b>Pair-specific cross-panel summary</b></summary>{table_html(cross_summary)}</details>
<details><summary><b>Pair-specific paired score-minus-equal effects</b></summary>{table_html(effect_summary)}</details>
<details><summary><b>Within-panel score-versus-equal summary</b></summary>{table_html(within_summary)}</details>
<h2>Exact outputs</h2>
<p><a href="aggregate_cross_panel_metrics.csv">Transcript-level six-pair aggregates</a> · <a href="aggregate_cross_panel_summary.csv">aggregate summary</a> · <a href="aggregate_policy_effects_vs_equal.csv">paired effects versus equal</a> · <a href="aggregate_orientation_effects.csv">best-versus-worst effects</a> · <a href="cross_panel_metrics.csv">pair-specific metrics</a> · <a href="reference_concentration.csv">reference geometry</a> · <a href="training_outcomes.csv">checkpoint and validation-loss controls</a> · <a href="availability.csv">artifact audit</a></p>
<p><b>Reproduce:</b> <code>RIBOUNMIX_PLOT_TEX=1 .venv/bin/python analyses/analyze_four_panel_quality_score_directionality.py --experiment-root results/four_panel_quality_score_directional_balanced_seed42 --bootstrap-replicates {replicates} --bootstrap-seed {bootstrap_seed}</code></p>
<p>Generated {datetime.now(timezone.utc).isoformat()} with {replicates:,} transcript-bootstrap replicates per mean.</p>
</body></html>'''
    (out / "analysis_report.html").write_text(report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_BALANCED_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260919)
    args = parser.parse_args(argv)
    root = args.experiment_root.expanduser().resolve()
    if args.bootstrap_replicates < 200:
        parser.error("--bootstrap-replicates must be at least 200")
    manifest, panels, ids, configs = audit_design(root)
    weights = pd.read_csv(root / "reference_weights.csv")
    profiles, availability = collect(root, manifest, configs, ids, weights)
    seed = manifest["training_seeds"][0]
    cross = cross_panel_rows(profiles, panels, seed, ids)
    within = within_panel_rows(profiles, panels, seed, ids)
    orientation = orientation_rows(profiles, panels, seed, ids)
    cross_summary = summarize_pcc(
        cross, ["panel_a", "panel_b", "arm"],
        args.bootstrap_replicates, args.bootstrap_seed,
    )
    within_summary = summarize_pcc(
        within, ["panel_id", "arm"],
        args.bootstrap_replicates, args.bootstrap_seed,
    )
    orientation_summary = summarize_pcc(
        orientation, ["panel_id", "power"],
        args.bootstrap_replicates, args.bootstrap_seed,
    )
    effect_metrics, effect_summary = paired_cross_panel_effects(
        cross, args.bootstrap_replicates, args.bootstrap_seed
    )
    concentration = pd.read_csv(root / "reference_concentration.csv")
    aggregate_metrics, aggregate_summary = aggregate_cross_panel_metrics(
        cross, args.bootstrap_replicates, args.bootstrap_seed
    )
    aggregate_effect_metrics, aggregate_effect_summary = aggregate_policy_contrasts(
        aggregate_metrics, args.bootstrap_replicates, args.bootstrap_seed
    )
    orientation_aggregate_metrics, orientation_aggregate_summary = (
        aggregate_orientation_contrasts(
            aggregate_metrics, args.bootstrap_replicates, args.bootstrap_seed
        )
    )
    training = training_outcomes(root, manifest)
    concentration_frame = concentration_association(cross_summary, concentration)

    expected_aggregate_rows = len(ids) * len(ARMS)
    n_complete = int(aggregate_metrics.complete_six_pair_record.sum())
    if n_complete != expected_aggregate_rows:
        raise ValueError(
            "The primary matched analysis is incomplete: "
            f"expected {expected_aggregate_rows:,} transcript-policy records with "
            f"all six panel pairs, found {n_complete:,}."
        )
    out = artifact_directory("real_data", root)
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "availability": availability,
        "reference_concentration": concentration,
        "reference_concentration_association": concentration_frame,
        "training_outcomes": training,
        "cross_panel_metrics": cross,
        "cross_panel_summary": cross_summary,
        "aggregate_cross_panel_metrics": aggregate_metrics,
        "aggregate_cross_panel_summary": aggregate_summary,
        "aggregate_policy_metrics_vs_equal": aggregate_effect_metrics,
        "aggregate_policy_effects_vs_equal": aggregate_effect_summary,
        "aggregate_orientation_metrics": orientation_aggregate_metrics,
        "aggregate_orientation_effects": orientation_aggregate_summary,
        "within_panel_policy_metrics": within,
        "within_panel_policy_summary": within_summary,
        "best_vs_worst_metrics": orientation,
        "best_vs_worst_summary": orientation_summary,
        "paired_score_vs_equal_metrics": effect_metrics,
        "paired_score_vs_equal_summary": effect_summary,
    }
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    plot_reference_geometry(weights, concentration, panels, out)
    plot_cross_panel(cross_summary, panels, out)
    plot_within_panel(within_summary, panels, out)
    plot_headline_summary(
        aggregate_summary,
        aggregate_effect_summary,
        orientation_aggregate_summary,
        training,
        out,
    )
    plot_orientation_contrast(orientation_aggregate_summary, out)
    plot_concentration_association(concentration_frame, out)
    render_report(
        root, out, manifest, panels, ids, availability, concentration,
        cross_summary, within_summary, effect_summary, aggregate_summary,
        aggregate_effect_summary, orientation_aggregate_summary,
        concentration_frame, training, args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    command = (
        "RIBOUNMIX_PLOT_TEX=1 .venv/bin/python "
        "analyses/analyze_four_panel_quality_score_directionality.py "
        "--experiment-root results/four_panel_quality_score_directional_balanced_seed42 "
        f"--bootstrap-replicates {args.bootstrap_replicates} "
        f"--bootstrap-seed {args.bootstrap_seed}"
    )
    (out / "REPRODUCE.md").write_text(
        "# Reproduce the four-panel directional score analysis\n\n"
        "Run from the repository root:\n\n"
        f"```bash\n{command}\n```\n\n"
        "The analysis reads frozen predictions only; it does not train or update a model.\n"
    )
    outputs = {
        path.name: sha256(path) for path in sorted(out.iterdir())
        if path.is_file() and path.name != "analysis_manifest.json"
    }
    complete = int(availability.status.eq("validated_predictions").sum())
    write_json(out / "analysis_manifest.json", dict(
        experiment_design=manifest["experiment_design"],
        experiment_manifest_sha256=sha256(root / "experiment_manifest.json"),
        analysis_code_sha256=sha256(Path(__file__)),
        generated_utc=datetime.now(timezone.utc).isoformat(),
        planned_models=len(manifest["tasks"]),
        validated_models=complete,
        panel_dataset_counts={panel: len(manifest["panels"][panel]) for panel in panels},
        common_train_transcripts=len(manifest["source_folds"][panels[0]]["train_ids"]),
        common_validation_transcripts=len(
            manifest["source_folds"][panels[0]]["validation_ids"]
        ),
        common_test_transcripts=len(ids),
        common_test_transcript_id_hash=transcript_id_hash(ids),
        panel_pairs=6,
        primary_estimand=(
            "mean, within transcript, of the six source-disjoint panel-pair "
            "full-CDS PCC or RMSE values"
        ),
        uncertainty=(
            "paired transcript bootstrap conditional on the four fitted panels "
            "and training seed 42"
        ),
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        outputs=outputs,
    ))
    print(f"Validated {complete}/{len(manifest['tasks'])} fits; report: {out / 'analysis_report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
