#!/usr/bin/env python3
"""Create panel D comparing available successive-size agreement summaries.

Ranked PCCs are recomputed from the saved profile arrays. The uniform series is
loaded from Figure 1's array-derived source table and independently rechecked.
The two series use different subset designs, so this is a descriptive overlay,
not an isolated ranked-minus-equal reference-weight effect.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyses import create_real_data_equal_figure as figure_one
from analyses.analyze_real_exp8_quality_rank_partial import load_export
from analyses.analyze_real_exp8_stability import _compare_profiles
from run_real_exp8_L_stability_quality_rank import inspect_ranking_components
from Utils.reliability_references import transcript_id_hash

DEFAULT_ROOT = (ROOT / "results/real_exp8_L_stability_quality_rank_10components"
                / "cumulative_qrank10components_p1.0_seed42")
DEFAULT_EQUAL_SOURCE = ROOT / "figures/real_data_equal_source"
MODEL_SIZES = (2, 5, 10, 20, 40, 80, 114)
SUCCESSIVE_PAIRS = tuple(zip(MODEL_SIZES[:-1], MODEL_SIZES[1:]))
TRAINING_SEED = 42
BOOTSTRAP_SEED = 20260910
BOOTSTRAP_DRAWS = 5000
RANKING_SHA256 = "5811cadf68c56740e83b232b2990299d527630326205db9cba8bc7024d3cf1f8"
ORANGE = "#D55E00"
GRAY = "#66727C"
SERIES_ORDER = ("equal_representative", "ranked_cumulative")
SERIES_LABELS = {
    "equal_representative": "Uniform representative",
    "ranked_cumulative": "Ranked cumulative",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--equal-source-dir", type=Path, default=DEFAULT_EQUAL_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--reference-figure", type=Path, default=ROOT / "figures/real_data_equal.pdf")
    parser.add_argument("--no-tex", action="store_true")
    args = parser.parse_args(argv)
    for name in ("run_root", "equal_source_dir", "output_dir", "reference_figure"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def validate_ranking(root: Path, manifest: dict) -> tuple[dict[str, float], dict]:
    path = root / "frozen_HEK_riboseq_profile_quality_rank_components.tsv"
    components = inspect_ranking_components(path, expected_count=10)
    digest = sha256(path)
    table = pd.read_csv(path, sep="\t")
    ranks = pd.to_numeric(table.quality_rank, errors="raise")
    if digest != RANKING_SHA256 or manifest.get("ranking_sha256") != digest:
        raise ValueError("Experiment does not use the frozen manuscript ten-component ranking.")
    if len(table) != 115 or ranks.max() != 115 or table.dataset.duplicated().any():
        raise ValueError("Expected the complete 115-row global rank universe with R=115.")
    rank = dict(zip(table.dataset.astype(str), ranks.astype(float), strict=True))
    return rank, dict(path=str(path), sha256=digest, rows=len(table), R=float(ranks.max()),
                      component_count=10, components=components["columns"], direction="1 = best",
                      formula="q_d=(R-r_d+1)/R; pi_d=q_d/sum_selected(q)")


def task_directory(root: Path, task: dict) -> Path:
    directory = root / task["directory"]
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return directory


def validate_weight_formula(root: Path, task: dict, rank: dict[str, float]) -> list[dict]:
    directory = task_directory(root, task)
    subset = read_json(directory / "subset_manifest.json")
    reference = subset["fixed_gamma_reference"]
    datasets = list(task["datasets"])
    if (subset["datasets"] != datasets or reference["dataset_names"] != datasets
            or reference["weighting"] != "quality_rank"
            or float(reference["quality_rank_power"]) != 1.0
            or reference.get("ranks_recomputed_within_panel") is not False):
        raise ValueError(f"{task['run_id']}: saved ranked-reference contract differs.")
    missing = sorted(set(datasets) - rank.keys())
    if missing:
        raise ValueError(f"Missing frozen global ranks: {missing}")
    q = np.array([(115-rank[d]+1)/115 for d in datasets], dtype=float)
    pi = q/q.sum()
    np.testing.assert_allclose([reference["base_quality_weight"][d] for d in datasets], q,
                               rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose([reference["pi"][d] for d in datasets], pi,
                               rtol=1e-12, atol=1e-14)
    return [dict(N=task["N"], run_id=task["run_id"], dataset_order=i,
                 dataset_id=d, global_rank=rank[d], q=float(q[i]), pi=float(pi[i]))
            for i, d in enumerate(datasets)]


def load_equal_adjacent_source(source: Path, transcript_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load and verify Figure 1's uniform-reference adjacent-size records."""
    manifest_path = source / "figure_manifest.json"
    values_path = source / "panel_b_adjacent_size_per_transcript.csv"
    pair_summary_path = source / "panel_b_adjacent_size_pair_summary.csv"
    transition_summary_path = source / "panel_b_adjacent_size_summary.csv"
    provenance_path = source / "run_provenance.csv"
    for path in (manifest_path, values_path, pair_summary_path, transition_summary_path, provenance_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = read_json(manifest_path)
    expected_hash = transcript_id_hash(transcript_ids)
    if (manifest.get("training_seed") != TRAINING_SEED
            or manifest.get("panel_B_test_count") != len(transcript_ids)
            or manifest.get("panel_B_test_hash") != expected_hash
            or manifest.get("uniform_gamma_reference") is not True
            or manifest.get("adjacent_size_models_are_not_nested") is not True
            or [list(pair) for pair in SUCCESSIVE_PAIRS]
               != manifest.get("panel_B_adjacent_size_transitions")):
        raise ValueError("Figure 1 uniform adjacent-size provenance is incompatible.")

    values = pd.read_csv(values_path, float_precision="round_trip")
    required = {"transcript_id", "N_from", "N_to", "run_from", "run_to",
                "profile_domain", "PCC", "status"}
    if not required <= set(values):
        raise ValueError(f"Uniform source table is missing columns: {sorted(required-set(values))}")
    if (values.status.ne("valid").any() or values.profile_domain.ne("full_CDS").any()
            or not np.isfinite(values.PCC).all()):
        raise ValueError("Uniform source contains invalid or non-full-CDS PCC records.")
    if values.duplicated(["transcript_id", "N_from", "N_to", "run_from", "run_to"]).any():
        raise ValueError("Uniform source contains duplicate transcript/model-pair records.")
    if set(values.transcript_id.astype(str)) != set(transcript_ids):
        raise ValueError("Uniform and ranked analyses do not use the same held-out transcripts.")

    expected_pair_counts = {(2, 5): 36, (5, 10): 36, (10, 20): 36,
                            (20, 40): 36, (40, 80): 18, (80, 114): 3}
    observed_transitions = set(map(tuple, values[["N_from", "N_to"]].drop_duplicates().to_numpy()))
    if observed_transitions != set(SUCCESSIVE_PAIRS):
        raise ValueError("Uniform adjacent-size transitions differ from the ranked chain.")

    pair_keys = ["N_from", "N_to", "run_from", "run_to"]
    recomputed_pairs = (values.groupby(pair_keys, sort=True, as_index=False)
                        .agg(mean_transcript_PCC=("PCC", "mean"),
                             valid_PCC=("PCC", "size"),
                             transcripts=("transcript_id", "nunique")))
    for pair, expected_count in expected_pair_counts.items():
        selected = recomputed_pairs.loc[
            recomputed_pairs.N_from.eq(pair[0]) & recomputed_pairs.N_to.eq(pair[1])]
        if (len(selected) != expected_count or selected.valid_PCC.ne(len(transcript_ids)).any()
                or selected.transcripts.ne(len(transcript_ids)).any()):
            raise ValueError(f"Uniform transition {pair} has incomplete model-pair records.")

    saved_pairs = pd.read_csv(pair_summary_path, float_precision="round_trip")
    checked = recomputed_pairs.merge(
        saved_pairs[pair_keys + ["mean_transcript_PCC"]], on=pair_keys,
        suffixes=("_recomputed", "_saved"), validate="one_to_one")
    if len(checked) != len(recomputed_pairs):
        raise ValueError("Uniform pair-summary mapping is incomplete.")
    np.testing.assert_allclose(checked.mean_transcript_PCC_recomputed,
                               checked.mean_transcript_PCC_saved, rtol=0, atol=5e-15)

    saved_transitions = pd.read_csv(transition_summary_path, float_precision="round_trip")
    recomputed_transitions = (recomputed_pairs.groupby(["N_from", "N_to"], as_index=False)
                              .mean_transcript_PCC.mean())
    checked_transitions = recomputed_transitions.merge(
        saved_transitions[["N_from", "N_to", "mean_over_model_pair_means"]],
        on=["N_from", "N_to"], validate="one_to_one")
    np.testing.assert_allclose(checked_transitions.mean_transcript_PCC,
                               checked_transitions.mean_over_model_pair_means,
                               rtol=0, atol=5e-15)

    run_provenance = pd.read_csv(provenance_path)
    used_runs = set(values.run_from) | set(values.run_to)
    used_provenance = run_provenance.loc[run_provenance.run_id.isin(used_runs)].copy()
    if (set(used_provenance.run_id) != used_runs
            or used_provenance.gamma_reference_weighting.ne("equal").any()
            or not np.allclose(used_provenance.pi_value,
                               1.0 / used_provenance.N, rtol=0, atol=1e-15)):
        raise ValueError("Uniform run provenance does not verify pi_d=1/N.")

    aggregate = (values.groupby(["transcript_id", "N_from", "N_to"], sort=True, as_index=False)
                 .agg(PCC=("PCC", "mean"), model_pairs_contributing=("PCC", "size")))
    for comparison_index, (N_a, N_b) in enumerate(SUCCESSIVE_PAIRS):
        mask = aggregate.N_from.eq(N_a) & aggregate.N_to.eq(N_b)
        if (mask.sum() != len(transcript_ids)
                or aggregate.loc[mask, "model_pairs_contributing"].ne(
                    expected_pair_counts[(N_a, N_b)]).any()):
            raise ValueError(f"Uniform per-transcript aggregation failed for {(N_a, N_b)}.")
        aggregate.loc[mask, "comparison_index"] = comparison_index
        aggregate.loc[mask, "comparison_id"] = f"N{N_a:03d}_vs_N{N_b:03d}"
    aggregate = aggregate.rename(columns={"N_from": "N_a", "N_to": "N_b"})
    aggregate["series"] = "equal_representative"
    aggregate["reference_policy"] = "equal"
    aggregate["subset_design"] = "representative_all_cross_size_model_pairs"
    metadata = {
        "source_directory": str(source),
        "figure_manifest": str(manifest_path),
        "figure_manifest_sha256": sha256(manifest_path),
        "per_transcript_source": str(values_path),
        "per_transcript_source_sha256": sha256(values_path),
        "model_pairs_per_transition": {f"{a}:{b}": count
                                       for (a, b), count in expected_pair_counts.items()},
        "models_are_nested": False,
        "aggregation": "mean PCC over transcripts within model pair, then unweighted mean over model pairs",
    }
    return aggregate, saved_pairs, metadata


def load_and_compare(root: Path, equal_source: Path):
    manifest = read_json(root / "experiment_manifest.json")
    if (manifest.get("experiment_design") != "cumulative_top_quality"
            or manifest.get("training_seeds") != [TRAINING_SEED]):
        raise ValueError("Expected the seed-42 cumulative top-quality experiment.")
    rank, ranking_provenance = validate_ranking(root, manifest)
    common = read_json(root / "common_test_manifest.json")
    transcript_ids = list(map(str, common["common_test_ids"]))
    if len(transcript_ids) != 1771 or len(set(transcript_ids)) != len(transcript_ids):
        raise ValueError("Expected 1,771 unique frozen common-test transcripts.")
    expected_hash = transcript_id_hash(transcript_ids)
    tasks = {int(task["N"]): task for task in manifest["tasks"]}
    if set(tasks) != {2, 5, 10, 20, 40, 80, 114}:
        raise ValueError("Cumulative task-size universe differs from the frozen experiment.")
    if any(task["training_seed"] != TRAINING_SEED for task in tasks.values()):
        raise ValueError("Mixed training seeds are not accepted.")
    previous: set[str] = set()
    for N in sorted(tasks):
        current = set(tasks[N]["datasets"])
        if previous and not previous < current:
            raise ValueError("Dataset collections are not strict cumulative prefixes.")
        previous = current

    profiles, provenance, weight_rows = {}, [], []
    for N in MODEL_SIZES:
        task = tasks[N]
        weight_rows.extend(validate_weight_formula(root, task, rank))
        profile, checks, record = load_export(root, task, transcript_ids, 1e-4,
                                               expected_weighting="quality_rank")
        if record["test_transcript_id_hash"] != expected_hash or len(checks) != len(transcript_ids):
            raise ValueError(f"N={N}: held-out export is incomplete.")
        profiles[N] = profile
        provenance.append(record)

    rows = []
    for comparison_index, (N_a, N_b) in enumerate(SUCCESSIVE_PAIRS):
        comparison_id = f"N{N_a:03d}_vs_N{N_b:03d}"
        compared = _compare_profiles(left=profiles[N_a], right=profiles[N_b],
                                     transcript_ids=transcript_ids)
        for row in compared:
            rows.append(dict(comparison_index=comparison_index,
                             comparison_id=comparison_id, N_a=N_a, N_b=N_b,
                             training_seed=TRAINING_SEED,
                             run_a=tasks[N_a]["run_id"], run_b=tasks[N_b]["run_id"], **row))
    ranked_values = pd.DataFrame(rows)
    comparison_ids = [f"N{a:03d}_vs_N{b:03d}" for a, b in SUCCESSIVE_PAIRS]
    ranked = ranked_values[["transcript_id", "comparison_index", "comparison_id",
                            "N_a", "N_b", "PCC"]].copy()
    ranked["model_pairs_contributing"] = 1
    ranked["series"] = "ranked_cumulative"
    ranked["reference_policy"] = "quality_rank"
    ranked["subset_design"] = "nested_cumulative_top_quality"
    equal, equal_pair_summary, equal_provenance = load_equal_adjacent_source(
        equal_source, transcript_ids)
    values = pd.concat([equal, ranked], ignore_index=True, sort=False)
    expected_columns = pd.MultiIndex.from_product(
        [SERIES_ORDER, comparison_ids], names=["series", "comparison_id"])
    pivot = values.pivot(index="transcript_id", columns=["series", "comparison_id"], values="PCC")
    pivot = pivot.reindex(columns=expected_columns)
    finite = np.isfinite(pivot.to_numpy()).all(axis=1)
    values["included_primary_common_cohort"] = values.transcript_id.isin(pivot.index[finite])
    values["exclusion_reason"] = np.where(values.included_primary_common_cohort, "",
                                           "undefined_PCC_in_at_least_one_successive_comparison")
    matrix = pivot.loc[finite].to_numpy(float).reshape(-1, len(SERIES_ORDER), len(SUCCESSIVE_PAIRS))
    if matrix.shape[0] == 0:
        raise ValueError("No complete finite transcript cohort.")
    point = matrix.mean(axis=0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty((BOOTSTRAP_DRAWS, len(SERIES_ORDER), len(SUCCESSIVE_PAIRS)), dtype=float)
    index_digest = hashlib.sha256()
    for draw in range(BOOTSTRAP_DRAWS):
        indices = rng.integers(0, len(matrix), size=len(matrix))
        index_digest.update(indices.astype("<i8").tobytes())
        samples[draw] = matrix[indices].mean(axis=0)
    low, high = np.quantile(samples, [.025, .975], axis=0, method="linear")
    summary_rows = []
    for series_index, series in enumerate(SERIES_ORDER):
        for comparison_index, (N_a, N_b) in enumerate(SUCCESSIVE_PAIRS):
            summary_rows.append(dict(series=series, series_label=SERIES_LABELS[series],
                comparison_index=comparison_index, comparison_id=comparison_ids[comparison_index],
                N_a=N_a, N_b=N_b, training_seed=TRAINING_SEED,
                n_transcripts=len(matrix), mean_PCC=point[series_index, comparison_index],
                ci_lower=low[series_index, comparison_index],
                ci_upper=high[series_index, comparison_index],
                bootstrap_draws=BOOTSTRAP_DRAWS, bootstrap_seed=BOOTSTRAP_SEED))
    summary = pd.DataFrame(summary_rows)
    difference_samples = samples[:, 1, :] - samples[:, 0, :]
    difference_low, difference_high = np.quantile(
        difference_samples, [.025, .975], axis=0, method="linear")
    difference = pd.DataFrame(dict(
        comparison_index=np.arange(len(SUCCESSIVE_PAIRS)), comparison_id=comparison_ids,
        N_a=[pair[0] for pair in SUCCESSIVE_PAIRS], N_b=[pair[1] for pair in SUCCESSIVE_PAIRS],
        ranked_minus_equal_descriptive=point[1]-point[0], ci_lower=difference_low,
        ci_upper=difference_high, n_transcripts=len(matrix), bootstrap_draws=BOOTSTRAP_DRAWS,
        bootstrap_seed=BOOTSTRAP_SEED,
        interpretation="confounded design contrast; not an isolated reference-weight effect"))
    bootstrap = dict(draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED,
        resampling_unit="held-out transcript carrying both series and all six successive-size comparisons",
        common_cohort_size=len(matrix), transcript_order_sha256=transcript_id_hash(pivot.index[finite]),
        bootstrap_index_stream_sha256=index_digest.hexdigest(),
        interval="2.5th and 97.5th percentiles; NumPy linear quantiles")
    return (values, summary, difference, equal_pair_summary, pd.DataFrame(provenance),
            pd.DataFrame(weight_rows), ranking_provenance, equal_provenance, bootstrap)


def reference_height(reference: Path) -> float:
    if not reference.is_file():
        return 2.65
    result = subprocess.run(["pdfinfo", str(reference)], capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        if line.startswith("Page size:"):
            return float(line.split()[4])/72
    raise ValueError("Could not read Figure 1 page height.")


def plot(summary: pd.DataFrame, height: float, no_tex: bool):
    style = dict(figure_one.FIGURE_RC)
    if no_tex:
        style.update({"text.usetex": False, "text.latex.preamble": "",
                      "font.serif": ["DejaVu Serif"], "mathtext.fontset": "cm"})
    with matplotlib.rc_context(style):
        fig, axis = plt.subplots(figsize=(3.75, height), layout="constrained")
        x = np.arange(len(SUCCESSIVE_PAIRS), dtype=float)
        styles = {
            "equal_representative": dict(color=GRAY, linestyle="--", marker="D", offset=-.035),
            "ranked_cumulative": dict(color=ORANGE, linestyle="-", marker="o", offset=.035),
        }
        for series in SERIES_ORDER:
            group = summary.loc[summary.series.eq(series)].sort_values("comparison_index")
            style_values = styles[series]
            positioned_x = x + style_values["offset"]
            axis.vlines(positioned_x, group.ci_lower, group.ci_upper,
                        color=style_values["color"], lw=1.0, zorder=2)
            axis.plot(positioned_x, group.mean_PCC, color=style_values["color"],
                      linestyle=style_values["linestyle"], marker=style_values["marker"],
                      markersize=5.2, markeredgecolor="white", markeredgewidth=.5,
                      lw=1.05, zorder=3, label=SERIES_LABELS[series])
        labels = [rf"${a}\!:\!{b}$" if not no_tex else f"{a}:{b}"
                  for a, b in SUCCESSIVE_PAIRS]
        axis.set_xticks(x, labels)
        axis.set_xlim(-.35, len(SUCCESSIVE_PAIRS)-.65)
        lo, hi = float(summary.ci_lower.min()), float(summary.ci_upper.max())
        pad = max(.02, .12*(hi-lo))
        axis.set_ylim(lo-pad, hi+pad)
        axis.set_xlabel("Dataset counts compared")
        axis.set_ylabel("Mean full-CDS PCC")
        axis.set_title(r"\textbf{D}\quad Successive-size agreement" if not no_tex
                       else "D  Successive-size agreement", loc="left")
        axis.grid(axis="both")
        axis.set_axisbelow(True)
        axis.legend(loc="lower right", frameon=False, handlelength=1.8)
    return fig


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.output_dir / "real_data_ranked_cumulative_panel_d_source"
    source.mkdir(parents=True, exist_ok=True)
    (values, summary, difference, equal_pair_summary, provenance, weights,
     ranking, equal_provenance, bootstrap) = load_and_compare(
        args.run_root, args.equal_source_dir)
    values.to_csv(source / "panel_d_per_transcript.csv", index=False, float_format="%.17g")
    summary.to_csv(source / "panel_d_summary.csv", index=False, float_format="%.17g")
    difference.to_csv(source / "panel_d_descriptive_difference.csv", index=False, float_format="%.17g")
    equal_pair_summary.to_csv(source / "equal_model_pair_summary.csv", index=False,
                              float_format="%.17g")
    provenance.to_csv(source / "prediction_provenance.csv", index=False)
    weights.to_csv(source / "reference_weights.csv", index=False, float_format="%.17g")
    command = [sys.executable, str(Path(__file__).resolve()), "--run-root", str(args.run_root),
               "--equal-source-dir", str(args.equal_source_dir),
               "--output-dir", str(args.output_dir), "--reference-figure", str(args.reference_figure)]
    if args.no_tex:
        command.append("--no-tex")
    height = reference_height(args.reference_figure)
    # Plot from the round-tripped numeric source table.
    plotted = pd.read_csv(source / "panel_d_summary.csv", float_precision="round_trip")
    pdf = args.output_dir / "real_data_ranked_cumulative_panel_d.pdf"
    png = args.output_dir / "real_data_ranked_cumulative_panel_d.png"
    # Keep the publication context active through Matplotlib's deferred draw.
    style = dict(figure_one.FIGURE_RC)
    if args.no_tex:
        style.update({"text.usetex": False, "text.latex.preamble": "",
                      "font.serif": ["DejaVu Serif"], "mathtext.fontset": "cm"})
    with matplotlib.rc_context(style):
        fig = plot(plotted, height, args.no_tex)
        axis = fig.axes[0]
        for line, series in zip(axis.lines, SERIES_ORDER):
            group = plotted.loc[plotted.series.eq(series)].sort_values("comparison_index")
            np.testing.assert_allclose(line.get_ydata(), group.mean_PCC)
        for collection, series in zip(axis.collections, SERIES_ORDER):
            group = plotted.loc[plotted.series.eq(series)].sort_values("comparison_index")
            np.testing.assert_allclose(np.asarray(collection.get_segments())[:, :, 1],
                                       group[["ci_lower", "ci_upper"]])
        fig.savefig(pdf)
        fig.savefig(png, dpi=600)
    plt.close(fig)
    caption = ("\\textbf{D, Successive-size agreement under uniform and quality-ranked references.} "
        "Each point is a mean full-CDS transcript-level Pearson correlation over the same 1,771 held-out "
        "transcripts. Orange circles compare the single pair of independently fitted ten-component "
        "quality-ranked models in the nested cumulative top-quality chain at each indicated transition. "
        "Gray diamonds reproduce the uniform-reference Figure~1 sensitivity: the unweighted mean over all "
        "available representative cross-size model-pair means (36, 36, 36, 36, 18 and 3 pairs). Bars are "
        "95\\% percentile intervals from 5,000 joint transcript-bootstrap draws (seed 20260910), with each "
        "sampled transcript carrying both series, all model pairs and all six transitions. Profiles are native, "
        "unsmoothed mean-one outputs from best-validation-loss checkpoints. The uniform subsets are not nested "
        "whereas the ranked subsets are nested and selected by quality rank; consequently their difference "
        "confounds reference policy, membership, overlap and construction and is not an isolated weighting "
        "effect or a causal effect of adding data. Intervals are conditional on the fitted models and selected "
        "collections; shared transcripts and models make estimates dependent. PCC denotes reproducibility, "
        "not biological accuracy.\n")
    (args.output_dir / "real_data_ranked_cumulative_panel_d_caption.tex").write_text(caption)
    (source / "regenerate.sh").write_text("#!/bin/bash\nset -euo pipefail\n" + shlex.join(command) + "\n")
    manifest = dict(status="complete", created_at_utc=datetime.now(timezone.utc).isoformat(),
        script=str(Path(__file__).resolve()), script_sha256=sha256(Path(__file__).resolve()),
        run_root=str(args.run_root), equal_source_directory=str(args.equal_source_dir),
        experiment_design="descriptive overlay of two distinct subset designs",
        interpretation="successive-size agreement; displayed series difference is NOT a weighting-only effect",
        ranking=ranking, equal_provenance=equal_provenance, bootstrap=bootstrap,
        model_sizes=list(MODEL_SIZES),
        successive_pairs=[list(pair) for pair in SUCCESSIVE_PAIRS],
        training_seed=TRAINING_SEED, profile_domain="complete valid CDS", smoothing=False,
        posthoc_rescaling=False, output_pdf=str(pdf), output_pdf_sha256=sha256(pdf),
        output_png=str(png), output_png_sha256=sha256(png), png_dpi=600,
        figure_height_inches=height, regeneration_command=shlex.join(command),
        source_sha256={p.name: sha256(p) for p in sorted(source.glob("*.csv"))},
        plotted_values_verified_against_saved_summary=True)
    (source / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    print(summary.to_string(index=False))
    print(f"Wrote {pdf}\nWrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
