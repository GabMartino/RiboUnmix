#!/usr/bin/env python3
"""Create Figure 2 panel C from matched four-panel PCC records.

The numerical input is the per-transcript table produced from the saved
full-CDS shared-profile arrays by ``compare_real_panel_weighting.py``.  The
script verifies that its equal-policy records reproduce Figure 1A exactly,
then estimates the ranked-minus-equal difference of medians for each of the
six panel pairs with a joint paired transcript bootstrap.

The historical ranked run used the six-component quality rank.  It is
accepted only with ``--allow-legacy-six-component`` and is exported with a
``_provisional`` suffix.  A ten-component run produces the final stem.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import shlex
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Utils.publication_plot_style import publication_rc
from Utils.reliability_references import transcript_id_hash
from analyses import create_real_data_equal_figure as figure_one
from analyses.compare_real_panel_weighting import audit_design
from run_real_exp8_L_stability_quality_rank import inspect_ranking_components


PANEL_NAMES = tuple(f"panel_{index:02d}" for index in range(1, 5))
INTERNAL_PAIRS = tuple(
    f"{left}__{right}" for left, right in itertools.combinations(PANEL_NAMES, 2)
)
PAIR_LABELS = tuple(
    f"P{left}\N{EN DASH}P{right}" for left, right in itertools.combinations(range(1, 5), 2)
)
PAIR_LABEL_BY_INTERNAL = dict(zip(INTERNAL_PAIRS, PAIR_LABELS, strict=True))

DEFAULT_COMPARISON = ROOT / "analyses/artifacts/real_data/panels_equal_vs_ranked"
DEFAULT_EQUAL_ROOT = ROOT / "results/my_panels_a100_b32_20260906_114323"
DEFAULT_RANKED_ROOT = ROOT / "results/my_panels_qrank_a100_b32_20260908_103510"
DEFAULT_FIGURE_ONE_SOURCE = ROOT / "figures/real_data_equal_source/panel_a_per_transcript.csv"
DEFAULT_LEGACY_RANKING = ROOT / "Datasets/data/HEK_riboseq_profile_quality_rank.tsv"
EXPECTED_LEGACY_SHA256 = "07f440ca13c9193f3814d8f529c1e50d8a09ec19fbc1c4a7aa91ecae860be125"
EXPECTED_TEN_COMPONENT_SHA256 = "5811cadf68c56740e83b232b2990299d527630326205db9cba8bc7024d3cf1f8"
BOOTSTRAP_DRAWS = 5_000
BOOTSTRAP_SEED = 20_260_910
ORANGE = "#D55E00"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--equal-root", type=Path, default=DEFAULT_EQUAL_ROOT)
    parser.add_argument("--ranked-root", type=Path, default=DEFAULT_RANKED_ROOT)
    parser.add_argument("--ranking-table", type=Path, default=DEFAULT_LEGACY_RANKING)
    parser.add_argument("--figure-one-source", type=Path, default=DEFAULT_FIGURE_ONE_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--allow-legacy-six-component", action="store_true")
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args(argv)
    for name in (
        "comparison_root",
        "equal_root",
        "ranked_root",
        "ranking_table",
        "figure_one_source",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.bootstrap_draws < 1:
        parser.error("--bootstrap-draws must be positive.")
    return args


def verify_ranking(
    ranked_root: Path,
    ranking_table: Path,
    allow_legacy: bool,
) -> dict:
    strategy = read_json(ranked_root / "panel_manifest.json").get("gamma_reference_strategy")
    if not isinstance(strategy, dict):
        raise ValueError("Ranked panel manifest has no gamma-reference strategy.")
    digest = sha256(ranking_table)
    if digest != strategy.get("ranking_table_sha256"):
        raise ValueError("The supplied ranking table does not match the trained ranked models.")
    component_info = inspect_ranking_components(ranking_table)
    component_count = int(component_info["count"])
    expected_digest = {
        6: EXPECTED_LEGACY_SHA256,
        10: EXPECTED_TEN_COMPONENT_SHA256,
    }.get(component_count)
    if expected_digest is None or digest != expected_digest:
        raise ValueError(
            f"Unsupported ranking definition: {component_count} components, SHA-256 {digest}."
        )
    if component_count == 6 and not allow_legacy:
        raise ValueError(
            "The matched historical models use the six-component ranking. Pass "
            "--allow-legacy-six-component for a clearly provisional figure, or provide "
            "the completed matched ten-component run."
        )

    table = pd.read_csv(ranking_table, sep="\t")
    ranks = pd.to_numeric(table["quality_rank"], errors="raise")
    if table["dataset"].duplicated().any() or not np.isfinite(ranks).all():
        raise ValueError("The frozen global ranking has duplicate IDs or non-finite ranks.")
    rank_universe = float(ranks.max())
    if rank_universe != 115 or len(table) != 115:
        raise ValueError("Expected the complete 115-row global ranking with R=115.")
    return {
        "path": str(ranking_table),
        "sha256": digest,
        "component_count": component_count,
        "components": component_info["columns"],
        "rows": len(table),
        "R": rank_universe,
        "direction": "1 = best",
        "formula": "q_d=(R-r_d+1)/R; pi_d=q_d/sum_selected(q)",
        "provisional": component_count != 10,
    }


def load_matched_records(
    comparison_root: Path,
    equal_root: Path,
    ranked_root: Path,
    figure_one_source: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest = read_json(comparison_root / "comparison_manifest.json")
    if not manifest.get("complete") or manifest.get("matched_panels") != list(PANEL_NAMES):
        raise ValueError("The saved comparison does not contain all four matched panels.")
    if Path(manifest["equal_root"]).resolve() != equal_root:
        raise ValueError("Comparison table was not generated from the requested equal root.")
    if Path(manifest["ranked_root"]).resolve() != ranked_root:
        raise ValueError("Comparison table was not generated from the requested ranked root.")

    path = comparison_root / "matched_inter_panel_metrics.csv"
    records = pd.read_csv(path, float_precision="round_trip")
    required = {
        "policy",
        "region",
        "pair",
        "transcript_id",
        "PCC",
        "valid_PCC",
        "reason",
    }
    if not required <= set(records.columns):
        raise ValueError(f"Matched table lacks columns: {sorted(required - set(records.columns))}")
    records = records.loc[
        records.region.eq("full_cds") & records.policy.isin(("equal", "ranked")),
        list(required),
    ].copy()
    if set(records.policy) != {"equal", "ranked"} or set(records.pair) != set(INTERNAL_PAIRS):
        raise ValueError("Expected both policies and exactly the six canonical panel pairs.")
    if records.duplicated(["policy", "pair", "transcript_id"]).any():
        raise ValueError("Duplicate policy/pair/transcript records.")

    expected_columns = pd.MultiIndex.from_product(
        [("equal", "ranked"), INTERNAL_PAIRS], names=("policy", "pair")
    )
    pivot = records.pivot(
        index="transcript_id", columns=("policy", "pair"), values="PCC"
    ).reindex(columns=expected_columns).sort_index()
    valid = np.isfinite(pivot.to_numpy(dtype=float)).all(axis=1)
    included_ids = pivot.index[valid]
    if len(included_ids) == 0:
        raise ValueError("No transcript is finite under both policies for all six pairs.")

    exclusions = pd.DataFrame(
        {
            "transcript_id": pivot.index[~valid],
            "included": False,
            "exclusion_reason": "undefined_PCC_in_at_least_one_policy_or_panel_pair",
        }
    )
    values = pivot.loc[included_ids]
    long = []
    for pair, label in zip(INTERNAL_PAIRS, PAIR_LABELS, strict=True):
        equal = values[("equal", pair)].to_numpy(float)
        ranked = values[("ranked", pair)].to_numpy(float)
        long.append(
            pd.DataFrame(
                {
                    "transcript_id": included_ids,
                    "comparison_id": pair,
                    "panel_pair": label,
                    "PCC_equal": equal,
                    "PCC_ranked": ranked,
                    "paired_PCC_difference": ranked - equal,
                    "included": True,
                }
            )
        )
    paired = pd.concat(long, ignore_index=True)

    # Figure 1A is the authoritative displayed equal-policy source.  Require
    # row-wise numerical identity, rather than comparing only its six medians.
    figure_one = pd.read_csv(figure_one_source, float_precision="round_trip")
    figure_one = figure_one.loc[
        figure_one.profile_domain.eq("full_CDS") & figure_one.status.eq("valid"),
        ["transcript_id", "panel_pair", "PCC"],
    ].rename(columns={"PCC": "PCC_figure_one"})
    check = paired.merge(
        figure_one,
        on=["transcript_id", "panel_pair"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if check._merge.ne("both").any():
        raise ValueError("The matched equal records and Figure 1A have different rows.")
    np.testing.assert_allclose(
        check.PCC_equal,
        check.PCC_figure_one,
        rtol=0,
        atol=5e-15,
        err_msg="Equal-policy values do not reproduce Figure 1A.",
    )
    source = {
        "matched_metrics": str(path),
        "matched_metrics_sha256": sha256(path),
        "comparison_manifest": str(comparison_root / "comparison_manifest.json"),
        "comparison_manifest_sha256": sha256(comparison_root / "comparison_manifest.json"),
        "figure_one_panel_a_source": str(figure_one_source),
        "figure_one_panel_a_source_sha256": sha256(figure_one_source),
        "equal_records_reproduce_figure_one_A": True,
        "matched_transcript_count": len(included_ids),
        "matched_transcript_hash": transcript_id_hash(included_ids),
        "excluded_transcript_count": int((~valid).sum()),
    }
    return paired, exclusions, source


def joint_paired_bootstrap(
    paired: pd.DataFrame,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, str]:
    equal = paired.pivot(
        index="transcript_id", columns="comparison_id", values="PCC_equal"
    ).reindex(columns=INTERNAL_PAIRS).sort_index()
    ranked = paired.pivot(
        index="transcript_id", columns="comparison_id", values="PCC_ranked"
    ).reindex(index=equal.index, columns=INTERNAL_PAIRS)
    a, b = equal.to_numpy(float), ranked.to_numpy(float)
    if a.shape != b.shape or a.shape[1] != 6 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Bootstrap requires a complete transcript-by-six-pair matrix.")

    point_equal = np.median(a, axis=0)
    point_ranked = np.median(b, axis=0)
    rng = np.random.default_rng(seed)
    samples_equal = np.empty((draws, 6), dtype=float)
    samples_ranked = np.empty((draws, 6), dtype=float)
    stream_digest = hashlib.sha256()
    for start in range(0, draws, 32):
        size = min(32, draws - start)
        indices = rng.integers(0, len(a), size=(size, len(a)))
        stream_digest.update(indices.astype("<i8").tobytes())
        samples_equal[start : start + size] = np.median(a[indices], axis=1)
        samples_ranked[start : start + size] = np.median(b[indices], axis=1)
    samples_delta = samples_ranked - samples_equal
    equal_ci = np.quantile(samples_equal, (0.025, 0.975), axis=0, method="linear")
    ranked_ci = np.quantile(samples_ranked, (0.025, 0.975), axis=0, method="linear")
    delta_ci = np.quantile(samples_delta, (0.025, 0.975), axis=0, method="linear")

    rows = []
    for index, (pair, label) in enumerate(zip(INTERNAL_PAIRS, PAIR_LABELS, strict=True)):
        rows.append(
            {
                "comparison_id": pair,
                "panel_pair": label,
                "training_seed": 42,
                "n_transcripts": len(a),
                "equal_median_PCC": point_equal[index],
                "equal_ci_lower": equal_ci[0, index],
                "equal_ci_upper": equal_ci[1, index],
                "ranked_median_PCC": point_ranked[index],
                "ranked_ci_lower": ranked_ci[0, index],
                "ranked_ci_upper": ranked_ci[1, index],
                "estimate": point_ranked[index] - point_equal[index],
                "ci_lower": delta_ci[0, index],
                "ci_upper": delta_ci[1, index],
                "statistic": "difference_of_medians",
                "bootstrap_draws": draws,
                "bootstrap_seed": seed,
            }
        )
    return pd.DataFrame(rows), stream_digest.hexdigest()


def draw_panel(summary: pd.DataFrame, output_stem: Path, provisional: bool, dpi: int) -> None:
    ordered = summary.set_index("comparison_id").loc[list(INTERNAL_PAIRS)].reset_index()
    plotted = ordered[["estimate", "ci_lower", "ci_upper"]].to_numpy(float)
    if not np.isfinite(plotted).all():
        raise ValueError("Every plotted estimate and interval endpoint must be finite.")
    low = min(0.0, float(ordered.ci_lower.min()))
    high = max(0.0, float(ordered.ci_upper.max()))
    padding = max(0.006, 0.11 * (high - low))

    # Match Figure 1's typography exactly while retaining the shared
    # compute-node fallback when an external TeX installation is unavailable.
    style = dict(figure_one.FIGURE_RC)
    rendering = publication_rc()
    if not rendering["text.usetex"]:
        style.update(
            {
                "text.usetex": False,
                "text.latex.preamble": "",
                "font.serif": ["DejaVu Serif"],
                "mathtext.fontset": "cm",
            }
        )
    with matplotlib.rc_context(style):
        figure, axis = plt.subplots(figsize=(3.55, 2.65), layout="constrained")
        y = np.arange(5, -1, -1)
        axis.axvline(0, color="#777777", linestyle="--", linewidth=0.8, zorder=0)
        axis.hlines(y, ordered.ci_lower, ordered.ci_upper, color=ORANGE, linewidth=1.15, zorder=2)
        axis.scatter(
            ordered.estimate,
            y,
            s=31,
            color=ORANGE,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        axis.set_xlim(low - padding, high + padding)
        axis.set_ylim(-0.65, 5.65)
        axis.set_yticks(y, [label.replace("\N{EN DASH}", "--") for label in PAIR_LABELS])
        axis.set_xlabel("Change in median PCC (ranked $-$ equal)")
        title = (
            r"\textbf{C}\quad Ranking effect on reproducibility"
            if style["text.usetex"]
            else "C  Ranking effect on reproducibility"
        )
        axis.set_title(title, loc="left")
        axis.grid(axis="x")
        axis.set_axisbelow(True)
        if provisional:
            axis.text(
                0.995,
                0.015,
                "Provisional: legacy 6-component rank",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.5,
                color="#666666",
            )
        figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
        figure.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
        plt.close(figure)


def write_caption(path: Path, provisional: bool) -> None:
    ranking = (
        "the historical six-component quality ranking (provisional layout; to be replaced by "
        "the frozen ten-component rerun)"
        if provisional
        else "the frozen ten-component quality ranking"
    )
    path.write_text(
        rf"""\textbf{{C, Effect of ranked reference weights on cross-panel reproducibility.}}
For each of the six pairs formed by four source-family-disjoint panels, the point is
the difference between the median full-CDS transcript-level PCC obtained with
quality-ranked and uniform gamma-reference weights. Ranked models use {ranking}.
The compared models have identical panel membership, transcript splits, training seed,
architecture, local reliability weights $w_{{dt}}$, and checkpoint-selection rule; the
selected checkpoints may differ. Bars are 95\% paired transcript-bootstrap intervals
(5,000 resamples; seed 20260910), with each sampled transcript carrying both policies
and all six panel pairs. The statistic is a difference of medians, not a median of
within-transcript differences. Intervals are conditional on the fitted models and
selected panels. Positive values indicate greater agreement, not demonstrated
biological accuracy.
""",
        encoding="utf-8",
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranking = verify_ranking(
        args.ranked_root, args.ranking_table, args.allow_legacy_six_component
    )

    # Reuse the production comparison audit rather than defining another
    # membership, split, reliability-weight, or pi_d matching rule here.
    ids, panels, weights, differences, design_audit = audit_design(
        args.equal_root, args.ranked_root, args.ranking_table
    )
    if panels != list(PANEL_NAMES):
        raise ValueError("The comparison does not contain the canonical four panels.")
    paired, exclusions, source = load_matched_records(
        args.comparison_root,
        args.equal_root,
        args.ranked_root,
        args.figure_one_source,
    )
    if set(paired.transcript_id) != set(ids):
        raise ValueError("Saved per-transcript records differ from the audited common test set.")

    summary, bootstrap_digest = joint_paired_bootstrap(
        paired, args.bootstrap_draws, args.bootstrap_seed
    )
    suffix = "_provisional" if ranking["provisional"] else ""
    stem = args.output_dir / f"real_data_ranking_effect_panel_c{suffix}"
    source_dir = args.output_dir / f"real_data_ranking_effect_panel_c{suffix}_source"
    source_dir.mkdir(parents=True, exist_ok=True)

    paired.to_csv(source_dir / "panel_c_paired_per_transcript.csv", index=False, float_format="%.17g")
    exclusions.to_csv(source_dir / "panel_c_exclusions.csv", index=False)
    summary.to_csv(source_dir / "panel_c_summary.csv", index=False, float_format="%.17g")
    weights.to_csv(source_dir / "reference_weights.csv", index=False, float_format="%.17g")
    differences.to_csv(source_dir / "configuration_differences.csv", index=False)
    draw_panel(summary, stem, ranking["provisional"], args.dpi)
    write_caption(stem.with_name(stem.name + "_caption.tex"), ranking["provisional"])

    command = shlex.join(
        [
            str(Path(sys.executable).resolve()),
            str(Path(__file__).resolve()),
            "--comparison-root",
            str(args.comparison_root),
            "--equal-root",
            str(args.equal_root),
            "--ranked-root",
            str(args.ranked_root),
            "--ranking-table",
            str(args.ranking_table),
            "--figure-one-source",
            str(args.figure_one_source),
            "--output-dir",
            str(args.output_dir),
            "--bootstrap-draws",
            str(args.bootstrap_draws),
            "--bootstrap-seed",
            str(args.bootstrap_seed),
            "--dpi",
            str(args.dpi),
        ]
        + (["--allow-legacy-six-component"] if args.allow_legacy_six_component else [])
    )
    regenerate = source_dir / "regenerate.sh"
    regenerate.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n")
    regenerate.chmod(0o755)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "provisional_legacy_six_component" if ranking["provisional"] else "final_ten_component",
        "figure_pdf": str(stem.with_suffix(".pdf")),
        "figure_png": str(stem.with_suffix(".png")),
        "ranking": ranking,
        "source": source,
        "equal_root": str(args.equal_root),
        "ranked_root": str(args.ranked_root),
        "panel_memberships_match": True,
        "common_split_and_training_seed_match": True,
        "local_reliability_weights_match": True,
        "design_audit": design_audit,
        "statistic": "median(PCC_ranked) - median(PCC_equal), separately per panel pair",
        "bootstrap": {
            "draws": args.bootstrap_draws,
            "seed": args.bootstrap_seed,
            "unit": "transcript carrying both policies and all six panel pairs",
            "index_stream_sha256": bootstrap_digest,
            "interval": "2.5th and 97.5th percentiles using NumPy linear quantiles",
            "conditional_on_fitted_models": True,
        },
        "profile_domain": "complete valid CDS; no smoothing, truncation, or zero filtering",
        "interpretation": "PCC agreement/reproducibility, not biological accuracy",
        "command": command,
        "script_sha256": sha256(Path(__file__)),
    }
    manifest_path = source_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    # The table written to disk is the sole plotting source: verify its values
    # before handing off the exports.
    saved = pd.read_csv(source_dir / "panel_c_summary.csv", float_precision="round_trip")
    np.testing.assert_allclose(
        saved[["estimate", "ci_lower", "ci_upper"]],
        summary[["estimate", "ci_lower", "ci_upper"]],
        rtol=0,
        atol=0,
    )
    print(summary[["panel_pair", "equal_median_PCC", "ranked_median_PCC", "estimate", "ci_lower", "ci_upper", "n_transcripts"]].to_string(index=False))
    print(f"Wrote {stem.with_suffix('.pdf')}")
    print(f"Wrote {stem.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
