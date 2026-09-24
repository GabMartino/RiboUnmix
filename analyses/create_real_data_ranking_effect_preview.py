#!/usr/bin/env python3
"""Assemble the available ranking-effect panels C and D for layout review.

Panel C is the matched four-panel ranked-minus-equal comparison.  Panel D is
the descriptive overlay of uniform representative and ten-component ranked
cumulative successive-size agreement.  The script plots only the audited
numeric source tables produced by their dedicated analysis scripts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
from analyses import create_real_data_equal_figure as figure_one
from analyses.create_real_data_ranked_cumulative_panel_d import (
    GRAY,
    ORANGE,
    SERIES_LABELS,
    SERIES_ORDER,
    SUCCESSIVE_PAIRS,
)
from analyses.create_real_data_ranked_panel_c import INTERNAL_PAIRS, PAIR_LABELS


DEFAULT_C_SOURCE = ROOT / "figures/real_data_ranking_effect_panel_c_provisional_source"
DEFAULT_D_SOURCE = ROOT / "figures/real_data_ranked_cumulative_panel_d_source"


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
    parser.add_argument("--panel-c-source", type=Path, default=DEFAULT_C_SOURCE)
    parser.add_argument("--panel-d-source", type=Path, default=DEFAULT_D_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument(
        "--output-stem",
        help="Filename stem. By default, provenance determines whether '_provisional' is used.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args(argv)
    for name in ("panel_c_source", "panel_d_source", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def load_sources(c_source: Path, d_source: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    c_path, d_path = c_source / "panel_c_summary.csv", d_source / "panel_d_summary.csv"
    c_manifest_path, d_manifest_path = c_source / "figure_manifest.json", d_source / "figure_manifest.json"
    for path in (c_path, d_path, c_manifest_path, d_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    c, d = (pd.read_csv(path, float_precision="round_trip") for path in (c_path, d_path))
    c_manifest, d_manifest = read_json(c_manifest_path), read_json(d_manifest_path)

    required_c = {"comparison_id", "estimate", "ci_lower", "ci_upper", "n_transcripts"}
    required_d = {
        "series",
        "comparison_index",
        "mean_PCC",
        "ci_lower",
        "ci_upper",
        "n_transcripts",
    }
    if not required_c <= set(c) or not required_d <= set(d):
        raise ValueError("A panel source table lacks required plotting columns.")
    if set(c.comparison_id) != set(INTERNAL_PAIRS) or len(c) != 6:
        raise ValueError("Panel C must contain exactly the six canonical panel pairs.")
    expected_d = {(series, index) for series in SERIES_ORDER for index in range(6)}
    observed_d = set(zip(d.series, d.comparison_index, strict=False))
    if observed_d != expected_d or len(d) != 12:
        raise ValueError("Panel D must contain both series at all six successive-size transitions.")
    if not np.isfinite(c[["estimate", "ci_lower", "ci_upper"]]).all().all():
        raise ValueError("Panel C contains a non-finite plotted value.")
    if not np.isfinite(d[["mean_PCC", "ci_lower", "ci_upper"]]).all().all():
        raise ValueError("Panel D contains a non-finite plotted value.")
    if c_manifest.get("source", {}).get("equal_records_reproduce_figure_one_A") is not True:
        raise ValueError("Panel C has not verified its equal records against Figure 1A.")
    if d_manifest.get("status") != "complete":
        raise ValueError("Panel D source analysis is not complete.")
    if d_manifest.get("ranking", {}).get("component_count") != 10:
        raise ValueError("Panel D is not the audited ten-component cumulative analysis.")
    return c, d, c_manifest, d_manifest


def plotting_style() -> dict:
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
    return style


def draw_panel_c(axis, summary: pd.DataFrame, use_tex: bool, provisional: bool) -> None:
    ordered = summary.set_index("comparison_id").loc[list(INTERNAL_PAIRS)].reset_index()
    y = np.arange(5, -1, -1)
    low = min(0.0, float(ordered.ci_lower.min()))
    high = max(0.0, float(ordered.ci_upper.max()))
    padding = max(0.006, 0.11 * (high - low))
    axis.axvline(0, color="#777777", linestyle="--", linewidth=0.8, zorder=0)
    axis.hlines(y, ordered.ci_lower, ordered.ci_upper, color=ORANGE, linewidth=1.1, zorder=2)
    axis.scatter(
        ordered.estimate,
        y,
        s=28,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    axis.set_xlim(low - padding, high + padding)
    axis.set_ylim(-0.65, 5.65)
    axis.set_yticks(y, [label.replace("\N{EN DASH}", "--") for label in PAIR_LABELS])
    axis.set_xlabel("Change in median PCC\n(ranked $-$ equal)")
    title = (
        r"\textbf{C}\quad Ranking effect on reproducibility"
        if use_tex
        else "C  Ranking effect on reproducibility"
    )
    axis.set_title(title, loc="left")
    axis.grid(axis="x")
    axis.set_axisbelow(True)
    if provisional:
        axis.text(
            0.99,
            0.015,
            "Legacy 6-component ranked run",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.8,
            color="#6F6F6F",
        )


def draw_panel_d(axis, summary: pd.DataFrame, use_tex: bool) -> None:
    x = np.arange(len(SUCCESSIVE_PAIRS), dtype=float)
    styles = {
        "equal_representative": {"color": GRAY, "linestyle": "--", "marker": "D", "offset": -0.035},
        "ranked_cumulative": {"color": ORANGE, "linestyle": "-", "marker": "o", "offset": 0.035},
    }
    for series in SERIES_ORDER:
        group = summary.loc[summary.series.eq(series)].sort_values("comparison_index")
        values = styles[series]
        positioned_x = x + values["offset"]
        axis.vlines(positioned_x, group.ci_lower, group.ci_upper, color=values["color"], lw=1.0, zorder=2)
        axis.plot(
            positioned_x,
            group.mean_PCC,
            color=values["color"],
            linestyle=values["linestyle"],
            marker=values["marker"],
            markersize=5.0,
            markeredgecolor="white",
            markeredgewidth=0.5,
            lw=1.0,
            zorder=3,
            label=SERIES_LABELS[series],
        )
    labels = [rf"${a}\!:\!{b}$" if use_tex else f"{a}:{b}" for a, b in SUCCESSIVE_PAIRS]
    axis.set_xticks(x, labels)
    axis.set_xlim(-0.35, len(SUCCESSIVE_PAIRS) - 0.65)
    low, high = float(summary.ci_lower.min()), float(summary.ci_upper.max())
    padding = max(0.02, 0.12 * (high - low))
    axis.set_ylim(low - padding, high + padding)
    axis.set_xlabel("Dataset counts compared")
    axis.set_ylabel("Mean full-CDS PCC")
    title = (
        r"\textbf{D}\quad Successive-size agreement"
        if use_tex
        else "D  Successive-size agreement"
    )
    axis.set_title(title, loc="left")
    axis.grid(axis="both")
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", frameon=False, handlelength=1.7)


def caption(c_provisional: bool) -> str:
    c_rank = (
        "the available historical six-component ranked runs (shown as a provisional layout pending the "
        "matched ten-component rerun)"
        if c_provisional
        else "the frozen ten-component ranked runs"
    )
    return (
        "\\textbf{Effect of ranked gamma-reference weights on shared-profile agreement.} "
        f"\\textbf{{C}}, for each pair among four source-family-disjoint panels, {c_rank} are compared "
        "with the exact uniform-reference models underlying Fig.~1A. Each point is the ranked minus "
        "uniform difference between median full-CDS transcript-level PCCs; bars are 95\\% paired "
        "transcript-bootstrap intervals. The models within each Panel-C contrast have matching dataset "
        "membership, transcript splits, training seed, architecture, local reliability weights $w_{dt}$, "
        "and checkpoint-selection rule. \\textbf{D}, orange circles show successive-size agreement for the "
        "nested cumulative top-quality models trained with the frozen ten-component ranking; gray diamonds "
        "reproduce the uniform-reference representative-subset sensitivity. Each value is the mean "
        "full-CDS transcript PCC. The latter two series differ in membership, nesting and reference policy, "
        "so their separation is descriptive and is not an isolated weighting effect. Intervals use 5,000 "
        "joint transcript resamples (seed 20260910) and are conditional on the fitted models and selected "
        "collections. PCC measures reproducibility, not biological accuracy.\n"
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    c, d, c_manifest, d_manifest = load_sources(args.panel_c_source, args.panel_d_source)
    provisional = c_manifest.get("ranking", {}).get("component_count") != 10
    output_stem = args.output_stem or (
        "real_data_ranking_effect_provisional" if provisional else "real_data_ranking_effect"
    )
    stem = args.output_dir / output_stem
    source_dir = args.output_dir / f"{output_stem}_source"
    source_dir.mkdir(parents=True, exist_ok=True)

    # Round-trip the exact plotting sources into the assembled figure directory.
    c.to_csv(source_dir / "panel_c_summary.csv", index=False, float_format="%.17g")
    d.to_csv(source_dir / "panel_d_summary.csv", index=False, float_format="%.17g")
    plotted_c = pd.read_csv(source_dir / "panel_c_summary.csv", float_precision="round_trip")
    plotted_d = pd.read_csv(source_dir / "panel_d_summary.csv", float_precision="round_trip")

    style = plotting_style()
    with matplotlib.rc_context(style):
        figure, (axis_c, axis_d) = plt.subplots(
            1,
            2,
            figsize=(7.15, 2.65),
            gridspec_kw={"width_ratios": (1.0, 1.08)},
            layout="constrained",
        )
        draw_panel_c(axis_c, plotted_c, style["text.usetex"], provisional)
        draw_panel_d(axis_d, plotted_d, style["text.usetex"])
        pdf, png = stem.with_suffix(".pdf"), stem.with_suffix(".png")
        figure.savefig(pdf, bbox_inches="tight")
        figure.savefig(png, dpi=args.dpi, bbox_inches="tight")
        plt.close(figure)

    caption_path = stem.with_name(stem.name + "_caption.tex")
    caption_path.write_text(caption(provisional), encoding="utf-8")
    command = shlex.join(
        [
            str(Path(sys.executable).resolve()),
            str(Path(__file__).resolve()),
            "--panel-c-source",
            str(args.panel_c_source),
            "--panel-d-source",
            str(args.panel_d_source),
            "--output-dir",
            str(args.output_dir),
            "--output-stem",
            output_stem,
            "--dpi",
            str(args.dpi),
        ]
    )
    regenerate = source_dir / "regenerate.sh"
    regenerate.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n")
    regenerate.chmod(0o755)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "provisional_mixed_ranking_definitions" if provisional else "complete",
        "output_pdf": str(pdf),
        "output_pdf_sha256": sha256(pdf),
        "output_png": str(png),
        "output_png_sha256": sha256(png),
        "png_dpi": args.dpi,
        "figure_size_inches_before_tight_bbox": [7.15, 2.65],
        "typography_source": str(Path(figure_one.__file__).resolve()),
        "panel_c_input": str(args.panel_c_source),
        "panel_c_input_manifest_sha256": sha256(args.panel_c_source / "figure_manifest.json"),
        "panel_c_ranking": c_manifest.get("ranking"),
        "panel_d_input": str(args.panel_d_source),
        "panel_d_input_manifest_sha256": sha256(args.panel_d_source / "figure_manifest.json"),
        "panel_d_ranking": d_manifest.get("ranking"),
        "interpretation": {
            "C": "matched reference-policy contrast",
            "D": "descriptive overlay confounded by different subset constructions",
            "PCC": "agreement/reproducibility, not biological accuracy",
        },
        "source_table_sha256": {
            "panel_c_summary.csv": sha256(source_dir / "panel_c_summary.csv"),
            "panel_d_summary.csv": sha256(source_dir / "panel_d_summary.csv"),
        },
        "regeneration_command": command,
        "script_sha256": sha256(Path(__file__)),
    }
    (source_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
