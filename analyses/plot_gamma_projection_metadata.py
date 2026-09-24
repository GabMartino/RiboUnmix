#!/usr/bin/env python3
"""Recolor the saved gamma projection; never infer, refit, or move its points."""
from __future__ import annotations

import argparse
import html
import json
import shlex
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analyses.analyze_cumulative_gamma_geometry import label_projection, sha256
from Utils.publication_plot_style import publication_rc

UNKNOWN = "Not reported / unclear"
KIT_COLORS = {
    "TruSeq Ribo Profile": "#0072B2",
    "NEXTflex small RNA": "#D55E00",
    "SMARTer smRNA-Seq": "#009E73",
    "QIAseq miRNA": "#CC79A7",
    "D-Plex Small RNA": "#B29200",
    UNKNOWN: "#ADB5BD",
}
PLATFORM_COLORS = {
    "HiSeq 2000": "#0072B2", "HiSeq 2500": "#56B4E9",
    "HiSeq 3000": "#009E73", "HiSeq 4000": "#D55E00",
    "NextSeq 500": "#CC79A7", "NovaSeq 6000": "#B29200",
    "NovaSeq X": "#6A51A3", "NovaSeq X Plus": "#503B28",
    "Mixed platforms": "#273744", UNKNOWN: "#ADB5BD",
}


def platform_label(value):
    """Keep exact instrument models; do not pick one from a mixed alias."""
    names = {item.strip().removeprefix("Illumina ") for item in str(value).split("|") if item.strip()}
    if len(names) > 1:
        return "Mixed platforms"
    return next(iter(names), UNKNOWN)


def attach_metadata(coordinates, datasets, kits):
    """One-to-one joins preserve the existing point order and coordinates."""
    merged = coordinates.merge(
        datasets[["dataset", "gse", "instruments", "gsms", "study_url"]],
        on=["dataset", "gse"], how="left", validate="one_to_one",
    ).merge(kits, on=["dataset", "gse"], how="left", validate="one_to_one")
    if merged[["instruments", "riboseq_library_kit"]].isna().any().any():
        raise ValueError("Some projected datasets lack reviewed metadata; annotate them before plotting.")
    np.testing.assert_array_equal(
        coordinates[["coordinate_1", "coordinate_2"]],
        merged[["coordinate_1", "coordinate_2"]],
    )
    np.testing.assert_array_equal(coordinates.dataset, merged.dataset)
    merged["sequencing_platform"] = merged.instruments.map(platform_label)
    for field, palette in (("riboseq_library_kit", KIT_COLORS), ("sequencing_platform", PLATFORM_COLORS)):
        if not set(merged[field]) <= set(palette):
            raise ValueError(f"Define display colors for new {field} values.")
    return merged


def plot_overlay(table, provenance, out, font_size):
    style = publication_rc()
    style.update({"font.size": font_size, "axes.labelsize": font_size,
                  "axes.titlesize": font_size + 2, "font.weight": "bold",
                  "axes.labelweight": "bold", "axes.titleweight": "bold"})
    if style["text.usetex"]:
        style["text.latex.preamble"] += r"\renewcommand{\seriesdefault}{\bfdefault}\AtBeginDocument{\boldmath}"
    escape = lambda text: text.replace("%", r"\%") if style["text.usetex"] else text
    xy = table[["coordinate_1", "coordinate_2"]].to_numpy()
    diag = provenance["projection"]
    fields = [("riboseq_library_kit", KIT_COLORS, "A  Ribo-seq library kit"),
              ("sequencing_platform", PLATFORM_COLORS, "B  Sequencing platform")]
    with matplotlib.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(18, 10))
        fig.subplots_adjust(left=.065, right=.985, bottom=.27, top=.87, wspace=.22)
        for ax, (field, colors, title) in zip(axes, fields):
            for category, color in colors.items():
                selected = table[field].eq(category).to_numpy()
                if selected.any():
                    ax.scatter(xy[selected, 0], xy[selected, 1], s=110, color=color,
                               edgecolor="white", linewidth=.7,
                               label=f"{category} (n={selected.sum()})", zorder=3)
            ax.set(xlabel=escape(f"Coordinate 1 ({diag['axis1_fraction']:.1%})"),
                   ylabel=escape(f"Coordinate 2 ({diag['axis2_fraction']:.1%})"), title=title)
            ax.set_aspect("equal", adjustable="box")
            ax.margins(.19)
            ax.grid(alpha=.25)
            ax.legend(loc="upper center", bbox_to_anchor=(.5, -.17),
                      ncol=2, fontsize=font_size * .85, handletextpad=.4, columnspacing=1.0)
        fig.suptitle(escape(
            f"Same gamma projection, different metadata colors | {len(table)} datasets, "
            f"{provenance['selected_transcripts']} transcripts\n"
            f"Fixed coordinates; two dimensions retain {diag['fraction_2d']:.1%} of variation"),
            fontsize=font_size + 2, y=.97)
        fig.canvas.draw()
        fig.set_layout_engine("none")
        for ax in axes:
            label_projection(ax, xy, fontsize=font_size * .85)
        fig.text(.5, .02, "Numbers identify the same datasets as in the original projection. "
                 "Gray kit labels mean unreported or unclear, not the absence of a kit.",
                 ha="center", fontsize=font_size * .8)
        for extension in ("pdf", "png", "svg"):
            fig.savefig(out / f"gamma_projection_metadata.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "analyses/artifacts/real_data/cumulative_stability/gamma_geometry/equal_N040")
    parser.add_argument("--metadata-dir", type=Path, default=ROOT / "analyses/artifacts/real_data/hek293_metadata")
    parser.add_argument("--font-size", type=float, default=14)
    args = parser.parse_args()
    geometry = args.analysis_dir
    out = geometry / "metadata_overlay"
    out.mkdir(parents=True, exist_ok=True)
    source_files = [geometry / "projection_coordinates.csv", geometry / "provenance.json",
                    args.metadata_dir / "dataset_metadata.tsv",
                    args.metadata_dir / "riboseq_library_kits.tsv"]
    coordinates = pd.read_csv(source_files[0])
    provenance = json.loads(source_files[1].read_text())
    table = attach_metadata(coordinates, pd.read_csv(source_files[2], sep="\t"),
                            pd.read_csv(source_files[3], sep="\t").fillna(""))
    # The existing annotation helper numbers points 1..N.
    np.testing.assert_array_equal(table.number, np.arange(1, len(table) + 1))
    table.to_csv(out / "projection_with_metadata.csv", index=False)
    counts = []
    for field in ("riboseq_library_kit", "sequencing_platform"):
        for category, group in table.groupby(field, sort=False):
            counts.append(dict(field=field, category=category, datasets=len(group), distinct_gse=group.gse.nunique()))
    pd.DataFrame(counts).to_csv(out / "metadata_counts.csv", index=False)
    plot_overlay(table, provenance, out, args.font_size)
    command = shlex.join([sys.executable, *sys.argv])
    cache_paths = [args.metadata_dir / "cache" / f"{gse}_gsm.soft" for gse in table.gse.unique()]
    audit = dict(command=command, source_sha256={str(p): sha256(p) for p in source_files + cache_paths},
                 script_sha256=sha256(__file__), coordinates_changed=False,
                 metadata_used_to_fit_projection=False, models_loaded=False,
                 annotation_rule="Reviewed exact GSM extraction protocols; distinguish Ribo-seq library kits from RNA-seq, extraction and depletion kits.",
                 unknown_is_not_absence=True)
    (out / "provenance.json").write_text(json.dumps(audit, indent=2) + "\n")
    (out / "command.txt").write_text(command + "\n")
    shown = table[["number", "dataset", "gse", "riboseq_library_kit", "instruments", "note"]].copy()
    shown = shown.map(lambda value: html.escape(str(value)))
    shown["sources"] = table.gsms.map(lambda value: " ".join(
        f'<a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gsm}">{gsm}</a>'
        for gsm in value.split(";")))
    n_known = int(table.riboseq_library_kit.ne(UNKNOWN).sum())
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Gamma projection colored by dataset metadata</title>
<style>body{{font:16px/1.5 system-ui;margin:30px;color:#233849}}img{{width:100%}}
td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}table{{border-collapse:collapse}}code{{overflow-wrap:anywhere}}</style>
<h1>Same projection, metadata colors</h1>
<p><a href="gamma_projection_metadata.pdf">PDF</a> · <a href="gamma_projection_metadata.png">PNG</a> ·
<a href="projection_with_metadata.csv">Dataset metadata and unchanged coordinates</a> ·
<a href="metadata_counts.csv">Category counts</a></p>
<img src="gamma_projection_metadata.png" alt="Unchanged gamma projection colored by Ribo-seq library kit and sequencing platform">
<p>Only point colors changed: no model was loaded, no distance was recalculated, and no projection was refitted.
There are {len(table)} datasets and {provenance['selected_transcripts']} matched transcripts.
The two coordinates retain {provenance['projection']['fraction_2d']:.1%} of the original geometry.</p>
<p>A Ribo-seq library kit is explicitly identifiable for {n_known}/{len(table)} aliases. Gray means the exact
sample record does not unambiguously name one, not that no kit was used. Kit modifications are not separate categories.
Sequencing platforms retain the reported model; a multi-platform alias is not assigned to just one instrument.</p>
<p>These are descriptive overlays, not evidence that a kit causes the observed gamma pattern.
For example, all five SMARTer aliases here come from one study.
GEO explicitly links the selected gillen_2021 samples to wilczynska_2019 as reanalyses:
<a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4793181">GSM4793181</a>.
Their proximity must not be counted as independent protocol replication.</p>
<h2>Dataset key and exact-sample sources</h2>{shown.to_html(index=False, escape=False)}
<p>Annotations were reviewed from the existing NCBI SOFT cache; file hashes are retained in
<a href="provenance.json">provenance.json</a>. The Ribo-seq kit is not inferred from a library kit
mentioned only for paired RNA-seq. Extraction kits and rRNA-depletion kits are not substituted.</p>
<h2>Reproduce</h2><pre>{html.escape(command)}</pre></html>"""
    (out / "report.html").write_text(body)
    print(f"Saved: {out / 'gamma_projection_metadata.pdf'}")
    print(f"Named Ribo-seq kit: {n_known}/{len(table)}; coordinates unchanged.")


if __name__ == "__main__":
    main()
