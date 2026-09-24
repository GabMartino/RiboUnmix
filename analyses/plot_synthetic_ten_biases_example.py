#!/usr/bin/env python3
"""One-page observations for ten synthetic biases at three read depths.

Rows are bias conditions and columns are nominal depths. Every cell overlays
the saved unbiased and corresponding biased NB2 observations for one transcript.
Counts are raw arithmetic means of rep1 and rep2: no depth normalization,
smoothing, resampling, pseudocount, zero filtering, or peak clipping is used.
Each cell has its own zero-based linear y scale so sparse profiles remain visible.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Utils.publication_plot_style import publication_rc
from analyses.plot_synthetic_bias_and_depth_example import (
    BIAS_LABELS,
    DEPTHS,
    read_transcript,
    replica_profile,
)


# Deterministic display choice: the shortest 100--149-codon transcript with at
# least one saved site for every bias. The absolute shortest K transcript has
# only two represented bias types, while the shortest all-bias transcript is 51 codons.
DEFAULT_TRANSCRIPT = "ENST00000306954.5"
ABSOLUTE_SHORTEST_TRANSCRIPT = "ENST00000711617.1"
SHORTEST_ALL_BIASES_TRANSCRIPT = "ENST00000321301.7"
BIAS_ORDER = (
    "3prime_aa", "3prime_cc", "3prime_gg", "3prime_uu",
    "5prime_aa", "5prime_cc", "5prime_gg", "5prime_uu",
    "gc_fraction_gt_0p7", "au_fraction_gt_0p7",
)
UNBIASED_COLOR = "#9573CC"
BIASED_COLOR = "#D86060"


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_record(path: Path, metadata: dict):
    return {"sha256": file_sha256(path), "size_bytes": path.stat().st_size,
            "metadata": metadata}


def load_observations(data_root: Path, transcript_id: str):
    """Load one transcript from K, annotations, and all 33 count files."""
    sources = {}
    kinetic_path = data_root / "artificial_ground_truth_kinetics_target_mean_one.parquet"
    kinetic_rows, kinetic_meta = read_transcript(kinetic_path, transcript_id)
    kinetic = kinetic_rows["kinetics_target"]
    sources[str(kinetic_path.resolve())] = _source_record(kinetic_path, kinetic_meta)

    annotations = {}
    fingerprints = {kinetic_meta.get("riboart.source_run_fingerprint")}
    for bias in BIAS_ORDER:
        path = (data_root / "bias_profile" /
                f"artificial_bias_{bias}_compendium_added_bias_only.parquet")
        rows, metadata = read_transcript(path, transcript_id, "added_bias")
        if not {"rep1", "rep2", "mean"}.issubset(rows):
            raise ValueError(f"{bias}: all three stored annotation rows are required.")
        added = rows["rep1"]
        if not all(np.array_equal(added, rows[role]) for role in ("rep2", "mean")):
            raise ValueError(f"{bias}: annotations differ across stored replicas.")
        if added.shape != kinetic.shape or metadata.get("riboart.sequence_bias_feature") != bias:
            raise ValueError(f"{bias}: annotation coordinates or metadata do not match.")
        annotations[bias] = added > 0
        fingerprints.add(metadata.get("riboart.source_run_fingerprint"))
        sources[str(path.resolve())] = _source_record(path, metadata)

    unbiased, biased = {}, {}
    for depth_slug, nominal_depth in DEPTHS:
        unbiased_path = (data_root / depth_slug /
                         f"artificial_ground_truth_psite_counts_{depth_slug}.parquet")
        rows, metadata = read_transcript(unbiased_path, transcript_id)
        if (float(metadata["riboart.counts_per_codon_unbiased_baseline"]) != nominal_depth
                or metadata.get("riboart.sequence_bias_enabled") != "false"):
            raise ValueError(f"{depth_slug}: invalid unbiased count metadata.")
        values = replica_profile(rows, "mean")
        if values.shape != kinetic.shape:
            raise ValueError(f"{depth_slug}: unbiased count coordinates do not match K.")
        unbiased[depth_slug] = values
        fingerprints.add(metadata.get("riboart.source_run_fingerprint"))
        sources[str(unbiased_path.resolve())] = _source_record(unbiased_path, metadata)

        biased[depth_slug] = {}
        for bias in BIAS_ORDER:
            path = (data_root / depth_slug /
                    f"artificial_bias_{bias}_psite_counts_{depth_slug}.parquet")
            rows, metadata = read_transcript(path, transcript_id)
            if (float(metadata["riboart.counts_per_codon_unbiased_baseline"]) != nominal_depth
                    or metadata.get("riboart.sequence_bias_enabled") != "true"
                    or metadata.get("riboart.sequence_bias_feature") != bias):
                raise ValueError(f"{depth_slug}/{bias}: invalid biased count metadata.")
            values = replica_profile(rows, "mean")
            if values.shape != kinetic.shape:
                raise ValueError(f"{depth_slug}/{bias}: count coordinates do not match K.")
            biased[depth_slug][bias] = values
            fingerprints.add(metadata.get("riboart.source_run_fingerprint"))
            sources[str(path.resolve())] = _source_record(path, metadata)

    if None in fingerprints or len(fingerprints) != 1:
        raise ValueError("Selected K, annotations, and observations do not share one simulator run.")
    return kinetic, annotations, unbiased, biased, sources


def figure_style(font_size: float):
    style = publication_rc()
    style.update({
        "font.size": font_size, "font.weight": "bold",
        "axes.labelsize": font_size, "axes.labelweight": "bold",
        "axes.titlesize": font_size, "axes.titleweight": "bold",
        "xtick.labelsize": font_size, "ytick.labelsize": font_size,
        "legend.fontsize": font_size, "axes.linewidth": 0.75,
        "xtick.major.width": 0.75, "ytick.major.width": 0.75,
    })
    if style["text.usetex"]:
        style["text.latex.preamble"] += (
            r"\renewcommand{\seriesdefault}{\bfdefault}\AtBeginDocument{\boldmath}")
    return style


def plot_observations(args, kinetic, unbiased, biased, selected):
    style = figure_style(args.font_size)
    x = np.arange(len(kinetic))[selected]
    with matplotlib.rc_context(style):
        fig, axes = plt.subplots(
            len(BIAS_ORDER), len(DEPTHS),
            figsize=(args.figure_width, args.figure_height), sharex=True, squeeze=False)
        for row, bias in enumerate(BIAS_ORDER):
            for column, (depth_slug, _) in enumerate(DEPTHS):
                ax = axes[row, column]
                reference = unbiased[depth_slug][selected]
                condition = biased[depth_slug][bias][selected]
                ax.plot(x, reference, color=UNBIASED_COLOR,
                        linewidth=args.line_width, zorder=3)
                ax.plot(x, condition, color=BIASED_COLOR,
                        linewidth=args.line_width, zorder=2)
                maximum = max(1.0, float(reference.max()), float(condition.max()))
                ax.set_ylim(0, maximum * 1.12)
                ax.set_xlim(x[0], x[-1])
                ax.yaxis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=2))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
                ax.grid(axis="y")
                ax.set_axisbelow(True)
                if row < len(BIAS_ORDER) - 1:
                    ax.tick_params(axis="x", labelbottom=False)
                if column == 0:
                    ax.text(
                        0.02, 0.90, BIAS_LABELS[bias], transform=ax.transAxes,
                        ha="left", va="top", fontsize=args.font_size, fontweight="bold",
                        bbox={"facecolor": "white", "edgecolor": "none",
                              "alpha": 0.80, "pad": 0.8})

        fig.supylabel("Raw P-site counts", x=0.014, fontweight="bold")
        fig.supxlabel("P-site codon index (0-based)", y=0.012, fontweight="bold")
        fig.suptitle(
            "Synthetic bias observations — " + args.transcript.replace("_", r"\_"),
            x=0.53, y=0.997, fontsize=args.font_size + 1.2, fontweight="bold")
        handles = [
            Line2D([0], [0], color=UNBIASED_COLOR, linewidth=args.line_width,
                   label="Unbiased observation"),
            Line2D([0], [0], color=BIASED_COLOR, linewidth=args.line_width,
                   label="Biased observation"),
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.965),
                   ncol=2, handlelength=2.2, columnspacing=1.8)
        fig.subplots_adjust(left=0.105, right=0.992, bottom=0.062,
                            top=0.890, hspace=0.24, wspace=0.28)
        for column, (_, nominal_depth) in enumerate(DEPTHS):
            position = axes[0, column].get_position()
            fig.text(
                (position.x0 + position.x1) / 2, 0.913,
                f"{nominal_depth:g} reads/codon",
                ha="center", va="center", fontsize=args.font_size + 0.5,
                fontweight="bold")
        for suffix in ("pdf", "png", "svg"):
            fig.savefig(args.output_dir / f"{args.output_stem}.{suffix}",
                        dpi=args.dpi if suffix == "png" else None)
        plt.close(fig)
    return {
        "text.usetex": style["text.usetex"], "font.family": style["font.family"],
        "font.size": style["font.size"], "font.weight": style["font.weight"],
        "figure_size_inches": [args.figure_width, args.figure_height],
        "line_width": args.line_width,
        "layout": "10 bias rows x 3 read-depth columns",
        "y_axis": "independent zero-based linear scale in every cell",
    }


def save_sources(args, kinetic, annotations, unbiased, biased, sources, selected, typography):
    output = args.output_dir / f"{args.output_stem}_source"
    output.mkdir(parents=True, exist_ok=True)
    fields = ["transcript_id", "codon_index", "displayed"]
    for depth_slug, _ in DEPTHS:
        fields.append(f"{depth_slug}_unbiased_counts")
        fields.extend(f"{depth_slug}_{bias}_biased_counts" for bias in BIAS_ORDER)
    with (output / "profiles.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(kinetic)):
            row = {"transcript_id": args.transcript, "codon_index": index,
                   "displayed": bool(selected[index])}
            for depth_slug, _ in DEPTHS:
                row[f"{depth_slug}_unbiased_counts"] = float(unbiased[depth_slug][index])
                for bias in BIAS_ORDER:
                    row[f"{depth_slug}_{bias}_biased_counts"] = float(
                        biased[depth_slug][bias][index])
            writer.writerow(row)

    summaries = []
    for bias in BIAS_ORDER:
        for depth_slug, nominal_depth in DEPTHS:
            reference = unbiased[depth_slug][selected]
            condition = biased[depth_slug][bias][selected]
            summaries.append({
                "bias": bias, "depth": nominal_depth,
                "annotated_sites": int((annotations[bias] & selected).sum()),
                "unbiased_mean_count": float(reference.mean()),
                "biased_mean_count": float(condition.mean()),
                "unbiased_max_count": float(reference.max()),
                "biased_max_count": float(condition.max()),
                "unbiased_zero_fraction": float(np.mean(reference == 0)),
                "biased_zero_fraction": float(np.mean(condition == 0)),
            })
    with (output / "observation_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    executable = Path(sys.executable)
    command = f"RIBOUNMIX_PLOT_TEX={int(typography['text.usetex'])} " + shlex.join([
        str(executable.relative_to(ROOT)) if executable.is_relative_to(ROOT) else str(executable),
        str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]])
    provenance = {
        "command": command, "transcript_id": args.transcript,
        "selection": ("Shortest 100--149-codon transcript with at least one positive saved annotation for all ten biases"
                      if args.transcript == DEFAULT_TRANSCRIPT else
                      "Transcript explicitly supplied through --transcript"),
        "absolute_shortest_transcript": ABSOLUTE_SHORTEST_TRANSCRIPT,
        "absolute_shortest_exclusion": "Only 3prime_aa and 3prime_cc have positive annotations",
        "shortest_all_biases_transcript": SHORTEST_ALL_BIASES_TRANSCRIPT,
        "default_selection_length_interval": "100 <= sense codons < 150",
        "bias_order": list(BIAS_ORDER),
        "depths_reads_per_codon": [depth for _, depth in DEPTHS],
        "sense_codons": len(kinetic),
        "displayed_interval_zero_based": [args.start, int(np.flatnonzero(selected)[-1]) + 1],
        "replicate": "arithmetic mean of stored rep1 and rep2; integerized mean row not used",
        "count_scale": "raw counts; independent zero-based linear y scale per cell",
        "depth_normalization": False, "pseudocount": 0, "smoothing": False,
        "resampling": False, "peak_clipping": False, "zero_filtering": False,
        "neural_models_used": False, "source_files": sources,
        "typography": typography, "script_sha256": file_sha256(Path(__file__)),
        "helper_script_sha256": file_sha256(
            ROOT / "analyses/plot_synthetic_bias_and_depth_example.py"),
        "summaries": summaries,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + command + "\n")
    caption = rf"""\textbf{{Synthetic bias observations across sequencing depths.}}
The same {len(kinetic)}-codon transcript, \texttt{{{args.transcript}}}, is shown for all ten
injected sequence-bias conditions (rows) at $C\in\{{0.25,2,20\}}$ expected
reads/codon (columns). It is the shortest transcript in the prespecified 100--149-codon
display range containing at least one annotated site for every bias condition. Purple and red curves show the saved
unbiased and corresponding biased NB2 observations, respectively, using the exact
arithmetic mean of the two sampled replicas. Counts are not divided by nominal depth
or by their observed mean. Each cell uses its own zero-based linear y-axis, and hence
apparent peak heights must not be compared across cells without reading the tick values.
All sampled zeros and peaks are retained; no smoothing, pseudocount, resampling,
clipping, or neural-model prediction is used.
"""
    (output / "caption.tex").write_text(caption)
    (output / "README.md").write_text(f"""# Ten biases across three read depths

![Figure](../{args.output_stem}.png)

Transcript `{args.transcript}` has {len(kinetic)} codons and is the shortest sequence
in the prespecified 100--149-codon display range with at least one saved annotation
for each bias. The absolute shortest generated transcript,
`{ABSOLUTE_SHORTEST_TRANSCRIPT}` (12 codons), represents only two bias types; the
globally shortest all-bias transcript, `{SHORTEST_ALL_BIASES_TRANSCRIPT}`, has 51 codons.

Rows are biases; columns are 0.25, 2, and 20 reads/codon. Curves are raw arithmetic
replica-mean counts. Each cell has an independent linear y-axis to prevent a single
large peak from compressing the other conditions. Tick values, rather than apparent
height, must be used for comparisons across cells.

Reproduce from the repository root:

```bash
{command}
```
""")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "Datasets/Synthetic_data")
    parser.add_argument("--transcript", default=DEFAULT_TRANSCRIPT)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--output-stem", default="synthetic_ten_biases_same_transcript")
    parser.add_argument("--font-size", type=float, default=8.0)
    parser.add_argument("--line-width", type=float, default=1.0)
    parser.add_argument("--figure-width", type=float, default=7.0)
    parser.add_argument("--figure-height", type=float, default=8.2)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args(argv)
    positive = (args.font_size, args.line_width, args.figure_width,
                args.figure_height, args.dpi)
    if not all(np.isfinite(value) and value > 0 for value in positive):
        parser.error("Typography, dimensions, and DPI must be positive.")
    args.data_root = args.data_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    kinetic, annotations, unbiased, biased, sources = load_observations(
        args.data_root, args.transcript)
    stop = len(kinetic) if args.stop is None else args.stop
    if not 0 <= args.start < stop <= len(kinetic) or stop - args.start < 2:
        parser.error("Require a valid source-CDS interval containing at least two codons.")
    selected = ((np.arange(len(kinetic)) >= args.start)
                & (np.arange(len(kinetic)) < stop))
    if args.transcript == DEFAULT_TRANSCRIPT and any(
            not (annotations[bias] & selected).any() for bias in BIAS_ORDER):
        raise ValueError("The default range-qualified transcript lost a bias after interval selection.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    typography = plot_observations(args, kinetic, unbiased, biased, selected)
    save_sources(args, kinetic, annotations, unbiased, biased, sources, selected, typography)
    print(f"Figure: {args.output_dir / (args.output_stem + '.pdf')}")
    print(f"{args.transcript}: {len(kinetic)} codons")
    for bias in BIAS_ORDER:
        print(f"{bias}: {int((annotations[bias] & selected).sum())} annotated sites")


if __name__ == "__main__":
    main()
