#!/usr/bin/env python3
"""Two-panel illustration of injected bias and read-depth-dependent observations.

Default A: saved deterministic K versus K*(1+added_bias), with annotated sites.
Occupancy A: the normalized two-trajectory occupancy consensus Q, with the
deterministic displacement to Q*(1+added_bias) shown only at affected sites.
Optional A: saved unbiased/biased NB2 observations, as in the supplied example.
B: the same transcript's raw counts at 0.25, 2 and 20 reads/codon.

Panel B is not normalized; three stacked linear plots use independent count
scales and a shared codon axis. Only optional observation-mode A divides counts by
nominal depth C. Replica means are recomputed from rep1/rep2, never taken from
the integerized 'mean' row.
No resampling, fitting, smoothing, zero filtering, or clipping of profile peaks.
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

import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Utils.publication_plot_style import publication_rc
from analyses.analyze_synthetic_tasep_occupancy_agreement import normalize_occupancy
from analyses.plot_synthetic_hierarchy import load_occupancy

DEFAULT_TRANSCRIPT = "ENST00000319974.6"
DEPTHS = (("0p25_per_codon", .25), ("2_per_codon", 2.), ("20_per_codon", 20.))
COLORS = ("#0072B2", "#E69F00", "#009E73")
BIAS_LABELS = {f"{end}prime_{bases.lower()}": rf"${end}^\prime$ {bases}"
               for end in (3, 5) for bases in ("AA", "CC", "GG", "UU")}
BIAS_LABELS.update({"gc_fraction_gt_0p7": r"GC $>0.7$", "au_fraction_gt_0p7": r"AU $>0.7$"})


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_transcript(path, transcript_id, value_column="rib_profile"):
    """Read only one transcript; use row-group statistics and small Arrow batches."""
    profiles = {}
    with pq.ParquetFile(path) as reader:
        metadata = {k.decode(): v.decode() for k, v in (reader.schema_arrow.metadata or {}).items()
                    if k != b"ARROW:schema"}
        tid_column = reader.schema.names.index("transcript_id")
        groups = []
        for i in range(reader.num_row_groups):
            stats = reader.metadata.row_group(i).column(tid_column).statistics
            if stats is not None and stats.has_min_max:
                low = stats.min.decode() if isinstance(stats.min, bytes) else stats.min
                high = stats.max.decode() if isinstance(stats.max, bytes) else stats.max
                if not low <= transcript_id <= high:
                    continue
            groups.append(i)
        for batch in reader.iter_batches(row_groups=groups, batch_size=32, use_threads=False,
                                         columns=["transcript_id", "sample", value_column]):
            for i, tid in enumerate(batch.column("transcript_id").to_pylist()):
                if tid != transcript_id:
                    continue
                sample = batch.column("sample")[i].as_py()
                role = sample.rsplit("_", 1)[-1] if sample != "kinetics_target" else sample
                if role in profiles:
                    raise ValueError(f"Duplicate {role} row for {transcript_id}: {path}")
                values = np.asarray(batch.column(value_column)[i].as_py(), dtype=float)
                if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
                    raise ValueError(f"Invalid profile for {transcript_id}: {path}")
                profiles[role] = values
    if not profiles:
        raise ValueError(f"Transcript {transcript_id} is absent from {path}")
    return profiles, metadata


def replica_profile(profiles, choice):
    if not {"rep1", "rep2"}.issubset(profiles):
        raise ValueError("Both original replicas are required.")
    a, b = profiles["rep1"], profiles["rep2"]
    if a.shape != b.shape or np.any(a < 0) or np.any(b < 0):
        raise ValueError("Unaligned or negative count replicas.")
    return (a+b)/2 if choice == "mean" else profiles[choice].copy()


def kinetic_bias_example(kinetic, added_bias):
    if kinetic.shape != added_bias.shape or np.any(added_bias < 0):
        raise ValueError("K and the nonnegative added-bias annotation must align.")
    multiplier = 1+added_bias
    return kinetic*multiplier, multiplier, added_bias > 0


def scale_counts(values, depth):
    if depth <= 0:
        raise ValueError("Nominal read depth must be positive.")
    return values/depth


def set_count_axis(ax, maximum):
    """Independent linear count scale; retain zeros and reserve label space."""
    ax.set_yscale("linear")
    ax.set_ylim(0, max(1., maximum)*1.35)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    ax.minorticks_off()


def prepare_data(args):
    sources = {}

    def load(path, column="rib_profile"):
        profiles, metadata = read_transcript(path, args.transcript, column)
        sources[str(path.resolve())] = dict(sha256=sha256(path), metadata=metadata)
        return profiles, metadata

    k_rows, k_meta = load(args.data_root / "artificial_ground_truth_kinetics_target_mean_one.parquet")
    k = k_rows["kinetics_target"]
    if not np.isclose(k.mean(), 1., atol=1e-10) or k_meta.get("riboart.read_sampling_applied") != "false":
        raise ValueError("Expected the saved deterministic mean-one kinetic target, not a count profile.")
    bias_rows, _ = load(args.data_root / "bias_profile" /
                        f"artificial_bias_{args.bias}_compendium_added_bias_only.parquet", "added_bias")
    added = bias_rows["rep1"]
    if not all(np.array_equal(added, bias_rows[role]) for role in ("rep2", "mean")):
        raise ValueError("The injected bias annotations differ across replicas.")
    biased_k, multiplier, sites = kinetic_bias_example(k, added)

    occupancy_path = args.data_root / "artificial_ground_truth_tasep_occupancy_replicates.parquet"
    O1, O2, occupancy_meta = load_occupancy(occupancy_path, args.transcript)
    sources[str(occupancy_path.resolve())] = {
        "sha256": sha256(occupancy_path),
        "metadata": occupancy_meta,
    }
    q1 = normalize_occupancy(O1)
    q2 = normalize_occupancy(O2)
    if q1.shape != k.shape or q2.shape != k.shape:
        raise ValueError("Kinetic and occupancy profiles have different P-site coordinates.")
    qbar = 0.5 * (q1 + q2)
    if not all(np.isclose(values.mean(), 1.0, atol=1e-12) for values in (q1, q2, qbar)):
        raise ValueError("Normalized occupancy profiles must have positional mean one.")
    biased_qbar = qbar * multiplier

    def counts(depth_slug, condition):
        name = "artificial_ground_truth" if condition == "unbiased" else f"artificial_bias_{args.bias}"
        profiles, metadata = load(args.data_root / depth_slug / f"{name}_psite_counts_{depth_slug}.parquet")
        nominal = dict(DEPTHS)[depth_slug]
        if (float(metadata["riboart.counts_per_codon_unbiased_baseline"]) != nominal
                or metadata["riboart.observation_model"] != "negative_binomial_NB2"
                or float(metadata["riboart.negative_binomial_dispersion_alpha"]) != .1
                or metadata["riboart.sequence_bias_enabled"] != str(condition != "unbiased").lower()):
            raise ValueError("The saved count-file metadata does not match the requested condition.")
        if any(v.shape != k.shape for v in profiles.values()):
            raise ValueError("Count and kinetic/bias profiles have different P-site coordinates.")
        return replica_profile(profiles, args.replicate), profiles

    profiles, source_replicas = {}, {}
    for slug, _ in DEPTHS:
        values, replicas = counts(slug, args.depth_condition)
        profiles[slug] = values
        source_replicas[slug] = replicas
    if args.left_mode == "kinetics":
        left = (k, biased_k)
    elif args.left_mode == "occupancy":
        left = (k, qbar, biased_qbar)
    else:
        slug = args.left_depth
        c = dict(DEPTHS)[slug]
        left = tuple(scale_counts(counts(slug, condition)[0], c) for condition in ("unbiased", "same-bias"))
    # The separately exported occupancy artifact predates the fingerprint field.
    # All artifacts that declare a simulator fingerprint must agree; exact array
    # alignment and the hierarchy audit provide the occupancy-side validation.
    fingerprints = {
        value
        for entry in sources.values()
        if (value := entry["metadata"].get("riboart.source_run_fingerprint")) is not None
    }
    if len(fingerprints) != 1:
        raise ValueError("Fingerprint-bearing source files do not share one simulator run.")
    return dict(k=k, q1=q1, q2=q2, qbar=qbar, biased_qbar=biased_qbar,
                biased_k=biased_k, multiplier=multiplier, sites=sites, left=left,
                depth_profiles=profiles, replicas=source_replicas, sources=sources)


def plot_example(args, data, selected, stem):
    style = publication_rc()
    style.update({"font.size": args.font_size, "font.weight": "bold",
                  "axes.labelsize": args.font_size + 0.5, "axes.labelweight": "bold",
                  "axes.titlesize": args.font_size+1.5, "axes.titleweight": "bold",
                  "xtick.labelsize": args.font_size, "ytick.labelsize": args.font_size,
                  "legend.fontsize": args.font_size - 0.4, "axes.linewidth": 1.5,
                  "xtick.major.width": 1.5, "ytick.major.width": 1.5})
    if style["text.usetex"]:
        # Font weight alone does not reliably bold TeX-generated ticks and math.
        style["text.latex.preamble"] += (
            r"\renewcommand{\seriesdefault}{\bfdefault}\AtBeginDocument{\boldmath}")
    x = np.arange(len(data["k"]))[selected]
    label = BIAS_LABELS[args.bias]
    with matplotlib.rc_context(style):
        fig = plt.figure(figsize=(15.2, 4.2), layout="constrained")
        subfigures = fig.subfigures(1, 2, wspace=0.055)
        left_ax = subfigures[0].subplots(1, 1)
        depth_axes = list(subfigures[1].subplots(3, 1, sharex=True))
        axes = [left_ax, *depth_axes]
        if args.left_mode == "occupancy":
            occupancy, = axes[0].plot(
                x, data["left"][1][selected], color="#0072B2",
                linewidth=args.line_width * 1.12,
                label=r"Simulated ribosome profile $\bar q_t$", zorder=4,
            )
            affected = data["sites"][selected]
            bias_x = x[affected]
            q_at_bias = data["left"][1][selected][affected]
            qb_at_bias = data["left"][2][selected][affected]
            axes[0].vlines(
                bias_x, q_at_bias, qb_at_bias, color="#D55E00",
                linewidth=args.line_width * 0.95, alpha=0.85, zorder=2,
            )
            biased = axes[0].scatter(
                bias_x, qb_at_bias, marker="v", s=42,
                facecolors="#D55E00", edgecolors="#A33F00",
                linewidths=0.65,
                label=rf"Biased expectation $\bar q_t b_t$ ({label})", zorder=5,
            )
            line_handles = [occupancy, biased]
        else:
            left_labels = ((r"Unbiased kinetic target $K_t$", rf"{label}-shaped $K_t b_t$")
                           if args.left_mode == "kinetics" else
                           ("Unbiased NB2 observation", f"{label}-biased NB2 observation"))
            # Draw the reference last so it remains visible where curves coincide.
            reference, = axes[0].plot(
                x, data["left"][0][selected], color="#9573CC",
                linewidth=args.line_width, label=left_labels[0], zorder=3,
            )
            biased, = axes[0].plot(
                x, data["left"][1][selected], color="#D86060",
                linewidth=args.line_width, label=left_labels[1], zorder=2,
            )
            line_handles = [reference, biased]
        upper = max(1., *(float(v[selected].max()) for v in data["left"]))*1.28
        bias_x = x[data["sites"][selected]]
        if args.left_mode == "occupancy":
            triangles = None
            legend_handles = line_handles
            axes[0].set_ylim(0, upper)
        else:
            triangles = axes[0].scatter(
                bias_x, np.full(len(bias_x), -.035*upper), marker="^",
                s=36, facecolors="white", edgecolors="#C83B36",
                linewidths=args.line_width*.85,
                label="Triangle: injected bias", zorder=4,
            )
            legend_handles = [*line_handles, triangles]
            axes[0].set_ylim(-.075*upper, upper)
        axes[0].axhline(0, color="#777777", linewidth=.6)
        ticks = MaxNLocator(nbins=4).tick_values(0, upper)
        axes[0].set_yticks(ticks[(ticks >= 0) & (ticks <= upper)])
        axes[0].legend(handles=legend_handles, loc="upper right", fontsize=args.font_size,
                       handlelength=1.5, labelspacing=.25)
        if args.left_mode == "kinetics":
            left_title = f"Kinetic profile and {label} bias"
            left_ylabel = "Relative kinetic profile"
        elif args.left_mode == "occupancy":
            left_title = f"Occupancy and deterministic {label} effect"
            left_ylabel = "Relative profile / unit depth"
        else:
            left_title = f"Unbiased and {label}-biased observations"
            left_ylabel = "Counts / nominal depth"
        axes[0].set_title("A  " + left_title, loc="left")
        axes[0].set_ylabel(left_ylabel)
        for ax, (slug, c), color in zip(depth_axes, DEPTHS, COLORS):
            values = data["depth_profiles"][slug][selected]
            ax.plot(x, values, color=color, linewidth=args.line_width)
            set_count_axis(ax, float(values.max()))
            ax.text(.98, .93, f"{c:g} reads/codon", transform=ax.transAxes,
                    ha="right", va="top", fontsize=args.font_size, color=color, fontweight="bold")
        subfigures[1].supylabel("Counts", fontsize=args.font_size, fontweight="bold", x=-0.035)
        axes[1].set_title("B  Read depth: " + ("unbiased observations" if args.depth_condition == "unbiased"
                                               else f"{label}-biased observations"), loc="left")
        for ax in axes:
            ax.set_xlim(x[0], x[-1])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            ax.grid(axis="y")
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", which="major", width=1.5, length=5.0)
            for tick_label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
                tick_label.set_fontweight("bold")
        for ax in depth_axes[:-1]:
            ax.tick_params(axis="x", labelbottom=False)
        for ax in (left_ax, depth_axes[-1]):
            ax.set_xlabel("P-site codon index (0-based)")
        fig.suptitle(args.transcript.replace("_", r"\_"), fontsize=args.font_size, color="#555555", fontweight="bold")
        for suffix in ("pdf", "png", "svg"):
            fig.savefig(args.output_dir / f"{stem}.{suffix}", dpi=600)
        plt.close(fig)
    return {**{k: style[k] for k in ("text.usetex", "font.family", "font.size", "font.weight")},
            "profile_line_width": args.line_width}


def save_sources(args, data, selected, stem, typography):
    out = args.output_dir / f"{stem}_source"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "profiles.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["transcript_id", "codon_index", "displayed", "K", "q1", "q2", "Q",
                         "bias_multiplier", "bias_site", "K_times_bias", "Q_times_bias"] +
                        [f"depth_counts_{slug}" for slug, _ in DEPTHS])
        for i in range(len(data["k"])):
            writer.writerow([args.transcript, i, selected[i], data["k"][i], data["q1"][i],
                             data["q2"][i], data["qbar"][i], data["multiplier"][i],
                             data["sites"][i], data["biased_k"][i], data["biased_qbar"][i]] +
                            [data["depth_profiles"][slug][i] for slug, _ in DEPTHS])
    summaries = []
    for slug, c in DEPTHS:
        replicas = data["replicas"][slug]
        values = replica_profile(replicas, args.replicate)
        mean = replica_profile(replicas, "mean")
        summaries.append(dict(depth=c, transcript_id=args.transcript, displayed_codons=int(selected.sum()),
                              raw_mean_count=float(values[selected].mean()),
                              raw_max_count=float(values[selected].max()),
                              zero_fraction=float(np.mean(values[selected] == 0)),
                              stored_integer_mean_abs_difference=float(np.abs(mean-replicas["mean"]).sum())))
    with (out / "read_depth_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    executable = Path(sys.executable)
    command = f"RIBOUNMIX_PLOT_TEX={int(typography['text.usetex'])} " + shlex.join(
        [str(executable.relative_to(ROOT)) if executable.is_relative_to(ROOT) else str(executable),
         str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]])
    selection = ("Previously used manuscript example; not selected by a depth-effect score"
                 if args.transcript == DEFAULT_TRANSCRIPT else
                 "Transcript explicitly supplied with --transcript; no performance-based selection by this script")
    provenance = dict(command=command, transcript_id=args.transcript,
                      selection=selection,
                      sense_codons=len(data["k"]), displayed_interval_zero_based=[args.start, int(np.flatnonzero(selected)[-1])+1],
                      bias=args.bias, n_bias_sites=int(data["sites"].sum()),
                      n_displayed_bias_sites=int(data["sites"][selected].sum()),
                      left_mode=args.left_mode, left_observation_depth=args.left_depth if args.left_mode == "observations" else None,
                      depth_condition=args.depth_condition, replicate=args.replicate,
                      count_scaling=dict(
                          left=("not applicable: kinetic profiles" if args.left_mode == "kinetics" else
                                "none: Q is mean one and Q*b is per unit nominal depth"
                                if args.left_mode == "occupancy" else
                                "selected raw replica or arithmetic replica mean divided by nominal C"),
                          right="none: selected raw replica or arithmetic replica mean, in count units"),
                      right_axis=dict(scale="linear", layout="three vertically stacked depth plots",
                                      shared_x=True, shared_y=False, pseudocount=0, zeros_retained=True),
                      left_profile_semantics=(
                          "separately normalized q1/q2 consensus Q as one continuous curve; "
                          "at affected coordinates only, stems and downward triangles show Q*(1+added_bias); "
                          "Q*b is the exact consensus expectation per unit nominal depth and is not renormalized"
                          if args.left_mode == "occupancy" else
                          "K*(1+added_bias), no renormalization; illustrative, not the simulator's traffic mean"
                          if args.left_mode == "kinetics" else
                          "saved sampled counts divided only by nominal depth"
                      ),
                      smoothing=False, count_resampling=False, peak_clipping=False, zero_filtering=False,
                      neural_models_used=False, source_files=data["sources"], typography=typography,
                      script_sha256=sha256(Path(__file__)), summaries=summaries)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    (out / "commands.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n"+command+"\n")
    replica_description = ("the arithmetic mean of the two original replicas" if args.replicate == "mean"
                           else f"original replica {args.replicate[-1]}")
    panel_a = (r"The saved mean-one kinetic target $K_{ti}$ and the illustrative product $K_{ti}b_{ti}$, "
               r"where $b_{ti}=1+\mathrm{added\_bias}_{ti}$. The product is not renormalized. "
               r"It is a visualization of the multiplier's effect on $K$, not a saved noisy observation "
               r"or the simulator's expected profile: actual counts originate from TASEP occupancy $q$, not $K$."
               if args.left_mode == "kinetics" else
               (r"The mean-one two-trajectory occupancy consensus $Q_{ti}$. At affected "
                r"coordinates only, vermillion stems and downward triangles show the exact biased "
                r"expectation per unit nominal depth $Q_{ti}b_{ti}$. The biased expectation is not "
                r"renormalized; omitting a second full curve prevents coincident values from obscuring $Q_t$."
                if args.left_mode == "occupancy" else
               rf"Actual unbiased and {BIAS_LABELS[args.bias]}-biased NB2 count profiles at "
               rf"$C={dict(DEPTHS)[args.left_depth]:g}$, each divided by the same nominal $C$. "
               r"These observations are not the deterministic kinetic target $K_t$."))
    caption = rf"""\textbf{{Synthetic example of bias and sequencing depth.}}
The same transcript, \texttt{{{args.transcript}}}, is used in both panels.
\textbf{{(A)}} {panel_a}
In Panel A, the downward triangles identify exactly the positions with positive saved
added-bias annotations, not positions chosen from observed count peaks.
\textbf{{(B)}} Saved {"unbiased" if args.depth_condition == "unbiased" else BIAS_LABELS[args.bias]+"-biased"}
NB2 observations at $C\in\{{0.25,2,20\}}$ expected reads/codon, using {replica_description}.
Panel B shows raw P-site counts in three vertically stacked plots, ordered by increasing
depth, with a shared codon axis and independent, zero-based linear count axes.
The y-axis ranges differ: compare count values using the ticks, not apparent peak heights.
There is no division by nominal depth $C$ or the observed mean, and no pseudocount
is added. Tick labels remain in count units. Zeros and all peak heights are retained;
no smoothing, new random sampling, or model prediction is used. For replica means,
the exact arithmetic mean is recomputed, not the source's integerized mean row.
Coordinates are zero-based P-sites; the source arrays already omit the terminal boundary.
This is an illustration of the synthetic observations, not evidence about neural-model
recovery of $K_t$. The two simulator traffic trajectories and bias multipliers are fixed
across depths; the stored NB2 dispersion is $\alpha=0.1$.
"""
    (out / "caption.tex").write_text(caption)
    zero_text = ", ".join(f"{s['zero_fraction']:.1%} at {s['depth']:g}" for s in summaries)
    (out / "README.md").write_text(f"""# Synthetic bias/read-depth example

![Figure](../{stem}.png)

Same transcript in both panels: `{args.transcript}`, {len(data['k'])} sense codons,
{int(data['sites'].sum())} annotated {args.bias} bias sites.
Selection: {selection}.

Left mode: **{args.left_mode}**. In kinetic mode, K times b is an explicit illustrative
transformation of two saved arrays, not a claim that the simulator samples counts from K.
Counts actually come from the traffic profile q. Neither K nor q depends on read depth.
The optional `--left-mode observations` shows the actual unbiased/biased count profiles
instead, matching the type of data in the supplied example image. Unlike a mean-one
display, both count curves are divided by the same nominal depth, retaining bias-induced
changes in total count mass. Peak heights are never clipped.

Right condition: **{args.depth_condition}**; replica choice: **{args.replicate}**.
Panel B shows raw P-site counts in three stacked linear plots, with no division by nominal depth C
or the observed mean. With replica means, half-integer counts are possible because the
two integer-valued count vectors are averaged. The count-scale differences across depths
are retained. The plots share codon coordinates but have independent, zero-based y-axis
ranges so sparse profiles remain readable. Compare counts using the y-axis ticks, not
apparent peak heights across rows. No pseudocount is added.
The zero fractions on the displayed interval are: {zero_text} reads/codon.
This is a one-transcript observation-level example, not a model-recovery benchmark.
The stored `mean` count row is integerized, so it is not used as the replica average.

Reproduce from the repository root:

```bash
{command}
```

Other options: `--transcript ID`, `--bias 3prime_cc`, `--replicate rep1`,
`--depth-condition same-bias`, and `--start 100 --stop 250` (zero-based, stop exclusive).
Text is bold. Use `--font-size 14` to increase all text sizes (default 12 points;
panel titles are one point larger), and `--line-width 2` for thicker profile lines
(default 1.6 points). These options do not change the data.
Default is the full source CDS, without trimming or additional terminal removal.
`profiles.csv` contains exact full-length plotted values and the displayed-position mask;
`read_depth_summary.csv`, `provenance.json`, and `caption.tex` give statistics and definitions.
""")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT/"Datasets/Synthetic_data")
    parser.add_argument("--transcript", default=DEFAULT_TRANSCRIPT)
    parser.add_argument("--bias", choices=list(BIAS_LABELS), default="3prime_cc")
    parser.add_argument("--left-mode", choices=("kinetics", "occupancy", "observations"), default="kinetics")
    parser.add_argument("--left-depth", choices=[s for s, _ in DEPTHS], default="20_per_codon")
    parser.add_argument("--depth-condition", choices=("unbiased", "same-bias"), default="unbiased")
    parser.add_argument("--replicate", choices=("mean", "rep1", "rep2"), default="mean")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"figures")
    parser.add_argument("--output-stem")
    parser.add_argument("--font-size", type=float, default=12., help="Base size for all text in points (default: 12).")
    parser.add_argument("--line-width", type=float, default=1.6, help="Profile line width in points (default: 1.6).")
    args = parser.parse_args()
    if not all(np.isfinite(v) and v > 0 for v in (args.font_size, args.line_width)):
        parser.error("Font size and line width must be finite and positive.")
    data = prepare_data(args)
    stop = len(data["k"]) if args.stop is None else args.stop
    if not 0 <= args.start < stop <= len(data["k"]) or stop-args.start < 2:
        parser.error("Require a valid source-CDS interval containing at least two codons.")
    selected = (np.arange(len(data["k"])) >= args.start) & (np.arange(len(data["k"])) < stop)
    stem = args.output_stem or ("synthetic_bias_read_depth_example" +
                               ("_observations" if args.left_mode == "observations" else ""))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    typography = plot_example(args, data, selected, stem)
    summaries = save_sources(args, data, selected, stem, typography)
    print(f"Figure: {args.output_dir / (stem+'.pdf')}")
    print(f"{args.transcript}: {len(data['k'])} codons; {int(data['sites'].sum())} bias sites")
    for row in summaries:
        print(f"C={row['depth']:g}: fraction of displayed zero-count positions={row['zero_fraction']:.4f}")


if __name__ == "__main__":
    main()
