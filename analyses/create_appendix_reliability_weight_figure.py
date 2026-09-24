#!/usr/bin/env python3
"""Create a publication-ready example of the current reliability weighting.

The figure is tied to one completed experiment: it reads that run's frozen,
training-only reliability-reference manifests and the scalar audit columns in
the weighted HEK Parquets.  It never loads or evaluates a trained model.

Four datasets are selected deterministically at the 12.5, 37.5, 62.5 and
87.5 percentiles of the 114 dataset-specific depth references ``tau_d``.  The
distribution panel averages dataset-specific densities, giving every dataset
equal mass irrespective of its transcript count.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "results/my_panels_a100_b32_20260906_114323"
DEFAULT_DATASET_DIRECTORY = (
    PROJECT_ROOT / "Datasets/data/weighted_HEK_riboseq_codon_replicas"
)
DEFAULT_SELECTION_QUANTILES = (0.125, 0.375, 0.625, 0.875)
GRID_SIZE = 240
WEIGHT_LIMIT = 3.0
DEFAULT_FONT_SCALE = 1.40

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Datasets.data.plot_current_reliability_weight_audit import (  # noqa: E402
    WEIGHT_CMAP,
    WeightReference,
    _load_experiment_references,
    normalized_reliability_weight,
)
from Utils.publication_plot_style import publication_rc  # noqa: E402
from analyses.paths import artifact_directory  # noqa: E402
from Utils.reliability_references import (  # noqa: E402
    WEIGHTING_MODE,
    apply_dataset_reliability_reference,
)


def select_reference_quantile_datasets(
    references: dict[str, WeightReference],
    quantiles: Sequence[float],
) -> pd.DataFrame:
    """Select the closest dataset to each requested tau quantile."""
    if not references:
        raise ValueError("No reliability references were supplied.")
    if len(set(quantiles)) != len(quantiles):
        raise ValueError("Selection quantiles must be unique.")
    reference_table = pd.DataFrame(
        {
            "dataset": sorted(references),
            "depth_reference_tau": [references[name].tau for name in sorted(references)],
        }
    )
    rows: list[dict[str, float | str]] = []
    selected: set[str] = set()
    for quantile in quantiles:
        quantile = float(quantile)
        if not 0.0 <= quantile <= 1.0:
            raise ValueError("Selection quantiles must lie in [0, 1].")
        target = float(reference_table["depth_reference_tau"].quantile(quantile))
        ranked = reference_table.assign(
            distance=(reference_table["depth_reference_tau"] - target).abs()
        ).sort_values(["distance", "dataset"], kind="mergesort")
        available = ranked.loc[~ranked["dataset"].isin(selected)]
        if available.empty:
            raise ValueError("Not enough distinct datasets for the requested quantiles.")
        chosen = available.iloc[0]
        dataset = str(chosen["dataset"])
        selected.add(dataset)
        rows.append(
            {
                "selection_quantile": quantile,
                "target_tau": target,
                "dataset": dataset,
                "depth_reference_tau": float(chosen["depth_reference_tau"]),
                "absolute_tau_distance": float(chosen["distance"]),
            }
        )
    return pd.DataFrame(rows)


def equal_dataset_quantile(sorted_weights: Sequence[np.ndarray], q: float) -> float:
    """Quantile of a mixture assigning equal probability to every dataset."""
    if not sorted_weights:
        raise ValueError("No weight arrays were supplied.")
    q = float(q)
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1].")
    if any(values.ndim != 1 or values.size == 0 for values in sorted_weights):
        raise ValueError("Every dataset must provide a non-empty one-dimensional array.")
    lower = min(float(values[0]) for values in sorted_weights)
    upper = max(float(values[-1]) for values in sorted_weights)
    if q == 0.0:
        return lower
    if q == 1.0:
        return upper
    for _ in range(64):
        midpoint = 0.5 * (lower + upper)
        cdf = float(
            np.mean(
                [
                    np.searchsorted(values, midpoint, side="right") / values.size
                    for values in sorted_weights
                ]
            )
        )
        if cdf < q:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def _read_runtime_weight_frame(
    dataset_path: Path,
    dataset_name: str,
    reference: WeightReference,
) -> pd.DataFrame:
    columns = ["id", "coverage", "read_density"]
    schema = set(pq.ParquetFile(dataset_path).schema_arrow.names)
    missing = set(columns).difference(schema)
    if missing:
        raise KeyError(f"{dataset_path.name}: missing {sorted(missing)}.")
    frame = pq.read_table(dataset_path, columns=columns).to_pandas()
    utility_reference = {
        "weighting_mode": WEIGHTING_MODE,
        "depth_reference_tau": reference.tau,
        "normalization_reference_median": reference.normalization_median,
        "depth_weight": reference.depth_weight,
        "coverage_weight": reference.coverage_weight,
    }
    frame["weight"] = apply_dataset_reliability_reference(
        frame,
        dataset_name=dataset_name,
        reference=utility_reference,
    ).astype(np.float64)
    frame["log_read_density"] = np.log1p(
        frame["read_density"].to_numpy(dtype=np.float64, copy=False)
    )
    return frame


def _weight_surface(
    reference: WeightReference,
    y_upper: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coverage = np.linspace(0.0, 1.0, GRID_SIZE)
    log_density = np.linspace(0.0, y_upper, GRID_SIZE)
    coverage_mesh, log_density_mesh = np.meshgrid(coverage, log_density)
    weight = normalized_reliability_weight(
        np.expm1(log_density_mesh),
        coverage_mesh,
        reference,
        allow_plot_boundary=True,
    )
    return coverage, log_density, weight


def _tex_escape(value: str, use_tex: bool) -> str:
    return value.replace("_", r"\_") if use_tex else value


def _distribution_source(
    weights_by_dataset: dict[str, np.ndarray],
    *,
    bin_count: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    sorted_weights = [np.sort(values) for values in weights_by_dataset.values()]
    maximum = max(float(values[-1]) for values in sorted_weights)
    upper = max(WEIGHT_LIMIT, math.ceil(maximum * 10.0) / 10.0)
    edges = np.linspace(0.0, upper, bin_count + 1)
    widths = np.diff(edges)
    densities = np.asarray(
        [
            np.histogram(values, bins=edges)[0] / (values.size * widths)
            for values in sorted_weights
        ],
        dtype=np.float64,
    )
    distribution = pd.DataFrame(
        {
            "weight_bin_left": edges[:-1],
            "weight_bin_right": edges[1:],
            "weight_bin_center": 0.5 * (edges[:-1] + edges[1:]),
            "equal_dataset_mean_density": densities.mean(axis=0),
            "dataset_density_q10": np.quantile(densities, 0.10, axis=0),
            "dataset_density_q90": np.quantile(densities, 0.90, axis=0),
        }
    )
    dataset_summary = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "n_transcripts": int(values.size),
                "mean_weight": float(values.mean()),
                "median_weight": float(np.median(values)),
                "q25_weight": float(np.quantile(values, 0.25)),
                "q75_weight": float(np.quantile(values, 0.75)),
                "min_weight": float(values.min()),
                "max_weight": float(values.max()),
            }
            for dataset, values in sorted(weights_by_dataset.items())
        ]
    )
    quantile_summary = {
        f"q{int(100 * q):02d}": equal_dataset_quantile(sorted_weights, q)
        for q in (0.025, 0.25, 0.50, 0.75, 0.975)
    }
    quantile_summary["equal_dataset_mean"] = float(
        np.mean([values.mean() for values in sorted_weights])
    )
    return distribution, dataset_summary, quantile_summary


def _write_markdown_explanation(
    *,
    output_dir: Path,
    run_root: Path,
    selected: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    quantiles: dict[str, float],
    font_scale: float = DEFAULT_FONT_SCALE,
) -> Path:
    """Write a result-linked interpretation that stays synchronized with the plot."""
    panel_rows: list[str] = []
    for panel_index, (_, row) in enumerate(selected.iterrows()):
        panel_rows.append(
            "| "
            + chr(ord("A") + panel_index)
            + f" | `{row['dataset']}` | {row['selection_quantile']:.3f} | "
            + f"{row['depth_reference_tau']:.6f} | {int(row['n_transcripts']):,} | "
            + f"{row['mean_weight']:.3f} | {row['median_weight']:.3f} | "
            + f"{row['q25_weight']:.3f}--{row['q75_weight']:.3f} |"
        )

    median_min = float(dataset_summary["median_weight"].min())
    median_max = float(dataset_summary["median_weight"].max())
    mean_min = float(dataset_summary["mean_weight"].min())
    mean_max = float(dataset_summary["mean_weight"].max())
    observed_min = float(dataset_summary["min_weight"].min())
    observed_max = float(dataset_summary["max_weight"].max())
    try:
        run_label = str(run_root.relative_to(PROJECT_ROOT))
    except ValueError:
        run_label = str(run_root)

    text = rf"""# Appendix figure: transcript--dataset reliability weighting

![Appendix reliability-weight figure](appendix_reliability_weight_example.png)

## Scope and provenance

This figure describes the transcript--dataset reliability weights used by the completed four-panel RiboUnmix experiment `{run_label}`. It is generated from the four run-specific `reliability_reference_manifest.json` files and the scalar `coverage` and `read_density` columns in the 114 weighted HEK Ribo-seq Parquets. No model is loaded and no prediction is recalculated.

The reliability references were fitted from **training transcripts only** within each run and then frozen. The plotted point clouds contain all eligible rows in the corresponding Parquet, including held-out rows transformed using those frozen references. Consequently, held-out data are visualized but did not influence either reference estimate.

This figure concerns the loss weight $w_{{dt}}$. The completed experiment used uniform fixed-reference gamma weights $\pi_d$; $w_{{dt}}$ and $\pi_d$ have different roles and must not be interpreted interchangeably.

## Exact definition

For dataset $d$, transcript $t$, and its stored arithmetic replica-consensus profile $r_{{dti}}$ over $L_t$ modeled codons,

$$
D_{{dt}}=\frac{{1}}{{L_t}}\sum_i r_{{dti}},
\qquad
C_{{dt}}=\frac{{1}}{{L_t}}\sum_i \mathbf{{1}}[r_{{dti}}>0].
$$

Thus, $D_{{dt}}$ is reads per modeled codon and $C_{{dt}}$ is the fraction of modeled codons with positive signal. The dataset-specific depth reference is

$$
\tau_d=\operatorname{{median}}_{{t\in\mathcal{{T}}_{{d,\mathrm{{train}}}}}}D_{{dt}}.
$$

The saturating depth score, raw reliability, and final normalized weight are

$$
q(D_{{dt}};\tau_d)=
\frac{{\sqrt{{D_{{dt}}}}}}{{\sqrt{{D_{{dt}}}}+\sqrt{{\tau_d}}}},
$$

$$
a_{{dt}}=0.70\,q(D_{{dt}};\tau_d)+0.30\,C_{{dt}},
\qquad
w_{{dt}}=\frac{{a_{{dt}}}}{{m_d}},
$$

where $m_d$ is the median $a_{{dt}}$ over the same dataset-specific training-reference rows. At $D_{{dt}}=\tau_d$, the depth component is exactly $q=0.5$. The square-root saturation prevents read-depth outliers from increasing the depth contribution linearly. Final weights are positive, are not clipped to $[0,1]$, and can exceed one.

With transcript-balanced reduction, $w_{{dt}}$ controls the relative contribution of dataset observations *within a transcript*:

$$
\bar{{\ell}}_t=
\frac{{\sum_{{d\in\mathcal{{A}}_t}}w_{{dt}}\ell_{{dt}}}}
{{\sum_{{d\in\mathcal{{A}}_t}}w_{{dt}}}},
\qquad
\mathcal{{L}}=\frac{{1}}{{|\mathcal{{T}}|}}\sum_{{t\in\mathcal{{T}}}}\bar{{\ell}}_t.
$$

This normalization means that $w_{{dt}}$ ranks the trust assigned to the available experimental views of the same transcript; it is not a probability and does not give a transcript with many datasets proportionally more outer loss mass.

## How to read panels A--D

Each blue point is one eligible transcript--dataset pair. The horizontal axis is positive-codon coverage $C_{{dt}}$ and the vertical axis is $\log(1+D_{{dt}})$. The yellow-to-red background and labeled contours show the exact normalized weight obtained from that dataset's frozen training-only $\tau_d$ and $m_d$. Increasing either density or coverage increases the weight, while the depth contribution saturates.

The four examples were selected deterministically, not visually hand-picked. They are the datasets nearest the 12.5th, 37.5th, 62.5th, and 87.5th percentiles of the empirical $\tau_d$ distribution across all 114 datasets.

| Panel | Dataset | target $\tau_d$ quantile | fitted $\tau_d$ | eligible pairs shown | mean $w_{{dt}}$ | median $w_{{dt}}$ | pair-level IQR |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(panel_rows)}

The displayed all-row medians need not equal one exactly because $m_d$ was fitted only on the training-reference rows. Their proximity to one shows that adding the held-out rows does not materially shift the aggregate scale. Because training rows dominate these all-row summaries, this is not a standalone test of held-out calibration and is not evidence that every dataset has identical reliability.

## How to read panel E

Panel E summarizes all 114 transformed weight distributions. Each dataset-specific density integrates to one, and the plotted density is their arithmetic mean. Therefore, each dataset contributes mass $1/114$ regardless of how many eligible transcripts it contains. This prevents large datasets from dominating the descriptive distribution.

- Equal-dataset mean: **{quantiles['equal_dataset_mean']:.3f}**
- Equal-dataset median: **{quantiles['q50']:.3f}**
- Interquartile range: **{quantiles['q25']:.3f}--{quantiles['q75']:.3f}**
- Central 95% descriptive range: **{quantiles['q02']:.3f}--{quantiles['q97']:.3f}**
- Dataset-specific all-row medians: **{median_min:.3f}--{median_max:.3f}**
- Dataset-specific means: **{mean_min:.3f}--{mean_max:.3f}**
- Complete observed range: **{observed_min:.3f}--{observed_max:.3f}**

The dark-blue curve is the equal-dataset mixture density. The blue band is the pointwise 10th--90th percentile of the 114 dataset-specific densities and describes between-dataset heterogeneity; **it is not a confidence interval**. The pale vertical band is the mixture IQR. The red dashed line marks the normalization reference $w_{{dt}}=1$, the blue dotted line marks the empirical mixture median, and the short rug marks the 114 dataset-specific medians.

## What the results show

1. **The weights retain substantial pair-level variation.** Half of the equal-dataset mixture lies between {quantiles['q25']:.3f} and {quantiles['q75']:.3f}; the procedure does not reduce all observations to weight one.
2. **The distribution is mildly right-skewed.** Its mean ({quantiles['equal_dataset_mean']:.3f}) is above its median ({quantiles['q50']:.3f}), reflecting a tail of well-covered, high-depth transcript--dataset pairs with weights above one.
3. **Depth is interpreted relative to each dataset.** Variation in $\tau_d$ changes the position of the same weight contours across panels. There is no single absolute read-density threshold applied to all experiments.
4. **The frozen reference remains numerically consistent on the complete eligible tables.** After applying it without refitting, the 114 all-row dataset medians remain in the narrow range {median_min:.3f}--{median_max:.3f}. This aggregate is still dominated by training rows.

The median near one is primarily imposed by the definition $w_{{dt}}=a_{{dt}}/m_d$. It is a useful implementation and scale-calibration check, but it is **not an empirical discovery**, a measure of model accuracy, or evidence that the datasets have equal biological quality. Similarly, the 70/30 coefficients are coefficients of two differently distributed components; they should not be described as an observed 70% versus 30% decomposition of loss influence.

## What the figure does not establish

- It does not measure predictive performance, shared-profile stability, or biological correctness.
- It does not visualize the fixed-reference gamma weights $\pi_d$ or the dataset-quality ranking.
- High $w_{{dt}}$ means stronger measured support under this reliability rule; it does not validate a stall, motif, or regulatory event.
- The equal-dataset aggregation in panel E is a visualization choice. The training objective remains transcript-balanced as written above.
- The 2.5th--97.5th percentile range is a descriptive population range, not a sampling-confidence interval.

## Reproduction and source data

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
RIBOUNMIX_PLOT_TEX=1 \\
.venv/bin/python analyses/create_appendix_reliability_weight_figure.py \\
  --font-scale {font_scale:g}
```

- [Vector PDF](appendix_reliability_weight_example.pdf)
- [300-dpi PNG](appendix_reliability_weight_example.png)
- [Representative-dataset source](representative_dataset_source.csv)
- [Equal-dataset distribution source](equal_dataset_weight_distribution.csv)
- [Per-dataset weight summaries](per_dataset_weight_summary.csv)
- [Analysis manifest](appendix_reliability_weight_example_manifest.json)
- [LaTeX caption](appendix_reliability_weight_example_caption.tex)
"""
    # Raw strings preserve Markdown/LaTeX commands; shell continuations need
    # one backslash rather than the two shown in the Python source literal.
    text = text.replace("\\\\\n", "\\\n")
    output_path = output_dir / "appendix_reliability_weight_example_explanation.md"
    output_path.write_text(text, encoding="utf-8")
    return output_path


def _render_figure(
    representative_frames: dict[str, pd.DataFrame],
    selected: pd.DataFrame,
    references: dict[str, WeightReference],
    distribution: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    quantiles: dict[str, float],
    *,
    output_stem: Path,
    dpi: int,
    request_tex: bool,
    font_scale: float,
) -> bool:
    style = publication_rc()
    if not request_tex:
        style.update(
            {
                "text.usetex": False,
                "text.latex.preamble": "",
                "font.serif": ["Latin Modern Roman", "DejaVu Serif"],
                "mathtext.fontset": "cm",
            }
        )
    use_tex = bool(style["text.usetex"])
    if not math.isfinite(font_scale) or font_scale <= 0.0:
        raise ValueError("font_scale must be finite and strictly positive.")
    style.update(
        {
            "font.size": 17.0 * font_scale,
            "axes.labelsize": 18.5 * font_scale,
            "axes.titlesize": 17.0 * font_scale,
            "xtick.labelsize": 15.0 * font_scale,
            "ytick.labelsize": 15.0 * font_scale,
            "legend.fontsize": 14.0 * font_scale,
            "legend.title_fontsize": 14.0 * font_scale,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
        }
    )
    if use_tex:
        # Bold the TeX glyphs too, including tick labels and mathematical notation.
        style["text.latex.preamble"] += (
            r"\renewcommand{\seriesdefault}{\bfdefault}\AtBeginDocument{\boldmath}"
        )

    with matplotlib.rc_context(style):
        # Reserve explicit title/legend margins: the staggered axes and shared
        # colorbar otherwise over-constrain Matplotlib's automatic layout.
        figure = plt.figure(figsize=(20.0, 13.0))
        # Four vertical grid units let panel E occupy one scatter-panel height
        # while remaining centered between the upper and lower rows.
        grid = figure.add_gridspec(
            4,
            3,
            width_ratios=(1.0, 1.0, 1.04),
            height_ratios=(1.0, 1.0, 1.0, 1.0),
            left=0.07,
            right=0.98,
            bottom=0.23,
            top=0.81,
            wspace=0.38,
            hspace=1.6,
        )
        surface_axes = [
            figure.add_subplot(grid[0:2, 0]),
            figure.add_subplot(grid[0:2, 1]),
            figure.add_subplot(grid[2:4, 0]),
            figure.add_subplot(grid[2:4, 1]),
        ]
        distribution_legend_axis = figure.add_axes((0.73, 0.70, 0.25, 0.18))
        distribution_legend_axis.set_axis_off()
        distribution_axis = figure.add_axes((0.73, 0.32, 0.25, 0.25))
        weight_norm = Normalize(vmin=0.0, vmax=WEIGHT_LIMIT)

        for panel_index, (_, selection) in enumerate(selected.iterrows()):
            axis = surface_axes[panel_index]
            dataset = str(selection["dataset"])
            frame = representative_frames[dataset]
            reference = references[dataset]
            observed_y = frame["log_read_density"].to_numpy(dtype=np.float64, copy=False)
            y_max = max(float(observed_y.max()), 0.5)
            y_upper = y_max + max(0.05 * y_max, 0.05)
            coverage_grid, density_grid, weight_grid = _weight_surface(reference, y_upper)
            if float(weight_grid.max()) > WEIGHT_LIMIT:
                raise ValueError(
                    f"{dataset}: weight surface exceeds the fixed publication color "
                    f"limit {WEIGHT_LIMIT:g}."
                )
            axis.imshow(
                weight_grid,
                extent=(0.0, 1.0, 0.0, y_upper),
                origin="lower",
                aspect="auto",
                cmap=WEIGHT_CMAP,
                norm=weight_norm,
                alpha=0.42,
                interpolation="bilinear",
                rasterized=True,
                zorder=0,
            )
            levels = np.asarray([0.5, 1.0, 1.5, 2.0, 2.5])
            levels = levels[
                (levels > float(weight_grid.min())) & (levels < float(weight_grid.max()))
            ]
            if levels.size:
                contours = axis.contour(
                    coverage_grid,
                    density_grid,
                    weight_grid,
                    levels=levels,
                    colors="#4A3B36",
                    linewidths=0.8,
                    alpha=0.55,
                    zorder=1,
                )
                axis.clabel(
                    contours,
                    fmt=r"$w=%.1f$",
                    fontsize=12.0 * font_scale,
                    inline=True,
                )
            axis.scatter(
                frame["coverage"],
                frame["log_read_density"],
                s=2.2,
                color="#1F5A7A",
                alpha=0.42,
                linewidths=0,
                rasterized=True,
                zorder=2,
            )
            weights = frame["weight"].to_numpy(dtype=np.float64, copy=False)
            axis.set_title(
                _tex_escape(dataset, use_tex)
                + "\n"
                + rf"$\tau_d={reference.tau:.3g}$; "
                + rf"median $w_{{dt}}={np.median(weights):.3f}$"
            )
            axis.set_xlim(0.0, 1.0)
            axis.set_ylim(0.0, y_upper)
            if panel_index % 2 == 0:
                axis.set_ylabel(r"$\log(1+D_{dt})$")
            if panel_index >= 2:
                axis.set_xlabel(r"Positive-codon coverage $C_{dt}$")
            axis.text(
                -0.16,
                1.05,
                chr(ord("A") + panel_index),
                transform=axis.transAxes,
                fontsize=23.0 * font_scale,
                fontweight="bold",
                va="top",
            )

        source = distribution
        distribution_axis.fill_between(
            source["weight_bin_center"].to_numpy(),
            source["dataset_density_q10"].to_numpy(),
            source["dataset_density_q90"].to_numpy(),
            color="#87B7D1",
            alpha=0.30,
            linewidth=0,
            label="10th--90th percentile\nacross datasets",
        )
        distribution_axis.plot(
            source["weight_bin_center"],
            source["equal_dataset_mean_density"],
            color="#145A7A",
            linewidth=2.5,
            label="Equal-dataset mixture",
        )
        distribution_axis.axvspan(
            quantiles["q25"],
            quantiles["q75"],
            color="#145A7A",
            alpha=0.09,
            label="Mixture IQR",
        )
        distribution_axis.axvline(
            1.0,
            color="#A12622",
            linestyle="--",
            linewidth=2.0,
            label="Normalization reference\n" + r"$w_{dt}=1$",
        )
        distribution_axis.axvline(
            quantiles["q50"],
            color="#145A7A",
            linestyle=":",
            linewidth=2.0,
            label=rf"Mixture median $={quantiles['q50']:.3f}$",
        )
        y_max = float(source["equal_dataset_mean_density"].max())
        distribution_axis.scatter(
            dataset_summary["median_weight"],
            np.full(len(dataset_summary), -0.025 * y_max),
            marker="|",
            s=55,
            linewidths=0.8,
            color="#494949",
            alpha=0.55,
            clip_on=False,
            label="114 dataset medians",
        )
        distribution_axis.set_xlim(0.0, float(source["weight_bin_right"].max()))
        distribution_axis.set_ylim(bottom=-0.055 * y_max)
        distribution_axis.set_xlabel("Normalized reliability\n" + r"weight $w_{dt}$")
        distribution_axis.set_ylabel("Probability density")
        distribution_axis.set_title(
            "Weight distribution\n"
            "each dataset contributes\n" + r"mass $1/114$",
            pad=10,
        )
        legend_handles, legend_labels = distribution_axis.get_legend_handles_labels()
        distribution_legend_axis.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            borderaxespad=0.0,
        )
        distribution_axis.text(
            -0.20,
            1.12,
            "E",
            transform=distribution_axis.transAxes,
            fontsize=23.0 * font_scale,
            fontweight="bold",
            # Grow upward into the deliberately unused margin.  This keeps a
            # large panel letter clear of the equally enlarged y-axis label.
            va="bottom",
        )

        color_mappable = ScalarMappable(norm=weight_norm, cmap=WEIGHT_CMAP)
        color_mappable.set_array([])
        colorbar = figure.colorbar(
            color_mappable,
            cax=figure.add_axes((0.07, 0.10, 0.575, 0.022)),
            orientation="horizontal",
        )
        colorbar.set_label(r"Normalized reliability weight $w_{dt}$")
        figure.suptitle(
            "Transcript--dataset reliability weighting\n"
            r"$w_{dt}=\left(0.70\,\frac{\sqrt{D_{dt}}}"
            r"{\sqrt{D_{dt}}+\sqrt{\tau_d}}+0.30\,C_{dt}\right)/m_d$",
            fontsize=22.0 * font_scale,
        )
        figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
        figure.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
        plt.close(figure)
    return use_tex


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIRECTORY)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--selection-quantiles",
        type=float,
        nargs=4,
        default=DEFAULT_SELECTION_QUANTILES,
        metavar=("Q1", "Q2", "Q3", "Q4"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--font-scale",
        type=float,
        default=DEFAULT_FONT_SCALE,
        help=(
            "Multiplier for all bold publication text relative to the base "
            f"style (default: {DEFAULT_FONT_SCALE:g})."
        ),
    )
    parser.add_argument("--no-tex", action="store_true")
    #parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_root = args.run_root.resolve()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else artifact_directory(
            "real_data", run_root, "reliability_weight_audit", "appendix_figure"
        )
    )
    manifest_paths = sorted(run_root.glob("panel_*/reliability_reference_manifest.json"))
    if len(manifest_paths) != 4:
        raise FileNotFoundError(
            f"Expected four panel reliability manifests below {run_root}; "
            f"found {len(manifest_paths)}."
        )
    references = _load_experiment_references(manifest_paths)
    if len(references) != 114:
        raise ValueError(f"Expected 114 unique dataset references; found {len(references)}.")
    if any(reference.reference_split != "training_only" for reference in references.values()):
        raise ValueError("Every reference must have reference_split=training_only.")

    available = {path.stem: path for path in dataset_dir.glob("*.parquet")}
    missing = sorted(set(references).difference(available))
    if missing:
        raise FileNotFoundError(f"Missing dataset Parquets: {missing[:8]}")
    selected = select_reference_quantile_datasets(references, args.selection_quantiles)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / "appendix_reliability_weight_example"
    expected_outputs = [output_stem.with_suffix(suffix) for suffix in (".pdf", ".png")]
    # if not args.overwrite and any(path.exists() for path in expected_outputs):
    #     raise FileExistsError(
    #         f"Appendix figure already exists in {output_dir}; use --overwrite."
    #     )

    weights_by_dataset: dict[str, np.ndarray] = {}
    representative_frames: dict[str, pd.DataFrame] = {}
    selected_names = set(selected["dataset"].astype(str))
    for index, dataset in enumerate(sorted(references), start=1):
        frame = _read_runtime_weight_frame(available[dataset], dataset, references[dataset])
        weights_by_dataset[dataset] = frame["weight"].to_numpy(dtype=np.float64, copy=True)
        if dataset in selected_names:
            representative_frames[dataset] = frame
        if index % 20 == 0 or index == len(references):
            print(f"Loaded scalar reliability data for {index}/114 datasets.")

    distribution, dataset_summary, quantile_summary = _distribution_source(
        weights_by_dataset
    )
    selected = selected.merge(
        dataset_summary,
        on="dataset",
        how="left",
        validate="one_to_one",
    )
    selected.to_csv(output_dir / "representative_dataset_source.csv", index=False)
    distribution.to_csv(output_dir / "equal_dataset_weight_distribution.csv", index=False)
    dataset_summary.to_csv(output_dir / "per_dataset_weight_summary.csv", index=False)

    request_tex = not args.no_tex
    latex_used = _render_figure(
        representative_frames,
        selected,
        references,
        distribution,
        dataset_summary,
        quantile_summary,
        output_stem=output_stem,
        dpi=args.dpi,
        request_tex=request_tex,
        font_scale=args.font_scale,
    )

    caption = (
        "Transcript--dataset reliability weighting in the completed uniform-reference "
        "four-panel experiment. Panels A--D show observed transcript--dataset pairs "
        "over the current SNR-depth/coverage weight surface for four datasets selected "
        "deterministically at the 12.5th, 37.5th, 62.5th and 87.5th percentiles of "
        "the dataset-specific training-only depth reference tau_d. Panel E shows an "
        "equal-dataset mixture of the 114 weight distributions; the band is the "
        "10th--90th percentile of dataset-specific densities. The normalization "
        "reference w_dt=1 is imposed by the within-dataset training-reference median "
        "and should be interpreted as a scale convention, not an empirical result. "
        "These reliability weights enter the training loss and are distinct from the "
        "uniform gamma-reference weights pi_d."
    )
    (output_dir / "appendix_reliability_weight_example_caption.txt").write_text(
        caption + "\n", encoding="utf-8"
    )
    latex_caption = (
        caption.replace("tau_d", r"$\tau_d$")
        .replace("w_dt", r"$w_{dt}$")
        .replace("pi_d", r"$\pi_d$")
        .replace("10th--90th", r"10th--90th")
    )
    (output_dir / "appendix_reliability_weight_example_caption.tex").write_text(
        latex_caption + "\n", encoding="utf-8"
    )
    explanation_path = _write_markdown_explanation(
        output_dir=output_dir,
        run_root=run_root,
        selected=selected,
        dataset_summary=dataset_summary,
        quantiles=quantile_summary,
        font_scale=args.font_scale,
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(SCRIPT_PATH),
        "run_root": str(run_root),
        "dataset_directory": str(dataset_dir),
        "reference_manifests": [str(path.resolve()) for path in manifest_paths],
        "weighting_mode": WEIGHTING_MODE,
        "reference_split": "training_only",
        "dataset_count": 114,
        "dataset_mixture_weighting": "equal mass per dataset",
        "selection_rule": "nearest tau_d to requested empirical quantiles",
        "selection_quantiles": list(map(float, args.selection_quantiles)),
        "selected_datasets": selected.to_dict(orient="records"),
        "distribution_quantiles": quantile_summary,
        "latex_rendering_requested": request_tex,
        "latex_rendering_used": latex_used,
        "font_scale": float(args.font_scale),
        "font_weight": "bold",
        "outputs": [str(path) for path in expected_outputs] + [str(explanation_path)],
        "scientific_caveat": (
            "Median weight near one is induced by within-dataset normalization; "
            "it is a scale check rather than an empirical discovery."
        ),
    }
    (output_dir / "appendix_reliability_weight_example_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(selected[["selection_quantile", "dataset", "depth_reference_tau"]].to_string(index=False))
    print(f"Equal-dataset mixture quantiles: {quantile_summary}")
    print(f"Wrote appendix figure and source files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
