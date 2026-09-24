#!/usr/bin/env python3
"""Write the data-driven methodological explanation for Figure 1."""

from __future__ import annotations

from datetime import datetime, timezone
import html
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"
SOURCE_DIR = FIGURE_DIR / "real_data_equal_source"
PANEL_ROOT = ROOT / "results/my_panels_a100_b32_20260906_114323"
STABILITY_ROOT = ROOT / "results/my_exp8_a100_b32_20260906_114340"
OUTPUT = FIGURE_DIR / "real_data_equal_explanation.html"


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def table(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, border=0, classes="data-table", escape=False)


def panel_membership_details(panel_manifest: dict) -> str:
    blocks = []
    for index in range(1, 5):
        key = f"panel_{index:02d}"
        datasets = list(map(str, panel_manifest["panels"][key]))
        sources = list(map(str, panel_manifest["panel_source_families"][key]))
        blocks.append(
            f"""
            <details>
              <summary>P{index}: {len(datasets)} datasets from {len(sources)} source families</summary>
              <p><strong>Datasets:</strong> {html.escape(', '.join(datasets))}</p>
              <p><strong>Source families:</strong> {html.escape(', '.join(sources))}</p>
            </details>
            """
        )
    return "\n".join(blocks)


def main() -> None:
    panel = pd.read_csv(SOURCE_DIR / "panel_a_summary.csv")
    same_N = pd.read_csv(SOURCE_DIR / "panel_b_N_summary.csv")
    N80 = pd.read_csv(SOURCE_DIR / "panel_b_N80_summary.csv")
    N80_pairs = pd.read_csv(SOURCE_DIR / "panel_b_N80_pair_summary.csv")
    adjacent = pd.read_csv(SOURCE_DIR / "panel_b_adjacent_size_summary.csv")
    adjacent_pairs = pd.read_csv(SOURCE_DIR / "panel_b_adjacent_size_pair_summary.csv")
    run_provenance = pd.read_csv(SOURCE_DIR / "run_provenance.csv")
    raw_audit = pd.read_csv(SOURCE_DIR / "raw_repeat_audit.csv")
    cohorts = read_json(SOURCE_DIR / "cohort_provenance.json")
    panel_manifest = read_json(PANEL_ROOT / "panel_manifest.json")
    experiment_manifest = read_json(STABILITY_ROOT / "experiment_manifest.json")

    expected = {
        "panel rows": (len(panel), 6),
        "same-N rows": (len(same_N), 5),
        "N80 pairs": (len(N80_pairs), 3),
        "adjacent transitions": (len(adjacent), 6),
    }
    failures = [name for name, values in expected.items() if values[0] != values[1]]
    if failures:
        raise ValueError(f"Incomplete Figure 1 source tables: {failures}.")

    panel_display = panel[
        ["panel_pair", "median_PCC", "p25_PCC", "p75_PCC", "p05_PCC", "p95_PCC"]
    ].copy()
    panel_display.columns = ["Pair", "Median", "P25", "P75", "P05", "P95"]
    for column in panel_display.columns[1:]:
        panel_display[column] = panel_display[column].map(lambda value: f"{value:.3f}")

    same_display = same_N[
        ["N", "number_of_designated_pairs", "R_N_mean_over_pair_means",
         "minimum_pair_mean_PCC", "maximum_pair_mean_PCC"]
    ].copy()
    same_display.columns = ["N", "Disjoint A/B pairs", "Mean PCC", "Minimum pair mean", "Maximum pair mean"]
    for column in same_display.columns[2:]:
        same_display[column] = same_display[column].map(lambda value: f"{value:.3f}")

    N80_display = N80_pairs[
        ["run_a", "run_b", "mean_transcript_PCC", "dataset_intersection",
         "dataset_jaccard", "source_family_intersection", "source_family_jaccard"]
    ].copy()
    N80_display.columns = [
        "Subset A", "Subset B", "Mean PCC", "Shared datasets", "Dataset Jaccard",
        "Shared source families", "Source-family Jaccard",
    ]
    N80_display["Mean PCC"] = N80_display["Mean PCC"].map(lambda value: f"{value:.3f}")
    N80_display["Dataset Jaccard"] = N80_display["Dataset Jaccard"].map(lambda value: f"{value:.3f}")
    N80_display["Source-family Jaccard"] = N80_display["Source-family Jaccard"].map(
        lambda value: f"{value:.3f}"
    )

    source_overlap = (
        adjacent_pairs.groupby(["N_from", "N_to"], sort=False)
        .agg(
            mean_source_jaccard=("source_family_jaccard", "mean"),
            min_source_jaccard=("source_family_jaccard", "min"),
            max_source_jaccard=("source_family_jaccard", "max"),
        )
        .reset_index()
    )
    adjacent_detail = adjacent.merge(source_overlap, on=["N_from", "N_to"], validate="one_to_one")
    adjacent_display = pd.DataFrame(
        {
            "Transition": adjacent_detail["transition"],
            "Plotted at N": adjacent_detail["plot_x_larger_endpoint"].astype(int),
            "Model pairs": adjacent_detail["number_of_model_pairs"].astype(int),
            "Mean PCC": adjacent_detail["mean_over_model_pair_means"].map(lambda value: f"{value:.3f}"),
            "Pair-mean range": adjacent_detail.apply(
                lambda row: f"{row.minimum_model_pair_mean_PCC:.3f}–{row.maximum_model_pair_mean_PCC:.3f}", axis=1
            ),
            "Dataset Jaccard, mean (range)": adjacent_detail.apply(
                lambda row: (
                    f"{row.mean_dataset_jaccard:.3f} "
                    f"({row.minimum_dataset_jaccard:.3f}–{row.maximum_dataset_jaccard:.3f})"
                ), axis=1
            ),
            "Source Jaccard, mean (range)": adjacent_detail.apply(
                lambda row: (
                    f"{row.mean_source_jaccard:.3f} "
                    f"({row.min_source_jaccard:.3f}–{row.max_source_jaccard:.3f})"
                ), axis=1
            ),
        }
    )

    construction = (
        run_provenance.groupby(["figure_panel", "N"], sort=False)
        .agg(
            models=("run_id", "size"),
            datasets_per_model=("number_of_datasets", "first"),
            minimum_source_families=("number_of_source_families", "min"),
            maximum_source_families=("number_of_source_families", "max"),
        )
        .reset_index()
    )
    construction["Source families/model"] = construction.apply(
        lambda row: (
            str(int(row.minimum_source_families))
            if row.minimum_source_families == row.maximum_source_families
            else f"{int(row.minimum_source_families)}–{int(row.maximum_source_families)}"
        ), axis=1
    )
    construction_display = construction[
        ["figure_panel", "N", "models", "datasets_per_model", "Source families/model"]
    ].copy()
    construction_display.columns = ["Figure role", "N", "Models", "Datasets/model", "Source families/model"]

    panel_a_test = int(cohorts["panel_A"]["common_test_count"])
    panel_b_test = int(cohorts["panel_B"]["common_test_count"])
    panel_rows = panel_a_test * 6
    same_rows = panel_b_test * 15
    N80_rows = panel_b_test * 3
    adjacent_rows = panel_b_test * int(adjacent.number_of_model_pairs.sum())
    repeated = raw_audit.loc[raw_audit.raw_export_mode == "repeated_dataset_rows"]
    sequence_only = raw_audit.loc[raw_audit.raw_export_mode == "sequence_only_one_row_per_transcript"]

    task_count = len(experiment_manifest.get("tasks", []))
    documented_seed = int(panel_manifest["random_seed"])
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Figure 1 — construction, estimands, and interpretation</title>
  <style>
    :root {{ --ink:#1f2933; --muted:#5c6770; --blue:#0b5a8c; --pale:#eef4f8; --line:#d8e0e6; --warn:#fff7df; }}
    body {{ margin:0; color:var(--ink); background:#fff; font-family:"Latin Modern Roman","Computer Modern Serif","STIX Two Text",Georgia,serif; line-height:1.55; }}
    main {{ max-width:1120px; margin:0 auto; padding:38px 34px 64px; }}
    h1 {{ font-size:2rem; line-height:1.15; margin:0 0 8px; }}
    h2 {{ margin-top:2.2rem; padding-bottom:.3rem; border-bottom:1px solid var(--line); font-size:1.45rem; }}
    h3 {{ margin-top:1.6rem; font-size:1.12rem; }}
    p, li {{ font-size:1rem; }}
    .subtitle {{ color:var(--muted); margin-top:0; }}
    .figure {{ margin:24px 0; padding:14px; border:1px solid var(--line); border-radius:8px; }}
    .figure img {{ display:block; width:100%; height:auto; }}
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; margin:18px 0; }}
    .card {{ background:var(--pale); border-left:4px solid var(--blue); padding:12px 14px; border-radius:4px; }}
    .card strong {{ display:block; font-size:1.25rem; color:var(--blue); }}
    .equation {{ margin:14px 0; padding:12px 16px; background:#f7f9fa; border:1px solid var(--line); font-family:"Latin Modern Mono",monospace; overflow-x:auto; }}
    .warning {{ background:var(--warn); border-left:4px solid #b7791f; padding:12px 15px; margin:16px 0; }}
    .data-table {{ width:100%; border-collapse:collapse; margin:14px 0 22px; font-variant-numeric:tabular-nums; font-size:.91rem; }}
    .data-table th {{ background:#eef2f5; text-align:left; }}
    .data-table th, .data-table td {{ border-bottom:1px solid var(--line); padding:7px 8px; vertical-align:top; }}
    details {{ border:1px solid var(--line); border-radius:5px; padding:8px 12px; margin:8px 0; }}
    summary {{ cursor:pointer; font-weight:bold; color:var(--blue); }}
    code {{ background:#f3f5f6; padding:.1em .3em; border-radius:3px; }}
    a {{ color:var(--blue); }}
    .small {{ color:var(--muted); font-size:.88rem; }}
  </style>
</head>
<body><main>
  <h1>Figure 1: reproducibility and stability across dataset scales</h1>
  <p class="subtitle">Precise construction, numerical results, and interpretation of <code>real_data_equal.pdf</code>. Generated {generated} from the saved source tables and experiment manifests.</p>

  <div class="figure"><img src="real_data_equal.png" alt="Two-panel Figure 1 showing cross-panel reproducibility and stability across dataset scales."></div>

  <div class="cards">
    <div class="card"><strong>{panel_a_test:,}</strong>Panel A held-out transcripts</div>
    <div class="card"><strong>{panel_b_test:,}</strong>Panel B held-out transcripts</div>
    <div class="card"><strong>42</strong>Single documented training seed</div>
    <div class="card"><strong>π<sub>d</sub> = 1/N</strong>Uniform reference weights in every displayed model</div>
  </div>

  <h2>1. Scientific question and common metric</h2>
  <p>The figure asks whether the dataset-independent shared profile inferred by RiboUnmix is reproducible when the training datasets change, and how that agreement behaves as the number of training datasets increases. It does <strong>not</strong> test recovery of an unavailable biological ground truth.</p>
  <p>For transcript <em>t</em> and two frozen models <em>a</em> and <em>b</em>, the basic observation is one Pearson correlation across aligned full-CDS codon positions:</p>
  <div class="equation">r<sub>t</sub>(a,b) = PCC(L<sub>t</sub><sup>(a)</sup>, L<sub>t</sub><sup>(b)</sup>)</div>
  <p>Each <em>L</em><sub>t</sub> is the model's dataset-independent, mean-one shared profile. Positions were not smoothed, filtered according to measured zeros, pooled across transcripts, or renormalized after export. Correlations were computed transcript by transcript. Missing, misaligned, constant, or nearly constant profiles would remain undefined rather than being replaced by zero.</p>
  <p>All displayed models use fixed-reference centering with uniform <em>gamma</em>-reference weights π<sub>d</sub>=1/N. This does not remove the original transcript–dataset reliability weights <em>w</em><sub>dt</sub>: those remain fitted from training data only and are explicitly separate from π.</p>

  <h2>2. How Panel A was constructed</h2>
  <h3>Panel selection</h3>
  <p>The 114-dataset collection was partitioned into four panels P1–P4 using seed {documented_seed}. They contain 29, 29, 28, and 28 datasets, drawn from 20, 21, 22, and 22 source families. Dataset identities are disjoint across panels, and source-family identities are also disjoint. This is stricter than merely training four models on different rows from the same studies: it prevents a source family from appearing in both models of any Panel A comparison.</p>
  <p>The evaluation cohort is the fixed intersection of {panel_a_test:,} held-out transcripts, with identity hash <code>{html.escape(cohorts['panel_A']['common_test_hash'])}</code>. The six rows are all unordered panel pairs, shown in a fixed, non-outcome-selected order.</p>
  {panel_membership_details(panel_manifest)}

  <h3>Displayed statistic</h3>
  <p>For each of the six model pairs, Panel A retains all {panel_a_test:,} transcript-level PCCs. The dot is the median; the thick interval is P25–P75; the thin interval is P05–P95. These are distribution intervals describing heterogeneity among transcripts—not confidence intervals for the median.</p>
  {table(panel_display)}
  <p>The pairwise medians range from {panel.median_PCC.min():.3f} to {panel.median_PCC.max():.3f}. The wider lower tails, especially P1–P4 and P2–P4, show that a high median does not imply uniform reproducibility for every transcript.</p>

  <h2>3. Why Panel B has three visual encodings</h2>
  <p>Panel B uses a different fixed cohort of {panel_b_test:,} transcripts, hash <code>{html.escape(cohorts['panel_B']['common_test_hash'])}</code>. The cohorts are kept separate because the two experiments used different saved split manifests. All {task_count} planned Exp8 tasks use seed 42; only the models required by the three estimands below are displayed.</p>
  {table(construction_display)}

  <h3>3.1 Filled blue line: genuinely disjoint same-N stability</h3>
  <p>At each N=2, 5, 10, 20, and 40, the experiment pre-designated three A/B pairs. Within each pair, both dataset identity and source-family identity are disjoint. Different pairs are design replicates and may overlap with one another.</p>
  <div class="equation">S[N,p] = mean<sub>t</sub> r<sub>t</sub>(N,p,A; N,p,B), &nbsp;&nbsp; S[N] = (1/3) Σ<sub>p=1</sub><sup>3</sup> S[N,p]</div>
  <p>Small light points are the three S[N,p] values. Large blue points are S[N], and only these aggregates are connected. The curve therefore does not pretend that pair01 at one N is a nested continuation of pair01 at another N.</p>
  {table(same_display)}

  <h3>3.2 Open square at N=80: overlapping same-N stability</h3>
  <p>Two disjoint 80-dataset subsets cannot be drawn from a universe of 114 datasets. The three saved N=80 subsets therefore overlap substantially. Their three pairwise mean transcript PCCs are {N80_pairs.mean_transcript_PCC.min():.3f}, {N80_pairs.mean_transcript_PCC.sort_values().iloc[1]:.3f}, and {N80_pairs.mean_transcript_PCC.max():.3f}; their unweighted mean is {float(N80.mean_over_pair_means.iloc[0]):.3f}. The plot uses one disconnected hollow square so this result is visible but cannot be mistaken for an extension of the disjoint blue curve.</p>
  {table(N80_display)}
  <div class="warning"><strong>Interpretation:</strong> the N=80 value is partly advantaged by shared training datasets and source families. Its PCC cannot be compared causally with the disjoint N≤40 values.</div>

  <h3>3.3 Gray dashed line: agreement between adjacent dataset scales</h3>
  <p>The second line answers a different question: how similar are models trained at adjacent available dataset counts? The transitions are 2–5, 5–10, 10–20, 20–40, 40–80, and 80–114. Across the 24 same-labelled comparisons from 2→5 through 20→40, none of the smaller dataset sets is nested inside its nominally corresponding larger set. Choosing only matching labels would therefore create an arbitrary pseudo-trajectory.</p>
  <p>Instead, every available Cartesian model pairing is used at each transition. For transition a→b:</p>
  <div class="equation">A[a,b] = (1 / (m<sub>a</sub>m<sub>b</sub>)) Σ<sub>i=1</sub><sup>m<sub>a</sub></sup> Σ<sub>j=1</sub><sup>m<sub>b</sub></sup> mean<sub>t</sub> r<sub>t</sub>(a,i; b,j)</div>
  <p>This gives 36 model pairs for each transition through 20–40, 18 for 40–80, and 3 for 80–114. Each gray diamond is the unweighted mean of the corresponding model-pair means. It is plotted at the larger endpoint: the diamond at x=114 therefore means “80 versus 114,” not same-N stability at 114. Individual cross-size model pairs are not drawn because 165 dependent points would obscure the estimand; their complete values and spreads remain in the source tables.</p>
  {table(adjacent_display)}
  <div class="warning"><strong>Important confounder:</strong> mean dataset Jaccard overlap rises from {adjacent.mean_dataset_jaccard.iloc[0]:.3f} for 2–5 to {adjacent.mean_dataset_jaccard.iloc[-1]:.3f} for 80–114. The last transition compares each 80-dataset subset with the 114-dataset collection that contains all 80 of those datasets. The gray line is descriptive cross-scale agreement, not an isolated effect of increasing N.</div>

  <h2>4. What the numerical pattern supports</h2>
  <ul>
    <li><strong>Source-disjoint reproducibility:</strong> the Panel A median PCCs of {panel.median_PCC.min():.3f}–{panel.median_PCC.max():.3f} indicate substantial recovery of common profile structure across entirely different source families.</li>
    <li><strong>Disjoint same-N stability is not monotonic:</strong> S[N] increases from {same_N.R_N_mean_over_pair_means.iloc[0]:.3f} at N=2 to {same_N.R_N_mean_over_pair_means.iloc[2]:.3f} at N=10, then is {same_N.R_N_mean_over_pair_means.iloc[3]:.3f} and {same_N.R_N_mean_over_pair_means.iloc[4]:.3f} at N=20 and 40.</li>
    <li><strong>Adjacent-scale agreement becomes moderately high:</strong> A[a,b] is {adjacent.mean_over_model_pair_means.iloc[0]:.3f} for 2–5 and {adjacent.mean_over_model_pair_means.iloc[-1]:.3f} for 80–114, but the trend contains a decrease at 20–40 and is increasingly confounded by overlap.</li>
    <li><strong>The N=80 point is informative but not exchangeable:</strong> its overlapping same-N mean of {float(N80.mean_over_pair_means.iloc[0]):.3f} should be reported with its distinct marker and overlap disclosure.</li>
  </ul>
  <p>The figure supports reproducibility of frozen inferred profiles. It does not by itself establish biological correctness, prove that more datasets always improve recovery, or isolate the causal effect of N from dataset composition, overlap, and single-seed optimization variability.</p>

  <h2>5. Validation and exclusions</h2>
  <ul>
    <li>Panel A: {panel_rows:,} valid transcript-pair PCCs ({panel_a_test:,} transcripts × 6 model pairs), zero undefined.</li>
    <li>Disjoint Panel B: {same_rows:,} valid PCCs ({panel_b_test:,} transcripts × 15 designated pairs), zero undefined.</li>
    <li>Overlapping N=80: {N80_rows:,} valid PCCs, zero undefined.</li>
    <li>Adjacent sizes: {adjacent_rows:,} valid PCCs ({panel_b_test:,} transcripts × {int(adjacent.number_of_model_pairs.sum())} model pairs), zero undefined.</li>
    <li>The four Panel A raw exports contained {int(repeated.raw_rows.sum()):,} rows. Before deduplication, {int(repeated.repeated_rows_checked.sum()):,} repeated rows were checked and had bitwise-identical shared profiles within transcript.</li>
    <li>The {len(sequence_only)} Panel B sequence-only exports contained {int(sequence_only.raw_rows.sum()):,} rows, exactly one per transcript per model. Compact exports exactly matched their raw float32 profiles.</li>
  </ul>

  <h2>6. Reproducibility files</h2>
  <ul>
    <li><a href="real_data_equal.pdf">Vector figure PDF</a> and <a href="real_data_equal.png">600-dpi PNG</a></li>
    <li><a href="real_data_equal.tex">LaTeX caption</a></li>
    <li><a href="real_data_equal_source/panel_a_per_transcript.csv">Panel A per-transcript PCCs</a></li>
    <li><a href="real_data_equal_source/panel_b_per_transcript.csv">Disjoint same-N per-transcript PCCs</a></li>
    <li><a href="real_data_equal_source/panel_b_N80_pair_summary.csv">N=80 overlap and agreement table</a></li>
    <li><a href="real_data_equal_source/panel_b_adjacent_size_pair_summary.csv">All 165 adjacent-size model-pair summaries</a></li>
    <li><a href="real_data_equal_source/panel_b_adjacent_size_per_transcript.csv">All adjacent-size per-transcript PCCs</a></li>
    <li><a href="real_data_equal_source/PROVENANCE_AND_EXCLUSIONS.md">Provenance and exclusions report</a></li>
    <li><a href="../results/my_panels_a100_b32_20260906_114323/panel_manifest.json">Four-panel manifest</a> and <a href="../results/my_exp8_a100_b32_20260906_114340/experiment_manifest.json">Exp8 task manifest</a></li>
    <li><a href="../analyses/create_real_data_equal_figure.py">Figure-generation script</a> and <a href="../analyses/create_real_data_equal_explanation.py">this HTML generator</a></li>
  </ul>
  <p class="small">No new training or checkpoint inference was performed for this explanation. All numbers were read from the frozen-output analysis tables generated from the saved experiment artifacts.</p>
</main></body></html>
"""
    OUTPUT.write_text(html_text, encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
