"""Summarize transcript counts and transcript weights per HEK dataset.

Reads every ``*.parquet`` in a weighted HEK replica directory
(``Datasets/data/weighted_HEK_riboseq_codon_replicas`` by default) and produces:

  1. A bar plot of the number of transcripts (rows) in each dataset.
  2. Per-dataset mean/median analysis of the transcript ``weight`` column (the
     normalized training weight) and of ``weight_raw`` (the pre-normalization
     local reliability score under the artifact's recorded weighting mode).
  3. A ``dataset_weight_summary.csv`` with the full per-dataset statistics.

Note on the normalized ``weight``: it is built as ``weight_raw / dataset_median``,
so its *median* is ~1.0 for every dataset by construction and carries no
cross-dataset signal. The discriminative quantities are the *mean* normalized
weight (reflects right-skew/spread) and the raw weight. Both are plotted.

Usage:
    python3 Utils/analyze_dataset_transcript_weights.py
    python3 Utils/analyze_dataset_transcript_weights.py --data-dir <dir> --outdir <dir>
    python3 Utils/analyze_dataset_transcript_weights.py --no-plots   # CSV/stdout only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

WEIGHT_COLUMN = "weight"
RAW_WEIGHT_COLUMN = "weight_raw"
ACCENT = "#4f46e5"       # bars
ACCENT_RAW = "#0284c7"   # raw-weight bars
MARKER = "#c2410c"       # median markers / reference lines


def collect_dataset_statistics(data_dir: Path) -> pd.DataFrame:
    """Return one row of transcript/weight statistics per dataset parquet."""
    parquet_paths = sorted(data_dir.glob("*.parquet"))
    if not parquet_paths:
        raise SystemExit(f"No parquet files found in {data_dir}")

    records: list[dict[str, object]] = []
    for path in parquet_paths:
        # Read the schema/metadata only, then load just the light weight columns
        # (avoids materializing the large per-codon list columns).
        parquet_file = pq.ParquetFile(path)
        n_rows = int(parquet_file.metadata.num_rows)
        available = set(parquet_file.schema_arrow.names)
        use_cols = [c for c in (WEIGHT_COLUMN, RAW_WEIGHT_COLUMN) if c in available]
        df = pd.read_parquet(path, columns=use_cols) if use_cols else pd.DataFrame()

        record: dict[str, object] = {
            "dataset": path.stem,
            "n_transcripts": n_rows,
        }
        for label, column in (("weight", WEIGHT_COLUMN), ("weight_raw", RAW_WEIGHT_COLUMN)):
            if column in df.columns:
                values = df[column].astype("float64")
                record[f"{label}_mean"] = float(values.mean())
                record[f"{label}_median"] = float(values.median())
                record[f"{label}_std"] = float(values.std())
                record[f"{label}_min"] = float(values.min())
                record[f"{label}_max"] = float(values.max())
        records.append(record)

    summary = pd.DataFrame.from_records(records)
    return summary.sort_values("n_transcripts", ascending=False).reset_index(drop=True)


def _bar_figure_height(n: int) -> float:
    """Tall enough that ~100+ horizontal category labels stay legible."""
    return max(4.0, 0.22 * n)


def plot_transcript_counts(summary: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = summary.sort_values("n_transcripts", ascending=True)
    n = len(ordered)
    fig, ax = plt.subplots(figsize=(9.0, _bar_figure_height(n)))
    ax.barh(ordered["dataset"], ordered["n_transcripts"], color=ACCENT, height=0.72)

    median_count = ordered["n_transcripts"].median()
    ax.axvline(median_count, color=MARKER, linestyle="--", linewidth=1.0,
               label=f"median = {median_count:,.0f}")
    ax.set_xlabel("Number of transcripts (rows in parquet)")
    ax.set_title(f"Transcripts per HEK dataset (n={n} datasets)")
    ax.tick_params(axis="y", labelsize=7)
    ax.margins(y=0.005)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_weight_analysis(
    summary: pd.DataFrame,
    out_path: Path,
    *,
    label: str,
    bar_color: str,
    annotate_flat_median: bool,
) -> None:
    """Horizontal mean bars with an overlaid median marker, per dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mean_col, median_col = f"{label}_mean", f"{label}_median"
    if mean_col not in summary.columns:
        print(f"[skip] column '{mean_col}' not present; skipping {label} plot")
        return

    ordered = summary.sort_values(mean_col, ascending=True)
    n = len(ordered)
    y = range(n)
    fig, ax = plt.subplots(figsize=(9.0, _bar_figure_height(n)))
    ax.barh(list(y), ordered[mean_col], color=bar_color, height=0.72,
            label=f"mean {label}")
    ax.scatter(ordered[median_col], list(y), color=MARKER, s=14, zorder=3,
               label=f"median {label}")
    ax.set_yticks(list(y))
    ax.set_yticklabels(ordered["dataset"], fontsize=7)
    ax.set_xlabel(f"Transcript {label} (per-dataset statistic)")
    ax.set_title(f"Per-dataset transcript {label}: mean (bar) vs median (dot)")
    ax.margins(y=0.005)
    ax.grid(axis="x", linewidth=0.4, alpha=0.5)

    if annotate_flat_median:
        ax.axvline(1.0, color="#6b7280", linestyle=":", linewidth=1.0)
        ax.text(
            0.99, 0.01,
            "median ≈ 1.0 for all datasets by construction\n"
            "(weight = weight_raw / dataset_median)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="#374151",
            bbox=dict(boxstyle="round", fc="#f3f4f6", ec="#d1d5db", alpha=0.9),
        )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def print_summary(summary: pd.DataFrame) -> None:
    n = len(summary)
    total = int(summary["n_transcripts"].sum())
    counts = summary["n_transcripts"]
    print(f"Datasets                 : {n}")
    print(f"Total transcript rows    : {total:,}")
    print(f"Transcripts per dataset  : "
          f"min={counts.min():,}  median={counts.median():,.0f}  "
          f"mean={counts.mean():,.0f}  max={counts.max():,}")
    if "weight_mean" in summary.columns:
        print(f"Normalized weight mean   : "
              f"min={summary['weight_mean'].min():.3f}  "
              f"max={summary['weight_mean'].max():.3f}  "
              f"(median per dataset ≈ 1.0 by construction)")
    if "weight_raw_mean" in summary.columns:
        print(f"Raw weight mean          : "
              f"min={summary['weight_raw_mean'].min():.3f}  "
              f"max={summary['weight_raw_mean'].max():.3f}")

    head = summary.nlargest(5, "n_transcripts")[["dataset", "n_transcripts"]]
    tail = summary.nsmallest(5, "n_transcripts")[["dataset", "n_transcripts"]]
    print("\nMost transcripts:")
    for _, r in head.iterrows():
        print(f"  {r['dataset']:<32} {int(r['n_transcripts']):>7,}")
    print("Fewest transcripts:")
    for _, r in tail.iterrows():
        print(f"  {r['dataset']:<32} {int(r['n_transcripts']):>7,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("../Datasets/data/weighted_HEK_riboseq_codon_replicas"),
        help="Directory of weighted HEK replica parquets.",
    )
    parser.add_argument(
        "--outdir", type=Path,
        default=Path("../analyses/weighted_hek_dataset_summary"),
        help="Where to write the CSV and PNG figures.",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Compute the CSV/stdout summary only, without rendering figures.",
    )
    args = parser.parse_args()

    summary = collect_dataset_statistics(args.data_dir)
    args.outdir.mkdir(parents=True, exist_ok=True)

    csv_path = args.outdir / "dataset_weight_summary.csv"
    summary.to_csv(csv_path, index=False)
    print_summary(summary)
    print(f"\nWrote summary table -> {csv_path}")

    if args.no_plots:
        return

    counts_png = args.outdir / "transcript_counts.png"
    weight_png = args.outdir / "transcript_weight_normalized.png"
    raw_png = args.outdir / "transcript_weight_raw.png"
    plot_transcript_counts(summary, counts_png)
    plot_weight_analysis(summary, weight_png, label="weight",
                         bar_color=ACCENT, annotate_flat_median=True)
    plot_weight_analysis(summary, raw_png, label="weight_raw",
                         bar_color=ACCENT_RAW, annotate_flat_median=False)
    print(f"Wrote figures       -> {counts_png}")
    print(f"                       {weight_png}")
    print(f"                       {raw_png}")


if __name__ == "__main__":
    main()
