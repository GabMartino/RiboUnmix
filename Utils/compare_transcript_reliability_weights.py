"""Compare legacy and SNR transcript-reliability weights without rewriting data.

The utility reads replica-aware, unweighted HEK parquets, applies the same
validation/filtering code as production preprocessing, and writes only audit
tables and figures under ``--output-dir``.  It never writes weighted parquets.

Example:
    python3 Utils/compare_transcript_reliability_weights.py \
        --datasets akichika_2019 martinez_2019
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Datasets.data import weight_hek_riboseq_codon_replicas as weighting


DEFAULT_INPUT_DIR = ROOT / "Datasets" / "data" / "HEK_riboseq_codon_replicas"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "transcript_reliability_weight_audit"
QUANTILES = (0.0, 0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0)


def compare_dataset(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return eligible pair rows carrying both formulas and one summary row."""
    data = pd.read_parquet(path, columns=["id", "ribo"])
    dataset_name = path.stem
    stats = weighting._profile_statistics(data, dataset_name)
    eligible = (
        (stats.lengths > 0)
        & (stats.total_reads > 0.0)
        & (stats.coverage > 0.0)
    )
    coverage = stats.coverage.loc[eligible]
    density = stats.read_density.loc[eligible]
    transcript_ids = data.loc[eligible, "id"]
    snr = weighting.calculate_transcript_weight_components(
        coverage,
        density,
        weighting_mode=weighting.SNR_DEPTH_COVERAGE_MODE,
        dataset_name=dataset_name,
        transcript_ids=transcript_ids,
    )
    legacy = weighting.calculate_transcript_weight_components(
        coverage,
        density,
        weighting_mode=weighting.LEGACY_COVERAGE_DENSITY_RANK_MODE,
        dataset_name=dataset_name,
        transcript_ids=transcript_ids,
    )
    snr_weight, snr_median = weighting.normalize_transcript_weights_by_median(
        snr.raw_weights,
        dataset_name=dataset_name,
        transcript_ids=transcript_ids,
    )
    legacy_weight, legacy_median = weighting.normalize_transcript_weights_by_median(
        legacy.raw_weights,
        dataset_name=dataset_name,
        transcript_ids=transcript_ids,
    )

    pairs = pd.DataFrame(
        {
            "dataset": dataset_name,
            "transcript_id": transcript_ids.astype(str),
            "coverage": coverage,
            "read_density": density,
            "depth_reference_tau": snr.depth_reference_tau,
            "depth_snr_score": snr.depth_snr_score,
            "legacy_density_percentile": legacy.density_percentile_legacy,
            "legacy_raw_weight": legacy.raw_weights,
            "snr_raw_weight": snr.raw_weights,
            "legacy_weight": legacy_weight,
            "snr_weight": snr_weight,
        }
    ).reset_index(drop=True)
    pairs["final_weight_change"] = pairs["snr_weight"] - pairs["legacy_weight"]
    summary: dict[str, object] = {
        "dataset": dataset_name,
        "input_rows": int(len(data)),
        "eligible_rows": int(eligible.sum()),
        "removed_rows": int((~eligible).sum()),
        "depth_reference_tau": float(snr.depth_reference_tau),
        "legacy_raw_normalization_median": float(legacy_median),
        "snr_raw_normalization_median": float(snr_median),
        "weight_pearson": float(pairs["legacy_weight"].corr(pairs["snr_weight"])),
        "weight_spearman": float(
            pairs["legacy_weight"].corr(pairs["snr_weight"], method="spearman")
        ),
        "coverage_mean": float(coverage.mean()),
        "coverage_median": float(coverage.median()),
        "coverage_iqr": float(coverage.quantile(0.75) - coverage.quantile(0.25)),
        "read_density_mean": float(density.mean()),
        "read_density_median": float(density.median()),
    }
    return pairs, summary


def quantile_rows(pairs: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    columns = (
        "legacy_weight",
        "snr_weight",
        "legacy_raw_weight",
        "snr_raw_weight",
        "depth_snr_score",
        "coverage",
        "read_density",
    )
    dataset = str(pairs["dataset"].iloc[0])
    for column in columns:
        values = pairs[column].quantile(QUANTILES)
        for quantile, value in values.items():
            rows.append(
                {
                    "dataset": dataset,
                    "quantity": column,
                    "quantile": float(quantile),
                    "value": float(value),
                }
            )
    return rows


def plot_raw_weight_contours(
    pairs: pd.DataFrame,
    output_path: Path,
    *,
    max_scatter_points: int,
) -> None:
    """Plot analytic SNR raw-score contours and final normalized row weights."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    dataset = str(pairs["dataset"].iloc[0])
    tau = float(pairs["depth_reference_tau"].iloc[0])
    y_values = np.log1p(pairs["read_density"].to_numpy(dtype=np.float64))
    y_low, y_high = np.quantile(y_values, [0.001, 0.999])
    if not np.isfinite(y_low) or not np.isfinite(y_high) or y_low == y_high:
        y_low, y_high = float(np.min(y_values)), float(np.max(y_values) + 1.0)
    coverage_grid = np.linspace(0.001, 1.0, 240)
    log_density_grid = np.linspace(y_low, y_high, 240)
    coverage_mesh, log_density_mesh = np.meshgrid(coverage_grid, log_density_grid)
    density_mesh = np.expm1(log_density_mesh)
    depth_mesh = np.sqrt(density_mesh) / (
        np.sqrt(density_mesh) + np.sqrt(tau)
    )
    raw_mesh = (
        weighting.DEPTH_WEIGHT * depth_mesh
        + weighting.COVERAGE_WEIGHT * coverage_mesh
    )

    if len(pairs) > max_scatter_points:
        positions = np.linspace(0, len(pairs) - 1, max_scatter_points).astype(int)
        scatter_data = pairs.iloc[positions]
    else:
        scatter_data = pairs

    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    contour = ax.contour(
        coverage_mesh,
        log_density_mesh,
        raw_mesh,
        levels=np.linspace(0.1, 0.9, 9),
        colors="#1f2937",
        linewidths=0.8,
    )
    ax.clabel(contour, inline=True, fontsize=7, fmt="raw %.1f")
    points = ax.scatter(
        scatter_data["coverage"],
        np.log1p(scatter_data["read_density"]),
        c=scatter_data["snr_weight"],
        s=8,
        alpha=0.45,
        cmap="viridis",
        linewidths=0,
    )
    colorbar = fig.colorbar(points, ax=ax)
    colorbar.set_label("Final median-normalized weight (may exceed 1)")
    ax.set_xlabel("Coverage: fraction of codons with positive reads")
    ax.set_ylabel("log1p(read density)")
    ax.set_title(
        f"{dataset}: SNR raw-weight contours and stored final weights\n"
        f"tau = median eligible read density = {tau:.6g}"
    )
    ax.text(
        0.02,
        0.98,
        "Contour labels are raw_weight in [0,1].\n"
        "Point colors are final weight = raw/median(raw).",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        metavar="NAME",
        help="Dataset parquet stems to compare.",
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--max-scatter-points", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_pairs: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    quantiles: list[dict[str, object]] = []
    positive: list[pd.DataFrame] = []
    negative: list[pd.DataFrame] = []

    for name in args.datasets:
        path = args.input_dir / f"{name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        pairs, summary = compare_dataset(path)
        all_pairs.append(pairs)
        summaries.append(summary)
        quantiles.extend(quantile_rows(pairs))
        positive.append(pairs.nlargest(args.top_n, "final_weight_change"))
        negative.append(pairs.nsmallest(args.top_n, "final_weight_change"))
        plot_raw_weight_contours(
            pairs,
            args.output_dir / f"{name}_snr_raw_weight_contours.png",
            max_scatter_points=max(1, int(args.max_scatter_points)),
        )
        print(
            f"{name}: eligible={summary['eligible_rows']} "
            f"tau={summary['depth_reference_tau']:.6g} "
            f"pearson={summary['weight_pearson']:.4f} "
            f"spearman={summary['weight_spearman']:.4f}"
        )

    pd.DataFrame(summaries).to_csv(
        args.output_dir / "dataset_mode_comparison.tsv", sep="\t", index=False
    )
    pd.DataFrame(quantiles).to_csv(
        args.output_dir / "weight_quantiles.tsv", sep="\t", index=False
    )
    pd.concat(all_pairs, ignore_index=True).to_csv(
        args.output_dir / "pair_weight_comparison.tsv", sep="\t", index=False
    )
    pd.concat(positive, ignore_index=True).to_csv(
        args.output_dir / "largest_positive_changes.tsv", sep="\t", index=False
    )
    pd.concat(negative, ignore_index=True).to_csv(
        args.output_dir / "largest_negative_changes.tsv", sep="\t", index=False
    )
    print(f"Audit outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
