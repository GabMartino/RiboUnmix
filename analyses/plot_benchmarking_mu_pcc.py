#!/usr/bin/env python3
"""Plot held-out μ-versus-target PCC for the four benchmarking datasets.

The benchmark training entry point writes one test-prediction parquet for each
checkpoint variant (``best_pcc`` and ``best_val_loss``).  This script selects
one variant per dataset, computes the same *raw* μ PCC diagnostic used by the
Lightning module, and produces one bar per organism/dataset.

Metric definition
-----------------
For every test transcript, use positions where ``mask`` is true and both
``mu`` and ``target`` are finite.  Compute Pearson correlation over those
positions.  As in ``RiboUnmixLightningModule._pearson_per_sample``, a
transcript whose prediction or target has negligible variance contributes
``0.0``.  The reported dataset value is the unweighted arithmetic mean over
all test transcripts.  No edge trimming, reliability weighting, or
Fisher-z aggregation is applied: this intentionally matches ``val_mu_pcc``
semantics, except that it is evaluated on held-out test predictions.

Examples
--------
Use the newest complete benchmarking bundle (default)::

    .venv/bin/python analyses/plot_benchmarking_mu_pcc.py

Plot the minimum-validation-loss checkpoint instead::

    .venv/bin/python analyses/plot_benchmarking_mu_pcc.py \
        --checkpoint-variant best_val_loss

Point explicitly to one benchmark bundle::

    .venv/bin/python analyses/plot_benchmarking_mu_pcc.py \
        --run-dir results/riboai_benchmarking_experiments/benchmark_20260829_194806
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyses.paths import artifact_directory

DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "riboai_benchmarking_experiments"
CHECKPOINT_VARIANTS = ("best_pcc", "best_val_loss")

# Keep this order stable rather than ordering bars by performance: the panel is
# meant to compare the four predefined benchmark datasets.
DATASETS: tuple[tuple[str, str, str], ...] = (
    ("human_iwasaki_2014", "Human\nIwasaki 2014", "#4C78A8"),
    ("yeast_stein_2021", "Yeast\nStein 2021", "#59A14F"),
    ("celegans_stein_2021", "C. elegans\nStein 2021", "#F28E2B"),
    ("ecoli_zhang_2016", "E. coli\nZhang 2016", "#E15759"),
)


@dataclass(frozen=True)
class DatasetMetric:
    dataset: str
    label: str
    checkpoint_variant: str
    mu_pcc: float
    n_test_transcripts: int
    n_valid_pcc_transcripts: int
    n_zero_variance_transcripts: int
    prediction_path: str
    checkpoint_path: str


def _manifest_prediction_path(
    manifest_path: Path,
    checkpoint_variant: str,
) -> tuple[Path, str]:
    """Resolve a prediction next to its manifest, surviving moved result trees."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON manifest: {manifest_path}") from exc

    entry = manifest.get(checkpoint_variant)
    if not isinstance(entry, dict):
        raise KeyError(
            f"{manifest_path} has no {checkpoint_variant!r} checkpoint entry."
        )

    recorded_path = entry.get("output_path")
    checkpoint_path = str(entry.get("checkpoint_path", ""))
    candidates: list[Path] = []
    if recorded_path:
        candidates.append(Path(str(recorded_path)))
        # Result bundles may have been copied between machines, so the absolute
        # recorded location can be stale.  The matching basename beside the
        # manifest is authoritative in that case.
        candidates.append(manifest_path.parent / Path(str(recorded_path)).name)
    candidates.extend(
        sorted(
            manifest_path.parent.glob(
                f"predictions_test_{checkpoint_variant}_*.parquet"
            )
        )
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate, checkpoint_path
    raise FileNotFoundError(
        f"Could not find a non-empty {checkpoint_variant} test-prediction parquet "
        f"for {manifest_path}."
    )


def _dataset_manifest(run_dir: Path, dataset: str) -> Path:
    candidates = sorted(
        (run_dir / "results" / dataset).glob("**/prediction_checkpoint_manifest.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No prediction checkpoint manifest found for {dataset!r} in {run_dir}."
        )
    return candidates[0]


def _complete_run_candidates(
    results_root: Path,
    checkpoint_variant: str,
) -> Iterable[Path]:
    """Yield bundles that contain usable selected-variant predictions for all four datasets."""
    roots = [results_root]
    if results_root.is_dir():
        roots.extend(sorted(path for path in results_root.iterdir() if path.is_dir()))

    for candidate in roots:
        try:
            for dataset, _, _ in DATASETS:
                manifest = _dataset_manifest(candidate, dataset)
                _manifest_prediction_path(manifest, checkpoint_variant)
        except (FileNotFoundError, KeyError, ValueError):
            continue
        yield candidate


def resolve_run_dir(
    *,
    results_root: Path,
    run_dir: Path | None,
    checkpoint_variant: str,
) -> Path:
    if run_dir is not None:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Benchmarking run directory does not exist: {run_dir}")
        return run_dir
    if not results_root.is_dir():
        raise FileNotFoundError(f"Benchmarking results root does not exist: {results_root}")

    candidates = list(_complete_run_candidates(results_root, checkpoint_variant))
    if not candidates:
        raise FileNotFoundError(
            "No complete benchmarking run with all four requested prediction files was found "
            f"under {results_root}."
        )
    return max(
        candidates,
        key=lambda path: max(
            manifest.stat().st_mtime_ns
            for manifest in path.glob("results/*/**/prediction_checkpoint_manifest.json")
        ),
    )


def _as_1d_array(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=dtype)
    return np.asarray(value, dtype=dtype).reshape(-1)


def transcript_mu_pcc(
    *,
    target_value: Any,
    mu_value: Any,
    mask_value: Any,
    pcc_prediction_floor: float,
    eps: float = 1.0e-8,
) -> tuple[float, bool]:
    """Return the model-compatible raw PCC and whether its variance was valid."""
    target = _as_1d_array(target_value, dtype=np.float64)
    mu = _as_1d_array(mu_value, dtype=np.float64)
    mask = _as_1d_array(mask_value, dtype=np.bool_)
    length = min(target.size, mu.size, mask.size)
    if length == 0:
        return 0.0, False

    target = target[:length]
    mu = mu[:length]
    mask = mask[:length] & np.isfinite(target) & np.isfinite(mu)
    if pcc_prediction_floor > 0.0:
        mu = np.where(mu >= pcc_prediction_floor, mu, 0.0)
        mask &= np.isfinite(mu)

    if not np.any(mask):
        return 0.0, False
    target_valid = target[mask]
    mu_valid = mu[mask]
    target_centered = target_valid - target_valid.mean()
    mu_centered = mu_valid - mu_valid.mean()
    target_var = float(np.dot(target_centered, target_centered))
    mu_var = float(np.dot(mu_centered, mu_centered))
    if target_var <= eps or mu_var <= eps:
        return 0.0, False
    correlation = float(
        np.dot(target_centered, mu_centered)
        / np.sqrt(target_var * mu_var + eps * eps)
    )
    if not np.isfinite(correlation):
        return 0.0, False
    return correlation, True


def compute_metric(
    *,
    dataset: str,
    label: str,
    checkpoint_variant: str,
    prediction_path: Path,
    checkpoint_path: str,
    pcc_prediction_floor: float,
) -> DatasetMetric:
    schema_columns = set(pq.ParquetFile(prediction_path).schema_arrow.names)
    required = {"target", "mu", "mask"}
    missing = sorted(required - schema_columns)
    if missing:
        raise KeyError(f"{prediction_path} is missing columns: {missing}")

    pccs: list[float] = []
    valid_count = 0
    parquet = pq.ParquetFile(prediction_path)
    for batch in parquet.iter_batches(
        columns=["target", "mu", "mask"], batch_size=128
    ):
        columns = batch.to_pydict()
        for target, mu, mask in zip(columns["target"], columns["mu"], columns["mask"]):
            value, valid = transcript_mu_pcc(
                target_value=target,
                mu_value=mu,
                mask_value=mask,
                pcc_prediction_floor=pcc_prediction_floor,
            )
            pccs.append(value)
            valid_count += int(valid)

    if not pccs:
        raise ValueError(f"No rows found in {prediction_path}")
    return DatasetMetric(
        dataset=dataset,
        label=label.replace("\n", " "),
        checkpoint_variant=checkpoint_variant,
        mu_pcc=float(np.mean(pccs)),
        n_test_transcripts=len(pccs),
        n_valid_pcc_transcripts=valid_count,
        n_zero_variance_transcripts=len(pccs) - valid_count,
        prediction_path=str(prediction_path),
        checkpoint_path=checkpoint_path,
    )


def plot_metrics(frame: pd.DataFrame, *, output_png: Path, title_suffix: str) -> None:
    # Some compute nodes have an unwritable home-level Matplotlib cache.  Keep
    # this analysis-only cache in the system temporary directory instead.
    matplotlib_cache = Path(tempfile.gettempdir()) / "riboai_matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display_labels = {dataset: label for dataset, label, _ in DATASETS}
    labels = [display_labels[dataset] for dataset in frame["dataset"]]
    scores = frame["mu_pcc"].to_numpy(dtype=float)
    colors = [color for _, _, color in DATASETS]
    fig, ax = plt.subplots(figsize=(8.7, 5.6), constrained_layout=True)
    bars = ax.bar(np.arange(len(frame)), scores, color=colors, width=0.68, edgecolor="0.2")
    ax.axhline(0.0, color="0.3", linewidth=0.8)
    lower = min(-0.05, float(np.nanmin(scores)) - 0.05)
    ax.set_ylim(lower, 1.0)
    ax.set_xticks(np.arange(len(frame)), labels)
    ax.set_ylabel("Test μ PCC")
    ax.set_title(f"Held-out μ PCC by benchmarking dataset\n{title_suffix}")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, score, n_rows in zip(bars, scores, frame["n_test_transcripts"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.025 if score >= 0 else score - 0.05,
            f"{score:.3f}\n(n={n_rows:,})",
            ha="center",
            va="bottom" if score >= 0 else "top",
            fontsize=10,
            fontweight="semibold",
        )
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_png.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory containing benchmark_* result bundles.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Explicit benchmark_* bundle; overrides automatic newest-complete selection.",
    )
    parser.add_argument(
        "--checkpoint-variant",
        choices=CHECKPOINT_VARIANTS,
        default="best_pcc",
        help="Which checkpoint's held-out predictions to plot (default: best_pcc).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <run-dir>/analysis/mu_pcc.",
    )
    parser.add_argument(
        "--pcc-prediction-floor",
        type=float,
        default=0.0,
        help="Optional μ floor used only for PCC, matching loss.pcc_prediction_floor.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pcc_prediction_floor < 0.0:
        raise ValueError("--pcc-prediction-floor must be non-negative.")

    run_dir = resolve_run_dir(
        results_root=args.results_root,
        run_dir=args.run_dir,
        checkpoint_variant=args.checkpoint_variant,
    )
    output_dir = args.output_dir or artifact_directory(
        "benchmarking", run_dir, "mu_pcc"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for dataset, label, _ in DATASETS:
        manifest = _dataset_manifest(run_dir, dataset)
        prediction_path, checkpoint_path = _manifest_prediction_path(
            manifest, args.checkpoint_variant
        )
        metric = compute_metric(
            dataset=dataset,
            label=label,
            checkpoint_variant=args.checkpoint_variant,
            prediction_path=prediction_path,
            checkpoint_path=checkpoint_path,
            pcc_prediction_floor=args.pcc_prediction_floor,
        )
        rows.append(asdict(metric))
        print(
            f"{dataset:24s} test_mu_pcc={metric.mu_pcc:.5f} "
            f"(n={metric.n_test_transcripts:,}; valid={metric.n_valid_pcc_transcripts:,})"
        )

    frame = pd.DataFrame(rows)
    tsv_path = output_dir / f"benchmark_test_mu_pcc_{args.checkpoint_variant}.tsv"
    frame.to_csv(tsv_path, sep="\t", index=False)
    png_path = output_dir / f"benchmark_test_mu_pcc_{args.checkpoint_variant}.png"
    plot_metrics(
        frame,
        output_png=png_path,
        title_suffix=(
            f"checkpoint variant: {args.checkpoint_variant.replace('_', ' ')}"
        ),
    )
    print(f"Run directory: {run_dir}")
    print(f"Saved metrics: {tsv_path}")
    print(f"Saved plot: {png_path}")
    print(f"Saved SVG: {png_path.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
