"""Evaluate whether transcript-reliability scores predict replica agreement.

This is an isolated diagnostic hook. It never fits or changes weighting
coefficients and never rewrites source parquets.  For each eligible
dataset--transcript pair with at least two informative replicas it exports the
legacy and SNR weights beside mean pairwise raw-PCC, fixed-alpha NB-VST PCC,
and Spearman agreement.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Datasets.data import weight_hek_riboseq_codon_replicas as weighting


DEFAULT_INPUT_DIR = ROOT / "Datasets" / "data" / "HEK_riboseq_codon_replicas"
DEFAULT_OUTPUT = (
    ROOT
    / "results"
    / "transcript_reliability_weight_audit"
    / "replica_reproducibility.tsv"
)


def _pcc(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or float(np.var(left)) <= 0.0 or float(np.var(right)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return float(pd.Series(left).corr(pd.Series(right), method="spearman"))


def _nb_vst(values: np.ndarray, alpha: float) -> np.ndarray:
    # Fixed alpha makes this a candidate reproducibility transform independent
    # of the trained model. It is not the model's learned NB2 dispersion.
    return 2.0 / np.sqrt(alpha) * np.arcsinh(np.sqrt(alpha * values))


def _replica_metrics(
    raw_replicas: object,
    *,
    dataset_name: str,
    transcript_id: str,
    expected_length: int,
    nb_vst_alpha: float,
) -> dict[str, float | int] | None:
    replicas: list[np.ndarray] = []
    for replica_index, raw in enumerate(list(raw_replicas)):
        replica = np.asarray(raw, dtype=np.float64)
        if replica.ndim != 1:
            raise ValueError(
                f"Dataset {dataset_name!r}, transcript {transcript_id!r}, replica "
                f"{replica_index}: expected one-dimensional profile, got {replica.shape}."
            )
        if replica.size != expected_length:
            raise ValueError(
                f"Dataset {dataset_name!r}, transcript {transcript_id!r}, replica "
                f"{replica_index}: length {replica.size} != consensus length "
                f"{expected_length}."
            )
        if not np.isfinite(replica).all() or bool((replica < 0.0).any()):
            raise ValueError(
                f"Dataset {dataset_name!r}, transcript {transcript_id!r}, replica "
                f"{replica_index}: counts must be finite and non-negative."
            )
        # A constant or all-zero profile has no defined correlation and does
        # not count as an informative replica for this diagnostic.
        if float(replica.sum()) > 0.0 and float(np.var(replica)) > 0.0:
            replicas.append(replica)
    if len(replicas) < 2:
        return None

    raw_pcc: list[float] = []
    nb_vst_pcc: list[float] = []
    spearman: list[float] = []
    for left, right in itertools.combinations(replicas, 2):
        raw_pcc.append(_pcc(left, right))
        nb_vst_pcc.append(_pcc(_nb_vst(left, nb_vst_alpha), _nb_vst(right, nb_vst_alpha)))
        spearman.append(_spearman(left, right))
    return {
        "informative_replicas": len(replicas),
        "pairwise_comparisons": len(raw_pcc),
        "pairwise_raw_pcc": float(np.nanmean(raw_pcc)),
        "pairwise_nb_vst_pcc": float(np.nanmean(nb_vst_pcc)),
        "pairwise_spearman": float(np.nanmean(spearman)),
    }


def analyze_dataset(
    path: Path,
    *,
    nb_vst_alpha: float,
    agreement_metric: str,
) -> pd.DataFrame:
    data = pd.read_parquet(path, columns=["id", "ribo", "ribo_cds_replicas"])
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
    snr_weight, _ = weighting.normalize_transcript_weights_by_median(
        snr.raw_weights,
        dataset_name=dataset_name,
        transcript_ids=transcript_ids,
    )
    legacy_weight, _ = weighting.normalize_transcript_weights_by_median(
        legacy.raw_weights,
        dataset_name=dataset_name,
        transcript_ids=transcript_ids,
    )

    rows: list[dict[str, object]] = []
    for index in coverage.index:
        transcript_id = str(data.at[index, "id"])
        metrics = _replica_metrics(
            data.at[index, "ribo_cds_replicas"],
            dataset_name=dataset_name,
            transcript_id=transcript_id,
            expected_length=int(stats.lengths.loc[index]),
            nb_vst_alpha=nb_vst_alpha,
        )
        if metrics is None:
            continue
        rows.append(
            {
                "transcript_id": transcript_id,
                "dataset": dataset_name,
                "coverage": float(coverage.loc[index]),
                "read_density": float(density.loc[index]),
                "depth_reference_tau": float(snr.depth_reference_tau),
                "legacy_weight": float(legacy_weight.loc[index]),
                "snr_weight": float(snr_weight.loc[index]),
                **metrics,
                "replica_agreement_metric": agreement_metric,
                "replica_agreement": float(metrics[agreement_metric]),
                "nb_vst_fixed_alpha": nb_vst_alpha,
            }
        )
    return pd.DataFrame(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", nargs="+", required=True, metavar="NAME")
    parser.add_argument("--nb-vst-alpha", type=float, default=1.0)
    parser.add_argument(
        "--agreement-metric",
        choices=("pairwise_raw_pcc", "pairwise_nb_vst_pcc", "pairwise_spearman"),
        default="pairwise_raw_pcc",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not np.isfinite(args.nb_vst_alpha) or args.nb_vst_alpha <= 0.0:
        raise ValueError("--nb-vst-alpha must be finite and strictly positive.")
    frames: list[pd.DataFrame] = []
    for name in args.datasets:
        path = args.input_dir / f"{name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = analyze_dataset(
            path,
            nb_vst_alpha=float(args.nb_vst_alpha),
            agreement_metric=args.agreement_metric,
        )
        frames.append(frame)
        print(f"{name}: informative transcript-dataset pairs={len(frame)}")
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, sep="\t", index=False)
    print(f"Replica-reproducibility audit: {args.output.resolve()}")


if __name__ == "__main__":
    main()
