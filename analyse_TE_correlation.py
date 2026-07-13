from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import pearsonr, spearmanr


SCALAR_METRICS = ("J", "scale_dt")
PROFILE_METRICS = (
    "target",
    "mu",
    "likelihood_positive_mean",
    "L_bio",
    "rho_bio",
    "gamma",
    "additive_bias",
    "log_sigma",
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Correlate transcript-level model outputs with measured translation efficiency."
    )
    parser.add_argument(
        "predictions",
        nargs="+",
        type=Path,
        help="Prediction parquet files or directories containing prediction parquet files.",
    )
    parser.add_argument(
        "--te-data",
        type=Path,
        default=repo_root / "Datasets/data/TE_ilr_residual.clr.median_across_datasets.csv",
        help="CSV containing transcript_id and TE_clr_median.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "results/te_correlations.csv",
        help="Destination for the tidy correlation table.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="Minimum number of matched transcripts required for a correlation.",
    )
    return parser.parse_args()


def resolve_prediction_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for input_path in inputs:
        if input_path.is_dir():
            paths.extend(sorted(input_path.rglob("predictions*.parquet")))
        elif input_path.is_file():
            paths.append(input_path)
        else:
            raise FileNotFoundError(f"Prediction input not found: {input_path}")

    unique_paths = list(dict.fromkeys(path.resolve() for path in paths))
    if not unique_paths:
        raise FileNotFoundError("No prediction parquet files were found.")
    return unique_paths


def profile_mean(values: object, length: object) -> float:
    if values is None:
        return np.nan
    array = np.asarray(values, dtype=float).reshape(-1)
    if pd.notna(length):
        array = array[: max(0, int(length))]
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else np.nan


def load_prediction_metrics(path: Path) -> pd.DataFrame:
    available = set(pq.read_schema(path).names)
    if "transcript_id" not in available:
        raise ValueError(f"Missing transcript_id in {path}")

    requested = {
        "transcript_id",
        "dataset_id",
        "length",
        "lengths",
        *SCALAR_METRICS,
        *PROFILE_METRICS,
    }
    frame = pd.read_parquet(path, columns=sorted(requested & available))
    frame["transcript_id"] = frame["transcript_id"].astype(str)
    if "dataset_id" not in frame:
        frame["dataset_id"] = "unknown"

    length_column = "length" if "length" in frame else "lengths" if "lengths" in frame else None
    lengths = frame[length_column] if length_column else pd.Series(np.nan, index=frame.index)

    result = frame[["transcript_id", "dataset_id"]].copy()
    for metric in SCALAR_METRICS:
        if metric in frame:
            result[metric] = pd.to_numeric(frame[metric], errors="coerce")
    for metric in PROFILE_METRICS:
        if metric in frame:
            summary_name = metric if metric.endswith("_mean") else f"{metric}_mean"
            result[summary_name] = [
                profile_mean(values, length)
                for values, length in zip(frame[metric], lengths, strict=True)
            ]
    return result


def correlate_group(
    frame: pd.DataFrame,
    te_data: pd.DataFrame,
    *,
    prediction_file: str,
    dataset_id: str,
    min_samples: int,
) -> list[dict[str, object]]:
    metric_columns = [
        column for column in frame.columns if column not in {"transcript_id", "dataset_id"}
    ]
    if not metric_columns:
        raise ValueError(f"No supported model metrics found in {prediction_file}")
    transcript_metrics = frame.groupby("transcript_id", as_index=False)[metric_columns].mean()
    merged = transcript_metrics.merge(te_data, on="transcript_id", how="inner", validate="one_to_one")

    rows: list[dict[str, object]] = []
    for metric in metric_columns:
        valid = merged[[metric, "TE_clr_median"]].replace([np.inf, -np.inf], np.nan).dropna()
        n_samples = len(valid)
        pearson_r = pearson_p = spearman_r = spearman_p = np.nan
        if (
            n_samples >= min_samples
            and valid[metric].nunique() > 1
            and valid["TE_clr_median"].nunique() > 1
        ):
            pearson = pearsonr(valid[metric], valid["TE_clr_median"])
            spearman = spearmanr(valid[metric], valid["TE_clr_median"])
            pearson_r, pearson_p = float(pearson.statistic), float(pearson.pvalue)
            spearman_r, spearman_p = float(spearman.statistic), float(spearman.pvalue)

        rows.append(
            {
                "prediction_file": prediction_file,
                "dataset_id": dataset_id,
                "metric": metric,
                "n_transcripts": n_samples,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
            }
        )
    return rows


def analyse_file(
    path: Path,
    te_data: pd.DataFrame,
    *,
    min_samples: int,
) -> list[dict[str, object]]:
    frame = load_prediction_metrics(path)
    label = str(path)
    rows = correlate_group(
        frame,
        te_data,
        prediction_file=label,
        dataset_id="all",
        min_samples=min_samples,
    )
    for dataset_id, group in frame.groupby("dataset_id", dropna=False):
        rows.extend(
            correlate_group(
                group,
                te_data,
                prediction_file=label,
                dataset_id=str(dataset_id),
                min_samples=min_samples,
            )
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.min_samples < 3:
        raise ValueError("--min-samples must be at least 3.")

    te_data = pd.read_csv(args.te_data, usecols=["transcript_id", "TE_clr_median"])
    te_data["transcript_id"] = te_data["transcript_id"].astype(str)
    te_data["TE_clr_median"] = pd.to_numeric(te_data["TE_clr_median"], errors="coerce")
    te_data = te_data.dropna().drop_duplicates("transcript_id", keep=False)

    rows: list[dict[str, object]] = []
    paths = resolve_prediction_paths(args.predictions)
    for path in paths:
        rows.extend(analyse_file(path, te_data, min_samples=args.min_samples))

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} correlations from {len(paths)} file(s) to {args.output}")


if __name__ == "__main__":
    main()
