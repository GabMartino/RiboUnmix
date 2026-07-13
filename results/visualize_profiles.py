from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ablation_utils import MIX_EXPERIMENT, safe_name


PROFILE_COLUMNS = (
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
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Render per-transcript model profile plots.")
    parser.add_argument(
        "predictions",
        nargs="*",
        type=Path,
        help="Prediction parquet files. The latest experiment run is used when omitted.",
    )
    parser.add_argument("--experiment", default=MIX_EXPERIMENT)
    parser.add_argument("--run", help="Run directory name within the selected experiment.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=repo_root / "results/riboai_queueing",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args()


def resolve_run_dir(base_results: Path, experiment: str, run: str | None = None) -> Path:
    experiment_dir = base_results / experiment
    if run is not None:
        run_dir = experiment_dir / run
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir

    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    run_dirs = [
        path
        for path in experiment_dir.iterdir()
        if path.is_dir() and any(path.glob(f"predictions_*_{experiment}.parquet"))
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No runs with prediction files found under {experiment_dir}")
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def read_predictions(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Prediction file not found: {path}")
        available = set(pq.read_schema(path).names)
        if "transcript_id" not in available:
            raise ValueError(f"Missing transcript_id in {path}")
        requested = {
            "transcript_id",
            "dataset_id",
            "length",
            "lengths",
            "mask",
            *PROFILE_COLUMNS,
        }
        frame = pd.read_parquet(path, columns=sorted(requested & available))
        frame["prediction_file"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def valid_profile(row: pd.Series, column: str) -> np.ndarray:
    values = np.asarray(row[column], dtype=float).reshape(-1)
    length = row.get("length", row.get("lengths", len(values)))
    if pd.notna(length):
        values = values[: max(0, int(length))]
    mask = row.get("mask")
    if mask is not None and not (np.isscalar(mask) and pd.isna(mask)):
        valid_mask = np.asarray(mask, dtype=bool).reshape(-1)[: len(values)]
        values = values[: len(valid_mask)][valid_mask]
    return values


def plot_profiles(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    limit: int = 10,
    dpi: int = 140,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for row_index, (_, row) in enumerate(frame.head(limit).iterrows()):
        columns = []
        for column in PROFILE_COLUMNS:
            if column not in frame:
                continue
            values = row[column]
            if values is None or (np.isscalar(values) and pd.isna(values)):
                continue
            if len(values) > 0:
                columns.append(column)
        if not columns:
            continue

        figure, axes = plt.subplots(
            nrows=len(columns),
            figsize=(12, max(2.4 * len(columns), 3.0)),
            squeeze=False,
            constrained_layout=True,
        )
        for axis, column in zip(axes[:, 0], columns, strict=True):
            values = valid_profile(row, column)
            axis.plot(np.arange(len(values)), values, linewidth=1.2)
            axis.set_ylabel(column)
            axis.grid(alpha=0.2)
        axes[-1, 0].set_xlabel("codon position")

        transcript_id = str(row["transcript_id"])
        dataset_id = str(row.get("dataset_id", "unknown"))
        figure.suptitle(f"{transcript_id} | dataset {dataset_id}")
        filename = f"{row_index:03d}_{safe_name(transcript_id)}_dataset-{safe_name(dataset_id)}.png"
        destination = output_dir / filename
        figure.savefig(destination, dpi=dpi)
        plt.close(figure)
        written.append(destination)

    return written


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive.")

    run_dir: Path | None = None
    if args.predictions:
        prediction_paths = [path.resolve() for path in args.predictions]
    else:
        run_dir = resolve_run_dir(args.results_root, args.experiment, args.run)
        prediction_paths = sorted(run_dir.glob(f"predictions_*_{args.experiment}.parquet"))
        if not prediction_paths:
            raise FileNotFoundError(f"No prediction parquet files found in {run_dir}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (run_dir / "profile_plots") if run_dir else Path(__file__).parent / "profile_plots"

    frame = read_predictions(prediction_paths)
    written = plot_profiles(frame, output_dir, limit=args.limit, dpi=args.dpi)
    print(f"Wrote {len(written)} profile plot(s) to {output_dir}")


if __name__ == "__main__":
    main()
