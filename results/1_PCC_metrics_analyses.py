from __future__ import annotations

from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


MIX_EXPERIMENT = "30_datasets_mix_3546d4"
COMPONENT_COLUMNS = {
    "mu": "mu",
    "rho": "rho_bio",
}
PREDICTION_PATTERNS = (
    "predictions_*.parquet",
    "comprehensive_predictions_rank*.parquet",
)


def sequence_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None

    if isinstance(value, float) and np.isnan(value):
        return None

    try:
        arr = np.asarray(value)
    except Exception:
        return None

    if arr.ndim == 0:
        if pd.isna(arr.item()):
            return None
        arr = arr.reshape(1)

    try:
        arr = arr.astype(np.float64, copy=False).reshape(-1)
    except (TypeError, ValueError):
        return None

    return arr if arr.size > 0 else None


def safe_pearsonr(x: Any, y: Any, eps: float = 1.0e-12) -> float:
    x_arr = sequence_or_none(x)
    y_arr = sequence_or_none(y)
    if x_arr is None or y_arr is None:
        return np.nan

    length = min(x_arr.size, y_arr.size)
    if length < 4:
        return np.nan

    x_arr = x_arr[:length]
    y_arr = y_arr[:length]
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    if x_arr.size < 4:
        return np.nan

    if np.std(x_arr) <= eps or np.std(y_arr) <= eps:
        return np.nan

    x_arr = x_arr - x_arr.mean()
    y_arr = y_arr - y_arr.mean()
    denom = np.sqrt(np.sum(x_arr**2) * np.sum(y_arr**2))
    if denom <= eps:
        return np.nan

    return float(np.sum(x_arr * y_arr) / denom)


def empty_metrics() -> dict[str, float]:
    return {
        "pcc": np.nan,
        "ci_lower": np.nan,
        "ci_upper": np.nan,
        "n_transcripts": 0,
        "median_pcc": np.nan,
        "unweighted_mean_pcc": np.nan,
    }


def fisher_weighted_pcc(pccs: list[float], lengths: list[int]) -> dict[str, float]:
    if len(pccs) == 0:
        return empty_metrics()

    pcc_arr = np.asarray(pccs, dtype=np.float64)
    length_arr = np.asarray(lengths, dtype=np.float64)
    z = np.arctanh(np.clip(pcc_arr, -0.9999, 0.9999))
    weights = np.maximum(length_arr - 3.0, 1.0)

    z_mean = np.average(z, weights=weights)
    z_se = 1.0 / np.sqrt(np.sum(weights))
    z_crit = NormalDist().inv_cdf(0.975)

    return {
        "pcc": float(np.tanh(z_mean)),
        "ci_lower": float(np.tanh(z_mean - z_crit * z_se)),
        "ci_upper": float(np.tanh(z_mean + z_crit * z_se)),
        "n_transcripts": int(len(pccs)),
        "median_pcc": float(np.median(pcc_arr)),
        "unweighted_mean_pcc": float(np.mean(pcc_arr)),
    }


def discover_prediction_files(base_path: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in PREDICTION_PATTERNS:
        files.extend(base_path.rglob(pattern))
    return sorted(set(files))


def prediction_metadata(path: Path, base_path: Path) -> dict[str, str]:
    rel = path.relative_to(base_path)
    split = "unknown"
    suffix = path.stem.removeprefix("predictions_")

    for candidate in ("main_val", "css_benchmark", "val", "test", "predict"):
        if suffix.startswith(candidate):
            split = candidate
            break

    return {
        "experiment": rel.parts[0],
        "run": rel.parts[1],
        "split": split,
        "file": str(path),
    }


def load_dataset_encoding(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return {int(value): str(name) for name, value in data.items()}


def dataset_label(dataset_id: Any, id_to_dataset: dict[int, str], fallback: str) -> str:
    try:
        dataset_int = int(dataset_id)
    except Exception:
        return fallback

    return id_to_dataset.get(dataset_int, f"dataset_{dataset_int}")


def compute_component_metrics(
    subset: pd.DataFrame,
    *,
    component: str,
    column: str,
) -> dict[str, Any]:
    if column not in subset.columns:
        return {
            "component": component,
            "resolved_column": None,
            **empty_metrics(),
        }

    pccs: list[float] = []
    lengths: list[int] = []

    for _, row in subset.iterrows():
        pred = sequence_or_none(row[column])
        target = sequence_or_none(row["target"])
        if pred is None or target is None:
            continue

        length = min(pred.size, target.size)
        pcc = safe_pearsonr(pred[:length], target[:length])
        if not np.isfinite(pcc):
            continue

        pccs.append(pcc)
        lengths.append(length)

    return {
        "component": component,
        "resolved_column": column,
        **fisher_weighted_pcc(pccs, lengths),
    }


def add_metrics_rows(
    rows: list[dict[str, Any]],
    *,
    df: pd.DataFrame,
    metadata: dict[str, str],
    id_to_dataset: dict[int, str],
) -> None:
    if "target" not in df.columns:
        raise KeyError("Prediction file is missing required column: target")

    if "dataset_id" in df.columns:
        groups = df.groupby("dataset_id", sort=True)
    else:
        groups = [(metadata["experiment"], df)]

    for dataset_id, subset in groups:
        dataset = dataset_label(
            dataset_id,
            id_to_dataset,
            fallback=metadata["experiment"],
        )

        for component, column in COMPONENT_COLUMNS.items():
            rows.append(
                {
                    **metadata,
                    "dataset_id": dataset_id,
                    "dataset": dataset,
                    **compute_component_metrics(
                        subset,
                        component=component,
                        column=column,
                    ),
                }
            )


def keep_mix_vs_single_rows(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in metrics_df.iterrows():
        experiment = str(row["experiment"])
        dataset = str(row["dataset"])

        if experiment == MIX_EXPERIMENT:
            run_type = "30-dataset mix"
        elif experiment == dataset:
            run_type = "single dataset"
        else:
            continue

        rows.append({**row.to_dict(), "run_type": run_type})

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(
        ["split", "component", "dataset", "run_type"],
        ascending=True,
    ).reset_index(drop=True)


def paired_plot_data(metrics_df: pd.DataFrame, component: str, split: str) -> pd.DataFrame:
    sub = metrics_df[
        (metrics_df["component"] == component)
        & (metrics_df["split"] == split)
        & (metrics_df["n_transcripts"].fillna(0).astype(int) > 0)
    ].copy()

    if sub.empty:
        return sub

    counts = sub.groupby("dataset")["run_type"].nunique()
    paired_datasets = set(counts[counts == 2].index)
    return sub[sub["dataset"].isin(paired_datasets)].copy()


def plot_component_comparison(
    metrics_df: pd.DataFrame,
    *,
    component: str,
    split: str,
    out_dir: Path,
) -> Path | None:
    sub = paired_plot_data(metrics_df, component, split)
    if sub.empty:
        print(f"[WARN] No paired rows for component={component}, split={split}.")
        return None

    if plt is None:
        return plot_component_comparison_with_pillow(
            sub,
            component=component,
            split=split,
            out_dir=out_dir,
        )

    mix = sub[sub["run_type"] == "30-dataset mix"][["dataset", "pcc"]]
    single = sub[sub["run_type"] == "single dataset"][["dataset", "pcc"]]
    single_perf = single.rename(columns={"pcc": "single_pcc"})
    datasets = single_perf.sort_values(
        "single_pcc",
        ascending=False,
    )["dataset"].tolist()

    y = np.arange(len(datasets), dtype=np.float64)
    bar_height = 0.34
    styles = {
        "single dataset": {"offset": -bar_height / 2.0, "color": "#4C78A8"},
        "30-dataset mix": {"offset": bar_height / 2.0, "color": "#F58518"},
    }

    fig, ax = plt.subplots(figsize=(11, max(5, 0.34 * len(datasets))))

    for run_type, style in styles.items():
        values = []
        err_lower = []
        err_upper = []

        for dataset in datasets:
            row = sub[
                (sub["dataset"] == dataset)
                & (sub["run_type"] == run_type)
            ].iloc[0]
            pcc = float(row["pcc"])
            values.append(pcc)
            err_lower.append(max(0.0, pcc - float(row["ci_lower"])))
            err_upper.append(max(0.0, float(row["ci_upper"]) - pcc))

        ax.barh(
            y + float(style["offset"]),
            values,
            height=bar_height,
            xerr=np.asarray([err_lower, err_upper], dtype=np.float64),
            capsize=1.5,
            color=str(style["color"]),
            edgecolor="black",
            linewidth=0.5,
            label=run_type,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(datasets, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Fisher-Z aggregated per-transcript PCC")
    ax.set_title(f"PCC({component}, target): single dataset vs {MIX_EXPERIMENT}")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(
        loc="lower right",
        fontsize=8,
        title_fontsize=8,
        framealpha=0.85,
        borderpad=0.4,
        labelspacing=0.3,
        handlelength=1.2,
    )

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pcc_{split}_{component}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def pil_font(size: int, *, bold: bool = False) -> Any:
    if ImageFont is None:
        return None

    font_name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = [
        f"/usr/share/fonts/truetype/dejavu/{font_name}",
        f"/usr/local/share/fonts/{font_name}",
    ]

    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass

    return ImageFont.load_default()


def plot_component_comparison_with_pillow(
    sub: pd.DataFrame,
    *,
    component: str,
    split: str,
    out_dir: Path,
) -> Path | None:
    if Image is None or ImageDraw is None:
        print("\n[WARN] Neither matplotlib nor Pillow is installed. Skipping plots.")
        return None

    mix = sub[sub["run_type"] == "30-dataset mix"][["dataset", "pcc"]]
    single = sub[sub["run_type"] == "single dataset"][["dataset", "pcc"]]
    single_perf = single.rename(columns={"pcc": "single_pcc"})
    datasets = single_perf.sort_values(
        "single_pcc",
        ascending=False,
    )["dataset"].tolist()

    width = 1300
    row_height = 24
    top = 70
    left = 230
    right = 50
    bottom = 78
    height = top + row_height * len(datasets) + bottom
    plot_width = width - left - right
    bg = "white"

    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    title_font = pil_font(16, bold=True)
    axis_font = pil_font(11)
    label_font = pil_font(10)
    legend_font = pil_font(10)

    pcc_min = float(np.nanmin(sub["ci_lower"].astype(float).to_numpy()))
    pcc_max = float(np.nanmax(sub["ci_upper"].astype(float).to_numpy()))
    x_min = min(0.0, np.floor((pcc_min - 0.03) * 10.0) / 10.0)
    x_max = min(1.0, max(0.7, np.ceil((pcc_max + 0.03) * 10.0) / 10.0))
    if x_max <= x_min:
        x_max = x_min + 1.0

    def x_pos(value: float) -> int:
        frac = (value - x_min) / (x_max - x_min)
        return int(round(left + frac * plot_width))

    title = f"PCC({component}, target): single dataset vs {MIX_EXPERIMENT}"
    draw.text((left, 18), title, fill="black", font=title_font)

    tick_start = int(np.ceil(x_min * 10.0))
    tick_end = int(np.floor(x_max * 10.0))
    axis_y = top + row_height * len(datasets) + 8
    for tick in range(tick_start, tick_end + 1):
        value = tick / 10.0
        x = x_pos(value)
        draw.line((x, top - 8, x, axis_y), fill="#E0E0E0", width=1)
        draw.text((x - 12, axis_y + 5), f"{value:.1f}", fill="black", font=axis_font)

    zero_x = x_pos(0.0)
    draw.line((zero_x, top - 8, zero_x, axis_y), fill="black", width=1)
    draw.line((left, axis_y, left + plot_width, axis_y), fill="black", width=1)
    draw.text(
        (left + plot_width // 2 - 140, height - 28),
        "Fisher-Z aggregated per-transcript PCC",
        fill="black",
        font=axis_font,
    )

    styles = {
        "single dataset": {"offset": -6, "color": "#4C78A8"},
        "30-dataset mix": {"offset": 6, "color": "#F58518"},
    }
    bar_height = 9

    for row_idx, dataset in enumerate(datasets):
        y_center = top + row_idx * row_height + row_height // 2
        draw.text((18, y_center - 7), dataset, fill="black", font=label_font)

        for run_type, style in styles.items():
            row = sub[
                (sub["dataset"] == dataset)
                & (sub["run_type"] == run_type)
            ].iloc[0]
            pcc = float(row["pcc"])
            ci_lower = float(row["ci_lower"])
            ci_upper = float(row["ci_upper"])
            y = y_center + int(style["offset"])
            x0 = x_pos(0.0)
            x1 = x_pos(pcc)
            lo = x_pos(ci_lower)
            hi = x_pos(ci_upper)
            x_left, x_right = sorted((x0, x1))

            draw.rectangle(
                (x_left, y - bar_height // 2, x_right, y + bar_height // 2),
                fill=str(style["color"]),
                outline="black",
                width=1,
            )
            draw.line((lo, y, hi, y), fill="black", width=1)
            draw.line((lo, y - 4, lo, y + 4), fill="black", width=1)
            draw.line((hi, y - 4, hi, y + 4), fill="black", width=1)

    legend_x = width - 210
    legend_y = 18
    draw.rectangle(
        (legend_x - 8, legend_y - 6, width - 45, legend_y + 44),
        fill="white",
        outline="#BBBBBB",
        width=1,
    )
    for idx, (run_type, style) in enumerate(styles.items()):
        y = legend_y + idx * 20
        draw.rectangle(
            (legend_x, y, legend_x + 16, y + 8),
            fill=str(style["color"]),
            outline="black",
            width=1,
        )
        draw.text((legend_x + 24, y - 3), run_type, fill="black", font=legend_font)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pcc_{split}_{component}.png"
    image.save(out_path)
    return out_path


def resolve_base_path() -> Path:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    candidates = [
        Path.cwd() / "riboai_queueing",
        repo_root / "results" / "riboai_queueing",
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    return candidates[-1]


def main() -> None:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    base_path = resolve_base_path()
    dataset_encoding_path = repo_root / "Datasets" / "encodings" / "dataset_encoding.yaml"

    prediction_files = discover_prediction_files(base_path)
    if len(prediction_files) == 0:
        raise RuntimeError(f"No prediction parquet files found under {base_path}.")

    id_to_dataset = load_dataset_encoding(dataset_encoding_path)
    rows: list[dict[str, Any]] = []

    print(f"Base path: {base_path}")
    print(f"Prediction files: {len(prediction_files)}")
    print(f"Comparison baseline: {MIX_EXPERIMENT}")
    print(f"Components: {', '.join(COMPONENT_COLUMNS)}")

    for path in prediction_files:
        metadata = prediction_metadata(path, base_path)
        print(f"Loading {path}")

        try:
            df = pd.read_parquet(path)
            add_metrics_rows(
                rows,
                df=df,
                metadata=metadata,
                id_to_dataset=id_to_dataset,
            )
        except Exception as exc:
            print(f"[WARN] Failed to process {path}: {exc}")

    metrics_df = keep_mix_vs_single_rows(pd.DataFrame(rows))
    if metrics_df.empty:
        raise RuntimeError("No single-dataset vs mix metrics were computed.")

    out_csv = base_path / "pcc_metrics_mu_rho_single_vs_mix.csv"
    metrics_df.to_csv(out_csv, index=False)
    print(f"\nSaved metrics table to: {out_csv}")

    visible = metrics_df[
        metrics_df["n_transcripts"].fillna(0).astype(int) > 0
    ][
        [
            "split",
            "dataset",
            "component",
            "run_type",
            "resolved_column",
            "pcc",
            "ci_lower",
            "ci_upper",
            "n_transcripts",
        ]
    ]
    print("\n=== Single-dataset vs mix PCCs ===")
    print(visible.to_string(index=False))

    plot_dir = base_path / "pcc_metric_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for old_plot in plot_dir.glob("pcc_*.png"):
        old_plot.unlink()

    for split in sorted(metrics_df["split"].dropna().unique()):
        for component in COMPONENT_COLUMNS:
            out_path = plot_component_comparison(
                metrics_df,
                component=component,
                split=str(split),
                out_dir=plot_dir,
            )
            if out_path is not None:
                print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
