#!/usr/bin/env python3
"""Analyze held-out shared-profile convergence across independent panels."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_NAME = "my_panels_a100_b32_20260906_114323"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "results" / DEFAULT_RUN_NAME
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utils.publication_plot_style import latex_paper_style
from analyses.paths import artifact_directory


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify common test identities and compare L_t from independently "
            "trained real-dataset panels."
        )
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help=(
            "Panel-experiment root. Default: "
            f"results/{DEFAULT_RUN_NAME} relative to the project root."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mean-one-tolerance", type=float, default=1.0e-4)
    parser.add_argument(
        "--require-all-panels",
        action="store_true",
        help=(
            "Fail unless every panel in panel_manifest.json has a usable "
            "best-validation-loss prediction. By default, analyze all currently "
            "available panels when at least two are complete."
        ),
    )
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.var(left) <= 0.0 or np.var(right) <= 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(left).rank(method="average").to_numpy(dtype=np.float64)
    right_rank = pd.Series(right).rank(method="average").to_numpy(dtype=np.float64)
    return _pearson(left_rank, right_rank)


def _resolve_artifact_path(
    raw: str,
    *,
    manifest_path: Path,
    local_search_root: Path | None = None,
) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (manifest_path.parent / path).resolve()
    if path.exists() or local_search_root is None:
        return path

    # Result directories are commonly copied from Leonardo to a workstation.
    # Runtime manifests retain their original absolute /leonardo_work path, so
    # recover the same named artifact inside the copied panel directory.
    matches = sorted(local_search_root.rglob(path.name))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise RuntimeError(
            f"Cannot remap stale artifact path {path}: found multiple local "
            f"matches below {local_search_root}: {matches}."
        )
    return path


def _locate_panel_prediction(
    panel_directory: Path,
) -> tuple[Path | None, Path | None, str]:
    """Locate the authoritative best-validation-loss prediction for one panel.

    The orchestrator writes ``scientific_checkpoint_manifest.json`` only after
    every panel process has returned successfully. A scheduler timeout or one
    failed panel can therefore leave valid completed-panel predictions without
    that convenience wrapper. In that case, use the training entrypoint's
    authoritative ``prediction_checkpoint_manifest.json`` directly.
    """

    scientific_path = panel_directory / "scientific_checkpoint_manifest.json"
    if scientific_path.exists():
        scientific = _read_json(scientific_path)
        if scientific.get("checkpoint_variant") != "best_val_loss":
            raise ValueError(
                f"{panel_directory.name} scientific prediction is not based on "
                "best_val_loss."
            )
        raw_prediction_path = scientific.get("prediction_path")
        if not raw_prediction_path:
            raise KeyError(f"{scientific_path} has no prediction_path.")
        source_manifest = scientific_path
        source_kind = "scientific_checkpoint_manifest"
    else:
        runtime_manifests = sorted(
            (panel_directory / "predictions").rglob(
                "prediction_checkpoint_manifest.json"
            )
        )
        if not runtime_manifests:
            return (
                None,
                None,
                "no scientific_checkpoint_manifest.json or "
                "prediction_checkpoint_manifest.json",
            )
        if len(runtime_manifests) != 1:
            raise RuntimeError(
                f"Expected one runtime prediction manifest for "
                f"{panel_directory.name}, found {len(runtime_manifests)}: "
                f"{runtime_manifests}."
            )
        source_manifest = runtime_manifests[0]
        runtime = _read_json(source_manifest)
        if set(runtime) != {"best_val_loss"}:
            raise ValueError(
                f"{panel_directory.name} runtime prediction variants are "
                f"{sorted(runtime)}; scientific analysis requires only "
                "best_val_loss."
            )
        record = runtime["best_val_loss"]
        if record.get("split_name") != "test":
            raise ValueError(
                f"{panel_directory.name} best_val_loss prediction is not from "
                "the held-out test split."
            )
        raw_prediction_path = record.get("output_path")
        if not raw_prediction_path:
            raise KeyError(f"{source_manifest} has no best_val_loss.output_path.")
        source_kind = "runtime_prediction_checkpoint_manifest"

    prediction_path = _resolve_artifact_path(
        str(raw_prediction_path),
        manifest_path=source_manifest,
        local_search_root=panel_directory / "predictions",
    )

    # Prefer an already materialized, compact L-only artifact. This is both
    # faster and more robust when a copied multi-dataset prediction parquet was
    # interrupted after the compact artifact had already been produced. The
    # authoritative best-validation-loss runtime/scientific manifest above is
    # still required; an unrelated compact file is never accepted on its own.
    compact_candidates = sorted(
        (panel_directory / "predictions").rglob(
            "common_test_L_profiles.parquet"
        )
    )
    valid_compact: list[Path] = []
    for candidate in compact_candidates:
        try:
            import pyarrow.parquet as pq

            columns = set(pq.read_schema(candidate).names)
        except Exception:
            continue
        if {"transcript_id", "transcript_length", "L_t"} <= columns:
            valid_compact.append(candidate.resolve())
    if len(valid_compact) > 1:
        raise RuntimeError(
            f"Expected at most one valid compact shared-profile artifact for "
            f"{panel_directory.name}, found {valid_compact}."
        )
    if valid_compact:
        return (
            valid_compact[0],
            source_manifest,
            f"{source_kind}+compact_common_test_L",
        )

    if not prediction_path.exists():
        return (
            None,
            source_manifest,
            f"prediction artifact is missing: {prediction_path}",
        )
    return prediction_path, source_manifest, source_kind


def _extract_panel_profiles(
    *,
    panel_name: str,
    run_identifier: str,
    prediction_path: Path,
    expected_ids: set[str],
    mean_one_tolerance: float,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    import pyarrow.parquet as pq

    available = set(pq.read_schema(prediction_path).names)
    compact_required = {"transcript_id", "transcript_length", "L_t"}
    raw_required = {"transcript_id", "length", "mask", "L_bio"}
    is_compact = compact_required <= available
    if not is_compact and not raw_required <= available:
        raise KeyError(
            f"Prediction artifact {prediction_path} has neither the compact "
            f"shared-profile schema {sorted(compact_required)} nor the raw "
            f"prediction schema {sorted(raw_required)}; columns={sorted(available)}."
        )

    if is_compact:
        frame = pd.read_parquet(
            prediction_path,
            columns=sorted(compact_required),
        )
        frame["transcript_id"] = frame["transcript_id"].astype(str)
        if frame["transcript_id"].duplicated().any():
            duplicates = sorted(
                frame.loc[
                    frame["transcript_id"].duplicated(keep=False),
                    "transcript_id",
                ].unique()
            )
            raise ValueError(
                f"{panel_name} compact shared-profile artifact contains "
                f"duplicate transcript IDs: {duplicates[:10]}."
            )
        observed = set(frame["transcript_id"])
        if observed != expected_ids:
            raise ValueError(
                f"{panel_name} compact test transcript identity mismatch: "
                f"missing={sorted(expected_ids - observed)[:10]}, "
                f"extra={sorted(observed - expected_ids)[:10]}."
            )
        profiles: dict[str, dict[str, Any]] = {}
        mean_rows: list[dict[str, Any]] = []
        for row in frame.itertuples(index=False):
            transcript_id = str(row.transcript_id)
            values = np.asarray(row.L_t, dtype=np.float64)
            length = int(row.transcript_length)
            if values.ndim != 1 or values.size != length or length <= 0:
                raise ValueError(
                    f"{panel_name}/{transcript_id}: invalid compact L_t/length."
                )
            if not np.isfinite(values).all() or np.any(values <= 0.0):
                raise ValueError(
                    f"{panel_name}/{transcript_id}: compact L_t is non-finite "
                    "or non-positive."
                )
            mean_value = float(values.mean())
            deviation = abs(mean_value - 1.0)
            if deviation > float(mean_one_tolerance):
                raise ValueError(
                    f"{panel_name}/{transcript_id}: compact L_t "
                    f"mean={mean_value:.8f}, outside mean-one tolerance "
                    f"{mean_one_tolerance}. No analysis renormalization was applied."
                )
            profiles[transcript_id] = {"values": values, "length": length}
            mean_rows.append(
                {
                    "panel": panel_name,
                    "transcript_id": transcript_id,
                    "transcript_length": length,
                    "L_mean": mean_value,
                    "absolute_mean_one_deviation": deviation,
                    "number_of_prediction_dataset_rows": 1,
                }
            )
        return profiles, pd.DataFrame(mean_rows)

    # Raw exports repeat each shared profile across many dataset rows and can
    # have a single enormous row group. Never materialize that table in pandas.
    # Retain only one unpadded L profile and mask per held-out transcript.
    profiles: dict[str, dict[str, Any]] = {}
    masks: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    with pq.ParquetFile(prediction_path) as reader:
        for batch in reader.iter_batches(
            batch_size=16, columns=sorted(raw_required), use_threads=False
        ):
            for row in batch.to_pylist():
                transcript_id = str(row["transcript_id"])
                if transcript_id not in expected_ids:
                    raise ValueError(f"{panel_name}: unexpected test ID {transcript_id}.")
                profile = np.asarray(row["L_bio"], dtype=np.float64)
                mask = np.asarray(row["mask"], dtype=bool)
                length = int(row["length"])
                if profile.ndim != 1 or mask.ndim != 1 or profile.shape != mask.shape:
                    raise ValueError(
                        f"{panel_name}/{transcript_id}: invalid L_bio or mask shape."
                    )
                if length != int(mask.sum()) or length <= 0:
                    raise ValueError(f"{panel_name}/{transcript_id}: length/mask disagreement.")
                values = profile[mask]
                if not np.isfinite(values).all() or np.any(values <= 0):
                    raise ValueError(f"{panel_name}/{transcript_id}: invalid L_bio values.")
                if transcript_id not in profiles:
                    profiles[transcript_id] = {"values": values, "length": length}
                    masks[transcript_id] = mask
                    counts[transcript_id] = 1
                else:
                    previous = profiles[transcript_id]
                    if (
                        previous["length"] != length
                        or not np.array_equal(masks[transcript_id], mask)
                        or not np.allclose(previous["values"], values, rtol=2e-5, atol=2e-6)
                    ):
                        raise ValueError(
                            f"{panel_name}/{transcript_id}: L_bio differs across dataset rows."
                        )
                    counts[transcript_id] += 1
            del batch
    observed = set(profiles)
    if observed != expected_ids:
        raise ValueError(
            f"{panel_name} test transcript identity mismatch: "
            f"missing={sorted(expected_ids - observed)[:10]}, "
            f"extra={sorted(observed - expected_ids)[:10]}."
        )
    del masks
    mean_rows: list[dict[str, Any]] = []
    for transcript_id in sorted(profiles):
        valid_profile = profiles[transcript_id]["values"]
        canonical_length = profiles[transcript_id]["length"]
        mean_value = float(valid_profile.mean())
        deviation = abs(mean_value - 1.0)
        if deviation > float(mean_one_tolerance):
            raise ValueError(
                f"{panel_name}/{transcript_id}: L_bio mean={mean_value:.8f}, "
                f"outside mean-one tolerance {mean_one_tolerance}. No analysis "
                "renormalization was applied."
            )
        mean_rows.append(
            {
                "panel": panel_name,
                "transcript_id": str(transcript_id),
                "transcript_length": canonical_length,
                "L_mean": mean_value,
                "absolute_mean_one_deviation": deviation,
                "number_of_prediction_dataset_rows": counts[transcript_id],
            }
        )
    import pyarrow as pa

    profile_artifact = prediction_path.parent / "common_test_L_profiles.parquet"
    temporary = profile_artifact.with_suffix(".parquet.partial")
    schema = pa.schema([
        ("transcript_id", pa.string()), ("panel", pa.string()),
        ("run_identifier", pa.string()), ("transcript_length", pa.int64()),
        ("L_t", pa.list_(pa.float32())),
    ])
    ids = sorted(profiles)
    with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
        for start in range(0, len(ids), 32):
            rows = [dict(transcript_id=tid, panel=panel_name,
                         run_identifier=run_identifier,
                         transcript_length=profiles[tid]["length"],
                         L_t=profiles[tid]["values"].astype(np.float32()))
                    for tid in ids[start:start + 32]]
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    temporary.replace(profile_artifact)
    return profiles, pd.DataFrame(mean_rows)


def _summary_table(agreement: pd.DataFrame) -> pd.DataFrame:
    groups: list[tuple[str, pd.DataFrame]] = [
        (str(pair), group)
        for pair, group in agreement.groupby("panel_pair", sort=True)
    ]
    groups.append(("pooled_all_pairs", agreement))
    rows: list[dict[str, Any]] = []
    for group_name, group in groups:
        for metric in ("PCC", "Spearman", "RMSE"):
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(
                dtype=np.float64
            )
            if not len(values):
                statistics = {key: float("nan") for key in ("mean", "median", "IQR", "p05", "p25", "p50", "p75", "p95")}
            else:
                percentiles = np.percentile(values, [5, 25, 50, 75, 95])
                statistics = {
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "IQR": float(percentiles[3] - percentiles[1]),
                    "p05": float(percentiles[0]),
                    "p25": float(percentiles[1]),
                    "p50": float(percentiles[2]),
                    "p75": float(percentiles[3]),
                    "p95": float(percentiles[4]),
                }
            rows.append(
                {
                    "group": group_name,
                    "metric": metric,
                    "n": int(len(values)),
                    **statistics,
                }
            )
    return pd.DataFrame(rows)


@latex_paper_style
def _save_pair_agreement_figure(agreement: pd.DataFrame, figure_dir: Path) -> None:
    pairs = sorted(agreement["panel_pair"].unique())
    data = [
        agreement.loc[agreement["panel_pair"] == pair, "PCC"]
        .dropna()
        .to_numpy(dtype=np.float64)
        for pair in pairs
    ]
    figure, axis = plt.subplots(figsize=(8.2, 4.7), constrained_layout=True)
    violin = axis.violinplot(
        data, positions=np.arange(1, len(pairs) + 1), showextrema=False, widths=0.82
    )
    for body in violin["bodies"]:
        body.set_facecolor("#4c78a8")
        body.set_edgecolor("#315b7d")
        body.set_alpha(0.42)
    axis.boxplot(
        data,
        positions=np.arange(1, len(pairs) + 1),
        widths=0.18,
        showfliers=False,
        medianprops={"color": "#d62728", "linewidth": 1.5},
    )
    axis.set_xticks(np.arange(1, len(pairs) + 1), pairs)
    axis.set_xlabel("Independent panel pair")
    axis.set_ylabel("Per-transcript PCC of held-out $L_t$")
    axis.set_ylim(-1.02, 1.02)
    axis.grid(axis="y", alpha=0.22)
    axis.set_title("Cross-panel shared-profile agreement")
    figure.savefig(figure_dir / "cross_panel_L_PCC_by_pair.pdf", bbox_inches="tight")
    figure.savefig(
        figure_dir / "cross_panel_L_PCC_by_pair.png", dpi=300, bbox_inches="tight"
    )
    plt.close(figure)

    pooled = agreement["PCC"].dropna().to_numpy(dtype=np.float64)
    figure, axis = plt.subplots(figsize=(6.3, 4.2), constrained_layout=True)
    axis.hist(pooled, bins=50, color="#4c78a8", alpha=0.82, edgecolor="white")
    axis.axvline(np.median(pooled), color="#d62728", linewidth=1.5, label=f"median={np.median(pooled):.3f}")
    axis.set_xlabel("PCC across all transcript–panel-pair comparisons")
    axis.set_ylabel("Count")
    axis.set_title("Pooled held-out shared-profile agreement")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.18)
    figure.savefig(figure_dir / "cross_panel_L_PCC_pooled.pdf", bbox_inches="tight")
    figure.savefig(
        figure_dir / "cross_panel_L_PCC_pooled.png", dpi=300, bbox_inches="tight"
    )
    plt.close(figure)


@latex_paper_style
def _representative_transcripts(
    agreement: pd.DataFrame,
    *,
    profiles: Mapping[str, Mapping[str, Mapping[str, Any]]],
    output_dir: Path,
    figure_dir: Path,
) -> None:
    pooled = (
        agreement.groupby("transcript_id", sort=True)[["PCC", "Spearman", "RMSE"]]
        .median()
        .rename(
            columns={
                "PCC": "median_pairwise_PCC",
                "Spearman": "median_pairwise_Spearman",
                "RMSE": "median_pairwise_RMSE",
            }
        )
        .reset_index()
    )
    pooled.to_csv(output_dir / "per_transcript_pooled_agreement.csv", index=False)
    specifications = [
        ("lower_agreement", 0.10),
        ("median_agreement", 0.50),
        ("high_agreement", 0.90),
    ]
    finite = pooled.loc[np.isfinite(pooled["median_pairwise_PCC"])].copy()
    if finite.empty:
        raise RuntimeError("No finite pooled PCC values for representative profiles.")
    selected_rows: list[dict[str, Any]] = []
    used: set[str] = set()
    for label, quantile in specifications:
        target = float(finite["median_pairwise_PCC"].quantile(quantile))
        ordered = finite.assign(
            distance=(finite["median_pairwise_PCC"] - target).abs()
        ).sort_values(["distance", "transcript_id"])
        row = next(
            candidate
            for candidate in ordered.to_dict(orient="records")
            if str(candidate["transcript_id"]) not in used or len(finite) < 3
        )
        transcript_id = str(row["transcript_id"])
        used.add(transcript_id)
        selected_rows.append(
            {
                "selection_label": label,
                "selection_quantile": quantile,
                "target_PCC_quantile_value": target,
                **row,
            }
        )
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(output_dir / "representative_transcripts.csv", index=False)

    panels = sorted(profiles)
    colors = dict(zip(panels, plt.get_cmap("tab10")(np.linspace(0, 0.75, len(panels))), strict=True))
    figure, axes = plt.subplots(
        len(selected), 1, figsize=(10.2, 2.9 * len(selected)), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    numeric_rows: list[dict[str, Any]] = []
    for axis, row in zip(axes, selected.itertuples(index=False), strict=True):
        transcript_id = str(row.transcript_id)
        for panel in panels:
            values = np.asarray(profiles[panel][transcript_id]["values"])
            positions = np.arange(1, len(values) + 1)
            axis.plot(
                positions,
                values,
                color=colors[panel],
                linewidth=0.9,
                alpha=0.9,
                label=panel,
            )
            numeric_rows.extend(
                {
                    "selection_label": row.selection_label,
                    "transcript_id": transcript_id,
                    "panel": panel,
                    "codon_position": int(position),
                    "L_t": float(value),
                }
                for position, value in zip(positions, values, strict=True)
            )
        axis.set_title(
            f"{row.selection_label.replace('_', ' ').title()}: {transcript_id} "
            f"(median pairwise PCC={row.median_pairwise_PCC:.3f})"
        )
        axis.set_xlabel("CDS codon position")
        axis.set_ylabel("Mean-one $L_t$")
        axis.grid(alpha=0.16)
    axes[0].legend(ncol=len(panels), frameon=False, loc="upper right")
    figure.savefig(figure_dir / "representative_heldout_L_profiles.pdf", bbox_inches="tight")
    figure.savefig(
        figure_dir / "representative_heldout_L_profiles.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)
    pd.DataFrame(numeric_rows).to_parquet(
        output_dir / "representative_heldout_L_profiles.parquet",
        engine="pyarrow",
        index=False,
    )


@latex_paper_style
def _agreement_matrix(
    agreement: pd.DataFrame, *, output_dir: Path, figure_dir: Path, panels: Sequence[str]
) -> None:
    matrix = pd.DataFrame(np.eye(len(panels)), index=panels, columns=panels)
    for row in agreement.groupby(["panel_a", "panel_b"], sort=True)["PCC"].median().reset_index().itertuples(index=False):
        matrix.loc[row.panel_a, row.panel_b] = row.PCC
        matrix.loc[row.panel_b, row.panel_a] = row.PCC
    matrix.to_csv(output_dir / "median_pairwise_L_agreement_matrix.csv", index_label="panel")
    figure, axis = plt.subplots(figsize=(5.0, 4.4), constrained_layout=True)
    image = axis.imshow(matrix.to_numpy(dtype=float), vmin=-1.0, vmax=1.0, cmap="coolwarm")
    labels = [panel.replace("panel_", "P") for panel in panels]
    axis.set_xticks(range(len(panels)), labels)
    axis.set_yticks(range(len(panels)), labels)
    for row_index in range(len(panels)):
        for column_index in range(len(panels)):
            axis.text(
                column_index,
                row_index,
                f"{matrix.iloc[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                fontsize=12,
                color="black",
            )
    axis.set_title("Median held-out $L_t$ PCC")
    figure.colorbar(image, ax=axis, shrink=0.82, label="Median PCC")
    figure.savefig(figure_dir / "median_pairwise_L_agreement_matrix.pdf", bbox_inches="tight")
    figure.savefig(
        figure_dir / "median_pairwise_L_agreement_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = args.run_root.expanduser().resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(
            f"Panel experiment directory does not exist: {run_root}. "
            "Pass --run-root to analyze another run."
        )
    print(f"Panel experiment root: {run_root}")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else artifact_directory("real_data", run_root, "panel_convergence")
    )
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    common_split = _read_json(run_root / "common_split_manifest.json")
    panel_manifest = _read_json(run_root / "panel_manifest.json")
    planned_panels = sorted(panel_manifest["panels"])
    if len(planned_panels) < 2:
        raise ValueError(
            f"Convergence analysis requires at least two panels, got "
            f"{planned_panels}."
        )
    expected_ids = set(map(str, common_split["common_test_ids"]))
    if not expected_ids:
        raise ValueError("Common test split is empty.")

    resolved_predictions: dict[str, Path] = {}
    availability_rows: list[dict[str, Any]] = []
    unavailable_panels: dict[str, str] = {}
    for panel_name in planned_panels:
        prediction_path, source_manifest, detail = _locate_panel_prediction(
            run_root / panel_name
        )
        if prediction_path is None:
            unavailable_panels[panel_name] = detail
            availability_rows.append(
                {
                    "panel": panel_name,
                    "status": "unavailable",
                    "prediction_path": None,
                    "source_manifest": (
                        str(source_manifest) if source_manifest is not None else None
                    ),
                    "detail": detail,
                }
            )
            continue
        resolved_predictions[panel_name] = prediction_path
        availability_rows.append(
            {
                "panel": panel_name,
                "status": "available",
                "prediction_path": str(prediction_path),
                "source_manifest": str(source_manifest),
                "detail": detail,
            }
        )

    availability = pd.DataFrame(availability_rows)
    availability.to_csv(output_dir / "panel_analysis_availability.csv", index=False)
    if unavailable_panels and args.require_all_panels:
        details = "; ".join(
            f"{panel}: {reason}"
            for panel, reason in sorted(unavailable_panels.items())
        )
        raise RuntimeError(f"Not all planned panels are available: {details}")
    panels = sorted(resolved_predictions)
    if len(panels) < 2:
        details = "; ".join(
            f"{panel}: {reason}"
            for panel, reason in sorted(unavailable_panels.items())
        )
        raise RuntimeError(
            "Convergence analysis needs predictions from at least two panels; "
            f"available={panels}. Unavailable: {details}"
        )
    analysis_is_complete = not unavailable_panels
    warning_path = output_dir / "PARTIAL_ANALYSIS_WARNING.txt"
    if analysis_is_complete:
        warning_path.unlink(missing_ok=True)
    else:
        warning = (
            "PARTIAL ANALYSIS: only "
            f"{', '.join(panels)} of the planned panels "
            f"{', '.join(planned_panels)} had usable best-validation-loss "
            "predictions. Re-run this analyzer after all panels finish before "
            "using results as the final four-panel scientific comparison.\n"
        )
        warning_path.write_text(warning, encoding="utf-8")
        print(f"WARNING: {warning.strip()}")

    profiles: dict[str, dict[str, dict[str, Any]]] = {}
    mean_checks: list[pd.DataFrame] = []
    panel_profile_artifacts: dict[str, str] = {}
    for panel_name in panels:
        prediction_path = resolved_predictions[panel_name]
        profiles[panel_name], checks = _extract_panel_profiles(
            panel_name=panel_name,
            run_identifier=str(panel_manifest["outer_run_identifier"]),
            prediction_path=prediction_path,
            expected_ids=expected_ids,
            mean_one_tolerance=args.mean_one_tolerance,
        )
        mean_checks.append(checks)
        panel_profile_artifacts[panel_name] = str(
            prediction_path.parent / "common_test_L_profiles.parquet"
        )
    observed_sets = [set(panel_profiles) for panel_profiles in profiles.values()]
    if any(observed != expected_ids for observed in observed_sets):
        raise AssertionError("Panel profile transcript IDs are not identical.")
    mean_check_table = pd.concat(mean_checks, ignore_index=True)
    mean_check_table.to_csv(output_dir / "L_mean_one_checks.csv", index=False)

    rows: list[dict[str, Any]] = []
    for panel_a, panel_b in itertools.combinations(panels, 2):
        pair_label = (
            f"{int(panel_a.removeprefix('panel_'))}-"
            f"{int(panel_b.removeprefix('panel_'))}"
        )
        for transcript_id in sorted(expected_ids):
            left = np.asarray(profiles[panel_a][transcript_id]["values"], dtype=np.float64)
            right = np.asarray(profiles[panel_b][transcript_id]["values"], dtype=np.float64)
            if left.shape != right.shape:
                raise ValueError(
                    f"Transcript length differs for {transcript_id}: "
                    f"{panel_a}={left.size}, {panel_b}={right.size}."
                )
            valid = np.isfinite(left) & np.isfinite(right)
            if int(valid.sum()) != left.size:
                raise ValueError(f"Non-finite valid position for {transcript_id}.")
            left_valid = left[valid]
            right_valid = right[valid]
            rows.append(
                {
                    "transcript_id": transcript_id,
                    "panel_a": panel_a,
                    "panel_b": panel_b,
                    "panel_pair": pair_label,
                    "PCC": _pearson(left_valid, right_valid),
                    "Spearman": _spearman(left_valid, right_valid),
                    "RMSE": float(np.sqrt(np.mean(np.square(left_valid - right_valid)))),
                    "transcript_length": int(valid.sum()),
                }
            )
    agreement = pd.DataFrame(rows)
    expected_pair_count = math.comb(len(panels), 2)
    if len(agreement) != len(expected_ids) * expected_pair_count:
        raise AssertionError(
            f"Expected {expected_pair_count} panel-pair rows per transcript."
        )
    agreement.to_csv(output_dir / "cross_panel_L_agreement_long.csv", index=False)
    agreement.to_parquet(
        output_dir / "cross_panel_L_agreement_long.parquet",
        engine="pyarrow",
        index=False,
    )
    summary = _summary_table(agreement)
    summary.to_csv(output_dir / "cross_panel_L_agreement_summary.csv", index=False)

    assignment = pd.read_csv(run_root / "panel_assignment.csv")
    from Utils.real_panel_convergence import plot_panel_balance

    plot_panel_balance(assignment, output_directory=figure_dir)
    _save_pair_agreement_figure(agreement, figure_dir)
    _representative_transcripts(
        agreement,
        profiles=profiles,
        output_dir=output_dir,
        figure_dir=figure_dir,
    )
    _agreement_matrix(
        agreement,
        output_dir=output_dir,
        figure_dir=figure_dir,
        panels=panels,
    )
    pooled_pcc = summary.loc[
        (summary["group"] == "pooled_all_pairs") & (summary["metric"] == "PCC")
    ].iloc[0]
    analysis_manifest = {
        "run_root": str(run_root),
        "analysis_complete_for_planned_panels": analysis_is_complete,
        "planned_panels": planned_panels,
        "panels": panels,
        "unavailable_panels": unavailable_panels,
        "common_test_transcript_count": len(expected_ids),
        "panel_profile_artifacts": panel_profile_artifacts,
        "normalization_applied_during_analysis": False,
        "maximum_absolute_L_mean_one_deviation": float(
            mean_check_table["absolute_mean_one_deviation"].max()
        ),
        "pooled_PCC_median": float(pooled_pcc["median"]),
        "pooled_PCC_IQR": float(pooled_pcc["IQR"]),
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.5g}"))
    print(f"\nAnalysis complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
