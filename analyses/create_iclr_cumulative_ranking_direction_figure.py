#!/usr/bin/env python3
"""Create the ICLR figure for cumulative ranking directionality.

The two panels use a common equal-reference N=2 anchor within each dataset
selection direction.  The best-first panel reads the audited transcript-level
comparisons from ``cumulative_stability_seed42``.  The worst-first panel is
computed from the validated compact profile exports because the existing
worst-first diagnostic used policy-specific anchors.

The plotted intervals resample held-out transcripts jointly across every cell
within a panel.  They quantify held-out-transcript sampling variation and do
not quantify variability over model initialization or optimization seeds.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utils.publication_plot_style import publication_rc


SIZES = [2, 5, 10, 20, 40, 80, 114]
POLICY_ORDER = ["equal", "ranked_p1", "ranked_p3", "reverse_p1", "reverse_p3"]
POLICY_LABELS = {
    "equal": "Equal reference",
    "ranked_p1": r"Ranked $p=1$",
    "ranked_p3": r"Ranked $p=3$",
    "reverse_p1": r"Reversed $p=1$",
    "reverse_p3": r"Reversed $p=3$",
}
POLICY_COLORS = {
    "equal": "#333333",
    "ranked_p1": "#0072B2",
    "ranked_p3": "#D55E00",
    "reverse_p1": "#56B4E9",
    "reverse_p3": "#E69F00",
}
POLICY_LINESTYLES = {
    "equal": "--",
    "ranked_p1": "-",
    "ranked_p3": "-",
    "reverse_p1": ":",
    "reverse_p3": ":",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--best-first-root",
        type=Path,
        default=PROJECT_ROOT / "results/cumulative_stability_seed42",
    )
    parser.add_argument(
        "--selection-direction-root",
        type=Path,
        default=PROJECT_ROOT / "results/cumulative_selection_direction_seed42",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "figures/assets_5_real_datasets_4_panels",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=5_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_recorded_path(recorded: str, result_root: Path) -> Path:
    """Resolve an availability-table path after moving an experiment root."""

    path = Path(recorded)
    if path.is_file():
        return path
    parts = path.parts
    matches = [index for index, part in enumerate(parts) if part == result_root.name]
    if matches:
        candidate = result_root.joinpath(*parts[matches[-1] + 1 :])
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot resolve validated profile export: {recorded}")


def validated_cell(
    availability: pd.DataFrame,
    result_root: Path,
    *,
    n_value: int,
    policy: str,
    selection_direction: str | None = None,
) -> tuple[Path, dict[str, object]]:
    selected = availability.loc[
        availability["N"].eq(n_value) & availability["reference_policy"].eq(policy)
        if "reference_policy" in availability.columns
        else availability["N"].eq(n_value) & availability["arm"].eq(policy)
    ].copy()
    if selection_direction is not None:
        directions = [selection_direction]
        if n_value == 114 and selection_direction == "worst_first":
            directions.append("shared_full")
        selected = selected.loc[selected["selection_direction"].isin(directions)]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one availability row for N={n_value}, policy={policy}, "
            f"direction={selection_direction}; found {len(selected)}."
        )
    row = selected.iloc[0]
    if row["status"] != "validated_predictions" or not bool(row["runtime_config_verified"]):
        raise ValueError(f"N={n_value}/{policy} is not a runtime-verified prediction export.")
    path = resolve_recorded_path(str(row["prediction_path"]), result_root)
    if int(row["n_test_transcripts"]) <= 0:
        raise ValueError(f"N={n_value}/{policy} has no audited test transcripts.")
    return path, row.to_dict()


def read_profiles(path: Path) -> pd.DataFrame:
    columns = [
        "transcript_id",
        "transcript_length",
        "L_t",
        "valid_position_mask",
        "run_id",
        "N",
        "L_mean",
    ]
    frame = pd.read_parquet(path, columns=columns)
    if frame.empty or frame["transcript_id"].duplicated().any():
        raise ValueError(f"{path} has an empty or duplicated transcript cohort.")
    if not np.isfinite(frame["L_mean"]).all() or float((frame["L_mean"] - 1.0).abs().max()) > 1e-4:
        raise ValueError(f"{path} contains a profile that is not mean-one normalized.")
    indexed = frame.set_index("transcript_id").sort_index()
    if not indexed.index.is_unique:
        raise ValueError(f"{path} has duplicated transcript identifiers.")
    return indexed


def pcc_rows_to_anchor(
    anchor: pd.DataFrame,
    current: pd.DataFrame,
    *,
    selection_path: str,
    policy: str,
    n_value: int,
) -> list[dict[str, object]]:
    if not anchor.index.equals(current.index):
        raise ValueError(f"{selection_path}/N={n_value}/{policy} uses a different transcript cohort.")

    rows: list[dict[str, object]] = []
    for transcript_id in anchor.index:
        left = anchor.loc[transcript_id]
        right = current.loc[transcript_id]
        if int(left["transcript_length"]) != int(right["transcript_length"]):
            raise ValueError(f"{transcript_id}: transcript lengths differ across compared models.")
        left_mask = np.asarray(left["valid_position_mask"], dtype=bool)
        right_mask = np.asarray(right["valid_position_mask"], dtype=bool)
        if left_mask.shape != right_mask.shape or not np.array_equal(left_mask, right_mask):
            raise ValueError(f"{transcript_id}: valid-position masks differ across compared models.")
        left_values = np.asarray(left["L_t"], dtype=float)
        right_values = np.asarray(right["L_t"], dtype=float)
        if left_values.shape != left_mask.shape or right_values.shape != right_mask.shape:
            raise ValueError(f"{transcript_id}: profile and mask lengths differ.")
        x, y = left_values[left_mask], right_values[right_mask]
        if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"{transcript_id}: invalid values in a profile comparison.")
        x_centered, y_centered = x - x.mean(), y - y.mean()
        denominator = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
        if denominator <= 0:
            raise ValueError(f"{transcript_id}: a profile is constant, so PCC is undefined.")
        pcc = float(np.clip(np.dot(x_centered, y_centered) / denominator, -1.0, 1.0))
        rows.append(
            {
                "selection_path": selection_path,
                "anchor_policy": "equal",
                "anchor_N": 2,
                "policy": policy,
                "N": n_value,
                "transcript_id": str(transcript_id),
                "PCC": pcc,
            }
        )
    return rows


def availability_audit_rows(
    availability: pd.DataFrame,
    result_root: Path,
    cells: set[tuple[int, str]],
    *,
    selection_direction: str | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n_value, policy in sorted(cells):
        path, record = validated_cell(
            availability,
            result_root,
            n_value=n_value,
            policy=policy,
            selection_direction=selection_direction,
        )
        rows.append(
            {
                "selection_path": selection_direction or "best_first",
                "N": n_value,
                "policy": policy,
                "status": record["status"],
                "runtime_config_verified": bool(record["runtime_config_verified"]),
                "n_test_transcripts": int(record["n_test_transcripts"]),
                "prediction_path": str(path),
                "prediction_sha256": str(record["prediction_sha256"]),
                "prediction_exists": path.is_file(),
            }
        )
    return rows


def load_best_first_metrics(best_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_path = best_root / "analysis/transcript_stability.csv"
    availability_path = best_root / "analysis/availability.csv"
    metrics = pd.read_csv(metric_path)
    availability = pd.read_csv(availability_path)
    require_columns(
        metrics,
        {"kind", "N_a", "N_b", "arm", "transcript_id", "PCC", "reason"},
        metric_path,
    )
    require_columns(
        availability,
        {
            "N",
            "arm",
            "status",
            "runtime_config_verified",
            "n_test_transcripts",
            "prediction_path",
            "prediction_sha256",
        },
        availability_path,
    )

    selected = metrics.loc[metrics["kind"].eq("shared_anchor")].copy()
    if selected.empty or not selected["N_a"].eq(2).all():
        raise ValueError("The best-first table lacks comparisons to the common N=2 anchor.")
    if selected["PCC"].isna().any() or not selected["reason"].eq("ok").all():
        raise ValueError("The best-first common-anchor table contains an invalid PCC.")
    selected = selected.rename(columns={"N_b": "N", "arm": "policy"})
    selected = selected[["N", "policy", "transcript_id", "PCC"]]
    selected.insert(0, "anchor_N", 2)
    selected.insert(0, "anchor_policy", "equal")
    selected.insert(0, "selection_path", "best_first")

    group_sizes = selected.groupby(["N", "policy"]).size()
    if not group_sizes.eq(1771).all():
        raise ValueError("A best-first comparison does not use all 1,771 audited transcripts.")
    expected_ids = None
    for _, group in selected.groupby(["N", "policy"], sort=False):
        ids = set(group["transcript_id"].astype(str))
        if len(ids) != 1771 or (expected_ids is not None and ids != expected_ids):
            raise ValueError("Best-first comparisons do not share one transcript cohort.")
        expected_ids = ids

    # The analysis table starts at N=5.  Add direct N=2 policy comparisons to
    # the same equal-reference N=2 model so both figure panels expose their
    # policy-specific displacement already at the anchor collection.
    availability_for_lookup = availability.copy()
    availability_for_lookup["reference_policy"] = availability_for_lookup["arm"]
    anchor_path, _ = validated_cell(
        availability_for_lookup, best_root, n_value=2, policy="equal"
    )
    anchor = read_profiles(anchor_path)
    if set(anchor.index.astype(str)) != expected_ids:
        raise ValueError("The best-first N=2 export differs from the audited comparison cohort.")
    n2_rows: list[dict[str, object]] = []
    for policy in POLICY_ORDER:
        current_path, _ = validated_cell(
            availability_for_lookup, best_root, n_value=2, policy=policy
        )
        current = read_profiles(current_path)
        n2_rows.extend(
            pcc_rows_to_anchor(
                anchor,
                current,
                selection_path="best_first",
                policy=policy,
                n_value=2,
            )
        )
    selected = pd.concat([pd.DataFrame(n2_rows), selected], ignore_index=True)

    cells = set(zip(selected["N"].astype(int), selected["policy"].astype(str), strict=False))
    audit = pd.DataFrame(
        availability_audit_rows(
            availability_for_lookup,
            best_root,
            cells,
            selection_direction=None,
        )
    )
    return selected, audit


def load_worst_first_metrics(selection_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    availability_path = selection_root / "analysis/availability.csv"
    availability = pd.read_csv(availability_path)
    require_columns(
        availability,
        {
            "N",
            "reference_policy",
            "selection_direction",
            "status",
            "runtime_config_verified",
            "n_test_transcripts",
            "prediction_path",
            "prediction_sha256",
        },
        availability_path,
    )

    anchor_path, _ = validated_cell(
        availability,
        selection_root,
        n_value=2,
        policy="equal",
        selection_direction="worst_first",
    )
    anchor = read_profiles(anchor_path)
    if len(anchor) != 714:
        raise ValueError("The worst-first common anchor must contain 714 test transcripts.")

    rows: list[dict[str, object]] = []
    cells: set[tuple[int, str]] = set()
    for n_value in SIZES:
        for source_policy, plotted_policy in (
            ("equal", "equal"),
            ("quality_p3", "ranked_p3"),
        ):
            current_path, _ = validated_cell(
                availability,
                selection_root,
                n_value=n_value,
                policy=source_policy,
                selection_direction="worst_first",
            )
            current = read_profiles(current_path)
            rows.extend(
                pcc_rows_to_anchor(
                    anchor,
                    current,
                    selection_path="worst_first",
                    policy=plotted_policy,
                    n_value=n_value,
                )
            )
            cells.add((n_value, source_policy))

    audit = pd.DataFrame(
        availability_audit_rows(
            availability,
            selection_root,
            cells,
            selection_direction="worst_first",
        )
    )
    audit["policy"] = audit["policy"].replace({"quality_p3": "ranked_p3"})
    return pd.DataFrame(rows), audit


def bootstrap_summary(
    metrics: pd.DataFrame,
    *,
    n_resamples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    pivot = metrics.pivot(index="transcript_id", columns=["N", "policy"], values="PCC")
    pivot = pivot.reindex(
        columns=sorted(
            pivot.columns,
            key=lambda cell: (SIZES.index(int(cell[0])), POLICY_ORDER.index(str(cell[1]))),
        )
    )
    if pivot.isna().any().any() or not np.isfinite(pivot.to_numpy(dtype=float)).all():
        raise ValueError("A common-anchor PCC matrix is incomplete or non-finite.")

    values = pivot.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty((n_resamples, values.shape[1]), dtype=float)
    digest = hashlib.sha256()
    chunk_size = 64
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        digest.update(indices.astype("<i8").tobytes())
        boot[start:stop] = values[indices].mean(axis=1)

    lower, upper = np.quantile(boot, [0.025, 0.975], axis=0)
    rows: list[dict[str, object]] = []
    for column_index, (n_value, policy) in enumerate(pivot.columns):
        rows.append(
            {
                "N": int(n_value),
                "policy": str(policy),
                "mean": float(values[:, column_index].mean()),
                "median": float(np.median(values[:, column_index])),
                "ci_lower": float(lower[column_index]),
                "ci_upper": float(upper[column_index]),
                "n_transcripts": len(values),
                "confidence": 0.95,
                "bootstrap_resamples": n_resamples,
                "bootstrap_seed": seed,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(boot, columns=pivot.columns), digest.hexdigest()


def paired_contrasts(
    summary: pd.DataFrame,
    boot: pd.DataFrame,
    contrasts: list[tuple[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    available = set(boot.columns)
    for n_value in SIZES:
        for left, right in contrasts:
            left_key, right_key = (n_value, left), (n_value, right)
            if left_key not in available or right_key not in available:
                continue
            differences = boot[left_key].to_numpy() - boot[right_key].to_numpy()
            observed = (
                float(summary.loc[(summary["N"].eq(n_value)) & summary["policy"].eq(left), "mean"].iloc[0])
                - float(summary.loc[(summary["N"].eq(n_value)) & summary["policy"].eq(right), "mean"].iloc[0])
            )
            lower, upper = np.quantile(differences, [0.025, 0.975])
            rows.append(
                {
                    "N": n_value,
                    "contrast": f"{left}_minus_{right}",
                    "mean_difference": observed,
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                }
            )
    return pd.DataFrame(rows)


def adaptive_limits(summary: pd.DataFrame) -> tuple[float, float]:
    lower = float(summary["ci_lower"].min())
    upper = float(summary["ci_upper"].max())
    span = max(upper - lower, 0.15)
    return max(-1.0, lower - 0.045 * span), min(1.015, upper + 0.045 * span)


def emphasize_axis_text(ax: plt.Axes) -> None:
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")


def plot_panel(ax: plt.Axes, summary: pd.DataFrame, policies: list[str]) -> None:
    for policy in policies:
        part = summary.loc[summary["policy"].eq(policy)].sort_values("N")
        if part.empty:
            continue
        x = part["N"].to_numpy(dtype=float)
        ax.fill_between(
            x,
            part["ci_lower"].to_numpy(dtype=float),
            part["ci_upper"].to_numpy(dtype=float),
            color=POLICY_COLORS[policy],
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x,
            part["mean"].to_numpy(dtype=float),
            color=POLICY_COLORS[policy],
            linestyle=POLICY_LINESTYLES[policy],
            marker="o",
            markersize=7.2,
            markeredgewidth=0.7,
            markeredgecolor="white",
            linewidth=2.7,
            label=POLICY_LABELS[policy],
            zorder=2,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES, [str(value) for value in SIZES])
    ax.set_xlim(1.8, 126)
    ax.set_ylim(*adaptive_limits(summary))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(axis="both", alpha=0.22)
    emphasize_axis_text(ax)


def create_figure(
    best_summary: pd.DataFrame,
    worst_summary: pd.DataFrame,
    best_root: Path,
    selection_root: Path,
    output_dir: Path,
) -> None:
    best_concentration = pd.read_csv(best_root / "reference_concentration.csv")
    best_p3 = best_concentration.loc[
        best_concentration["arm"].eq("ranked_p3") & best_concentration["N"].le(20)
    ].copy()
    best_fraction = float((best_p3["N_ref"] / best_p3["N"]).min())

    worst_concentration = pd.read_csv(selection_root / "reference_concentration.csv")
    worst_n2 = worst_concentration.loc[
        worst_concentration["selection_direction"].eq("worst_first")
        & worst_concentration["reference_policy"].eq("quality_p3")
        & worst_concentration["N"].eq(2)
    ]
    if len(worst_n2) != 1:
        raise ValueError("Missing the worst-first N=2 reference-concentration diagnostic.")
    worst_fraction = float(worst_n2.iloc[0]["N_ref"] / 2.0)

    style = publication_rc()
    style.update(
        {
            "text.usetex": False,
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 14.5,
            "font.weight": "bold",
            "axes.labelsize": 15.0,
            "axes.labelweight": "bold",
            "axes.titlesize": 15.5,
            "axes.titleweight": "bold",
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "legend.fontsize": 12.1,
        }
    )
    with matplotlib.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.75), constrained_layout=True)

        plot_panel(axes[0], best_summary, POLICY_ORDER)
        axes[0].axvspan(1.8, 22.5, color="#6B7280", alpha=0.065, zorder=-4)
        axes[0].set_title(
            "A  Best-first: lower-ranked datasets are added",
            loc="left",
            pad=12,
        )
        axes[0].set_xlabel(r"Number of selected datasets, $N$")
        axes[0].set_ylabel(
            "Mean transcript PCC of " + r"$\mathbf{L}_t$" + "\n"
            + r"to the common equal-reference best-$N=2$ anchor"
        )
        axes[0].text(
            2.15,
            axes[0].get_ylim()[0] + 0.035 * np.diff(axes[0].get_ylim())[0],
            rf"$p=3$: $N_{{\mathrm{{eff}}}}/N\geq {best_fraction:.2f}$ through $N=20$",
            color="#5F6368",
            fontsize=11.2,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

        plot_panel(axes[1], worst_summary, ["equal", "ranked_p3"])
        axes[1].set_title(
            "B  Worst-first: higher-ranked datasets are added",
            loc="left",
            pad=12,
        )
        axes[1].set_xlabel(r"Number of selected datasets, $N$")
        axes[1].set_ylabel(
            "Mean transcript PCC of " + r"$\mathbf{L}_t$" + "\n"
            + r"to the common equal-reference worst-$N=2$ anchor"
        )
        axes[1].text(
            2.15,
            axes[1].get_ylim()[0] + 0.035 * np.diff(axes[1].get_ylim())[0],
            rf"$p=3$ is concentrated already at $N=2$ "
            rf"($N_{{\mathrm{{eff}}}}/N={worst_fraction:.2f}$)",
            color="#5F6368",
            fontsize=11.2,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.075),
            ncol=5,
            frameon=False,
            columnspacing=1.25,
            handlelength=2.5,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = output_dir / "appendix_cumulative_ranking_direction_L_PCC"
        fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(prefix.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    best_root = args.best_first_root.expanduser().resolve()
    selection_root = args.selection_direction_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    best_metrics, best_audit = load_best_first_metrics(best_root)
    worst_metrics, worst_audit = load_worst_first_metrics(selection_root)
    best_summary, best_boot, best_digest = bootstrap_summary(
        best_metrics,
        n_resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    worst_summary, worst_boot, worst_digest = bootstrap_summary(
        worst_metrics,
        n_resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed + 1,
    )
    best_summary.insert(0, "selection_path", "best_first")
    worst_summary.insert(0, "selection_path", "worst_first")
    best_summary["anchor"] = "common_equal_best_N2"
    worst_summary["anchor"] = "common_equal_worst_N2"
    best_summary["bootstrap_index_sha256"] = best_digest
    worst_summary["bootstrap_index_sha256"] = worst_digest
    source = pd.concat([best_summary, worst_summary], ignore_index=True)

    best_contrasts = paired_contrasts(
        best_summary,
        best_boot,
        [
            ("ranked_p1", "reverse_p1"),
            ("ranked_p3", "reverse_p3"),
            ("ranked_p1", "equal"),
            ("ranked_p3", "equal"),
        ],
    )
    best_contrasts.insert(0, "selection_path", "best_first")
    worst_contrasts = paired_contrasts(
        worst_summary,
        worst_boot,
        [("ranked_p3", "equal")],
    )
    worst_contrasts.insert(0, "selection_path", "worst_first")
    contrasts = pd.concat([best_contrasts, worst_contrasts], ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "appendix_cumulative_ranking_direction_L_PCC"
    source.to_csv(prefix.with_name(prefix.name + "_source.csv"), index=False)
    pd.concat([best_metrics, worst_metrics], ignore_index=True).to_csv(
        prefix.with_name(prefix.name + "_transcript_metrics.csv"), index=False
    )
    pd.concat([best_audit, worst_audit], ignore_index=True).to_csv(
        prefix.with_name(prefix.name + "_audit.csv"), index=False
    )
    contrasts.to_csv(prefix.with_name(prefix.name + "_paired_contrasts.csv"), index=False)

    create_figure(best_summary, worst_summary, best_root, selection_root, output_dir)
    print(f"Wrote {prefix.with_suffix('.pdf')}")
    print(f"Wrote {prefix.with_suffix('.png')}")
    print(f"Wrote source, transcript-level, audit, and paired-contrast CSV files.")


if __name__ == "__main__":
    main()
