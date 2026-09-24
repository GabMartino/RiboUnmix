#!/usr/bin/env python3
"""Validate benchmark table values and emphasis against the merged source CSV."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ORGANISM_GROUPS = (
    ("celegans_stein_2021", "ecoli_zhang_2016"),
    ("human_iwasaki_2014", "yeast_stein_2021"),
)
METRICS = ("pcc", "scc", "rmse_normalized")
ORGANISMS = (*ORGANISM_GROUPS[0], *ORGANISM_GROUPS[1])
PROTOCOLS = {
    "iXnos": ("Own preprocessing", "RiboUnmix unweighted", "RiboUnmix weighted"),
    "RiboExp": ("Own preprocessing", "RiboUnmix unweighted", "RiboUnmix weighted"),
    "Riboformer": ("Own preprocessing", "RiboUnmix unweighted", "RiboUnmix weighted"),
    "Seq2Ribo": ("Own preprocessing", "RiboUnmix unweighted", "RiboUnmix weighted"),
    "RiboMIMO": ("Own preprocessing", "RiboUnmix unweighted", "RiboUnmix weighted"),
    "RiboUnmix": ("Weighted",),
}
EXTERNAL_TRAINING = {
    "iXnos": {
        "Own preprocessing": "readme_filter__keep_zeros_full_canonical_test",
        "RiboUnmix unweighted": "unweighted",
        "RiboUnmix weighted": "weighted",
    },
    "RiboExp": {
        "Own preprocessing": "paper_top500__mask_zeros_full_canonical_test",
        "RiboUnmix unweighted": "unweighted",
        "RiboUnmix weighted": "weighted",
    },
    "Riboformer": {
        "Own preprocessing": "canonical_native_filtered_unweighted",
        "RiboUnmix unweighted": "unweighted",
        "RiboUnmix weighted": "weighted",
    },
    "Seq2Ribo": {
        "Own preprocessing": "paper_filtered_native_train_validation",
        "RiboUnmix unweighted": "unweighted",
        "RiboUnmix weighted": "weighted",
    },
}
CELL = re.compile(
    r"\\(?:(mci|bmci|bestmci)\{([0-9.]+)\}\{([0-9.]+)\}|(naentry))"
)
PROVENANCE_CELL = re.compile(
    r"([0-9][0-9,]*) \(([0-9]+)(?:\\textsuperscript\{\*\})?\)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser.parse_args()


def expected_cell(
    summary: pd.DataFrame,
    architecture: str,
    protocol: str,
    organism: str,
    metric: str,
) -> tuple[str, str, str]:
    selected = summary[
        summary["architecture"].eq(architecture)
        & summary["protocol"].eq(protocol)
        & summary["organism"].eq(organism)
        & summary["metric"].eq(metric)
    ]
    if len(selected) != 1:
        raise ValueError(
            "Expected one source row for "
            f"{architecture}/{protocol}/{organism}/{metric}; found {len(selected)}"
        )
    row = selected.iloc[0]
    if pd.isna(row["median"]) or pd.isna(row["ci_95_half_width"]):
        return "naentry", "", ""
    candidates = summary[
        summary["organism"].eq(organism) & summary["metric"].eq(metric)
    ].dropna(subset=["median", "ci_95_half_width"])
    best_index = (
        candidates["median"].idxmin()
        if metric == "rmse_normalized"
        else candidates["median"].idxmax()
    )
    best = candidates.loc[best_index]
    lower = best["median"] - best["ci_95_half_width"]
    upper = best["median"] + best["ci_95_half_width"]
    if row.name == best_index:
        macro = "bestmci"
    elif lower <= row["median"] <= upper:
        macro = "bmci"
    else:
        macro = "mci"
    return macro, f"{row['median']:.3f}", f"{row['ci_95_half_width']:.3f}"


def extract_cells(text: str) -> list[tuple[str, str, str]]:
    cells: list[tuple[str, str, str]] = []
    for macro, median, half_width, missing in CELL.findall(text):
        cells.append(("naentry", "", "") if missing else (macro, median, half_width))
    return cells


def validate_appendix(summary: pd.DataFrame, table_path: Path) -> int:
    text = table_path.read_text(encoding="utf-8")
    marker = "& \\multicolumn{3}{c}{Human (HEK293T)}"
    blocks = text.split(marker)
    if len(blocks) != 2:
        raise ValueError("Could not identify the two organism blocks in appendix table")

    checked = 0
    architectures = tuple(PROTOCOLS)
    for block, organisms in zip(blocks, ORGANISM_GROUPS, strict=True):
        for index, architecture in enumerate(architectures):
            start = block.index(f"\\textbf{{{architecture}}}")
            later = [
                block.find(f"\\textbf{{{candidate}}}", start + 1)
                for candidate in architectures[index + 1 :]
            ]
            later = [position for position in later if position >= 0]
            end = min(later) if later else len(block)
            observed = extract_cells(block[start:end])
            expected_count = len(PROTOCOLS[architecture]) * 6
            if len(observed) != expected_count:
                raise ValueError(
                    f"Expected {expected_count} cells for {architecture} in "
                    f"{organisms}; found {len(observed)}"
                )
            cell_index = 0
            for protocol in PROTOCOLS[architecture]:
                for organism in organisms:
                    for metric in METRICS:
                        expected = expected_cell(
                            summary, architecture, protocol, organism, metric
                        )
                        actual = observed[cell_index]
                        cell_index += 1
                        if actual != expected:
                            raise ValueError(
                                f"Appendix mismatch for {architecture}/{protocol}/"
                                f"{organism}/{metric}: table={actual}, source={expected}"
                            )
                        checked += 1
    return checked


def validate_main(summary: pd.DataFrame, table_path: Path) -> int:
    text = table_path.read_text(encoding="utf-8")
    checked = 0
    for architecture in PROTOCOLS:
        label = "\\textbf{RiboUnmix}" if architecture == "RiboUnmix" else architecture
        start = text.index(label + "\n")
        end = text.index("\\\\", start)
        observed = extract_cells(text[start:end])
        protocol = "Weighted" if architecture == "RiboUnmix" else "RiboUnmix weighted"
        expected = [
            expected_cell(summary, architecture, protocol, organism, "pcc")
            for organism in (*ORGANISM_GROUPS[0], *ORGANISM_GROUPS[1])
        ]
        if observed != expected:
            raise ValueError(
                f"Main-table mismatch for {architecture}: "
                f"table={observed}, source={expected}"
            )
        checked += len(expected)
    return checked


def validate_external_provenance(root: Path, table_path: Path) -> int:
    source = pd.read_csv(
        root / "results/four_model_canonical_training_checkpoint_provenance.tsv",
        sep="\t",
    )
    text = table_path.read_text(encoding="utf-8")
    architectures = tuple(PROTOCOLS)
    checked = 0
    for index, architecture in enumerate(EXTERNAL_TRAINING):
        start = text.index(f"\\textbf{{{architecture}}}")
        later = [
            text.find(f"\\textbf{{{candidate}}}", start + 1)
            for candidate in architectures[index + 1 :]
        ]
        later = [position for position in later if position >= 0]
        end = min(later) if later else len(text)
        observed = PROVENANCE_CELL.findall(text[start:end])
        expected: list[tuple[str, str]] = []
        for protocol in PROTOCOLS[architecture]:
            training = EXTERNAL_TRAINING[architecture][protocol]
            for organism in ORGANISMS:
                selected = source[
                    source["model"].eq(architecture)
                    & source["training"].eq(training)
                    & source["dataset"].eq(organism)
                ]
                if len(selected) != 1:
                    raise ValueError(
                        "Expected one provenance row for "
                        f"{architecture}/{training}/{organism}; found {len(selected)}"
                    )
                row = selected.iloc[0]
                if row["epoch_numbering"] != "1-based":
                    raise ValueError(
                        f"Unexpected epoch convention for {architecture}/{training}/"
                        f"{organism}: {row['epoch_numbering']}"
                    )
                expected.append(
                    (
                        f"{int(row['n_training_transcripts']):,}",
                        str(int(row["selected_epoch"])),
                    )
                )
        if observed != expected:
            raise ValueError(
                f"Training-provenance mismatch for {architecture}: "
                f"table={observed}, source={expected}"
            )
        checked += len(expected)
    return checked


def main() -> int:
    root = parse_args().project_root.resolve()
    summary = pd.read_csv(
        root
        / "results/cross_organism_benchmark_reanalysis/"
        "combined_summary_trim5_normalized_rmse.csv"
    )
    duplicate_keys = summary.duplicated(
        ["architecture", "protocol", "organism", "metric"]
    )
    if duplicate_keys.any():
        raise ValueError(f"Merged source has {int(duplicate_keys.sum())} duplicate keys")

    appendix_count = validate_appendix(
        summary, root / "ICLR_draft/tables/four_benchmark_gt_0.tex"
    )
    main_count = validate_main(
        summary, root / "ICLR_draft/tables/four_benchmark_gt_0_pearson_only.tex"
    )
    provenance_count = validate_external_provenance(
        root, root / "ICLR_draft/tables/four_model_training_provenance.tex"
    )
    print(
        f"PASS: {appendix_count} appendix cells and {main_count} main-table cells "
        "match the full-precision source and emphasis rule; "
        f"{provenance_count} external training/epoch cells match the consolidated "
        "checkpoint provenance."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
