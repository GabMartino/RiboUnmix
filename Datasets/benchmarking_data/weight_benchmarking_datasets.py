"""Validate, adapt, filter, and reliability-weight the benchmark datasets.

The four benchmark sources contain one codon-resolution profile rather than
the replica-aware schema used by the training pipeline.  This script performs
only the format adaptation and CDS validation here, then delegates all profile
eligibility filtering and reliability calculations to
``Datasets.data.weight_hek_riboseq_codon_replicas``.

The source profile is represented explicitly as one replica.  This is a schema
adapter, not a claim that an unobserved biological replicate exists.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Datasets.data.weight_hek_riboseq_codon_replicas import (  # noqa: E402
    COVERAGE_WEIGHT,
    DEFAULT_WEIGHTING_MODE,
    DEPTH_WEIGHT,
    WEIGHTING_MODES,
    add_weights,
)


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = DATA_DIR / "weighted"


@dataclass(frozen=True)
class BenchmarkDatasetSpec:
    name: str
    ribo_filename: str
    cds_filename: str


DATASET_SPECS = (
    BenchmarkDatasetSpec(
        "human_iwasaki_2014",
        "human_iwasaki_2014.parquet",
        "human_iwasaki_2014_cds.parquet",
    ),
    BenchmarkDatasetSpec(
        "yeast_stein_2021",
        "yeast_stein_2021_yeast.parquet",
        "yeast_stein_2021_yeast_cds.parquet",
    ),
    BenchmarkDatasetSpec(
        "celegans_stein_2021",
        "celegans_stein_2021.parquet",
        "celegans_stein_2021_cds.parquet",
    ),
    BenchmarkDatasetSpec(
        "ecoli_zhang_2016",
        "ecoli_zhang_2016.parquet",
        "ecoli_zhang_2016_cds.parquet",
    ),
)


def _load_valid_codons(codon_encoding_path: Path) -> set[str]:
    with codon_encoding_path.open("r", encoding="utf-8") as handle:
        encoding = yaml.safe_load(handle)
    if not isinstance(encoding, dict) or not encoding:
        raise ValueError(f"Invalid codon encoding: {codon_encoding_path}")
    return {str(codon).upper().replace("U", "T") for codon in encoding}


def _validate_unique_ids(data: pd.DataFrame, *, path: Path) -> None:
    if "id" not in data.columns:
        raise KeyError(f"{path}: missing required column 'id'.")
    duplicate = data["id"].astype(str).duplicated(keep=False)
    if bool(duplicate.any()):
        first = str(data.loc[duplicate, "id"].iloc[0])
        raise ValueError(f"{path}: duplicate transcript ID {first!r}.")


def prepare_benchmark_dataset(
    *,
    spec: BenchmarkDatasetSpec,
    input_dir: Path,
    valid_codons: set[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return one validated replica-aware dataframe and audit metadata."""
    ribo_path = input_dir / spec.ribo_filename
    cds_path = input_dir / spec.cds_filename
    if not ribo_path.is_file():
        raise FileNotFoundError(f"Missing benchmark profile parquet: {ribo_path}")
    if not cds_path.is_file():
        raise FileNotFoundError(f"Missing benchmark CDS parquet: {cds_path}")

    ribo_data = pd.read_parquet(ribo_path)
    cds_data = pd.read_parquet(cds_path)
    required_ribo = {"id", "ribo"}
    required_cds = {"id", "cds_seq"}
    if missing := required_ribo.difference(ribo_data.columns):
        raise KeyError(f"{ribo_path}: missing columns {sorted(missing)}.")
    if missing := required_cds.difference(cds_data.columns):
        raise KeyError(f"{cds_path}: missing columns {sorted(missing)}.")
    _validate_unique_ids(ribo_data, path=ribo_path)
    _validate_unique_ids(cds_data, path=cds_path)

    ribo_data = ribo_data.copy()
    cds_data = cds_data.copy()
    ribo_data["id"] = ribo_data["id"].astype(str)
    cds_data["id"] = cds_data["id"].astype(str)
    ribo_ids = set(ribo_data["id"])
    cds_ids = set(cds_data["id"])
    if ribo_ids != cds_ids:
        missing_cds = sorted(ribo_ids - cds_ids)
        missing_ribo = sorted(cds_ids - ribo_ids)
        raise ValueError(
            f"Dataset {spec.name!r}: profile/CDS ID sets differ; "
            f"profiles_without_cds={missing_cds[:5]}, "
            f"cds_without_profiles={missing_ribo[:5]}."
        )

    cds_by_id = cds_data.set_index("id", drop=False)
    profile_lengths: list[int] = []
    terminal_stop_count = 0
    stop_codons = {"TAA", "TAG", "TGA"}
    for row in ribo_data.itertuples(index=False):
        transcript_id = str(row.id)
        profile = np.asarray(row.ribo, dtype=np.float32)
        if profile.ndim != 1:
            raise ValueError(
                f"Dataset {spec.name!r}, transcript {transcript_id!r}: "
                f"ribo must be one-dimensional, got {profile.shape}."
            )
        cds = np.asarray(cds_by_id.at[transcript_id, "cds_seq"])
        if cds.ndim != 1:
            raise ValueError(
                f"Dataset {spec.name!r}, transcript {transcript_id!r}: "
                f"cds_seq must be one-dimensional, got {cds.shape}."
            )
        codons = np.char.replace(np.char.upper(cds.astype(str)), "U", "T")
        if len(profile) != len(codons):
            raise ValueError(
                f"Dataset {spec.name!r}, transcript {transcript_id!r}: "
                f"profile/CDS length mismatch ({len(profile)} != {len(codons)})."
            )
        if len(codons) == 0:
            raise ValueError(
                f"Dataset {spec.name!r}, transcript {transcript_id!r}: empty CDS."
            )
        invalid = sorted({str(codon) for codon in codons if codon not in valid_codons})
        if invalid:
            raise ValueError(
                f"Dataset {spec.name!r}, transcript {transcript_id!r}: "
                f"codons absent from codon_encoding: {invalid[:10]}."
            )
        profile_lengths.append(len(profile))
        terminal_stop_count += int(str(codons[-1]) in stop_codons)

    # Preserve the original consensus profile. The new nested column explicitly
    # records the only observed source profile as the one available replica.
    ribo_data["ribo_cds_replicas"] = [
        [np.asarray(profile, dtype=np.float32)] for profile in ribo_data["ribo"]
    ]
    ribo_data["replica_ids"] = [["source_profile"] for _ in range(len(ribo_data))]

    audit = {
        "dataset": spec.name,
        "source_ribo_path": str(ribo_path),
        "source_cds_path": str(cds_path),
        "aligned_input_rows": int(len(ribo_data)),
        "profile_length_min": int(min(profile_lengths)),
        "profile_length_median": float(np.median(profile_lengths)),
        "profile_length_mean": float(np.mean(profile_lengths)),
        "profile_length_max": int(max(profile_lengths)),
        "terminal_stop_rows": int(terminal_stop_count),
        "replica_semantics": "single_observed_source_profile",
    }
    return ribo_data, audit


def preprocess_benchmark_datasets(
    *,
    input_dir: Path = DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    dataset_names: set[str] | None = None,
    codon_encoding_path: Path = REPOSITORY_ROOT
    / "Datasets/encodings/codon_encoding.yaml",
    overwrite: bool = False,
    dry_run: bool = False,
    weighting_mode: str = DEFAULT_WEIGHTING_MODE,
    depth_weight: float = DEPTH_WEIGHT,
    coverage_weight: float = COVERAGE_WEIGHT,
) -> list[dict[str, object]]:
    """Preprocess selected benchmark sources with the shared weight pipeline."""
    selected = [
        spec
        for spec in DATASET_SPECS
        if dataset_names is None or spec.name in dataset_names
    ]
    if dataset_names is not None:
        missing = dataset_names.difference(spec.name for spec in selected)
        if missing:
            raise KeyError(
                f"Unknown benchmark dataset(s) {sorted(missing)}; available: "
                f"{[spec.name for spec in DATASET_SPECS]}."
            )
    if not selected:
        raise ValueError("No benchmark datasets selected.")

    valid_codons = _load_valid_codons(codon_encoding_path)
    summaries: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="riboai_benchmark_weight_") as temporary:
        temporary_dir = Path(temporary)
        for index, spec in enumerate(selected, start=1):
            adapted, audit = prepare_benchmark_dataset(
                spec=spec,
                input_dir=input_dir,
                valid_codons=valid_codons,
            )
            adapted_path = temporary_dir / f"{spec.name}.parquet"
            adapted.to_parquet(adapted_path, engine="pyarrow", index=False)
            output_path = output_dir / f"{spec.name}.parquet"
            summary = add_weights(
                adapted_path,
                output_path,
                overwrite=overwrite,
                dry_run=dry_run,
                error_on_empty_dataset=True,
                weighting_mode=weighting_mode,
                depth_weight=depth_weight,
                coverage_weight=coverage_weight,
            )
            summary.update(audit)
            summary["output_path"] = str(output_path)
            summaries.append(summary)
            print(
                f"[{index}/{len(selected)}] {spec.name}: "
                f"input={summary['input_rows']}, eligible={summary['eligible_rows']}, "
                f"removed_zero={summary['removed_zero_rows']}, "
                f"tau={float(summary['depth_reference_tau']):.6g}, "
                f"median_weight={float(summary['final_weight_median']):.6g}"
            )

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = pd.DataFrame(summaries)
        manifest.to_csv(output_dir / "weight_manifest.tsv", sep="\t", index=False)
        metadata = {
            "weighting_implementation": (
                "Datasets/data/weight_hek_riboseq_codon_replicas.py"
            ),
            "weighting_mode": weighting_mode,
            "source_profile_replica_semantics": "single_observed_source_profile",
            "terminal_stop_handling": (
                "preserved because each source profile is already aligned one-to-one "
                "with its provided CDS codon array"
            ),
            "datasets": [asdict(spec) for spec in selected],
        }
        (output_dir / "preprocessing_manifest.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return summaries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--codon-encoding", type=Path, default=REPOSITORY_ROOT / "Datasets/encodings/codon_encoding.yaml")
    parser.add_argument("--datasets", nargs="*", choices=[spec.name for spec in DATASET_SPECS])
    parser.add_argument("--weighting-mode", choices=WEIGHTING_MODES, default=DEFAULT_WEIGHTING_MODE)
    parser.add_argument("--depth-weight", type=float, default=DEPTH_WEIGHT)
    parser.add_argument("--coverage-weight", type=float, default=COVERAGE_WEIGHT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    preprocess_benchmark_datasets(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset_names=set(args.datasets) if args.datasets else None,
        codon_encoding_path=args.codon_encoding,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        weighting_mode=args.weighting_mode,
        depth_weight=args.depth_weight,
        coverage_weight=args.coverage_weight,
    )


if __name__ == "__main__":
    main()
