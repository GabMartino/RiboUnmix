#!/usr/bin/env python3
"""Build the project's sequence-table schema from MANE codon sequences.

The nucleotide representation (``ref``), amino-acid representation, tAI, and
CDS/frame markers are reproducible from codons.  Other optional annotations
(``dom``, ``exo``, ``gmp``, ``tmp``, and ``openen``) are not present in the
MANE input.  For transcripts already present in the existing sequence table,
this script preserves those annotations after verifying identical codons.  For
new transcripts it writes NaN placeholders rather than inventing measurements.
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANE = ROOT / "Datasets/data/sequence/MANE.selection.cds_codons.parquet"
DEFAULT_TEMPLATE = ROOT / "Datasets/data/sequence/sequence_embeddings_with_css.parquet"
DEFAULT_CSS = ROOT / "Datasets/conserved_stalling_sites/stalling_sites.parquet"
DEFAULT_OUTPUT = ROOT / "Datasets/data/sequence/MANE.selection.sequence_embeddings_with_css.parquet"
DEFAULT_NT_ENCODING = ROOT / "Datasets/encodings/nt_encoding.yaml"
DEFAULT_CODON_TO_AA = ROOT / "Datasets/encodings/codon2aa.yaml"
DEFAULT_AA_ENCODING = ROOT / "Datasets/encodings/aa_encoding.yaml"

CORE_COLUMNS = {"transcript_id", "codons", "ref", "aas", "conserved_stalling_sites"}
STOP_CODONS = {"TAA", "TAG", "TGA"}
# These unusually long transcripts create extreme padded batches in the current
# pair-row-based sampler and exhaust memory in the two-GRU model.  Keep every
# exclusion version-specific so no other transcript or future isoform is
# silently removed.
EXCLUDED_TRANSCRIPT_IDS = frozenset(
    {
        "ENST00000589042.5",  # 35,992 codons
        "ENST00000680850.1",  # 8,926 codons
        "ENST00000367255.10", # 8,798
        "ENST00000397345.8",  # 8,526
        "ENST00000680361.1", #7,819
        "ENST00000564288.6", #7,556
    }
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _base_id(value: Any) -> str:
    return str(value).split(".", maxsplit=1)[0]


def _plain(value: Any) -> Any:
    """Recursively turn NumPy/Pandas objects into Arrow-friendly Python data."""
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value]
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _infer_tai_lookup(template: pd.DataFrame) -> dict[str, float]:
    """Infer the fixed sense-codon tAI table already used by this project."""
    lookup: dict[str, float] = {}
    for codons, values in zip(template["codons"], template["tAI_profile_codon"]):
        for codon, raw_value in zip(codons, values):
            codon = str(codon).upper()
            value = float(raw_value)
            if codon in STOP_CODONS or not math.isfinite(value):
                continue
            previous = lookup.setdefault(codon, value)
            if not math.isclose(previous, value, rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError(
                    f"tAI is not codon-constant for {codon}: {previous} versus {value}."
                )
        if len(lookup) == 61:
            break
    if len(lookup) != 61:
        missing = sorted(set(_all_sense_codons()) - set(lookup))
        raise ValueError(f"Could not infer tAI for all 61 sense codons; missing {missing}.")
    return lookup


def _all_sense_codons() -> set[str]:
    bases = "ACTG"
    return {
        a + b + c
        for a in bases
        for b in bases
        for c in bases
        if a + b + c not in STOP_CODONS
    }


def _validate_schema(schema: pa.Schema) -> None:
    expected = [
        "aas",
        "transcript_id",
        "ref",
        "dom",
        "exo",
        "fra",
        "gmp",
        "mod",
        "tmp",
        "openen",
        "tAI_profile_codon",
        "codons",
        "conserved_stalling_sites",
    ]
    if schema.names != expected:
        raise ValueError(
            "The template does not have the expected project sequence schema.\n"
            f"Expected: {expected}\nFound:    {schema.names}"
        )


def _validate_codons(transcript_id: str, raw_codons: Any) -> tuple[list[str], list[str]]:
    codons = [str(codon).upper() for codon in raw_codons]
    invalid = [codon for codon in codons if len(codon) != 3 or set(codon) - set("ACTG")]
    if not codons:
        invalid.append("<empty sequence>")
    return codons, invalid


def _generated_features(
    codons: list[str],
    *,
    nt_encoding: dict[str, list[float]],
    codon_to_aa: dict[str, str],
    aa_encoding: dict[str, int],
    tai_lookup: dict[str, float],
) -> dict[str, Any]:
    ref = [[nt_encoding[base] for base in codon] for codon in codons]

    # This reproduces the legacy `aas` representation in the template.  The
    # active dataloader does not consume it: it reconstructs AA one-hots from ref.
    aas = []
    for codon in codons:
        aa_name = codon_to_aa[codon]
        aa_value = 1.0 if aa_name == "*" else float(aa_encoding[aa_name]) / 24.0
        aas.append([aa_value, aa_value, aa_value])

    length = len(codons)
    nan_triplets = [[float("nan"), float("nan"), float("nan")] for _ in codons]
    tai = [
        (-1000.0 if index == length - 1 else float("nan"))
        if codon in STOP_CODONS
        else tai_lookup[codon]
        for index, codon in enumerate(codons)
    ]
    return {
        "aas": aas,
        "ref": ref,
        "dom": [row.copy() for row in nan_triplets],
        "exo": [row.copy() for row in nan_triplets],
        "fra": [[1.0, 0.0, 0.0] for _ in codons],
        "gmp": [row.copy() for row in nan_triplets],
        "mod": [[[0.0, 1.0, 0.0] for _ in range(3)] for _ in codons],
        "tmp": [row.copy() for row in nan_triplets],
        "openen": [row.copy() for row in nan_triplets],
        "tAI_profile_codon": tai,
    }


def build(args: argparse.Namespace) -> None:
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}. Pass --overwrite to replace it.")

    template_schema = pq.read_schema(args.template)
    _validate_schema(template_schema)

    mane = pd.read_parquet(args.input, columns=["transcript_id", "gene_id", "codons"])
    input_mane_rows = len(mane)
    mane_transcript_ids = mane["transcript_id"].astype(str)
    excluded_mask = mane_transcript_ids.isin(EXCLUDED_TRANSCRIPT_IDS)
    excluded_count = int(excluded_mask.sum())
    mane = mane.loc[~excluded_mask].copy()
    template = pd.read_parquet(args.template)
    css = pd.read_parquet(args.css)
    if args.limit is not None:
        mane = mane.iloc[: args.limit].copy()

    if mane["transcript_id"].duplicated().any():
        duplicate = mane.loc[mane["transcript_id"].duplicated(), "transcript_id"].iloc[0]
        raise ValueError(f"Duplicate MANE transcript_id: {duplicate}")

    template_by_base = {_base_id(value): index for index, value in template["transcript_id"].items()}
    if len(template_by_base) != len(template):
        raise ValueError("The template has duplicate versionless transcript IDs.")
    css_by_base = {
        _base_id(row.transcript_id): _plain(row.conserved_stalling_sites)
        for row in css.itertuples(index=False)
    }

    nt_encoding = _read_yaml(args.nt_encoding)
    codon_to_aa = _read_yaml(args.codon_to_aa)
    aa_encoding = _read_yaml(args.aa_encoding)
    tai_lookup = _infer_tai_lookup(template)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    records: list[dict[str, Any]] = []
    written = copied = generated = invalid_count = feature_mismatch = css_out_of_range = 0

    def flush() -> None:
        nonlocal records, writer
        if not records:
            return
        table = pa.Table.from_pylist(records, schema=template_schema)
        if writer is None:
            writer = pq.ParquetWriter(args.output, template_schema, compression="zstd")
        writer.write_table(table)
        records = []

    try:
        for row in mane.itertuples(index=False):
            transcript_id = str(row.transcript_id)
            codons, invalid = _validate_codons(transcript_id, row.codons)
            if invalid:
                invalid_count += 1
                message = f"{transcript_id} contains invalid codon token(s): {invalid[:5]}"
                if args.invalid_codon_policy == "error":
                    raise ValueError(message)
                warnings.warn(message + "; dropping this transcript.", stacklevel=2)
                continue

            base_id = _base_id(transcript_id)
            features = _generated_features(
                codons,
                nt_encoding=nt_encoding,
                codon_to_aa=codon_to_aa,
                aa_encoding=aa_encoding,
                tai_lookup=tai_lookup,
            )

            template_index = template_by_base.get(base_id)
            if args.preserve_existing_features and template_index is not None:
                old = template.loc[template_index]
                if list(old["codons"]) == codons:
                    for column in template_schema.names:
                        if column not in CORE_COLUMNS:
                            features[column] = _plain(old[column])
                    copied += 1
                else:
                    feature_mismatch += 1
                    generated += 1
            else:
                generated += 1

            sites = css_by_base.get(base_id)
            if sites is not None:
                invalid_sites = [int(site) for site in sites if int(site) < 0 or int(site) >= len(codons)]
                if invalid_sites:
                    css_out_of_range += 1

            record = {
                **features,
                "transcript_id": transcript_id,
                "codons": codons,
                "conserved_stalling_sites": sites,
            }
            records.append({name: record[name] for name in template_schema.names})
            written += 1
            if len(records) >= args.chunk_rows:
                flush()
        flush()
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise RuntimeError("No valid rows were produced.")
    output_file = pq.ParquetFile(args.output)
    if output_file.schema_arrow != template_schema:
        raise AssertionError("Written schema differs from the template schema.")

    print(f"Input MANE rows:                       {input_mane_rows:,}")
    print(f"Configured transcripts excluded:       {excluded_count:,}")
    print(f"MANE rows after exclusion/limit:        {len(mane):,}")
    print(f"Output rows:                           {written:,}")
    print(f"Malformed transcripts dropped:         {invalid_count:,}")
    print(f"Existing optional-feature rows copied: {copied:,}")
    print(f"Rows with generated/missing features:  {generated:,}")
    print(f"Template codon mismatches:              {feature_mismatch:,}")
    print(f"CSS rows with out-of-range sites:       {css_out_of_range:,}")
    print(f"Output schema matches template:         yes")
    print(f"Wrote: {args.output}")
    if generated:
        print(
            "WARNING: generated rows contain NaN for dom/exo/gmp/tmp/openen. "
            "The current baseline is valid because these optional features are routed "
            "to 'none'. Recompute them before enabling their feature routes."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_MANE)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--css", type=Path, default=DEFAULT_CSS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nt-encoding", type=Path, default=DEFAULT_NT_ENCODING)
    parser.add_argument("--codon-to-aa", type=Path, default=DEFAULT_CODON_TO_AA)
    parser.add_argument("--aa-encoding", type=Path, default=DEFAULT_AA_ENCODING)
    parser.add_argument("--chunk-rows", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="Convert only the first N rows (smoke tests).")
    parser.add_argument("--invalid-codon-policy", choices=("drop", "error"), default="drop")
    parser.add_argument(
        "--no-preserve-existing-features",
        action="store_false",
        dest="preserve_existing_features",
        help="Do not copy optional annotations for matching rows from the template.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
