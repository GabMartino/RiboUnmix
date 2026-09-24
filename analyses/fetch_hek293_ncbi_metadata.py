#!/usr/bin/env python3
"""Retrieve GEO metadata for the exact GSE/BioSample pairs in the supplied table.

Only metadata are downloaded, never sequencing reads or expression matrices.
GEO's BioSample relations establish the match; dataset aliases do not.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "analyses/artifacts/real_data/hek293_metadata"
GEO = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc="
REQUEST_LOCK = threading.Lock()
LAST_REQUEST = 0.0


def parse_soft(text: str) -> dict[str, dict[str, list[str]]]:
    """Preserve repeated SOFT fields, restricting parsing to metadata lines."""
    records = {}
    current = None
    for line in text.splitlines():
        if line.startswith("^") and " = " in line:
            _, accession = line.split(" = ", 1)
            current = records.setdefault(accession.strip(), {})
        elif line.startswith("!") and " = " in line and current is not None:
            key, value = line[1:].split(" = ", 1)
            current.setdefault(key, []).append(value.strip())
    return records


def joined(record, key):
    return " | ".join(dict.fromkeys(record.get(key, [])))


def characteristics(record):
    values = defaultdict(list)
    for item in record.get("Sample_characteristics_ch1", []):
        key, separator, value = item.partition(":")
        values[key.strip().lower() if separator else "unspecified"].append(
            value.strip() if separator else item
        )
    return dict(values)


def relation_ids(record, prefix):
    return sorted(set(re.findall(rf"\b{prefix}\d+\b", joined(record, "Sample_relation"))))


def download(accession: str, target: str, cache: Path, offline: bool):
    """Cache one metadata response, with a global rate below three requests/s."""
    global LAST_REQUEST
    path = cache / f"{accession}_{target}.soft"
    url = f"{GEO}{accession}&targ={target}&form=text&view=quick"
    if path.exists():
        data = path.read_bytes()
    else:
        if offline:
            raise FileNotFoundError(f"Missing cached response: {path}")
        for attempt in range(3):
            try:
                with REQUEST_LOCK:
                    time.sleep(max(0, 0.4 - (time.monotonic() - LAST_REQUEST)))
                    LAST_REQUEST = time.monotonic()
                request = urllib.request.Request(url, headers={"User-Agent": "RiboUnmix-metadata-audit/1.0"})
                with urllib.request.urlopen(request, timeout=45) as response:
                    data = response.read(25_000_001)
                if len(data) > 25_000_000:
                    raise ValueError(f"Unexpectedly large metadata response: {url}")
                records = parse_soft(data.decode("utf-8-sig"))
                required = accession in records if target == "self" else any(k.startswith("GSM") for k in records)
                if not required:
                    raise ValueError(f"No expected GEO records returned: {url}")
                path.write_bytes(data)
                break
            except (OSError, ValueError) as exc:
                if attempt == 2:
                    raise RuntimeError(f"{url}: {exc}") from exc
                time.sleep(1 + attempt)
    return parse_soft(data.decode("utf-8-sig")), {
        "url": url, "cache_file": str(path.relative_to(cache.parent)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "cached_at_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
    }


def fetch_study(gse, cache, offline):
    study, study_provenance = download(gse, "self", cache, offline)
    samples, sample_provenance = download(gse, "gsm", cache, offline)
    return study[gse], samples, [study_provenance, sample_provenance]


def write_tsv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def sample_row(dataset, gse, biosample, gsm, record):
    attrs = characteristics(record)
    line_keys = ("cell line", "cell_line", "cell-line", "cellline", "cell line/type", "cells")
    line = " | ".join(v for key in line_keys for v in attrs.get(key, []))
    line_source = "sample cell-line characteristic" if line else "not explicitly annotated"
    if not line:
        # Keep the literal source text (including engineered derivatives).
        # Never infer a line from a local alias or a generic kidney-cell label.
        for evidence, value in (
            ("cell-type characteristic", " | ".join(attrs.get("cell type", []))),
            ("source-name field", joined(record, "Sample_source_name_ch1")),
        ):
            if re.search(r"293", value):
                line, line_source = value, evidence
                break
        else:
            line = "not reported"
    replicate = " | ".join(f"{k}: {v}" for k, values in attrs.items() if "replic" in k for v in values)
    flags = []
    if gse not in record.get("Sample_series_id", []):
        flags.append("gse_not_in_sample_series_ids")
    if line == "not reported":
        flags.append("cell_line_not_explicitly_annotated")
    title = joined(record, "Sample_title")
    if re.search(r"rna[\s_-]?seq", title, re.I) and not re.search(r"ribo|rpf|footprint", title, re.I):
        flags.append("check_assay_title_mentions_RNAseq")
    return {
        "dataset": dataset, "gse": gse, "biosample": biosample, "gsm": gsm,
        "sample_title": title, "organism": joined(record, "Sample_organism_ch1"),
        "cell_line": line, "cell_line_evidence": line_source,
        "source_name": joined(record, "Sample_source_name_ch1"),
        "characteristics": " | ".join(record.get("Sample_characteristics_ch1", [])),
        "characteristics_json": json.dumps(attrs, ensure_ascii=False),
        "replicate_annotation": replicate or "not explicitly annotated",
        "biological_replication_explicit_in_sample_label": bool(re.search(
            r"biol(?:ogical)?[\s_-]*rep", title + " " + joined(record, "Sample_source_name_ch1"), re.I
        )),
        "instrument": joined(record, "Sample_instrument_model"),
        "library_strategy": joined(record, "Sample_library_strategy"),
        "library_source": joined(record, "Sample_library_source"),
        "library_selection": joined(record, "Sample_library_selection"),
        "extracted_molecule": joined(record, "Sample_molecule_ch1"),
        "platform": joined(record, "Sample_platform_id"),
        "sra_experiments": ";".join(relation_ids(record, "SRX")),
        "public_status": joined(record, "Sample_status"),
        "last_update": joined(record, "Sample_last_update_date"),
        "sample_url": GEO + gsm,
        "biosample_url": "https://www.ncbi.nlm.nih.gov/biosample/" + biosample,
        "validation_flags": ";".join(flags),
    }


def render_html(datasets, samples, audit, output):
    esc = lambda value: html.escape(str(value))
    by_dataset = defaultdict(list)
    for sample in samples:
        by_dataset[sample["dataset"]].append(sample)
    rows = []
    for row in datasets:
        selected = by_dataset[row["dataset"]]
        sample_details = []
        for sample in selected:
            sample_details.append(
                f'<p><a href="{esc(sample["sample_url"])}">{esc(sample["gsm"])}</a> '
                f'→ <a href="{esc(sample["biosample_url"])}">{esc(sample["biosample"])}</a>'
                f'<br><b>{esc(sample["sample_title"])}</b><br>{esc(sample["characteristics"])}'
                f'<br>Source: {esc(sample["source_name"])}; instrument: {esc(sample["instrument"])}'
                f'<br>Library strategy: {esc(sample["library_strategy"])}; '
                f'replicate annotation: {esc(sample["replicate_annotation"])}.</p>'
            )
        flags = row["review_note"] or row["validation_flags"] or "GSE–BioSample links verified"
        rows.append(
            f'<tr><td><b>{esc(row["dataset"])}</b><br>{row["listed_biosamples"]} BioSamples</td>'
            f'<td><a href="{GEO}{esc(row["gse"])}">{esc(row["gse"])}</a><br>'
            f'{esc(row["study_title"])}<br><small>{esc(row["study_public_status"])}</small></td>'
            f'<td>{esc(row["cell_lines"])}<br><small>{esc(row["source_names"])}</small></td>'
            f'<td>{esc(row["selected_condition"] or row["sample_titles"])}<details><summary>Sample-level annotations and links</summary>'
            + "".join(sample_details) + '</details></td>'
            f'<td>{esc(row["instruments"])}<br><small>{esc(row["library_strategies"])}</small></td>'
            f'<td>{esc(flags)}</td></tr>'
        )
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>HEK293-family datasets — NCBI metadata audit</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;margin:2em;color:#192c39}}h1{{font-size:27px}}
a{{color:#14618b}}table{{border-collapse:collapse;width:100%}}th,td{{padding:12px;border:1px solid #dbe3e8;text-align:left;vertical-align:top}}
th{{background:#edf3f6;position:sticky;top:0}}tr:nth-child(even){{background:#f8fafb}}small{{color:#52636d}}
details{{margin-top:8px}}summary{{color:#14618b;cursor:pointer}}input{{padding:10px;width:60%;font:inherit;margin:14px 0}}
.note{{background:#eef5fa;padding:16px;border-left:4px solid #277298}}td:first-child{{overflow-wrap:anywhere;min-width:140px}}</style>
<h1>HEK293-family Ribo-seq collection: exact-sample metadata</h1>
<p>{audit['dataset_count']} dataset aliases; {audit['unique_gse_count']} distinct GSE accessions;
{audit['unique_listed_biosamples']} distinct listed BioSamples. Retrieved {esc(audit['generated_utc'])}.</p>
<p><a href="dataset_metadata.tsv">Dataset table (TSV)</a> · <a href="sample_metadata.tsv">Exact samples (TSV)</a> ·
<a href="study_metadata.tsv">Study table (TSV)</a> · <a href="README.md">Findings and caveats</a> ·
<a href="audit.json">Provenance and validation</a></p>
<div class="note"><b>How to read this list.</b> Each alias is matched to the supplied BioSample IDs using GEO sample relations
within its specified GSE, not inferred from the alias or the study title. The titles and characteristics describe the selected
samples, not all conditions in the study. When the cell-line field is absent, explicit cell-type or source-name text is retained,
with its origin recorded in the sample TSV. “Not reported” means these fields did not identify the line.
A GEO library strategy of “RNA-Seq” alone does not distinguish RNA-seq from ribosome profiling.
<p><b>Counts are not proof of biological replication.</b> The original “Replicate Number” is checked against the number of distinct
BioSample accessions; independent biological replication requires protocol-level evidence. Study release dates are not publication
years. Shared GSE accessions are not independent studies. HEK293, HEK293T and derived engineered lines should not be treated as
identical cell lines. Shared extraction protocols may describe multiple conditions; we do not automatically assign every treatment
mentioned there to every sample.</p></div>
<p><b>Validation:</b> {audit['matched_biosamples']} distinct BioSamples matched; {len(audit['missing_inputs'])} missing mappings;
{len(audit['ambiguous_mappings'])} BioSamples linked to more than one GSM. See audit.json for exact exceptions.</p>
<input id="filter" placeholder="Filter by dataset, GSE, sample title, cell line, instrument…" aria-label="Filter datasets">
<table><thead><tr><th>Dataset alias</th><th>NCBI study</th><th>Cell line / source</th><th>Selected sample titles / conditions</th>
<th>Sequencing instrument</th><th>Checks / caveats</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<script>document.getElementById('filter').addEventListener('input',function(){{const q=this.value.toLowerCase();
document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));}});</script></html>"""
    (output / "metadata.html").write_text(body, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT / "input_accessions.tsv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offline", action="store_true", help="Rebuild tables from cached responses only.")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.output_dir / "cache"
    cache.mkdir(exist_ok=True)
    with args.input.open(encoding="utf-8") as handle:
        inputs = list(csv.DictReader(handle, delimiter="\t"))
    review_path = args.output_dir / "reviewed_annotations.tsv"
    reviews = {}
    if review_path.exists():
        with review_path.open(encoding="utf-8") as handle:
            reviews = {r["dataset"]: r for r in csv.DictReader(handle, delimiter="\t")}
    if len({r["dataset"] for r in inputs}) != len(inputs):
        raise ValueError("Duplicate dataset aliases in input table")
    gses = sorted({r["gse"] for r in inputs})
    fetched, failures, provenance = {}, {}, []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(fetch_study, gse, cache, args.offline): gse for gse in gses}
        for number, future in enumerate(as_completed(futures), 1):
            gse = futures[future]
            try:
                study, records, sources = future.result()
                fetched[gse] = (study, records)
                provenance.extend(sources)
                print(f"[{number}/{len(gses)}] {gse}: {len(records)} GEO sample records", flush=True)
            except Exception as exc:
                failures[gse] = str(exc)
                print(f"[{number}/{len(gses)}] {gse}: FAILED {exc}", flush=True)
    datasets, samples, study_rows, missing, ambiguous = [], [], [], [], []
    for gse in gses:
        study = fetched.get(gse, ({}, {}))[0]
        study_rows.append({
            "gse": gse, "title": joined(study, "Series_title"),
            "public_status": joined(study, "Series_status"),
            "submission_date": joined(study, "Series_submission_date"),
            "last_update": joined(study, "Series_last_update_date"),
            "organisms": joined(study, "Series_sample_organism"),
            "experiment_types": joined(study, "Series_type"),
            "pubmed_ids": joined(study, "Series_pubmed_id"),
            "all_series_gsm_count": len(study.get("Series_sample_id", [])),
            "study_url": GEO + gse, "retrieval_error": failures.get(gse, ""),
        })
    for entry in inputs:
        gse, name = entry["gse"], entry["dataset"]
        study, records = fetched.get(gse, ({}, {}))
        index = defaultdict(list)
        for gsm, record in records.items():
            for biosample in relation_ids(record, "SAMN"):
                index[biosample].append((gsm, record))
        biosamples = entry["biosamples"].split(";")
        flags, selected, matched = [], [], []
        if int(entry["reported_replicate_number"]) != len(set(biosamples)):
            flags.append("reported_count_differs_from_unique_biosamples")
        for biosample in biosamples:
            matches = index[biosample]
            if not matches:
                missing.append({"dataset": name, "gse": gse, "biosample": biosample})
                flags.append(f"missing:{biosample}")
            else:
                matched.append(biosample)
            if len(matches) > 1:
                ambiguous.append({"dataset": name, "gse": gse, "biosample": biosample, "gsms": [m[0] for m in matches]})
                flags.append(f"multiple_GSMs:{biosample}")
            for gsm, record in matches:
                row = sample_row(name, gse, biosample, gsm, record)
                selected.append(row)
                samples.append(row)
                flags.extend(f for f in row["validation_flags"].split(";") if f)
        aggregate = lambda key: " | ".join(dict.fromkeys(r[key] for r in selected if r[key]))
        datasets.append({
            "dataset": name, "gse": gse, "listed_biosamples": len(set(biosamples)),
            "reported_replicate_number": entry["reported_replicate_number"],
            "matched_biosamples": len(set(matched)), "matched_gsms": len(selected),
            "biosamples": entry["biosamples"], "gsms": ";".join(r["gsm"] for r in selected),
            "study_title": joined(study, "Series_title"),
            "study_public_status": joined(study, "Series_status"),
            "pubmed_ids": joined(study, "Series_pubmed_id"),
            "organisms": aggregate("organism"), "cell_lines": aggregate("cell_line"),
            "source_names": aggregate("source_name"), "sample_titles": aggregate("sample_title"),
            "sample_characteristics": aggregate("characteristics"),
            "instruments": aggregate("instrument"), "library_strategies": aggregate("library_strategy"),
            "replicate_annotations": aggregate("replicate_annotation"),
            "study_url": GEO + gse, "validation_flags": ";".join(dict.fromkeys(flags)),
            "selected_condition": reviews.get(name, {}).get("selected_condition", ""),
            "review_note": reviews.get(name, {}).get("review_note", ""),
        })
    listed = [s for entry in inputs for s in entry["biosamples"].split(";")]
    audit = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_count": len(inputs), "unique_gse_count": len(gses),
        "listed_biosample_occurrences": len(listed), "unique_listed_biosamples": len(set(listed)),
        "matched_biosamples": len({r["biosample"] for r in samples}),
        "unique_matched_gsms": len({r["gsm"] for r in samples}),
        "duplicate_biosamples_in_input": {s: n for s, n in Counter(listed).items() if n > 1},
        "missing_inputs": missing, "ambiguous_mappings": ambiguous, "retrieval_failures": failures,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reviewed_annotations_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest() if review_path.exists() else None,
        "command": " ".join(sys.argv), "sources": sorted(provenance, key=lambda p: p["url"]),
        "method": "Exact supplied SAMN accession matched against Sample_relation in the specified GSE; no alias-based condition inference.",
    }
    write_tsv(args.output_dir / "dataset_metadata.tsv", datasets)
    write_tsv(args.output_dir / "sample_metadata.tsv", samples)
    write_tsv(args.output_dir / "study_metadata.tsv", study_rows)
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    render_html(datasets, samples, audit, args.output_dir)
    print(json.dumps({k: v for k, v in audit.items() if k not in {"sources", "script_sha256"}}, indent=2))


if __name__ == "__main__":
    main()
