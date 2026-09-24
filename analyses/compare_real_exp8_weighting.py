#!/usr/bin/env python3
"""Compare saved Exp8 designs without treating them as a weighting-only ablation.

Default: PCC between every available same-seed model pair at successive planned
N values, separately within each experiment. No missing N is bridged. The
original equal-weight same-N disjoint statistic is also regenerated separately:
the cumulative ranked experiment has no corresponding disjoint replication.
Run from any directory; no training, checkpoint loading, or profile rescaling.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Utils.publication_plot_style import LATEX_PAPER_RC
from Utils.reliability_references import transcript_id_hash
from analyses.analyze_real_exp8_stability import _load_profiles, _task_directory
from analyses.analyze_real_exp8_quality_rank_partial import load_export, relocated

DEFAULT_EQUAL = ROOT / "results/my_exp8_a100_b32_20260906_114340"
DEFAULT_RANKED = ROOT / "results/real_exp8_L_stability_quality_rank_10components/cumulative_qrank10components_p1.0_seed42"


def read_json(path):
    return json.loads(path.read_text())


def load_equal(root, task, ids):
    directory = _task_directory(root, task)
    paths = list(directory.glob("predictions/**/prediction_checkpoint_manifest.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one runtime manifest; found {len(paths)}")
    runtime = read_json(paths[0])["best_val_loss"]
    if (runtime.get("sequence_only_shared_profile_prediction") is not True
            or runtime.get("split_name") != "test"
            or runtime["transcript_id_hash"] != transcript_id_hash(ids)
            or runtime["transcript_count"] != len(ids)):
        raise ValueError("Sequence-only test identity/count mismatch")
    path = relocated(root, directory, runtime["shared_profile_output_path"])
    if path.parent != paths[0].parent:
        raise ValueError("Prediction and runtime manifest are not co-located")
    identity = pd.read_parquet(path, columns=["run_id", "N"])
    if set(identity.run_id) != {task["run_id"]} or set(identity.N) != {task["N"]}:
        raise ValueError("Prediction run ID/N mismatch")
    subset = read_json(directory / "subset_manifest.json")
    gamma = read_json(path.parent / "gamma_reference_manifest.json")
    if (subset["datasets"] != task["datasets"]
            or set(gamma["reference_dataset_names"]) != set(task["datasets"])
            or gamma["centering_mode"] != "fixed_reference"
            or gamma["weighting"] != "equal"):
        raise ValueError("Dataset membership/gamma reference mismatch")
    np.testing.assert_allclose(gamma["reference_pi"], 1 / task["N"], rtol=1e-6)
    np.testing.assert_allclose(gamma["reference_pi"],
        [subset["fixed_gamma_reference"]["pi"][d] for d in gamma["reference_dataset_names"]],
        rtol=1e-6)
    before = path.stat()
    profiles, checks = _load_profiles(path=path, expected_ids=ids, mean_one_tolerance=1e-4)
    for row in profiles.values():
        if not row["mask"][:row["length"]].all() or (row["values"][row["mask"]] < 0).any():
            raise ValueError("Incomplete CDS or negative shared profile")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Prediction changed while reading")
    return profiles, checks, dict(profile_path=str(path), profile_sha256=digest,
        checkpoint_path=runtime["checkpoint_path"], checkpoint_variant="best_val_loss")


def collect(root, manifest, ids, experiment):
    profiles, audit = {}, []
    for task in manifest["tasks"]:
        directory = root / task["directory"] if experiment == "ranked" else _task_directory(root, task)
        row = dict(experiment=experiment, run_id=task["run_id"], N=task["N"],
                   training_seed=task["training_seed"], available=False, status="missing")
        if list(directory.glob("predictions/**/prediction_checkpoint_manifest.json")):
            try:
                p, checks, source = (load_export(root, task, ids, 1e-4) if experiment == "ranked"
                                     else load_equal(root, task, ids))
            except Exception as exc:
                row.update(status="invalid", reason=f"{type(exc).__name__}: {exc}")
            else:
                profiles[task["run_id"]] = p
                row.update(source, available=True, status="verified", transcript_count=len(ids),
                    max_mean_one_deviation=float(checks.absolute_mean_one_deviation.max()))
        audit.append(row)
    return profiles, audit


def planned_pairs(tasks, available, disjoint=False):
    """Use the original size schedule, never the compressed available schedule."""
    sizes = sorted({t["N"] for t in tasks})
    transitions = set(zip(sizes, sizes[1:]))
    for a, b in itertools.combinations(sorted(tasks, key=lambda t: (t["N"], t["run_id"])), 2):
        if a["run_id"] not in available or b["run_id"] not in available:
            continue
        if a["training_seed"] != b["training_seed"]:
            continue
        if disjoint:
            if not (a["kind"] == b["kind"] == "designated_disjoint_pair"
                    and a["N"] == b["N"] and a["pair_id"] == b["pair_id"]
                    and a["side"] != b["side"]):
                continue
            if set(a["datasets"]) & set(b["datasets"]):
                raise ValueError("Designated disjoint pair shares datasets")
            if set(a["source_families"]) & set(b["source_families"]):
                raise ValueError("Designated disjoint pair shares source families")
        elif (a["N"], b["N"]) not in transitions:
            continue
        yield a, b


def profile_pcc(a, b):
    if (a["values"].shape != b["values"].shape
            or not np.array_equal(a["mask"], b["mask"])):
        raise ValueError("Transcript position alignment differs; refusing truncation")
    x, y = a["values"][a["mask"]], b["values"][b["mask"]]
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.clip(np.dot(x, y) / denominator, -1, 1)) if denominator > 0 else np.nan


def compare(tasks, profiles, ids, experiment, disjoint=False):
    rows = []
    for a, b in planned_pairs(tasks, profiles, disjoint):
        sa, sb = set(a["datasets"]), set(b["datasets"])
        fa, fb = set(a["source_families"]), set(b["source_families"])
        meta = dict(experiment=experiment, N=a["N"], N_next=b["N"],
            run_a=a["run_id"], run_b=b["run_id"], training_seed=a["training_seed"],
            dataset_jaccard=len(sa & sb) / len(sa | sb),
            source_jaccard=len(fa & fb) / len(fa | fb))
        for tid in ids:
            rows.append(dict(meta, transcript_id=tid,
                PCC=profile_pcc(profiles[a["run_id"]][tid], profiles[b["run_id"]][tid])))
    return pd.DataFrame(rows)


def summarize(values):
    rows = []
    for (experiment, n, nxt), g in values.groupby(["experiment", "N", "N_next"]):
        v = g.PCC.dropna()
        rows.append(dict(experiment=experiment, N=n, N_next=nxt, mean_PCC=v.mean(),
            std_PCC=v.std(), median_PCC=v.median(), n_usable=len(v), n_undefined=g.PCC.isna().sum(),
            n_transcripts=g.transcript_id.nunique(), n_pairs=len(g[["run_a", "run_b"]].drop_duplicates()),
            mean_dataset_jaccard=g.dataset_jaccard.mean(), mean_source_jaccard=g.source_jaccard.mean()))
    return pd.DataFrame(rows)


def plot(summary, out, use_tex, disjoint=False):
    style = dict(LATEX_PAPER_RC)
    if not use_tex:
        style.update({"text.usetex": False, "font.serif": ["DejaVu Serif"], "mathtext.fontset": "cm"})
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        labels = {"equal": r"Equal $\pi$: quality-balanced panels",
                  "ranked": r"Ranked $\pi$: nested top-quality panels"}
        for experiment, color, marker in [("equal", "#0072B2", "o"), ("ranked", "#D55E00", "s")]:
            g = summary[summary.experiment == experiment].sort_values("N")
            if g.empty:
                continue
            ax.errorbar(g.N, g.mean_PCC, yerr=g.std_PCC.fillna(0), fmt=marker + "-",
                        color=color, capsize=3, lw=1.8, markersize=6, label=labels[experiment])
        ticks = summary[["N", "N_next"]].drop_duplicates().sort_values("N")
        ax.set_xticks(ticks.N, [str(int(r.N)) if disjoint else
            rf"${int(r.N)}\!\rightarrow\!{int(r.N_next)}$" for r in ticks.itertuples()])
        ax.set_xlabel("Number of datasets ($N$)" if disjoint else "Dataset counts compared ($N$ to next planned $N$)")
        ax.set_ylabel(r"Shared-profile $L_t$ PCC")
        ax.set_title("Independent shared-profile stability (available pairs)" if disjoint else
                     "Shared-profile agreement as dataset count increases")
        ax.grid(alpha=.2)
        ax.legend(fontsize=10, loc="lower right")
        note = ("Equal weights only: ranked runs have no same-N disjoint pairs." if disjoint else
                "Different panel selection and overlap; this does not isolate the effect of weights.")
        fig.text(.5, .055, note, ha="center", fontsize=9)
        fig.text(.5, .015, "Points: mean PCC; bars: descriptive SD across transcript/model-pair values (not CI).",
                 ha="center", fontsize=8)
        fig.subplots_adjust(left=.105, right=.98, top=.91, bottom=.22)
        for extension in ["png", "pdf"]:
            fig.savefig(out.with_suffix("." + extension), dpi=300, bbox_inches="tight")
        plt.close(fig)


def flatten(data, prefix=""):
    result = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def design_audit(roots, manifests, out):
    membership, configs = [], []
    baseline_task = manifests["equal"]["tasks"][0]
    baseline = flatten(yaml.safe_load((_task_directory(roots["equal"], baseline_task) / "resolved_config.yaml").read_text()))
    for experiment, manifest in manifests.items():
        for task in manifest["tasks"]:
            directory = roots[experiment] / task["directory"] if experiment == "ranked" else _task_directory(roots[experiment], task)
            subset = read_json(directory / "subset_manifest.json")
            for name, pi in subset["fixed_gamma_reference"]["pi"].items():
                membership.append(dict(experiment=experiment, run_id=task["run_id"], N=task["N"], dataset=name, pi=pi))
            config = flatten(yaml.safe_load((directory / "resolved_config.yaml").read_text()))
            for key in sorted(baseline.keys() | config.keys()):
                if baseline.get(key) != config.get(key):
                    configs.append(dict(experiment=experiment, run_id=task["run_id"], key=key,
                        baseline=json.dumps(baseline.get(key)), value=json.dumps(config.get(key))))
    pd.DataFrame(membership).to_csv(out / "dataset_membership_and_pi.csv", index=False)
    pd.DataFrame(configs).to_csv(out / "config_differences_from_equal_N002_pair01_A.csv", index=False)
    matched = []
    for a in manifests["equal"]["tasks"]:
        for b in manifests["ranked"]["tasks"]:
            if a["N"] == b["N"] and set(a["datasets"]) == set(b["datasets"]):
                matched.append(dict(N=a["N"], equal=a["run_id"], ranked=b["run_id"]))
    return matched


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equal-root", type=Path, default=DEFAULT_EQUAL)
    parser.add_argument("--ranked-root", type=Path, default=DEFAULT_RANKED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "analyses/artifacts/real_data/exp8_equal_vs_qrank10",
    )
    parser.add_argument("--no-tex", action="store_true", help="Use Matplotlib fonts instead of installed LaTeX")
    args = parser.parse_args(argv)
    roots = dict(equal=args.equal_root.resolve(), ranked=args.ranked_root.resolve())
    manifests = {k: read_json(r / "experiment_manifest.json") for k, r in roots.items()}
    tests = {k: read_json(r / "common_test_manifest.json") for k, r in roots.items()}
    ids = sorted(tests["equal"]["common_test_ids"])
    for test in tests.values():
        if (len(test["common_test_ids"]) != len(set(test["common_test_ids"]))
                or set(test["common_test_ids"]) != set(ids)
                or test["transcript_id_hash"] != transcript_id_hash(ids)):
            raise ValueError("Experiments must have identical, correctly hashed held-out IDs")
    if sorted({t["N"] for t in manifests["equal"]["tasks"]}) != sorted({t["N"] for t in manifests["ranked"]["tasks"]}):
        raise ValueError("Experiments must have the same planned N schedule")
    out = args.output_dir.resolve()
    if any(out == r or out.is_relative_to(r) for r in roots.values()):
        raise ValueError("Use a separate output directory to preserve source experiments")
    out.mkdir(parents=True, exist_ok=True)
    matched = design_audit(roots, manifests, out)
    values, audits, disjoint = [], [], pd.DataFrame()
    for experiment, root in roots.items():
        print(f"Verifying {experiment} exports...", flush=True)
        profiles, audit = collect(root, manifests[experiment], ids, experiment)
        audits.extend(audit)
        print(f"Comparing {len(profiles)} verified {experiment} models...", flush=True)
        values.append(compare(manifests[experiment]["tasks"], profiles, ids, experiment))
        if experiment == "equal":
            disjoint = compare(manifests[experiment]["tasks"], profiles, ids, experiment, disjoint=True)
        del profiles
    pd.DataFrame(audits).to_csv(out / "export_audit.csv", index=False)
    values = pd.concat(values, ignore_index=True)
    if values.empty:
        raise ValueError("No verified successive-size comparisons; see export_audit.csv")
    values.to_parquet(out / "successive_N_per_transcript.parquet", index=False)
    summary = summarize(values)
    summary.to_csv(out / "successive_N_summary.csv", index=False)
    # A joint plot uses only transitions available in both experiments. All
    # one-sided transitions remain in the CSV rather than silently disappearing.
    common = summary.groupby(["N", "N_next"]).experiment.nunique()
    keys = set(common[common == 2].index)
    joint = summary[[tuple(x) in keys for x in summary[["N", "N_next"]].to_numpy()]]
    if joint.empty:
        raise ValueError("No transition has verified results in both experiments")
    plot(joint, out / "stability_equal_vs_ranked", not args.no_tex)
    if not disjoint.empty:
        disjoint.to_parquet(out / "equal_disjoint_per_transcript.parquet", index=False)
        ds = summarize(disjoint)
        ds.to_csv(out / "equal_disjoint_summary.csv", index=False)
        plot(ds, out / "stability_equal_disjoint_updated", not args.no_tex, disjoint=True)
    caption = (f"Shared-profile agreement on the same {len(ids):,} held-out transcripts, using original "
        "mean-one full-CDS sequence-only best_val_loss predictions. At each transition, all available "
        "same-training-seed model pairs at N and its next planned size enter the arithmetic mean PCC "
        "and sample SD. Constant-profile PCC is undefined and excluded with counts reported. "
        "Equal-weight panels are quality-balanced; ranked panels are nested top-quality prefixes. "
        "Selection, dataset/source overlap, task-specific training/validation eligibility and gamma "
        "reference weights differ. Shared transcripts and reused models make these comparisons "
        "dependent; error bars are descriptive, not confidence intervals. Higher agreement does not "
        "establish biological accuracy or a causal benefit of pi weighting. The ranked design has "
        "no same-N disjoint replication, so the original disjoint metric is plotted separately. "
        "Only transitions available in both experiments enter the joint figure; all available "
        "transitions are retained in successive_N_summary.csv. No smaller model substitutes for N=114.\n")
    (out / "figure_caption.txt").write_text(caption)
    (out / "analysis_manifest.json").write_text(json.dumps(dict(
        source_roots={k: str(r) for k, r in roots.items()},
        source_manifest_sha256={k: hashlib.sha256((r / "experiment_manifest.json").read_bytes()).hexdigest() for k, r in roots.items()},
        common_test_hash=transcript_id_hash(ids), transcript_count=len(ids),
        identical_membership_task_pairs=matched, plotted_transitions=sorted(keys),
        comparison="all same-seed model pairs at consecutive planned dataset counts",
        weighting_only_ablation=False, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ), indent=2) + "\n")
    print(joint.to_string(index=False))
    print(f"Saved comparison figures, source tables and audits to {out}")


if __name__ == "__main__":
    main()
