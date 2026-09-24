#!/usr/bin/env python3
"""Partial-results analysis for Experiment 8 (L-profile stability).

Unlike the complete analysis, this script never requires every planned run to
exist.  It discovers only the exported ``common_test_L_profiles.parquet``
artifacts (which are produced from the best-validation-loss checkpoint),
records unavailable tasks, and computes every comparison possible from the
available artifacts.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Utils.publication_plot_style import latex_paper_style


def _discover_run_root() -> Path:
    """Return the newest Experiment-8 directory containing predictions."""
    results_root = PROJECT_ROOT / "results"
    candidates = []
    for manifest in results_root.rglob("experiment_manifest.json"):
        root = manifest.parent
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("experiment_name") != "real_exp8_L_stability":
            continue
        n_profiles = sum(1 for _ in root.rglob("common_test_L_profiles.parquet"))
        candidates.append((n_profiles > 0, manifest.stat().st_mtime, root))
    if not candidates:
        raise FileNotFoundError(
            f"No Experiment-8 experiment_manifest.json found under {results_root}. "
            "Pass --run-root explicitly."
        )
    # Prefer a run with usable predictions; among ties use the newest design.
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _planned_runs(root: Path) -> pd.DataFrame:
    p = root / "subset_assignments.csv"
    if not p.exists():
        return pd.DataFrame(columns=["run_id", "N", "kind", "pair_id", "side"])
    x = pd.read_csv(p)
    cols = ["run_id", "N", "kind", "pair_id", "side"]
    for c in cols:
        if c not in x:
            x[c] = np.nan
    return x[cols].drop_duplicates("run_id")


def _profile_files(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in root.rglob("common_test_L_profiles.parquet"):
        try:
            x = pd.read_parquet(p, columns=["run_id"])
            if len(x) and str(x.iloc[0]["run_id"]) not in out:
                out[str(x.iloc[0]["run_id"])] = p
        except Exception:
            continue
    return out


def _load_profile(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    x = pd.read_parquet(path)
    result = {}
    for _, r in x.iterrows():
        tid = str(r["transcript_id"])
        l = np.asarray(r["L_t"], dtype=float)
        mask = np.asarray(r["valid_position_mask"], dtype=bool) if "valid_position_mask" in r else np.ones(l.size, bool)
        n = min(l.size, mask.size)
        result[tid] = (l[:n], mask[:n], int(r.get("transcript_length", n)))
    return result


def _metrics(a, b, ma, mb):
    n = min(len(a), len(b), len(ma), len(mb))
    z = np.isfinite(a[:n]) & np.isfinite(b[:n]) & ma[:n] & mb[:n]
    a, b = a[:n][z], b[:n][z]
    if len(a) < 2:
        return (np.nan, np.nan, np.nan, 0)
    pcc = np.corrcoef(a, b)[0, 1] if np.std(a) > 0 and np.std(b) > 0 else np.nan
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    sp = np.corrcoef(ra, rb)[0, 1] if np.std(ra) > 0 and np.std(rb) > 0 else np.nan
    return float(pcc), float(sp), float(np.sqrt(np.mean((a - b) ** 2))), int(len(a))


def _summary(x: pd.Series, group: str | None = None) -> pd.DataFrame:
    if group:
        rows = []
        for key, g in x.groupby(group):
            v = pd.to_numeric(g["PCC"], errors="coerce").dropna()
            rows.append({"N": key, "n": len(v), "mean_PCC": v.mean(), "median_PCC": v.median(),
                         "p05_PCC": v.quantile(.05), "p25_PCC": v.quantile(.25),
                         "p75_PCC": v.quantile(.75), "p95_PCC": v.quantile(.95),
                         "mean_RMSE": g["RMSE"].mean(), "median_RMSE": g["RMSE"].median()})
        return pd.DataFrame(rows).sort_values("N") if rows else pd.DataFrame()
    v = pd.to_numeric(x["PCC"], errors="coerce").dropna()
    return pd.DataFrame([{"n": len(v), "mean_PCC": v.mean(), "median_PCC": v.median(),
                          "p05_PCC": v.quantile(.05), "p25_PCC": v.quantile(.25),
                          "p75_PCC": v.quantile(.75), "p95_PCC": v.quantile(.95),
                          "mean_RMSE": x["RMSE"].mean(), "median_RMSE": x["RMSE"].median()}])


def _save_availability(plan, paths, out):
    p = plan.copy()
    p["available"] = p["run_id"].isin(paths)
    p["profile_path"] = p["run_id"].map(lambda r: str(paths[r]) if r in paths else "")
    p.to_csv(out / "available_runs.csv", index=False)
    p.loc[~p.available].to_csv(out / "missing_runs.csv", index=False)
    return p


def _save_figure(figure: plt.Figure, stem: Path) -> None:
    """Write publication-quality vector and raster companions."""
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


@latex_paper_style
def _plot_availability(p, out):
    q = p.groupby("N", as_index=False).agg(
        expected=("run_id", "size"), available=("available", "sum")
    )
    q.to_csv(out / "availability_by_N.csv", index=False)
    figure, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    x = np.arange(len(q))
    width = 0.38
    axis.bar(
        x - width / 2,
        q.expected,
        width,
        label="Planned",
        color="#bdbdbd",
    )
    axis.bar(
        x + width / 2,
        q.available,
        width,
        label="Valid best-validation-loss output",
        color="#377eb8",
    )
    axis.set_xticks(x, q.N.astype(int))
    axis.set_xlabel("Number of datasets ($N$)")
    axis.set_ylabel("Runs")
    axis.legend(frameon=False)
    axis.set_title("Experiment 8 run availability")
    _save_figure(figure, out / "availability_by_N")


@latex_paper_style
def _plot_stability(s, out):
    if s.empty:
        return
    g = s.groupby("N")["PCC"].agg(["mean", "median", "count", "std"]).reset_index()
    g.to_csv(out / "stability_summary_by_N.csv", index=False)
    figure, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    axis.errorbar(
        g.N,
        g["mean"],
        yerr=g["std"].fillna(0),
        fmt="o-",
        capsize=3,
    )
    axis.set_xlabel("Number of datasets ($N$)")
    axis.set_ylabel(r"Shared-profile $L_t$ PCC (disjoint pairs)")
    axis.set_title("Independent shared-profile stability (available pairs)")
    axis.grid(alpha=0.2)
    _save_figure(figure, out / "stability_vs_N_partial")


@latex_paper_style
def _plot_overlap(s, out):
    if s.empty:
        return
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    axis.scatter(s.jaccard, s.mean_PCC, c=s.N, cmap="viridis", s=35)
    axis.set_xlabel("Dataset Jaccard overlap")
    axis.set_ylabel(r"Mean shared-profile $L_t$ PCC")
    axis.set_title("Overlap versus agreement (secondary)")
    _save_figure(figure, out / "overlap_vs_agreement_partial")


@latex_paper_style
def _plot_quality(root, out):
    path = root / "subset_quality_report.csv"
    if not path.exists():
        return
    q = pd.read_csv(path)
    q.to_csv(out / "subset_quality_source.csv", index=False)
    column = "quality_mismatch" if "quality_mismatch" in q else q.columns[-1]
    figure, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    q.boxplot(column=column, by="N", ax=axis)
    axis.set_title("Quality mismatch of selected subsets")
    figure.suptitle("")
    axis.set_xlabel("Number of datasets ($N$)")
    axis.set_ylabel("Quality mismatch")
    _save_figure(figure, out / "subset_quality_balance")


@latex_paper_style
def _plot_representative(st, loaded, out):
    if st.empty:
        return
    # Choose deterministic low/median/high agreement transcripts from the
    # available disjoint comparisons, then overlay the two profiles.
    means = st.groupby(["run_a", "run_b", "transcript_id"], as_index=False).PCC.mean()
    chosen = means.iloc[np.unique(np.clip([0, len(means)//2, len(means)-1], 0, max(0, len(means)-1)))]
    figure, axes = plt.subplots(
        len(chosen),
        1,
        figsize=(10.2, 3.0 * len(chosen)),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, (_, row) in zip(axes[:, 0], chosen.iterrows()):
        a, b, tid = row.run_a, row.run_b, row.transcript_id
        xa, ma, _ = loaded[a][tid]; xb, mb, _ = loaded[b][tid]
        n=min(len(xa),len(xb)); mask=ma[:n]&mb[:n]
        label_a = str(a).removeprefix("real_exp8_").replace("_", " ")
        label_b = str(b).removeprefix("real_exp8_").replace("_", " ")
        axis.plot(np.flatnonzero(mask), xa[:n][mask], label=label_a, lw=1)
        axis.plot(np.flatnonzero(mask), xb[:n][mask], label=label_b, lw=1)
        axis.set_title(f"{tid}  (PCC={row.PCC:.3f})")
        axis.set_ylabel(r"Shared profile $L_t$")
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("CDS codon position")
    axes[0, 0].legend(frameon=False, ncol=2)
    figure.suptitle("Representative shared profiles (available disjoint pairs)")
    _save_figure(figure, out / "representative_L_profiles_partial")


@latex_paper_style
def _plot_convergence(summary: pd.DataFrame, out: Path) -> None:
    """Plot convergence to the complete $N=114$ collection when available."""
    figure, axis = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    axis.plot(summary.N, summary["mean"], "o-")
    axis.set_xlabel("Number of datasets ($N$)")
    axis.set_ylabel(r"Shared-profile $L_t$ PCC to $N=114$")
    axis.set_title("Convergence to the full collection")
    axis.grid(alpha=0.2)
    _save_figure(figure, out / "convergence_to_full_vs_N_partial")


def main(argv: Sequence[str] | None = None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-root", type=Path, default=None,
        help="Experiment root. Default: newest Experiment-8 run containing predictions.",
    )
    ap.add_argument("--output-dir",type=Path)
    args=ap.parse_args(argv); root=(args.run_root or _discover_run_root()).expanduser().resolve(); out=(args.output_dir or root/"analysis_partial").expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    print(f"Experiment root: {root}")
    plan=_planned_runs(root); paths=_profile_files(root); status=_save_availability(plan,paths,out); _plot_availability(status,out); _plot_quality(root,out)
    loaded={k:_load_profile(v) for k,v in paths.items()}
    rec=[]; pairrec=[]; conv=[]; allpairs=[]
    overlap = {}
    op = root / "overlap_report.csv"
    if op.exists():
        ox = pd.read_csv(op)
        for _, rr in ox.iterrows():
            overlap[(str(rr.run_a), str(rr.run_b))] = rr
            overlap[(str(rr.run_b), str(rr.run_a))] = rr
    meta=status.set_index("run_id").to_dict("index") if len(status) else {}
    for a,b in itertools.combinations(sorted(paths),2):
        ma,mb=meta.get(a,{}),meta.get(b,{})
        if ma.get("N")!=mb.get("N"): continue
        n=int(ma["N"]); designated=bool(ma.get("pair_id")==mb.get("pair_id") and ma.get("side") in ("A","B") and mb.get("side") in ("A","B") and ma.get("side")!=mb.get("side"))
        vals=[]
        for tid in sorted(set(loaded[a])&set(loaded[b])):
            x,mx,lx=loaded[a][tid]; y,my,ly=loaded[b][tid]; pcc,sp,rm,npos=_metrics(x,y,mx,my)
            vals.append(pcc); row={"N":n,"pair_id":ma.get("pair_id",""),"run_a":a,"run_b":b,"transcript_id":tid,"PCC":pcc,"Spearman":sp,"RMSE":rm,"n_positions":npos,"designated_disjoint":designated}
            if designated: pairrec.append(row)
        if vals:
            ov = overlap.get((a, b))
            allpairs.append({"N":n,"run_a":a,"run_b":b,
                             "intersection_count": getattr(ov, "intersection_count", np.nan),
                             "jaccard": getattr(ov, "jaccard", np.nan),
                             "mean_PCC":np.nanmean(vals),"median_PCC":np.nanmedian(vals),"designated_disjoint":designated})
    st=pd.DataFrame(pairrec); st.to_parquet(out/"stability_disjoint_per_transcript.parquet",index=False) if not st.empty else pd.DataFrame().to_parquet(out/"stability_disjoint_per_transcript.parquet",index=False)
    _plot_stability(st,out)
    _plot_representative(st, loaded, out)
    apairs=pd.DataFrame(allpairs); apairs.to_csv(out/"same_N_all_pairs.csv",index=False); _plot_overlap(apairs,out)
    # Full-reference convergence is only meaningful when an N=114 profile exists.
    full=[r for r in paths if int(meta.get(r,{}).get("N",-1))==114]
    if full:
        ref=loaded[full[0]]
        for r in paths:
            if r==full[0]: continue
            for tid in sorted(set(loaded[r])&set(ref)):
                x,mx,lx=loaded[r][tid]; y,my,ly=ref[tid]; pcc,sp,rm,np_= _metrics(x,y,mx,my); conv.append({"N":meta[r].get("N"),"run_id":r,"transcript_id":tid,"PCC":pcc,"Spearman":sp,"RMSE":rm,"n_positions":np_})
    cv=pd.DataFrame(conv); cv.to_parquet(out/"convergence_to_full_per_transcript.parquet",index=False) if not cv.empty else pd.DataFrame().to_parquet(out/"convergence_to_full_per_transcript.parquet",index=False)
    if not cv.empty:
        cs=cv.groupby("N")["PCC"].agg(["mean","median","count"]).reset_index(); cs.to_csv(out/"convergence_to_full_summary.csv",index=False)
        _plot_convergence(cs, out)
    manifest={"run_root":str(root),"planned_runs":int(len(plan)),"available_runs":int(len(paths)),"available_N":sorted({int(meta[r]["N"]) for r in paths if r in meta}),"n_disjoint_pair_comparisons":int(len(st)),"full_114_reference_available":bool(full),"best_validation_loss_only":True,"pdf_written":True,"svg_written":False,"plot_typography":"LaTeX/Latin Modern","minimum_plot_font_size_pt":12}
    (out/"partial_analysis_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(f"Available valid outputs: {len(paths)}/{len(plan)}; N values: {manifest['available_N']}; N=114 reference: {'yes' if full else 'no'}")
    print(f"Wrote partial analysis to {out}")
    return 0

if __name__=="__main__": raise SystemExit(main())
