#!/usr/bin/env python3
"""Pilot dataset geometry from frozen, transcript-matched log-gamma profiles.

One checkpoint at a time; no training, observation-based position filtering,
cross-model pooling, or nonlinear embedding tuning. Default: largest completed
uniform-reference model and 50 deterministically selected held-out transcripts.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import hashlib
import html
import json
import shlex
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Utils.publication_plot_style import publication_rc


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


SOURCE_SHA256 = sha256(__file__)


def id_hash(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def log_distance_squared(profiles):
    """Pairwise mean squared log contrasts on one common transcript domain."""
    values = np.asarray(profiles, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("Expected finite dataset-by-position log profiles.")
    return squareform(pdist(values, metric="sqeuclidean")) / values.shape[1]


def classical_mds(distance):
    """Principal coordinates; report distortion instead of implying perfect 2D fit."""
    distance = np.asarray(distance, dtype=np.float64)
    n = len(distance)
    center = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * center @ (distance ** 2) @ center
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    tolerance = max(float(eigenvalues[0]), 1.0) * 1e-10
    if eigenvalues.min() < -tolerance:
        raise ValueError("Distance matrix is not Euclidean on the shared cohort.")
    positive = np.maximum(eigenvalues, 0)
    if positive.sum() <= tolerance:
        raise ValueError("All dataset profiles coincide; no meaningful projection.")
    coordinates = eigenvectors[:, :2] * np.sqrt(positive[:2])
    for k in range(coordinates.shape[1]):
        if coordinates[np.argmax(np.abs(coordinates[:, k])), k] < 0:
            coordinates[:, k] *= -1
    full, projected = squareform(distance, checks=False), pdist(coordinates)
    diagnostics = {
        "axis1_fraction": float(positive[0] / positive.sum()),
        "axis2_fraction": float(positive[1] / positive.sum()),
        "fraction_2d": float(positive[:2].sum() / positive.sum()),
        "normalized_distance_stress": float(np.linalg.norm(projected - full) / np.linalg.norm(full)),
        "distance_spearman": float(spearmanr(full, projected).statistic) if n > 2 else None,
        "minimum_eigenvalue": float(eigenvalues.min()),
    }
    return coordinates, eigenvalues, diagnostics


def choose_task(root, manifest, arm, n):
    inventory = []
    eligible = []
    for task in manifest["tasks"]:
        directory = root / task["directory"]
        state_path = directory / "execution_status.json"
        state = read_json(state_path) if state_path.exists() else {}
        exported = list(directory.glob("predictions/**/prediction_checkpoint_manifest.json"))
        ready = state.get("status") == "completed" and len(exported) == 1
        inventory.append(dict(run_id=task["run_id"], arm=task["arm"], N=task["N"],
                              recorded_status=state.get("status", "absent"), completed_export=ready))
        if ready and task["arm"] == arm and (n is None or task["N"] == n):
            eligible.append((task, state, exported[0]))
    if not eligible:
        raise FileNotFoundError(f"No completed export for arm={arm}, N={n}.")
    return max(eligible, key=lambda x: x[0]["N"]), inventory


def selected_raw_rows(path, selected):
    columns = ["transcript_id", "dataset_id", "length", "mask", "codon_ids", "log_gamma", "L_bio"]
    seen = set()
    with pq.ParquetFile(path) as reader:
        for batch in reader.iter_batches(batch_size=8, columns=columns, use_threads=False):
            for row in batch.to_pylist():
                tid = row["transcript_id"]
                if tid not in selected:
                    continue
                if tid in seen:
                    raise ValueError("Pilot expects the verified one-row-per-transcript sequence export.")
                seen.add(tid)
                mask = np.asarray(row["mask"], bool)
                if mask.sum() != row["length"] or not np.array_equal(np.flatnonzero(mask), np.arange(mask.sum())):
                    raise ValueError(f"Noncontiguous/misaligned valid CDS: {tid}")
                for key in ("codon_ids", "log_gamma", "L_bio"):
                    values = np.asarray(row[key])
                    if values.shape != mask.shape:
                        raise ValueError(f"{tid}: {key} is not aligned to the valid mask")
                    row[key] = values[mask]
                yield row
    if seen != selected:
        raise ValueError(f"Missing test transcripts in saved export: {sorted(selected - seen)}")


def resolve_inputs(root, manifest, task, state, export_manifest):
    from omegaconf import OmegaConf
    from analyses.analyze_real_exp8_reference_directionality import verify_runtime_config
    original_root = Path(manifest["output_root"])
    original_repo = original_root.parent.parent

    def local(path):
        path = Path(path)
        return root / path.relative_to(original_root) if path.is_relative_to(original_root) else ROOT / path.relative_to(original_repo)

    directory = root / task["directory"]
    config_path = directory / "hydra/.hydra/config.yaml"
    cfg = OmegaConf.load(config_path)
    prepared_path = local(task["config_path"])
    if sha256(prepared_path) != task["config_sha256"]:
        raise ValueError("Prepared configuration changed.")
    verify_runtime_config(OmegaConf.to_container(OmegaConf.load(prepared_path)), OmegaConf.to_container(cfg), state)
    runtime = read_json(export_manifest)["best_val_loss"]
    if not runtime["sequence_only_shared_profile_prediction"] or runtime["split_name"] != "test":
        raise ValueError("Expected frozen sequence-only held-out export.")
    from Utils.reliability_references import transcript_id_hash
    test_ids = manifest["source_folds"][str(task["N"])]["test_ids"]
    if runtime["transcript_count"] != len(test_ids) or runtime["transcript_id_hash"] != transcript_id_hash(test_ids):
        raise ValueError("Saved export does not match the frozen test cohort.")
    checkpoint = local(runtime["checkpoint_path"])
    if sha256(checkpoint) != state["outputs"]["checkpoint_sha256"]:
        raise ValueError("Best-checkpoint hash differs from the completion record.")
    gamma_path = export_manifest.parent / "gamma_reference_manifest.json"
    reference = read_json(gamma_path)
    names = task["datasets"]
    if reference["reference_dataset_names"] != names or reference["centering_mode"] != "fixed_reference":
        raise ValueError("Unexpected gamma reference panel.")
    weights_path = root / "reference_weights.csv"
    weights = pd.read_csv(weights_path)
    weights = weights[(weights.N == task["N"]) & (weights.arm == task["arm"])].set_index("dataset_id").loc[names]
    np.testing.assert_allclose(reference["reference_pi"], weights.pi, rtol=1e-6, atol=1e-9)
    if task["arm"] == "equal":
        np.testing.assert_allclose(reference["reference_pi"], np.ones(len(names)) / len(names))
    if list(cfg.experiment.dataset) != names or int(cfg.experiment.seed) != int(task["training_seed"]):
        raise ValueError("Runtime seed/membership does not match the selected task.")
    if any(str(v.get("route", "none")) != "none" for v in cfg.model.additional_sequence_features.values()):
        raise ValueError("This pilot's encoder requires the saved baseline (no additional routed features).")
    for key in cfg.paths.encodings:
        cfg.paths.encodings[key] = str(local(cfg.paths.encodings[key]))
    cfg.data.dataset_quality_ranking.path = str(local(cfg.data.dataset_quality_ranking.path))
    raw_path = local(runtime["output_path"])
    provenance = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                  "runtime_config": str(config_path), "runtime_config_sha256": sha256(config_path),
                  "raw_export": str(raw_path), "raw_export_sha256": sha256(raw_path),
                  "reference_manifest_sha256": sha256(gamma_path),
                  "reference_weights_sha256": sha256(weights_path), "reference": reference}
    return cfg, checkpoint, raw_path, reference, weights, provenance


def infer_profiles(cfg, checkpoint, raw_path, selected, reference, args, out):
    import torch
    from lightning_fabric.utilities.apply_func import move_data_to_device
    from analyses.analyze_real_panel_posthoc_robustness_streaming import _make_frozen_model_and_encoder, _autocast_context
    torch.set_num_threads(1)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    module, encoder, names = _make_frozen_model_and_encoder(cfg, checkpoint=checkpoint, device=device)
    module.requires_grad_(False)
    model = module.model
    np.testing.assert_array_equal(model.gamma_reference_dataset_ids.cpu(), reference["reference_dataset_ids"])
    pi = model.gamma_reference_weights.detach().cpu().numpy().astype(float)
    pi /= pi.sum()
    np.testing.assert_allclose(pi, reference["reference_pi"], rtol=1e-6, atol=1e-9)
    inverse_codons = {int(v): k for k, v in encoder.codon_encoding.items()}
    dataset_ids = reference["reference_dataset_ids"]
    schema = pa.schema([("transcript_id", pa.string()), ("dataset", pa.string()),
                        ("codon_ids", pa.list_(pa.int16())), ("log_gamma", pa.list_(pa.float32())),
                        ("gamma", pa.list_(pa.float32()))])
    replay_rows, profile_rows, memory_rows, matrices, tids = [], [], [], [], []
    with pq.ParquetWriter(out / "gamma_profiles.parquet", schema, compression="zstd") as writer:
        for number, row in enumerate(selected_raw_rows(raw_path, selected), 1):
            tid, codon_ids = row["transcript_id"], row["codon_ids"]
            n = len(codon_ids)
            chunk_size = min(args.dataset_batch_size, args.max_padded_tokens // n)
            if chunk_size < 1:
                raise ValueError(f"{tid} exceeds the padded-token limit without truncation.")
            model.gamma_reference_chunk_size = chunk_size  # execution only; panel/pi unchanged
            batch = move_data_to_device(encoder.collate([(tid, tuple(inverse_codons[int(c)] for c in codon_ids))]), device)
            np.testing.assert_array_equal(batch[6][0].cpu(), codon_ids)
            logs = np.empty((len(names), n), dtype=np.float64)
            anchor_L = None
            for start in range(0, len(names), chunk_size):
                count = min(chunk_size, len(names) - start)
                with torch.inference_mode(), _autocast_context(cfg, device):
                    mu, log_sigma, result = model(x_packed=batch[2], codon_ids=batch[6].expand(count, -1),
                                   id_datasets=torch.tensor(dataset_ids[start:start + count], device=device),
                                   mask=batch[5].expand(count, -1), target=batch[3].expand(count, -1),
                                   sample_ids=[tid] * count, transcript_group_index=torch.zeros(count, dtype=torch.long, device=device))
                logs[start:start + count] = result["log_gamma"].detach().float().cpu().numpy()
                if start == 0:
                    anchor_L = result["L_bio"][0].detach().float().cpu().numpy()
                del result, mu, log_sigma
            del batch
            anchor_index = dataset_ids.index(int(row["dataset_id"]))
            delta = logs[anchor_index] - row["log_gamma"]
            l_delta = anchor_L - row["L_bio"]
            replay = dict(transcript_id=tid, log_gamma_rmse=float(np.sqrt(np.mean(delta ** 2))),
                          log_gamma_max_abs_error=float(np.max(np.abs(delta))),
                          L_relative_rmse=float(np.sqrt(np.mean(l_delta ** 2)) / np.sqrt(np.mean(row["L_bio"] ** 2))),
                          positional_gauge_error=float(np.max(np.abs(logs.mean(axis=1)))),
                          reference_gauge_error=float(np.max(np.abs(pi @ logs))))
            replay_rows.append(replay)
            if replay["log_gamma_rmse"] > args.replay_rmse_tolerance or replay["L_relative_rmse"] > args.replay_rmse_tolerance:
                pd.DataFrame(replay_rows).to_csv(out / "replay_validation.csv", index=False)
                raise ValueError(f"Frozen replay failed: {replay}")
            if max(replay["positional_gauge_error"], replay["reference_gauge_error"]) > 2e-4:
                raise ValueError(f"Gamma gauge check failed: {replay}")
            if not np.isfinite(logs).all() or not np.isfinite(np.exp(logs)).all():
                raise ValueError(f"Nonfinite gamma for {tid}")
            domain = slice(args.trim, n - args.trim)
            if n <= 2 * args.trim + 1:
                raise ValueError(f"{tid}: too short for the requested trimming; no pairwise cohort substitution.")
            matrices.append(log_distance_squared(logs[:, domain]))
            tids.append(tid)
            for index, name in enumerate(names):
                values = logs[index, domain]
                profile_rows.append(dict(transcript_id=tid, dataset=name, n_positions=len(values),
                                         log_gamma_mean=float(values.mean()), log_gamma_sd=float(values.std()),
                                         near_constant=bool(values.std() < 1e-6)))
            writer.write_table(pa.Table.from_pylist([
                dict(transcript_id=tid, dataset=name, codon_ids=codon_ids.tolist(),
                     log_gamma=logs[index].astype(np.float32).tolist(), gamma=np.exp(logs[index]).astype(np.float32).tolist())
                for index, name in enumerate(names)], schema=schema))
            rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 30
            memory_rows.append(dict(transcript_index=number, transcript_id=tid, rss_gb=rss,
                                    cuda_allocated_gb=torch.cuda.memory_allocated() / 2 ** 30 if device.type == "cuda" else 0,
                                    cuda_peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0))
            if rss > args.max_rss_gb:
                raise MemoryError(f"RSS {rss:.2f} GiB exceeded {args.max_rss_gb:g} GiB budget.")
            if number % 5 == 0 or number == len(selected):
                print(f"{number}/{len(selected)} transcripts, {len(names)} datasets; RSS={rss:.2f} GiB; replay RMSE={replay['log_gamma_rmse']:.3g}", flush=True)
            del logs, row, anchor_L
    del model, module, encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    pd.DataFrame(replay_rows).to_csv(out / "replay_validation.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(out / "profile_diagnostics.csv", index=False)
    pd.DataFrame(memory_rows).to_csv(out / "memory_log.csv", index=False)
    return tids, np.stack(matrices), str(device)


def label_projection(axis, coordinates, fontsize):
    """Place short labels without text/point collisions; never move the data."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.transforms import Bbox
    axis.figure.canvas.draw()
    renderer = axis.figure.canvas.get_renderer()
    pixels = axis.transData.transform(coordinates)
    bounds = axis.get_window_extent()
    point_boxes = [Bbox.from_bounds(x - 5, y - 5, 10, 10) for x, y in pixels]
    used = []
    # Approximate bounding boxes with an installed font; the visible labels
    # still use the publication LaTeX style set by plot_geometry.
    font = FontProperties(size=fontsize, weight="bold", family="DejaVu Serif")
    for index, ((x, y), data_xy) in enumerate(zip(pixels, coordinates)):
        width, height, _ = renderer.get_text_width_height_descent(str(index + 1), font, ismath=False)
        candidates = []
        for radius in (13, 21, 30, 40, 52):
            for angle in np.deg2rad((45, 135, -45, -135, 90, -90, 0, 180)):
                dx, dy = radius * np.cos(angle), radius * np.sin(angle)
                box = Bbox.from_bounds(x + dx - width / 2 - 2, y + dy - height / 2 - 2, width + 4, height + 4)
                overlap = sum(box.overlaps(other) for other in used + point_boxes)
                outside = not (bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1))
                candidates.append((10000 * outside + 1000 * overlap + radius, dx, dy, box))
        _, dx, dy, box = min(candidates, key=lambda item: item[0])
        used.append(box)
        axis.annotate(str(index + 1), data_xy, xytext=(dx, dy), textcoords="offset pixels",
                      ha="center", va="center", fontsize=fontsize,
                      arrowprops=dict(arrowstyle="-", color="#78858f", lw=.5, shrinkA=1, shrinkB=5))


def plot_geometry(distance, coordinates, diagnostics, metadata, out, stem, title, font_size):
    n = len(metadata)
    order = leaves_list(linkage(squareform(distance, checks=False), method="average", optimal_ordering=True))
    counts = Counter(metadata.gse)
    shared = [gse for gse, count in counts.items() if gse and count > 1]
    palette = ["#2878a0", "#af573c", "#67844f", "#896299", "#bc9a31"]
    colors = {gse: palette[i % len(palette)] for i, gse in enumerate(shared)}
    point_colors = [colors.get(gse, "#929da6") for gse in metadata.gse]
    style = publication_rc()
    style.update({"font.size": font_size, "axes.labelsize": font_size, "axes.titlesize": font_size + 1,
                  "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold"})
    if style["text.usetex"]:
        style["text.latex.preamble"] += r"\renewcommand{\seriesdefault}{\bfdefault}\AtBeginDocument{\boldmath}"
    escape = lambda s: str(s).replace("_", r"\_").replace("%", r"\%") if style["text.usetex"] else str(s)
    with matplotlib.rc_context(style):
        fig = plt.figure(figsize=(22, 12), layout="constrained")
        grid = fig.add_gridspec(1, 3, width_ratios=(1.12, 1, .82))
        ax, projection, key = [fig.add_subplot(grid[0, k]) for k in range(3)]
        im = ax.imshow(distance[np.ix_(order, order)], cmap="Blues", vmin=0, interpolation="nearest")
        ax.set_xticks(range(n), metadata.number.to_numpy()[order], rotation=90, fontsize=10)
        ax.set_yticks(range(n), metadata.number.to_numpy()[order], fontsize=10)
        ax.set(xlabel="Dataset number (clustered order)", ylabel="Dataset number", title=r"A  Matched-profile distance")
        fig.colorbar(im, ax=ax, orientation="horizontal", fraction=.045, pad=.09, label=r"RMS log-$\gamma$ difference")
        projection.scatter(coordinates[:, 0], coordinates[:, 1], c=point_colors, s=90, edgecolors="white", linewidths=.7)
        projection.set(xlabel=escape(f"Coordinate 1 ({diagnostics['axis1_fraction']:.1%})"),
                       ylabel=escape(f"Coordinate 2 ({diagnostics['axis2_fraction']:.1%})"),
                       title=escape(f"B  Classical MDS: {diagnostics['fraction_2d']:.1%} retained"))
        projection.set_aspect("equal", adjustable="box")
        projection.margins(.17)
        projection.grid(alpha=.25)
        key.axis("off")
        key.set_title("Dataset key", loc="left")
        for index, row in enumerate(metadata.itertuples(index=False)):
            key.text(0, 1 - (index + 1) / (n + 2), f"{row.number:2d}  {escape(row.dataset)}",
                     transform=key.transAxes, fontsize=12 * font_size / 14,
                     color=colors.get(row.gse, "#596977"), va="top")
        key.text(0, 0, "Color: shared GEO study; gray: other studies", fontsize=10, transform=key.transAxes)
        fig.suptitle(escape(title), fontsize=font_size + 3)
        # Freeze the axes geometry before placing collision-aware annotations.
        fig.canvas.draw()
        fig.set_layout_engine("none")
        label_projection(projection, coordinates, fontsize=12 * font_size / 14)
        for extension in ("pdf", "png", "svg"):
            fig.savefig(out / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=ROOT / "results/cumulative_stability_seed42")
    parser.add_argument("--arm", default="equal", choices=("equal", "ranked_p1", "reverse_p1", "ranked_p3", "reverse_p3"))
    parser.add_argument("--n-datasets", type=int)
    parser.add_argument("--max-transcripts", type=int, default=50, help="0 uses all held-out transcripts.")
    parser.add_argument("--transcript-id", help="Use only this specified test transcript.")
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--trim", type=int, default=0)
    parser.add_argument("--dataset-batch-size", type=int, default=4)
    parser.add_argument("--max-padded-tokens", type=int, default=20000)
    parser.add_argument("--max-rss-gb", type=float, default=4.0)
    parser.add_argument("--replay-rmse-tolerance", type=float, default=.01)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--font-size", type=float, default=14)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plot-only", action="store_true", help="Replot existing matrices; do not load a model.")
    args = parser.parse_args(argv)
    if args.max_transcripts < 0 or args.trim < 0 or min(args.dataset_batch_size, args.max_padded_tokens, args.bootstrap) <= 0:
        parser.error("Invalid cohort, trimming, batch, or bootstrap size.")
    root = args.experiment_root.resolve()
    manifest_path = root / "experiment_manifest.json"
    manifest = read_json(manifest_path)
    (task, state, export_manifest), inventory = choose_task(root, manifest, args.arm, args.n_datasets)
    out = args.output_dir or root / "analysis/gamma_geometry" / f"{args.arm}_N{task['N']:03d}"
    out.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        provenance = read_json(out / "provenance.json")
        metadata = pd.read_csv(out / "projection_coordinates.csv").fillna({"gse": ""})
        distance = pd.read_csv(out / "distance_matrix.csv", index_col=0).to_numpy(float)
        coordinates = metadata[["coordinate_1", "coordinate_2"]].to_numpy(float)
        plot_geometry(distance, coordinates, provenance["projection"], metadata, out, "gamma_distance_projection",
                      f"Dataset-specific gamma geometry | N={provenance['N']}, {provenance['arm']}, training seed {provenance['seed']} | {provenance['selected_transcripts']} held-out transcripts", args.font_size)
        example = pd.read_csv(out / "example_transcript_distance_matrix.csv", index_col=0).to_numpy(float)
        example_coordinates, _, example_diagnostics = classical_mds(example)
        plot_geometry(example, example_coordinates, example_diagnostics, metadata, out, "gamma_example_transcript_projection",
                      f"One matched transcript: {provenance['example_transcript']} | N={provenance['N']}, {provenance['arm']}", args.font_size)
        provenance["plot_script_sha256"] = SOURCE_SHA256
        provenance["plot_font_size"] = args.font_size
        provenance["plot_command"] = shlex.join([sys.executable, *sys.argv])
        write_json(out / "provenance.json", provenance)
        write_report(out, provenance, pd.read_csv(out / "pairwise_distances.csv"))
        return
    pd.DataFrame(inventory).to_csv(out / "run_inventory.csv", index=False)
    folds = manifest["source_folds"][str(task["N"])]; test_ids = set(folds["test_ids"])
    if test_ids & (set(folds["train_ids"]) | set(folds["validation_ids"])):
        raise ValueError("Test cohort overlaps training or validation.")
    ordered = sorted(test_ids, key=lambda tid: hashlib.sha256(f"{args.sampling_seed}:{tid}".encode()).hexdigest())
    selected_ids = [args.transcript_id] if args.transcript_id else ordered[:args.max_transcripts or None]
    if not set(selected_ids) <= test_ids:
        raise ValueError("Requested transcript is not in the frozen test cohort.")
    cfg, checkpoint, raw_path, reference, weights, provenance = resolve_inputs(root, manifest, task, state, export_manifest)
    names = task["datasets"]
    print(f"Pilot: {task['run_id']}; {len(selected_ids)}/{len(test_ids)} held-out transcripts", flush=True)
    tids, squared, device = infer_profiles(cfg, checkpoint, raw_path, set(selected_ids), reference, args, out)
    indices = [tids.index(tid) for tid in selected_ids]
    squared = squared[indices]
    np.savez_compressed(out / "transcript_distances.npz", transcript_ids=selected_ids, datasets=names, squared_distances=squared)
    pd.DataFrame({"transcript_id": selected_ids}).to_csv(out / "selected_transcripts.csv", index=False)
    distance = np.sqrt(squared.mean(axis=0))
    coordinates, eigenvalues, diagnostics = classical_mds(distance)
    pd.DataFrame(distance, index=names, columns=names).to_csv(out / "distance_matrix.csv", index_label="dataset")
    metadata_path = ROOT / "analyses/artifacts/real_data/hek293_metadata/dataset_metadata.tsv"
    info = pd.read_csv(metadata_path, sep="\t").set_index("dataset") if metadata_path.exists() else pd.DataFrame()
    metadata = pd.DataFrame(dict(number=np.arange(1, len(names) + 1), dataset=names,
                                 gse=[str(info.loc[n, "gse"]) if n in info.index else "" for n in names],
                                 global_rank=weights.global_rank.to_numpy(), reference_pi=reference["reference_pi"]))
    metadata.assign(coordinate_1=coordinates[:, 0], coordinate_2=coordinates[:, 1]).to_csv(out / "projection_coordinates.csv", index=False)
    pd.DataFrame({"eigenvalue": eigenvalues}).to_csv(out / "projection_eigenvalues.csv", index=False)
    pair_i, pair_j = np.triu_indices(len(names), 1)
    per_transcript_pairs = squared[:, pair_i, pair_j]
    rng = np.random.default_rng(args.sampling_seed)
    boot = np.empty((args.bootstrap, len(pair_i)))
    for k in range(args.bootstrap):
        boot[k] = np.sqrt(per_transcript_pairs[rng.integers(len(selected_ids), size=len(selected_ids))].mean(axis=0))
    lo, hi = np.quantile(boot, [.025, .975], axis=0)
    pairs = pd.DataFrame(dict(dataset_a=np.array(names)[pair_i], dataset_b=np.array(names)[pair_j],
                              distance=distance[pair_i, pair_j], ci_lower=lo, ci_upper=hi,
                              same_gse=[bool(metadata.gse[i] and metadata.gse[i] == metadata.gse[j]) for i, j in zip(pair_i, pair_j)]))
    pairs.sort_values("distance").to_csv(out / "pairwise_distances.csv", index=False)
    plot_geometry(distance, coordinates, diagnostics, metadata, out, "gamma_distance_projection",
                  f"Dataset-specific gamma geometry | N={len(names)}, {args.arm}, training seed {task['training_seed']} | {len(selected_ids)} held-out transcripts", args.font_size)
    example = np.sqrt(squared[0])
    example_coordinates, _, example_diagnostics = classical_mds(example)
    pd.DataFrame(example, index=names, columns=names).to_csv(out / "example_transcript_distance_matrix.csv", index_label="dataset")
    plot_geometry(example, example_coordinates, example_diagnostics, metadata, out, "gamma_example_transcript_projection",
                  f"One matched transcript: {selected_ids[0]} | N={len(names)}, {args.arm}", args.font_size)
    provenance.update(created_utc=datetime.now(timezone.utc).isoformat(), task=task["run_id"], seed=task["training_seed"],
                      arm=args.arm, N=len(names), test_cohort_size=len(test_ids), selected_transcripts=len(selected_ids),
                      selection="first SHA256(sampling_seed:transcript_id), independent of predictions", selected_id_hash=id_hash(selected_ids),
                      trim_codons_each_end=args.trim, device=device, dataset_batch_size=args.dataset_batch_size,
                      experiment_manifest_sha256=sha256(manifest_path), script_sha256=SOURCE_SHA256,
                      metadata_sha256=sha256(metadata_path) if metadata_path.exists() else None,
                      projection=diagnostics, example_transcript=selected_ids[0], example_projection=example_diagnostics,
                      bootstrap_draws=args.bootstrap, command=shlex.join([sys.executable, *sys.argv]),
                      neural_parameters_updated=False, observation_dummy_outputs_used=False,
                      metric="sqrt(mean_transcripts(mean_positions((log_gamma_d - log_gamma_e)^2)))")
    provenance["supporting_code_sha256"] = {str(path.relative_to(ROOT)): sha256(path) for path in (
        ROOT / "Models/RiboUnmixModel/RiboUnmixModel.py", ROOT / "main_ribounmix_multidataset.py",
        ROOT / "analyses/analyze_real_panel_posthoc_robustness_streaming.py",
        ROOT / "Dataloaders/RiboUnmixMultiDataset/RiboUnmixMultiDataset.py")}
    command = [sys.executable, str(Path(__file__).resolve()), "--experiment-root", str(root),
               "--arm", args.arm, "--n-datasets", str(task["N"]), "--max-transcripts", str(args.max_transcripts),
               "--sampling-seed", str(args.sampling_seed), "--trim", str(args.trim), "--device", device,
               "--dataset-batch-size", str(args.dataset_batch_size), "--max-padded-tokens", str(args.max_padded_tokens),
               "--max-rss-gb", str(args.max_rss_gb), "--replay-rmse-tolerance", str(args.replay_rmse_tolerance),
               "--bootstrap", str(args.bootstrap), "--font-size", str(args.font_size), "--output-dir", str(out)]
    if args.transcript_id:
        command += ["--transcript-id", args.transcript_id]
    provenance["reproduce_command"] = shlex.join(command)
    write_json(out / "provenance.json", provenance)
    write_report(out, provenance, pairs)
    print(json.dumps(diagnostics, indent=2), flush=True)
    print(f"Results: {out / 'report.html'}", flush=True)


def write_report(out, provenance, pairs):
    p = provenance; diag = p["projection"]
    closest = pairs.nsmallest(10, "distance").to_html(index=False, float_format=lambda x: f"{x:.4f}")
    shared = pairs.groupby("same_gse").distance.agg(["count", "mean", "median"])
    shared.to_csv(out / "same_gse_distance_summary.csv")
    replay = pd.read_csv(out / "replay_validation.csv")
    profiles = pd.read_csv(out / "profile_diagnostics.csv")
    memory = pd.read_csv(out / "memory_log.csv")
    rank_correlation = f"{diag['distance_spearman']:.3f}" if diag['distance_spearman'] is not None else "undefined (only one pair)"
    body = rf"""<!doctype html><html><head><meta charset="utf-8"><title>Frozen gamma geometry pilot</title>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>body{{font:16px/1.6 system-ui;max-width:1150px;margin:35px auto;padding:0 25px;color:#233849}}img{{width:100%}}
td,th{{padding:6px;border-bottom:1px solid #ddd}}table{{border-collapse:collapse}}code{{overflow-wrap:anywhere}}.note{{background:#eef4f8;padding:15px}}</style></head><body>
<h1>Dataset-specific gamma: distance matrix and projection</h1>
<p>{p['N']} datasets from <code>{html.escape(p['task'])}</code>, training seed {p['seed']};
{p['selected_transcripts']} of {p['test_cohort_size']} held-out transcripts. This is a pilot, not the full test cohort.</p>
<p class="note">The existing sequence-only exports contain one dataset's gamma per transcript, not every dataset's gamma.
The missing profiles were obtained from the saved best-validation-loss checkpoint, in evaluation/inference mode,
using the resolved runtime configuration, strict repository checkpoint loader and repository sequence encoder.
No parameters, reference weights or observation reliability weights were fitted or changed. Dummy targets in sequence-only
inference were not interpreted as measured observations. Saved log-gamma and L profiles were replay-checked.</p>
<h2>Distance definition</h2><p>For dataset d and transcript t, let \(g_{{dti}}=\log\gamma_{{dti}}\).
For the same aligned positions \(I_t\) in both datasets,</p>
\[\delta_t(d,e)^2=\frac{{1}}{{|I_t|}}\sum_{{i\in I_t}}(g_{{dti}}-g_{{eti}})^2,
\qquad D(d,e)=\sqrt{{\frac{{1}}{{|T|}}\sum_{{t\in T}}\delta_t(d,e)^2}}.\]
<p>Each transcript has equal total weight, regardless of length. Positions follow the full exported valid-CDS mask,
with {p['trim_codons_each_end']} codons excluded from each end for this analysis. No smoothing, count-zero filtering,
imputation or variance standardization is used. All pairs use exactly the same transcript cohort. Distance retains correction amplitude;
unlike PCC, it remains defined for constant profiles. Near-constant log-gamma profiles are disclosed in the diagnostics CSV.</p>
<p>Log distance treats multiplicative corrections symmetrically. Within one checkpoint, adding the same codon-dependent
reference curve to every dataset cancels in \(g_d-g_e\). This does <em>not</em> imply that retraining with different
reference weights must give the same distances: training can change the learned dataset effects.</p>
<p>Validation: {len(profiles)} dataset–transcript profiles; {int(profiles.near_constant.sum())} near-constant profiles.
Maximum frozen-replay log-gamma RMSE: {replay.log_gamma_rmse.max():.5f}; maximum relative L RMSE:
{replay.L_relative_rmse.max():.2%}. Small replay differences were observed under separate mixed-precision execution;
profiles are not claimed to be bitwise identical. Maximum observed reference-centering error:
{replay.reference_gauge_error.max():.2g}; maximum RSS: {memory.rss_gb.max():.2f} GiB.</p>
<h2>Projection and results</h2><p>Classical multidimensional scaling (principal coordinates) embeds this Euclidean matrix.
The first two coordinates retain {diag['fraction_2d']:.1%} of the total centered squared-distance variation;
normalized distance stress is {diag['normalized_distance_stress']:.3f}.
Projection axes have no intrinsic biological meaning. Nearby points indicate similar learned correction profiles,
not necessarily identical protocols or technical ground truth. Trust the full matrix when a 2D view distorts distances.</p>
<p class="note"><b>Projection limitation:</b> this 2D display retains only {diag['fraction_2d']:.1%} of the geometry.
Its pair-distance rank correlation with the full matrix is {rank_correlation}.
Do not use apparent overlap or separation in this projection alone to rank dataset similarity.</p>
<img src="gamma_distance_projection.png" alt="Transcript-balanced distance matrix and classical MDS projection">
<p>Heatmap order uses average-linkage clustering for display only. Numbers map to dataset names; colors identify repeated
GSEs from the preceding metadata audit, and gray marks other studies. GEO annotation conflicts remain unresolved;
no sample was silently relabelled. This top-quality cumulative prefix is not representative of all 114 datasets.</p>
<h3>Closest dataset pairs</h3>{closest}
<h3>Shared-GSE descriptive comparison</h3>{shared.to_html(float_format=lambda x: f'{x:.4f}')}
<p>True denotes pairs within one GSE, not proof of identical protocols. These summaries are descriptive; there are
many more between-GSE pairs, pairs share datasets, and only a few studies have multiple aliases in this prefix.</p>
<p>The 95% intervals resample transcripts jointly ({p['bootstrap_draws']} draws), conditional on this one trained model
and pilot sample; they do not represent independent training runs. Pairwise distances are dependent and are not treated
as independent observations for hypothesis tests. No clustering p-values or claims of biological/technical disentanglement are made.</p>
<h2>Single-transcript example</h2><p>Prespecified as the first hash-selected transcript, not chosen for visual clustering:
<code>{p['example_transcript']}</code>.</p><img src="gamma_example_transcript_projection.png" alt="One transcript distance matrix and projection">
<h2>Files</h2><p><a href="distance_matrix.csv">Full distance matrix</a> · <a href="pairwise_distances.csv">Pairs and intervals</a> ·
<a href="projection_coordinates.csv">Projection coordinates / dataset key</a> · <a href="gamma_distance_projection.pdf">Vector PDF</a> ·
<a href="replay_validation.csv">Frozen replay checks</a> · <a href="profile_diagnostics.csv">Profile diagnostics</a> ·
<a href="memory_log.csv">Memory log</a> · <a href="provenance.json">Provenance</a>.</p>
<p>Extracted profiles: <code>gamma_profiles.parquet</code>. Per-transcript distance matrices:
<code>transcript_distances.npz</code>. Full and example plots are exported as PDF, PNG and SVG.</p>
<h2>Reproduce</h2><pre>{html.escape(p['reproduce_command'])}</pre>
<p>Use <code>--max-transcripts 0</code> for all held-out transcripts, <code>--transcript-id ENST...</code> for one,
or <code>--arm ranked_p1 --n-datasets 40</code> for a separate frozen-model analysis. Do not pool models into one matrix.</p>
</body></html>"""
    (out / "report.html").write_text(body)
    (out / "command.txt").write_text(p["reproduce_command"] + "\n")


if __name__ == "__main__":
    main()
