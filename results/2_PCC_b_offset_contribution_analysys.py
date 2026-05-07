from __future__ import annotations

import glob
import os
from pathlib import Path

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# ============================================================
# Robust PCC aggregation
# ============================================================

def safe_pearsonr(x, y, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    L = min(len(x), len(y))
    if L < 4:
        return np.nan

    x = x[:L]
    y = y[:L]

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 4:
        return np.nan

    if np.std(x) <= eps or np.std(y) <= eps:
        return np.nan

    x = x - x.mean()
    y = y - y.mean()

    denom = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    if denom <= eps:
        return np.nan

    return float(np.sum(x * y) / denom)


def fisher_weighted_pcc(pred_list, target_list, alpha: float = 0.05):
    pcc_s = []
    n_s = []

    for pred, target in zip(pred_list, target_list):
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)
        target = np.asarray(target, dtype=np.float64).reshape(-1)

        L = min(len(pred), len(target))
        if L < 4:
            continue

        r = safe_pearsonr(pred[:L], target[:L])
        if np.isfinite(r):
            pcc_s.append(r)
            n_s.append(L)

    if not pcc_s:
        return np.nan, np.nan, np.nan, 0

    pcc_array = np.asarray(pcc_s, dtype=np.float64)
    n_array = np.asarray(n_s, dtype=np.float64)

    r_clipped = np.clip(pcc_array, -0.9999, 0.9999)
    z_scores = np.arctanh(r_clipped)

    weights = np.maximum(n_array - 3.0, 1.0)
    z_mean = np.average(z_scores, weights=weights)

    se_z_mean = 1.0 / np.sqrt(np.sum(weights))
    z_critical = norm.ppf(1.0 - alpha / 2.0)

    ci_lower = np.tanh(z_mean - z_critical * se_z_mean)
    ci_upper = np.tanh(z_mean + z_critical * se_z_mean)
    mean_pcc = np.tanh(z_mean)

    return float(mean_pcc), float(ci_lower), float(ci_upper), int(len(pcc_s))


# ============================================================
# Component reconstruction
# ============================================================

def scalar_smean(s):
    arr = np.asarray(s)
    return float(arr.reshape(-1)[0])


def scale_by_smean(profile, s):
    return np.asarray(profile, dtype=np.float32) * scalar_smean(s)


def multiply_arrays(a, b):
    return np.asarray(a, dtype=np.float32) * np.asarray(b, dtype=np.float32)


def get_component_lists(df: pd.DataFrame) -> dict[str, list[np.ndarray]]:
    """
    Constructs comparable prediction components.

    Required:
      target
      mu_obs
      L_queue
      S_mean

    Optional:
      L_effective
      multiplier
      mu_base
      additive_bg
    """
    target = df["target"].values

    mu_obs = df["mu_obs"].values
    L_queue = df["L_queue"].values
    S_mean = df["S_mean"].values

    raw_biology = [
        scale_by_smean(lq, s)
        for lq, s in zip(L_queue, S_mean)
    ]

    if "L_effective" in df.columns and df["L_effective"].notna().any():
        L_effective = df["L_effective"].values
        shifted_biology = [
            scale_by_smean(le, s)
            for le, s in zip(L_effective, S_mean)
        ]
    else:
        shifted_biology = raw_biology

    if "mu_base" in df.columns and df["mu_base"].notna().any():
        mu_base = df["mu_base"].values
    elif "multiplier" in df.columns and df["multiplier"].notna().any():
        multiplier = df["multiplier"].values
        mu_base = [
            multiply_arrays(sb, m)
            for sb, m in zip(shifted_biology, multiplier)
        ]
    else:
        mu_base = shifted_biology

    if "additive_bg" in df.columns and df["additive_bg"].notna().any():
        additive_bg = df["additive_bg"].values
    else:
        additive_bg = [np.zeros_like(np.asarray(m, dtype=np.float32)) for m in mu_obs]

    return {
        "target": target,
        "raw_biology": raw_biology,
        "shifted_biology": shifted_biology,
        "mu_base": mu_base,
        "mu_obs": mu_obs,
        "additive_bg": additive_bg,
    }


def compute_decomposition_metrics(df: pd.DataFrame) -> dict[str, float]:
    comps = get_component_lists(df)
    target = comps["target"]

    pcc_raw, raw_l, raw_u, n_raw = fisher_weighted_pcc(comps["raw_biology"], target)
    pcc_shifted, shifted_l, shifted_u, _ = fisher_weighted_pcc(comps["shifted_biology"], target)
    pcc_base, base_l, base_u, _ = fisher_weighted_pcc(comps["mu_base"], target)
    pcc_obs, obs_l, obs_u, _ = fisher_weighted_pcc(comps["mu_obs"], target)

    return {
        "pcc_raw_biology": pcc_raw,
        "pcc_shifted_biology": pcc_shifted,
        "pcc_mu_base": pcc_base,
        "pcc_mu_obs": pcc_obs,

        "ci_raw_lower": raw_l,
        "ci_raw_upper": raw_u,
        "ci_shifted_lower": shifted_l,
        "ci_shifted_upper": shifted_u,
        "ci_base_lower": base_l,
        "ci_base_upper": base_u,
        "ci_obs_lower": obs_l,
        "ci_obs_upper": obs_u,

        "shift_gain": pcc_shifted - pcc_raw,
        "b_gain": pcc_base - pcc_shifted,
        "additive_gain": pcc_obs - pcc_base,
        "total_gain": pcc_obs - pcc_raw,

        "n_valid": n_raw,
    }


# ============================================================
# Loading
# ============================================================

def load_prediction_folder(folder: Path) -> pd.DataFrame | None:
    parquet_files = sorted(glob.glob(str(folder / "comprehensive_predictions_rank*.parquet")))

    if not parquet_files:
        return None

    return pd.concat(
        [pd.read_parquet(f) for f in parquet_files],
        ignore_index=True,
    )


# ============================================================
# Main
# ============================================================

def main():
    base_path = Path("./riboai_queueing")
    mixed_folder_name = "eichhorn_2014_grimson_2019"

    dataset_encoding_path = Path("../Datasets/encodings/dataset_encoding.yaml")
    with dataset_encoding_path.open("r", encoding="utf-8") as f:
        dataset_encoding = yaml.safe_load(f)

    id2dataset = {int(v): str(k) for k, v in dataset_encoding.items()}

    # ------------------------------------------------------------
    # Individual runs
    # ------------------------------------------------------------
    indiv_metrics = {}

    for folder in base_path.iterdir():
        if not folder.is_dir():
            continue

        folder_name = folder.name

        if folder_name == mixed_folder_name:
            continue

        # Skip all-dataset aggregate folders if present.
        if folder_name.startswith("33_datasets_mix"):
            continue

        df = load_prediction_folder(folder)
        if df is None:
            continue

        if "target" not in df.columns or "mu_obs" not in df.columns:
            print(f"[WARN] Missing required columns in {folder_name}. Skipping.")
            continue

        indiv_metrics[folder_name] = compute_decomposition_metrics(df)

    # ------------------------------------------------------------
    # Mixed run split by dataset_id
    # ------------------------------------------------------------
    mixed_folder = base_path / mixed_folder_name
    df_mix = load_prediction_folder(mixed_folder)

    if df_mix is None:
        raise FileNotFoundError(f"No prediction parquet files found in {mixed_folder}")

    results = []

    for dataset_id in sorted(df_mix["dataset_id"].unique()):
        dataset_id = int(dataset_id)
        dataset_name = id2dataset.get(dataset_id, f"dataset_{dataset_id}")

        subset = df_mix[df_mix["dataset_id"] == dataset_id]
        mix_metrics = compute_decomposition_metrics(subset)

        indiv = indiv_metrics.get(dataset_name, {})

        row = {
            "dataset": dataset_name,

            "mix_pcc_raw_biology": mix_metrics["pcc_raw_biology"],
            "mix_pcc_shifted_biology": mix_metrics["pcc_shifted_biology"],
            "mix_pcc_mu_base": mix_metrics["pcc_mu_base"],
            "mix_pcc_mu_obs": mix_metrics["pcc_mu_obs"],

            "mix_shift_gain": mix_metrics["shift_gain"],
            "mix_b_gain": mix_metrics["b_gain"],
            "mix_additive_gain": mix_metrics["additive_gain"],
            "mix_total_gain": mix_metrics["total_gain"],
            "mix_n_valid": mix_metrics["n_valid"],

            "indiv_pcc_raw_biology": indiv.get("pcc_raw_biology", np.nan),
            "indiv_pcc_shifted_biology": indiv.get("pcc_shifted_biology", np.nan),
            "indiv_pcc_mu_base": indiv.get("pcc_mu_base", np.nan),
            "indiv_pcc_mu_obs": indiv.get("pcc_mu_obs", np.nan),

            "indiv_shift_gain": indiv.get("shift_gain", np.nan),
            "indiv_b_gain": indiv.get("b_gain", np.nan),
            "indiv_additive_gain": indiv.get("additive_gain", np.nan),
            "indiv_total_gain": indiv.get("total_gain", np.nan),
            "indiv_n_valid": indiv.get("n_valid", np.nan),
        }

        results.append(row)

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("mix_pcc_mu_obs", ascending=True).reset_index(drop=True)

    print(res_df)

    out_csv = base_path / f"decomposition_metrics_{mixed_folder_name}.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"Saved metrics to: {out_csv}")

    # ------------------------------------------------------------
    # Plot 1: final prediction quality
    # ------------------------------------------------------------
    y = np.arange(len(res_df))
    bar_width = 0.35

    plt.figure(figsize=(12, max(5, 0.5 * len(res_df))))

    plt.barh(
        y - bar_width / 2,
        res_df["indiv_pcc_mu_obs"],
        height=bar_width,
        color="lightsteelblue",
        edgecolor="black",
        label="Individual: PCC(mu_obs, target)",
    )

    plt.barh(
        y + bar_width / 2,
        res_df["mix_pcc_mu_obs"],
        height=bar_width,
        color="navy",
        edgecolor="black",
        label="Mixed: PCC(mu_obs, target)",
    )

    plt.axvline(0, color="black", linewidth=1)
    plt.yticks(y, res_df["dataset"])
    plt.xlabel("Fisher-Z aggregated per-transcript PCC")
    plt.title("Final prediction quality: individual vs mixed")
    plt.legend()
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # Plot 2: decomposition gains in the mixed model
    # ------------------------------------------------------------
    plt.figure(figsize=(12, max(5, 0.5 * len(res_df))))

    left = np.zeros(len(res_df))

    for col, color, label in [
        ("mix_shift_gain", "lightblue", "Shift gain"),
        ("mix_b_gain", "orange", "Multiplicative b gain"),
        ("mix_additive_gain", "darkred", "Additive gain"),
    ]:
        values = res_df[col].fillna(0.0).values

        plt.barh(
            y,
            values,
            left=left,
            color=color,
            edgecolor="black",
            label=label,
        )

        left = left + values

    plt.axvline(0, color="black", linewidth=1)
    plt.yticks(y, res_df["dataset"])
    plt.xlabel("PCC gain relative to previous component")
    plt.title(
        "Mixed-model component gains\n"
        "raw biology → shifted biology → mu_base → mu_obs"
    )
    plt.legend()
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # Plot 3: b gain individual vs mixed
    # ------------------------------------------------------------
    plt.figure(figsize=(12, max(5, 0.5 * len(res_df))))

    plt.barh(
        y - bar_width / 2,
        res_df["indiv_b_gain"],
        height=bar_width,
        color="lightsalmon",
        edgecolor="black",
        label="Individual b gain",
    )

    plt.barh(
        y + bar_width / 2,
        res_df["mix_b_gain"],
        height=bar_width,
        color="darkorange",
        edgecolor="black",
        label="Mixed b gain",
    )

    plt.axvline(0, color="black", linewidth=1)
    plt.yticks(y, res_df["dataset"])
    plt.xlabel("b gain = PCC(mu_base) - PCC(shifted_biology)")
    plt.title("Impact of multiplicative dataset/codon bias b")
    plt.legend()
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()