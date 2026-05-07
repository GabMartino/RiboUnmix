from __future__ import annotations

import glob
import os
import yaml

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def safe_pcc(x, y, eps: float = 1e-12) -> float:
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

        r = safe_pcc(pred[:L], target[:L])
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


def scalar_smean(s) -> float:
    return float(np.asarray(s).reshape(-1)[0])


def divide_by_smean(y, s, eps: float = 1e-8):
    return np.asarray(y, dtype=np.float32) / max(scalar_smean(s), eps)


def debiased_effective_target(y, s, multiplier, additive_bg=None, eps: float = 1e-8):
    y = np.asarray(y, dtype=np.float32)
    multiplier = np.asarray(multiplier, dtype=np.float32)

    if additive_bg is not None:
        y = y - np.asarray(additive_bg, dtype=np.float32)
        y = np.clip(y, 0.0, None)

    S = max(scalar_smean(s), eps)
    denom = S * np.clip(multiplier, eps, None)

    return y / denom


def calc_err(df, mean_col, lower_col, upper_col):
    return np.asarray([
        df[mean_col] - df[lower_col],
        df[upper_col] - df[mean_col],
    ])


def main():
    base_path = "./riboai_queueing/"
    mix_folder = "eichhorn_2014_grimson_2019"

    path_mix = os.path.join(
        base_path,
        mix_folder,
        "comprehensive_predictions_rank*.parquet",
    )

    dataset_encoding = yaml.load(
        open("../Datasets/encodings/dataset_encoding.yaml"),
        Loader=yaml.FullLoader,
    )
    id2dataset = {int(v): str(k) for k, v in dataset_encoding.items()}

    parquet_files_mix = glob.glob(path_mix)
    if not parquet_files_mix:
        raise FileNotFoundError(f"No parquet files found at: {path_mix}")

    df_mix = pd.concat(
        [pd.read_parquet(f) for f in parquet_files_mix],
        ignore_index=True,
    )

    required = {"dataset_id", "target", "S_mean", "L_queue", "mu_obs"}
    missing = required - set(df_mix.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    results = []

    for d_id in sorted(df_mix["dataset_id"].unique()):
        dataset_name = id2dataset.get(int(d_id), f"dataset_{int(d_id)}")
        subset = df_mix.loc[df_mix["dataset_id"] == d_id]

        targets = subset["target"].values
        S_mean = subset["S_mean"].values
        L_queue = subset["L_queue"].values
        mu_obs = subset["mu_obs"].values

        target_over_S = [
            divide_by_smean(y, s)
            for y, s in zip(targets, S_mean)
        ]

        pcc_L_queue, ci_l_L_queue, ci_u_L_queue, n_L_queue = fisher_weighted_pcc(
            L_queue,
            target_over_S,
        )

        pcc_mu_obs, ci_l_mu_obs, ci_u_mu_obs, n_mu_obs = fisher_weighted_pcc(
            mu_obs,
            targets,
        )

        row = {
            "dataset": dataset_name,

            "pcc_L_queue_vs_y_over_S": pcc_L_queue,
            "ci_L_queue_lower": ci_l_L_queue,
            "ci_L_queue_upper": ci_u_L_queue,

            "pcc_mu_obs_vs_y": pcc_mu_obs,
            "ci_mu_obs_lower": ci_l_mu_obs,
            "ci_mu_obs_upper": ci_u_mu_obs,

            "n_valid_L_queue": n_L_queue,
            "n_valid_mu_obs": n_mu_obs,
        }

        if "L_effective" in subset.columns:
            L_effective = subset["L_effective"].values

            pcc_L_eff, ci_l_L_eff, ci_u_L_eff, n_L_eff = fisher_weighted_pcc(
                L_effective,
                target_over_S,
            )

            row.update({
                "pcc_L_effective_vs_y_over_S": pcc_L_eff,
                "ci_L_effective_lower": ci_l_L_eff,
                "ci_L_effective_upper": ci_u_L_eff,
                "n_valid_L_effective": n_L_eff,
            })

        if (
            "L_effective" in subset.columns
            and "multiplier" in subset.columns
        ):
            L_effective = subset["L_effective"].values
            multiplier = subset["multiplier"].values

            additive_bg = (
                subset["additive_bg"].values
                if "additive_bg" in subset.columns
                else [None] * len(subset)
            )

            target_debiased = [
                debiased_effective_target(y, s, m, a)
                for y, s, m, a in zip(targets, S_mean, multiplier, additive_bg)
            ]

            pcc_debiased, ci_l_deb, ci_u_deb, n_deb = fisher_weighted_pcc(
                L_effective,
                target_debiased,
            )

            row.update({
                "pcc_L_effective_vs_debiased_target": pcc_debiased,
                "ci_debiased_lower": ci_l_deb,
                "ci_debiased_upper": ci_u_deb,
                "n_valid_debiased": n_deb,
            })

        results.append(row)

    res_df = pd.DataFrame(results)

    sort_col = (
        "pcc_L_effective_vs_debiased_target"
        if "pcc_L_effective_vs_debiased_target" in res_df.columns
        else "pcc_L_queue_vs_y_over_S"
    )

    res_df = (
        res_df
        .dropna(subset=["pcc_L_queue_vs_y_over_S"])
        .sort_values(sort_col, ascending=True)
        .reset_index(drop=True)
    )

    print(res_df)

    out_csv = os.path.join(base_path, f"biological_reconstruction_{mix_folder}.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"Saved metrics to: {out_csv}")

    y_indices = np.arange(len(res_df))
    bar_width = 0.25

    plt.figure(figsize=(14, max(8, 0.45 * len(res_df))))

    err_L_queue = calc_err(
        res_df,
        "pcc_L_queue_vs_y_over_S",
        "ci_L_queue_lower",
        "ci_L_queue_upper",
    )

    plt.barh(
        y_indices - bar_width,
        res_df["pcc_L_queue_vs_y_over_S"],
        height=bar_width,
        xerr=err_L_queue,
        capsize=3,
        color="lightsteelblue",
        edgecolor="black",
        label=r"$PCC(L_{queue}, y/S)$",
    )

    if "pcc_L_effective_vs_y_over_S" in res_df.columns:
        err_L_eff = calc_err(
            res_df,
            "pcc_L_effective_vs_y_over_S",
            "ci_L_effective_lower",
            "ci_L_effective_upper",
        )

        plt.barh(
            y_indices,
            res_df["pcc_L_effective_vs_y_over_S"],
            height=bar_width,
            xerr=err_L_eff,
            capsize=3,
            color="steelblue",
            edgecolor="black",
            label=r"$PCC(L_{effective}, y/S)$",
        )

    if "pcc_L_effective_vs_debiased_target" in res_df.columns:
        err_deb = calc_err(
            res_df,
            "pcc_L_effective_vs_debiased_target",
            "ci_debiased_lower",
            "ci_debiased_upper",
        )

        plt.barh(
            y_indices + bar_width,
            res_df["pcc_L_effective_vs_debiased_target"],
            height=bar_width,
            xerr=err_deb,
            capsize=3,
            color="darkgreen",
            edgecolor="black",
            label=r"$PCC(L_{effective}, (y-A)/(S \cdot multiplier))$",
        )

    plt.axvline(0, color="black", linewidth=1)
    plt.yticks(y_indices, res_df["dataset"])
    plt.xlabel("Fisher-Z aggregated per-transcript PCC")

    plt.title(
        "Biological reconstruction diagnostics\n"
        "Raw target scaling vs shift-corrected and de-biased biological target",
        pad=15,
        fontsize=14,
    )

    plt.legend(loc="lower right", framealpha=0.9)
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()