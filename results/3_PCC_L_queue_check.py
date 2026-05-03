import glob
import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, norm


def pcc_mu_obs_vs_y(mu_obs_list, y_target_list, alpha=0.05):
    """
    Computes the weighted mean Pearson Correlation Coefficient and its
    confidence interval across multiple independent samples using Fisher Z.
    """
    pcc_s = []
    n_s = []

    for mu_obs, y in zip(mu_obs_list, y_target_list):
        r = pearsonr(mu_obs, y)[0]
        n = len(mu_obs)
        if n > 3 and not np.isnan(r):
            pcc_s.append(r)
            n_s.append(n)

    if not pcc_s:
        return np.nan, np.nan, np.nan

    pcc_array = np.array(pcc_s)
    n_array = np.array(n_s)

    r_clipped = np.clip(pcc_array, -0.9999, 0.9999)
    z_scores = np.arctanh(r_clipped)

    weights = n_array - 3
    z_mean = np.average(z_scores, weights=weights)

    se_z_mean = 1 / np.sqrt(np.sum(weights))
    z_critical = norm.ppf(1 - alpha / 2)

    z_ci_lower = z_mean - z_critical * se_z_mean
    z_ci_upper = z_mean + z_critical * se_z_mean

    mean_pcc = np.tanh(z_mean)
    ci_lower = np.tanh(z_ci_lower)
    ci_upper = np.tanh(z_ci_upper)

    return mean_pcc, ci_lower, ci_upper


def main():
    base_path = "./riboai_queueing/"

    # ---------------------------------------------------------
    # 1. Process Mixed Dataset
    # ---------------------------------------------------------
    path_mix = os.path.join(base_path, "33_datasets_mix_6e5e33", "comprehensive_predictions_rank*.parquet")
    dataset_encoding = yaml.load(open("../Datasets/encodings/dataset_encoding.yaml"), Loader=yaml.FullLoader)
    id2dataset = {id: name for name, id in dataset_encoding.items()}

    parquet_files_mix = glob.glob(path_mix)
    df_mix = pd.concat([pd.read_parquet(f) for f in parquet_files_mix], ignore_index=True)
    datasets_id = df_mix["dataset_id"].unique()

    results = []
    for d_id in datasets_id:
        dataset_name = id2dataset[d_id]
        subset = df_mix.loc[df_mix["dataset_id"] == d_id]

        targets = subset["target"].values
        l_queue = subset["L_queue"].values
        total_scale = subset["total_scale"].values
        b_offset = subset["b_offset"].values

        # FIXED: Vectorized zero-division handling for arrays of arrays
        with np.errstate(divide='ignore', invalid='ignore'):
            target_scaled_L = [
                np.where(s != 0, y / s, np.nan)
                for y, s in zip(targets, total_scale)
            ]

            target_scaled_L_base = [
                np.where(s != 0, y / (s / np.exp(b)), np.nan)
                for y, s, b in zip(targets, total_scale, b_offset)
            ]

        pcc_L, ci_l_L, ci_u_L = pcc_mu_obs_vs_y(l_queue, target_scaled_L)
        pcc_L_base, ci_l_L_base, ci_u_L_base = pcc_mu_obs_vs_y(l_queue, target_scaled_L_base)

        results.append({
            "dataset": dataset_name,
            "mix_pcc_L": pcc_L,
            "mix_ci_L_lower": ci_l_L,
            "mix_ci_L_upper": ci_u_L,
            "mix_pcc_L_base": pcc_L_base,
            "mix_ci_L_base_lower": ci_l_L_base,
            "mix_ci_L_base_upper": ci_u_L_base
        })

    res_df = pd.DataFrame(results)
    res_df = res_df.dropna(subset=['mix_pcc_L']).sort_values("mix_pcc_L", ascending=True).reset_index(drop=True)

    # ---------------------------------------------------------
    # 2. Plotting Setup (2 Bars per Dataset)
    # ---------------------------------------------------------
    def calc_err(mean_col, lower_col, upper_col):
        return np.array([
            res_df[mean_col] - res_df[lower_col],
            res_df[upper_col] - res_df[mean_col]
        ])

    err_mix_L = calc_err("mix_pcc_L", "mix_ci_L_lower", "mix_ci_L_upper")
    err_mix_L_base = calc_err("mix_pcc_L_base", "mix_ci_L_base_lower", "mix_ci_L_base_upper")

    y_indices = np.arange(len(res_df))
    bar_width = 0.35

    plt.figure(figsize=(12, 12))

    plt.barh(y_indices - bar_width / 2, res_df["mix_pcc_L_base"], height=bar_width,
             xerr=err_mix_L_base, capsize=3, color='lightsalmon', edgecolor='black',
             label=r'Base Target: $PCC_{L\_base\_target}$')

    plt.barh(y_indices + bar_width / 2, res_df["mix_pcc_L"], height=bar_width,
             xerr=err_mix_L, capsize=3, color='steelblue', edgecolor='black',
             label=r'Scaled Target: $PCC_L$')

    plt.axvline(0, color='black', linewidth=1)
    plt.yticks(y_indices, res_df["dataset"])
    plt.xlabel("Pearson Correlation Coefficient (PCC)")

    plt.title(
        r"Biological Reconstruction Check: $PCC_L$ vs $PCC_{L\_base\_target}$" + "\n" +
        r"$PCC_L = PCC(L_{queue}, \frac{y}{total\_scale})$  |  $PCC_{L\_base\_target} = PCC(L_{queue}, \frac{y}{S})$",
        pad=15, fontsize=14
    )
    plt.legend(loc='lower right', framealpha=0.9)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()