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

    # Step 1: Collect valid correlations and sample sizes
    for mu_obs, y in zip(mu_obs_list, y_target_list):
        r = pearsonr(mu_obs, y)[0]
        n = len(mu_obs)
        # Ensure mathematical validity for Fisher Z
        if n > 3 and not np.isnan(r):
            pcc_s.append(r)
            n_s.append(n)

    if not pcc_s:
        return np.nan, np.nan, np.nan

    pcc_array = np.array(pcc_s)
    n_array = np.array(n_s)

    # Step 2: Transform to Fisher z-space
    r_clipped = np.clip(pcc_array, -0.9999, 0.9999)
    z_scores = np.arctanh(r_clipped)

    # Step 3: Compute weighted mean of z-scores
    weights = n_array - 3
    z_mean = np.average(z_scores, weights=weights)

    # Step 4: Compute Standard Error and CI in z-space
    se_z_mean = 1 / np.sqrt(np.sum(weights))
    z_critical = norm.ppf(1 - alpha / 2)

    z_ci_lower = z_mean - z_critical * se_z_mean
    z_ci_upper = z_mean + z_critical * se_z_mean

    # Step 5: Transform back to r-space
    mean_pcc = np.tanh(z_mean)
    ci_lower = np.tanh(z_ci_lower)
    ci_upper = np.tanh(z_ci_upper)

    return mean_pcc, ci_lower, ci_upper


def main():
    base_path = "./riboai_queueing/"

    # ---------------------------------------------------------
    # 1. Process Individual Datasets
    # ---------------------------------------------------------
    datasets_results = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if "33_datasets_mix_6e5e33" in datasets_results:
        datasets_results.remove("33_datasets_mix_6e5e33")

    indiv_metrics = {}
    for folder_name in datasets_results:
        base_path_d = os.path.join(base_path, folder_name, "comprehensive_predictions_rank*.parquet")
        parquet_files = glob.glob(base_path_d)

        if not parquet_files:
            continue

        dataset_df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

        # Because these are columns of arrays, we can pass them directly to our Fisher Z function
        targets = dataset_df["target"].values
        mu_obs = dataset_df["mu_obs"].values
        mu_total = dataset_df["mu_total"].values

        pcc_obs, ci_l_obs, ci_u_obs = pcc_mu_obs_vs_y(mu_obs, targets)
        pcc_total, ci_l_total, ci_u_total = pcc_mu_obs_vs_y(mu_total, targets)

        indiv_metrics[folder_name] = {
            "indiv_pcc_total": pcc_total,
            "indiv_ci_total_lower": ci_l_total,
            "indiv_ci_total_upper": ci_u_total,
            "indiv_pcc_obs": pcc_obs,
            "indiv_ci_obs_lower": ci_l_obs,
            "indiv_ci_obs_upper": ci_u_obs
        }

    # ---------------------------------------------------------
    # 2. Process Mixed Dataset
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
        mu_obs = subset["mu_obs"].values
        mu_total = subset["mu_total"].values

        pcc_obs, ci_l_obs, ci_u_obs = pcc_mu_obs_vs_y(mu_obs, targets)
        pcc_total, ci_l_total, ci_u_total = pcc_mu_obs_vs_y(mu_total, targets)

        # Merge with individual metrics.
        # Note: Ensure the folder names in 'indiv_metrics' match 'dataset_name' exactly.
        indiv_data = indiv_metrics.get(dataset_name, {
            "indiv_pcc_total": np.nan, "indiv_ci_total_lower": np.nan, "indiv_ci_total_upper": np.nan,
            "indiv_pcc_obs": np.nan, "indiv_ci_obs_lower": np.nan, "indiv_ci_obs_upper": np.nan
        })

        results.append({
            "dataset": dataset_name,
            # Mixed Metrics
            "mix_pcc_total": pcc_total,
            "mix_ci_total_lower": ci_l_total,
            "mix_ci_total_upper": ci_u_total,
            "mix_pcc_obs": pcc_obs,
            "mix_ci_obs_lower": ci_l_obs,
            "mix_ci_obs_upper": ci_u_obs,
            # Individual Metrics
            **indiv_data
        })

    res_df = pd.DataFrame(results)
    # Drop rows where the mixed dataset correlation failed to compute, and sort
    res_df = res_df.dropna(subset=['mix_pcc_total']).sort_values("mix_pcc_total", ascending=True).reset_index(drop=True)

    # ---------------------------------------------------------
    # 3. Plotting Setup (4 Bars per Dataset)
    # ---------------------------------------------------------
    def calc_err(mean_col, lower_col, upper_col):
        return np.array([
            res_df[mean_col] - res_df[lower_col],
            res_df[upper_col] - res_df[mean_col]
        ])

    err_mix_total = calc_err("mix_pcc_total", "mix_ci_total_lower", "mix_ci_total_upper")
    err_mix_obs = calc_err("mix_pcc_obs", "mix_ci_obs_lower", "mix_ci_obs_upper")
    err_indiv_total = calc_err("indiv_pcc_total", "indiv_ci_total_lower", "indiv_ci_total_upper")
    err_indiv_obs = calc_err("indiv_pcc_obs", "indiv_ci_obs_lower", "indiv_ci_obs_upper")

    y_indices = np.arange(len(res_df))
    bar_width = 0.2

    plt.figure(figsize=(14, 16))

    # Plot 1: Individual Obs
    plt.barh(y_indices - bar_width * 1.5, res_df["indiv_pcc_obs"], height=bar_width,
             xerr=err_indiv_obs, capsize=2, color='lightsalmon', edgecolor='black', label='Indiv: $\mu_{obs}$')
    # Plot 2: Individual Total
    plt.barh(y_indices - bar_width * 0.5, res_df["indiv_pcc_total"], height=bar_width,
             xerr=err_indiv_total, capsize=2, color='lightblue', edgecolor='black', label='Indiv: $\mu_{total}$')
    # Plot 3: Mix Obs
    plt.barh(y_indices + bar_width * 0.5, res_df["mix_pcc_obs"], height=bar_width,
             xerr=err_mix_obs, capsize=2, color='darkred', edgecolor='black', label='Mix: $\mu_{obs}$')
    # Plot 4: Mix Total
    plt.barh(y_indices + bar_width * 1.5, res_df["mix_pcc_total"], height=bar_width,
             xerr=err_mix_total, capsize=2, color='navy', edgecolor='black', label='Mix: $\mu_{total}$')

    plt.axvline(0, color='black', linewidth=1)
    plt.yticks(y_indices, res_df["dataset"])
    plt.xlabel("Pearson Correlation Coefficient (PCC)")

    plt.title(
        r"Comparison of Individual vs Mixed Dataset Predictions" + "\n" +
        r"Where: $\mu_{total} = (1 - \pi)\mu_{obs}e^{\sigma^2/2}$",
        pad=15, fontsize=14
    )
    plt.legend(loc='lower right', framealpha=0.9)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()