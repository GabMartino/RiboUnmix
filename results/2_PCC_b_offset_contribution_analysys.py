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

        targets = dataset_df["target"].values
        mu_obs = dataset_df["mu_obs"].values

        # Extract components to build mu_base
        l_queue = dataset_df["L_queue"].values
        total_scale = dataset_df["total_scale"].values
        b_offset = dataset_df["b_offset"].values

        # Reconstruct mu_base: L_queue * (total_scale / e^b)
        # Using a list comprehension to handle the element-wise array math
        mu_base = [l * (s / np.exp(b)) for l, s, b in zip(l_queue, total_scale, b_offset)]

        pcc_obs, _, _ = pcc_mu_obs_vs_y(mu_obs, targets)
        pcc_base, _, _ = pcc_mu_obs_vs_y(mu_base, targets)

        # Calculate Delta_b
        delta_b = pcc_obs - pcc_base

        indiv_metrics[folder_name] = {
            "indiv_delta_b": delta_b,
            "indiv_pcc_obs": pcc_obs,
            "indiv_pcc_base": pcc_base
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

        l_queue = subset["L_queue"].values
        S_mean = subset["S_mean"].values
        total_scale = subset["total_scale"].values
        b_offset = subset["b_offset"].values

        mu_base = [l * s for l, s in zip(l_queue, S_mean)]

        pcc_obs, _, _ = pcc_mu_obs_vs_y(mu_obs, targets)
        pcc_base, _, _ = pcc_mu_obs_vs_y(mu_base, targets)

        delta_b = pcc_obs - pcc_base

        indiv_data = indiv_metrics.get(dataset_name, {
            "indiv_delta_b": np.nan, "indiv_pcc_obs": np.nan, "indiv_pcc_base": np.nan
        })

        results.append({
            "dataset": dataset_name,
            "mix_delta_b": delta_b,
            "mix_pcc_obs": pcc_obs,
            "mix_pcc_base": pcc_base,
            **indiv_data
        })

    res_df = pd.DataFrame(results)
    res_df = res_df.dropna(subset=['mix_delta_b']).sort_values("mix_delta_b", ascending=True).reset_index(drop=True)

    # ---------------------------------------------------------
    # 3. Plotting Setup
    # ---------------------------------------------------------
    y_indices = np.arange(len(res_df))
    bar_width = 0.35

    plt.figure(figsize=(14, 12))

    # Plot Individual Delta_b
    plt.barh(y_indices - bar_width / 2, res_df["indiv_delta_b"], height=bar_width,
             color='lightcoral', edgecolor='black', label='Individual Model $\Delta_b$')

    # Plot Mix Delta_b
    plt.barh(y_indices + bar_width / 2, res_df["mix_delta_b"], height=bar_width,
             color='darkred', edgecolor='black', label='Mix Model $\Delta_b$')

    # Formatting
    plt.axvline(0, color='black', linewidth=1.5, linestyle='-')
    plt.yticks(y_indices, res_df["dataset"])
    plt.xlabel(r"Gain in Correlation ($\Delta_b = PCC_{\mu_{obs}} - PCC_{\mu_{base}}$)")

    plt.title(
        r"Impact of `b_offset` on Profile Reconstruction ($\Delta_b$)" + "\n" +
        r"Positive = Improvement | Near Zero = No Effect | Negative = Overfitting/Hurting",
        pad=15, fontsize=14
    )
    plt.legend(loc='lower right', framealpha=0.9)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()