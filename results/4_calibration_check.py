import glob
import os
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


EPS = 1e-8


def safe_array(x, dtype=np.float32):
    return np.asarray(x, dtype=dtype).reshape(-1)


def compute_mu_total_from_median(mu_obs, pi, sigma):
    """
    Positive-component lognormal median = mu_obs.

    mean_positive = mu_obs * exp(sigma^2 / 2)
    mu_total = (1 - pi) * mean_positive
    """
    mu_obs = np.maximum(safe_array(mu_obs), EPS)
    pi = np.clip(safe_array(pi), 0.0, 1.0)
    sigma = np.maximum(safe_array(sigma), 0.0)

    L = min(len(mu_obs), len(pi), len(sigma))
    mu_obs = mu_obs[:L]
    pi = pi[:L]
    sigma = sigma[:L]

    sigma = np.clip(sigma, 0.0, 10.0)
    return (1.0 - pi) * mu_obs * np.exp(0.5 * sigma * sigma)


def compute_ziln_variance_from_median(mu_obs, pi, sigma):
    """
    Total variance of zero-inflated lognormal.

    Positive component:
        mean_pos = mu_obs * exp(sigma^2 / 2)
        var_pos  = (exp(sigma^2) - 1) * mu_obs^2 * exp(sigma^2)

    Zero-inflated total:
        Var[Y] = (1 - pi) * var_pos + pi * (1 - pi) * mean_pos^2
    """
    mu_obs = np.maximum(safe_array(mu_obs), EPS)
    pi = np.clip(safe_array(pi), 0.0, 1.0)
    sigma = np.maximum(safe_array(sigma), 0.0)

    L = min(len(mu_obs), len(pi), len(sigma))
    mu_obs = mu_obs[:L]
    pi = pi[:L]
    sigma = sigma[:L]

    sigma = np.clip(sigma, 0.0, 10.0)
    s2 = sigma * sigma

    mean_pos = mu_obs * np.exp(0.5 * s2)
    var_pos = (np.exp(s2) - 1.0) * (mu_obs ** 2) * np.exp(s2)

    var_total = (1.0 - pi) * var_pos + pi * (1.0 - pi) * (mean_pos ** 2)
    return np.maximum(var_total, EPS)


def compute_pi_ece(pi_values, zero_values, n_bins=10):
    """
    Expected calibration error for zero-inflation probability.

        ECE_pi = sum_b (n_b / N) * |mean(pi)_b - observed_zero_rate_b|
    """
    pi_values = np.clip(np.asarray(pi_values, dtype=np.float64), 0.0, 1.0)
    zero_values = np.asarray(zero_values, dtype=np.float64)

    valid = np.isfinite(pi_values) & np.isfinite(zero_values)
    pi_values = pi_values[valid]
    zero_values = zero_values[valid]

    if len(pi_values) == 0:
        return np.nan

    if np.unique(pi_values).size < 2:
        return float(abs(np.mean(pi_values) - np.mean(zero_values)))

    df = pd.DataFrame({"pi": pi_values, "is_zero": zero_values})

    try:
        df["bin"] = pd.qcut(df["pi"], q=n_bins, duplicates="drop")
    except Exception:
        return float(abs(df["pi"].mean() - df["is_zero"].mean()))

    bin_df = (
        df.groupby("bin", observed=True)
        .agg(
            n=("is_zero", "size"),
            mean_pi=("pi", "mean"),
            observed_zero_rate=("is_zero", "mean"),
        )
        .reset_index()
    )

    N = bin_df["n"].sum()
    if N <= 0:
        return np.nan

    ece = np.sum(
        (bin_df["n"] / N)
        * np.abs(bin_df["mean_pi"] - bin_df["observed_zero_rate"])
    )

    return float(ece)


def compute_dataset_calibration_metrics(subset, n_bins=10):
    """
    Computes pi_ece, pi_brier, and median_z2 for one dataset.
    """
    all_pi = []
    all_zero = []
    all_z2 = []

    mu_col = "mu_obs" if "mu_obs" in subset.columns else "mu"

    has_mu_total = "mu_total" in subset.columns

    for _, row in subset.iterrows():
        L = int(row["length"])

        y = safe_array(row["target"])[:L]
        mu_obs = safe_array(row[mu_col])[:L]
        pi = safe_array(row["pi"])[:L]
        sigma = safe_array(row["sigma"])[:L]

        L_eff = min(len(y), len(mu_obs), len(pi), len(sigma))
        if L_eff == 0:
            continue

        y = y[:L_eff]
        mu_obs = mu_obs[:L_eff]
        pi = np.clip(pi[:L_eff], 0.0, 1.0)
        sigma = np.maximum(sigma[:L_eff], 0.0)

        if has_mu_total:
            mu_total = safe_array(row["mu_total"])[:L_eff]
        else:
            mu_total = compute_mu_total_from_median(mu_obs, pi, sigma)

        pred_var = compute_ziln_variance_from_median(mu_obs, pi, sigma)

        is_zero = (y <= 0.0).astype(np.float32)
        z2 = ((y - mu_total) ** 2) / np.maximum(pred_var, EPS)

        all_pi.append(pi)
        all_zero.append(is_zero)
        all_z2.append(z2)

    if not all_pi:
        return {
            "pi_ece": np.nan,
            "pi_brier": np.nan,
            "median_z2": np.nan,
            "n_positions": 0,
        }

    pi_all = np.concatenate(all_pi)
    zero_all = np.concatenate(all_zero)
    z2_all = np.concatenate(all_z2)

    valid_pi = np.isfinite(pi_all) & np.isfinite(zero_all)
    pi_all = pi_all[valid_pi]
    zero_all = zero_all[valid_pi]

    valid_z2 = np.isfinite(z2_all)
    z2_all = z2_all[valid_z2]

    pi_ece = compute_pi_ece(pi_all, zero_all, n_bins=n_bins)
    pi_brier = float(np.mean((pi_all - zero_all) ** 2)) if len(pi_all) else np.nan
    median_z2 = float(np.median(z2_all)) if len(z2_all) else np.nan

    return {
        "pi_ece": pi_ece,
        "pi_brier": pi_brier,
        "median_z2": median_z2,
        "n_positions": int(len(pi_all)),
    }


def plot_calibration_metrics(res_df, out_path):
    """
    Makes one figure with three horizontal bar plots.
    """
    res_df = res_df.copy()

    # Sort by pi_ece because this is the clearest calibration error.
    res_df = res_df.sort_values("pi_ece", ascending=False).reset_index(drop=True)

    y = np.arange(len(res_df))

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(18, max(8, 0.35 * len(res_df))),
        sharey=True,
    )

    # ---------------------------------------------------------
    # pi_ece
    # ---------------------------------------------------------
    ax = axes[0]
    ax.barh(y, res_df["pi_ece"], color="steelblue", edgecolor="black")
    ax.set_yticks(y)
    ax.set_yticklabels(res_df["dataset"])
    ax.invert_yaxis()
    ax.set_xlabel(r"$ECE_{\pi}$")
    ax.set_title(
        "Zero-inflation calibration error\n"
        r"$ECE_{\pi}=\sum_b \frac{n_b}{N}|\overline{\pi}_b-\widehat{P}(y=0)_b|$"
    )
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # ---------------------------------------------------------
    # pi_brier
    # ---------------------------------------------------------
    ax = axes[1]
    ax.barh(y, res_df["pi_brier"], color="darkorange", edgecolor="black")
    ax.set_xlabel(r"Brier score for zero prediction")
    ax.set_title(
        "Zero-inflation Brier score\n"
        r"$\frac{1}{N}\sum_i(\pi_i-\mathbf{1}[y_i=0])^2$"
    )
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # ---------------------------------------------------------
    # median_z2
    # ---------------------------------------------------------
    ax = axes[2]
    ax.barh(y, res_df["median_z2"], color="seagreen", edgecolor="black")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.2, label="ideal = 1")
    ax.set_xlabel(r"Median $z^2$")
    ax.set_title(
        "Uncertainty calibration\n"
        r"$z_i^2=\frac{(y_i-\mu_{total,i})^2}{Var_{pred}(Y_i)}$"
    )
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    fig.suptitle(
        "Zero-inflation and uncertainty calibration by dataset\n"
        "Lower is better for ECE and Brier. Median z² should be close to 1.",
        fontsize=15,
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    base_path = "./riboai_queueing/"
    run_name = "33_datasets_mix_6e5e33"
    n_bins = 10

    prediction_glob = os.path.join(
        base_path,
        run_name,
        "comprehensive_predictions_rank*.parquet",
    )

    dataset_encoding_path = "../Datasets/encodings/dataset_encoding.yaml"
    dataset_encoding = yaml.load(open(dataset_encoding_path), Loader=yaml.FullLoader)
    id2dataset = {int(v): k for k, v in dataset_encoding.items()}

    parquet_files = glob.glob(prediction_glob)
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found at: {prediction_glob}")

    print(f"Loading {len(parquet_files)} parquet files...")
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

    if "transcripts_id" in df.columns and "transcript_id" not in df.columns:
        df = df.rename(columns={"transcripts_id": "transcript_id"})

    required_cols = {"dataset_id", "transcript_id", "length", "target", "pi", "sigma"}
    if "mu_obs" not in df.columns and "mu" not in df.columns:
        required_cols.add("mu_obs")

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    before = len(df)
    df = df.drop_duplicates(subset=["dataset_id", "transcript_id"]).reset_index(drop=True)
    print(f"Dropped {before - len(df)} duplicated DDP rows.")

    results = []

    for d_id in sorted(df["dataset_id"].unique()):
        dataset_name = id2dataset.get(int(d_id), f"dataset_{d_id}")
        subset = df[df["dataset_id"] == d_id]

        print(f"Computing calibration metrics for {dataset_name} ({len(subset)} transcripts)...")

        metrics = compute_dataset_calibration_metrics(subset, n_bins=n_bins)

        results.append(
            {
                "dataset_id": int(d_id),
                "dataset": dataset_name,
                **metrics,
            }
        )

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("pi_ece", ascending=False).reset_index(drop=True)

    out_dir = "calibration_diagnostics"
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "pi_sigma_calibration_metrics.csv")
    plot_path = os.path.join(out_dir, "pi_sigma_calibration_metrics.png")

    res_df.to_csv(csv_path, index=False)

    print("\nSaved table:")
    print(csv_path)

    print("\nCalibration summary:")
    print(res_df.to_string(index=False))

    plot_calibration_metrics(res_df, plot_path)

    print("\nSaved plot:")
    print(plot_path)


if __name__ == "__main__":
    main()