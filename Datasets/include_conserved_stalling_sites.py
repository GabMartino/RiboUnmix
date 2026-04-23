"""
Include conserved stalling sites + compute pausing sites by z-score thresholds.

- Robust against:
  - all-zero ribo vectors
  - constant ribo vectors (std ~ 0)
  - very short non-zero subsets
  - NaNs / inf in ribo
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass(frozen=True)
class ZScoreConfig:
    thresholds: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0)
    eps: float = 1e-8
    min_count: int = 2  # need at least 2 points to define a std meaningfully


def safe_zscore(x: np.ndarray, *, ignore_zeros: bool, cfg: ZScoreConfig) -> np.ndarray:
    """
    Return z-scored array. If not possible (empty/constant), return an array of NaNs.
    """
    x = np.asarray(x, dtype=np.float32)

    # drop non-finite values from stats computation (but keep shape for return)
    finite_mask = np.isfinite(x)

    if ignore_zeros:
        stat_mask = finite_mask & (x > 0)
    else:
        stat_mask = finite_mask

    vals = x[stat_mask]
    if vals.size < cfg.min_count:
        return np.full_like(x, np.nan, dtype=np.float32)

    mu = float(vals.mean())
    sd = float(vals.std(ddof=0))

    if not np.isfinite(sd) or sd < cfg.eps:
        return np.full_like(x, np.nan, dtype=np.float32)

    z = (x - mu) / sd
    # keep non-finite original positions as NaN
    z[~finite_mask] = np.nan
    return z


def pausing_sites_from_z(z: np.ndarray, thresholds: Iterable[float]) -> dict[str, list[int] | None]:
    """
    Compute pausing sites indices for each threshold.
    If z is all NaN, return None for all thresholds.
    """
    out: dict[str, list[int] | None] = {}
    if np.all(~np.isfinite(z)):
        for t in thresholds:
            out[f"pausing_sites_std_{int(t)}"] = None
        return out

    for t in thresholds:
        idxs = np.flatnonzero(z >= t).tolist()
        out[f"pausing_sites_std_{int(t)}"] = idxs if idxs else None
    return out


def compute_pausing_columns(ribo_series: pd.Series, cfg: ZScoreConfig) -> pd.DataFrame:
    """
    Compute 8 columns:
      pausing_sites_std_{2,3,4,5}
      pausing_sites_std_{2,3,4,5}_non_zeros
    """
    std_cols = []
    std_nz_cols = []

    ribos = ribo_series.to_list()

    for ribo in tqdm(ribos, desc="Computing pausing sites", leave=False):
        z_all = safe_zscore(ribo, ignore_zeros=False, cfg=cfg)
        z_nz = safe_zscore(ribo, ignore_zeros=True, cfg=cfg)

        d_all = pausing_sites_from_z(z_all, cfg.thresholds)
        d_nz = pausing_sites_from_z(z_nz, cfg.thresholds)

        std_cols.append(d_all)
        std_nz_cols.append({f"{k}_non_zeros": v for k, v in d_nz.items()})

    df_all = pd.DataFrame(std_cols)
    df_nz = pd.DataFrame(std_nz_cols)
    return pd.concat([df_all, df_nz], axis=1)


def main() -> None:
    css_path = "conserved_stalling_sites/stalling_sites.parquet"
    base_path = "data/raw_datasets"
    out_path = base_path + "_with_css"

    cfg = ZScoreConfig()

    conserved = pd.read_parquet(css_path)

    datasets_paths = sorted(str(p) for p in pathlib.Path(base_path).glob("*.parquet"))
    os.makedirs(out_path, exist_ok=True)
    datasets_paths = [d for d in datasets_paths]
    for path in tqdm(datasets_paths, desc="Datasets"):
        dataset_name = pathlib.Path(path).stem
        data = pd.read_parquet(path)

        # normalize transcript id
        data = data.copy()
        data["transcript_id"] = data["id"].astype(str).str.split(".").str[0]

        # Keep all dataset rows; attach conserved info if present
        # (left join from data to conserved)
        merged = data.merge(conserved, on="transcript_id", how="left", suffixes=("", "_css"))

        # compute pausing sites columns from ribo
        pausing_df = compute_pausing_columns(merged["ribo"], cfg)
        merged = pd.concat([merged, pausing_df], axis=1)

        merged.to_parquet(
            os.path.join(out_path, f"{dataset_name}.parquet"),
            compression="gzip",
            engine="pyarrow",
            index=False,
        )


if __name__ == "__main__":
    main()