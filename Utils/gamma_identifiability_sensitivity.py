from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


SEQUENCE_COLUMNS = (
    "mu",
    "L_bio",
    "gamma",
    "log_gamma",
    "additive_bias",
)


def _as_float_array(value) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.float64)
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False).reshape(-1)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float64).reshape(-1)
    return np.asarray([value], dtype=np.float64)


def _safe_pcc(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2:
        return 0.0
    a = a[valid] - a[valid].mean()
    b = b[valid] - b[valid].mean()
    denom = np.sqrt(np.square(a).sum() * np.square(b).sum())
    if denom <= 0.0:
        return 0.0
    return float((a * b).sum() / denom)


def _relative_l1(a: np.ndarray, b: np.ndarray, eps: float = 1.0e-8) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if not valid.any():
        return 0.0
    return float(np.abs(a[valid] - b[valid]).sum() / (np.abs(a[valid]).sum() + eps))


def _relative_l2(a: np.ndarray, b: np.ndarray, eps: float = 1.0e-8) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if not valid.any():
        return 0.0
    return float(
        np.sqrt(np.square(a[valid] - b[valid]).sum())
        / (np.sqrt(np.square(a[valid]).sum()) + eps)
    )


def _stack_column(df: pd.DataFrame, column: str) -> np.ndarray:
    arrays = [_as_float_array(value) for value in df[column].values]
    if not arrays:
        return np.asarray([], dtype=np.float64)
    return np.concatenate(arrays)


def compare_prediction_files(path_a: Path, path_b: Path) -> dict[str, float | str]:
    df_a = pd.read_parquet(path_a)
    df_b = pd.read_parquet(path_b)
    key_cols = ["transcript_id", "dataset_id"]
    missing = [c for c in key_cols if c not in df_a.columns or c not in df_b.columns]
    if missing:
        raise KeyError(f"Missing key column(s) in prediction files: {missing}")
    merged = df_a.merge(df_b, on=key_cols, suffixes=("_a", "_b"))
    out: dict[str, float | str] = {
        "run_a": str(path_a),
        "run_b": str(path_b),
        "matched_rows": float(len(merged)),
    }
    for column in SEQUENCE_COLUMNS:
        col_a = f"{column}_a"
        col_b = f"{column}_b"
        if col_a not in merged.columns or col_b not in merged.columns:
            continue
        a = _stack_column(merged, col_a)
        b = _stack_column(merged, col_b)
        n = min(a.size, b.size)
        a = a[:n]
        b = b[:n]
        out[f"{column}_pcc"] = _safe_pcc(a, b)
        out[f"{column}_relative_l1"] = _relative_l1(a, b)
        out[f"{column}_relative_l2"] = _relative_l2(a, b)
        out[f"{column}_mean_a"] = float(np.nanmean(a)) if a.size else 0.0
        out[f"{column}_mean_b"] = float(np.nanmean(b)) if b.size else 0.0
        out[f"{column}_std_a"] = float(np.nanstd(a)) if a.size else 0.0
        out[f"{column}_std_b"] = float(np.nanstd(b)) if b.size else 0.0
    return out


def compare_grid(paths: list[Path]) -> pd.DataFrame:
    rows = [compare_prediction_files(a, b) for a, b in combinations(paths, 2)]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if {"mu_pcc", "L_bio_relative_l2", "gamma_relative_l2", "additive_bias_relative_l2"}.issubset(df.columns):
        df["decomposition_ambiguity_flag"] = (
            (df["mu_pcc"] > 0.98)
            & (
                (df["L_bio_relative_l2"] > 0.1)
                | (df["gamma_relative_l2"] > 0.1)
                | (df["additive_bias_relative_l2"] > 0.1)
            )
        )
    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare RiboUnmix prediction exports across regularization/anchoring "
            "settings to diagnose remaining L_bio/gamma/additive ambiguity."
        )
    )
    parser.add_argument("prediction_files", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    df = compare_grid(args.prediction_files)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
    else:
        print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
