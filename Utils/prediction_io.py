from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def flatten_predictions(obj: Any) -> List[Dict[str, Any]]:
    """Flatten Lightning `trainer.predict` output into a list of dict rows."""

    rows: List[Dict[str, Any]] = []

    def rec(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, dict):
            rows.append(x)
            return
        if isinstance(x, (list, tuple)):
            for y in x:
                rec(y)
            return
        # ignore unknown types instead of crashing

    rec(obj)
    return rows


def save_predictions_parquet(rows: Sequence[Dict[str, Any]], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows))
    df.to_parquet(out_path, index=False, engine="pyarrow")
    return out_path


def plot_example_profile(
    df: pd.DataFrame,
    *,
    out_dir: str | Path,
    example_idx: int = 0,
    trim: int = 5,
    pred_preference: Iterable[str] = ("mu_total_profile", "mu_phys_profile", "rho_profile"),
) -> Path:
    """Plot one example curve vs GT. Returns path to saved figure."""

    if len(df) == 0:
        raise ValueError("Empty dataframe; cannot plot.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    row = df.iloc[int(example_idx)]

    gt = np.asarray(row["gt_profile"], dtype=float)

    pr = None
    pr_label = None
    for col in pred_preference:
        if col in df.columns:
            pr = np.asarray(row[col], dtype=float)
            pr_label = f"Pred {col.replace('_profile','')}"
            break

    if pr is None:
        raise KeyError(f"None of {list(pred_preference)} found in df columns.")

    # hide trimmed edges if requested
    m = np.ones_like(gt, dtype=bool)
    if trim > 0 and gt.size > 2 * trim:
        m[:trim] = False
        m[-trim:] = False

    gt_plot = gt.copy()
    pr_plot = pr.copy()
    gt_plot[~m] = np.nan
    pr_plot[~m] = np.nan

    plt.figure(figsize=(12, 4))
    plt.plot(gt_plot, label="GT ribo")
    plt.plot(pr_plot, label=pr_label)
    plt.title(f"Transcript: {row['transcript_id']}  (L={int(row['length'])})")
    plt.xlabel("Position")
    plt.ylabel("Ribo / model space")
    plt.legend()

    fig_path = out_dir / f"example_{row['transcript_id']}.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    return fig_path
