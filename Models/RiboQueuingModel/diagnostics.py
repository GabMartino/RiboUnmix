from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class ProfileDiagnosticCache:
    transcript_id: str
    gt: Any
    mu_phys: Any
    mu_total: Any
    pi: Any
    rho: Any
    rho_target: Any
    sigma: Any
    w_prob: Any
    w_target: Any
    J_pressure: float
    J_eff: float
    S: float
    mu_pcc: float
    rho_pcc: float
    w_pcc: float



def make_profile_diagnostic_figure(
    cache: Dict[str, Any],
    *,
    epoch: int,
    S_quantile: float,
    css_linestyle: str = "--",
    css_linewidth: float = 0.8,
    css_alpha: float = 0.35,
    x_is_one_based: bool = False,
):
    """
    Expects `cache` keys:
      gt, mu_phys, mu_total, rho, rho_target, w_prob, w_target, pi, sigma
      optional: css_idx (list[int])
    """
    gt = np.asarray(cache["gt"])
    L = int(gt.shape[0])
    x = np.arange(1, L + 1) if x_is_one_based else np.arange(L)

    css_idx: Sequence[int] = cache.get("css_idx", []) or []
    if x_is_one_based:
        css_x = [int(i) + 1 for i in css_idx]
    else:
        css_x = [int(i) for i in css_idx]

    fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # --- panel 1: GT vs mu ---
    axs[0].plot(x, cache["gt"], label="GT")
    axs[0].plot(x, cache["mu_phys"], label="mu_phys")
    axs[0].plot(x, cache["mu_total"], label="mu_total")
    axs[0].set_ylabel("counts / mean")
    axs[0].legend(loc="upper right")

    # --- panel 2: rho vs rho_target ---
    axs[1].plot(x, cache["rho"], label="rho")
    axs[1].plot(x, cache["rho_target"], label="rho_target")
    axs[1].set_ylabel("rho")
    axs[1].legend(loc="upper right")

    # --- panel 3: w ---
    axs[2].plot(x, cache["w_prob"], label="w_prob")
    axs[2].plot(x, cache["w_target"], label="w_target")
    axs[2].set_ylabel("w")
    axs[2].legend(loc="upper right")

    # --- panel 4: pi + sigma ---
    axs[3].plot(x, cache["pi"], label="pi")
    axs[3].plot(x, cache["sigma"], label="sigma")
    axs[3].set_ylabel("pi / sigma")
    axs[3].set_xlabel("position")
    axs[3].legend(loc="upper right")

    # === CSS vertical lines (THIS is the part you asked for) ===
    if css_x:
        for ax in axs:
            for xx in css_x:
                ax.axvline(xx, linestyle=css_linestyle, linewidth=css_linewidth, alpha=css_alpha)

    title = (
        f"epoch={epoch} | id={cache.get('transcript_id','?')} | "
        f"S_q={S_quantile} | "
        f"mu_pcc={cache.get('mu_pcc', float('nan')):.3f} "
        f"rho_pcc={cache.get('rho_pcc', float('nan')):.3f} "
        f"w_pcc={cache.get('w_pcc', float('nan')):.3f}"
    )
    fig.suptitle(title)
    fig.tight_layout()
    return fig
