from __future__ import annotations

from typing import Any, Optional, Sequence
import numpy as np


def log_plot_validation(
    target_profile: Any,
    *,
    mu_phys: Any = None,
    mu_total: Any = None,
    pi: Any = None,
    w_prob: Any = None,
    sigma: Any = None,
    css: Optional[Sequence[Any]] = None,
    lengths: Any = None,
    sample: int = 0,
    tag: str = "Validation_Profile_Diagnostic",
    step: Optional[int] = None,
    logger: Any = None,
    experiment: Any = None,
    one_based_x: bool = False,
    max_len: Optional[int] = None,
    close: bool = True,
    w_norm_eps: float = 1e-8,
    css_linestyle: str = "--",
    css_linewidth: float = 0.8,
    css_alpha: float = 0.35,
    # NEW:
    mu_phys_is_median: bool = True,
    ln_var_eps: float = 1e-8,
):
    # -------- rank-zero guard (DDP-safe) --------
    if not _is_rank_zero():
        return None

    import matplotlib.pyplot as plt

    def to_np(x: Any) -> Optional[np.ndarray]:
        if x is None:
            return None
        try:
            import torch
            if torch.is_tensor(x):
                x = x.detach().float().cpu().numpy()
        except Exception:
            pass
        return np.asarray(x)

    def select_1d(x: Any) -> Optional[np.ndarray]:
        arr = to_np(x)
        if arr is None:
            return None
        if arr.ndim == 0:
            return arr.reshape(1)
        if arr.ndim == 1:
            return arr
        return np.asarray(arr[sample]).reshape(-1)

    def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a).reshape(-1)
        b = np.asarray(b).reshape(-1)
        m = np.isfinite(a) & np.isfinite(b)
        a = a[m]
        b = b[m]
        if a.size < 3:
            return float("nan")
        if a.std() <= 1e-12 or b.std() <= 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    # -------- slice sample + determine valid length --------
    y = select_1d(target_profile)
    if y is None or y.size == 0:
        return None

    L = None
    if lengths is not None:
        L_arr = to_np(lengths).reshape(-1)
        if 0 <= sample < L_arr.size:
            L = int(L_arr[sample])

    T = int(y.size if L is None else max(0, min(L, y.size)))
    if max_len is not None:
        T = min(T, int(max_len))
    if T <= 0:
        return None

    def cut(a: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if a is None:
            return None
        return a[:T]

    mp = cut(select_1d(mu_phys))
    mt = cut(select_1d(mu_total))
    p = cut(select_1d(pi))
    wp = cut(select_1d(w_prob))
    sg = cut(select_1d(sigma))
    y = y[:T]

    x = (np.arange(1, T + 1) if one_based_x else np.arange(T))

    # -------- CSS indices (0-based input) --------
    css_idx: list[int] = []
    if css is not None:
        try:
            css_i = to_np(css[sample]).reshape(-1)
            css_i = css_i[np.isfinite(css_i)].astype(np.int64, copy=False)
            css_i = css_i[(css_i >= 0) & (css_i < T)]
            css_idx = sorted(set(int(v) for v in css_i.tolist()))
        except Exception:
            css_idx = []
    css_x = [i + 1 for i in css_idx] if one_based_x else css_idx

    # -------- w_target (true probability distribution) --------
    y_clip = np.maximum(y, 0.0)
    denom = float(y_clip.sum())
    if denom <= w_norm_eps:
        w_t = np.zeros_like(y_clip, dtype=np.float32)
    else:
        w_t = (y_clip / denom).astype(np.float32, copy=False)

    # -------- correlations --------
    corr_mu_phys = safe_corr(y, mp) if mp is not None else float("nan")
    corr_mu_total = safe_corr(y, mt) if mt is not None else float("nan")
    corr_w = safe_corr(wp, w_t) if (wp is not None) else float("nan")

    # -------- NEW: LogNormal variance (positive component) --------
    ln_var = None
    if (mp is not None) and (sg is not None):
        m = np.maximum(mp.astype(np.float32, copy=False), ln_var_eps)
        s = np.maximum(sg.astype(np.float32, copy=False), 0.0)
        s2 = s * s

        if mu_phys_is_median:
            # median m: Var = (exp(s^2)-1) * exp(2*log(m) + s^2)
            ln_var = (np.exp(s2) - 1.0) * np.exp(2.0 * np.log(m) + s2)
        else:
            # mean m: Var = (exp(s^2)-1) * m^2
            ln_var = (np.exp(s2) - 1.0) * (m * m)

        ln_var = np.where(np.isfinite(ln_var), ln_var, np.nan).astype(np.float32, copy=False)

    # -------- plot --------
    nrows = 5 if ln_var is not None else 4
    fig, axs = plt.subplots(nrows, 1, figsize=(14, 11 if nrows == 5 else 10), sharex=True)

    axs[0].plot(x, y, label="target")
    if mp is not None:
        axs[0].plot(x, mp, label="mu_phys")
    if mt is not None:
        axs[0].plot(x, mt, label="mu_total")
    axs[0].set_ylabel("y / mu")
    axs[0].legend(loc="upper right")

    axs[1].plot(x, w_t, label="w_target (= target / sum(target))")
    if wp is not None:
        axs[1].plot(x, wp, label="w_prob")
    axs[1].set_ylabel("w")
    axs[1].legend(loc="upper right")

    if p is not None:
        axs[2].plot(x, p, label="pi")
        axs[2].legend(loc="upper right")
    axs[2].set_ylabel("pi")

    if sg is not None:
        axs[3].plot(x, sg, label="sigma")
        axs[3].legend(loc="upper right")
    axs[3].set_ylabel("sigma")

    if ln_var is not None:
        axs[4].plot(x, ln_var, label="Var[LogNormal] (positive comp.)")
        axs[4].legend(loc="upper right")
        axs[4].set_ylabel("variance")
        axs[4].set_xlabel("position")
    else:
        axs[3].set_xlabel("position")

    if css_x:
        for ax in axs:
            for xx in css_x:
                ax.axvline(xx, linestyle=css_linestyle, linewidth=css_linewidth, alpha=css_alpha)

    fig.suptitle(
        f"{tag} | sample={sample} | "
        f"corr(y,mu_phys)={corr_mu_phys:.3f}  corr(y,mu_total)={corr_mu_total:.3f}  "
        f"corr(w_prob,w_target)={corr_w:.3f} | css_n={len(css_idx)}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    # -------- logging --------
    exp = experiment
    if exp is None and logger is not None:
        exp = getattr(logger, "experiment", None)

    gs = 0 if step is None else int(step)
    logged = False

    if exp is not None and hasattr(exp, "add_figure"):
        exp.add_figure(tag, fig, global_step=gs)
        logged = True

    if not logged:
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({tag: wandb.Image(fig)}, step=gs)
                logged = True
        except Exception:
            pass

    if close:
        plt.close(fig)
        return None
    return fig


def _is_rank_zero() -> bool:
    try:
        import torch
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
    except Exception:
        pass
    return True