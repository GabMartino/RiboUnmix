from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


def log_plot_validation(
    target_profile: Any,
    mu_phys: Any = None,
    mu_total: Any = None,
    pi: Any = None,
    w_prob: Any = None,
    sigma: Any = None,
    *,
    l_queue: Any = None,
    alpha: Any = None,
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
    mu_phys_is_median: bool = True,
    ln_var_eps: float = 1e-8,
):
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
            return arr.reshape(-1)

        if not (0 <= sample < arr.shape[0]):
            raise IndexError(
                f"sample index {sample} out of bounds for array with shape {arr.shape}"
            )

        return np.asarray(arr[sample]).reshape(-1)

    def safe_corr(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        if a is None or b is None:
            return float("nan")

        a = np.asarray(a).reshape(-1)
        b = np.asarray(b).reshape(-1)

        n = min(a.size, b.size)
        if n < 3:
            return float("nan")

        a = a[:n]
        b = b[:n]

        m = np.isfinite(a) & np.isfinite(b)
        a = a[m]
        b = b[m]

        if a.size < 3:
            return float("nan")
        if a.std() <= 1e-12 or b.std() <= 1e-12:
            return float("nan")

        return float(np.corrcoef(a, b)[0, 1])

    def cut(a: Optional[np.ndarray], T: int) -> Optional[np.ndarray]:
        if a is None:
            return None
        return np.asarray(a[:T]).reshape(-1)

    def extract_css_indices(css_obj: Any, T: int) -> list[int]:
        if css_obj is None:
            return []

        try:
            arr = to_np(css_obj)
            if arr is None:
                return []

            if arr.ndim == 0:
                vals = arr.reshape(1)
            elif arr.ndim == 1:
                vals = arr
            else:
                if not (0 <= sample < arr.shape[0]):
                    return []
                vals = np.asarray(arr[sample]).reshape(-1)

            vals = vals[np.isfinite(vals)].astype(np.int64, copy=False)
            vals = vals[(vals >= 0) & (vals < T)]
            return sorted(set(int(v) for v in vals.tolist()))
        except Exception:
            return []

    y = select_1d(target_profile)
    if y is None or y.size == 0:
        return None

    L = None
    if lengths is not None:
        L_arr = to_np(lengths)
        if L_arr is not None:
            L_arr = np.asarray(L_arr).reshape(-1)
            if 0 <= sample < L_arr.size:
                L = int(L_arr[sample])

    T = int(y.size if L is None else max(0, min(L, y.size)))
    if max_len is not None:
        T = min(T, int(max_len))
    if T <= 0:
        return None

    y = y[:T]
    mp = cut(select_1d(mu_phys), T)
    mt = cut(select_1d(mu_total), T)
    p = cut(select_1d(pi), T)
    wp = cut(select_1d(w_prob), T)
    sg = cut(select_1d(sigma), T)
    lq = cut(select_1d(l_queue), T)
    al = cut(select_1d(alpha), T)

    x = np.arange(1, T + 1) if one_based_x else np.arange(T)

    css_idx = extract_css_indices(css, T)
    css_x = [i + 1 for i in css_idx] if one_based_x else css_idx

    y_clip = np.maximum(y, 0.0)
    denom = float(y_clip.sum())
    if denom <= w_norm_eps:
        w_t = np.zeros_like(y_clip, dtype=np.float32)
    else:
        w_t = (y_clip / denom).astype(np.float32, copy=False)

    corr_mu_phys = safe_corr(y, mp)
    corr_mu_total = safe_corr(y, mt)
    corr_w = safe_corr(wp, w_t)
    corr_lq = safe_corr(lq, y) if lq is not None else float("nan")

    ln_var = None
    if (mp is not None) and (sg is not None):
        m = np.maximum(mp.astype(np.float32, copy=False), ln_var_eps)
        s = np.maximum(sg.astype(np.float32, copy=False), 0.0)
        s2 = s * s

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if mu_phys_is_median:
                ln_var = (np.exp(s2) - 1.0) * np.exp(2.0 * np.log(m) + s2)
            else:
                ln_var = (np.exp(s2) - 1.0) * (m * m)

        ln_var = np.where(np.isfinite(ln_var), ln_var, np.nan).astype(np.float32, copy=False)

    panels: list[tuple[str, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]] = [
        ("y / mu", y, mp, mt),
        ("w", w_t, wp, None),
        ("pi", p, None, None),
        ("sigma", sg, None, None),
    ]

    if lq is not None:
        panels.append(("L_queue", lq, None, None))

    if al is not None:
        panels.append(("alpha", al, None, None))

    if ln_var is not None:
        panels.append(("variance", ln_var, None, None))

    nrows = len(panels)
    fig, axs = plt.subplots(
        nrows,
        1,
        figsize=(14, max(8, 2.1 * nrows)),
        sharex=True,
    )

    if nrows == 1:
        axs = [axs]

    row = 0

    axs[row].plot(x, y, label="target")
    if mp is not None:
        axs[row].plot(x, mp, label="mu_phys")
    if mt is not None:
        axs[row].plot(x, mt, label="mu_total")
    axs[row].set_ylabel("y / mu")
    axs[row].legend(loc="upper right")
    row += 1

    axs[row].plot(x, w_t, label="w_target (= target / sum(target))")
    if wp is not None:
        axs[row].plot(x, wp, label="w_prob")
    axs[row].set_ylabel("w")
    axs[row].legend(loc="upper right")
    row += 1

    axs[row].set_ylabel("pi")
    if p is not None:
        axs[row].plot(x, p, label="pi")
        axs[row].legend(loc="upper right")
    row += 1

    axs[row].set_ylabel("sigma")
    if sg is not None:
        axs[row].plot(x, sg, label="sigma")
        axs[row].legend(loc="upper right")
    row += 1

    if lq is not None:
        axs[row].plot(x, lq, label="L_queue")
        axs[row].set_ylabel("L_queue")
        axs[row].legend(loc="upper right")
        row += 1

    if al is not None:
        axs[row].plot(x, al, label="alpha")
        axs[row].set_ylabel("alpha")
        axs[row].legend(loc="upper right")
        row += 1

    if ln_var is not None:
        axs[row].plot(x, ln_var, label="Var[LogNormal] (positive comp.)")
        axs[row].set_ylabel("variance")
        axs[row].legend(loc="upper right")
        row += 1

    axs[-1].set_xlabel("position")

    if css_x:
        for ax in axs:
            for xx in css_x:
                ax.axvline(
                    xx,
                    linestyle=css_linestyle,
                    linewidth=css_linewidth,
                    alpha=css_alpha,
                )

    title_parts = [
        f"{tag}",
        f"sample={sample}",
        f"corr(y,mu_phys)={corr_mu_phys:.3f}",
        f"corr(y,mu_total)={corr_mu_total:.3f}",
        f"corr(w_prob,w_target)={corr_w:.3f}",
        f"css_n={len(css_idx)}",
    ]
    if lq is not None:
        title_parts.insert(-1, f"corr(L_queue,y)={corr_lq:.3f}")

    fig.suptitle(" | ".join(title_parts))
    fig.tight_layout(rect=(0, 0, 1, 0.96))

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