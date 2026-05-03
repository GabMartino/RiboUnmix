from __future__ import annotations

from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt


def _to_numpy(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _safe_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    L = min(len(a), len(b))
    if L < 3:
        return np.nan

    a = a[:L]
    b = b[:L]

    if np.var(a) <= eps or np.var(b) <= eps:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def _normalize_profile(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, None)

    s = np.sum(x)
    if not np.isfinite(s) or s <= eps:
        return np.zeros_like(x, dtype=np.float64)

    return x / s


def _normalize_css(css_i: Any, L: int) -> np.ndarray:
    if css_i is None:
        return np.array([], dtype=int)

    css_arr = _to_numpy(css_i)
    css_arr = np.asarray(css_arr).reshape(-1)

    if css_arr.size == 0:
        return np.array([], dtype=int)

    if np.issubdtype(css_arr.dtype, np.number):
        css_arr = css_arr[np.isfinite(css_arr)]
        css_arr = css_arr.astype(int, copy=False)
        css_arr = css_arr[(css_arr >= 0) & (css_arr < L)]
        return css_arr

    return np.array([], dtype=int)


def _slice_optional(arr, sample: int, L: int, default=None):
    arr_np = _to_numpy(arr)
    if arr_np is None:
        return default
    return np.asarray(arr_np[sample, :L], dtype=np.float64)


def log_plot_validation(
    profiles_target,
    mu_pred,
    mu_total=None,
    pi=None,
    w_prob=None,
    phi=None,
    *,
    L_queue=None,
    L_effective=None,
    total_scale=None,
    b_offset=None,
    additive_bg=None,
    bg_q=None,
    p_bio=None,
    mu_bio=None,
    M_y=None,
    css=None,
    lengths=None,
    sample: int = 0,
    experiment=None,
    step: int = 0,
    tag: str = "val/profile_diag",
    eps: float = 1e-8,
):
    """
    Hurdle-Gamma + mass-conserved validation diagnostic plot.

    Current model meaning:

        mu_pred / mu_total:
            expected observed profile E[Y]

        p_bio:
            normalized biological profile:

                p_bio_i ∝ L_effective_i * exp(b_i)

        mu_bio:
            biological contribution to expected signal:

                mu_bio_i = M_y * (1 - beta_d) * p_bio_i

        additive_bg:
            additive background contribution:

                A_i = M_y * beta_d * q_i

        phi:
            Gamma relative dispersion:

                Var[Y | positive] = phi * mu_positive^2

    The important mass-conservation diagnostic is:

        sum_i mu_i ≈ sum_i y_i
    """

    y_all = _to_numpy(profiles_target)
    mu_all = _to_numpy(mu_pred)

    if y_all is None or mu_all is None:
        return None

    pi_all = _to_numpy(pi)
    w_all = _to_numpy(w_prob)
    phi_all = _to_numpy(phi)

    L_queue_all = _to_numpy(L_queue)
    L_eff_all = _to_numpy(L_effective)
    total_scale_all = _to_numpy(total_scale)
    b_all = _to_numpy(b_offset)
    additive_all = _to_numpy(additive_bg)
    bg_q_all = _to_numpy(bg_q)
    p_bio_all = _to_numpy(p_bio)
    mu_bio_all = _to_numpy(mu_bio)
    M_y_all = _to_numpy(M_y)
    lengths_np = _to_numpy(lengths)

    if lengths_np is not None:
        L = int(np.asarray(lengths_np).reshape(-1)[sample])
    else:
        L = int(y_all.shape[1])

    y = np.asarray(y_all[sample, :L], dtype=np.float64)
    mu = np.asarray(mu_all[sample, :L], dtype=np.float64)

    if mu_total is not None:
        mu_total_all = _to_numpy(mu_total)
        mu_plot = np.asarray(mu_total_all[sample, :L], dtype=np.float64)
    else:
        mu_plot = mu

    pi_i = _slice_optional(pi_all, sample, L, default=np.zeros(L, dtype=np.float64))
    phi_i = _slice_optional(phi_all, sample, L, default=np.zeros(L, dtype=np.float64))
    w_i = _slice_optional(w_all, sample, L, default=np.zeros(L, dtype=np.float64))

    L_queue_i = _slice_optional(L_queue_all, sample, L, default=None)
    L_eff_i = _slice_optional(L_eff_all, sample, L, default=L_queue_i)

    total_scale_i = _slice_optional(total_scale_all, sample, L, default=None)
    b_i = _slice_optional(b_all, sample, L, default=np.zeros(L, dtype=np.float64))
    A_i = _slice_optional(additive_all, sample, L, default=np.zeros(L, dtype=np.float64))
    q_i = _slice_optional(bg_q_all, sample, L, default=np.zeros(L, dtype=np.float64))
    p_bio_i = _slice_optional(p_bio_all, sample, L, default=None)
    mu_bio_i = _slice_optional(mu_bio_all, sample, L, default=None)

    # ------------------------------------------------------------
    # Fallbacks for old models / partial logging
    # ------------------------------------------------------------
    if p_bio_i is None:
        if L_eff_i is not None:
            p_bio_score = np.clip(L_eff_i, 0.0, None) * np.exp(
                np.clip(b_i, -20.0, 20.0)
            )
            p_bio_i = _normalize_profile(p_bio_score, eps=eps)
        else:
            p_bio_i = np.zeros(L, dtype=np.float64)
    else:
        p_bio_i = _normalize_profile(p_bio_i, eps=eps)

    if mu_bio_i is None:
        mu_bio_i = np.clip(mu - A_i, 0.0, None)

    if total_scale_i is not None and L_eff_i is not None:
        old_multiplicative_mu = L_eff_i * total_scale_i
    else:
        old_multiplicative_mu = None

    # ------------------------------------------------------------
    # Shapes / mass diagnostics
    # ------------------------------------------------------------
    w_target = _normalize_profile(y, eps=eps)
    w_prob_norm = _normalize_profile(w_i, eps=eps)
    q_norm = _normalize_profile(q_i, eps=eps)
    mu_shape = _normalize_profile(mu_plot, eps=eps)
    mu_bio_shape = _normalize_profile(mu_bio_i, eps=eps)

    bio_target_mass = np.clip(y - A_i, 0.0, None)
    bio_target_shape = _normalize_profile(bio_target_mass, eps=eps)

    additive_fraction = A_i / np.clip(mu, eps, None)

    target_mass = float(np.sum(y))
    pred_mass = float(np.sum(mu))
    bio_mass = float(np.sum(mu_bio_i))
    additive_mass = float(np.sum(A_i))

    if M_y_all is not None:
        M_y_i = float(np.asarray(M_y_all[sample]).reshape(-1)[0])
    else:
        M_y_i = target_mass

    mass_rel_error = abs(pred_mass - target_mass) / max(abs(target_mass), eps)

    beta_hat = additive_mass / max(pred_mass, eps)

    # ------------------------------------------------------------
    # Hurdle-Gamma variance diagnostics
    # ------------------------------------------------------------
    pi_i = np.nan_to_num(pi_i, nan=0.0, posinf=0.0, neginf=0.0)
    phi_i = np.nan_to_num(phi_i, nan=0.0, posinf=0.0, neginf=0.0)

    pi_i = np.clip(pi_i, 0.0, 1.0 - eps)
    phi_i = np.clip(phi_i, 0.0, None)

    mu_pos = mu / np.clip(1.0 - pi_i, eps, None)
    var_pos = phi_i * mu_pos**2
    var_total = (mu**2) * (phi_i + pi_i) / np.clip(1.0 - pi_i, eps, None)

    # ------------------------------------------------------------
    # Correlations
    # ------------------------------------------------------------
    corr_y_mu = _safe_corr(y, mu_plot)
    corr_w = _safe_corr(w_target, w_prob_norm)
    corr_p_bio = _safe_corr(bio_target_shape, p_bio_i)
    corr_mu_shape = _safe_corr(w_target, mu_shape)

    bg_q_max = float(np.max(q_norm)) if q_norm.size else 0.0
    p_bio_max = float(np.max(p_bio_i)) if p_bio_i.size else 0.0

    finite_add_frac = additive_fraction[np.isfinite(additive_fraction)]
    additive_frac_mean = float(np.mean(finite_add_frac)) if finite_add_frac.size else 0.0

    phi_mean = float(np.mean(phi_i))
    pi_mean = float(np.mean(pi_i))

    x = np.arange(L)

    if css is not None:
        try:
            css_i = _normalize_css(css[sample], L)
        except Exception:
            css_i = np.array([], dtype=int)
    else:
        css_i = np.array([], dtype=int)

    fig, axes = plt.subplots(
        7,
        1,
        figsize=(16, 16),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1.6, 1.25, 1.2, 1.05, 1.05, 1.05, 1.05],
        },
    )

    fig.suptitle(
        f"{tag} | sample={sample} | "
        f"corr(y,mu)={corr_y_mu:.3f} | "
        f"corr(shape_mu,y)={corr_mu_shape:.3f} | "
        f"corr(w,target)={corr_w:.3f} | "
        f"corr(p_bio,bio_target)={corr_p_bio:.3f} | "
        f"mass_err={mass_rel_error:.2e} | "
        f"beta≈{beta_hat:.3f} | "
        f"mean(pi)={pi_mean:.3f} | mean(phi)={phi_mean:.3f} | "
        f"css_n={len(css_i)}",
        fontsize=11,
    )

    # ------------------------------------------------------------
    # 1. Mean reconstruction and mass-conserved decomposition
    # ------------------------------------------------------------
    ax = axes[0]
    ax.plot(x, y, label="target y", linewidth=1.2)
    ax.plot(x, mu_plot, label="mu = E[Y]", linewidth=1.4)
    ax.plot(x, mu_bio_i, label="mu_bio = M_y(1-beta)p_bio", linewidth=1.1)
    ax.plot(x, A_i, label="additive A = M_y beta q", linewidth=1.1)

    ax.set_ylabel("signal")
    ax.set_title(
        "Mass-conserved mean decomposition | "
        f"sum(y)={target_mass:.3g}, sum(mu)={pred_mass:.3g}, "
        f"sum(mu_bio)={bio_mass:.3g}, sum(A)={additive_mass:.3g}, M_y={M_y_i:.3g}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # ------------------------------------------------------------
    # 2. Shape decomposition
    # ------------------------------------------------------------
    ax = axes[1]
    ax.plot(x, w_target, label="target shape y / sum(y)", linewidth=1.1)
    ax.plot(x, mu_shape, label="mu / sum(mu)", linewidth=1.1)
    ax.plot(x, w_prob_norm, label="w_prob", linewidth=1.15)
    ax.plot(x, p_bio_i, label="p_bio ∝ L_eff exp(b)", linewidth=1.2)

    ax.set_ylabel("probability")
    ax.set_title(
        "Profile-shape diagnostics: w_prob is intrinsic; p_bio includes shift + multiplicative b"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # ------------------------------------------------------------
    # 3. Background-corrected biological target
    # ------------------------------------------------------------
    ax = axes[2]
    ax.plot(x, bio_target_shape, label="bio target shape ∝ max(y - A, 0)", linewidth=1.1)
    ax.plot(x, p_bio_i, label="p_bio", linewidth=1.2)

    if np.any(q_norm > 0):
        ax.plot(x, q_norm, label="bg_q", linewidth=1.0, alpha=0.9)

    ax.set_ylabel("probability")
    ax.set_title(
        f"Biological shape vs background shape | max(p_bio)={p_bio_max:.3g}, max(bg_q)={bg_q_max:.3g}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # ------------------------------------------------------------
    # 4. Intrinsic / shifted queue profiles
    # ------------------------------------------------------------
    ax = axes[3]

    if L_queue_i is not None:
        ax.plot(
            x,
            _normalize_profile(L_queue_i, eps=eps),
            label="normalized L_queue",
            linewidth=1.05,
        )

    if L_eff_i is not None:
        ax.plot(
            x,
            _normalize_profile(L_eff_i, eps=eps),
            label="normalized L_effective",
            linewidth=1.05,
        )

    if old_multiplicative_mu is not None:
        ax.plot(
            x,
            _normalize_profile(old_multiplicative_mu, eps=eps),
            label="old-style normalized L_eff * total_scale",
            linewidth=0.95,
            alpha=0.8,
        )

    ax.set_ylabel("normalized")
    ax.set_title("Queueing profile before/after dataset-level shift")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # ------------------------------------------------------------
    # 5. Additive usage diagnostic
    # ------------------------------------------------------------
    ax = axes[4]
    ax.plot(x, additive_fraction, label="A / mu", linewidth=1.1)

    if np.any(q_norm > 0):
        q_scaled = q_norm / np.max(q_norm) if np.max(q_norm) > eps else q_norm
        ax.plot(
            x,
            q_scaled,
            label="bg_q / max(bg_q)",
            linewidth=1.0,
            alpha=0.85,
        )

    ax.set_ylim(bottom=0.0)
    ax.set_ylabel("fraction")
    ax.set_title(
        "Additive background usage; sharp bg_q or high A/mu at peaks means background is stealing signal"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # ------------------------------------------------------------
    # 6. Multiplicative bias
    # ------------------------------------------------------------
    ax = axes[5]
    ax.plot(x, b_i, label="b_offset", linewidth=1.1)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)

    ax.set_ylabel("log offset")
    ax.set_title("Centered multiplicative technical bias b; positive values amplify p_bio locally")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # ------------------------------------------------------------
    # 7. Hurdle-Gamma uncertainty
    # ------------------------------------------------------------
    ax = axes[6]
    ax.plot(x, pi_i, label="pi = P(zero/censored)", linewidth=1.05)
    ax.plot(x, phi_i, label="phi = Gamma relative dispersion", linewidth=1.05)
    ax.plot(
        x,
        np.log10(var_total + eps),
        label="log10 Var[Y]",
        linewidth=0.95,
        alpha=0.75,
    )

    ax.set_ylabel("pi / phi / log-var")
    ax.set_xlabel("codon position")
    ax.set_title("Hurdle probability, Gamma dispersion, and implied variance")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    for ax in axes:
        for j, css_pos in enumerate(css_i):
            ax.axvline(
                css_pos,
                color="purple",
                linestyle="--",
                linewidth=0.8,
                alpha=0.35,
                label="CSS" if j == 0 else None,
            )

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if experiment is not None:
        if hasattr(experiment, "add_figure"):
            experiment.add_figure(tag, fig, global_step=step)
        elif hasattr(experiment, "log_figure"):
            experiment.log_figure(figure_name=tag, figure=fig, step=step)

    plt.close(fig)
    return fig