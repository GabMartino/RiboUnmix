from __future__ import annotations

import torch
import torch.nn as nn

from Models.RiboQueuingModel.DatasetBiasSubmodel import DatasetBiasSubmodel
from Models.RiboQueuingModel.QueuingBiologicalModel import QueuingBiologicalModel


class RiboQueuingModel(nn.Module):
    """
    Queue-length-support model with multiplicative dataset allocation bias.

    Biological branch:

        a_t       = biological allocation logits
        w_bio     = entmax(a_t)
        h_bio_i   = J_t * T_t * w_bio_i
        L_bio_i   = exp(h_bio_i) - 1

    Dataset/protocol branch:

        b_raw_{d,t,i} = gate_{d,t,i} * amplitude_{d,t,i}

        b_eff_{d,t,i}
            = b_raw_{d,t,i} / sum_j w_bio_{t,j} b_raw_{d,t,j}

        w_obs_{d,t,i}
            = w_bio_{t,i} * b_eff_{d,t,i}

    Therefore:

        sum_i w_obs_{d,t,i} = 1

    Observation support:

        h_obs_i = J_t * T_t * w_obs_i
        L_obs_i = exp(h_obs_i) - 1

    Profile mean:

        mu_i = total_mass * L_obs_i / sum_j L_obs_j

    Important:
        L_bio is the shared biological signal.
        L_obs is the dataset-observed signal after multiplicative allocation bias.
    """

    def __init__(
        self,
        model_configs: dict,
        eps: float = 1e-8,
        mu_max: float = 1e8,
    ):
        super().__init__()

        self.eps = float(eps)
        self.mu_max = float(mu_max)

        dataset_bias_params = model_configs["dataset_bias_params"]
        biological_params = model_configs["biological_params"]

        self.position_features = list(dataset_bias_params["position_features"])
        self.position_scale = float(dataset_bias_params.get("position_scale", 5000.0))
        self.position_edge_tau = float(dataset_bias_params.get("position_edge_tau", 30.0))

        self.hazard_max = float(
            biological_params.get(
                "hazard_max",
                dataset_bias_params.get("hazard_max", 8.0),
            )
        )

        self.biological_model = QueuingBiologicalModel(
            config_params=biological_params,
        )

        use_biological_context = bool(
            dataset_bias_params.get("use_biological_context", True)
        )

        if use_biological_context:
            biological_context_size = (
                    int(biological_params["num_layers"])
                    * 2
                    * int(biological_params["hidden_size"])
            )
        else:
            biological_context_size = 0

        self.dataset_bias_model = DatasetBiasSubmodel(
            config_params=dataset_bias_params,
            biological_context_size=biological_context_size,
        )

    # ============================================================
    # Position features
    # ============================================================

    def make_position_features(
        self,
        mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        mask_b = mask.bool()
        B, T = mask_b.shape
        device = mask_b.device

        pos = torch.arange(T, device=device, dtype=dtype).unsqueeze(0).expand(B, T)

        lengths = mask_b.sum(dim=1, keepdim=True).to(dtype=dtype).clamp_min(1.0)
        last_pos = (lengths - 1.0).clamp_min(1.0)

        rel_pos = pos / last_pos
        abs_pos = pos / float(self.position_scale)

        abs_pos_log = torch.log1p(pos) / torch.log1p(
            torch.tensor(
                float(self.position_scale),
                device=device,
                dtype=dtype,
            )
        )

        tau = max(float(self.position_edge_tau), 1.0e-6)

        dist_start = pos
        dist_stop = (lengths - 1.0 - pos).clamp_min(0.0)

        start_window = torch.exp(-dist_start / tau)
        stop_window = torch.exp(-dist_stop / tau)

        feature_map = {
            "rel_pos": rel_pos,
            "abs_pos": abs_pos,
            "abs_pos_log": abs_pos_log,
            "start_window": start_window,
            "stop_window": stop_window,
            "dist_to_start": rel_pos,
            "dist_to_stop": 1.0 - rel_pos,
        }

        missing = [name for name in self.position_features if name not in feature_map]
        if missing:
            raise KeyError(f"Unknown position feature(s): {missing}")

        x_pos = torch.stack(
            [feature_map[name] for name in self.position_features],
            dim=-1,
        )

        x_pos = x_pos * mask_b.unsqueeze(-1).to(dtype=dtype)

        return x_pos

    # ============================================================
    # Support/profile helpers
    # ============================================================

    def _profile_from_support(
        self,
        *,
        q: torch.Tensor,
        fallback_q: torch.Tensor,
        y_raw_target: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=q.dtype)

        q = q.clamp_min(0.0) * mask_f
        fallback_q = fallback_q.clamp_min(0.0) * mask_f

        q_mass_raw = q.sum(dim=1, keepdim=True)
        use_q = q_mass_raw > self.eps

        fallback_mass = fallback_q.sum(dim=1, keepdim=True)
        use_fallback = (~use_q) & (fallback_mass > self.eps)

        q_uniform = mask_f

        q_for_profile = torch.where(
            use_q,
            q,
            torch.where(use_fallback, fallback_q, q_uniform),
        )

        q_for_profile = q_for_profile * mask_f
        q_mass = q_for_profile.sum(dim=1, keepdim=True).clamp_min(self.eps)

        profile_prob = q_for_profile / q_mass
        profile_prob = profile_prob * mask_f

        total_mass = (
            y_raw_target.float().clamp_min(0.0) * mask_f.float()
        ).sum(dim=1, keepdim=True).detach().clamp_min(self.eps)

        mu = total_mass * profile_prob
        mu = mu * mask_f

        mu = torch.nan_to_num(
            mu,
            nan=self.eps,
            posinf=self.mu_max,
            neginf=self.eps,
        )

        mu = mu.clamp(min=self.eps, max=self.mu_max)
        mu = torch.where(mask_b, mu, torch.ones_like(mu))

        return {
            "mu": mu,
            "profile_prob": profile_prob,
            "q_for_profile": q_for_profile,
            "q_mass": q_mass,
            "q_mass_raw": q_mass_raw,
            "total_mass": total_mass,
            "q_used_fallback": use_fallback.reshape(-1).float(),
            "q_used_uniform_fallback": ((~use_q) & (~use_fallback)).reshape(-1).float(),
        }

    def _mu_from_support(
        self,
        *,
        support: torch.Tensor,
        total_mass: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_f = mask.bool().to(dtype=support.dtype)

        support = support.clamp_min(0.0) * mask_f
        mass = support.sum(dim=1, keepdim=True).clamp_min(self.eps)

        return total_mass * support / mass * mask_f

    def _zero_frac(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        threshold: float = 0.0,
    ) -> torch.Tensor:
        mask_b = mask.bool()
        mask_f = mask_b.float()

        return (
            ((x <= threshold) & mask_b).float().sum(dim=1)
            / mask_f.sum(dim=1).clamp_min(1.0)
        )

    # ============================================================
    # Multiplicative allocation-bias helper
    # ============================================================

    def _apply_multiplicative_allocation_bias(
        self,
        *,
        w_bio: torch.Tensor,
        b_raw: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Applies multiplicative dataset bias to biological allocation.

        Given:

            sum_i w_bio_i = 1

        and positive/gated raw bias b_raw_i, define:

            b_eff_i = b_raw_i / sum_j w_bio_j b_raw_j

            w_obs_i = w_bio_i * b_eff_i

        Then:

            sum_i w_obs_i = 1

        If all gates close and weighted_b_mass is zero, falls back to w_bio.
        """
        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=w_bio.dtype)

        w = w_bio.clamp_min(0.0) * mask_f
        b = b_raw.to(dtype=w_bio.dtype).clamp_min(0.0) * mask_f

        # Biological allocation should already sum to 1, but renormalize
        # defensively to avoid numerical drift.
        w = w / w.sum(dim=1, keepdim=True).clamp_min(self.eps)
        w = w * mask_f

        weighted_b_mass = (w * b).sum(dim=1, keepdim=True)
        has_mass = weighted_b_mass > self.eps

        b_eff = b / weighted_b_mass.clamp_min(self.eps)
        w_obs_candidate = w * b_eff
        w_obs_candidate = w_obs_candidate * mask_f

        w_obs = torch.where(
            has_mass,
            w_obs_candidate,
            w,
        )

        w_obs = w_obs * mask_f
        w_obs = w_obs / w_obs.sum(dim=1, keepdim=True).clamp_min(self.eps)
        w_obs = w_obs * mask_f

        # For diagnostics, padding should be neutral.
        b_eff_diag = torch.where(mask_b, b_eff, torch.ones_like(b_eff))

        return w_obs, b_eff_diag, weighted_b_mass.reshape(-1)

    # ============================================================
    # Forward
    # ============================================================

    def forward(
        self,
        x_packed,
        codon_ids: torch.Tensor,
        id_datasets: torch.Tensor,
        y_raw_target: torch.Tensor,
    ):
        # ------------------------------------------------------------
        # 1. Shared biological branch
        # ------------------------------------------------------------
        bio = self.biological_model(x_packed)

        if not isinstance(bio, dict):
            raise RuntimeError(
                "This RiboQueuingModel expects QueuingBiologicalModel to return a dict "
                "with keys: w_bio, J, lengths, mask, h_bio, rho_bio, L_bio."
            )

        w_bio = bio["w_bio"]
        J = bio["J"]
        lengths = bio["lengths"]
        mask = bio["mask"]

        h_bio = bio["h_bio"]
        rho_bio = bio["rho_bio"]
        L_bio = bio["L_bio"]

        B, T = mask.shape
        mask_b = mask.bool()
        mask_f = mask_b.to(dtype=w_bio.dtype)

        # ------------------------------------------------------------
        # 2. Dataset/protocol multiplicative allocation bias
        # ------------------------------------------------------------
        position_features = self.make_position_features(
            mask=mask_b,
            dtype=w_bio.dtype,
        )

        h_n = bio.get("h_n", None)

        if h_n is None and getattr(self.dataset_bias_model, "biological_context_size", 0) > 0:
            raise RuntimeError(
                "Dataset bias model expects biological_context, but biological model "
                "did not return 'h_n'."
            )

        bias = self.dataset_bias_model(
            dataset_ids=id_datasets,
            codon_ids=codon_ids,
            mask=mask_b,
            position_features=position_features,
            biological_context=h_n.detach() if h_n is not None else None,
        )

        b_raw = bias["obs_bias_raw"].to(dtype=w_bio.dtype)
        b_raw = b_raw.clamp_min(0.0) * mask_f

        w_obs, b_eff, weighted_b_mass = self._apply_multiplicative_allocation_bias(
            w_bio=w_bio,
            b_raw=b_raw,
            mask=mask_b,
        )

        # ------------------------------------------------------------
        # 3. Observed hazard/support
        # ------------------------------------------------------------
        h_obs = self.biological_model.hazard_from_allocation(
            w=w_obs,
            J=J,
            lengths=lengths,
            mask=mask_b,
        )

        rho_obs = self.biological_model.rho_from_hazard(
            h_obs,
            mask_b,
        )

        L_obs = self.biological_model.L_queue_from_hazard(
            h_obs,
            mask_b,
        )

        # ------------------------------------------------------------
        # 4. Profile mean from observed support
        # ------------------------------------------------------------
        prof = self._profile_from_support(
            q=L_obs,
            fallback_q=L_bio,
            y_raw_target=y_raw_target,
            mask=mask_b,
        )

        mu = prof["mu"]
        total_mass = prof["total_mass"]

        # ------------------------------------------------------------
        # 5. Dispersion / kappa input
        # ------------------------------------------------------------
        phi_raw = bias["phi"].to(dtype=w_bio.dtype)

        if phi_raw.shape == mask.shape:
            phi = torch.where(mask_b, phi_raw, torch.ones_like(phi_raw))
        else:
            phi = phi_raw

        # ------------------------------------------------------------
        # 6. Mean-profile diagnostics
        # ------------------------------------------------------------
        mu_L_bio = self._mu_from_support(
            support=L_bio,
            total_mass=total_mass,
            mask=mask_b,
        )

        mu_L_obs = self._mu_from_support(
            support=L_obs,
            total_mass=total_mass,
            mask=mask_b,
        )

        # Compatibility aliases for existing Lightning logging.
        mu_L_only = mu_L_bio
        mu_bio_smooth = mu_L_obs
        mu_bio_only = mu_L_obs

        # ------------------------------------------------------------
        # 7. Mass / zero / stability diagnostics
        # ------------------------------------------------------------
        L_mass_bio = (L_bio * mask_f).sum(dim=1)
        L_mass_obs = (L_obs * mask_f).sum(dim=1)

        h_mass_bio = (h_bio * mask_f).sum(dim=1)
        h_mass_obs = (h_obs * mask_f).sum(dim=1)

        w_bio_zero_frac = self._zero_frac(w_bio, mask_b, threshold=0.0)
        w_obs_zero_frac = self._zero_frac(w_obs, mask_b, threshold=0.0)

        L_bio_zero_frac = self._zero_frac(L_bio, mask_b, threshold=0.0)
        L_obs_zero_frac = self._zero_frac(L_obs, mask_b, threshold=0.0)

        hazard_cap_frac_bio = (
            ((h_bio >= 0.99 * self.hazard_max) & mask_b).float().sum(dim=1)
            / mask_f.sum(dim=1).clamp_min(1.0)
        )

        hazard_cap_frac_obs = (
            ((h_obs >= 0.99 * self.hazard_max) & mask_b).float().sum(dim=1)
            / mask_f.sum(dim=1).clamp_min(1.0)
        )

        delta_w_l1 = ((w_obs - w_bio).abs() * mask_f).sum(dim=1)

        log_b_eff = torch.log(b_eff.clamp_min(self.eps))

        b_abs_log_mean = (
            log_b_eff.abs() * mask_f
        ).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)

        b_abs_log_w_bio_weighted = (
            w_bio.detach().float().clamp_min(0.0) * log_b_eff.detach().float().abs() * mask_f
        ).sum(dim=1)

        obs_bias_keep_prob = bias.get("obs_bias_keep_prob")
        obs_bias_keep_gate = bias.get("obs_bias_keep_gate")
        obs_bias_keep_hard = bias.get("obs_bias_keep_hard")
        obs_bias_amp = bias.get("obs_bias_amp")
        obs_bias_amp_logits = bias.get("obs_bias_amp_logits")
        obs_bias_gate_logits = bias.get("obs_bias_gate_logits")
        h_n_flat = None
        if h_n is not None:
            h_n_flat = h_n.detach().permute(1, 0, 2).reshape(B, -1)
        # ------------------------------------------------------------
        # 8. Extras
        # ------------------------------------------------------------
        extras = {
            # --------------------------------------------------------
            # Biological branch
            # --------------------------------------------------------
            "w_logits": bio.get("w_logits"),
            "w_bio": w_bio,
            "w_prob": w_bio,

            "J": J,
            "biological_context": h_n_flat,
            "biological_context_norm": (
                h_n_flat.float().norm(dim=1) if h_n_flat is not None else None
            ),
            "h_bio": h_bio,
            "rho_bio": rho_bio,
            "L_bio": L_bio,
            "L_queue": L_bio,
            "bio_q_base": L_bio,

            # --------------------------------------------------------
            # Multiplicative observation bias
            # --------------------------------------------------------
            "obs_bias_raw": b_raw,
            "obs_bias_effective": b_eff,
            "obs_bias_weighted_mass": weighted_b_mass,

            "obs_bias_amp": obs_bias_amp,
            "obs_bias_amp_logits": obs_bias_amp_logits,

            "obs_bias_keep_prob": obs_bias_keep_prob,
            "obs_bias_keep_gate": obs_bias_keep_gate,
            "obs_bias_keep_hard": obs_bias_keep_hard,
            "obs_bias_gate_logits": obs_bias_gate_logits,

            # Compatibility aliases for existing b/gate diagnostics.
            "b_shape": b_eff,
            "b_effective": b_eff,
            "b_smooth": b_raw,
            "log_b_shape": log_b_eff,
            "log_b": log_b_eff,

            "keep_gate": obs_bias_keep_gate,
            "keep_prob": obs_bias_keep_prob,
            "keep_hard": obs_bias_keep_hard,
            "gate_logits": obs_bias_gate_logits,

            # --------------------------------------------------------
            # Observed branch
            # --------------------------------------------------------
            "w_obs": w_obs,
            "h_obs": h_obs,
            "rho_obs": rho_obs,
            "L_obs": L_obs,
            "L_queue_obs": L_obs,

            "bio_q_smooth": L_obs,
            "bio_q": L_obs,
            "q": L_obs,
            "q_for_profile": prof["q_for_profile"],
            "q_mass": prof["q_mass"].reshape(B),
            "q_mass_raw": prof["q_mass_raw"].reshape(B),
            "profile_prob": prof["profile_prob"],

            "q_used_fallback": prof["q_used_fallback"],
            "q_used_uniform_fallback": prof["q_used_uniform_fallback"],

            "total_mass": total_mass.reshape(B),

            # --------------------------------------------------------
            # Mean profiles
            # --------------------------------------------------------
            "mu_obs": mu,
            "mu_L_only": mu_L_only,
            "mu_L_bio": mu_L_bio,
            "mu_L_obs": mu_L_obs,
            "mu_bio_smooth": mu_bio_smooth,
            "mu_bio_only": mu_bio_only,

            # --------------------------------------------------------
            # Diagnostics
            # --------------------------------------------------------
            "L_mass_bio": L_mass_bio,
            "L_mass_base": L_mass_bio,
            "L_mass_obs": L_mass_obs,
            "L_mass_smooth": L_mass_obs,
            "L_mass_eff": L_mass_obs,

            "h_mass_bio": h_mass_bio,
            "h_mass_obs": h_mass_obs,

            "w_bio_zero_frac": w_bio_zero_frac,
            "w_obs_zero_frac": w_obs_zero_frac,
            "L_bio_zero_frac": L_bio_zero_frac,
            "L_obs_zero_frac": L_obs_zero_frac,

            "delta_w_obs_bio_l1": delta_w_l1,

            "b_abs_log_mean": b_abs_log_mean,
            "b_abs_log_w_bio_weighted": b_abs_log_w_bio_weighted,

            "hazard_cap_frac": hazard_cap_frac_obs,
            "hazard_cap_frac_bio": hazard_cap_frac_bio,
            "hazard_cap_frac_obs": hazard_cap_frac_obs,

            # --------------------------------------------------------
            # Dispersion / kappa input
            # --------------------------------------------------------
            "phi": phi,
            "phi_raw": phi_raw,
            "kappa_input": phi,
        }

        return mu, phi, extras