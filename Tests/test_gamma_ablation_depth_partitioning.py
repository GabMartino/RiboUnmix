from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _load_numbered_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "analyses" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {filename}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RECOVERY = _load_numbered_module(
    "gamma_ablation_recovery_depth_test",
    "synthetic_gamma_ablation_lbio_recovery.py",
)
BIOLOGY = _load_numbered_module(
    "gamma_ablation_biology_depth_test",
    "synthetic_gamma_ablation_biological_quality.py",
)
REPORT = _load_numbered_module(
    "gamma_ablation_report_depth_test",
    "synthetic_gamma_ablation_report.py",
)


class GammaAblationDepthPartitioningTests(unittest.TestCase):
    def test_recovery_summary_preserves_depth_and_mass_condition(self) -> None:
        rows = []
        for index, depth in enumerate(("0p25_per_codon", "20_per_codon")):
            rows.append(
                {
                    "run_id": f"run_{index}",
                    "strategy": "within",
                    "training_scope": "multi_dataset",
                    "depth": depth,
                    "mass_condition": "mass_free",
                    "n_datasets": 2,
                    "quality_rank_power": 0.0,
                    "gamma_weighting": "equal",
                    "feature_preset": "Baseline",
                    "seed": 42,
                    "model_id": f"run_{index}",
                    "split": "main_val",
                    "dataset": "artificial_bias_3prime_aa",
                    "component": "L_bio",
                    "reference_column": "rib_profile",
                    "reference_kind": "latent_ground_truth",
                    "scope": "shared_latent_truth_interior",
                    "transcript_id": f"tx_{index}",
                    "pearson": 0.8,
                    "spearman": 0.7,
                    "shape_rmse": 0.2,
                    "n_positions": 100,
                }
            )
        summary = pd.DataFrame(
            RECOVERY.summarize_transcript_rows(
                rows,
                n_bootstrap=0,
                seed=42,
            )
        )
        self.assertEqual(len(summary), 2)
        self.assertEqual(
            set(summary["depth"]),
            {"0p25_per_codon", "20_per_codon"},
        )
        self.assertEqual(set(summary["mass_condition"]), {"mass_free"})

    def test_biological_summary_preserves_depth_and_mass_condition(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "run_id": f"run_{index}",
                    "strategy": "within",
                    "training_scope": "multi_dataset",
                    "depth": depth,
                    "mass_condition": "mass_free",
                    "n_datasets": 2,
                    "quality_rank_power": 0.0,
                    "gamma_weighting": "equal",
                    "feature_preset": "Baseline",
                    "seed": 42,
                    "split": "main_val",
                    "signal": "L_bio",
                    "metric": "motif_P",
                    "n": 10,
                    "mean": 0.1,
                }
                for index, depth in enumerate(
                    ("0p25_per_codon", "20_per_codon")
                )
            ]
        )
        summary = BIOLOGY.condition_summaries(metrics, pd.DataFrame())
        self.assertEqual(len(summary), 2)
        self.assertEqual(
            set(summary["depth"]),
            {"0p25_per_codon", "20_per_codon"},
        )

    def test_plotters_reject_mixed_depth_input(self) -> None:
        recovery = pd.DataFrame(
            {
                "split": ["main_val", "main_val"],
                "depth": ["0p25_per_codon", "20_per_codon"],
                "mass_condition": ["mass_free", "mass_free"],
            }
        )
        with self.assertRaisesRegex(ValueError, "exactly one read depth"):
            RECOVERY.plot_recovery(recovery, Path("unused.png"))

        biology = recovery.assign(signal="L_bio", metric="motif_P")
        with self.assertRaisesRegex(ValueError, "exactly one read depth"):
            BIOLOGY.plot_biological_summary(biology, Path("unused.png"))

    def test_report_condition_identity_includes_depth_and_mass(self) -> None:
        self.assertIn("depth", REPORT.CONDITION_KEYS)
        self.assertIn("mass_condition", REPORT.CONDITION_KEYS)


if __name__ == "__main__":
    unittest.main()
