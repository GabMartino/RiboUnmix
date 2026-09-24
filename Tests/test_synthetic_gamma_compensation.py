"""Focused tests for the isolated synthetic gamma-compensation report tool."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analyses.analyze_synthetic_gamma_compensation import (
    _historical_config,
    alpha_calibration_metrics,
    classify_sites,
    find_best_val_loss_prediction,
    summarize_boundary_diagnostics,
)


class SyntheticGammaCompensationTests(unittest.TestCase):
    def test_alpha_calibration_uses_interior_positions_and_known_truth(self) -> None:
        pairs = {
            ("t1", "A"): {"alpha": np.asarray([9.0] * 5 + [0.1, 0.1] + [9.0] * 5)},
            ("t1", "B"): {"alpha": np.asarray([9.0] * 5 + [0.2, 0.05] + [9.0] * 5)},
        }
        metrics = alpha_calibration_metrics(pairs, true_alpha=0.1)
        self.assertEqual(metrics["alpha_validation_pairs"], 2)
        self.assertEqual(metrics["alpha_validation_positions"], 4)
        self.assertAlmostEqual(metrics["alpha_mean"], 0.1125)
        self.assertAlmostEqual(metrics["alpha_mae_from_true"], 0.0375)
        self.assertAlmostEqual(metrics["alpha_fraction_within_2fold"], 1.0)

    def test_boundary_compensation_diagnostics_do_not_enter_primary_rate(self) -> None:
        frame = pd.DataFrame(
            {
                "run": ["r"] * 4,
                "bias_name": ["b"] * 4,
                "is_interior": [True, True, False, False],
                "is_strong": [True, True, True, True],
                "is_missed_strong": [False, False, True, True],
            }
        )
        row = summarize_boundary_diagnostics(
            frame, ["run", "bias_name"]
        ).iloc[0]
        self.assertEqual(row["interior_strong_site_miss_rate"], 0.0)
        self.assertEqual(row["boundary_strong_site_miss_rate"], 1.0)
        self.assertEqual(row["fraction_of_all_misses_at_boundary"], 1.0)

    def test_legacy_or_pcc_prediction_is_never_accepted_as_best_val_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "predictions_main_val_panel.parquet").touch()
            prediction, reason = find_best_val_loss_prediction(root)
            self.assertIsNone(prediction)
            self.assertIn("legacy", reason)

    def test_manifest_requires_val_loss_checkpoint_and_variant_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction = root / "predictions_main_val_best_val_loss_panel.parquet"
            prediction.touch()
            (root / "prediction_checkpoint_manifest.json").write_text(json.dumps({
                "best_val_loss": {
                    "checkpoint_path": str(root / "epoch=2-val_loss=0.2.ckpt"),
                    "output_path": str(prediction),
                }
            }), encoding="utf-8")
            found, reason = find_best_val_loss_prediction(root)
            self.assertEqual(found, prediction)
            self.assertEqual(reason, "manifest")

    def test_remote_manifest_path_relocates_to_same_named_local_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local = root / "results" / "predictions_main_val_best_val_loss_panel.parquet"
            local.parent.mkdir()
            local.touch()
            remote = Path("/remote/cluster/results") / local.name
            (root / "prediction_checkpoint_manifest.json").write_text(json.dumps({
                "best_val_loss": {
                    "checkpoint_path": "/remote/epoch=2-val_loss=0.2.ckpt",
                    "output_path": str(remote),
                }
            }), encoding="utf-8")
            found, reason = find_best_val_loss_prediction(root)
            self.assertEqual(found, local)
            self.assertEqual(reason, "manifest (relocated local artifact)")

    def test_strong_miss_and_joint_compensation_are_explicit(self) -> None:
        frame = pd.DataFrame({
            "run": ["r", "r", "r"], "dataset": ["d", "d", "d"],
            "g_true": [1.2, 1.2, 0.0], "g_learned": [0.2, 1.15, 0.0],
            "programmed_bias": [True, True, False],
            "L_true": [1.0, 1.0, 1.0], "L_learned": [1.4, 1.0, 1.0],
            "alpha": [2.0, 1.0, 1.0], "consensus": [3.0, 3.0, 3.0],
            "replicate_cv": [0.1, 0.1, 0.1], "log1p_consensus": np.log1p([3.0, 3.0, 3.0]),
        })
        output = classify_sites(frame)
        self.assertTrue(bool(output.loc[0, "is_missed_strong"]))
        self.assertEqual(output.loc[1, "site_group"], "correct_strong")
        self.assertTrue(bool(output.loc[0, "L_inflated"]))
        self.assertGreater(float(output.loc[0, "gamma_underestimate"]), 0.0)

    def test_original_direct_config_wins_over_replay_version_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "logs" / "x" / "direct_old" / "config.yaml"
            replay = root / "logs" / "x" / "version_0" / "config.yaml"
            original.parent.mkdir(parents=True); replay.parent.mkdir(parents=True)
            original.write_text("name: original\n", encoding="utf-8")
            replay.write_text("name: replay\n", encoding="utf-8")
            self.assertEqual(_historical_config(root), original)


if __name__ == "__main__":
    unittest.main()
