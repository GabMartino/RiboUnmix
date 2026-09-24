from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analyses.analyze_benchmark_loss_ablation import paired_effects, transcript_metrics
from run_benchmark_loss_ablation import (
    DEFAULT_DESIGN,
    build_tasks,
    build_training_command,
    load_design,
    resolve_python,
)


class BenchmarkLossAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.design = load_design(DEFAULT_DESIGN)

    def test_default_matrix_is_four_arms_by_four_datasets(self) -> None:
        tasks = build_tasks(self.design, (42,))
        self.assertEqual(len(tasks), 16)
        self.assertEqual(tasks[0].task_id, "seed42__full__human_iwasaki_2014")
        self.assertEqual(tasks[3].task_id, "seed42__full__ecoli_zhang_2016")
        self.assertEqual(tasks[4].task_id, "seed42__nb_only__human_iwasaki_2014")
        self.assertEqual(tasks[-1].task_id, "seed42__nb_vst_pcc__ecoli_zhang_2016")
        self.assertEqual(
            {
                (
                    task.replica_nb_weight,
                    task.consensus_raw_pcc_weight,
                    task.consensus_nb_vst_pcc_weight,
                )
                for task in tasks
            },
            {(1.0, 0.5, 0.5), (1.0, 0.0, 0.0), (1.0, 0.5, 0.0), (1.0, 0.0, 0.5)},
        )

    def test_multiple_training_seeds_keep_split_seed_fixed(self) -> None:
        tasks = build_tasks(self.design, (42, 43, 44))
        self.assertEqual(len(tasks), 48)
        self.assertEqual({task.split_seed for task in tasks}, {42})
        self.assertEqual({task.training_seed for task in tasks}, {42, 43, 44})
        self.assertEqual(tasks[16].task_id, "seed43__full__human_iwasaki_2014")

    def test_command_changes_only_declared_loss_coefficients(self) -> None:
        task = build_tasks(self.design, (42,))[4]
        with tempfile.TemporaryDirectory() as temporary:
            command = build_training_command(
                python_executable=Path("/usr/bin/python3"),
                task=task,
                attempt_dir=Path(temporary),
                design=self.design,
                num_workers=8,
                predict_num_workers=0,
            )
        self.assertIn("loss.replica_nb_weight=1.0", command)
        self.assertIn("loss.consensus_raw_pcc_weight=0.0", command)
        self.assertIn("loss.consensus_nb_vst_pcc_weight=0.0", command)
        self.assertIn("loss.gamma_reg_weight=0.0001", command)
        self.assertIn("split.seed=42", command)
        self.assertIn(
            "prediction.checkpoint_variants=[best_nb_nll,best_val_loss,best_pcc]",
            command,
        )

    def test_python_resolution_preserves_virtual_environment_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            venv_python = Path(temporary) / "venv" / "bin" / "python"
            venv_python.parent.mkdir(parents=True)
            venv_python.symlink_to(Path(sys.executable).resolve())

            selected = resolve_python(str(venv_python))

            self.assertEqual(selected, venv_python.absolute())
            self.assertNotEqual(str(selected), str(venv_python.resolve()))

    def test_perfect_profile_metrics_have_unit_correlations_and_zero_error(self) -> None:
        values = np.asarray([0.0, 1.0, 4.0, 2.0], dtype=np.float64)
        metrics = transcript_metrics(
            {
                "transcript_id": "t1",
                "target": values,
                "mu": values,
                "mask": np.ones(values.size, dtype=bool),
                "log_sigma": np.full(values.size, -2.0),
            }
        )
        self.assertAlmostEqual(metrics["raw_pcc"], 1.0)
        self.assertAlmostEqual(metrics["nb_vst_pcc"], 1.0)
        self.assertAlmostEqual(metrics["rmse"], 0.0)
        self.assertAlmostEqual(metrics["relative_rmse"], 0.0)
        self.assertAlmostEqual(metrics["mean_ratio"], 1.0)
        self.assertTrue(np.isfinite(metrics["nb2_nll"]))

    def test_effect_is_difference_of_paired_transcript_means(self) -> None:
        rows = []
        for transcript, full, ablated in (("t1", 0.2, 0.3), ("t2", 0.6, 0.9)):
            for arm, raw in (("full", full), ("nb_only", ablated)):
                rows.append(
                    {
                        "dataset": "human_iwasaki_2014",
                        "training_seed": 42,
                        "arm": arm,
                        "transcript_id": transcript,
                        "target_hash": transcript,
                        "mask_hash": transcript,
                        "raw_pcc": raw,
                        "nb_vst_pcc": raw - 0.05,
                        "nb2_nll": 1.0 - raw,
                    }
                )
        effects, seed_effects, exclusions = paired_effects(
            pd.DataFrame(rows),
            design=self.design,
            draws=200,
            seed=7,
        )
        raw = effects[
            (effects["arm"] == "nb_only") & (effects["metric"] == "raw_pcc")
        ].iloc[0]
        self.assertAlmostEqual(float(raw["estimate"]), 0.2)
        self.assertEqual(int(raw["n_transcripts"]), 2)
        self.assertFalse(seed_effects.empty)
        self.assertEqual(int(exclusions.iloc[0]["excluded_nonfinite_or_unmatched"]), 0)


if __name__ == "__main__":
    unittest.main()
