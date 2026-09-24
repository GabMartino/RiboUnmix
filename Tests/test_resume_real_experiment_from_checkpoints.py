from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_resume_module():
    path = ROOT / "resume_real_experiment_from_checkpoints.py"
    spec = importlib.util.spec_from_file_location("real_resume_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RESUME = _load_resume_module()


class ResumeCheckpointTests(unittest.TestCase):
    def test_frozen_config_preserves_saved_defaults_and_embedded_dataset_mapping(self):
        import yaml
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            original = {"defaults": [{"dataset_config": "obsolete_group"}, "_self_"],
                        "name": "saved_panel", "model": {"historical_parameter": 17},
                        "orchestrator": {"run_id": "saved_run", "N": 114},
                        "dataset_config": {"dataset_path": {"d1": "data.parquet"}},
                        "paths": {"sequences_path": "/old/project/sequences.parquet"}}
            source = task / "resolved_config.yaml"
            source.write_text(yaml.safe_dump(original))
            before = source.read_bytes()
            launch = ("/usr/bin/python3 -u /old/project/main_ribounmix_multidataset.py "
                      "--config-path=/old/project/config --config-name=base "
                      "dataset_config=obsolete_group name=saved_panel "
                      "+orchestrator.run_id=exp8_qrank_N114_seed42 +orchestrator.N=114")
            (task / "launch_command.sh").write_text(launch + "\n")
            command = RESUME._read_launch_command(task / "launch_command.sh")
            audit = RESUME._use_saved_resolved_config(command, task)
            frozen = yaml.safe_load(Path(audit["snapshot"]).read_text())
            self.assertEqual(frozen["model"], original["model"])
            self.assertEqual(frozen["dataset_config"], original["dataset_config"])
            self.assertNotIn("defaults", frozen)
            self.assertNotIn("dataset_config=obsolete_group", command)
            self.assertIn("++orchestrator.run_id=exp8_qrank_N114_seed42", command)
            self.assertIn("++orchestrator.N=114", command)
            self.assertNotIn("+orchestrator.run_id=exp8_qrank_N114_seed42", command)
            self.assertEqual(frozen["paths"]["sequences_path"], str(ROOT / "sequences.parquet"))
            self.assertEqual(source.read_bytes(), before)
            self.assertTrue(audit["hydra_composition_checked"])

    def test_dataset_size_and_explicit_run_filters_form_a_union(self) -> None:
        command = [
            sys.executable,
            "-u",
            str(ROOT / "main_ribounmix_multidataset.py"),
            "name=real_exp8_N040_pair01_A",
            "+orchestrator.N=40",
        ]
        size = RESUME._task_dataset_size(
            command,
            Path("N040/pair01_A"),
            "real_exp8_N040_pair01_A",
        )
        self.assertEqual(size, 40)
        self.assertTrue(
            RESUME._matches_task_filter(
                run_id="real_exp8_N040_pair01_A",
                dataset_size=size,
                dataset_sizes=(40, 80, 114),
                include_run_ids=("real_exp8_N010_pair01_B",),
            )
        )
        self.assertTrue(
            RESUME._matches_task_filter(
                run_id="real_exp8_N010_pair01_B",
                dataset_size=10,
                dataset_sizes=(40, 80, 114),
                include_run_ids=("real_exp8_N010_pair01_B",),
            )
        )
        self.assertFalse(
            RESUME._matches_task_filter(
                run_id="real_exp8_N020_pair01_A",
                dataset_size=20,
                dataset_sizes=(40, 80, 114),
                include_run_ids=("real_exp8_N010_pair01_B",),
            )
        )

    def test_corrupt_checkpoint_is_rejected_and_full_state_is_preferred(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            checkpoint_dir = task / "checkpoints"
            checkpoint_dir.mkdir()
            full = checkpoint_dir / "val-loss.ckpt"
            torch.save(
                {
                    "epoch": 4,
                    "global_step": 50,
                    "optimizer_states": [{"state": {}}],
                    "lr_schedulers": [{}],
                },
                full,
            )
            # A later weights-only artifact must not displace an exact resume.
            torch.save(
                {
                    "epoch": 5,
                    "global_step": 60,
                    "optimizer_states": [],
                    "lr_schedulers": [],
                },
                checkpoint_dir / "pcc.ckpt",
            )
            (checkpoint_dir / "last.ckpt").write_bytes(b"")

            selected, metadata, rejected = RESUME._select_resume_checkpoint(task)

            self.assertEqual(selected, full)
            self.assertEqual(metadata["global_step"], 50)
            self.assertTrue(metadata["has_optimizer_state"])
            self.assertEqual(len(rejected), 1)
            self.assertIn("empty", rejected[0]["reason"])

    def test_prepared_task_keeps_command_as_argv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_dir = root / "panel_01"
            task_dir.mkdir()
            launch = task_dir / "launch_command.sh"
            launch.write_text(
                "#!/usr/bin/env bash\n"
                f"{sys.executable} -u "
                f"{ROOT / 'main_ribounmix_multidataset.py'} "
                f"--config-path={ROOT / 'config'} name=test_panel\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                allow_weights_only_warm_resume=False,
                throughput_profile="unchanged",
                max_pair_rows_per_forward=None,
                max_padded_codon_tokens_per_forward=None,
                reference_chunk_size=None,
                log_every_n_steps=None,
                data_num_workers=None,
            )

            task = RESUME._prepare_task(args, root, launch)

            self.assertIsInstance(task["command"], list)
            self.assertEqual(task["command"][1], "-u")
            self.assertEqual(task["resume_mode"], "fresh_start")
            self.assertNotIn("trainer.detect_anomaly=true", task["command"])
            self.assertIsNone(task["bias_gru_precision_override"])
            self.assertEqual(task["gru_compute_policy"], "cuda-amp-gru-fp32-v1")

            args.detect_anomaly = True
            diagnostic_task = RESUME._prepare_task(args, root, launch)
            self.assertEqual(
                diagnostic_task["command"],
                task["command"] + ["trainer.detect_anomaly=true"],
            )

            args.bias_gru_precision = "float32"
            fixed = RESUME._prepare_task(args, root, launch)
            override = "++model.dataset_bias_params.context_gru_precision=float32"
            self.assertEqual(fixed["command"], diagnostic_task["command"] + [override])
            self.assertEqual(fixed["bias_gru_precision_source"], "cli")
            saved = json.loads((task_dir / "resume_manifest.json").read_text())
            self.assertEqual(saved["bias_gru_precision_override"], "float32")

            args.bias_gru_precision = None
            repeated = RESUME._prepare_task(args, root, launch)
            self.assertEqual(repeated["command"], fixed["command"])
            self.assertEqual(repeated["bias_gru_precision_source"], "previous_resume_manifest")

            args.bias_gru_precision = "inherit"
            restored = RESUME._prepare_task(args, root, launch)
            self.assertNotIn(override, restored["command"])
            self.assertIn("++model.dataset_bias_params.context_gru_precision=inherit", restored["command"])

    def test_override_replaces_hydra_add_forms_and_rejects_duplicates(self):
        key = "model.dataset_bias_params.context_gru_precision"
        for prefix in ("", "+", "++"):
            command = [sys.executable, f"{prefix}{key}=inherit"]
            RESUME._set_override(command, f"++{key}", "float32")
            self.assertEqual(command, [sys.executable, f"++{key}=float32"])
        with self.assertRaisesRegex(ValueError, "duplicate override"):
            RESUME._set_override([f"{key}=inherit", f"++{key}=float32"], key, "float32")

    def test_tbptt_resume_policy_is_explicit_persistent_and_checkpointed(self):
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            launch = task / 'launch_command.sh'
            launch.write_text(f'{sys.executable} {ROOT / "main_ribounmix_multidataset.py"} name=tbptt_test\n')
            original = launch.read_bytes()
            (task / 'checkpoints').mkdir()
            checkpoint = task / 'checkpoints/last.ckpt'
            torch.save({'epoch': 19, 'global_step': 8040, 'optimizer_states': [{'state': {}}],
                        'hyper_parameters': {'bias_gru_tbptt_window': 512}}, checkpoint)
            args = RESUME.parse_args(['--run-root', temporary, '--dry-run'])
            restored = RESUME._prepare_task(args, task, launch)
            self.assertEqual(restored['bias_gru_tbptt_window_override'], 512)
            self.assertEqual(restored['bias_gru_tbptt_window_source'], 'checkpoint')
            args.bias_gru_tbptt_window = 1024
            changed = RESUME._prepare_task(args, task, launch)
            self.assertIn('++model.dataset_bias_params.context_gru_tbptt_window=1024', changed['command'])
            self.assertEqual(changed['checkpoint']['global_step'], 8040)
            self.assertEqual(changed['bias_gru_tbptt_window_source'], 'cli')
            self.assertIn('biased training gradient', changed['training_gradient_note'])
            args.bias_gru_tbptt_window = None
            repeated = RESUME._prepare_task(args, task, launch)
            self.assertEqual(repeated['command'], changed['command'])
            self.assertEqual(repeated['bias_gru_tbptt_window_source'], 'previous_resume_manifest')
            args.capture_gru_failure = True
            with self.assertRaisesRegex(ValueError, 'not compatible'):
                RESUME._prepare_task(args, task, launch)
            args.capture_gru_failure = False
            args.bias_gru_tbptt_window = 0
            disabled = RESUME._prepare_task(args, task, launch)
            self.assertIn('++model.dataset_bias_params.context_gru_tbptt_window=0', disabled['command'])
            args.bias_gru_tbptt_window = None
            repeated = RESUME._prepare_task(args, task, launch)
            self.assertEqual(repeated['bias_gru_tbptt_window_override'], 0)
            self.assertEqual(launch.read_bytes(), original)

    def test_n114_dry_run_keeps_full_state_and_other_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for n in (40, 114):
                task_dir = root / f"N{n:03d}" / "full"
                task_dir.mkdir(parents=True)
                (task_dir / "launch_command.sh").write_text(
                    f"{sys.executable} -u "
                    f"{ROOT / 'main_ribounmix_multidataset.py'} "
                    f"name=real_exp8_N{n:03d}_full +orchestrator.N={n} "
                    "data.batch_size=32 loss.experiment_mode=standard_nb "
                    "trainer.precision=bf16-mixed "
                    "model.gamma_centering.reference.weighting=equal\n",
                    encoding="utf-8",
                )
            full = root / "N114" / "full"
            checkpoints = full / "checkpoints"
            checkpoints.mkdir()
            checkpoint = checkpoints / "last.ckpt"
            torch.save({"epoch": 8, "global_step": 3753, "optimizer_states": [{"state": {}}],
                        "lr_schedulers": [{}]}, checkpoint)
            with patch.object(RESUME, "_launch") as launch, contextlib.redirect_stdout(io.StringIO()):
                status = RESUME.main([
                    "--run-root", str(root), "--dataset-sizes", "114", "--gpus", "0",
                    "--bias-gru-precision", "float32", "--data-num-workers", "0", "--dry-run",
                ])
                launch.assert_not_called()
            self.assertEqual(status, 0)
            saved = json.loads((full / "resume_manifest.json").read_text())
            self.assertEqual(saved["resume_mode"], "exact_full_state")
            self.assertEqual(saved["checkpoint"]["global_step"], 3753)
            self.assertIn("experiment.resume_training_state=true", saved["command"])
            self.assertIn("experiment.allow_weights_only_resume=false", saved["command"])
            for setting in ("data.batch_size=32", "loss.experiment_mode=standard_nb",
                            "trainer.precision=bf16-mixed", "model.gamma_centering.reference.weighting=equal"):
                self.assertIn(setting, saved["command"])
            self.assertEqual(saved["bias_gru_precision_override"], "float32")
            self.assertFalse((root / "N040" / "full" / "resume_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
