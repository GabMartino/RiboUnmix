"""Matched-config and split tests; model gradient tests reuse reference-campaign probes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from run_synthetic_gamma_unity import ROOT, make_config, validate_source
from Utils.external_transcript_split import assert_expected_transcript_split


class SyntheticUnityDesignTests(unittest.TestCase):
    def source(self):
        cfg = yaml.safe_load((ROOT / "config/config_ribounmix_synthetic.yaml").read_text())
        cfg["dataset_config"] = {"_name_": "synthetic_2_per_codon"}
        cfg["model"]["mass_conservation"] = False
        cfg["loss"].update(experiment_mode="standard_nb", nb_mean_gradient_beta=0.)
        cfg["model"]["gamma_centering"]["reference"]["weighting"] = "equal"
        cfg["training"]["grouped_optimizer_batch"]["resolved"] = {"old_runtime": True}
        return cfg

    def split(self, cfg):
        return dict(train_ids=["t1", "t2"], validation_ids=["v1"],
                    experiment_datasets=cfg["experiment"]["dataset"])

    def test_clone_preserves_all_scientific_settings_except_gamma(self):
        source = self.source()
        original = copy.deepcopy(source)
        cfg = make_config(source, arm="unity", task_root=Path("/new"), split_path=Path("/split.json"))
        self.assertEqual(source, original)
        expected_model = dict(source["model"], mean_correction="unity")
        self.assertEqual(cfg["model"], expected_model)
        for key in ("data", "loss", "optim", "metrics", "dataset_config", "synthetic_ground_truth"):
            self.assertEqual(cfg[key], source[key])
        self.assertEqual(cfg["trainer"], source["trainer"])
        self.assertEqual(cfg["training"]["execution_microbatching"], source["training"]["execution_microbatching"])
        self.assertNotIn("resolved", cfg["training"]["grouped_optimizer_batch"])
        self.assertFalse(cfg["experiment"]["from_checkpoint"])
        self.assertEqual(cfg["prediction"]["checkpoint_variants"], ["best_val_loss"])

    def test_contemporaneous_control_only_changes_intervention_and_name(self):
        cfg = self.source()
        kwargs = dict(task_root=Path("/new"), split_path=Path("/split.json"))
        a = make_config(cfg, arm="unity", **kwargs)
        b = make_config(cfg, arm="learned", **kwargs)
        a["model"]["mean_correction"] = "learned"
        a["name"] = b["name"]
        self.assertEqual(a, b)

    def test_wrong_source_loss_rejected(self):
        cfg = self.source()
        self.assertTrue(all(validate_source(cfg, self.split(cfg), 2).values()))
        cfg["loss"]["experiment_mode"] = "mean_gradient_reweighted_nb"
        with self.assertRaisesRegex(ValueError, "violates design"):
            validate_source(cfg, self.split(cfg), 2)

    def test_depth_specific_source_validation(self):
        for name in (
            "synthetic_0p25_per_codon",
            "synthetic_2_per_codon",
            "synthetic_20_per_codon",
        ):
            cfg = self.source()
            cfg["dataset_config"]["_name_"] = name
            checks = validate_source(
                cfg,
                self.split(cfg),
                2,
                expected_dataset_config=name,
            )
            self.assertTrue(all(checks.values()))
            with self.assertRaisesRegex(ValueError, "violates design"):
                validate_source(
                    cfg,
                    self.split(cfg),
                    2,
                    expected_dataset_config="different_depth",
                )

    def check_split(self, payload, train=None, val=None, datasets=None):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            path.write_text(json.dumps(payload))
            assert_expected_transcript_split(path,
                train_ids=payload["train_ids"] if train is None else train,
                validation_ids=payload["validation_ids"] if val is None else val,
                experiment_datasets=payload["experiment_datasets"] if datasets is None else datasets)

    def test_exact_split_passes(self):
        self.check_split(self.split(self.source()))

    def test_split_reordering_rejected(self):
        with self.assertRaisesRegex(ValueError, "historical split"):
            self.check_split(self.split(self.source()), train=["t2", "t1"])

    def test_validation_membership_drift_rejected(self):
        with self.assertRaisesRegex(ValueError, "historical split"):
            self.check_split(self.split(self.source()), val=["v2"])

    def test_leakage_rejected(self):
        payload = self.split(self.source())
        payload["validation_ids"] = ["t1"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.check_split(payload)

    def test_wrong_panel_rejected(self):
        with self.assertRaisesRegex(ValueError, "different experiment"):
            self.check_split(self.split(self.source()), datasets=["different"])


if __name__ == "__main__":
    unittest.main()
