"""Regression checks for the reviewer-facing synthetic smoke fixture."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "synthetic_smoke"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PublicSmokeExampleTests(unittest.TestCase):
    def test_published_data_match_manifest(self) -> None:
        manifest = json.loads(
            (EXAMPLE / "data_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["selection"]["transcript_count"], 64)
        self.assertEqual(len(manifest["selection"]["transcript_ids"]), 64)
        for relative_path, metadata in manifest["published_files"].items():
            path = EXAMPLE / relative_path
            self.assertTrue(path.is_file())
            self.assertEqual(path.stat().st_size, metadata["bytes"])
            self.assertEqual(sha256(path), metadata["sha256"])

    def test_smoke_config_is_cpu_bounded(self) -> None:
        with initialize_config_dir(
            version_base=None,
            config_dir=str(ROOT / "config"),
        ):
            cfg = compose(config_name="config_ribounmix_smoke")
        self.assertEqual(cfg.trainer.accelerator, "cpu")
        self.assertEqual(cfg.trainer.precision, "32-true")
        self.assertEqual(cfg.trainer.max_epochs, 2)
        self.assertEqual(cfg.data.num_workers, 0)
        self.assertEqual(len(cfg.experiment.dataset), 2)
        self.assertEqual(cfg.model.gamma_centering.reference.weighting, "equal")

    def test_reference_checkpoint_and_predictions(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(EXAMPLE / "verify_reference.py")],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(completed.stdout)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["prediction_rows"], 32)
        self.assertEqual(report["validation_transcripts"], 16)


if __name__ == "__main__":
    unittest.main()
