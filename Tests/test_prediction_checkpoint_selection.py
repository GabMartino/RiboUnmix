import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from main_ribounmix_multidataset import (
    find_prediction_checkpoint,
    resolve_prediction_checkpoint_variants,
)


class PredictionCheckpointSelectionTests(unittest.TestCase):
    def test_finds_best_loss_and_best_pcc_independently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            loss_worse = root / "val-loss-epoch=3-val_loss=1.5000.ckpt"
            loss_best = root / "val-loss-epoch=7-val_loss=1.2000.ckpt"
            pcc_worse = root / "pcc-epoch=4-val_mu_pcc=0.7000.ckpt"
            pcc_best = root / "pcc-epoch=6-val_mu_pcc=0.8000.ckpt"
            for path in (loss_worse, loss_best, pcc_worse, pcc_best):
                path.touch()

            self.assertEqual(
                find_prediction_checkpoint(root, "best_val_loss"),
                str(loss_best),
            )
            self.assertEqual(
                find_prediction_checkpoint(root, "best_pcc"),
                str(pcc_best),
            )

    def test_last_checkpoint_is_never_used_as_a_metric_best(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "last.ckpt").touch()
            self.assertIsNone(find_prediction_checkpoint(root, "best_val_loss"))
            self.assertIsNone(find_prediction_checkpoint(root, "best_pcc"))

    def test_configured_variants_are_ordered_and_validated(self) -> None:
        cfg = OmegaConf.create(
            {
                "prediction": {
                    "checkpoint_variants": ["best_val_loss", "best_pcc"]
                }
            }
        )
        self.assertEqual(
            resolve_prediction_checkpoint_variants(cfg),
            ("best_val_loss", "best_pcc"),
        )

        duplicate = OmegaConf.create(
            {"prediction": {"checkpoint_variants": ["best_pcc", "best_pcc"]}}
        )
        with self.assertRaisesRegex(ValueError, "duplicates"):
            resolve_prediction_checkpoint_variants(duplicate)


if __name__ == "__main__":
    unittest.main()
