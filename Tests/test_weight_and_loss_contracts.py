from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDataset import (
    RiboUnmixMultiDataset,
)
from Models.RiboUnmixLightningModule import (
    NegativeBinomialProfileLoss,
    RiboUnmixLightningModule,
    masked_pcc,
    reduce_dataset_balanced_weighted_mean,
)
from main_ribounmix_multidataset import load_weights_only


def _load_weighting_module():
    path = ROOT / "Datasets" / "data" / "weight_hek_riboseq_codon_replicas.py"
    spec = importlib.util.spec_from_file_location("weight_hek_test_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import weighting script from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_loss_test_module() -> RiboUnmixLightningModule:
    """Build the loss helpers without constructing the full trainable model."""
    module = RiboUnmixLightningModule.__new__(
        RiboUnmixLightningModule
    )
    nn.Module.__init__(module)
    module.loss_fn = NegativeBinomialProfileLoss(
        log_alpha_min=-5.0,
        log_alpha_max=1.0,
    )
    module.replica_nb_weight = 1.0
    module.consensus_raw_pcc_weight = 0.5
    module.consensus_nb_vst_pcc_weight = 0.5
    module.pcc_detach_alpha = True
    module.min_pcc_target_var = 1.0e-6
    module.pcc_prediction_floor = 0.0
    module.eps = 1.0e-8
    return module


class WeightAndLossContractTests(unittest.TestCase):
    def test_legacy_replica_pcc_coefficients_raise(self) -> None:
        loss = SimpleNamespace(
            eps=1.0e-8,
            nb_log_alpha_min=-5.0,
            nb_log_alpha_max=1.0,
            nb_sequence_reduction="mean",
            nb_length_temper_gamma=0.75,
            nb_length_temper_ref=1000.0,
            nb_length_temper_min_weight=0.5,
            nb_length_temper_max_weight=2.0,
            replica_nb_weight=1.0,
            replica_raw_pcc_weight=0.5,
            replica_nb_vst_pcc_weight=0.5,
            gamma_reg_weight=0.0,
            sample_reduction="transcript_balanced",
        )
        with self.assertRaisesRegex(ValueError, "arithmetic replica consensus"):
            RiboUnmixLightningModule(
                nn.Identity(),
                SimpleNamespace(loss=loss),
                {},
            )

    def test_checkpoint_loading_rejects_partial_state(self) -> None:
        model = nn.Linear(2, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partial.ckpt"
            torch.save({"state_dict": {"weight": model.weight.detach().clone()}}, path)
            with self.assertRaisesRegex(RuntimeError, "Partial or legacy"):
                load_weights_only(model, path)

    def test_zero_coverage_row_is_absent_and_weights_above_one_survive(self) -> None:
        weighting = _load_weighting_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "synthetic.parquet"
            output_path = root / "weighted.parquet"
            pd.DataFrame(
                {
                    "id": ["t1", "t2", "t3"],
                    "ribo": [
                        np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
                        np.asarray([0.0, 2.0, 0.0], dtype=np.float32),
                        np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
                    ],
                    "ribo_cds_replicas": [
                        [np.asarray([0.0, 0.0, 0.0], dtype=np.float32)],
                        [np.asarray([0.0, 2.0, 0.0], dtype=np.float32)],
                        [np.asarray([1.0, 1.0, 1.0], dtype=np.float32)],
                    ],
                    "replica_ids": [["r1"], ["r1"], ["r1"]],
                }
            ).to_parquet(input_path, index=False)

            summary = weighting.add_weights(input_path, output_path)
            output = pd.read_parquet(output_path)

            self.assertEqual(output["id"].tolist(), ["t2", "t3"])
            self.assertEqual(summary["removed_transcripts"], 1)
            self.assertTrue(bool((output["weight"] > 0.0).all()))
            self.assertGreater(float(output["weight"].max()), 1.0)
            self.assertAlmostEqual(float(output["weight"].median()), 1.0, places=6)
            for column in ("weight", "weight_raw", "coverage", "read_density"):
                self.assertEqual(output[column].dtype, np.dtype("float32"))

    def test_runtime_rejects_a_zero_transcript_weight(self) -> None:
        dataset = RiboUnmixMultiDataset.__new__(
            RiboUnmixMultiDataset
        )
        dataset._sample_weight_cache = {}
        dataset.data_records = {"sample_weights": {"t": {"d": 0.0}}}
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            dataset._get_sample_weight("t", "d")

    def test_dataset_balanced_reducer_preserves_weights_and_gradients(self) -> None:
        values = torch.tensor([1.0, 3.0, 10.0], requires_grad=True)
        dataset_ids = torch.tensor([0, 0, 1])
        weights = torch.tensor([3.0, 1.0, 2.0])

        result = reduce_dataset_balanced_weighted_mean(
            values,
            dataset_ids,
            weights,
        )
        self.assertAlmostEqual(float(result.detach()), 5.75)
        result.backward()
        torch.testing.assert_close(
            values.grad,
            torch.tensor([0.375, 0.125, 0.5]),
        )

    def test_dataset_balanced_reducer_rejects_zero_weight(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            reduce_dataset_balanced_weighted_mean(
                torch.tensor([1.0, 2.0]),
                torch.tensor([0, 1]),
                torch.tensor([1.0, 0.0]),
            )

    def test_reducer_accumulates_mixed_precision_values_in_float32(self) -> None:
        values = torch.tensor([1.0, 3.0], dtype=torch.float16, requires_grad=True)
        result = reduce_dataset_balanced_weighted_mean(
            values,
            torch.tensor([0, 0]),
            torch.tensor([1.25, 0.75]),
        )
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(result)))
        result.backward()
        self.assertTrue(bool(torch.isfinite(values.grad).all()))

    def test_nb_vst_uses_the_same_dispersion_bounds_as_nb(self) -> None:
        module = RiboUnmixLightningModule.__new__(
            RiboUnmixLightningModule
        )
        nn.Module.__init__(module)
        module.loss_fn = NegativeBinomialProfileLoss(
            log_alpha_min=-5.0,
            log_alpha_max=1.0,
        )
        module.pcc_detach_alpha = False
        alpha = module._pcc_alpha(
            torch.tensor([-100.0, 100.0]),
            target_shape=(2,),
        )
        torch.testing.assert_close(
            alpha,
            torch.exp(torch.tensor([-5.0, 1.0])),
        )

    def test_pcc_flat_prediction_has_finite_bounded_gradient(self) -> None:
        """A nearly flat prediction must not create a singular PCC gradient."""
        prediction = torch.zeros(1, 1_000, dtype=torch.float32, requires_grad=True)
        target = torch.linspace(-1.0, 1.0, 1_000, dtype=torch.float32).reshape(1, -1)
        target = 100.0 * (target + 1.0)
        result = masked_pcc(
            prediction,
            target,
            torch.ones_like(prediction, dtype=torch.bool),
            min_target_var=1.0e-6,
            eps=1.0e-8,
        )
        self.assertTrue(bool(result["valid"].all()))
        (1.0 - result["pcc_per_sample"].mean()).backward()

        self.assertIsNotNone(prediction.grad)
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
        # The old summed-variance denominator yields ~1e10 in this case.
        self.assertLess(float(prediction.grad.abs().max()), 1.0e4)

    def test_pcc_matches_ordinary_pearson_when_well_conditioned(self) -> None:
        prediction = torch.linspace(0.5, 1.5, 1_000, dtype=torch.float32).reshape(1, -1)
        target = torch.linspace(0.2, 1.3, 1_000, dtype=torch.float32).reshape(1, -1)
        result = masked_pcc(
            prediction,
            target,
            torch.ones_like(prediction, dtype=torch.bool),
        )
        expected = torch.corrcoef(torch.cat((prediction, target), dim=0))[0, 1]
        torch.testing.assert_close(
            result["pcc_per_sample"],
            expected.reshape(1),
            rtol=1.0e-5,
            atol=1.0e-6,
        )

    def test_replica_average_excludes_masked_replicas(self) -> None:
        values = torch.tensor([1.0, 100.0, 100.0, 3.0])
        valid = torch.tensor([[True, False], [False, True]])
        result = RiboUnmixLightningModule._average_over_valid_replicas(
            values,
            valid,
        )
        torch.testing.assert_close(result, torch.tensor([1.0, 3.0]))

    def test_pcc_uses_arithmetic_replica_consensus_once(self) -> None:
        module = _make_loss_test_module()
        replicas = torch.tensor(
            [[[4.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 4.0]]]
        )
        consensus = replicas.mean(dim=1)
        mask = torch.ones_like(consensus, dtype=torch.bool)
        log_sigma = torch.zeros_like(consensus)

        consensus_terms = module._compute_consensus_loss_terms(
            {
                "mu": consensus.clone().requires_grad_(True),
                "target": consensus,
                "mask": mask,
                "log_sigma": log_sigma,
            },
            consensus,
            mask,
        )
        self.assertLess(
            float(consensus_terms["raw_pcc_diag"]["loss_per_sample"].item()),
            1.0e-6,
        )
        self.assertLess(
            float(
                consensus_terms["nb_vst_pcc_diag"]["loss_per_sample"].item()
            ),
            1.0e-6,
        )

        # This is the old computation: compare the same pair prediction with
        # each raw replica and then average. It is intentionally different.
        replica_level = module._pcc_loss_per_sample(
            {
                "mu": consensus.expand(2, -1),
                "target": replicas.reshape(2, 4),
                "mask": torch.ones(2, 4, dtype=torch.bool),
                "log_sigma": log_sigma.expand(2, -1),
            },
            transform="raw",
        )
        self.assertGreater(
            float(replica_level["loss_per_sample"].mean().item()),
            0.1,
        )

    def test_validation_transcript_pcc_distribution_is_unweighted(self) -> None:
        module = _make_loss_test_module()
        module.log_validation_transcript_mu_pcc_distribution = True
        module._validation_transcript_mu_pcc = {}
        module._record_validation_transcript_mu_pcc(
            {
                "ids": ["t1", "t1", "t2"],
                "dataset_ids": torch.tensor([0, 1, 0]),
                "transcript_group_index": torch.tensor([0, 0, 1]),
            },
            {
                "mu_pcc_per_sample": torch.tensor([0.2, 0.8, -0.4]),
            },
        )

        transcript_ids, values = (
            module._collect_validation_transcript_mu_pcc()
        )
        self.assertEqual(transcript_ids, ["t1", "t2"])
        # t1 is the equal arithmetic mean of its two dataset observations;
        # transcript reliability weights are deliberately not consulted.
        torch.testing.assert_close(values, torch.tensor([0.5, -0.4]))

    def test_replica_loss_helper_keeps_only_raw_replica_nb(self) -> None:
        module = _make_loss_test_module()
        replicas = torch.tensor(
            [[[4.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 4.0]]]
        )
        consensus = replicas.mean(dim=1)
        mask = torch.ones_like(consensus, dtype=torch.bool)
        out = {
            "replica_profiles": replicas,
            "replica_mask": torch.tensor([[True, True]]),
            "mask": mask,
            "log_sigma": torch.zeros_like(consensus),
            "extras": {"normalized_shape": consensus},
        }

        result = module._compute_replica_loss_terms(
            out,
            optimize_with_reweighted_nb=True,
        )
        self.assertIn("nll_per_sample", result)
        self.assertNotIn("raw_pcc_diag", result)
        self.assertNotIn("nb_vst_pcc_diag", result)

        flat_target = replicas.reshape(2, 4)
        expected = module.loss_fn(
            mu_phys=consensus.expand(2, -1),
            log_sigma=torch.zeros_like(flat_target),
            y_true=flat_target,
            mask=torch.ones_like(flat_target, dtype=torch.bool),
            return_per_sample=True,
        ).mean()
        torch.testing.assert_close(result["nll_per_sample"], expected.reshape(1))


if __name__ == "__main__":
    unittest.main()
