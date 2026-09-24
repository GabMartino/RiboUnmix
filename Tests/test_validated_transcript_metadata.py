from __future__ import annotations

import copy
import pickle
import unittest
from unittest.mock import patch

import numpy as np
import torch
from lightning_fabric.utilities.apply_func import move_data_to_device

from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDataset import (
    RiboUnmixMultiDataset,
)
from Models.RiboUnmixModel.RiboUnmixModel import RiboUnmixModel
from Models.RiboUnmixLightningModule import RiboUnmixLightningModule
from Utils.transcript_batch_metadata import ValidatedTranscriptMetadata


def _collate_fixture(*, with_bias_features: bool = True, execution=None):
    # Exercise the real collator with compact samples, without loading a panel.
    dataset = RiboUnmixMultiDataset.__new__(
        RiboUnmixMultiDataset
    )
    dataset.biological_extra_dim = 1
    dataset.dataset_bias_extra_dim = int(with_bias_features)
    dataset.num_datasets_by_transcript = {"long": 2, "short": 2}
    dataset.base_biological_feature_lut = (
        np.arange(40, dtype=np.float32).reshape(8, 5) / 40.0
    )
    samples = []
    for transcript, dataset_id in (("short", 1), ("long", 0), ("short", 0), ("long", 1)):
        codons = np.asarray(
            [1, 2, 3, 4] if transcript == "long" else [4, 3, 2],
            dtype=np.uint8,
        )
        extra = np.arange(len(codons), dtype=np.float32)[:, None] / 3.0
        profile = np.arange(1, len(codons) + 1, dtype=np.float32) + dataset_id
        sample = (
            dataset_id,
            transcript,
            extra.copy(),
            codons,
            profile,
            [],
            1.0,
            float(dataset_id + 1),
            1.0,
        )
        if with_bias_features:
            sample = (*sample, extra.copy() + 0.25)
        samples.append((*sample, profile[None, :].copy(), execution))
    return dataset, samples


def _batch(*, with_bias_features: bool = True, execution=None):
    dataset, samples = _collate_fixture(
        with_bias_features=with_bias_features, execution=execution
    )
    return dataset.collate_fn(samples)


def _small_model(*, with_bias_features: bool = True, biological_dropout: float = 0.0):
    features = {"bio": {"route": "biological", "dimension": 1}}
    if with_bias_features:
        features["bias"] = {"route": "dataset_bias", "dimension": 1}
    model = RiboUnmixModel(
        {
            "mass_conservation": True,
            "alpha_mode": "learned",
            "init_gamma": 1.0,
            "gamma_centering": {
                "mode": "fixed_reference",
                "reference": {
                    "weighting": "equal",
                    "minimum_datasets": 2,
                    "chunk_size": 1,
                },
            },
            "additional_sequence_features": features,
            "biological_params": {
                "input_size": 5,
                "hidden_size": 4,
                "num_layers": 2,
                "dropout": biological_dropout,
                "init_local_hazard_factor": 1.0,
                "init_local_hazard_weight_std": 1.0e-2,
            },
            "dataset_bias_params": {
                "position_features": ["rel_pos", "stop_window"],
                "position_scale": 100.0,
                "position_edge_tau": 10.0,
                "num_datasets": 2,
                "dataset_embeddings_size": 2,
                "num_codons": 8,
                "codon_embeddings_size": 3,
                "use_nucleotide_amino_acid_embeddings": False,
                "context_gru_hidden_size": 3,
                "context_gru_num_layers": 1,
                "context_gru_dropout": 0.0,
                "dataset_multiplicative_allocation_bias_submodule_params": {
                    "hidden_size": 4,
                    "dropout": 0.25,
                },
                "dataset_log_sigma_submodule_params": {
                    "hidden_size": 4,
                    "dropout": 0.25,
                },
            },
        },
        selected_dataset_names=("d0", "d1"),
        selected_dataset_ids=(0, 1),
        reference_dataset_names=("d0", "d1"),
        reference_dataset_ids=(0, 1),
        reference_dataset_quality_weights=(1.0, 1.0),
    )
    # Neutral final heads would hide errors in recurrent outputs and gradients.
    with torch.no_grad():
        model.dataset_bias_model.observation_bias_head.log_bias_head.weight.normal_(
            mean=0.0, std=0.2
        )
        model.dataset_bias_model.log_sigma_head.ff[-1].weight.normal_(
            mean=0.0, std=0.2
        )
    return model


def _module(model):
    module = RiboUnmixLightningModule.__new__(RiboUnmixLightningModule)
    torch.nn.Module.__init__(module)
    module.model = model
    return module


def _assert_model_outputs_equal(first, second):
    for name in ("mu", "log_sigma"):
        torch.testing.assert_close(first[name], second[name])
    assert first["extras"].keys() == second["extras"].keys()
    for name, value in first["extras"].items():
        other = second["extras"][name]
        if torch.is_tensor(value):
            torch.testing.assert_close(value, other, msg=lambda msg: f"{name}: {msg}")
        else:
            assert value == other, name


def _check_collate_rejects_inconsistent_repeated_inputs_on_cpu(test, field, message):
    dataset, samples = _collate_fixture()
    altered = list(samples[2])
    altered[field] = altered[field].copy()
    altered[field].flat[0] += 1
    samples[2] = tuple(altered)
    with test.assertRaisesRegex(ValueError, message):
        dataset.collate_fn(samples)


def _check_collate_accepts_matching_nan_features_and_keeps_metadata_on_host():
    dataset, samples = _collate_fixture()
    for index in (0, 2):
        samples[index][2][0, 0] = np.nan
        samples[index][9][0, 0] = np.nan
    batch = dataset.collate_fn(samples)
    metadata = batch[-2]
    assert isinstance(metadata, ValidatedTranscriptMetadata)
    assert metadata.group_indices == (0, 0, 1, 1)
    assert metadata.rows_by_transcript == ((0, 1), (2, 3))
    assert metadata.lengths == (4, 4, 3, 3)
    assert metadata.canonical_rows == (0, 2)
    assert metadata.unique_lengths == (4, 3)
    assert batch[-1] is None
    assert pickle.loads(pickle.dumps(metadata)) == metadata

    # A real non-CPU transfer detects accidentally tensorized metadata on a
    # CPU-only test host too. PackedSequence retains its CPU batch_sizes.
    transferred = move_data_to_device(batch, torch.device("meta"))
    assert transferred[0].device.type == "meta"
    assert isinstance(transferred[-2], ValidatedTranscriptMetadata)
    assert transferred[-2] == metadata
    assert all(type(value) is int for value in transferred[-2].lengths)


def _check_validated_path_preserves_outputs_and_gradients(training, biological_dropout):
    torch.manual_seed(31)
    batch = _batch()
    fast_model = _small_model(biological_dropout=biological_dropout)
    legacy_model = copy.deepcopy(fast_model)
    fast_model.train(training)
    legacy_model.train(training)
    fast = _module(fast_model)
    legacy = _module(legacy_model)
    legacy_batch = (*batch[:-2], batch[-1])
    encoder_lengths = []

    def record_lengths(_module, _args, kwargs):
        encoder_lengths.append(kwargs.get("cpu_lengths"))

    hook = fast_model.dataset_bias_model.local_context_gru.register_forward_pre_hook(
        record_lengths, with_kwargs=True
    )
    try:
        torch.manual_seed(57)
        expected = legacy.forward_batch(legacy_batch)
        torch.manual_seed(57)
        with (
            patch.object(
                fast_model,
                "_assert_transcript_tensor_consistency",
                side_effect=AssertionError("Repeated inputs must be checked in CPU collate."),
            ),
            patch.object(
                fast_model,
                "_canonical_transcript_rows",
                side_effect=AssertionError("Canonical rows must come from CPU collate."),
            ),
        ):
            actual = fast.forward_batch(batch)
    finally:
        hook.remove()

    _assert_model_outputs_equal(actual, expected)
    assert encoder_lengths == (
        [(4, 4, 3, 3), (4, 3), (4, 3)] if training else [(4, 4, 3, 3)]
    )
    coefficients = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4) / 16.0
    for output in (actual, expected):
        ((output["mu"] * coefficients).sum() + output["log_sigma"].square().sum()).backward()
    for (name, parameter), (other_name, other_parameter) in zip(
        fast_model.named_parameters(), legacy_model.named_parameters(), strict=True
    ):
        assert name == other_name
        assert (parameter.grad is None) == (other_parameter.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(
                parameter.grad,
                other_parameter.grad,
                msg=lambda msg: f"{name}: {msg}",
            )
    assert fast_model.dataset_bias_model.local_context_gru.rnn.weight_ih_l0.grad.abs().sum() > 0


def _check_lightning_keeps_optional_bias_replicas_and_execution_schema(
    with_bias_features, with_execution
):
    execution = (
        {
            "logical_batch_index": 0,
            "execution_chunk_index": 0,
            "execution_chunk_count": 1,
            "logical_group_count": 2,
            "execution_group_count": 2,
            "logical_pair_count": 4,
        }
        if with_execution else None
    )
    batch = _batch(with_bias_features=with_bias_features, execution=execution)
    module = _module(_small_model(with_bias_features=with_bias_features).eval())
    current = module.forward_batch(batch)
    legacy_batch = (*batch[:-2], batch[-1])
    legacy = module.forward_batch(legacy_batch)
    _assert_model_outputs_equal(current, legacy)
    assert current["execution_microbatch_metadata"] == execution
    assert module._execution_metadata(batch) == execution
    assert current["replica_profiles"].shape == (4, 1, 4)
    assert current["replica_mask"].shape == (4, 1)
    assert (current["dataset_bias_sequence_features"] is not None) == with_bias_features
    if not with_execution:
        # Older manually assembled batches also omit the execution sentinel.
        manual = module.forward_batch(batch[:-2])
        _assert_model_outputs_equal(current, manual)


class ValidatedTranscriptMetadataTests(unittest.TestCase):
    def test_collate_rejects_inconsistent_repeated_inputs_on_cpu(self):
        for field, message in (
            (3, "codon IDs"),
            (2, "biological sequence features"),
            (9, "dataset-bias optional sequence features"),
        ):
            with self.subTest(field=field):
                _check_collate_rejects_inconsistent_repeated_inputs_on_cpu(
                    self, field, message
                )

    def test_collate_accepts_matching_nan_features_and_keeps_metadata_on_host(self):
        _check_collate_accepts_matching_nan_features_and_keeps_metadata_on_host()

    def test_validated_path_preserves_outputs_and_gradients(self):
        for training in (False, True):
            for biological_dropout in (0.0, 0.2):
                with self.subTest(training=training, biological_dropout=biological_dropout):
                    _check_validated_path_preserves_outputs_and_gradients(
                        training, biological_dropout
                    )

    def test_lightning_keeps_optional_bias_replicas_and_execution_schema(self):
        for with_bias_features in (False, True):
            for with_execution in (False, True):
                with self.subTest(
                    with_bias_features=with_bias_features, with_execution=with_execution
                ):
                    _check_lightning_keeps_optional_bias_replicas_and_execution_schema(
                        with_bias_features, with_execution
                    )


if __name__ == "__main__":
    unittest.main()
