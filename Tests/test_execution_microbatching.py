import numpy as np
import torch

from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDatasetDataModule import (
    TranscriptGroupedMultiDatasetBatchSampler,
)
from Models.RiboUnmixLightningModule import reduce_per_sample_quantity


def test_execution_microbatching_preserves_complete_transcript_groups():
    transcript_ids = np.asarray(
        [transcript for transcript in ("t0", "t1", "t2", "t3") for _ in range(3)]
    )
    dataset_ids = np.asarray([dataset for _ in range(4) for dataset in range(3)])
    sampler = TranscriptGroupedMultiDatasetBatchSampler(
        flat_transcript_ids=transcript_ids,
        flat_dataset_ids=dataset_ids,
        lengths=np.full(12, 100, dtype=np.int64),
        batch_size=2,
        seed=42,
        sort_by_length=True,
        shuffle_batches=False,
        require_multidataset=True,
        minimum_distinct_datasets=2,
        execution_microbatch_max_transcript_groups=1,
    )

    statistics = sampler.preview_epoch_batch_statistics()
    execution_batches = list(iter(sampler))

    assert statistics.number_of_microbatches == 2
    assert statistics.unique_transcripts_per_microbatch == (2, 2)
    assert len(execution_batches) == 4
    for batch in execution_batches:
        flat_indices = np.asarray([encoded_index[1] for encoded_index in batch])
        assert len(set(transcript_ids[flat_indices])) == 1
        assert set(dataset_ids[flat_indices]) == {0, 1, 2}
        assert all(encoded_index[0] == "execution_microbatch_v1" for encoded_index in batch)
        assert all(encoded_index[6] == 1 for encoded_index in batch)


def test_pair_row_budget_adapts_to_dataset_count_without_splitting_groups():
    transcript_ids = np.asarray(
        [transcript for transcript in ("t0", "t1", "t2", "t3") for _ in range(3)]
    )
    dataset_ids = np.asarray([dataset for _ in range(4) for dataset in range(3)])
    sampler = TranscriptGroupedMultiDatasetBatchSampler(
        flat_transcript_ids=transcript_ids,
        flat_dataset_ids=dataset_ids,
        lengths=np.full(12, 100, dtype=np.int64),
        batch_size=4,
        seed=42,
        sort_by_length=True,
        shuffle_batches=False,
        require_multidataset=True,
        minimum_distinct_datasets=2,
        execution_microbatch_max_pair_rows=7,
    )

    execution_batches = list(iter(sampler))
    assert [len(batch) for batch in execution_batches] == [6, 6]
    for batch in execution_batches:
        flat_indices = np.asarray([encoded_index[1] for encoded_index in batch])
        observed = transcript_ids[flat_indices]
        for transcript_id in set(observed):
            assert int(np.sum(observed == transcript_id)) == 3


def test_padded_token_budget_separates_long_complete_groups():
    transcript_ids = np.asarray(
        [transcript for transcript in ("short", "long") for _ in range(2)]
    )
    dataset_ids = np.asarray([0, 1, 0, 1])
    sampler = TranscriptGroupedMultiDatasetBatchSampler(
        flat_transcript_ids=transcript_ids,
        flat_dataset_ids=dataset_ids,
        lengths=np.asarray([100, 100, 1000, 1000]),
        batch_size=2,
        seed=42,
        sort_by_length=True,
        shuffle_batches=False,
        require_multidataset=True,
        minimum_distinct_datasets=2,
        execution_microbatch_max_padded_codon_tokens=2500,
    )

    execution_batches = list(iter(sampler))
    assert [len(batch) for batch in execution_batches] == [2, 2]
    for batch in execution_batches:
        flat_indices = np.asarray([encoded_index[1] for encoded_index in batch])
        assert len(set(transcript_ids[flat_indices])) == 1


def test_execution_epoch_length_is_invariant_across_shuffle_epochs():
    rng = np.random.default_rng(4)
    transcript_ids = []
    dataset_ids = []
    lengths = []
    for group_index in range(30):
        selected = sorted(
            rng.choice(6, size=int(rng.integers(2, 7)), replace=False).tolist()
        )
        length = int(rng.choice([100, 100, 100, 200, 300]))
        for dataset_id in selected:
            transcript_ids.append(f"t{group_index:02d}")
            dataset_ids.append(dataset_id)
            lengths.append(length)

    sampler = TranscriptGroupedMultiDatasetBatchSampler(
        flat_transcript_ids=transcript_ids,
        flat_dataset_ids=dataset_ids,
        lengths=lengths,
        batch_size=4,
        seed=42,
        sort_by_length=True,
        shuffle_batches=True,
        minimum_distinct_datasets=2,
        execution_microbatch_max_pair_rows=12,
        execution_microbatch_max_padded_codon_tokens=1800,
    )

    epoch_lengths = [
        len(sampler._build_epoch_plan(iteration_index)[0])
        for iteration_index in range(8)
    ]
    assert len(set(epoch_lengths)) == 1


def test_transcript_balanced_chunk_scaling_reconstructs_loss_and_gradient():
    values = torch.tensor([1.0, 3.0, 5.0, 9.0, 2.0, 8.0], requires_grad=True)
    weights = torch.tensor([1.0, 2.0, 3.0, 1.0, 4.0, 2.0])
    dataset_ids = torch.tensor([0, 1, 2, 0, 1, 2])
    transcript_ids = torch.tensor([0, 0, 0, 1, 1, 1])

    full_loss = reduce_per_sample_quantity(
        values,
        weights,
        dataset_ids,
        transcript_ids,
        "transcript_balanced",
    )
    full_loss.backward()
    full_gradient = values.grad.detach().clone()
    values.grad.zero_()

    scaled_chunks = []
    for transcript_id in (0, 1):
        mask = transcript_ids == transcript_id
        chunk_loss = reduce_per_sample_quantity(
            values[mask],
            weights[mask],
            dataset_ids[mask],
            transcript_ids[mask],
            "transcript_balanced",
        )
        scaled_chunks.append(chunk_loss * 0.5)
    chunked_loss = sum(scaled_chunks)
    chunked_loss.backward()

    torch.testing.assert_close(chunked_loss, full_loss)
    torch.testing.assert_close(values.grad, full_gradient)
