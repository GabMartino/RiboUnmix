from __future__ import annotations

import sys
import unittest
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDatasetDataModule import (
    GroupedBatchStatistics,
    TranscriptGroupedMultiDatasetBatchSampler,
    resolve_grouped_optimizer_batch_plan,
)
from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDataset import (
    transcript_group_indices_from_ids,
)
from main_ribounmix_multidataset import (
    resolve_training_grouped_optimizer_batching,
)


def _statistics(unique_transcripts: tuple[int, ...]) -> GroupedBatchStatistics:
    return GroupedBatchStatistics(
        iteration_index=0,
        number_of_microbatches=len(unique_transcripts),
        eligible_transcript_groups=sum(unique_transcripts),
        physical_pair_capacity=16,
        pair_rows_per_microbatch=tuple(value * 2 for value in unique_transcripts),
        unique_transcripts_per_microbatch=unique_transcripts,
        datasets_per_transcript_group=(2,) * sum(unique_transcripts),
        minimum_group_size=2,
        median_group_size=2.0,
        mean_group_size=2.0,
        maximum_group_size=2,
        minimum_unique_transcripts_per_microbatch=min(unique_transcripts),
        median_unique_transcripts_per_microbatch=4.0,
        mean_unique_transcripts_per_microbatch=(
            sum(unique_transcripts) / len(unique_transcripts)
        ),
        maximum_unique_transcripts_per_microbatch=max(unique_transcripts),
    )


class _PreviewDatamodule:
    def __init__(self, statistics: GroupedBatchStatistics) -> None:
        self.statistics = statistics
        self.setup_stages: list[str] = []

    def setup(self, stage: str) -> None:
        self.setup_stages.append(stage)

    def preview_train_grouped_batch_statistics(
        self,
        iteration_index: int = 0,
    ) -> GroupedBatchStatistics:
        self.asserted_iteration_index = iteration_index
        return self.statistics


class GroupedOptimizerBatchingTests(unittest.TestCase):
    def test_transcript_group_identity_and_complete_group_assertion(self) -> None:
        group_ids = transcript_group_indices_from_ids(
            ["t1", "t1", "t2"],
            expected_pair_rows_by_transcript={"t1": 2, "t2": 1},
        )
        self.assertEqual(group_ids.tolist(), [0, 0, 1])
        with self.assertRaisesRegex(RuntimeError, "Incomplete transcript group"):
            transcript_group_indices_from_ids(
                ["t1"],
                expected_pair_rows_by_transcript={"t1": 2},
            )

    def test_batch_size_is_enforced_per_dataset(self) -> None:
        transcript_ids: list[str] = []
        dataset_ids: list[int] = []
        lengths: list[int] = []
        for transcript_index in range(40):
            for dataset_id in (0, 1):
                transcript_ids.append(f"t{transcript_index:02d}")
                dataset_ids.append(dataset_id)
                lengths.append(100)

        sampler = TranscriptGroupedMultiDatasetBatchSampler(
            flat_transcript_ids=transcript_ids,
            flat_dataset_ids=dataset_ids,
            lengths=lengths,
            batch_size=16,
            sort_by_length=False,
            shuffle_batches=False,
            require_multidataset=True,
        )
        batches = list(iter(sampler))

        self.assertEqual([len(batch) for batch in batches], [32, 32, 16])
        self.assertEqual(
            [
                len(set(transcript_ids[index] for index in batch))
                for batch in batches
            ],
            [16, 16, 8],
        )
        for batch in batches:
            counts = {
                dataset_id: sum(dataset_ids[index] == dataset_id for index in batch)
                for dataset_id in (0, 1)
            }
            self.assertLessEqual(counts[0], 16)
            self.assertLessEqual(counts[1], 16)
            self.assertEqual(counts[0], counts[1])

        statistics = sampler.last_epoch_statistics
        self.assertIsNotNone(statistics)
        self.assertEqual(statistics.per_dataset_pair_capacity, 16)
        self.assertEqual(statistics.physical_pair_capacity, 32)
        self.assertEqual(statistics.selected_dataset_ids, (0, 1))
        plan = resolve_grouped_optimizer_batch_plan(
            statistics,
            target_unique_transcripts_per_optimizer_step=32,
            accumulation_statistic="median",
        )
        self.assertEqual(plan.resolved_accumulate_grad_batches, 2)

    def test_two_dataset_ddp_target_scope_with_per_dataset_batch(self) -> None:
        statistics = GroupedBatchStatistics(
            iteration_index=0,
            number_of_microbatches=8,
            eligible_transcript_groups=128,
            physical_pair_capacity=32,
            pair_rows_per_microbatch=(32,) * 8,
            unique_transcripts_per_microbatch=(16,) * 8,
            datasets_per_transcript_group=(2,) * 128,
            minimum_group_size=2,
            median_group_size=2.0,
            mean_group_size=2.0,
            maximum_group_size=2,
            minimum_unique_transcripts_per_microbatch=16,
            median_unique_transcripts_per_microbatch=16.0,
            mean_unique_transcripts_per_microbatch=16.0,
            maximum_unique_transcripts_per_microbatch=16,
            selected_dataset_ids=(0, 1),
            per_dataset_pair_capacity=16,
            dataset_pair_rows_per_microbatch=((16, 16),) * 8,
        )
        per_rank = resolve_grouped_optimizer_batch_plan(
            statistics,
            target_unique_transcripts_per_optimizer_step=32,
            accumulation_statistic="mean",
            target_scope="per_rank",
            world_size=2,
        )
        globally_scoped = resolve_grouped_optimizer_batch_plan(
            statistics,
            target_unique_transcripts_per_optimizer_step=32,
            accumulation_statistic="mean",
            target_scope="global",
            world_size=2,
        )

        self.assertEqual(per_rank.resolved_accumulate_grad_batches, 2)
        self.assertEqual(
            per_rank.estimated_global_unique_transcripts_per_optimizer_step,
            64,
        )
        self.assertEqual(
            per_rank.estimated_global_pair_rows_per_optimizer_step,
            128,
        )
        self.assertEqual(globally_scoped.resolved_accumulate_grad_batches, 1)
        self.assertEqual(
            globally_scoped.estimated_global_unique_transcripts_per_optimizer_step,
            32,
        )
        self.assertEqual(
            globally_scoped.estimated_global_pair_rows_per_optimizer_step,
            64,
        )

    def test_target_scope_changes_global_batch_semantics(self) -> None:
        statistics = _statistics((4, 4, 2))
        per_rank = resolve_grouped_optimizer_batch_plan(
            statistics,
            target_unique_transcripts_per_optimizer_step=32,
            target_scope="per_rank",
            world_size=2,
        )
        globally_scoped = resolve_grouped_optimizer_batch_plan(
            statistics,
            target_unique_transcripts_per_optimizer_step=32,
            target_scope="global",
            world_size=2,
        )

        self.assertEqual(per_rank.resolved_accumulate_grad_batches, 8)
        self.assertEqual(per_rank.estimated_global_unique_transcripts_per_optimizer_step, 64)
        self.assertEqual(globally_scoped.resolved_accumulate_grad_batches, 4)
        self.assertEqual(globally_scoped.estimated_global_unique_transcripts_per_optimizer_step, 32)

    def test_resolver_supports_all_transcript_grouped_sampling(self) -> None:
        cfg = OmegaConf.create(
            {
                "data": {"train_sampling_strategy": "transcript_grouped_pairs"},
                "training": {
                    "grouped_optimizer_batch": {
                        "enabled": True,
                        "auto_accumulate_grad_batches": True,
                        "target_unique_transcripts_per_optimizer_step": 32,
                        "accumulation_statistic": "median",
                        "max_accumulate_grad_batches": 32,
                        "target_scope": "global",
                    }
                },
                "trainer": {
                    "devices": [0, 1],
                    "accumulate_grad_batches": 1,
                },
            }
        )
        datamodule = _PreviewDatamodule(_statistics((4, 4, 2)))
        statistics, plan = resolve_training_grouped_optimizer_batching(
            cfg=cfg,
            datamodule=datamodule,
        )

        self.assertIsNotNone(statistics)
        self.assertIsNotNone(plan)
        self.assertEqual(datamodule.setup_stages, ["fit"])
        self.assertEqual(cfg.trainer.accumulate_grad_batches, 4)
        self.assertEqual(
            cfg.training.grouped_optimizer_batch.resolved.resolved_accumulate_grad_batches,
            4,
        )

    def test_multidataset_strategy_excludes_singleton_groups(self) -> None:
        common_kwargs = dict(
            flat_transcript_ids=["singleton", "paired", "paired"],
            flat_dataset_ids=[0, 0, 1],
            lengths=[100, 100, 100],
            batch_size=3,
            sort_by_length=False,
            shuffle_batches=False,
        )
        all_groups = TranscriptGroupedMultiDatasetBatchSampler(
            **common_kwargs,
            require_multidataset=False,
        )
        multidataset_groups = TranscriptGroupedMultiDatasetBatchSampler(
            **common_kwargs,
            require_multidataset=True,
        )

        self.assertEqual(len(all_groups.groups), 2)
        self.assertEqual(len(multidataset_groups.groups), 1)
        self.assertEqual(multidataset_groups.group_transcript_ids, ("paired",))

    def test_positive_support_is_counted_before_packing_and_preview(self) -> None:
        sampler = TranscriptGroupedMultiDatasetBatchSampler(
            flat_transcript_ids=["t1", "t1", "t2"],
            flat_dataset_ids=[0, 1, 0],
            considered_transcript_ids=["t1", "t2", "t3"],
            lengths=[100, 100, 100],
            batch_size=4,
            sort_by_length=False,
            shuffle_batches=False,
            require_multidataset=True,
            minimum_distinct_datasets=2,
        )
        statistics = sampler.preview_epoch_batch_statistics()

        self.assertEqual(sampler.group_transcript_ids, ("t1",))
        self.assertEqual(statistics.transcripts_considered, 3)
        self.assertEqual(statistics.transcripts_with_positive_k0, 1)
        self.assertEqual(statistics.transcripts_with_positive_k1, 1)
        self.assertEqual(statistics.transcripts_with_positive_k2_or_more, 1)
        self.assertEqual(statistics.transcripts_admitted, 1)
        self.assertEqual(
            statistics.transcripts_excluded_for_insufficient_support,
            2,
        )
        self.assertEqual(statistics.positive_pair_rows, 2)
        self.assertEqual(statistics.pair_rows_per_microbatch, (2,))

    def test_duplicate_transcript_dataset_pair_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one row"):
            TranscriptGroupedMultiDatasetBatchSampler(
                flat_transcript_ids=["t1", "t1", "t1"],
                flat_dataset_ids=[0, 0, 1],
                lengths=[100, 100, 100],
                batch_size=4,
                require_multidataset=True,
                minimum_distinct_datasets=2,
            )

    def test_explicit_singleton_compatible_grouped_strategy(self) -> None:
        sampler = TranscriptGroupedMultiDatasetBatchSampler(
            flat_transcript_ids=["t1"],
            flat_dataset_ids=[0],
            lengths=[100],
            batch_size=1,
            require_multidataset=False,
            minimum_distinct_datasets=1,
            sort_by_length=False,
            shuffle_batches=False,
        )
        self.assertEqual(sampler.group_transcript_ids, ("t1",))
        self.assertEqual(list(iter(sampler)), [[0]])


if __name__ == "__main__":
    unittest.main()
