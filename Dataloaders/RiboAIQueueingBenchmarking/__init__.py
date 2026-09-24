"""Deprecated import compatibility for the pre-RiboUnmix package name."""

from Dataloaders.RiboUnmixBenchmarking import (
    RiboUnmixBenchmarkingDataModule,
    RiboUnmixBenchmarkingDataset,
    available_ids_for_dataset,
)

RiboAIQueueingBenchmarkingDatamodule = RiboUnmixBenchmarkingDataModule
RiboAIQueueingBenchmarkingDataset = RiboUnmixBenchmarkingDataset

__all__ = [
    "RiboAIQueueingBenchmarkingDatamodule",
    "RiboAIQueueingBenchmarkingDataset",
    "available_ids_for_dataset",
]
