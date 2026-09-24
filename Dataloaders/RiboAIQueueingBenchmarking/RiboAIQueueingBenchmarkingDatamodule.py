"""Deprecated compatibility adapter; use RiboUnmixBenchmarkingDataModule."""

from Dataloaders.RiboUnmixBenchmarking.RiboUnmixBenchmarkingDataModule import *  # noqa: F401,F403
from Dataloaders.RiboUnmixBenchmarking.RiboUnmixBenchmarkingDataModule import (
    RiboUnmixBenchmarkingDataModule as RiboAIQueueingBenchmarkingDatamodule,
)
