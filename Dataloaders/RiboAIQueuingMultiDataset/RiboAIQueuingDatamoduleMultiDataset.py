"""Deprecated compatibility adapter; use RiboUnmixMultiDatasetDataModule."""

from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDatasetDataModule import *  # noqa: F401,F403
from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDatasetDataModule import (
    RiboUnmixMultiDatasetDataModule as RiboAIQueuingDatamoduleMultiDataset,
)
