from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from Dataloaders.RiboUnmixBenchmarking import RiboUnmixBenchmarkingDataModule
from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDataset import RiboUnmixMultiDataset
from Models.RiboUnmixModel import RiboUnmixModel
from Models.RiboUnmixLightningModule import RiboUnmixLightningModule
from Utils.publication_plot_style import publication_rc
from main_ribounmix_multidataset import shared_logger_version_from_environment


class RiboUnmixNamingTests(unittest.TestCase):
    def test_legacy_imports_resolve_to_canonical_classes(self) -> None:
        from Dataloaders.RiboAIQueueingBenchmarking import (
            RiboAIQueueingBenchmarkingDatamodule,
        )
        from Dataloaders.RiboAIQueuingMultiDataset.RiboAIQueuingMultiDataset import (
            RiboAIQueuingDatasetMultiDataset,
        )
        from Models.RiboQueuingModel import RiboQueuingModel
        from Models.RiboQueuingModelLighningModule import (
            RiboQueuingModelLightningModule,
        )

        self.assertIs(RiboAIQueueingBenchmarkingDatamodule, RiboUnmixBenchmarkingDataModule)
        self.assertIs(RiboAIQueuingDatasetMultiDataset, RiboUnmixMultiDataset)
        self.assertIs(RiboQueuingModel, RiboUnmixModel)
        self.assertIs(RiboQueuingModelLightningModule, RiboUnmixLightningModule)

    def test_canonical_logger_environment_has_precedence(self) -> None:
        environment = {
            "RIBOUNMIX_LOGGER_VERSION": "canonical",
            "RIBOAI_LOGGER_VERSION": "legacy",
        }
        with patch.dict(os.environ, environment, clear=True):
            self.assertEqual(shared_logger_version_from_environment(), "canonical")

    def test_legacy_logger_environment_remains_a_fallback(self) -> None:
        with patch.dict(os.environ, {"RIBOAI_LOGGER_VERSION": "legacy"}, clear=True):
            self.assertEqual(shared_logger_version_from_environment(), "legacy")

    def test_canonical_plot_environment_has_precedence(self) -> None:
        environment = {"RIBOUNMIX_PLOT_TEX": "0", "RIBOAI_PLOT_TEX": "1"}
        with patch.dict(os.environ, environment, clear=True):
            self.assertFalse(publication_rc()["text.usetex"])


if __name__ == "__main__":
    unittest.main()
