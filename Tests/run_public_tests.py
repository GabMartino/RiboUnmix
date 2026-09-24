"""Run the self-contained test profile shipped with the public repository.

The complete ``Tests`` directory also contains integration checks for the full
experimental data and historical result trees.  Those artifacts are not part
of the lightweight reviewer release, so they are deliberately outside this
profile.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


PUBLIC_TEST_MODULES = (
    "Tests.test_public_smoke_example",
    "Tests.test_ribounmix_naming",
    "Tests.test_gamma_centering",
    "Tests.test_stable_training_numerics",
    "Tests.test_gamma_exponential_range",
    "Tests.test_sample_reductions",
    "Tests.test_nb_mean_gradient_reweighting",
    "Tests.test_alpha_branch_optimization",
    "Tests.test_alpha_causality_modes",
    "Tests.test_bias_gru_precision",
    "Tests.test_bias_gru_tbptt",
    "Tests.test_dataset_bias_sequence_embeddings",
    "Tests.test_compact_sequence_collate",
    "Tests.test_execution_microbatching",
    "Tests.test_grouped_optimizer_batching",
    "Tests.test_unique_biological_forward",
    "Tests.test_validated_transcript_metadata",
    "Tests.test_prediction_checkpoint_selection",
    "Tests.test_common_weight_stratified_split",
    "Tests.test_nonfinite_gradient_diagnostics",
)


def main() -> int:
    """Load and run the deterministic, data-independent public test profile."""

    suite = unittest.defaultTestLoader.loadTestsFromNames(PUBLIC_TEST_MODULES)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
