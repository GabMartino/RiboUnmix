"""Scientific membership controls for the quality-selection follow-up."""
import unittest

import numpy as np

from Utils.quality_selection_experiments import (
    CUMULATIVE_SCORE_DESIGN,
    SCORE_REFERENCE_POLICIES,
    cumulative_collections,
    directional_quality_score_weights,
    four_quality_collections,
    inverse_quality_score_weights,
    reference_variants,
)


class QualitySelectionExperimentTests(unittest.TestCase):
    def setUp(self):
        self.names = [f"rank_{rank:03d}" for rank in range(1, 115)]

    def test_cumulative_paths_change_membership_and_are_nested(self):
        collections = {row["collection_id"]: row for row in cumulative_collections(self.names)}
        self.assertEqual(collections["best_first_N002"]["datasets"], self.names[:2])
        self.assertEqual(collections["worst_first_N002"]["datasets"], self.names[-2:])
        self.assertEqual(collections["full_N114"]["datasets"], self.names)
        for n in (2, 5, 10, 20, 40):
            best = set(collections[f"best_first_N{n:03d}"]["datasets"])
            worst = set(collections[f"worst_first_N{n:03d}"]["datasets"])
            self.assertTrue(best.isdisjoint(worst))
        for direction in ("best_first", "worst_first"):
            previous = set()
            for n in (2, 5, 10, 20, 40, 80):
                current = set(collections[f"{direction}_N{n:03d}"]["datasets"])
                self.assertLessEqual(previous, current)
                previous = current
            self.assertLessEqual(previous, set(collections["full_N114"]["datasets"]))

    def test_four_quality_strata_are_ordered_exhaustive_blocks(self):
        panels = four_quality_collections(self.names)
        self.assertEqual([panel["N"] for panel in panels], [29, 29, 28, 28])
        self.assertEqual(panels[0]["datasets"], self.names[:29])
        self.assertEqual(panels[-1]["datasets"], self.names[-28:])
        flattened = [dataset for panel in panels for dataset in panel["datasets"]]
        self.assertEqual(flattened, self.names)
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_score_weight_policies_are_prespecified(self):
        self.assertEqual(
            SCORE_REFERENCE_POLICIES,
            (("equal", 0), ("score_p1", 1), ("score_p3", 3), ("score_p5", 5)),
        )

    def test_inverse_score_weights_favor_lower_better_scores(self):
        scores = np.array([200.0, 400.0, 800.0])
        np.testing.assert_allclose(
            inverse_quality_score_weights(scores, 0, global_minimum=200.0),
            np.ones(3),
        )
        np.testing.assert_allclose(
            inverse_quality_score_weights(scores, 1, global_minimum=200.0),
            [1.0, 0.5, 0.25],
        )
        p3 = inverse_quality_score_weights(scores, 3, global_minimum=200.0)
        p5 = inverse_quality_score_weights(scores, 5, global_minimum=200.0)
        self.assertTrue(np.all(np.diff(p3) < 0))
        self.assertGreater((p5 / p5.sum())[0], (p3 / p3.sum())[0])

    def test_inverse_score_weights_reject_invalid_inputs(self):
        for scores in ([], [1.0, 0.0], [1.0, np.nan]):
            with self.subTest(scores=scores):
                with self.assertRaises(ValueError):
                    inverse_quality_score_weights(scores, 1)
        with self.assertRaises(ValueError):
            inverse_quality_score_weights([1.0, 2.0], -1)

    def test_directional_score_weights_favor_the_named_extreme(self):
        scores = np.array([200.0, 400.0, 800.0])
        best = directional_quality_score_weights(
            scores, 1, "best_first", global_minimum=200.0, global_maximum=800.0)
        worst = directional_quality_score_weights(
            scores, 1, "worst_first", global_minimum=200.0, global_maximum=800.0)
        np.testing.assert_allclose(best, [1.0, 0.5, 0.25])
        np.testing.assert_allclose(worst, [0.25, 0.5, 1.0])
        self.assertEqual(int(np.argmax(best)), 0)
        self.assertEqual(int(np.argmax(worst)), 2)
        with self.assertRaises(ValueError):
            directional_quality_score_weights(
                scores, 1, "shared_full", global_minimum=200.0, global_maximum=800.0)

    def test_score_campaign_has_55_nonredundant_fits(self):
        collections = cumulative_collections(self.names)
        variants = [(collection["N"], *variant)
                    for collection in collections
                    for variant in reference_variants(collection, CUMULATIVE_SCORE_DESIGN)]
        self.assertEqual(len(variants), 55)
        full = [variant for variant in variants if variant[0] == 114]
        self.assertEqual(len(full), 7)
        self.assertEqual(full.count((114, "shared_full", "equal", 0)), 1)
        self.assertIn((114, "best_first", "score_p5", 5), full)
        self.assertIn((114, "worst_first", "score_p5", 5), full)


if __name__ == "__main__":
    unittest.main()
