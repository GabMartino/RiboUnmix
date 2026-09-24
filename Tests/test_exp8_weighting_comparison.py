"""Scientific comparison contracts; all profiles here are synthetic fixtures."""
import unittest

import numpy as np
import pandas as pd

from analyses.compare_real_exp8_weighting import planned_pairs, profile_pcc, summarize


class Exp8WeightingComparisonTests(unittest.TestCase):
    def task(self, n, label, seed=42):
        return dict(N=n, run_id=label, training_seed=seed, kind="cumulative_top_quality")

    def test_missing_middle_size_is_not_bridged(self):
        tasks = [self.task(n, str(n)) for n in [2, 5, 10]]
        self.assertEqual(list(planned_pairs(tasks, {"2", "10"})), [])

    def test_all_pairs_without_mixing_training_seeds(self):
        tasks = [self.task(2, "a"), self.task(2, "b"), self.task(5, "c"), self.task(5, "d", 43)]
        pairs = list(planned_pairs(tasks, {t["run_id"] for t in tasks}))
        self.assertEqual([(a["run_id"], b["run_id"]) for a, b in pairs], [("a", "c"), ("b", "c")])

    def test_disjoint_mode_rejects_overlap(self):
        tasks = [dict(self.task(2, side), kind="designated_disjoint_pair", pair_id="pair01",
                      side=side, datasets=["shared", side], source_families=["shared", side])
                 for side in ["A", "B"]]
        with self.assertRaisesRegex(ValueError, "shares datasets"):
            list(planned_pairs(tasks, {"A", "B"}, disjoint=True))
        # One ranked model per N cannot supply a same-N disjoint statistic.
        ranked = [self.task(n, str(n)) for n in [2, 5]]
        self.assertEqual(list(planned_pairs(ranked, {"2", "5"}, disjoint=True)), [])

    def test_full_profile_pcc_and_undefined_constant(self):
        def profile(v):
            return dict(values=np.array(v, dtype=float), mask=np.ones(len(v), dtype=bool))
        a, b = profile([.5, 1, 1.5]), profile([1.4, 1, .6])
        self.assertAlmostEqual(profile_pcc(a, b), -1)
        self.assertTrue(np.isnan(profile_pcc(a, profile([1, 1, 1]))))
        with self.assertRaisesRegex(ValueError, "alignment"):
            profile_pcc(a, profile([.5, 1.5]))

    def test_summary_counts_dependence_and_undefined(self):
        values = pd.DataFrame([dict(experiment="equal", N=2, N_next=5, run_a=a,
            run_b="c", transcript_id=t, PCC=v, dataset_jaccard=0., source_jaccard=0.)
            for a, t, v in [("a", "t1", .2), ("a", "t2", .8), ("b", "t1", np.nan)]])
        s = summarize(values).iloc[0]
        self.assertAlmostEqual(s.mean_PCC, .5)
        self.assertAlmostEqual(s.std_PCC, np.std([.2, .8], ddof=1))
        self.assertEqual((s.n_usable, s.n_undefined, s.n_pairs, s.n_transcripts), (2, 1, 2, 2))


if __name__ == "__main__":
    unittest.main()
