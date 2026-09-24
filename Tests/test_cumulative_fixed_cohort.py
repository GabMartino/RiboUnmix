"""Scientific controls for the common observed cohort and cumulative split audit."""
import unittest
from unittest.mock import patch

import pandas as pd

from Utils.cumulative_fixed_cohort import build_fixed_cumulative_split, partition_complete_cohort
from analyses.reevaluate_cumulative_fixed_observations import audit_folds


class FixedCohortTests(unittest.TestCase):
    def setUp(self):
        ids = [f't{i:03d}' for i in range(103)]
        self.metadata = pd.DataFrame({'css_bin': ['css_0'] * len(ids)}, index=ids)
        self.supports = {'best': set(ids), 'middle': set(ids[:-1]), 'worst': set(ids[:-3])}
        self.weights = {d: {t: (int(t[1:]) + 1) / 104 for t in supported}
                        for d, supported in self.supports.items()}

    def test_complete_intersection_is_partitioned_once(self):
        folds = partition_complete_cohort(self.metadata, self.supports, self.weights, seed=42)
        train, val, test = (set(folds[k]) for k in ('train', 'validation', 'test'))
        self.assertEqual([len(train), len(val), len(test)], [80, 10, 10])
        self.assertFalse(train & val or train & test or val & test)
        self.assertEqual(train | val | test, self.supports['worst'])
        # Reordering dataset inputs cannot change transcript roles.
        reordered = dict(reversed(list(self.supports.items())))
        self.assertEqual(folds, partition_complete_cohort(self.metadata, reordered, self.weights, seed=42))

    def test_builder_reuses_all_folds_when_worse_datasets_are_added(self):
        tasks = [dict(run_id='N2', datasets=['best', 'middle']),
                 dict(run_id='N3', datasets=['best', 'middle', 'worst'])]
        with patch('Utils.cumulative_fixed_cohort.load_sequence_metadata', return_value=(self.metadata, {})), \
             patch('Utils.cumulative_fixed_cohort.load_support_and_stored_weights',
                   return_value=(self.supports, self.weights, {})):
            split = build_fixed_cumulative_split(experiment_name='test', tasks=tasks,
                dataset_mapping={d: d+'.parquet' for d in self.supports}, sequences_path='sequences.parquet',
                subset_seed=42, validation_fraction=.1, test_fraction=.1, reliability_bins=10,
                maximum_cds_codons=None)
        self.assertEqual(split['panel_train_eligible_ids']['N2'], split['panel_train_eligible_ids']['N3'])
        self.assertEqual(split['panel_validation_ids']['N2'], split['panel_validation_ids']['N3'])
        self.assertTrue(set(split['complete_transcript_ids']) <= self.supports['worst'])
        manifest = dict(sizes=[2, 3], source_folds={str(n): dict(
            train_ids=split['panel_train_eligible_ids'][f'N{n}'],
            validation_ids=split['panel_validation_ids'][f'N{n}'],
            test_ids=split['common_test_ids']) for n in [2, 3]})
        _, transitions = audit_folds(manifest)
        self.assertEqual(transitions.iloc[0].old_train_now_validation, 0)
        self.assertEqual(transitions.iloc[0].old_validation_now_train, 0)
        self.assertEqual(transitions.iloc[0].training_removed, 0)

    def test_audit_distinguishes_role_swapping_from_test_leakage(self):
        manifest = dict(sizes=[2, 5], source_folds={
            '2': dict(train_ids=['a'], validation_ids=['b'], test_ids=['heldout']),
            '5': dict(train_ids=['b'], validation_ids=['a'], test_ids=['heldout'])})
        _, changes = audit_folds(manifest)
        self.assertEqual(changes.iloc[0].old_validation_now_train, 1)
        self.assertEqual(changes.iloc[0].old_train_now_validation, 1)
        manifest['source_folds']['5']['train_ids'].append('heldout')
        with self.assertRaisesRegex(ValueError, 'within-model disjointness'):
            audit_folds(manifest)


if __name__ == '__main__':
    unittest.main()
