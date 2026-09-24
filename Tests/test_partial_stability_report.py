"""Scientific handling of missing endpoints and descriptive partial summaries."""
import unittest

import pandas as pd

from Utils.partial_stability_report import audit_transcript_folds, comparison_readiness, effect_sentence, snapshot_text


class PartialReportTests(unittest.TestCase):
    def test_legacy_role_changes_are_measured_and_rejected_for_a_fixed_design(self):
        folds = {'2': dict(train_ids=['a'], validation_ids=['b'], test_ids=['t']),
                 '5': dict(train_ids=['b'], validation_ids=['a'], test_ids=['t'])}
        controls, _, pairs = audit_transcript_folds(folds)
        self.assertTrue(controls['identical_test_ids'])
        self.assertFalse(controls['identical_train_ids'])
        self.assertEqual(pairs.iloc[0].a_train_in_b_validation, 1)
        self.assertEqual(pairs.iloc[0].a_validation_in_b_train, 1)
        with self.assertRaisesRegex(ValueError, 'identical validation'):
            audit_transcript_folds(folds, require_common_validation=True)

    def test_panels_can_have_different_training_coverage_with_fixed_heldout_roles(self):
        folds = {'panel_1': dict(train_ids=['a', 'b'], validation_ids=['v'], test_ids=['t']),
                 'panel_2': dict(train_ids=['b', 'c'], validation_ids=['v'], test_ids=['t'])}
        controls, _, pairs = audit_transcript_folds(folds, require_common_validation=True)
        self.assertFalse(controls['identical_train_ids'])
        self.assertEqual(pairs.iloc[0].shared_train, 1)
        self.assertEqual(pairs.iloc[0].a_train_in_b_validation, 0)
        self.assertEqual(pairs.iloc[0].a_validation_in_b_train, 0)
        with self.assertRaisesRegex(ValueError, 'identical training'):
            audit_transcript_folds(folds, require_common_training=True)

    def test_test_leakage_and_different_test_cohorts_are_rejected(self):
        folds = {'a': dict(train_ids=['train', 'test'], validation_ids=['val'], test_ids=['test'])}
        with self.assertRaisesRegex(ValueError, 'roles overlap'):
            audit_transcript_folds(folds)
        folds['a']['train_ids'] = ['train']
        folds['b'] = dict(train_ids=['train'], validation_ids=['val'], test_ids=['other_test'])
        with self.assertRaisesRegex(ValueError, 'share test'):
            audit_transcript_folds(folds)

    def test_a_checkpoint_is_not_a_completed_comparison_endpoint(self):
        availability = pd.DataFrame([
            dict(N=2, arm='equal', status='validated_predictions', recorded_status='completed'),
            dict(N=5, arm='equal', status='checkpoint_without_export', recorded_status='running'),
            dict(N=10, arm='equal', status='validated_predictions', recorded_status='completed'),
        ])
        readiness = comparison_readiness(availability, 'N', [(2, 5), (5, 10)], ['equal'])
        self.assertFalse(readiness.ready.any())
        self.assertEqual(readiness.missing_endpoints.tolist(), ['5', '5'])
        self.assertIn('2/3 validated model exports', snapshot_text(availability, 1771))
        self.assertIn('does not query the live scheduler', snapshot_text(availability, 1771))

    def test_interpretation_reports_pcc_and_rmse_disagreement(self):
        effects = pd.DataFrame([
            dict(kind='cross_panel', baseline='shared_only', policy='equal', metric=metric, mean_improvement=value)
            for metric, value in [('PCC', .03), ('PCC', -.02), ('RMSE', .2), ('RMSE', .05)]
        ])
        sentence = effect_sentence(effects, 'shared_only', 'equal', unit='panel pairs')
        self.assertIn('PCC is higher in 1/2', sentence)
        self.assertIn('RMSE is lower in 2/2', sentence)
        self.assertIn('-0.0200 to +0.0300', sentence)
        self.assertIn('no matched', effect_sentence(effects, 'equal', 'ranked_p1'))

    def test_adjacent_caption_excludes_anchor_effects(self):
        effects = pd.DataFrame([
            dict(kind=kind, baseline='equal', policy='ranked_p1', metric='PCC', mean_improvement=value)
            for kind, value in [('adjacent', -.01), ('anchor', .3)]
        ])
        sentence = effect_sentence(effects, 'equal', 'ranked_p1', kind='adjacent', unit='steps')
        self.assertIn('PCC is higher in 0/1', sentence)
        self.assertNotIn('+0.3000', sentence)


if __name__ == '__main__':
    unittest.main()
