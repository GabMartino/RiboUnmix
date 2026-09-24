"""Failed audit reports must never be treated as resumable successful designs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from Tests.test_panel_reference_quality_audit import fixture
from run_reproducibility_reference_campaign import parse_args
from Utils.panel_reference_audit import sha256
from Utils.reference_campaign import prepare_campaign, require_successful_preparation_audit


class PreparationResumeTests(unittest.TestCase):
    def test_real_failed_audit_preserves_original_error_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            historical, ranking, _ = fixture(root)
            # An input-loading failure occurs before run_audit can save `ranking`.
            pd.read_csv(ranking, sep='\t').drop(columns='quality_rank').to_csv(
                ranking, sep='\t', index=False)
            args = parse_args(['--stage', 'prepare', '--output-root', str(root/'campaign'),
                               '--panel-manifest', str(historical/'panel_manifest.json'),
                               '--ranking-table', str(ranking), '--no-tex', '--dry-run'])
            inputs_before = {str(p): sha256(p) for p in root.rglob('*') if p.is_file()}
            with patch('Utils.panel_reference_preparation.prepare_design') as design:
                with self.assertRaisesRegex(ValueError, "KeyError: 'quality_rank'") as fresh:
                    prepare_campaign(args)
                manifest_path = args.output_root/'partition_preparation/audit_manifest.json'
                self.assertNotIn('ranking', json.loads(manifest_path.read_text()))
                saved_before = {str(p): sha256(p) for p in root.rglob('*') if p.is_file()}
                args.resume = True
                with patch('Utils.panel_reference_audit.run_audit') as audit:
                    with self.assertRaises(ValueError) as resumed:
                        prepare_campaign(args)
                    audit.assert_not_called()
                design.assert_not_called()
            self.assertEqual(str(fresh.exception), str(resumed.exception))
            self.assertIn(str(manifest_path), str(resumed.exception))
            self.assertEqual(saved_before, {str(p): sha256(p) for p in root.rglob('*') if p.is_file()})
            self.assertTrue(all(sha256(p) == digest for p, digest in inputs_before.items()))
            self.assertFalse((args.output_root/'candidate_partition_manifest.json').exists())
            self.assertFalse((args.output_root/'campaign_manifest.json').exists())

    def test_failed_audit_with_ranking_also_blocks_before_hash_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pre = root/'partition_preparation'
            pre.mkdir()
            manifest = dict(hard_errors=['Missing global ranks: dataset_X'],
                            ranking={'sha256': 'unused'},
                            input_sha256={'/absent/frozen/input.tsv': 'unused'})
            (pre/'audit_manifest.json').write_text(json.dumps(manifest))
            args = parse_args(['--output-root', str(root), '--resume'])
            with patch('Utils.reference_campaign.sha256') as digest, \
                 patch('Utils.panel_reference_preparation.prepare_design') as design:
                with self.assertRaisesRegex(ValueError, 'Missing global ranks: dataset_X'):
                    prepare_campaign(args)
                digest.assert_not_called()
                design.assert_not_called()

    def test_issue_level_failure_cannot_be_hidden_by_empty_summary(self):
        manifest = dict(hard_errors=[], issues=[dict(severity='hard_error', detail='Source family crosses panels')])
        with self.assertRaisesRegex(ValueError, 'Source family crosses panels'):
            require_successful_preparation_audit(manifest, Path('audit_manifest.json'))

    def test_incomplete_audit_has_actionable_message_not_keyerror(self):
        for manifest in ({}, {'hard_errors': []}, {'hard_errors': [], 'ranking': None}):
            with self.subTest(manifest=manifest):
                with self.assertRaisesRegex(ValueError, 'Incomplete CPU preparation audit.*ranking.sha256'):
                    require_successful_preparation_audit(manifest, Path('audit_manifest.json'))

    def test_successful_audit_is_not_changed(self):
        manifest = dict(hard_errors=[], ranking={'sha256': 'frozen'}, input_sha256={},
                        observed_design={'datasets': 114}, issues=[])
        before = json.dumps(manifest, sort_keys=True)
        require_successful_preparation_audit(manifest, Path('audit_manifest.json'))
        self.assertEqual(before, json.dumps(manifest, sort_keys=True))


if __name__ == '__main__':
    unittest.main()
