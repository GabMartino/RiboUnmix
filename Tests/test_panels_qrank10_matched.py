from contextlib import redirect_stdout
from io import StringIO
import argparse
import copy
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

import run_real_independent_panel_convergence as equal
import run_real_independent_panel_convergence_quality_rank as ranked
import run_real_panels_qrank10_matched as matched
from analyses.compare_real_panel_weighting import audit_design, resolve_ranking
from resume_real_experiment_from_checkpoints import _use_saved_resolved_config


class MatchedPanelsTests(unittest.TestCase):
    def test_task_mapping(self):
        self.assertEqual(matched.task_spec(0), ('equal', 'panel_01', 'real_panel_convergence_panel01'))
        self.assertEqual(matched.task_spec(7), ('qrank10components_p1', 'panel_04', 'real_panel_qrank_convergence_panel04'))
        self.assertEqual(len({matched.task_spec(i) for i in range(8)}), 8)
        with self.assertRaises(ValueError):
            matched.task_spec(8)

    def test_prepare_option_is_not_dry_run(self):
        for runner, extra in ((equal, []), (ranked, ['--standalone-design'])):
            args = runner.parse_args(['--prepare-only', *extra])
            self.assertTrue(args.prepare_only)
            self.assertFalse(args.dry_run)

    def test_reference_checks_panel_sources_and_ordered_split_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assignment = pd.DataFrame(dict(dataset_name=['a', 'b'],
                panel=['panel_01', 'panel_02'], source_identifier=['source_a', 'source_b']))
            split = dict(common_validation_ids=['v1', 'v2'], common_test_ids=['t1', 't2'],
                         panel_train_eligible_ids={'panel_01': ['a1', 'a2'], 'panel_02': ['b1', 'b2']})
            assignment.to_csv(root/'panel_assignment.csv', index=False)
            (root/'common_split_manifest.json').write_text(json.dumps(split))
            reference = matched.load_reference_design(argparse.Namespace(reference_design_root=root))
            matched.validate_historical_design(root, reference)
            # CSV row order and machine-local data paths are not design identity.
            assignment.iloc[::-1].assign(dataset_path='/another/cluster').to_csv(root/'panel_assignment.csv', index=False)
            matched.validate_historical_design(root, reference)
            for column in ('dataset_name', 'panel', 'source_identifier'):
                with self.subTest(column=column):
                    changed = assignment.copy()
                    changed.loc[0, column] = 'different'
                    changed.to_csv(root/'panel_assignment.csv', index=False)
                    with self.assertRaisesRegex(RuntimeError, 'panel_assignment'):
                        matched.validate_historical_design(root, reference)
            assignment.to_csv(root/'panel_assignment.csv', index=False)
            for key in ('common_validation_ids', 'common_test_ids', 'panel_train_eligible_ids'):
                with self.subTest(split=key):
                    changed = copy.deepcopy(split)
                    if key == 'panel_train_eligible_ids':
                        changed[key]['panel_01'].reverse()
                    else:
                        changed[key].reverse()
                    (root/'common_split_manifest.json').write_text(json.dumps(changed))
                    with self.assertRaisesRegex(RuntimeError, key):
                        matched.validate_historical_design(root, reference)

    def test_missing_explicit_reference_does_not_silently_fall_back(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'missing'
            with self.assertRaisesRegex(FileNotFoundError, 'Incomplete explicit reference'):
                matched.load_reference_design(argparse.Namespace(reference_design_root=root))
            with self.assertRaisesRegex(FileNotFoundError, 'Deploy this JSON'):
                matched.load_reference_design(argparse.Namespace(reference_design_root=None,
                    reference_design_manifest=root/'missing.json'))

    def test_bundled_reference_is_available_without_result_folders(self):
        reference = matched.load_reference_design(argparse.Namespace(reference_design_root=None))
        identity = reference['design_identity']
        self.assertEqual(reference['schema_version'], 1)
        assignment = pd.DataFrame(identity['panel_assignment'])
        self.assertEqual(assignment.groupby('panel').size().tolist(), [29, 29, 28, 28])
        self.assertEqual(assignment.dataset_name.nunique(), 114)
        self.assertEqual(identity['common_test_ids']['count'], 1593)
        self.assertEqual(identity['common_validation_ids']['count'], 1593)

    def test_complete_toy_preparation_and_resume_contract(self):
        """Run both real design builders on eight small replica-aware datasets."""
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            ids = [f't{i:03}' for i in range(80)]
            seq = temp / 'sequences.parquet'
            pd.DataFrame(dict(transcript_id=ids, codons=[['ATG','AAA','TAA']]*len(ids),
                              css=[[]]*len(ids))).to_parquet(seq)
            mapping = {}
            for n in range(8):
                name = f'source{n}_2020'
                path = temp / f'{name}.parquet'
                replicas = [[[1.+n/20, 0., 2.], [1., 1., 2.+n/20]]] * len(ids)
                pd.DataFrame(dict(id=ids, weight=np.ones(len(ids)),
                                  read_density=[float(np.mean(x)) for x in replicas],
                                  coverage=np.ones(len(ids)),
                                  ribo=[np.mean(x, axis=0) for x in replicas],
                                  ribo_cds_replicas=replicas)).to_parquet(path)
                mapping[name] = str(path)
            dataset_config = temp / 'datasets.yaml'
            dataset_config.write_text(yaml.safe_dump(dict(dataset_path=mapping)))
            ranking = temp / 'rank10.tsv'
            table = pd.DataFrame(dict(dataset=list(mapping), quality_rank=range(1,9), rank_component_count=10))
            for n in range(10):
                table[f'rank_component{n}'] = range(1,9)
            table.to_csv(ranking, sep='\t', index=False)
            extra = ['--dataset-config', str(dataset_config), '--sequences-path', str(seq), '--expected-dataset-count', '8']
            reference = temp / 'historical'
            with redirect_stdout(StringIO()), patch.object(equal, 'plot_panel_balance'):
                equal.main(['--prepare-only', '--output-root', str(temp), '--run-id', 'historical', *extra])
            config = temp/'config.yaml'
            cfg = yaml.safe_load(equal.DEFAULT_CONFIG.read_text())
            cfg['paths']['sequences_path'] = str(seq)
            config.write_text(yaml.safe_dump(cfg))
            args = argparse.Namespace(output_root=temp/'new', reference_design_root=reference,
                                      ranking_table=ranking, config=config, dataset_config=dataset_config)
            portable = temp/'historical_design.json'
            portable.write_text(json.dumps(matched.load_reference_design(args)))
            reference.rename(temp/'historical_not_at_original_path')
            self.assertFalse(reference.exists())
            args.reference_design_root = None
            calls = []
            original_run = matched.subprocess.run

            def local_design(command, **kwargs):
                if command[0] == 'git':
                    return original_run(command, **kwargs)
                calls.append(command)
                runner = ranked if 'quality_rank.py' in command[2] else equal
                with redirect_stdout(StringIO()), patch.object(runner, 'plot_panel_balance'):
                    self.assertEqual(runner.main([*command[3:], *extra]), 0)

            with patch.object(matched, 'DEFAULT_REFERENCE_DESIGN_MANIFEST', portable), \
                    patch.object(matched.subprocess, 'run', side_effect=local_design), redirect_stdout(StringIO()):
                matched.prepare(args)
                matched.prepare(args)
            self.assertEqual(len(calls), 2)  # no recomposition on resubmission
            self.assertFalse((temp/'new/equal_dry_run').exists())
            # Shared split, local reliability, ranks and all configured model
            # hyperparameters are verified by the actual comparison audit.
            audit_design(args.output_root/'equal', args.output_root/'qrank10components_p1', ranking)
            for arm in matched.ARM_NAMES:
                for panel in range(1,5):
                    task = args.output_root/arm/f'panel_{panel:02}'
                    cfg = yaml.safe_load((task/'resolved_config.yaml').read_text())
                    self.assertEqual(cfg['trainer']['precision'], 'bf16-mixed')
                    self.assertEqual(cfg['model']['dataset_bias_params']['context_gru_precision'], 'float32')
                    self.assertEqual(cfg['model']['dataset_bias_params']['context_gru_tbptt_window'], 0)
                    self.assertFalse(cfg['experiment']['from_checkpoint'])
                    command = shlex.split((task/'launch_command.sh').read_text().splitlines()[-1])
                    self.assertTrue(_use_saved_resolved_config(command, task)['hydra_composition_checked'])
            # The unchanged files survive normal resume-input creation.
            # Resuming a prepared experiment uses its frozen reference, not the external bundle.
            portable.rename(temp/'historical_design_not_at_original_path.json')
            matched.prepare(args)
            marker = json.loads((args.output_root/'matched_experiment.json').read_text())
            self.assertEqual(marker['ranking_metadata']['count'], 10)
            self.assertIsNone(marker['reference_design_root'])
            self.assertIn('frozen_reference_design.json', marker['frozen_files'])
            with patch.object(matched, 'execution_identity', return_value={'different': True}):
                with self.assertRaisesRegex(RuntimeError, 'source/environment changed'):
                    matched.prepare(args)
            frozen_reference = args.output_root/'frozen_reference_design.json'
            frozen_reference.write_text(frozen_reference.read_text() + ' ')
            with self.assertRaisesRegex(RuntimeError, 'Frozen design inputs changed'):
                matched.prepare(args)

    def test_wrong_ranking_and_partial_preparation_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'new'
            args = argparse.Namespace(output_root=root, reference_design_root=Path(temporary),
                ranking_table=matched.ROOT/'Datasets/data/HEK_riboseq_profile_quality_rank.tsv')
            with self.assertRaisesRegex(ValueError, '10-component'):
                matched.prepare(args)
            root.mkdir()
            (root/'partial').write_text('do not overwrite')
            with self.assertRaisesRegex(RuntimeError, 'Incomplete preparation'):
                matched.prepare(args)
            self.assertEqual((root/'partial').read_text(), 'do not overwrite')

    def test_relocated_ranking_requires_matching_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen = root/'frozen_quality_rank_10components.tsv'
            frozen.write_text('frozen content')
            strategy = dict(ranking_table='/missing/table.tsv', ranking_table_sha256=matched.sha(frozen))
            self.assertEqual(resolve_ranking(root/'ranked', strategy), frozen)
            bad = root/'bad.tsv'
            bad.write_text('different')
            with self.assertRaisesRegex(ValueError, 'matches the frozen'):
                resolve_ranking(root/'ranked', strategy, bad)


if __name__ == '__main__':
    unittest.main()
