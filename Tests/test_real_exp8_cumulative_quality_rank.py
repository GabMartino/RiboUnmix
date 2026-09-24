from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

import run_real_exp8_L_stability_quality_rank as runner
from analyses.analyze_real_exp8_cumulative_quality_rank import main as analyze
from Tests.test_real_exp8_stability import toy_quality
from Utils.reliability_references import transcript_id_hash


class CumulativeRankTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.ranking = self.root / 'ranking.tsv'
        pd.DataFrame({'dataset': ['outside', 'd00', 'd02', 'd01', 'd03'],
                      'quality_rank': [10, 1, 3, 3, 7],
                      'rank_component_count': [2] * 5,
                      'rank_alpha': [1, 2, 3, 4, 5],
                      'rank_beta': [5, 4, 3, 2, 1]}).to_csv(
                          self.ranking, sep='\t', index=False)
        self.names = ['d00', 'd01', 'd02', 'd03']

    def test_exact_nested_prefixes_and_lexicographic_ties(self):
        tasks, order = runner.build_cumulative_tasks(self.names, self.ranking, [4, 2, 3], [42, 43])
        self.assertEqual(order.dataset.tolist(), self.names)
        self.assertEqual(len(tasks), 6)
        self.assertEqual(tasks[0]['datasets'], self.names[:2])
        self.assertEqual(tasks[2]['kind'], 'full_collection')
        for a, b in zip(tasks[:2], tasks[1:3]):
            self.assertLess(set(a['datasets']), set(b['datasets']))

    def test_ranking_component_count_is_verified(self):
        metadata = runner.inspect_ranking_components(self.ranking, expected_count=2)
        self.assertEqual(metadata['count'], 2)
        self.assertEqual(metadata['columns'], ['rank_alpha', 'rank_beta'])
        with self.assertRaisesRegex(ValueError, 'Expected a 10-component ranking'):
            runner.inspect_ranking_components(self.ranking, expected_count=10)

    def test_prepare_only_is_distinct_from_dry_run(self):
        args = runner.parse_args(['--prepare-only'])
        self.assertTrue(args.prepare_only)
        self.assertFalse(args.dry_run)
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            runner.parse_args(['--prepare-only', '--dry-run'])

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "run_real_exp8_L_stability_quality_rank.slurm").is_file(),
        "Site-specific Slurm launchers are intentionally excluded from the public tree.",
    )
    def test_slurm_array_has_one_job_and_gpu_per_size(self):
        script = runner.PROJECT_ROOT / 'run_real_exp8_L_stability_quality_rank.slurm'
        source = script.read_text()
        self.assertIn('#SBATCH --array=0-6', source)
        self.assertIn('#SBATCH --gres=gpu:1', source)
        self.assertNotIn('#SBATCH --gres=gpu:2', source)
        result = subprocess.run(['bash', str(script), '--list-tasks'],
                                capture_output=True, text=True, check=True)
        rows = [line.split('\t') for line in result.stdout.splitlines()]
        self.assertEqual([int(row[0]) for row in rows], list(range(7)))
        self.assertEqual([row[1] for row in rows],
                         ['N114', 'N80', 'N40', 'N20', 'N10', 'N5', 'N2'])
        self.assertEqual(len({row[2] for row in rows}), 7)

    def test_invalid_designs_rejected(self):
        for sizes in ([], [4], [2, 3], [2, 2, 4], [1, 4]):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                runner.build_cumulative_tasks(self.names, self.ranking, sizes, [42])
        with self.assertRaises(ValueError):
            runner.build_cumulative_tasks(self.names + ['absent'], self.ranking, [2, 5], [42])

    def test_config_and_command_use_global_not_subset_ranks(self):
        args = runner.parse_args(['--quality-ranking-table', str(self.ranking),
                                  '--expected-rank-components', '2'])
        args.ranking_component_count = 2
        args.ranking_component_columns = ['rank_alpha', 'rank_beta']
        tasks, _ = runner.build_cumulative_tasks(self.names, self.ranking, [2, 4], [42])
        config, command, reference = runner.configure_task(args, tasks[0], self.root,
            self.root / 'split.json', self.root / 'reliability.json', self.root / 'seq.parquet',
            runner.base._load_yaml(args.config), runner.base._load_yaml(args.dataset_config), self.ranking)
        self.assertEqual(config['model']['gamma_centering']['reference']['weighting'], 'quality_rank')
        self.assertEqual(config['model']['gamma_centering']['reference']['quality_rank_power'], 1.)
        self.assertEqual(config['data']['dataset_quality_ranking']['path'], str(self.ranking))
        for key, value in runner.ranked_overrides(args, self.ranking).items():
            self.assertEqual([part for part in command if part.startswith(key + '=')],
                             [f'{key}={json.dumps(value)}'])
        # R=10 from the *whole* frozen table, not rank 3 or rank 7 from a subset.
        self.assertAlmostEqual(reference['base_quality_weight']['d01'], .8)
        self.assertAlmostEqual(reference['pi']['d00'], 1 / 1.8)
        self.assertAlmostEqual(sum(reference['pi'].values()), 1.)
        self.assertFalse(reference['ranks_recomputed_within_panel'])
        self.assertIn('experiment.from_checkpoint=false', command)

    def test_resume_hash_ignores_only_audit_timestamps(self):
        a = {'created_at_utc': 'yesterday', 'nested': [{'created_at_utc': 'old', 'rank': 2}]}
        b = {'created_at_utc': 'today', 'nested': [{'created_at_utc': 'new', 'rank': 2}]}
        self.assertEqual(runner.stable_hash(a), runner.stable_hash(b))
        b['nested'][0]['rank'] = 3
        self.assertNotEqual(runner.stable_hash(a), runner.stable_hash(b))

    def test_dry_run_real_split_reliability_and_resume(self):
        ids = [f't{i:03d}' for i in range(60)]
        sequence = self.root / 'sequences.parquet'
        pd.DataFrame({'transcript_id': ids, 'codons': [['ATG', 'AAA', 'TAA']] * len(ids),
                      'css': [[]] * len(ids)}).to_parquet(sequence, index=False)
        mapping = {}
        for name in self.names:
            path = self.root / f'{name}.parquet'
            pd.DataFrame({'id': ids, 'weight': np.linspace(.5, 1.5, len(ids)),
                'coverage': np.full(len(ids), 2 / 3), 'read_density': np.ones(len(ids)),
                'ribo': [np.array([1., 0., 2.])] * len(ids)}).to_parquet(path, index=False)
            mapping[name] = str(path)
        dataset_config = self.root / 'datasets.yaml'
        dataset_config.write_text(yaml.safe_dump(mapping))
        argv = ['--quality-ranking-table', str(self.ranking), '--expected-rank-components', '2',
            '--dataset-config', str(dataset_config),
            '--sequences-path', str(sequence), '--dataset-sizes', '2,4', '--expected-dataset-count', '4',
            '--output-root', str(self.root), '--run-id', 'test', '--dry-run']
        with patch.object(runner.base, 'load_dataset_mapping', return_value=mapping), \
             patch.object(runner.base, '_inspect_weighted_dataset_sources', return_value={'status': 'PASS'}), \
             patch.object(runner.base, '_load_or_compute_quality', return_value=(toy_quality(4), [], {})), \
             patch.object(runner.base, '_queue_tasks') as launch:
            self.assertEqual(runner.main(argv), 0)
            run = self.root / 'test_dry_run'
            self.assertTrue((run / 'frozen_ranking.tsv').is_file())
            split = json.loads((run / 'experiment_split_manifest.json').read_text())
            self.assertTrue(split['common_test_ids'])
            manifests = list(run.glob('N*/trainseed*/subset_manifest.json'))
            self.assertEqual(len(manifests), 2)
            for path in manifests:
                task = json.loads(path.read_text())
                name = task['run_id']
                self.assertFalse(set(split['panel_train_eligible_ids'][name]) & set(split['common_test_ids']))
                self.assertEqual(task['fixed_gamma_reference']['weighting'], 'quality_rank')
                self.assertTrue((path.parent / 'logs').is_dir())
            self.assertEqual(runner.main(argv + ['--resume']), 0)
            launch.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'hash changed'):
                runner.main(argv + ['--resume', '--quality-rank-power', '2'])

    def test_analysis_from_original_profile_arrays(self):
        tasks, _ = runner.build_cumulative_tasks(self.names, self.ranking, [2, 4], [42])
        ids = ['t1', 't2']
        profile_arrays = [[np.array([.5, 1., 1.5]), np.ones(3)],
                          [np.array([.6, .9, 1.5]), np.ones(3)]]
        for task, arrays in zip(tasks, profile_arrays):
            directory = runner.task_directory(self.root, task)
            directory.mkdir(parents=True)
            task['directory'] = str(directory.relative_to(self.root))
            profile = directory / 'L.parquet'
            pd.DataFrame({'transcript_id': ids, 'transcript_length': [3, 3],
                'L_t': arrays, 'valid_position_mask': [[True] * 3] * 2}).to_parquet(profile, index=False)
            (directory / 'selected_checkpoint.json').write_text(json.dumps(dict(
                checkpoint_variant='best_val_loss', test_transcript_id_hash=transcript_id_hash(ids),
                shared_profile_path=str(profile), checkpoint_path='fixture-only.ckpt')))
        (self.root / 'experiment_manifest.json').write_text(json.dumps(dict(
            experiment_design='cumulative_top_quality', tasks=tasks)))
        (self.root / 'common_test_manifest.json').write_text(json.dumps({'common_test_ids': ids}))
        self.assertEqual(analyze(['--run-root', str(self.root)]), 0)
        summary = pd.read_csv(self.root / 'analysis/cumulative_agreement_summary.csv')
        self.assertTrue((summary.usable_transcripts == 1).all())
        self.assertTrue((summary.unusable_transcripts == 1).all())
        expected = np.corrcoef(profile_arrays[0][0], profile_arrays[1][0])[0, 1]
        np.testing.assert_allclose(summary.mean_PCC, expected)
        for suffix in ('pdf', 'png'):
            self.assertGreater((self.root / f'analysis/cumulative_convergence.{suffix}').stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
