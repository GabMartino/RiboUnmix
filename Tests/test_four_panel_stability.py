import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from run_cumulative_stability import ROOT, object_sha256, run_task, sha256
from run_four_panel_stability import load_panels, prepare
from analyses.analyze_four_panel_stability import cross_panel_rows, peak_overlap, summarize_pairs
from analyses.analyze_real_exp8_reference_directionality import collect


def profile(values):
    return dict(values=np.asarray(values, dtype=float), length=len(values), coordinate_hash='same_cds')


class FourPanelSetupTests(unittest.TestCase):
    def test_shared_panel_inputs_and_global_weights_without_previous_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = [f'source{i}_2020' for i in range(8)]
            panels = {f'panel_{i+1:02d}': names[2*i:2*i+2] for i in range(4)}
            mapping = {name: f'/data/{name}.parquet' for name in names}
            (root / 'datasets.yaml').write_text(yaml.safe_dump({'dataset_path': mapping}))
            assignment = pd.DataFrame([dict(dataset_name=name, panel=panel)
                                       for panel, datasets in panels.items() for name in datasets])
            assignment.to_csv(root / 'panels.csv', index=False)
            pd.DataFrame(dict(dataset=names + ['unused'], quality_rank=list(range(1, 9)) + [10])).to_csv(
                root / 'ranking.tsv', sep='\t', index=False)
            args = argparse.Namespace(output_root=root / 'experiment', seed=42,
                                      config=ROOT / 'config/config_ribounmix_multidataset.yaml',
                                      datasets=root / 'datasets.yaml', panels=root / 'panels.csv',
                                      ranking=root / 'ranking.tsv', task_index=19, dry_run=True)
            split = dict(common_test_ids=['test'], common_validation_ids=['validation'],
                         panel_train_eligible_ids={p: [f'train_{p}'] for p in panels})
            with patch('run_four_panel_stability.build_common_transcript_split', return_value=split) as builder, \
                 patch('run_four_panel_stability.fit_panel_reliability_manifest', return_value={}) as fitter:
                plan = prepare(args)
                self.assertEqual(prepare(args), plan)
                self.assertEqual(builder.call_count, 1)
                self.assertEqual(fitter.call_count, 4)
                for call in fitter.call_args_list:
                    panel = call.kwargs['panel_name']
                    self.assertEqual(call.kwargs['panel_training_ids'], [f'train_{panel}'])
                    self.assertEqual(call.kwargs['test_ids'], ['test'])
                    self.assertEqual(call.kwargs['validation_ids'], ['validation'])
            self.assertEqual(len(plan['tasks']), 24)
            self.assertEqual(plan['tasks'][19]['panel_id'], 'panel_04')
            self.assertEqual([(t['panel_id'], t['arm']) for t in plan['tasks'][20:]],
                             [(p, 'shared_only') for p in panels])
            self.assertEqual(run_task(args, plan), 0)
            reference_paths = set()
            for panel in panels:
                configs = {t['arm']: yaml.safe_load(Path(t['config_path']).read_text())
                           for t in plan['tasks'] if t['panel_id'] == panel}
                paths = {cfg['data']['reliability_reference_manifest'] for cfg in configs.values()}
                self.assertEqual(len(paths), 1)
                reference_paths.update(paths)
                for cfg in configs.values():
                    self.assertEqual(cfg['split'], configs['equal']['split'])
                    self.assertEqual(cfg['orchestrator']['panel_id'], panel)
                    self.assertEqual(cfg['experiment']['seed'], 42)
                shared = copy.deepcopy(configs['shared_only'])
                self.assertEqual(shared['model']['mean_correction'], 'unity')
                self.assertEqual(shared['model']['alpha_mode'], 'learned')
                self.assertFalse(shared['experiment']['from_checkpoint'])
                expected_hash = shared['orchestrator'].pop('task_contract_sha256')
                self.assertEqual(expected_hash, object_sha256(shared))
                # Every scientific setting except the mean correction is identical.
                shared['model']['mean_correction'] = 'learned'
                shared['name'] = configs['equal']['name']
                shared['orchestrator'] = configs['equal']['orchestrator']
                for key in ('checkpoints', 'logs', 'results'):
                    shared['paths'][key] = configs['equal']['paths'][key]
                self.assertEqual(shared, configs['equal'])
                reference = lambda arm: configs[arm]['model']['gamma_centering']['reference']['explicit_weights']
                forward, reverse = reference('ranked_p3'), reference('reverse_p3')
                np.testing.assert_allclose(list(forward.values()), list(reverse.values())[::-1])
                first = panels[panel][0]
                rank = names.index(first) + 1
                self.assertAlmostEqual(forward[first], ((11-rank)/10) ** 3)
            self.assertEqual(len(reference_paths), 4)

            # An existing 20-task setup can be extended without refitting inputs
            # or changing task identities already used by queued/running models.
            old = copy.deepcopy(plan)
            old['tasks'] = old['tasks'][:20]
            old['tasks_sha256'] = object_sha256(old['tasks'])
            old.pop('shared_only_ablation')
            old['frozen_file_sha256'] = {p: h for p, h in old['frozen_file_sha256'].items() if 'shared_only' not in p}
            weights_path = args.output_root / 'reference_weights.csv'
            weights = pd.read_csv(weights_path)
            weights[weights.arm != 'shared_only'].to_csv(weights_path, index=False)
            old['frozen_file_sha256'][str(weights_path)] = sha256(weights_path)
            (args.output_root / 'experiment_manifest.json').write_text(json.dumps(old))
            original_files = {p: Path(p).read_bytes() for p in old['frozen_file_sha256'] if Path(p) != weights_path}
            with patch('run_four_panel_stability.build_common_transcript_split') as builder, \
                 patch('run_four_panel_stability.fit_panel_reliability_manifest') as fitter:
                extended = prepare(args)
                builder.assert_not_called()
                fitter.assert_not_called()
            self.assertEqual(extended['tasks'][:20], old['tasks'])
            self.assertEqual(len(extended['tasks']), 24)
            for path, data in original_files.items():
                self.assertEqual(Path(path).read_bytes(), data)

    def test_source_family_cannot_cross_panels(self):
        names = ['author_2020_a', 'author_2020_b', 'b_2021', 'c_2022']
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'panels.csv'
            pd.DataFrame(dict(dataset_name=names, panel=['p1', 'p2', 'p3', 'p4'])).to_csv(path, index=False)
            with self.assertRaisesRegex(AssertionError, 'Related datasets'):
                load_panels(path, dict.fromkeys(names), pd.DataFrame(dict(dataset=names, quality_rank=[1,2,3,4])))


class FourPanelAnalysisTests(unittest.TestCase):
    def test_full_vs_shared_effect_sign_and_missing_baseline(self):
        panels = ['panel_01', 'panel_02']
        profiles = {(42, arm, panel): {'t': profile([.5, 1, 1.5]), 'flat': profile([1, 1, 1])}
                    for arm in ('equal', 'shared_only') for panel in panels}
        profiles[42, 'shared_only', panels[1]]['t'] = profile([1.5, 1, .5])
        rows = cross_panel_rows(profiles, ['t', 'flat'], panels, 42)
        _, effects, _ = summarize_pairs(rows, ['t', 'flat'])
        contrast = effects[effects.baseline == 'shared_only'].set_index('metric')
        self.assertEqual(set(contrast.policy), {'equal'})
        self.assertAlmostEqual(contrast.loc['PCC', 'mean_improvement'], 2.)
        self.assertGreater(contrast.loc['RMSE', 'mean_improvement'], 0)
        self.assertAlmostEqual(contrast.loc['peak_Jaccard', 'mean_improvement'], 1.)
        self.assertEqual(contrast.loc['PCC', 'n_valid'], 1)
        del profiles[42, 'shared_only', panels[1]]
        _, effects, _ = summarize_pairs(cross_panel_rows(profiles, ['t', 'flat'], panels, 42), ['t', 'flat'])
        self.assertTrue(effects.empty)

    def test_flat_or_tied_peaks_do_not_create_perfect_agreement(self):
        self.assertTrue(np.isnan(peak_overlap(profile([1, 1, 1]), profile([1, 1, 1]))))
        self.assertTrue(np.isnan(peak_overlap(profile([0, 2, 2]), profile([0, 2, 2]))))
        self.assertEqual(peak_overlap(profile([.5, 1, 1.5]), profile([.5, 1, 1.5])), 1.)

    def test_equal_sized_panels_do_not_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks, configs = [], {}
            for panel in ('panel_01', 'panel_02'):
                task = dict(panel_id=panel, training_seed=42, arm='equal', N=29,
                            directory=panel, array_index=len(tasks), run_id=panel)
                tasks.append(task)
                configs[42, 'equal', panel] = {}
                directory = root / panel
                (directory / 'hydra/.hydra').mkdir(parents=True)
                (directory / 'hydra/.hydra/config.yaml').write_text('{}')
                (directory / 'predictions').mkdir()
                (directory / 'predictions/prediction_checkpoint_manifest.json').write_text('{}')
                (directory / 'execution_status.json').write_text(json.dumps(dict(status='completed')))
            with patch('analyses.analyze_real_exp8_reference_directionality.read_export',
                       side_effect=lambda root, task, *args: ({'t': profile([.5, 1, 1.5])}, {})):
                profiles, availability = collect(root, {'tasks': tasks}, configs, ['t'], pd.DataFrame())
            self.assertEqual(len(profiles), 2)
            self.assertEqual(set(availability.panel_id), {'panel_01', 'panel_02'})
            self.assertTrue(availability.status.eq('validated_predictions').all())

    def test_six_pairs_and_paired_effects_use_matched_finite_transcripts(self):
        panels = [f'panel_{i:02d}' for i in range(1, 5)]
        profiles = {(42, arm, panel): {'t': profile([.5,1,1.5]), 'flat': profile([1,1,1])}
                    for arm in ('equal', 'ranked_p1') for panel in panels}
        profiles[42, 'equal', panels[1]]['t'] = profile([1.5,1,.5])
        rows = cross_panel_rows(profiles, ['t','flat'], panels, 42)
        self.assertEqual(len(rows[['panel_a','panel_b']].drop_duplicates()), 6)
        summary, effects, cohorts = summarize_pairs(rows, ['t','flat'])
        comparison = effects[(effects.panel_a == panels[0]) & (effects.panel_b == panels[1])].set_index('metric')
        self.assertAlmostEqual(comparison.loc['PCC', 'mean_improvement'], 2.)
        self.assertEqual(comparison.loc['PCC', 'n_valid'], 1)
        self.assertEqual(comparison.loc['RMSE', 'n_valid'], 2)
        self.assertGreater(comparison.loc['RMSE', 'mean_improvement'], 0)
        del profiles[42, 'ranked_p1', panels[1]]
        rows = cross_panel_rows(profiles, ['t','flat'], panels, 42)
        _, effects, _ = summarize_pairs(rows, ['t','flat'])
        self.assertFalse(((effects.panel_a == panels[1]) | (effects.panel_b == panels[1])).any())


if __name__ == '__main__':
    unittest.main()
