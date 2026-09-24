from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from analyses import analyze_real_exp8_quality_rank_partial as partial
from Utils.reliability_references import transcript_id_hash


class PartialRankedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ids = ['t1', 't2']
        self.tasks = [dict(run_id=f'run{n}', N=n, training_seed=42,
            directory=f'N{n:03d}/trainseed42', datasets=[f'd{i}' for i in range(n)],
            kind='full_collection' if n == 4 else 'cumulative_top_quality') for n in [2,3,4]]

    def export(self, task, values=None):
        directory = self.root/task['directory']
        pred = directory/'predictions'/'model'
        pred.mkdir(parents=True)
        path = pred/'common_test_L_profiles.parquet'
        if values is None:
            values = [[.5,1.,1.5],[1.,1.,1.]]
        pd.DataFrame(dict(transcript_id=self.ids, transcript_length=[3,3],
            L_t=values, valid_position_mask=[[True]*3]*2, run_id=[task['run_id']]*2,
            N=[task['N']]*2)).to_parquet(path,index=False)
        prefix='/cluster/results/experiment/'+task['directory']
        runtime=dict(checkpoint_path=prefix+'/checkpoints/best.ckpt',
            sequence_only_shared_profile_prediction=True,split_name='test',
            transcript_id_hash=transcript_id_hash(self.ids),transcript_count=2,
            shared_profile_output_path=prefix+'/predictions/model/common_test_L_profiles.parquet')
        (pred/'prediction_checkpoint_manifest.json').write_text(json.dumps({'best_val_loss':runtime}))
        selected=dict(checkpoint_variant='best_val_loss',run_id=task['run_id'],
            test_transcript_id_hash=transcript_id_hash(self.ids),
            source_runtime_manifest=prefix+'/predictions/model/prediction_checkpoint_manifest.json',
            shared_profile_path=runtime['shared_profile_output_path'],checkpoint_path=runtime['checkpoint_path'])
        (directory/'selected_checkpoint.json').write_text(json.dumps(selected))
        pi=[1/task['N']]*task['N']
        (pred/'gamma_reference_manifest.json').write_text(json.dumps(dict(centering_mode='fixed_reference',
            weighting='quality_rank',quality_rank_power=1.,reference_dataset_names=task['datasets'],reference_pi=pi)))
        (directory/'subset_manifest.json').write_text(json.dumps(dict(datasets=task['datasets'],
            fixed_gamma_reference=dict(quality_rank_power=1.,pi=dict(zip(task['datasets'],pi))))))
        return path

    def test_relocated_export_and_missing_runs(self):
        self.export(self.tasks[0])
        profiles, checks, provenance, availability = partial.collect(self.root,
            {'tasks':self.tasks},self.ids,1e-4)
        self.assertEqual(list(profiles),['run2'])
        self.assertEqual(availability.available.tolist(),[True,False,False])
        self.assertEqual(len(provenance[0]['profile_sha256']),64)
        self.assertTrue(provenance[0]['profile_path'].startswith(str(self.root)))

    def test_bad_hash_is_audited_and_omitted(self):
        self.export(self.tasks[0])
        path=self.root/self.tasks[0]['directory']/'selected_checkpoint.json'
        payload=json.loads(path.read_text());payload['test_transcript_id_hash']='wrong'
        path.write_text(json.dumps(payload))
        profiles, _, _, availability=partial.collect(self.root,{'tasks':self.tasks},self.ids,1e-4)
        self.assertFalse(profiles)
        self.assertEqual(availability.iloc[0].export_status,'invalid')
        self.assertIn('hash mismatch',availability.iloc[0].reason)

    def test_unconsolidated_export_is_allowed_but_not_last_checkpoint(self):
        self.export(self.tasks[0])
        path=self.root/self.tasks[0]['directory']/'selected_checkpoint.json'
        # Fixture-only removal simulates the interval before orchestration consolidation.
        path.unlink()
        _,_,source=partial.load_export(self.root,self.tasks[0],self.ids,1e-4)
        self.assertFalse(source['selected_checkpoint_present'])
        runtime=next((self.root/self.tasks[0]['directory']).rglob('prediction_checkpoint_manifest.json'))
        content=json.loads(runtime.read_text())
        runtime.write_text(json.dumps({'last':content['best_val_loss']}))
        with self.assertRaises(KeyError):
            partial.load_export(self.root,self.tasks[0],self.ids,1e-4)

    def test_wrong_amplitudes_not_silently_rescaled(self):
        self.export(self.tasks[0],[[1.,2.,3.],[1.,1.,1.]])
        with self.assertRaisesRegex(ValueError,'not mean-one'):
            partial.load_export(self.root,self.tasks[0],self.ids,1e-4)

    def test_missing_middle_size_is_not_adjacent_or_fake_full(self):
        self.export(self.tasks[0])
        self.export(self.tasks[2],[[.6,.9,1.5],[1.,1.,1.]])
        profiles,_,_,_=partial.collect(self.root,{'tasks':self.tasks},self.ids,1e-4)
        values=partial.compare_available(self.tasks,profiles,self.ids)
        self.assertFalse(values.adjacent_planned.any())
        self.assertTrue(values.to_full.all())
        summary=partial.summarize(values)
        self.assertEqual(summary.iloc[0].n_usable,1)
        self.assertEqual(summary.iloc[0].n_undefined,1)
        np.testing.assert_allclose(summary.iloc[0].median_PCC,np.corrcoef([.5,1.,1.5],[.6,.9,1.5])[0,1])
        # A subset must not become a full reference merely because it is the largest export.
        partial_tasks=[dict(t,kind='cumulative_top_quality') for t in self.tasks]
        values=partial.compare_available(partial_tasks,profiles,self.ids)
        self.assertFalse(values.to_full.any())

    def test_example_ties_lexicographic_and_missing_pair_excluded(self):
        values=pd.DataFrame([dict(transcript_id=t,N_a=a,N_b=b,PCC=v)
            for t,v in [('z',.8),('a',.8),('bad',np.nan)] for a,b in [(2,3),(2,4),(3,4)]])
        table,chosen=partial.select_examples(values)
        self.assertEqual(chosen,'a')
        self.assertFalse(table.set_index('transcript_id').loc['bad','eligible'])

    def test_path_relocation_never_uses_another_task(self):
        path=self.export(self.tasks[0])
        with self.assertRaises(ValueError):
            partial.relocated(self.root,self.root/self.tasks[1]['directory'],str(path))

    def test_equal_design_resolves_directories_and_requires_uniform_pi(self):
        manifest = dict(
            experiment_name='real_exp8_L_stability',
            gamma_centering_mode='fixed_reference',
            gamma_pi_strategy='uniform within each selected subset',
            sampling_mode='quality_matched',
            tasks=[
                dict(run_id='equal_A', N=2, training_seed=42,
                     datasets=['a', 'b'], source_families=['fa', 'fb'],
                     kind='designated_disjoint_pair', pair_id='pair01', side='A'),
                dict(run_id='equal_B', N=2, training_seed=42,
                     datasets=['c', 'd'], source_families=['fc', 'fd'],
                     kind='designated_disjoint_pair', pair_id='pair01', side='B'),
            ],
        )
        contract = partial.identify_design(manifest)
        tasks = partial.prepare_tasks(self.root, manifest, contract['key'])
        self.assertEqual(tasks[0]['directory'], 'N002/pair01_A')
        self.export(tasks[0])
        directory = self.root / tasks[0]['directory']
        gamma_path = next(directory.rglob('gamma_reference_manifest.json'))
        gamma = json.loads(gamma_path.read_text())
        gamma.update(weighting='equal', quality_rank_power=0.0)
        gamma_path.write_text(json.dumps(gamma))
        subset_path = directory / 'subset_manifest.json'
        subset = json.loads(subset_path.read_text())
        subset['fixed_gamma_reference'] = dict(
            weighting='equal', pi={'a': 0.5, 'b': 0.5}
        )
        subset_path.write_text(json.dumps(subset))
        _, _, source = partial.load_export(
            self.root, tasks[0], self.ids, 1e-4, expected_weighting='equal'
        )
        self.assertEqual(source['gamma_reference_weighting'], 'equal')
        self.assertAlmostEqual(source['gamma_effective_reference_count'], 2.0)

        gamma['reference_pi'] = [0.6, 0.4]
        gamma_path.write_text(json.dumps(gamma))
        subset['fixed_gamma_reference']['pi'] = {'a': 0.6, 'b': 0.4}
        subset_path.write_text(json.dumps(subset))
        with self.assertRaises(AssertionError):
            partial.load_export(
                self.root, tasks[0], self.ids, 1e-4, expected_weighting='equal'
            )

    def test_equal_design_compares_only_prespecified_estimands(self):
        def task(run_id, N, kind, **metadata):
            return dict(run_id=run_id, N=N, training_seed=42, kind=kind,
                        datasets=[f'{run_id}_{i}' for i in range(N)], **metadata)

        tasks = [
            task('n2_p1a', 2, 'designated_disjoint_pair', pair_id='pair01', side='A'),
            task('n2_p1b', 2, 'designated_disjoint_pair', pair_id='pair01', side='B'),
            task('n2_p2a', 2, 'designated_disjoint_pair', pair_id='pair02', side='A'),
            task('n2_p2b', 2, 'designated_disjoint_pair', pair_id='pair02', side='B'),
            task('n80_s1', 80, 'large_N_subset', subset_id='subset01'),
            task('n80_s2', 80, 'large_N_subset', subset_id='subset02'),
            task('full', 114, 'full_collection'),
        ]
        comparisons = partial.planned_comparisons(tasks, design=partial.EQUAL_DESIGN)
        pairs = {(row['left']['run_id'], row['right']['run_id']): row for row in comparisons}
        self.assertIn(('n2_p1a', 'n2_p1b'), pairs)
        self.assertIn(('n2_p2a', 'n2_p2b'), pairs)
        self.assertNotIn(('n2_p1a', 'n2_p2a'), pairs)
        self.assertNotIn(('n2_p1a', 'n80_s1'), pairs)
        self.assertTrue(pairs[('n2_p1a', 'n2_p1b')]['primary_comparison'])
        self.assertFalse(pairs[('n80_s1', 'n80_s2')]['primary_comparison'])
        self.assertIn(('n2_p1a', 'full'), pairs)
        self.assertTrue(pairs[('n2_p1a', 'full')]['to_full'])


if __name__=='__main__':
    unittest.main()
