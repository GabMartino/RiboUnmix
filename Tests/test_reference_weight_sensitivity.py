import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from run_reference_weight_sensitivity import reference_policy, task_matrix, _assert_arm_match, object_sha256, sha256, load_cumulative_source, DESIGN, transcript_id_hash
from analyses.analyze_reference_weight_sensitivity import compare_profiles, strength_effect, summarize, reference_design_table, main as analyze


class ReferenceSensitivityTests(unittest.TestCase):
    def cumulative_fixture(self, root):
        recorded=Path('/cluster/project/results/cumulative')
        names=[f'd{r:03d}' for r in range(1,116) if r!=110]
        panel='old_N114_seed42'
        ranking=pd.DataFrame(dict(dataset=[f'd{r:03d}' for r in range(1,116)],quality_rank=range(1,116),rank_component_count=[10]*115))
        for i in range(10):
            ranking[f'rank_feature{i}']=range(1,116)
        ranking.to_csv(root/'ranking.tsv',sep='\t',index=False)
        folds=dict(source_panel=panel,train_ids=['tr1','tr2'],validation_ids=['val'],test_ids=['test'])
        (root/'split.json').write_text(json.dumps(dict(manifest_version=1,panels={panel:names},
            common_test_ids=folds['test_ids'],common_validation_ids=folds['validation_ids'],
            panel_train_eligible_ids={panel:folds['train_ids']})))
        (root/'reliability.json').write_text(json.dumps(dict(manifest_version=1,reference_split='training_only',
            heldout_rows_used_for_fitting=0,panel_training_transcript_count=2,
            panel_training_transcript_id_hash=transcript_id_hash(folds['train_ids']),datasets={d:{} for d in names})))
        cfg=dict(experiment=dict(dataset=names,seed=42),split=dict(external_manifest=str(recorded/'split.json'),external_panel_name=panel),
                 data=dict(dataset_quality_ranking=dict(path=str(recorded/'ranking.tsv')),reliability_reference_manifest=str(recorded/'reliability.json')))
        (root/'template.yaml').write_text(yaml.safe_dump(cfg))
        tasks=[dict(N=114,arm='equal',training_seed=42,datasets=names,source_panel=panel,
                    config_path=str(recorded/'template.yaml'),config_sha256=sha256(root/'template.yaml')),
               dict(N=2,arm='equal',training_seed=42,datasets=names[:2])]
        manifest=dict(experiment_design='matched_cumulative_reference_directionality',output_root=str(recorded),
                      tasks=tasks,tasks_sha256=object_sha256(tasks),source_folds={'114':folds},
                      ranking=dict(sha256=sha256(root/'ranking.tsv')),
                      frozen_file_sha256={str(recorded/p.name):sha256(p) for p in root.iterdir()})
        (root/'experiment_manifest.json').write_text(json.dumps(manifest))
        return manifest

    def test_cumulative_endpoint_retains_rank_gap_folds_and_reversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest=self.cumulative_fixture(root)
            base=load_cumulative_source(root)
            self.assertEqual(len(base['names']),114)
            self.assertEqual(base['names'][109],'d111')
            self.assertEqual(base['folds'],manifest['source_folds']['114'])
            _,rows=reference_policy(base['names'],base['ranks'],base['quality'],1,'ranked')
            self.assertAlmostEqual(1/sum(row['pi']**2 for row in rows),86.4736639185)
            tasks=task_matrix(base['names'],42,base['folds']['source_panel'])
            self.assertTrue(all(t['N']==114 and t['source_panel']=='old_N114_seed42' for t in tasks))

    def test_cumulative_source_rejects_changed_reliability_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.cumulative_fixture(root)
            with (root/'reliability.json').open('a') as f:
                f.write('\n')
            with self.assertRaisesRegex(ValueError,'Source frozen artifact changed'):
                load_cumulative_source(root)

    def test_design_comparison_keeps_previous_endpoint_equal_to_new_linear_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=self.cumulative_fixture(root)
            base=load_cumulative_source(root)
            rows=[]
            for task in task_matrix(base['names'],42):
                _,records=reference_policy(base['names'],base['ranks'],base['quality'],task['power'],task['direction'])
                rows.extend(dict(arm=task['arm'],**row) for row in records)
            table=reference_design_table(pd.DataFrame(rows),source)
            old=table[(table.design=='previous') & (table.N==114)].set_index('arm')
            new=table[table.design=='new'].set_index('arm')
            for previous,current in [('equal','equal'),('ranked','ranked_p1'),('reverse','reverse_p1')]:
                for metric in ('N_ref','reference_fraction','weighted_mean_rank'):
                    self.assertAlmostEqual(old.loc[previous,metric],new.loc[current,metric])
            self.assertAlmostEqual(new.loc['ranked_p3','reference_fraction'],50.53360045924242/114)
            self.assertAlmostEqual(new.loc['ranked_p3','N_ref'],new.loc['reverse_p3','N_ref'])
            self.assertLess(new.loc['ranked_p3','weighted_mean_rank'],new.loc['ranked_p1','weighted_mean_rank'])
            self.assertGreater(new.loc['reverse_p3','weighted_mean_rank'],new.loc['reverse_p1','weighted_mean_rank'])

    def test_cumulative_source_rejects_nonprefix_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest=self.cumulative_fixture(root)
            manifest['tasks'][1]['datasets']=['d002','d003']
            manifest['tasks_sha256']=object_sha256(manifest['tasks'])
            (root/'experiment_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'exact cumulative prefixes'):
                load_cumulative_source(root)

    def test_power_and_reverse_use_global_scores_with_gaps_and_original_order(self):
        names=['middle','worst','best']
        ranks={'best':5.,'middle':57.,'worst':106.}
        quality={name:(116-rank)/115 for name,rank in ranks.items()}
        expected=np.array([59.,10.,111.])**3
        expected/=expected.sum()
        implementation,forward=reference_policy(names,ranks,quality,3,'ranked')
        _,reverse=reference_policy(names,ranks,quality,3,'reverse')
        np.testing.assert_allclose([r['pi'] for r in forward],expected)
        np.testing.assert_allclose([r['pi'] for r in reverse],expected[[0,2,1]])
        self.assertEqual(implementation['weighting'],'explicit')
        self.assertEqual(reverse[1]['assigned_from_dataset'],'best')
        self.assertEqual(forward[0]['original_q'],59/115)
        self.assertEqual(forward[0]['assigned_q'],(59/115)**3)
        self.assertAlmostEqual(sum(r['pi']**2 for r in forward),sum(r['pi']**2 for r in reverse))

    def test_uniform_baseline_and_stronger_direction_separation(self):
        names=['best','middle','worst']
        ranks=dict(zip(names,[5.,57.,106.]))
        quality={name:(116-rank)/115 for name,rank in ranks.items()}
        _,equal=reference_policy(names,ranks,quality,0,'equal')
        np.testing.assert_allclose([r['pi'] for r in equal],np.ones(3)/3)
        distances=[]
        for power in (1,3):
            _,a=reference_policy(names,ranks,quality,power,'ranked')
            _,b=reference_policy(names,ranks,quality,power,'reverse')
            distances.append(.5*sum(abs(x['pi']-y['pi']) for x,y in zip(a,b)))
        self.assertGreater(distances[1],distances[0])
        with self.assertRaises(ValueError):
            reference_policy(names,ranks,quality,0,'reverse')

    def test_five_tasks_share_one_seed_and_membership(self):
        tasks=task_matrix(['b','a','c'],42)
        self.assertEqual(len(tasks),5)
        self.assertEqual({t['training_seed'] for t in tasks},{42})
        self.assertTrue(all(t['datasets']==['b','a','c'] for t in tasks))
        self.assertEqual([t['array_index'] for t in tasks],list(range(5)))
        self.assertEqual(len({t['directory'] for t in tasks}),5)

    def test_pair_contract_rejects_optimizer_changes(self):
        equal={'optim':{'lr':.001},'model':{'gamma_centering':{'reference':{'explicit_weights':{'a':1.}}}}}
        other=copy.deepcopy(equal)
        other['model']['gamma_centering']['reference']['explicit_weights']['a']=.2
        _assert_arm_match(equal,other)
        other['optim']['lr']=.01
        with self.assertRaisesRegex(ValueError,'optim.lr'):
            _assert_arm_match(equal,other)

    def test_strength_effect_is_paired_and_detects_amplitude_without_pcc_change(self):
        tasks=task_matrix(['a','b'],42)
        def profile(values):
            return {'values':np.array(values,dtype=float),'length':4,'coordinate_hash':'same'}
        profiles={(42,t['arm'],2):{'t':profile([.5,1.,1.,1.5])} for t in tasks}
        profiles[42,'reverse_p3',2]['t']=profile([0.,1.,1.,2.])
        frame=compare_profiles(profiles,tasks,['t'])
        delta=strength_effect(frame).iloc[0]
        self.assertAlmostEqual(delta.delta_one_minus_PCC,0.)
        self.assertAlmostEqual(delta.delta_RMSE,np.sqrt(.125))
        self.assertEqual(len(frame),8)
        self.assertTrue((summarize(frame).n_valid==1).all())
        # Missing p=3 is not replaced by another arm or a zero-effect observation.
        del profiles[42,'reverse_p3',2]
        self.assertTrue(strength_effect(compare_profiles(profiles,tasks,['t'])).empty)

    def test_empty_and_undefined_metrics_are_not_imputed(self):
        tasks=task_matrix(['a','b'],42)
        empty=compare_profiles({},tasks,['t'])
        self.assertTrue(summarize(empty).empty)
        self.assertEqual(strength_effect(empty).delta_RMSE.dtype,np.dtype(float))
        rows=pd.DataFrame([dict(arm_a='ranked_p3',arm_b='reverse_p3',PCC=np.nan,RMSE=0.,variance_a=0.,variance_b=0.)])
        result=summarize(rows).set_index('metric')
        self.assertEqual(result.loc['PCC','n_excluded'],1)
        self.assertTrue(np.isnan(result.loc['PCC','mean']))

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "run_reference_weight_sensitivity_univie.slurm").is_file(),
        "Site-specific Slurm launchers are intentionally excluded from the public tree.",
    )
    def test_single_command_submits_cpu_prepare_five_gpu_tasks_and_cpu_analysis(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            mock=d/'sbatch'
            mock.write_text('''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
p=Path(os.environ['CALLS'])
with p.open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')
print(str(8000+len(p.read_text().splitlines()))+';cluster')
''')
            mock.chmod(0o755)
            calls=d/'calls'
            env=dict(os.environ,PATH=f'{d}:{os.environ["PATH"]}',CALLS=str(calls),PROJECT_DIR=str(root),MAX_CONCURRENT='2')
            subprocess.run(['bash',str(root/'submit_reference_weight_sensitivity_univie.sh')],env=env,check=True,capture_output=True)
            commands=[json.loads(line) for line in calls.read_text().splitlines()]
            self.assertEqual(len(commands),3)
            self.assertIn('--gres=none',commands[0])
            self.assertIn('--dependency=afterok:8001',commands[1])
            self.assertIn('--array=0-4%2',commands[1])
            self.assertIn('--kill-on-invalid-dep=yes',commands[1])
            self.assertIn('--dependency=afterany:8002',commands[2])
            self.assertIn('--gres=none',commands[2])

    def test_analysis_renders_available_profiles_and_paired_strength_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            tasks=task_matrix(['a','b'],42)
            profiles={}
            frozen={}
            for task in tasks:
                path=root/f'{task["arm"]}.yaml'
                path.write_text('{}\n')
                task['config_path']=str(path)
                frozen[str(path)]=sha256(path)
                scale=.5 if task['arm']=='reverse_p3' else .25
                profiles[42,task['arm'],2]={'t':dict(values=np.array([1-scale,1.,1.,1+scale]),length=4,coordinate_hash='same')}
            (root/'reference_weights.csv').write_text('N,arm,dataset_id,pi\n')
            manifest=dict(experiment_design=DESIGN,output_root=str(root),
                          tasks=tasks,tasks_sha256=object_sha256(tasks),frozen_file_sha256=frozen,
                          training_seeds=[42],source_folds={'2':{'test_ids':['t']}})
            (root/'experiment_manifest.json').write_text(json.dumps(manifest))
            availability=pd.DataFrame([dict(arm=t['arm'],status='validated_predictions',recorded_status='completed',selected_epoch=1) for t in tasks])
            with patch('analyses.analyze_reference_weight_sensitivity.collect',return_value=(profiles,availability)):
                self.assertEqual(analyze(['--experiment-root',str(root)]),0)
            self.assertTrue((root/'analysis/direction_sensitivity.svg').is_file())
            summary=pd.read_csv(root/'analysis/strength_effect_summary.csv').set_index('metric')
            self.assertGreater(summary.loc['delta_RMSE','mean'],0)
            self.assertIn('5/5 validated exports',(root/'analysis/analysis_report.html').read_text())


if __name__=='__main__':
    unittest.main()
