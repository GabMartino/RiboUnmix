import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd
from analyses.analyze_cumulative_reference_weight_stability import observed_fit_tables, stability_rows, summarize_stability
from Utils.cumulative_dataset_pcc_report import dataset_mu_grid


def profile(values):
    return dict(values=np.array(values,dtype=float),length=len(values),coordinate_hash='same')


class ObservedFitTests(unittest.TestCase):
    def setUp(self):
        self.tasks = [dict(run_id='ready', datasets=['a', 'b', 'c'])]
        self.availability = pd.DataFrame([dict(task_id='ready', training_seed=42, arm='equal', N=3,
            status='validated_predictions', selected_epoch=2)])
        self.logs = pd.DataFrame([
            dict(task_id='ready', epoch=epoch, step=epoch * 10, wall_time=epoch,
                 source='run', tag=tag, value=value if epoch == 2 else .99)
            for epoch in (1, 2) for tag, value in [
                ('val_mu_pcc/a', .8), ('val_L_bio_pcc/a', .3),
                ('val_mu_pcc/b', .4), ('val_L_bio_pcc/b', .2),
                ('val_mu_pcc/c', .1), ('val_L_bio_pcc/c', -.1),
                ('val_mu_pcc', .9), ('val_L_bio_pcc', -.9)]])

    def test_selected_epoch_and_equal_dataset_reduction_ignore_global_averages(self):
        datasets, summary = observed_fit_tables(self.logs, self.availability, self.tasks, ['a', 'b'])
        result = summary.set_index('scope')
        self.assertAlmostEqual(result.loc['all_selected', 'mu_pcc'], 1.3 / 3)
        self.assertAlmostEqual(result.loc['all_selected', 'correction_gain'], .3)
        self.assertAlmostEqual(result.loc['best2_datasets', 'mu_pcc'], .6)
        self.assertAlmostEqual(result.loc['best2_datasets', 'L_bio_pcc'], .25)
        self.assertTrue(datasets.matched.all())
        self.assertEqual(set(datasets.mu_pcc_step), {20})

    def test_missing_or_unpaired_tags_do_not_silently_change_dataset_cohort(self):
        for mode in ('missing', 'different_pass'):
            with self.subTest(mode=mode):
                logs = self.logs.copy()
                selected = (logs.epoch == 2) & (logs.tag == 'val_L_bio_pcc/c')
                if mode == 'missing':
                    logs = logs[~selected]
                else:
                    logs.loc[selected, 'step'] = 21
                _, summary = observed_fit_tables(logs, self.availability, self.tasks, ['a', 'b'])
                result = summary.set_index('scope')
                self.assertTrue(np.isnan(result.loc['all_selected', 'mu_pcc']))
                self.assertEqual(result.loc['all_selected', 'n_matched_datasets'], 2)
                self.assertEqual(result.loc['all_selected', 'missing_datasets'], 'c')
                self.assertAlmostEqual(result.loc['best2_datasets', 'mu_pcc'], .6)

    def test_unfinished_model_does_not_contribute_selected_fit(self):
        self.availability['status'] = 'checkpoint_without_export'
        _, summary = observed_fit_tables(self.logs, self.availability, self.tasks, ['a', 'b'])
        self.assertFalse(summary.complete.any())
        self.assertTrue(summary.mu_pcc.isna().all())


class DatasetMuGridTests(unittest.TestCase):
    def test_mu_survives_missing_lbio_and_membership_is_not_a_zero(self):
        weights = pd.DataFrame([
            dict(N=n, arm='equal', dataset_id=dataset, global_rank=rank)
            for n, members in [(2, [('b', 2), ('a', 1)]), (5, [('a', 1), ('b', 2), ('c', 3)])]
            for dataset, rank in members])
        observed = weights[['dataset_id', 'N', 'arm']].copy()
        observed['status'] = 'validated_predictions'
        observed['selected_epoch'] = 3
        observed['mu_pcc'] = .4
        observed['matched'] = False  # L_bio is absent; mu is still a valid diagnostic.
        observed.loc[(observed.dataset_id == 'a') & (observed.N == 2), 'mu_pcc'] = 0.
        grid = dataset_mu_grid(observed, weights, [2, 5], ['equal'])
        self.assertEqual(grid.dataset_id.drop_duplicates().tolist(), ['a', 'b', 'c'])
        rows = grid.set_index(['dataset_id', 'N'])
        self.assertEqual(rows.loc[('a', 2), 'cell_status'], 'available')
        self.assertEqual(rows.loc[('a', 2), 'mu_pcc'], 0.)
        self.assertAlmostEqual(rows.loc[('a', 5), 'mu_pcc'], .4)
        self.assertEqual(rows.loc[('c', 2), 'cell_status'], 'not_in_subset')
        self.assertTrue(np.isnan(rows.loc[('c', 2), 'mu_pcc']))

    def test_unselected_models_and_absent_tags_remain_distinct_gaps(self):
        weights = pd.DataFrame([dict(N=2, arm=arm, dataset_id='a', global_rank=1)
                                for arm in ['equal', 'ranked_p1', 'reverse_p1']])
        observed = weights[['dataset_id', 'N', 'arm']].copy()
        observed['status'] = ['checkpoint_without_export', 'validated_predictions', 'validated_predictions']
        observed['selected_epoch'] = [3, 3, np.nan]
        observed['mu_pcc'] = [.8, np.nan, .9]
        grid = dataset_mu_grid(observed, weights, [2], ['equal', 'ranked_p1', 'reverse_p1'])
        self.assertTrue(grid.mu_pcc.isna().all())
        self.assertEqual(grid.cell_status.tolist(), ['model_unavailable', 'mu_not_logged', 'model_unavailable'])


class CumulativeStabilityTests(unittest.TestCase):
    def test_missing_endpoint_never_bridges_adjacent_steps_but_anchor_can_exist(self):
        profiles={(42,'equal',2):{'t':profile([.5,1,1.5])},(42,'equal',10):{'t':profile([.6,1,1.4])}}
        rows=stability_rows(profiles,['t'],[2,5,10],42)
        self.assertTrue(rows[rows.kind=='adjacent'].empty)
        self.assertEqual(rows[['kind','N_a','N_b']].values.tolist(),[['anchor',2,10]])

    def test_paired_improvement_has_correct_sign_and_common_finite_cohort(self):
        profiles={}
        for arm in ('equal','ranked_p1'):
            profiles[42,arm,2]={'t':profile([.5,1,1.5]),'constant':profile([1,1,1])}
            profiles[42,arm,5]={'t':profile([1.5,1,.5] if arm=='equal' else [.5,1,1.5]),'constant':profile([1,1,1])}
        rows=stability_rows(profiles,['t','constant'],[2,5],42)
        summaries,effects,cohorts=summarize_stability(rows,['t','constant'])
        adjacent=effects[effects.kind=='adjacent'].set_index('metric')
        self.assertAlmostEqual(adjacent.loc['PCC','mean_improvement'],2.)
        self.assertEqual(adjacent.loc['PCC','n_valid'],1)
        self.assertEqual(adjacent.loc['PCC','n_excluded'],1)
        self.assertAlmostEqual(adjacent.loc['RMSE','mean_improvement'],np.sqrt(2/3)/2)
        self.assertEqual(adjacent.loc['RMSE','n_valid'],2)
        pcc=summaries[(summaries.kind=='adjacent')&(summaries.metric=='PCC')]
        self.assertEqual(set(pcc.n_valid),{1})
        self.assertFalse(cohorts[(cohorts.metric=='PCC')&(cohorts.transcript_id=='constant')].included.any())

    def test_no_ranked_endpoint_does_not_create_a_policy_effect(self):
        profiles={(42,'equal',n):{'t':profile([.5,1,1.5])} for n in (2,5)}
        rows=stability_rows(profiles,['t'],[2,5],42)
        summary,effects,_=summarize_stability(rows,['t'])
        self.assertFalse(summary.empty)
        self.assertTrue(effects.empty)

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "run_cumulative_reference_weight_stability_univie.slurm").is_file(),
        "Site-specific Slurm launchers are intentionally excluded from the public tree.",
    )
    def test_submitter_maps_all_35_single_seed_tasks(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);calls=directory/'calls';mock=directory/'sbatch'
            mock.write_text('''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
p=Path(os.environ['CALLS'])
with p.open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')
print(str(8000+len(p.read_text().splitlines()))+';cluster')
''')
            mock.chmod(0o755)
            env=dict(os.environ,PATH=f'{directory}:{os.environ["PATH"]}',CALLS=str(calls),PROJECT_DIR=str(root),MAX_CONCURRENT='2')
            subprocess.run(['bash',str(root/'submit_cumulative_reference_weight_stability_univie.sh')],env=env,check=True,capture_output=True)
            commands=[json.loads(line) for line in calls.read_text().splitlines()]
            self.assertEqual(len(commands),1)
            self.assertIn('--array=0-34%2',commands[0])
            worker=(root/'run_cumulative_reference_weight_stability_univie.slurm').read_text()
            self.assertIn('#SBATCH --array=0-34%2',worker)
            self.assertIn('run_cumulative_stability.py',worker)
            self.assertNotIn('SOURCE_ROOT',worker)


if __name__=='__main__':
    unittest.main()
