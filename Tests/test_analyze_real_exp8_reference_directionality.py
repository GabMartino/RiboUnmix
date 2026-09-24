from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyses.analyze_real_exp8_reference_directionality import (
    SLOT_COLUMNS, adjacent_summaries, bootstrap_means, comparison_availability,
    descriptive_summary, evaluation_tables, pair_record, plot_stability,
    verify_runtime_config,
)


def profile(values, coordinate_hash='same_cds'):
    return dict(values=np.asarray(values, dtype=float), length=len(values),
                coordinate_hash=coordinate_hash)


class CumulativeDirectionalityAnalysisTests(unittest.TestCase):
    def test_partial_exports_do_not_manufacture_policy_effects(self):
        x=np.linspace(.2,1.8,80)
        profiles={(43,'ranked',2):{'t':profile(x)},
                  (43,'ranked',5):{'t':profile(x*.8+.2)},
                  (43,'reverse',2):{'t':profile(x*.9+.1)}}
        adjacent,sensitivity,optimization,diagnostics=evaluation_tables(profiles,['t'],[2,5,10])
        self.assertEqual(set(adjacent.arm),{'ranked'})
        self.assertEqual(set(zip(adjacent.N_a,adjacent.N_b)),{(2,5)})
        self.assertEqual(set(sensitivity.contrast),{'ranked_vs_reverse'})
        self.assertTrue(optimization.empty)
        self.assertEqual(len(diagnostics),6)
        summary,effects,cohorts=adjacent_summaries(adjacent,['t'],2,40,42)
        self.assertEqual(len(summary),6)
        self.assertTrue(effects.empty)
        self.assertTrue(cohorts.included.all())
        availability=pd.DataFrame([dict(training_seed=s,arm=a,N=n,status='validated_predictions')
                                   for s,a,n in profiles])
        missing=comparison_availability(availability,[2,5,10],[43])
        self.assertFalse(missing.ready.any())
        ranked_reverse=missing[(missing.contrast=='ranked_minus_reverse')&(missing.N_a==2)].iloc[0]
        self.assertEqual(ranked_reverse.missing_models,'seed43/reverse/N005')

    def test_never_bridge_missing_size_or_mix_training_seeds(self):
        x=profile(np.linspace(.2,1.8,80))
        profiles={(42,'equal',2):{'t':x},(42,'equal',10):{'t':x},
                  (43,'equal',5):{'t':x},(43,'ranked',2):{'t':x}}
        adjacent,sensitivity,optimization,_=evaluation_tables(profiles,['t'],[2,5,10])
        self.assertTrue(adjacent.empty)
        self.assertTrue(sensitivity.empty)
        self.assertTrue(optimization.empty)
        profiles[43,'equal',2]={'t':x}
        _,_,optimization,_=evaluation_tables(profiles,['t'],[2,5,10])
        self.assertEqual(set(zip(optimization.seed_a,optimization.seed_b)),{(42,43)})
        self.assertEqual(set(optimization.N),{2})

    def test_alignment_undefined_metrics_and_original_interior_amplitudes(self):
        flat=profile(np.ones(80))
        result=pair_record(flat,flat,'full_cds')
        self.assertTrue(np.isnan(result['PCC']))
        self.assertEqual(result['RMSE'],0)
        self.assertEqual(result['reason'],'nearly_constant_profile')
        for other in (profile(np.ones(79)),profile(np.ones(80),'different_cds')):
            result=pair_record(flat,other,'full_cds')
            self.assertTrue(np.isnan(result['RMSE']))
            self.assertEqual(result['reason'],'coordinate_or_length_mismatch')
        x=np.linspace(.1,2.5,80);y=x*.7+.3
        result=pair_record(profile(x),profile(y),'interior_20')
        self.assertAlmostEqual(result['RMSE'],np.sqrt(np.mean((x[20:-20]-y[20:-20])**2)))
        self.assertEqual(result['n_positions'],40)

    def test_joint_bootstrap_matches_manual_resampling(self):
        x=np.array([[.1,.4],[.2,.1],[.8,.9],[.9,.5]])
        point,samples,digest=bootstrap_means(x,73,17)
        idx=np.random.default_rng(17).integers(0,len(x),size=(73,len(x)))
        np.testing.assert_array_equal(samples,x[idx].mean(axis=1))
        np.testing.assert_array_equal(point,x.mean(axis=0))
        self.assertEqual(digest,bootstrap_means(x,73,17)[2])
        repeated=np.column_stack([x[:,0],x[:,0]+.1])
        _,samples,_=bootstrap_means(repeated,100,17)
        np.testing.assert_allclose(samples[:,1]-samples[:,0],.1,atol=1e-15)
        for invalid in (np.empty((0,2)),np.array([[np.nan]])):
            with self.assertRaises(ValueError):bootstrap_means(invalid,100,17)

    def test_difference_of_means_and_global_available_cohort(self):
        a=np.array([0.,.1,.2,.8,.9]);b=np.array([.05,.11,.15,.81,.9])
        ids=[f't{i}' for i in range(5)]
        rows=[]
        for seed in (42,43):
            for j,(n,m) in enumerate(((2,5),(5,10))):
                for arm,values in [('equal',a),('ranked',b)]:
                    for tid,x in zip(ids,values):
                        rows.append(dict(training_seed=seed,arm=arm,transition=j,N_a=n,N_b=m,
                            transcript_id=tid,domain='full_cds',PCC=x,Spearman=x,RMSE=1-x))
        frame=pd.DataFrame(rows)
        summary,effects,cohorts=adjacent_summaries(frame,ids,2,100,42)
        pcc=effects[effects.metric=='PCC']
        np.testing.assert_allclose(pcc.estimate,b.mean()-a.mean())
        self.assertNotAlmostEqual(pcc.iloc[0].estimate,np.median(b)-np.median(a))
        self.assertTrue(pcc.all_transitions_matched.all())
        self.assertEqual(pcc.bootstrap_index_sha256.nunique(),1)
        self.assertEqual(pcc.ci_low.nunique(),1)
        self.assertEqual(pcc.ci_high.nunique(),1)
        frame.loc[(frame.transcript_id=='t0')&(frame.training_seed==43)&(frame.N_a==5)&(frame.arm=='ranked'),'PCC']=np.nan
        summary,effects,cohorts=adjacent_summaries(frame,ids,2,100,42)
        self.assertTrue((summary.loc[summary.metric=='PCC','n_transcripts']==4).all())
        self.assertTrue((summary.loc[summary.metric=='RMSE','n_transcripts']==5).all())
        self.assertEqual(effects.loc[effects.metric=='PCC','cohort_hash'].nunique(),1)
        self.assertFalse(cohorts[(cohorts.transcript_id=='t0')&(cohorts.metric=='PCC')].included.any())

    def test_empty_csv_schemas_and_future_three_arm_plots(self):
        empty,_,_,_=evaluation_tables({},['t'],[2,5,10])
        summary,effects,cohorts=adjacent_summaries(empty,['t'],2)
        with tempfile.TemporaryDirectory() as temp,plt.rc_context({'text.usetex':False}):
            out=Path(temp)
            self.assertEqual(plot_stability(summary,effects,[2,5,10],out,50),[])
            for i,table in enumerate((summary,effects,cohorts,descriptive_summary(empty,SLOT_COLUMNS+['domain']))):
                path=out/f'empty{i}.csv';table.to_csv(path,index=False)
                self.assertTrue(pd.read_csv(path).empty)
            x=np.linspace(.2,1.8,80)
            profiles={(43,arm,n):{'t1':profile(x*(1-.1*j)+.1*j),'t2':profile(x[::-1])}
                      for j,arm in enumerate(('equal','ranked','reverse')) for n in (2,5,10)}
            adjacent,_,_,_=evaluation_tables(profiles,['t1','t2'],[2,5,10])
            summary,effects,_=adjacent_summaries(adjacent,['t1','t2'],2,40,42)
            plot_stability(summary,effects,[2,5,10],out,50)
            for stem in ('adjacent_stability_seed43','adjacent_policy_effects_seed43'):
                for suffix in ('.pdf','.svg','.png'):self.assertTrue((out/(stem+suffix)).is_file())

    def test_runtime_configuration_only_allows_recorded_full_state_resume(self):
        cfg={'experiment':{'from_checkpoint':False,'resume_training_state':False,
                           'resume_checkpoint_path':None,'allow_weights_only_resume':False},
             'optimizer':{'lr':.001}}
        verify_runtime_config(cfg,deepcopy(cfg),{})
        actual=deepcopy(cfg);actual['optimizer']['lr']=.01
        with self.assertRaises(ValueError):verify_runtime_config(cfg,actual,{})
        actual=deepcopy(cfg);actual['experiment'].update(from_checkpoint=True,resume_training_state=True,
                                                        resume_checkpoint_path='/run/last.ckpt')
        state={'attempts':[{'resume_checkpoint':{'path':'/run/last.ckpt'}}]}
        verify_runtime_config(cfg,actual,state)
        with self.assertRaises(ValueError):verify_runtime_config(cfg,actual,{})


if __name__=='__main__':unittest.main()
