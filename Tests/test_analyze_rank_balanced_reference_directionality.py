from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyses.analyze_rank_balanced_reference_directionality import (
    aligned_metrics, cross_policy_effects, evaluation_tables, example_selections,
    plot_cross_panel, plot_sensitivity, reference_tables,
)


def profile(values, coordinate_hash='same_cds'):
    return dict(values=np.asarray(values,dtype=float),length=len(values),coordinate_hash=coordinate_hash)


class DirectionalityAnalysisTests(unittest.TestCase):
    def test_partial_download_is_sensitivity_not_cross_panel_ranking_effect(self):
        x=np.linspace(.1,1.9,80)
        profiles={(42,'equal','panel_01'):{'t':profile(x)},
                  (42,'equal','panel_02'):{'t':profile(x[::-1])},
                  (42,'ranked','panel_01'):{'t':profile(x*.8+.2)}}
        cross,sensitivity,diagnostics=evaluation_tables(profiles,['t'])
        self.assertEqual(set(cross.arm),{'equal'})
        self.assertEqual(set(sensitivity.contrast),{'ranked_vs_equal'})
        effects,cohorts=cross_policy_effects(cross,['t'],20,42)
        self.assertTrue(effects.empty)
        self.assertTrue(cohorts.empty)
        self.assertEqual(len(diagnostics),6)
        row=sensitivity[sensitivity.domain=='interior_20'].iloc[0]
        self.assertAlmostEqual(row.RMSE,np.sqrt(np.mean((x[20:-20]-(x*.8+.2)[20:-20])**2)))

    def test_alignment_and_degenerate_metrics(self):
        x=profile(np.ones(60))
        result=aligned_metrics(x,x,'full_cds')
        self.assertTrue(np.isnan(result['PCC']))
        self.assertEqual(result['RMSE'],0)
        self.assertEqual(result['reason'],'nearly_constant_profile')
        with self.assertRaisesRegex(ValueError,'codon sequences'):
            aligned_metrics(x,profile(np.ones(60),'different_cds'),'full_cds')
        with self.assertRaisesRegex(ValueError,'positions differ'):
            aligned_metrics(x,profile(np.ones(59)),'full_cds')

    def test_difference_of_medians_shared_resampling_and_common_cohort(self):
        a=np.array([.0,.1,.2,.8,.9]);b=np.array([.05,.11,.15,.81,.9])
        ids=[f't{i}' for i in range(5)]
        rows=[]
        for i,tid in enumerate(ids):
            for pair in ('panel_01__panel_02','panel_01__panel_03'):
                for arm,x in [('equal',a),('ranked',b)]:
                    rows.append(dict(training_seed=42,arm=arm,domain='full_cds',pair=pair,transcript_id=tid,
                                     PCC=x[i],Spearman=x[i],RMSE=1-x[i]))
        frame=pd.DataFrame(rows)
        result,cohorts=cross_policy_effects(frame,ids,100,42)
        repeated,_=cross_policy_effects(frame,ids,100,42)
        pd.testing.assert_frame_equal(result,repeated)
        pcc=result[result.metric=='PCC']
        np.testing.assert_allclose(pcc.estimate,np.median(b)-np.median(a))
        self.assertNotAlmostEqual(pcc.iloc[0].estimate,np.median(b-a))
        np.testing.assert_allclose(pcc.ci_low,pcc.iloc[0].ci_low)
        np.testing.assert_allclose(pcc.ci_high,pcc.iloc[0].ci_high)
        frame.loc[(frame.transcript_id=='t0')&(frame.pair=='panel_01__panel_03'),'PCC']=np.nan
        result,cohorts=cross_policy_effects(frame,ids,100,42)
        self.assertTrue((result.loc[result.metric=='PCC','n_transcripts']==4).all())
        self.assertEqual(result.loc[result.metric=='PCC','cohort_hash'].nunique(),1)

    def test_future_three_arm_figures_and_collapse_do_not_crash(self):
        x=np.linspace(.2,1.8,60)
        profiles={(42,arm,panel):{'t1':profile(x*(1-.1*j)+.1*j),'t2':profile(x[::-1])}
                  for j,arm in enumerate(('equal','ranked','reverse')) for panel in ('panel_01','panel_02')}
        cross,sensitivity,_=evaluation_tables(profiles,['t1','t2'])
        effects,_=cross_policy_effects(cross,['t1','t2'],10,42)
        with tempfile.TemporaryDirectory() as temp, plt.rc_context({'text.usetex':False}):
            out=Path(temp)
            plot_sensitivity(sensitivity,out,40)
            plot_cross_panel(cross,effects,out,40)
            self.assertEqual(len(list(out.rglob('policy_sensitivity*.pdf'))),6)
            self.assertTrue((out/'paired_reproducibility_effects_seed42.pdf').is_file())
            sensitivity[['variance_a','variance_b']]=0.
            sensitivity['PCC']=np.nan
            plot_sensitivity(sensitivity,out,40)

    def test_reference_concentration_and_source_shares(self):
        rows=[]
        for arm,pi in [('equal',[1/3]*3),('ranked',[.6,.3,.1]),('reverse',[.1,.3,.6])]:
            for i,value in enumerate(pi):
                rows.append(dict(arm=arm,panel_id='panel_01',dataset_id=f'd{i}',global_rank=i+1,
                                 quality_group=i+1,source_family='family_a' if i<2 else 'family_b',pi=value))
        mass,sources,c=reference_tables(pd.DataFrame(rows))
        c=c.set_index('arm')
        self.assertAlmostEqual(c.loc['equal','N_ref'],3)
        self.assertAlmostEqual(c.loc['ranked','N_ref'],c.loc['reverse','N_ref'])
        self.assertNotAlmostEqual(c.loc['ranked','max_source_mass'],c.loc['reverse','max_source_mass'])
        np.testing.assert_allclose(mass.groupby('arm').reference_mass.sum(),1)
        np.testing.assert_allclose(sources.groupby('arm').source_mass.sum(),1)

    def test_example_rule_and_frozen_reuse(self):
        frame=pd.DataFrame([dict(training_seed=42,panel_id='panel_01',contrast='ranked_vs_equal',
                                domain='full_cds',transcript_id=t,PCC=p) for t,p in [('z',.9),('a',.9),('b',.8)]])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'selection.csv'
            selected=example_selections(frame,path)
            self.assertEqual(selected.loc[selected['quantile']==.5,'transcript_id'].item(),'a')
            selected.to_csv(path,index=False)
            changed=frame.copy();changed['PCC']=[0,.1,1]
            pd.testing.assert_frame_equal(selected,example_selections(changed,path))


if __name__=='__main__':
    unittest.main()
