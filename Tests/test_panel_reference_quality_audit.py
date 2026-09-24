from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from analyses.audit_four_panel_reference_quality import parse_args
from Utils.panel_reference_audit import (add_weights, canonical_ids, check_saved_weights,
    global_quality_groups, load_inputs, run_audit, sha256, summarize)
from Utils.panel_reference_preparation import (ROOT, balance_objective, DEFAULT_SEARCH,
    finish_preparation, freeze_heldout_split, paired_config_differences, prepare_design,
    verify_prepared_inputs)
from Utils.real_panel_convergence import fit_panel_reliability_manifest, refine_source_atomic_panel_assignment, assert_panel_partition
from Utils.reliability_references import transcript_id_hash


def fixture(root):
    historical=root/'historical'; historical.mkdir()
    names=[f'source{i}_2020' for i in range(8)]
    panels={f'panel_{i+1:02d}':names[2*i:2*i+2] for i in range(4)}
    ids=[f't{i:03}' for i in range(40)]
    seq=root/'sequences.parquet'
    pd.DataFrame(dict(transcript_id=ids,codons=[['ATG','AAA','GGG','TAA']]*40,css=[[]]*40)).to_parquet(seq)
    mapping={}
    for i,name in enumerate(names):
        path=root/f'{name}.parquet'
        pd.DataFrame(dict(id=ids,ribo=[[1.,2.,3.,0.]]*40,ribo_cds_replicas=[[[1.,2.,3.,0.],[1.,2.,3.,0.]]]*40,
            read_density=[1.5]*40,coverage=[.75]*40,weight=[1.]*40)).to_parquet(path)
        mapping[name]=str(path)
    split=dict(manifest_version=1,panels=panels,common_validation_ids=ids[:4],common_test_ids=ids[4:8],
        panel_train_eligible_ids={p:ids[8:] for p in panels},minimum_usable_datasets_per_panel=2,
        maximum_cds_codons=None,random_seed=42,fold_id_hashes={
            'validation':transcript_id_hash(ids[:4]),'test':transcript_id_hash(ids[4:8])})
    split_path=historical/'common_split_manifest.json';split_path.write_text(json.dumps(split))
    quality=pd.DataFrame(dict(dataset_name=names,source_identifier=names,eligible=True,
        log1p_median_read_density=np.arange(8)/5.,median_positive_codon_coverage=np.arange(8)/10.,
        number_of_eligible_transcripts=40,median_replica_PCC=[np.nan,*np.arange(7)/8.]))
    quality.to_csv(historical/'dataset_quality_table.csv',index=False)
    manifest=dict(panels=panels,retained_datasets=names,panel_source_families={p:ds for p,ds in panels.items()},
        random_seed=42,git={'commit':None},dataset_quality_summaries=quality.to_dict('records'))
    (historical/'panel_manifest.json').write_text(json.dumps(manifest))
    config=yaml.safe_load((ROOT/'config/config_ribounmix_multidataset.yaml').read_text())
    config['dataset_config']={'dataset_path':mapping}
    config['model']['mass_conservation']=False
    config['model']['gamma_centering']['mode']='fixed_reference'
    config['model']['gamma_centering']['reference'].update(weighting='equal',dataset_names=None,quality_rank_power=0.)
    config['data']['train_sampling_strategy']='transcript_grouped_multidataset_pairs'
    config['loss']['sample_reduction']='transcript_balanced'
    config['loss']['experiment_mode']='standard_nb'
    config['paths']['sequences_path']=str(seq)
    for panel,ds in panels.items():
        task=historical/panel;task.mkdir()
        c=copy.deepcopy(config);c['experiment'].update(dataset=ds,seed=42)
        (task/'resolved_config.yaml').write_text(yaml.safe_dump(c))
        reliability=fit_panel_reliability_manifest(experiment_name='fixture',panel_name=panel,panel_datasets=ds,
            dataset_mapping=mapping,panel_training_ids=ids[8:],validation_ids=ids[:4],test_ids=ids[4:8],source_split_manifest=split_path)
        (task/'reliability_reference_manifest.json').write_text(json.dumps(reliability))
    ranking=root/'ranks.tsv'
    ranks=pd.DataFrame(dict(dataset=[*names,'outside_universe'],quality_rank=np.arange(1,10),rank_component_count=6))
    for c in ['periodicity','cds_enrichment','depth','transcript_support','rpf_length_center','rpf_length_spread']:
        ranks[f'rank_{c}']=np.arange(1,10)
    ranks.to_csv(ranking,sep='\t',index=False)
    return historical,ranking,split


class PanelReferenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.historical,self.ranking,self.split=fixture(self.root)

    def args(self,mode='audit',suffix='audit'):
        return parse_args(['--mode',mode,'--panel-manifest',str(self.historical/'panel_manifest.json'),
            '--ranking-table',str(self.ranking),'--output-root',str(self.root/suffix),'--no-tex'])

    def test_global_conversion_and_one_fixed_group_definition(self):
        args=self.args();table,*_=load_inputs(args,{},[])
        self.assertEqual(table.loc[table.dataset_id=='source7_2020','raw_reference_score_q'].item(),2/9)
        table=add_weights(table);labels,definition=global_quality_groups(table.global_rank)
        table['global_quality_group']=labels
        self.assertEqual(definition['boundaries'],[2.75,4.5,6.25])
        for policy in ('equal','ranked'):
            np.testing.assert_allclose(table.groupby('panel_id')[f'pi_{policy}'].sum(),1)
            self.assertTrue(table[f'pi_{policy}'].gt(0).all())
        results=summarize(table,['rank_depth','median_replica_PCC'])
        np.testing.assert_allclose(results['panel_concentration_summary'].query('policy=="equal"').N_ref,2)
        np.testing.assert_allclose(results['source_family_reference_mass'].groupby(['panel_id','policy']).source_mass.sum(),1)
        np.testing.assert_allclose(results['panel_quality_group_reference_mass'].groupby(['panel_id','policy']).reference_mass.sum(),1)
        self.assertEqual(results['panel_component_balance'].query('panel_id=="panel_01" and feature=="median_replica_PCC"').n_missing.tolist(),[1,1])

    def test_tied_ranks_stay_together_and_grouping_is_deterministic(self):
        values=np.array([1,2,2,2,2,4,5,5])
        labels,meta=global_quality_groups(values)
        self.assertEqual(len(set(labels[values==2])),1)
        np.testing.assert_array_equal(labels,global_quality_groups(values)[0])
        self.assertTrue(meta['unequal_group_sizes'])

    def test_missing_rank_blocks_preparation_without_imputation(self):
        ranks=pd.read_csv(self.ranking,sep='\t');ranks=ranks.iloc[1:];ranks.to_csv(self.ranking,sep='\t',index=False)
        result=run_audit(self.args())
        self.assertTrue(result['manifest']['hard_errors'])
        self.assertIn('source0_2020',' '.join(result['manifest']['hard_errors']))
        self.assertTrue(result['table'].rank_missing.any())

    def test_invalid_rank_reports_the_dataset_id(self):
        ranks=pd.read_csv(self.ranking,sep='\t');ranks.loc[2,'quality_rank']=np.nan
        ranks.to_csv(self.ranking,sep='\t',index=False)
        with self.assertRaisesRegex(ValueError,'source2_2020'):
            load_inputs(self.args(),{},[])

    def test_rank_only_summary_does_not_invent_components(self):
        table,*_=load_inputs(self.args(),{},[]);table=add_weights(table)
        table['global_quality_group'],_=global_quality_groups(table.global_rank)
        result=summarize(table,[])
        self.assertTrue(result['panel_component_balance'].empty)
        self.assertIn('feature',result['panel_component_balance'])
        self.assertEqual(len(result['panel_rank_summary']),4)

    def test_ambiguous_alias_and_duplicate_weights(self):
        with self.assertRaisesRegex(ValueError,'ambiguous'):
            canonical_ids(['a','b'],{'a':'same','b':'same'},'ranking')
        table,*_=load_inputs(self.args(),{},[]);table=add_weights(table)
        weights=table[['panel_id','dataset_id','pi_ranked']].rename(columns={'pi_ranked':'pi'}).assign(policy='ranked')
        duplicate=pd.concat([weights,weights.iloc[:1]],ignore_index=True)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            check_saved_weights(duplicate,table,{})
        checks=check_saved_weights(duplicate,table,{},'collapse-identical')
        self.assertTrue(checks.status.eq('match').all())
        duplicate.loc[len(duplicate)-1,'pi']=.99
        with self.assertRaisesRegex(ValueError,'conflicting=True'):
            check_saved_weights(duplicate,table,{},'collapse-identical')
        weights.loc[0,'pi']+=.001
        self.assertEqual(check_saved_weights(weights,table,{}).status.eq('mismatch').sum(),1)

    def test_cpu_audit_reads_no_arrays_or_checkpoints(self):
        with patch('pandas.read_parquet',side_effect=AssertionError('Audit must not load arrays')), \
             patch('torch.load',side_effect=AssertionError('Audit must not load checkpoints')), \
             patch('Utils.panel_reference_audit.plot_audit'):
            result=run_audit(self.args())
        self.assertFalse(result['manifest']['hard_errors'])
        self.assertTrue(result['manifest']['cpu_only'])

    def test_restricted_gamma_reference_is_a_hard_error(self):
        path=self.historical/'panel_01/resolved_config.yaml'
        cfg=yaml.safe_load(path.read_text())
        cfg['model']['gamma_centering']['reference']['dataset_names']=['source0_2020']
        path.write_text(yaml.safe_dump(cfg))
        result=run_audit(self.args())
        self.assertTrue(any('reference universe' in e for e in result['manifest']['hard_errors']))

    def test_source_atomicity_and_exact_capacities(self):
        table,*_=load_inputs(self.args(),{},[]);table=add_weights(table)
        table['global_quality_group'],_=global_quality_groups(table.global_rank)
        # An intact two-dataset family cannot be split by a refinement.
        table.loc[:1,'source_family']='family'
        assignment=table.rename(columns={'dataset_id':'dataset_name','panel_id':'panel','source_family':'source_identifier'})
        objective=balance_objective(table,DEFAULT_SEARCH)
        a,record=refine_source_atomic_panel_assignment(assignment,objective=objective,seed=42,proposal_budget=30)
        b,_=refine_source_atomic_panel_assignment(assignment,objective=objective,seed=42,proposal_budget=30)
        pd.testing.assert_frame_equal(a,b)
        self.assertEqual(a.groupby('panel').size().tolist(),[2,2,2,2])
        self.assertEqual(a.loc[a.source_identifier=='family','panel'].nunique(),1)
        self.assertLessEqual(record['final_objective'],record['initial_objective'])
        broken=a.copy();broken.loc[0,'panel']='panel_04'
        with self.assertRaises(AssertionError): assert_panel_partition(broken,expected_datasets=assignment.dataset_name.tolist())

    def test_frozen_heldout_support_and_no_test_dropping(self):
        ids={f't{i:03}' for i in range(40)}
        supports={d:set(ids) for ds in self.split['panels'].values() for d in ds}
        for d in self.split['panels']['panel_01']: supports[d].discard('t004')
        new,report,ok=freeze_heldout_split(self.split,self.split['panels'],supports,ids)
        self.assertTrue(ok)
        self.assertEqual(new['common_test_ids'],self.split['common_test_ids'])
        self.assertFalse(set(sum(new['panel_train_eligible_ids'].values(),[]))&set(self.split['common_test_ids']))
        for d in self.split['panels']['panel_01']: supports[d].discard('t000')
        _,report,ok=freeze_heldout_split(self.split,self.split['panels'],supports,ids)
        self.assertFalse(ok)
        self.assertTrue(report.blocks_preparation.any())

    def test_both_preparation_modes_no_training_and_paired_contract(self):
        before={str(p):sha256(p) for p in self.historical.rglob('*') if p.is_file()}
        search=self.root/'search.json';search.write_text(json.dumps({'restarts':1,'proposal_budget_per_restart':10}))
        for index,mode in enumerate(('prepare-existing','prepare-rank-balanced')):
            args=self.args(mode,f'audit_{index}');args.design_config=search
            if mode=='prepare-rank-balanced':
                # Explicit nonstandard input columns must still work with the
                # production parser and existing matched-analysis contract.
                ranks=pd.read_csv(self.ranking,sep='\t').rename(columns={'dataset':'dataset_id','quality_rank':'rank_global'})
                ranks.to_csv(self.ranking,sep='\t',index=False)
                args.ranking_dataset_column='dataset_id';args.ranking_rank_column='rank_global'
            with patch('Utils.panel_reference_audit.plot_audit'), patch('Utils.panel_reference_preparation.plot_audit'):
                audit=run_audit(args)
                self.assertFalse(audit['manifest']['hard_errors'])
                prepare_design(args,audit)
            destination=next(p.parent for p in args.output_root.glob('*/rerun_plan.json'))
            plan=json.loads((destination/'rerun_plan.json').read_text())
            self.assertEqual(plan['status'],'prepared_not_launched')
            self.assertEqual(plan['number_of_fresh_trainings'],8)
            self.assertFalse(plan['training_launched'])
            self.assertEqual(len(plan['tasks']),8)
            self.assertEqual(pd.read_csv(destination/'frozen_global_ranking.tsv',sep='\t').quality_rank.tolist(),list(range(1,10)))
            a=yaml.safe_load((destination/'equal/panel_01/resolved_config.yaml').read_text())
            b=yaml.safe_load((destination/'ranked/panel_01/resolved_config.yaml').read_text())
            paired_config_differences(a,b)
            b['loss']['consensus_raw_pcc_weight']+=1
            with self.assertRaisesRegex(ValueError,'consensus_raw_pcc_weight'): paired_config_differences(a,b)
            verify_prepared_inputs(destination)
            blocked=subprocess.run(['bash',str(destination/'launch_commands.sh'),'0'],capture_output=True,text=True)
            self.assertEqual(blocked.returncode,2)
            self.assertIn('requires subsequent explicit authorization',blocked.stderr)
        self.assertEqual(before,{str(p):sha256(p) for p in self.historical.rglob('*') if p.is_file()})

    def failed_final_audit(self, suffix):
        args=self.args('prepare-rank-balanced',suffix)
        search=self.root/'recovery_search.json'
        search.write_text(json.dumps({'restarts':1,'proposal_budget_per_restart':10}))
        args.design_config=search
        with patch('Utils.panel_reference_audit.plot_audit'), \
             patch('Utils.panel_reference_preparation.plot_audit'), \
             patch('analyses.compare_real_panel_weighting.audit_design',
                   side_effect=ValueError('Current global quality ranking does not match the frozen experiment.')):
            audit=run_audit(args)
            with self.assertRaisesRegex(ValueError,'ranking does not match'):
                prepare_design(args,audit)
        return args.output_root/'global_rank_balanced_partition_v2'

    def test_resume_failed_final_audit_without_repartitioning_or_refitting(self):
        for legacy in (False,True):
            with self.subTest(legacy=legacy):
                destination=self.failed_final_audit(f'recovery_{legacy}')
                plan=json.loads((destination/'rerun_plan.json').read_text())
                self.assertEqual(plan['status'],'blocked_before_training')
                self.assertFalse((destination/'frozen_execution.json').exists())
                if legacy:
                    # Older preparations saved no finalization state or task list.
                    (destination/'preparation_state.json').unlink()
                    for key in ('tasks','training_seeds','audit_manifest','frozen_ranking_sha256'):
                        plan.pop(key,None)
                    (destination/'rerun_plan.json').write_text(json.dumps(plan))
                before={str(p):sha256(p) for p in destination.rglob('*')
                        if p.is_file() and p.suffix in ('.json','.yaml','.tsv','.csv','.py')
                        and p.name!='rerun_plan.json'}
                with patch('Utils.panel_reference_preparation.prepare_partition',side_effect=AssertionError('No new search')), \
                     patch('Utils.panel_reference_preparation.fit_panel_reliability_manifest',side_effect=AssertionError('No refitting')), \
                     patch('pandas.read_parquet',side_effect=AssertionError('No new observation loading')):
                    recovered=finish_preparation(destination)
                self.assertEqual(recovered['status'],'prepared_not_launched')
                self.assertFalse(recovered['training_launched'])
                self.assertEqual(len(recovered['tasks']),8)
                self.assertEqual(before,{path:sha256(path) for path in before})
                verify_prepared_inputs(destination)
                recovery=json.loads((destination/'preparation_recovery.json').read_text())
                self.assertEqual(recovery['legacy_recovery'],legacy)
                self.assertFalse(recovery['partition_search_repeated'])
                plan_hash=sha256(destination/'rerun_plan.json')
                frozen_hash=sha256(destination/'frozen_execution.json')
                finish_preparation(destination)
                self.assertEqual(plan_hash,sha256(destination/'rerun_plan.json'))
                self.assertEqual(frozen_hash,sha256(destination/'frozen_execution.json'))

    def test_recovery_does_not_accept_changed_ranking_or_incomplete_design(self):
        destination=self.failed_final_audit('recovery_reject')
        ranking=destination/'frozen_global_ranking.tsv'
        saved=ranking.read_bytes()
        ranking.write_bytes(saved+b'\n')
        with self.assertRaisesRegex(ValueError,'Input changed'):
            finish_preparation(destination)
        ranking.write_bytes(saved)
        (destination/'resolved_configs/panel_04_ranked.yaml').unlink()
        with self.assertRaises(FileNotFoundError):
            finish_preparation(destination)
        self.assertFalse((destination/'frozen_execution.json').exists())
        self.assertEqual(json.loads((destination/'rerun_plan.json').read_text())['status'],'blocked_before_training')


if __name__=='__main__': unittest.main()
