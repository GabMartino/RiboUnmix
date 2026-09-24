from __future__ import annotations

import copy
import fcntl
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import yaml

from Utils.campaign_training import assert_campaign_pair, selected_tasks, run_task, verify_campaign_execution
from Utils.panel_reference_preparation import validation_support_constraint
from Utils.real_panel_convergence import refine_source_atomic_panel_assignment, write_json
from Utils.reference_campaign import object_hash, task_matrix
from Utils.panel_reference_audit import sha256


class SupportConstrainedSearchTests(unittest.TestCase):
    def test_rejects_infeasible_swaps_before_scoring(self):
        table=pd.DataFrame([dict(dataset_id=f'd{i}',source_family=f'f{i}',panel_id=f'panel_{i//2+1:02}') for i in range(8)])
        supports={f'd{i}':({'v'} if i%2==0 else set()) for i in range(8)}
        constraint=validation_support_constraint(table,supports,dict(common_validation_ids=['v'],minimum_usable_datasets_per_panel=1))
        original=table.rename(columns={'dataset_id':'dataset_name','panel_id':'panel','source_family':'source_identifier'})
        def objective(frame):
            self.assertTrue(constraint(frame))
            return float(sum(int(p[-2:])*(i+1) for i,p in enumerate(frame.panel)))
        result,record=refine_source_atomic_panel_assignment(original,objective=objective,seed=42,proposal_budget=100,feasible=constraint)
        self.assertTrue(constraint(result))
        self.assertGreater(record['support_rejected_proposals'],0)
        self.assertEqual(result.groupby('panel').size().tolist(),[2,2,2,2])
        again,_=refine_source_atomic_panel_assignment(original,objective=objective,seed=42,proposal_budget=100,feasible=constraint)
        pd.testing.assert_frame_equal(result,again)
        broken=original.copy();broken.loc[[0,3],'panel']=broken.loc[[3,0],'panel'].to_numpy()
        with self.assertRaisesRegex(ValueError,'Initial'):
            refine_source_atomic_panel_assignment(broken,objective=objective,seed=42,proposal_budget=100,feasible=constraint)


class CampaignExecutionTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.cfg=dict(trainer=dict(precision='bf16-mixed'),orchestrator=dict(task_contract_sha256='contract'))
        config=self.root/'input.yaml';config.write_text(yaml.safe_dump(self.cfg))
        self.task=dict(task_id='B_equal_seed42_panel01',root=str(self.root),config=str(config),
            command=[sys.executable,'-u',str(self.root/'production.py'),'--config-name=frozen'])

    def test_task_cap_is_not_repeated_by_array_index(self):
        matrix=task_matrix('main')
        plan={'execution_tasks':matrix.to_dict('records')}
        self.assertEqual(len(selected_tasks(plan,8)),8)
        for i in range(8):
            self.assertEqual(selected_tasks(plan,8,i)[0]['task_id'],matrix.iloc[i].task_id)
        for i in (-1,8,47):
            with self.assertRaisesRegex(ValueError,'cap'):selected_tasks(plan,8,i)

    def test_loss_and_local_reliability_cannot_change_between_arms(self):
        base={'loss':{'sample_reduction':'transcript_balanced'},'data':{'reliability_reference_manifest':'same'},'name':'equal'}
        other=copy.deepcopy(base);other['name']='ranked'
        self.assertEqual(len(assert_campaign_pair(base,other)),1)
        for block,key in [('loss','sample_reduction'),('data','reliability_reference_manifest')]:
            other=copy.deepcopy(base);other[block][key]='changed'
            with self.assertRaisesRegex(ValueError,'Unapproved'):assert_campaign_pair(base,other)

    def fake_run(self, **kwargs):
        with patch('torch.cuda.is_available',return_value=True), patch('torch.cuda.device_count',return_value=1), \
             patch('torch.cuda.is_bf16_supported',return_value=True), patch('torch.cuda.get_device_name',return_value='test_gpu'), \
             patch('Utils.campaign_training.subprocess.run',return_value=subprocess.CompletedProcess([],0)) as launch, \
             patch('Utils.campaign_training.validate_predictions',return_value={'verified':'native_L'}):
            result=run_task(self.task,'approved_plan',**kwargs)
        return result,launch

    def test_one_production_process_and_explicit_repeat_protection(self):
        state,launch=self.fake_run()
        self.assertEqual(state['status'],'completed_native_L_validated')
        self.assertEqual(launch.call_count,1)
        self.assertEqual(launch.call_args.args[0],self.task['command'])
        with self.assertRaisesRegex(ValueError,'--resume'):self.fake_run()
        _,launch=self.fake_run(resume=True)
        self.assertEqual(launch.call_count,0)

    def test_lock_prevents_duplicate_execution(self):
        with (self.root/'execution.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError,'already running'):self.fake_run()

    def test_resume_rejects_foreign_checkpoint(self):
        write_json(self.root/'execution_status.json',dict(plan_hash='approved_plan',status='failed',attempts=[{}]))
        checkpoint=self.root/'last.ckpt';checkpoint.write_bytes(b'fixture')
        metadata=dict(has_optimizer_state=True,has_scheduler_state=True)
        with patch('resume_real_experiment_from_checkpoints._select_resume_checkpoint',return_value=(checkpoint,metadata,[])), \
             patch('torch.load',return_value={'hyper_parameters':{'campaign_task_contract_sha256':'foreign'}}):
            with self.assertRaisesRegex(ValueError,'exact campaign'):self.fake_run(resume=True)

    def test_full_state_resume_keeps_scientific_config(self):
        write_json(self.root/'execution_status.json',dict(plan_hash='approved_plan',status='failed',attempts=[{}]))
        checkpoint=self.root/'last.ckpt';checkpoint.write_bytes(b'fixture')
        metadata=dict(has_optimizer_state=True,has_scheduler_state=True)
        with patch('resume_real_experiment_from_checkpoints._select_resume_checkpoint',return_value=(checkpoint,metadata,[])), \
             patch('torch.load',return_value={'hyper_parameters':{'campaign_task_contract_sha256':'contract'}}):
            state,launch=self.fake_run(resume=True)
        args=launch.call_args.args[0]
        self.assertIn('experiment.resume_training_state=true',args)
        self.assertIn('experiment.allow_weights_only_resume=false',args)
        self.assertEqual(len(state['attempts']),2)

    def test_frozen_input_tampering_is_rejected(self):
        packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
        frozen=self.root/'frozen_execution.json'
        write_json(frozen,dict(software={'python':platform.python_version()},environment_packages=packages,
                             file_sha256={self.task['config']:sha256(Path(self.task['config']))}))
        plan={'execution_tasks':[self.task],'frozen_execution_sha256':sha256(frozen)}
        write_json(self.root/'plan_definition.json',plan)
        manifest={'plan_hash':object_hash(plan)}
        verify_campaign_execution(self.root,manifest)
        Path(self.task['config']).write_text('changed')
        with self.assertRaisesRegex(ValueError,'input changed'):verify_campaign_execution(self.root,manifest)


if __name__=='__main__':unittest.main()
