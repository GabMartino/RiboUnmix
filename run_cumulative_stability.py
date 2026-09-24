#!/usr/bin/env python3
"""Cumulative profile stability from dataset YAML + ranking TSV, with no prior run.

The first invocation creates shared splits, train-only reliability references and
35 configs. An array task then trains one config through the production entrypoint.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

from Utils.real_exp8_stability import build_exp8_transcript_split
from Utils.real_panel_convergence import fit_panel_reliability_manifest, infer_source_identifier

ROOT=Path(__file__).resolve().parent
DEFAULT_OUTPUT=ROOT/'results/cumulative_stability_seed42'
DESIGN='cumulative_stability_from_data_v1'
FIXED_DESIGN='cumulative_stability_fixed_complete_cohort_v2'
FIXED_DEFAULT_OUTPUT=ROOT/'results/cumulative_stability_fixed_cohort_seed42'
SIZES=(2,5,10,20,40,80,114)
POLICIES=(('equal',0,False),('ranked_p1',1,False),('reverse_p1',1,True),
          ('ranked_p3',3,False),('reverse_p3',3,True))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha256(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def make_tasks(mapping,ranking,sizes,seed):
    ranks=ranking.set_index('dataset').quality_rank
    if ranks.index.has_duplicates or not np.isfinite(ranks).all() or (ranks<=0).any():
        raise ValueError('Ranking must contain unique dataset names and finite positive ranks.')
    missing=set(mapping)-set(ranks.index)
    if missing:
        raise ValueError(f'Datasets missing from the ranking: {sorted(missing)}')
    names=sorted(mapping,key=lambda d:(float(ranks[d]),d))
    if list(sizes)!=sorted(set(sizes)) or min(sizes)<2 or max(sizes)>len(names):
        raise ValueError(f'Use increasing sizes from 2 to the configured dataset count ({len(names)}).')
    tasks=[]
    for n in sizes:
        for arm,power,reverse in POLICIES:
            tasks.append(dict(array_index=len(tasks),N=n,arm=arm,power=power,reverse=reverse,
                training_seed=seed,datasets=names[:n],source_panel=f'exp8_qrank_N{n:03d}_seed{seed}',
                run_id=f'cumulative_N{n:03d}_{arm}_seed{seed}',directory=f'runs/seed{seed}/{arm}/N{n:03d}'))
    return tasks,ranks


def make_config(base,mapping,task,root,raw_weights):
    cfg=copy.deepcopy(base);cfg.pop('defaults',None)
    cfg['name']=task['run_id']
    cfg['dataset_config']={'dataset_path':{d:mapping[d] for d in task['datasets']}}
    cfg['experiment'].update(dataset=task['datasets'],seed=task['training_seed'],train=True,predict=True,
        from_checkpoint=False,resume_training_state=False,resume_checkpoint_path=None,allow_weights_only_resume=False)
    cfg['prediction'].update(checkpoint_variants=['best_val_loss'],sequence_only_shared_profile=True)
    cfg['split'].update(master_dataset_universe=list(mapping),external_manifest=str(root/'inputs/split.json'),external_panel_name=task['source_panel'])
    reference_key=task.get('panel_id',f'N{task["N"]:03d}')
    cfg['data'].update(num_workers=0,predict_num_workers=0,reliability_reference_manifest=str(root/f'inputs/reliability_{reference_key}.json'))
    cfg['data']['dataset_quality_ranking'].update(path=str(root/'inputs/ranking.tsv'),dataset_column='dataset',rank_column='quality_rank',strict=True)
    cfg['model'].update(mass_conservation=False,alpha_mode='learned',mean_correction='learned')
    cfg['model']['dataset_bias_params'].update(context_gru_precision='float32',context_gru_tbptt_window=0)
    cfg['model']['gamma_centering']['mode']='fixed_reference'
    cfg['model']['gamma_centering']['reference'].update(dataset_names=None,weighting='explicit',quality_rank_power=1.,explicit_weights=raw_weights)
    cfg['trainer'].update(accelerator='gpu',devices=[0],num_nodes=1,precision='bf16-mixed',use_distributed_sampler=False)
    cfg['callbacks']['save_best_pcc_checkpoint']=False
    cfg['metrics'].update(log_example_plot=False,log_validation_transcript_mu_pcc_distribution=False)
    directory=root/task['directory']
    cfg['paths'].update(checkpoints=str(directory/'checkpoints'),logs=str(directory/'logs'),results=str(directory/'predictions'))
    cfg['orchestrator']=dict(experiment=task.get('experiment_design',DESIGN),N=task['N'],arm=task['arm'],training_seed=task['training_seed'])
    if 'panel_id' in task:
        cfg['orchestrator']['panel_id']=task['panel_id']
    cfg['orchestrator']['task_contract_sha256']=object_sha256(cfg)
    return cfg


def prepare(args):
    root=args.output_root
    fixed_cohort=bool(getattr(args,'fixed_cohort',False))
    design=FIXED_DESIGN if fixed_cohort else DESIGN
    root.mkdir(parents=True,exist_ok=True)
    # Array workers share one setup; the first creates it while the others wait.
    with (root/'.setup.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        request=dict(seed=args.seed,sizes=args.sizes,config_sha256=sha256(args.config),
                     dataset_yaml_sha256=sha256(args.datasets),ranking_sha256=sha256(args.ranking))
        if fixed_cohort:
            request['split_protocol']='fixed_complete_transcripts'
        plan_path=root/'experiment_manifest.json'
        if plan_path.exists():
            plan=json.loads(plan_path.read_text())
            if plan.get('output_root')!=str(root):
                raise ValueError('This setup was generated at another location. Use a fresh --output-root so paths are generated on this machine.')
            if plan.get('setup_request')!=request or plan.get('experiment_design')!=design:
                raise ValueError('This output directory has a different setup; choose a new --output-root.')
            return plan
        base=yaml.safe_load(args.config.read_text())
        mapping=yaml.safe_load(args.datasets.read_text())['dataset_path']
        mapping={name:str((ROOT/Path(path)).resolve()) for name,path in mapping.items()}
        base['paths']['sequences_path']=str((ROOT/Path(base['paths']['sequences_path'])).resolve())
        base['paths']['encodings']={name:str((ROOT/Path(path)).resolve()) for name,path in base['paths']['encodings'].items()}
        tasks,ranks=make_tasks(mapping,pd.read_csv(args.ranking,sep='\t'),args.sizes,args.seed)
        if fixed_cohort:
            for task in tasks:
                task['experiment_design']=design
        inputs=root/'inputs';inputs.mkdir(exist_ok=True)
        configs=root/'configs';configs.mkdir(exist_ok=True)
        shutil.copy2(args.ranking,inputs/'ranking.tsv')
        # One split per N, never a separate split per policy.
        prefixes=[dict(N=t['N'],datasets=t['datasets'],run_id=t['source_panel']) for t in tasks if t['arm']=='equal']
        if fixed_cohort:
            from Utils.cumulative_fixed_cohort import build_fixed_cumulative_split
            builder=build_fixed_cumulative_split
            print('Building ONE complete train/validation/test cohort shared by every dataset and N...',flush=True)
        else:
            builder=build_exp8_transcript_split
            print('Legacy design: validation and training membership change across N; use run_cumulative_fixed_cohort.py for controlled new runs.',flush=True)
        split=builder(experiment_name=design,tasks=prefixes,dataset_mapping=mapping,
            sequences_path=base['paths']['sequences_path'],subset_seed=args.seed,validation_fraction=.1,
            test_fraction=.1,reliability_bins=10,maximum_cds_codons=base['data'].get('max_cds_codons'))
        write_json(inputs/'split.json',split)
        folds={}
        for prefix in prefixes:
            n,panel=prefix['N'],prefix['run_id']
            folds[str(n)]=dict(source_panel=panel,train_ids=split['panel_train_eligible_ids'][panel],
                validation_ids=split['panel_validation_ids'][panel],test_ids=split['common_test_ids'])
            print(f'N={n}: fitting reliability references on training transcripts only...',flush=True)
            reliability=fit_panel_reliability_manifest(experiment_name=design,panel_name=panel,
                panel_datasets=prefix['datasets'],dataset_mapping=mapping,panel_training_ids=folds[str(n)]['train_ids'],
                validation_ids=folds[str(n)]['validation_ids'],test_ids=split['common_test_ids'],source_split_manifest=inputs/'split.json')
            write_json(inputs/f'reliability_N{n:03d}.json',reliability)
        weights=[];R=float(ranks.max())
        for task in tasks:
            names=task['datasets'];q=(R-ranks.loc[names].to_numpy(float)+1)/R
            raw=q**task['power']
            if task['reverse']: raw=raw[::-1]
            pi=raw/raw.sum()
            weights.extend(dict(N=task['N'],arm=task['arm'],power=task['power'],dataset_id=d,
                global_rank=float(ranks[d]),original_q=float(original),assigned_q=float(value),pi=float(weight),
                source_family=infer_source_identifier(d)) for d,original,value,weight in zip(names,q,raw,pi))
            cfg=make_config(base,mapping,task,root,dict(zip(names,map(float,raw))))
            path=configs/f'{task["run_id"]}.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
            task.update(config_path=str(path),config_sha256=sha256(path),task_contract_sha256=cfg['orchestrator']['task_contract_sha256'])
            (root/task['directory']).mkdir(parents=True,exist_ok=True)
            task['command']=[sys.executable,'-u',str(ROOT/'main_ribounmix_multidataset.py'),
                f'--config-path={configs}',f'--config-name={path.stem}',f'hydra.run.dir={root/task["directory"]}/hydra','hydra.job.chdir=false']
        weights=pd.DataFrame(weights);weights.to_csv(root/'reference_weights.csv',index=False)
        concentration=pd.DataFrame([dict(N=n,arm=arm,N_ref=1/np.square(g.pi).sum(),
            weighted_mean_rank=np.dot(g.pi,g.global_rank)) for (n,arm),g in weights.groupby(['N','arm'],sort=False)])
        concentration.to_csv(root/'reference_concentration.csv',index=False)
        weights.groupby(['N','arm','source_family'],as_index=False).pi.sum().to_csv(root/'source_family_reference_mass.csv',index=False)
        pd.DataFrame(tasks).drop(columns=['command','datasets']).to_csv(root/'task_matrix.csv',index=False)
        plan=dict(experiment_design=design,output_root=str(root),setup_request=request,training_seeds=[args.seed],
            sizes=args.sizes,tasks=tasks,tasks_sha256=object_sha256(tasks),source_folds=folds,
            frozen_file_sha256={str(p):sha256(p) for p in [*inputs.iterdir(),*configs.iterdir(),root/'reference_weights.csv']})
        write_json(plan_path,plan)
        print(f'Ready: {len(tasks)} tasks. {plan_path}',flush=True)
        return plan


def _execution_runtime_overrides(args):
    """Return execution-only Hydra overrides requested by a launcher.

    These limits change how a logical transcript batch is partitioned across
    forward passes.  They deliberately remain outside the frozen scientific
    task contract, which defines the datasets, split, objective and optimizer
    batch.
    """
    fields = (
        ('max_pair_rows_per_forward',
         'training.execution_microbatching.max_pair_rows_per_forward'),
        ('max_padded_codon_tokens_per_forward',
         'training.execution_microbatching.max_padded_codon_tokens_per_forward'),
    )
    overrides = {}
    for attribute, key in fields:
        value = getattr(args, attribute, None)
        if value is None:
            continue
        value = int(value)
        if value <= 0:
            raise ValueError(f'{attribute} must be a positive integer.')
        overrides[key] = value
    return overrides


def run_task(args,plan):
    task=plan['tasks'][args.task_index];directory=args.output_root/task['directory']
    command=list(task['command']);command[0]=sys.executable
    runtime_overrides=_execution_runtime_overrides(args)
    command.extend(f'{key}={value}' for key,value in runtime_overrides.items())
    if args.dry_run:
        print(' '.join(command));return 0
    if sha256(task['config_path'])!=task['config_sha256']:
        raise ValueError('Task configuration was edited after setup.')
    from run_real_exp8_reference_directionality import validate_outputs
    with (directory/'.training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_path=directory/'execution_status.json'
        previous=json.loads(state_path.read_text()) if state_path.exists() else {}
        ids=plan['source_folds'][task.get('panel_id',str(task['N']))]['test_ids']
        if previous.get('status')=='completed':
            validate_outputs(directory,task,ids)
            print(f'Already completed: {task["run_id"]}');return 0
        checkpoint=None
        checkpoint_files=list((directory/'checkpoints').rglob('*.ckpt'))
        if getattr(args,'require_resume_checkpoint',False) and not checkpoint_files:
            raise FileNotFoundError(
                f'Resume was required, but {directory} contains no checkpoint.')
        if checkpoint_files:
            from resume_real_experiment_from_checkpoints import _select_resume_checkpoint
            path,checkpoint,_=_select_resume_checkpoint(
                directory,
                expected_task_contract_sha256=task['task_contract_sha256'],
            )
            if path is None or not checkpoint['has_optimizer_state'] or not checkpoint['has_scheduler_state']:
                raise ValueError(
                    'An interrupted run has no matching usable full-state checkpoint.')
            command+=['experiment.from_checkpoint=true','experiment.resume_training_state=true',f'experiment.resume_checkpoint_path={path}']
        attempt=dict(started_unix=time.time(),resume_checkpoint=checkpoint,
                     runtime_overrides=runtime_overrides)
        state=dict(task_id=task['run_id'],config_sha256=task['config_sha256'],status='running',attempts=[*previous.get('attempts',[]),attempt])
        write_json(state_path,state)
        try:
            result=subprocess.run(command,cwd=ROOT,check=False)
            attempt.update(exit_code=result.returncode,finished_unix=time.time())
            if result.returncode:
                state['status']='failed'
            else:
                state.update(status='completed',outputs=validate_outputs(directory,task,ids))
        except Exception as error:
            state.update(status='failed',reason=str(error));write_json(state_path,state);raise
        write_json(state_path,state)
        return result.returncode


def main(argv=None, *, fixed_cohort=False):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/config_ribounmix_multidataset.yaml')
    parser.add_argument('--datasets',type=Path,default=ROOT/'config/dataset_config/weighted_hek_riboseq_codon_replicas.yaml')
    parser.add_argument('--ranking',type=Path,default=ROOT/'Datasets/data/HEK_riboseq_profile_quality_rank_components.tsv')
    parser.add_argument('--output-root',type=Path,default=FIXED_DEFAULT_OUTPUT if fixed_cohort else DEFAULT_OUTPUT)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--sizes',type=lambda text:[int(n) for n in text.split(',')],default=list(SIZES))
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare-only',action='store_true')
    action.add_argument('--task-index',type=int)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    args.fixed_cohort=fixed_cohort
    for key in ('config','datasets','ranking','output_root'): setattr(args,key,getattr(args,key).expanduser().resolve())
    if args.task_index is not None and not 0<=args.task_index<len(args.sizes)*len(POLICIES):
        parser.error(f'--task-index must be in 0..{len(args.sizes)*len(POLICIES)-1}')
    plan=prepare(args)
    return 0 if args.prepare_only else run_task(args,plan)


if __name__=='__main__':
    raise SystemExit(main())
