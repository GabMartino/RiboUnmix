"""CPU campaign planning and explicit prerequisite gates.

This module does not submit jobs. A failed validation-support audit is a design
decision, not an infrastructure failure to retry with another split or seed.
"""
from __future__ import annotations

import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import shlex
import shutil
import sys
import yaml

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from Utils.panel_reference_audit import ROOT, read_json, sha256, software_provenance
from Utils.panel_reference_preparation import snapshot_code
from Utils.real_panel_convergence import write_json


def object_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def derived_common_reference(log_L, log_gamma, valid_mask):
    """NEW descriptive common-reference shape; never a checkpoint mutation.

    The full-CDS result must be computed once, then sliced for interior metrics.
    log_Z remains representable even if Z itself exceeds float64 support.
    """
    log_L=np.asarray(log_L,dtype=float);g=np.asarray(log_gamma,dtype=float)
    mask=np.asarray(valid_mask,dtype=bool)
    if log_L.ndim!=1 or mask.shape!=log_L.shape or g.ndim!=2 or g.shape[1:]!=log_L.shape or not g.shape[0] or not mask.any():
        raise ValueError('Expected aligned log_L[T], g[D,T] and a nonempty valid mask.')
    if not np.isfinite(log_L[mask]).all() or not np.isfinite(g[:,mask]).all():
        raise ValueError('Nonfinite native factors; no repair or imputation.')
    z=log_L[mask]+g[:,mask].mean(axis=0)
    log_Z=float(logsumexp(z)-np.log(mask.sum()))
    common=np.full(log_L.shape,np.nan);common[mask]=np.exp(z-log_Z)
    return dict(values=common,log_Z=log_Z,Z=float(np.exp(log_Z)) if log_Z<=np.log(np.finfo(float).max) else None,
                label='derived common-reference shape',normalization_domain='full_valid_CDS')


def shared_extreme_mask(profiles,valid_mask,fraction):
    """Deterministic union over the complete declared comparison-model list."""
    x=np.asarray(profiles,float);valid=np.asarray(valid_mask,bool)
    if x.ndim!=2 or x.shape[1:]!=valid.shape or not 0<fraction<1 or not valid.any():
        raise ValueError('Invalid aligned profiles/mask or removal fraction.')
    if not np.isfinite(x[:,valid]).all(): raise ValueError('Nonfinite profile; no peak-selection imputation.')
    positions=np.flatnonzero(valid);number=int(np.ceil(fraction*len(positions)))
    removed=np.zeros_like(valid)
    for row in x:
        order=np.argsort(-row[positions],kind='stable')
        removed[positions[order[:number]]]=True
    return removed


def matched_random_masks(valid_mask,removed_count,seed,draws=100):
    valid=np.asarray(valid_mask,bool);positions=np.flatnonzero(valid)
    if not 0<=removed_count<=len(positions) or draws<1: raise ValueError('Invalid matched removal budget.')
    rng=np.random.default_rng(seed)
    for _ in range(draws):
        removed=np.zeros_like(valid);removed[rng.choice(positions,removed_count,replace=False)]=True
        yield removed


def task_matrix(campaign, training_seeds=(42,43,44), reference_permutations=3):
    seeds=list(training_seeds)
    if len(seeds)!=len(set(seeds)) or 42 not in seeds:
        raise ValueError('Distinct training seeds including seed 42 are required for the prespecified shuffled comparison.')
    if not 0<=reference_permutations<=3:
        raise ValueError('At most three reference permutations; a smaller number is an explicitly incomplete campaign.')
    rows=[]
    def append(partition,arm,seed,permutation=None):
        for panel in range(1,5):
            first=partition=='B' and arm in ('equal','ranked') and seed==42
            rows.append(dict(task_id=f'{partition}_{arm}_seed{seed}_panel{panel:02}',partition=partition,
                panel=f'panel_{panel:02}',arm=arm,training_seed=seed,permutation_id=permutation,
                milestone=1 if first else 2,initialization='fresh',status='planned_not_trained'))
    for seed in seeds:
        for arm in ('equal','ranked','shared_only'):
            append('B',arm,seed)
    if campaign=='extended':
        for seed in seeds:
            for arm in ('equal','ranked'): append('O',arm,seed)
    for permutation in range(1,reference_permutations+1):
        for seed in (seeds if campaign=='extended' else [42]):
            append('B',f'shuffled_{permutation:02}',seed,permutation)
    return pd.DataFrame(rows).sort_values(['milestone','partition','training_seed','arm','panel']).reset_index(drop=True)


def permute_reference_weights(table, permutation_seeds=(11001,11002,11003)):
    """One draw per mapping/panel. Never redraw to obtain a QC correlation."""
    rows=[]; diagnostics=[]
    for panel_index,(panel,frame) in enumerate(table.groupby('panel_id',sort=True)):
        frame=frame.sort_values('dataset_order')
        names=frame.dataset_id.tolist();q=frame.raw_reference_score_q.to_numpy(float)
        if not np.isfinite(q).all() or not (q>0).all(): raise ValueError('Invalid audited reference scores.')
        real=q/q.sum();seen=set()
        for permutation,seed in enumerate(permutation_seeds,1):
            order=np.random.default_rng(np.random.SeedSequence([seed,panel_index])).permutation(len(q))
            identity=tuple(order.tolist())
            if identity in seen:
                raise ValueError(f'{panel}: duplicate one-shot reference mappings; report this draw, do not silently resample.')
            seen.add(identity)
            assigned=real[order]
            np.testing.assert_array_equal(np.sort(assigned),np.sort(real))
            np.testing.assert_allclose([assigned.sum(),assigned.max(),1/np.square(assigned).sum()],
                                       [1.,real.max(),1/np.square(real).sum()],rtol=1e-13,atol=1e-13)
            family_mass=pd.Series(assigned).groupby(frame.source_family.to_numpy()).sum()
            diagnostics.append(dict(panel_id=panel,permutation_id=permutation,permutation_seed=seed,
                panel_seed_index=panel_index,assigned_weight_q_correlation=float(np.corrcoef(assigned,q)[0,1]),
                N_ref=float(1/np.square(assigned).sum()),maximum_dataset_weight=float(assigned.max()),
                maximum_source_mass=float(family_mass.max()),largest_source=family_mass.idxmax()))
            for index,name in enumerate(names):
                rows.append(dict(panel_id=panel,dataset_id=name,source_family=frame.source_family.iloc[index],
                    permutation_id=permutation,permutation_seed=seed,panel_seed_index=panel_index,
                    donor_dataset_id=names[order[index]],global_rank=float(frame.global_rank.iloc[index]),
                    true_q=float(q[index]),assigned_q=float(q[order[index]]),pi=float(assigned[index])))
    return pd.DataFrame(rows),pd.DataFrame(diagnostics)


def check_execution_gate(manifest, *, approved_plan_hash, approved_partition_manifest,
                         max_new_trainings, authorize_training):
    """Approval never overrides failed scientific prerequisites."""
    if manifest['prerequisite_failures']:
        raise ValueError('Training blocked by scientific prerequisites: '+'; '.join(manifest['prerequisite_failures']))
    if not authorize_training:
        raise ValueError('Explicit --authorize-training is required.')
    if max_new_trainings is None or not 0<max_new_trainings<=manifest['planned_training_count']:
        raise ValueError('An explicit positive task cap within the frozen budget is required.')
    if approved_plan_hash!=manifest['plan_hash']:
        raise ValueError('Approved task-plan hash does not match.')
    if approved_partition_manifest is None or sha256(approved_partition_manifest)!=manifest['candidate_partition_sha256']:
        raise ValueError('Approved partition bytes do not match the proposed partition.')


def comparison_contract():
    return dict(
        primary_aggregation='Per seed/domain: median of pooled transcript-panel-pair values; never average profiles first.',
        sensitivity_aggregation='Separately named transcript-first median over pairs.',
        metrics=['PCC','Spearman','RMSE'],domains=['full_valid_CDS','interior_20'],interior_renormalization=False,
        bootstrap=dict(draws=5000,seed=20260911,cluster='transcript, keeping all paired panels, policies and corresponding training seeds together',
                       interpretation='conditional on fitted models; not independent panel pairs, codons or retraining on datasets'),
        shuffled_scope_main='B, seed 42 only; compare each mapping separately with equal/ranked seed 42',
        shuffled_scope_extended='B, each frozen mapping crossed with all training seeds',
        controls='equal and ranked both have learned gamma; one additional gamma=1 mean arm',
        local_weights='Fit once per partition/panel on training IDs only; reuse identical numerical references across arms/seeds',
        common_reference='NEW derived shape from uniform-panel mean(log L + g), normalized once on full valid CDS using logsumexp; save Z and log Z; do not change checkpoints',
        boundary_interaction='paired Delta_policy(full)-Delta_policy(interior)',
        peak_masks='union of top 1% or 5% positions over the declared comparison-family models; for equal/ranked at one seed use all eight models',
        random_removal=dict(draws=100,seed=20260912,rule='match union size and use identical random mask across compared models'),
        additive_diagnostics=dict(fit='each model own frozen predictions on common training intersection; total weight one per transcript',
            internal_knots=[.2,.4,.6,.8],boundary_knots=[0,1],codon_reference='AAA',
            signed_residuals=True,stop_codons='retained as codon categories',unknown_codons='existing design raises an error',
            reused_implementation='analyses/analyze_real_panel_posthoc_robustness_streaming.py'),
        illustration_selection='pending validation-only or pre-existing documented fixed-ID selection; no test improvement selection',
        historical_ranked_outputs='excluded',QC_is_biological_ground_truth=False,
        partitions_are_independent_compendia=False,
        exp8=dict(optional=True,separate_authorization=True,one_seed_fresh_training_budget=68,
                  existing_launcher='run_real_exp8_L_stability.py',primary='designated disjoint pairs; N80 subset overlap is unavoidable'))


def write_campaign_report(root):
    manifest=read_json(root/'campaign_manifest.json')
    matrix=pd.read_csv(root/'task_matrix.csv')
    availability=[]
    for row in matrix.itertuples():
        task_root=root/'runs'/row.partition/row.arm/f'seed{row.training_seed}'/row.panel
        state_path=task_root/'execution_status.json'
        state=read_json(state_path) if state_path.exists() else {}
        availability.append(dict(task_id=row.task_id,status=state.get('status',row.status),
            attempts=len(state.get('attempts',[])),completed=state.get('status')=='completed_native_L_validated'))
    pd.DataFrame(availability).to_csv(root/'training_availability.csv',index=False)
    manifest['completed_training_count']=sum(r['completed'] for r in availability)
    before_after=pd.read_csv(root/'balance_objective_before_after.csv')
    support=pd.read_csv(root/'validation_support_failures.csv')
    sections=[ '<h1>RiboUnmix reproducibility/reference campaign</h1>',
        '<p>Campaign status from recorded per-task attempts. No historical ranked performance is used. Native L validation is not completion of all proposed factor/robustness analyses.</p>',
        '<h2>Training availability</h2>'+pd.DataFrame(availability).to_html(index=False),
        '<h2>Current status</h2><pre>'+html.escape(json.dumps({k:manifest[k] for k in (
            'status','prerequisite_failures','planned_training_count','completed_training_count','plan_hash','candidate_partition_sha256')},indent=2))+'</pre>',
        '<h2>Original versus candidate QC objective</h2>'+before_after.to_html(index=False,float_format=lambda v:f'{v:.7g}'),
        '<p>The combined score is not a balance certificate. All blocks are reported. Source families remain intact; their concentration is not eliminated. No performance was used.</p>',
        '<h2>Failed frozen validation support</h2>'+support.to_html(index=False),
        '<p>No held-out IDs were moved to training or dropped. Test support changes do not justify dropping sequence-only test predictions.</p>',
        '<h2>Planned run budget, not completed models</h2>'+matrix.groupby(['partition','arm','milestone']).size().rename('planned_trainings').reset_index().to_html(index=False),
        '<p>Milestone 1: eight B equal/ranked seed-42 models. The remaining controls stay in the frozen plan; staging must not drop unfavorable controls.</p>',
        '<h2>Gamma=1 gradient consequence</h2><p>The production alpha head consumes detached dataset-context features. Setting gamma=1 removes the context encoder\'s learning path. The shared encoder and alpha head remain trainable. This is not identical learned dispersion features under an isolated mean intervention. No detachment or loss coefficient is changed.</p>',
        '<h2>Production formulation</h2><p>The resolved configuration is mass-free. Raw-replicate NB2 uses replicate-specific supplied mean scales; consensus PCC terms use the consensus-scaled model prediction. A generic model docstring still displays optional mass normalization: it is not the active resolved mass-free contract.</p>',
        '<h2>Provenance limitations</h2><p>The six-component ranking is frozen by content; its historical transcript scope and freeze chronology are not verified. This limitation is separate from train-only local reliability references. Historical code/environment/data identity is not inferred from matching YAML.</p>',
        '<h2>Completed versus pending implementation</h2><pre>'+html.escape(json.dumps(manifest['implementation_status'],indent=2))+'</pre>',
        '<h2>Design artifacts</h2><ul><li><a href="partition_preparation/audit_report.html">Original-partition audit</a></li>'
        '<li><a href="partition_preparation/global_rank_balanced_partition_v2/proposed_panel_rank_summary.csv">Candidate rank summary</a></li>'
        '<li><a href="reference_weights_and_permutations.csv">Prespecified reference mappings</a></li>'
        '<li><a href="comparison_contract.json">Frozen proposed analysis contract</a></li></ul>',
        '<h2>Commands and provenance</h2><pre>'+html.escape(json.dumps(manifest,indent=2))+'</pre>' ]
    (root/'campaign_report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>RiboUnmix campaign preparation</title><style>body{font:15px/1.5 system-ui;max-width:1100px;margin:32px auto;padding:0 20px}table{border-collapse:collapse;display:block;overflow:auto}td,th{padding:6px;border:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><body>'+''.join(sections)+'</body></html>')


def require_successful_preparation_audit(manifest, manifest_path):
    """A saved audit may be a failure report, not a completed input contract."""
    errors = list(dict.fromkeys([
        *manifest.get('hard_errors', []),
        *(issue['detail'] for issue in manifest.get('issues', [])
          if issue.get('severity') == 'hard_error'),
    ]))
    if errors:
        raise ValueError(
            f'CPU preparation audit failed: {manifest_path}\n'
            + '\n'.join(f'  - {error}' for error in errors)
            + '\nResolve these original audit errors before preparing a design. '
              '--resume cannot turn a failed audit into a successful one. '
              'Saved artifacts were not modified; no partition search or training was started.'
        )
    missing = [key for key in ('hard_errors', 'ranking', 'input_sha256', 'observed_design')
               if key not in manifest]
    if not isinstance(manifest.get('ranking'), dict) or not manifest['ranking'].get('sha256'):
        missing.append('ranking.sha256')
    if missing:
        raise ValueError(
            f'Incomplete CPU preparation audit: {manifest_path}; missing fields: {", ".join(missing)}. '
            'No successful audit is established. Inspect audit_report.html and the original preparation log; '
            'no ranking was inferred and no search was repeated.'
        )


def prepare_campaign(args):
    """Run/reuse the existing CPU prerequisite check; never launch training."""
    from analyses.audit_four_panel_reference_quality import parse_args as audit_args
    from Utils.panel_reference_audit import run_audit
    from Utils.panel_reference_preparation import prepare_design
    root=args.output_root
    if (root/'campaign_manifest.json').exists():
        raise ValueError('A campaign manifest already exists. Use a new versioned root; report does not rerun preparation.')
    root.mkdir(parents=True,exist_ok=True)
    pre=root/'partition_preparation'
    options=audit_args(['--mode','prepare-rank-balanced','--panel-manifest',str(args.panel_manifest),
        '--ranking-table',str(args.ranking_table),'--partition-seed',str(args.partition_seed),
        '--training-seed','42','--output-root',str(pre),'--gpus','inherit','--dry-run',
        '--require-validation-support']+
        (['--no-tex'] if args.no_tex else []))
    prerequisite_failure=None
    if (pre/'audit_manifest.json').exists():
        if not args.resume: raise ValueError('Existing CPU preparation requires --resume; the search will not be repeated.')
        audit_manifest=read_json(pre/'audit_manifest.json')
        require_successful_preparation_audit(audit_manifest, pre/'audit_manifest.json')
        for path,digest in audit_manifest['input_sha256'].items():
            if sha256(path)!=digest: raise ValueError(f'Preparation input changed: {path}')
        if audit_manifest['ranking']['sha256']!=sha256(args.ranking_table):
            raise ValueError('Requested ranking differs from the existing audit.')
    else:
        audit=run_audit(options);audit_manifest=audit['manifest']
        require_successful_preparation_audit(audit_manifest, pre/'audit_manifest.json')
        try: prepare_design(options,audit)
        except (ValueError,FileNotFoundError,RuntimeError) as exc: prerequisite_failure=str(exc)
    observed=audit_manifest['observed_design']
    expected=dict(datasets=114,source_families=85,panel_sizes=[29,29,28,28],global_rank_max=115)
    if observed!=expected: raise ValueError(f'Campaign universe conflicts with its declared contract: {observed}')
    candidate_dir=pre/'global_rank_balanced_partition_v2'
    preparation=read_json(candidate_dir/'rerun_plan.json')
    candidate=pd.read_csv(candidate_dir/'proposed_dataset_rank_reference_table.csv')
    search=read_json(candidate_dir/'partition_search.json')
    search_config=read_json(candidate_dir/'design_config.json')
    if not search_config.get('validation_support_constraint',{}).get('enabled'):
        raise ValueError('This campaign requires the new support-constrained search; use a new output root, not the failed v1 candidate.')
    split=read_json(args.panel_manifest.parent/'common_split_manifest.json')
    if any(len(split[k])!=1593 for k in ('common_validation_ids','common_test_ids')):
        raise ValueError('Frozen held-out counts differ from the declared 1593/1593; report the conflict before proceeding.')
    candidate_manifest=dict(partition='B',status='proposed_not_approved',
        panels={p:g.sort_values('dataset_order').dataset_id.tolist() for p,g in candidate.groupby('panel_id')},
        source_families={d:s for d,s in zip(candidate.dataset_id,candidate.source_family)},
        ranking_sha256=sha256(args.ranking_table),partition_seed=args.partition_seed,
        original_panel_manifest_sha256=sha256(args.panel_manifest),
        frozen_split_sha256=sha256(args.panel_manifest.parent/'common_split_manifest.json'),
        search_configuration_sha256=sha256(candidate_dir/'design_config.json'))
    write_json(root/'candidate_partition_manifest.json',candidate_manifest)
    shutil.copy2(args.ranking_table,root/'frozen_global_ranking.tsv')
    write_json(root/'ranking_provenance.json',audit_manifest['ranking'])
    permutations,perm_diagnostics=permute_reference_weights(candidate,args.permutation_seeds[:args.reference_permutations])
    permutations.to_csv(root/'reference_weights_and_permutations.csv',index=False)
    perm_diagnostics.to_csv(root/'reference_permutation_diagnostics.csv',index=False)
    blocks=[]
    for block,before in search['original_objective']['blocks'].items():
        after=search['selected']['objective']['blocks'][block]
        blocks.append(dict(block=block,weight=read_json(candidate_dir/'design_config.json')['block_weights'][block],
            original=before,candidate=after,change=after-before))
    blocks.append(dict(block='combined',weight=1.,original=search['original_objective']['total'],
        candidate=search['selected']['objective']['total'],change=search['selected']['objective']['total']-search['original_objective']['total']))
    pd.DataFrame(blocks).to_csv(root/'balance_objective_before_after.csv',index=False)
    support=pd.read_csv(candidate_dir/'heldout_support_report.csv')
    failed=support.loc[support.blocks_preparation]
    failed.to_csv(root/'validation_support_failures.csv',index=False)
    matrix=task_matrix(args.campaign,args.training_seeds,args.reference_permutations)
    failures=[]
    if not failed.empty:
        failures.append(f'{failed.transcript_id.nunique()} frozen validation transcripts fail minimum support in the proposed partition')
    if preparation['status']!='prepared_not_launched': failures.append(preparation.get('reason',prerequisite_failure or preparation['status']))
    if not search['improved']: failures.append('Prespecified search did not improve the combined objective; a design decision is required')
    matrix['status']='blocked_prerequisite' if failures else 'pending_partition_approval'
    matrix.to_csv(root/'task_matrix.csv',index=False)
    matrix[['task_id','status']].assign(attempts=0,completed=False).to_csv(root/'training_availability.csv',index=False)
    contract=comparison_contract();contract['bootstrap']['seed']=args.analysis_seed
    write_json(root/'comparison_contract.json',contract)
    code_hashes=snapshot_code(root/'code_snapshot')
    preparations={'B':candidate_dir}
    if not failures and args.campaign=='extended':
        original_options=audit_args(['--mode','prepare-existing','--panel-manifest',str(args.panel_manifest),
            '--ranking-table',str(args.ranking_table),'--output-root',str(root/'original_preparation'),
            '--gpus','inherit','--dry-run']+(['--no-tex'] if args.no_tex else []))
        original_audit=run_audit(original_options)
        if original_audit['manifest']['hard_errors']:
            raise ValueError('Original-partition audit failed: '+str(original_audit['manifest']['hard_errors']))
        prepare_design(original_options,original_audit)
        preparations['O']=original_options.output_root/'existing_partition_ranked_rerun_v2'
    execution_tasks=[]
    if not failures:
        from Utils.campaign_training import prepare_tasks, sequence_split_audit
        execution_tasks=prepare_tasks(root,matrix,preparations,permutations)
        sequence_split_audit(root,preparations,execution_tasks)
    environment={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    immutable={**code_hashes,**audit_manifest['input_sha256']}
    # The live orchestration code is also part of the approval, not only the
    # production snapshot executed by each model process.
    for relative in ('run_reproducibility_reference_campaign.py','Utils/reference_campaign.py',
                     'Utils/campaign_training.py','Utils/panel_reference_preparation.py',
                     'run_reproducibility_reference_campaign_univie.slurm',
                     'submit_reproducibility_reference_campaign_univie.sh'):
        immutable[str(ROOT/relative)]=sha256(ROOT/relative)
    for directory in preparations.values():
        # Freeze the observed data, shared split, fitted reliability, and templates.
        identity=pd.read_csv(directory/'data_artifact_identity.csv')
        immutable.update(dict(zip(identity.path,identity.sha256)))
        for path in (directory/'partition_and_split_manifests').glob('*'):
            if path.is_file(): immutable[str(path)]=sha256(path)
    for task in execution_tasks:
        immutable[task['config']]=task['config_sha256']
        cfg=yaml.safe_load(Path(task['config']).read_text())
        for path in (cfg['paths']['sequences_path'],cfg['data']['dataset_quality_ranking']['path']):
            if path not in immutable: immutable[path]=sha256(path)
        for name in ('split_manifest.json','reliability_reference_manifest.json'):
            path=Path(task['root'])/name;immutable[str(path)]=sha256(path)
    for name in ('candidate_partition_manifest.json','reference_weights_and_permutations.csv',
                 'comparison_contract.json','frozen_global_ranking.tsv','task_matrix.csv'):
        immutable[str(root/name)]=sha256(root/name)
    if execution_tasks:
        for name in ('sequence_coordinates.csv','common_training_intersection.json','initialization_and_gradient_audit.csv'):
            immutable[str(root/name)]=sha256(root/name)
    write_json(root/'frozen_execution.json',dict(status='ready_for_approval' if execution_tasks else 'blocked',
        file_sha256=immutable,software=software_provenance(),environment_packages=environment,
        preparation_root=str(ROOT),gpu_execution='One production process per allocated GPU; no DDP',
        environment_relocation='Prepare on the target cluster; do not copy a local frozen environment and claim exact equivalence.'))
    plan=dict(campaign=args.campaign,partition=candidate_manifest,tasks=json.loads(matrix.to_json(orient='records')),
        seeds=dict(partition=args.partition_seed,training=args.training_seeds,permutations=args.permutation_seeds[:args.reference_permutations],analysis=args.analysis_seed),
        comparison_contract=contract,resource_milestones=[8,len(matrix)-8],
        execution_tasks=execution_tasks,frozen_execution_sha256=sha256(root/'frozen_execution.json'),
        incomplete_scientific_coverage=args.training_seeds!=[42,43,44] or args.reference_permutations!=3)
    write_json(root/'plan_definition.json',plan)
    manifest=dict(status='blocked_prerequisite' if failures else 'awaiting_design_approval',
        prerequisite_failures=failures,campaign=args.campaign,planned_training_count=len(matrix),
        completed_training_count=0,training_launched=False,performance_used_for_design=False,
        execution_plan_ready=bool(execution_tasks),approval_status='not_eligible_for_training_approval' if failures else 'pending_design_review',
        plan_hash=object_hash(plan),candidate_partition_sha256=sha256(root/'candidate_partition_manifest.json'),
        ranking_sha256=sha256(root/'frozen_global_ranking.tsv'),
        source_hashes={str(Path(p).relative_to(root/'code_snapshot')):h for p,h in code_hashes.items()},
        wall_time_estimate='not estimated: no campaign training logs exist',
        implementation_status=dict(completed=['CPU original/candidate audit','bounded QC-only search','validation-support check',
            'planned main/extended task matrix','reference-permutation preservation checks','production explicit-reference and gamma=1 options',
            'standalone common-reference and symmetric peak/random-mask numerical primitives',
            *(['resolved production configurations','production initialization/gradient audit','per-task GPU execution and full-state resume'] if execution_tasks else [])],
            pending=['user approval and GPU training','native factor/training-intersection exports','multi-seed matched campaign analysis',
            'campaign robustness/observation-fit integration and performance figures']),
        command=shlex.join([sys.executable,*sys.argv]))
    write_json(root/'campaign_manifest.json',manifest)
    write_campaign_report(root)
    return manifest


def execute_campaign(args):
    from Utils.campaign_training import verify_campaign_execution, selected_tasks, run_task
    root=args.output_root
    manifest=read_json(root/'campaign_manifest.json')
    check_execution_gate(manifest,approved_plan_hash=args.approved_plan_hash,
        approved_partition_manifest=args.approved_partition_manifest,max_new_trainings=args.max_new_trainings,
        authorize_training=args.authorize_training)
    if not manifest.get('execution_plan_ready'):
        raise ValueError('This campaign has no prepared execution tasks; prepare the support-constrained v2 design.')
    if args.gpus!='inherit':
        raise ValueError('Campaign jobs require --gpus inherit; allocate one GPU per process using the scheduler.')
    plan=verify_campaign_execution(root,manifest)
    tasks=selected_tasks(plan,args.max_new_trainings,args.task_index)
    if args.dry_run:
        print(f'Execution preflight OK: {len(tasks)} tasks selected within cap {args.max_new_trainings}; no training launched.')
        return 0
    failures=[]
    for task in tasks:
        try:
            run_task(task,manifest['plan_hash'],resume=args.resume)
        except (ValueError,RuntimeError,OSError) as exc:
            failures.append(dict(task_id=task['task_id'],reason=str(exc)))
            print(f'{task["task_id"]}: {exc}',file=sys.stderr)
    return 2 if failures else 0
