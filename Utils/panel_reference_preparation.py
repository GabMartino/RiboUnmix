"""Preparation only: reuse production training, source groups and split loaders."""
from __future__ import annotations

import copy
import fcntl
import importlib.metadata
import json
from pathlib import Path
import platform
import shlex
import shutil
import sys

import numpy as np
import pandas as pd
import yaml
from hydra import compose, initialize_config_dir

from Utils.panel_reference_audit import ROOT, ORIGINAL_QC, add_weights, read_json, sha256, summarize, plot_audit, software_provenance
from Utils.real_panel_convergence import (assert_panel_partition, deterministic_stratified_panel_assignment,
    refine_source_atomic_panel_assignment, load_sequence_metadata, load_support_and_stored_weights,
    fit_panel_reliability_manifest, write_json, _support_summary)
from Utils.external_transcript_split import load_external_transcript_split
from Utils.reliability_references import load_reliability_reference_manifest, transcript_id_hash
from run_real_independent_panel_convergence_quality_rank import _flatten_config, _assert_only_gamma_reference_changed

DEFAULT_SEARCH = dict(restarts=4,proposal_budget_per_restart=2000,
    block_weights=dict(original_qc=1.,global_rank_distribution=1.,quality_groups=1.,component_ranks=1.,
                       ranked_reference_groups=1.,ranked_reference_components=.5))


def balance_objective(table, spec):
    """Fixed QC-only loss: mean squared block discrepancies from full-universe targets."""
    names = table.dataset_id.tolist()
    panels = sorted(table.panel_id.unique())
    q = table.raw_reference_score_q.to_numpy(float)
    rank = table.global_rank.to_numpy(float)
    cuts = np.quantile(rank,np.arange(.125,1,.125),method='linear')
    groups = np.column_stack([(table.global_quality_group==g).to_numpy(float) for g in range(1,5)])

    def qc_matrix(columns):
        blocks=[]
        for c in columns:
            x=pd.to_numeric(table[c],errors='coerce').to_numpy(float)
            valid=np.isfinite(x)
            if not valid.any():
                continue
            sd=x[valid].std(ddof=0)
            if sd>0:
                blocks.append((x-x[valid].mean())/sd)
            if not valid.all():
                blocks.append((~valid).astype(float))
        return np.column_stack(blocks) if blocks else np.empty((len(table),0))

    components=qc_matrix([c for c in table if c.startswith('rank_') and c not in ('rank_component_count','rank_missing')])
    blocks=dict(original_qc=(qc_matrix([c for c in ORIGINAL_QC if c in table]),np.ones(len(table))),
        global_rank_distribution=(np.column_stack([(rank-rank.mean())/(rank.std(ddof=0) or 1),*(rank<=c for c in cuts)]),np.ones(len(table))),
        quality_groups=(groups,np.ones(len(table))),component_ranks=(components,np.ones(len(table))),
        ranked_reference_groups=(groups,q),ranked_reference_components=(components,q))

    def moments(x,w):
        observed=np.isfinite(x)
        denominator=(observed*w[:,None]).sum(axis=0)
        return np.divide((np.where(observed,x,0)*w[:,None]).sum(axis=0),denominator,
                         out=np.full(x.shape[1],np.nan),where=denominator>0)
    targets={name:moments(x,w) for name,(x,w) in blocks.items()}
    def score(assignment,details=False):
        membership=assignment.set_index('dataset_name').panel.reindex(names).to_numpy()
        results={}
        for name,(x,w) in blocks.items():
            if not x.shape[1]:
                results[name]=0.; continue
            delta=np.vstack([moments(x[membership==p],w[membership==p])-targets[name] for p in panels])
            # A wholly missing panel feature is not evidence of good balance.
            results[name]=float(np.mean(np.where(np.isfinite(delta),delta**2,1.)))
        total=sum(spec['block_weights'][k]*v for k,v in results.items())
        return dict(total=total,blocks=results) if details else total
    return score


def validation_support_constraint(table, supports, split):
    """Observed eligibility only; never a validation loss or prediction criterion."""
    names = table.dataset_id.tolist()
    ids = list(split['common_validation_ids'])
    minimum = int(split.get('minimum_usable_datasets_per_panel', 2))
    support = np.array([[t in supports[d] for t in ids] for d in names], dtype=np.int16)
    panels = sorted(table.panel_id.unique())

    def feasible(assignment):
        membership = assignment.set_index('dataset_name').panel.reindex(names).to_numpy()
        return all(bool((support[membership == p].sum(axis=0) >= minimum).all()) for p in panels)

    return feasible


def prepare_partition(args,audit,destination,*,supports=None):
    spec=copy.deepcopy(DEFAULT_SEARCH)
    if args.design_config:
        given=read_json(args.design_config)
        if set(given)-set(spec):
            raise ValueError(f'Unknown design configuration keys: {set(given)-set(spec)}')
        for key,value in given.items():
            if key=='block_weights':
                if set(value)-set(spec[key]): raise ValueError('Unknown balance block.')
                spec[key].update(value)
            else: spec[key]=value
    if spec['restarts']<1 or spec['proposal_budget_per_restart']<0 or any(not np.isfinite(v) or v<0 for v in spec['block_weights'].values()):
        raise ValueError('Invalid prespecified search budget/block weights.')
    spec.update(partition_seed=args.partition_seed,global_quality_groups=audit['manifest']['grouping'],
        capacities=[len(v) for v in audit['panels'].values()],
        objective='Sum of declared weights times mean squared per-panel block deviations. Unweighted targets: full retained uniform collection; ranked targets: full retained q-weighted collection. Numeric features use full retained observed-value population SD; ECDF/proportions are in [0,1].',
        missingness='Observed-value means renormalized; missingness indicators included. Wholly unobserved panel features receive a fixed squared-discrepancy penalty of 1; no rank imputation.',
        tie_handling='Strict >1e-12 improvement within seeded swaps; final ties lexicographic by dataset-sorted panel assignment.',
        neighborhood='Existing source-atomic greedy initialization and equal-size source-family swaps; bounded search, no optimality guarantee.',
        performance_inputs='none')
    constrained = bool(getattr(args, 'require_validation_support', False))
    spec['validation_support_constraint'] = dict(enabled=constrained,
        minimum=int(audit['split'].get('minimum_usable_datasets_per_panel', 2)),
        validation_id_hash=transcript_id_hash(audit['split']['common_validation_ids']),
        initialization='Frozen feasible original partition for each seeded restart' if constrained else 'Historical search initializations',
        rule='Reject any swap leaving any frozen validation transcript below minimum support in any panel; no losses or predictions used.')
    # Commit the protocol to disk before any candidate is scored.
    write_json(destination/'design_config.json',spec)
    table=audit['table']
    original=table.rename(columns={'dataset_id':'dataset_name','panel_id':'panel','source_family':'source_identifier'}).copy()
    original['eligible']=True
    feasible = validation_support_constraint(table, supports, audit['split']) if constrained else None
    if feasible is not None and not feasible(original):
        raise ValueError('Original partition fails validation support on current observations; no constrained search started.')
    objective=balance_objective(table,spec)
    candidates=[(original,dict(initial='frozen existing panels',objective=objective(original,True)))]
    for index in range(spec['restarts']):
        seed=args.partition_seed+index
        if index==0 or constrained:
            candidate,method=refine_source_atomic_panel_assignment(original,objective=objective,seed=seed,
                proposal_budget=spec['proposal_budget_per_restart'],feasible=feasible)
        else:
            try:
                candidate,method=deterministic_stratified_panel_assignment(original,seed=seed,
                    additional_balance_objective=objective,additional_swap_budget=spec['proposal_budget_per_restart'])
            except RuntimeError as exc:
                candidates.append((None,dict(seed=seed,status='initialization_failed',reason=str(exc))))
                continue
        candidates.append((candidate,dict(seed=seed,method=method,objective=objective(candidate,True))))
    valid=[(frame,meta) for frame,meta in candidates if frame is not None]
    best,selected=min(valid,key=lambda item:(item[1]['objective']['total'],tuple(item[0].sort_values('dataset_name').panel)))
    history=[meta for _,meta in candidates]
    write_json(destination/'partition_search.json',dict(candidates=history,selected=selected,
        original_objective=candidates[0][1]['objective'],improved=objective(best)<objective(original)-1e-12,
        no_improvement_is_not_infeasibility_proof=True))
    assert_panel_partition(best,expected_datasets=table.dataset_id.tolist())
    if best.groupby('panel').size().tolist()!=spec['capacities']:
        raise ValueError('Proposed partition violated exact capacities.')
    # Preserve original relative ID order inside each newly selected subset.
    membership=best.set_index('dataset_name').panel.to_dict()
    panels={p:[d for d in table.dataset_id if membership[d]==p] for p in audit['panels']}
    proposed=table.copy(); proposed['panel_id']=proposed.dataset_id.map(membership)
    proposed=add_weights(proposed)
    proposed.to_csv(destination/'proposed_dataset_rank_reference_table.csv',index=False)
    features=list(audit['summaries']['panel_component_balance'].feature.unique())
    summaries=summarize(proposed,features)
    for name,frame in summaries.items(): frame.to_csv(destination/f'proposed_{name}.csv',index=False)
    plot_audit(proposed,summaries,audit['manifest']['grouping'],args,destination/'proposed_figures')
    return panels,proposed,spec


def relocated_input(raw):
    """Exact repository-relative suffix relocation; never basename/fuzzy matching."""
    path=Path(raw)
    if path.is_file(): return path.resolve()
    if not path.is_absolute() and (ROOT/path).is_file(): return (ROOT/path).resolve()
    if 'Datasets' in path.parts:
        candidate=ROOT.joinpath(*path.parts[path.parts.index('Datasets'):])
        if candidate.is_file(): return candidate.resolve()
    raise FileNotFoundError(f'Data artifact missing: {raw}; supply the relocated dataset config / sequence path.')


def freeze_heldout_split(old,panels,supports,sequence_ids,*,existing=False):
    """Preserve validation/test IDs; recompute only panel training eligibility."""
    heldout=set(old['common_validation_ids'])|set(old['common_test_ids'])
    missing=sorted(heldout-set(sequence_ids))
    if missing: raise ValueError(f'Frozen held-out sequences missing: {missing}')
    result=copy.deepcopy(old); result['panels']=panels
    result['panel_train_eligible_ids']={}; result['panel_support_statistics']={}
    rows=[]; minimum=int(old.get('minimum_usable_datasets_per_panel',2))
    feasible=True
    for panel,names in panels.items():
        count={t:sum(t in supports[d] for d in names) for t in sequence_ids}
        train=(list(old['panel_train_eligible_ids'][panel]) if existing else
               sorted(t for t,n in count.items() if t not in heldout and n>=minimum))
        if not train or set(train)&heldout or any(count.get(t,0)<minimum for t in train):
            raise ValueError(f'{panel}: empty, leaked or unsupported training fold.')
        result['panel_train_eligible_ids'][panel]=train
        for fold,ids in [('validation',old['common_validation_ids']),('test',old['common_test_ids'])]:
            for t in ids:
                supported=count[t]>=minimum
                rows.append(dict(panel_id=panel,transcript_id=t,fold=fold,usable_dataset_count=count[t],
                    minimum=minimum,supported=supported,blocks_preparation=fold=='validation' and not supported))
                if fold=='validation' and not supported: feasible=False
        result['panel_support_statistics'][panel]={
            'number_of_train_eligible_transcripts':len(train),
            'train_support':_support_summary([count[t] for t in train]),
            'validation_support':_support_summary([count[t] for t in old['common_validation_ids']]),
            'test_support':_support_summary([count[t] for t in old['common_test_ids']])}
    if not existing:
        result['common_evaluation_ids']=sorted(t for t in sequence_ids if all(sum(t in supports[d] for d in ds)>=minimum for ds in panels.values()))
        result['number_of_common_evaluation_transcripts']=len(result['common_evaluation_ids'])
        result['train_excluded_ids']=sorted(heldout)
        result['stratification']={'method':'Frozen historical held-out IDs, no resampling; panel-specific training eligibility recomputed.'}
        result['assertions']={'train_validation_overlap':0,'train_test_overlap':0,'validation_test_overlap':0,
            'all_train_transcripts_meet_panel_support':True,'all_validation_transcripts_meet_panel_support':feasible,
            'test_list_retained_for_sequence_only_prediction':True}
    return result,pd.DataFrame(rows),feasible


def assert_training_contract(cfg):
    checks={'grouped sampling':cfg['data']['train_sampling_strategy']=='transcript_grouped_multidataset_pairs',
        'transcript-balanced loss':cfg['loss']['sample_reduction']=='transcript_balanced',
        'raw-replicate NB2':cfg['loss']['experiment_mode']=='standard_nb' and cfg['loss']['replica_nb_weight']>0,
        'consensus shape terms':cfg['loss']['consensus_raw_pcc_weight']>0 and cfg['loss']['consensus_nb_vst_pcc_weight']>0,
        'mass conservation disabled':cfg['model']['mass_conservation'] is False,
        'fixed full-panel gamma reference':cfg['model']['gamma_centering']['mode']=='fixed_reference' and
            (cfg['model']['gamma_centering']['reference'].get('dataset_names') is None or cfg['model']['gamma_centering']['reference']['dataset_names']==cfg['experiment']['dataset']),
        'best validation loss only':cfg['prediction']['checkpoint_variants']==['best_val_loss'],
        'fresh initialization':not cfg['experiment'].get('from_checkpoint') and not cfg['experiment'].get('resume_training_state')}
    if not all(checks.values()): raise ValueError(f'Production training contract failed: {[k for k,v in checks.items() if not v]}')


def paired_config_differences(equal,ranked):
    a,b=_flatten_config(equal),_flatten_config(ranked)
    allowed={'name','model.gamma_centering.reference.weighting',
             'model.gamma_centering.reference.quality_rank_power'}
    differences=[]
    for key in sorted(set(a)|set(b)):
        if a.get(key)!=b.get(key):
            approved=key in allowed or key.startswith(('paths.checkpoints','paths.logs','paths.results','orchestrator.'))
            if not approved: raise ValueError(f'Paired configurations differ unexpectedly: {key}')
            differences.append(dict(key=key,equal=a.get(key),ranked=b.get(key),approved=True))
    return differences


def snapshot_code(destination):
    paths=set(ROOT.glob('*.py'))
    for folder in ('Models','Dataloaders','Utils','results/gamma_ablation'):
        paths.update((ROOT/folder).rglob('*.py'))
    paths.update((ROOT/'results').glob('*.py'))
    for folder in ('config','Datasets/encodings'):
        paths.update((ROOT/folder).rglob('*.yaml'))
        paths.update((ROOT/folder).rglob('*.json'))
    hashes={}
    for path in sorted(paths):
        target=destination/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        hashes[str(target)]=sha256(target)
    return hashes


def verify_prepared_inputs(root):
    """Called by generated commands only after explicit authorization."""
    frozen=read_json(Path(root)/'frozen_execution.json')
    for name,digest in frozen['file_sha256'].items():
        if sha256(name)!=digest: raise RuntimeError(f'Frozen input/code/config changed: {name}')
    if platform.python_version()!=frozen['software']['python']:
        raise RuntimeError('Prepared Python version changed.')
    for package,expected in frozen['environment_packages'].items():
        if importlib.metadata.version(package)!=expected:
            raise RuntimeError(f'Prepared environment changed: {package}')


def _atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    write_json(temporary, payload)
    temporary.replace(path)


def _preparation_files(destination):
    """Immutable artifacts at the final-audit boundary; status files are separate."""
    excluded = {'rerun_plan.json', 'preparation_state.json', 'frozen_execution.json',
                'launch_commands.sh', 'preparation_recovery.json'}
    return {str(p): sha256(p) for p in sorted(destination.rglob('*'))
            if p.is_file() and p.name not in excluded
            and p.suffix in ('.py', '.yaml', '.json', '.sh', '.tsv', '.csv')}


def _legacy_preparation_state(destination, plan):
    """Recover the final-audit boundary written by older preparation versions.

    The failed versions had already saved all scientific artifacts, but did not
    save tasks/input hashes together until after the matched audit succeeded.
    Read those artifacts in place; never repeat assignment or reliability fits.
    """
    audit_path = destination.parent / 'audit_manifest.json'
    audit = read_json(audit_path)
    if audit.get('hard_errors') or audit.get('mode') != plan['design']:
        raise ValueError('Cannot resume a failed or incompatible initial CPU audit.')
    frozen_rank = destination / 'frozen_global_ranking.tsv'
    original = destination / 'original_global_ranking.tsv'
    original = original if original.is_file() else frozen_rank
    if sha256(original) != audit['ranking']['sha256']:
        raise ValueError('Saved ranking differs from the ranking used to select the partition.')
    common_path = destination / 'partition_and_split_manifests/common_split_manifest.json'
    common = read_json(common_path)
    historical = read_json(destination / 'equal/panel_manifest.json')
    old_split = Path(historical['source_historical_panel_manifest']).parent / 'common_split_manifest.json'
    original_split = read_json(old_split)
    for fold in ('common_validation_ids', 'common_test_ids'):
        if common[fold] != original_split[fold]:
            raise ValueError(f'Recovery would change frozen {fold}.')
    if audit['input_sha256'].get(str(old_split)) != sha256(old_split):
        raise ValueError('Original common split changed since the CPU audit.')

    assignment = pd.read_csv(destination / 'partition_and_split_manifests/panel_assignment.csv')
    canonical = assignment.rename(columns={'dataset_id': 'dataset_name', 'panel_id': 'panel',
                                           'source_family': 'source_identifier'})
    assert_panel_partition(canonical, expected_datasets=historical['retained_datasets'])
    expected_sizes = historical['panel_sizes']
    if assignment.groupby('panel_id').size().to_dict() != expected_sizes:
        raise ValueError('Saved partition capacities differ from the panel manifests.')
    read_json(destination / 'comparison_contract.json')  # present only after all eight arms exist
    if plan['design'] == 'prepare-rank-balanced':
        read_json(destination / 'partition_search.json')
        read_json(destination / 'design_config.json')

    artifacts = pd.read_csv(destination / 'data_artifact_identity.csv')
    input_hashes = dict(zip(artifacts.path, artifacts.sha256))
    tasks, training_seeds = [], {}
    for panel, datasets in historical['panels'].items():
        train, val, test, _ = load_external_transcript_split(
            common_path, panel_name=panel, experiment_datasets=datasets)
        assigned = assignment.loc[assignment.panel_id == panel, 'dataset_id'].tolist()
        if set(assigned) != set(datasets):
            raise ValueError(f'{panel}: saved assignment differs from its configuration.')
        sources = sorted(assignment.loc[assignment.panel_id == panel, 'source_family'].unique())
        if sources != sorted(historical['panel_source_families'][panel]):
            raise ValueError(f'{panel}: source families differ from the saved manifest.')
        for policy in ('equal', 'ranked'):
            config = destination / f'resolved_configs/{panel}_{policy}.yaml'
            cfg = yaml.safe_load(config.read_text())
            task = destination / policy / panel
            if config.read_bytes() != (task / 'resolved_config.yaml').read_bytes():
                raise ValueError(f'{panel}/{policy}: prepared and run configurations differ.')
            assert_training_contract(cfg)
            if read_json(task / 'split_manifest.json') != dict(train_ids=train, validation_ids=val, test_ids=test):
                raise ValueError(f'{panel}/{policy}: split differs from the common manifest.')
            reference = destination / f'partition_and_split_manifests/{panel}_reliability_reference_manifest.json'
            reliability = read_json(reference)
            if (read_json(task / 'reliability_reference_manifest.json') != reliability
                    or reliability.get('reference_split') != 'training_only'
                    or reliability.get('heldout_rows_used_for_fitting') != 0
                    or reliability.get('panel_training_transcript_id_hash') != transcript_id_hash(train)):
                raise ValueError(f'{panel}/{policy}: train-only reliability contract differs.')
            for name in datasets:
                path = str(Path(cfg['dataset_config']['dataset_path'][name]).resolve())
                if path not in input_hashes:
                    raise ValueError(f'{panel}: data identity missing for {name}.')
            sequences = str(Path(cfg['paths']['sequences_path']).resolve())
            if sequences not in input_hashes:
                input_hashes[sequences] = sha256(sequences)
            training_seeds[panel] = int(cfg['experiment']['seed'])
            command = [sys.executable, '-u', str(destination / 'code_snapshot/main_ribounmix_multidataset.py'),
                       f'--config-path={config.parent}', f'--config-name={config.stem}',
                       f'hydra.run.dir={task / "hydra"}', 'hydra.job.chdir=false']
            tasks.append(dict(index=len(tasks), panel=panel, policy=policy, command=command, config=str(config)))
    if len(tasks) != plan['number_of_fresh_trainings']:
        raise ValueError('Not all prepared panel/policy configurations are present.')
    environment = read_json(destination / 'environment.json')
    plan = dict(plan, tasks=tasks, training_seeds=training_seeds, audit_manifest=str(audit_path),
                dry_run_requested=True,
                frozen_ranking_sha256=sha256(frozen_rank), original_ranking_sha256=sha256(original))
    return dict(plan=plan, input_sha256=input_hashes, file_sha256=_preparation_files(destination),
                software=environment['software'], environment_packages=environment['packages'],
                legacy_recovery=True,
                provenance_note='Legacy interrupted preparation: dataset hashes checked against data_artifact_identity.csv; '
                    'sequence and snapshot/config hashes first recorded at recovery. No claim of prior hash verification for those files.')


def finish_preparation(destination, *, gpus='inherit'):
    """Resume only final validation/freezing of an already selected design."""
    from analyses.compare_real_panel_weighting import audit_design

    destination = Path(destination).expanduser().resolve()
    if not (destination / 'rerun_plan.json').is_file():
        raise FileNotFoundError(f'Missing preparation plan: {destination / "rerun_plan.json"}')
    with (destination / '.finalize.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = read_json(destination / 'rerun_plan.json')
        if previous.get('status') == 'prepared_not_launched' and (destination / 'frozen_execution.json').is_file():
            verify_prepared_inputs(destination)
            print(f'Reusing completed preparation: {destination}', flush=True)
            return previous
        if previous.get('training_launched') or next(destination.rglob('*.ckpt'), None) is not None:
            raise ValueError('Finalization recovery is limited to designs with no training/checkpoints.')
        try:
            state_path = destination / 'preparation_state.json'
            state = read_json(state_path) if state_path.is_file() else _legacy_preparation_state(destination, previous)
            for path, expected in {**state['input_sha256'], **state['file_sha256']}.items():
                if sha256(path) != expected:
                    raise ValueError(f'Input changed after preparation: {path}')
            if platform.python_version() != state['software']['python']:
                raise ValueError('Prepared Python version changed.')
            for package, expected in state['environment_packages'].items():
                if importlib.metadata.version(package) != expected:
                    raise ValueError(f'Prepared environment changed: {package}')
            support = pd.read_csv(destination / 'heldout_support_report.csv')
            validation = support.loc[support.fold == 'validation']
            common = read_json(destination / 'partition_and_split_manifests/common_split_manifest.json')
            expected_validation = {(p, t) for p in common['panels'] for t in common['common_validation_ids']}
            if (len(validation) != len(expected_validation)
                    or set(zip(validation.panel_id, validation.transcript_id)) != expected_validation
                    or not validation.supported.eq(True).all()):
                raise ValueError('Frozen validation support is incomplete; cannot finalize.')
            frozen_rank = destination / 'frozen_global_ranking.tsv'
            if sha256(frozen_rank) != state['plan']['frozen_ranking_sha256']:
                raise ValueError('Frozen ranking changed after panel selection.')
            # Explicit input avoids old analysis versions consulting the six-component default.
            audit_design(destination / 'equal', destination / 'ranked', ranking_table=frozen_rank)
            plan = dict(state['plan'])
            tasks = plan['tasks']
            snapshots = destination / 'code_snapshot'
            tokens = gpus.split(',')
            if not tokens or any(not token.strip() for token in tokens):
                raise ValueError('Supply GPU visibility tokens or inherit.')
            lines = ['#!/usr/bin/env bash', 'set -euo pipefail',
                     'if [[ "${RIBOUNMIX_TRAINING_AUTHORIZED:-}" != "1" ]]; then',
                     '  echo "Preparation only. GPU training requires subsequent explicit authorization." >&2',
                     '  exit 2', 'fi', f'cd {shlex.quote(str(snapshots))}',
                     shlex.join([sys.executable, '-c', 'from Utils.panel_reference_preparation import verify_prepared_inputs; import sys; verify_prepared_inputs(sys.argv[1])', str(destination)]),
                     'case "${1:?Select one task index; see rerun_plan.json}" in']
            for task in tasks:
                visibility = '' if gpus == 'inherit' else f'export CUDA_VISIBLE_DEVICES={shlex.quote(tokens[task["index"] % len(tokens)].strip())}; '
                lines.append(f'  {task["index"]}) {visibility}exec {shlex.join(task["command"])} ;;')
            lines += ['  *) echo "Invalid task index" >&2; exit 2 ;;', 'esac']
            (destination / 'launch_commands.sh').write_text('\n'.join(lines) + '\n')
            if previous.get('status') == 'blocked_before_training' or state.get('legacy_recovery'):
                _atomic_json(destination / 'preparation_recovery.json', dict(
                    previous_plan=previous, legacy_recovery=bool(state.get('legacy_recovery')),
                    provenance_note=state.get('provenance_note'), partition_search_repeated=False,
                    scientific_artifacts_rewritten=False, finalizer_sha256=sha256(Path(__file__)),
                    matched_audit_file=str(Path(sys.modules[audit_design.__module__].__file__)),
                    matched_audit_sha256=sha256(Path(sys.modules[audit_design.__module__].__file__)),
                ))
            plan.update(status='prepared_not_launched', training_launched=False, gpu_visibility=gpus)
            plan.pop('reason', None)
            # Publish readiness last. A failure before this write remains resumable.
            hashes = {**state['input_sha256'], **_preparation_files(destination)}
            for name in ('launch_commands.sh', 'preparation_state.json', 'preparation_recovery.json'):
                path = destination / name
                if path.is_file(): hashes[str(path)] = sha256(path)
            temporary_plan = destination / 'rerun_plan.json.tmp'
            write_json(temporary_plan, plan)
            hashes[str(destination / 'rerun_plan.json')] = sha256(temporary_plan)
            _atomic_json(destination / 'frozen_execution.json', dict(file_sha256=hashes,
                         software=state['software'], environment_packages=state['environment_packages']))
            temporary_plan.replace(destination / 'rerun_plan.json')
            print(f'Prepared {len(tasks)} fresh runs at {destination}. No training launched.', flush=True)
            return plan
        except (ValueError, KeyError, FileNotFoundError, RuntimeError, AssertionError, TypeError) as exc:
            previous.update(status='blocked_before_training', reason=f'{type(exc).__name__}: {exc}', training_launched=False)
            _atomic_json(destination / 'rerun_plan.json', previous)
            raise


def prepare_design(args,audit):
    destination=args.output_root/('existing_partition_ranked_rerun_v2' if args.mode=='prepare-existing' else 'global_rank_balanced_partition_v2')
    destination.mkdir()
    plan=dict(design=args.mode,status='preparing',training_launched=False,historical_equal_reuse='unverified',
        fresh_ranked_models=4,fresh_equal_models=4,number_of_fresh_trainings=8,
        reason_for_fresh_equal='Historical source/environment/data identity is not established; configurations alone are insufficient.',
        old_ranked_checkpoints_used=False,performance_used_for_design=False)
    write_json(destination/'rerun_plan.json',plan)
    try:
        if audit['split'] is None or set(audit['configs'])!=set(audit['panels']):
            raise ValueError('Preparation requires the common split and every panel training configuration.')
        table=audit['table']
        configs={p:copy.deepcopy(c['config']) for p,c in audit['configs'].items()}
        base=next(iter(configs.values()))
        mapping_payload=yaml.safe_load(args.dataset_config.read_text()) if args.dataset_config else base['dataset_config']
        mapping=mapping_payload['dataset_path']
        universe=set(table.dataset_id)
        if not universe<=set(mapping): raise ValueError(f'Dataset config misses {sorted(universe-set(mapping))}.')
        paths={d:str(relocated_input(mapping[d])) for d in table.dataset_id}
        sequences=relocated_input(args.sequences_path or base['paths']['sequences_path'])
        input_hashes={path:sha256(path) for path in [*paths.values(),str(sequences)]}
        artifact_rows=[]
        for d,path in paths.items():
            saved=table.loc[table.dataset_id==d]
            expected_size=(saved.source_file_size_bytes.iloc[0] if 'source_file_size_bytes' in saved else None)
            observed_size=Path(path).stat().st_size
            matches=bool(observed_size==expected_size) if expected_size is not None and pd.notna(expected_size) else None
            artifact_rows.append(dict(dataset_id=d,path=path,sha256=input_hashes[path],
                current_size_bytes=observed_size,historical_size_bytes=expected_size,size_matches=matches,
                historical_cryptographic_identity='not verified'))
        pd.DataFrame(artifact_rows).to_csv(destination/'data_artifact_identity.csv',index=False)
        if any(row['size_matches'] is False for row in artifact_rows):
            raise ValueError('Dataset byte sizes differ from the frozen QC artifact provenance; see data_artifact_identity.csv. No automatic data substitution.')
        plan['historical_data_hash_match']='not verified: current input hashes frozen, historical training-linked cryptographic hashes absent'
        metadata,_=load_sequence_metadata(sequences,max_cds_codons=audit['split'].get('maximum_cds_codons'))
        supports,_,_=load_support_and_stored_weights(paths,eligible_sequence_ids=set(metadata.index))
        if args.mode=='prepare-existing':
            panels=audit['panels']
            plan['interpretation']='Reference policy changes within existing panels; this does not repair between-panel quality imbalance.'
        else:
            panels,table,_=prepare_partition(args,audit,destination,supports=supports)
            plan['interpretation']='New QC-only partition with within-design matched equal/ranked controls; not an isolated weighting comparison against old models.'
        common,support_report,feasible=freeze_heldout_split(audit['split'],panels,supports,set(metadata.index),existing=args.mode=='prepare-existing')
        support_report.to_csv(destination/'heldout_support_report.csv',index=False)
        if not feasible:
            raise ValueError('Frozen common validation support is infeasible. See heldout_support_report.csv; no split redrawn or training prepared. Frozen test IDs are retained.')
        partition_dir=destination/'partition_and_split_manifests';partition_dir.mkdir()
        common_path=partition_dir/'common_split_manifest.json';write_json(common_path,common)
        table[['dataset_id','panel_id','source_family']].to_csv(partition_dir/'panel_assignment.csv',index=False)
        frozen_rank=destination/'frozen_global_ranking.tsv'
        frozen_rank.write_bytes(args.ranking_table.read_bytes())
        aliases=audit['manifest']['ranking']['alias_mapping']
        column_mapping={args.ranking_dataset_column:'dataset',args.ranking_rank_column:'quality_rank'}
        if aliases or any(k!=v for k,v in column_mapping.items()):
            shutil.copy2(frozen_rank,destination/'original_global_ranking.tsv')
            full=pd.read_csv(frozen_rank,sep='\t')
            for old,new in column_mapping.items():
                if old!=new and new in full and new not in column_mapping:
                    raise ValueError(f'Canonical ranking column {new} already exists; resolve its meaning explicitly.')
            full[args.ranking_dataset_column]=full[args.ranking_dataset_column].map(lambda d:aliases.get(str(d),str(d)))
            full=full.rename(columns=column_mapping)
            full.to_csv(frozen_rank,sep='\t',index=False)
        plan['ranking_normalization']=dict(column_mapping=column_mapping,aliases=aliases,
            note='Complete table retained with unchanged rank values; canonical column names support the existing matched analysis.')
        source_families={p:sorted(table.loc[table.panel_id==p,'source_family'].unique()) for p in panels}
        snapshots=destination/'code_snapshot';code_hashes=snapshot_code(snapshots)
        environment={dist.metadata['Name']:dist.version for dist in importlib.metadata.distributions() if dist.metadata['Name']}
        write_json(destination/'environment.json',dict(software=software_provenance(),packages=environment))
        tasks=[];differences=[];historical_differences=[];training_seeds={}
        config_dir=destination/'resolved_configs';config_dir.mkdir()
        for panel,ds in panels.items():
            train,val,test,_=load_external_transcript_split(common_path,panel_name=panel,experiment_datasets=ds)
            if args.mode=='prepare-existing':
                if panel not in audit['reliability']: raise ValueError(f'{panel}: missing train-only reliability reference.')
                reliability=copy.deepcopy(load_reliability_reference_manifest(audit['reliability'][panel]['path']))
                if (reliability.get('reference_split')!='training_only' or reliability.get('heldout_rows_used_for_fitting')!=0
                    or reliability.get('panel_training_transcript_id_hash')!=transcript_id_hash(train)
                    or set(reliability['datasets'])!=set(ds)):
                    raise ValueError(f'{panel}: historical w_dt training-reference contract is not verified.')
                for d in ds: reliability['datasets'][d]['source_dataset_path']=paths[d]
            else:
                reliability=fit_panel_reliability_manifest(experiment_name=plan['design'],panel_name=panel,
                    panel_datasets=ds,dataset_mapping=paths,panel_training_ids=train,
                    validation_ids=val,test_ids=test,source_split_manifest=common_path)
            reliability['source_split_manifest']=str(common_path)
            reliability_path=partition_dir/f'{panel}_reliability_reference_manifest.json';write_json(reliability_path,reliability)
            saved_seed=int(configs[panel]['experiment']['seed'])
            seed=args.training_seed if args.training_seed is not None else saved_seed
            if args.mode=='prepare-existing' and seed!=saved_seed:
                raise ValueError('Existing-panel rerun must preserve its historical training seed.')
            training_seeds[panel]=seed
            pair={}
            for policy in ('equal','ranked'):
                task=destination/policy/panel;task.mkdir(parents=True)
                cfg=copy.deepcopy(configs[panel]);cfg.pop('defaults',None);cfg.pop('hydra',None)
                cfg['name']=f'{panel}_{policy}'
                cfg['experiment'].update(dataset=ds,seed=seed,from_checkpoint=False,resume_training_state=False,
                    resume_checkpoint_path=None,allow_weights_only_resume=False,train=True,predict=True)
                cfg['prediction'].update(checkpoint_variants=['best_val_loss'],sequence_only_shared_profile=True)
                cfg['split'].update(external_manifest=str(common_path),external_panel_name=panel,master_dataset_universe=ds)
                cfg['dataset_config']['dataset_path']=paths
                cfg['data']['reliability_reference_manifest']=str(reliability_path)
                cfg['data']['dataset_quality_ranking'].update(path=str(frozen_rank),dataset_column='dataset',
                    rank_column='quality_rank',strict=True)
                cfg['model']['gamma_centering']['reference'].update(dataset_names=None,
                    weighting='equal' if policy=='equal' else 'quality_rank',quality_rank_power=0. if policy=='equal' else 1.)
                cfg['trainer']['devices']=[0]
                cfg['paths'].update(sequences_path=str(sequences),checkpoints=str(task/'checkpoints'),logs=str(task/'logs'),results=str(task/'predictions'))
                cfg['orchestrator']={'experiment':plan['design'],'panel':panel,'run_name':cfg['name']}
                assert_training_contract(cfg)
                config_path=config_dir/f'{panel}_{policy}.yaml'
                config_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
                with initialize_config_dir(version_base=None,config_dir=str(config_dir)):
                    compose(config_name=config_path.stem)
                (task/'resolved_config.yaml').write_bytes(config_path.read_bytes())
                write_json(task/'split_manifest.json',dict(train_ids=train,validation_ids=val,test_ids=test))
                write_json(task/'reliability_reference_manifest.json',reliability)
                frame=table.loc[table.panel_id==panel].set_index('dataset_id')
                pi={d:float(frame.at[d,f'pi_{policy}']) for d in ds}
                write_json(task/'run_manifest.json',dict(selected_datasets=ds,checkpoint_selection='best_val_loss',
                    fixed_gamma_reference=dict(dataset_names=ds,weighting='equal' if policy=='equal' else 'quality_rank',
                        quality_rank_power=0. if policy=='equal' else 1.,pi=pi)))
                command=[sys.executable,'-u',str(snapshots/'main_ribounmix_multidataset.py'),
                    f'--config-path={config_dir}',f'--config-name={config_path.stem}',
                    f'hydra.run.dir={task / "hydra"}','hydra.job.chdir=false']
                tasks.append(dict(index=len(tasks),panel=panel,policy=policy,command=command,config=str(config_path)))
                pair[policy]=cfg
                before,after=_flatten_config(configs[panel]),_flatten_config(cfg)
                historical_differences.extend(dict(panel=panel,policy=policy,key=k,old=before.get(k),new=after.get(k))
                    for k in sorted(set(before)|set(after)) if before.get(k)!=after.get(k))
            differences.extend(dict(panel=panel,**d) for d in paired_config_differences(pair['equal'],pair['ranked']))
            _assert_only_gamma_reference_changed(reference_root=destination/'equal',panel_name=panel,resolved_config=pair['ranked'])
        for policy in ('equal','ranked'):
            # Build a new manifest: do not copy stale historical panel_1..4 or
            # stratification metadata into a newly optimized partition.
            policy_manifest=dict(manifest_version=1,experiment_name=plan['design'],
                panels=panels,panel_source_families=source_families,outer_run_identifier=destination.name+'_'+policy,
                number_of_panels=len(panels),panel_sizes={p:len(ds) for p,ds in panels.items()},
                panel_source_family_counts={p:len(f) for p,f in source_families.items()},
                retained_datasets=table.dataset_id.tolist(),all_candidate_datasets=table.dataset_id.tolist(),
                random_seed=next(iter(training_seeds.values())) if len(set(training_seeds.values()))==1 else None,
                training_seed_by_panel=training_seeds,source_historical_panel_manifest=str(args.panel_manifest),
                source_historical_panel_manifest_sha256=sha256(args.panel_manifest),
                source_family_partition=dict(atomic=True,identifier_column='source_family',number_of_source_families=table.source_family.nunique()),
                design_configuration=str(destination/'design_config.json') if args.mode=='prepare-rank-balanced' else 'Frozen existing memberships; no search',
                reliability_weight_mode='train-only-snr',prediction_checkpoint_variant='best_val_loss')
            if policy=='ranked':
                policy_manifest['gamma_reference_strategy']=dict(ranking_table=str(frozen_rank),ranking_table_sha256=sha256(frozen_rank),quality_rank_power=1.)
            write_json(destination/policy/'panel_manifest.json',policy_manifest)
            write_json(destination/policy/'common_split_manifest.json',common)
        contract=dict(paired_config_differences=differences,historical_config_differences=historical_differences,
            checkpoint_selection='best_val_loss',metrics=['PCC','Spearman','RMSE'],domains=['full_CDS','interior_trim20'],
            interior_renormalization=False,paired_finite_positions=True,bootstrap='paired transcript-cluster; all pairs and policies together',
            six_pairs_are_not_six_independent_training_replicates=True,
            analysis_command=shlex.join([sys.executable,str(snapshots/'analyses/compare_real_panel_weighting.py'),
                '--equal-root',str(destination/'equal'),'--ranked-root',str(destination/'ranked'),
                '--ranking-table',str(frozen_rank),'--output-dir',str(destination/'comparison'),'--require-all-panels']))
        write_json(destination/'comparison_contract.json',contract)
        plan.update(tasks=tasks,training_seeds=training_seeds,
            audit_manifest=str(args.output_root/'audit_manifest.json'),frozen_ranking_sha256=sha256(frozen_rank),
            original_ranking_sha256=sha256(args.ranking_table),gpu_visibility=args.gpus,dry_run_requested=args.dry_run)
        # Save the completed scientific work before the final audit. Retries
        # validate this state and finish; they never repeat the partition search.
        _atomic_json(destination/'preparation_state.json',dict(plan=plan,input_sha256=input_hashes,
            file_sha256={**_preparation_files(destination),**code_hashes},
            software=software_provenance(),environment_packages=environment))
        finish_preparation(destination,gpus=args.gpus)
    except (ValueError,KeyError,FileNotFoundError,RuntimeError,AssertionError,TypeError) as exc:
        plan.update(status='blocked_before_training',reason=str(exc),training_launched=False)
        write_json(destination/'rerun_plan.json',plan)
        raise
