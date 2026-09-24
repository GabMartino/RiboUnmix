"""Frozen campaign configurations and one-process production execution.

No training loop lives here. Preparation is CPU-only; each authorized array
element executes one immutable production configuration on its allocated GPU.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from Utils.panel_reference_audit import ROOT, read_json, sha256
from Utils.panel_reference_preparation import assert_training_contract, relocated_input
from Utils.real_panel_convergence import write_json
from run_real_independent_panel_convergence_quality_rank import _flatten_config


def parameter_hash(model, prefix=''):
    """Hash named FP32 initial parameters, excluding reference and other buffers."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if name.startswith(prefix):
            value = parameter.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str((tuple(value.shape), str(value.dtype))).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def initialization_audit(cfg):
    """Use the production constructor/reference resolver, not a second model."""
    import lightning as pl
    import torch
    from torch.nn.utils.rnn import pack_padded_sequence
    from Models.RiboUnmixModel import RiboUnmixModel
    from Models.RiboUnmixLightningModule import NegativeBinomialProfileLoss
    from main_ribounmix_multidataset import resolve_gamma_reference_panel

    pl.seed_everything(int(cfg['experiment']['seed']), workers=True)
    encoding = {k: yaml.safe_load(Path(v).read_text()) for k, v in cfg['paths']['encodings'].items()}
    reference = resolve_gamma_reference_panel(cfg=OmegaConf.create(cfg),
        experiment_datasets=cfg['experiment']['dataset'], dataset_encoding=encoding['datasets'])
    model = RiboUnmixModel(model_configs=cfg['model'], eps=float(cfg['model'].get('eps', 1e-8)),
        selected_dataset_names=reference['selected_names'], selected_dataset_ids=reference['selected_ids'],
        reference_dataset_names=reference['reference_names'], reference_dataset_ids=reference['reference_ids'],
        reference_dataset_quality_weights=reference['reference_quality'],
        nt_encoding=encoding['nt'], codon_encoding=encoding['codon'],
        codon_to_aa_encoding=encoding['codon_to_aa'], aa_encoding=encoding['aa'])
    result = dict(parameter_sha256=parameter_hash(model),
        shared_parameter_sha256=parameter_hash(model, 'biological_model.'),
        reference_buffer_sha256=hashlib.sha256(model.gamma_reference_weights.numpy().tobytes()).hexdigest(),
        initialization_check='Production constructor on CPU; runtime rechecks parameter hash before checkpoint load',
        gradient_probe='Synthetic 8-codon batch, not a training epoch or performance measurement')
    model.eval()
    # This bounded probe checks connectivity only. It neither uses held-out
    # observations nor tunes architecture/optimization using a successful result.
    x = torch.linspace(-1, 1, 8 * model.biological_model.input_size).reshape(1, 8, -1)
    target = torch.tensor([[1., 2., 3., 1., 4., 2., 1., 3.]])
    mask = torch.ones(1, 8, dtype=torch.bool)
    _, alpha, extra = model(x_packed=pack_padded_sequence(x, [8], batch_first=True),
        mask=mask, codon_ids=torch.tensor([[encoding['codon']['AAA']] * 8]),
        id_datasets=torch.tensor([reference['selected_ids'][0]]), target=target,
        sample_ids=['initialization_probe'], transcript_group_index=torch.tensor([0]))
    loss = NegativeBinomialProfileLoss(experiment_mode='standard_nb', nb_mean_gradient_beta=0)(
        mu_phys=None, log_mu_phys=extra['log_mu'], log_sigma=alpha, y_true=target, mask=mask)
    loss.backward()
    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise ValueError('Nonfinite gradient in production initialization probe.')
    for label, module in [('shared_encoder', model.biological_model),
                          ('alpha_head', model.dataset_bias_model.log_sigma_head),
                          ('dataset_context', model.dataset_bias_model.local_context_gru)]:
        result[f'{label}_has_gradient'] = any(p.grad is not None and bool((p.grad != 0).any()) for p in module.parameters())
    if cfg['model']['mean_correction'] == 'unity':
        if not torch.equal(extra['gamma'], torch.ones_like(extra['gamma'])):
            raise ValueError('gamma=1 intervention failed.')
        if result['dataset_context_has_gradient'] or not result['shared_encoder_has_gradient'] or not result['alpha_head_has_gradient']:
            raise ValueError('Unexpected gamma=1 gradient paths; do not silently reconnect detached features.')
    return result


def assert_campaign_pair(equal, other):
    a, b = _flatten_config(equal), _flatten_config(other)
    exact = {'name', 'model.mean_correction', 'model.gamma_centering.reference.weighting',
             'model.gamma_centering.reference.quality_rank_power'}
    prefixes = ('paths.checkpoints', 'paths.logs', 'paths.results', 'orchestrator.',
                'model.gamma_centering.reference.explicit_weights')
    differences = []
    for key in sorted(set(a) | set(b)):
        if a.get(key) != b.get(key):
            if key not in exact and not key.startswith(prefixes):
                raise ValueError(f'Unapproved within-seed arm difference: {key}')
            differences.append(dict(key=key, equal=a.get(key), other=b.get(key)))
    return differences


def prepare_tasks(root, matrix, preparations, permutations):
    """Expand the audited two-policy templates; reuse each panel's fitted w_dt."""
    config_dir = root / 'resolved_configs'
    config_dir.mkdir()
    tasks, audits, differences, baselines = [], [], [], {}
    for index, row in matrix.iterrows():
        partition, panel, arm, seed = row.partition, row.panel, row.arm, int(row.training_seed)
        prepared = preparations[partition]
        template = prepared / 'equal' / panel
        cfg = yaml.safe_load((template / 'resolved_config.yaml').read_text())
        cfg['name'] = row.task_id
        cfg['experiment']['seed'] = seed
        cfg['model']['mean_correction'] = 'unity' if arm == 'shared_only' else 'learned'
        if cfg['model']['alpha_mode'] != 'learned' or float(cfg['loss']['gamma_reg_weight']) != 0:
            raise ValueError('This campaign requires the resolved learned-alpha, zero-gamma-penalty contract.')
        ref = cfg['model']['gamma_centering']['reference']
        ref.pop('explicit_weights', None)
        ref.update(weighting='quality_rank' if arm == 'ranked' else 'equal',
                   quality_rank_power=1. if arm == 'ranked' else 0., dataset_names=None)
        if arm.startswith('shuffled_'):
            mapping = permutations[(permutations.panel_id == panel) &
                                   (permutations.permutation_id == int(row.permutation_id))]
            ref.update(weighting='explicit', quality_rank_power=1.,
                       explicit_weights=dict(zip(mapping.dataset_id, mapping.assigned_q)))
            if set(ref['explicit_weights']) != set(cfg['experiment']['dataset']):
                raise ValueError('Permutation identities do not match selected panel.')
        task_root = root / 'runs' / partition / arm / f'seed{seed}' / panel
        task_root.mkdir(parents=True)
        for key in ('checkpoints', 'logs'):
            cfg['paths'][key] = str(task_root / key)
        cfg['paths']['results'] = str(task_root / 'predictions')
        for key, value in cfg['paths']['encodings'].items():
            source = relocated_input(value)
            cfg['paths']['encodings'][key] = str(root / 'code_snapshot' / source.relative_to(ROOT))
        cfg['orchestrator'] = dict(experiment='reproducibility_reference_campaign', partition=partition,
            panel=panel, arm=arm, training_seed=seed,
            permutation_id=int(row.permutation_id) if pd.notna(row.permutation_id) else None)
        assert_training_contract(cfg)
        check = initialization_audit(cfg)
        key = (partition, panel, seed)
        if arm == 'equal':
            baselines[key] = (copy.deepcopy(cfg), check)
        else:
            baseline, initial = baselines[key]
            if check['parameter_sha256'] != initial['parameter_sha256']:
                raise ValueError(f'{row.task_id}: initial parameters differ from equal arm.')
            differences.extend(dict(task_id=row.task_id, **d) for d in assert_campaign_pair(baseline, cfg))
        cfg['orchestrator']['initialization_sha256'] = check['parameter_sha256']
        cfg['orchestrator']['task_contract_sha256'] = hashlib.sha256(
            json.dumps(cfg, sort_keys=True, allow_nan=False).encode()).hexdigest()
        audits.append(dict(task_id=row.task_id, **check))
        config = config_dir / f'{row.task_id}.yaml'
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            compose(config_name=config.stem)
        for name in ('split_manifest.json', 'reliability_reference_manifest.json'):
            (task_root / name).write_bytes((template / name).read_bytes())
        (task_root / 'resolved_config.yaml').write_bytes(config.read_bytes())
        command = [sys.executable, '-u', str(root / 'code_snapshot/main_ribounmix_multidataset.py'),
            f'--config-path={config_dir}', f'--config-name={config.stem}',
            f'hydra.run.dir={task_root / "hydra"}', 'hydra.job.chdir=false']
        tasks.append(dict(index=int(index), task_id=row.task_id, partition=partition, panel=panel,
            arm=arm, training_seed=seed, milestone=int(row.milestone), root=str(task_root),
            config=str(config), config_sha256=sha256(config), command=command,
            reliability_sha256=sha256(Path(cfg['data']['reliability_reference_manifest'])),
            split_sha256=sha256(Path(cfg['split']['external_manifest']))))
    pd.DataFrame(audits).to_csv(root / 'initialization_and_gradient_audit.csv', index=False)
    write_json(root / 'approved_configuration_differences.json', differences)
    return tasks


def sequence_split_audit(root, preparations, tasks):
    """Record exact CDS identities and cross-split duplicates; never redraw IDs."""
    import pyarrow.parquet as pq
    cfg = yaml.safe_load(Path(tasks[0]['config']).read_text())
    path = cfg['paths']['sequences_path']
    schema = set(pq.read_schema(path).names)
    column = 'codons' if 'codons' in schema else 'ref'
    columns = ['transcript_id', column] + (['gene_id'] if 'gene_id' in schema else [])
    sequences = pd.read_parquet(path, columns=columns)
    rows = []
    for record in sequences.to_dict('records'):
        codons = list(map(str, record[column]))
        rows.append(dict(transcript_id=str(record['transcript_id']), length=len(codons),
            coordinate_sha256=hashlib.sha256(json.dumps(codons, separators=(',', ':')).encode()).hexdigest(),
            gene_id=record.get('gene_id')))
    coordinates = pd.DataFrame(rows)
    coordinates.to_csv(root / 'sequence_coordinates.csv', index=False)
    duplicates, common_training = [], {}
    for partition, directory in preparations.items():
        split = read_json(directory / 'partition_and_split_manifests/common_split_manifest.json')
        val, test = set(split['common_validation_ids']), set(split['common_test_ids'])
        train_sets = [set(ids) for ids in split['panel_train_eligible_ids'].values()]
        train = set.union(*train_sets)
        if train & (val | test) or val & test:
            raise ValueError('Frozen transcript IDs overlap across splits.')
        common_training[partition] = sorted(set.intersection(*train_sets))
        table = coordinates.copy()
        table['fold'] = table.transcript_id.map(lambda t: 'validation' if t in val else 'test' if t in test else 'training' if t in train else 'excluded')
        table = table[table.fold != 'excluded']
        for kind in ('coordinate_sha256', 'gene_id'):
            for identity, group in table.dropna(subset=[kind]).groupby(kind):
                if group.fold.nunique() > 1:
                    duplicates.append(dict(partition=partition, identity_type=kind, identity=identity,
                        transcript_ids=json.dumps(group.transcript_id.tolist()), folds=json.dumps(group.fold.tolist())))
    pd.DataFrame(duplicates, columns=['partition','identity_type','identity','transcript_ids','folds']).to_csv(
        root / 'cross_split_sequence_gene_duplicates.csv', index=False)
    write_json(root / 'common_training_intersection.json', common_training)
    write_json(root / 'sequence_split_audit.json', dict(sequence_file_sha256=sha256(path),
        cross_split_duplicate_groups=len(duplicates), gene_mapping='gene_id in sequence artifact' if 'gene_id' in schema else 'not supplied; not verified',
        action='Reported only; no split or transcript changed.',
        common_training_intersection_sizes={p:len(ids) for p,ids in common_training.items()}))


def verify_campaign_execution(root, manifest):
    from Utils.reference_campaign import object_hash
    plan = read_json(root / 'plan_definition.json')
    if object_hash(plan) != manifest['plan_hash']:
        raise ValueError('Campaign plan bytes/content changed since approval.')
    if sha256(root / 'frozen_execution.json') != plan['frozen_execution_sha256']:
        raise ValueError('Frozen execution contract changed.')
    frozen = read_json(root / 'frozen_execution.json')
    if platform.python_version() != frozen['software']['python']:
        raise ValueError('Python version differs from preparation. Prepare on the cluster before approval.')
    actual = {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
    if actual != frozen['environment_packages']:
        raise ValueError('Python environment differs from the frozen campaign; no implicit environment migration.')
    for path, expected in frozen['file_sha256'].items():
        if sha256(path) != expected:
            raise ValueError(f'Frozen campaign input changed: {path}')
    return plan


def selected_tasks(plan, cap, task_index=None):
    """Cap the admitted plan prefix, globally consistent across array elements."""
    tasks = plan['execution_tasks'][:cap]
    if task_index is not None:
        if not 0 <= task_index < len(tasks):
            raise ValueError('Task index lies outside the approved task cap.')
        return [tasks[task_index]]
    return tasks


def validate_predictions(task):
    from analyses.analyze_real_panel_convergence import _locate_panel_prediction, _extract_panel_profiles
    task_root = Path(task['root'])
    prediction, source, reason = _locate_panel_prediction(task_root)
    if prediction is None:
        raise ValueError(f'{task["task_id"]}: {reason}')
    expected = set(read_json(task_root / 'split_manifest.json')['test_ids'])
    profiles, _ = _extract_panel_profiles(panel_name=task['panel'], run_identifier=task['task_id'],
        prediction_path=prediction, expected_ids=expected, mean_one_tolerance=1e-4)
    coordinates = pd.read_csv(Path(task['config']).parent.parent / 'sequence_coordinates.csv').set_index('transcript_id')
    for transcript, profile in profiles.items():
        if profile['length'] != int(coordinates.at[transcript, 'length']):
            raise ValueError(f'{transcript}: exported length differs from the frozen CDS coordinates.')
    runtime = read_json(source)
    checkpoint = Path(runtime['best_val_loss']['checkpoint_path'])
    checkpoint_identity = sha256(checkpoint)
    return dict(prediction_path=str(prediction), prediction_sha256=sha256(prediction),
                source_manifest=str(source), source_manifest_sha256=sha256(source),
                checkpoint_path=str(checkpoint), checkpoint_sha256=checkpoint_identity,
                validated_test_transcripts=len(profiles), mean_one_tolerance=1e-4,
                validation_scope='Native L: frozen IDs, finite positive values, masks and duplicate-row consistency; factor diagnostics pending')


def run_task(task, plan_hash, *, resume=False):
    """Exclusive per-task lock; keep attempt history and require full-state resume."""
    root = Path(task['root'])
    with (root / 'execution.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Task already running: {task["task_id"]}') from exc
        state_path = root / 'execution_status.json'
        previous = read_json(state_path) if state_path.exists() else None
        if previous and previous['plan_hash'] != plan_hash:
            raise ValueError('Cannot resume a task belonging to a different frozen plan.')
        if previous and not resume:
            raise ValueError('Task has an earlier attempt; explicit --resume is required.')
        if previous and previous['status'] == 'completed_native_L_validated':
            validation = validate_predictions(task)
            if validation != previous['prediction_validation']:
                raise ValueError('Completed prediction artifacts changed after validation.')
            return previous
        command = list(task['command'])
        cfg = yaml.safe_load(Path(task['config']).read_text())
        checkpoint, checkpoint_info, rejected = None, None, []
        if previous:
            from resume_real_experiment_from_checkpoints import _select_resume_checkpoint
            checkpoint, checkpoint_info, rejected = _select_resume_checkpoint(root)
            if checkpoint is not None:
                if not checkpoint_info['has_optimizer_state'] or not checkpoint_info['has_scheduler_state']:
                    raise ValueError('Only full optimizer/scheduler-state resume is allowed.')
                import torch
                payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
                if payload.get('hyper_parameters', {}).get('campaign_task_contract_sha256') != cfg['orchestrator']['task_contract_sha256']:
                    raise ValueError('Checkpoint does not belong to this exact campaign task/configuration.')
                del payload
                command += ['experiment.from_checkpoint=true', 'experiment.resume_training_state=true',
                            f'experiment.resume_checkpoint_path={checkpoint}', 'experiment.allow_weights_only_resume=false']
            elif rejected:
                raise ValueError(f'No usable checkpoint; refusing a silent fresh restart: {rejected}')
        elif list((root / 'checkpoints').rglob('*.ckpt')):
            raise ValueError('Untracked checkpoints exist in a fresh campaign task; no historical initialization allowed.')
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError('One visible allocated CUDA GPU is required per campaign process.')
        if 'bf16' in str(cfg['trainer']['precision']) and not torch.cuda.is_bf16_supported(including_emulation=False):
            raise ValueError('Allocated GPU lacks native BF16; request a compatible node, do not change precision.')
        attempt_number = 1 + (len(previous['attempts']) if previous else 0)
        attempt = dict(number=attempt_number, command=command, started_unix=time.time(),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu=torch.cuda.get_device_name(0),
            checkpoint=checkpoint_info, rejected_checkpoints=rejected,
            checkpoint_sha256=sha256(checkpoint) if checkpoint else None)
        state = dict(task_id=task['task_id'], plan_hash=plan_hash, status='running',
                     attempts=[*(previous['attempts'] if previous else []), attempt])
        write_json(state_path, state)
        log_path = root / f'attempt_{attempt_number:03d}.log'
        try:
            with log_path.open('x') as log:
                result = subprocess.run(command, cwd=Path(command[2]).parent, stdout=log, stderr=subprocess.STDOUT)
            attempt.update(exit_code=result.returncode, ended_unix=time.time(), log=str(log_path))
            if result.returncode:
                raise RuntimeError(f'Production task exited {result.returncode}; see {log_path}')
            state['prediction_validation'] = validate_predictions(task)
            state['status'] = 'completed_native_L_validated'
        except Exception as exc:
            state.update(status='failed', reason=str(exc))
            attempt.setdefault('ended_unix', time.time())
            write_json(state_path, state)
            raise
        write_json(state_path, state)
        return state
