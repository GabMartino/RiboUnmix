"""Real multiprocess CPU/Gloo tests; no multi-GPU hardware is assumed."""
import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import lightning as pl
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset, DataLoader

from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDatasetDataModule import (
    TranscriptGroupedMultiDatasetBatchSampler as Sampler,
)
from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDataset import RiboUnmixMultiDataset
from Models.RiboUnmixLightningModule import RiboUnmixLightningModule, reduce_per_sample_quantity
from Utils.global_batch import empty_execution_metadata
from Utils.global_batch_strategy import GlobalBatchStrategy, METRICS


def fixture():
    tids, datasets = [], []
    for i in range(13):
        for d in range(2 + (i % 2)):
            tids.append(f't{i:03}')
            datasets.append(d)
    return np.asarray(tids), np.asarray(datasets)


def sampler(world, rank, global_mode=True):
    tids, ds = fixture()
    return Sampler(flat_transcript_ids=tids, flat_dataset_ids=ds, lengths=np.full(len(ds), 10),
                   batch_size=4, seed=42, sort_by_length=False, shuffle_batches=False,
                   num_replicas=world, rank=rank, global_batch_distributed=global_mode,
                   execution_microbatch_max_pair_rows=7)


class TinyDataset(Dataset):
    def __len__(self):
        return len(fixture()[0])

    def __getitem__(self, token):
        metadata = dict(zip(('logical_batch_index', 'execution_chunk_index', 'execution_chunk_count',
                             'logical_group_count', 'execution_group_count', 'logical_pair_count'), token[2:]))
        if token[1] == -1:
            return {'empty_execution': metadata}
        return token[1], metadata

    @staticmethod
    def collate_fn(rows):
        if isinstance(rows[0], dict):
            return rows[0]
        return torch.tensor([r[0] for r in rows]), rows[0][1]


class TinyModule(RiboUnmixLightningModule):
    """Exercise production training/epoch/optimizer methods with a tiny loss."""
    def __init__(self, output, accumulation=1):
        pl.LightningModule.__init__(self)
        self.weight = torch.nn.Parameter(torch.tensor(.7, dtype=torch.float64))
        self.unused = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))
        self.output = Path(output)
        self.automatic_optimization = False
        self.global_batch_distributed = True
        self.execution_microbatching_enabled = True
        self.execution_gradient_clip_val = .4
        self.execution_gradient_clip_algorithm = 'norm'
        self._grouped_optimizer_batch_plan = {'resolved_accumulate_grad_batches': accumulation}
        self._execution_logical_batches_since_step = 0
        self._execution_optimizer_steps = 0
        self._execution_has_pending_gradients = False
        self._grouped_batch_logging_enabled = False
        self._global_epoch_totals = {}
        self.log_validation_transcript_mu_pcc_distribution = False
        self._synthetic_ground_truth = None
        self.config = OmegaConf.create({'optim': {'scheduler': {'monitor': 'val_loss'}}})
        self.updates = []

    def _forward_batch(self, batch):
        indices, metadata = batch
        tids, ds = fixture()
        indices = indices.cpu().numpy()
        group_ids = torch.tensor([int(t[1:]) for t in tids[indices]], device=self.device)
        ds = torch.tensor(ds[indices], device=self.device)
        x = group_ids.to(torch.float64) / 7 + ds.to(torch.float64) / 11 + 1
        target = torch.ones_like(x)
        return dict(x=x, target=target, dataset_ids=ds, transcript_group_index=group_ids,
                    weights=1.+ds.to(torch.float64)/3, ids=list(tids[indices]),
                    lengths=torch.ones_like(ds), execution_microbatch_metadata=metadata)

    def _compute_loss_and_metrics(self, out, **kwargs):
        loss = reduce_per_sample_quantity((self.weight*out['x']-out['target']).square(),
            out['weights'], out['dataset_ids'], out['transcript_group_index'], 'transcript_balanced')
        return {key: loss if key != 'mu_pcc_unweighted' else out['x'].mean() for key in METRICS}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=.01, weight_decay=.1)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=0, factor=.5)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sched, 'monitor': 'val_loss'}}

    def on_before_optimizer_step(self, optimizer):
        super().on_before_optimizer_step(optimizer)
        self.updates.append(float(self.weight.grad))

    def train_dataloader(self):
        return DataLoader(TinyDataset(), batch_sampler=sampler(self.trainer.world_size, self.global_rank),
                          collate_fn=TinyDataset.collate_fn, num_workers=0)

    def val_dataloader(self):
        return self.train_dataloader()

    def on_train_end(self):
        state = dict(weight=float(self.weight.detach()), unused=float(self.unused.detach()),
                     gradients=self.updates, steps=self.global_step,
                     val_loss=float(self.trainer.callback_metrics['val_loss']),
                     lr=self.optimizers().param_groups[0]['lr'],
                     optimizer=self.optimizers().state_dict(),
                     scheduler=self.lr_schedulers().state_dict())
        torch.save(state, self.output/f'rank{self.global_rank}.pt')
        self.trainer.save_checkpoint(self.output/'epoch_boundary.ckpt')


def run_training(output, world, accumulation=1, max_epochs=2, ckpt_path=None):
    torch.set_num_threads(1)
    model = TinyModule(output, accumulation)
    # Keep CPU/Gloo tests independent of the workstation GPU. Spawn (not fork)
    # also permits comparing with an independent autograd run in this process.
    with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
        trainer = pl.Trainer(accelerator='cpu', devices=world,
            strategy=GlobalBatchStrategy(start_method='spawn', process_group_backend='gloo'),
            precision='64-true', max_epochs=max_epochs, logger=False, enable_progress_bar=False,
            enable_model_summary=False, default_root_dir=str(output), num_sanity_val_steps=0,
            use_distributed_sampler=False)
        trainer.fit(model, ckpt_path=ckpt_path)
    return [torch.load(Path(output)/f'rank{i}.pt', weights_only=False) for i in range(world)]


def monolithic_reference(accumulation):
    """Independent full-logical-batch objective, without distributed helpers."""
    tids, datasets = fixture()
    weight = torch.nn.Parameter(torch.tensor(.7, dtype=torch.float64))
    opt = torch.optim.AdamW([weight], lr=.01, weight_decay=.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=0, factor=.5)
    logical = {}
    for chunk in sampler(1, 0, False):
        logical.setdefault(chunk[0][2], []).extend(row[1] for row in chunk)

    def losses(indices):
        values = []
        for tid in sorted(set(tids[indices])):
            ds = torch.tensor(datasets[[i for i in indices if tids[i] == tid]], dtype=torch.float64)
            x = int(tid[1:]) / 7 + ds / 11 + 1
            reliability = 1 + ds / 3
            values.append(((weight*x - 1).square()*reliability).sum()/reliability.sum())
        return torch.stack(values)

    gradients = []
    for epoch in range(2):
        for i, indices in enumerate(logical.values()):
            (losses(indices).mean()/accumulation).backward()
            if (i+1) % accumulation == 0 or i+1 == len(logical):
                torch.nn.utils.clip_grad_norm_([weight], .4)
                gradients.append(float(weight.grad))
                opt.step()
                opt.zero_grad()
        val_loss = float(losses(list(range(len(tids)))).mean().detach())
        scheduler.step(val_loss)
    return dict(weight=float(weight.detach()), gradients=gradients, val_loss=val_loss)


class GlobalBatchTests(unittest.TestCase):
    def test_global_membership_and_idle_slots(self):
        reference = list(sampler(1, 0, False))
        expected = {}
        for chunk in reference:
            expected.setdefault(chunk[0][2], []).extend(t[1] for t in chunk)
        for world in (1, 2, 3, 8):
            plans = [list(sampler(world, r)) for r in range(world)]
            self.assertEqual(len({len(p) for p in plans}), 1)
            actual = {}
            for slots in zip(*plans):
                self.assertEqual(len({(s[0][2], s[0][3], s[0][4]) for s in slots}), 1)
                for chunk in slots:
                    actual.setdefault(chunk[0][2], []).extend(t[1] for t in chunk if t[1] != -1)
            self.assertEqual({k: sorted(v) for k,v in expected.items()}, {k: sorted(v) for k,v in actual.items()})

    def test_empty_token_never_reads_a_real_sample(self):
        dataset = object.__new__(RiboUnmixMultiDataset)
        token = ('execution_microbatch_v1', -1, 0, 0, 1, 1, 0, 2)
        sample = dataset[token]
        batch = dataset.collate_fn([sample])
        self.assertEqual(empty_execution_metadata(batch)['execution_group_count'], 0)

    def test_multiprocess_updates_metrics_and_unused_parameter(self):
        with tempfile.TemporaryDirectory() as temporary:
            for accumulation in (1, 3):  # includes an incomplete final window
                results = []
                for world in (1, 2, 3):
                    output = Path(temporary)/f'a{accumulation}_w{world}'
                    output.mkdir()
                    states = run_training(output, world, accumulation)
                    for s in states[1:]:
                        self.assertEqual(s['gradients'], states[0]['gradients'])
                        self.assertEqual(s['weight'], states[0]['weight'])
                        self.assertEqual(s['val_loss'], states[0]['val_loss'])
                    results.append(states[0])
                for state in results[1:]:
                    self.assertAlmostEqual(state['weight'], results[0]['weight'], places=12)
                    self.assertAlmostEqual(state['val_loss'], results[0]['val_loss'], places=12)
                    np.testing.assert_allclose(state['gradients'], results[0]['gradients'], atol=1e-12, rtol=1e-12)
                    self.assertEqual(state['steps'], results[0]['steps'])
                    self.assertEqual(state['lr'], results[0]['lr'])
                    self.assertEqual(state['unused'], 2.)
                    self.assertEqual(len(state['optimizer']['state']), 1)
                reference = monolithic_reference(accumulation)
                self.assertAlmostEqual(results[0]['weight'], reference['weight'], places=12)
                self.assertAlmostEqual(results[0]['val_loss'], reference['val_loss'], places=12)
                np.testing.assert_allclose(results[0]['gradients'], reference['gradients'], atol=1e-12, rtol=1e-12)

    def test_epoch_checkpoint_can_change_world_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)/'base'
            base.mkdir()
            reference = run_training(base, 1, accumulation=3)[0]
            for before, after in ((1, 2), (2, 1)):
                first, second = (Path(temporary)/f'{before}_{after}_{suffix}' for suffix in ('first', 'second'))
                first.mkdir()
                second.mkdir()
                run_training(first, before, accumulation=3, max_epochs=1)
                state = run_training(second, after, accumulation=3, ckpt_path=first/'epoch_boundary.ckpt')[0]
                for key in ('weight', 'val_loss', 'lr'):
                    self.assertAlmostEqual(state[key], reference[key], places=12)
                self.assertEqual(state['steps'], reference['steps'])
                for key in ('best', '_last_lr', 'num_bad_epochs', 'last_epoch'):
                    np.testing.assert_allclose(state['scheduler'][key], reference['scheduler'][key], atol=1e-12, rtol=1e-12)
                for p, params in state['optimizer']['state'].items():
                    for key, value in params.items():
                        torch.testing.assert_close(value, reference['optimizer']['state'][p][key], atol=1e-12, rtol=1e-12)

    def test_planner_keeps_global_accumulation_and_batch_size(self):
        from main_ribounmix_multidataset import resolve_training_grouped_optimizer_batching
        from Tests.test_grouped_optimizer_batching import _statistics, _PreviewDatamodule
        for world in (1, 2, 4):
            cfg = OmegaConf.create(dict(
                data=dict(batch_size=32, train_sampling_strategy='transcript_grouped_multidataset_pairs'),
                training=dict(execution_microbatching=dict(enabled=True, distributed_mode='global_batch'),
                              grouped_optimizer_batch=dict(enabled=True, target_unique_transcripts_per_optimizer_step=32)),
                trainer=dict(devices=list(range(world)), accumulate_grad_batches=1)))
            _, plan = resolve_training_grouped_optimizer_batching(
                cfg=cfg, datamodule=_PreviewDatamodule(_statistics((4, 4, 2))))
            self.assertEqual(plan.resolved_accumulate_grad_batches, 8)
            self.assertEqual(plan.estimated_global_unique_transcripts_per_optimizer_step, 32)
            self.assertEqual(plan.microbatches_per_epoch_per_rank, 3)
            self.assertEqual(cfg.data.batch_size, 32)
            self.assertEqual(cfg.trainer.accumulate_grad_batches, 1)  # manual windows

    def test_slurm_local_children_do_not_write_rank_zero_artifacts(self):
        from main_ribounmix_multidataset import env_global_rank
        with patch.dict(os.environ, {'LOCAL_RANK': '1', 'SLURM_PROCID': '0', 'SLURM_NTASKS': '1'}, clear=True):
            self.assertEqual(env_global_rank(), 1)

    def test_resume_override_does_not_divide_saved_batch_or_change_objective(self):
        from Tests.test_exp8_parallel_resume import Exp8ParallelResumeTests, RESUME
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task, run_id = Exp8ParallelResumeTests._write_launcher(root, 'N040/pair02_A')
            original = (task/'launch_command.sh').read_text()
            args = RESUME.parse_args(['--run-root', temporary, '--gpus', 'inherit',
                                      '--global-batch-gpus', '2', '--dry-run'])
            prepared = RESUME._prepare_task(args, root, task/'launch_command.sh')
            command = prepared['command']
            self.assertEqual(prepared['global_batch_gpus_override'], 2)
            self.assertIn('trainer.devices=[0, 1]', command)
            for value in ('data.batch_size=32', 'loss.experiment_mode=standard_nb',
                          'model.gamma_centering.reference.weighting=equal',
                          'training.grouped_optimizer_batch.target_unique_transcripts_per_optimizer_step=32',
                          '++training.execution_microbatching.distributed_mode=global_batch'):
                self.assertIn(value, command)
            self.assertEqual((task/'launch_command.sh').read_text(), original)

    def test_univie_launcher_dry_run_composes_saved_config_for_two_gpus(self):
        from Tests.test_exp8_parallel_resume import Exp8ParallelResumeTests, ROOT
        with tempfile.TemporaryDirectory() as temporary:
            task, run_id = Exp8ParallelResumeTests._write_launcher(Path(temporary), 'N040/pair02_A')
            cfg = OmegaConf.load(ROOT/'config/config_ribounmix_multidataset.yaml')
            cfg.dataset_config = {'dataset_path': {'d0': 'dummy.parquet'}}
            OmegaConf.save(cfg, task/'resolved_config.yaml')
            env = os.environ | dict(RUN_ROOT=temporary, RUN_ID=run_id, PROJECT_DIR=str(ROOT),
                UNIVIE_VENV_PATH=str(Path(sys.executable).absolute().parent.parent),
                GLOBAL_BATCH_GPUS='2', DRY_RUN='1', SLURM_JOB_ID='test_global_batch')
            shell = 'module() { :; }; srun() { shift; "$@"; }; export -f module srun; bash "$1"'
            result = subprocess.run(['bash', '-c', shell, 'smoke', str(ROOT/'resume_real_global_batch_univie.slurm')],
                                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=40)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            audit = json.loads((task/'resume_manifest.json').read_text())
            self.assertTrue(audit['saved_config']['hydra_composition_checked'])
            self.assertEqual(audit['global_batch_gpus_override'], 2)
            self.assertIn('data.batch_size=32', audit['command'])


if __name__ == '__main__':
    unittest.main()
