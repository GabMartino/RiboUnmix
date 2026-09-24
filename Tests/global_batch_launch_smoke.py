"""End-to-end real-model smoke test using Lightning's production local launcher.

Run from the repository root, e.g.:
python -m Tests.global_batch_launch_smoke --accelerator cpu --devices 2 --output /tmp/ribo-smoke
python -m Tests.global_batch_launch_smoke --accelerator gpu --devices 2 --precision bf16-mixed --output ./results/global_batch_smoke

Uses synthetic arrays, both real GRUs, quality-weighted fixed-reference gamma,
all active loss terms, real training/validation/prediction methods and AdamW.
This is a correctness smoke test, not a representative speed benchmark.
"""
import argparse
from pathlib import Path

import lightning as pl
from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDatasetDataModule import TranscriptGroupedMultiDatasetBatchSampler
from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDataset import transcript_group_indices_from_ids
from Models.RiboUnmixLightningModule import RiboUnmixLightningModule
from Tests.test_alpha_causality_modes import _small_model
from Utils.global_batch_strategy import GlobalBatchStrategy


class Profiles(Dataset):
    def __len__(self):
        return 14  # seven transcripts, each with two datasets

    def __getitem__(self, token):
        index = token[1]
        metadata = dict(zip(('logical_batch_index', 'execution_chunk_index', 'execution_chunk_count',
                             'logical_group_count', 'execution_group_count', 'logical_pair_count'), token[2:]))
        if index == -1:
            return {'empty_execution': metadata}
        t, d = divmod(index, 2)
        rng = torch.Generator().manual_seed(123 + t)
        features = torch.randn(12+t, 5, generator=rng)
        codons = torch.randint(0, 8, (12+t,), generator=rng)
        counts = torch.poisson(torch.full((2, 12+t), 2.+d), generator=rng)
        return t, d, features, codons, counts, metadata

    @staticmethod
    def collate_fn(rows):
        if isinstance(rows[0], dict):
            return rows[0]
        tids = [f't{r[0]}' for r in rows]
        ds = torch.tensor([r[1] for r in rows])
        lengths = torch.tensor([len(r[2]) for r in rows])
        canonical = [tids.index(t) for t in dict.fromkeys(tids)]
        features = pad_sequence([rows[i][2] for i in canonical], batch_first=True)
        packed = pack_padded_sequence(features, lengths[canonical], batch_first=True, enforce_sorted=False)
        codons = pad_sequence([r[3] for r in rows], batch_first=True)
        replicas = pad_sequence([r[4].T for r in rows], batch_first=True).transpose(1, 2)
        target = replicas.mean(1)
        mask = torch.arange(target.shape[1])[None, :] < lengths[:, None]
        return (ds, tids, packed, target, lengths, mask, codons, torch.zeros_like(target),
                1.+ds.float()/3, ds+1, 1./(ds.float()+1), transcript_group_indices_from_ids(tids),
                replicas, torch.ones(len(rows), 2, dtype=torch.bool), rows[0][-1])


class ProfileModule(RiboUnmixLightningModule):
    def __init__(self, output):
        config = OmegaConf.load(Path(__file__).resolve().parents[1]/'config/config_ribounmix_multidataset.yaml')
        config.training.execution_microbatching.distributed_mode = 'global_batch'
        model = _small_model('learned')
        model.selected_dataset_names = ('d0', 'd1')
        model.selected_dataset_ids = (0, 1)
        model._configure_gamma_centering(
            {'gamma_centering': {'mode': 'fixed_reference', 'reference': {
                'weighting': 'quality_rank', 'quality_rank_power': 1., 'chunk_size': 2}}},
            reference_dataset_names=('d0', 'd1'), reference_dataset_ids=(0, 1),
            reference_dataset_quality_weights=(1., .5))
        super().__init__(model, config, {'d0': 0, 'd1': 1})
        self.output = output
        self._grouped_optimizer_batch_plan = {'resolved_accumulate_grad_batches': 3}

    def train_dataloader(self):
        sampler = TranscriptGroupedMultiDatasetBatchSampler(
            flat_transcript_ids=np.repeat([f't{i}' for i in range(7)], 2),
            flat_dataset_ids=np.tile([0, 1], 7), lengths=np.repeat(np.arange(12, 19), 2),
            batch_size=4, seed=42, shuffle_batches=False,
            global_batch_distributed=True, num_replicas=self.trainer.world_size, rank=self.global_rank,
            execution_microbatch_max_transcript_groups=1)
        return DataLoader(Profiles(), batch_sampler=sampler, collate_fn=Profiles.collate_fn)

    val_dataloader = train_dataloader
    predict_dataloader = train_dataloader

    def on_train_end(self):
        torch.save(dict(parameters={n: p.detach().cpu() for n, p in self.named_parameters()},
                        val_loss=float(self.trainer.callback_metrics['val_loss']),
                        steps=self.global_step), self.output/f'rank{self.global_rank}.pt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accelerator', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--devices', type=int, default=2)
    parser.add_argument('--precision', default='32-true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    pl.seed_everything(42)
    model = ProfileModule(args.output)
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, precision=args.precision,
        strategy=GlobalBatchStrategy(cluster_environment=LightningEnvironment()),
        max_epochs=2, logger=False, enable_progress_bar=False, enable_model_summary=False,
        num_sanity_val_steps=0, use_distributed_sampler=False, default_root_dir=str(args.output),
        callbacks=[ModelCheckpoint(monitor='val_loss', save_last=True)])
    trainer.fit(model)
    predictions = trainer.predict(model)
    ids = [(t, int(d)) for chunk in predictions if chunk is not None
           for t, d in zip(chunk['ids'], chunk['dataset_id'])]
    torch.save(ids, args.output/f'prediction_ids_rank{trainer.global_rank}.pt')
    trainer.strategy.barrier()
    if trainer.is_global_zero:
        states = [torch.load(args.output/f'rank{r}.pt', weights_only=False) for r in range(args.devices)]
        for state in states:
            assert np.isfinite(state['val_loss']) and state['steps'] == 2, state
            for name, value in state['parameters'].items():
                assert torch.isfinite(value).all(), name
                torch.testing.assert_close(value, states[0]['parameters'][name], atol=0, rtol=0)
            assert state['val_loss'] == states[0]['val_loss']
        ids = [pair for r in range(args.devices)
               for pair in torch.load(args.output/f'prediction_ids_rank{r}.pt', weights_only=False)]
        assert sorted(ids) == [(f't{t}', d) for t in range(7) for d in (0, 1)], ids
        print(f'PASS: {args.devices} {args.accelerator} processes, {args.precision}; '
              f'identical rank parameters, 2 optimizer steps, 14 distinct predictions; '
              f'val_loss={states[0]["val_loss"]:.8f}')


if __name__ == '__main__':
    main()
