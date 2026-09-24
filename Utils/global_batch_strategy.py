"""Synchronous data parallelism with reduction at global optimizer boundaries.

No DDP forward/backward wrapper: local chunks may be uneven or absent. Each
rank accumulates its contribution to the global objective, then SUM-reduces
gradients once before clipping/Adam. Hence losses must NOT contain a world-size
multiplier (unlike DDP's default averaged-gradient convention).
"""
import torch
import torch.distributed as dist
from lightning.pytorch.strategies import DDPStrategy


class GlobalBatchStrategy(DDPStrategy):
    def configure_ddp(self):
        if self.lightning_module.automatic_optimization:
            raise ValueError('GlobalBatchStrategy requires manual execution-microbatch optimization.')
        # Lightning provides launch, process groups, precision and checkpoint
        # restoration; only model wrapping/reduction differs from standard DDP.
        for tensor in (*self.model.parameters(), *self.model.buffers()):
            dist.broadcast(tensor.detach(), src=0)


@torch.no_grad()
def sum_global_gradients(parameters, bucket_bytes=16 * 1024 * 1024):
    """Sum accumulated gradients, retaining None for globally unused parameters.

    A rank without a local observation contributes zero, not a repeated sample.
    Keeping globally unused gradients as None preserves AdamW's skip semantics.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    params = [p for p in parameters if p.requires_grad]
    used = torch.tensor([p.grad is not None for p in params], device=params[0].device, dtype=torch.int32)
    dist.all_reduce(used, op=dist.ReduceOp.SUM)
    active = [p for p, n in zip(params, used.tolist()) if n]
    bucket, size = [], 0

    def reduce_bucket():
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in bucket])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        offset = 0
        for p in bucket:
            value = flat[offset:offset + p.numel()].view_as(p)
            if p.grad is None:
                p.grad = value.clone()
            else:
                p.grad.copy_(value)
            offset += p.numel()

    for p in active:
        nbytes = p.numel() * p.element_size()
        if bucket and (size + nbytes > bucket_bytes or p.dtype != bucket[0].dtype):
            reduce_bucket()
            bucket, size = [], 0
        bucket.append(p)
        size += nbytes
    if bucket:
        reduce_bucket()


# These are the primary loss/likelihood diagnostics and checkpoint metrics.
# Each column carries its own denominator, avoiding mean-of-rank-means bias.
METRICS = ('loss', 'nb_nll_raw', 'nb_nll_mean_reweighted', 'nb_nll_alpha_branch',
           'optimization_surrogate', 'mu_pcc_unweighted')


def accumulate_metrics(totals, metrics, *, transcripts, pairs):
    for i, key in enumerate(METRICS):
        weight = pairs if key == 'mu_pcc_unweighted' else transcripts
        totals[i, 0] += metrics[key].detach().to(torch.float64) * weight
        totals[i, 1] += weight


def finish_metrics(totals):
    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if bool((totals[:, 1] <= 0).any()):
        raise RuntimeError('No observations for global epoch metrics.')
    return dict(zip(METRICS, (totals[:, 0] / totals[:, 1]).unbind()))
