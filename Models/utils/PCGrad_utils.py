import torch
from torch import nn


def per_dataset_losses(
    loss_per_sample: torch.Tensor,
    dataset_ids: torch.Tensor,
) -> list[torch.Tensor]:
    dataset_ids = dataset_ids.reshape(-1)
    loss_per_sample = loss_per_sample.reshape(-1)

    losses = []

    for dataset_id in torch.unique(dataset_ids.detach()):
        ds_mask = dataset_ids == dataset_id

        if torch.any(ds_mask):
            losses.append(loss_per_sample[ds_mask].mean())

    return losses



def pcgrad_combine(
    flat_grads: list[torch.Tensor],
    eps: float = 1e-12,
) -> torch.Tensor:
    if len(flat_grads) == 0:
        raise ValueError("No gradients passed to PCGrad.")

    if len(flat_grads) == 1:
        return flat_grads[0]

    projected = []

    for i, g_i_original in enumerate(flat_grads):
        g_i = g_i_original.clone()

        order = torch.randperm(len(flat_grads), device=g_i.device)

        for j_tensor in order:
            j = int(j_tensor.item())

            if j == i:
                continue

            g_j = flat_grads[j]

            dot = torch.dot(g_i, g_j)
            denom = torch.dot(g_j, g_j).clamp_min(eps)

            if dot < 0:
                g_i = g_i - (dot / denom) * g_j

        projected.append(g_i)

    return torch.stack(projected, dim=0).mean(dim=0)



def assign_flat_grads(
    params: list[nn.Parameter],
    flat_grad: torch.Tensor,
) -> None:
    offset = 0

    for p in params:
        n = p.numel()
        g = flat_grad[offset:offset + n].view_as(p)
        offset += n

        if p.grad is None:
            p.grad = g.detach().clone()
        else:
            p.grad.detach().copy_(g)
