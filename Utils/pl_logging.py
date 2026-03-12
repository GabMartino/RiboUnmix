from __future__ import annotations

from typing import Dict, Iterable, Set

import torch


def log_scalars(
    module,
    scalars: Dict[str, torch.Tensor | float],
    *,
    batch_size: int,
    on_step: bool,
    on_epoch: bool,
    prog_bar: Iterable[str] = (),
    sync_dist: bool = True,
) -> None:
    """Small wrapper around LightningModule.log for consistent logging.

    - Forces passing `batch_size` (avoids Lightning warnings and ensures weighted epoch means).
    - Avoids repeating on_step/on_epoch/sync_dist everywhere.

    Parameters
    ----------
    module:
        A LightningModule (or anything exposing `.log`).
    scalars:
        Mapping name -> scalar tensor/float.
    prog_bar:
        Iterable of keys that should be shown in the progress bar.
    """

    prog: Set[str] = set(prog_bar)
    for k, v in scalars.items():
        module.log(
            k,
            v,
            prog_bar=(k in prog),
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=sync_dist,
            batch_size=batch_size,
        )
