from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional


_CKPT_RE = re.compile(
    r"(?:epoch=(?P<epoch>\d+))?.*?(?:val_loss_epoch=(?P<val>[-+]?\d+(?:\.\d+)?))|^(?P<epoch2>\d+)-(?P<val2>[-+]?\d+(?:\.\d+)?)\.ckpt$"
)


def parameter_hash(model: object, prefix: str = "") -> str:
    """Hash named trainable parameters for initialization verification.

    Reference tensors and other registered buffers are intentionally excluded.
    Values are serialized after a deterministic CPU-contiguous conversion.
    """

    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if name.startswith(prefix):
            value = parameter.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str((tuple(value.shape), str(value.dtype))).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def find_checkpoint(path: str | Path, *, prefer: str = "best") -> Optional[Path]:
    """Find a checkpoint under `path`.

    prefer:
      - "best": smallest val metric parsed from filename (fallback latest)
      - "last": prefer last.ckpt
      - "latest": most recently modified
    """

    root = Path(path)
    if not root.exists():
        return None

    last = next(root.rglob("last.ckpt"), None)
    if prefer == "last" and last is not None:
        return last

    ckpts = [p for p in root.rglob("*.ckpt") if p.is_file() and p.name != "last.ckpt"]
    if not ckpts:
        return last

    if prefer == "latest":
        return max(ckpts, key=lambda p: p.stat().st_mtime)

    # prefer == "best"
    scored: list[tuple[float, int, Path]] = []
    for p in ckpts:
        m = _CKPT_RE.search(p.name)
        if not m:
            scored.append((float("inf"), 0, p))
            continue

        if m.group("val") is not None:
            val = float(m.group("val"))
            epoch = int(m.group("epoch")) if m.group("epoch") is not None else -1
            scored.append((val, -epoch, p))
        elif m.group("val2") is not None:
            val = float(m.group("val2"))
            epoch = int(m.group("epoch2")) if m.group("epoch2") is not None else -1
            scored.append((val, -epoch, p))
        else:
            scored.append((float("inf"), 0, p))

    if any(v != float("inf") for v, _, _ in scored):
        return min(scored, key=lambda t: (t[0], t[1]))[2]

    return max(ckpts, key=lambda p: p.stat().st_mtime)
