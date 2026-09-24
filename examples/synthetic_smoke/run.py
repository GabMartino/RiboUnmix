#!/usr/bin/env python3
"""Run the public RiboUnmix smoke workflow without cluster infrastructure."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


EXAMPLE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXAMPLE_ROOT.parents[1]
ENTRYPOINT = REPOSITORY_ROOT / "main_ribounmix_synthetic.py"


def _run(arguments: list[str]) -> None:
    subprocess.run(
        [sys.executable, str(ENTRYPOINT), *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train the small public model, replay its committed checkpoint, "
            "or verify the committed artifacts."
        )
    )
    parser.add_argument("action", choices=("train", "replay", "verify"))
    args = parser.parse_args()

    if args.action == "train":
        _run(["--config-name=config_ribounmix_smoke"])
    elif args.action == "replay":
        _run(
            [
                "--config-name=config_ribounmix_smoke",
                "experiment.train=false",
                "experiment.predict=true",
                "experiment.from_checkpoint=true",
                "paths.checkpoints=examples/synthetic_smoke/reference/checkpoints",
                "paths.logs=outputs/synthetic_smoke/replay/logs",
                "paths.results=outputs/synthetic_smoke/replay/results",
            ]
        )
    else:
        subprocess.run(
            [sys.executable, str(EXAMPLE_ROOT / "verify_reference.py")],
            cwd=REPOSITORY_ROOT,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
