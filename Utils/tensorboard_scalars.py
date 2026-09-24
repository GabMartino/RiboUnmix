from __future__ import annotations

import argparse
from pathlib import Path


def _import_event_accumulator():
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: tensorboard. Install requirements or run "
            "`python3 -m pip install tensorboard`."
        ) from exc

    return event_accumulator


def find_event_runs(logdir: Path) -> list[Path]:
    event_files = sorted(logdir.rglob("events.out.tfevents*"))
    return sorted({path.parent for path in event_files})


def load_scalars(run_dir: Path) -> dict[str, list]:
    event_accumulator = _import_event_accumulator()
    accumulator = event_accumulator.EventAccumulator(
        str(run_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()
    return {
        tag: accumulator.Scalars(tag)
        for tag in accumulator.Tags().get("scalars", [])
    }


def load_config(run_dir: Path) -> str | None:
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        return cfg_path.read_text()
    return None


def format_row(
    *,
    run_dir: Path,
    logdir: Path,
    tag: str,
    step: int,
    value: float,
) -> str:
    run_name = str(run_dir.relative_to(logdir))
    return f"{run_name}\t{tag}\tstep={step}\tvalue={value:.6g}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print TensorBoard scalar values from event files.",
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        default=Path("logs/riboai_queueing"),
        help="Root directory containing TensorBoard event files.",
    )
    parser.add_argument(
        "--run-contains",
        default=None,
        help="Only include run paths containing this substring.",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="Scalar tags to print. Defaults to every scalar tag.",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=1,
        help="Number of latest values to print per tag.",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List run directories and exit.",
    )
    parser.add_argument(
        "--list-tags",
        action="store_true",
        help="List scalar tags for matching runs and exit.",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print the saved config.yaml for each matching run and exit.",
    )
    args = parser.parse_args()

    logdir = args.logdir.resolve()
    runs = find_event_runs(logdir)

    if args.run_contains:
        runs = [run for run in runs if args.run_contains in str(run)]

    if args.list_runs:
        for run in runs:
            print(run.relative_to(logdir))
        return

    if not runs:
        raise SystemExit(f"No TensorBoard event runs found under {logdir}.")

    if args.show_config:
        for run in runs:
            print(f"=== {run.relative_to(logdir)} ===")
            cfg_text = load_config(run)
            if cfg_text:
                print(cfg_text)
            else:
                print("  (no config.yaml — run predates config snapshot feature)")
            print()
        return

    for run in runs:
        scalars = load_scalars(run)

        if args.list_tags:
            print(run.relative_to(logdir))
            for tag in sorted(scalars):
                print(f"  {tag}")
            continue

        tags = args.tags if args.tags else sorted(scalars)

        for tag in tags:
            values = scalars.get(tag)
            if not values:
                continue

            for event in values[-args.tail :]:
                print(
                    format_row(
                        run_dir=run,
                        logdir=logdir,
                        tag=tag,
                        step=event.step,
                        value=event.value,
                    )
                )


if __name__ == "__main__":
    main()
