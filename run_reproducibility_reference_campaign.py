#!/usr/bin/env python3
"""Staged RiboUnmix campaign. Preparation stops at failed scientific prerequisites."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shlex
import sys
import traceback


def parse_args(argv=None):
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','run','export','analyze','report'],default='prepare')
    p.add_argument('--campaign',choices=['main','extended'],default='main')
    p.add_argument('--output-root',type=Path,default=root/'results/reproducibility_reference_campaign_main_v2')
    p.add_argument('--panel-manifest',type=Path,default=root/'results/my_panels_a100_b32_20260906_114323/panel_manifest.json')
    p.add_argument('--ranking-table',type=Path,default=root/'Datasets/data/HEK_riboseq_profile_quality_rank.tsv')
    p.add_argument('--training-seeds',default='42,43,44')
    p.add_argument('--partition-seed',type=int,default=42)
    p.add_argument('--permutation-seeds',default='11001,11002,11003')
    p.add_argument('--reference-permutations',type=int,default=3)
    p.add_argument('--analysis-seed',type=int,default=20260911)
    p.add_argument('--approved-partition-manifest',type=Path)
    p.add_argument('--approved-plan-hash')
    p.add_argument('--max-new-trainings',type=int)
    p.add_argument('--task-index',type=int,help='One index from the frozen task matrix; suitable for one-GPU array jobs.')
    p.add_argument('--gpus',default='inherit')
    p.add_argument('--authorize-training',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--no-tex',action='store_true')
    p.add_argument('--check-dependencies',action='store_true',
                   help='Check imports in this Python environment, then exit without reading data or preparing a design.')
    args=p.parse_args(argv)
    for key,value in vars(args).items():
        if isinstance(value,Path): setattr(args,key,value.expanduser().resolve())
    args.training_seeds=[int(x) for x in args.training_seeds.split(',')]
    args.permutation_seeds=[int(x) for x in args.permutation_seeds.split(',')]
    if len(args.permutation_seeds)!=3 or len(set(args.permutation_seeds))!=3:
        p.error('Declare exactly three distinct permutation seeds; reduced plans may use fewer mappings.')
    return args


def dependency_error_message(exc):
    """Explain an absent package without installing anything in a batch job."""
    missing = (exc.name or '').split('.')[0]
    distributions = {
        'numpy': 'numpy', 'pandas': 'pandas', 'scipy': 'scipy',
        'matplotlib': 'matplotlib', 'pyarrow': 'pyarrow', 'yaml': 'PyYAML',
        'hydra': 'hydra-core', 'omegaconf': 'omegaconf', 'torch': 'torch',
        'lightning': 'lightning', 'torchmetrics': 'torchmetrics', 'tqdm': 'tqdm',
        'PIL': 'Pillow', 'psutil': 'psutil', 'tensorboard': 'tensorboard',
    }
    message = f"Campaign import failed: {exc}\nActive Python: {sys.executable}\n"
    if missing in distributions:
        package = distributions[missing]
        # Reuse the repository's declared version, not an independently chosen
        # numerical dependency. Only suggest the absent package, not a bulk upgrade.
        requirements = Path(__file__).resolve().with_name('requirements.txt')
        specification = package
        if requirements.is_file():
            specification = next((line.strip() for line in requirements.read_text().splitlines()
                                  if line.strip().split('==')[0].lower() == package.lower()), package)
        command = shlex.join([sys.executable, '-m', 'pip', 'install', specification])
        message += f"Install the missing dependency in this venv on the login node:\n  {command}\n"
    else:
        message += ('Verify the deployed repository files and environment; '
                    'no PyPI package is inferred for this missing module.\n')
    message += ('Then run: python run_reproducibility_reference_campaign.py --check-dependencies\n'
                'No packages were installed automatically. Dependency checks do not authorize training.')
    return message


def main(argv=None):
    args=parse_args(argv)
    if args.stage in ('prepare','report') or args.check_dependencies:
        os.environ['CUDA_VISIBLE_DEVICES']=''
    try:
        from Utils.reference_campaign import (prepare_campaign,read_json,write_campaign_report,check_execution_gate)
        if args.check_dependencies:
            if args.stage == 'prepare':
                # The audit reuses this production rank loader. Importing it is
                # CPU-only and checks its torch/lightning dependencies as well.
                from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDatasetDataModule import load_dataset_quality_ranking
            print(f'Campaign {args.stage} imports OK. Active Python: {sys.executable}')
            print('Import check only: no data read, design preparation, training, or GPU validation.')
            return 0
        if args.stage=='prepare':
            m=prepare_campaign(args)
            print(f"Campaign: {m['status']}; planned={m['planned_training_count']}, trained=0")
            print(f"Proposed partition SHA256: {m['candidate_partition_sha256']}")
            print(f"Plan hash: {m['plan_hash']}")
            return 2 if m['prerequisite_failures'] else 0
        if args.stage=='report':
            write_campaign_report(args.output_root);return 0
        if args.stage=='run':
            from Utils.reference_campaign import execute_campaign
            return execute_campaign(args)
        m=read_json(args.output_root/'campaign_manifest.json')
        check_execution_gate(m,approved_plan_hash=args.approved_plan_hash,
            approved_partition_manifest=args.approved_partition_manifest,max_new_trainings=args.max_new_trainings,
            authorize_training=args.authorize_training)
        # These stages are deliberately not advertised as implemented while
        # preparation has stopped for a required design decision.
        raise ValueError('Campaign factor-export/multi-seed analysis integration remains pending. No job was launched.')
    except ModuleNotFoundError as exc:
        print(dependency_error_message(exc),file=sys.stderr);return 2
    except ImportError as exc:
        print(f'Campaign dependency import failed in {sys.executable}: {exc}\n'
              'Check installed package compatibility with python -m pip check; '
              'no automatic installation or environment change was made.',file=sys.stderr)
        return 2
    except KeyError:
        # A bare dictionary key hides which artifact/code path failed on the cluster.
        # Expected audit failures have actionable messages; unexpected schema errors
        # retain their traceback in the batch stderr log.
        traceback.print_exc(file=sys.stderr)
        return 2
    except (ValueError,FileNotFoundError,RuntimeError) as exc:
        print(str(exc),file=sys.stderr);return 2


if __name__=='__main__': raise SystemExit(main())
