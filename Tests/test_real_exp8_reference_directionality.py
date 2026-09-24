import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from run_real_exp8_reference_directionality import (
    ARMS, SIZES, _assert_arm_match, _reference_policy, load_source, main,
    sha256, task_config, task_matrix, transcript_id_hash, verify_hashes,
)


def template(names):
    return dict(name='old', defaults=[{'dataset_config': 'stale'}, '_self_'],
        experiment=dict(dataset=names, seed=42, from_checkpoint=False),
        prediction=dict(checkpoint_variants=['best_val_loss']), split={},
        paths={}, data=dict(dataset_quality_ranking={}, batch_size=32,
            train_sampling_strategy='transcript_grouped_multidataset_pairs'),
        model=dict(alpha_mode='learned', mass_conservation=False, dataset_bias_params={},
            gamma_centering=dict(mode='fixed_reference', reference=dict(chunk_size=16))),
        trainer={}, loss=dict(sample_reduction='transcript_balanced', experiment_mode='standard_nb',
            replica_nb_weight=1., consensus_raw_pcc_weight=.5, consensus_nb_vst_pcc_weight=.5),
        optim=dict(lr_biological=.0005, lr_rest=.001))


def source_fixture(root):
    table = pd.DataFrame(dict(dataset=['a','b','c','outside'], quality_rank=[1,2,3,115],
                              rank_component_count=[10]*4))
    for i in range(10):
        table[f'rank_feature{i}'] = [1,2,3,4]
    rank_path = root / 'rank.tsv'
    table.to_csv(rank_path, sep='\t', index=False)
    tasks = [dict(run_id=f'old{n}', N=n, training_seed=42, datasets=list('abc')[:n],
                  directory=f'N{n:03d}/seed42') for n in (2,3)]
    split = dict(manifest_version=1, common_test_ids=['test'], common_validation_ids=['v2'],
        panels={t['run_id']:t['datasets'] for t in tasks},
        panel_validation_ids={'old2':['v2'], 'old3':['v3']},
        panel_train_eligible_ids={'old2':['tr'], 'old3':['tr','v2']})
    (root / 'experiment_split_manifest.json').write_text(json.dumps(split))
    (root / 'experiment_manifest.json').write_text(json.dumps(dict(
        experiment_design='cumulative_top_quality', frozen_ranking_table=str(rank_path),
        ranking_sha256=sha256(rank_path), tasks=tasks)))
    for t in tasks:
        directory = root / t['directory']
        directory.mkdir(parents=True)
        (directory / 'resolved_config.yaml').write_text(yaml.safe_dump(template(t['datasets'])))
        ids = split['panel_train_eligible_ids'][t['run_id']]
        (directory / 'reliability_reference_manifest.json').write_text(json.dumps(dict(
            manifest_version=1, reference_split='training_only', heldout_rows_used_for_fitting=0,
            panel_training_transcript_id_hash=transcript_id_hash(ids), panel_training_transcript_count=len(ids),
            datasets={d:{} for d in t['datasets']})))


class CumulativeDirectionalityTests(unittest.TestCase):
    def test_stable_full_and_noncontiguous_indices(self):
        tasks = task_matrix(SIZES, [42,43,44])
        self.assertEqual(len(tasks), 63)
        self.assertEqual([t['arm'] for t in tasks[:3]], list(ARMS))
        self.assertEqual([(tasks[i]['N'], tasks[i]['training_seed'], tasks[i]['arm']) for i in (20,21,41,62)],
                         [(114,42,'reverse'), (2,43,'equal'), (114,43,'reverse'), (114,44,'reverse')])
        self.assertEqual(len({t['run_id'] for t in tasks}), 63)

    def test_global_conversion_and_reversal_at_small_N(self):
        ranks = {'a':1., 'b':2., 'outside':115.}
        q = {d:(115-r+1)/115 for d,r in ranks.items()}
        policies = [_reference_policy(arm, ['a','b'], ranks, q) for arm in ARMS]
        equal, ranked, reverse = [np.array([r['pi'] for r in rows]) for _,rows in policies]
        np.testing.assert_allclose(equal, [.5,.5])
        np.testing.assert_allclose(ranked, [115/229,114/229])
        np.testing.assert_allclose(reverse, ranked[::-1])
        self.assertAlmostEqual(1/(ranked@ranked), 1/(reverse@reverse))

    def test_preserve_N_specific_validation_and_common_test_without_input_edits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_fixture(root)
            before = {p:sha256(p) for p in root.rglob('*') if p.is_file()}
            loaded = load_source(root, [2,3])
            self.assertEqual(loaded['folds'][3]['train_ids'], ['tr','v2'])
            self.assertEqual(loaded['folds'][3]['validation_ids'], ['v3'])
            self.assertEqual(loaded['folds'][2]['test_ids'], loaded['folds'][3]['test_ids'])
            self.assertEqual(loaded['quality']['b'], 114/115)
            self.assertEqual(before, {p:sha256(p) for p in before})
            p=root/'experiment_split_manifest.json'
            bad=json.loads(p.read_text());bad['panel_train_eligible_ids']['old3'].append('test')
            p.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'overlap'):
                load_source(root, [2,3])

    def test_checksum_missing_rank_and_reliability_mismatch_fail(self):
        for failure in ('checksum','rank','reliability'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                root=Path(temp);source_fixture(root)
                if failure in ('checksum','rank'):
                    path=root/'rank.tsv'; frame=pd.read_csv(path,sep='\t')
                    frame.loc[frame.dataset=='b','dataset']='not_b'
                    frame.to_csv(path,sep='\t',index=False)
                    if failure=='rank':
                        path=root/'experiment_manifest.json';m=json.loads(path.read_text())
                        m['ranking_sha256']=sha256(root/'rank.tsv');path.write_text(json.dumps(m))
                else:
                    path=root/'N002/seed42/reliability_reference_manifest.json'
                    ref=json.loads(path.read_text());ref['panel_training_transcript_id_hash']='wrong'
                    path.write_text(json.dumps(ref))
                with self.assertRaises(ValueError):load_source(root,[2,3])

    def test_paired_configs_and_only_approved_changes(self):
        tasks=task_matrix([2], [42]);cfg=template(['a','b']);original=copy.deepcopy(cfg)
        ranks={'a':1.,'b':2.};q={'a':1.,'b':114/115}
        configs={t['arm']:task_config(cfg,t,Path('/new'),'old2',_reference_policy(t['arm'],['a','b'],ranks,q)[0])
                 for t in tasks}
        self.assertEqual(cfg, original)
        for c in configs.values():
            self.assertNotIn('defaults',c)
            self.assertEqual(c['trainer']['devices'],[0])
            self.assertEqual(c['data']['batch_size'],32)
            self.assertEqual(c['model']['alpha_mode'],'learned')
            self.assertEqual(c['model']['dataset_bias_params']['context_gru_tbptt_window'],0)
            self.assertFalse(c['experiment']['from_checkpoint'])
        for arm in ['ranked','reverse']:_assert_arm_match(configs['equal'],configs[arm])
        configs['ranked']['optim']['lr_rest']=.01
        with self.assertRaisesRegex(ValueError,'optim.lr_rest'):
            _assert_arm_match(configs['equal'],configs['ranked'])

    def test_authorization_before_preparation_and_unchanged_cuda_visibility(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':'7'}), patch(
                'run_real_exp8_reference_directionality.freeze_plan') as freeze:
            with self.assertRaises(SystemExit): main(['--task-index','0'])
            freeze.assert_not_called()
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'7')

    def test_frozen_bytes_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'input';path.write_text('old')
            hashes={str(path):sha256(path)};verify_hashes(hashes)
            path.write_text('new')
            with self.assertRaisesRegex(ValueError,'changed'):verify_hashes(hashes)

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "run_real_exp8_reference_directionality_univie.slurm").is_file(),
        "Site-specific Slurm launchers are intentionally excluded from the public tree.",
    )
    def test_univie_worker_forwards_one_task_quotes_and_exit_code(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);venv=directory/'venv with spaces';(venv/'bin').mkdir(parents=True)
            (venv/'bin/activate').write_text('export TEST_ACTIVATED=1\n')
            module=directory/'module';module.write_text('#!/bin/bash\nexit 0\n');module.chmod(0o755)
            srun=directory/'srun'
            srun.write_text('#!/usr/bin/env python3\nimport os,json,sys\nfrom pathlib import Path\n'
                'Path(os.environ["TEST_ARGS"]).write_text(json.dumps(dict(argv=sys.argv[1:],cuda=os.environ["CUDA_VISIBLE_DEVICES"],active=os.environ["TEST_ACTIVATED"])))\n'
                'raise SystemExit(7)\n');srun.chmod(0o755)
            output=directory/'args.json'
            env=dict(os.environ, PATH=f'{directory}:{os.environ["PATH"]}', PROJECT_DIR=str(root),
                UNIVIE_VENV_PATH=str(venv), SLURM_ARRAY_TASK_ID='41',CUDA_VISIBLE_DEVICES='7',
                TEST_ARGS=str(output), OUTPUT_ROOT=str(directory/'outputs with spaces'),DRY_RUN='0')
            result=subprocess.run(['bash',str(root/'run_real_exp8_reference_directionality_univie.slurm')],env=env)
            self.assertEqual(result.returncode,7)
            args=json.loads(output.read_text());argv=args['argv']
            self.assertEqual(args['cuda'],'7');self.assertEqual(args['active'],'1')
            self.assertEqual(argv[argv.index('--task-index')+1],'41')
            self.assertEqual(argv[argv.index('--output-root')+1],str(directory/'outputs with spaces'))
            self.assertIn('--authorize-training',argv);self.assertIn('--resume',argv)
            self.assertNotIn('--dry-run',argv)


if __name__ == '__main__':unittest.main()
