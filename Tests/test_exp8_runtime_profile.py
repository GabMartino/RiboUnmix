import contextlib
import io
import json
import shlex
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from Utils.exp8_runtime_profile import execution_profile
from Tests import test_exp8_parallel_resume as resume_tests
from Tests.test_gamma_centering import _fixed_reference_helper

RESUME = resume_tests.RESUME


class Exp8RuntimeTests(unittest.TestCase):
    def test_resume_applies_profile_without_mutating_saved_design(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task, run_id = resume_tests.Exp8ParallelResumeTests._write_launcher(root, 'N114/test')
            launch = task / 'launch_command.sh'
            original = launch.read_bytes()
            # Compose a fully resolved frozen config with the real override keys.
            config = yaml.safe_load((RESUME.PROJECT_ROOT / 'config/config_ribounmix_multidataset.yaml').read_text())
            config['dataset_config'] = {'dataset_path': {'d': 'fixture.parquet'}}
            for token in shlex.split(original.decode()):
                if '=' not in token:
                    continue
                key, value = token.split('=', 1)
                node = config
                parts = key.lstrip('+').split('.')
                for part in parts[:-1]:
                    node = node.setdefault(part, {})
                node[parts[-1]] = yaml.safe_load(value)
            (task / 'resolved_config.yaml').write_text(yaml.safe_dump(config))
            checkpoints = task / 'checkpoints'
            checkpoints.mkdir()
            torch.save({'epoch': 5, 'global_step': 60, 'optimizer_states': [{'state': {}}],
                        'lr_schedulers': [{}]}, checkpoints / 'last.ckpt')
            args = RESUME.parse_args([
                '--run-root', str(root), '--gpus', 'inherit', '--dry-run',
                '--exp8-runtime-profile', 'auto', '--profile-gpu-memory-gib', '80',
                '--use-saved-resolved-config', '--max-pair-rows-per-forward', '768',
            ])
            with contextlib.redirect_stdout(io.StringIO()):
                record = RESUME._prepare_task(args, root, launch)
            command = record['command']
            get = lambda key: RESUME._command_override(command, key)
            self.assertEqual(get('data.batch_size'), '32')
            self.assertEqual(get('training.grouped_optimizer_batch.target_unique_transcripts_per_optimizer_step'), '32')
            self.assertEqual(get('model.gamma_centering.reference.weighting'), 'equal')
            self.assertEqual(get('training.execution_microbatching.max_pair_rows_per_forward'), '768')
            self.assertEqual(get('training.execution_microbatching.max_padded_codon_tokens_per_forward'), '512000')
            self.assertEqual(get('model.gamma_centering.reference.chunk_size'), '32')
            self.assertEqual(get('model.gamma_centering.reference.chunk_size_override_on_load'), 'true')
            self.assertEqual(record['resume_mode'], 'exact_full_state')
            self.assertTrue(record['saved_config']['hydra_composition_checked'])
            self.assertEqual(launch.read_bytes(), original)
            self.assertIsNone(args.reference_chunk_size)  # per-task copy
            self.assertEqual(json.loads((task / 'resume_manifest.json').read_text())
                             ['exp8_runtime_profile']['N'], 114)

    def test_gpu_memory_and_n_bound_the_candidate_profile(self):
        self.assertEqual(execution_profile(114, 24)['max_pair_rows_per_forward'], 256)
        self.assertEqual(execution_profile(10, 80)['max_pair_rows_per_forward'], 512)
        self.assertEqual(execution_profile(114, 80)['reference_chunk_size'], 32)
        self.assertEqual(execution_profile(114, 80, 'aggressive')['reference_chunk_size'], 64)
        self.assertEqual(execution_profile(2, 80)['reference_chunk_size'], 2)

    def test_explicit_chunk_override_survives_checkpoint_extra_state(self):
        model = _fixed_reference_helper([0, 1], [1., 2.])
        model.alpha_mode = 'learned'
        model.fixed_alpha = 1.
        model.fixed_log_alpha = torch.tensor(0.)
        model.dataset_bias_model.raw_log_gamma_bound = 8.
        model.gamma_centering_mode = 'fixed_reference'
        model.gamma_reference_dataset_names = ('a', 'b')
        model.gamma_reference_manifest_hash = 'test'
        model.gamma_centering_weighting = 'quality_rank'
        model.gamma_centering_quality_rank_power = 1.
        model.selected_dataset_names = ('a', 'b')
        model.selected_dataset_ids = (0, 1)
        state = {'gamma_reference_chunk_size': 16}
        model.gamma_reference_chunk_size = 32
        model.gamma_reference_chunk_size_override_on_load = True
        model.set_extra_state(state)
        self.assertEqual(model.gamma_reference_chunk_size, 32)
        model.gamma_reference_chunk_size_override_on_load = False
        model.set_extra_state(state)
        self.assertEqual(model.gamma_reference_chunk_size, 16)
