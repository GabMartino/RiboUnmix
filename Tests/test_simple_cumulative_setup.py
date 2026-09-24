"""Check scientific controls without loading large parquets."""
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from run_cumulative_stability import ROOT, prepare


class StandaloneSetupTests(unittest.TestCase):
    def test_prepare_from_current_inputs_and_reuse_shared_folds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            datasets = root / 'datasets.yaml'
            datasets.write_text(yaml.safe_dump({'dataset_path': {
                'd5': '/data/d5.parquet', 'd1': '/data/d1.parquet', 'd3': '/data/d3.parquet'}}))
            ranking = root / 'ranking.tsv'
            ranking.write_text('dataset\tquality_rank\nd1\t1\nd3\t3\nd5\t5\nunselected\t10\n')
            args = argparse.Namespace(
                output_root=root / 'new_experiment', seed=42, sizes=[2, 3],
                config=ROOT / 'config/config_ribounmix_multidataset.yaml',
                datasets=datasets, ranking=ranking)
            panels = ['exp8_qrank_N002_seed42', 'exp8_qrank_N003_seed42']
            split = dict(common_test_ids=['test'],
                         panel_train_eligible_ids={p: ['train'] for p in panels},
                         panel_validation_ids={p: ['validation'] for p in panels})
            with patch('run_cumulative_stability.build_exp8_transcript_split', return_value=split) as split_builder, \
                 patch('run_cumulative_stability.fit_panel_reliability_manifest', return_value={}) as fitter:
                plan = prepare(args)
                self.assertEqual(prepare(args), plan)
                self.assertEqual(split_builder.call_count, 1)
                self.assertEqual(fitter.call_count, 2)
                for call in fitter.call_args_list:
                    self.assertEqual(call.kwargs['panel_training_ids'], ['train'])
                    self.assertEqual(call.kwargs['validation_ids'], ['validation'])
                    self.assertEqual(call.kwargs['test_ids'], ['test'])

            self.assertEqual(len(plan['tasks']), 10)
            self.assertEqual(plan['tasks'][0]['datasets'], ['d1', 'd3'])
            self.assertEqual(plan['tasks'][-1]['datasets'], ['d1', 'd3', 'd5'])
            configs = {t['arm']: yaml.safe_load(Path(t['config_path']).read_text())
                       for t in plan['tasks'] if t['N'] == 2}
            for cfg in configs.values():
                self.assertEqual(cfg['experiment']['seed'], 42)
                self.assertEqual(cfg['split'], configs['equal']['split'])
                self.assertEqual(cfg['data']['reliability_reference_manifest'],
                                 configs['equal']['data']['reliability_reference_manifest'])

            def weights(arm):
                return configs[arm]['model']['gamma_centering']['reference']['explicit_weights']

            self.assertEqual(weights('equal'), {'d1': 1., 'd3': 1.})
            # Scores use the full ranking universe, including unselected rows.
            self.assertAlmostEqual(weights('ranked_p3')['d3'], .8 ** 3)
            self.assertEqual(list(weights('ranked_p3').values()),
                             list(weights('reverse_p3').values())[::-1])


if __name__ == '__main__':
    unittest.main()
