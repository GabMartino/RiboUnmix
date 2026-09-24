import numpy as np
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from prepare_rank_balanced_reference_directionality import (
    _reference_policy,
    normalized,
    parse_training_seeds,
    reverse_weight_assignment,
)


class ReferenceDirectionalityTests(unittest.TestCase):
    def test_reverse_assignment_preserves_multiset_and_reverses_direction(self):
        datasets = ["d_best", "d_middle", "d_worst"]
        ranks = {"d_best": 1.0, "d_middle": 4.0, "d_worst": 9.0}
        quality = {name: (10.0 - rank) / 9.0 for name, rank in ranks.items()}

        reverse, donors = reverse_weight_assignment(datasets, ranks, quality)

        self.assertEqual(
            donors,
            {
                "d_best": "d_worst",
                "d_middle": "d_middle",
                "d_worst": "d_best",
            },
        )
        np.testing.assert_allclose(
            np.sort(normalized(quality[name] for name in datasets)),
            np.sort(normalized(reverse[name] for name in datasets)),
            rtol=0.0,
            atol=1e-15,
        )
        self.assertLess(reverse["d_best"], reverse["d_middle"])
        self.assertLess(reverse["d_middle"], reverse["d_worst"])

    def test_three_policies_are_positive_normalized_and_reverse_is_explicit(self):
        datasets = ["a", "b", "c", "d"]
        ranks = {name: float(index + 1) for index, name in enumerate(datasets)}
        quality = {name: (5.0 - ranks[name]) / 4.0 for name in datasets}

        policies = {
            arm: _reference_policy(arm, datasets, ranks, quality)
            for arm in ("equal", "ranked", "reverse")
        }
        for _, rows in policies.values():
            pi = np.array([row["pi"] for row in rows])
            self.assertTrue(np.all(pi > 0))
            np.testing.assert_allclose(pi.sum(), 1.0, rtol=0.0, atol=1e-15)

        self.assertEqual(
            policies["equal"][0], {"weighting": "equal", "quality_rank_power": 0.0}
        )
        self.assertEqual(
            policies["ranked"][0],
            {"weighting": "quality_rank", "quality_rank_power": 1.0},
        )
        self.assertEqual(policies["reverse"][0]["weighting"], "explicit")
        ranked_pi = sorted(row["pi"] for row in policies["ranked"][1])
        reverse_pi = sorted(row["pi"] for row in policies["reverse"][1])
        np.testing.assert_allclose(ranked_pi, reverse_pi, rtol=0.0, atol=1e-15)

    def test_seed_parser_rejects_duplicates(self):
        self.assertEqual(parse_training_seeds("42,43,44"), (42, 43, 44))
        with self.assertRaisesRegex(Exception, "distinct"):
            parse_training_seeds("42,42")

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "prepare_rank_balanced_reference_directionality_univie.slurm").is_file(),
        "Site-specific Slurm launchers are intentionally excluded from the public tree.",
    )
    def test_univie_preparation_has_no_historical_results_dependency(self):
        root = Path(__file__).resolve().parents[1]
        preparation = (root / "prepare_rank_balanced_reference_directionality_univie.slurm").read_text()
        worker = (root / "run_rank_balanced_reference_directionality_univie.slurm").read_text()
        submitter = (root / "submit_rank_balanced_reference_directionality_univie.sh").read_text()

        self.assertNotIn("results/my_panels_a100_b32_20260906_114323", preparation)
        self.assertIn("run_real_independent_panel_convergence.py", preparation)
        self.assertIn("panels_equal_seed42_20260906_114323.json", preparation)
        self.assertIn("validate_historical_design", preparation)
        expected = "${AUDIT_ROOT}/global_rank_balanced_partition_v2"
        self.assertIn(expected, preparation)
        self.assertIn(expected, worker)
        self.assertIn('afterok:${PREPARATION_JOB_ID}', submitter)
        self.assertIn('ARRAY_RANGE="0-11"', submitter)
        self.assertIn('--resume-preparation "${PREPARED_ROOT}"', preparation)

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "run_rank_balanced_reference_directionality_univie.slurm").is_file(),
        "Site-specific Slurm launchers are intentionally excluded from the public tree.",
    )
    def test_submitter_preserves_afterok_and_cancels_impossible_dependency(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            calls=directory/'calls.jsonl'
            sbatch=directory/'sbatch'
            sbatch.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
calls=Path(os.environ['MOCK_SBATCH_CALLS'])
with calls.open('a') as handle:
    handle.write(json.dumps(sys.argv[1:])+'\\n')
index=len(calls.read_text().splitlines())
if os.environ.get('FAIL_PREPARATION_SUBMIT')=='1' and index==1:
    raise SystemExit(7)
print(f'{8000+index};testcluster')
''')
            sbatch.chmod(0o755)
            env=dict(os.environ,PATH=f'{directory}:{os.environ["PATH"]}',
                     MOCK_SBATCH_CALLS=str(calls),PROJECT_DIR=str(root),MAX_CONCURRENT='4')
            result=subprocess.run(['bash',str(root/'submit_rank_balanced_reference_directionality_univie.sh'),'seed42'],
                                  env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            commands=[json.loads(line) for line in calls.read_text().splitlines()]
            self.assertEqual(len(commands),2)
            self.assertEqual(commands[0],['--parsable',str(root/'prepare_rank_balanced_reference_directionality_univie.slurm')])
            self.assertIn('--dependency=afterok:8001',commands[1])
            self.assertIn('--kill-on-invalid-dep=yes',commands[1])
            self.assertIn('--array=0-11%4',commands[1])
            self.assertIn('squeue -j 8001,8002',result.stdout)
            calls.unlink()
            env['FAIL_PREPARATION_SUBMIT']='1'
            failed=subprocess.run(['bash',str(root/'submit_rank_balanced_reference_directionality_univie.sh'),'seed42'],
                                  env=env,capture_output=True,text=True)
            self.assertEqual(failed.returncode,7)
            self.assertEqual(len(calls.read_text().splitlines()),1)


if __name__ == "__main__":
    unittest.main()
