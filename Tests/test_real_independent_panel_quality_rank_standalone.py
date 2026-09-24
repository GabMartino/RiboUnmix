from __future__ import annotations

import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

import run_real_independent_panel_convergence_quality_rank as runner


class QualityRankStandaloneDesignTests(unittest.TestCase):
    def test_standalone_mode_does_not_require_a_reference_directory(self):
        args = runner.parse_args(["--standalone-design"])
        self.assertTrue(args.standalone_design)
        self.assertIsNone(args.reference_design_root)

    def test_reference_and_standalone_modes_are_mutually_exclusive(self):
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                runner.parse_args([])
            with self.assertRaises(SystemExit):
                runner.parse_args([
                    "--standalone-design",
                    "--reference-design-root",
                    str(Path("equal-run")),
                ])


if __name__ == "__main__":
    unittest.main()
