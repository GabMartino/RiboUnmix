"""Campaign CLI dependency failures must not reach data or design preparation."""
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_reproducibility_reference_campaign as campaign_cli


ROOT = Path(__file__).resolve().parents[1]


class CampaignDependencyTests(unittest.TestCase):
    def test_missing_scipy_uses_declared_version_and_active_interpreter(self):
        message = campaign_cli.dependency_error_message(
            ModuleNotFoundError("No module named 'scipy'", name='scipy'))
        self.assertIn('scipy==1.17.1', message)
        self.assertIn(sys.executable, message)
        self.assertIn('--check-dependencies', message)
        self.assertNotIn('pip install -r', message)
        self.assertNotIn('--upgrade', message)

    def test_missing_repository_module_is_not_treated_as_pypi_dependency(self):
        message = campaign_cli.dependency_error_message(
            ModuleNotFoundError("No module named 'Utils.panel_reference_audit'",
                                name='Utils.panel_reference_audit'))
        self.assertIn('deployed repository', message)
        self.assertNotIn('pip install', message)

    def test_univie_import_error_is_caught_before_any_outputs(self):
        original_import = __import__

        def missing_scipy(name, *args, **kwargs):
            if name == 'Utils.reference_campaign':
                raise ModuleNotFoundError("No module named 'scipy'", name='scipy')
            return original_import(name, *args, **kwargs)

        for flags in ([], ['--check-dependencies']):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'must_not_exist'
                error = io.StringIO()
                with patch.dict('os.environ'), patch('builtins.__import__', side_effect=missing_scipy), redirect_stderr(error):
                    code = campaign_cli.main(['--output-root', str(output), *flags])
                self.assertEqual(code, 2)
                self.assertFalse(output.exists())
                self.assertIn('scipy==1.17.1', error.getvalue())
                self.assertNotIn('Traceback', error.getvalue())

    def test_help_does_not_require_numerical_dependencies(self):
        original_import = __import__

        def no_scientific_imports(name, *args, **kwargs):
            if name.startswith(('numpy', 'pandas', 'scipy', 'Utils.reference_campaign')):
                raise AssertionError('Help must not import scientific packages.')
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=no_scientific_imports), redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                campaign_cli.main(['--help'])
        self.assertEqual(error.exception.code, 0)

    def test_unexpected_keyerror_keeps_traceback_and_nonzero_exit(self):
        error = io.StringIO()
        with patch.dict('os.environ'), \
             patch('Utils.reference_campaign.prepare_campaign', side_effect=KeyError('ranking')), \
             redirect_stderr(error):
            code = campaign_cli.main(['--stage', 'prepare'])
        self.assertEqual(code, 2)
        self.assertIn('Traceback', error.getvalue())
        self.assertIn("KeyError: 'ranking'", error.getvalue())
        self.assertIn('prepare_campaign', error.getvalue())

    def test_real_import_check_leaves_requested_output_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'must_not_exist'
            result = subprocess.run(
                [sys.executable, str(ROOT / 'run_reproducibility_reference_campaign.py'),
                 '--check-dependencies', '--output-root', str(output)],
                cwd=ROOT, text=True, capture_output=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('imports OK', result.stdout)
            self.assertIn('no data read', result.stdout)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
