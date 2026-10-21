"""Check standalone-run settings without executing or importing pySEBAL."""
from pathlib import Path
import contextlib
import importlib.util
import io
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

path = Path(__file__).resolve().parents[1] / 'extra code/SEBAL/Run_py3.py'
spec = importlib.util.spec_from_file_location('central_runner', path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class CentralRunnerTests(unittest.TestCase):
    def test_study_defaults_override_legacy_lower_member(self):
        model = SimpleNamespace(ALPHA=-.201, BETA=.245, SEBALcode=Mock())
        with tempfile.NamedTemporaryFile(suffix='.xlsx') as workbook:
            with patch.dict('sys.modules', {'pysebal_py3': model}), contextlib.redirect_stdout(io.StringIO()):
                runner.main(['--excel', workbook.name, '--end-row', '3'])
            self.assertEqual((model.ALPHA, model.BETA), (-.089, .343))
            self.assertTrue(model.CALIBRATED)
            self.assertFalse(model.CALIBRATIONS)
            self.assertEqual(model.SEBALcode.call_count, 2)
            model.SEBALcode.assert_any_call(2, workbook.name)

    def test_explicit_coefficients_and_invalid_values(self):
        model = SimpleNamespace(SEBALcode=Mock())
        with tempfile.NamedTemporaryFile(suffix='.xlsx') as workbook:
            args = ['--excel', workbook.name, '--end-row', '2']
            with patch.dict('sys.modules', {'pysebal_py3': model}), contextlib.redirect_stdout(io.StringIO()):
                runner.main(args + ['--alpha', '.2', '--beta', '.3'])
            self.assertEqual((model.ALPHA, model.BETA), (.2, .3))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main(args + ['--alpha', 'nan'])


if __name__ == '__main__':
    unittest.main()
