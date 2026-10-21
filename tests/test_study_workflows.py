"""Task wiring, private path overrides, and promoted analysis invariants."""
from pathlib import Path
from tempfile import TemporaryDirectory
import contextlib
import io
import json
import runpy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import main
from workflows import catalog
from modules.regional_volatility import interval
from modules.validation_maps import eligible_correlations


class StudyWorkflowTests(unittest.TestCase):
    def test_cli_list_does_not_execute_science(self):
        with patch('workflows.catalog.run') as execute, contextlib.redirect_stdout(io.StringIO()) as out:
            main.main(['--list'])
        execute.assert_not_called()
        self.assertIn('rice-seasonal', out.getvalue())
        self.assertIn('uncertainty-samples', out.getvalue())

    def test_cli_all_orders_stages_and_excludes_raw_preparation(self):
        with patch('workflows.catalog.preflight', return_value=[]), patch('workflows.catalog.run') as execute:
            main.main(['--all'])
        names = execute.call_args.args[0]
        self.assertLess(names.index('validation'), names.index('maps'))
        self.assertLess(names.index('rice'), names.index('rice-seasonal'))
        self.assertNotIn('tower-extract', names)
        self.assertNotIn('uncertainty-inputs', names)
        self.assertTrue(all('pysebal' not in n.lower() for n in names))

    def test_check_and_failure_never_execute(self):
        with patch('workflows.catalog.preflight', return_value=[]), patch('workflows.catalog.run') as execute:
            main.main(['--task', 'maps', '--check'])
            execute.assert_not_called()
        with patch('workflows.catalog.preflight', return_value=['maps: missing input']), patch('workflows.catalog.run') as execute:
            with self.assertRaises(SystemExit):
                main.main(['--task', 'maps'])
            execute.assert_not_called()

    def test_preflight_accepts_only_planned_producer(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'not_created.csv'
            fake = {
                'producer': catalog.Task('producer', 'make input', Path(directory)),
                'consumer': catalog.Task('consumer', 'read input', Path(directory), ((path, None, 'producer'),)),
            }
            with patch('workflows.catalog.tasks', return_value=fake):
                self.assertEqual(catalog.preflight(['producer', 'consumer']), [])
                self.assertTrue(catalog.preflight(['consumer']))

    def test_optional_bootstrap_preparation_orders_before_fitting(self):
        names = catalog.ordered_tasks(['calibration-fit', 'uncertainty-inputs', 'validation'])
        self.assertLess(names.index('validation'), names.index('uncertainty-inputs'))
        self.assertLess(names.index('uncertainty-inputs'), names.index('calibration-fit'))

    def test_calibration_fit_uses_precalibration_pairs_not_evaluation_workbook(self):
        from workflows import study_workflows
        with patch('modules.calibration_fit.run') as fit:
            study_workflows.calibration_fit()
        self.assertEqual(fit.call_args.args[0], catalog.cfg.CALIBRATION_FIT_INPUT)
        self.assertNotEqual(fit.call_args.args[0], catalog.cfg.CALIBRATION_MERGED)

    def test_local_base_paths_derive_outputs_without_editing_public_config(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.py').write_text((Path(__file__).parents[1] / 'config.py').read_text())
            (root / 'config_local.py').write_text("from pathlib import Path\nOUTPUT_ROOT = Path('D:/private/results')\n")
            config = runpy.run_path(str(root / 'config.py'))
            self.assertEqual(config['SCALING_OUTPUT_BASE'], Path('D:/private/results/validation'))

    def test_variability_uses_only_consecutive_days_and_preserves_mssd(self):
        dates = pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-04', '2021-01-05'])
        sensor = pd.DataFrame({'date': dates, 'observed': [.1, .2, .8, .6]})
        result = interval(sensor, dates[0], dates[-1])
        self.assertEqual(result['valid_increments'], 2)
        self.assertAlmostEqual(result['AV_norm'], .15)
        self.assertAlmostEqual(result['QV_norm'], .025)
        self.assertNotAlmostEqual(result['QV_norm'], result['AV_norm']**2)

    def test_correlation_filter_does_not_select_by_performance(self):
        series = pd.DataFrame({'overlaps': [5, 6, 10], 'p_rho': [.9, -.8, .7], 's_rho': [.95, -.6, .8]})
        selected = eligible_correlations(series, 6)
        self.assertEqual(selected.index.tolist(), [1, 2])
        self.assertEqual(selected.iloc[0].p_rho, -.8)

    def test_failed_task_records_failure_and_stops_dependents(self):
        with TemporaryDirectory() as directory:
            tasks = {n: catalog.Task(n, n, Path(directory) / n) for n in ['climate', 'coverage']}
            with patch('workflows.catalog.cfg.OUTPUT_ROOT', Path(directory)), patch('workflows.catalog.tasks', return_value=tasks), \
                 patch('workflows.study_workflows.climate', side_effect=ValueError('bad inputs')), \
                 patch('workflows.study_workflows.coverage') as later:
                with self.assertRaisesRegex(ValueError, 'bad inputs'):
                    catalog.run(['climate', 'coverage'])
                later.assert_not_called()
            manifest = json.loads((Path(directory) / 'run_manifest.json').read_text())
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(manifest['tasks'][0]['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
