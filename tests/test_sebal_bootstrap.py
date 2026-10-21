"""Compare the ensemble path to the original scientific functions without GDAL.

Extract original function definitions via AST; no upstream model or TIFF IO is
mocked into a claim of actual regional reproduction. Geometry/output tests use
a small synthetic scene and an explicit identity projection.
"""
from pathlib import Path
import ast
import contextlib
import csv
import gzip
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEBAL = ROOT/'extra code'/'SEBAL'
sys.path.insert(0, str(SEBAL))
import sebal_bootstrap_core as core
import Run_py3_bootstrap as runner

NAMES = ['Calc_instantaneous_ET_fraction_calibrations', 'Calc_ETact', 'Calc_Soil_Moisture', 'Separate_E_T']
ORIGINAL_AST = ast.parse((SEBAL/'pysebal_py3.py').read_text())
FUNCTIONS = {'np': np}
exec(compile(ast.Module(body=[n for n in ORIGINAL_AST.body if isinstance(n, ast.FunctionDef) and n.name in NAMES],
                        type_ignores=[]), '<original SEBAL helpers>', 'exec'), FUNCTIONS)


def synthetic_scene():
    shape = (8, 8)
    values = np.arange(64).reshape(shape)
    array = lambda v: np.full(shape, v, dtype=float)
    state = {name: array(.1) for name in core.STATE_NAMES}
    state.update(LE_inst=60.+values*9, rn_inst=array(500), g_inst=array(70),
                 esat_24=array(3), eact_24=array(1.5), Rn_24=array(170), Refl_rad_water=array(10),
                 Lhv=array(2.45e6), ETpot_24=array(6), QC_Map=array(0), water_mask=array(0),
                 vegt_cover=.04+values*.015, Theta_sat_top=array(.45), Theta_sat_sub=array(.48),
                 Theta_res_top=array(.02), Theta_res_sub=array(.025), depl_factor=array(.5),
                 Field_Capacity=array(.3), FPAR=array(.5), Soil_moisture_wilting_point=array(.1),
                 Light_use_extinction_factor=.5, LAI=array(2), sl_es_24=array(.2), Psychro_c=array(.065))
    state['QC_Map'][0, 0] = 1
    state['water_mask'][0, 1] = 1
    return state


def original_sequence(s, alpha, beta):
    _, ef = FUNCTIONS[NAMES[0]](s['LE_inst'], s['rn_inst'], s['g_inst'], alpha, beta, True)
    eta, _ = FUNCTIONS[NAMES[1]](s['esat_24'], s['eact_24'], ef, s['Rn_24'], s['Refl_rad_water'], s['Lhv'])
    etp = np.where(s['ETpot_24'] < eta, eta, s['ETpot_24'])
    trigger, total, first, stress, top, _ = FUNCTIONS[NAMES[2]](
        eta, ef, s['QC_Map'], s['water_mask'], s['vegt_cover'], s['Theta_sat_top'], s['Theta_sat_sub'],
        s['Theta_res_top'], s['Theta_res_sub'], s['depl_factor'], s['Field_Capacity'], s['FPAR'], s['Soil_moisture_wilting_point'])
    return FUNCTIONS[NAMES[3]](s['Light_use_extinction_factor'], s['LAI'], etp,
        s['Theta_res_top'], s['Theta_res_sub'], s['Theta_sat_top'], s['Theta_sat_sub'], top,
        s['sl_es_24'], s['Psychro_c'], stress, s['vegt_cover'], eta, trigger, first, total)[6]


class FakeTemplate:
    RasterYSize, RasterXSize = 8, 8
    def GetGeoTransform(self):
        return (70., 1., 0., 40., 0., -1.)
    def GetProjection(self):
        return 'synthetic identity projection'


class EnsembleTests(unittest.TestCase):
    def test_original_helper_definitions_identical(self):
        changed = ast.parse((SEBAL/'pysebal_bootstrap.py').read_text())
        original = {n.name: ast.dump(n, include_attributes=False) for n in ORIGINAL_AST.body if isinstance(n, ast.FunctionDef)}
        variant = {n.name: ast.dump(n, include_attributes=False) for n in changed.body if isinstance(n, ast.FunctionDef)}
        for name in original:
            if name != 'SEBALcode':
                self.assertEqual(original[name], variant[name], name)
        function = next(n for n in changed.body if isinstance(n, ast.FunctionDef) and n.name == 'SEBALcode')
        source = ast.unparse(function)
        output_assignments = [n for n in ast.walk(function) if isinstance(n, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == 'output_folder' for t in n.targets)]
        self.assertEqual(len(output_assignments), 1)
        self.assertIn('scratch_dir', ast.unparse(output_assignments[0]))
        self.assertIn('return ensemble_context.process_scene', source)

    def test_whole_scene_matches_original_lower_mean_upper_and_other_pair(self):
        state = synthetic_scene()
        original_inputs = {key: np.copy(value) for key, value in state.items()}
        with contextlib.redirect_stdout(io.StringIO()), np.errstate(all='ignore'):
            for alpha, beta in [(-.201, .245), (-.089, .343), (.023, .441), (.8, .6)]:
                actual = core.downstream_root_zone(state, alpha, beta, FUNCTIONS)
                expected = original_sequence(state, alpha, beta)
                np.testing.assert_array_equal(actual.astype('float32'), expected.astype('float32'))
            first = core.downstream_root_zone(state, -.089, .343, FUNCTIONS)
            core.downstream_root_zone(state, .8, .6, FUNCTIONS)
            np.testing.assert_array_equal(first, core.downstream_root_zone(state, -.089, .343, FUNCTIONS))
        for key in state:
            np.testing.assert_array_equal(state[key], original_inputs[key])

    def test_raw_ef_application_not_silently_clipped(self):
        state = synthetic_scene()
        state['LE_inst'][3, 3] = 430*3
        _, actual = FUNCTIONS[NAMES[0]](state['LE_inst'], state['rn_inst'], state['g_inst'], -.089, .343, True)
        self.assertAlmostEqual(actual[3, 3], -.089*3+.343)

    def test_nested_subsets_keep_joint_pairs(self):
        bank = [dict(replicate=i, alpha=i/1000., beta=1-i/1000.) for i in range(300)]
        first = core.select_coefficients(bank, 100, 23)
        self.assertEqual(first, core.select_coefficients(bank, 200, 23)[:100])
        self.assertTrue(all(p['alpha']+p['beta'] == 1 for p in first))
        with self.assertRaises(ValueError):
            core.select_coefficients(bank, 301)

    def test_pixel_projection_supports_swath_rotation_and_outside(self):
        locations = [dict(gpi='a', longitude=1.9, latitude=2.8, swaths=['149039']),
                     dict(gpi='b', longitude=-3., latitude=20., swaths=[]),
                     dict(gpi='c', longitude=1., latitude=1., swaths=['150039'])]
        # x=col+.2*row, y=.1*col+row. First point falls in (row2,col1).
        p = core.project_locations(locations, '149039', (8, 8), (0, 1, .2, 0, .1, 1), lambda x,y:(x,y))
        self.assertEqual(len(p), 2)
        self.assertEqual((p[0]['pixel_row'], p[0]['pixel_col']), (2, 1))
        self.assertFalse(p[1]['inside_raster'])

    def test_interval_width_nonnegative_with_signed_shift_separate(self):
        summary = core.summary_record(np.array([.1, .2, .3]), .4, True)
        self.assertGreater(summary['interval_width'], 0)
        self.assertLess(summary['median_shift'], 0)
        self.assertEqual(core.summary_record(np.array([0., 0.]), 0., True)['interval_width'], 0)
        self.assertEqual(core.summary_record(np.array([0., 0.]), 0., True)['relative_interval_width'], '')

    def test_missing_or_masked_members_do_not_get_narrow_intervals(self):
        for values, valid in [(np.array([.1, np.nan, .2]), True), (np.array([0., 0., 0.]), False)]:
            summary = core.summary_record(values, .2, valid)
            self.assertFalse(summary['complete_ensemble'])
            self.assertEqual(summary['interval_width'], '')

    def test_reference_mismatch_prevents_publishing_scene(self):
        loc = [dict(gpi='clear', longitude=73.5, latitude=36.5, swaths=['149039'])]
        pairs = [dict(ensemble_order=i+1, replicate=i+1, alpha=-.08, beta=.34) for i in range(2)]
        meta = dict(workbook='example.xlsx', excel_row=2, scene='LC08_L1TP_149039_20210101_X', date='2021-01-01', swath='149039')
        with tempfile.TemporaryDirectory() as directory, patch.object(core, 'gdal_projector', return_value=lambda x,y:(x,y)):
            context = core.EnsembleContext(coefficients=pairs, locations=loc, output=directory)
            with patch.object(context, 'compare_reference', return_value=[dict(gpi='clear', difference=.01)]):
                with self.assertRaisesRegex(ValueError, 'reproduction failed'):
                    context.process_scene(synthetic_scene(), FUNCTIONS, FakeTemplate(), meta)
            self.assertFalse(context.scene_directory(meta).exists())
            self.assertFalse(list(Path(directory).rglob('*.csv.gz')))

    def test_existing_run_settings_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner.initialize_output(root, dict(workflow='sebal_calibration_bootstrap', n=200), False)
            with self.assertRaises(ValueError):
                runner.initialize_output(root, dict(workflow='sebal_calibration_bootstrap', n=500), True)
            with self.assertRaises(ValueError):
                runner.initialize_output(root, dict(workflow='sebal_calibration_bootstrap', n=200), False)
            runner.initialize_output(root, dict(workflow='sebal_calibration_bootstrap', n=200), True)

    def test_scene_outputs_count_flags_float32_and_atomic_no_overwrite(self):
        loc = [dict(gpi='clear', longitude=73.5, latitude=36.5, swaths=['149039']),
               dict(gpi='cloud', longitude=70.5, latitude=39.5, swaths=['149039']),
               dict(gpi='water', longitude=71.5, latitude=39.5, swaths=['149039']),
               dict(gpi='outside', longitude=30., latitude=30., swaths=['149039'])]
        pairs = [dict(ensemble_order=1, replicate=7, alpha=-.08, beta=.34),
                 dict(ensemble_order=2, replicate=3, alpha=-.1, beta=.35)]
        meta = dict(workbook='example.xlsx', excel_row=2, scene='LC08_L1TP_149039_20210101_X', date='2021-01-01', swath='149039')
        with tempfile.TemporaryDirectory() as directory, patch.object(core, 'gdal_projector', return_value=lambda x,y:(x,y)):
            context = core.EnsembleContext(coefficients=pairs, locations=loc, output=directory)
            with contextlib.redirect_stdout(io.StringIO()):
                target = context.process_scene(synthetic_scene(), FUNCTIONS, FakeTemplate(), meta)
            with gzip.open(target/'point_realizations.csv.gz', 'rt') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), len(pairs)*len(loc))
            reasons = {r['gpi']: r['exclusion_reason'] for r in rows}
            self.assertEqual(reasons, {'clear':'', 'cloud':'qc_cloud_or_snow', 'water':'water', 'outside':'outside_raster'})
            self.assertFalse(list(target.glob('*.tif')))
            self.assertTrue(json.loads((target/'scene_manifest.json').read_text())['complete'])
            with self.assertRaises(FileExistsError):
                context.process_scene(synthetic_scene(), FUNCTIONS, FakeTemplate(), meta)
            for row in rows:
                if row['rzsm']:
                    self.assertEqual(float(row['rzsm']), float(np.float32(float(row['rzsm']))))
            runner.combine_tables(Path(directory), [(meta, target)])
            with gzip.open(Path(directory)/'by_workbook'/'example_point_realizations.csv.gz', 'rt') as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 8)

    def test_real_workbooks_have_96_rows_and_preflight_checks_inputs(self):
        for swath in ['149039', '150039']:
            rows = runner.scene_rows(SEBAL/f'InputEXCEL_LBDC_{swath}.xlsx')
            self.assertEqual(len(rows), 96)
            self.assertEqual({r['metadata']['swath'] for r in rows}, {swath})
            self.assertFalse(any('VIIRS_' in p for r in rows for p in r['required_paths']))
            locations = [dict(gpi='a', swaths=[swath])]
            check = runner.preflight(rows, locations)
            self.assertEqual(check['n_scenes'], 96)
            # Current workbook inputs are on the original Windows/NAS machine.
            self.assertFalse(check['ready'])
            self.assertGreater(check['n_missing_paths'], 0)


if __name__ == '__main__':
    unittest.main()
