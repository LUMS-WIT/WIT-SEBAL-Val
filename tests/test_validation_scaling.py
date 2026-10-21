"""Scientific invariants of location-held-out scaling and legacy-format exports."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from modules.validation_scaling_module import (
    cross_validate, geographic_folds, export_validation_results, run_validation_scaling,
)


def example_pairs():
    records = []
    for gpi in range(11):
        # The last two GPIs are distinct deployments at the same coordinates.
        loc = min(gpi, 9)
        for day in range(4):
            raw = .05 + .008 * gpi + .013 * day
            records.append(dict(gpi=gpi, row="149039", date=pd.Timestamp(2020, 1, day + 1),
                                latitude=29 + loc * .1, longitude=71 + loc * .13,
                                raw=raw, observed=2 * raw + .1))
    # One GPI appears in both satellite rows, as in the real data.
    records.append({**records[0], "row": "150039", "raw": .06, "observed": .22})
    return pd.DataFrame(records)


class FiveFoldTests(unittest.TestCase):
    def test_grouping_and_coordinate_only_assignment(self):
        pairs = example_pairs()
        assigned, locations = geographic_folds(pairs)
        changed = pairs.sample(frac=1, random_state=4).copy()
        changed["observed"] = np.arange(len(changed)) * .3
        again, _ = geographic_folds(changed)
        pd.testing.assert_series_equal(assigned.groupby("gpi").fold.first(),
                                       again.groupby("gpi").fold.first())
        self.assertEqual(locations.groupby("fold").size().tolist(), [2] * 5)
        self.assertTrue(assigned.groupby("gpi").fold.nunique().eq(1).all())
        self.assertTrue(assigned.groupby(["latitude", "longitude"]).fold.nunique().eq(1).all())

    def test_known_transform_and_exact_coverage(self):
        pairs = example_pairs()
        held, _, folds, sites, _ = cross_validate(pairs)
        np.testing.assert_allclose(held.scaled, held.observed, atol=1e-14)
        np.testing.assert_allclose(folds.slope, 2)
        np.testing.assert_allclose(folds.intercept, .1)
        self.assertEqual(len(held), len(pairs))
        self.assertFalse(held.duplicated(["gpi", "row", "date"]).any())
        for field in ["Pearson", "Spearman"]:
            table = sites.pivot(index="gpi", columns="method", values=field)
            np.testing.assert_allclose(table.raw, table.scaled, atol=1e-12)

    def test_heldout_observations_do_not_enter_fit(self):
        pairs = example_pairs()
        baseline, _, _, _, _ = cross_validate(pairs)
        test_gpis = baseline.loc[baseline.fold.eq(1), "gpi"].unique()
        changed = pairs.copy()
        changed.loc[changed.gpi.isin(test_gpis), "observed"] += 100
        held, _, _, _, _ = cross_validate(changed)
        first = baseline[baseline.fold.eq(1)]
        second = held[held.fold.eq(1)]
        np.testing.assert_allclose(first[["slope", "intercept", "scaled"]],
                                   second[["slope", "intercept", "scaled"]])

    def test_invalid_inputs_fail_instead_of_silent_drops(self):
        pairs = example_pairs()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            cross_validate(pd.concat([pairs, pairs.iloc[:1]]))
        with self.assertRaisesRegex(ValueError, "Fold count"):
            cross_validate(pairs, n_folds=11)
        with self.assertRaisesRegex(ValueError, "standard deviations"):
            cross_validate(pairs.assign(raw=.1))
        with self.assertRaisesRegex(ValueError, "finite"):
            cross_validate(pairs.assign(raw=np.inf))

    def test_export_has_no_many_to_many_join_or_performance_filter(self):
        pairs = example_pairs()
        # Retain a negative-correlation GPI and a one-pair satellite-row series.
        mask = pairs.gpi.eq(0) & pairs.row.eq("149039")
        pairs.loc[mask, "observed"] = [.4, .3, .2, .1]
        held, _, _, sites, _ = cross_validate(pairs)
        metadata = pairs[["gpi", "latitude", "longitude"]].drop_duplicates().assign(
            start_date="2020-01-01", end_date="2020-01-04", count=4, overlaps=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            import metrics
            from modules.validation_scaling_module import error_metrics
            with patch('utils.plot_box_and_whiskers') as box_plot, patch('utils.plot_metric_with_ci') as ci_plot:
                export_validation_results(held, metadata, root, ["149039", "150039"], "mean", 0)
            for method in ["unscaled", "scaled"]:
                result = root / method / "results/validations_tw_0.xlsx"
                with pd.ExcelFile(result) as workbook:
                    self.assertEqual(workbook.sheet_names, ["MetaData", "Summary"])
                table = pd.read_excel(result, sheet_name="MetaData")
                summary = pd.read_excel(result, sheet_name="Summary")
                self.assertEqual(len(table), 11)  # one record per GPI across both rows
                self.assertTrue(table.gpi.is_unique)
                self.assertEqual(table.overlaps.sum(), len(pairs))
                self.assertEqual(summary.iloc[0]["mean"], len(pairs))
                self.assertTrue((table.p_rho.dropna() < 0).any())
                valid = table.dropna(subset=["bias_cl", "bias_cu"])
                np.testing.assert_allclose((valid.bias_cl + valid.bias_cu) / 2, valid.bias)
                one = pd.read_excel(root / method / 'results/validations_150039_tw_0.xlsx',
                                    sheet_name='MetaData').iloc[0]
                self.assertTrue(np.isnan(one.p_rho))
                self.assertTrue(np.isfinite(one.mse))
                prediction_column = 'scaled' if method == 'scaled' else 'raw'
                expected = sites[sites.method.eq(prediction_column)].sort_values('gpi')
                for key, field in [('bias', 'bias'), ('mse', 'MSE'), ('ubrmsd', 'ubRMSD')]:
                    np.testing.assert_allclose(table.sort_values('gpi')[key], expected[field], atol=1e-12)
                    np.testing.assert_allclose(summary.set_index('Metric').loc[key, 'median'],
                                               expected[field].median(), atol=1e-12)
                merged = table[table.gpi.eq(0)].iloc[0]
                site = held[held.gpi.eq(0)]
                x, y = site[prediction_column].to_numpy(), site.observed.to_numpy()
                scores = error_metrics(x, y)
                self.assertEqual(merged.overlaps, 5)
                self.assertEqual(merged['count'], 4)  # sensor metadata is not doubled
                np.testing.assert_allclose([merged.bias_cl, merged.bias_cu], metrics.bias_ci(x, y, scores['bias']))
                np.testing.assert_allclose([merged.ubrmsd_cl, merged.ubrmsd_cu], metrics.ubrmsd_ci(x, y, scores['ubRMSD']))
                for row in ['149039', '150039']:
                    row_table = pd.read_excel(root / method / 'results' / f'validations_{row}_tw_0.xlsx')
                    for record in row_table.to_dict('records'):
                        subset = held[held.gpi.eq(record['gpi']) & held.row.eq(row)]
                        score = error_metrics(subset[prediction_column], subset.observed)
                        np.testing.assert_allclose(record['mse'], score['MSE'])
                combined_boxes = [call.args[0] for call in box_plot.call_args_list
                                  if f'/{method}/figs/boxplot_mean_149039_150039' in Path(call.kwargs['filename']).as_posix()]
                self.assertEqual(len(combined_boxes), 1)
                for key in ['bias', 'mse', 'ubrmsd']:
                    np.testing.assert_allclose(combined_boxes[0][key], table[key])
                combined_cis = [call for call in ci_plot.call_args_list
                                if f'/{method}/figs/ci_' in Path(call.kwargs['filename']).as_posix()]
                self.assertEqual(len(combined_cis), 2)
                for call in combined_cis:
                    key = call.kwargs['metric']
                    for field in [key, key + '_cl', key + '_cu']:
                        np.testing.assert_allclose(call.args[0][field], table[field])

    def test_failed_run_preserves_previous_output(self):
        pairs = example_pairs()
        metadata = pairs[["gpi", "latitude", "longitude"]].drop_duplicates()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            output.mkdir()
            (output / "summary.json").write_text('{"previous": true}')
            (output / "manifest.json").write_text(json.dumps({
                "workflow": "validation_scaling", "generated_files": ["summary.json"]}))
            with patch('modules.validation_scaling_module.extract_pairs',
                       return_value=(pairs, metadata, {})), patch(
                           'modules.validation_scaling_module.export_validation_results',
                           side_effect=RuntimeError("Export failed")):
                with self.assertRaisesRegex(RuntimeError, "Export failed"):
                    run_validation_scaling(wit_sms_path="unused", raster_base="unused",
                                           output_base=output, cohort_file=None, rows=["149039", "150039"])
            self.assertEqual((output / "summary.json").read_text(), '{"previous": true}')
            self.assertEqual(list(Path(directory).iterdir()), [output])
            (output / "manual_notes.md").write_text("Keep this")
            with self.assertRaisesRegex(ValueError, "manually added"):
                run_validation_scaling(wit_sms_path="unused", raster_base="unused",
                                       output_base=output, cohort_file=None, rows=["149039"])


if __name__ == "__main__":
    unittest.main()
