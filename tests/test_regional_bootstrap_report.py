import unittest
import numpy as np
import pandas as pd

from modules.regional_bootstrap_report import calculate_results


def data():
    summaries, members = [], []
    for gpi, pixel, central in [('a', 0, .1), ('b', 1, .2), ('c', 1, .2), ('d', 2, .3)]:
        values = np.array([central-.01, central, central+.01, central+.02])
        low, median, high = np.percentile(values, [2.5, 50, 97.5])
        identity = dict(scene='scene', swath='149039', date='2020-09-10', excel_row=2, gpi=gpi,
                        pixel_row=0, pixel_col=pixel, inside_raster=True, mask_valid=True)
        summaries.append(dict(**identity, n_realizations=4, n_finite=4, n_valid=4,
                              central_rzsm=central, complete_ensemble=True,
                              ensemble_mean=values.mean(), ensemble_median=median,
                              ensemble_sd=values.std(ddof=1), p025=low, p975=high,
                              interval_width=high-low, median_shift=median-central,
                              relative_interval_width=(high-low)/central))
        for order, value in enumerate(values, 1):
            members.append(dict(**identity, ensemble_order=order, replicate=order,
                                alpha=-.01*order, beta=.2+.01*order, rzsm=value, valid_rzsm=True))
    return pd.DataFrame(summaries), pd.DataFrame(members)


class ReportingTests(unittest.TestCase):
    def test_shared_pixels_count_once_and_scene_usr_uses_spatial_iqr(self):
        summary, members = data()
        pixels, daily, ranks, convergence, errors = calculate_results(summary, members)
        self.assertEqual(len(pixels), 3)
        self.assertEqual(pixels.set_index('gpi').loc['b', 'sensor_ids'], 'b;c')
        self.assertEqual(daily.iloc[0].n_valid_sensor_cases, 4)
        self.assertAlmostEqual(daily.iloc[0].central_spatial_iqr, .1)
        self.assertAlmostEqual(daily.iloc[0].usr, np.std([-.01, 0, .01, .02], ddof=1)/.1)
        np.testing.assert_allclose(ranks.spearman_vs_central, 1)
        self.assertLess(max(errors.values()), 1e-10)

    def test_false_small_interval_rejected_by_recomputation(self):
        summary, members = data()
        summary.loc[0, 'interval_width'] = 0
        with self.assertRaisesRegex(ValueError, 'does not reproduce'):
            calculate_results(summary, members)

    def test_nonjoint_parameters_rejected(self):
        summary, members = data()
        members.loc[0, 'alpha'] += 1
        with self.assertRaisesRegex(ValueError, 'calibration pairs differ'):
            calculate_results(summary, members)

    def test_missing_member_rejected(self):
        summary, members = data()
        with self.assertRaisesRegex(ValueError, 'member count differs'):
            calculate_results(summary, members.iloc[1:])


if __name__ == '__main__':
    unittest.main()
