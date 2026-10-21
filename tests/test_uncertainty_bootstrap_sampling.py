"""Scientific checks for paired acquisition sampling and coordinate-only inputs."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from modules.uncertainty_bootstrap_sampling import bootstrap_bank, draw_bootstrap_sample, export_locations, fit_line


def example_pairs():
    records = []
    for stratum, sites in [('Idaho', ['one', 'two', 'three']), ('Arizona', ['four'])]:
        for scene in range(5):
            for i, site in enumerate(sites):
                x = .12 + .21*scene + .03*i
                records.append(dict(site=site, stratum=stratum, acquisition_group=f'{stratum}_{scene}',
                                    ef_model=x, ef_tower=.32-.1*x+.004*scene*i))
    return pd.DataFrame(records)


class CalibrationSamplingTests(unittest.TestCase):
    def test_acquisition_blocks_preserve_all_simultaneous_sites(self):
        pairs = example_pairs()
        sample, draws = draw_bootstrap_sample(pairs, np.random.default_rng(2))
        self.assertEqual(len(draws), 10)
        for draw in draws:
            actual = sample[(sample.stratum == draw['stratum']) & (sample.draw_slot == draw['draw_slot'])]
            expected = pairs[pairs.acquisition_group == draw['acquisition_group']]
            self.assertEqual(set(actual.site), set(expected.site))
            self.assertEqual(len(actual), len(expected))
        self.assertEqual(sample.groupby('stratum').size().to_dict(), {'Arizona': 5, 'Idaho': 15})

    def test_refits_reproducible_and_extendable(self):
        pairs = example_pairs()
        a, draws = bootstrap_bank(pairs, 20, 17)
        b, _ = bootstrap_bank(pairs, 25, 17)
        pd.testing.assert_frame_equal(a, b.iloc[:20].reset_index(drop=True))
        sample, _ = draw_bootstrap_sample(pairs, np.random.default_rng(17))
        fit = fit_line(sample.ef_model, sample.ef_tower)
        self.assertAlmostEqual(a.iloc[0].alpha, fit['alpha'])
        self.assertAlmostEqual(a.iloc[0].beta, fit['beta'])
        self.assertEqual(len(draws), 20*10)

    def test_observations_not_required_or_read_for_locations(self):
        coordinates = pd.DataFrame(dict(gpi=[10, 10, 11], row=['149039', '150039', '150039'],
                                        latitude=[30., 30., 30.], longitude=[72., 72., 72.]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'locations.csv'
            coordinates.to_csv(path, index=False)
            first = export_locations(path)
            coordinates['observed'] = ['INVALID', 'DO NOT READ', 'NaN']
            coordinates.to_csv(path, index=False)
            pd.testing.assert_frame_equal(first, export_locations(path))
        self.assertEqual(len(first), 2)  # retain colocated deployment identities
        self.assertEqual(first.iloc[0].swaths, '149039;150039')

    def test_degenerate_fit_and_ambiguous_locations_rejected(self):
        with self.assertRaises(ValueError):
            fit_line([1, 1, 1], [.1, .2, .3])
        with self.assertRaises(ValueError):
            bootstrap_bank(example_pairs(), 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'locations.csv'
            pd.DataFrame(dict(gpi=[1, 1], row=['149039']*2, latitude=[30, 31], longitude=[72, 72])).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'inconsistent'):
                export_locations(path)


if __name__ == '__main__':
    unittest.main()
