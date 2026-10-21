"""Input and formula guards for historical climate-result reproduction."""
import unittest
import numpy as np
import pandas as pd
from modules.climate_analog import calculate, YEARS, VARIABLES


def annual_input():
    # Orthogonal, centered annual descriptors give identity sample covariance.
    rng = np.random.default_rng(12)
    matrix = rng.normal(size=(30, 4))
    matrix -= matrix.mean(axis=0)
    basis = np.linalg.qr(matrix)[0] * np.sqrt(29)
    records = []
    for site, offset in [('Bari_Doab', 0.), ('Candidate', 4.)]:
        row = dict(Site=site, AI=1. if not offset else 2.)
        for j, variable in enumerate(VARIABLES):
            for year, value in zip(YEARS, basis[:, j] + (offset if j == 0 else 0.)):
                row[f'{variable}_{year}'] = value
        records.append(row)
    return pd.DataFrame(records)


class ClimateTests(unittest.TestCase):
    def test_historical_composite_is_distinct_from_quadratic_alternative(self):
        result, metadata = calculate(annual_input())
        # Q=16, z_AI**2=2: legacy sqrt(4+2), alternative sqrt(16+2).
        self.assertAlmostEqual(result.iloc[1].Mahalanobis_D, np.sqrt(6))
        self.assertEqual(result.iloc[0].Mahalanobis_D, 0.)
        self.assertAlmostEqual(metadata['quadratic_plus_aridity_diagnostic']['Candidate'], np.sqrt(18))
        self.assertIn('not a conventional', metadata['method_note'])

    def test_column_order_and_earth_engine_prefixes_do_not_change_results(self):
        data = annual_input()
        expected, _ = calculate(data)
        shuffled = data.sample(frac=1, axis=1, random_state=42)
        shuffled = shuffled.rename(columns={c: f'12_{c}' for c in data if c not in ['Site', 'AI']})
        actual, _ = calculate(shuffled)
        np.testing.assert_array_equal(actual.Mahalanobis_D, expected.Mahalanobis_D)

    def test_summary_csv_is_not_an_annual_input(self):
        with self.assertRaisesRegex(ValueError, 'annual-input CSV'):
            calculate(annual_input().drop(columns='VPD_2000'))

    def test_duplicate_and_nonfinite_inputs_rejected(self):
        data = annual_input()
        data.loc[1, 'Site'] = 'Bari_Doab'
        with self.assertRaisesRegex(ValueError, 'unique'):
            calculate(data)
        data = annual_input()
        data.loc[1, 'VPD_2000'] = np.nan
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            calculate(data)
        data = annual_input().assign(AI=1.)
        with self.assertRaisesRegex(ValueError, 'positive between-site'):
            calculate(data)


if __name__ == '__main__':
    unittest.main()
