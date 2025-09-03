
"""
Extra File

Load AmeriFlux daily CSV, clean it, and subset to a date range and desired variables. 
This data is meant for SEBAL (meteo) as well as for calibrations (energy modules)

Returns
-------
pd.DataFrame
    Cleaned dataframe indexed by DATE with the selected variables.

Raises
------
ValueError
    If none of the expected key variables are present in the CSV.
"""


import pandas as pd
import numpy as np

# Column definitions
COL_LE  = "LE_F_MDS"     # latent heat flux (W m-2), gap-filled
COL_H   = "H_F_MDS"      # sensible heat flux (W m-2)
COL_RN  = "NETRAD"       # net radiation (W m-2)
COL_G   = "G_F_MDS"      # soil heat flux (W m-2)


def load_flux_data(csv_path, initial_date, final_date, key_vars=None):
    """
    Load AmeriFlux daily CSV, clean it, and subset to a date range and desired variables.

    Parameters
    ----------
    csv_path : str
        Path to the AmeriFlux CSV file.
    initial_date : str
        Start date in YYYY-MM-DD format.
    final_date : str
        End date in YYYY-MM-DD format.
    key_vars : list of str, optional
        Variables to keep. Defaults to [COL_LE, COL_H, COL_RN, COL_G].

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe indexed by DATE with the selected variables.
    """
    if key_vars is None:
        key_vars = [COL_LE, COL_H, COL_RN, COL_G]

    data = pd.read_csv(csv_path)

    # Parse date
    data['DATE'] = pd.to_datetime(data['TIMESTAMP'], format='%Y%m%d', errors='coerce')

    # Replace missing codes with NaN
    data.replace(-9999, np.nan, inplace=True)

    # Ensure requested vars exist
    present = [c for c in key_vars if c in data.columns]
    if not present:
        raise ValueError("None of the expected key variables are present in the CSV.")

    # Keep only DATE + variables
    df = data[['DATE'] + present].copy()

    # Drop rows with NaNs in the key variables
    df = df.dropna(subset=present)

    # Index by DATE and subset
    df.set_index('DATE', inplace=True)
    df = df.loc[initial_date:final_date].copy()

    return df


def compare_le(df):
    """
    Compute LE from energy balance and compare with observed LE.
    Also computes energy balance closure ratio and gap.
    """

    # Compute LE from energy balance
    df['LE_est_Wm2'] = df[COL_RN] - df[COL_G] - df[COL_H]
    df['LE_obs_Wm2'] = df[COL_LE]
    df['LE_diff_Wm2'] = df['LE_est_Wm2'] - df['LE_obs_Wm2']

    # Convert to ET (mm/day)
    rho_w, Lv, sec_day = 1000.0, 2.45e6, 86400.0
    df['ET_est_mm_d'] = (df['LE_est_Wm2'] / (rho_w * Lv)) * sec_day
    df['ET_obs_mm_d'] = (df['LE_obs_Wm2'] / (rho_w * Lv)) * sec_day
    df['ET_diff_mm_d'] = df['ET_est_mm_d'] - df['ET_obs_mm_d']

    # Energy balance closure: (H + LE_obs) / (Rn - G)
    df['closure_ratio'] = (df[COL_H] + df[COL_LE]) / (df[COL_RN] - df[COL_G])
    df['closure_gap'] = 1 - df['closure_ratio']  # positive means under-closure

    # Quick stats
    rmse = np.sqrt(np.nanmean((df['LE_est_Wm2'] - df['LE_obs_Wm2']) ** 2))
    bias = np.nanmean(df['LE_est_Wm2'] - df['LE_obs_Wm2'])
    closure_mean = np.nanmean(df['closure_ratio'])
    closure_std = np.nanstd(df['closure_ratio'])

    print(f"RMSE (W/m²): {rmse:.2f}, Bias (W/m²): {bias:.2f}")
    print(f"Rows compared: {len(df)}")
    print(f"Closure ratio mean: {closure_mean:.3f} ± {closure_std:.3f}")
    print(f"Typical closure gap: {(1-closure_mean)*100:.1f}%")

    return df


# ===== Example usage =====
csv_path = r"ec_flux\Ameriflux\BsH\AMF_US-Rls_FLUXNET_SUBSET_2014-2023_5-7\AMF_US-Rls_FLUXNET_SUBSET_DD_2014-2023_5-7.csv"
initial_date = '2021-02-01'
final_date   = '2021-12-01'

df_clean = load_flux_data(csv_path, initial_date, final_date)
df_compared = compare_le(df_clean)
print(df_compared[['LE_obs_Wm2','LE_est_Wm2','closure_ratio','closure_gap']].head())

