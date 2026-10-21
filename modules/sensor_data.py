"""Shared date normalization used by the unfiltered daily-record analysis."""
import pandas as pd


def to_daily_frame(dates, values, value_col):
    d = pd.to_datetime(pd.Series(list(dates)), errors='coerce').dt.normalize()
    v = pd.to_numeric(pd.Series(list(values)), errors='coerce')
    frame = pd.DataFrame({'date': d, value_col: v}).dropna()
    return frame.drop_duplicates('date').sort_values('date').reset_index(drop=True)
