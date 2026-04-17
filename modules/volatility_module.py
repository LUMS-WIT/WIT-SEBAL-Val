from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import pandas as pd

from utils import SoilMoistureData


@dataclass(frozen=True)
class SiteId:
    gpi: str
    lat: Optional[float] = None
    lon: Optional[float] = None


# Matches: sebal_149039_mean_witgpi_2021052_lat_30.892122_lon_73.374222.xlsx
_OVERLAP_FNAME_RE = re.compile(
    r"witgpi_(?P<gpi>[^_]+)_lat_(?P<lat>-?\d+(\.\d+)?)_lon_(?P<lon>-?\d+(\.\d+)?)",
    re.IGNORECASE,
)


def parse_site_id_from_filename(path: Path) -> SiteId:
    m = _OVERLAP_FNAME_RE.search(path.name)
    if not m:
        return SiteId(gpi="unknown", lat=None, lon=None)
    return SiteId(gpi=str(m.group("gpi")), lat=float(m.group("lat")), lon=float(m.group("lon")))


def _norm_dt(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    return out.dt.normalize()


def read_model_series_from_overlap_file(overlap_xlsx: Path) -> tuple[SiteId, pd.DataFrame]:
    """
    Reads SEBAL (model) dates and values from an overlap-point XLSX.

    Returns:
      site_id: SiteId from filename
      model_df: columns [date, y] where date is normalized datetime64 and y is float
    """
    df = pd.read_excel(overlap_xlsx)

    if "Timestamp" not in df.columns or "sebal_sm" not in df.columns:
        raise ValueError(f"Missing Timestamp/sebal_sm in {overlap_xlsx}")

    model_df = pd.DataFrame(
        {
            "date": _norm_dt(df["Timestamp"]),
            "y": pd.to_numeric(df["sebal_sm"], errors="coerce"),
        }
    ).dropna(subset=["date", "y"])

    model_df = model_df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    return parse_site_id_from_filename(overlap_xlsx), model_df


def load_daily_wit_series(soil: SoilMoistureData, lat: float, lon: float) -> pd.DataFrame:
    """
    Load DAILY in-situ WIT series for one site (no interpolation).
    Returns df columns: [date, x]
    """
    dates, values = soil.get_soil_moisture_by_location(lat, lon)

    d = pd.to_datetime(pd.Series(list(dates)), errors="coerce").dt.normalize()
    x = pd.to_numeric(pd.Series(list(values)), errors="coerce")

    out = pd.DataFrame({"date": d, "x": x}).dropna(subset=["date"])
    out = out.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out


def compute_inter_overpass_windows(
    *,
    sensor_daily_df: pd.DataFrame,   # [date, x]
    model_df: pd.DataFrame,          # [date, y] sparse (SEBAL dates)
    min_valid_increments: int = 2,
    epsilon: float = 1e-9,
) -> pd.DataFrame:
    """
    For each consecutive pair of SEBAL dates (tau_k, tau_{k+1}), compute
    daily-increment variation metrics on the WIT daily series inside the window.

    Gap rule:
      Only compute increment x(t)-x(t-1) if both days exist and are consecutive (1-day gap).
    """
    if model_df is None or len(model_df) < 2:
        return pd.DataFrame()

    sensor_map = {
        d.date(): float(v)
        for d, v in zip(sensor_daily_df["date"], sensor_daily_df["x"])
        if pd.notna(v)
    }

    model_df = model_df.dropna(subset=["date", "y"]).sort_values("date").reset_index(drop=True)

    rows = []
    model_dates = list(model_df["date"])
    model_vals = list(model_df["y"])

    for k in range(len(model_dates) - 1):
        start_dt = model_dates[k]
        end_dt = model_dates[k + 1]
        if end_dt <= start_dt:
            continue

        y_start = float(model_vals[k])
        y_end = float(model_vals[k + 1])
        delta_y = float(y_end - y_start)

        w = sensor_daily_df[(sensor_daily_df["date"] >= start_dt) & (sensor_daily_df["date"] <= end_dt)].copy()
        w = w.dropna(subset=["x"]).sort_values("date").reset_index(drop=True)

        incs = []
        for i in range(1, len(w)):
            d0 = w.loc[i - 1, "date"].date()
            d1 = w.loc[i, "date"].date()
            if (d1 - d0).days == 1:
                incs.append(float(w.loc[i, "x"] - w.loc[i - 1, "x"]))

        m = len(incs)
        if m < min_valid_increments:
            continue

        incs = np.asarray(incs, dtype=float)
        AV = float(np.nansum(np.abs(incs)))
        QV = float(np.nansum(incs ** 2))
        AV_norm = AV / m
        QV_norm = QV / m

        x_start = sensor_map.get(start_dt.date(), np.nan)
        x_end = sensor_map.get(end_dt.date(), np.nan)
        if np.isfinite(x_start) and np.isfinite(x_end) and AV > 0:
            delta_x_end = float(x_end - x_start)
            Missed = float(AV - abs(delta_x_end))
            Missed_norm = float((AV - abs(delta_x_end)) / (AV + epsilon))
        else:
            delta_x_end = np.nan
            Missed = np.nan
            Missed_norm = np.nan

        rows.append(
            dict(
                start_date=start_dt.date(),
                end_date=end_dt.date(),
                window_length_days=int((end_dt.date() - start_dt.date()).days),
                y_start=y_start,
                y_end=y_end,
                delta_y=delta_y,
                x_start=x_start,
                x_end=x_end,
                delta_x_end=delta_x_end,
                valid_increments=int(m),
                AV=AV,
                QV=QV,
                AV_norm=AV_norm,
                QV_norm=QV_norm,
                Missed=Missed,
                Missed_norm=Missed_norm,
            )
        )

    return pd.DataFrame(rows)


def summarize_site_windows(
    *,
    site_id: SiteId,
    row_path: str,
    raster_stat: str,
    windows_df: pd.DataFrame,
    overlap_file: Path,
) -> dict:
    if windows_df is None or windows_df.empty:
        return dict(
            gpi=site_id.gpi,
            latitude=site_id.lat,
            longitude=site_id.lon,
            row_path=row_path,
            raster_stat=raster_stat,
            overlap_file=str(overlap_file),
            number_of_windows=0,
            total_valid_increment_count=0,
            mean_window_length_days=np.nan,
            median_window_length_days=np.nan,
            median_AV_norm=np.nan,
            median_QV_norm=np.nan,
            median_Missed_norm=np.nan,
        )

    return dict(
        gpi=site_id.gpi,
        latitude=site_id.lat,
        longitude=site_id.lon,
        row_path=row_path,
        raster_stat=raster_stat,
        overlap_file=str(overlap_file),
        number_of_windows=int(len(windows_df)),
        total_valid_increment_count=int(windows_df["valid_increments"].sum()),
        mean_window_length_days=float(windows_df["window_length_days"].mean()),
        median_window_length_days=float(windows_df["window_length_days"].median()),
        median_AV_norm=float(windows_df["AV_norm"].median()),
        median_QV_norm=float(windows_df["QV_norm"].median()),
        median_Missed_norm=float(windows_df["Missed_norm"].median(skipna=True)),
    )


def load_validation_metadata_metrics(validation_xlsx: Path) -> pd.DataFrame:
    """
    Load per-site validation metrics from MetaData sheet.
    Deduplicate by gpi (keep the row with highest overlaps if available).
    """
    df = pd.read_excel(validation_xlsx, sheet_name="MetaData")
    if "gpi" not in df.columns:
        raise ValueError(f"MetaData missing gpi in {validation_xlsx}")
    df["gpi"] = df["gpi"].astype(str)

    if "overlaps" in df.columns:
        df = df.sort_values(["gpi", "overlaps"], ascending=[True, False]).drop_duplicates("gpi", keep="first")
    else:
        df = df.drop_duplicates("gpi", keep="first")

    return df


def maybe_add_delta_r(
    merged_df: pd.DataFrame,
    *,
    validations_results_dir: Path,
    row_path: str,
    baseline_tw: int = 0,
    candidate_tws: Iterable[int] = (3, 5, 7),
) -> pd.DataFrame:
    """
    Optional: add delta_r_3/5/7 columns if corresponding validation workbooks exist.
    """
    out = merged_df.copy()
    out["gpi"] = out["gpi"].astype(str)

    base = validations_results_dir / f"validations_{row_path}_tw_{baseline_tw}.xlsx"
    if not base.exists():
        return out

    base_df = pd.read_excel(base, sheet_name="MetaData")[["gpi", "p_rho"]].copy()
    base_df["gpi"] = base_df["gpi"].astype(str)
    base_df = base_df.drop_duplicates("gpi", keep="first").rename(columns={"p_rho": f"p_rho_tw{baseline_tw}"})
    out = out.merge(base_df, on="gpi", how="left")

    for tw in candidate_tws:
        p = validations_results_dir / f"validations_{row_path}_tw_{tw}.xlsx"
        if not p.exists():
            continue
        df_tw = pd.read_excel(p, sheet_name="MetaData")[["gpi", "p_rho"]].copy()
        df_tw["gpi"] = df_tw["gpi"].astype(str)
        df_tw = df_tw.drop_duplicates("gpi", keep="first").rename(columns={"p_rho": f"p_rho_tw{tw}"})
        out = out.merge(df_tw, on="gpi", how="left")
        out[f"delta_r_{tw}"] = out[f"p_rho_tw{tw}"] - out[f"p_rho_tw{baseline_tw}"]

    return out


def run_case_from_files(
    *,
    overlap_files: list[Path],
    wit_sms_path: str,
    row_path_label: str,
    raster_stat: str,
    min_valid_increments: int,
    epsilon: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Same as run_case, but the caller supplies overlap_files (used for COMBINED runs).
    """
    soil = SoilMoistureData(wit_sms_path)
    soil.read_data()

    window_all = []
    site_all = []

    for f in overlap_files:
        try:
            site_id, model_df = read_model_series_from_overlap_file(f)
        except Exception as e:
            print(f"[volatility] SKIP {f.name}: cannot read SEBAL series ({e})")
            continue

        if site_id.lat is None or site_id.lon is None:
            print(f"[volatility] SKIP {f.name}: missing lat/lon in filename")
            continue

        sensor_daily_df = load_daily_wit_series(soil, site_id.lat, site_id.lon)

        windows_df = compute_inter_overpass_windows(
            sensor_daily_df=sensor_daily_df,
            model_df=model_df,
            min_valid_increments=min_valid_increments,
            epsilon=epsilon,
        )

        if not windows_df.empty:
            windows_df.insert(0, "gpi", site_id.gpi)
            windows_df.insert(1, "latitude", site_id.lat)
            windows_df.insert(2, "longitude", site_id.lon)
            windows_df.insert(3, "row_path", row_path_label)
            windows_df.insert(4, "raster_stat", raster_stat)
            windows_df.insert(5, "overlap_file", str(f))
            window_all.append(windows_df)

        site_all.append(
            summarize_site_windows(
                site_id=site_id,
                row_path=row_path_label,
                raster_stat=raster_stat,
                windows_df=windows_df,
                overlap_file=f,
            )
        )

    window_df = pd.concat(window_all, ignore_index=True) if window_all else pd.DataFrame()
    site_df = pd.DataFrame(site_all) if site_all else pd.DataFrame()
    return window_df, site_df


def run_case(
    *,
    overlap_folder: Path,
    wit_sms_path: str,
    row_path: str,
    raster_stat: str,
    min_valid_increments: int,
    epsilon: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overlap_files = sorted(overlap_folder.glob("*.xlsx"))
    return run_case_from_files(
        overlap_files=overlap_files,
        wit_sms_path=wit_sms_path,
        row_path_label=row_path,
        raster_stat=raster_stat,
        min_valid_increments=min_valid_increments,
        epsilon=epsilon,
    )