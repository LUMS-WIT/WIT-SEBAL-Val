from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from utils import SoilMoistureData, SebalSoilMoistureData, remove_nan_entries
from sms_calibration import sms_calibrations
from scaling import scaling


@dataclass(frozen=True)
class SiteMeta:
    gpi: str
    lat: float
    lon: float


def _to_df(dates, values, value_col: str) -> pd.DataFrame:
    d = pd.to_datetime(pd.Series(list(dates)), errors="coerce").dt.normalize()
    v = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    df = pd.DataFrame({"date": d, value_col: v})
    df = df.dropna(subset=["date", value_col])
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def _safe_corr(x: pd.Series, y: pd.Series, method: str = "pearson") -> float:
    d = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 3:
        return np.nan
    return float(d["x"].corr(d["y"], method=method))


def _rmse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((y - x) ** 2))) if len(x) else np.nan


def _bias(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.nanmean(y - x)) if len(x) else np.nan


def _ubrmsd(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0:
        return np.nan
    xd = x - np.nanmean(x)
    yd = y - np.nanmean(y)
    return float(np.sqrt(np.nanmean((yd - xd) ** 2)))


def load_site_sensor_daily(
    soil: SoilMoistureData,
    lat: float,
    lon: float,
    apply_sms_calibration: bool = True,
) -> pd.DataFrame:
    dates, values = soil.get_soil_moisture_by_location(lat, lon)
    dates, values = remove_nan_entries(dates, values)

    data = (dates, values)
    if apply_sms_calibration:
        data = sms_calibrations(data)

    sensor_df = _to_df(data[0], data[1], "x")
    return sensor_df


def load_site_model_full(
    raster_data: SebalSoilMoistureData,
    lat: float,
    lon: float,
) -> pd.DataFrame:
    dates, values = raster_data.get_data(lat, lon)
    if dates is None or values is None:
        return pd.DataFrame(columns=["date", "y"])

    dates, values = remove_nan_entries(dates, values)
    model_df = _to_df(dates, values, "y")
    return model_df


def maybe_rescale_model_to_sensor(
    sensor_df: pd.DataFrame,
    model_df: pd.DataFrame,
    apply_rescaling: bool = True,
) -> pd.DataFrame:
    if not apply_rescaling or sensor_df.empty or model_df.empty:
        return model_df

    sensor_data = (list(sensor_df["date"]), list(sensor_df["x"]))
    model_data = (list(model_df["date"]), list(model_df["y"]))

    try:
        scaled_model_data, _ = scaling(model_data, sensor_data)
        scaled_df = _to_df(scaled_model_data[0], scaled_model_data[1], "y")
        return scaled_df
    except Exception:
        return model_df


def _sensor_window_mean(
    sensor_map: dict,
    center_date: pd.Timestamp,
    total_window: int,
) -> tuple[float, int]:
    if total_window == 0:
        v = sensor_map.get(center_date.date(), np.nan)
        return float(v) if np.isfinite(v) else np.nan, int(np.isfinite(v))

    half = total_window // 2
    vals = []
    for off in range(-half, half + 1):
        dt = (center_date + pd.Timedelta(days=off)).date()
        v = sensor_map.get(dt, np.nan)
        if np.isfinite(v):
            vals.append(float(v))

    if total_window == 3:
        min_valid = 2
    elif total_window == 5:
        min_valid = 3
    elif total_window == 7:
        min_valid = 4
    else:
        min_valid = max(1, half + 1)

    if len(vals) < min_valid:
        return np.nan, len(vals)

    return float(np.nanmean(vals)), len(vals)


def compute_site_validation_metrics(
    sensor_df: pd.DataFrame,
    model_df: pd.DataFrame,
    temporal_windows: Iterable[int] = (0,),
    min_pairs: int = 3,
) -> dict:
    if sensor_df.empty or model_df.empty:
        return {}

    sensor_map = {
        d.date(): float(v)
        for d, v in zip(sensor_df["date"], sensor_df["x"])
        if pd.notna(v)
    }

    out = {}
    for tw in temporal_windows:
        x_vals = []
        y_vals = []
        counts_used = []

        for dt, y in zip(model_df["date"], model_df["y"]):
            x_agg, n_used = _sensor_window_mean(sensor_map, dt, tw)
            if np.isfinite(x_agg) and np.isfinite(y):
                x_vals.append(float(x_agg))
                y_vals.append(float(y))
                counts_used.append(int(n_used))

        x_arr = np.asarray(x_vals, dtype=float)
        y_arr = np.asarray(y_vals, dtype=float)

        out[f"overlap_count_tw{tw}"] = int(len(x_arr))

        if len(x_arr) < min_pairs:
            out[f"p_rho_tw{tw}"] = np.nan
            out[f"s_rho_tw{tw}"] = np.nan
            out[f"bias_tw{tw}"] = np.nan
            out[f"rmse_tw{tw}"] = np.nan
            out[f"ubrmsd_tw{tw}"] = np.nan
            continue

        out[f"p_rho_tw{tw}"] = _safe_corr(pd.Series(x_arr), pd.Series(y_arr), method="pearson")
        out[f"s_rho_tw{tw}"] = _safe_corr(pd.Series(x_arr), pd.Series(y_arr), method="spearman")
        out[f"bias_tw{tw}"] = _bias(x_arr, y_arr)
        out[f"rmse_tw{tw}"] = _rmse(x_arr, y_arr)
        out[f"ubrmsd_tw{tw}"] = _ubrmsd(x_arr, y_arr)

    if 0 in temporal_windows:
        base = out.get("p_rho_tw0", np.nan)
        for tw in temporal_windows:
            if tw == 0:
                continue
            out[f"delta_r_{tw}"] = (
                out.get(f"p_rho_tw{tw}", np.nan) - base
                if np.isfinite(base) and np.isfinite(out.get(f"p_rho_tw{tw}", np.nan))
                else np.nan
            )

    return out


def compute_inter_overpass_windows(
    sensor_daily_df: pd.DataFrame,
    model_df: pd.DataFrame,
    min_valid_increments: int = 2,
    epsilon: float = 1e-9,
    require_complete_path_for_missed: bool = True,
) -> pd.DataFrame:
    if model_df is None or len(model_df) < 2:
        return pd.DataFrame()

    sensor_daily_df = sensor_daily_df.dropna(subset=["date", "x"]).sort_values("date").reset_index(drop=True)
    model_df = model_df.dropna(subset=["date", "y"]).sort_values("date").reset_index(drop=True)

    sensor_map = {
        d.date(): float(v)
        for d, v in zip(sensor_daily_df["date"], sensor_daily_df["x"])
        if pd.notna(v)
    }

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
        delta_y = y_end - y_start

        w = sensor_daily_df[
            (sensor_daily_df["date"] >= start_dt) &
            (sensor_daily_df["date"] <= end_dt)
        ].copy()

        if w.empty:
            continue

        w = w.dropna(subset=["x"]).sort_values("date").reset_index(drop=True)

        incs = []
        valid_inc_dates = []

        for i in range(1, len(w)):
            d0 = w.loc[i - 1, "date"].date()
            d1 = w.loc[i, "date"].date()
            if (d1 - d0).days == 1:
                incs.append(float(w.loc[i, "x"] - w.loc[i - 1, "x"]))
                valid_inc_dates.append((d0, d1))

        m = len(incs)
        expected_increments = int((end_dt.date() - start_dt.date()).days)

        if m < min_valid_increments:
            continue

        incs = np.asarray(incs, dtype=float)

        AV = float(np.nansum(np.abs(incs)))
        QV = float(np.nansum(incs ** 2))
        AV_norm = AV / m
        QV_norm = QV / m
        coverage_ratio = (m / expected_increments) if expected_increments > 0 else np.nan

        x_start = sensor_map.get(start_dt.date(), np.nan)
        x_end = sensor_map.get(end_dt.date(), np.nan)

        complete_path = False
        if expected_increments > 0 and not w.empty:
            complete_path = (
                pd.Timestamp(w.iloc[0]["date"]).date() == start_dt.date() and
                pd.Timestamp(w.iloc[-1]["date"]).date() == end_dt.date() and
                m == expected_increments
            )

        if np.isfinite(x_start) and np.isfinite(x_end):
            delta_x_end = float(x_end - x_start)
        else:
            delta_x_end = np.nan

        Missed = np.nan
        Missed_norm = np.nan

        can_compute_missed = (
            np.isfinite(delta_x_end) and
            (
                (complete_path and require_complete_path_for_missed) or
                (not require_complete_path_for_missed)
            )
        )

        if can_compute_missed:
            Missed = float(AV - abs(delta_x_end))
            if AV <= epsilon and abs(delta_x_end) <= epsilon:
                Missed_norm = 0.0
            elif AV > epsilon:
                Missed_norm = float((AV - abs(delta_x_end)) / (AV + epsilon))

        rows.append(
            {
                "start_date": start_dt.date(),
                "end_date": end_dt.date(),
                "window_length_days": expected_increments,
                "y_start": y_start,
                "y_end": y_end,
                "delta_y": delta_y,
                "x_start": x_start,
                "x_end": x_end,
                "delta_x_end": delta_x_end,
                "valid_increments": int(m),
                "expected_increments": int(expected_increments),
                "coverage_ratio": coverage_ratio,
                "complete_path": bool(complete_path),
                "AV": AV,
                "QV": QV,
                "AV_norm": AV_norm,
                "QV_norm": QV_norm,
                "Missed": Missed,
                "Missed_norm": Missed_norm,
            }
        )

    return pd.DataFrame(rows)


def summarize_site_windows(
    site: SiteMeta,
    row_path: str,
    raster_stat: str,
    windows_df: pd.DataFrame,
    validation_metrics: dict | None = None,
) -> dict:
    base = {
        "gpi": str(site.gpi),
        "latitude": float(site.lat),
        "longitude": float(site.lon),
        "row_path": row_path,
        "raster_stat": raster_stat,
        "number_of_windows": 0,
        "total_valid_increment_count": 0,
        "mean_window_length_days": np.nan,
        "median_window_length_days": np.nan,
        "median_coverage_ratio": np.nan,
        "median_AV_norm": np.nan,
        "median_QV_norm": np.nan,
        "median_Missed_norm": np.nan,
        "p75_AV_norm": np.nan,
        "p75_QV_norm": np.nan,
    }

    if windows_df is not None and not windows_df.empty:
        base.update({
            "number_of_windows": int(len(windows_df)),
            "total_valid_increment_count": int(windows_df["valid_increments"].sum()),
            "mean_window_length_days": float(windows_df["window_length_days"].mean()),
            "median_window_length_days": float(windows_df["window_length_days"].median()),
            "median_coverage_ratio": float(windows_df["coverage_ratio"].median()),
            "median_AV_norm": float(windows_df["AV_norm"].median()),
            "median_QV_norm": float(windows_df["QV_norm"].median()),
            "median_Missed_norm": float(windows_df["Missed_norm"].median(skipna=True)),
            "p75_AV_norm": float(windows_df["AV_norm"].quantile(0.75)),
            "p75_QV_norm": float(windows_df["QV_norm"].quantile(0.75)),
        })

    if validation_metrics:
        base.update(validation_metrics)

    return base


def run_case(
    wit_sms_path: str,
    raster_base: str,
    row_path: str,
    raster_stat: str,
    min_valid_increments: int,
    epsilon: float,
    require_complete_path_for_missed: bool,
    apply_sms_calibration: bool,
    apply_rescaling: bool,
    temporal_windows_for_corr: Iterable[int],
    min_pairs_corr: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    soil = SoilMoistureData(wit_sms_path)
    soil.read_data()
    metadata = soil.get_metadata()

    raster_folder = Path(raster_base) / raster_stat / row_path
    raster = SebalSoilMoistureData(str(raster_folder), pattern="Root_zone_moisture")

    window_all = []
    site_all = []

    for meta in metadata:
        try:
            lat = float(meta["latitude"])
            lon = float(meta["longitude"])
            gpi = str(meta["gpi"])
        except Exception:
            continue

        site = SiteMeta(gpi=gpi, lat=lat, lon=lon)

        sensor_df = load_site_sensor_daily(
            soil,
            lat,
            lon,
            apply_sms_calibration=apply_sms_calibration,
        )
        if sensor_df.empty:
            continue

        model_df = load_site_model_full(raster, lat, lon)
        if model_df.empty:
            continue

        model_df = maybe_rescale_model_to_sensor(
            sensor_df,
            model_df,
            apply_rescaling=apply_rescaling,
        )

        validation_metrics = compute_site_validation_metrics(
            sensor_df=sensor_df,
            model_df=model_df,
            temporal_windows=temporal_windows_for_corr,
            min_pairs=min_pairs_corr,
        )

        windows_df = compute_inter_overpass_windows(
            sensor_daily_df=sensor_df,
            model_df=model_df,
            min_valid_increments=min_valid_increments,
            epsilon=epsilon,
            require_complete_path_for_missed=require_complete_path_for_missed,
        )

        if not windows_df.empty:
            windows_df.insert(0, "gpi", site.gpi)
            windows_df.insert(1, "latitude", site.lat)
            windows_df.insert(2, "longitude", site.lon)
            windows_df.insert(3, "row_path", row_path)
            windows_df.insert(4, "raster_stat", raster_stat)
            window_all.append(windows_df)

        site_all.append(
            summarize_site_windows(
                site=site,
                row_path=row_path,
                raster_stat=raster_stat,
                windows_df=windows_df,
                validation_metrics=validation_metrics,
            )
        )

    window_df = pd.concat(window_all, ignore_index=True) if window_all else pd.DataFrame()
    site_df = pd.DataFrame(site_all) if site_all else pd.DataFrame()

    return window_df, site_df