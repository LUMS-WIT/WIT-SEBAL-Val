"""Location-held-out scaling. Shared validation helpers remain unchanged."""
from pathlib import Path
import hashlib
import json
import shutil
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def error_metrics(x, y):
    """Population errors; undefined correlations are missing, never zero."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not len(x) or not np.isfinite(np.r_[x, y]).all():
        raise ValueError("Metrics require nonempty, finite matched pairs.")
    error = x - y
    r = rho = float("nan")
    if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
        r, rho = float(pearsonr(x, y).statistic), float(spearmanr(x, y).statistic)
    mse = float(np.mean(error ** 2))
    return dict(N=len(x), bias=float(error.mean()), MSE=mse, RMSE=mse ** .5,
                ubRMSD=float(np.std(error)), Pearson=r, Spearman=rho)


def validate_pairs(pairs):
    required = ["gpi", "row", "date", "latitude", "longitude", "raw", "observed"]
    if pairs.empty or not set(required).issubset(pairs.columns):
        raise ValueError(f"Matched pairs must be nonempty with columns {required}.")
    if pairs[required].isna().any().any():
        raise ValueError("Missing identity, date, coordinate or observation in matched pairs.")
    if not np.isfinite(pairs[["latitude", "longitude", "raw", "observed"]]).all().all():
        raise ValueError("Matched coordinates and observations must be finite.")
    if pairs.duplicated(["gpi", "row", "date"]).any():
        raise ValueError("Duplicate GPI/row/date pairs would double-count observations.")
    if (pairs.groupby("gpi")[["latitude", "longitude"]].nunique() > 1).any().any():
        raise ValueError("A GPI has multiple coordinates; resolve its location before folding.")


def geographic_folds(pairs, n_folds=5):
    """Equal-location-count strips along the leading geographic principal axis.

    Coordinates alone determine folds. Exact co-locations stay together; there
    is no spatial buffer. All dates and satellite rows of a GPI share a fold.
    """
    validate_pairs(pairs)
    locations = (pairs[["latitude", "longitude"]].drop_duplicates()
                 .sort_values(["latitude", "longitude"]).reset_index(drop=True))
    if not isinstance(n_folds, int) or not 2 <= n_folds <= len(locations):
        raise ValueError("Fold count must be between 2 and the number of unique locations.")
    locations["location_group"] = np.arange(len(locations))
    lat0, lon0 = locations.latitude.mean(), locations.longitude.mean()
    xy = np.column_stack([(locations.longitude - lon0) * 111.32 * np.cos(np.deg2rad(lat0)),
                          (locations.latitude - lat0) * 111.32])
    _, vectors = np.linalg.eigh(np.cov(xy, rowvar=False))
    axis = vectors[:, -1]
    if axis[1] < 0 or (axis[1] == 0 and axis[0] < 0):
        axis = -axis
    locations["axis_distance_km"] = xy @ axis
    order = locations.sort_values(["axis_distance_km", "location_group"]).index.to_numpy()
    for fold, indices in enumerate(np.array_split(order, n_folds), 1):
        locations.loc[indices, "fold"] = fold
    locations["fold"] = locations.fold.astype(int)
    assigned = pairs.drop(columns=["fold", "location_group"], errors="ignore").merge(
        locations.drop(columns="axis_distance_km"), on=["latitude", "longitude"],
        how="left", validate="many_to_one")
    return assigned, locations


def cross_validate(pairs, n_folds=5):
    assigned, locations = geographic_folds(pairs, n_folds)
    predictions, fold_records, site_records = [], [], []
    for fold in range(1, n_folds + 1):
        train = assigned[assigned.fold.ne(fold)]
        test = assigned[assigned.fold.eq(fold)].copy()
        if set(train.gpi) & set(test.gpi) or set(train.location_group) & set(test.location_group):
            raise ValueError("Training and test locations overlap.")
        sx, sy = train.raw.std(ddof=0), train.observed.std(ddof=0)
        if not np.isfinite([sx, sy]).all() or sx <= 0 or sy <= 0:
            raise ValueError(f"Fold {fold}: both calibration standard deviations must be positive.")
        slope = float(sy / sx)
        intercept = float(train.observed.mean() - slope * train.raw.mean())
        test["scaled"] = slope * test.raw + intercept
        test["slope"], test["intercept"] = slope, intercept
        predictions.append(test)
        common = dict(fold=fold, train_locations=train.location_group.nunique(),
                      test_locations=test.location_group.nunique(), train_gpis=train.gpi.nunique(),
                      test_gpis=test.gpi.nunique(), train_pairs=len(train), slope=slope, intercept=intercept)
        for method in ["raw", "scaled"]:
            fold_records.append(dict(**common, method=method, **error_metrics(test[method], test.observed)))
            for gpi, site in test.groupby("gpi"):
                site_records.append(dict(fold=fold, gpi=gpi, method=method,
                                         **error_metrics(site[method], site.observed)))
    heldout = pd.concat(predictions).sort_values(["gpi", "date", "row"]).reset_index(drop=True)
    if len(heldout) != len(pairs) or not heldout.groupby("gpi").fold.nunique().eq(1).all():
        raise ValueError("Every input pair must be held out exactly once.")
    folds, sites = pd.DataFrame(fold_records), pd.DataFrame(site_records)
    summary = {m: error_metrics(heldout[m], heldout.observed) for m in ["raw", "scaled"]}
    fields = ["bias", "MSE", "RMSE", "ubRMSD", "Pearson", "Spearman"]
    for name, frame in [("fold", folds), ("gpi", sites)]:
        summary[name + "_means"] = frame.groupby("method")[fields].mean().to_dict("index")
        summary[name + "_medians"] = frame.groupby("method")[fields].median().to_dict("index")
    return heldout, locations, folds, sites, summary


def extract_pairs(wit_sms_path, raster_base, rows, member, temporal_win, cohort_file):
    """Fresh raw extraction by GPI, without any fit or correlation filtering."""
    from utils import SoilMoistureData, SebalSoilMoistureData, remove_nan_entries
    from sms_calibration import sms_calibrations
    from scaling import temporal_matching, temporal_matching_windowed
    from modules.validation_module import tw_to_window_params

    wit_sms_path, raster_base = Path(wit_sms_path), Path(raster_base)
    if not wit_sms_path.is_dir():
        raise FileNotFoundError(f"SCALING_WIT_SMS_PATH does not exist: {wit_sms_path}")
    half_window, min_valid = tw_to_window_params(temporal_win)
    sources = []
    for row in rows:
        folder = raster_base / member / str(row)
        files = sorted(folder.glob("*Root_zone_moisture*.tif"))
        if not files:
            raise FileNotFoundError(f"No Root_zone_moisture rasters in {folder}")
        sources.extend(dict(path=str(p.resolve()), size=p.stat().st_size,
                            mtime_ns=p.stat().st_mtime_ns) for p in files)
    cohort = json.loads(Path(cohort_file).read_text()) if cohort_file else None
    wanted = set(map(int, cohort["gpis"])) if cohort else None
    sensors = SoilMoistureData(str(wit_sms_path))
    sensors.read_data()
    metadata = pd.DataFrame(sensors.get_metadata())
    if metadata.empty:
        raise ValueError("No readable sensor series.")
    metadata["gpi"] = metadata.gpi.astype(int)
    metadata[["latitude", "longitude"]] = metadata[["latitude", "longitude"]].astype(float)
    if wanted is not None:
        metadata = metadata[metadata.gpi.isin(wanted)].copy()
        if set(metadata.gpi) != wanted:
            raise ValueError(f"Missing cohort GPIs: {sorted(wanted - set(metadata.gpi))}")
    if metadata.gpi.duplicated().any():
        raise ValueError("Multiple sensor files for one GPI; resolve duplicates before validation.")
    pairs = []
    for row in rows:
        reader = SebalSoilMoistureData(str(raster_base / member / str(row)), pattern="Root_zone_moisture")
        cache = {}
        for m in metadata.sort_values("gpi").to_dict("records"):
            loc = (m["latitude"], m["longitude"])
            if loc not in cache:
                cache[loc] = reader.get_data(*loc)
            rd, rv = cache[loc]
            if rd is None or rv is None:
                continue
            # Location lookup can pick a different deployment at identical coordinates.
            sd, sv = sensors.get_soil_moisture_by_location(gpi=str(m["gpi"]))
            sd, sv = remove_nan_entries(sd, sv)
            rd, rv = remove_nan_entries(rd, rv)
            sd, sv = sms_calibrations((sd, sv))
            if not len(sd) or not len(rd):
                continue
            if temporal_win == 0:
                dates, observed, raw, _ = temporal_matching((rd, rv), (sd, sv), 0)
            else:
                dates, observed, raw, _, _, _ = temporal_matching_windowed(
                    model_data=(rd, rv), sensor_data=(sd, sv),
                    half_window=half_window, min_valid=min_valid)
            pairs.extend(dict(gpi=m["gpi"], row=str(row), date=pd.Timestamp(d), latitude=loc[0],
                              longitude=loc[1], raw=float(x), observed=float(y))
                         for d, x, y in zip(dates, raw, observed) if np.isfinite(x) and np.isfinite(y))
        print(f"[scaling] Extracted row {row}; {len(pairs)} cumulative raw pairs", flush=True)
    pairs = pd.DataFrame(pairs)
    validate_pairs(pairs)
    if wanted is not None and set(pairs.gpi) != wanted:
        raise ValueError(f"Cohort GPIs without matched pairs: {sorted(wanted - set(pairs.gpi))}")
    import re
    sensor_hashes = {}
    for path in sorted(wit_sms_path.glob("*witsms_gpi*.csv")):
        match = re.search(r"gpi=(\d+)", path.name)
        if match and int(match[1]) in set(pairs.gpi):
            sensor_hashes[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = dict(cohort=cohort or {"description": "All GPIs with finite matched pairs"},
                      sensor_sha256=sensor_hashes, raster_files=sources)
    return pairs, metadata, provenance


def plot_error_comparison(site_metrics, output_stem, show=False):
    """One box per method/metric; each GPI has equal weight."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = [("raw", "Unscaled", "#59758F"), ("scaled", "Scaled", "#D98948")]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for offset, (method, _, color) in zip([-.18, .18], colors):
        values = [site_metrics.loc[site_metrics.method.eq(method), key].dropna().to_numpy()
                  for key in ["bias", "MSE", "ubRMSD"]]
        bp = ax.boxplot(values, positions=np.arange(1, 4) + offset, widths=.29,
                        patch_artist=True, manage_ticks=False, whis=1.5,
                        boxprops=dict(facecolor=color, edgecolor=color, alpha=.85, linewidth=1.3),
                        medianprops=dict(color="#202A35", linewidth=1.8),
                        whiskerprops=dict(color=color, linewidth=1.3),
                        capprops=dict(color=color, linewidth=1.3),
                        flierprops=dict(marker="o", markerfacecolor=color, markeredgecolor="white",
                                        markeredgewidth=.45, markersize=4.5, alpha=.8))
        for line in bp["medians"]:
            xs, ys = line.get_xdata(), line.get_ydata()
            # Labels extend outward from the paired boxes, leaving their lines visible.
            left = method == "raw"
            ax.annotate(f"{ys[0]:.4f}", xy=(xs[0] if left else xs[-1], ys[0]),
                        xytext=(-6 if left else 6, 0), textcoords="offset points",
                        ha="right" if left else "left", va="center", fontsize=10,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=1.2))
    ax.axhline(0, color="#7D8790", lw=.9, linestyle=(0, (4, 3)), zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#E9EDF0", linewidth=.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#B6BEC6")
    ax.tick_params(axis="both", length=0, pad=8)
    ax.set_xticks([1, 2, 3], ["Bias\n(m³/m³)", "MSE\n((m³/m³)²)", "ubRMSD\n(m³/m³)"])
    ax.set_xlim(.25, 3.75)
    ax.set_ylabel("Metric value", fontsize=12)
    ax.legend(handles=[Patch(facecolor=c, label=l, alpha=.85) for _, l, c in colors],
              loc="upper right", ncol=2, frameon=False)
    fig.tight_layout()
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(output_stem.with_suffix("." + suffix), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_fold_results(locations, folds, output_stem, show=False):
    """Retain the geographic-block map and per-fold RMSE comparison."""
    import matplotlib.pyplot as plt

    fig, (map_ax, error_ax) = plt.subplots(1, 2, figsize=(11, 5))
    palette = ["#286482", "#32A3A3", "#99B154", "#D99A35", "#9C6092"]
    for (fold, group), color in zip(locations.groupby("fold"),
                                    np.resize(palette, locations.fold.nunique())):
        map_ax.scatter(group.longitude, group.latitude, s=32, color=color,
                       edgecolors="white", linewidths=.4,
                       label=f"Fold {fold}: {len(group)} locations")
    map_ax.set_xlabel("Longitude (°E)")
    map_ax.set_ylabel("Latitude (°N)")
    map_ax.set_aspect(1 / np.cos(np.deg2rad(locations.latitude.mean())))
    map_ax.legend(frameon=False, fontsize=9)
    for offset, method, label, color in [(-.18, "raw", "Unscaled", "#59758F"),
                                         (.18, "scaled", "Scaled", "#D98948")]:
        data = folds[folds.method.eq(method)].sort_values("fold")
        bars = error_ax.bar(data.fold + offset, data.RMSE, width=.34, label=label, color=color)
        error_ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    error_ax.set_xticks(sorted(locations.fold.unique()),
                        [f"Fold {i}" for i in sorted(locations.fold.unique())])
    error_ax.set_ylabel("RMSE (m³/m³)")
    error_ax.set_ylim(0, folds.RMSE.max() * 1.3)
    error_ax.legend(frameon=False)
    for ax in [map_ax, error_ax]:
        ax.set_axisbelow(True)
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(Path(output_stem).with_suffix("." + suffix), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def export_validation_results(heldout, metadata, output, rows, member, temporal_win,
                              show_plots=False, save_site_plots=False):
    """Preserve the legacy sheets/columns and plots without its score filtering.

    Combined MetaData retains one row per GPI/satellite-row series, as intended
    by the legacy format. It does not perform the legacy many-to-many GPI join.
    The separate per_gpi_metrics.csv aggregates each GPI across satellite rows.
    """
    import matplotlib.pyplot as plt
    import metrics
    from utils import compute_statistics, plot_box_and_whiskers, plot_metric_with_ci, save_to_plot
    from modules.validation_module import make_paired_scatter_figure, save_fig

    meta_columns = ["gpi", "latitude", "longitude", "start_date", "end_date", "count", "overlaps"]
    metric_columns = ["bias_cl", "bias", "bias_cu", "mse", "ubrmsd_cl", "ubrmsd", "ubrmsd_cu", "p_rho", "s_rho"]
    lookup = metadata.set_index("gpi")
    if not lookup.index.is_unique:
        raise ValueError("Metadata must have one record per GPI.")
    for method, folder in [("raw", "unscaled"), ("scaled", "scaled")]:
        base = output / folder
        tables = []
        for row in rows:
            subset = heldout[heldout.row.astype(str).eq(str(row))]
            records = []
            points_dir = base / "validation_points" / member / f"{row}_{temporal_win}"
            points_dir.mkdir(parents=True, exist_ok=True)
            for gpi, site in subset.groupby("gpi"):
                m = lookup.loc[gpi].to_dict()
                m.update(gpi=str(gpi), overlaps=len(site))
                x, y = site[method].to_numpy(), site.observed.to_numpy()
                scores = error_metrics(x, y)
                b, u = scores["bias"], scores["ubRMSD"]
                # Legacy code passed ubRMSD as the bias centre; use the actual bias.
                bl, bu = metrics.bias_ci(x, y, b) if len(site) > 1 else (np.nan, np.nan)
                ul, uu = metrics.ubrmsd_ci(x, y, u) if len(site) > 1 else (np.nan, np.nan)
                records.append({**m, "bias_cl": bl, "bias": b, "bias_cu": bu,
                                "mse": scores["MSE"], "ubrmsd_cl": ul, "ubrmsd": u,
                                "ubrmsd_cu": uu, "p_rho": scores["Pearson"], "s_rho": scores["Spearman"]})
                name = f"sebal_{row}_{member}_witgpi_{gpi}_lat_{m['latitude']}_lon_{m['longitude']}"
                point_frame = pd.DataFrame({"Timestamp": pd.to_datetime(site.date), "wit_sm": y, "sebal_sm": x})
                point_frame.to_excel(points_dir / (name + ".xlsx"), index=False, engine="openpyxl")
                if save_site_plots:
                    image_dir = base / "figs" / member / str(row)
                    image_dir.mkdir(parents=True, exist_ok=True)
                    save_to_plot(point_frame.Timestamp, y, point_frame.Timestamp, x,
                                 m["latitude"], m["longitude"], str(image_dir / (name + ".png")))
                    plt.close("all")
            table = pd.DataFrame(records, columns=meta_columns + metric_columns)
            table[meta_columns].to_excel(points_dir.parent / f"metadata_{row}_tw_{temporal_win}.xlsx",
                                         index=False, engine="openpyxl")
            tables.append(table)
        groups = [(str(row), tables[i], heldout[heldout.row.astype(str).eq(str(row))]) for i, row in enumerate(rows)]
        groups.append((None, pd.concat(tables, ignore_index=True), heldout))
        for row, table, pairs in groups:
            if pairs.empty:
                continue
            metrics_dict = table[["gpi"] + metric_columns].to_dict("list")
            usable = {key: values for key, values in metrics_dict.items()
                      if key == "gpi" or np.isfinite(np.asarray(values, float)).any()}
            stats = compute_statistics(usable)
            for key in metric_columns:
                stats.setdefault(key, dict(mean=np.nan, median=np.nan, IQR=np.nan))
            summary = [{"Metric": "Observations", "mean": len(pairs), "median": "", "IQR": ""}]
            summary.extend({"Metric": key, **stats[key]} for key in ["bias", "mse", "ubrmsd", "p_rho", "s_rho"])
            name = f"validations_{row}" if row else "validations"
            result_dir = base / "results"
            result_dir.mkdir(parents=True, exist_ok=True)
            with pd.ExcelWriter(result_dir / f"{name}_tw_{temporal_win}.xlsx", engine="xlsxwriter") as writer:
                table.to_excel(writer, sheet_name="MetaData", index=False)
                pd.DataFrame(summary, columns=["Metric", "mean", "median", "IQR"]).to_excel(writer, sheet_name="Summary", index=False)
                # Preserve the plain tabular style, but keep headers/numeric values legible.
                writer.sheets["MetaData"].set_column(0, 15, 14)
                writer.sheets["Summary"].set_column(0, 3, 16)
            figs = base / "figs" / row if row else base / "figs"
            figs.mkdir(parents=True, exist_ok=True)
            tag = f"{member}_{row}" if row else f"{member}_{'_'.join(map(str, rows))}"
            paired = pairs.rename(columns={method: "sebal_sm", "observed": "wit_sm"})
            fig = make_paired_scatter_figure(paired, stats, tag,
                                            metrics_override=error_metrics(pairs[method], pairs.observed))
            save_fig(fig, figs / f"paired_scatter_tw_{temporal_win}.png")
            if show_plots:
                plt.show()
            plt.close(fig)
            plot_box_and_whiskers(metrics_dict, filename=str(figs / f"boxplot_{tag}_tw_{temporal_win}.png"), save=True, show=show_plots)
            for key in ["bias", "ubrmsd"]:
                plot_metric_with_ci(metrics_dict, metric=key,
                                    filename=str(figs / f"ci_{key}_{tag}_tw_{temporal_win}.png"), save=True, show=show_plots)


def run_validation_scaling(*, wit_sms_path, raster_base, output_base, cohort_file,
                           rows, member="mean", temporal_win=0, n_folds=5,
                           show_plots=False, save_site_plots=False):
    """Extract, calibrate, validate and export both methods into a dedicated folder."""
    rows = list(map(str, rows))
    if not rows or len(set(rows)) != len(rows):
        raise ValueError("Provide one or more distinct satellite rows.")
    output = Path(output_base).resolve()
    if output.exists():
        marker = output / "manifest.json"
        if not marker.exists() or json.loads(marker.read_text()).get("workflow") != "validation_scaling":
            raise ValueError(f"Refusing to replace an unmanaged output folder: {output}")
        manifest = json.loads(marker.read_text())
        existing = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}
        unknown = existing - set(manifest["generated_files"]) - {"manifest.json", ".DS_Store"}
        if unknown:
            raise ValueError(f"Output contains manually added files; move them before rerunning: {sorted(unknown)}")
    pairs, metadata, provenance = extract_pairs(wit_sms_path, raster_base, rows, member, temporal_win, cohort_file)
    heldout, locations, folds, sites, summary = cross_validate(pairs, n_folds)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".validation_scaling-", dir=output.parent))
    try:
        pairs.to_csv(staging / "raw_pairs.csv", index=False)
        heldout.to_csv(staging / "heldout_predictions.csv", index=False)
        locations.to_csv(staging / "location_fold_assignments.csv", index=False)
        gpis = heldout[["gpi", "latitude", "longitude", "location_group", "fold"]].drop_duplicates()
        gpis["paired_observations"] = gpis.gpi.map(heldout.groupby("gpi").size())
        gpis.to_csv(staging / "gpi_fold_assignments.csv", index=False)
        folds.to_csv(staging / "fold_metrics.csv", index=False)
        sites.to_csv(staging / "per_gpi_metrics.csv", index=False)
        (staging / "summary.json").write_text(json.dumps(summary, indent=2))
        export_validation_results(heldout, metadata, staging, rows, member, temporal_win, show_plots, save_site_plots)
        plot_error_comparison(sites, staging / "figs/error_metric_boxplots", show=show_plots)
        plot_fold_results(locations, folds, staging / "figs/five_fold_results", show=show_plots)
        report = ["# Five-fold validation results", "",
                  f"{len(gpis)} GPIs at {len(locations)} exact coordinate locations; {len(pairs)} matched pairs.", "",
                  "Each location is held out once. Fit a common positive mean/std transform on the other folds only. "
                  "All dates, satellite rows and co-located GPIs remain together. Folds use coordinates alone; no buffer is imposed.", "",
                  "| Pooled held-out metric | Unscaled | Scaled |", "|---|---:|---:|"]
        report.extend(f"| {key} | {summary['raw'][key]:.6f} | {summary['scaled'][key]:.6f} |"
                      for key in ["bias", "MSE", "RMSE", "ubRMSD", "Pearson", "Spearman"])
        report.extend(["", "The comparison box plot uses one value per GPI, combining its satellite rows. "
                       "The legacy-format workbooks retain one row per GPI/satellite-row series. "
                       "Their Summary sheet uses the existing Fisher-z mean for correlations, not the pooled correlation above.", "",
                       "Per-GPI and within-fold correlations are invariant under their common positive affine transform. "
                       "Concatenating folds uses different transforms, so pooled correlations may change.", "",
                       provenance["cohort"]["description"], "",
                       "The workflow applies no new performance filtering. Any prior selection in the configured cohort "
                       "still limits inference. Raw results remain primary. "
                       f"This fresh GPI-based extraction has {len(pairs)} pairs; the historical reference workbook had 422.", "",
                       "Bias and ubRMSD CI plots retain the legacy parametric formulas; these assume independent errors "
                       "and do not quantify spatial/temporal dependence or calibration uncertainty. The bias CI is centered "
                       "on the actual bias (correcting the legacy exporter’s wrong argument).", "",
                       "Input provenance, fitted coefficients, assignments, predictions and metrics are saved alongside the workbooks."])
        (staging / "REPORT.md").write_text("\n".join(report) + "\n")
        provenance.update(workflow="validation_scaling", folds=n_folds, rows=rows, member=member,
                          temporal_win=temporal_win, gpis=len(gpis), locations=len(locations), pairs=len(pairs),
                          calibration="Training pairs only; common mean/std affine transform; ddof=0; pair weighted",
                          assignment="Coordinate-only equal-count strips along leading geographic principal axis; no buffer",
                          filtering="Finite matched pairs, fixed sensor QC 0.12–0.90; no correlation-score filtering",
                          workbook_summary="Legacy site-row summary; correlations use legacy Fisher-z mean",
                          raw_pairs_sha256=hashlib.sha256((staging / "raw_pairs.csv").read_bytes()).hexdigest())
        provenance["generated_files"] = sorted(str(p.relative_to(staging)) for p in staging.rglob("*") if p.is_file())
        (staging / "manifest.json").write_text(json.dumps(provenance, indent=2))
        # Publish only a completed run. Existing managed results survive failures.
        backup = None
        if output.exists():
            backup = Path(tempfile.mkdtemp(prefix=".validation_scaling_previous-", dir=output.parent))
            backup.rmdir()
            output.rename(backup)
        try:
            staging.rename(output)
        except Exception:
            if backup:
                backup.rename(output)
            raise
        if backup:
            shutil.rmtree(backup)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(f"[scaling] Complete: {len(gpis)} GPIs, {len(locations)} locations, {len(pairs)} pairs -> {output}")
    print(pd.DataFrame({m: summary[m] for m in ["raw", "scaled"]}).to_string())
    return output
