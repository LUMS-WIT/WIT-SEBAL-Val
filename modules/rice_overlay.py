"""Read-only overlay of the agreed 445-pair validation and a public rice map."""
from pathlib import Path
import hashlib
import json
import math
import zipfile

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds, Window
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

import config as cfg
ROOT = cfg.PROJECT_ROOT
OUT = Path(cfg.RICE_OUTPUT)
BOOK = Path(cfg.SCALING_OUTPUT_BASE) / f"scaled/results/validations_tw_{cfg.TEMPORAL_WIN}.xlsx"
PAIRS = Path(cfg.SCALING_OUTPUT_BASE) / "heldout_predictions.csv"
ARCHIVE = Path(cfg.RICE_ARCHIVE)
RADIUS = [100, 500, 1000, 5000]
EARTH = 6371008.8


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(group):
    y, x = group.observed.to_numpy(), group.scaled.to_numpy()
    e = x - y
    return pd.Series({
        "overlaps": len(group), "bias": e.mean(), "mse": (e**2).mean(),
        "ubrmsd": np.sqrt(((e - e.mean())**2).mean()),
        "p_rho": np.corrcoef(x, y)[0, 1] if len(x) > 2 else np.nan,
    })


def subset_tiles(tiles, bounds):
    l, b, r, t = bounds
    return [s for s in tiles if s.bounds.left < r and s.bounds.right > l
            and s.bounds.bottom < t and s.bounds.top > b]


def merge_buffer(tiles, bounds):
    """Single-band north-up mosaic with explicit nearest-pixel grid alignment.

    Explicit dimensions/window rounding avoid unfilled tile-seam pixels
    across Rasterio releases. Nearest-neighbor, first-valid semantics
    follow https://github.com/rasterio/rasterio/blob/1.4.3/rasterio/merge.py.
    """
    left, bottom, right, top = bounds
    xres, yres = tiles[0].res
    width, height = round((right-left)/xres), round((top-bottom)/yres)
    transform = rasterio.transform.from_origin(left, top, xres, yres)
    mosaic = np.full((height, width), 255, dtype=np.uint8)
    east, south = left + width*xres, top - height*yres
    for tile in tiles:
        if tile.crs.to_epsg() != 4326 or tile.transform.b or tile.transform.d:
            raise ValueError('Rice buffers require north-up EPSG:4326 tiles.')
        overlap = (max(left, tile.bounds.left), max(south, tile.bounds.bottom),
                   min(east, tile.bounds.right), min(top, tile.bounds.top))
        if overlap[0] >= overlap[2] or overlap[1] >= overlap[3]:
            continue
        source_window = from_bounds(*overlap, tile.transform)
        target = from_bounds(*overlap, transform)
        x0, y0 = math.floor(target.col_off+.5), math.floor(target.row_off+.5)
        nx, ny = math.floor(target.width+.5), math.floor(target.height+.5)
        data = tile.read(1, window=source_window, out_shape=(ny, nx), masked=True)
        region = mosaic[y0:y0+ny, x0:x0+nx]
        if region.shape != data.shape:
            raise ValueError('Rice mosaic window does not align with its destination.')
        valid = (region == 255) & ~np.ma.getmaskarray(data)
        np.copyto(region, data.data, where=valid)
    return mosaic, transform


def rice_around(tiles, lat, lon, radius):
    dy = math.degrees(radius / EARTH)
    dx = math.degrees(radius / EARTH / math.cos(math.radians(lat)))
    selected = subset_tiles(tiles, (lon-dx, lat-dy, lon+dx, lat+dy))
    if not selected:
        raise ValueError('The rice map does not cover a requested buffer.')
    a, transform = merge_buffer(selected, (lon-dx, lat-dy, lon+dx, lat+dy))
    ys = transform.f + (np.arange(a.shape[0]) + .5) * transform.e
    xs = transform.c + (np.arange(a.shape[1]) + .5) * transform.a
    phi, lam = np.radians(ys[:, None]), np.radians(xs[None, :])
    h = np.sin((phi-math.radians(lat))/2)**2 + np.cos(phi)*math.cos(math.radians(lat))*np.sin((lam-math.radians(lon))/2)**2
    inside = (2*EARTH*np.arcsin(np.sqrt(np.minimum(h, 1)))) <= radius
    assert np.all(a[inside] != 255), "A buffer contains unmapped pixels; do not count those as non-rice."
    weights = np.broadcast_to(np.cos(phi), a.shape)
    positive = (a >= 1) & (a <= 3)
    frac = float(weights[inside & positive].sum() / weights[inside].sum())
    return frac, int((inside & positive).sum()), int(inside.sum())


def summary(frame):
    results = {"n": len(frame), "groups": {}, "spearman": {}}
    for label, mask in [("MSE_upper_quartile", frame.high_mse),
                        ("MSE_other", ~frame.high_mse),
                        ("ubRMSD_upper_quartile", frame.high_ubrmsd),
                        ("ubRMSD_other", ~frame.high_ubrmsd)]:
        d = frame.loc[mask]
        results["groups"][label] = {"n": len(d)}
        for r in RADIUS:
            col = f"rice_fraction_{r}m"
            results["groups"][label][col] = {
                "median": float(d[col].median()), "mean": float(d[col].mean()),
                "any_rice": int((d[col] > 0).sum()),
                "at_least_10pct": int((d[col] >= .10).sum()),
                "at_least_25pct": int((d[col] >= .25).sum()),
                "at_least_50pct": int((d[col] >= .50).sum()),
            }
    for metric in ["mse", "ubrmsd", "bias", "p_rho"]:
        results["spearman"][metric] = {}
        for r in RADIUS:
            v = frame[[metric, f"rice_fraction_{r}m"]].dropna().rank()
            results["spearman"][metric][str(r)] = float(v.corr().iloc[0, 1])
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_excel(BOOK, sheet_name="MetaData")
    pairs = pd.read_csv(PAIRS)
    assert len(pairs) > 0 and len(d) == pairs.gpi.nunique()
    assert d.gpi.is_unique
    calculated = pairs.groupby("gpi", sort=False).apply(metrics, include_groups=False)
    joined = d.set_index("gpi").join(calculated, rsuffix="_recomputed")
    delta = {c: float((joined[c] - joined[c+"_recomputed"]).abs().max())
             for c in ["overlaps", "bias", "mse", "ubrmsd", "p_rho"]}
    assert max(delta.values()) < 1e-10, delta
    q_mse, q_ubr = float(d.mse.quantile(.75)), float(d.ubrmsd.quantile(.75))
    d["high_mse"] = d.mse >= q_mse
    d["high_ubrmsd"] = d.ubrmsd >= q_ubr
    extent = (float(d.longitude.min()-.15), float(d.latitude.min()-.15),
              float(d.longitude.max()+.15), float(d.latitude.max()+.15))
    tile_dir = OUT / "rice_tiles"
    tile_dir.mkdir(exist_ok=True)
    tiles, meta = [], []
    with zipfile.ZipFile(ARCHIVE) as z:
        for name in z.namelist():
            if not name.endswith(".tif"):
                continue
            with rasterio.open("zip://"+str(ARCHIVE)+"!"+name) as src:
                b = src.bounds
                if b.left >= extent[2] or b.right <= extent[0] or b.bottom >= extent[3] or b.top <= extent[1]:
                    continue
            path = tile_dir / Path(name).name
            if not path.exists():
                path.write_bytes(z.read(name))
            src = rasterio.open(path)
            tiles.append(src)
            meta.append({"file": path.name, "bounds": list(src.bounds),
                         "shape": list(src.shape), "crs": str(src.crs),
                         "resolution": list(src.res), "nodata": src.nodata})
    print("Relevant tiles:", len(tiles), flush=True)
    cache = {}
    for i, row in d.iterrows():
        key = (row.latitude, row.longitude)
        if key not in cache:
            cache[key] = {}
            for r in RADIUS:
                cache[key][r] = rice_around(tiles, *key, r)
        for r, (frac, n_rice, n_all) in cache[key].items():
            d.loc[i, f"rice_fraction_{r}m"] = frac
            d.loc[i, f"rice_pixels_{r}m"] = n_rice
            d.loc[i, f"buffer_pixels_{r}m"] = n_all
        center = next(s for s in tiles if s.bounds.left <= row.longitude < s.bounds.right
                      and s.bounds.bottom <= row.latitude < s.bounds.top)
        d.loc[i, "rice_class_at_point"] = int(next(center.sample([(row.longitude, row.latitude)]))[0])
    d.to_csv(OUT / "gpi_rice_overlap.csv", index=False)
    loc = pairs.groupby(["latitude", "longitude"], sort=False).apply(metrics, include_groups=False).reset_index()
    loc = loc.merge(d.drop_duplicates(["latitude", "longitude"])[
        ["latitude", "longitude", "rice_class_at_point"] + [f"rice_fraction_{r}m" for r in RADIUS]],
        on=["latitude", "longitude"], validate="one_to_one")
    loc["high_mse"] = loc.mse >= loc.mse.quantile(.75)
    loc["high_ubrmsd"] = loc.ubrmsd >= loc.ubrmsd.quantile(.75)
    loc.to_csv(OUT / "distinct_coordinate_rice_overlap.csv", index=False)
    out = {
        "workbook": str(BOOK), "sheet": "MetaData", "range": "A1:P67",
        "workbook_sha256": sha256(BOOK), "pairs_sha256": sha256(PAIRS),
        "n_pairs": len(pairs), "n_gpi": len(d), "n_distinct_coordinates": len(loc),
        "audit_max_absolute_difference": delta,
        "dataset": "https://zenodo.org/records/20923150", "map_year": 2021,
        "rice_classes": [1, 2, 3], "other_class": 0,
        "definition": "Elevated errors: within-cohort upper quartile, separately by metric. Buffers: circular great-circle radius; rice fraction weighted by cos(latitude). No exact-pixel membership is inferred from neighborhood overlap.",
        "mse_q75": q_mse, "ubrmsd_q75": q_ubr, "tiles": meta,
        "gpi_summary": summary(d), "distinct_coordinate_summary": summary(loc),
    }
    (OUT / "overlay_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"mse_q75": q_mse, "ubrmsd_q75": q_ubr,
                      "gpi_summary": out["gpi_summary"],
                      "distinct_coordinate_summary": out["distinct_coordinate_summary"]}, indent=2), flush=True)
    print("HIGH MSE", flush=True)
    print(d.loc[d.high_mse, ["gpi", "latitude", "longitude", "overlaps", "mse", "ubrmsd", "rice_class_at_point", "rice_fraction_1000m", "rice_fraction_5000m"]].sort_values("mse", ascending=False).round(5).to_string(index=False), flush=True)
    make_map(d, tiles, extent)
    for src in tiles:
        src.close()


def make_map(d, tiles, extent):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.6), sharex=True, sharey=True)
    rice_cmap = ListedColormap(["#168b42"])
    for src in tiles:
        b = src.bounds
        clip = (max(b.left, extent[0]), max(b.bottom, extent[1]),
                min(b.right, extent[2]), min(b.top, extent[3]))
        raw = from_bounds(*clip, src.transform)
        win = Window(math.floor(raw.col_off), math.floor(raw.row_off),
                     math.ceil(raw.width)+1, math.ceil(raw.height)+1).intersection(Window(0, 0, src.width, src.height))
        a = src.read(1, window=win)
        assert np.isin(a, [0, 1, 2, 3]).all(), np.unique(a)
        # Aggregate a binary rice map for display only; calculations retain 10 m grid.
        img = Image.fromarray((np.isin(a, [1, 2, 3]).astype(np.uint8)*255))
        small = np.asarray(img.resize((max(1, a.shape[1]//20), max(1, a.shape[0]//20)), resample=Image.Resampling.BOX))/255
        tile_extent = rasterio.windows.bounds(win, src.transform)
        ex = [tile_extent[0], tile_extent[2], tile_extent[1], tile_extent[3]]
        for ax in axs:
            ax.imshow(np.ones_like(small), extent=ex, origin="upper", cmap=rice_cmap,
                      alpha=np.minimum(small*1.6, .9), interpolation="nearest")
    for ax, flag, title in zip(axs, ["high_mse", "high_ubrmsd"], ["Elevated MSE", "Elevated ubRMSD"]):
        ax.scatter(d.loc[~d[flag], "longitude"], d.loc[~d[flag], "latitude"],
                   s=28, c="#c1c6cd", edgecolors="#48515d", linewidths=.55, zorder=4, label="Other validation GPIs")
        ax.scatter(d.loc[d[flag], "longitude"], d.loc[d[flag], "latitude"],
                   s=70, c="#bc2845", marker="o", edgecolors="white", linewidths=.9, zorder=5, label="Upper-quartile error GPIs")
        ax.scatter([], [], marker="s", s=70, color="#168b42", label="Mapped rice (2021)")
        ax.set_title(title, loc="left", fontweight="bold", fontsize=13)
        ax.set_xlim(extent[0], extent[2]); ax.set_ylim(extent[1], extent[3])
        ax.set_aspect(1/math.cos(math.radians(30.4)))
        ax.set_facecolor("#faf9f5")
        ax.grid(alpha=.14, linewidth=.7)
        ax.set_xlabel("Longitude (°E)")
        ax.spines[["right", "top"]].set_visible(False)
        ax.legend(loc="lower right", fontsize=8, framealpha=.95)
    axs[0].set_ylabel("Latitude (°N)")
    fig.suptitle("Validation errors and mapped rice near the Bari Doab network", fontsize=16, x=.06, ha="left", fontweight="bold")
    fig.text(.06,.91,"Validation locations at exact coordinates, without map jitter",fontsize=10,color="#444")
    fig.text(.06,.055,"Rice: Open-SARice-10, 2020–2021 (Zenodo 20923150). Green display aggregates the 10 m rice map to about 200 m.\nOverlap describes rice in the surrounding landscape; it does not establish the crop at a sensor on a validation date.",fontsize=8.5,color="#444")
    fig.subplots_adjust(left=.06,right=.985,top=.85,bottom=.16,wspace=.11)
    fig.savefig(OUT/"rice_error_overlap.png",dpi=210,facecolor="white")
    fig.savefig(OUT/"rice_error_overlap.pdf",facecolor="white")
    plt.close(fig)


def secondary():
    from rasterio.vrt import WarpedVRT
    d = pd.read_csv(OUT / "gpi_rice_overlap.csv")
    source = Path(cfg.RICE_2024_RASTER)
    with rasterio.open(source) as original:
        print("2024 raster:", original.crs, original.bounds, original.res, original.nodata, flush=True)
        src = original if original.crs.to_epsg() == 4326 else WarpedVRT(original, crs="EPSG:4326")
        cache = {}
        for i,row in d.iterrows():
            key = (row.latitude,row.longitude)
            if key not in cache:
                cache[key] = {r:rice_around([src], *key, r) for r in RADIUS}
            for r,(fraction,n_rice,n_all) in cache[key].items():
                d.loc[i,f"rice_fraction_{r}m"] = fraction
                d.loc[i,f"rice_pixels_{r}m"] = n_rice
                d.loc[i,f"buffer_pixels_{r}m"] = n_all
            point = int(next(src.sample([(row.longitude,row.latitude)]))[0])
            assert point in [0,1], point
            d.loc[i,"rice_class_at_point"] = point
        if src is not original:
            src.close()
    d.to_csv(OUT/"gpi_rice_overlap_2024.csv",index=False)
    result={"dataset":"https://zenodo.org/records/18468250", "map_year":2024,
            "interpretation":"Out-of-validation-period regional cross-check; not evidence of crops at a validation date.",
            "summary":summary(d),"exact_point_rice":int((d.rice_class_at_point==1).sum())}
    (OUT/"overlay_summary_2024.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    print(d[d.high_mse][["gpi","latitude","longitude","mse","rice_fraction_1000m","rice_fraction_5000m"]].sort_values("mse",ascending=False).round(5).to_string(index=False),flush=True)


def compare_maps():
    d=pd.read_csv(OUT/"gpi_rice_overlap.csv")
    extent=(float(d.longitude.min()-.15),float(d.latitude.min()-.15),float(d.longitude.max()+.15),float(d.latitude.max()+.15))
    sources_2021=[rasterio.open(p)for p in sorted((OUT/"rice_tiles").glob("*.tif"))]
    sources_2024=[rasterio.open(cfg.RICE_2024_RASTER)]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
    fig,axs=plt.subplots(1,2,figsize=(13,6.6),sharex=True,sharey=True)
    for ax,sources,title in zip(axs,[sources_2021,sources_2024],["2021 rice map (SARice)","2024 rice map (PHAROS)"]):
        for gx in range(math.floor(extent[0]),math.ceil(extent[2])):
            for gy in range(math.floor(extent[1]),math.ceil(extent[3])):
                grid=(max(gx,extent[0]),max(gy,extent[1]),min(gx+1,extent[2]),min(gy+1,extent[3]))
                for src in subset_tiles(sources,grid):
                    b=src.bounds
                    clip=(max(grid[0],b.left),max(grid[1],b.bottom),min(grid[2],b.right),min(grid[3],b.top))
                    raw=from_bounds(*clip,src.transform)
                    win=Window(math.floor(raw.col_off),math.floor(raw.row_off),math.ceil(raw.width)+1,math.ceil(raw.height)+1).intersection(Window(0,0,src.width,src.height))
                    a=src.read(1,window=win)
                    binary=(np.isin(a,[1,2,3]).astype(np.uint8)*255)
                    image=Image.fromarray(binary)
                    small=np.asarray(image.resize((max(1,a.shape[1]//20),max(1,a.shape[0]//20)),resample=Image.Resampling.BOX))/255
                    e=rasterio.windows.bounds(win,src.transform)
                    ax.imshow(np.ones_like(small),extent=[e[0],e[2],e[1],e[3]],origin="upper",cmap=ListedColormap(["#138442"]),alpha=np.minimum(small*1.6,.9),interpolation="nearest")
        ax.scatter(d.loc[~d.high_mse,"longitude"],d.loc[~d.high_mse,"latitude"],s=28,c="#bac1c8",edgecolors="#48515d",linewidths=.55,zorder=4,label="Other validation GPIs")
        ax.scatter(d.loc[d.high_mse,"longitude"],d.loc[d.high_mse,"latitude"],s=70,c="#bc2845",edgecolors="white",linewidths=.9,zorder=5,label="Highest-MSE quartile")
        ax.scatter([],[],marker="s",s=70,color="#138442",label="Mapped rice")
        ax.set_xlim(extent[0],extent[2]);ax.set_ylim(extent[1],extent[3]);ax.set_aspect(1/math.cos(math.radians(30.4)))
        ax.set_facecolor("#faf9f5");ax.grid(alpha=.14,linewidth=.7);ax.spines[["right","top"]].set_visible(False)
        ax.set_title(title,loc="left",fontsize=13,fontweight="bold");ax.set_xlabel("Longitude (°E)")
        ax.legend(loc="lower right",fontsize=8,framealpha=.95)
    axs[0].set_ylabel("Latitude (°N)")
    fig.suptitle("Rice maps and high-MSE validation locations",fontsize=16,x=.06,ha="left",fontweight="bold")
    fig.text(.06,.91,"Validation locations at exact coordinates, without map jitter",fontsize=10,color="#444")
    fig.text(.06,.055,"A: 31.06°N, 73.87°E     B: 30.85°N, 73.70°E     C: 30.66°N, 73.40°E\nSources: Zenodo 20923150 (2021) and 18468250 (2024). Green display aggregates the 10 m maps to about 200 m.\nDifferent map methods and years limit direct comparison; 2024 lies outside the 2021–2023 validation record.",fontsize=8.5,color="#444")
    fig.subplots_adjust(left=.06,right=.985,top=.85,bottom=.17,wspace=.11)
    fig.savefig(OUT/"rice_overlap_comparison.png",dpi=210,facecolor="white")
    fig.savefig(OUT/"rice_overlap_comparison.pdf",facecolor="white")
    plt.close(fig)
    for s in sources_2021+sources_2024:s.close()


