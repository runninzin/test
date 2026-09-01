# -*- coding: utf-8 -*-
"""
make_figs.py — ECMWF IFS open data の「最新解析」から総観場の図を作る（速報用・静止画）
データ: ECMWF IFS open data (CC BY 4.0) — 出典表示で商用利用可。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs"
FIGS = SITE / "figs"
MANIFEST = SITE / "manifest.json"

EXTENT = [113.0, 157.0, 15.0, 55.0]
LEVELS_HPA = [500, 850]
CREDIT = "Source: ECMWF IFS open data (CC BY 4.0)  /  作図: 走る人参"
A_EARTH = 6.371e6
OMEGA_E = 7.292e-5


def _herbie(cycle: dt.datetime, fxx: int, cache_dir: Path):
    from herbie import Herbie
    return Herbie(cycle.strftime("%Y-%m-%d %H:%M"), model="ifs", product="oper",
                  fxx=fxx, save_dir=str(cache_dir), verbose=False)


def _floor6(t: dt.datetime) -> dt.datetime:
    return t.replace(hour=(t.hour // 6) * 6, minute=0, second=0, microsecond=0)


def latest_cycle(now_utc: dt.datetime, cache_dir: Path) -> dt.datetime:
    c0 = _floor6(now_utc)
    for back in range(0, 10):
        c = c0 - dt.timedelta(hours=6 * back)
        try:
            if _herbie(c, 0, cache_dir).grib:
                return c
        except Exception:
            pass
    raise RuntimeError("最新の IFS サイクルが見つかりません（少し待って再実行）。")


def fetch_analysis(cycle: dt.datetime, cache_dir: Path):
    import xarray as xr
    H = _herbie(cycle, 0, cache_dir)
    lvl = "|".join(str(x) for x in LEVELS_HPA)
    pl = H.xarray(rf":(?:gh|t|u|v):(?:{lvl}):", remove_grib=False)
    sfc = H.xarray(r":msl:", remove_grib=False)

    def _std(d):
        ren = {}
        if "latitude" in d.coords:
            ren["latitude"] = "lat"
        if "longitude" in d.coords:
            ren["longitude"] = "lon"
        if "isobaricInhPa" in d.coords:
            ren["isobaricInhPa"] = "level"
        d = d.rename(ren)
        d = d.assign_coords(lon=(d["lon"] % 360)).sortby("lon").sortby("lat")
        lo0, lo1, la0, la1 = EXTENT
        return d.sel(lat=slice(la0, la1), lon=slice(lo0 % 360, lo1 % 360))

    out = {}
    for d in (pl if isinstance(pl, list) else [pl]):
        d = _std(d)
        for v in d.data_vars:
            out[v] = d[v]
    for d in (sfc if isinstance(sfc, list) else [sfc]):
        d = _std(d)
        for v in d.data_vars:
            out[v] = d[v]
    return out


def _ddx(f, lat, lon):
    dlon = np.deg2rad(np.gradient(lon))
    dx = A_EARTH * np.cos(np.deg2rad(lat))[:, None] * dlon[None, :]
    dx = np.where(np.abs(dx) < 1.0, 1.0, dx)
    return np.gradient(np.asarray(f, float), axis=-1) / dx


def _ddy(f, lat, lon):
    dlat = np.deg2rad(np.gradient(lat))
    dy = A_EARTH * dlat
    return np.gradient(np.asarray(f, float), axis=-2) / dy[:, None]


def absolute_vorticity(u, v, lat, lon, smooth=1.6):
    if smooth and smooth > 0:
        from scipy.ndimage import gaussian_filter
        u = gaussian_filter(np.asarray(u, float), smooth, mode="nearest")
        v = gaussian_filter(np.asarray(v, float), smooth, mode="nearest")
    zeta = _ddx(v, lat, lon) - _ddy(u, lat, lon)
    f = 2.0 * OMEGA_E * np.sin(np.deg2rad(lat))[:, None]
    return zeta + f


def _setup_mpl():
    import logging
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    avail = {f.name for f in fm.fontManager.ttflist}
    for cand in ("Yu Gothic", "Meiryo", "MS Gothic", "BIZ UDGothic",
                 "Noto Sans CJK JP", "IPAexGothic", "TakaoPGothic"):
        if cand in avail:
            plt.rcParams["font.family"] = [cand, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _basemap(ax):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    try:
        coast = cfeature.COASTLINE.with_scale("50m")
        border = cfeature.BORDERS.with_scale("50m")
        ax.add_feature(coast, edgecolor="white", linewidth=2.4, zorder=6)
        ax.add_feature(coast, edgecolor="#101010", linewidth=1.0, zorder=6.1)
        ax.add_feature(border, edgecolor="#333", linewidth=0.5, zorder=5)
    except Exception:
        ax.coastlines(linewidth=1.0, color="black", zorder=6)
    gl = ax.gridlines(draw_labels=True, linestyle="--", color="gray", alpha=0.3)
    gl.top_labels = gl.right_labels = False


def _title(fig, text, valid_jst):
    fig.text(0.06, 0.955, text, fontsize=13, fontweight="bold", va="center")
    fig.text(0.94, 0.955, valid_jst, fontsize=10, ha="right", va="center", color="0.25")


def fig_500(data, cycle, valid_jst, path):
    import cartopy.crs as ccrs
    plt = _setup_mpl()
    gh = data["gh"].sel(level=500)
    u = data["u"].sel(level=500)
    v = data["v"].sel(level=500)
    lat = gh["lat"].values
    lon = gh["lon"].values
    av = absolute_vorticity(u.values, v.values, lat, lon) * 1e5

    fig = plt.figure(figsize=(8.2, 8.2))
    ax = fig.add_axes([0.06, 0.05, 0.88, 0.86], projection=ccrs.PlateCarree())
    _basemap(ax)
    cf = ax.contourf(lon, lat, av, levels=np.arange(0, 42, 3), cmap="YlOrRd",
                     extend="max", transform=ccrs.PlateCarree(), zorder=2)
    cs = ax.contour(lon, lat, gh.values, levels=np.arange(4800, 6100, 60),
                    colors="black", linewidths=1.0, transform=ccrs.PlateCarree(), zorder=4)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%d")
    cb = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("絶対渦度 (10$^{-5}$ s$^{-1}$)")
    _title(fig, "500 hPa 高度 [m] ・ 絶対渦度", valid_jst)
    fig.text(0.5, 0.012, CREDIT, ha="center", fontsize=7.5, color="0.3")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_850(data, cycle, valid_jst, path):
    import cartopy.crs as ccrs
    plt = _setup_mpl()
    msl = data["msl"] / 100.0
    t850 = data["t"].sel(level=850) - 273.15
    u850 = data["u"].sel(level=850)
    v850 = data["v"].sel(level=850)
    lat = t850["lat"].values
    lon = t850["lon"].values

    fig = plt.figure(figsize=(8.2, 8.2))
    ax = fig.add_axes([0.06, 0.05, 0.88, 0.86], projection=ccrs.PlateCarree())
    _basemap(ax)
    cf = ax.contourf(lon, lat, t850.values, levels=np.arange(-30, 33, 3),
                     cmap="RdYlBu_r", extend="both", transform=ccrs.PlateCarree(), zorder=2)
    cs = ax.contour(lon, lat, msl.values, levels=np.arange(940, 1052, 4),
                    colors="black", linewidths=1.0, transform=ccrs.PlateCarree(), zorder=4)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%d")
    sj = max(1, len(lat) // 20)
    si = max(1, len(lon) // 20)
    ax.barbs(lon[::si], lat[::sj],
             u850.values[::sj, ::si] * 1.94384, v850.values[::sj, ::si] * 1.94384,
             length=5, linewidth=0.5, transform=ccrs.PlateCarree(), zorder=5)
    cb = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("850 hPa 気温 (°C)")
    _title(fig, "地上気圧 [hPa] ・ 850 hPa 気温・風", valid_jst)
    fig.text(0.5, 0.012, CREDIT, ha="center", fontsize=7.5, color="0.3")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def load_manifest() -> dict:
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cache", default=os.environ.get("WX_CACHE", ".herbie_cache"))
    args = ap.parse_args()

    FIGS.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    cycle = latest_cycle(now, cache_dir)
    cycle_iso = cycle.strftime("%Y-%m-%dT%H:00:00Z")
    print(f"[wx] 最新サイクル: {cycle_iso}")

    prev = load_manifest()
    if prev.get("init_time") == cycle_iso and not args.force:
        print("[wx] 既に最新サイクルの図がある。何もしない。")
        return 0

    data = fetch_analysis(cycle, cache_dir)
    valid_jst = (cycle + dt.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M JST 初期値")

    fig_500(data, cycle, valid_jst, FIGS / "500hPa.png")
    print("[wx]  -> docs/figs/500hPa.png")
    fig_850(data, cycle, valid_jst, FIGS / "850hPa.png")
    print("[wx]  -> docs/figs/850hPa.png")

    next_cycle = cycle + dt.timedelta(hours=6)
    next_expected = next_cycle + dt.timedelta(hours=7)
    manifest = {
        "init_time": cycle_iso,
        "init_time_jst": valid_jst,
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "next_cycle": next_cycle.strftime("%Y-%m-%dT%H:00:00Z"),
        "next_update_expected": next_expected.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "ECMWF IFS open data (CC BY 4.0)",
        "figures": [
            {"file": "figs/500hPa.png", "title": "500 hPa 高度・絶対渦度"},
            {"file": "figs/850hPa.png", "title": "地上気圧・850 hPa 気温・風"},
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[wx] manifest 更新: {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
