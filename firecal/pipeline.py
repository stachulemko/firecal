"""End-to-end analysis of an area of interest: data -> harmonized record -> calendar -> forecasts."""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from shapely.geometry import box
from sklearn.cluster import KMeans

from . import calendar as cal
from . import forecast as fcst
from . import harmonize as hz
from .config import AOI_DIR, SENSOR_CODE, SENSORS, WEATHER_POINTS_MAX, firms_map_key
from .geo import AOI, Grid, country_shapes, select_countries
from .sources import firms
from .sources import weather as wsrc

log = logging.getLogger(__name__)
Progress = Callable[[float, str], None]
PIPELINE_VERSION = 7

PRESETS: dict[str, dict] = {
    "poland": {"name": "Poland", "country": "Poland",
               "note": "spring meadow and field burning (March-April), summer stubble and forest fires"},
    "portugal": {"name": "Portugal (mainland)", "country": "Portugal", "clip": (-9.6, 36.9, -6.1, 42.2),
                 "note": "Mediterranean summer fire regime; extreme years 2003, 2005, 2017, 2024"},
    "delta-parana": {"name": "Paraná Delta wetlands (Argentina)", "bbox": (-60.9, -34.4, -58.3, -32.0),
                     "note": "wetland and pasture burning; record fires 2020 and 2022"},
    "cordoba": {"name": "Sierras de Córdoba (Argentina)", "bbox": (-65.4, -32.7, -63.9, -30.3),
                "note": "late-winter/spring fire season (Aug-Oct)"},
    "corrientes": {"name": "Corrientes / Iberá (Argentina)", "bbox": (-59.6, -30.4, -55.7, -27.3),
                   "note": "grassland and wetland fires; extreme drought fires in early 2022"},
    "galicia": {"name": "Galicia & Asturias (Spain)", "bbox": (-9.4, 41.8, -4.5, 43.8),
                "note": "Atlantic Iberia with spring and autumn agricultural burning"},
    "greece": {"name": "Greece", "country": "Greece", "clip": (19.3, 34.7, 26.7, 41.8),
               "note": "summer fire regime; extreme years 2007, 2021, 2023"},
}


def preset_aoi(key: str) -> AOI:
    p = PRESETS[key]
    if "bbox" in p:
        return AOI.from_bbox(*p["bbox"], name=p["name"])
    geom = next(g for n, g in country_shapes() if n == p["country"])
    if "clip" in p:
        geom = geom.intersection(box(*p["clip"]))
    return AOI(p["name"], geom.buffer(0))


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, dt.date, np.datetime64)):
        return str(pd.Timestamp(o).date())
    return o


@dataclass
class Analysis:
    aoi: AOI
    grid: Grid
    result: dict                              # JSON-ready products
    daily: pd.DataFrame
    bundle: dict = field(default_factory=dict)  # models and arrays for on-demand forecasts
    _cache: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def dir(self) -> Path:
        return AOI_DIR / self.aoi.id

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "result.json").write_text(json.dumps(_jsonable(self.result)))
        self.daily.to_parquet(self.dir / "daily.parquet")
        joblib.dump(self.bundle, self.dir / "bundle.joblib", compress=3)

    @classmethod
    def load(cls, aoi_id: str) -> "Analysis | None":
        d = AOI_DIR / aoi_id
        if not (d / "result.json").exists():
            return None
        result = json.loads((d / "result.json").read_text())
        if result.get("meta", {}).get("pipeline_version") != PIPELINE_VERSION:
            return None
        aoi = AOI.from_geojson(result["meta"]["aoi"], result["meta"]["name"])
        grid = Grid(**result["meta"]["grid"])
        daily = pd.read_parquet(d / "daily.parquet")
        bundle = joblib.load(d / "bundle.joblib") if (d / "bundle.joblib").exists() else {}
        return cls(aoi, grid, result, daily, bundle)

    # -- on-demand forecasts (operational re-issue or historical backtest) ------------------------
    def _tables(self, use_eval: bool):
        key = "eval" if use_eval else "final"
        with self._lock:
            if key not in self._cache:
                b = self.bundle
                harm = _HarmLite(b["H"], b["source"], b["cube_meta"], self.grid, b["sensors"], b["y_ml"])
                fc: fcst.DailyForecaster = b["forecaster"]
                ft = fcst.FireTables(harm, fc.eval_clim_years if use_eval else fc.clim_years)
                if "wx" not in self._cache:
                    self._cache["wx"] = fcst.WeatherCube.from_frames(
                        b["weather_points"], b["weather_frames"], b["cell_point"], b["cube_meta"]["day0"],
                        b["weather_ndays"], b["weather_last_observed"])
                self._cache[key] = ft
            return self._cache[key], self._cache["wx"]

    def forecast_from(self, issue_date: str) -> dict:
        """Backtest from a past date. Dates in years the evaluation model never saw are out-of-sample."""
        fc: fcst.DailyForecaster = self.bundle["forecaster"]
        day = hz.day_number(issue_date)
        use_eval = pd.Timestamp(issue_date).year >= fc.first_unseen_year
        ft, wx = self._tables(use_eval)
        idx = day - ft.day0
        if idx < 8 or idx >= ft.ndays:
            raise ValueError("issue date outside the record")
        if np.isnan(ft.H[:, idx]).all():
            raise ValueError("no fire observations on the issue date")
        out = fcst.predict_daily(fc, ft, wx, self.bundle["oni"], idx, use_eval=use_eval)
        out["mode"] = "out-of-sample backtest" if use_eval else "in-sample backtest"
        out["model_trained_until"] = (fc.first_unseen_year - 1) if use_eval else max(fc.clim_years)
        return _jsonable(out)


class _CubeLite:
    """The parts of ``CellCube`` the forecast tables need, rebuilt from a saved bundle."""

    def __init__(self, grid: Grid, cells: np.ndarray, in_aoi: np.ndarray, day0: int, ndays: int):
        self.grid, self.cells, self.in_aoi, self.day0, self.ndays = grid, cells, in_aoi, day0, ndays
        self.days = np.arange(day0, day0 + ndays)
        self.ncell = len(cells)

    def local_neighbors(self, radius: int = 1) -> np.ndarray:
        lut = np.full(self.grid.ncells, -1, dtype=np.int64)
        lut[self.cells] = np.arange(self.ncell)
        nb = self.grid.neighbors(radius)[self.cells]
        return np.where(nb >= 0, lut[np.maximum(nb, 0)], -1)


class _HarmLite:
    """Just enough of ``Harmonized`` to rebuild forecast tables from a saved bundle."""

    def __init__(self, H: np.ndarray, source: np.ndarray, cube_meta: dict, grid: Grid, sensors: dict,
                 y_ml: np.ndarray):
        self.H, self.source, self.sensors, self.y_ml = H, source, sensors, y_ml
        self.cube = _CubeLite(grid, cube_meta["cells"], cube_meta["in_aoi"], cube_meta["day0"], H.shape[1])


# ---------------------------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------------------------

def _stage(progress: Progress | None, lo: float, hi: float) -> Progress:
    def f(frac: float, msg: str):
        if progress:
            progress(lo + (hi - lo) * max(0.0, min(1.0, frac)), msg)
    return f


def run(aoi: AOI, progress: Progress | None = None, force: bool = False, max_age_hours: float = 6.0) -> Analysis:
    if not force:
        cached = Analysis.load(aoi.id)
        if cached is not None:
            age = time.time() - cached.result["meta"]["created_ts"]
            if age < max_age_hours * 3600:
                if progress:
                    progress(1.0, "loaded cached analysis")
                return cached
    t_start = time.time()
    say = lambda f, m: progress(f, m) if progress else log.info(m)  # noqa: E731
    today = dt.datetime.now(dt.timezone.utc).date()
    today_day = hz.day_number(today)

    # ---- 1. grid & FIRMS archive -------------------------------------------------------------
    grid = Grid.for_aoi(aoi)
    gbounds = (grid.lon0, grid.lat0, grid.lon0 + grid.nx * grid.res, grid.lat0 + grid.ny * grid.res)
    countries, skipped = select_countries(aoi.geometry, gbounds, country_shapes())
    if not countries:
        raise ValueError("the area does not overlap any land covered by FIRMS")
    say(0.01, f"grid {grid.ny}x{grid.nx} @ {grid.res} deg; FIRMS countries: {', '.join(countries)}"
              + (f" (skipped, barely in the grid: {', '.join(skipped)})" if skipped else ""))
    if not aoi.named:  # e.g. "Greece · 36.4°N–38.4°N, 21.0°E–24.2°E"
        aoi = AOI(f"{' / '.join(c.replace('_', ' ') for c in countries)} · {aoi.name}", aoi.geometry)
    last_year = firms.latest_archive_year()
    pts, counts = firms.load_archive(countries, gbounds, last_year=last_year, progress=_stage(progress, 0.02, 0.40))
    static = hz.static_source_keys(pts)

    # ---- 2. recent data: FIRMS API (if a MAP_KEY is configured) + keyless NRT week -----------
    extra = {}
    recent_parts = []
    api_used = False
    gap_start = dt.date(last_year + 1, 1, 1)
    if firms_map_key():
        try:
            api_pts = firms.load_recent_via_api(gbounds, gap_start, today - dt.timedelta(days=1),
                                                _stage(progress, 0.40, 0.48))
            recent_parts.append(api_pts)
            for code, g in api_pts.groupby("sensor"):
                extra.setdefault(int(code), []).append(np.arange(hz.day_number(gap_start), today_day))
            api_used = True
        except Exception as e:  # keep going with the NRT week only
            log.warning("FIRMS API failed: %s", e)
            say(0.48, f"FIRMS API unavailable ({e}); continuing with the public 7-day feed")
    nrt_global = []
    try:
        for feed in ("VIIRS_SNPP", "VIIRS_NOAA20", "VIIRS_NOAA21", "MODIS"):
            g = firms.fetch_nrt(feed)
            nrt_global.append(g.groupby(["day", "sensor"]).size().rename("n").reset_index())
            w, s, e, n = gbounds
            recent_parts.append(g[(g.lon >= w) & (g.lon <= e) & (g.lat >= s) & (g.lat <= n)])
            say(0.50, f"FIRMS near-real-time feed {feed}")
    except Exception as e:
        log.warning("NRT feed failed: %s", e)
        say(0.50, f"near-real-time feed unavailable ({e})")
    nrt_first_day = None
    if nrt_global:
        # a sensor covers an NRT day if its global total is a normal full day (drops partial days)
        cnt = pd.concat(nrt_global)
        for code, g in cnt.groupby("sensor"):
            c = g.groupby("day")["n"].sum()
            c = c[c.index < today_day]
            if len(c):
                full_days = c.index[c >= 0.5 * c.median()].to_numpy()
                extra.setdefault(int(code), []).append(full_days)
                if len(full_days):
                    nrt_first_day = int(min(full_days)) if nrt_first_day is None else min(nrt_first_day, int(min(full_days)))
    recent = pd.concat(recent_parts, ignore_index=True) if recent_parts else firms.empty_points()
    if not recent.empty:
        recent = recent[recent["day"] > hz.day_number(f"{last_year}-12-31")]
        recent = recent.drop_duplicates(subset=["lat", "lon", "day", "hhmm", "sensor"])
    all_pts = pd.concat([pts, recent], ignore_index=True)
    extra = {k: np.unique(np.concatenate(v)) for k, v in extra.items()}

    # ---- 3. QA, availability, gridding ---------------------------------------------------------
    q = hz.qa_filter(all_pts, static)
    last_day = int(max(q["day"].max(), hz.day_number(f"{last_year}-12-31")))
    last_day = min(last_day, today_day - 1)
    ndays = last_day - hz.DAY0 + 1
    avail = hz.sensor_availability(counts, hz.DAY0, ndays, extra_coverage=extra)
    aoi_mask = grid.cell_mask(aoi.geometry)
    cube = hz.build_cube(q, grid, aoi_mask, hz.DAY0, last_day, avail)
    say(0.52, f"{len(q):,} quality-controlled hotspots in {cube.ncell} active cells "
              f"({int(cube.in_aoi.sum())} inside the area)")

    # ---- 4. harmonization ----------------------------------------------------------------------
    tm = hz.fit_transfer(cube, log_fn=lambda m: say(0.60, m))
    harm = hz.harmonize(cube, tm)
    say(0.66, "harmonized record assembled")

    # ---- 5. calendar ---------------------------------------------------------------------------
    calendar, daily, clim = cal.build_calendar(harm.daily)
    say(0.68, "burning activity calendar built")

    # ---- 6. weather & climate drivers ----------------------------------------------------------
    lat, lon = grid.centers()
    clat, clon = lat[cube.cells], lon[cube.cells]
    wts = np.nansum(harm.H, axis=1) + 1.0
    area = (gbounds[2] - gbounds[0]) * (gbounds[3] - gbounds[1])
    k = int(np.clip(round(area / 2.0), 1, min(WEATHER_POINTS_MAX, cube.ncell)))
    km = KMeans(k, n_init=10, random_state=0).fit(np.column_stack([clat, clon]), sample_weight=wts)
    wpts = sorted({wsrc.snap_power(a, b) for a, b in km.cluster_centers_})
    power = wsrc.fetch_power(wpts, progress=_stage(progress, 0.68, 0.74))
    try:
        om = wsrc.fetch_openmeteo(wpts)
    except Exception as e:
        log.warning("Open-Meteo failed: %s", e)
        om = None
    frames, winfo = [], []
    for i, p in enumerate(power):
        if om is not None:
            m, inf = wsrc.blend_forecast(p, om[i])
        else:
            m, inf = p.dropna(subset=["T2M_MAX"]), {"last_observed": str(p.dropna(subset=["T2M_MAX"]).index.max().date())}
        frames.append(m)
        winfo.append(inf)
    w_last_obs = min(hz.day_number(i["last_observed"]) for i in winfo)
    w_end = min(f.index.max() for f in frames)
    w_ndays = hz.day_number(w_end.date()) - hz.DAY0 + 1
    d2 = (clat[:, None] - np.array([a for a, _ in wpts])[None]) ** 2 + \
         (clon[:, None] - np.array([b for _, b in wpts])[None]) ** 2
    cell_point = np.argmin(d2, axis=1)
    wx = fcst.WeatherCube.from_frames(wpts, frames, cell_point, hz.DAY0, w_ndays, w_last_obs)
    oni = wsrc.fetch_oni()
    say(0.75, f"weather drivers: {len(wpts)} NASA POWER points, Open-Meteo forecast to {w_end.date()}")

    # ---- 7. forecasting models -----------------------------------------------------------------
    fc, ft = fcst.train_daily(harm, wx, oni, log_fn=lambda m: say(0.85, m))
    say(0.93, "daily forecast model trained and evaluated")
    ol, monthly = fcst.train_outlook(ft, wx, oni)
    outlook = fcst.predict_outlook(ol, monthly)

    # operational forecast from the last complete day with fire observations
    has_data = np.flatnonzero(harm.source > 0)
    issue_idx = int(has_data[-1]) if len(has_data) else None
    operational = None
    if issue_idx is not None and (today_day - (hz.DAY0 + issue_idx)) <= 3:
        operational = fcst.predict_daily(fc, ft, wx, oni, issue_idx)
        operational["mode"] = "operational"
    say(0.97, "forecasts issued")

    # ---- 8. products ---------------------------------------------------------------------------
    dates = hz.to_dates(cube.days)
    yrs = hz.years_of(cube.days)
    full = [y for y in calendar["baseline_years"]]
    Hs = harm.H[:, np.isin(yrs, full)]
    mean_annual = np.nansum(Hs, axis=1) / max(len(full), 1)
    yearly = np.stack([np.nansum(harm.H[:, yrs == y], axis=1) for y in full], axis=1) if full else np.zeros((cube.ncell, 1))
    months = (dates.astype("datetime64[M]").astype(int) % 12)
    monthly_cell = np.stack([np.nanmean(harm.H[:, (months == m) & np.isin(yrs, full)], axis=1) for m in range(12)], 1)
    recent_mask = (cube.days > last_day - 7) & (harm.source > 0)
    cells = {
        "res": grid.res,
        "lat": np.round(clat, 4), "lon": np.round(clon, 4), "in_aoi": cube.in_aoi.astype(int),
        "mean_annual": np.round(mean_annual, 2),
        "burn_frequency": np.round((yearly >= 1).mean(axis=1), 3),
        "peak_month": np.argmax(monthly_cell, axis=1) + 1,
        "recent7": np.round(np.nansum(harm.H[:, recent_mask], axis=1), 2) if recent_mask.any() else None,
    }
    recent_pts = q[(q["day"] > last_day - 7)]
    recent_pts = recent_pts[aoi_mask[np.maximum(grid.cell_of(recent_pts.lat.values, recent_pts.lon.values), 0)]
                            & (grid.cell_of(recent_pts.lat.values, recent_pts.lon.values) >= 0)]
    recent_pts = recent_pts.sort_values("frp", ascending=False).head(5000)
    code_to_key = {v: k for k, v in SENSOR_CODE.items()}
    hotspots = {
        "lat": recent_pts.lat.round(4).tolist(), "lon": recent_pts.lon.round(4).tolist(),
        "date": [str(d) for d in hz.to_dates(recent_pts.day.values)],
        "sensor": [code_to_key[int(s)] for s in recent_pts.sensor], "frp": recent_pts.frp.round(1).tolist(),
    }
    val = tm.validation
    validation = {
        "date": [str(d.date()) for d in pd.DatetimeIndex(val["date"])],
        "observed": val["observed_snpp"].round(1).tolist(), "harmonized": val["harmonized_modis"].round(1).tolist(),
        "raw_modis": val["raw_modis"].round(1).tolist(), "kind": val["kind"].tolist(),
    }
    hd = harm.daily.assign(year=pd.DatetimeIndex(harm.daily["date"]).year)
    annual_sensors = []
    for y, g in hd.groupby("year"):
        row = {"year": int(y), "harmonized": float(g["y"].sum()), "harmonized_days": int(g["y"].notna().sum()),
               "modis": float((g["n_MODIS_T"].fillna(0) + g["n_MODIS_A"].fillna(0)).sum()),
               "modis_days": int((g["n_MODIS_T"].notna() | g["n_MODIS_A"].notna()).sum())}
        for k in ("VIIRS_SNPP", "VIIRS_NOAA20", "VIIRS_NOAA21"):
            row[k.lower()] = float(g[f"n_{k}"].sum())
            row[f"{k.lower()}_days"] = int(g[f"n_{k}"].notna().sum())
        annual_sensors.append(row)
    src_days = pd.Series(harm.source).value_counts().to_dict()
    data_end = str(hz.to_dates(np.array([hz.DAY0 + issue_idx]))[0]) if issue_idx is not None else None
    sensors_used = {k: {"label": s.label, "first": None, "last": None, "days": 0} for k, s in SENSORS.items()}
    for key, code in SENSOR_CODE.items():
        av = avail[code]
        if av.any():
            sensors_used[key].update(first=str(dates[av][0]), last=str(dates[av][-1]), days=int(av.sum()))
    result = {
        "meta": {
            "id": aoi.id, "name": aoi.name, "aoi": aoi.to_geojson(), "bounds": aoi.bounds, "grid": grid.to_dict(),
            "countries": countries, "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "created_ts": time.time(), "runtime_s": round(time.time() - t_start, 1),
            "pipeline_version": PIPELINE_VERSION, "archive_last_year": last_year,
            "data_start": str(dates[0]), "data_end": data_end, "firms_api_used": api_used,
            "gap_without_key": None if api_used or nrt_first_day is None else
            [f"{last_year + 1}-01-01", str(hz.to_dates(np.array([nrt_first_day - 1]))[0])],
            "sensors": sensors_used, "hotspots_qc": int(len(q)),
            "days_by_source": {hz.SOURCE_LABELS[int(k)]: int(v) for k, v in src_days.items()},
            "weather_points": wpts, "weather_info": winfo[0] if winfo else None,
        },
        "calendar": calendar,
        "harmonization": {"metrics": tm.metrics, "validation": validation, "annual": annual_sensors,
                          "monthly_ratio": tm.level.ratios["both"].tolist()},
        "forecast": {
            "operational": operational, "outlook": outlook,
            "daily_metrics": fc.metrics, "outlook_metrics": ol.metrics, "feature_importance": fc.importance,
        },
        "cells": cells,
        "hotspots": hotspots,
    }
    bundle = {
        "H": harm.H.astype(np.float32), "source": harm.source, "sensors": harm.sensors, "y_ml": harm.y_ml,
        "cube_meta": {"cells": cube.cells, "in_aoi": cube.in_aoi, "day0": cube.day0},
        "forecaster": fc, "weather_points": wpts, "weather_frames": frames, "cell_point": cell_point,
        "weather_ndays": w_ndays, "weather_last_observed": w_last_obs, "oni": oni,
    }
    analysis = Analysis(aoi, grid, _jsonable(result), daily, bundle)
    analysis._cache["final"], analysis._cache["wx"] = ft, wx
    analysis.save()
    say(1.0, f"done in {time.time() - t_start:.0f}s")
    return analysis
