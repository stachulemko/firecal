"""Weather and climate drivers.

* NASA POWER (MERRA-2 / GEOS based, satellite-assimilated) daily point data since 1981 - history
  used for training, no API key, ~3 s for 25 years at one point.
* Open-Meteo forecast API - the last month and the next 16 days, bias-corrected to POWER on the
  overlapping days so the forecast model sees the same climate it was trained on.
* NOAA CPC Oceanic Nino Index (ENSO) - monthly.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from ..config import RAW_DIR
from .http import get_json, get_text

log = logging.getLogger(__name__)

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_VARS = ["T2M_MAX", "T2M_MIN", "T2MDEW", "RH2M", "PRECTOTCORR", "WS10M_MAX", "GWETTOP", "GWETROOT",
              "ALLSKY_SFC_SW_DWN"]
POWER_START = dt.date(2000, 1, 1)
POWER_DIR = RAW_DIR / "power"

OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"
# Open-Meteo daily variable -> (POWER variable, scale)
OPENMETEO_VARS = {
    "temperature_2m_max": ("T2M_MAX", 1.0),
    "temperature_2m_min": ("T2M_MIN", 1.0),
    "dew_point_2m_mean": ("T2MDEW", 1.0),
    "relative_humidity_2m_mean": ("RH2M", 1.0),
    "precipitation_sum": ("PRECTOTCORR", 1.0),
    "wind_speed_10m_max": ("WS10M_MAX", 1 / 3.6),  # km/h -> m/s
    "shortwave_radiation_sum": ("ALLSKY_SFC_SW_DWN", 1.0),
}
ADDITIVE = {"T2M_MAX", "T2M_MIN", "T2MDEW", "RH2M"}  # bias-corrected by offset, the rest by ratio

ONI_URL = "https://psl.noaa.gov/data/correlation/oni.data"


def snap_power(lat: float, lon: float) -> tuple[float, float]:
    """Snap to the MERRA-2 grid POWER serves (0.5 x 0.625 deg) so nearby requests share a cache."""
    return round(round(lat * 2) / 2, 3), round(round(lon / 0.625) * 0.625, 4)


def fetch_power_point(lat: float, lon: float, end: dt.date | None = None) -> pd.DataFrame:
    """Daily POWER series for one grid point from 2000 to ``end`` (cached, extended incrementally)."""
    lat, lon = snap_power(lat, lon)
    end = end or dt.date.today()
    path = POWER_DIR / f"{lat:+08.3f}_{lon:+09.4f}.parquet"
    cached = pd.read_parquet(path) if path.exists() else None
    if cached is not None:
        last_valid = cached.dropna(subset=["T2M_MAX"]).index.max()
        fresh = time.time() - path.stat().st_mtime < 12 * 3600
        if fresh or (last_valid is not None and (pd.Timestamp(end) - last_valid).days <= 3):
            return cached
        start = (last_valid - pd.Timedelta(days=10)).date() if last_valid is not None else POWER_START
    else:
        start = POWER_START
    params = {"parameters": ",".join(POWER_VARS), "community": "AG", "latitude": lat, "longitude": lon,
              "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"), "format": "JSON"}
    js = get_json(POWER_URL, params=params, timeout=300)
    p = js["properties"]["parameter"]
    df = pd.DataFrame({k: pd.Series(v) for k, v in p.items()})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.replace(-999.0, np.nan).astype("float32")
    if cached is not None:
        df = pd.concat([cached[cached.index < df.index.min()], df])
    POWER_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def fetch_power(points: list[tuple[float, float]], end: dt.date | None = None, progress=None) -> list[pd.DataFrame]:
    out: list[pd.DataFrame | None] = [None] * len(points)
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(fetch_power_point, la, lo, end): i for i, (la, lo) in enumerate(points)}
        for n, fut in enumerate(futs):
            out[futs[fut]] = fut.result()
            if progress:
                progress((n + 1) / len(points), f"NASA POWER weather point {n + 1}/{len(points)}")
    return out  # type: ignore[return-value]


def fetch_openmeteo(points: list[tuple[float, float]], past_days: int = 45, forecast_days: int = 16) -> list[pd.DataFrame]:
    """Recent past + forecast daily weather for several points in one request (POWER units)."""
    cache = RAW_DIR / "openmeteo"
    cache.mkdir(parents=True, exist_ok=True)
    key = "_".join(f"{la:.3f},{lo:.3f}" for la, lo in points)
    path = cache / f"{abs(hash((key, past_days, forecast_days))) % 10**12}.json"
    if path.exists() and time.time() - path.stat().st_mtime < 3 * 3600:
        js = json.loads(path.read_text())
    else:
        params = {"latitude": ",".join(f"{la:.4f}" for la, _ in points),
                  "longitude": ",".join(f"{lo:.4f}" for _, lo in points),
                  "daily": ",".join(OPENMETEO_VARS), "past_days": past_days, "forecast_days": forecast_days,
                  "timezone": "UTC"}
        js = get_json(OPENMETEO_URL, params=params, timeout=120)
        path.write_text(json.dumps(js))
    if isinstance(js, dict):
        js = [js]
    out = []
    for item in js:
        d = item["daily"]
        df = pd.DataFrame({pw: np.asarray(d[k], dtype=float) * scale for k, (pw, scale) in OPENMETEO_VARS.items()},
                          index=pd.to_datetime(d["time"]))
        out.append(df.astype("float32"))
    return out


def blend_forecast(power: pd.DataFrame, om: pd.DataFrame, min_overlap: int = 10) -> tuple[pd.DataFrame, dict]:
    """POWER history extended with bias-corrected Open-Meteo data after POWER's last valid day.

    Offsets (temperatures, humidity) or ratios (precipitation, wind, radiation) are estimated on the
    days both sources cover. Soil wetness is not forecast and is held at its last observed value.
    """
    last = power.dropna(subset=["T2M_MAX"]).index.max()
    both = om.index[(om.index <= last) & om.index.isin(power.index)]
    corr = {}
    ext = om[om.index > last].copy()
    for col in om.columns:
        if len(both) >= min_overlap:
            a = power.loc[both, col].astype(float)
            b = om.loc[both, col].astype(float)
            ok = a.notna() & b.notna()
            if col in ADDITIVE:
                corr[col] = ("offset", float((a[ok] - b[ok]).mean()))
            else:
                corr[col] = ("ratio", float(np.clip((a[ok].mean() + 1e-3) / (b[ok].mean() + 1e-3), 0.5, 2.0)))
        else:
            corr[col] = ("offset", 0.0) if col in ADDITIVE else ("ratio", 1.0)
        kind, v = corr[col]
        ext[col] = ext[col] + v if kind == "offset" else ext[col] * v
    for col in power.columns:
        if col not in ext.columns:
            ext[col] = power.loc[last, col]
    ext["RH2M"] = ext["RH2M"].clip(1, 100)
    ext["PRECTOTCORR"] = ext["PRECTOTCORR"].clip(lower=0)
    merged = pd.concat([power[power.index <= last], ext[power.columns]])
    return merged.astype("float32"), {"last_observed": last.date().isoformat(), "corrections": corr}


def fetch_oni() -> pd.Series:
    """Monthly Oceanic Nino Index (3-month running mean SST anomaly in Nino 3.4), indexed by month start."""
    path = RAW_DIR / "oni.csv"
    if path.exists() and time.time() - path.stat().st_mtime < 7 * 86400:
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        return s
    text = get_text(ONI_URL)
    rows = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) != 13 or not parts[0].isdigit():
            continue
        year = int(parts[0])
        for m, v in enumerate(parts[1:], start=1):
            val = float(v)
            if val > -90:
                rows.append((pd.Timestamp(year=year, month=m, day=1), val))
    s = pd.Series(dict(rows), name="oni").sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(path)
    return s
