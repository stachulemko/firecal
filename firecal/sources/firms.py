"""NASA FIRMS active fire hotspots: yearly country archive, global NRT feeds and the area API.

All readers return the same normalized point schema:

    lat, lon (float32) | day (int32, days since 1970-01-01) | hhmm (int16, UTC) | sensor (int8 code)
    conf (int8: 0 low, 1 nominal, 2 high) | frp (float32, MW) | is_day (bool)
    type (int8: 0 vegetation, 1 volcano, 2 static land source, 3 offshore, -1 unknown/NRT)
    scan, track (float32, km) | bt (float32, K: MODIS 21/22 um or VIIRS I4 brightness temperature)
"""
from __future__ import annotations

import datetime as dt
import io
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from ..config import (ARCHIVE_PRODUCTS, DOWNLOAD_WORKERS, FIRMS_BASE, MODIS_CONF_HIGH, MODIS_CONF_NOMINAL,
                      NRT_FEEDS, NRT_TTL_HOURS, RAW_DIR, SENSOR_CODE, SENSORS, firms_map_key)
from .http import NotFound, download, get_text, session

log = logging.getLogger(__name__)
Progress = Callable[[float, str], None]

COLUMNS = ["lat", "lon", "day", "hhmm", "sensor", "conf", "frp", "is_day", "type", "scan", "track", "bt"]
ARCHIVE_DIR = RAW_DIR / "firms_archive"
NRT_DIR = RAW_DIR / "firms_nrt"
API_DIR = RAW_DIR / "firms_api"


def empty_points() -> pd.DataFrame:
    return pd.DataFrame({
        "lat": pd.Series(dtype="float32"), "lon": pd.Series(dtype="float32"), "day": pd.Series(dtype="int32"),
        "hhmm": pd.Series(dtype="int16"), "sensor": pd.Series(dtype="int8"), "conf": pd.Series(dtype="int8"),
        "frp": pd.Series(dtype="float32"), "is_day": pd.Series(dtype="bool"), "type": pd.Series(dtype="int8"),
        "scan": pd.Series(dtype="float32"), "track": pd.Series(dtype="float32"), "bt": pd.Series(dtype="float32"),
    })


def _sensor_codes(raw: pd.DataFrame, family: str, default_sensor: str | None) -> np.ndarray:
    if family == "MODIS":
        sat = raw["satellite"].astype(str).str.strip().str.upper().str[0]
        codes = np.where(sat == "A", SENSOR_CODE["MODIS_A"], SENSOR_CODE["MODIS_T"])
        return codes.astype(np.int8)
    return np.full(len(raw), SENSOR_CODE[default_sensor], dtype=np.int8)


def normalize(raw: pd.DataFrame, family: str, sensor: str | None = None) -> pd.DataFrame:
    """Convert a FIRMS CSV frame (archive, NRT or API flavour) to the normalized schema."""
    if raw is None or len(raw) == 0:
        return empty_points()
    if family == "MODIS":
        c = pd.to_numeric(raw["confidence"], errors="coerce").fillna(0).to_numpy()
        conf = np.where(c >= MODIS_CONF_HIGH, 2, np.where(c >= MODIS_CONF_NOMINAL, 1, 0))
        bt = raw["brightness"]
    else:
        first = raw["confidence"].astype(str).str.strip().str.lower().str[0]
        conf = np.select([first == "h", first == "n"], [2, 1], 0)
        bt = raw["bright_ti4"]
    day = pd.to_datetime(raw["acq_date"].astype(str), format="%Y-%m-%d").to_numpy().astype("datetime64[D]")
    out = pd.DataFrame({
        "lat": raw["latitude"].astype("float32").to_numpy(),
        "lon": raw["longitude"].astype("float32").to_numpy(),
        "day": day.astype(np.int64).astype(np.int32),
        "hhmm": pd.to_numeric(raw["acq_time"], errors="coerce").fillna(0).astype("int16").to_numpy(),
        "sensor": _sensor_codes(raw, family, sensor),
        "conf": conf.astype(np.int8),
        "frp": pd.to_numeric(raw["frp"], errors="coerce").fillna(0).astype("float32").to_numpy(),
        "is_day": (raw["daynight"].astype(str).str.upper().str[0] == "D").to_numpy(),
        "type": (pd.to_numeric(raw["type"], errors="coerce").fillna(-1).astype("int8").to_numpy()
                 if "type" in raw.columns else np.full(len(raw), -1, dtype=np.int8)),
        "scan": pd.to_numeric(raw["scan"], errors="coerce").astype("float32").to_numpy(),
        "track": pd.to_numeric(raw["track"], errors="coerce").astype("float32").to_numpy(),
        "bt": pd.to_numeric(bt, errors="coerce").astype("float32").to_numpy(),
    })
    return out[COLUMNS]


def _read_csv(path_or_buf) -> pd.DataFrame:
    try:
        return pd.read_csv(path_or_buf, engine="pyarrow")
    except Exception:  # pyarrow chokes on some edge cases (e.g. empty body): fall back
        if hasattr(path_or_buf, "seek"):
            path_or_buf.seek(0)
        return pd.read_csv(path_or_buf)


# ---------------------------------------------------------------------------------------------
# Yearly country archive (public, standard processing; MODIS C6.1 since 2000, VIIRS since 2012)
# ---------------------------------------------------------------------------------------------

def archive_url(product: str, year: int, country: str) -> str:
    return f"{FIRMS_BASE}/data/country/{product}/{year}/{product}_{year}_{country}.csv"


def latest_archive_year(max_age_days: float = 7.0) -> int:
    """Most recent complete year published in the country archive (probed, cached for a week)."""
    meta = ARCHIVE_DIR / "meta.json"
    if meta.exists():
        info = json.loads(meta.read_text())
        if time.time() - info.get("checked", 0) < max_age_days * 86400:
            return int(info["latest_year"])
    year = dt.date.today().year
    latest = None
    for y in range(year, year - 6, -1):
        r = session().head(archive_url("modis", y, "United_States"), timeout=30)
        if r.status_code == 200:
            latest = y
            break
    if latest is None:
        raise RuntimeError("could not determine the latest FIRMS archive year")
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps({"latest_year": latest, "checked": time.time()}))
    return latest


def _archive_path(product: str, year: int, country: str) -> Path:
    return ARCHIVE_DIR / product / str(year) / f"{country}.parquet"


def fetch_archive_file(product: str, year: int, country: str) -> Path:
    """Download one country-year CSV and store it as Parquet (an empty file marks 'no detections')."""
    path = _archive_path(product, year, country)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = path.with_suffix(".csv")
    family = "MODIS" if product == "modis" else "VIIRS"
    sensor = ARCHIVE_PRODUCTS[product][1][0]
    try:
        download(archive_url(product, year, country), csv_path)
        df = normalize(_read_csv(csv_path), family, sensor)
    except NotFound:
        df = empty_points()
    finally:
        csv_path.unlink(missing_ok=True)
    df.to_parquet(path, compression="zstd", index=False)
    return path


def archive_jobs(countries: Iterable[str], products: Iterable[str] = tuple(ARCHIVE_PRODUCTS),
                 last_year: int | None = None) -> list[tuple[str, int, str]]:
    last_year = last_year or latest_archive_year()
    jobs = []
    for product in products:
        first_year = ARCHIVE_PRODUCTS[product][0]
        for year in range(first_year, last_year + 1):
            for country in countries:
                jobs.append((product, year, country))
    return jobs


def ensure_archive(countries: list[str], progress: Progress | None = None,
                   products: Iterable[str] = tuple(ARCHIVE_PRODUCTS), last_year: int | None = None) -> list[Path]:
    """Download (in parallel) every missing country-year file. Returns all parquet paths."""
    jobs = archive_jobs(countries, products, last_year)
    todo = [j for j in jobs if not _archive_path(*j).exists()]
    paths = [_archive_path(*j) for j in jobs]
    if not todo:
        return paths
    done = 0
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
        futs = {ex.submit(fetch_archive_file, *j): j for j in todo}
        for fut in as_completed(futs):
            j = futs[fut]
            done += 1
            try:
                fut.result()
            except Exception as e:  # keep going; report at the end
                errors.append(f"{j}: {e}")
                log.warning("archive download failed %s: %s", j, e)
            if progress:
                progress(done / len(todo), f"FIRMS archive {j[0]} {j[1]} {j[2]} ({done}/{len(todo)})")
    if errors:
        raise RuntimeError(f"{len(errors)} FIRMS archive downloads failed, e.g. {errors[0]}")
    return paths


def load_archive(countries: list[str], bounds: tuple[float, float, float, float] | None,
                 products: Iterable[str] = tuple(ARCHIVE_PRODUCTS), last_year: int | None = None,
                 progress: Progress | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Points within ``bounds`` plus country-level daily counts per sensor (for outage detection)."""
    paths = ensure_archive(countries, progress, products, last_year)
    parts, daily = [], []
    for p in paths:
        df = pd.read_parquet(p)
        if df.empty:
            continue
        daily.append(df.groupby(["day", "sensor"]).size().rename("n").reset_index())
        if bounds is not None:
            w, s, e, n = bounds
            df = df[(df.lon >= w) & (df.lon <= e) & (df.lat >= s) & (df.lat <= n)]
        parts.append(df)
    points = pd.concat(parts, ignore_index=True) if parts else empty_points()
    counts = (pd.concat(daily).groupby(["day", "sensor"], as_index=False)["n"].sum() if daily
              else pd.DataFrame({"day": [], "sensor": [], "n": []}))
    return points, counts


# ---------------------------------------------------------------------------------------------
# Global near-real-time feeds (keyless, last 7 days)
# ---------------------------------------------------------------------------------------------

def fetch_nrt(feed: str, max_age_hours: float = NRT_TTL_HOURS) -> pd.DataFrame:
    path = NRT_DIR / f"{feed}.parquet"
    if path.exists() and time.time() - path.stat().st_mtime < max_age_hours * 3600:
        return pd.read_parquet(path)
    family = "MODIS" if feed == "MODIS" else "VIIRS"
    csv_path = NRT_DIR / f"{feed}.csv"
    download(NRT_FEEDS[feed], csv_path)
    df = normalize(_read_csv(csv_path), family, None if family == "MODIS" else feed)
    csv_path.unlink(missing_ok=True)
    df.to_parquet(path, compression="zstd", index=False)
    return df


def load_nrt(bounds: tuple[float, float, float, float], feeds: Iterable[str] = tuple(NRT_FEEDS),
             progress: Progress | None = None) -> pd.DataFrame:
    w, s, e, n = bounds
    parts = []
    feeds = list(feeds)
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(fetch_nrt, f): f for f in feeds}
        for i, fut in enumerate(as_completed(futs)):
            f = futs[fut]
            try:
                df = fut.result()
            except Exception as err:
                log.warning("NRT feed %s failed: %s", f, err)
                continue
            parts.append(df[(df.lon >= w) & (df.lon <= e) & (df.lat >= s) & (df.lat <= n)])
            if progress:
                progress((i + 1) / len(feeds), f"FIRMS near-real-time feed {f}")
    return pd.concat(parts, ignore_index=True) if parts else empty_points()


# ---------------------------------------------------------------------------------------------
# FIRMS area API (requires a free MAP_KEY) - closes the gap between the archive and the NRT week
# ---------------------------------------------------------------------------------------------

def api_availability(map_key: str) -> pd.DataFrame:
    text = get_text(f"{FIRMS_BASE}/api/data_availability/csv/{map_key}/all")
    df = pd.read_csv(io.StringIO(text))
    return df.set_index("data_id")


def _api_family(source: str) -> tuple[str, str | None]:
    if source.startswith("MODIS"):
        return "MODIS", None
    for key, s in SENSORS.items():
        if source in (s.api_sp, s.api_nrt):
            return "VIIRS", key
    raise ValueError(source)


def fetch_api_range(source: str, bounds: tuple[float, float, float, float], start: dt.date, end: dt.date,
                    map_key: str | None = None, progress: Progress | None = None) -> pd.DataFrame:
    """Query the FIRMS area API in 5-day chunks (cached; chunks older than 10 days are final)."""
    map_key = map_key or firms_map_key()
    if not map_key:
        raise RuntimeError("FIRMS_MAP_KEY not set")
    family, sensor = _api_family(source)
    w, s, e, n = bounds
    area = f"{w:.4f},{s:.4f},{e:.4f},{n:.4f}"
    cache = API_DIR / source / area.replace(",", "_")
    cache.mkdir(parents=True, exist_ok=True)
    parts = []
    chunks = []
    d = start
    while d <= end:
        chunks.append(d)
        d += dt.timedelta(days=5)
    today = dt.date.today()
    for i, d0 in enumerate(chunks):
        days = min(5, (end - d0).days + 1)
        path = cache / f"{d0.isoformat()}_{days}.parquet"
        final = (today - (d0 + dt.timedelta(days=days))).days > 10
        if path.exists() and (final or time.time() - path.stat().st_mtime < NRT_TTL_HOURS * 3600):
            parts.append(pd.read_parquet(path))
            continue
        url = f"{FIRMS_BASE}/api/area/csv/{map_key}/{source}/{area}/{days}/{d0.isoformat()}"
        text = get_text(url, timeout=180)
        if text.lstrip().lower().startswith(("invalid", "error")):
            raise RuntimeError(f"FIRMS API error for {source}: {text[:200]}")
        raw = pd.read_csv(io.StringIO(text)) if text.strip() else pd.DataFrame()
        df = normalize(raw, family, sensor) if len(raw) else empty_points()
        df.to_parquet(path, compression="zstd", index=False)
        parts.append(df)
        if progress:
            progress((i + 1) / len(chunks), f"FIRMS API {source} {d0}")
        time.sleep(0.3)  # stay well below the 5000 requests / 10 min limit
    return pd.concat(parts, ignore_index=True) if parts else empty_points()


def load_recent_via_api(bounds: tuple[float, float, float, float], start: dt.date, end: dt.date,
                        progress: Progress | None = None) -> pd.DataFrame:
    """Fill [start, end] for every sensor, preferring standard-processing over NRT data."""
    key = firms_map_key()
    avail = api_availability(key)
    parts = []
    for skey, s in SENSORS.items():
        if skey == "MODIS_A":  # MODIS sources return Terra and Aqua together
            continue
        covered_until = start - dt.timedelta(days=1)
        for source in (s.api_sp, s.api_nrt):
            if not source or source not in avail.index:
                continue
            lo = max(start, covered_until + dt.timedelta(days=1),
                     dt.date.fromisoformat(str(avail.loc[source, "min_date"])))
            hi = min(end, dt.date.fromisoformat(str(avail.loc[source, "max_date"])))
            if lo > hi:
                continue
            parts.append(fetch_api_range(source, bounds, lo, hi, key, progress))
            covered_until = hi
    return pd.concat(parts, ignore_index=True) if parts else empty_points()
