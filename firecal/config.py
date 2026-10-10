"""Global configuration: paths, sensor registry and modelling constants."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines); real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.environ.get("FIRECAL_DATA", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
STATIC_DIR = DATA_DIR / "static"
AOI_DIR = DATA_DIR / "aoi"

USER_AGENT = "firecal/0.1 (NASA Space Apps burning activity calendar)"
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov"
DOWNLOAD_WORKERS = int(os.environ.get("FIRECAL_DOWNLOAD_WORKERS", "6"))


def firms_map_key() -> str:
    """FIRMS MAP_KEY (free, https://firms.modaps.eosdis.nasa.gov/api/map_key/). Optional."""
    return os.environ.get("FIRMS_MAP_KEY", "").strip()


@dataclass(frozen=True)
class Sensor:
    key: str
    platform: str
    instrument: str
    pixel_m: int
    first_date: str
    family: str
    nrt_feed: str | None
    api_sp: str | None
    api_nrt: str | None
    label: str


SENSORS: dict[str, Sensor] = {
    s.key: s
    for s in [
        Sensor("MODIS_T", "Terra", "MODIS", 1000, "2000-11-01", "MODIS", "MODIS", "MODIS_SP", "MODIS_NRT",
               "MODIS Terra (1 km)"),
        Sensor("MODIS_A", "Aqua", "MODIS", 1000, "2002-07-04", "MODIS", "MODIS", "MODIS_SP", "MODIS_NRT",
               "MODIS Aqua (1 km)"),
        Sensor("VIIRS_SNPP", "Suomi NPP", "VIIRS", 375, "2012-01-20", "VIIRS", "VIIRS_SNPP", "VIIRS_SNPP_SP",
               "VIIRS_SNPP_NRT", "VIIRS S-NPP (375 m)"),
        Sensor("VIIRS_NOAA20", "NOAA-20", "VIIRS", 375, "2018-04-01", "VIIRS", "VIIRS_NOAA20", "VIIRS_NOAA20_SP",
               "VIIRS_NOAA20_NRT", "VIIRS NOAA-20 (375 m)"),
        Sensor("VIIRS_NOAA21", "NOAA-21", "VIIRS", 375, "2024-01-17", "VIIRS", "VIIRS_NOAA21", None,
               "VIIRS_NOAA21_NRT", "VIIRS NOAA-21 (375 m)"),
    ]
}
SENSOR_ORDER = list(SENSORS)
SENSOR_CODE = {k: i for i, k in enumerate(SENSOR_ORDER)}  # compact int8 codes for storage

# FIRMS yearly per-country archive (public, no key): folder -> (first year, sensors contained)
ARCHIVE_PRODUCTS: dict[str, tuple[int, tuple[str, ...]]] = {
    "modis": (2000, ("MODIS_T", "MODIS_A")),
    "viirs-snpp": (2012, ("VIIRS_SNPP",)),
    "viirs-jpss1": (2018, ("VIIRS_NOAA20",)),
}

# Keyless global near-real-time feeds (last 7 days)
NRT_FEEDS: dict[str, str] = {
    "MODIS": f"{FIRMS_BASE}/data/active_fire/modis-c6.1/csv/MODIS_C6_1_Global_7d.csv",
    "VIIRS_SNPP": f"{FIRMS_BASE}/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_Global_7d.csv",
    "VIIRS_NOAA20": f"{FIRMS_BASE}/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_Global_7d.csv",
    "VIIRS_NOAA21": f"{FIRMS_BASE}/data/active_fire/noaa-21-viirs-c2/csv/J2_VIIRS_C2_Global_7d.csv",
}
NRT_TTL_HOURS = 3

# ---- harmonization ---------------------------------------------------------------------------
REFERENCE_SENSOR = "VIIRS_SNPP"  # harmonized unit = S-NPP-equivalent hotspot detections
MODIS_CONF_NOMINAL = 30  # MODIS C6.1 confidence classes: <30 low, 30-79 nominal, >=80 high
MODIS_CONF_HIGH = 80
GRID_RESOLUTIONS = (0.1, 0.25, 0.5, 1.0)
MAX_GRID_CELLS = 1600
# Terra/Aqua orbits drift after ~2020, so the MODIS->VIIRS transfer is fitted on the stable-orbit overlap
TRANSFER_TRAIN_YEARS = tuple(range(2012, 2020))
TRANSFER_VALID_YEAR = 2020

# ---- forecasting -----------------------------------------------------------------------------
FORECAST_LEADS = (1, 2, 3, 5, 7, 10, 14)  # days ahead used for training the lead-aware model
MAX_LEAD = 14
WEATHER_POINTS_MAX = 12


for _d in (RAW_DIR, STATIC_DIR, AOI_DIR):
    _d.mkdir(parents=True, exist_ok=True)
