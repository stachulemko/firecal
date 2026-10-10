"""The whole world as areas of interest: every country, plus the regions of the nine largest ones.

Countries come from Natural Earth admin-0 (only those with a FIRMS archive), regions from Natural
Earth admin-1, which covers exactly the countries too large to analyse in one piece. Small islands
far from the main territory are left out of a country's area (they would only stretch the grid over
the ocean) and areas crossing the antimeridian keep their larger side.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.prepared import prep

from .config import GRID_RESOLUTIONS, MAX_GRID_CELLS, STATIC_DIR
from .geo import AOI, country_features
from .sources.http import download

NE_ADMIN1_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
                 "ne_50m_admin_1_states_provinces.geojson")
REGION_COUNTRY_NAMES = {"United States of America": "United States", "Russia": "Russia", "China": "China",
                        "India": "India", "Indonesia": "Indonesia", "Brazil": "Brazil", "Canada": "Canada",
                        "Australia": "Australia", "South Africa": "South Africa"}


@dataclass(frozen=True)
class Area:
    key: str               # "c:<Natural Earth ADMIN>" or "r:<adm1_code>"
    name: str
    kind: str              # "country" | "region"
    group: str             # continent, or "<country>: regions"
    geometry: BaseGeometry

    @property
    def bounds(self) -> list[float]:
        return [round(float(v), 3) for v in self.geometry.bounds]

    @property
    def fits(self) -> bool:
        """Small enough for one analysis grid: at most MAX_GRID_CELLS cells at the coarsest resolution."""
        w, s, e, n = self.geometry.bounds
        res = GRID_RESOLUTIONS[-1]
        nx = math.ceil(e / res) - math.floor(w / res)
        ny = math.ceil(n / res) - math.floor(s / res)
        return e - w <= 180 and nx * ny <= MAX_GRID_CELLS

    def aoi(self) -> AOI:
        return AOI(self.name, self.geometry)


def _parts(geom: BaseGeometry) -> list[BaseGeometry]:
    return list(geom.geoms) if hasattr(geom, "geoms") else [geom]


def core(geom: BaseGeometry, min_share: float = 0.05, max_gap_deg: float = 5.0) -> BaseGeometry:
    """Drop small parts far from the main territory (Azores, Galapagos, Easter Island...) and keep one
    side of the antimeridian (Chukotka, Fiji...)."""
    parts = _parts(geom)
    if len(parts) > 1:
        total = sum(p.area for p in parts)
        main = max(parts, key=lambda p: p.area)
        parts = [p for p in parts if p.area >= min_share * total or p.distance(main) <= max_gap_deg]
    w, _, e, _ = unary_union(parts).bounds
    if e - w > 180:
        east = [p for p in parts if p.centroid.x >= 0]
        west = [p for p in parts if p.centroid.x < 0]
        parts = max((east, west), key=lambda ps: sum(p.area for p in ps))
    return unary_union(parts)


@lru_cache(maxsize=1)
def areas() -> tuple[Area, ...]:
    out = [Area(f"c:{c.admin}", c.name, "country", c.continent or "Other", core(c.geometry))
           for c in country_features() if c.continent != "Antarctica"]
    path = STATIC_DIR / "ne_50m_admin_1_states_provinces.geojson"
    if not path.exists():
        download(NE_ADMIN1_URL, path)
    from shapely.geometry import shape
    for f in json.loads(path.read_text())["features"]:
        p = f["properties"]
        country = REGION_COUNTRY_NAMES.get(p.get("admin"))
        if country is None or not p.get("adm1_code"):
            continue
        out.append(Area(f"r:{p['adm1_code']}", f"{country}: {p['name']}", "region", f"{country}: regions",
                        core(shape(f["geometry"]))))
    return tuple(out)


@lru_cache(maxsize=1)
def _index() -> dict[str, Area]:
    return {a.key: a for a in areas()}


def area(key: str) -> Area:
    a = _index().get(key)
    if a is None:
        raise KeyError(f"unknown area {key}")
    return a


@lru_cache(maxsize=1)
def _prepared() -> list[tuple[Area, object]]:
    # regions first: a click in Siberia should pick the region, not all of Russia
    ordered = sorted(areas(), key=lambda a: a.kind != "region")
    return [(a, prep(a.geometry)) for a in ordered]


def area_at(lat: float, lon: float) -> Area | None:
    pt = Point(lon, lat)
    hit = next((a for a, pg in _prepared() if pg.contains(pt)), None)
    if hit is None:  # coastline generalization: take the nearest area within ~25 km
        near = [(a.geometry.distance(pt), a) for a, _ in _prepared()]
        d, a = min(near, key=lambda t: t[0])
        hit = a if d < 0.25 else None
    return hit


def listing() -> list[dict]:
    return [{"key": a.key, "name": a.name, "kind": a.kind, "group": a.group, "bounds": a.bounds, "fits": a.fits}
            for a in areas()]


# ---------------------------------------------------------------------------------------------
# fires on Earth in the last 7 days (keyless near-real-time VIIRS feeds)
# ---------------------------------------------------------------------------------------------

_recent_cache: dict = {}


def recent_world(res: float = 1.0) -> dict:
    """VIIRS detections of the last 7 days on a global grid: the mean over the VIIRS satellites that
    reported (S-NPP, NOAA-20, NOAA-21), so the unit matches one S-NPP-equivalent sensor. Industrial
    heat sources are not masked here (the near-real-time feeds have no type column)."""
    from . import harmonize as hz
    from .sources import firms

    if _recent_cache.get("t", 0) > time.time() - 1800 and _recent_cache.get("res") == res:
        return _recent_cache["data"]
    counts, first, last = [], None, None
    for feed in ("VIIRS_SNPP", "VIIRS_NOAA20", "VIIRS_NOAA21"):
        try:
            g = hz.qa_filter(firms.fetch_nrt(feed))
        except Exception:  # one feed down: use the others
            continue
        if g.empty:
            continue
        iy = np.floor((g["lat"].to_numpy() + 90) / res).astype(np.int64)
        ix = np.floor((g["lon"].to_numpy() + 180) / res).astype(np.int64)
        counts.append(pd.Series(1, index=iy * 100000 + ix).groupby(level=0).sum())
        first = int(g["day"].min()) if first is None else min(first, int(g["day"].min()))
        last = int(g["day"].max()) if last is None else max(last, int(g["day"].max()))
    if not counts:
        raise RuntimeError("no near-real-time feed available")
    n = pd.concat(counts, axis=1).fillna(0).mean(axis=1)
    n = n[n >= 0.5]
    iy, ix = n.index.to_numpy() // 100000, n.index.to_numpy() % 100000
    data = {
        "res": res, "lat": np.round(-90 + (iy + 0.5) * res, 3).tolist(), "lon": np.round(-180 + (ix + 0.5) * res, 3).tolist(),
        "n": np.round(n.to_numpy(), 1).tolist(), "sensors": len(counts),
        "first": str(hz.to_dates(np.array([first]))[0]), "last": str(hz.to_dates(np.array([last]))[0]),
        "total": float(n.sum()),
    }
    _recent_cache.update(t=time.time(), res=res, data=data)
    return data
