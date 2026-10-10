"""Area-of-interest handling, country lookup for FIRMS archives and the analysis grid."""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np
from shapely import contains_xy
from shapely.geometry import box, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .config import GRID_RESOLUTIONS, MAX_GRID_CELLS, STATIC_DIR
from .sources.http import download

NE_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
          "ne_50m_admin_0_countries.geojson")

# Natural Earth ADMIN name -> FIRMS archive file name, where automatic matching fails
_ALIASES = {
    "Ivory Coast": "Cote_d_Ivoire",
    "North Macedonia": "Macedonia_Former_Yugoslav_Republic_of",
    "Republic of the Congo": "Republic_of_Congo",
    "eSwatini": "Swaziland",
    "Somaliland": "Somalia",
    "Northern Cyprus": "Cyprus",
    "Western Sahara": "Morocco",
    "Aland": "Aland_Islands",
    "Siachen Glacier": "India",
    "South Georgia and the Islands": "South_Georgia_and_South_Sandwich_Islands",
    "Saint Barthelemy": "Saint-Barthelemy",
    "Wallis and Futuna": "Wallis_and_Futuna_Islands",
    "Federated States of Micronesia": "Federated_States_of_Micronesia",
}
# French overseas departments are separate files in FIRMS but part of France in Natural Earth
_OVERSEAS = {
    "French_Guiana": (-54.7, 2.0, -51.5, 5.9),
    "Guadeloupe": (-61.9, 15.8, -60.9, 16.6),
    "Martinique": (-61.3, 14.35, -60.75, 14.95),
    "Mayotte": (44.9, -13.1, 45.4, -12.5),
    "Reunion": (55.1, -21.5, 55.9, -20.8),
}
_OVERSEAS_CONTINENT = {"French_Guiana": "South America", "Guadeloupe": "North America", "Martinique": "North America",
                       "Mayotte": "Africa", "Reunion": "Africa"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


@lru_cache(maxsize=1)
def firms_country_names() -> tuple[str, ...]:
    text = resources.files("firecal.resources").joinpath("firms_countries.txt").read_text()
    return tuple(n for n in text.split() if n)


@dataclass(frozen=True)
class Country:
    firms: str      # FIRMS archive file name
    admin: str      # Natural Earth ADMIN (unique key)
    name: str       # display name
    continent: str
    geometry: BaseGeometry


@lru_cache(maxsize=1)
def country_features() -> list[Country]:
    """Every country or territory with a FIRMS archive, from Natural Earth 1:50m admin-0 boundaries."""
    path = STATIC_DIR / "ne_50m_admin_0_countries.geojson"
    if not path.exists():
        download(NE_URL, path)
    gj = json.loads(path.read_text())
    firms = {_norm(n): n for n in firms_country_names()}
    out: list[Country] = []
    for feat in gj["features"]:
        p = feat["properties"]
        geom = shape(feat["geometry"])
        name = _ALIASES.get(p.get("ADMIN", ""))
        if name is None:
            for k in ("NAME", "NAME_LONG", "ADMIN", "GEOUNIT", "SUBUNIT", "FORMAL_EN", "NAME_EN", "BRK_NAME"):
                if p.get(k) and _norm(p[k]) in firms:
                    name = firms[_norm(p[k])]
                    break
        if name is None:
            continue
        if name == "France" and geom.geom_type == "MultiPolygon":
            overseas = [box(*b) for b in _OVERSEAS.values()]
            parts = [g for g in geom.geoms if not any(o.contains(g.centroid) for o in overseas)]
            geom = unary_union(parts)
        out.append(Country(name, p["ADMIN"], p.get("NAME_LONG") or p.get("NAME") or p["ADMIN"],
                           p.get("CONTINENT", ""), geom))
    out.extend(Country(n, n, n.replace("_", " "), _OVERSEAS_CONTINENT[n], box(*b)) for n, b in _OVERSEAS.items())
    return out


@lru_cache(maxsize=1)
def country_shapes() -> list[tuple[str, BaseGeometry]]:
    """[(FIRMS name, geometry)] built from Natural Earth 1:50m admin-0 boundaries."""
    return [(c.firms, c.geometry) for c in country_features()]


def countries_for(geom: BaseGeometry, buffer_deg: float = 0.25) -> list[str]:
    """FIRMS country files whose territory intersects the (buffered) AOI."""
    g = geom.buffer(buffer_deg)
    return sorted({name for name, cg in country_shapes() if cg.intersects(g)})


def select_countries(aoi_geom: BaseGeometry, grid_bounds: tuple[float, float, float, float],
                     shapes: list[tuple[str, BaseGeometry]]) -> tuple[list[str], list[str]]:
    """(countries to download, skipped neighbours) for an analysis.

    Every country with territory inside the area is needed. Neighbours only fill the grid around it
    (context for the models), so they are taken when they cover a useful part of the grid, unless the
    country is huge and barely touches it: Russia's Kaliningrad exclave next to Poland would mean
    downloading gigabytes of Siberian fires for a few context cells.
    """
    gbox = box(*grid_bounds)
    g_area, a_area = gbox.area, max(aoi_geom.area, 1e-12)
    need, skipped = set(), set()
    for name, cg in shapes:
        if not cg.intersects(gbox):
            continue
        if cg.intersection(aoi_geom).area > 1e-4 * a_area:
            need.add(name)
            continue
        in_grid = cg.intersection(gbox).area
        if in_grid < 0.005 * g_area:
            continue
        if cg.area > 5 * g_area and in_grid < 0.05 * cg.area:
            skipped.add(name)
            continue
        need.add(name)
    return sorted(need), sorted(skipped - need)


def _deg(v: float, pos: str, neg: str) -> str:
    return f"{abs(v):.1f}°{pos if v >= 0 else neg}"


def bbox_label(west: float, south: float, east: float, north: float) -> str:
    return f"{_deg(south, 'N', 'S')}–{_deg(north, 'N', 'S')}, {_deg(west, 'E', 'W')}–{_deg(east, 'E', 'W')}"


@dataclass
class AOI:
    name: str
    geometry: BaseGeometry
    named: bool = True  # False: the name is a placeholder; the pipeline prefixes the countries

    @classmethod
    def from_bbox(cls, west: float, south: float, east: float, north: float, name: str | None = None) -> "AOI":
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError("bbox must be west<east, south<north within valid lon/lat ranges")
        return cls(name or bbox_label(west, south, east, north), box(west, south, east, north), named=bool(name))

    @classmethod
    def from_geojson(cls, gj: dict, name: str | None = None) -> "AOI":
        if gj.get("type") == "FeatureCollection":
            geom = unary_union([shape(f["geometry"]) for f in gj["features"]])
        elif gj.get("type") == "Feature":
            geom = shape(gj["geometry"])
        else:
            geom = shape(gj)
        if geom.is_empty or geom.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError("AOI must be a (multi)polygon")
        return cls(name or "custom area", geom.buffer(0), named=bool(name))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return tuple(float(v) for v in self.geometry.bounds)

    @property
    def id(self) -> str:
        # Stable id from the rounded geometry, so equal areas share the cache
        rounded = json.dumps(mapping(self.geometry), sort_keys=True, default=lambda v: round(v, 4))
        return hashlib.sha1(re.sub(r"(\d+\.\d{4})\d+", r"\1", rounded).encode()).hexdigest()[:12]

    def to_geojson(self) -> dict:
        return {"type": "Feature", "properties": {"name": self.name}, "geometry": mapping(self.geometry)}


@dataclass(frozen=True)
class Grid:
    """Regular lat/lon grid snapped to multiples of ``res`` covering the AOI bounds."""
    res: float
    lat0: float  # southern edge
    lon0: float  # western edge
    ny: int
    nx: int

    @classmethod
    def for_aoi(cls, aoi: AOI, res: float | None = None) -> "Grid":
        w, s, e, n = aoi.bounds
        if res is None:
            res = GRID_RESOLUTIONS[-1]
            for r in GRID_RESOLUTIONS:
                if math.ceil((e - w) / r) * math.ceil((n - s) / r) <= MAX_GRID_CELLS:
                    res = r
                    break
        lon0 = math.floor(w / res) * res
        lat0 = math.floor(s / res) * res
        nx = max(1, math.ceil(round((e - lon0) / res, 6)))
        ny = max(1, math.ceil(round((n - lat0) / res, 6)))
        return cls(float(res), float(round(lat0, 6)), float(round(lon0, 6)), int(ny), int(nx))

    @property
    def ncells(self) -> int:
        return self.ny * self.nx

    def cell_of(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Cell index for each point, -1 outside the grid."""
        iy = np.floor((np.asarray(lat, dtype=np.float64) - self.lat0) / self.res).astype(np.int64)
        ix = np.floor((np.asarray(lon, dtype=np.float64) - self.lon0) / self.res).astype(np.int64)
        ok = (iy >= 0) & (iy < self.ny) & (ix >= 0) & (ix < self.nx)
        return np.where(ok, iy * self.nx + ix, -1)

    def centers(self) -> tuple[np.ndarray, np.ndarray]:
        """(lat, lon) of every cell centre, indexed by cell id."""
        iy, ix = np.divmod(np.arange(self.ncells), self.nx)
        return self.lat0 + (iy + 0.5) * self.res, self.lon0 + (ix + 0.5) * self.res

    def cell_mask(self, geom: BaseGeometry) -> np.ndarray:
        """Cells whose centre lies inside the AOI (cells touching a thin AOI are kept too)."""
        lat, lon = self.centers()
        inside = contains_xy(geom, lon, lat)
        if not inside.any():
            inside = np.array([geom.intersects(self.cell_box(c)) for c in range(self.ncells)])
        return inside

    def cell_box(self, c: int) -> BaseGeometry:
        iy, ix = divmod(int(c), self.nx)
        return box(self.lon0 + ix * self.res, self.lat0 + iy * self.res,
                   self.lon0 + (ix + 1) * self.res, self.lat0 + (iy + 1) * self.res)

    def neighbors(self, radius: int = 1) -> np.ndarray:
        """(ncells, k) neighbour cell ids within a (2r+1)^2 window, -1 where outside the grid."""
        iy, ix = np.divmod(np.arange(self.ncells), self.nx)
        cols = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                y, x = iy + dy, ix + dx
                ok = (y >= 0) & (y < self.ny) & (x >= 0) & (x < self.nx)
                cols.append(np.where(ok, y * self.nx + x, -1))
        return np.stack(cols, axis=1)

    def to_dict(self) -> dict:
        return {"res": self.res, "lat0": self.lat0, "lon0": self.lon0, "ny": self.ny, "nx": self.nx}
