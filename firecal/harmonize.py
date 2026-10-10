"""Cross-sensor harmonization of active fire hotspots into one consistent daily record.

The record is expressed in **VIIRS S-NPP-equivalent hotspot detections** (375 m), the sensor that
will continue after MODIS. Steps:

1. QA: drop low-confidence detections and non-vegetation sources (volcanoes, static industrial
   sources, offshore flares). NRT data has no ``type`` column, so static sources are masked with
   locations that the archive flags as static.
2. Grid every sensor onto a common lat/lon grid as daily cell counts.
3. Infer sensor outages (a sensor reporting nothing while the others see many fires) so that
   "no data" is never mistaken for "no fire".
4. Learn a MODIS -> S-NPP transfer model (LightGBM, Poisson) on the 2012-2019 overlap at
   cell-day level. It uses Terra/Aqua day/night counts, FRP, pixel footprint, temporal and spatial
   context and the cell's fire regime, so it can also estimate the small fires that only VIIRS
   resolves. It is validated on later, unseen years against simpler harmonization baselines.
5. Assemble the record: observed S-NPP where available, NOAA-20/21 rescaled to S-NPP during S-NPP
   gaps, and the transfer model for the MODIS-only era (2000-2012) and remaining gaps.
6. For reconstructed days, also give the most likely daily count (``QuantileMap``): the expected
   value is right for totals but smoother than any observed day.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import calendar as cal
from .config import SENSOR_CODE, SENSORS, TRANSFER_TRAIN_YEARS, TRANSFER_VALID_YEAR
from .geo import Grid

log = logging.getLogger(__name__)

EPOCH = np.datetime64("1970-01-01", "D")
DAY0 = int((np.datetime64("2000-11-01", "D") - EPOCH).astype(int))  # first MODIS Terra day in FIRMS

T, A, V, J1, J2 = (SENSOR_CODE[k] for k in ("MODIS_T", "MODIS_A", "VIIRS_SNPP", "VIIRS_NOAA20", "VIIRS_NOAA21"))
SOURCE_LABELS = {
    0: "no data",
    1: "VIIRS S-NPP (observed)",
    2: "VIIRS NOAA-20 rescaled to S-NPP",
    3: "VIIRS NOAA-21 rescaled to S-NPP",
    4: "MODIS Terra+Aqua -> S-NPP (transfer model)",
    5: "MODIS Terra -> S-NPP (transfer model)",
}


def day_number(date) -> int:
    return int((np.datetime64(str(date), "D") - EPOCH).astype(int))


def to_dates(days: np.ndarray) -> np.ndarray:
    return (np.asarray(days, dtype=np.int64) + EPOCH.astype(np.int64)).astype("datetime64[D]")


def years_of(days: np.ndarray) -> np.ndarray:
    return to_dates(days).astype("datetime64[Y]").astype(int) + 1970


# ---------------------------------------------------------------------------------------------
# 1. quality control
# ---------------------------------------------------------------------------------------------

_KEY_RES = 0.004  # ~400 m: static-source matching tolerance (VIIRS 375 m pixels jitter)


def _loc_keys(lat: np.ndarray, lon: np.ndarray, dy: int = 0, dx: int = 0) -> np.ndarray:
    ky = np.round(np.asarray(lat, dtype=np.float64) / _KEY_RES).astype(np.int64) + dy + 50_000
    kx = np.round(np.asarray(lon, dtype=np.float64) / _KEY_RES).astype(np.int64) + dx + 100_000
    return ky * 1_000_000 + kx


def static_source_keys(points: pd.DataFrame) -> np.ndarray:
    """Location keys (with a 1-key halo) of detections FIRMS classifies as non-vegetation."""
    p = points[points["type"] > 0]
    if p.empty:
        return np.empty(0, dtype=np.int64)
    keys = [_loc_keys(p.lat.values, p.lon.values, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    return np.unique(np.concatenate(keys))


def qa_filter(points: pd.DataFrame, static_keys: np.ndarray | None = None) -> pd.DataFrame:
    keep = points["conf"].to_numpy() >= 1
    typ = points["type"].to_numpy()
    unknown = typ == -1
    if unknown.any() and static_keys is not None and len(static_keys):
        is_static = np.isin(_loc_keys(points.lat.values, points.lon.values), static_keys)
        keep &= (typ == 0) | (unknown & ~is_static)
    else:
        keep &= (typ == 0) | unknown
    return points[keep]


# ---------------------------------------------------------------------------------------------
# 3. sensor availability
# ---------------------------------------------------------------------------------------------

def sensor_availability(counts: pd.DataFrame, day0: int, ndays: int,
                        extra_coverage: dict[int, np.ndarray] | None = None,
                        min_expected: float = 15.0) -> np.ndarray:
    """(nsensor, ndays) bool: was the sensor delivering data that day?

    ``counts`` are detection counts per (day, sensor) over the whole downloaded countries.
    A sensor is considered down on a day when it reports nothing although

    * a sensor of the same instrument family, rescaled by the long-term ratio between the two,
      implies at least ``min_expected`` detections (cross-family evidence needs 3x more, because
      MODIS legitimately misses the small fires VIIRS resolves), or
    * it reports nothing for 7+ consecutive days while its own seasonal climatology implies at
      least 100 detections over that run (catches outages when no other sensor is flying).

    ``extra_coverage`` marks days covered by NRT/API downloads (sensor code -> days array).
    """
    ns = len(SENSOR_CODE)
    C = np.zeros((ndays, ns))
    if len(counts):
        d = counts["day"].to_numpy().astype(np.int64) - day0
        ok = (d >= 0) & (d < ndays)
        np.add.at(C, (d[ok], counts["sensor"].to_numpy()[ok].astype(int)), counts["n"].to_numpy()[ok])
    avail = np.zeros((ns, ndays), dtype=bool)
    present = C > 0
    family = {SENSOR_CODE[k]: s.family for k, s in SENSORS.items()}
    for key, s in SENSORS.items():
        code = SENSOR_CODE[key]
        if not present[:, code].any():
            continue
        first = max(day_number(s.first_date) - day0, int(np.argmax(present[:, code])))
        last = int(ndays - 1 - np.argmax(present[::-1, code]))
        avail[code, first:last + 1] = True
    lifetime = avail.copy()
    doy = (to_dates(np.arange(day0, day0 + ndays)) - to_dates(np.arange(day0, day0 + ndays)).astype("datetime64[Y]")
           ).astype(int)
    for s in range(ns):
        if not lifetime[s].any():
            continue
        # (a) ratio evidence from the other sensors
        for same_family in (True, False):
            others = [o for o in range(ns) if o != s and lifetime[o].any() and (family[o] == family[s]) == same_family]
            exp = []
            for o in others:
                both = lifetime[s] & lifetime[o] & present[:, s] & present[:, o]
                if both.sum() < 30:
                    continue
                ratio = C[both, s].sum() / max(C[both, o].sum(), 1.0)
                exp.append(np.where(lifetime[o] & present[:, o], ratio * C[:, o], np.nan))
            if not exp:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN days are expected
                expected = np.nanmedian(np.vstack(exp), axis=0)
            thr = min_expected if same_family else 3 * min_expected
            avail[s, lifetime[s] & ~present[:, s] & (np.nan_to_num(expected) >= thr)] = False
        # (b) long silent runs against the sensor's own seasonal climatology
        life = lifetime[s]
        clim = np.zeros(366)
        for k in range(366):
            sel = life & (np.abs(((doy - k + 183) % 366) - 183) <= 15)
            clim[k] = C[sel, s].mean() if sel.any() else 0.0
        silent = life & ~present[:, s]
        start = None
        for t in range(ndays + 1):
            if t < ndays and silent[t]:
                start = t if start is None else start
                continue
            if start is not None and t - start >= 7 and clim[doy[start:t]].sum() >= 100:
                avail[s, start:t] = False
            start = None
    if extra_coverage:
        for code, days in extra_coverage.items():
            d = np.asarray(days, dtype=np.int64) - day0
            d = d[(d >= 0) & (d < ndays)]
            avail[code, d] = True
    return avail


# ---------------------------------------------------------------------------------------------
# 2. gridding
# ---------------------------------------------------------------------------------------------

@dataclass
class CellCube:
    """Dense daily grids for the active cells of an AOI (cells x days per sensor)."""
    grid: Grid
    cells: np.ndarray          # global grid ids of active cells
    in_aoi: np.ndarray         # bool per active cell
    day0: int
    ndays: int
    counts: np.ndarray         # (nsensor, ncell, ndays) float32 detections
    day_counts: np.ndarray     # (nsensor, ncell, ndays) float32 daytime detections
    frp: np.ndarray            # (nsensor, ncell, ndays) float32 summed FRP [MW]
    area: np.ndarray           # (nsensor, ncell, ndays) float32 summed pixel footprint [km2]
    avail: np.ndarray          # (nsensor, ndays) bool
    meta: dict = field(default_factory=dict)

    @property
    def ncell(self) -> int:
        return len(self.cells)

    @property
    def days(self) -> np.ndarray:
        return np.arange(self.day0, self.day0 + self.ndays)

    def local_neighbors(self, radius: int = 1) -> np.ndarray:
        lut = np.full(self.grid.ncells, -1, dtype=np.int64)
        lut[self.cells] = np.arange(self.ncell)
        nb = self.grid.neighbors(radius)[self.cells]
        return np.where(nb >= 0, lut[np.maximum(nb, 0)], -1)


def build_cube(points: pd.DataFrame, grid: Grid, aoi_mask: np.ndarray, day0: int, day1: int,
               avail: np.ndarray) -> CellCube:
    """Aggregate QA-filtered points to (sensor, cell, day) arrays. ``day1`` is inclusive."""
    ndays = day1 - day0 + 1
    cell = grid.cell_of(points.lat.values, points.lon.values)
    d = points["day"].to_numpy().astype(np.int64) - day0
    ok = (cell >= 0) & (d >= 0) & (d < ndays)
    cell, d = cell[ok], d[ok]
    sensor = points["sensor"].to_numpy()[ok].astype(np.int64)
    active = np.unique(np.concatenate([cell, np.flatnonzero(aoi_mask)]))
    lut = np.full(grid.ncells, -1, dtype=np.int64)
    lut[active] = np.arange(len(active))
    ns, nc = len(SENSOR_CODE), len(active)
    flat = (sensor * nc + lut[cell]) * ndays + d
    size = ns * nc * ndays

    def acc(weights=None):
        return np.bincount(flat, weights=weights, minlength=size).astype(np.float32).reshape(ns, nc, ndays)

    sub = points[ok]
    cube = CellCube(
        grid=grid, cells=active, in_aoi=aoi_mask[active], day0=day0, ndays=ndays,
        counts=acc(), day_counts=acc(sub["is_day"].to_numpy().astype(np.float64)),
        frp=acc(sub["frp"].to_numpy().astype(np.float64)),
        area=acc((sub["scan"] * sub["track"]).fillna(1.0).to_numpy().astype(np.float64)),
        avail=avail,
    )
    return cube


# ---------------------------------------------------------------------------------------------
# 4. MODIS -> S-NPP transfer model
# ---------------------------------------------------------------------------------------------

def _window_mean(x: np.ndarray, valid: np.ndarray, w: int) -> np.ndarray:
    """Centered +-w day mean over valid days only (x: cells x days, valid: days)."""
    xv = np.where(valid[None, :], x, 0.0)
    pad = np.pad(xv, ((0, 0), (w + 1, w)))
    cs = np.cumsum(pad, axis=1, dtype=np.float64)
    s = cs[:, 2 * w + 1:] - cs[:, :-2 * w - 1]
    vp = np.cumsum(np.pad(valid.astype(np.float64), (w + 1, w)))
    n = vp[2 * w + 1:] - vp[:-2 * w - 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (s / n[None, :]).astype(np.float32)


def _neighbor_sum(x: np.ndarray, nb: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    for k in range(nb.shape[1]):
        idx = nb[:, k]
        ok = idx >= 0
        out[ok] += x[idx[ok]]
    return out


TRANSFER_FEATURES = [
    "t_day", "t_night", "a_day", "a_night", "t_frp", "a_frp", "m_area", "m_w1", "m_w3", "m_w7",
    "m_nb", "m_nb_w3", "clim_m_cm", "clim_m_c", "lat", "lon", "doy_sin", "doy_cos",
]
# NB: features derived from the *target* sensor (e.g. a cell's VIIRS climatology) were tested and
# rejected - they leak the training years' targets and cost ~25% bias out of sample.


class TransferFeatures:
    """Pre-computes all cell-day transfer features once for a cube.

    Context features (temporal windows, neighbourhood sums) exist in two variants: from Terra+Aqua
    and from Terra alone, so that rows simulating the Terra-only era (Nov 2000 - Jul 2002) see
    exactly what the model will see when backcasting that era.
    """

    def __init__(self, cube: CellCube):
        self.cube = cube
        c = cube.counts
        avT, avA = cube.avail[T], cube.avail[A]
        nb = cube.local_neighbors(1)
        dates = to_dates(cube.days)
        self.month = dates.astype("datetime64[M]").astype(int) % 12
        doy = (dates - dates.astype("datetime64[Y]")).astype(int)
        self.doy_sin = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
        self.doy_cos = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
        self.ctx = {}
        for variant, valid, M in (
            ("both", avT | avA, np.where(avT[None], c[T], 0) + np.where(avA[None], c[A], 0)),
            ("terra", avT, np.where(avT[None], c[T], 0)),
        ):
            Mnb = _neighbor_sum(M, nb)
            self.ctx[variant] = {
                "m_w1": _window_mean(M, valid, 1), "m_w3": _window_mean(M, valid, 3),
                "m_w7": _window_mean(M, valid, 7), "m_nb": np.where(valid[None], Mnb, np.nan).astype(np.float32),
                "m_nb_w3": _window_mean(Mnb, valid, 3),
            }
            if variant == "both":
                self.M, self.m_valid = M.astype(np.float32), valid
        # cell fire regime as seen by MODIS over its whole record (no target information)
        nm = np.zeros((cube.ncell, 12), np.float64)
        dm = np.zeros(12)
        for mth in range(12):
            sel = (self.month == mth) & self.m_valid
            nm[:, mth] = self.M[:, sel].sum(axis=1)
            dm[mth] = sel.sum()
        self.clim_m_cm = (nm / np.maximum(dm, 1)).astype(np.float32)
        self.clim_m_c = (nm.sum(1) / max(dm.sum(), 1)).astype(np.float32)
        lat, lon = cube.grid.centers()
        self.lat = lat[cube.cells].astype(np.float32)
        self.lon = lon[cube.cells].astype(np.float32)

    def frame(self, d_idx: np.ndarray, drop_aqua: bool = False) -> pd.DataFrame:
        """Features for every active cell on the given day indices (cell-major order)."""
        cube = self.cube
        nc, nd = cube.ncell, len(d_idx)
        c, dc = cube.counts, cube.day_counts

        def take(arr):
            return arr[:, d_idx].reshape(-1)

        def sensor_feat(code, arr, drop=False):
            unavailable = np.repeat(~cube.avail[code, d_idx][None, :], nc, axis=0).reshape(-1)
            if drop:
                unavailable[:] = True
            return np.where(unavailable, np.nan, take(arr)).astype(np.float32)

        ctx = self.ctx["terra" if drop_aqua else "both"]
        area = cube.area[T] * cube.avail[T][None]
        if not drop_aqua:
            area = area + cube.area[A] * cube.avail[A][None]
        f = {
            "t_day": sensor_feat(T, dc[T]), "t_night": sensor_feat(T, c[T] - dc[T]),
            "a_day": sensor_feat(A, dc[A], drop_aqua), "a_night": sensor_feat(A, c[A] - dc[A], drop_aqua),
            "t_frp": sensor_feat(T, cube.frp[T]), "a_frp": sensor_feat(A, cube.frp[A], drop_aqua),
            "m_area": take(area),
            **{k: take(ctx[k]) for k in ("m_w1", "m_w3", "m_w7", "m_nb", "m_nb_w3")},
            "clim_m_cm": self.clim_m_cm[:, self.month[d_idx]].reshape(-1),
            "clim_m_c": np.repeat(self.clim_m_c, nd),
            "lat": np.repeat(self.lat, nd), "lon": np.repeat(self.lon, nd),
            "doy_sin": np.tile(self.doy_sin[d_idx], nc), "doy_cos": np.tile(self.doy_cos[d_idx], nc),
        }
        return pd.DataFrame(f)[TRANSFER_FEATURES]


_LGB_PARAMS = dict(
    objective="poisson", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    # Caps every tree's leaf output at lr * 10 = 0.5 in log space. Without it, Poisson boosting on
    # mostly-zero cell-days produces runaway leaves (tiny hessians) predicting ~1e14 detections.
    max_delta_step=10.0,
    max_bin=255, verbose=-1, num_threads=0, seed=7,
)
TRANSFER_ROUNDS = 250
LEVEL_WINDOW = 15  # +-days over which the ML daily pattern is rescaled to the ratio-based level


def _sample_rows(feats: TransferFeatures, d_idx: np.ndarray, rng: np.random.Generator,
                 quiet_keep: float = 0.2, drop_aqua: bool = False):
    """Training rows; cell-days with no fire signal at all are subsampled (and up-weighted)."""
    X = feats.frame(d_idx, drop_aqua=drop_aqua)
    y = feats.cube.counts[V][:, d_idx].reshape(-1)
    quiet = (y == 0) & (np.nan_to_num(X["m_w7"].to_numpy()) == 0) & (np.nan_to_num(X["m_nb_w3"].to_numpy()) == 0)
    keep = ~quiet | (rng.random(len(y)) < quiet_keep)
    w = np.where(quiet, 1.0 / quiet_keep, 1.0).astype(np.float32)
    return X[keep].reset_index(drop=True), y[keep], w[keep]


def _fit_booster(feats: TransferFeatures, d_tr: np.ndarray, rng: np.random.Generator) -> lgb.Booster:
    X, y, w = _sample_rows(feats, d_tr, rng)
    # extra rows with Aqua masked teach the model the Terra-only era (Nov 2000 - Jul 2002)
    sub = d_tr[(rng.random(len(d_tr)) < 0.15) & feats.cube.avail[A, d_tr]]
    Xa, ya, wa = _sample_rows(feats, sub, rng, drop_aqua=True)
    data = lgb.Dataset(pd.concat([X, Xa], ignore_index=True), np.concatenate([y, ya]),
                       weight=np.concatenate([w, wa]))
    return lgb.train(_LGB_PARAMS, data, num_boost_round=TRANSFER_ROUNDS)


class LevelModel:
    """AOI-level calendar-month S-NPP/MODIS ratios (Terra+Aqua, Terra-only and Aqua-only days)."""

    def __init__(self, cube: CellCube, d_tr: np.ndarray, month: np.ndarray, shrink: float = 20.0):
        aoi = cube.in_aoi
        v = cube.counts[V][aoi][:, d_tr].sum(0)
        t = cube.counts[T][aoi][:, d_tr].sum(0)
        a = cube.counts[A][aoi][:, d_tr].sum(0)
        avT, avA = cube.avail[T, d_tr], cube.avail[A, d_tr]
        mon = month[d_tr]
        self.ratios = {}
        for name, num, den, ok in (("both", v, t + a, avT & avA), ("terra", v, t, avT), ("aqua", v, a, avA)):
            g = num[ok].sum() / max(den[ok].sum(), 1.0)
            r = np.full(12, g)
            for k in range(12):
                sel = ok & (mon == k)
                r[k] = (num[sel].sum() + shrink * g) / (den[sel].sum() + shrink)
            self.ratios[name] = r
        self.cube, self.month = cube, month

    def estimate(self, d_idx: np.ndarray, drop_aqua: bool = False) -> np.ndarray:
        cube, aoi = self.cube, self.cube.in_aoi
        t = cube.counts[T][aoi][:, d_idx].sum(0)
        a = cube.counts[A][aoi][:, d_idx].sum(0)
        avT, avA = cube.avail[T, d_idx], cube.avail[A, d_idx] & (not drop_aqua)
        mon = self.month[d_idx]
        return np.select([avT & avA, avT, avA],
                         [self.ratios["both"][mon] * (t + a), self.ratios["terra"][mon] * t,
                          self.ratios["aqua"][mon] * a], np.nan)


def level_factor(ml: np.ndarray, ratio: np.ndarray, w: int = LEVEL_WINDOW) -> np.ndarray:
    """Per-day factor rescaling the ML estimate to the ratio-based level over +-w days.

    Inputs are full-length daily AOI series with NaN on days that are not modelled; only modelled
    days enter the windows. The ML model gets the daily pattern and the extremes right, the ratio
    method gets multi-week totals right; combined they beat both (see transfer metrics).
    """
    valid = ~np.isnan(ml)
    zm, zr = np.where(valid, ml, 0.0), np.where(valid, ratio, 0.0)
    k = np.ones(2 * w + 1)
    sm = np.convolve(zm, k, mode="same")
    sr = np.convolve(np.nan_to_num(zr), k, mode="same")
    return np.where(valid, (sr + 1.0) / (sm + 1.0), np.nan)


QM_WINDOW = 30  # +-calendar days pooled by the daily quantile mapping
QM_LEVELS = np.linspace(0.005, 0.99, 198)


class QuantileMap:
    """Season-conditional empirical quantile mapping of expected AOI totals onto observed S-NPP days.

    The transfer model estimates the *expected* S-NPP count given what MODIS saw. An expectation is
    smoother than a single day's observation: ~150 cells' small chances of a fire MODIS cannot see
    add up to about one detection on days when VIIRS sees none (2012-2020, Portugal: 6% of
    reconstructed days without fire against 35% observed). Expectations are right for totals -
    months, seasons, 7-day anomalies - but not for single days: the calendar heat map, daily
    percentiles, the chance of a day with fire. This map sends an expected value to the observed
    count at the same quantile for the same time of year (+-``window`` days), fitted on
    out-of-sample predictions: the most likely daily count, with the observed share of quiet days
    and the observed extremes. Mapping is done on log1p values; above the fitted range it keeps
    the top offset (a constant factor), so record days stay records.
    """

    def __init__(self, pred: np.ndarray, obs: np.ndarray, md: np.ndarray, window: int = QM_WINDOW):
        lp, lo = np.log1p(np.asarray(pred, float)), np.log1p(np.asarray(obs, float))
        ok = ~np.isnan(lp) & ~np.isnan(lo)  # e.g. Terra outages in the overlap years
        lp, lo, md = lp[ok], lo[ok], np.asarray(md)[ok]
        if not len(lp):
            raise ValueError("no prediction/observation pairs to fit the quantile map")
        self.xq = np.empty((cal.NMD, len(QM_LEVELS)))
        self.yq = np.empty_like(self.xq)
        for k in range(cal.NMD):
            dist = np.abs(md - k)
            sel = np.minimum(dist, cal.NMD - dist) <= window
            if sel.sum() < 60:  # too few days at this time of year: pool the whole year
                sel = np.ones(len(lp), bool)
            self.xq[k] = np.quantile(lp[sel], QM_LEVELS)
            self.yq[k] = np.quantile(lo[sel], QM_LEVELS)
        # strictly increasing knots (np.interp needs them; ties appear when many values are equal)
        self.xq = np.maximum.accumulate(self.xq + 1e-9 * np.arange(len(QM_LEVELS)), axis=1)

    def apply(self, pred: np.ndarray, md: np.ndarray) -> np.ndarray:
        lp = np.log1p(np.asarray(pred, float))
        out = np.full(len(lp), np.nan)
        for k in np.unique(md):
            i = np.flatnonzero((md == k) & ~np.isnan(lp))
            xq, yq = self.xq[k], self.yq[k]
            x = lp[i]
            out[i] = np.where(x > xq[-1], yq[-1] + (x - xq[-1]), np.interp(x, xq, yq))
        return np.expm1(np.maximum(out, 0.0))


def _distribution_metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    """How closely the daily values are distributed like observed ones (not whether days match)."""
    ok = ~np.isnan(obs) & ~np.isnan(pred)
    o, p = obs[ok], pred[ok]
    lv = np.linspace(0.05, 0.95, 19)
    return {
        "no_fire_days": float(np.mean(p < 0.5)), "no_fire_days_observed": float(np.mean(o < 0.5)),
        "quantile_error_log": float(np.mean(np.abs(np.log1p(np.quantile(o, lv)) - np.log1p(np.quantile(p, lv))))),
    }


@dataclass
class TransferModel:
    booster: lgb.Booster
    features: TransferFeatures
    level: LevelModel
    ratio_noaa20: float
    ratio_noaa21: float
    metrics: dict
    residual_q: dict          # AOI-level log-residual quantiles for uncertainty bands
    validation: pd.DataFrame  # out-of-sample daily AOI series used for the metrics
    qmap: dict                # "both" / "terra" -> QuantileMap (expected -> most likely daily count)

    def predict_days(self, d_idx: np.ndarray, drop_aqua: bool = False) -> np.ndarray:
        X = self.features.frame(d_idx, drop_aqua=drop_aqua)
        return self.booster.predict(X).reshape(self.features.cube.ncell, len(d_idx))


def _agg_metrics(obs: np.ndarray, pred: np.ndarray, days: np.ndarray) -> dict:
    """Agreement of AOI totals at daily / weekly / monthly / yearly aggregation."""
    out = {}
    s = pd.DataFrame({"obs": obs, "pred": pred}, index=pd.DatetimeIndex(to_dates(days))).dropna()
    for name, rule in (("daily", None), ("weekly", "W"), ("monthly", "MS"), ("yearly", "YS")):
        g = s if rule is None else s.resample(rule).sum()
        o, p = g["obs"].to_numpy(), g["pred"].to_numpy()
        if rule is not None:  # drop empty resample bins (gaps between test years)
            keep = s["obs"].resample(rule).count().to_numpy() > 0
            o, p = o[keep], p[keep]
        if len(o) < 3:
            continue
        lo, lp = np.log1p(o), np.log1p(p)
        out[name] = {
            "n": int(len(o)),
            "r_log": float(np.corrcoef(lo, lp)[0, 1]) if lo.std() > 0 and lp.std() > 0 else None,
            "r": float(np.corrcoef(o, p)[0, 1]) if o.std() > 0 and p.std() > 0 else None,
            "bias_pct": float(100 * (p.sum() - o.sum()) / max(o.sum(), 1e-9)),
            "mae_log": float(np.mean(np.abs(lo - lp))),
        }
    if len(s) > 30:
        o, p = s["obs"].to_numpy(), s["pred"].to_numpy()
        top = o >= np.quantile(o, 0.9)
        out["extreme_days_hit"] = float(((p >= np.quantile(p, 0.9)) & top).sum() / max(top.sum(), 1))
    return out


def fit_transfer(cube: CellCube, rng_seed: int = 7, log_fn=None) -> TransferModel:
    """Fit and validate the MODIS -> S-NPP transfer on the stable-orbit sensor overlap."""
    say = log_fn or (lambda m: log.info(m))
    rng = np.random.default_rng(rng_seed)
    yrs = years_of(cube.days)
    both = cube.avail[V] & (cube.avail[T] | cube.avail[A])
    stable = [y for y in (*TRANSFER_TRAIN_YEARS, TRANSFER_VALID_YEAR) if ((yrs == y) & both).sum() > 100]
    if len(stable) < 3:
        raise RuntimeError("not enough MODIS/VIIRS overlap to fit the transfer model")
    feats = TransferFeatures(cube)
    aoi = cube.in_aoi
    nd = cube.ndays
    obs_full = np.where(cube.avail[V], cube.counts[V][aoi].sum(0), np.nan)

    # ---- blocked cross-validation over the stable years (3 interleaved folds) -----------------
    folds = [stable[i::3] for i in range(3)]
    oos = {k: np.full(nd, np.nan) for k in ("ml", "ratio", "ml_t", "ratio_t")}
    say(f"transfer model: 3-fold blocked CV over {stable[0]}-{stable[-1]}")
    for fold in folds:
        d_te = np.flatnonzero(np.isin(yrs, fold) & both)
        d_tr = np.flatnonzero(np.isin(yrs, [y for y in stable if y not in fold]) & both)
        b = _fit_booster(feats, d_tr, rng)
        lvl = LevelModel(cube, d_tr, feats.month)
        oos["ml"][d_te] = b.predict(feats.frame(d_te)).reshape(cube.ncell, -1)[aoi].sum(0)
        oos["ml_t"][d_te] = b.predict(feats.frame(d_te, drop_aqua=True)).reshape(cube.ncell, -1)[aoi].sum(0)
        oos["ratio"][d_te] = lvl.estimate(d_te)
        oos["ratio_t"][d_te] = lvl.estimate(d_te, drop_aqua=True)
    harm = oos["ml"] * level_factor(oos["ml"], oos["ratio"])
    harm_t = oos["ml_t"] * level_factor(oos["ml_t"], oos["ratio_t"])
    d_cv = np.flatnonzero(~np.isnan(oos["ml"]))
    days = cube.days[d_cv]
    raw_m = feats.M[aoi].sum(0)
    metrics: dict = {"stable_years": stable, "folds": folds}
    metrics["cv"] = {
        "harmonized": _agg_metrics(obs_full[d_cv], harm[d_cv], days),
        "ml_only": _agg_metrics(obs_full[d_cv], oos["ml"][d_cv], days),
        "monthly_ratio": _agg_metrics(obs_full[d_cv], oos["ratio"][d_cv], days),
        "raw_modis": _agg_metrics(obs_full[d_cv], raw_m[d_cv], days),
        "harmonized_terra_only": _agg_metrics(obs_full[d_cv], harm_t[d_cv], days),
        "monthly_ratio_terra_only": _agg_metrics(obs_full[d_cv], oos["ratio_t"][d_cv], days),
    }
    # empirical log-residual quantiles by predicted magnitude -> uncertainty bands
    lp, lo = np.log1p(harm[d_cv]), np.log1p(obs_full[d_cv])
    edges = np.quantile(lp, [0.25, 0.5, 0.75])
    bins = np.searchsorted(edges, lp)
    residual_q = {"edges": edges.tolist(), "q10": [], "q90": []}
    for i in range(4):
        r = (lo - lp)[bins == i]
        residual_q["q10"].append(float(np.quantile(r, 0.1)) if len(r) > 10 else -0.7)
        residual_q["q90"].append(float(np.quantile(r, 0.9)) if len(r) > 10 else 0.7)

    # expected value -> most likely daily count, validated leaving one whole year out
    md_cv, yrs_cv, obs_cv = cal.md_index(to_dates(days)), yrs[d_cv], obs_full[d_cv]
    qmap, ml_cv = {}, {}
    for name, pred in (("both", harm[d_cv]), ("terra", harm_t[d_cv])):
        qmap[name] = QuantileMap(pred, obs_cv, md_cv)
        ml_cv[name] = np.empty(len(d_cv))
        for y in np.unique(yrs_cv):
            te = yrs_cv == y
            ml_cv[name][te] = QuantileMap(pred[~te], obs_cv[~te], md_cv[~te]).apply(pred[te], md_cv[te])
    metrics["cv"]["most_likely"] = _agg_metrics(obs_cv, ml_cv["both"], days)
    metrics["cv"]["most_likely_terra_only"] = _agg_metrics(obs_cv, ml_cv["terra"], days)
    for key, pred in (("harmonized", harm[d_cv]), ("ml_only", oos["ml"][d_cv]), ("monthly_ratio", oos["ratio"][d_cv]),
                      ("raw_modis", raw_m[d_cv]), ("harmonized_terra_only", harm_t[d_cv]),
                      ("most_likely", ml_cv["both"]), ("most_likely_terra_only", ml_cv["terra"])):
        metrics["cv"][key]["distribution"] = _distribution_metrics(obs_cv, pred)
    cvh = metrics["cv"]["harmonized"]
    say(f"transfer CV: daily r={cvh['daily']['r']:.3f} r(log)={cvh['daily']['r_log']:.3f}, "
        f"monthly r(log)={cvh['monthly']['r_log']:.3f}, bias {cvh['daily']['bias_pct']:+.1f}% "
        f"(monthly-ratio baseline daily r={metrics['cv']['monthly_ratio']['daily']['r']:.3f})")
    dm, dh = metrics["cv"]["most_likely"]["distribution"], cvh["distribution"]
    say(f"daily distribution: days without fire {dh['no_fire_days']:.0%} expected / {dm['no_fire_days']:.0%} most likely "
        f"/ {dm['no_fire_days_observed']:.0%} observed; quantile error {dh['quantile_error_log']:.2f} -> "
        f"{dm['quantile_error_log']:.2f}")

    # ---- production model on all stable years ---------------------------------------------------
    d_all = np.flatnonzero(np.isin(yrs, stable) & both)
    booster = _fit_booster(feats, d_all, rng)
    level = LevelModel(cube, d_all, feats.month)
    imp = booster.feature_importance("gain")
    metrics["feature_importance"] = {f: float(v) for f, v in
                                     sorted(zip(TRANSFER_FEATURES, imp / max(imp.sum(), 1)), key=lambda t: -t[1])}
    metrics["monthly_ratio_both"] = level.ratios["both"].round(3).tolist()

    # ---- robustness under Terra/Aqua orbital drift (years after the stable period) -------------
    drift = np.flatnonzero((yrs > max(stable)) & both)
    ml_d = np.full(nd, np.nan)
    rt_d = np.full(nd, np.nan)
    if len(drift):
        ml_d[drift] = booster.predict(feats.frame(drift)).reshape(cube.ncell, -1)[aoi].sum(0)
        rt_d[drift] = level.estimate(drift)
        harm_d = ml_d * level_factor(ml_d, rt_d)
        metrics["drift_test"] = {
            "years": sorted({int(y) for y in yrs[drift]}),
            "harmonized": _agg_metrics(obs_full[drift], harm_d[drift], cube.days[drift]),
            "monthly_ratio": _agg_metrics(obs_full[drift], rt_d[drift], cube.days[drift]),
        }
        harm = np.where(np.isnan(harm), harm_d, harm)

    # VIIRS NOAA-20 / NOAA-21 -> S-NPP scaling (same instrument, different orbit phase)
    def ratio(code):
        ok = cube.avail[V] & cube.avail[code]
        a, b = cube.counts[V][:, ok].sum(), cube.counts[code][:, ok].sum()
        return float(a / b) if b > 100 else None

    r20 = ratio(J1) or 1.0
    r21 = ratio(J2) or r20
    metrics["ratio_noaa20_to_snpp"], metrics["ratio_noaa21_to_snpp"] = r20, r21
    val_days = np.flatnonzero(~np.isnan(harm))
    ml_all = np.full(nd, np.nan)
    ml_all[d_cv] = ml_cv["both"]
    if len(drift):
        ml_all[drift] = qmap["both"].apply(harm_d[drift], cal.md_index(to_dates(cube.days[drift])))
    validation = pd.DataFrame({
        "date": to_dates(cube.days[val_days]), "observed_snpp": obs_full[val_days],
        "harmonized_modis": harm[val_days], "most_likely_modis": ml_all[val_days], "raw_modis": raw_m[val_days],
        "kind": np.where(np.isin(yrs[val_days], stable), "cross-validation", "orbital-drift test"),
    })
    return TransferModel(booster, feats, level, r20, r21, metrics, residual_q, validation, qmap)


# ---------------------------------------------------------------------------------------------
# 5. assemble the harmonized record
# ---------------------------------------------------------------------------------------------

@dataclass
class Harmonized:
    cube: CellCube
    H: np.ndarray            # (ncell, ndays) float32 S-NPP-equivalent detections, NaN = no data
    source: np.ndarray       # (ndays,) int8 SOURCE_LABELS code
    daily: pd.DataFrame      # AOI daily table: y = expected (for totals), y_ml = most likely daily count
    transfer: TransferModel
    sensors: dict            # multi-sensor arrays used as forecast predictors (see multi_sensor)

    @property
    def y_ml(self) -> np.ndarray:
        return self.daily["y_ml"].to_numpy(dtype=np.float32)


def multi_sensor(cube: CellCube, r20: float, r21: float) -> dict[str, np.ndarray]:
    """Per cell-day observations from *all* sensors, used as predictors of tomorrow's activity.

    * ``viirs``: mean of the available VIIRS sensors (S-NPP, NOAA-20, NOAA-21) rescaled to S-NPP;
      more satellites -> a less noisy estimate of the current activity, same expected value.
    * ``frp``: mean VIIRS fire radiative power (MW) - intense fires persist for days.
    * ``modis``: Terra + Aqua detections (morning/afternoon/night overpasses at other times).
    """
    av = cube.avail
    scale = {V: 1.0, J1: r20, J2: r21}
    num = np.zeros((cube.ncell, cube.ndays), np.float32)
    frp = np.zeros_like(num)
    n = np.zeros(cube.ndays, np.float32)
    for code, s in scale.items():
        num += np.where(av[code][None], cube.counts[code] * s, 0)
        frp += np.where(av[code][None], cube.frp[code], 0)
        n += av[code]
    with np.errstate(invalid="ignore", divide="ignore"):
        viirs = np.where(n[None] > 0, num / n[None], np.nan).astype(np.float32)
        frp = np.where(n[None] > 0, frp / n[None], np.nan).astype(np.float32)
    m_ok = av[T] | av[A]
    modis = np.where(m_ok[None], cube.counts[T] * av[T][None] + cube.counts[A] * av[A][None], np.nan)
    return {"viirs": viirs, "frp": frp, "modis": modis.astype(np.float32)}


def _bands(pred: np.ndarray, rq: dict) -> tuple[np.ndarray, np.ndarray]:
    lp = np.log1p(pred)
    idx = np.searchsorted(np.asarray(rq["edges"]), lp)
    lo = np.expm1(np.maximum(lp + np.asarray(rq["q10"])[idx], 0))
    hi = np.expm1(lp + np.asarray(rq["q90"])[idx])
    return np.minimum(lo, pred), np.maximum(hi, pred)


def harmonize(cube: CellCube, model: TransferModel) -> Harmonized:
    av = cube.avail
    ndays = cube.ndays
    source = np.zeros(ndays, np.int8)
    H = np.full((cube.ncell, ndays), np.nan, np.float32)
    s_v = av[V]
    s_j1 = ~s_v & av[J1]
    s_j2 = ~s_v & ~s_j1 & av[J2]
    s_m = ~s_v & ~s_j1 & ~s_j2 & (av[T] | av[A])
    H[:, s_v] = cube.counts[V][:, s_v]
    H[:, s_j1] = model.ratio_noaa20 * cube.counts[J1][:, s_j1]
    H[:, s_j2] = model.ratio_noaa21 * cube.counts[J2][:, s_j2]
    source[s_v], source[s_j1], source[s_j2] = 1, 2, 3
    aoi = cube.in_aoi
    d_m = np.flatnonzero(s_m)
    if len(d_m):
        for chunk in np.array_split(d_m, max(1, len(d_m) // 1500)):
            H[:, chunk] = model.predict_days(chunk)
        ml = np.full(ndays, np.nan)
        rt = np.full(ndays, np.nan)
        ml[d_m] = H[aoi][:, d_m].sum(0)
        rt[d_m] = model.level.estimate(d_m)
        H[:, d_m] *= level_factor(ml, rt)[d_m].astype(np.float32)
        source[d_m] = np.where(av[A, d_m], 4, 5)

    y = np.where(source > 0, np.nansum(H[aoi], axis=0), np.nan)
    lo, hi, y_ml = y.copy(), y.copy(), y.copy()
    if len(d_m):
        lo[d_m], hi[d_m] = _bands(y[d_m], model.residual_q)
        md = cal.md_index(to_dates(cube.days[d_m]))
        terra = source[d_m] == 5
        y_ml[d_m] = model.qmap["both"].apply(y[d_m], md)
        if terra.any():
            y_ml[d_m[terra]] = model.qmap["terra"].apply(y[d_m[terra]], md[terra])
    raw = {}
    for key in ("MODIS_T", "MODIS_A", "VIIRS_SNPP", "VIIRS_NOAA20", "VIIRS_NOAA21"):
        code = SENSOR_CODE[key]
        raw[key] = np.where(av[code], cube.counts[code][aoi].sum(0), np.nan)
    frp_v = np.where(av[V], cube.frp[V][aoi].sum(0), np.nan)
    frp_m = np.where(av[T] | av[A], (cube.frp[T][aoi] * av[T] + cube.frp[A][aoi] * av[A]).sum(0), np.nan)
    daily = pd.DataFrame({
        "date": to_dates(cube.days), "y": y, "y_ml": y_ml, "y_lo": lo, "y_hi": hi, "source": source,
        **{f"n_{k}": v for k, v in raw.items()}, "frp_viirs": frp_v, "frp_modis": frp_m,
        "cells_burning": np.where(source > 0, (np.nan_to_num(H[aoi]) >= 0.5).sum(0), np.nan),
    })
    return Harmonized(cube, H, source, daily, model, multi_sensor(cube, model.ratio_noaa20, model.ratio_noaa21))
