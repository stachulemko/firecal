"""Fire activity forecasting on top of the harmonized record.

Daily model (1-14 days ahead)
    Unit: grid cell x target day, issued on day t0 for lead L = t - t0. Two LightGBM models share
    one feature set - a classifier for P(>=1 S-NPP-equivalent detection) and a Poisson regressor for
    the expected count. Features: the cell's and its neighbourhood's recent burning (persistence),
    the harmonized climatology of that cell and calendar day, weather on the target day (NASA POWER
    in training, bias-corrected Open-Meteo forecasts in operation - "perfect prognosis"), antecedent
    drought (rain deficits, days since rain, soil wetness), ENSO and the lead time.
    AOI totals are the sum of cell expectations; a negative-binomial dispersion fitted on held-out
    data turns them into prediction intervals and the probability of a *critical day* (AOI activity
    above the 90th percentile for that calendar day).

Seasonal outlook (1-3 months ahead)
    Ridge regression of monthly log-anomalies of AOI activity on ENSO, rainfall / temperature /
    soil-moisture anomalies and recent fire anomalies, evaluated by leave-one-year-out CV.

Every metric is computed on years the models never saw.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.metrics import average_precision_score, roc_auc_score

from . import calendar as cal
from .config import FORECAST_LEADS, MAX_LEAD
from .fwi import fwi_system
from .harmonize import Harmonized, to_dates, years_of

log = logging.getLogger(__name__)

WEATHER_TARGET = ["tmax", "rh", "vpd", "wind", "rain", "solar", "rain7", "rain30", "dsr", "tmax_anom", "vpd_anom",
                  "ffmc", "dmc", "dc", "isi", "bui", "fwi", "fwi_anom"]
WEATHER_ISSUE = ["rain90_anom", "rain30_anom", "gwet_top", "gwet_root", "gwet_root_anom"]
FIRE = ["y0", "y1", "y2", "s7", "dsf", "nb_1", "nb_7", "nb5_7", "aoi_1", "aoi_7", "cell_anom7", "aoi_anom7",
        "viirs0", "viirs_nb", "frp0", "frp7", "modis0"]
STATIC = ["clim_c", "clim_nb", "pfire_c", "clim_aoi", "doy_sin", "doy_cos", "lat", "lon", "lead", "oni"]
FEATURES = FIRE + STATIC + WEATHER_TARGET + WEATHER_ISSUE

LGB_BIN = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=0, seed=11)
LGB_CNT = dict(objective="poisson", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_delta_step=10.0, verbose=-1,
               num_threads=0, seed=13)


# ---------------------------------------------------------------------------------------------
# weather cube
# ---------------------------------------------------------------------------------------------

def _es(t):
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


@dataclass
class WeatherCube:
    """Daily weather per point aligned to the fire record's day index (and beyond, for forecasts)."""
    points: list[tuple[float, float]]
    cell_point: np.ndarray       # (ncell,) weather point of every active cell
    day0: int
    ndays: int
    v: dict[str, np.ndarray]     # name -> (npoint, ndays) float32
    last_observed_day: int       # last day of observed (POWER) weather; later days are forecasts
    info: dict = field(default_factory=dict)

    @classmethod
    def from_frames(cls, points, frames: list[pd.DataFrame], cell_point: np.ndarray, day0: int, ndays: int,
                    last_observed_day: int, info: dict | None = None) -> "WeatherCube":
        idx = pd.DatetimeIndex(to_dates(np.arange(day0, day0 + ndays)))
        raw = {k: np.stack([f.reindex(idx)[k].to_numpy(dtype=np.float64) for f in frames])
               for k in frames[0].columns}
        for k in raw:  # short gaps (e.g. radiation lagging other variables) -> carry forward
            raw[k] = pd.DataFrame(raw[k].T).ffill(limit=5).to_numpy().T
        tmax, tdew = raw["T2M_MAX"], raw["T2MDEW"]
        rain = np.nan_to_num(raw["PRECTOTCORR"])
        v = {
            "tmax": tmax, "rh": raw["RH2M"], "vpd": np.maximum(_es(tmax) - _es(tdew), 0.0),
            "wind": raw["WS10M_MAX"], "rain": raw["PRECTOTCORR"], "solar": raw["ALLSKY_SFC_SW_DWN"],
            "gwet_top": raw["GWETTOP"], "gwet_root": raw["GWETROOT"],
        }
        cs = np.cumsum(np.pad(rain, ((0, 0), (1, 0))), axis=1)
        for w in (7, 30, 90):
            s = cs[:, 1:] - cs[:, np.maximum(np.arange(ndays) + 1 - w, 0)]
            v[f"rain{w}"] = s
        wet = rain >= 2.0
        dsr = np.zeros_like(rain)
        run = np.zeros(rain.shape[0])
        for t in range(ndays):
            run = np.where(wet[:, t], 0, np.minimum(run + 1, 120))
            dsr[:, t] = run
        v["dsr"] = dsr
        md = cal.md_index(idx)
        # Canadian FWI system; noon humidity approximated at the daily maximum temperature
        rh_noon = np.clip(100.0 * _es(tdew) / _es(tmax), 1.0, 100.0)
        fw = fwi_system(tmax, rh_noon, raw["WS10M_MAX"] * 3.6, rain, idx.month.to_numpy(),
                        np.array([p[0] for p in points]))
        v.update(fw)
        obs = np.arange(ndays) <= (last_observed_day - day0)
        for name in ("tmax", "vpd", "rain30", "rain90", "gwet_root", "fwi"):
            clim = _md_climatology(v[name], md, obs)
            if name.startswith("rain"):
                v[f"{name}_anom"] = np.log((v[name] + 1.0) / (clim + 1.0))
            else:
                v[f"{name}_anom"] = v[name] - clim
        v = {k: a.astype(np.float32) for k, a in v.items()}
        return cls(list(points), cell_point, day0, ndays, v, last_observed_day, info or {})


def _md_climatology(x: np.ndarray, md: np.ndarray, valid: np.ndarray, window: int = 15) -> np.ndarray:
    """Per-point calendar-day mean (+-window days), mapped back onto every day."""
    npnt = x.shape[0]
    s = np.zeros((npnt, cal.NMD))
    n = np.zeros(cal.NMD)
    ok = valid & ~np.isnan(x).any(axis=0)
    np.add.at(s.T, md[ok], x[:, ok].T)
    np.add.at(n, md[ok], 1)
    k = np.ones(2 * window + 1)
    s3 = np.concatenate([s, s, s], axis=1)
    n3 = np.concatenate([n, n, n])
    ss = np.stack([np.convolve(r, k, mode="same") for r in s3])[:, cal.NMD:2 * cal.NMD]
    nn = np.convolve(n3, k, mode="same")[cal.NMD:2 * cal.NMD]
    clim = ss / np.maximum(nn, 1)
    return clim[:, md]


# ---------------------------------------------------------------------------------------------
# fire-history tables
# ---------------------------------------------------------------------------------------------

def _neighbor_sum(x: np.ndarray, nb: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    for k in range(nb.shape[1]):
        idx = nb[:, k]
        ok = idx >= 0
        out[ok] += np.nan_to_num(x[idx[ok]])
    return out


def _trailing_sum(x: np.ndarray, w: int) -> np.ndarray:
    """Sum over [t-w+1, t] scaled up for missing days (NaN only if all days are missing)."""
    v = ~np.isnan(x)
    cs = np.cumsum(np.pad(np.nan_to_num(x), ((0, 0), (1, 0))), axis=1)
    cn = np.cumsum(np.pad(v, ((0, 0), (1, 0))), axis=1)
    idx = np.arange(x.shape[1])
    lo = np.maximum(idx + 1 - w, 0)
    s = cs[:, idx + 1] - cs[:, lo]
    n = cn[:, idx + 1] - cn[:, lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s * w / n, np.nan)


class FireTables:
    """Persistence and climatology tables derived from the harmonized cell record."""

    def __init__(self, harm: Harmonized, clim_years: list[int]):
        # climatology years must be given up front so tests never see their own years
        cube = harm.cube
        H = harm.H.astype(np.float64)
        self.H = H.astype(np.float32)
        self.ncell, self.ndays = H.shape
        self.day0 = cube.day0
        self.aoi = cube.in_aoi
        self.dates = to_dates(cube.days)
        self.md = cal.md_index(self.dates)
        self.years = years_of(cube.days)
        nb = cube.local_neighbors(1)
        nb5 = cube.local_neighbors(2)
        s7 = _trailing_sum(H, 7)
        f32 = np.float32
        self.s7 = s7.astype(f32)
        self.nb_1 = np.where(np.isnan(H).all(0)[None], np.nan, _neighbor_sum(H, nb)).astype(f32)
        self.nb_7 = np.where(np.isnan(s7).all(0)[None], np.nan, _neighbor_sum(s7, nb)).astype(f32)
        self.nb5_7 = np.where(np.isnan(s7).all(0)[None], np.nan, _neighbor_sum(s7, nb5)).astype(f32)
        aoi_daily = np.where(np.isnan(H[self.aoi]).all(0), np.nan, np.nansum(H[self.aoi], axis=0))
        self.aoi_1 = aoi_daily
        # most likely single-day values (differ from aoi_1 on reconstructed days): daily quantiles
        y_ml = getattr(harm, "y_ml", None)
        self.aoi_ml = aoi_daily if y_ml is None else np.asarray(y_ml, dtype=np.float64)
        self.aoi_7 = _trailing_sum(aoi_daily[None], 7)[0]
        fire = (H >= 0.5).astype(np.float64)
        fire[np.isnan(H)] = np.nan
        # days since fire within the last 7 days (8 = none)
        dsf = np.full(H.shape, 8.0)
        for k in range(7, -1, -1):
            shifted = np.full(H.shape, np.nan)
            shifted[:, k:] = fire[:, :self.ndays - k] if k else fire
            dsf = np.where(shifted == 1, k, dsf)
        self.dsf = dsf.astype(np.float32)
        lat, lon = cube.grid.centers()
        self.lat, self.lon = lat[cube.cells], lon[cube.cells]
        self.nb = nb
        ms = getattr(harm, "sensors", None) or {}
        nan = np.full(H.shape, np.nan, np.float32)
        self.viirs0 = ms.get("viirs", nan)
        self.viirs_nb = np.where(np.isnan(self.viirs0).all(0)[None], np.nan,
                                 _neighbor_sum(self.viirs0.astype(np.float64), nb)).astype(np.float32)
        self.frp0 = ms.get("frp", nan)
        self.frp7 = _trailing_sum(self.frp0.astype(np.float64), 7).astype(np.float32)
        self.modis0 = ms.get("modis", nan)
        self.set_climatology(clim_years)

    def set_climatology(self, years: list[int]):
        """Cell / AOI calendar-day climatology from the given years only (no look-ahead in tests)."""
        sel = np.isin(self.years, years) & ~np.isnan(self.H).all(0)
        s = np.zeros((self.ncell, cal.NMD))
        f = np.zeros((self.ncell, cal.NMD))
        n = np.zeros(cal.NMD)
        Hs = np.nan_to_num(self.H[:, sel]).astype(np.float64)
        np.add.at(s.T, self.md[sel], Hs.T)
        np.add.at(f.T, self.md[sel], (Hs >= 0.5).T.astype(float))
        np.add.at(n, self.md[sel], 1)
        k = np.ones(31)

        def smooth(a):
            a3 = np.concatenate([a, a, a], axis=-1)
            if a.ndim == 1:
                return np.convolve(a3, k, mode="same")[cal.NMD:2 * cal.NMD]
            return np.stack([np.convolve(r, k, mode="same") for r in a3])[:, cal.NMD:2 * cal.NMD]

        nn = np.maximum(smooth(n), 1)
        self.clim_c = smooth(s) / nn
        self.pfire_c = smooth(f) / nn
        self.clim_nb = _neighbor_sum(self.clim_c, self.nb) / 9.0
        self.clim_aoi = self.clim_c[self.aoi].sum(0)
        self.clim_years = list(years)
        aoi_daily = pd.DataFrame({"date": self.dates, "y": self.aoi_1, "y_ml": self.aoi_ml})
        self.aoi_clim = cal.Climatology(aoi_daily, years)


# ---------------------------------------------------------------------------------------------
# feature assembly
# ---------------------------------------------------------------------------------------------

def _oni_lookup(oni: pd.Series, dates: np.ndarray, lag_months: int = 2) -> np.ndarray:
    m = pd.DatetimeIndex(dates).to_period("M") - lag_months
    s = oni.copy()
    s.index = s.index.to_period("M")
    s = s.reindex(pd.period_range(min(s.index.min(), m.min()), max(s.index.max(), m.max()), freq="M")).ffill()
    return s.reindex(m).to_numpy(dtype=np.float32)


def build_block(ft: FireTables, wx: WeatherCube, oni: pd.Series, i0: np.ndarray, lead: int,
                persist_weather: bool = False) -> pd.DataFrame:
    """Features for all cells x issue days ``i0`` at one lead (issue-major row order)."""
    i = i0 + lead
    nc, ni = ft.ncell, len(i0)

    def cell_at(arr, idx):  # (ncell, ndays) -> rows
        return arr[:, idx].T.reshape(-1)

    def aoi_at(arr, idx):  # (ndays,) -> rows
        return np.repeat(arr[idx], nc)

    md_t = cal.md_index(to_dates(ft.day0 + i))
    doy = 2 * np.pi * md_t / 366.0
    f = {
        "y0": cell_at(ft.H, i0), "y1": cell_at(ft.H, i0 - 1), "y2": cell_at(ft.H, i0 - 2),
        "s7": cell_at(ft.s7, i0), "dsf": cell_at(ft.dsf, i0),
        "viirs0": cell_at(ft.viirs0, i0), "viirs_nb": cell_at(ft.viirs_nb, i0), "frp0": cell_at(ft.frp0, i0),
        "frp7": cell_at(ft.frp7, i0), "modis0": cell_at(ft.modis0, i0),
        "nb_1": cell_at(ft.nb_1, i0), "nb_7": cell_at(ft.nb_7, i0), "nb5_7": cell_at(ft.nb5_7, i0),
        "aoi_1": aoi_at(ft.aoi_1, i0), "aoi_7": aoi_at(ft.aoi_7, i0),
        "clim_c": ft.clim_c[:, md_t].T.reshape(-1), "clim_nb": ft.clim_nb[:, md_t].T.reshape(-1),
        "pfire_c": ft.pfire_c[:, md_t].T.reshape(-1), "clim_aoi": np.repeat(ft.clim_aoi[md_t], nc),
        "doy_sin": np.repeat(np.sin(doy), nc), "doy_cos": np.repeat(np.cos(doy), nc),
        "lat": np.tile(ft.lat, ni), "lon": np.tile(ft.lon, ni),
        "lead": np.full(nc * ni, lead, np.float32),
        "oni": np.repeat(_oni_lookup(oni, to_dates(ft.day0 + i0)), nc),
    }
    md0 = cal.md_index(to_dates(ft.day0 + i0))
    clim7_c = 7 * ft.clim_c[:, md0].T.reshape(-1)
    f["cell_anom7"] = np.log((np.nan_to_num(f["s7"], nan=0) + 0.1) / (clim7_c + 0.1))
    f["aoi_anom7"] = np.repeat(np.log((np.nan_to_num(ft.aoi_7[i0]) + 1) / (7 * ft.clim_aoi[md0] + 1)), nc)
    off = ft.day0 - wx.day0
    p = wx.cell_point[:, None]
    tw = (i0 + off if persist_weather else i + off)[None, :]  # persistence = no knowledge of future weather
    for name in WEATHER_TARGET:
        f[name] = wx.v[name][p, tw].T.reshape(-1)
    for name in WEATHER_ISSUE:
        f[name] = wx.v[name][p, (i0 + off)[None, :]].T.reshape(-1)
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})[FEATURES]


def _target(ft: FireTables, i: np.ndarray) -> np.ndarray:
    return ft.H[:, i].T.reshape(-1)


# ---------------------------------------------------------------------------------------------
# training & evaluation
# ---------------------------------------------------------------------------------------------

def _rows(ft, wx, oni, i0_all, rng, observed_target, quiet_keep=0.25, leads=FORECAST_LEADS):
    """Random lead per (cell, issue day); quiet rows subsampled and up-weighted."""
    Xs, ys, ws, meta = [], [], [], []
    for lead in leads:
        i0 = i0_all[observed_target[i0_all + lead]]
        i0 = i0[rng.random(len(i0)) < 1.0 / len(leads)]
        if not len(i0):
            continue
        X = build_block(ft, wx, oni, i0, lead)
        y = _target(ft, i0 + lead)
        quiet = (y == 0) & (np.nan_to_num(X["nb5_7"].to_numpy()) == 0) & (X["pfire_c"].to_numpy() < 0.02)
        keep = ~np.isnan(y) & (~quiet | (rng.random(len(y)) < quiet_keep))
        Xs.append(X[keep])
        ys.append(y[keep])
        ws.append(np.where(quiet[keep], 1 / quiet_keep, 1.0))
        meta.append(np.column_stack([np.repeat(i0, ft.ncell)[keep], np.full(keep.sum(), lead)]))
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys), np.concatenate(ws), np.vstack(meta)


class ResidualModel:
    """Empirical predictive distribution of AOI totals: log1p(obs) - log1p(expected).

    Residuals come from days the model did not see, grouped by lead time and by the size of the
    expected value (the spread is very different for quiet and active days). It converts the
    expected value into a median, prediction intervals and exceedance probabilities, and on
    held-out years its 80% intervals cover ~80% of outcomes (a fitted negative binomial covered
    ~98% - far too wide).
    """

    LEAD_GROUPS = ((1, 2), (3, 5), (6, 9), (10, 14))

    def __init__(self, lead: np.ndarray, lmu: np.ndarray, resid: np.ndarray, nbins: int = 3):
        self.edges = np.quantile(lmu, np.linspace(0, 1, nbins + 1)[1:-1]) if len(lmu) else np.array([])
        self.table = {}
        for g, (a, b) in enumerate(self.LEAD_GROUPS):
            in_g = (lead >= a) & (lead <= b)
            bins = np.searchsorted(self.edges, lmu[in_g])
            for k in range(nbins):
                r = np.sort(resid[in_g][bins == k])
                if len(r) < 20:  # fall back to the whole lead group
                    r = np.sort(resid[in_g])
                self.table[(g, k)] = r

    def _res(self, lead: int, lmu: float) -> np.ndarray:
        g = next((i for i, (a, b) in enumerate(self.LEAD_GROUPS) if a <= lead <= b), len(self.LEAD_GROUPS) - 1)
        return self.table[(g, int(np.searchsorted(self.edges, lmu)))]

    def quantiles(self, mu: np.ndarray, lead: int, qs=(0.1, 0.5, 0.9)) -> dict:
        lmu = np.log1p(np.atleast_1d(np.asarray(mu, float)))
        out = {q: np.empty(len(lmu)) for q in qs}
        for i, v in enumerate(lmu):
            r = self._res(lead, v)
            for q in qs:
                out[q][i] = np.expm1(max(v + np.quantile(r, q), 0.0))
        return out

    def exceed(self, mu: np.ndarray, lead: int, thr: np.ndarray) -> np.ndarray:
        lmu = np.log1p(np.atleast_1d(np.asarray(mu, float)))
        thr = np.broadcast_to(np.asarray(thr, float), lmu.shape)
        out = np.empty(len(lmu))
        for i, (v, t) in enumerate(zip(lmu, thr)):
            r = self._res(lead, v)
            need = np.log1p(t) - v
            out[i] = (np.sum(r >= need) + 0.5) / (len(r) + 1.0)  # smoothed so no probability is exactly 0/1
        return out


def _aoi_eval(ft, wx, oni, models, i0, lead, persist_weather=False):
    """AOI-level predictions for issue days i0 at one lead."""
    X = build_block(ft, wx, oni, i0, lead, persist_weather)
    mu_cell = models["cnt"].predict(X).reshape(len(i0), ft.ncell)
    p_cell = models["occ"].predict(X).reshape(len(i0), ft.ncell)
    mu = mu_cell[:, ft.aoi].sum(1)
    obs = ft.aoi_1[i0 + lead]
    return X, mu_cell, p_cell, mu, obs


def _mae_log(obs, pred):
    return float(np.mean(np.abs(np.log1p(obs) - np.log1p(pred))))


def _skill(obs, pred, ref):
    """1 - MAE/MAE_ref on the log1p scale (1 = perfect, 0 = no better than the reference)."""
    er = _mae_log(obs, ref)
    return float(1 - _mae_log(obs, pred) / er) if er > 0 else None


@dataclass
class DailyForecaster:
    models: dict                 # operational models (all observed years)
    residuals: ResidualModel     # calibrated on three unseen years of the evaluation models
    metrics: dict
    importance: dict
    best_iter: dict
    eval_models: dict            # trained only on years < first_unseen_year: honest backtests
    eval_residuals: ResidualModel
    eval_clim_years: list
    first_unseen_year: int
    clim_years: list


# A single validation year can stop the count model after a few dozen trees (Sierras de Cordoba: 36,
# with a moderate 2022 after the extreme 2020). On the unseen test years, 150 trees ranked days better
# (critical-day AUC 0.72 -> 0.75, area Spearman 0.43 -> 0.50) and left Portugal and the Delta unchanged.
MIN_COUNT_ROUNDS = 150


def _train_pair(X, y, w, Xv=None, yv=None, wv=None, rounds=None):
    occ_y = (y >= 0.5).astype(float)
    if Xv is not None:
        occ = lgb.train(LGB_BIN, lgb.Dataset(X, occ_y, weight=w), 2000,
                        valid_sets=[lgb.Dataset(Xv, (yv >= 0.5).astype(float), weight=wv)],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        cnt = lgb.train(LGB_CNT, lgb.Dataset(X, y, weight=w), 2000,
                        valid_sets=[lgb.Dataset(Xv, yv, weight=wv)],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        if cnt.best_iteration < MIN_COUNT_ROUNDS:
            cnt = lgb.train(LGB_CNT, lgb.Dataset(X, y, weight=w), MIN_COUNT_ROUNDS)
        return {"occ": occ, "cnt": cnt}, {"occ": occ.best_iteration, "cnt": max(cnt.best_iteration, MIN_COUNT_ROUNDS)}
    occ = lgb.train(LGB_BIN, lgb.Dataset(X, occ_y, weight=w), max(50, rounds["occ"]))
    cnt = lgb.train(LGB_CNT, lgb.Dataset(X, y, weight=w), max(50, rounds["cnt"]))
    return {"occ": occ, "cnt": cnt}, rounds


def _collect_residuals(ft, wx, oni, models, i0_all, obs_mask, years, ys):
    leads, lmus, res = [], [], []
    for lead in range(1, MAX_LEAD + 1):
        t = np.minimum(i0_all + lead, ft.ndays - 1)
        i0 = i0_all[np.isin(years[t], ys) & obs_mask[t]]
        if not len(i0):
            continue
        _, _, _, mu, obs = _aoi_eval(ft, wx, oni, models, i0, lead)
        ok = ~np.isnan(obs)
        leads.append(np.full(ok.sum(), lead))
        lmus.append(np.log1p(mu[ok]))
        res.append(np.log1p(obs[ok]) - np.log1p(mu[ok]))
    if not leads:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(leads), np.concatenate(lmus), np.concatenate(res)


def train_daily(harm: Harmonized, wx: WeatherCube, oni: pd.Series, log_fn=None, seed: int = 3) -> tuple[DailyForecaster, FireTables]:
    """Fit, evaluate on held-out years, then refit on everything for operational use."""
    say = log_fn or (lambda m: log.info(m))
    rng = np.random.default_rng(seed)
    observed = np.isin(harm.source, (1, 2, 3))  # VIIRS-observed days are the training truth
    years = years_of(harm.cube.days)
    viirs_years = sorted({int(y) for y in years[observed]})
    full_years = [y for y in viirs_years if observed[years == y].mean() > 0.9]
    if len(full_years) < 5:
        raise RuntimeError("need at least 5 years of VIIRS observations to train the forecaster")
    test_years = full_years[-2:]
    valid_year = full_years[-3]
    train_years = [y for y in full_years if y < valid_year]
    hist_years = sorted({int(y) for y in years if y < valid_year and (harm.source[years == y] > 0).mean() > 0.9})
    ft = FireTables(harm, hist_years)  # climatology may use the MODIS-era backcast, never the test years
    first_issue = max(30, int(np.argmax(observed)) + 8)
    last_issue = ft.ndays - 1 - MAX_LEAD
    i0_all = np.arange(first_issue, last_issue + 1)

    def rows_for(ys, keep_quiet=0.25):
        return _rows(ft, wx, oni, i0_all, rng, observed & np.isin(years, ys), keep_quiet)

    say(f"forecast model: train {train_years[0]}-{train_years[-1]}, validation {valid_year}, test {test_years}")
    X, y, w, _ = rows_for(train_years)
    Xv, yv, wv, _ = rows_for([valid_year], 1.0)
    say(f"forecast model: {len(y):,} training rows, {len(FEATURES)} features")
    models, best = _train_pair(X, y, w, Xv, yv, wv)
    resid_valid = ResidualModel(*_collect_residuals(ft, wx, oni, models, i0_all, observed, years, [valid_year]))

    # ---- held-out test years ------------------------------------------------------------------
    metrics = {"train_years": train_years, "valid_year": valid_year, "test_years": test_years, "by_lead": {},
               "note": "Scores on years never used for training or calibration. Weather on the target day is "
                       "observed reanalysis (upper bound for operational skill); the *_no_weather_forecast "
                       "scores hold the issue-day weather constant (lower bound)."}
    for lead in (1, 3, 7, 14):
        t = np.minimum(i0_all + lead, ft.ndays - 1)
        i0 = i0_all[np.isin(years[t], test_years) & observed[t]]
        if len(i0) < 30:
            continue
        X_t, mu_cell, p_cell, mu, obs = _aoi_eval(ft, wx, oni, models, i0, lead)
        y_cell = _target(ft, i0 + lead)
        ok_c = ~np.isnan(y_cell)
        occ = (y_cell[ok_c] >= 0.5).astype(int)
        p = p_cell.reshape(-1)[ok_c]
        pclim = X_t["pfire_c"].to_numpy()[ok_c]
        bs, bs_clim = np.mean((p - occ) ** 2), np.mean((pclim - occ) ** 2)
        ok = ~np.isnan(obs)
        obs, mu, i0 = obs[ok], mu[ok], i0[ok]
        md_t = ft.md[i0 + lead]
        clim_med = ft.aoi_clim.q[0.5][md_t]
        persist = np.nan_to_num(ft.aoi_7[i0]) / 7.0
        thr = ft.aoi_clim.q[0.9][md_t]
        crit = (obs >= thr) & (thr >= 1)
        q = resid_valid.quantiles(mu, lead, (0.1, 0.5, 0.9))
        p_crit = resid_valid.exceed(mu, lead, np.maximum(thr, 1))
        _, _, _, mu_pw, _ = _aoi_eval(ft, wx, oni, models, i0, lead, persist_weather=True)
        med_pw = resid_valid.quantiles(mu_pw, lead, (0.5,))[0.5]
        base_rate = float(np.mean(crit))
        m = {
            "n_days": int(len(obs)),
            "cell_auc": float(roc_auc_score(occ, p)) if 0 < occ.mean() < 1 else None,
            "cell_auc_climatology": float(roc_auc_score(occ, pclim)) if 0 < occ.mean() < 1 else None,
            "cell_avg_precision": float(average_precision_score(occ, p)) if occ.any() else None,
            "cell_brier_skill": float(1 - bs / bs_clim) if bs_clim > 0 else None,
            "aoi_spearman": float(stats.spearmanr(obs, q[0.5]).statistic),
            "aoi_spearman_persistence": float(stats.spearmanr(obs, persist).statistic),
            "aoi_mae_log": _mae_log(obs, q[0.5]),
            "aoi_skill_vs_climatology": _skill(obs, q[0.5], clim_med),
            "aoi_skill_vs_persistence": _skill(obs, q[0.5], persist),
            "aoi_spearman_no_weather_forecast": float(stats.spearmanr(obs, med_pw).statistic),
            "aoi_skill_vs_climatology_no_weather_forecast": _skill(obs, med_pw, clim_med),
            "interval80_coverage": float(np.mean((obs >= q[0.1]) & (obs <= q[0.9]))),
            "critical_days": int(crit.sum()),
            "critical_auc": float(roc_auc_score(crit, p_crit)) if 0 < crit.mean() < 1 else None,
            "critical_brier": float(np.mean((p_crit - crit) ** 2)),
            # reference: always forecasting the climatological 10% chance of a critical day
            "critical_brier_climatology": float(np.mean((0.1 - crit) ** 2)),
            "critical_base_rate": base_rate,
        }
        m["critical_brier_skill"] = (1 - m["critical_brier"] / m["critical_brier_climatology"]
                                     if m["critical_brier_climatology"] > 0 else None)
        metrics["by_lead"][str(lead)] = m
        say(f"forecast test lead {lead:2d}d: cell AUC {m['cell_auc']:.3f} (clim {m['cell_auc_climatology']:.3f}) | "
            f"AOI rho {m['aoi_spearman']:.2f} (persistence {m['aoi_spearman_persistence']:.2f}), "
            f"skill vs clim {m['aoi_skill_vs_climatology']:+.2f}, vs persistence {m['aoi_skill_vs_persistence']:+.2f}, "
            f"80% interval coverage {m['interval80_coverage']:.2f}, critical AUC {m['critical_auc'] or float('nan'):.2f}")

    # ---- operational refit on all observed years -----------------------------------------------
    # The predictive distribution of the refit model is calibrated with the out-of-sample residuals of
    # the evaluation model on the validation *and* test years (three unseen years).
    resid_all = ResidualModel(*_collect_residuals(ft, wx, oni, models, i0_all, observed, years,
                                                  [valid_year] + test_years))
    all_hist = sorted({int(y) for y in years if (harm.source[years == y] > 0).mean() > 0.9})
    ft.set_climatology(all_hist)
    Xf, yf, wf, _ = rows_for(full_years + [y for y in viirs_years if y not in full_years])
    final, _ = _train_pair(Xf, yf, wf, rounds={k: int(v * 1.1) for k, v in best.items()})
    imp = final["cnt"].feature_importance("gain")
    importance = {f: float(v) for f, v in sorted(zip(FEATURES, imp / max(imp.sum(), 1)), key=lambda t: -t[1])}
    return DailyForecaster(final, resid_all, metrics, importance, best, models, resid_valid, hist_years,
                           valid_year, all_hist), ft


# EFFIS fire danger classes of the Fire Weather Index (European Forest Fire Information System)
FWI_CLASSES = ((5.2, "very low"), (11.2, "low"), (21.3, "moderate"), (38.0, "high"), (50.0, "very high"),
               (float("inf"), "extreme"))


def fwi_class(v: float) -> str:
    return next(name for upper, name in FWI_CLASSES if v < upper)


def aoi_weather_weights(ft: FireTables, wx: WeatherCube) -> np.ndarray:
    """Weights of the weather points for area summaries: where the area's fires usually are."""
    w_cell = np.where(ft.aoi, ft.clim_c.mean(axis=1), 0.0) + 1e-6 * ft.aoi
    pw = np.bincount(wx.cell_point, weights=w_cell, minlength=len(wx.points))
    return pw / max(pw.sum(), 1e-12)


def predict_daily(fc: DailyForecaster, ft: FireTables, wx: WeatherCube, oni: pd.Series, issue_idx: int,
                  max_lead: int = MAX_LEAD, use_eval: bool = False) -> dict:
    """Forecast from issue day index ``issue_idx`` for leads 1..max_lead.

    ``use_eval`` selects the evaluation models (trained before ``fc.first_unseen_year``); ``ft`` must
    then carry the matching climatology (``fc.eval_clim_years``).
    """
    models = fc.eval_models if use_eval else fc.models
    residuals = fc.eval_residuals if use_eval else fc.residuals
    off = ft.day0 - wx.day0
    max_lead = int(min(max_lead, wx.ndays - 1 - (issue_idx + off)))
    pw = aoi_weather_weights(ft, wx)
    rows, cells = [], []
    for lead in range(1, max_lead + 1):
        X = build_block(ft, wx, oni, np.array([issue_idx]), lead)
        mu_cell = models["cnt"].predict(X)
        p_cell = models["occ"].predict(X)
        mu = float(mu_cell[ft.aoi].sum())
        day = ft.day0 + issue_idx + lead
        md = int(cal.md_index(to_dates(np.array([day])))[0])
        thr = max(float(ft.aoi_clim.q[0.9][md]), 1.0)
        q = residuals.quantiles(mu, lead)
        obs = (float(ft.aoi_1[issue_idx + lead])
               if issue_idx + lead < ft.ndays and not np.isnan(ft.aoi_1[issue_idx + lead]) else None)
        rows.append({
            "date": str(to_dates(np.array([day]))[0]), "lead": lead, "expected": mu,
            "q10": float(q[0.1][0]), "median": float(q[0.5][0]), "q90": float(q[0.9][0]),
            "p_critical": float(residuals.exceed(mu, lead, thr)[0]), "critical_threshold": thr,
            "normal": float(ft.aoi_clim.q[0.5][md]), "normal_mean": float(ft.aoi_clim.mean[md]),
            "normal_q90": float(ft.aoi_clim.q[0.9][md]), "observed": obs,
            "weather": "forecast" if day > wx.last_observed_day else "observed",
            "fire_weather": _fire_weather(wx, pw, issue_idx + lead + off),
        })
        cells.append({"lead": lead, "p_fire": np.round(p_cell, 4).tolist(), "expected": np.round(mu_cell, 3).tolist()})
    return {"issue_date": str(to_dates(np.array([ft.day0 + issue_idx]))[0]), "days": rows, "cells": cells}


def _fire_weather(wx: WeatherCube, pw: np.ndarray, t: int) -> dict:
    """Area-weighted fire weather of one day (the model's weather inputs, for display)."""
    v = {k: float(np.nansum(wx.v[k][:, t] * pw)) for k in ("fwi", "fwi_anom", "isi", "tmax", "rh", "wind", "rain")}
    v["fwi_class"] = fwi_class(v["fwi"])
    return v


# ---------------------------------------------------------------------------------------------
# seasonal (monthly) outlook
# ---------------------------------------------------------------------------------------------

def _monthly_frame(ft: FireTables, wx: WeatherCube, oni: pd.Series) -> pd.DataFrame:
    d = pd.DataFrame({"date": pd.DatetimeIndex(ft.dates), "y": ft.aoi_1})
    m = d.set_index("date")["y"].resample("MS").agg(["sum", "count"])
    m["total"] = np.where(m["count"] >= 0.8 * m.index.days_in_month, m["sum"], np.nan)  # unfinished month -> NaN
    m["log"] = np.log1p(m["total"])
    m["month"] = m.index.month
    m["anom"] = m["log"] - m.groupby("month")["log"].transform("mean")
    off = ft.day0 - wx.day0
    idx = pd.DatetimeIndex(to_dates(np.arange(wx.day0, wx.day0 + wx.ndays)))
    obs_mask = np.arange(wx.ndays) <= wx.last_observed_day - wx.day0
    w_aoi = ft.aoi.astype(float)
    pw = np.bincount(wx.cell_point, weights=w_aoi, minlength=len(wx.points))
    pw = pw / max(pw.sum(), 1)
    def series(name):
        return pd.Series(np.nansum(wx.v[name] * pw[:, None], axis=0), index=idx)[obs_mask]
    for name in ("rain30_anom", "rain90_anom", "tmax_anom", "gwet_root_anom"):
        s = series(name).resample("MS").last() if "rain" in name else series(name).resample("MS").mean()
        m[name] = s.reindex(m.index)
    # ONI of the previous month: the 3-month mean centred on the issue month is not yet published
    prev = (m.index - pd.DateOffset(months=1))
    m["oni"] = oni.reindex(oni.index.union(prev)).ffill().reindex(prev).to_numpy()
    return m


OUTLOOK_FEATURES = ["anom_0", "anom_1", "oni", "rain30_anom", "rain90_anom", "tmax_anom", "gwet_root_anom",
                    "tsin", "tcos"]


def _outlook_rows(m: pd.DataFrame, lead: int) -> pd.DataFrame:
    """Row per issue month: drivers at the end of that month, target = anomaly ``lead`` months later."""
    r = pd.DataFrame(index=m.index)
    r["anom_0"] = m["anom"]
    r["anom_1"] = m["anom"].shift(1)
    for c in ("oni", "rain30_anom", "rain90_anom", "tmax_anom", "gwet_root_anom"):
        r[c] = m[c]
    tm = (m.index.month - 1 + lead) % 12
    r["tsin"], r["tcos"] = np.sin(2 * np.pi * tm / 12), np.cos(2 * np.pi * tm / 12)
    r["target"] = m["anom"].shift(-lead)
    r["target_month"] = tm + 1
    return r


@dataclass
class Outlook:
    models: dict          # (lead, with_fire) -> fitted pipeline
    metrics: dict
    rmse: dict


def train_outlook(ft: FireTables, wx: WeatherCube, oni: pd.Series) -> tuple[Outlook, pd.DataFrame]:
    m = _monthly_frame(ft, wx, oni)
    models, metrics, rmse = {}, {}, {}
    for lead in (1, 2, 3):
        rows = _outlook_rows(m, lead).dropna(subset=["target"])
        for with_fire in (True, False):
            cols = OUTLOOK_FEATURES if with_fire else [c for c in OUTLOOK_FEATURES if not c.startswith("anom")]
            data = rows.dropna(subset=cols)
            if len(data) < 48:
                continue
            years = data.index.year
            pred = np.full(len(data), np.nan)
            for y in np.unique(years):  # leave-one-year-out
                tr, te = years != y, years == y
                mdl = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(data.loc[tr, cols], data.loc[tr, "target"])
                pred[te] = mdl.predict(data.loc[te, cols])
            t = data["target"].to_numpy()
            err = pred - t
            lo_t, hi_t = np.quantile(t, [1 / 3, 2 / 3])
            cls_t = np.digitize(t, [lo_t, hi_t])
            cls_p = np.digitize(pred, [lo_t, hi_t])
            key = f"{lead}_{'fire' if with_fire else 'nofire'}"
            metrics[key] = {
                "n": int(len(t)), "r": float(np.corrcoef(t, pred)[0, 1]),
                "rmse": float(np.sqrt(np.mean(err ** 2))), "rmse_climatology": float(np.sqrt(np.mean(t ** 2))),
                "skill": float(1 - np.mean(err ** 2) / np.mean(t ** 2)),
                "tercile_hit_rate": float(np.mean(cls_t == cls_p)),
            }
            rmse[key] = metrics[key]["rmse"]
            models[key] = (RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(data[cols], data["target"]), cols)
    return Outlook(models, metrics, rmse), m


def predict_outlook(ol: Outlook, m: pd.DataFrame, issue_month: pd.Timestamp | None = None) -> list[dict]:
    """Expected AOI activity for the next three months versus normal, with tercile probabilities."""
    if issue_month is None:
        issue_month = m.dropna(subset=["rain30_anom"]).index.max()
    clim_log = m.groupby("month")["log"].mean()
    clim_tot = m.groupby("month")["total"].mean()
    terc = m.groupby("month")["anom"].quantile([1 / 3, 2 / 3]).unstack()
    out = []
    for lead in (1, 2, 3):
        row = _outlook_rows(m, lead).loc[[issue_month]]
        has_fire = not row[["anom_0", "anom_1"]].isna().any(axis=1).iloc[0]
        key = f"{lead}_{'fire' if has_fire else 'nofire'}"
        if key not in ol.models:
            continue
        mdl, cols = ol.models[key]
        x = row[cols]
        if x.isna().any(axis=1).iloc[0]:
            continue
        a = float(mdl.predict(x)[0])
        s = ol.rmse[key]
        tm = int(row["target_month"].iloc[0])
        lo, hi = terc.loc[tm, 1 / 3], terc.loc[tm, 2 / 3]
        p_below = float(stats.norm.cdf(lo, a, s))
        p_above = float(1 - stats.norm.cdf(hi, a, s))
        month = (issue_month + pd.DateOffset(months=lead)).strftime("%Y-%m")
        out.append({
            "month": month, "lead_months": lead, "anomaly_log": a, "ratio_to_normal": float(np.exp(a)),
            "expected_total": float(np.expm1(clim_log[tm] + a)), "normal_total": float(clim_tot[tm]),
            "p_below": p_below, "p_near": max(0.0, 1 - p_below - p_above), "p_above": p_above,
            "uses_recent_fire": has_fire, "skill": ol.metrics[key]["skill"], "r": ol.metrics[key]["r"],
        })
    return out
