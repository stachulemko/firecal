"""Burning activity calendar: climatology, anomalies, fire-season metrics, critical periods, events.

All functions take the harmonized AOI daily table (``date``, ``y`` = S-NPP-equivalent detections,
optionally ``y_ml``). On reconstructed (MODIS-era) days ``y`` is the expected count, which is right
for sums: totals, means, 7-day anomalies, seasons, events. ``y_ml`` is the most likely count of the
single day; everything that looks at single days (the heat map, daily quantiles, the chance of a
day with fire) uses it so that reconstructed and observed years are distributed alike.
Days are indexed on a leap-year calendar ("month-day index", 0..365) so that the same calendar day
lines up across years; Feb 29 simply has fewer samples.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

NMD = 366
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_LEAP_STARTS = np.cumsum([0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30])


def md_index(dates) -> np.ndarray:
    d = pd.DatetimeIndex(dates)
    return (_LEAP_STARTS[d.month.to_numpy() - 1] + d.day.to_numpy() - 1).astype(int)


def md_label(i: int) -> str:
    i = int(i) % NMD
    m = int(np.searchsorted(_LEAP_STARTS, i, side="right")) - 1
    return f"{MONTH_ABBR[m]} {i - _LEAP_STARTS[m] + 1}"


def year_matrix(daily: pd.DataFrame, col: str = "y") -> tuple[np.ndarray, np.ndarray]:
    """(years, matrix[year, md_index]) with NaN for missing days / Feb 29 of common years."""
    dates = pd.DatetimeIndex(daily["date"])
    years = np.arange(dates.year.min(), dates.year.max() + 1)
    Y = np.full((len(years), NMD), np.nan)
    Y[dates.year.to_numpy() - years[0], md_index(dates)] = daily[col].to_numpy(dtype=float)
    return years, Y


def complete_years(daily: pd.DataFrame, min_coverage: float = 0.9) -> list[int]:
    d = daily.assign(year=pd.DatetimeIndex(daily["date"]).year)
    cov = d.groupby("year")["y"].apply(lambda s: s.notna().mean() * len(s) / (366 if len(s) == 366 else 365))
    n = d.groupby("year").size()
    return [int(y) for y in cov.index if cov[y] >= min_coverage and n[y] >= 360]


# ---------------------------------------------------------------------------------------------
# climatology & anomalies
# ---------------------------------------------------------------------------------------------

def daily_col(daily: pd.DataFrame) -> str:
    """Column with single-day values: the most likely count if the table has one."""
    return "y_ml" if "y_ml" in daily.columns else "y"


class Climatology:
    """Day-of-year statistics pooled over +-``window`` days and all baseline years.

    Means come from ``y``; quantiles, the chance of fire and the log statistics from the
    single-day values (``daily_col``). With ``smooth`` everything uses the rolling mean of ``y``.
    """

    QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95)

    def __init__(self, daily: pd.DataFrame, baseline_years: list[int], window: int = 15, smooth: int = 0):
        self.window = window
        self.baseline_years = list(baseline_years)
        years, Y = year_matrix(daily)
        if smooth:
            Y = _rolling_rows(daily, smooth)
            Q = Y
        else:
            Q = year_matrix(daily, daily_col(daily))[1]
        sel = np.isin(years, baseline_years)
        Yb, Qb = Y[sel], Q[sel]
        offsets = np.arange(-window, window + 1)
        self.pools = []
        means = np.full(NMD, np.nan)
        for k in range(NMD):
            cols = (k + offsets) % NMD
            vals = Qb[:, cols].ravel()
            self.pools.append(np.sort(vals[~np.isnan(vals)]))
            mv = Yb[:, cols].ravel()
            mv = mv[~np.isnan(mv)]
            if len(mv):
                means[k] = mv.mean()
        self.mean = means
        self.q = {q: np.array([np.quantile(p, q) if len(p) else np.nan for p in self.pools]) for q in self.QUANTILES}
        self.p_fire = np.array([(p >= 0.5).mean() if len(p) else np.nan for p in self.pools])
        self.log_mean = np.array([np.log1p(p).mean() if len(p) else np.nan for p in self.pools])
        self.log_std = np.array([max(np.log1p(p).std(), 0.25) if len(p) else np.nan for p in self.pools])
        # mean of each baseline year's annual total (a robust "normal year")
        yb = Y[sel]
        self.annual_mean = float(np.nanmean(np.nansum(yb, axis=1))) if len(yb) else float("nan")

    def percentile(self, md: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Mid-rank percentile (0..100) of each value within its calendar day's pooled distribution."""
        out = np.full(len(values), np.nan)
        for i, (k, v) in enumerate(zip(md, values)):
            p = self.pools[k]
            if np.isnan(v) or not len(p):
                continue
            lo = np.searchsorted(p, v, side="left")
            hi = np.searchsorted(p, v, side="right")
            out[i] = 100.0 * (lo + hi) / (2 * len(p))
        return out

    def to_dict(self) -> dict:
        return {
            "mean": _clean(self.mean), "p_fire": _clean(self.p_fire),
            **{f"q{int(q * 100)}": _clean(v) for q, v in self.q.items()},
            "window": self.window, "baseline_years": self.baseline_years, "annual_mean": _f(self.annual_mean),
        }


def _rolling_rows(daily: pd.DataFrame, w: int) -> np.ndarray:
    s = daily.set_index("date")["y"]
    r = s.rolling(w, center=True, min_periods=max(1, w // 2)).mean()
    _, Y = year_matrix(pd.DataFrame({"date": r.index, "y": r.to_numpy()}))
    return Y


def add_anomalies(daily: pd.DataFrame, clim: Climatology, clim7: Climatology) -> pd.DataFrame:
    """Adds 7-day mean, ratio to normal, percentiles and anomaly classes to the daily table."""
    d = daily.copy()
    md = md_index(d["date"])
    one_day = d[daily_col(d)]
    d["y7"] = d["y"].rolling(7, center=True, min_periods=4).mean()
    d["normal"] = clim.mean[md]
    d["ratio"] = (d["y"] + 1.0) / (d["normal"] + 1.0)
    d["pct"] = clim.percentile(md, one_day.to_numpy())
    d["pct7"] = clim7.percentile(md, d["y7"].to_numpy())
    d["z"] = (np.log1p(one_day) - clim.log_mean[md]) / clim.log_std[md]
    bins = [-1, 10, 33.3, 66.7, 90, 97.5, 101]
    labels = ["much below", "below", "normal", "above", "much above", "extreme"]
    d["anomaly"] = pd.cut(d["pct7"], bins=bins, labels=labels).astype(object)
    return d


# ---------------------------------------------------------------------------------------------
# fire-season metrics
# ---------------------------------------------------------------------------------------------

def fire_year_start_month(clim: Climatology) -> int:
    """Calendar month (1-12) with the least climatological activity: the fire year starts there."""
    months = np.searchsorted(_LEAP_STARTS, np.arange(NMD), side="right")
    monthly = np.array([np.nanmean(clim.mean[months == m]) for m in range(1, 13)])
    return int(np.nanargmin(monthly)) + 1


def season_metrics(daily: pd.DataFrame, start_month: int, min_coverage: float = 0.9) -> pd.DataFrame:
    """Per fire year: total, onset (10%), peak, end (90%), length; plus anomalies vs the mean year."""
    s = daily.set_index("date")["y"]
    first = pd.Timestamp(s.index.min())
    last = pd.Timestamp(s.index.max())
    y0 = first.year if first.month <= start_month else first.year + 1
    rows = []
    for y in range(y0 - 1, last.year + 1):
        a = pd.Timestamp(year=y, month=start_month, day=1)
        b = pd.Timestamp(year=y + 1, month=start_month, day=1) - pd.Timedelta(days=1)
        seg = s[(s.index >= a) & (s.index <= b)]
        if len(seg) == 0:
            continue
        coverage = seg.notna().sum() / ((b - a).days + 1)
        if coverage < min_coverage:
            continue
        v = seg.fillna(0.0).to_numpy()
        total = float(v.sum())
        if total <= 0:
            continue
        cum = np.cumsum(v) / total
        i10, i50, i90 = (int(np.searchsorted(cum, q)) for q in (0.1, 0.5, 0.9))
        sm = pd.Series(v).rolling(15, center=True, min_periods=1).mean().to_numpy()
        ip = int(np.argmax(sm))
        # name the fire year after the calendar year holding most of its months
        major = y if start_month <= 7 else y + 1
        label = f"{major}" if start_month == 1 else \
            f"{major} ({MONTH_ABBR[start_month - 1]} {y} – {MONTH_ABBR[(start_month - 2) % 12]} {y + 1})"
        rows.append({
            "fire_year": label, "year": major, "start": a.date().isoformat(), "coverage": round(float(coverage), 3),
            "total": total, "onset": seg.index[i10].date().isoformat(), "center": seg.index[i50].date().isoformat(),
            "end": seg.index[i90].date().isoformat(), "peak": seg.index[ip].date().isoformat(),
            "peak_value": float(v[int(np.argmax(v))]), "peak_15d_mean": float(sm[ip]),
            "length_days": i90 - i10 + 1, "onset_off": i10, "peak_off": ip, "end_off": i90,
            "active_days": int((v >= 1).sum()),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ("total", "onset_off", "peak_off", "end_off", "length_days"):
        df[f"{col}_mean"] = df[col].mean()
    df["total_ratio"] = df["total"] / df["total_mean"]
    df["onset_shift_days"] = df["onset_off"] - df["onset_off_mean"]
    df["end_shift_days"] = df["end_off"] - df["end_off_mean"]
    df["length_anomaly_days"] = df["length_days"] - df["length_days_mean"]
    df["rank"] = df["total"].rank(ascending=False, method="min").astype(int)
    return df


# ---------------------------------------------------------------------------------------------
# critical periods (climatological)
# ---------------------------------------------------------------------------------------------

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Circular runs of True as (start, end) inclusive indices."""
    n = len(mask)
    if mask.all():
        return [(0, n - 1)]
    if not mask.any():
        return []
    shift = int(np.argmin(mask))  # start scanning at a False so runs don't wrap
    rolled = np.roll(mask, -shift)
    runs, start = [], None
    for i, v in enumerate(np.append(rolled, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append(((start + shift) % n, (i - 1 + shift) % n))
            start = None
    return runs


def critical_periods(clim: Climatology, shares=(0.5, 0.8), merge_gap: int = 10, min_len: int = 7) -> list[dict]:
    """Shortest sets of calendar days holding 50% / 80% of normal annual activity, as date ranges."""
    m = np.nan_to_num(pd.Series(np.tile(clim.mean, 3)).rolling(15, center=True, min_periods=1).mean()
                      .to_numpy()[NMD:2 * NMD])
    total = m.sum()
    if total <= 0:
        return []
    order = np.argsort(-m)
    out = []
    for share, level in zip(shares, ("peak", "high")):
        k = int(np.searchsorted(np.cumsum(m[order]) / total, share)) + 1
        mask = np.zeros(NMD, bool)
        mask[order[:k]] = True
        # close short gaps, then drop fragments
        for s, e in _runs(~mask):
            length = (e - s) % NMD + 1
            if length <= merge_gap:
                idx = (s + np.arange(length)) % NMD
                mask[idx] = True
        for s, e in _runs(mask):
            length = (e - s) % NMD + 1
            if length < min_len:
                continue
            idx = (s + np.arange(length)) % NMD
            out.append({"level": level, "target_share": share, "start_md": int(s), "end_md": int(e),
                        "start": md_label(s), "end": md_label(e), "days": int(length),
                        "share": float(m[idx].sum() / total)})
    return out


# ---------------------------------------------------------------------------------------------
# unusual events
# ---------------------------------------------------------------------------------------------

def detect_events(d: pd.DataFrame, clim: Climatology, pct_threshold: float = 95.0, max_gap: int = 2,
                  top: int = 40) -> list[dict]:
    """Runs of days whose 7-day activity is unusually high for the time of year.

    Requires ``add_anomalies`` columns. A minimum absolute level (a quarter of the mean daily
    activity) keeps a handful of detections in the quiet season from counting as an event.
    """
    mean_daily = clim.annual_mean / 365.0 if np.isfinite(clim.annual_mean) else 1.0
    min_abs = max(3.0, 0.25 * mean_daily)
    flag = (d["pct7"].to_numpy() >= pct_threshold) & (d["y7"].to_numpy() >= min_abs)
    idx = np.flatnonzero(np.nan_to_num(flag).astype(bool))
    if not len(idx):
        return []
    groups, cur = [], [idx[0]]
    for i in idx[1:]:
        if i - cur[-1] <= max_gap + 1:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    events = []
    md_all = md_index(d["date"])
    clim_q50 = clim.q[0.5]
    peak_season = clim.mean >= np.nanmean(clim.mean)
    for g in groups:
        a, b = g[0], g[-1]
        seg = d.iloc[a:b + 1]
        y = seg["y"].fillna(0).to_numpy()
        md = md_all[a:b + 1]
        excess = float(np.sum(np.maximum(y - clim_q50[md], 0)))
        ip = int(np.argmax(y))
        events.append({
            "start": seg["date"].iloc[0].date().isoformat(), "end": seg["date"].iloc[-1].date().isoformat(),
            "days": int(b - a + 1), "peak_date": seg["date"].iloc[ip].date().isoformat(),
            "peak_value": float(y[ip]), "total": float(y.sum()), "excess": excess,
            "normal_total": float(np.sum(clim.mean[md])),
            "max_pct7": float(np.nanmax(seg["pct7"])), "ratio": float((y.sum() + 1) / (np.sum(clim.mean[md]) + 1)),
            "out_of_season": bool(not peak_season[md_all[a + ip]]),
            "modeled": bool((seg["source"] >= 4).any()),
        })
    events.sort(key=lambda e: -e["excess"])
    return events[:top]


# ---------------------------------------------------------------------------------------------
# monthly table & payload
# ---------------------------------------------------------------------------------------------

def monthly_table(daily: pd.DataFrame, baseline_years: list[int]) -> pd.DataFrame:
    d = daily.assign(year=pd.DatetimeIndex(daily["date"]).year, month=pd.DatetimeIndex(daily["date"]).month)
    g = d.groupby(["year", "month"]).agg(total=("y", "sum"), days=("y", "count")).reset_index()
    g["n"] = pd.to_datetime(pd.DataFrame({"year": g["year"], "month": g["month"], "day": 1})).dt.days_in_month
    g.loc[g["days"] < 0.8 * g["n"], "total"] = np.nan  # incl. the current, unfinished month
    base = g[g["year"].isin(baseline_years)].groupby("month")["total"]
    g["normal"] = g["month"].map(base.median())  # median: a few catastrophic years must not define "normal"
    g["mean"] = g["month"].map(base.mean())
    g["ratio"] = (g["total"] + 1) / (g["normal"] + 1)
    g["rank"] = g.groupby("month")["total"].rank(ascending=False, method="min")
    g["years_ranked"] = g.groupby("month")["total"].transform("count")
    return g


def _f(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v)


def _clean(a) -> list:
    return [None if not np.isfinite(v) else round(float(v), 3) for v in np.asarray(a, dtype=float)]


def build_calendar(daily: pd.DataFrame, baseline_years: list[int] | None = None) -> dict:
    """Everything the calendar views need, JSON-ready."""
    baseline_years = baseline_years or complete_years(daily)
    clim = Climatology(daily, baseline_years)
    clim7 = Climatology(daily, baseline_years, smooth=7)
    d = add_anomalies(daily, clim, clim7)
    start_month = fire_year_start_month(clim)
    seasons = season_metrics(d, start_month)
    events = detect_events(d, clim)
    periods = critical_periods(clim)
    monthly = monthly_table(d, baseline_years)
    years, Y = year_matrix(d, daily_col(d))
    _, E = year_matrix(d)
    _, P = year_matrix(d, "pct7")
    _, S = year_matrix(d.assign(source=d["source"].astype(float)), "source")
    annual = d.assign(year=pd.DatetimeIndex(d["date"]).year).groupby("year").agg(
        total=("y", "sum"), days=("y", "count"), modeled_days=("source", lambda s: int((s >= 4).sum())))
    return {
        "baseline_years": baseline_years,
        "fire_year_start_month": start_month,
        "years": years.tolist(),
        "matrix": [_clean(r) for r in Y],      # single-day values (most likely count)
        "expected": [_clean(r) for r in E],    # expected values (differ on reconstructed days only)
        "pct7": [_clean(r) for r in P],
        "source": [[None if not np.isfinite(v) else int(v) for v in r] for r in S],
        "climatology": clim.to_dict(),
        # 7-day means (what the anomalies and the "normal year" chart compare against)
        "climatology7": {k: v for k, v in clim7.to_dict().items() if k != "p_fire"},
        "critical_periods": periods,
        "seasons": seasons.replace({np.nan: None}).to_dict(orient="records"),
        "events": events,
        "monthly": monthly.replace({np.nan: None}).to_dict(orient="records"),
        "annual": [{"year": int(y), "total": float(r.total), "days": int(r.days), "modeled_days": int(r.modeled_days)}
                   for y, r in annual.iterrows()],
    }, d, clim
