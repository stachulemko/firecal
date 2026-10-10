"""Offline tests on synthetic data (no network)."""
import numpy as np
import pandas as pd
import pytest

from firecal import calendar as cal
from firecal import forecast as fc
from firecal import harmonize as hz
from firecal.config import SENSOR_CODE
from firecal.fwi import fwi_system
from firecal.geo import AOI, Grid


# --------------------------------------------------------------------------------------------- geo
def test_grid_indexing_and_neighbors():
    aoi = AOI.from_bbox(-9.6, 36.9, -6.1, 42.2)
    g = Grid.for_aoi(aoi)
    assert g.res == 0.25 and g.ncells <= 1600
    lat, lon = g.centers()
    assert np.array_equal(g.cell_of(lat, lon), np.arange(g.ncells))
    assert g.cell_of(np.array([0.0]), np.array([0.0]))[0] == -1
    nb = g.neighbors(1)
    assert nb.shape == (g.ncells, 9)
    assert (nb[0] >= 0).sum() == 4  # corner cell sees itself + 3 neighbours


def test_aoi_id_is_stable():
    a = AOI.from_bbox(1.0, 2.0, 3.0, 4.0)
    b = AOI.from_bbox(1.0, 2.0, 3.0, 4.0)
    assert a.id == b.id
    with pytest.raises(ValueError):
        AOI.from_bbox(3.0, 2.0, 1.0, 4.0)


# ------------------------------------------------------------------------------------ harmonize
def _points(lat, lon, day, sensor, conf=1, typ=0):
    n = len(lat)
    return pd.DataFrame({
        "lat": np.asarray(lat, np.float32), "lon": np.asarray(lon, np.float32), "day": np.asarray(day, np.int32),
        "hhmm": np.full(n, 1330, np.int16), "sensor": np.full(n, sensor, np.int8), "conf": np.full(n, conf, np.int8),
        "frp": np.full(n, 10, np.float32), "is_day": np.ones(n, bool), "type": np.full(n, typ, np.int8),
        "scan": np.ones(n, np.float32), "track": np.ones(n, np.float32), "bt": np.full(n, 320, np.float32),
    })


def test_qa_filter_removes_low_confidence_and_static_sources():
    archive = _points([10.0, 10.0], [20.0, 20.5], [100, 100], 2, typ=2)  # static source at (10, 20)
    keys = hz.static_source_keys(archive)
    nrt = pd.concat([
        _points([10.0005], [20.0005], [200], 2, typ=-1),  # near the static source -> dropped
        _points([11.0], [21.0], [200], 2, typ=-1),        # unknown type, elsewhere -> kept
        _points([12.0], [22.0], [200], 2, conf=0),        # low confidence -> dropped
        _points([13.0], [23.0], [200], 2, typ=1),         # volcano -> dropped
    ], ignore_index=True)
    out = hz.qa_filter(nrt, keys)
    assert out["lat"].round(3).tolist() == [11.0]


def test_sensor_availability_detects_outage():
    days = np.arange(hz.DAY0, hz.DAY0 + 6000)
    rows = []
    for d in days:
        for key in ("VIIRS_SNPP", "VIIRS_NOAA20"):
            code = SENSOR_CODE[key]
            if d - hz.DAY0 < 4500:
                continue
            if key == "VIIRS_SNPP" and 5000 <= d - hz.DAY0 < 5010:  # 10-day S-NPP outage
                continue
            rows.append((d, code, 100))
    counts = pd.DataFrame(rows, columns=["day", "sensor", "n"])
    av = hz.sensor_availability(counts, hz.DAY0, len(days))
    v = av[SENSOR_CODE["VIIRS_SNPP"]]
    assert not v[5000:5010].any()
    assert v[4900:5000].all() and v[5010:5100].all()


def test_level_factor_rescales_to_ratio_level():
    ml = np.full(100, 10.0)
    ratio = np.full(100, 20.0)
    ml[:10] = np.nan
    f = hz.level_factor(ml, ratio, w=5)
    assert np.isnan(f[:10]).all()
    assert np.allclose(f[30:70] * 10, 20, rtol=0.02)  # ML level x factor = ratio level


def _synthetic_cube(seed=0, ncell_side=6, years=range(2008, 2023)):
    """MODIS and VIIRS points with VIIRS ~ 3x MODIS plus small fires only VIIRS sees."""
    rng = np.random.default_rng(seed)
    aoi = AOI.from_bbox(0.0, 0.0, ncell_side * 0.25, ncell_side * 0.25)
    grid = Grid.for_aoi(aoi, res=0.25)
    lat, lon = grid.centers()
    day0, day1 = hz.day_number("2008-01-01"), hz.day_number(f"{max(years)}-12-31")
    days = np.arange(day0, day1 + 1)
    doy = (hz.to_dates(days) - hz.to_dates(days).astype("datetime64[Y]")).astype(int)
    season = np.exp(-((doy - 220) / 30.0) ** 2)
    parts = []
    for c in range(grid.ncells):
        lam = 0.6 * season * (1 + rng.random())
        fires = rng.poisson(lam)
        m_t = rng.binomial(fires * 2, 0.5)
        m_a = rng.binomial(fires * 2, 0.5)
        v = rng.poisson(3 * fires + 0.2 * season)
        for code, cnt in ((SENSOR_CODE["MODIS_T"], m_t), (SENSOR_CODE["MODIS_A"], m_a), (SENSOR_CODE["VIIRS_SNPP"], v)):
            dd = np.repeat(days, cnt)
            if code == SENSOR_CODE["VIIRS_SNPP"]:
                dd = dd[dd >= hz.day_number("2012-01-20")]
            parts.append(_points(np.full(len(dd), lat[c]), np.full(len(dd), lon[c]), dd, code))
    pts = pd.concat(parts, ignore_index=True)
    counts = pts.groupby(["day", "sensor"]).size().rename("n").reset_index()
    av = hz.sensor_availability(counts, day0, len(days))
    return hz.build_cube(pts, grid, grid.cell_mask(aoi.geometry), day0, day1, av)


def test_transfer_model_recovers_viirs_from_modis():
    cube = _synthetic_cube()
    tm = hz.fit_transfer(cube, log_fn=lambda m: None)
    cv = tm.metrics["cv"]
    assert cv["harmonized"]["monthly"]["r_log"] > 0.9
    assert abs(cv["harmonized"]["daily"]["bias_pct"]) < 15
    assert cv["raw_modis"]["daily"]["bias_pct"] < -25  # T+A ~ 2/3 of VIIRS here: raw MODIS is too low
    h = hz.harmonize(cube, tm)
    yrs = pd.DatetimeIndex(h.daily["date"]).year
    pre = h.daily.loc[(yrs >= 2008) & (yrs <= 2011), "y"].sum() / 4
    post = h.daily.loc[(yrs >= 2013) & (yrs <= 2020), "y"].sum() / 8
    assert 0.7 < pre / post < 1.3  # stationary synthetic climate: eras must agree after harmonization


# ------------------------------------------------------------------------------------- calendar
def _daily(years=range(2001, 2021), peak=220, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31", freq="D")
    doy = dates.dayofyear.to_numpy()
    lam = 50 * np.exp(-((doy - peak) / 25.0) ** 2) + 0.5
    y = rng.poisson(lam).astype(float)
    return pd.DataFrame({"date": dates, "y": y, "source": np.ones(len(dates), np.int8)})


def test_md_index_aligns_leap_years():
    idx = cal.md_index(pd.to_datetime(["2020-02-29", "2021-03-01", "2020-03-01", "2021-12-31"]))
    assert idx.tolist() == [59, 60, 60, 365]
    assert cal.md_label(59) == "Feb 29"


def test_climatology_and_critical_periods():
    d = _daily()
    c = cal.Climatology(d, list(range(2001, 2021)))
    assert np.argmax(c.mean) in range(200, 240)
    p = cal.critical_periods(c)
    peak = [x for x in p if x["level"] == "peak"][0]
    assert 180 <= peak["start_md"] <= 215 and 225 <= peak["end_md"] <= 260
    assert 0.45 <= peak["share"] <= 0.6


def test_season_metrics_handle_year_wrap():
    d = _daily(peak=15)  # fire season around mid-January -> fire year must wrap the calendar year
    c = cal.Climatology(d, list(range(2001, 2021)))
    start = cal.fire_year_start_month(c)
    assert start in (6, 7, 8)
    s = cal.season_metrics(d, start)
    assert len(s) >= 18
    peaks = pd.to_datetime(s["peak"]).dt.month
    assert peaks.isin([12, 1, 2]).mean() > 0.9


def test_detect_events_finds_injected_extreme():
    d = _daily()
    sel = (d["date"] >= "2015-04-10") & (d["date"] <= "2015-04-16")
    d.loc[sel, "y"] = 400.0  # big out-of-season episode
    payload, dd, clim = cal.build_calendar(d)
    top = payload["events"][0]
    assert top["start"] <= "2015-04-10" <= top["end"] or top["start"] <= "2015-04-13" <= top["end"]
    assert top["out_of_season"]


# ------------------------------------------------------------------------------------------ FWI
def test_fwi_matches_reference_values():
    T = np.array([[17, 20, 8.5, 6.5, 13]], float)
    H = np.array([[42, 21, 40, 25, 34]], float)
    W = np.array([[25, 25, 17, 6, 24]], float)
    P = np.array([[0, 2.4, 0, 0, 0]], float)
    out = fwi_system(T, H, W, P, np.full(5, 4), np.array([46.0]))
    # Van Wagner & Pickett (1985) test data
    assert np.allclose(out["ffmc"][0], [87.7, 86.2, 87.0, 88.8, 89.1], atol=0.1)
    assert np.allclose(out["dmc"][0], [8.5, 10.4, 11.8, 13.2, 15.4], atol=0.1)
    assert np.allclose(out["dc"][0], [19.0, 23.6, 26.1, 28.2, 31.5], atol=0.1)
    assert np.allclose(out["fwi"][0], [10.1, 9.3, 7.6, 6.2, 14.8], atol=0.1)


# ------------------------------------------------------------------------------------- forecast
def test_trailing_sum_scales_for_missing_days():
    x = np.array([[1, 1, np.nan, 1, 1, 1, 1, 1]], float)
    s = fc._trailing_sum(x, 7)
    assert s[0, 6] == pytest.approx(7.0)  # 6 valid days of 1, scaled to 7
    assert s[0, 0] == pytest.approx(7.0)


def test_residual_model_quantiles_and_exceedance():
    rng = np.random.default_rng(0)
    n = 3000
    lead = rng.integers(1, 15, n)
    lmu = rng.uniform(0, 5, n)
    resid = rng.normal(0, 0.5, n)
    rm = fc.ResidualModel(lead, lmu, resid)
    q = rm.quantiles(np.array([np.expm1(3.0)]), 3, (0.1, 0.5, 0.9))
    assert q[0.1][0] < q[0.5][0] < q[0.9][0]
    assert np.log1p(q[0.5][0]) == pytest.approx(3.0, abs=0.15)
    p_hi = rm.exceed(np.array([np.expm1(3.0)]), 3, np.expm1(3.0 + 0.64))[0]  # ~ +1.28 sigma
    assert 0.05 < p_hi < 0.16


def test_quantile_map_gives_observed_share_of_quiet_days():
    rng = np.random.default_rng(1)
    md = np.tile(np.arange(365), 9)
    lam = 4 * np.exp(1.2 * np.sin(2 * np.pi * md / 365))       # seasonal activity, 1.2-13 a day
    obs = rng.poisson(lam * rng.gamma(0.3, 1 / 0.3, len(md)))  # bursty days, many without fire
    pred = lam * np.clip(1 + 0.2 * rng.standard_normal(len(md)), 0.3, None)  # smooth, like an expectation
    qm = hz.QuantileMap(pred, obs, md)
    out = qm.apply(pred, md)
    assert np.mean(obs < 0.5) > 0.25 and np.mean(pred < 0.5) < 0.05  # the expectation has too few quiet days
    # mapped values have about the observed share (pooling +-30 days of a steep seasonal cycle costs a few points)
    assert abs(np.mean(out < 0.5) - np.mean(obs < 0.5)) < 0.05
    y = qm.apply(np.array([0.1, 1.0, 5.0, 50.0, 500.0]), np.full(5, 100))
    assert np.all(np.diff(y) >= 0) and y[-1] > 100              # monotone; record days stay records
    gappy = pred.copy()
    gappy[100:140] = np.nan                                     # a sensor outage in the overlap years
    assert not np.isnan(hz.QuantileMap(gappy, obs, md).apply(pred, md)).any()


def test_fwi_danger_classes():
    assert [fc.fwi_class(v) for v in (0, 5.2, 11.1, 21.3, 37.9, 38.0, 49.9, 50.0, 120)] == [
        "very low", "low", "low", "high", "high", "very high", "very high", "extreme", "extreme"]


# ---------------------------------------------------------------------------------------- world
def test_select_countries_skips_huge_neighbour_that_barely_touches_the_grid():
    from shapely.geometry import box as sbox
    from firecal.geo import select_countries
    aoi = sbox(14, 49, 24, 54.4)                                  # "Poland", next to an exclave-like corner
    shapes = [("Poland", aoi),
              ("Germany", sbox(6, 47, 15, 55)),                   # big neighbour, a fair strip inside the grid
              ("Russia", sbox(19, 54.4, 180, 78)),                # huge, only an exclave-sized corner in the grid
              ("Faraway", sbox(100, 0, 110, 10))]
    need, skipped = select_countries(aoi, (13.75, 48.75, 24.25, 55.25), shapes)
    assert need == ["Germany", "Poland"] and skipped == ["Russia"]
    # a huge country inside the area itself is always needed
    need, _ = select_countries(sbox(30, 55, 40, 60), (29.75, 54.75, 40.25, 60.25), shapes)
    assert need == ["Russia"]


def test_world_core_drops_far_islands_and_keeps_one_side_of_the_antimeridian():
    from shapely.geometry import MultiPolygon, box as sbox
    from firecal.world import core
    mainland, far_isle, near_isle = sbox(-10, 37, -6, 42), sbox(-31, 37, -30.5, 37.5), sbox(-5, 39, -4.5, 39.5)
    c = core(MultiPolygon([mainland, far_isle, near_isle]))
    assert c.bounds == (-10.0, 37.0, -4.5, 42.0)                  # Azores-like island dropped, near one kept
    chukotka = core(MultiPolygon([sbox(160, 62, 180, 70), sbox(-180, 64, -172, 67)]))
    assert chukotka.bounds[0] >= 160                              # the small western side is dropped
