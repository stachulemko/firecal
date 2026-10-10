"""Canadian Forest Fire Weather Index (FWI) System, vectorized over weather points.

Van Wagner (1987) equations as implemented in the cffdrs reference code (Wang et al. 2015), with
latitude-dependent day-length factors. Inputs are daily values standing in for local noon:
temperature (deg C), relative humidity (%), wind speed (km/h) and 24-h precipitation (mm).
The codes carry fuel-moisture memory: FFMC (hours), DMC (~2 weeks), DC (~2 months of drought).
"""
from __future__ import annotations

import numpy as np

_ELL = {  # DMC effective day length by latitude band
    "n33": [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0],
    "n15": [7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8],
    "s15": [10.1, 9.6, 9.1, 8.5, 8.1, 7.8, 7.9, 8.3, 8.9, 9.4, 9.9, 10.2],
    "s30": [11.5, 10.5, 9.2, 7.9, 6.8, 6.2, 6.5, 7.4, 8.7, 10.0, 11.2, 11.8],
}
_FL_N = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]
_FL_S = [6.4, 5.0, 2.4, 0.4, -1.6, -1.6, -1.6, -1.6, -1.6, 0.9, 3.8, 5.8]


def _day_length_tables(lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(npoint, 12) DMC day-length and DC day-length adjustment factors."""
    ell = np.empty((len(lat), 12))
    fl = np.empty((len(lat), 12))
    for i, la in enumerate(lat):
        if la > 33:
            ell[i] = _ELL["n33"]
        elif la > 15:
            ell[i] = _ELL["n15"]
        elif la >= -15:
            ell[i] = 9.0
        elif la >= -30:
            ell[i] = _ELL["s15"]
        else:
            ell[i] = _ELL["s30"]
        fl[i] = _FL_N if la > 20 else (_FL_S if la <= -20 else 1.4)
    return ell, fl


def fwi_system(temp: np.ndarray, rh: np.ndarray, wind: np.ndarray, rain: np.ndarray, month: np.ndarray,
               lat: np.ndarray, ffmc0: float = 85.0, dmc0: float = 6.0, dc0: float = 15.0) -> dict[str, np.ndarray]:
    """Daily FWI components for arrays shaped (npoint, ndays); ``month`` is 1..12 per day."""
    temp = np.nan_to_num(np.asarray(temp, float), nan=15.0)
    rh = np.clip(np.nan_to_num(np.asarray(rh, float), nan=60.0), 0.0, 100.0)
    wind = np.maximum(np.nan_to_num(np.asarray(wind, float), nan=10.0), 0.0)
    rain = np.maximum(np.nan_to_num(np.asarray(rain, float), nan=0.0), 0.0)
    npnt, nd = temp.shape
    ell, fl = _day_length_tables(np.asarray(lat, float))
    out = {k: np.empty((npnt, nd), np.float32) for k in ("ffmc", "dmc", "dc", "isi", "bui", "fwi")}
    ffmc = np.full(npnt, ffmc0)
    dmc = np.full(npnt, dmc0)
    dc = np.full(npnt, dc0)
    rows = np.arange(npnt)
    for t in range(nd):
        T, H, W, P = temp[:, t], rh[:, t], wind[:, t], rain[:, t]
        m = int(month[t]) - 1
        # --- FFMC
        mo = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
        wet = P > 0.5
        rf = np.where(wet, P - 0.5, 1.0)
        add = 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf))
        add = np.where(mo > 150.0, add + 0.0015 * (mo - 150.0) ** 2 * np.sqrt(rf), add)
        mo = np.where(wet, np.minimum(mo + add, 250.0), mo)
        ed = 0.942 * H ** 0.679 + 11.0 * np.exp((H - 100.0) / 10.0) + 0.18 * (21.1 - T) * (1.0 - np.exp(-0.115 * H))
        ew = 0.618 * H ** 0.753 + 10.0 * np.exp((H - 100.0) / 10.0) + 0.18 * (21.1 - T) * (1.0 - np.exp(-0.115 * H))
        ko = 0.424 * (1.0 - (H / 100.0) ** 1.7) + 0.0694 * np.sqrt(W) * (1.0 - (H / 100.0) ** 8)
        kd = ko * 0.581 * np.exp(0.0365 * T)
        k1 = 0.424 * (1.0 - ((100.0 - H) / 100.0) ** 1.7) + 0.0694 * np.sqrt(W) * (1.0 - ((100.0 - H) / 100.0) ** 8)
        kw = k1 * 0.581 * np.exp(0.0365 * T)
        mm = np.where(mo > ed, ed + (mo - ed) * 10.0 ** (-kd), np.where(mo < ew, ew - (ew - mo) * 10.0 ** (-kw), mo))
        ffmc = np.clip(59.5 * (250.0 - mm) / (147.2 + mm), 0.0, 101.0)
        # --- DMC
        tt = np.maximum(T, -1.1)
        rk = 1.894 * (tt + 1.1) * (100.0 - H) * ell[rows, m] * 1e-4
        wet = P > 1.5
        re = 0.92 * P - 1.27
        mo_d = 20.0 + np.exp(5.6348 - dmc / 43.43)
        b = np.where(dmc <= 33.0, 100.0 / (0.5 + 0.3 * dmc),
                     np.where(dmc <= 65.0, 14.0 - 1.3 * np.log(np.maximum(dmc, 1e-6)), 6.2 * np.log(np.maximum(dmc, 1e-6)) - 17.2))
        mr = mo_d + 1000.0 * re / (48.77 + b * re)
        pr = np.maximum(244.72 - 43.43 * np.log(np.maximum(mr - 20.0, 1e-6)), 0.0)
        dmc = np.maximum(np.where(wet, pr, dmc) + rk, 0.0)
        # --- DC
        td = np.maximum(T, -2.8)
        pe = np.maximum((0.36 * (td + 2.8) + fl[rows, m]) / 2.0, 0.0)
        wet = P > 2.8
        rd = 0.83 * P - 1.27
        qr = 800.0 * np.exp(-dc / 400.0) + 3.937 * rd
        dr = np.maximum(400.0 * np.log(800.0 / np.maximum(qr, 1e-6)), 0.0)
        dc = np.where(wet, dr, dc) + pe
        # --- ISI, BUI, FWI
        fm = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
        sf = 19.115 * np.exp(-0.1386 * fm) * (1.0 + fm ** 5.31 / 4.93e7)
        isi = sf * np.exp(0.05039 * W)
        bui = np.where(dmc <= 0.4 * dc, 0.8 * dc * dmc / np.maximum(dmc + 0.4 * dc, 1e-6),
                       dmc - (1.0 - 0.8 * dc / np.maximum(dmc + 0.4 * dc, 1e-6)) * (0.92 + (0.0114 * dmc) ** 1.7))
        bui = np.maximum(bui, 0.0)
        fd = np.where(bui <= 80.0, 0.626 * bui ** 0.809 + 2.0, 1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui)))
        bb = 0.1 * isi * fd
        fwi = np.where(bb > 1.0, np.exp(2.72 * (0.434 * np.log(np.maximum(bb, 1.0))) ** 0.647), bb)
        for k, v in (("ffmc", ffmc), ("dmc", dmc), ("dc", dc), ("isi", isi), ("bui", bui), ("fwi", fwi)):
            out[k][:, t] = v
    return out
