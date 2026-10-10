# FireCal: harmonized burning activity calendar

NASA Space Apps challenge: *build a web application that harmonizes satellite active-fire records into
a burning activity calendar for a selected area of interest, for early warning, science and land management.*

FireCal is a Python model and web app. It downloads every active-fire hotspot NASA FIRMS has for an area
(MODIS Terra/Aqua since 2000, VIIRS S-NPP since 2012, NOAA-20 since 2018, NOAA-21 since 2024). It converts
them into **one consistent daily record in VIIRS S-NPP-equivalent detections**, and then builds:

* **a calendar**: year × day heat map, raw or as anomaly percentiles, plus a month-by-month view;
* **the normal pattern**: day-of-year climatology with percentile bands, fire-season onset, peak and end for
  every year, and the *critical periods*, i.e. the fewest calendar days that hold 50% and 80% of normal activity;
* **unusual episodes**: runs of days above the 95th percentile for the time of year, ranked, including
  out-of-season ones;
* **forecasts**:
  * daily fire probability per grid cell and area totals for **1 to 14 days ahead**, with calibrated 80% ranges
    and the chance of a *critical day*, next to the area's Fire Weather Index and its danger class;
  * a **1 to 3 month outlook**;
  * an out-of-sample **backtest from any past date**.

## Quick start

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/firecal run --preset portugal --backtest 2024-09-12   # CLI: analysis + backtest
.venv/bin/firecal serve --port 8000                              # web app on http://127.0.0.1:8000
.venv/bin/python -m pytest                                       # offline tests (synthetic data)
```

Optional: put a free FIRMS key (<https://firms.modaps.eosdis.nasa.gov/api/map_key/>) in `.env` as
`FIRMS_MAP_KEY=...`. Without it, the public yearly archive ends with the last complete year and the keyless
near-real-time feed only covers the last 7 days, so the period in between is shown as *no data*. With the key,
the FIRMS area API fills that gap. This path is implemented but could not be tested here without a key.

**Any place on Earth.** Pick any of the 233 countries and territories with FIRMS data, or one of the 294 states and
provinces of the nine largest countries (USA, Russia, China, Canada, Brazil, India, Indonesia, Australia, South
Africa); search them by name, click them on the map, or draw a rectangle. The map can also show the whole world:
VIIRS fires of the last 7 days on a 1° grid, from the keyless near-real-time feeds. The USA, Russia, China, Canada
and Brazil are too large for one analysis grid and are analysed by region.

The first analysis of a new country downloads its yearly FIRMS archives: Poland, Portugal or Greece take about a
minute, the most fire-prone countries (Brazil, DR Congo, Angola, Russia) 2–3 GB. Neighbouring countries are
downloaded only when they cover a useful part of the grid: Poland skips Russia, whose Kaliningrad exclave would
cost 2 GB of Siberian fires. Everything is cached as Parquet under `data/`, so later runs take 1–3 minutes, most
of it model training. Analysed areas stay listed under "Your areas" and open instantly.

A single analysis of the whole Earth is not offered: the archives of all countries are tens of GB, and the
calendar is meant for an area of interest.

## Data sources

| Source | What | Access |
|---|---|---|
| NASA FIRMS country archive | MODIS C6.1 (2000–), VIIRS S-NPP (2012–), NOAA-20 (2018–) standard processing | public, no key |
| NASA FIRMS NRT feeds | MODIS, S-NPP, NOAA-20, NOAA-21, last 7 days | public, no key |
| NASA FIRMS area API | everything after the archive | free MAP_KEY |
| NASA POWER | daily MERRA-2 weather since 2000 (temperature, humidity, dew point, rain, wind, radiation, soil wetness) | public |
| Open-Meteo | last 45 days + 16-day weather forecast | public (non-commercial) |
| NOAA PSL / CPC | Oceanic Niño Index (ENSO) | public |
| Natural Earth | country borders, used to find the FIRMS archive files covering an area | public |

## The model

### 1. Harmonization (`firecal/harmonize.py`)

* **Quality control.**
  * Drops low-confidence detections, volcanoes, static industrial sources and offshore flares.
  * The NRT feeds have no `type` column, so static sources are matched against locations the archive flags.
* **Sensor outage detection.** "No data" must never be read as "no fire". A sensor is flagged as down when either:
  * it reports nothing while a same-instrument sensor (rescaled) implies at least 15 detections, or
  * it is silent for 7+ days where its own climatology expects at least 100.

  It finds the known gaps: Terra in June–July 2001 and the S-NPP safe-hold from 27 July 2022.
* **MODIS → VIIRS transfer model.**
  * A LightGBM Poisson model at grid-cell × day level.
  * Trained on the **stable-orbit overlap 2012–2020**, because Terra and Aqua drift after that.
  * Inputs: Terra/Aqua day and night counts, FRP, pixel footprint, ±1/3/7-day and neighbouring-cell context, and
    the cell's MODIS fire regime.
  * A Terra-only variant covers Nov 2000 – Jul 2002.
  * Its daily pattern is rescaled to the calendar-month VIIRS/MODIS ratio over ±15 days. This *ML + level*
    combination beats both parts alone.
* **Assembly.**
  * Observed S-NPP wherever it is available.
  * NOAA-20/21 rescaled to S-NPP during S-NPP gaps.
  * The transfer model for the MODIS-only era, with empirical uncertainty bands.
* **Expected value vs most likely day.** The transfer model estimates the *expected* S-NPP count. Summed over
  ~150 cells, the small chances of fires MODIS cannot see add up to about one detection on days VIIRS sees none,
  so the reconstruction almost never says "no fire" (7% of days against 35% observed in Portugal). That is right
  for totals and wrong for single days, and it showed as a visible seam at 2012 in the calendar.
  * A season-aware **quantile mapping** (±30 calendar days, fitted on out-of-sample predictions) turns the
    expected value into the **most likely count of the day**, with the observed share of quiet days and extremes.
  * The most likely count feeds the heat map and daily statistics (daily quantiles, chance of a day with fire,
    critical-day thresholds). The expected value feeds every total: months, years, seasons, 7-day anomalies and
    episodes, where it is more accurate.
  * Matching the observed distribution cannot tell *which* quiet days had a small fire, so the most likely
    series is worse for monthly totals in quiet seasons (Paraná Delta: monthly r 0.83 against 0.92). That is why
    both are kept, each where it is more accurate.

Accuracy on held-out years: 3-fold cross-validation over whole years (2012–2020) for the transfer model, and
leave-one-year-out for the quantile mapping:

| Area | series | daily r | monthly r (log) | bias | days without fire (observed) |
|---|---|---|---|---|---|
| Portugal | expected value | 0.927 | 0.972 | −1.9% | 7% (35%) |
| Portugal | most likely day | 0.925 | 0.952 | −2.1% | 35% (35%) |
| Portugal | raw MODIS | 0.827 | 0.973 | −62.6% | 55% (35%) |
| Paraná Delta (Argentina) | expected value | 0.934 | 0.916 | −5.7% | 3% (34%) |
| Paraná Delta (Argentina) | most likely day | 0.915 | 0.829 | −2.6% | 35% (34%) |
| Paraná Delta (Argentina) | raw MODIS | 0.920 | 0.879 | −75.6% | 53% (34%) |
| Poland | expected value | 0.807 | 0.889 | −0.7% | 21% (34%) |
| Poland | most likely day | 0.758 | 0.774 | +7.5% | 34% (34%) |
| Poland | raw MODIS | 0.810 | 0.902 | −73.7% | 68% (34%) |

Rejected along the way, with measurements:
* Features derived from the target sensor leaked training-year targets (−29% bias out of sample).
* Pure ratio scaling had lower daily skill (r 0.85) and found fewer extreme days (73% vs 82%).

### 2. Calendar (`firecal/calendar.py`)

* **Climatology.** Pooled over ±15 calendar days and all complete years (2001 to the last complete year).
  Daily quantiles use the most likely daily counts, means use expected values. The "normal year" chart compares
  7-day means with the climatology of 7-day means.
* **Anomalies.** Percentiles of 7-day activity.
* **Fire year.** Starts at the quietest month, so seasons that span New Year (southern hemisphere, tropics) stay whole.
* **Season metrics.** Onset is the date 10% of the year's activity had burned, end is the 90% date, and peak is
  the highest 15-day mean.

### 3. Forecast (`firecal/forecast.py`, `firecal/fwi.py`)

* **Two LightGBM models per cell × target day:**
  * a classifier for P(≥1 detection);
  * a Poisson regressor for the expected count.
* **Inputs:**
  * **multi-sensor recent burning:** VIIRS consensus of S-NPP/NOAA-20/NOAA-21, MODIS, FRP, neighbourhood and area totals;
  * **harmonized climatology** of the cell and calendar day;
  * **weather on the target day:** NASA POWER in training, bias-corrected Open-Meteo forecasts in operation;
  * the **Canadian Fire Weather Index system** (FFMC, DMC, DC, ISI, BUI, FWI), verified against the
    Van Wagner & Pickett reference data;
  * **drought:** rain deficits, days since rain, soil wetness anomalies;
  * **ENSO**;
  * **lead time**.
* **Area totals.** The sum of cell expectations. The median, 80% range and the probability of a critical day
  (above the 90th percentile for the date) come from the empirical out-of-sample residual distribution, by lead
  time and magnitude. A negative binomial was tried and covered 98% instead of 80%.
* **Fire weather.** Every forecast day also shows the weather the model used, averaged where the area usually
  burns: the Fire Weather Index with its EFFIS danger class and its anomaly for the date, ISI, temperature,
  humidity, wind and rain.
* **Training length.** Early stopping on a single validation year can stop the count model after a few dozen trees
  (Sierras de Córdoba: 36, after a moderate 2022). A minimum of 150 trees raised the critical-day AUC there from
  0.72 to 0.75 on the test years and left Portugal and the Paraná Delta unchanged.
* **Seasonal outlook.** Ridge regression on monthly anomalies, with leave-one-year-out skill reported honestly;
  skill is low for Mediterranean regions.

Rejected along the way, with measurements on the test years:
* An area-level second model (critical-day classifier and quantile regression on area fire weather): critical-day
  AUC 0.937 → 0.935 in Portugal, Brier score 10% better in the Paraná Delta, but area medians worse in all three
  test regions.
* Longer, more regularized training (400 trees): better day ranking, but in Córdoba the cell expectations on
  unseen years summed to about half of what burned (0.9 with early stopping): the model starts to memorize where
  the training years burned.

Test-year skill on **2023–2024, never used for training or calibration**:

| Area | lead | cell AUC (climatology) | critical-day AUC | skill vs climatology | 80% range hit |
|---|---|---|---|---|---|
| Poland | 1 d | 0.864 (0.815) | 0.75 | +0.28 | 0.76 |
| Poland | 7 d | 0.863 | 0.77 | +0.28 | 0.76 |
| Portugal | 1 d | 0.828 (0.743) | 0.95 | +0.40 | 0.76 |
| Portugal | 7 d | 0.820 | 0.93 | +0.37 | 0.77 |
| Paraná Delta | 1 d | 0.861 (0.763) | 0.98 | +0.44 | 0.89 |
| Paraná Delta | 14 d | 0.853 | 0.99 | +0.43 | 0.91 |

Skill is the reduction of the typical log error of the area total against the climatological daily median. These
scores use observed weather on the target day, so they are an upper bound. The app also reports a lower bound
with no weather forecast: skill +0.34 at 1 day, falling to about 0 at 7–14 days. Real skill sits between the
two, depending on how good the weather forecast is.

## Code layout

```
firecal/
  config.py        sensors, paths, constants
  geo.py           AOI, Natural Earth -> FIRMS country files, analysis grid, which countries to download
  world.py         every country and region of the world as an area; fires on Earth in the last 7 days
  sources/         firms.py (archive / NRT / API), weather.py (POWER, Open-Meteo, ONI), http.py
  harmonize.py     QA, outage detection, gridding, MODIS->VIIRS transfer model, assembly
  calendar.py      climatology, anomalies, seasons, critical periods, events
  fwi.py           Canadian Fire Weather Index system (vectorized)
  forecast.py      daily forecaster, residual calibration, seasonal outlook
  pipeline.py      end-to-end analysis, presets, caching, backtests
  cli.py           `firecal run | presets | serve`
app/
  main.py          FastAPI: jobs, results, backtests, CSV export
  static/          single-page front end (Leaflet + dependency-free SVG/canvas charts)
tests/             offline tests on synthetic data
```

## Limits

* Hotspots are thermal anomalies at overpass time, not burned area. Clouds, smoke and canopy hide fires.
* The pre-2012 record is reconstructed. The small fires MODIS cannot see are estimated from each cell's regime;
  on a single quiet day the most likely count is a best guess, not an observation.
* Forecast skill degrades with weather forecast error, mostly beyond 5–7 days. Extreme magnitudes, like
  Portugal in September 2024, are flagged as elevated risk but under-predicted in size.

Credits: NASA FIRMS (LANCE/EOSDIS), NASA POWER, Open-Meteo (CC BY 4.0), NOAA PSL/CPC, Natural Earth,
OpenStreetMap.
