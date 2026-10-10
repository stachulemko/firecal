import { css, fmt, mdLabel, MONTH_NAMES, MONTH_STARTS, tooltip, seriesChart, columnChart, scatterLog,
  rampColor, heatStops, divStops } from "/static/charts.js";

const $ = (s) => document.querySelector(s);
const state = { presets: [], data: null, aoiId: null, year: null, calMode: "activity", event: null,
  forecast: null, live: null, mapLayer: "mean_annual", lead: 1, drawing: false };

const SOURCE = { 0: "no data", 1: "VIIRS S-NPP, observed", 2: "VIIRS NOAA-20, rescaled to S-NPP",
  3: "VIIRS NOAA-21, rescaled to S-NPP", 4: "reconstructed from MODIS Terra + Aqua", 5: "reconstructed from MODIS Terra" };
const FEATURE_NAMES = {
  clim_c: "normal activity in the cell (day of year)", clim_nb: "normal activity nearby", pfire_c: "normal chance of fire in the cell",
  clim_aoi: "normal activity of the area", nb_1: "fires nearby yesterday", nb_7: "fires nearby, last 7 days",
  nb5_7: "fires within ~2 cells, last 7 days", y0: "fires in the cell on the issue day", y1: "fires in the cell 1 day before",
  y2: "fires in the cell 2 days before", s7: "fires in the cell, last 7 days", dsf: "days since the cell last burned",
  aoi_1: "area activity on the issue day", aoi_7: "area activity, last 7 days", cell_anom7: "cell activity vs normal, 7 days",
  aoi_anom7: "area activity vs normal, 7 days", viirs0: "VIIRS consensus (S-NPP+NOAA-20/21)", viirs_nb: "VIIRS consensus nearby",
  frp0: "fire radiative power", frp7: "fire radiative power, 7 days", modis0: "MODIS Terra+Aqua detections",
  tmax: "max temperature", rh: "relative humidity", vpd: "vapour pressure deficit", wind: "wind speed", rain: "rain",
  solar: "solar radiation", rain7: "rain, last 7 days", rain30: "rain, last 30 days", dsr: "days since rain",
  tmax_anom: "temperature anomaly", vpd_anom: "dryness (VPD) anomaly", ffmc: "FWI: fine fuel moisture code",
  dmc: "FWI: duff moisture code", dc: "FWI: drought code", isi: "FWI: initial spread index", bui: "FWI: buildup index",
  fwi: "Fire Weather Index", fwi_anom: "Fire Weather Index anomaly", rain90_anom: "rain deficit, 90 days",
  rain30_anom: "rain deficit, 30 days", gwet_top: "topsoil wetness", gwet_root: "root-zone soil wetness",
  gwet_root_anom: "soil wetness anomaly", doy_sin: "season", doy_cos: "season", lat: "latitude", lon: "longitude",
  lead: "days ahead", oni: "El Niño / La Niña (ONI)",
};

// ---------------------------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------------------------
async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch { /* keep status text */ }
    throw new Error(msg);
  }
  return r.json();
}

// ---------------------------------------------------------------------------------------------
// map
// ---------------------------------------------------------------------------------------------
let map, aoiLayer, cellLayer, hotLayer, drawRect, canvasRenderer;
function initMap() {
  map = L.map("map", { zoomSnap: 0.25, worldCopyJump: true }).setView([20, 0], 2);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 12, attribution: "© OpenStreetMap contributors",
  }).addTo(map);
  canvasRenderer = L.canvas({ padding: 0.3 });
  let start = null;
  map.on("mousedown", (e) => {
    if (!state.drawing) return;
    start = e.latlng;
    if (drawRect) drawRect.remove();
    drawRect = L.rectangle([start, start], { color: css("--accent"), weight: 2, fill: false }).addTo(map);
  });
  map.on("mousemove", (e) => { if (state.drawing && start) drawRect.setBounds([start, e.latlng]); });
  let justDrew = false;
  map.on("mouseup", (e) => {
    if (!state.drawing || !start) return;
    const b = L.latLngBounds(start, e.latlng);
    start = null;
    setDrawing(false);
    justDrew = true;
    setTimeout(() => { justDrew = false; }, 300);
    setBBox([b.getWest(), b.getSouth(), b.getEast(), b.getNorth()]);
    $("#preset-select").value = "custom";
  });
  // click anywhere on Earth: select the country (or the region of a large country) under the pointer
  map.on("click", async (e) => {
    if (state.drawing || justDrew) return;
    const { lat, lng } = e.latlng.wrap();
    const hit = await api(`/api/area_at?lat=${lat.toFixed(4)}&lon=${lng.toFixed(4)}`).catch(() => null);
    if (hit) selectArea(hit.key, hit.fits ? null : `${hit.name} is too large for one analysis: pick one of its regions.`);
  });
}
function setDrawing(on) {
  state.drawing = on;
  $("#draw-btn").classList.toggle("active", on);
  $("#draw-btn").textContent = on ? "Drag on the map…" : "Draw rectangle";
  if (on) map.dragging.disable(); else map.dragging.enable();
  map.getContainer().style.cursor = on ? "crosshair" : "";
}
function setBBox(b) {
  ["w", "s", "e", "n"].forEach((k, i) => { $(`#bb-${k}`).value = b[i].toFixed(2); });
}
function showAoi(geojson, fit = true) {
  if (aoiLayer) aoiLayer.remove();
  aoiLayer = L.geoJSON(geojson, { style: { color: css("--ink"), weight: 1.5, fill: false } }).addTo(map);
  if (fit) map.fitBounds(aoiLayer.getBounds(), { padding: [20, 20] });
}

function cellValues() {
  const c = state.data.cells;
  if (state.mapLayer === "risk") {
    const fc = state.forecast || state.live;
    const day = fc?.cells?.find((d) => d.lead === state.lead);
    return { values: day ? day.p_fire : null, scale: "linear", max: Math.max(0.3, ...(day?.p_fire || [0])), unit: "chance of ≥1 detection", pct: true };
  }
  if (state.mapLayer === "burn_frequency") return { values: c.burn_frequency, scale: "linear", max: 1, unit: "share of years with fire", pct: true };
  if (state.mapLayer === "recent7") return { values: c.recent7, scale: "log", max: Math.max(1, ...(c.recent7 || [0])), unit: "detections, last 7 days" };
  return { values: c.mean_annual, scale: "log", max: Math.max(1, ...c.mean_annual), unit: "detections per normal year" };
}

function renderMap() {
  if (cellLayer) cellLayer.remove();
  if (hotLayer) hotLayer.remove();
  cellLayer = hotLayer = null;
  if (state.mapLayer === "world") { renderWorldLayer(); return; }
  if (!state.data) return;
  const c = state.data.cells, h = c.res / 2;
  const { values, scale, max, unit, pct } = cellValues();
  const stops = heatStops();
  cellLayer = L.layerGroup();
  const tip = (i, v) => `${pct ? fmt.pct(v) : fmt.num(v, 1)} · ${unit}`;
  if (values) {
    for (let i = 0; i < c.lat.length; i++) {
      const v = values[i];
      if (v == null || v <= 0) continue;
      const t = scale === "log" ? Math.log1p(v) / Math.log1p(max) : v / max;
      const r = L.rectangle([[c.lat[i] - h, c.lon[i] - h], [c.lat[i] + h, c.lon[i] + h]], {
        renderer: canvasRenderer, stroke: false, fillColor: rampColor(stops, 0.08 + 0.92 * t),
        fillOpacity: c.in_aoi[i] ? 0.8 : 0.3,
      });
      r.bindTooltip(tip(i, v), { sticky: true });
      cellLayer.addLayer(r);
    }
  }
  cellLayer.addTo(map);
  if ($("#show-hotspots").checked && state.data.hotspots?.lat?.length) {
    hotLayer = L.layerGroup();
    const hs = state.data.hotspots;
    for (let i = 0; i < hs.lat.length; i++) {
      const m = L.circleMarker([hs.lat[i], hs.lon[i]], { renderer: canvasRenderer, radius: 3, weight: 1,
        color: css("--surface"), fillColor: css("--ink"), fillOpacity: 0.9 });
      m.bindTooltip(`${hs.date[i]} · ${hs.sensor[i].replace("_", " ")} · FRP ${hs.frp[i]} MW`);
      hotLayer.addLayer(m);
    }
    hotLayer.addTo(map);
  }
  aoiLayer?.bringToFront();
  const lg = $("#map-legend");
  lg.replaceChildren();
  const lab = document.createElement("span");
  lab.textContent = unit;
  const ramp = document.createElement("span");
  ramp.className = "ramp";
  ramp.style.background = `linear-gradient(90deg, ${stops.join(",")})`;
  const lo = document.createElement("span"); lo.textContent = pct ? "0%" : "0";
  const hi = document.createElement("span"); hi.textContent = pct ? fmt.pct(max) : fmt.compact(max);
  lg.append(lab, lo, ramp, hi);
  if (scale === "log") { const n = document.createElement("span"); n.textContent = "(log scale)"; lg.append(n); }
  if (hotLayer) { const n = document.createElement("span"); n.className = "sw"; n.innerHTML = "<i style='border-radius:50%;background:var(--ink)'></i>"; n.append("hotspot, last 7 days"); lg.append(n); }
}

async function renderWorldLayer() {
  const token = (state.mapToken = (state.mapToken || 0) + 1);
  const lg = $("#map-legend");
  if (!state.world) {
    lg.textContent = "Loading fires of the last 7 days on Earth…";
    try { state.world = await api("/api/world/recent"); } catch (e) { lg.textContent = `World layer unavailable: ${e.message}`; return; }
  }
  if (state.mapLayer !== "world" || token !== state.mapToken) return;  // switched away while loading
  const w = state.world, h = w.res / 2, stops = heatStops();
  const sorted = w.n.slice().sort((a, b) => a - b), vmax = Math.max(10, sorted[Math.floor(sorted.length * 0.995)] || 1);
  cellLayer = L.layerGroup();
  for (let i = 0; i < w.lat.length; i++) {
    const t = Math.min(1, Math.log1p(w.n[i]) / Math.log1p(vmax));
    const r = L.rectangle([[w.lat[i] - h, w.lon[i] - h], [w.lat[i] + h, w.lon[i] + h]], {
      renderer: canvasRenderer, stroke: false, fillColor: rampColor(stops, 0.12 + 0.88 * t), fillOpacity: 0.85 });
    r.bindTooltip(`${fmt.int(w.n[i])} VIIRS detections · ${w.first} – ${w.last} · click to select this area`, { sticky: true });
    cellLayer.addLayer(r);
  }
  cellLayer.addTo(map);
  aoiLayer?.bringToFront();
  lg.replaceChildren();
  const lab = document.createElement("span");
  lab.textContent = `VIIRS detections per 1° cell, ${w.first} – ${w.last} (${fmt.compact(w.total)} on Earth)`;
  const lo = document.createElement("span"); lo.textContent = "1";
  const ramp = document.createElement("span"); ramp.className = "ramp";
  ramp.style.background = `linear-gradient(90deg, ${stops.join(",")})`;
  const hi = document.createElement("span"); hi.textContent = `${fmt.compact(vmax)}+ (log scale) · includes industrial heat sources`;
  lg.append(lab, lo, ramp, hi);
}

// ---------------------------------------------------------------------------------------------
// presets, areas of the world & analysis jobs
// ---------------------------------------------------------------------------------------------
async function loadPresets() {
  const [presets, analyses] = await Promise.all([api("/api/presets"), api("/api/analyses").catch(() => [])]);
  state.presets = presets;
  const presetIds = new Set(presets.map((p) => p.id));
  const sel = $("#preset-select");
  sel.replaceChildren();
  const option = (value, text, parent) => {
    const o = document.createElement("option");
    o.value = value; o.textContent = text;
    parent.appendChild(o);
  };
  const examples = document.createElement("optgroup");
  examples.label = "Example areas";
  for (const p of presets) option(p.key, p.name + (p.cached ? " ✓" : ""), examples);
  sel.appendChild(examples);
  const own = analyses.filter((a) => !presetIds.has(a.id));
  if (own.length) {
    const g = document.createElement("optgroup");
    g.label = "Your areas (analysed)";
    for (const a of own) option(`aoi:${a.id}`, a.name, g);
    sel.appendChild(g);
  }
  if (!state.areas) state.areas = await api("/api/areas").catch(() => []);
  const byGroup = new Map();
  for (const a of state.areas) {
    if (!byGroup.has(a.group)) byGroup.set(a.group, []);
    byGroup.get(a.group).push(a);
  }
  const groups = [...byGroup.keys()].sort((x, y) => (x.includes(": regions") - y.includes(": regions")) || x.localeCompare(y));
  for (const name of groups) {
    const g = document.createElement("optgroup");
    g.label = name.includes(": regions") ? name : `${name}: countries`;
    for (const a of byGroup.get(name).sort((x, y) => x.name.localeCompare(y.name))) {
      option(`area:${a.key}`, a.fits ? a.name : `${a.name} (too large: pick a region)`, g);
      if (!a.fits) g.lastChild.disabled = true;
    }
    sel.appendChild(g);
  }
  option("custom", "Custom rectangle…", sel);
  const dl = $("#area-list");
  dl.replaceChildren();
  for (const a of state.areas) if (a.fits) { const o = document.createElement("option"); o.value = a.name; dl.appendChild(o); }
  sel.onchange = () => {
    $("#area-hint").textContent = "Any place on Earth: pick a country or region, click it on the map, or draw a rectangle.";
    const p = state.presets.find((x) => x.key === sel.value);
    if (p) { setBBox(p.bounds); showAoi(p.geojson); }
    else if (sel.value.startsWith("aoi:")) loadAnalysis(sel.value.slice(4)).catch((e) => alert(e.message));
    else if (sel.value.startsWith("area:")) previewArea(sel.value.slice(5));
  };
}

async function previewArea(key) {
  const a = await api(`/api/area?key=${encodeURIComponent(key)}`);
  setBBox(a.bounds);
  showAoi(a.geojson);
  $("#area-hint").textContent = `${a.name} selected. Press “Analyse area”: the first analysis of a new country downloads its ` +
    "FIRMS archive (a few MB to ~3 GB for the most fire-prone countries); it is cached afterwards.";
}

function selectArea(key, message) {
  const sel = $("#preset-select"), value = `area:${key}`;
  const opt = [...sel.options].find((o) => o.value === value);
  if (!opt || opt.disabled) { if (message) $("#area-hint").textContent = message; return; }
  sel.value = value;
  sel.onchange();
}

async function runAnalysis() {
  const key = $("#preset-select").value;
  const body = { force: $("#force").checked };
  if (state.presets.some((p) => p.key === key)) body.preset = key;
  else if (key.startsWith("area:")) body.area = key.slice(5);
  else {  // a drawn or typed rectangle, or one of "your areas" again
    const b = ["w", "s", "e", "n"].map((k) => parseFloat($(`#bb-${k}`).value));
    if (b.some((v) => !isFinite(v))) { alert("Draw a rectangle or type its bounds first."); return; }
    body.bbox = b;
  }
  const btn = $("#run-btn");
  btn.disabled = true;
  const prog = $("#progress");
  prog.hidden = false;
  try {
    const job = await api("/api/analyses", { method: "POST", body: JSON.stringify(body) });
    for (;;) {
      const s = await api(`/api/jobs/${job.job_id}`);
      prog.querySelector(".fill").style.width = `${Math.round(100 * s.progress)}%`;
      prog.querySelector(".msg").textContent = `${Math.round(100 * s.progress)}% · ${s.message || ""}`;
      if (s.status === "done") { await loadAnalysis(s.aoi_id); break; }
      if (s.status === "error") throw new Error(s.error);
      await new Promise((r) => setTimeout(r, 1000));
    }
    const done = body.preset || `aoi:${job.aoi_id}`;
    loadPresets().then(() => { $("#preset-select").value = done; });
  } catch (e) {
    prog.querySelector(".msg").textContent = `Error: ${e.message}`;
  } finally {
    btn.disabled = false;
  }
}

async function loadAnalysis(id) {
  const d = await api(`/api/analyses/${id}`);
  state.data = d;
  state.aoiId = id;
  state.live = d.forecast.operational;
  state.forecast = null;
  state.event = null;
  const full = d.calendar.annual.filter((a) => a.days >= 300).map((a) => a.year);
  state.year = full.length ? full[full.length - 1] : d.calendar.years[d.calendar.years.length - 1];
  $("#csv-link").href = `/api/analyses/${id}/daily.csv`;
  showAoi(d.meta.aoi);
  setBBox(d.meta.bounds);
  renderAll();
}

// ---------------------------------------------------------------------------------------------
// helpers on the calendar matrix
// ---------------------------------------------------------------------------------------------
const isLeap = (y) => (y % 4 === 0 && y % 100 !== 0) || y % 400 === 0;
function mdOf(date) {
  const [y, m, d] = date.split("-").map(Number);
  return MONTH_STARTS[m - 1] + d - 1;
}
function valueOn(date, field = "matrix") {
  const c = state.data.calendar, y = +date.slice(0, 4);
  const r = c.years.indexOf(y);
  return r < 0 ? null : c[field][r][mdOf(date)];
}
function addDays(date, n) {
  const t = new Date(date + "T00:00:00Z");
  t.setUTCDate(t.getUTCDate() + n);
  return t.toISOString().slice(0, 10);
}
const dayNum = (date) => Date.parse(date + "T00:00:00Z") / 864e5;
const prettyDate = (date) => new Date(date + "T00:00:00Z").toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
function modeledYear(y) {
  const a = state.data.calendar.annual.find((r) => r.year === y);
  return a && a.modeled_days > 180;
}

// ---------------------------------------------------------------------------------------------
// summary
// ---------------------------------------------------------------------------------------------
function tile(label, value, sub) {
  const t = document.createElement("div");
  t.className = "tile";
  for (const [cls, txt] of [["label", label], ["value", value], ["sub", sub]]) {
    if (txt == null) continue;
    const e = document.createElement("div");
    e.className = cls;
    e.textContent = txt;
    t.appendChild(e);
  }
  return t;
}
function anomalyWord(p) {
  if (p == null) return "";
  if (p >= 97.5) return "extreme for the time of year";
  if (p >= 90) return "much above normal";
  if (p >= 66.7) return "above normal";
  if (p > 33.3) return "near normal";
  if (p > 10) return "below normal";
  return "much below normal";
}

function renderSummary() {
  const d = state.data, m = d.meta, c = d.calendar;
  $("#summary-empty").hidden = true;
  $("#summary").hidden = false;
  $("#aoi-name").textContent = m.name;
  const nSensors = Object.values(m.sensors).filter((s) => s.days > 0).length;
  $("#aoi-meta").textContent = `${m.countries.join(", ").replaceAll("_", " ")} · ${m.grid.res}° grid · ` +
    `${m.data_start} → ${m.data_end} · ${nSensors} satellite sensors · ${fmt.int(m.hotspots_qc)} quality-checked hotspots`;
  // hero: last 7 days with data
  const end = m.data_end;
  let tot = 0, n = 0;
  for (let i = 0; i < 7; i++) { const v = valueOn(addDays(end, -i)); if (v != null) { tot += v; n++; } }
  const p7 = valueOn(end, "pct7") ?? valueOn(addDays(end, -3), "pct7");
  $("#hero-label").textContent = `Last 7 days (to ${prettyDate(end)})`;
  $("#hero-value").textContent = n ? fmt.int(tot) : "–";
  $("#hero-delta").textContent = n ? `VIIRS-equivalent detections · ${p7 != null ? `${Math.round(p7)}th percentile, ${anomalyWord(p7)}` : ""}` : "no recent data";
  const tiles = $("#tiles");
  tiles.replaceChildren();
  const op = state.live;
  if (op?.days?.length) {
    const next = op.days.slice(0, 7);
    const sumMed = next.reduce((s, x) => s + x.median, 0), sumNorm = next.reduce((s, x) => s + x.normal, 0);
    const pk = next.reduce((a, b) => (b.p_critical > a.p_critical ? b : a));
    tiles.appendChild(tile("Next 7 days (forecast)", fmt.int(sumMed), `normal ${fmt.int(sumNorm)} · max chance of a critical day ${fmt.pct(pk.p_critical)} (${prettyDate(pk.date).slice(0, -5)})`));
  }
  const peak = c.critical_periods.find((p) => p.level === "peak");
  if (peak) tiles.appendChild(tile("Peak fire period (normal)", `${peak.start} – ${peak.end}`, `${fmt.pct(peak.share)} of yearly activity in ${peak.days} days`));
  const seasons = c.seasons;
  if (seasons.length) {
    const top = seasons.reduce((a, b) => (b.total > a.total ? b : a));
    tiles.appendChild(tile("Normal year", fmt.compact(c.climatology.annual_mean), `detections · worst fire year: ${top.fire_year.split(" (")[0]} (×${fmt.num(top.total_ratio, 1)})`));
  }
  const hm = d.harmonization.metrics.cv.harmonized;
  tiles.appendChild(tile("MODIS → VIIRS harmonization", `r = ${fmt.num(hm.daily.r, 2)}`, `daily, held-out years · bias ${fmt.num(hm.daily.bias_pct, 1)}%`));
  const s1 = d.forecast.daily_metrics.by_lead["1"], s7 = d.forecast.daily_metrics.by_lead["7"];
  if (s1) tiles.appendChild(tile("Critical-day forecast skill", `AUC ${fmt.num(s1.critical_auc, 2)}`, `1 day ahead (${fmt.num(s7?.critical_auc, 2)} at 7 days) · unseen ${d.forecast.daily_metrics.test_years.join("–")}`));
  // alerts
  const al = $("#alert");
  al.replaceChildren();
  al.hidden = true;
  const notes = [];
  const worst = op?.days?.reduce((a, b) => (b.p_critical > a.p_critical ? b : a), { p_critical: 0 });
  if (worst && worst.p_critical >= 0.3) notes.push({ level: worst.p_critical >= 0.5 ? "critical" : "serious", icon: "!",
    text: `Elevated chance (${fmt.pct(worst.p_critical)}) of a critical burning day on ${prettyDate(worst.date)} – activity above the 90th percentile for the date.` });
  // fire weather can be extreme while the count model still under-reacts (Portugal, Sept 2024)
  const dangerous = (op?.days || []).slice(0, 7).filter((d) => ["very high", "extreme"].includes(d.fire_weather?.fwi_class));
  if (dangerous.length) {
    const top = dangerous.reduce((x, y) => (y.fire_weather.fwi > x.fire_weather.fwi ? y : x)), fw = top.fire_weather;
    notes.push({ level: fw.fwi_class === "extreme" ? "critical" : "serious", icon: "!",
      text: `${fw.fwi_class === "extreme" ? "Extreme" : "Very high"} fire weather forecast: Fire Weather Index ${fmt.num(fw.fwi, 0)} on ${prettyDate(top.date)} (${fw.fwi_anom >= 0 ? "+" : ""}${fmt.num(fw.fwi_anom, 0)} vs normal); ${dangerous.length} of the next 7 days ${dangerous.length === 1 ? "is" : "are"} very high or extreme.` });
  }
  if (m.gap_without_key) notes.push({ level: "warning", icon: "i",
    text: `No FIRMS MAP_KEY configured: the public archive ends ${m.archive_last_year}-12-31 and live data covers the last 7 days, so ${m.gap_without_key[0]} – ${m.gap_without_key[1]} shows as “no data”. Add FIRMS_MAP_KEY to .env to fill the gap.` });
  for (const nt of notes) {
    al.hidden = false;
    const row = document.createElement("div");
    row.style.display = "flex"; row.style.gap = "10px"; row.style.alignItems = "flex-start";
    const ic = document.createElement("span");
    ic.className = "status-icon"; ic.style.background = css(`--${nt.level}`); ic.textContent = nt.icon;
    ic.setAttribute("aria-label", nt.level);
    const tx = document.createElement("div");
    const strong = document.createElement("strong");
    strong.textContent = nt.level === "warning" ? "Data note. " : "Alert. ";
    tx.append(strong, nt.text);
    row.append(ic, tx);
    al.appendChild(row);
  }
}

// ---------------------------------------------------------------------------------------------
// calendar heat map (canvas)
// ---------------------------------------------------------------------------------------------
const CAL = { left: 58, top: 20, row: 14, gap: 2 };
function anomalyColor(p, stops) {
  if (p == null) return null;
  if (p < 10) return stops[0];
  if (p < 33.3) return stops[1];
  if (p < 66.7) return stops[2];
  if (p < 90) return stops[3];
  if (p < 97.5) return stops[4];
  return stops[5];
}
function renderCalendar() {
  const c = state.data.calendar;
  const cv = $("#cal-canvas");
  const W = cv.parentElement.clientWidth;
  const H = CAL.top + c.years.length * CAL.row + 4;
  const dpr = window.devicePixelRatio || 1;
  cv.width = W * dpr; cv.height = H * dpr; cv.style.height = `${H}px`;
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = css("--surface"); ctx.fillRect(0, 0, W, H);
  const cw = (W - CAL.left - 4) / 366;
  // no-data texture
  const pat = document.createElement("canvas"); pat.width = 6; pat.height = 6;
  const pc = pat.getContext("2d"); pc.strokeStyle = css("--nodata"); pc.lineWidth = 1.5;
  pc.beginPath(); pc.moveTo(0, 6); pc.lineTo(6, 0); pc.stroke();
  const noData = ctx.createPattern(pat, "repeat");
  const vals = c.matrix.flat().filter((v) => v != null).sort((a, b) => a - b);
  const vmax = Math.max(5, vals[Math.floor(vals.length * 0.995)] || 1);
  state.calVmax = vmax;
  const heat = heatStops(), div = divStops();
  ctx.font = `11px ${css("--font")}`;
  // month header
  ctx.fillStyle = css("--muted");
  ctx.textBaseline = "middle";
  MONTH_NAMES.forEach((mn, i) => {
    const x = CAL.left + MONTH_STARTS[i] * cw;
    ctx.fillText(mn, x + 2, 9);
    ctx.fillStyle = css("--axis"); ctx.fillRect(x, 14, 1, 5); ctx.fillStyle = css("--muted");
  });
  c.years.forEach((y, r) => {
    const top = CAL.top + r * CAL.row;
    ctx.fillStyle = css("--ink-2");
    ctx.textAlign = "right";
    ctx.fillText(`${modeledYear(y) ? "○ " : ""}${y}`, CAL.left - 8, top + (CAL.row - CAL.gap) / 2);
    ctx.textAlign = "left";
    for (let k = 0; k < 366; k++) {
      const kk = k === 59 && !isLeap(y) ? 58 : k;  // common years: Feb 29 column repeats Feb 28
      const v = c.matrix[r][kk];
      let col;
      if (state.calMode === "activity") col = v == null ? null : rampColor(heat, Math.log1p(v) / Math.log1p(vmax));
      else col = anomalyColor(c.pct7[r][kk], div);
      ctx.fillStyle = col || noData;
      ctx.fillRect(CAL.left + k * cw, top, Math.ceil(cw), CAL.row - CAL.gap);
    }
  });
  // selected year & event outlines
  ctx.strokeStyle = css("--ink"); ctx.lineWidth = 1.5;
  const ry = c.years.indexOf(state.year);
  if (ry >= 0) ctx.strokeRect(CAL.left - 1, CAL.top + ry * CAL.row - 1, 366 * cw + 2, CAL.row - CAL.gap + 2);
  if (state.event) {
    let d = state.event.start;
    while (d <= state.event.end) {
      const r = c.years.indexOf(+d.slice(0, 4)), k = mdOf(d);
      if (r >= 0) { ctx.strokeStyle = css("--ink"); ctx.lineWidth = 2; ctx.strokeRect(CAL.left + k * cw, CAL.top + r * CAL.row - 2, Math.ceil(cw), CAL.row - CAL.gap + 4); }
      d = addDays(d, 1);
    }
  }
  renderCalLegend(vmax);
}
function calCellAt(ev) {
  const c = state.data.calendar, cv = $("#cal-canvas"), rect = cv.getBoundingClientRect();
  const W = rect.width, cw = (W - CAL.left - 4) / 366;
  const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
  const r = Math.floor((y - CAL.top) / CAL.row), k = Math.floor((x - CAL.left) / cw);
  if (r < 0 || r >= c.years.length || k < 0 || k > 365) return null;
  return { r, k, year: c.years[r] };
}
function calTooltip(cell, cx, cy) {
  const c = state.data.calendar, tip = tooltip();
  const { r, k, year } = cell;
  if (k === 59 && !isLeap(year)) { tip.hide(); return; }
  const v = c.matrix[r][k], p = c.pct7[r][k], src = c.source[r][k];
  const date = `${mdLabel(k)} ${year}`;
  const cl = c.climatology;
  const rows = [];
  if (v == null) rows.push({ value: "no data", label: SOURCE[0] });
  else {
    const modeled = src >= 4, e = c.expected?.[r]?.[k];
    rows.push({ value: fmt.num(v, v < 10 ? 1 : 0), label: modeled ? "detections, most likely for the day" : "detections (VIIRS S-NPP equivalent)" });
    if (modeled && e != null) rows.push({ value: fmt.num(e, e < 10 ? 1 : 0), label: "expected value (used for totals)" });
    rows.push({ value: fmt.num(cl.q50[k], 1), label: `normal median (mean ${fmt.num(cl.mean[k], 0)})` });
    if (p != null) rows.push({ value: `${Math.round(p)}th`, label: `percentile of 7-day activity · ${anomalyWord(p)}` });
    rows.push({ value: "", label: SOURCE[src] || "" });
  }
  tip.show(cx, cy, date, rows);
}
function renderCalLegend(vmax) {
  const lg = $("#cal-legend");
  lg.replaceChildren();
  const add = (html) => { const s = document.createElement("span"); s.className = "sw"; s.innerHTML = html; lg.appendChild(s); return s; };
  if (state.calMode === "activity") {
    const t = document.createElement("span"); t.textContent = "Detections per day"; lg.appendChild(t);
    const lo = document.createElement("span"); lo.textContent = "0"; lg.appendChild(lo);
    const ramp = document.createElement("span"); ramp.className = "ramp";
    ramp.style.background = `linear-gradient(90deg, ${heatStops().join(",")})`; lg.appendChild(ramp);
    const hi = document.createElement("span"); hi.textContent = `${fmt.compact(vmax)}+ (log scale)`; lg.appendChild(hi);
  } else {
    const labels = ["much below (<10th pct)", "below", "near normal", "above", "much above (>90th)", "extreme (>97.5th)"];
    divStops().forEach((col, i) => { const s = add(`<i style="background:${col}"></i>`); s.append(labels[i]); });
  }
  const nd = add('<i class="nodata"></i>'); nd.append("no data");
  const mo = document.createElement("span"); mo.textContent = "○ year reconstructed from MODIS"; lg.appendChild(mo);
}

function renderMonthGrid() {
  const c = state.data.calendar, g = $("#month-grid");
  g.replaceChildren();
  g.style.gridTemplateColumns = `48px repeat(12, minmax(0, 1fr))`;
  const tip = tooltip(), div = divStops();
  g.appendChild(document.createElement("div"));
  for (const mn of MONTH_NAMES) { const e = document.createElement("div"); e.className = "ml"; e.textContent = mn; g.appendChild(e); }
  const byKey = new Map(c.monthly.map((r) => [`${r.year}-${r.month}`, r]));
  for (const y of c.years) {
    const yl = document.createElement("div"); yl.className = "yl"; yl.textContent = `${modeledYear(y) ? "○ " : ""}${y}`; g.appendChild(yl);
    for (let mth = 1; mth <= 12; mth++) {
      const r = byKey.get(`${y}-${mth}`);
      const e = document.createElement("div");
      e.className = "cell";
      if (!r || r.total == null) e.classList.add("empty");
      else {
        const l = Math.log2(r.ratio);
        e.style.background = l <= -2 ? div[0] : l <= -0.585 ? div[1] : l < 0.585 ? div[2] : l < 2 ? div[3] : l < 3.32 ? div[4] : div[5];
        e.tabIndex = 0;
        const show = (x, yy) => tip.show(x, yy, `${MONTH_NAMES[mth - 1]} ${y}`, [
          { value: fmt.int(r.total), label: "detections" },
          { value: fmt.int(r.normal), label: "normal (median of years)" },
          { value: `×${fmt.num(r.ratio, 1)}`, label: "of normal" },
          { value: `${r.rank} / ${r.years_ranked}`, label: "rank among years" }]);
        e.addEventListener("pointermove", (ev) => show(ev.clientX, ev.clientY));
        e.addEventListener("pointerleave", () => tip.hide());
        e.addEventListener("focus", () => { const b = e.getBoundingClientRect(); show(b.right, b.top); });
        e.addEventListener("blur", () => tip.hide());
      }
      g.appendChild(e);
    }
  }
  const lg = $("#month-legend");
  lg.replaceChildren();
  ["≤ ¼ of normal", "≤ ⅔", "near normal", "1.5–4×", "4–10×", "> 10×"].forEach((t, i) => {
    const s = document.createElement("span"); s.className = "sw"; s.innerHTML = `<i style="background:${div[i]}"></i>`; s.append(t); lg.appendChild(s);
  });
}

// ---------------------------------------------------------------------------------------------
// normal pattern (climatology) & seasons
// ---------------------------------------------------------------------------------------------
function legend(elm, items) {
  elm.replaceChildren();
  for (const it of items) {
    const s = document.createElement("span"); s.className = "item";
    const k = document.createElement("span"); k.className = `${it.kind || "line"}-key`;
    k.style.background = it.color; if (it.opacity) k.style.opacity = it.opacity;
    const t = document.createElement("span"); t.textContent = it.label;
    s.append(k, t); elm.appendChild(s);
  }
}
function rolling7(arr) {
  return arr.map((_, i) => {
    let s = 0, n = 0;
    for (let j = i - 3; j <= i + 3; j++) if (j >= 0 && j < arr.length && arr[j] != null) { s += arr[j]; n++; }
    return n >= 4 ? s / n : null;
  });
}
function renderSeason() {
  const c = state.data.calendar, cl = c.climatology, cl7 = c.climatology7 || cl;
  const sel = $("#year-select");
  if (sel.options.length !== c.years.length) {
    sel.replaceChildren();
    for (const y of c.years.slice().reverse()) { const o = document.createElement("option"); o.value = y; o.textContent = `${y}${modeledYear(y) ? " (from MODIS)" : ""}`; sel.appendChild(o); }
  }
  sel.value = state.year;
  $("#clim-sub").textContent = `7-day mean activity. Normal = all ${cl.baseline_years.length} years ${cl.baseline_years[0]}–${cl.baseline_years[cl.baseline_years.length - 1]}, pooled ±${cl.window} days. Shaded: peak period. Log scale.`;
  const x = Array.from({ length: 366 }, (_, i) => i);
  const r = c.years.indexOf(state.year);
  const yearVals = r >= 0 ? c.matrix[r] : [];
  const year7 = rolling7(r >= 0 ? (c.expected || c.matrix)[r] : []);
  const s1 = css("--s1"), s2 = css("--s2");
  legend($("#clim-legend"), [
    { label: "normal (median)", color: s1 }, { label: "middle 50% of years", color: s1, kind: "box", opacity: 0.3 },
    { label: "80% of years", color: s1, kind: "box", opacity: 0.15 }, { label: `${state.year}, 7-day mean`, color: s2 }]);
  const spans = c.critical_periods.slice().sort((a, b) => (a.level === "peak") - (b.level === "peak")).map((p) => ({
    from: p.start_md, to: p.end_md, color: p.level === "peak" ? css("--span") : "transparent",
    label: p.level === "peak" ? `Peak: ${p.start} – ${p.end}` : "",
  }));
  seriesChart($("#clim-chart"), {
    x, yScale: "log1p", height: 300, ariaLabel: "Normal yearly cycle of burning activity",
    xTicks: MONTH_STARTS.map((s, i) => ({ at: s + 15, label: MONTH_NAMES[i] })),
    bands: [{ lo: cl7.q10, hi: cl7.q90, color: s1, opacity: 0.1 }, { lo: cl7.q25, hi: cl7.q75, color: s1, opacity: 0.18 }],
    lines: [{ values: cl7.q50, color: s1, endLabel: "normal" }, { values: year7, color: s2, endLabel: String(state.year) }],
    spans,
    tooltip: (i) => ({ title: mdLabel(i), rows: [
      { value: fmt.num(year7[i], 1), label: `${state.year}, 7-day mean`, color: s2 },
      { value: fmt.num(yearVals[i], 0), label: `${state.year}, that day` },
      { value: fmt.num(cl7.q50[i], 1), label: "normal 7-day mean (median)", color: s1 },
      { value: `${fmt.num(cl7.q25[i], 1)}–${fmt.num(cl7.q75[i], 1)}`, label: "middle 50% of years" },
      { value: `${fmt.num(cl7.q10[i], 1)}–${fmt.num(cl7.q90[i], 1)}`, label: "80% of years" },
      { value: fmt.pct(cl.p_fire[i]), label: "chance of a day with fire" }] }),
  });
  const pl = $("#periods");
  pl.replaceChildren();
  for (const p of c.critical_periods) {
    const e = document.createElement("div"); e.className = "period";
    const s = document.createElement("strong"); s.textContent = `${p.start} – ${p.end}`;
    const t = document.createElement("span"); t.className = "muted";
    t.textContent = `${p.level === "peak" ? "Peak period" : "High-activity period"} · ${p.days} days · ${fmt.pct(p.share)} of yearly activity`;
    e.append(s, t); pl.appendChild(e);
  }
  // seasons table
  const fy = c.fire_year_start_month;
  $("#season-sub").textContent = `Fire year starts in ${MONTH_NAMES[fy - 1]} (the quietest month). Onset/end = dates when 10% / 90% of the year's activity had burned; peak = highest 15-day mean.`;
  const tb = $("#season-table");
  tb.replaceChildren();
  const head = tb.createTHead().insertRow();
  for (const [h, num] of [["Fire year", 0], ["Total", 1], ["vs mean", 1], ["Onset", 0], ["Peak", 0], ["End", 0], ["Length", 1], ["Rank", 1]]) {
    const th = document.createElement("th"); th.textContent = h; if (num) th.className = "num"; head.appendChild(th);
  }
  const body = tb.createTBody();
  for (const s of c.seasons.slice().reverse()) {
    const tr = body.insertRow();
    tr.className = "clickable" + (s.year === state.year ? " selected" : "");
    const shift = (v) => (v == null ? "" : ` (${v > 0 ? "+" : ""}${Math.round(v)} d)`);
    const cells = [[`${s.fire_year}${modeledYear(s.year) ? " ○" : ""}`], [fmt.int(s.total), 1], [`×${fmt.num(s.total_ratio, 2)}`, 1],
      [`${s.onset.slice(5)}${shift(s.onset_shift_days)}`], [s.peak.slice(5)], [`${s.end.slice(5)}${shift(s.end_shift_days)}`],
      [`${s.length_days} d`, 1], [`${s.rank}`, 1]];
    for (const [v, num] of cells) { const td = tr.insertCell(); td.textContent = v; if (num) td.className = "num"; }
    tr.onclick = () => { state.year = s.year; renderSeason(); renderCalendar(); };
  }
}

// ---------------------------------------------------------------------------------------------
// events
// ---------------------------------------------------------------------------------------------
function renderEvents() {
  const ev = state.data.calendar.events, tb = $("#event-table");
  tb.replaceChildren();
  const head = tb.createTHead().insertRow();
  for (const [h, num] of [["#", 1], ["Period", 0], ["Days", 1], ["Peak day", 0], ["Peak", 1], ["Total", 1], ["× normal", 1], ["Notes", 0]]) {
    const th = document.createElement("th"); th.textContent = h; if (num) th.className = "num"; head.appendChild(th);
  }
  const body = tb.createTBody();
  ev.forEach((e, i) => {
    const tr = body.insertRow();
    tr.className = "clickable";
    const notes = [e.out_of_season ? "out of season" : "", e.modeled ? "MODIS era" : ""].filter(Boolean).join(" · ");
    const cells = [[i + 1, 1], [`${prettyDate(e.start)} – ${prettyDate(e.end)}`], [e.days, 1], [prettyDate(e.peak_date)],
      [fmt.int(e.peak_value), 1], [fmt.int(e.total), 1], [`×${fmt.num(e.ratio, 1)}`, 1], [notes]];
    for (const [v, num] of cells) { const td = tr.insertCell(); td.textContent = v; if (num) td.className = "num"; }
    tr.onclick = () => {
      state.event = e; state.year = +e.start.slice(0, 4);
      selectTab("calendar"); renderCalendar();
      $("#cal-canvas").scrollIntoView({ behavior: "smooth", block: "center" });
    };
  });
}

// ---------------------------------------------------------------------------------------------
// forecast
// ---------------------------------------------------------------------------------------------
function renderForecast() {
  const fc = state.forecast || state.live;
  const s1 = css("--s1"), s2 = css("--s2"), ink2 = css("--ink-2");
  const title = $("#fc-title"), sub = $("#fc-sub");
  $("#bt-reset").hidden = !state.forecast;
  if (!fc) {
    title.textContent = "Next 14 days";
    sub.textContent = "No live forecast: recent fire observations are missing. Use a backtest date to see the model at work.";
    ["#fc-chart", "#fc-prob", "#fc-fwi", "#fc-table"].forEach((s) => $(s).replaceChildren());
  } else {
    const live = !state.forecast;
    title.textContent = live ? `Next ${fc.days.length} days` : `Backtest issued ${prettyDate(fc.issue_date)}`;
    sub.textContent = live
      ? `Issued from observations up to ${prettyDate(fc.issue_date)}, with Open-Meteo weather forecasts bias-corrected to NASA POWER.`
      : `${fc.mode === "out-of-sample backtest" ? "Out-of-sample" : "In-sample"}: model trained on data until ${fc.model_trained_until}; observed weather used (upper bound for skill). Dots = what actually happened.`;
    // history (14 days) + forecast
    const hist = [];
    for (let i = 13; i >= 0; i--) hist.push(addDays(fc.issue_date, -i));
    const dates = hist.concat(fc.days.map((d) => d.date));
    const xs = dates.map(dayNum);
    const nH = hist.length;
    const pad = (arr) => Array(nH).fill(null).concat(arr);
    const obs = hist.map((d) => valueOn(d)).concat(fc.days.map((d) => d.observed));
    const normal = dates.map((d) => state.data.calendar.climatology.q50[mdOf(d)]);
    const lo = pad(fc.days.map((d) => d.q10)), hi = pad(fc.days.map((d) => d.q90));
    const med = Array(nH - 1).fill(null).concat([valueOn(fc.issue_date)], fc.days.map((d) => d.median));
    legend($("#fc-legend"), [{ label: "forecast (median)", color: s1 }, { label: "80% range", color: s1, kind: "box", opacity: 0.2 },
      { label: "normal median", color: ink2 }, { label: "observed", color: s2, kind: "dot" }]);
    seriesChart($("#fc-chart"), {
      x: xs, yScale: "log1p", height: 260, ariaLabel: "Forecast of daily burning activity",
      xTicks: dates.filter((_, i) => i % 2 === 0).map((d) => ({ at: dayNum(d), label: d.slice(5) })),
      spans: [{ from: dayNum(fc.issue_date) + 0.5, to: xs[xs.length - 1], color: css("--span"), label: "forecast" }],
      bands: [{ lo, hi, color: s1, opacity: 0.14 }],
      lines: [{ values: normal, color: ink2, width: 1.5 }, { values: med, color: s1 }],
      dots: [{ values: obs, color: s2 }],
      tooltip: (i) => {
        const d = dates[i], f = i >= nH ? fc.days[i - nH] : null;
        const rows = [];
        if (f) rows.push({ value: fmt.num(f.median, 0), label: "forecast median", color: s1 }, { value: `${fmt.num(f.q10, 0)}–${fmt.num(f.q90, 0)}`, label: "80% range" },
          { value: fmt.pct(f.p_critical), label: "chance of a critical day" });
        if (obs[i] != null) rows.push({ value: fmt.num(obs[i], 0), label: "observed", color: s2 });
        rows.push({ value: fmt.num(normal[i], 1), label: "normal median", color: ink2 });
        if (f?.fire_weather) rows.push({ value: fmt.num(f.fire_weather.fwi, 0), label: `FWI, ${f.fire_weather.fwi_class} (${f.weather})` });
        else if (f) rows.push({ value: "", label: `weather: ${f.weather}` });
        return { title: prettyDate(d), rows };
      },
    });
    columnChart($("#fc-prob"), {
      labels: fc.days.map((d) => d.date.slice(5)), values: fc.days.map((d) => d.p_critical), color: s1, yMax: Math.max(0.3, ...fc.days.map((d) => d.p_critical)),
      valueFmt: (v) => fmt.pct(v), refs: [{ y: 0.1, label: "normal chance (10%)" }], height: 150, ariaLabel: "Chance of a critical day",
      tooltip: (i) => ({ title: prettyDate(fc.days[i].date), rows: [{ value: fmt.pct(fc.days[i].p_critical), label: "chance of a critical day" },
        { value: fmt.int(fc.days[i].critical_threshold), label: "critical threshold (detections)" }] }),
    });
    const fwOk = fc.days.every((d) => d.fire_weather);
    if (fwOk) {
      const heat = heatStops(), cls = ["very low", "low", "moderate", "high", "very high", "extreme"];
      columnChart($("#fc-fwi"), {
        labels: fc.days.map((d) => d.date.slice(5)), values: fc.days.map((d) => d.fire_weather.fwi),
        // the palest heat step would vanish on the surface, so the two lowest classes share the next one
        colors: fc.days.map((d) => heat[[1, 1, 2, 3, 4, 5][Math.max(0, cls.indexOf(d.fire_weather.fwi_class))]]),
        yMax: Math.max(40, ...fc.days.map((d) => d.fire_weather.fwi)), valueFmt: (v) => fmt.num(v, 0), height: 150,
        refs: [{ y: 21.3, label: "high" }, { y: 38, label: "very high" }], ariaLabel: "Fire Weather Index by day",
        tooltip: (i) => {
          const w = fc.days[i].fire_weather;
          return { title: prettyDate(fc.days[i].date), rows: [
            { value: fmt.num(w.fwi, 0), label: `Fire Weather Index · ${w.fwi_class}` },
            { value: `${w.fwi_anom >= 0 ? "+" : ""}${fmt.num(w.fwi_anom, 0)}`, label: "vs normal for the date" },
            { value: fmt.num(w.isi, 0), label: "initial spread index" },
            { value: `${fmt.num(w.tmax, 0)} °C`, label: "max temperature" },
            { value: `${fmt.num(w.rh, 0)}%`, label: "relative humidity" },
            { value: `${fmt.num(w.wind, 0)} m/s`, label: "max wind" },
            { value: `${fmt.num(w.rain, 1)} mm`, label: `rain (${fc.days[i].weather})` }] };
        },
      });
    } else $("#fc-fwi").replaceChildren();
    const tb = $("#fc-table");
    tb.replaceChildren();
    const h = tb.createTHead().insertRow();
    for (const [t, n] of [["Date", 0], ["Median", 1], ["80% range", 1], ["Normal", 1], ["P(critical)", 1], ["Observed", 1], ["Fire weather (FWI)", 0], ["Weather", 0]]) {
      const th = document.createElement("th"); th.textContent = t; if (n) th.className = "num"; h.appendChild(th);
    }
    const b = tb.createTBody();
    for (const d of fc.days) {
      const tr = b.insertRow();
      const fw = d.fire_weather;
      const fwTxt = fw ? `${fmt.num(fw.fwi, 0)} ${fw.fwi_class} (${fw.fwi_anom >= 0 ? "+" : ""}${fmt.num(fw.fwi_anom, 0)} vs normal)` : "–";
      for (const [v, n] of [[prettyDate(d.date)], [fmt.int(d.median), 1], [`${fmt.int(d.q10)}–${fmt.int(d.q90)}`, 1], [fmt.num(d.normal, 0), 1],
        [fmt.pct(d.p_critical), 1], [d.observed == null ? "–" : fmt.int(d.observed), 1], [fwTxt], [d.weather]]) {
        const td = tr.insertCell(); td.textContent = v; if (n) td.className = "num";
      }
    }
  }
  renderOutlook();
  renderSkill();
}

function renderOutlook() {
  const ol = state.data.forecast.outlook || [], box = $("#outlook"), div = divStops();
  box.replaceChildren();
  $("#ol-sub").textContent = "Chance that the monthly total falls in the lower, middle or upper third of past years, from ENSO, rainfall, temperature, soil moisture and recent fire anomalies. Low skill = close to climatology.";
  if (!ol.length) { box.textContent = "Not enough data for an outlook."; return; }
  for (const o of ol) {
    const row = document.createElement("div"); row.className = "outlook-row";
    const lab = document.createElement("div"); lab.className = "lab";
    const [yy, mm] = o.month.split("-");
    lab.textContent = `${MONTH_NAMES[+mm - 1]} ${yy}`;
    const sm = document.createElement("small");
    sm.textContent = o.skill > 0.05 ? `skill ${fmt.num(o.skill, 2)}` : "no skill: climatology";
    lab.appendChild(sm);
    const st = document.createElement("div"); st.className = "stack";
    const parts = [["below", o.p_below, div[1]], ["near", o.p_near, div[2]], ["above", o.p_above, div[4]]];
    for (const [name, p, col] of parts) {
      const seg = document.createElement("div");
      seg.style.width = `${Math.max(p * 100, 0.5)}%`;
      seg.style.background = col;
      seg.style.color = name === "near" ? css("--ink") : "#ffffff";
      if (name === "below") seg.style.color = css("--ink");
      seg.textContent = p >= 0.14 ? `${name} ${Math.round(p * 100)}%` : "";
      seg.title = `${name} normal: ${Math.round(p * 100)}%`;
      st.appendChild(seg);
    }
    if (o.skill <= 0.05) st.style.opacity = 0.45;
    row.append(lab, st);
    box.appendChild(row);
  }
}

function renderSkill() {
  const m = state.data.forecast.daily_metrics, tb = $("#skill-table");
  $("#skill-sub").textContent = `Scored on ${m.test_years.join(" and ")}, never seen in training (trained ${m.train_years[0]}–${m.train_years[m.train_years.length - 1]}, calibrated on ${m.valid_year}). Skill = reduction of the typical (log) error; 0 = no better than the reference. Upper bound uses observed weather, lower bound none.`;
  tb.replaceChildren();
  const h = tb.createTHead().insertRow();
  for (const t of ["Days ahead", "Cell AUC (normal)", "Area ρ (persistence)", "Skill vs normal", "Skill vs normal, no weather forecast", "Critical-day AUC", "80% range hit"]) {
    const th = document.createElement("th"); th.textContent = t; th.className = "num"; h.appendChild(th);
  }
  const b = tb.createTBody();
  for (const [lead, s] of Object.entries(m.by_lead)) {
    const tr = b.insertRow();
    for (const v of [lead, `${fmt.num(s.cell_auc, 3)} (${fmt.num(s.cell_auc_climatology, 3)})`,
      `${fmt.num(s.aoi_spearman, 2)} (${fmt.num(s.aoi_spearman_persistence, 2)})`, fmt.num(s.aoi_skill_vs_climatology, 2),
      fmt.num(s.aoi_skill_vs_climatology_no_weather_forecast, 2), fmt.num(s.critical_auc, 2), fmt.pct(s.interval80_coverage)]) {
      const td = tr.insertCell(); td.textContent = v; td.className = "num";
    }
  }
  // importance: grouped, top 12
  const imp = Object.entries(state.data.forecast.feature_importance).slice(0, 12);
  const box = $("#importance");
  box.replaceChildren();
  const max = imp[0]?.[1] || 1;
  for (const [f, v] of imp) {
    const row = document.createElement("div");
    row.style.cssText = "display:grid;grid-template-columns:minmax(0,46%) 1fr 44px;gap:8px;align-items:center;font-size:12px;margin:3px 0";
    const l = document.createElement("span"); l.textContent = FEATURE_NAMES[f] || f; l.style.color = css("--ink-2");
    const bar = document.createElement("span");
    bar.style.cssText = `height:10px;border-radius:0 3px 3px 0;background:${css("--s1")};width:${Math.max(2, (100 * v) / max)}%`;
    const n = document.createElement("span"); n.textContent = fmt.pct(v, 1); n.style.textAlign = "right";
    row.append(l, bar, n);
    box.appendChild(row);
  }
}

async function runBacktest() {
  const d = $("#bt-date").value;
  if (!d) return;
  $("#bt-run").disabled = true;
  try {
    state.forecast = await api(`/api/analyses/${state.aoiId}/forecast?issue_date=${d}`);
    renderForecast();
    if (state.mapLayer === "risk") renderMap();
  } catch (e) {
    alert(`Backtest failed: ${e.message}`);
  } finally {
    $("#bt-run").disabled = false;
  }
}

// ---------------------------------------------------------------------------------------------
// sensors & harmonization
// ---------------------------------------------------------------------------------------------
function renderSensors() {
  const h = state.data.harmonization, an = h.annual.filter((r) => r.harmonized_days > 300 || r.year === new Date().getFullYear());
  const s1 = css("--s1"), s2 = css("--s2"), s3 = css("--s3");
  const full = an.filter((r) => r.harmonized_days > 300);
  const x = full.map((r) => r.year);
  legend($("#annual-legend"), [{ label: "harmonized (VIIRS S-NPP equivalent)", color: s1 }, { label: "MODIS Terra + Aqua, raw", color: s2 },
    { label: "VIIRS S-NPP, raw", color: s3 }]);
  const firstV = full.find((r) => r.viirs_snpp_days > 300)?.year;
  seriesChart($("#annual-chart"), {
    x, height: 280, ariaLabel: "Yearly hotspot totals by sensor",
    xTicks: x.filter((y) => y % 2 === 0).map((y) => ({ at: y, label: String(y) })),
    spans: firstV ? [{ from: x[0] - 0.5, to: firstV - 0.5, color: css("--span"), label: "MODIS only: reconstructed" }] : [],
    lines: [{ values: full.map((r) => r.harmonized), color: s1 },
      { values: full.map((r) => (r.modis_days > 300 ? r.modis : null)), color: s2 },
      { values: full.map((r) => (r.viirs_snpp_days > 300 ? r.viirs_snpp : null)), color: s3 }],
    dots: [{ values: full.map((r) => r.harmonized), color: s1 }],
    tooltip: (i) => ({ title: String(x[i]), rows: [
      { value: fmt.int(full[i].harmonized), label: "harmonized", color: s1 },
      { value: fmt.int(full[i].modis), label: "MODIS raw", color: s2 },
      { value: full[i].viirs_snpp_days ? fmt.int(full[i].viirs_snpp) : "–", label: "VIIRS S-NPP raw", color: s3 },
      { value: full[i].viirs_noaa20_days ? fmt.int(full[i].viirs_noaa20) : "–", label: "VIIRS NOAA-20 raw" }] }),
  });
  // validation scatter (monthly)
  const v = h.validation, months = new Map();
  v.date.forEach((d, i) => {
    const k = d.slice(0, 7);
    if (!months.has(k)) months.set(k, { x: 0, y: 0, kind: v.kind[i], k });
    const m = months.get(k); m.x += v.observed[i]; m.y += v.harmonized[i];
  });
  const pts = [...months.values()];
  legend($("#val-legend"), [{ label: "cross-validation (stable orbits)", color: s1, kind: "dot" }, { label: "orbital-drift years", color: s2, kind: "dot" }]);
  scatterLog($("#val-chart"), {
    height: 300, xLabel: "VIIRS S-NPP observed (detections per month)", yLabel: "reconstructed from MODIS",
    groups: [{ color: s1, points: pts.filter((p) => p.kind === "cross-validation") }, { color: s2, points: pts.filter((p) => p.kind !== "cross-validation") }],
    tooltip: (p) => ({ title: p.k, rows: [{ value: fmt.int(p.x), label: "observed by VIIRS" }, { value: fmt.int(p.y), label: "reconstructed from MODIS" }] }),
  });
  const met = h.metrics, tb = $("#harm-table");
  tb.replaceChildren();
  const hd = tb.createTHead().insertRow();
  const obsNoFire = met.cv.harmonized.distribution?.no_fire_days_observed;
  for (const t of ["Method", "daily r", "monthly r (log)", "yearly r", "bias", "extremes found", `days w/o fire (obs ${fmt.pct(obsNoFire)})`]) {
    const th = document.createElement("th"); th.textContent = t; th.className = t === "Method" ? "" : "num"; hd.appendChild(th);
  }
  const rows = [["FireCal: expected value", met.cv.harmonized], ["FireCal: most likely day", met.cv.most_likely],
    ["  machine learning only", met.cv.ml_only], ["  monthly ratio only", met.cv.monthly_ratio],
    ["raw MODIS (no harmonization)", met.cv.raw_modis], ["Terra only (2000–2002 case)", met.cv.harmonized_terra_only]];
  if (met.drift_test) rows.push([`orbital drift ${met.drift_test.years[0]}–${met.drift_test.years[met.drift_test.years.length - 1]}`, met.drift_test.harmonized]);
  const b = tb.createTBody();
  for (const [name, s] of rows) {
    const tr = b.insertRow();
    if (!s) continue;
    for (const [val, n] of [[name], [fmt.num(s.daily?.r, 3), 1], [fmt.num(s.monthly?.r_log, 3), 1],
      [fmt.num(s.yearly?.r, 3), 1], [`${fmt.num(s.daily?.bias_pct, 1)}%`, 1], [fmt.pct(s.extreme_days_hit), 1],
      [s.distribution ? fmt.pct(s.distribution.no_fire_days) : "–", 1]]) {
      const td = tr.insertCell(); td.textContent = val; if (n) td.className = "num";
    }
  }
  const st = $("#sensor-table");
  st.replaceChildren();
  const sh = st.createTHead().insertRow();
  for (const t of ["Sensor", "First day", "Last day", "Days", "Role"]) { const th = document.createElement("th"); th.textContent = t; sh.appendChild(th); }
  const roles = { MODIS_T: "predictor, 2000–2012 reconstruction", MODIS_A: "predictor, 2002–2012 reconstruction",
    VIIRS_SNPP: "reference unit", VIIRS_NOAA20: `gap filler ×${fmt.num(met.ratio_noaa20_to_snpp, 2)}, predictor`,
    VIIRS_NOAA21: "gap filler, predictor" };
  const sb = st.createTBody();
  for (const [k, s] of Object.entries(state.data.meta.sensors)) {
    const tr = sb.insertRow();
    for (const v of [s.label, s.first || "–", s.last || "–", fmt.int(s.days), roles[k]]) { const td = tr.insertCell(); td.textContent = v; }
  }
}

// ---------------------------------------------------------------------------------------------
// tabs, theme, wiring
// ---------------------------------------------------------------------------------------------
function selectTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach((p) => { p.hidden = p.id !== `tab-${name}`; });
  state.tab = name;
  if (location.hash.slice(1) !== name) history.replaceState(null, "", `#${name}`);
  renderTab();
}
function renderTab() {
  if (!state.data) return;
  const t = state.tab || "calendar";
  if (t === "calendar") { renderCalendar(); renderMonthGrid(); }
  if (t === "season") renderSeason();
  if (t === "events") renderEvents();
  if (t === "forecast") renderForecast();
  if (t === "sensors") renderSensors();
}
function renderAll() {
  renderSummary();
  renderMap();
  renderTab();
}

function wire() {
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => selectTab(b.dataset.tab)));
  document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => {
    state.calMode = b.dataset.mode;
    document.querySelectorAll(".seg button").forEach((x) => x.setAttribute("aria-pressed", x === b));
    renderCalendar();
  }));
  const cv = $("#cal-canvas");
  cv.addEventListener("pointermove", (ev) => { const c = calCellAt(ev); if (c) calTooltip(c, ev.clientX, ev.clientY); else tooltip().hide(); });
  cv.addEventListener("pointerleave", () => tooltip().hide());
  cv.addEventListener("click", (ev) => { const c = calCellAt(ev); if (c) { state.year = c.year; state.event = null; renderCalendar(); } });
  cv.addEventListener("keydown", (ev) => {
    const ys = state.data?.calendar.years; if (!ys) return;
    let i = ys.indexOf(state.year);
    if (ev.key === "ArrowUp") i = Math.max(0, i - 1); else if (ev.key === "ArrowDown") i = Math.min(ys.length - 1, i + 1); else return;
    state.year = ys[i]; renderCalendar(); ev.preventDefault();
  });
  $("#year-select").addEventListener("change", (e) => { state.year = +e.target.value; renderSeason(); });
  $("#run-btn").addEventListener("click", runAnalysis);
  $("#draw-btn").addEventListener("click", () => setDrawing(!state.drawing));
  $("#map-layer").addEventListener("change", (e) => {
    const was = state.mapLayer;
    state.mapLayer = e.target.value;
    $("#lead-wrap").hidden = state.mapLayer !== "risk";
    $("#show-hotspots").parentElement.hidden = state.mapLayer === "world";
    if (state.mapLayer === "world") map.setView([25, 10], 2);
    else if (was === "world" && aoiLayer) map.fitBounds(aoiLayer.getBounds(), { padding: [20, 20] });
    renderMap();
  });
  $("#area-search").addEventListener("change", (e) => {
    const q = e.target.value.trim().toLowerCase();
    const a = (state.areas || []).find((x) => x.fits && x.name.toLowerCase() === q) ||
      (state.areas || []).find((x) => x.fits && x.name.toLowerCase().startsWith(q));
    if (a) { selectArea(a.key); e.target.value = ""; }
  });
  $("#map-lead").addEventListener("input", (e) => { state.lead = +e.target.value; $("#map-lead-out").textContent = `+${state.lead}`; renderMap(); });
  $("#show-hotspots").addEventListener("change", renderMap);
  $("#bt-run").addEventListener("click", runBacktest);
  $("#bt-reset").addEventListener("click", () => { state.forecast = null; renderForecast(); });
  $("#theme-toggle").addEventListener("click", () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("firecal-theme", root.dataset.theme); } catch { /* storage unavailable */ }
    if (aoiLayer) aoiLayer.setStyle({ color: css("--ink") });
    renderAll();
  });
  let rt;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => state.data && renderTab(), 150); });
}

async function main() {
  try { const t = localStorage.getItem("firecal-theme"); if (t) document.documentElement.dataset.theme = t; } catch { /* ignore */ }
  const qTheme = new URLSearchParams(location.search).get("theme");
  if (qTheme === "light" || qTheme === "dark") document.documentElement.dataset.theme = qTheme;
  const hashTab = location.hash.slice(1);
  if (document.querySelector(`.tabs button[data-tab="${hashTab}"]`)) selectTab(hashTab);
  initMap();
  wire();
  const st = await api("/api/status").catch(() => ({}));
  $("#key-status").textContent = st.firms_map_key ? "FIRMS API key: on" : "FIRMS API key: off (archive + last 7 days)";
  await loadPresets();
  const first = state.presets.find((p) => p.cached) || state.presets[0];
  if (first) {
    $("#preset-select").value = first.key;
    setBBox(first.bounds);
    showAoi(first.geojson);
    if (first.cached) await loadAnalysis(first.id).catch((e) => console.warn(e));
  }
  const today = new Date();
  $("#bt-date").max = today.toISOString().slice(0, 10);
  $("#bt-date").value = `${today.getFullYear() - 2}-08-01`;
}
main();
