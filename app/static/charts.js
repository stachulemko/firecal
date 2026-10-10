// Small dependency-free SVG/canvas chart helpers following the data-viz rules:
// thin marks, hairline solid grid, one y-axis, crosshair tooltips, text in ink tokens only.

const SVG = "http://www.w3.org/2000/svg";

export const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

export const fmt = {
  int: (v) => (v == null || !isFinite(v) ? "–" : Math.round(v).toLocaleString("en-US")),
  num: (v, d = 1) => {
    if (v == null || !isFinite(v)) return "–";
    const s = Number(v).toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: 0 });
    return s === "-0" ? "0" : s;  // -0.004 rounds to "-0"
  },
  compact: (v) => {
    if (v == null || !isFinite(v)) return "–";
    const a = Math.abs(v);
    if (a >= 1e6) return (v / 1e6).toFixed(1).replace(/\.0$/, "") + "M";
    if (a >= 1e4) return Math.round(v / 1e3) + "K";
    if (a >= 1e3) return (v / 1e3).toFixed(1).replace(/\.0$/, "") + "K";
    return a >= 10 ? Math.round(v).toString() : (Math.round(v * 10) / 10).toString();
  },
  pct: (v, d = 0) => (v == null || !isFinite(v) ? "–" : (100 * v).toFixed(d) + "%"),
};

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const LEAP_STARTS = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335];
export const mdLabel = (i) => {
  let m = 0;
  while (m < 11 && LEAP_STARTS[m + 1] <= i) m++;
  return `${MONTHS[m]} ${i - LEAP_STARTS[m] + 1}`;
};
export const MONTH_NAMES = MONTHS;
export const MONTH_STARTS = LEAP_STARTS;

// ---------------------------------------------------------------------------------------------
// tooltip (one shared element; content built with textContent - labels are untrusted data)
// ---------------------------------------------------------------------------------------------
let tipEl;
export function tooltip() {
  if (!tipEl) {
    tipEl = document.createElement("div");
    tipEl.className = "tooltip";
    tipEl.setAttribute("role", "status");
    document.body.appendChild(tipEl);
  }
  return {
    show(x, y, title, rows) {
      tipEl.replaceChildren();
      if (title) {
        const t = document.createElement("div");
        t.className = "tt-title";
        t.textContent = title;
        tipEl.appendChild(t);
      }
      for (const r of rows) {
        const row = document.createElement("div");
        row.className = "tt-row";
        if (r.color) {
          const k = document.createElement("span");
          k.className = r.key === "box" ? "tt-box" : "tt-key";
          k.style.background = r.color;
          row.appendChild(k);
        }
        const v = document.createElement("strong");
        v.textContent = r.value;
        row.appendChild(v);
        const l = document.createElement("span");
        l.className = "tt-label";
        l.textContent = r.label;
        row.appendChild(l);
        tipEl.appendChild(row);
      }
      tipEl.style.display = "block";
      const w = tipEl.offsetWidth, h = tipEl.offsetHeight;
      const px = x + 16 + w > window.innerWidth ? x - w - 16 : x + 16;
      const py = Math.min(Math.max(8, y - h / 2), window.innerHeight - h - 8);
      tipEl.style.left = `${px + window.scrollX}px`;
      tipEl.style.top = `${py + window.scrollY}px`;
    },
    hide() { if (tipEl) tipEl.style.display = "none"; },
  };
}

// ---------------------------------------------------------------------------------------------
// scales
// ---------------------------------------------------------------------------------------------
export function linear(d0, d1, r0, r1) {
  const f = (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
  f.invert = (p) => d0 + ((p - r0) / (r1 - r0 || 1)) * (d1 - d0);
  return f;
}

export function log1pScale(max, r0, r1) {
  const lm = Math.log1p(Math.max(max, 1));
  const f = (v) => r0 + (Math.log1p(Math.max(v, 0)) / lm) * (r1 - r0);
  f.ticks = () => {
    const out = [0];
    for (let p = 1; p <= max * 1.0001; p *= 10) out.push(p);
    if (out.length < 3) out.push(Math.round(max));
    return out;
  };
  return f;
}

function niceTicks(max, n = 4) {
  if (max <= 0) return [0];
  const step0 = max / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) || step0;
  const out = [];
  for (let v = 0; v <= max * 1.0001; v += step) out.push(+v.toPrecision(10));
  return out;
}

// ---------------------------------------------------------------------------------------------
// time-series chart: bands (washes), lines, dots, shaded spans, reference lines, crosshair
// ---------------------------------------------------------------------------------------------
/**
 * opts: {
 *   x: number[] (positions), xLabel(i) -> string, xTicks: [{at, label}],
 *   yScale: "linear"|"log1p", yMax?, yLabel,
 *   bands: [{name, lo[], hi[], color, opacity}], lines: [{name, values[], color, width, dash?}],
 *   dots: [{name, values[], color}], spans: [{from, to, label}], refs: [{y, label}],
 *   tooltip(i) -> {title, rows}, height
 * }
 */
export function seriesChart(container, opts) {
  container.replaceChildren();
  const W = Math.max(container.clientWidth, 280);
  const H = opts.height || 260;
  const m = { l: 52, r: 16, t: opts.spans?.some((s) => s.label) ? 22 : 10, b: 28 };
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": opts.ariaLabel || "chart" }, container);
  const xs = opts.x;
  const x = linear(xs[0], xs[xs.length - 1], m.l, W - m.r);
  let ymax = opts.yMax ?? 0;
  if (opts.yMax == null) {
    for (const b of opts.bands || []) for (const v of b.hi) if (v != null && v > ymax) ymax = v;
    for (const l of opts.lines || []) for (const v of l.values) if (v != null && v > ymax) ymax = v;
    for (const d of opts.dots || []) for (const v of d.values) if (v != null && v > ymax) ymax = v;
    for (const r of opts.refs || []) if (r.y > ymax) ymax = r.y;
  }
  ymax = Math.max(ymax, 1);
  const y = opts.yScale === "log1p" ? log1pScale(ymax * 1.1, H - m.b, m.t) : linear(0, ymax * 1.08, H - m.b, m.t);
  const ticks = opts.yScale === "log1p" ? y.ticks() : niceTicks(ymax * 1.05);

  // shaded spans (e.g. critical periods), behind everything
  for (const s of opts.spans || []) {
    const segs = s.from <= s.to ? [[s.from, s.to]] : [[s.from, xs[xs.length - 1]], [xs[0], s.to]];
    for (const [a, b] of segs) {
      el("rect", { x: x(a), y: m.t, width: Math.max(1, x(b) - x(a)), height: H - m.b - m.t, fill: s.color || css("--span") }, svg);
    }
    if (s.label) {
      const t = el("text", { x: x(segs[0][0]) + 4, y: m.t - 7, class: "chart-annot" }, svg);
      t.textContent = s.label;
    }
  }
  // grid + y axis
  for (const v of ticks) {
    const py = y(v);
    el("line", { x1: m.l, x2: W - m.r, y1: py, y2: py, class: "grid" }, svg);
    const t = el("text", { x: m.l - 8, y: py + 4, class: "tick", "text-anchor": "end" }, svg);
    t.textContent = fmt.compact(v);
  }
  el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }, svg);
  for (const tk of opts.xTicks || []) {
    const px = x(tk.at);
    if (px < m.l - 1 || px > W - m.r + 1) continue;
    el("line", { x1: px, x2: px, y1: H - m.b, y2: H - m.b + 4, class: "axis" }, svg);
    const t = el("text", { x: px, y: H - m.b + 17, class: "tick", "text-anchor": tk.anchor || "middle" }, svg);
    t.textContent = tk.label;
  }
  // bands
  for (const b of opts.bands || []) {
    let d = "", started = false;
    const pts = [];
    for (let i = 0; i < xs.length; i++) if (b.lo[i] != null && b.hi[i] != null) pts.push(i);
    if (!pts.length) continue;
    for (const i of pts) { d += `${started ? "L" : "M"}${x(xs[i]).toFixed(1)},${y(b.hi[i]).toFixed(1)}`; started = true; }
    for (const i of pts.slice().reverse()) d += `L${x(xs[i]).toFixed(1)},${y(b.lo[i]).toFixed(1)}`;
    el("path", { d: d + "Z", fill: b.color, "fill-opacity": b.opacity ?? 0.12, stroke: "none" }, svg);
  }
  // reference lines
  for (const r of opts.refs || []) {
    el("line", { x1: m.l, x2: W - m.r, y1: y(r.y), y2: y(r.y), class: "ref" }, svg);
    if (r.label) {
      const t = el("text", { x: W - m.r - 4, y: y(r.y) - 5, class: "chart-annot", "text-anchor": "end" }, svg);
      t.textContent = r.label;
    }
  }
  // lines (gaps where value is null)
  const endLabels = [];
  for (const l of opts.lines || []) {
    let d = "", pen = false;
    for (let i = 0; i < xs.length; i++) {
      const v = l.values[i];
      if (v == null || !isFinite(v)) { pen = false; continue; }
      d += `${pen ? "L" : "M"}${x(xs[i]).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
    }
    el("path", { d, fill: "none", stroke: l.color, "stroke-width": l.width || 2, "stroke-linejoin": "round",
      "stroke-linecap": "round", ...(l.dash ? { "stroke-dasharray": l.dash } : {}) }, svg);
    if (l.endLabel) {
      let last = -1;
      for (let i = xs.length - 1; i >= 0; i--) if (l.values[i] != null) { last = i; break; }
      if (last >= 0) endLabels.push({ px: x(xs[last]), py: y(l.values[last]) - 6, text: l.endLabel });
    }
  }
  // end labels: keep at least one line of text apart
  endLabels.sort((a, b) => a.py - b.py);
  for (let i = 1; i < endLabels.length; i++) endLabels[i].py = Math.max(endLabels[i].py, endLabels[i - 1].py + 14);
  for (const lb of endLabels) {
    const t = el("text", { x: Math.min(lb.px + 6, W - m.r - 2), y: Math.max(lb.py, m.t + 10), class: "chart-label",
      "text-anchor": lb.px > W - 80 ? "end" : "start" }, svg);
    t.textContent = lb.text;
  }
  for (const s of opts.dots || []) {
    for (let i = 0; i < xs.length; i++) {
      const v = s.values[i];
      if (v == null || !isFinite(v)) continue;
      el("circle", { cx: x(xs[i]), cy: y(v), r: 4.5, fill: s.color, stroke: css("--surface"), "stroke-width": 2 }, svg);
    }
  }
  // crosshair + tooltip
  if (opts.tooltip) {
    const tip = tooltip();
    const cross = el("line", { y1: m.t, y2: H - m.b, class: "crosshair", visibility: "hidden" }, svg);
    const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent",
      tabindex: 0 }, svg);
    let focusIdx = xs.length - 1;
    const showAt = (i, cx, cy) => {
      const px = x(xs[i]);
      cross.setAttribute("x1", px);
      cross.setAttribute("x2", px);
      cross.setAttribute("visibility", "visible");
      const tt = opts.tooltip(i);
      if (tt) tip.show(cx, cy, tt.title, tt.rows);
    };
    const nearest = (px) => {
      const v = x.invert(px);
      let best = 0, bd = Infinity;
      for (let i = 0; i < xs.length; i++) { const d = Math.abs(xs[i] - v); if (d < bd) { bd = d; best = i; } }
      return best;
    };
    hit.addEventListener("pointermove", (ev) => {
      const r = svg.getBoundingClientRect();
      focusIdx = nearest(ev.clientX - r.left);
      showAt(focusIdx, ev.clientX, ev.clientY);
    });
    hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); tip.hide(); });
    hit.addEventListener("keydown", (ev) => {
      if (ev.key !== "ArrowLeft" && ev.key !== "ArrowRight") return;
      focusIdx = Math.max(0, Math.min(xs.length - 1, focusIdx + (ev.key === "ArrowRight" ? 1 : -1)));
      const r = svg.getBoundingClientRect();
      showAt(focusIdx, r.left + x(xs[focusIdx]), r.top + m.t + 20);
      ev.preventDefault();
    });
    hit.addEventListener("blur", () => { cross.setAttribute("visibility", "hidden"); tip.hide(); });
  }
  return { svg, x, y };
}

// ---------------------------------------------------------------------------------------------
// column chart (one series, optional per-bar color), 4px rounded tops, <=24px wide
// ---------------------------------------------------------------------------------------------
export function columnChart(container, { labels, values, color, colors, yMax, height = 160, refs = [], tooltip: tt,
  valueFmt = fmt.compact, labelEvery = 1, ariaLabel }) {
  container.replaceChildren();
  const W = Math.max(container.clientWidth, 260), H = height;
  const m = { l: 44, r: 10, t: 12, b: 24 };
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": ariaLabel || "chart" }, container);
  const max = Math.max(yMax ?? Math.max(...values.map((v) => v ?? 0)), 1e-9);
  const y = linear(0, max * 1.08, H - m.b, m.t);
  for (const v of niceTicks(max * 1.02, 3)) {
    el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: "grid" }, svg);
    const t = el("text", { x: m.l - 6, y: y(v) + 4, class: "tick", "text-anchor": "end" }, svg);
    t.textContent = valueFmt(v);
  }
  const band = (W - m.l - m.r) / labels.length;
  const bw = Math.max(2, Math.min(24, band - 2));
  const tip = tooltip();
  labels.forEach((lab, i) => {
    const v = values[i];
    const cx = m.l + band * (i + 0.5);
    if (v != null && isFinite(v) && v > 0) {
      const top = y(v), h = H - m.b - top, r = Math.min(4, bw / 2, h);
      const x0 = cx - bw / 2;
      const d = `M${x0},${H - m.b}V${top + r}Q${x0},${top} ${x0 + r},${top}H${x0 + bw - r}Q${x0 + bw},${top} ${x0 + bw},${top + r}V${H - m.b}Z`;
      el("path", { d, fill: colors?.[i] || color, class: "bar" }, svg);
    }
    if (i % labelEvery === 0) {
      const t = el("text", { x: cx, y: H - m.b + 15, class: "tick", "text-anchor": "middle" }, svg);
      t.textContent = lab;
    }
    if (tt) {
      const hit = el("rect", { x: cx - band / 2, y: m.t, width: band, height: H - m.t - m.b, fill: "transparent", tabindex: 0 }, svg);
      const show = (cxp, cyp) => { const o = tt(i); tip.show(cxp, cyp, o.title, o.rows); };
      hit.addEventListener("pointermove", (ev) => show(ev.clientX, ev.clientY));
      hit.addEventListener("pointerleave", () => tip.hide());
      hit.addEventListener("focus", () => { const r = hit.getBoundingClientRect(); show(r.right, r.top + 20); });
      hit.addEventListener("blur", () => tip.hide());
    }
  });
  el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: "axis" }, svg);
  for (const r of refs) {
    el("line", { x1: m.l, x2: W - m.r, y1: y(r.y), y2: y(r.y), class: "ref" }, svg);
    const t = el("text", { x: W - m.r - 2, y: y(r.y) - 5, class: "chart-annot", "text-anchor": "end" }, svg);
    t.textContent = r.label;
  }
  return svg;
}

// ---------------------------------------------------------------------------------------------
// log-log scatter with 1:1 line and nearest-point hover
// ---------------------------------------------------------------------------------------------
export function scatterLog(container, { groups, height = 300, xLabel, yLabel, tooltip: tt }) {
  container.replaceChildren();
  const W = Math.max(container.clientWidth, 260), H = height;
  const m = { l: 52, r: 14, t: 12, b: 40 };
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${yLabel} vs ${xLabel}` }, container);
  let max = 1;
  for (const g of groups) for (const p of g.points) max = Math.max(max, p.x, p.y);
  const sx = log1pScale(max * 1.1, m.l, W - m.r), sy = log1pScale(max * 1.1, H - m.b, m.t);
  for (const v of sx.ticks()) {
    el("line", { x1: sx(v), x2: sx(v), y1: m.t, y2: H - m.b, class: "grid" }, svg);
    const t = el("text", { x: sx(v), y: H - m.b + 16, class: "tick", "text-anchor": "middle" }, svg);
    t.textContent = fmt.compact(v);
    el("line", { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), class: "grid" }, svg);
    const t2 = el("text", { x: m.l - 6, y: sy(v) + 4, class: "tick", "text-anchor": "end" }, svg);
    t2.textContent = fmt.compact(v);
  }
  el("line", { x1: sx(0), y1: sy(0), x2: sx(max * 1.1), y2: sy(max * 1.1), class: "ref" }, svg);
  const lab = el("text", { x: sx(max * 0.9), y: sy(max * 0.9) - 8, class: "chart-annot", "text-anchor": "end" }, svg);
  lab.textContent = "1:1";
  const xt = el("text", { x: (m.l + W - m.r) / 2, y: H - 6, class: "tick", "text-anchor": "middle" }, svg);
  xt.textContent = xLabel;
  const all = [];
  for (const g of groups) {
    for (const p of g.points) {
      el("circle", { cx: sx(p.x), cy: sy(p.y), r: 4, fill: g.color, "fill-opacity": 0.85, stroke: css("--surface"), "stroke-width": 1.5 }, svg);
      all.push({ ...p, g, px: sx(p.x), py: sy(p.y) });
    }
  }
  if (tt) {
    const tip = tooltip();
    const ring = el("circle", { r: 7, fill: "none", stroke: css("--ink"), "stroke-width": 1.5, visibility: "hidden" }, svg);
    const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", (ev) => {
      const r = svg.getBoundingClientRect();
      const px = ev.clientX - r.left, py = ev.clientY - r.top;
      let best = null, bd = 24 * 24;
      for (const p of all) { const d = (p.px - px) ** 2 + (p.py - py) ** 2; if (d < bd) { bd = d; best = p; } }
      if (!best) { tip.hide(); ring.setAttribute("visibility", "hidden"); return; }
      ring.setAttribute("cx", best.px); ring.setAttribute("cy", best.py); ring.setAttribute("visibility", "visible");
      const o = tt(best);
      tip.show(ev.clientX, ev.clientY, o.title, o.rows);
    });
    hit.addEventListener("pointerleave", () => { tip.hide(); ring.setAttribute("visibility", "hidden"); });
  }
  return svg;
}

// ---------------------------------------------------------------------------------------------
// colour ramps
// ---------------------------------------------------------------------------------------------
function hexToRgb(h) {
  const v = parseInt(h.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}
export function rampColor(stops, t) {
  t = Math.max(0, Math.min(1, t));
  const p = t * (stops.length - 1);
  const i = Math.min(Math.floor(p), stops.length - 2);
  const f = p - i;
  const a = hexToRgb(stops[i]), b = hexToRgb(stops[i + 1]);
  return `rgb(${Math.round(a[0] + (b[0] - a[0]) * f)},${Math.round(a[1] + (b[1] - a[1]) * f)},${Math.round(a[2] + (b[2] - a[2]) * f)})`;
}
export const heatStops = () => [1, 2, 3, 4, 5, 6].map((i) => css(`--heat-${i}`));
export const divStops = () => ["--div-n2", "--div-n1", "--div-mid", "--div-p1", "--div-p2", "--div-p3"].map(css);
