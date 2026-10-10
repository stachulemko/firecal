"""Command line interface: ``firecal run --preset portugal`` / ``firecal serve``."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time


def _progress():
    """Prints every new kind of message; repeats of one kind (e.g. download counters) at most twice a second."""
    last = {"t": 0.0, "kind": None}

    def f(frac: float, msg: str):
        now = time.time()
        kind = msg.split(" ")[:2]
        if kind != last["kind"] or now - last["t"] > 0.5 or frac >= 1.0:
            print(f"[{frac:6.1%}] {msg}", flush=True)
            last.update(t=now, kind=kind)
    return f


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="firecal", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="analyse an area of interest")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--preset", help="preset key (see 'firecal presets')")
    g.add_argument("--bbox", help="west,south,east,north in degrees")
    g.add_argument("--geojson", help="path to a GeoJSON polygon")
    r.add_argument("--name", default=None)
    r.add_argument("--force", action="store_true", help="ignore cached results")
    r.add_argument("--backtest", default=None, help="also issue a forecast from this past date (YYYY-MM-DD)")
    sub.add_parser("presets", help="list preset areas")
    s = sub.add_parser("serve", help="start the web application")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    from . import pipeline
    if a.cmd == "presets":
        for k, p in pipeline.PRESETS.items():
            print(f"{k:14s} {p['name']} - {p['note']}")
        return 0
    if a.cmd == "serve":
        import uvicorn
        uvicorn.run("app.main:app", host=a.host, port=a.port)
        return 0
    from .geo import AOI
    if a.preset:
        aoi = pipeline.preset_aoi(a.preset)
    elif a.bbox:
        aoi = AOI.from_bbox(*[float(v) for v in a.bbox.split(",")], name=a.name)
    else:
        aoi = AOI.from_geojson(json.load(open(a.geojson)), name=a.name)
    res = pipeline.run(aoi, progress=_progress(), force=a.force)
    m = res.result
    h = m["harmonization"]["metrics"]["cv"]["harmonized"]
    print(f"\n== {m['meta']['name']} ({m['meta']['data_start']} .. {m['meta']['data_end']})")
    print(f"harmonization (cross-validated, S-NPP vs MODIS-derived): daily r={h['daily']['r']:.3f}, "
          f"monthly r(log)={h['monthly']['r_log']:.3f}, bias {h['daily']['bias_pct']:+.1f}%")
    for p in m["calendar"]["critical_periods"]:
        print(f"critical period ({p['level']}): {p['start']} - {p['end']} ({p['share']:.0%} of activity)")
    for e in m["calendar"]["events"][:5]:
        print(f"event {e['start']}..{e['end']}: {e['total']:.0f} detections, x{e['ratio']:.1f} normal")
    for lead, s in m["forecast"]["daily_metrics"]["by_lead"].items():
        print(f"forecast lead {lead:>2}d (test {m['forecast']['daily_metrics']['test_years']}): cell AUC {s['cell_auc']:.3f}, "
              f"AOI skill vs climatology {s['aoi_skill_vs_climatology']:+.2f}, critical-day AUC "
              f"{'n/a' if s['critical_auc'] is None else format(s['critical_auc'], '.2f')}")
    op = m["forecast"]["operational"]
    if op:
        print(f"operational forecast issued {op['issue_date']}:")
        for d in op["days"][:7]:
            fw = d["fire_weather"]
            print(f"  {d['date']}: median {d['median']:.0f} (80%: {d['q10']:.0f}-{d['q90']:.0f}), "
                  f"normal {d['normal']:.0f}, P(critical) {d['p_critical']:.2f}, FWI {fw['fwi']:.0f} ({fw['fwi_class']})")
    if a.backtest:
        bt = res.forecast_from(a.backtest)
        print(f"{bt['mode']} from {bt['issue_date']} (model trained until {bt['model_trained_until']}):")
        for d in bt["days"]:
            print(f"  {d['date']}: median {d['median']:.0f}, P(critical) {d['p_critical']:.2f}, observed {d['observed']}, "
                  f"FWI {d['fire_weather']['fwi']:.0f} ({d['fire_weather']['fwi_class']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
