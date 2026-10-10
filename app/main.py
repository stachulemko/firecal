"""FastAPI web application: analysis jobs, results and the single-page front end."""
from __future__ import annotations

import io
import json
import logging
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from shapely.geometry import mapping

from firecal import pipeline, world
from firecal.config import AOI_DIR, GRID_RESOLUTIONS, MAX_GRID_CELLS, firms_map_key
from firecal.geo import AOI

log = logging.getLogger("firecal.app")
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="FireCal - burning activity calendar", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_executor = ThreadPoolExecutor(max_workers=1)  # one heavy analysis at a time; others queue
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_analyses: "OrderedDict[str, pipeline.Analysis]" = OrderedDict()
_analyses_lock = threading.Lock()
MAX_LOADED = 4


class AnalysisRequest(BaseModel):
    preset: str | None = None
    area: str | None = None  # a country or region of the world (see /api/areas)
    bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    geojson: dict | None = None
    name: str | None = None
    force: bool = False


def _remember(a: pipeline.Analysis) -> None:
    with _analyses_lock:
        _analyses[a.aoi.id] = a
        _analyses.move_to_end(a.aoi.id)
        while len(_analyses) > MAX_LOADED:
            _analyses.popitem(last=False)


def _get(aoi_id: str) -> pipeline.Analysis:
    with _analyses_lock:
        a = _analyses.get(aoi_id)
    if a is None:
        a = pipeline.Analysis.load(aoi_id)
        if a is None:
            raise HTTPException(404, "analysis not found - run it first")
        _remember(a)
    return a


def _aoi_from(req: AnalysisRequest) -> AOI:
    if req.preset:
        if req.preset not in pipeline.PRESETS:
            raise HTTPException(400, f"unknown preset {req.preset}")
        return pipeline.preset_aoi(req.preset)
    if req.area:
        try:
            a = world.area(req.area)
        except KeyError:
            raise HTTPException(400, f"unknown area {req.area}")
        if not a.fits:
            raise HTTPException(400, f"{a.name} is too large to analyse in one piece: pick one of its regions "
                                     "or draw a rectangle")
        aoi = a.aoi()
        for k in pipeline.PRESETS:  # the same land as an example area: share its cached analysis
            p = pipeline.preset_aoi(k)
            if p.geometry.equals(aoi.geometry):
                return p
        return aoi
    try:
        if req.bbox:
            w, s, e, n = req.bbox
            cells = (e - w) / GRID_RESOLUTIONS[-1] * (n - s) / GRID_RESOLUTIONS[-1]
            if cells > MAX_GRID_CELLS:
                raise HTTPException(400, "area too large: keep it under ~1600 square degrees (e.g. 40 x 40)")
            return AOI.from_bbox(w, s, e, n, name=req.name)
        if req.geojson:
            return AOI.from_geojson(req.geojson, name=req.name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    raise HTTPException(400, "give a preset, an area, a bbox or a GeoJSON polygon")


def _run_job(job_id: str, aoi: AOI, force: bool) -> None:
    job = _jobs[job_id]
    job["status"] = "running"
    job["started"] = time.time()

    def progress(frac: float, msg: str) -> None:
        job["progress"] = round(float(frac), 4)
        job["message"] = msg
        if not job["log"] or job["log"][-1] != msg:
            job["log"].append(msg)
            del job["log"][:-60]

    try:
        a = pipeline.run(aoi, progress=progress, force=force)
        _remember(a)
        job.update(status="done", progress=1.0, aoi_id=a.aoi.id)
    except Exception as e:  # report to the client, keep the server alive
        log.exception("analysis failed")
        job.update(status="error", error=str(e), trace=traceback.format_exc()[-2000:])


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status():
    return {"firms_map_key": bool(firms_map_key()), "loaded": list(_analyses)}


@app.get("/api/presets")
def presets():
    out = []
    for k, p in pipeline.PRESETS.items():
        aoi = pipeline.preset_aoi(k)
        cached = (AOI_DIR / aoi.id / "result.json").exists()
        out.append({"key": k, "name": p["name"], "note": p["note"], "bounds": aoi.bounds, "id": aoi.id,
                    "cached": cached, "geojson": aoi.to_geojson()})
    return out


@app.get("/api/areas")
def list_areas():
    """Every country of the world with a FIRMS archive, and the regions of the largest ones."""
    return world.listing()


@app.get("/api/area")
def get_area(key: str):
    try:
        a = world.area(key)
    except KeyError:
        raise HTTPException(404, "unknown area")
    geom = a.geometry.simplify(0.01, preserve_topology=True)
    return {"key": a.key, "name": a.name, "kind": a.kind, "group": a.group, "bounds": a.bounds, "fits": a.fits,
            "geojson": {"type": "Feature", "properties": {"name": a.name}, "geometry": mapping(geom)}}


@app.get("/api/area_at")
def get_area_at(lat: float, lon: float):
    a = world.area_at(lat, lon)
    if a is None:
        raise HTTPException(404, "no country here")
    return {"key": a.key, "name": a.name, "fits": a.fits}


@app.get("/api/world/recent")
def world_recent():
    """VIIRS fire detections of the last 7 days on a 1-degree global grid."""
    try:
        return world.recent_world()
    except RuntimeError as e:
        raise HTTPException(503, str(e))


@app.get("/api/analyses")
def list_analyses():
    out = []
    for d in sorted(AOI_DIR.glob("*/result.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            meta = json.loads(d.read_text())["meta"]
            out.append({"id": meta["id"], "name": meta["name"], "created": meta["created"],
                        "bounds": meta["bounds"]})
        except Exception:
            continue
    return out


@app.post("/api/analyses")
def start_analysis(req: AnalysisRequest):
    aoi = _aoi_from(req)
    with _jobs_lock:
        for j in _jobs.values():  # same area already queued/running -> reuse that job
            if j["aoi_id"] == aoi.id and j["status"] in ("queued", "running"):
                return {"job_id": j["id"], "aoi_id": aoi.id}
        job_id = uuid.uuid4().hex[:10]
        _jobs[job_id] = {"id": job_id, "status": "queued", "progress": 0.0, "message": "queued", "log": [],
                         "aoi_id": aoi.id, "name": aoi.name, "created": time.time()}
    _executor.submit(_run_job, job_id, aoi, req.force)
    return {"job_id": job_id, "aoi_id": aoi.id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return {k: v for k, v in job.items() if k != "trace"}


@app.get("/api/analyses/{aoi_id}")
def get_analysis(aoi_id: str):
    return JSONResponse(_get(aoi_id).result)


@app.get("/api/analyses/{aoi_id}/forecast")
def forecast_from(aoi_id: str, issue_date: str):
    a = _get(aoi_id)
    try:
        return a.forecast_from(issue_date)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/analyses/{aoi_id}/daily.csv")
def daily_csv(aoi_id: str):
    a = _get(aoi_id)
    cols = ["date", "y", "y_ml", "y_lo", "y_hi", "source", "normal", "pct7", "anomaly", "n_MODIS_T", "n_MODIS_A",
            "n_VIIRS_SNPP", "n_VIIRS_NOAA20", "n_VIIRS_NOAA21", "frp_viirs", "frp_modis"]
    df = a.daily[[c for c in cols if c in a.daily.columns]].copy()
    df = df.rename(columns={"y": "activity_snpp_equivalent", "y_ml": "activity_most_likely_day",
                            "y_lo": "activity_p10", "y_hi": "activity_p90",
                            "normal": "normal_mean", "pct7": "percentile_7day"})
    buf = io.StringIO()
    df.to_csv(buf, index=False, float_format="%.3f")
    buf.seek(0)
    name = f"firecal_{aoi_id}_daily.csv"
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename={name}"})
