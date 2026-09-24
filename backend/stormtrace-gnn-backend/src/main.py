"""
Alerting API (blueprint deliverable #4) + pipeline trigger + historical replay.

The pipeline (mesh build + GNN training + diffusion training/sampling) takes
real wall-clock time, so /pipeline/run does NOT block the request: it enqueues
a background job and returns a job_id immediately, matching the blueprint's
"Diffusion inference latency" guardrail ("keep the live path optional and
clearly labelled as slower"). Historical Replay Mode uses a FIXED seed per
named event so the same "Cyclone Amphan" replay is reproducible and offline-
safe for judging, exactly as blueprint section 11 specifies.
"""
import uuid
import threading
import traceback
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from src.db import init_db, get_latest_alerts, get_alert, get_metrics
from src.pipeline import run_full_pipeline
from src.schemas import PipelineRunRequest, JobStatus

app = FastAPI(
    title="STORMTRACE-GNN Backend",
    description="Spherical-mesh GNN anomaly tracking + physics-constrained diffusion downscaling API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# in-memory job registry. Fine for a single-process demo/API; swap for Redis
# if this is ever run with multiple uvicorn workers sharing job state.
_JOBS = {}
_JOBS_LOCK = threading.Lock()

# fixed seeds -> reproducible, offline-safe "historical replay" cases
HISTORICAL_REPLAY_SEEDS = {
    "cyclone_amphan": 2020,
    "north_india_heatwave": 2022,
}


@app.on_event("startup")
def startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "backend is working", "version": app.version}


def _execute_job(job_id: str, event_name: str, seed: Optional[int]):
    with _JOBS_LOCK:
        _JOBS[job_id]["status"] = "running"

    def progress(msg: str):
        with _JOBS_LOCK:
            _JOBS[job_id]["progress"] = msg

    try:
        result = run_full_pipeline(event_name=event_name, seed=seed, progress_cb=progress)
        with _JOBS_LOCK:
            _JOBS[job_id]["status"] = "done"
            _JOBS[job_id]["result"] = result
    except Exception as e:
        with _JOBS_LOCK:
            _JOBS[job_id]["status"] = "failed"
            _JOBS[job_id]["error"] = f"{e}\n{traceback.format_exc()}"


@app.post("/pipeline/run", response_model=JobStatus)
def trigger_pipeline(req: PipelineRunRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    with _JOBS_LOCK:
        _JOBS[job_id] = {"status": "queued", "progress": "", "result": None, "error": None}
    background_tasks.add_task(_execute_job, job_id, req.event_name, req.seed)
    return JobStatus(job_id=job_id, status="queued")


@app.get("/pipeline/status/{job_id}", response_model=JobStatus)
def pipeline_status(job_id: str):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatus(job_id=job_id, **job)


@app.post("/pipeline/run_sync")
def trigger_pipeline_sync(req: PipelineRunRequest):
    """Blocking variant for quick local testing / curl. Prefer /pipeline/run for real clients."""
    try:
        return run_full_pipeline(event_name=req.event_name, seed=req.seed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/replay/{event_key}")
def historical_replay(event_key: str, background_tasks: BackgroundTasks):
    """Offline-safe judging path: known historical event -> fixed seed -> reproducible run."""
    if event_key not in HISTORICAL_REPLAY_SEEDS:
        raise HTTPException(
            status_code=404,
            detail=f"unknown replay event. Available: {list(HISTORICAL_REPLAY_SEEDS.keys())}",
        )
    job_id = str(uuid.uuid4())
    with _JOBS_LOCK:
        _JOBS[job_id] = {"status": "queued", "progress": "", "result": None, "error": None}
    background_tasks.add_task(_execute_job, job_id, event_key, HISTORICAL_REPLAY_SEEDS[event_key])
    return JobStatus(job_id=job_id, status="queued")


@app.get("/alerts/latest")
def latest_alerts(
    limit: int = Query(10, ge=1, le=200),
    severity: Optional[str] = Query(None, pattern="^(low|moderate|severe)$"),
    event_name: Optional[str] = None,
):
    return get_latest_alerts(limit=limit, severity=severity, event_name=event_name)


@app.get("/alerts/{alert_id}")
def alert_by_id(alert_id: int):
    alert = get_alert(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="alert not found")
    return alert


@app.get("/metrics")
def metrics():
    """Verification card per blueprint section 11 step 7: aggregate stats across all
    alerts, e.g. average diffusion-vs-U-Net amplitude gain and physics score."""
    return get_metrics()


@app.get("/replay/events")
def list_replay_events():
    return {"available_events": list(HISTORICAL_REPLAY_SEEDS.keys())}
