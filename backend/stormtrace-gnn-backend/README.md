# STORMTRACE-GNN — Backend

A working implementation of the two-stage STORMTRACE-GNN pipeline from the
blueprint: **spherical-mesh GNN anomaly tracking** (Stage 1) feeding a
**physics-constrained conditional diffusion downscaler** (Stage 2), wired
end-to-end behind a FastAPI alerting API with Postgres persistence.

Uses realistic **synthetic** multivariable ensemble data (same shortcut the
blueprint itself recommends) so the entire pipeline runs today with zero
external downloads. Swap `src/data_gen.py` for a real GRIB2/NetCDF loader
later — every file from `mesh.py` onward is unchanged.

## What's actually implemented (every pipeline stage, not a subset)

| Stage | File | Detail |
|---|---|---|
| Synthetic ensemble | `src/data_gen.py` | 4 variables (t2m, rainfall, wind u/v), 8 forecast steps (D+3..D+10), an anomaly that physically advects across the domain over time, ensemble spread that grows with member index like a real EPS |
| Spherical mesh | `src/mesh.py` | Icosahedral mesh (configurable subdivision), vectorized grid↔mesh reprojection (both directions, so round-trip fidelity is unit-testable), fully vectorized sparse adjacency build with GCN-style symmetric normalization |
| EFI | `src/efi.py` | Per-mesh-point percentile rank against that point's own 30-year baseline distribution (vectorized broadcast, not a python loop, and not collapsed to one global scalar) |
| Stage 1 — GNN | `src/gnn.py` | Multi-layer sparse message-passing GNN (`torch.sparse.mm`, not a dense P×P matmul), residual connections + LayerNorm, class-balanced loss (anomalies are rare) |
| 4D tracking | `src/tracking.py` | Per-timestep connected-component clustering on the anomalous subgraph + Hungarian (`linear_sum_assignment`) cross-timestep cluster linking → one 4D (x,y,z,t) bounding box per tracked anomaly |
| Baseline downscaler | `src/unet.py` | CNN/U-Net trained to minimize pixel MSE — deliberately included as the spectral-smoothing benchmark the diffusion model is measured against |
| Stage 2 — Diffusion | `src/diffusion.py` | Real DDPM: linear beta schedule, closed-form `q_sample`, sinusoidal timestep embeddings, ancestral sampling with correct posterior variance, conditioned on the coarse field, z-score normalized (so the reverse process actually recovers real-world amplitude instead of collapsing to the noise prior's scale) |
| Physics constraints | `src/physics.py` | A differentiable moisture-convergence penalty used **inside** the diffusion training loss (not just post-hoc), plus a post-hoc ensemble-agreement consistency score |
| Severity | `src/severity.py` | Calibrated blend of amplitude, EFI, and physics score; a "severe" call with low physics confidence is automatically downgraded, so the alert layer never over-calls off an unsupported generative peak |
| Persistence | `src/db.py` | Postgres via SQLAlchemy, `events` + `alerts` tables (event-based, so historical-replay holdouts are a clean query), connection pooling |
| API | `src/main.py` | FastAPI; pipeline runs as a background job (`/pipeline/run` returns immediately, poll `/pipeline/status/{id}`), fixed-seed offline-safe historical replay (`/replay/cyclone_amphan`, `/replay/north_india_heatwave`), alerts listing/filtering, aggregate `/metrics` |

## Efficiency choices ("efficiently heavy")

- **Sparse everywhere on the mesh.** Adjacency is a scipy CSR → torch sparse
  COO; every GNN layer is `torch.sparse.mm`, not a dense matrix multiply.
  At the default `MESH_SUBDIVISIONS=4` (2562 nodes) a dense adjacency would
  be a 2562×2562 matmul per layer for no reason — sparse costs `O(nnz)`.
- **Mesh built once, cached.** The icosahedral mesh and its adjacency don't
  depend on forecast data, so `pipeline.py` builds them once per process
  (`_MESH_CACHE`) instead of on every request.
- **Vectorized EFI and reprojection.** No python loops over mesh points or
  ensemble years — single broadcasted numpy comparisons / one KDTree query.
- **Background jobs, not blocking requests.** Training + diffusion sampling
  take real wall-clock time; `/pipeline/run` enqueues and returns a `job_id`
  immediately rather than holding the HTTP connection open.
- **Stage 1 / Stage 2 stay separate models with a fixed interface** (the
  bounding-box crop) per the blueprint's own feasibility guardrail — no
  expensive joint end-to-end training.

## Running it

```bash
# 1. start Postgres
docker compose up -d db

# 2. install deps
pip install -r requirements.txt

# 3. run the API (auto-creates tables on startup)
uvicorn src.main:app --reload --port 8000
```

Or the whole stack in containers:

```bash
docker compose up --build
```

### Try it

```bash
curl http://127.0.0.1:8000/health

# offline-safe historical replay (fixed seed, reproducible)
curl -X POST http://127.0.0.1:8000/replay/cyclone_amphan
curl http://127.0.0.1:8000/pipeline/status/<job_id>

# a fresh live run with a random seed
curl -X POST http://127.0.0.1:8000/pipeline/run \
  -H "Content-Type: application/json" -d '{"event_name": "live_run"}'

curl http://127.0.0.1:8000/alerts/latest
curl "http://127.0.0.1:8000/alerts/latest?severity=severe"
curl http://127.0.0.1:8000/metrics
```

`/pipeline/run_sync` runs the pipeline inline (blocking) if you just want to
`curl` a full result in one shot during local dev — it's slower and not
meant for production clients.

## Tests

```bash
pytest -q
```

Covers the guardrails named explicitly in the blueprint's Test Strategy
section: mesh reprojection round-trip fidelity, EFI against a known
distribution, and the physics penalty correctly ranking an unsupported peak
worse than a supported one.

## Config

Every size knob (`GRID_SIZE`, `MESH_SUBDIVISIONS`, `N_ENSEMBLE_MEMBERS`,
`N_FORECAST_STEPS`, `DIFFUSION_STEPS`, etc.) is in `config.py` / `.env` —
copy `.env.example` to `.env` and scale up once the default (student-laptop,
fast-iteration) sizing is confirmed working end-to-end.

## What's simplified, and how to upgrade later

- `data_gen.py` fakes NEPS-G/ERA5 — only this file changes when you point at
  real GRIB2/NetCDF; everything downstream is untouched.
- `mesh.py`'s reprojection uses nearest-neighbour KDTree matching on a
  synthetic flat grid rather than true lat/lon geodesic interpolation — swap
  in real lat/lon coordinates and this still works unchanged.
- `physics.py` implements one conservation check (moisture convergence), as
  the blueprint itself recommends starting with; add the full fluid-dynamics
  set (via MetPy) once this is stable.
- The in-memory job registry in `main.py` is fine for a single process; swap
  for Redis/Celery if you run multiple uvicorn workers.
