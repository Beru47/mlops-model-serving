"""FastAPI service: loads the Production alias from the registry, exposes /predict and /metrics."""

import asyncio
import os
import time
from collections import deque
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from mlflow.tracking import MlflowClient
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel, Field
from threadpoolctl import threadpool_limits

from . import config as C
from .drift import drift_report

REQS = Counter("api_requests_total", "Requests", ["endpoint", "status"])
LAT = Histogram(
    "api_request_latency_seconds",
    "Request latency",
    ["endpoint"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
)
PRED = Histogram(
    "model_prediction_value",
    "Distribution of predictions",
    buckets=(0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7, 10),
)
DRIFT_PSI = Gauge("feature_drift_psi", "PSI vs training reference", ["feature"])
DRIFT_KS_P = Gauge("feature_drift_ks_pvalue", "KS p-value vs training reference", ["feature"])
DRIFT_FLAG = Gauge("drift_detected", "1 if any feature drifted (PSI and KS agree)")
MODEL_VERSION = Gauge("model_version", "Registry version served")

WINDOW = int(os.environ.get("DRIFT_WINDOW", "1000"))
MIN_ROWS = int(os.environ.get("DRIFT_MIN_ROWS", "200"))
INTERVAL = float(os.environ.get("DRIFT_INTERVAL_S", "15"))


class House(BaseModel):
    med_inc: float = Field(ge=0, le=100)
    house_age: float = Field(ge=0, le=150)
    avg_rooms: float = Field(gt=0, le=100)
    population: float = Field(ge=0, le=1e7)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class Batch(BaseModel):
    rows: list[House] = Field(min_length=1, max_length=10_000)


class State:
    model = None
    version = 0
    reference: pd.DataFrame | None = None
    window: deque = deque(maxlen=WINDOW)


S = State()


def load_production() -> None:
    # single-row inference: OpenMP thread fan-out costs far more than it saves
    threadpool_limits(1)
    mlflow.set_tracking_uri(C.tracking_uri())
    mv = MlflowClient().get_model_version_by_alias(C.model_name(), C.PRODUCTION)
    S.model = mlflow.sklearn.load_model(f"models:/{C.model_name()}@{C.PRODUCTION}")
    path = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="reference.csv")
    S.reference = pd.read_csv(path)
    S.version = int(mv.version)
    S.window.clear()
    MODEL_VERSION.set(S.version)


def run_drift_check() -> dict | None:
    if S.reference is None or len(S.window) < MIN_ROWS:
        return None
    rep = drift_report(S.reference, pd.DataFrame(list(S.window), columns=C.FEATURES))
    for f, m in rep["features"].items():
        DRIFT_PSI.labels(f).set(m["psi"])
        DRIFT_KS_P.labels(f).set(m["ks_pvalue"])
    DRIFT_FLAG.set(int(rep["drift_detected"]))
    rep["window_rows"] = len(S.window)
    return rep


async def _drift_loop() -> None:
    while True:
        await asyncio.sleep(INTERVAL)
        await asyncio.to_thread(run_drift_check)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_production()
    task = asyncio.create_task(_drift_loop())
    yield
    task.cancel()


app = FastAPI(title="mlops-model-serving", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def observe(request: Request, call_next):
    ep = request.url.path
    if ep.startswith("/metrics"):
        return await call_next(request)
    t0 = time.perf_counter()
    status = "500"
    try:
        resp = await call_next(request)
        status = str(resp.status_code)
        return resp
    finally:
        LAT.labels(ep).observe(time.perf_counter() - t0)
        REQS.labels(ep, status).inc()


def _score(rows: list[House]) -> list[float]:
    df = pd.DataFrame([r.model_dump() for r in rows])[C.FEATURES]
    preds = [float(p) for p in S.model.predict(df)]
    S.window.extend(df.itertuples(index=False, name=None))
    for p in preds:
        PRED.observe(p)
    return preds


@app.get("/health")
def health():
    return {"status": "ok", "model_version": S.version}


@app.post("/predict")
def predict(house: House):
    return {"prediction": _score([house])[0], "model_version": S.version}


@app.post("/predict/batch")
def predict_batch(batch: Batch):
    return {"predictions": _score(batch.rows), "model_version": S.version}


@app.get("/drift")
def drift():
    rep = run_drift_check()
    if rep is None:
        raise HTTPException(409, f"need at least {MIN_ROWS} live rows, have {len(S.window)}")
    return rep


@app.post("/reload")
def reload():
    """Pick up a newly promoted Production version without restarting."""
    load_production()
    return {"model_version": S.version}
