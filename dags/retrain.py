"""Airflow DAGs: weekly retrain, plus an hourly drift check that triggers the retrain DAG.

Tasks are thin wrappers over mlops.pipeline and idempotent: the run key is the Airflow
run_id, so a retried task reuses the already registered version and promotion is a no-op
if the version is already Production.
"""
import json
import os
import zlib
import urllib.request
from datetime import datetime

from airflow.decorators import dag, task
from airflow.operators.python import ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

API = os.environ.get("MODEL_API_URL", "http://api:8000")


@dag(schedule="@weekly", start_date=datetime(2025, 1, 1), catchup=False, tags=["mlops"])
def retrain():
    @task
    def train(run_id=None, dag_run=None) -> int:
        from mlops.pipeline import train_candidate

        conf = (dag_run.conf if dag_run else None) or {}
        # fresh seed per run so retraining sees a new sample; same run_id -> same version
        seed = int(conf.get("seed", zlib.crc32(str(run_id).encode()) % 10_000))
        return train_candidate(
            run_key=run_id, seed=seed, n=int(conf.get("n", 5000)), shift=float(conf.get("shift", 0))
        )

    @task
    def evaluate(version: int) -> dict:
        from mlops.pipeline import evaluate as ev

        return ev(version)

    @task
    def promote(version: int, metrics: dict) -> dict:
        # evaluate output is logged for the run; promote re-checks the gate itself (idempotent)
        from mlops.pipeline import promote_if_better

        return promote_if_better(version)

    @task
    def reload_api(result: dict) -> None:
        if result["promoted"]:
            urllib.request.urlopen(urllib.request.Request(f"{API}/reload", method="POST"))

    v = train()
    m = evaluate(v)
    reload_api(promote(v, m))


retrain()


@dag(schedule="@hourly", start_date=datetime(2025, 1, 1), catchup=False, tags=["mlops"])
def drift_check():
    def _drifted() -> bool:
        try:
            with urllib.request.urlopen(f"{API}/drift") as r:
                return bool(json.load(r)["drift_detected"])
        except Exception:  # 409 (not enough live rows) or API down -> do not retrain
            return False

    gate = ShortCircuitOperator(task_id="drift_detected", python_callable=_drifted)
    gate >> TriggerDagRunOperator(task_id="trigger_retrain", trigger_dag_id="retrain")


drift_check()
