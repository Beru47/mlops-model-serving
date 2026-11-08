"""Train -> evaluate -> conditional promote, on top of the MLflow Model Registry (aliases)."""

import argparse
import json

import mlflow
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from . import config as C
from .data import make_data
from .promotion import should_promote

EXPERIMENT = "housing-regressor"


def _client() -> MlflowClient:
    mlflow.set_tracking_uri(C.tracking_uri())
    mlflow.set_experiment(EXPERIMENT)
    return MlflowClient()


def holdout() -> pd.DataFrame:
    """Fixed, drift-free holdout shared by every candidate and by Production."""
    return make_data(C.HOLDOUT_SIZE, C.HOLDOUT_SEED)


def rmse(model, df: pd.DataFrame) -> float:
    return float(np.sqrt(mean_squared_error(df[C.TARGET], model.predict(df[C.FEATURES]))))


def train_candidate(
    run_key: str | None = None, seed: int = 0, n: int = 5000, shift: float = 0.0
) -> int:
    """Train, log, register and set alias `staging`. Idempotent per run_key: a retried task
    returns the version already registered for that key instead of training again."""
    client = _client()
    name = C.model_name()
    if run_key:
        try:
            for mv in client.search_model_versions(f"name='{name}'"):
                if mv.tags.get("run_key") == run_key:
                    return int(mv.version)
        except MlflowException:
            pass  # registered model does not exist yet
    train = make_data(n, seed, shift)
    model = HistGradientBoostingRegressor(random_state=seed, max_iter=200).fit(
        train[C.FEATURES], train[C.TARGET]
    )
    with mlflow.start_run():
        mlflow.log_params(
            {"seed": seed, "n_train": n, "shift": shift, "model": "HistGradientBoosting"}
        )
        mlflow.log_metric("holdout_rmse", rmse(model, holdout()))
        sig = infer_signature(train[C.FEATURES], model.predict(train[C.FEATURES]))
        info = mlflow.sklearn.log_model(
            model, name="model", signature=sig, serialization_format="cloudpickle"
        )
        # Reference distribution for drift monitoring travels with the model version's run.
        train[C.FEATURES].sample(min(n, 2000), random_state=0).to_csv("reference.csv", index=False)
        mlflow.log_artifact("reference.csv")
    mv = mlflow.register_model(info.model_uri, name)
    if run_key:
        client.set_model_version_tag(name, mv.version, "run_key", run_key)
    client.set_registered_model_alias(name, C.STAGING, mv.version)
    return int(mv.version)


def production_version() -> int | None:
    try:
        v = _client().get_model_version_by_alias(C.model_name(), C.PRODUCTION).version
        return int(v)
    except MlflowException:
        return None


def evaluate(version: int) -> dict:
    """Score the candidate and the current Production model on the SAME holdout."""
    test, name = holdout(), C.model_name()
    cand = mlflow.sklearn.load_model(f"models:/{name}/{version}")
    prod_v = production_version()
    prod = mlflow.sklearn.load_model(f"models:/{name}/{prod_v}") if prod_v else None
    return {
        "candidate_version": version,
        "candidate_rmse": rmse(cand, test),
        "production_version": prod_v,
        "production_rmse": rmse(prod, test) if prod else None,
    }


def promote_if_better(version: int, min_improvement: float = 0.0) -> dict:
    """Move alias `production` to `version` only if the gate passes. Idempotent."""
    res = evaluate(version)
    res["promoted"] = res["production_version"] == version  # retry of an earlier success
    if not res["promoted"] and should_promote(
        res["candidate_rmse"], res["production_rmse"], min_improvement
    ):
        _client().set_registered_model_alias(C.model_name(), C.PRODUCTION, version)
        res["promoted"] = True
    return res


def main() -> None:
    p = argparse.ArgumentParser(description="Train a candidate; promote it if it beats Production")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--shift", type=float, default=0.0)
    p.add_argument("--run-key")
    p.add_argument("--min-improvement", type=float, default=0.0)
    a = p.parse_args()
    v = train_candidate(a.run_key, a.seed, a.n, a.shift)
    print(json.dumps(promote_if_better(v, a.min_improvement), indent=2))


if __name__ == "__main__":
    main()
