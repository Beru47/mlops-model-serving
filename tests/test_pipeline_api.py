import pytest
from fastapi.testclient import TestClient

from mlops import pipeline, serve
from mlops.config import FEATURES
from mlops.data import make_data

GOOD = make_data(1, 5)[FEATURES].iloc[0].to_dict()


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    import os

    d = tmp_path_factory.mktemp("mlflow")
    old = os.getcwd()
    os.chdir(d)  # artifacts + reference.csv land in the temp dir
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{(d / 'mlflow.db').as_posix()}"
    yield d
    os.chdir(old)


def test_first_candidate_is_promoted(registry):
    v = pipeline.train_candidate(run_key="t1", seed=0, n=1500)
    assert pipeline.promote_if_better(v)["promoted"]
    assert pipeline.production_version() == v


def test_train_is_idempotent_per_run_key(registry):
    assert pipeline.train_candidate(run_key="t1") == pipeline.production_version()


def test_worse_candidate_is_rejected(registry):
    prod = pipeline.production_version()
    weak = pipeline.train_candidate(run_key="t-weak", seed=1, n=60)
    res = pipeline.promote_if_better(weak)
    assert not res["promoted"]
    assert pipeline.production_version() == prod


def test_better_candidate_is_promoted(registry):
    better = pipeline.train_candidate(run_key="t-big", seed=2, n=6000)
    assert pipeline.promote_if_better(better)["promoted"]
    assert pipeline.production_version() == better


def test_api(registry):
    with TestClient(serve.app) as c:
        assert c.get("/health").json()["model_version"] == pipeline.production_version()
        r = c.post("/predict", json=GOOD)
        assert r.status_code == 200 and isinstance(r.json()["prediction"], float)
        r = c.post("/predict/batch", json={"rows": [GOOD, GOOD]})
        assert len(r.json()["predictions"]) == 2
        assert c.post("/predict", json={**GOOD, "latitude": 999}).status_code == 422
        assert c.post("/predict", json={"med_inc": 1}).status_code == 422
        assert c.get("/drift").status_code == 409  # not enough live rows yet
        rows = make_data(400, 3, shift=1.5)[FEATURES].to_dict("records")
        assert c.post("/predict/batch", json={"rows": rows}).status_code == 200
        assert c.get("/drift").json()["drift_detected"] is True
        m = c.get("/metrics").text
        assert "api_requests_total" in m and "drift_detected 1.0" in m
