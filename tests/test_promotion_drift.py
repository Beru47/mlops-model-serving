import numpy as np
import pandas as pd

from mlops.drift import drift_report, psi
from mlops.promotion import should_promote


def test_gate_first_model_promotes():
    assert should_promote(1.0, None)


def test_gate_requires_strict_improvement():
    assert should_promote(0.9, 1.0)
    assert not should_promote(1.0, 1.0)  # tie
    assert not should_promote(1.1, 1.0)  # regression


def test_gate_min_improvement():
    assert not should_promote(0.99, 1.0, min_improvement=0.05)
    assert should_promote(0.94, 1.0, min_improvement=0.05)


def test_psi_same_distribution_is_small():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=5000), rng.normal(size=5000)) < 0.05


def test_psi_shift_is_large():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(size=5000), rng.normal(1.5, 1, size=5000)) > 0.2


def test_report_flags_only_shifted_feature():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"a": rng.normal(size=3000), "b": rng.normal(size=3000)})
    cur = pd.DataFrame({"a": rng.normal(size=1000), "b": rng.normal(2, 1, size=1000)})
    rep = drift_report(ref, cur)
    assert rep["drift_detected"]
    assert not rep["features"]["a"]["drifted"]
    assert rep["features"]["b"]["drifted"]
