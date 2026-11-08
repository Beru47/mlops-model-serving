"""Drift math: PSI (effect size) + two-sample KS test (significance) per feature."""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PSI_THRESHOLD = 0.2  # rule of thumb: <0.1 stable, 0.1-0.2 moderate, >0.2 significant
KS_ALPHA = 0.01


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index with quantile bins taken from the reference."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    eps = 1e-4
    r = np.histogram(ref, edges)[0] / len(ref) + eps
    c = np.histogram(cur, edges)[0] / len(cur) + eps
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Per-feature PSI / KS. Drifted only if PSI and KS both agree."""
    feats = {}
    for col in reference.columns:
        stat, p = ks_2samp(reference[col], current[col])
        score = psi(reference[col], current[col])
        feats[col] = {
            "psi": score,
            "ks_stat": float(stat),
            "ks_pvalue": float(p),
            "drifted": bool(score > PSI_THRESHOLD and p < KS_ALPHA),
        }
    return {"drift_detected": any(f["drifted"] for f in feats.values()), "features": feats}
