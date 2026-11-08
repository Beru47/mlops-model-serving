"""Seeded synthetic housing-style dataset (offline, deterministic, drift can be injected)."""
import argparse

import numpy as np
import pandas as pd

from .config import FEATURES, TARGET


def make_data(n: int, seed: int, shift: float = 0.0) -> pd.DataFrame:
    """Generate n rows. `shift` moves med_inc/population (covariate drift); the label rule is unchanged."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "med_inc": rng.lognormal(1.2, 0.45, n) + shift * 2.0,
            "house_age": rng.uniform(1, 52, n),
            "avg_rooms": rng.normal(5.4, 1.1, n).clip(2, 10),
            "population": rng.lognormal(7.2, 0.6, n) * (1 + shift),
            "latitude": rng.uniform(32.5, 42.0, n),
            "longitude": rng.uniform(-124.3, -114.3, n),
        }
    )
    signal = (
        0.45 * df.med_inc
        + 0.012 * df.house_age
        + 0.08 * df.avg_rooms
        - 0.00002 * df.population
        - 0.05 * (df.latitude - 37)
        + 0.3 * np.sin(df.longitude / 3)
    )
    df[TARGET] = signal + rng.normal(0, 0.25, n)
    return df[[*FEATURES, TARGET]]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Write a synthetic CSV (features only)")
    p.add_argument("out")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--shift", type=float, default=0.0)
    a = p.parse_args()
    make_data(a.n, a.seed, a.shift)[FEATURES].to_csv(a.out, index=False)
