"""Batch scoring CLI: python -m mlops.batch --input in.csv --output out.csv"""

import argparse
import time

import mlflow
import pandas as pd

from . import config as C


def main() -> None:
    p = argparse.ArgumentParser(description="Score a CSV with the Production model")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--alias", default=C.PRODUCTION)
    p.add_argument("--chunk", type=int, default=50_000)
    a = p.parse_args()
    mlflow.set_tracking_uri(C.tracking_uri())
    model = mlflow.pyfunc.load_model(f"models:/{C.model_name()}@{a.alias}")
    t0, rows, first = time.perf_counter(), 0, True
    for chunk in pd.read_csv(a.input, chunksize=a.chunk):
        missing = set(C.FEATURES) - set(chunk.columns)
        if missing:
            raise SystemExit(f"missing columns: {sorted(missing)}")
        chunk = chunk[C.FEATURES].astype("float64")
        chunk["prediction"] = model.predict(chunk)
        chunk.to_csv(a.output, mode="w" if first else "a", header=first, index=False)
        rows, first = rows + len(chunk), False
    dt = time.perf_counter() - t0
    print(f"scored {rows} rows in {dt:.2f}s ({rows / dt:,.0f} rows/s) -> {a.output}")


if __name__ == "__main__":
    main()
