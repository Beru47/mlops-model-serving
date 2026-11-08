"""Async load test: python -m mlops.loadtest --url http://localhost:8000 --n 3000 --concurrency 16"""
import argparse
import asyncio
import json
import platform
import time

import httpx
import numpy as np

from .config import FEATURES
from .data import make_data


async def run(url: str, n: int, conc: int, shift: float, batch: int) -> dict:
    df = make_data(n, seed=99, shift=shift)[FEATURES]
    if batch > 1:
        path = "/predict/batch"
        payloads = [
            {"rows": df.iloc[i : i + batch].to_dict("records")} for i in range(0, n, batch)
        ]
    else:
        path, payloads = "/predict", df.to_dict("records")
    lat: list[float] = []
    errors = 0
    sem = asyncio.Semaphore(conc)
    async with httpx.AsyncClient(base_url=url, timeout=30) as c:

        async def one(p):
            nonlocal errors
            async with sem:
                t0 = time.perf_counter()
                r = await c.post(path, json=p)
                lat.append((time.perf_counter() - t0) * 1000)
                errors += r.status_code != 200

        for p in payloads[:50]:  # warm-up, not measured
            await c.post(path, json=p)
        t0 = time.perf_counter()
        await asyncio.gather(*(one(p) for p in payloads))
        wall = time.perf_counter() - t0
    p50, p95, p99 = np.percentile(lat, [50, 95, 99])
    return {
        "endpoint": path,
        "requests": len(payloads),
        "rows_per_request": batch,
        "concurrency": conc,
        "errors": errors,
        "rps": round(len(payloads) / wall, 1),
        "p50_ms": round(float(p50), 2),
        "p95_ms": round(float(p95), 2),
        "p99_ms": round(float(p99), 2),
        "machine": f"{platform.processor() or platform.machine()} / {platform.system()}"
        f" / python {platform.python_version()}",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--batch", type=int, default=1, help="rows per request (>1 -> /predict/batch)")
    p.add_argument("--shift", type=float, default=0.0, help="inject covariate drift into traffic")
    p.add_argument("--p95-target-ms", type=float, default=50.0)
    a = p.parse_args()
    res = asyncio.run(run(a.url, a.n, a.concurrency, a.shift, a.batch))
    res["p95_target_ms"] = a.p95_target_ms
    res["target_met"] = res["p95_ms"] <= a.p95_target_ms and res["errors"] == 0
    print(json.dumps(res, indent=2))
    raise SystemExit(0 if res["target_met"] else 1)


if __name__ == "__main__":
    main()
