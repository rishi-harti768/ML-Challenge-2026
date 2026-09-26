import os

from joblib import Parallel, delayed


def _apply_chunk(func, chunk):
    return [func(x) for x in chunk]


def default_n_jobs() -> int:
    cpus = os.cpu_count() or 1
    return max(1, cpus - 2)


def parallel_map(func, values, n_jobs: int = -1):
    n = len(values)
    if n == 0:
        return []
    if n_jobs == -1:
        n_jobs = default_n_jobs()
    n_jobs = max(1, min(n_jobs, max(1, n // 5000)))
    if n_jobs == 1:
        return [func(v) for v in values]
    chunk_size = max(1, -(-n // n_jobs))
    chunks = [values[i : i + chunk_size] for i in range(0, n, chunk_size)]
    results = Parallel(n_jobs=n_jobs)(delayed(_apply_chunk)(func, c) for c in chunks)
    out = []
    for r in results:
        out.extend(r)
    return out
