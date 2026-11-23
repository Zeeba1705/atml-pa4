import time

import numpy as np
import torch


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize_times(times_ms, unit="ms"):
    q25, q50, q75= np.percentile(times_ms, [25, 50, 75])

    return {
        "unit": unit,
        "reps": len(times_ms),
        "median": float(q50),
        "iqr": float(q75 - q25),
        "q25": float(q25),
        "q75": float(q75),
        "all": [float(t) for t in times_ms]
    }


def time_fn(fn, device, warmup, reps, setup=None):
    #setup runs untimed before every call (e.g. prefill before timing decode); fn gets its result
    times= []

    for i in range(warmup + reps):
        state= setup() if setup is not None else None
        sync(device)

        start= time.perf_counter()
        fn(state)
        sync(device)
        end= time.perf_counter()

        if i >= warmup:
            times.append((end - start) * 1000.0)

    return summarize_times(times)
