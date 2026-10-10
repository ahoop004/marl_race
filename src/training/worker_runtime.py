"""Process setup, transport errors and cleanup shared by collector workers."""
from contextlib import contextmanager
from copy import deepcopy
import os
import time


_WORKER_TIMEOUT_SECONDS = 120
_THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")


@contextmanager
def worker_thread_limits():
    """Limit native thread pools before spawn, restoring the parent environment."""
    previous = {key: os.environ.get(key) for key in _THREAD_VARIABLES}
    try:
        for key in _THREAD_VARIABLES:
            os.environ[key] = "1"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def collector_scenario(scenario, environment_id):
    """Derive policy and physical seeds for one collector without mutating input."""
    result = deepcopy(scenario)
    base_seed = int(result["experiment"]["seed"])
    seed = (base_seed + environment_id) % (2 ** 32)
    result["experiment"]["seed"] = seed
    env = result["environment"]
    base_env_seed = base_seed if env.get("seed") is None else int(env["seed"])
    env["seed"] = (base_env_seed + environment_id) % (2 ** 32)
    env["render"] = False
    return result, seed


def receive_worker(connection, process, *, label, timeout=None, on_received=None):
    if timeout is not None and not connection.poll(timeout):
        raise RuntimeError(f"{label} timed out after {timeout}s "
                           f"(pid={process.pid}, exitcode={process.exitcode})")
    try:
        kind, payload = connection.recv()
    except (EOFError, ConnectionResetError) as exc:
        raise RuntimeError(f"{label} disconnected (exitcode={process.exitcode})") from exc
    if on_received is not None:
        on_received()
    if kind == "error":
        raise RuntimeError(f"{label} failed:\n{payload}")
    return kind, payload


def worker_settings(scenario):
    defaults = {"worker_startup_batch_size": 16, "worker_startup_timeout_s": 600,
                "worker_response_timeout_s": _WORKER_TIMEOUT_SECONDS}
    settings = {key: scenario.get("experiment", {}).get(key, default)
                for key, default in defaults.items()}
    for key, value in settings.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"experiment.{key} must be a positive integer")
    return settings


def close_workers(connections, processes, *, failed):
    # Signal the entire group before joining: per-process waits multiplied by
    # hundreds of workers otherwise turn error handling into a long shutdown.
    if failed:
        for process in processes:
            if process.is_alive():
                process.terminate()
    for connection in connections:
        connection.close()
    deadline = time.monotonic() + 5.0
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    remaining = [process for process in processes if process.is_alive()]
    for process in remaining:
        process.kill()
    for process in remaining:
        process.join()


def report_worker_error(connection):
    import sys
    import traceback
    error = traceback.format_exc()
    try:
        connection.send(("error", error))
    except (BrokenPipeError, EOFError, ConnectionResetError):
        # Retain the original exception if its parent can no longer receive it.
        print(error, file=sys.stderr, flush=True)


