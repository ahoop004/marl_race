"""Collector scheduling, event buffering, and CPU worker lifecycle helpers."""
import time
import os
from multiprocessing.connection import wait
from pathlib import Path


def cpu_affinity_count():
    """CPUs this process can use, which can be fewer than the node's CPUs."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


def cpu_affinity_core_count(*, sysfs_root='/sys/devices/system/cpu'):
    """Physical cores represented in the affinity mask, when Linux exposes them.

    Sibling lists identify a core across sockets without counting its hardware
    threads twice. This describes accessible topology, not exclusive ownership
    or a container's CPU-time quota.
    """
    try:
        cpus = os.sched_getaffinity(0)
        siblings = {(Path(sysfs_root) / f'cpu{cpu}' / 'topology' /
                     'thread_siblings_list').read_text().strip() for cpu in cpus}
        return len(siblings) if siblings and '' not in siblings else None
    except (AttributeError, OSError):
        return None


class CollectorEventSink:
    """Buffer worker events until the next collector message."""

    def __init__(self):
        self.events = []

    def send(self, event):
        self.events.append(event)

    def take(self):
        events, self.events = self.events, []
        return events


class CollectorScheduler:
    def __init__(self, mode, timeout):
        if mode not in {'synchronous', 'ready'}:
            raise ValueError('collector_scheduling must be synchronous or ready')
        self.mode = mode
        self.timeout = timeout
        self.last_response = {}

    def reset(self):
        # Policy updates and evaluation deliberately pause all collectors.
        self.last_response.clear()

    def workers(self, connections, paused):
        active = [worker for worker in connections if worker not in paused]
        if self.mode == 'synchronous' or not active:
            return active
        now = time.monotonic()
        for worker in active:
            self.last_response.setdefault(worker, now)
        remaining = min(self.timeout - (now - self.last_response[w]) for w in active)
        if remaining <= 0:
            worker = min(active, key=self.last_response.__getitem__)
            raise RuntimeError(f'Collector worker {worker} timed out after {self.timeout}s')
        ready = wait([connections[w] for w in active], timeout=remaining)
        if not ready:
            worker = min(active, key=self.last_response.__getitem__)
            raise RuntimeError(f'Collector worker {worker} timed out after {self.timeout}s')
        return [w for w in active if connections[w] in ready]

    def exclude_parent_time(self, seconds):
        # Logging/evaluation can block the parent while a response is already
        # queued; that delay must not be blamed on a healthy collector.
        for worker in self.last_response:
            self.last_response[worker] += seconds

    def received(self, worker):
        self.last_response[worker] = time.monotonic()
