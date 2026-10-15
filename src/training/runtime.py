"""Process randomness and inference mode belong to execution, not task creation."""
from contextlib import contextmanager
import random

import numpy as np
import torch


def tensor_payload(data, *, source_device=None):
    """Metadata-only union of tensor storage byte spans, deduplicating aliases.

    Exact for contiguous leaves; strided leaves can include holes in their span.
    This is a logical footprint, not allocated storage size or PCIe traffic.
    No tensor contents are read, copied, or moved to calculate it.
    """
    spans, leaves = {}, 0
    for value in data.values(True, True):
        if not isinstance(value, torch.Tensor):
            continue
        if source_device is not None and value.device.type != source_device:
            continue
        leaves += 1
        if not value.numel():
            continue
        size = value.element_size()
        start = value.storage_offset() * size
        end = start + (1 + sum((n - 1) * stride for n, stride in
                              zip(value.shape, value.stride()))) * size
        key = (value.device, value.untyped_storage().data_ptr())
        spans.setdefault(key, []).append((start, end))
    total = 0
    for intervals in spans.values():
        previous_end = 0
        for start, end in sorted(intervals):
            total += max(0, end - max(start, previous_end))
            previous_end = max(previous_end, end)
    return leaves, total


def profiling_sample_indices(total, limit):
    """Bounded deterministic samples including endpoints, without touching RNG."""
    count = min(total, limit)
    if count < 2:
        return {0} if count else set()
    return {index * (total - 1) // (count - 1) for index in range(count)}


def seed_process(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # manual_seed also seeds CUDA lazily; CPU workers need no CUDA probe.


@contextmanager
def preserve_random_state():
    numpy_state, python_state = np.random.get_state(), random.getstate()
    torch_state = torch.random.get_rng_state()
    cuda_devices = ([torch.cuda.current_device()] if torch.distributed.is_initialized()
                    else range(torch.cuda.device_count())) if torch.cuda.is_initialized() else ()
    cuda_states = [(device, torch.cuda.get_rng_state(device)) for device in cuda_devices]
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        torch.random.set_rng_state(torch_state)
        for device, state in cuda_states:
            torch.cuda.set_rng_state(state, device)


@contextmanager
def evaluation_mode(policy):
    actor_was_training = policy.actor.training
    with preserve_random_state():
        policy.actor.eval()
        try:
            with torch.no_grad():
                yield
        finally:
            policy.actor.train(actor_was_training)


class CudaCollectionProfile:
    """Optional CUDA memcpy timing, including action copies owned by TorchRL.

    Device-to-host events include other policy outputs stored with actions.
    Kernel durations exclude host waiting and must not be added to wall time.
    """
    def __init__(self, enabled):
        self.enabled = enabled
        self.device_to_cpu_seconds = 0.
        self.device_to_cpu_calls = 0
        self.cpu_to_device_seconds = 0.
        self.cpu_to_device_calls = 0

    def __enter__(self):
        if self.enabled:
            from torch.profiler import profile, ProfilerActivity
            self.profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            self.profiler.__enter__()
        return self

    def __exit__(self, *exc):
        if self.enabled:
            self.profiler.__exit__(*exc)
            copies = [event for event in self.profiler.events()
                      if event.device_type == torch.autograd.DeviceType.CUDA
                      and event.name.startswith("Memcpy DtoH")]
            self.device_to_cpu_seconds = sum(event.device_time_total for event in copies) / 1e6
            self.device_to_cpu_calls = len(copies)
            copies = [event for event in self.profiler.events()
                      if event.device_type == torch.autograd.DeviceType.CUDA
                      and event.name.startswith("Memcpy HtoD")]
            self.cpu_to_device_seconds = sum(event.device_time_total for event in copies) / 1e6
            self.cpu_to_device_calls = len(copies)
