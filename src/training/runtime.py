"""Process randomness and inference mode belong to execution, not task creation."""
from contextlib import contextmanager
import random

import numpy as np
import torch


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
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


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
