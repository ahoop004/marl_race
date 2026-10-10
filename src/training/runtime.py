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
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


@contextmanager
def preserve_random_state():
    numpy_state, python_state = np.random.get_state(), random.getstate()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
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
