"""Single-node launcher ownership and shared synchronous on-policy collectives."""
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
import math
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


class _LossForward(torch.nn.Module):
    """DDP sees the actual TorchRL objective, entropy and critic computation."""
    def __init__(self, agent):
        super().__init__()
        self.objective = agent.loss_module
        self.agent = agent

    def forward(self, batch, group=None, detail=None):
        if batch is None:
            # An exhausted rank must still enter backward. No fake actor row is
            # introduced; globally unused grads are restored to None afterwards.
            zero = sum(p.sum() * 0 for p in self.parameters() if p.requires_grad)
            return zero, zero.detach().new_zeros(4)
        from agents.torchrl_updates import loss_forward
        return loss_forward(self.agent, batch, group=group, detail=detail)


@dataclass
class DistributedExecution:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def primary(self):
        return self.rank == 0

    def sum(self, values):
        tensor = torch.as_tensor(values, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def maximum(self, value):
        tensor = torch.as_tensor(value, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return tensor

    def gather(self, value):
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def broadcast(self, value):
        values = [value if self.primary else None]
        dist.broadcast_object_list(values, src=0)
        return values[0]

    def layout(self, workers, envs_per_worker, threads=1):
        if workers < self.world_size or workers % self.world_size:
            raise ValueError("Distributed parallel_envs is global and must be divisible by world_size")
        if hasattr(os, "sched_getaffinity"):
            minimum = (workers + self.world_size) * threads
            if len(os.sched_getaffinity(0)) < minimum:
                raise ValueError(f"Distributed collectors require at least {minimum} allocated logical CPUs")
        return workers // self.world_size

    def attach(self, agent):
        if any(agent.compilation_config[key] for key in ("actor", "critic")):
            raise ValueError("Distributed training currently requires eager networks (compile_networks=false)")
        objective = _LossForward(agent)
        registered = {id(p) for p in objective.parameters() if p.requires_grad}
        if registered != {id(p) for p in agent._optim_parameters}:
            raise ValueError("DDP objective must own exactly the optimizer's trainable parameters")
        agent.distributed = self
        agent.distributed_loss = DistributedDataParallel(
            objective, device_ids=[self.local_rank] if self.device.type == "cuda" else None,
            forward_sync_buffers=False, find_unused_parameters=True)
        self.parameters = tuple(agent._optim_parameters)
        self.used = torch.zeros(len(self.parameters), dtype=torch.int32, device=self.device)
        self.has_samples = False
        def record(index, grad):
            if self.has_samples:
                self.used[index] = 1
        from functools import partial
        self.handles = [p.register_hook(partial(record, index)) for index, p in enumerate(self.parameters)]

    def detach(self, agent):
        for handle in self.handles:
            handle.remove()
        agent.distributed_loss = None
        agent.distributed = None

    def begin_backward(self, valid):
        self.used.zero_()
        self.has_samples = valid

    def finish_backward(self):
        dist.all_reduce(self.used, op=dist.ReduceOp.MAX)
        for parameter, used in zip(self.parameters, self.used.tolist()):
            if not used:
                # Preserve Adam's absent-gradient behavior, including momentum
                # and weight decay, for globally absent independent actors.
                parameter.grad = None

    def normalize_advantage(self, advantage, total):
        if self.world_size == 1:
            return (advantage - advantage.mean()) / (advantage.std(correction=0) + 1e-8)
        value = advantage.double() if advantage is not None else None
        local_sum = value.sum() if value is not None else torch.zeros((), device=self.device, dtype=torch.float64)
        mean = self.sum(local_sum) / total
        local_variance = ((value - mean).square().sum() if value is not None else torch.zeros_like(mean))
        std = (self.sum(local_variance) / total).sqrt()
        return ((advantage - mean.to(advantage.dtype)) / (std.to(advantage.dtype) + 1e-8)
                if advantage is not None else None)

    def minibatches(self, local_samples, global_batch_size):
        """Rank quotas exhaust every local sample once per epoch, without padding.

        Balanced batches split the configured GLOBAL minibatch equally. If a
        rank runs out, its unused quota goes to a rank with remaining samples.
        Empty ranks execute a zero loss through DDP at the same optimizer step.
        """
        remaining = self.gather(local_samples)
        while sum(remaining):
            size = min(global_batch_size, sum(remaining))
            quotas = [min(count, size // self.world_size) for count in remaining]
            extra = size - sum(quotas)
            for rank in range(self.world_size):
                take = min(extra, remaining[rank] - quotas[rank])
                quotas[rank] += take
                extra -= take
            remaining = [count - quota for count, quota in zip(remaining, quotas)]
            yield quotas[self.rank], size

    def metadata(self):
        return {"mode": "ddp", "backend": dist.get_backend(), "world_size": self.world_size,
                "rank": self.rank, "local_rank": self.local_rank,
                "device_mapping": {str(rank): f"cuda:{rank}" if self.device.type == "cuda" else "cpu"
                                   for rank in range(self.world_size)}}


@contextmanager
def distributed_execution(enabled=False, *, timeout_seconds=1800):
    """Production uses NCCL only; CPU/Gloo tests initialize their own groups."""
    launched = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if not enabled:
        if launched:
            raise ValueError("torchrun with multiple ranks requires experiment.distributed=true")
        yield None
        return
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("experiment.distributed_timeout_seconds must be positive and finite")
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    if any(key not in os.environ for key in required):
        raise ValueError("Distributed training must be launched with single-node torchrun")
    rank, local_rank, size, local_size = (int(os.environ[key]) for key in required[:4])
    if size not in {1, 2} or size != local_size or rank != local_rank or not 0 <= rank < size:
        raise ValueError("Only single-node torchrun with one or two local ranks is supported")
    if not torch.cuda.is_available() or torch.cuda.device_count() < size:
        raise ValueError(f"NCCL training requests {size} visible CUDA GPUs; found {torch.cuda.device_count()}")
    if not dist.is_nccl_available():
        raise ValueError("This PyTorch build does not provide NCCL")
    if dist.is_initialized():
        raise ValueError("A process group is already initialized")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    try:
        dist.init_process_group("nccl", rank=rank, world_size=size, device_id=device,
                                timeout=timedelta(seconds=timeout_seconds))
        yield DistributedExecution(rank, local_rank, size, device)
    finally:
        # Do not insert a barrier on exception: torchrun terminates sibling
        # ranks, and the process-group timeout bounds failed collectives.
        if dist.is_initialized():
            dist.destroy_process_group()
