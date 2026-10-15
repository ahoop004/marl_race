"""Bounded, order-preserving preparation of independent TorchRL trajectories."""
from contextlib import nullcontext

import torch
from tensordict import TensorDictBase


def rollout_groups(agent, rollouts, *, max_bytes=32 * 1024**2, max_trajectories=64, max_rows=16384):
    """Group consecutive equal layouts, without padding or reordering.

    Byte, trajectory, and row limits bound stacking storage, Python metadata,
    and value-network intermediates independently of collector topology.
    Oversized or incompatible trajectories stay single.
    """
    group, group_bytes, group_rows, previous = [], 0, 0, None
    region = getattr(agent, "profile_region", None)
    for rollout in rollouts:
        if not isinstance(rollout, TensorDictBase):
            raise TypeError("TorchRL requires a TensorDict rollout")
        if not rollout.batch_size.numel():
            continue
        with region("rollout_key_preparation") if region else nullcontext():
            data = agent.select_rollout(rollout)
            leaves = list(data.items(True, True))
            compatible = data.ndim == 1 and all(isinstance(value, torch.Tensor)
                and value.layout == torch.strided for _, value in leaves)
            nested = tuple((key, tuple(value.batch_size), tuple(value.names))
                           for key, value in data.items(True, False) if isinstance(value, TensorDictBase))
            signature = (tuple(data.batch_size), tuple(data.names), nested, tuple(
                (key, tuple(value.shape), value.dtype, value.device) for key, value in leaves)) if compatible else None
            size = sum(value.numel() * value.element_size() for _, value in leaves
                       if isinstance(value, torch.Tensor))
            rows = data.batch_size.numel()
        if group and (signature is None or signature != previous
                      or group_bytes + size > max_bytes or len(group) >= max_trajectories
                      or group_rows + rows > max_rows):
            yield group
            group, group_bytes, group_rows = [], 0, 0
        group.append(data)
        group_bytes += size
        group_rows += rows
        previous = signature
        if signature is None or group_bytes >= max_bytes or group_rows >= max_rows:
            yield group
            group, group_bytes, group_rows = [], 0, 0
    if group:
        yield group


def transfer_rollout_group(agent, group):
    """Return owned data and a CPU-only all-active check for sample selection."""
    region = getattr(agent, "profile_region", None)
    profile = getattr(agent, "profile_call", None)
    count = getattr(agent, "profile_count", None)
    with region("rollout_batch_stack") if region else nullcontext():
        data = group[0] if len(group) == 1 else torch.stack(group, dim=0)
    active = data.get(("agents", "active"), None)
    # No GPU scalar reads: same-device CUDA callers use the indexed fallback.
    all_active = active is None or (active.device.type == "cpu" and bool(active.all()))
    devices = [value.device for value in data.values(True, True) if isinstance(value, torch.Tensor)]
    destination = torch.device(agent.device)
    if destination.type == "cuda" and destination.index is None:
        destination = torch.device("cuda", torch.cuda.current_device())
    elif destination.type == "cpu":
        destination = torch.device("cpu")
    # Stack allocates every leaf. A singleton's .to() instead reuses any leaves
    # already on the destination, so same-device and mixed-device inputs still
    # need a clone. A complete CPU-to-CUDA move already owns every GPU tensor.
    needs_clone = len(group) == 1 and any(device == destination for device in devices)
    payload = getattr(agent, "profile_payload", None)
    if payload:
        payload("cpu_to_cuda", data, destination=agent.device, trajectories=len(group))
    if count:
        count("transfer_batch_count")
        if len(group) > 1:
            count("rollout_stack_bytes", sum(value.numel() * value.element_size()
                  for value in data.values(True, True) if isinstance(value, torch.Tensor)))
    moving = any(device != destination for device in devices)
    if profile:
        category = "rollout_device_transfer" if moving else "tensor_device_conversion"
        data = profile(category, data.to, agent.device, device=agent.device)
    else:
        data = data.to(agent.device)
    with region("learner_rollout_clone") if region else nullcontext():
        if needs_clone:
            data = profile("tensor_copy", data.clone, device=agent.device) if profile else data.clone()
            if count:
                count("learner_clone_batches")
    return data, all_active


def batch_gae(agent, data, trajectories):
    """Keep GAE recursion on time, independent of trajectory and agent axes."""
    with torch.no_grad():
        if trajectories > 1:
            agent.gae(data, time_dim=1)  # [trajectory, time, (agent), features]
        else:
            agent.gae(data)
    count = getattr(agent, "profile_count", None)
    if count:
        count("gae_batch_count")
        count("gae_trajectories_processed", trajectories)
