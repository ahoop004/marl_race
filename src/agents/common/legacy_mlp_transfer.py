"""Legacy MLP tensor layout and appended-observation transfer support."""
from collections.abc import Mapping

import torch


def mlp_transfer_state(state, expected, *, source_width=None, expand_inputs=False):
    """Copy weights and zero-fill appended inputs without renaming tensors."""
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint has no network state")
    result = dict(state)
    weight = result.get("net.0.weight")
    target = expected["net.0.weight"]
    if (not isinstance(weight, torch.Tensor) or weight.ndim != 2
            or (source_width is not None and weight.shape != (target.shape[0], source_width))):
        raise ValueError("Pretrained first-layer weights do not match the source observation width")
    if expand_inputs and weight.shape[0] == target.shape[0] and weight.shape[1] < target.shape[1]:
        expanded = torch.zeros_like(target)
        expanded[:, :weight.shape[1]] = weight
        result["net.0.weight"] = expanded
    return result
