"""Validated PyTorch optimization shared by on-policy learners and policies."""
from collections.abc import Mapping
import inspect
import math

import numpy as np
import torch


OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}
# These are ordinary optimization hyperparameters, not autograd/capture modes.
OPTIMIZER_ARGUMENTS = {
    "adam": {"betas", "eps", "weight_decay", "amsgrad", "foreach", "maximize", "fused",
             "decoupled_weight_decay"},
    "adamw": {"betas", "eps", "weight_decay", "amsgrad", "foreach", "maximize", "fused"},
    "sgd": {"momentum", "dampening", "weight_decay", "nesterov", "foreach", "maximize", "fused"},
    "rmsprop": {"alpha", "eps", "weight_decay", "momentum", "centered", "foreach", "maximize"},
}
SCHEDULERS = {
    "constant": None, "linear": None, "cosine": None,
    "step": torch.optim.lr_scheduler.StepLR,
    "exponential": torch.optim.lr_scheduler.ExponentialLR,
}
SCHEDULER_ALIASES = {"StepLR": "step", "ExponentialLR": "exponential"}
OBJECTIVES = {"ppo": "clip_ppo", "mappo": "mappo"}
CRITIC_LOSSES = frozenset({"l2", "l1", "smooth_l1"})


def _mapping(value, name):
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    return dict(value)


def _number(value, name, *, minimum=0, maximum=None, upper_exclusive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < minimum or (maximum is not None and (
                value >= maximum if upper_exclusive else value > maximum))):
        upper = "" if maximum is None else f" and {'<' if upper_exclusive else '<='} {maximum}"
        raise ValueError(f"{name} must be finite, >= {minimum}{upper}")
    return value


def resolve_optimizer(name="adam", kwargs=None):
    """Resolve only registered classes; learning_rate is the sole initial LR."""
    if not isinstance(name, str) or name not in OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer {name!r}; choose from {sorted(OPTIMIZERS)}")
    supplied = _mapping({} if kwargs is None else kwargs, "optimizer_kwargs")
    allowed = OPTIMIZER_ARGUMENTS[name]
    unknown = set(supplied) - allowed
    if unknown:
        raise ValueError(f"Unsupported {name} optimizer_kwargs {sorted(unknown)}; allowed: {sorted(allowed)}. "
                         "Use learning_rate for the initial LR")
    signature = inspect.signature(OPTIMIZERS[name])
    resolved = {key: signature.parameters[key].default for key in allowed}
    resolved.update(supplied)
    for key, value in resolved.items():
        label = f"{name} optimizer_kwargs.{key}"
        if key == "betas":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"{label} must contain two numbers in [0, 1)")
            resolved[key] = tuple(_number(v, label, maximum=1, upper_exclusive=True) for v in value)
        elif key in {"foreach", "fused"}:
            if value is not None and type(value) is not bool:
                raise ValueError(f"{label} must be boolean or null")
        elif key in {"amsgrad", "maximize", "nesterov", "centered", "decoupled_weight_decay"}:
            if type(value) is not bool:
                raise ValueError(f"{label} must be boolean")
        else:
            _number(value, label, maximum=1 if key == "alpha" else None)
    if resolved.get("fused") and resolved.get("foreach"):
        raise ValueError("optimizer_kwargs.fused and foreach cannot both be true")
    if resolved.get("nesterov") and (resolved["momentum"] <= 0 or resolved["dampening"] != 0):
        raise ValueError("SGD nesterov requires momentum > 0 and dampening = 0")
    return name, resolved


def make_optimizer(parameters, name, kwargs, learning_rate):
    """Keep one optimizer and exclude frozen actor/LoRA parameters."""
    name, kwargs = resolve_optimizer(name, kwargs)
    trainable = tuple(p for p in parameters if p.requires_grad)
    return OPTIMIZERS[name](trainable, lr=learning_rate, **kwargs), trainable


def validate_objective(algorithm, loss, critic_loss):
    if loss != OBJECTIVES.get(algorithm):
        raise ValueError(f"Unsupported loss {loss!r} for {algorithm}; expected {OBJECTIVES.get(algorithm)!r}")
    if not isinstance(critic_loss, str) or critic_loss not in CRITIC_LOSSES:
        raise ValueError(f"Unsupported critic_loss {critic_loss!r}; choose from {sorted(CRITIC_LOSSES)}")


def make_objective(algorithm, loss, critic_loss, actor, critic, **kwargs):
    """Use TorchRL's clipped objectives with the existing caller-supplied keys."""
    from torchrl.objectives import ClipPPOLoss
    from torchrl.objectives.multiagent import MAPPOLoss

    validate_objective(algorithm, loss, critic_loss)
    registry = {"clip_ppo": ClipPPOLoss, "mappo": MAPPOLoss}
    return registry[loss](actor, critic, loss_critic_type=critic_loss, **kwargs)


class OnPolicyOptimizationSettings:
    def configure_optimization(self, params):
        self.optimizer_name, self.optimizer_kwargs = resolve_optimizer(
            params.get("optimizer", "adam"), _mapping(params.get("optimizer_kwargs", {}), "optimizer_kwargs"))
        self.loss = params.get("loss", OBJECTIVES[self.algorithm])
        self.critic_loss = params.get("critic_loss", "l2")
        validate_objective(self.algorithm, self.loss, self.critic_loss)
        _number(params.get("learning_rate", 3e-4), "learning_rate")
        self.lr = float(params.get("learning_rate", 3e-4))
        self.lr_schedule = str(params.get("lr_schedule", "constant"))
        self.lr_schedule = SCHEDULER_ALIASES.get(self.lr_schedule, self.lr_schedule)
        if self.lr_schedule not in SCHEDULERS:
            raise ValueError(f"Unsupported lr_schedule {self.lr_schedule!r}; choose from {sorted(SCHEDULERS)}")
        if self.lr_schedule in {"linear", "cosine"} and params.get("learning_rate_end") is None:
            raise ValueError(f"{self.lr_schedule.capitalize()} lr_schedule requires learning_rate_end")
        end = params.get("learning_rate_end")
        if end is not None:
            _number(end, "learning_rate_end")
        self.lr_end = self.lr if end is None else float(end)
        if not all(np.isfinite(rate) and rate > 0 for rate in (self.lr, self.lr_end)):
            raise ValueError("Learning rates must be finite and positive")
        if self.lr_schedule == "constant" and self.lr_end != self.lr:
            raise ValueError("A different learning_rate_end requires lr_schedule: linear or cosine")
        if self.lr_schedule == "cosine" and self.lr_end > self.lr:
            raise ValueError("Cosine learning_rate_end must not exceed learning_rate")
        self.scheduler_kwargs = _mapping(params.get("lr_scheduler_kwargs", {}), "lr_scheduler_kwargs")
        native = SCHEDULERS[self.lr_schedule] is not None
        self.scheduler_clock = "learner_updates" if native else "training_budget_progress"
        allowed = {"step_size", "gamma"} if self.lr_schedule == "step" else {"gamma"} if native else set()
        unknown = set(self.scheduler_kwargs) - allowed
        if unknown:
            raise ValueError(f"Unsupported {self.lr_schedule} lr_scheduler_kwargs: {sorted(unknown)}")
        if native:
            if end is not None:
                raise ValueError("Native LR schedulers use lr_scheduler_kwargs; learning_rate_end must be null")
            if self.lr_schedule == "step":
                size = self.scheduler_kwargs.get("step_size")
                if type(size) is not int or size < 1:
                    raise ValueError("StepLR requires a positive integer lr_scheduler_kwargs.step_size (learner updates)")
                self.scheduler_kwargs.setdefault("gamma", 0.1)
            elif "gamma" not in self.scheduler_kwargs:
                raise ValueError("ExponentialLR requires lr_scheduler_kwargs.gamma")
            gamma = _number(self.scheduler_kwargs["gamma"], "lr_scheduler_kwargs.gamma", maximum=1)
            if gamma == 0:
                raise ValueError("lr_scheduler_kwargs.gamma must be > 0")
        self.scheduler = None
        self.target_kl = params.get("target_kl")
        if self.target_kl is not None:
            self.target_kl = float(self.target_kl)
            if not np.isfinite(self.target_kl) or self.target_kl <= 0:
                raise ValueError("target_kl must be finite and positive")

    def build_optimizer(self, parameters, *, training):
        self._optim_parameters = tuple(p for p in parameters if p.requires_grad)
        self.optimizer = None
        if training:
            self.optimizer, self._optim_parameters = make_optimizer(
                self._optim_parameters, self.optimizer_name, self.optimizer_kwargs, self.lr)
            scheduler_class = SCHEDULERS[self.lr_schedule]
            if scheduler_class is not None:
                self.scheduler = scheduler_class(self.optimizer, **self.scheduler_kwargs)

    def step_update_scheduler(self, optimizer_steps):
        """Advance native schedulers once after an update that optimized parameters.

        Their next LR is used by the next learner update. The metric for the
        completed update still reports the LR that actually produced it.
        """
        if self.scheduler is not None and optimizer_steps:
            self.scheduler.step()

    def set_training_progress(self, progress):
        if self.lr_schedule in {"linear", "cosine"}:
            fraction = float(np.clip(progress, 0.0, 1.0))
            if self.lr_schedule == "cosine":
                fraction = (1.0 - math.cos(math.pi * fraction)) / 2.0
            rate = self.lr + fraction * (self.lr_end - self.lr)
            for group in self.optimizer.param_groups:
                group["lr"] = rate
