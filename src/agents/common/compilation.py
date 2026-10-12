"""Optional process-local compilation of deterministic neural computations."""
from collections.abc import Mapping
import time
from weakref import WeakSet

import torch

_active_calls = WeakSet()


def resolve_compile_config(value=False):
    defaults = dict(actor=False, critic=False, backend="inductor", mode="default")
    if type(value) is bool:
        return {**defaults, "actor": value, "critic": value}
    if not isinstance(value, Mapping) or set(value) - defaults.keys():
        raise ValueError("compile_networks must be boolean or a mapping with actor, critic, backend, mode")
    config = {**defaults, **value}
    if any(type(config[key]) is not bool for key in ("actor", "critic")):
        raise ValueError("compile_networks.actor and critic must be boolean")
    if not isinstance(config["backend"], str) or config["backend"] not in {"inductor", "aot_eager"}:
        raise ValueError("compile_networks.backend must be inductor or aot_eager (correctness diagnostics)")
    if not isinstance(config["mode"], str) or config["mode"] not in {"default", "max-autotune-no-cudagraphs"}:
        raise ValueError("compile_networks.mode must be default or max-autotune-no-cudagraphs")
    if config["backend"] != "inductor" and config["mode"] != "default":
        raise ValueError("Compilation autotuning requires the inductor backend")
    return config


class _CompiledCall:
    def __init__(self, module, method, config, role):
        # An unbound function avoids pickle rebinding a replaced method to this
        # wrapper. Keeping the original module preserves parameter ownership.
        self.module, self.function = module, getattr(type(module), method)
        self.config, self.role = config, role
        self.compiled = None
        self.first_call_seconds = 0.
        self.calls = 0
        self.device = None

    def __getstate__(self):
        # TorchRL deepcopies policies and spawns workers. Compiled closures and
        # timings belong to the executing copy, never to the sending process.
        return {**self.__dict__, "compiled": None, "first_call_seconds": 0., "calls": 0, "device": None}

    def __call__(self, inputs, *args, **kwargs):
        if self.compiled is not None:
            self.calls += 1
            return self.compiled(self.module, inputs, *args, **kwargs)
        self.device = inputs.device
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        options = None
        if self.config["backend"] == "inductor":
            # Graph replay can overwrite stored rollout outputs. Keep it off
            # even if an external Inductor configuration enables CUDA Graphs.
            options = {"triton.cudagraphs": False}
            if self.config["mode"] == "max-autotune-no-cudagraphs":
                options.update(max_autotune=True, coordinate_descent_tuning=True)
        self.compiled = torch.compile(self.function, backend=self.config["backend"],
                                      fullgraph=True, dynamic=True, options=options,
                                      isolate_recompiles=True)
        result = self.compiled(self.module, inputs, *args, **kwargs)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.first_call_seconds = time.perf_counter() - started
        self.calls = 1
        _active_calls.add(self)
        return result


def configure_compilation(policy, value=False):
    config = resolve_compile_config(value)
    policy.compilation_config = config
    if config["actor"]:
        actors = getattr(policy.actor, "actors", {"shared": policy.actor})
        for actor in actors.values():
            # LoRA's tensor computation includes residual banks. Its variable
            # per-agent row selection stays in the ordinary forward method.
            module, method = (actor, "_mean") if hasattr(actor, "_mean") else (actor.net, "forward")
            setattr(module, method, _CompiledCall(module, method, config, "actor"))
    if config["critic"]:
        net = policy.critic.net
        net.forward = _CompiledCall(net, "forward", config, "critic")


def compilation_metrics():
    result = {}
    for call in list(_active_calls):
        prefix = f"{call.role}/{call.device}/"
        for name, value in (("first_call_seconds", call.first_call_seconds), ("initializations", 1),
                            ("calls", call.calls)):
            result[prefix + name] = result.get(prefix + name, 0) + value
    return result
