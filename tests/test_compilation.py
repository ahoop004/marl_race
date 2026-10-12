"""Small compilation correctness checks; throughput experiments belong on HPC."""
from copy import deepcopy
import pickle

import pytest
import torch

from agents.common.compilation import resolve_compile_config, _CompiledCall
from agents.common.ppo_policy import PPOPolicy
from test_torchrl_ppo import agent, rollout
from test_torchrl_mappo import learner, transition, IDS, BOUNDS

OPTIONS = dict(actor=True, critic=True, backend="aot_eager")


def test_compile_options_are_strict_and_default_is_eager():
    assert not any(resolve_compile_config()[key] for key in ("actor", "critic"))
    for value in (None, 1, {"actors": True}, {"actor": 1}, {"backend": []},
                  {"mode": "reduce-overhead"}, {"backend": "aot_eager", "mode": "max-autotune-no-cudagraphs"}):
        with pytest.raises(ValueError, match="compil|Compil"):
            resolve_compile_config(value)


def test_compiled_copies_own_parameters_and_reset_caches():
    policy = agent(hidden_dims=[4], compile_networks=OPTIONS)
    source = policy.actor.net
    expected = source(torch.ones(2, 1)).detach().clone()
    assert source.forward.compiled is not None
    for copied in (deepcopy(source), pickle.loads(pickle.dumps(source))):
        assert copied.forward.module is copied and copied.forward.compiled is None
        assert copied.forward.calls == 0
        torch.testing.assert_close(copied(torch.ones(2, 1)), expected)
        assert copied[0].weight is not source[0].weight
        copied.load_state_dict({key: torch.zeros_like(value) for key, value in source.state_dict().items()})
        torch.testing.assert_close(copied(torch.ones(3, 1)), torch.zeros(3, 2))
        torch.testing.assert_close(source(torch.ones(2, 1)), expected)


@pytest.mark.parametrize("mode", ["ppo", "shared", "independent", "lora"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_compiled_sampling_updates_and_checkpoints_match_eager(mode, device, backend, tmp_path):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware unavailable")
    if mode == "ppo":
        compiled = agent(hidden_dims=[4], device=device, compile_networks={**OPTIONS, "backend": backend})
        eager = agent(hidden_dims=[4], device=device)
    else:
        options = dict(hidden_dims=[4], device=device, actor_mode="shared" if mode == "lora" else mode)
        if mode == "lora":
            options["lora"] = dict(mode="per_agent", rank=1, alpha=2, per_agent_log_std=True)
            source = tmp_path / "source.pt"
            PPOPolicy(1, *BOUNDS, {"hidden_dims": [4], "device": "cpu"}).save_checkpoint(source)
        compiled = learner(**options, compile_networks={**OPTIONS, "backend": backend})
        eager = learner(**options)
        if mode == "lora":
            for policy in (compiled, eager):
                policy.initialize_from_checkpoint(source, scope="actor_only")
    eager.actor.load_state_dict(compiled.actor.state_dict())
    eager.critic.load_state_dict(compiled.critic.state_dict())
    # Rebuild behavior samples under the matching weights, including stochastic
    # sampling outside the compiled deterministic cores.
    if mode == "ppo":
        def sample(data):
            data.update(eager.policy(data.to(device)).to("cpu"))
        data = rollout(sample, [0, 1, 2, 3], done=[False, False, False, True],
                       terminated=[False, False, False, True])
    else:
        data = torch.stack([transition(eager, ids, step, step + 1, terminal=step == 2)
                            for step, ids in enumerate((IDS, ["car_1"], ["car_1"]))])
    current = data.to(device).clone()
    before_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device == "cuda" else None
    eager.policy(current)
    torch.random.set_rng_state(before_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    compiled_current = data.to(device).clone()
    compiled.policy(compiled_current)
    key = "raw_action" if mode == "ppo" else ("agents", "raw_action")
    torch.testing.assert_close(compiled_current[key], current[key])
    # Optimizer minibatch order uses the same RNG; exercise GAE, gradients and a
    # retired learner as well as the actor/critic forward computations.
    torch.manual_seed(91)
    eager.update(data.clone())
    torch.manual_seed(91)
    compiled.update(data.clone())
    for module_name in ("actor", "critic"):
        actual, expected = getattr(compiled, module_name), getattr(eager, module_name)
        assert actual.state_dict().keys() == expected.state_dict().keys()
        for key, value in actual.state_dict().items():
            torch.testing.assert_close(value, expected.state_dict()[key], rtol=2e-5, atol=2e-6)
    checkpoint = tmp_path / "compiled.pt"
    compiled.save_checkpoint(checkpoint)
    eager.load_for_evaluation(checkpoint)
    for key, value in compiled.actor.state_dict().items():
        torch.testing.assert_close(value, eager.actor.state_dict()[key], rtol=0, atol=0)


def test_inductor_options_disable_cuda_graphs(monkeypatch):
    calls = []
    def compile_fn(function, **kwargs):
        calls.append(kwargs)
        return function
    monkeypatch.setattr(torch, "compile", compile_fn)
    policy = agent(hidden_dims=[4], compile_networks={"actor": True, "mode": "max-autotune-no-cudagraphs"})
    policy.actor(torch.ones(2, 1))
    assert calls[0]["options"] == {"triton.cudagraphs": False, "max_autotune": True,
                                  "coordinate_descent_tuning": True}
    assert calls[0]["fullgraph"] and calls[0]["dynamic"]
    assert isinstance(policy.actor.net.forward, _CompiledCall)
    assert not isinstance(policy.critic.net.forward, _CompiledCall)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_inductor_forward_backward_and_output_lifetime(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware unavailable")
    compiled = agent(hidden_dims=[4], device=device, compile_networks=True)
    eager = agent(hidden_dims=[4], device=device)
    for role in ("actor", "critic"):
        actual, reference = getattr(compiled, role).net, getattr(eager, role).net
        reference.load_state_dict(actual.state_dict())
        inputs = torch.ones(2, 1, device=device)
        output = actual(inputs)
        snapshot = output.detach().clone()
        torch.testing.assert_close(output, reference(inputs), rtol=1e-4, atol=1e-6)
        output.sum().backward()
        reference(inputs).sum().backward()
        for left, right in zip(actual.parameters(), reference.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=1e-4, atol=1e-6)
        actual(torch.zeros(3, 1, device=device))
        torch.testing.assert_close(output.detach(), snapshot, rtol=0, atol=0)


@pytest.mark.parametrize("algorithm", ["ppo", "mappo"])
@pytest.mark.parametrize("backend", ["process", "centralized"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_compiled_collectors_sync_and_reload(algorithm, backend, device, tmp_path, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware unavailable")
    import training.algorithms as algorithms
    from training.on_policy import OnPolicyTrainer
    from test_on_policy_pipeline import test_collection_devices_sync_batches_and_checkpoint_loading
    original_params = algorithms.learner_params
    monkeypatch.setattr(algorithms, "learner_params", lambda *args, **kwargs: {
        **original_params(*args, **kwargs), "compile_networks": OPTIONS})
    original_metrics = OnPolicyTrainer.performance_metrics
    seen = []
    def metrics(trainer):
        result = original_metrics(trainer)
        seen.append(result)
        return result
    monkeypatch.setattr(OnPolicyTrainer, "performance_metrics", metrics)
    test_collection_devices_sync_batches_and_checkpoint_loading(
        algorithm, device, "cpu", 2, 2, backend, tmp_path, monkeypatch)
    result = seen[-1]
    if backend == "process":
        for worker in range(2):
            assert result[f"perf/workers/{worker}/compilation/actor/cpu/calls"] > 0
    else:
        assert result["perf/compilation/actor/cpu/calls"] > 0
    assert any(value > 0 for key, value in result.items()
               if key.startswith("perf/compilation/critic/") and key.endswith("/calls"))
