"""Learner selection and contracts, independent of environment construction."""
import numpy as np

from training.configuration import POLICY_ALGORITHMS, resolve_mappo_config
from core.provenance import physics_contract
from wrappers.actions.composer import ActionComposer


def resolve_training_params(agent_cfg: dict, scenario: dict) -> dict:
    """Merge training_defaults with agent params — agent params win."""
    defaults = scenario.get("training_defaults", {})
    params = agent_cfg.get("params", {})
    environment = scenario.get("environment", {})
    decision_dt = float(environment.get("timestep", 0.01)) * int(environment.get("action_repeat", 1))
    network = scenario.get("network", {})
    # Routing/input choices are validated at the application boundary; the
    # existing MLP factory accepts these four architecture fields.
    if "_configuration" in scenario and "algorithm" in scenario:
        network = {key: value for key, value in network.items() if key in {
            "architecture", "actor_hidden_dims", "critic_hidden_dims", "activation"}}
    return {"network": network, **defaults, **params,
            "_physics_contract": physics_contract(environment),
            "_action_contract": ActionComposer.contract_from_config(
        agent_cfg.get("action_constraints", {}), decision_dt,
    )}


def select_algorithm(scenario, policy_agents):
    algorithms = {str(scenario["agents"][aid]["algorithm"]).strip().lower()
                  for aid in policy_agents}
    if len(algorithms) != 1 or not algorithms <= POLICY_ALGORITHMS:
        raise ValueError("Select one PPO learner or a homogeneous MAPPO team")
    algorithm = algorithms.pop()
    if algorithm == "ppo" and len(policy_agents) != 1:
        raise ValueError("PPO requires one policy agent")
    return algorithm


def learner_params(scenario, spec, algorithm):
    focal = spec.agent_ids[0]
    params = resolve_training_params(scenario["agents"][focal], scenario)
    params["_observation_contract"] = (
        spec.observation_contracts[focal] if params["_physics_contract"] is not None else None)
    if algorithm == "mappo":
        params.update(resolve_mappo_config(scenario))
        params.update(_observation_dims=spec.observation_dims,
                      _observation_contracts=spec.observation_contracts,
                      _global_state_contract_version=spec.state_version)
    return params


def create_learner(algorithm, spec, params, *, training=True):
    """Construct policy state from specifications; never create or reset a task."""
    focal = spec.agent_ids[0]
    low, high = spec.action_lows[focal], spec.action_highs[focal]
    if algorithm == "ppo":
        if len(spec.agent_ids) != 1:
            raise ValueError("PPO requires one policy agent")
        if training:
            from agents.torchrl_ppo import TorchRLPPOAgent as Policy
        else:
            from agents.common.ppo_policy import PPOPolicy as Policy
        return Policy(spec.observation_dims[focal], low, high, params)
    if algorithm != "mappo":
        raise ValueError(f"Unsupported learner algorithm: {algorithm!r}")
    if len(set(spec.observation_dims.values())) != 1 and (params.get("lora") or {}).get("mode") != "per_agent":
        raise ValueError("Shared MAPPO actors require identical local observation dimensions")
    if any(not np.array_equal(spec.action_lows[aid], low)
           or not np.array_equal(spec.action_highs[aid], high) for aid in spec.agent_ids):
        raise ValueError("MAPPO requires identical physical action bounds")
    if training:
        from agents.torchrl_mappo import TorchRLMAPPOAgent as Policy
    else:
        from agents.common.mappo_policy import MAPPOPolicy as Policy
    return Policy(spec.observation_dims[focal], spec.state_dim, low, high,
                  list(spec.agent_ids), params)


def create_trainer(algorithm, task, learner, **options):
    if algorithm == "ppo":
        from training.torchrl_ppo_trainer import TorchRLPPOTrainer as Trainer
    elif algorithm == "mappo":
        from training.torchrl_mappo_trainer import TorchRLMAPPOTrainer as Trainer
    else:
        raise ValueError(f"Unsupported trainer algorithm: {algorithm!r}")
    return Trainer(task, learner, **options)


def check_evaluation_spec(training_spec, evaluation_spec):
    if (training_spec.agent_ids != evaluation_spec.agent_ids
            or training_spec.observation_dims != evaluation_spec.observation_dims
            or training_spec.observation_contracts != evaluation_spec.observation_contracts
            or training_spec.state_dim != evaluation_spec.state_dim
            or training_spec.state_version != evaluation_spec.state_version
            or any(not np.array_equal(training_spec.action_lows[aid], evaluation_spec.action_lows[aid])
                   or not np.array_equal(training_spec.action_highs[aid], evaluation_spec.action_highs[aid])
                   for aid in training_spec.agent_ids)):
        raise ValueError("Training and evaluation task specifications must match")
