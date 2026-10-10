"""Algorithm configuration and supported training experiment validation."""
from typing import Any, Dict

from core.scenario import ScenarioError
from core.agent_builder import fixed_controller_names
from core.agent_roles import is_policy_agent
from training.evaluation_config import resolve_evaluation_protocol


POLICY_ALGORITHMS = frozenset({"ppo", "mappo"})


MAPPO_DEFAULTS: Dict[str, str] = {
    "actor_mode": "shared",
    "reward_mode": "team_shared",
    "critic_mode": "shared_team",
    "team_reward_reduction": "mean",
}


def resolve_mappo_config(scenario: Dict[str, Any]) -> Dict[str, str]:
    """Return the normalized MAPPO reward/critic experiment contract."""
    raw = scenario.get("mappo", {}) or {}
    if not isinstance(raw, dict):
        raise ScenarioError("'mappo' must be a dictionary when provided.")
    unknown = sorted(set(raw) - set(MAPPO_DEFAULTS))
    if unknown:
        raise ScenarioError(f"Unknown MAPPO config field(s): {unknown}.")
    config = dict(MAPPO_DEFAULTS)
    config.update({key: str(value).strip().lower() for key, value in raw.items()})
    return config


def validate_training_scenario(scenario):
    """Validate learner eligibility after the task configuration is validated."""
    fixed_algos = set(fixed_controller_names())
    known_algos = POLICY_ALGORITHMS | fixed_algos
    experiment = scenario["experiment"]
    environment = scenario["environment"]
    agents = scenario["agents"]
    if "evaluation_only" in experiment and not isinstance(experiment["evaluation_only"], bool):
        raise ScenarioError("experiment.evaluation_only must be boolean.")
    checkpoint = experiment.get("checkpoint")
    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint.strip()):
        raise ScenarioError("'experiment.checkpoint' must be a nonempty path string or null.")
    total_steps = experiment.get("total_steps")
    if total_steps is not None and (isinstance(total_steps, bool)
            or not isinstance(total_steps, int) or total_steps <= 0):
        raise ScenarioError("'experiment.total_steps' must be a positive integer or null.")
    if experiment.get("ppo_backend", "torchrl") != "torchrl":
        raise ScenarioError("Legacy PPO backends have been removed; experiment.ppo_backend must be torchrl")
    if experiment.get("mappo_backend", "torchrl") != "torchrl":
        raise ScenarioError("Legacy MAPPO backends have been removed; experiment.mappo_backend must be torchrl")

    if scenario.get("evaluation"):
        resolve_evaluation_protocol(scenario, "selection")

    # --- Per-agent checks ---
    for agent_id, agent_cfg in agents.items():
        if not isinstance(agent_cfg, dict):
            raise ScenarioError(
                f"Agent '{agent_id}' config must be a dictionary, got {type(agent_cfg).__name__}."
            )

        algo = str(agent_cfg.get("algorithm", "")).strip().lower()
        if not algo:
            raise ScenarioError(
                f"Agent '{agent_id}' is missing required 'algorithm' field."
            )

        if algo not in known_algos:
            raise ScenarioError(
                f"Agent '{agent_id}' has unknown algorithm '{algo}'. "
                f"Known RL algorithms: {sorted(POLICY_ALGORITHMS)}. "
                f"Known fixed controllers: {sorted(fixed_algos)}."
            )

        if agent_cfg.get("role") in {"attacker", "defender"}:
            raise ScenarioError("Attacker/defender roles are unsupported for completion experiments")

        explicit = agent_cfg.get("trainable")
        if explicit is not None and not isinstance(explicit, bool):
            raise ScenarioError(f"Agent '{agent_id}' trainable must be a boolean.")
        if explicit is not None and explicit != (algo in POLICY_ALGORITHMS):
            raise ScenarioError(
                f"Agent '{agent_id}': algorithm '{algo}' does not support trainable={explicit}. "
                "PPO/MAPPO are trainable; fixed opponents use a fixed controller."
            )

    trainable_ids = [aid for aid, cfg in agents.items() if is_policy_agent(cfg)]
    trainable_algos = {
        str(agents[aid]["algorithm"]).strip().lower() for aid in trainable_ids
    }
    if len(trainable_algos) > 1:
        raise ScenarioError("Mixed trainable algorithms are unsupported; use one PPO agent or a MAPPO team.")
    if trainable_algos == {"ppo"} and len(agents) != 1:
        raise ScenarioError("PPO completion experiments require one vehicle")
    if experiment.get("ppo_backend") == "torchrl":
        if trainable_algos != {"ppo"}:
            raise ScenarioError("The TorchRL PPO backend requires one PPO learner")
    if experiment.get("mappo_backend") == "torchrl":
        if trainable_algos != {"mappo"}:
            raise ScenarioError("The TorchRL MAPPO backend requires MAPPO learners")
    evaluation_strategy = scenario.get("evaluation", {}).get("selection_strategy")
    if evaluation_strategy is not None:
        supported = {"team_completion"} if trainable_algos == {"mappo"} else {"completion_progress", "lap_time"}
        if evaluation_strategy not in supported:
            raise ScenarioError(f"Unsupported evaluation selection strategy for {sorted(trainable_algos)}: {evaluation_strategy}")
    if total_steps is not None and trainable_algos not in ({"ppo"}, {"mappo"}):
        raise ScenarioError("A total_steps budget requires PPO or MAPPO.")
    if experiment.get("collector_scheduling", "synchronous") not in {"synchronous", "ready"}:
        raise ScenarioError("experiment.collector_scheduling must be synchronous or ready")
    eval_workers = scenario.get("evaluation", {}).get("num_workers", 1)
    if eval_workers != 'auto' and (isinstance(eval_workers, bool)
            or not isinstance(eval_workers, int) or eval_workers < 1):
        raise ScenarioError("evaluation.num_workers must be a positive integer or auto")
    num_envs = experiment.get("num_envs", 1)
    for name in ("num_envs", "num_workers", "torch_threads", "worker_startup_batch_size",
                 "worker_startup_timeout_s", "worker_response_timeout_s"):
        value = experiment.get(name, 1)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ScenarioError(f"'experiment.{name}' must be a positive integer.")
    if num_envs > 1:
        if trainable_algos not in ({"ppo"}, {"mappo"}):
            raise ScenarioError("Parallel environments require PPO or MAPPO.")
        if environment.get("render"):
            raise ScenarioError("Parallel training requires headless training.")
        seed = experiment.get("seed")
        env_seed = environment.get("seed", seed)
        if env_seed is None:
            env_seed = seed
        if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < 2 ** 32
               for v in (seed, env_seed)):
            raise ScenarioError("Parallel training requires explicit integer seeds in [0, 2**32).")
        if total_steps is not None and total_steps < num_envs:
            raise ScenarioError("Parallel training total_steps must be at least num_envs")
        if total_steps is None and int(experiment.get("episodes", 1000)) < num_envs:
            raise ScenarioError("Parallel training requires at least num_envs total episodes.")
        params = {**scenario.get("training_defaults", {}), **agents[trainable_ids[0]].get("params", {})}
        n_steps = params.get("n_steps", 2048)
        if trainable_algos == {"ppo"} and (isinstance(n_steps, bool) or not isinstance(n_steps, int)
                or n_steps < num_envs or n_steps % num_envs):
            raise ScenarioError("Parallel PPO n_steps must be a positive multiple of num_envs.")
        if trainable_algos == {"mappo"}:
            horizon = scenario.get("training_defaults", {}).get("rollout_steps_per_env", 256)
            if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
                raise ScenarioError("MAPPO rollout_steps_per_env must be a positive integer")
    if trainable_algos == {"mappo"}:
        if len(trainable_ids) != 2 or len(agents) != 4:
            raise ScenarioError("MAPPO completion experiments require two learners and two fixed opponents")
        mappo = resolve_mappo_config(scenario)
        if mappo["actor_mode"] not in {"shared", "independent"}:
            raise ScenarioError("mappo.actor_mode must be shared or independent")
        if (mappo["reward_mode"] != "team_shared" or mappo["critic_mode"] != "shared_team"
                or mappo["team_reward_reduction"] != "mean"):
            raise ScenarioError("MAPPO completion requires team_shared rewards, shared_team critic and mean reduction")
        reference_id = trainable_ids[0]
        params = {**scenario.get("training_defaults", {}), **agents[reference_id].get("params", {})}
        if "adapter_transfer" in params:
            raise ScenarioError("adapter_transfer is unsupported; use pretrained_actor_checkpoint for PPO actor transfer")
        if not isinstance(params.get("require_pretrained_actor", False), bool):
            raise ScenarioError("require_pretrained_actor must be boolean")
        lora = params.get("lora")
        if lora is not None and (not isinstance(lora, dict)
                or mappo["actor_mode"] != "shared" or lora.get("mode") != "per_agent"):
            raise ScenarioError("LoRA completion experiments require a shared actor with per_agent adapters")
        if (params.get("team_return_mode") != "joint"
                or int(environment.get("action_repeat", 1)) != 1
                or environment.get("episode_termination", {}).get("mode") not in {"all_agents", "all_trainable"}):
            raise ScenarioError("MAPPO completion requires joint team returns, action_repeat=1 and all_agents/all_trainable termination")
        reference = agents[reference_id]
        for agent_id in trainable_ids[1:]:
            for field in ("observation", "reward", "params", "action_constraints"):
                if agents[agent_id].get(field, {}) != reference.get(field, {}):
                    raise ScenarioError(f"MAPPO completion learners require identical {field} config; {reference_id} and {agent_id} differ")


