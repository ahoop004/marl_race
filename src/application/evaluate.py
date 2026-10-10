"""Standalone checkpoint evaluation using the shared task episode runner."""
import argparse
from contextlib import ExitStack
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

from application.checkpoints import resolve_checkpoint_path
from adapters.rewards import RewardMapping
from core.agent_roles import resolve_agent_roles
from training.evaluation_config import resolve_evaluation_protocol
from core.task_builder import create_race_task
from core.run_id import resolve_run_id
from core.provenance import build_run_provenance, provenance_mismatches
from loggers.console import ConsoleLogger
from loggers.metric_policy import MetricPolicy
from loggers.lap_completion import episode_lap_summary
from metrics.racing_eval import aggregate_eval_episodes, episode_race_record, team_finish_result
from training.contracts import validate_team_reward_composers
from training.algorithms import select_algorithm, learner_params, create_learner
from training.evaluation import run_evaluation_episode
from training.runtime import evaluation_mode, seed_process
from utils.torch_io import resolve_device


def run_evaluation(scenario, args, console, scenario_dir):
    with ExitStack() as resources:
        _run_evaluation(scenario, args, console, scenario_dir, resources)


def _run_evaluation(
    scenario: Dict,
    args: argparse.Namespace,
    console: "ConsoleLogger",
    scenario_dir: Path,
    resources,
) -> None:
    """Evaluate a trained PPO or MAPPO checkpoint with deterministic actions."""
    checkpoint = args.checkpoint
    if not checkpoint:
        console.print_error("--eval requires --checkpoint or experiment.checkpoint in the scenario")
        sys.exit(1)

    checkpoint_path = resolve_checkpoint_path(checkpoint)
    compact_laps = MetricPolicy(scenario.get("wandb", {}).get("logging"), scenario).lap_completion

    agent_configs = scenario.get("agents", {})
    trainable_ids = list(resolve_agent_roles(agent_configs).policy_agents)
    if not trainable_ids:
        console.print_error(
            "--eval requires at least one trainable agent in the scenario."
        )
        sys.exit(1)

    algorithm = select_algorithm(scenario, trainable_ids)

    focal_agent_id = trainable_ids[0]
    focal_cfg = agent_configs[focal_agent_id]
    provenance_scenario = scenario
    protocol_name = getattr(args, "eval_protocol", None)
    if protocol_name:
        protocol = resolve_evaluation_protocol(scenario, protocol_name)
        scenario = copy.deepcopy(scenario)
        scenario["experiment"]["seed"] = protocol["seed"]
        scenario["experiment"]["episodes"] = protocol["episodes"]
        scenario["environment"]["max_steps"] = protocol["max_steps"]
        scenario.setdefault("evaluation", {})["max_steps"] = protocol["max_steps"]
        if "map_bundles" in protocol:
            scenario["environment"]["map_bundles_eval"] = protocol["map_bundles"]
        if "target_laps" in protocol:
            scenario["evaluation"]["target_laps"] = protocol["target_laps"]
    exp_cfg = scenario.get("experiment", {})
    env_cfg = scenario.get("environment", {})
    eval_episodes = (
        args.eval_episodes
        if args.eval_episodes is not None
        else int(exp_cfg.get("episodes", 1))
    )
    eval_episodes = max(1, int(eval_episodes))
    base_seed = int(exp_cfg.get("seed", 0) or 0)
    render = bool(env_cfg.get("render", False))
    action_repeat = int(env_cfg.get("action_repeat", 1))

    seed_process(base_seed)
    task = create_race_task(scenario, mode="eval", scenario_dir=scenario_dir,
                            render_mode="human" if render else None)
    resources.callback(task.close)
    env = task.env
    spec = task.spec
    params = learner_params(scenario, spec, algorithm)
    action_dim = len(spec.action_lows[focal_agent_id])
    has_team_rewards = validate_team_reward_composers(
        task.reward_composers, trainable_ids=trainable_ids,
        opponent_ids=list(task.fixed_policy_agents), reward_mode=params.get("reward_mode", "individual"),
        critic_mode=params.get("critic_mode", "agent_conditioned"),
        team_return_mode=params.get("team_return_mode", "per_agent"), action_repeat=action_repeat,
    )
    agent = create_learner(algorithm, spec, params, training=False)
    from utils.torch_io import safe_load
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    checkpoint_payload = safe_load(str(checkpoint_path), map_location="cpu")
    selection = checkpoint_payload.get('checkpoint_selection') or {}
    checkpoint_steps = checkpoint_payload.get('environment_steps', selection.get('environment_steps'))
    stored_provenance = (
        checkpoint_payload.get("provenance")
        if isinstance(checkpoint_payload, dict)
        else None
    )
    mismatches = []
    if isinstance(stored_provenance, dict):
        current_provenance = build_run_provenance(
            provenance_scenario,
            scenario_path=args.scenario,
            run_id="evaluation",
            algorithm=algorithm,
            trainable_agents=trainable_ids,
        )
        mismatches = provenance_mismatches(stored_provenance, current_provenance)
        if mismatches and not args.allow_provenance_mismatch:
            raise ValueError(
                "Checkpoint provenance does not match the evaluation scenario: "
                + "; ".join(mismatches)
                + ". Pass --allow-provenance-mismatch only for an intentional cross-scenario evaluation."
            )
        if mismatches:
            console.print_warning("Checkpoint provenance mismatch explicitly allowed: " + "; ".join(mismatches))
        else:
            console.print_info("Checkpoint provenance matches scenario/config/map hashes.")
    else:
        console.print_warning(
            "Checkpoint has no provenance block; scenario/config/map compatibility cannot be verified."
        )
    agent.load(str(checkpoint_path))
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != checkpoint_hash:
        raise ValueError("Checkpoint changed while loading for evaluation; use a stable checkpoint file.")

    trainable_set = set(trainable_ids)
    other_agents = task.fixed_controllers

    device_str = str(resolve_device([params.get("device", "cpu")]))
    fixed_str = ", ".join(other_agents) or "none"
    trainable_str = ", ".join(trainable_ids)
    console.print_header(
        f"Evaluation: {exp_cfg.get('name', 'unnamed')}",
        f"algorithm={algorithm}  trainable=({trainable_str})  fixed=({fixed_str})",
    )
    console.print_info(
        f"checkpoint={checkpoint_path}  episodes={eval_episodes}  "
        f"seed={base_seed}  device={device_str}  obs_dim={spec.observation_dims[focal_agent_id]}  "
        f"action_dim={action_dim}"
    )

    run_id = args.run_id or resolve_run_id(
        scenario_name=exp_cfg.get("name"), algorithm=f"{algorithm}-eval", seed=base_seed)
    output_dir = Path(args.output_dir) if args.output_dir else Path(scenario.get("paths", {}).get("output_root", "outputs")) / exp_cfg.get("name", "unnamed") / "evaluation" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    from omegaconf import OmegaConf
    OmegaConf.save(OmegaConf.create(scenario), output_dir / "resolved_config.yaml")
    evaluation_provenance = build_run_provenance(scenario, scenario_path=args.scenario,
        run_id=run_id, algorithm=algorithm, trainable_agents=trainable_ids)
    all_agent_ids = list(getattr(env, "possible_agents", list(agent_configs)))
    opponent_ids = [aid for aid in all_agent_ids if aid not in trainable_set]
    target_id = str(focal_cfg.get("target_id", "") or "")
    opponent_agent_id = target_id if target_id in opponent_ids else (opponent_ids[0] if opponent_ids else None)
    eval_episodes_facts = []
    eval_physics = {}
    eval_maps = {}
    eval_spawns = {}
    eval_records = {}
    team_results = []

    mapping = RewardMapping(params.get("reward_mode", "individual"),
                            params.get("team_reward_reduction", "mean"))
    with evaluation_mode(agent):
        for episode in range(eval_episodes):
            result = run_evaluation_episode(task, agent.evaluation_actions,
                episode=episode, seed=base_seed + episode, reward_mapping=mapping, render=render)
            episode_facts = result.facts
            info_dict = result.snapshot.infos
            env_steps = result.snapshot.physics_steps
            eval_maps[episode], eval_spawns[episode] = result.map_id, result.spawn_context
            eval_records[episode] = result.record
            if result.physics is not None:
                eval_physics[episode] = result.physics
            if has_team_rewards:
                team_results.append({
                    **team_finish_result(info_dict, trainable_ids, opponent_ids),
                    "team_episode_reward": result.team_return,
                })
            eval_episodes_facts.append(episode_facts)
            episode_summary = aggregate_eval_episodes(
                [episode_facts],
                focal_agent_id=focal_agent_id,
                opponent_agent_id=opponent_agent_id,
            )
            focal_outcome = episode_facts.agents[focal_agent_id].outcome
            reward_total = sum(
                episode_facts.agents[aid].reward_total
                for aid in trainable_ids
                if aid in episode_facts.agents
            )
            if not opponent_ids and len(trainable_ids) > 1:
                win_value = episode_summary.get("team_both_finished_rate", 0.0)
            else:
                win_value = episode_summary.get(
                    "team_win_rate", episode_summary.get("win_rate", 0.0)
                )
            if has_team_rewards:
                objective = task.reward_composers[focal_agent_id].team_contract[0]["objective"]
                win_value = team_results[-1]["both_finished" if objective == "combined" else objective]

            if compact_laps:
                laps, lap_time, outcomes = episode_lap_summary({}, {
                    "race_record": episode_race_record(episode_facts, timestep=float(env.timestep), include_rewards=False),
                    "agent_outcomes": {aid: episode_facts.agents[aid].outcome for aid in trainable_ids}})
                laps_text = "n/a" if laps is None else f"{laps:g}"
                time_text = "n/a" if lap_time is None else f"{lap_time:.2f}s"
                console.print_info(
                    f"eval ep {episode + 1:4d}/{eval_episodes}  reward={reward_total:+.2f}  "
                    f"laps={laps_text}  lap_time={time_text}  outcome={outcomes}")
            else:
                console.print_info(
                    f"eval ep {episode + 1:4d}/{eval_episodes}  "
                    f"reward={reward_total:+.2f}  steps={env_steps:5d}  "
                    f"win={win_value:.0f}  outcome={focal_outcome}"
                )
    summary = aggregate_eval_episodes(
        eval_episodes_facts,
        focal_agent_id=focal_agent_id,
        opponent_agent_id=opponent_agent_id,
        timestep=float(env.timestep),
    )
    if not opponent_ids and len(trainable_ids) > 1:
        summary["success_rate"] = summary.get("team_both_finished_rate", 0.0)
    elif "team_win_rate" in summary:
        summary["success_rate"] = summary["team_win_rate"]
    elif "win_rate" in summary:
        summary["success_rate"] = summary["win_rate"]
    if has_team_rewards:
        objective = task.reward_composers[focal_agent_id].team_contract[0]["objective"]
        summary["team_objective"] = objective
        summary["team_result_means"] = {
            key: float(np.mean([row[key] for row in team_results])) for key in team_results[0]
        }
        summary["team_results_by_episode"] = team_results
        summary["success_rate"] = summary["team_result_means"][
            "both_finished" if objective == "combined" else objective
        ]

    if compact_laps:
        keys = ("race_count", "mean_episode_reward", "completion_rate",
                "team_both_finished_rate", "learner_failure_rate", "timeout_rate",
                "mean_net_progress", "mean_valid_lap_time_s", "valid_laps",
                "mean_clean_finish_time_s", "clean_finish_count")
        console.print_summary({key: summary[key] for key in keys if key in summary},
                              title="Evaluation Summary")
    elif len(trainable_ids) == 2 and len(opponent_ids) == 2:
        # Historical win/success aliases can mean beating just one opponent.
        # Keep them in the report for compatibility, but use explicit headlines.
        console.print_summary({key + "_rate" if key in {"team_first_place", "team_sweep"} else key: summary[key] for key in (
            "race_count", "team_both_finished_rate", "both_finished_count",
            "at_least_one_finished_count", "team_first_place", "first_place_count",
            "team_sweep", "sweep_count", "team_rank_score", "any_learner_collision_dnf_count",
            "mean_clean_finish_time_s", "finish_time_sample_count", "valid_laps",
            "mean_valid_lap_time_s") if key in summary}, title="Evaluation Summary")
    else:
        console.print_summary(summary, title="Evaluation Summary")
    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_provenance": stored_provenance,
        "provenance_mismatches": mismatches,
        "evaluation_provenance": evaluation_provenance,
        "protocol": protocol_name or "custom",
        "environment_steps": checkpoint_steps,
        "seeds": list(range(base_seed, base_seed + eval_episodes)),
        "max_steps": env.max_steps,
        "target_laps": getattr(env, 'target_laps', None),
        "timestep_s": float(env.timestep),
        "horizon_s": env.max_steps * float(env.timestep) if env.max_steps > 0 else None,
        "action_repeat": action_repeat,
        "summary": summary,
        "per_map": {
            map_name: aggregate_eval_episodes(
                [facts for facts in eval_episodes_facts if eval_maps[facts.episode] == map_name],
                focal_agent_id=focal_agent_id, opponent_agent_id=opponent_agent_id,
                timestep=float(env.timestep),
            ) for map_name in sorted({name for name in eval_maps.values() if name is not None})
        },
        "episode_results": [
            {"seed": base_seed + facts.episode,
             "map_bundle": eval_maps[facts.episode],
             "race_record": {**eval_records[facts.episode],
                             "phase": "evaluation", "run_id": run_id,
                             "environment_id": "evaluation", "environment_episode": facts.episode,
                             "environment_seed": base_seed + facts.episode,
                             "map_id": eval_maps[facts.episode],
                             "spawn_configuration": eval_spawns[facts.episode],
                             "action_repeat": action_repeat,
                             "episode_id": f"{run_id}_standalone_ep{facts.episode:06d}",
                             "checkpoint_sha256": checkpoint_hash},
             **({"physics": eval_physics[facts.episode]} if facts.episode in eval_physics else {}),
             **aggregate_eval_episodes(
                [facts], focal_agent_id=focal_agent_id,
                opponent_agent_id=opponent_agent_id, timestep=float(env.timestep),
            )}
            for facts in eval_episodes_facts
        ],
    }
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    console.print_info(f"Evaluation report: {report_path}")


