"""Run scenarios with only fixed controllers."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
from typing import Dict

import numpy as np

from core.setup import create_environment_setup
from core.run_id import resolve_run_id, set_run_id_env
from core.provenance import build_run_provenance
from loggers.console import ConsoleLogger
from loggers.csv_logger import CSVLogger
from loggers.wandb_logger import WandbLogger
from training.runtime import seed_process


def run_heuristic(scenario, args, console):
    with ExitStack() as resources:
        _run_heuristic(scenario, args, console, resources)


def _run_heuristic(
    scenario: Dict,
    args: argparse.Namespace,
    console: "ConsoleLogger",
    resources,
) -> None:
    """Episode loop for scenarios where every agent is a heuristic/fixed policy.

    Replaces the RL training path when ``run.py`` detects no trainable agents.
    Supports the same ``--render``, ``--episodes``, ``--seed``, and ``--wandb``
    flags as the RL path.
    """
    agent_configs = scenario.get("agents", {})
    exp_cfg = scenario.get("experiment", {})
    env_cfg = scenario.get("environment", {})
    wandb_cfg = scenario.get("wandb", {})

    n_episodes = int(exp_cfg.get("episodes", 100))
    render = bool(env_cfg.get("render", False))
    agent_ids = list(agent_configs.keys())
    algos = {aid: agent_configs[aid].get("algorithm", "?") for aid in agent_ids}
    agents_str = "  ".join(f"{aid}={v}" for aid, v in algos.items())
    algorithm = "+".join(sorted({str(value) for value in algos.values()})) or "heuristic"
    run_id = args.run_id or resolve_run_id(
        scenario_name=exp_cfg.get("name"),
        algorithm=algorithm,
        seed=exp_cfg.get("seed"),
    )
    set_run_id_env(run_id)
    output_dir = args.output_dir or os.path.join(
        scenario.get("paths", {}).get("output_root", "outputs"), exp_cfg.get("name", "unnamed"), run_id
    )
    provenance = build_run_provenance(
        scenario,
        scenario_path=args.scenario,
        run_id=run_id,
        algorithm=algorithm,
        trainable_agents=[],
    )
    csv_logger = CSVLogger(output_dir, scenario, provenance=provenance)
    resources.callback(csv_logger.close)

    console.print_header(
        f"Heuristic: {exp_cfg.get('name', 'unnamed')}",
        agents_str,
    )

    wandb_logger = None
    if wandb_cfg.get("enabled", False):
        wandb_logger = WandbLogger(
            project=wandb_cfg.get("project", "f110"),
            name=wandb_cfg.get("name", exp_cfg.get("name")),
            config=scenario,
            tags=wandb_cfg.get("tags", []),
            group=wandb_cfg.get("group"),
            job_type=wandb_cfg.get("job_type", "heuristic"),
            entity=wandb_cfg.get("entity"),
            notes=wandb_cfg.get("notes"),
            mode=wandb_cfg.get("mode", "online"),
            logging_config=wandb_cfg.get("logging"),
        )
        resources.callback(wandb_logger.finish)

    seed_process(exp_cfg.get("seed"))
    env, agents = create_environment_setup(
        scenario, mode="train", scenario_dir=Path(args.scenario).resolve().parent
    )

    resources.callback(env.close)
    for episode in range(n_episodes):
        obs_dict, info = env.reset()
        for ag in agents.values():
            if hasattr(ag, "reset"):
                ag.reset()

        step = 0
        done_set: set = set()
        any_collision = False
        timeout = False

        while True:
            if not obs_dict or done_set.issuperset(agent_ids):
                break

            actions: Dict[str, np.ndarray] = {}
            for aid, obs in obs_dict.items():
                if aid not in agents:
                    continue
                try:
                    act = agents[aid].act(obs)
                except Exception:
                    act = np.zeros(2, dtype=np.float32)
                actions[aid] = np.asarray(act, dtype=np.float32)

            if not actions:
                break

            obs_dict, _rewards, dones, truncs, info = env.step(actions)
            step += 1

            for aid in agent_ids:
                if info.get(aid, {}).get("collision", False):
                    any_collision = True
                if dones.get(aid, False) or truncs.get(aid, False):
                    done_set.add(aid)
                    if truncs.get(aid, False):
                        timeout = True

            if render:
                env.render()

        status = "TIMEOUT" if timeout else ("COLLISION" if any_collision else "ok")
        console.print_info(f"ep {episode+1:4d}/{n_episodes}  steps={step:5d}  {status}")

        final_infos = {aid: dict(info.get(aid, {})) for aid in agent_ids}
        focal_info = final_infos.get(agent_ids[0], {}) if agent_ids else {}
        focal_info["outcome"] = focal_info.get("terminal_reason") or status.lower()
        csv_logger.log_training_episode(
            episode,
            reward=0.0,
            info=focal_info,
            metrics={
                "episode_steps": step,
                "collision": int(any_collision),
                "timeout": int(timeout),
                "agent_outcomes": {
                    aid: str(payload.get("terminal_reason") or status.lower())
                    for aid, payload in final_infos.items()
                },
                "agent_terminal_reasons": {
                    aid: payload.get("terminal_reason")
                    for aid, payload in final_infos.items()
                },
                "agent_finish_positions": {
                    aid: payload.get("finish_position")
                    for aid, payload in final_infos.items()
                },
                "agent_lap_counts": {
                    aid: int(payload.get("lap_count", 0))
                    for aid, payload in final_infos.items()
                },
            },
        )

        if wandb_logger:
            wandb_logger.log_metrics({
                "episode/number": episode,
                "episode/steps": step,
                "episode/failed": int(any_collision),
                "episode/timeout": int(timeout),
            })
    console.print_info("Done.")


