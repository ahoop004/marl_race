"""Assemble one task and learner, then run training with owned resources."""
from contextlib import ExitStack
import hashlib
import os
from pathlib import Path
from typing import Optional

from application.checkpoints import resolve_checkpoint_path, resolve_scenario_relative_path
from core.task_builder import create_race_task
from training.algorithms import resolve_training_params
from core.run_id import resolve_run_id, set_run_id_env
from core.provenance import build_run_provenance
from loggers.csv_logger import CSVLogger
from loggers.wandb_logger import WandbLogger
from loggers.metric_policy import MetricPolicy
from training.algorithms import select_algorithm, learner_params, create_learner, create_trainer
from training.selection import create_selection_evaluator
from training.runtime import seed_process
from training.hooks import (CSVHook, CheckpointHook, ConsoleHook,
                            EvaluationCheckpointHook, MAPPOConsoleHook, WandbHook)


def run_training(scenario, args, console, scenario_dir, roles):
    with ExitStack() as resources:
        _run_training(scenario, args, console, scenario_dir, roles, resources)


def _run_training(scenario, args, console, scenario_dir, roles, resources):
    agent_configs = scenario["agents"]
    exp_cfg = scenario["experiment"]
    env_cfg = scenario["environment"]
    trainable_ids = list(roles.policy_agents)
    initial_checkpoint = None
    if args.checkpoint:
        if len(trainable_ids) != 1 or str(
            agent_configs[trainable_ids[0]]["algorithm"]
        ).strip().lower() != "ppo":
            raise ValueError("--checkpoint for training currently supports one PPO learner only.")
        initial_checkpoint = resolve_checkpoint_path(args.checkpoint)

    rl_agent_id = trainable_ids[0]
    algorithm = select_algorithm(scenario, trainable_ids)
    agent_cfg = agent_configs[rl_agent_id]

    run_id = args.run_id or resolve_run_id(
        scenario_name=exp_cfg.get("name"),
        algorithm=algorithm,
        seed=exp_cfg.get("seed"),
    )
    output_dir = args.output_dir or os.path.join(
        "outputs", exp_cfg.get("name", "unnamed"), run_id
    )
    if initial_checkpoint is not None and Path(output_dir).resolve() == initial_checkpoint.parent:
        raise ValueError("Use a new --output-dir for PPO transfer to preserve the source checkpoints.")
    set_run_id_env(run_id)

    # Loggers
    wandb_cfg = scenario.get("wandb", {})
    wandb_enabled = wandb_cfg.get("enabled", False)
    wandb_logger: Optional[WandbLogger] = None
    if wandb_enabled:
        wandb_logger = WandbLogger(
            project=wandb_cfg.get("project", "f110"),
            name=wandb_cfg.get("name", exp_cfg.get("name")),
            config=scenario,
            tags=wandb_cfg.get("tags", []),
            group=wandb_cfg.get("group"),
            job_type=wandb_cfg.get("job_type", algorithm),
            entity=wandb_cfg.get("entity"),
            notes=wandb_cfg.get("notes"),
            mode=wandb_cfg.get("mode", "online"),
            logging_config=wandb_cfg.get("logging"),
            run_id=run_id,
        )
        resources.callback(wandb_logger.finish)

    render = bool(env_cfg.get("render", False))
    seed_process(exp_cfg.get("seed"))
    task = create_race_task(scenario, scenario_dir=scenario_dir, roles=roles,
                            render_mode="human" if render else None)
    resources.callback(task.close)
    # Preserve the legacy MAPPO training reset sequence; current physics needs
    # no sizing reset and exposes the state contract before episode zero.
    if algorithm == "mappo" and resolve_training_params(agent_cfg, scenario)["_physics_contract"] is None:
        task.env.reset()
    spec = task.spec
    params = learner_params(scenario, spec, algorithm)
    if initial_checkpoint is not None:
        params["_initial_checkpoint_sha256"] = hashlib.sha256(initial_checkpoint.read_bytes()).hexdigest()

    pretrained_actor_path: Optional[Path] = None
    if getattr(args, "pretrained_actor", None):
        params["pretrained_actor_checkpoint"] = str(resolve_checkpoint_path(args.pretrained_actor))
    pretrained_actor_value = params.get("pretrained_actor_checkpoint")
    if algorithm == "mappo" and pretrained_actor_value:
        pretrained_actor_path = resolve_scenario_relative_path(
            str(pretrained_actor_value), scenario_dir
        )
        if pretrained_actor_path.is_dir():
            pretrained_actor_path = pretrained_actor_path / "best_model.pt"
        if not pretrained_actor_path.is_file():
            raise FileNotFoundError(
                f"Pretrained actor checkpoint not found: {pretrained_actor_path}"
            )
        params["_resolved_pretrained_actor_checkpoint"] = str(pretrained_actor_path)

    if params.get("require_pretrained_actor", False) and pretrained_actor_path is None:
        raise ValueError("This comparison requires --pretrained-actor or pretrained_actor_checkpoint for both training arms")

    if params.get("lora") is not None:
        if algorithm != "mappo":
            raise ValueError("LoRA is supported for MAPPO actor transfer only")
        if pretrained_actor_path is None:
            raise ValueError("LoRA training requires --pretrained-actor or pretrained_actor_checkpoint")

    # --- Startup banner ---
    maps_raw = env_cfg.get(
        "maps", env_cfg.get("map_bundles", env_cfg.get("map", "?"))
    )
    maps_str = ", ".join(maps_raw) if isinstance(maps_raw, list) else str(maps_raw)
    seed_str = str(exp_cfg.get("seed", "random"))
    from utils.torch_io import resolve_device
    device_str = str(resolve_device([params.get("device", "cpu")]))
    trainable_str = ", ".join(trainable_ids)
    fixed_str = ", ".join(task.fixed_policy_agents) or "none"

    console.print_header(
        f"Training: {exp_cfg.get('name', 'unnamed')}",
        f"algorithm={algorithm}  trainable=({trainable_str})  fixed=({fixed_str})",
    )
    console.print_info(
        f"map={maps_str}  seed={seed_str}  device={device_str}  "
        f"obs_dim={spec.observation_dims[rl_agent_id]}  action_dim={len(spec.action_lows[rl_agent_id])}"
    )

    learner = create_learner(algorithm, spec, params)
    if initial_checkpoint is not None:
        learner.load(str(initial_checkpoint), load_optimizer=False,
                     observation_extension=params.get("pretrained_observation_extension"))
        if hashlib.sha256(initial_checkpoint.read_bytes()).hexdigest() != params["_initial_checkpoint_sha256"]:
            raise ValueError("Checkpoint changed while loading for training; use a stable checkpoint file.")
        console.print_info(f"Initialized PPO actor and critic from {initial_checkpoint}; fresh optimizer and schedule.")
    if pretrained_actor_path is not None:
        learner.load_pretrained_actor(str(pretrained_actor_path))
        console.print_info(f"Initialized MAPPO actors from checkpoint: {pretrained_actor_path}")
    if getattr(learner, "lora_config", None) is not None:
        trainable = sum(p.numel() for p in learner.actor.parameters() if p.requires_grad)
        total = sum(p.numel() for p in learner.actor.parameters())
        console.print_info(f"LoRA mode={learner.lora_config['mode']} rank={learner.lora_config['rank']} "
                           f"actor_trainable={trainable}/{total}; centralized critic fully trainable")

    provenance = build_run_provenance(
        scenario,
        scenario_path=args.scenario,
        run_id=run_id,
        algorithm=algorithm,
        trainable_agents=trainable_ids,
    )
    num_envs = int(exp_cfg.get("num_envs", 1))
    if initial_checkpoint is not None:
        provenance["initial_checkpoint"] = {
            "path": str(initial_checkpoint),
            "sha256": params["_initial_checkpoint_sha256"],
            "load_scope": "actor_and_critic",
            "observation_extension": params.get("pretrained_observation_extension"),
            "optimizer_restored": False,
            "training_progress_restored": False,
        }
    if num_envs > 1 and algorithm == "ppo":
        env_seed = env_cfg.get("seed")
        env_seed = exp_cfg["seed"] if env_seed is None else env_seed
        provenance["ppo_collection"] = {
            "backend": "torchrl",
            "advantage_estimator": "torchrl.GAE",
            "mode": "synchronous_grouped_workers_v1",
            "num_envs": num_envs, "worker_threads": 1,
            "collector_scheduling": exp_cfg.get("collector_scheduling", "synchronous"),
            "num_workers": min(num_envs, int(exp_cfg.get("num_workers", num_envs))),
            "worker_seeds": [(exp_cfg["seed"] + i) % (2 ** 32) for i in range(num_envs)],
            "environment_seeds": [(env_seed + i) % (2 ** 32) for i in range(num_envs)],
            "max_steps_per_worker_rollout": int(params.get("n_steps", 2048)) // num_envs,
        }
    if num_envs > 1 and algorithm == "mappo":
        provenance["mappo_collection"] = {
            "backend": "torchrl",
            "advantage_estimator": "torchrl.MultiAgentGAE",
            "mode": "synchronous_grouped_workers_v1", "num_envs": num_envs,
            "num_workers": min(num_envs, int(exp_cfg.get("num_workers", num_envs))),
            "worker_threads": 1,
            "collector_scheduling": exp_cfg.get("collector_scheduling", "synchronous"),
            "rollout_steps_per_env": scenario.get("training_defaults", {}).get("rollout_steps_per_env", 256),
            "environment_step_unit": "joint_environment_decisions_including_opponent_only_steps",
            "policy_seeds": ("parent RNG; arrival ordering" if exp_cfg.get("collector_scheduling") == "ready"
                             else "parent RNG; deterministic worker/environment ordering"),
            "environment_seeds": [((env_cfg.get("seed") if env_cfg.get("seed") is not None
                                    else exp_cfg["seed"]) + i) % (2 ** 32) for i in range(num_envs)],
        }
    if pretrained_actor_path is not None:
        provenance["pretrained_actor"] = {
            "path": str(pretrained_actor_path),
            "sha256": hashlib.sha256(pretrained_actor_path.read_bytes()).hexdigest(),
            "load_scope": "actor_only",
            "observation_extension": params.get("pretrained_actor_observation_extension"),
            "lora": params.get("lora"),
        }
    csv_logger = CSVLogger(
        output_dir=output_dir,
        scenario_config=scenario,
        provenance=provenance,
    )

    resources.callback(csv_logger.close)

    # Hooks
    eval_cfg = scenario.get("evaluation", {}) or {}
    if not isinstance(eval_cfg, dict):
        raise ValueError("Scenario 'evaluation' must be a mapping when provided.")
    evaluation_selection_enabled = bool(eval_cfg.get("enabled", False)) and algorithm in {"ppo", "mappo"}
    compact_laps = MetricPolicy(scenario.get("wandb", {}).get("logging"), scenario).lap_completion

    hooks = [
        ConsoleHook(
            logger=console,
            log_every=int(os.environ.get("F110_LOG_EVERY", "1")),
            summary_every=int(os.environ.get("F110_SUMMARY_EVERY", "25")),
            lap_completion=compact_laps,
            episode_only=compact_laps and num_envs > 1,
        ),
        CSVHook(csv_logger),
        CheckpointHook(
            agent=learner,
            output_dir=output_dir,
            save_every=int(params.get("checkpoint_every", os.environ.get("F110_CHECKPOINT_EVERY", 100))),
            provenance=provenance,
            save_best_training_reward=not evaluation_selection_enabled,
            save_final=algorithm in {"ppo", "mappo"},
            save_every_steps=(int(params.get("checkpoint_every_steps", 4096000))
                              if exp_cfg.get("total_steps") is not None or
                              (algorithm == "mappo" and params.get("checkpoint_every_steps") is not None)
                              else None),
        ),
    ]
    if wandb_logger:
        hooks.append(WandbHook(wandb_logger))

    if params.get("_physics_contract") is not None:
        from training.hooks import PhysicsEpisodeHook
        hooks.append(PhysicsEpisodeHook(output_dir))

    if algorithm == "mappo" and num_envs > 1 and not compact_laps:
        if not exp_cfg.get("terminal_episode_detail", False):
            hooks = [hook for hook in hooks if not isinstance(hook, ConsoleHook)]
        hooks.insert(0, MAPPOConsoleHook(console,
            window=int(exp_cfg.get("terminal_recent_episodes", 100)),
            every_updates=int(exp_cfg.get("terminal_every_updates", 10)),
            diagnostic_every=int(exp_cfg.get("terminal_diagnostic_every_updates", 100))))
    if evaluation_selection_enabled:
        evaluator = create_selection_evaluator(algorithm, scenario, scenario_dir, spec, params)
        resources.callback(evaluator.close)
        evaluator.bind_agent(learner)
        hooks.append(EvaluationCheckpointHook(
            learner, output_dir, evaluator,
            evaluate_every=int(eval_cfg.get("every_episodes", 100)),
            selection_strategy=eval_cfg.get("selection_strategy",
                "team_completion" if algorithm == "mappo" else "completion_progress"),
            evaluate_every_steps=eval_cfg.get("every_steps"), provenance=provenance,
            console=console, wandb_logger=wandb_logger,
        ))
        console.print_info(f"Selection evaluation: {evaluator.episodes} fixed-seed races; "
                           f"completion={evaluator.completion}; workers={getattr(evaluator, 'num_workers', 1)}")
    trainer = create_trainer(algorithm, task, learner, hooks=hooks, render=render, run_id=run_id)
    trainer.console = console
    total_steps = exp_cfg.get("total_steps")
    episodes = 0 if total_steps is not None else int(exp_cfg.get("episodes", 1000))
    budget = f"{total_steps} joint environment decisions" if total_steps is not None else f"{episodes} episodes"
    console.print_info(f"Starting {algorithm.upper()} training for {budget} | num_envs={num_envs}")
    options = {"total_steps": total_steps} if total_steps is not None else {}
    if num_envs > 1:
        trainer.train_parallel(scenario, scenario_dir, num_envs, episodes, **options)
    else:
        trainer.train(episodes, **options)
