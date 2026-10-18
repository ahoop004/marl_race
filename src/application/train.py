"""Assemble one task and learner, then run training with owned resources."""
from contextlib import ExitStack
import os
import time
from pathlib import Path
from typing import Optional

from application.checkpoints import resolve_checkpoint_path, validate_resume_experiment
from agents.common.checkpoints import read_checkpoint
from core.task_builder import create_race_task
from core.run_id import resolve_run_id, set_run_id_env
from core.provenance import build_run_provenance, learner_parameter_counts
from loggers.csv_logger import CSVLogger
from loggers.wandb_logger import WandbLogger
from loggers.metric_policy import MetricPolicy
from training.algorithms import select_algorithm, learner_params, create_learner, create_trainer
from training.selection import create_selection_evaluator
from training.runtime import seed_process
from training.hooks import (CSVHook, CheckpointHook, ConsoleHook,
                            EvaluationCheckpointHook, WandbHook)


def run_training(scenario, args, console, scenario_dir, roles):
    from training.distributed import distributed_execution
    with distributed_execution(
            scenario["experiment"].get("distributed", False),
            timeout_seconds=scenario["experiment"].get("distributed_timeout_seconds", 1800)) as distributed:
        with ExitStack() as resources:
            _run_training(scenario, args, console, scenario_dir, roles, resources, distributed)


def _run_training(scenario, args, console, scenario_dir, roles, resources, distributed=None):
    initialization_started = time.perf_counter()
    primary = distributed is None or distributed.primary
    agent_configs = scenario["agents"]
    exp_cfg = scenario["experiment"]
    env_cfg = scenario["environment"]
    trainable_ids = list(roles.policy_agents)
    resume_path = scenario.get("checkpoint", {}).get("resume")
    resume_path = resolve_checkpoint_path(resume_path, operation="resume") if resume_path else None
    resume_payload = read_checkpoint(resume_path) if resume_path else None
    if resume_payload:
        validate_resume_experiment(resume_payload["metadata"], scenario)
        saved = (resume_payload.get("training") or {}).get("distributed")
        if bool(saved) != bool(distributed):
            raise ValueError("Recovery must retain distributed execution mode; use transfer to change mode")
        if distributed:
            workers = distributed.layout(exp_cfg.get("parallel_envs", 1), exp_cfg.get("envs_per_worker", 1),
                                         exp_cfg.get("torch_threads", 1))
            if (saved["world_size"], saved["workers_per_rank"], saved["envs_per_worker"]) != (
                    distributed.world_size, workers, exp_cfg.get("envs_per_worker", 1)):
                raise ValueError("Distributed recovery requires the same world size and collector layout")
    initial_checkpoint = None
    if args.checkpoint and not resume_path:
        if len(trainable_ids) != 1 or str(
            agent_configs[trainable_ids[0]]["algorithm"]
        ).strip().lower() != "ppo":
            raise ValueError("--checkpoint for training currently supports one PPO learner only.")
        initial_checkpoint = resolve_checkpoint_path(args.checkpoint, operation="transfer")

    rl_agent_id = trainable_ids[0]
    algorithm = select_algorithm(scenario, trainable_ids)

    run_id = args.run_id or (resume_payload["metadata"]["provenance"].get("run_id") if resume_payload else None) or resolve_run_id(
        scenario_name=exp_cfg.get("name"),
        algorithm=algorithm,
        seed=exp_cfg.get("seed"),
    )
    if distributed:
        run_id = distributed.broadcast(run_id)
    output_dir = args.output_dir or (str(resume_path.parent) if resume_path else None) or os.path.join(
        scenario.get("paths", {}).get("output_root", "outputs"), exp_cfg.get("name", "unnamed"), run_id
    )
    set_run_id_env(run_id)

    # Loggers
    wandb_cfg = scenario.get("wandb", {})
    wandb_enabled = wandb_cfg.get("enabled", False) and primary
    wandb_logger: Optional[WandbLogger] = None
    if wandb_enabled:
        saved_wandb_id = (resume_payload["metadata"]["provenance"].get("wandb_run_id")
                          if resume_payload else None)
        wandb_identity = {}
        if saved_wandb_id:
            wandb_identity = {"id": saved_wandb_id, "resume": "must"}
        elif wandb_cfg.get("id"):
            wandb_identity = {"id": wandb_cfg["id"], "resume": "allow"}
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
            **wandb_identity,
            run_id=run_id,
        )
        resources.callback(wandb_logger.finish)

    render = bool(env_cfg.get("render", False))
    seed_process(exp_cfg.get("seed"))
    task = create_race_task(scenario, scenario_dir=scenario_dir, roles=roles,
                            render_mode="human" if render else None)
    resources.callback(task.close)
    spec = task.spec
    params = learner_params(scenario, spec, algorithm)
    if distributed:
        if render:
            raise ValueError("Distributed training requires headless CPU environments")
        params["device"] = str(distributed.device)

    pretrained_actor_path: Optional[Path] = None
    if getattr(args, "pretrained_actor", None):
        params["pretrained_actor_checkpoint"] = str(resolve_checkpoint_path(args.pretrained_actor, operation="transfer"))
    pretrained_actor_value = params.get("pretrained_actor_checkpoint")
    if algorithm == "mappo" and pretrained_actor_value and not resume_path:
        pretrained_actor_path = resolve_checkpoint_path(
            str(pretrained_actor_value), operation="transfer", base_dir=scenario_dir)
        params["pretrained_actor_checkpoint"] = str(pretrained_actor_path)
    for source in (initial_checkpoint, pretrained_actor_path):
        if source is not None and Path(output_dir).resolve() == source.parent:
            raise ValueError("Use a new --output-dir for transfer to preserve the source checkpoints.")
    if params.get("require_pretrained_actor", False) and pretrained_actor_path is None and not resume_path:
        raise ValueError("This comparison requires --pretrained-actor or pretrained_actor_checkpoint for both training arms")

    if params.get("lora") is not None:
        if algorithm != "mappo":
            raise ValueError("LoRA is supported for MAPPO actor transfer only")
        if pretrained_actor_path is None and not resume_path:
            raise ValueError("LoRA training requires --pretrained-actor or pretrained_actor_checkpoint")

    # --- Startup banner ---
    maps_raw = env_cfg.get(
        "maps", env_cfg.get("map_bundles", env_cfg.get("map", "?"))
    )
    maps_str = ", ".join(maps_raw) if isinstance(maps_raw, list) else str(maps_raw)
    seed_str = str(exp_cfg.get("seed", "random"))
    from utils.torch_io import resolve_device, resolve_collection_device
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
    workers = exp_cfg.get("parallel_envs", 1)
    envs_per_worker = exp_cfg.get("envs_per_worker", 1)
    if distributed:
        workers = distributed.layout(workers, envs_per_worker, exp_cfg.get("torch_threads", 1))
    total_envs = workers * envs_per_worker
    collection_backend = exp_cfg.get("collection_backend", "process")
    centralized = collection_backend == "centralized"
    process_collection = total_envs > 1 or centralized or distributed is not None
    collection_device = resolve_collection_device(
        str(distributed.device) if distributed else params.get("collection_device"),
        learner.device, workers, envs_per_worker, collection_backend, distributed=distributed is not None)
    console.print_info(f"collection_device={collection_device}  learner_device={learner.device}  "
                       f"workers={workers}  envs_per_worker={envs_per_worker}  environment_slots={total_envs}")
    recovery_state = None
    if resume_payload:
        recovery_state, _ = learner.resume_training(resume_payload)
        console.print_info(f"Recovering from {resume_path}; partial episode discarded, environment schedules restart at the training seed.")
    elif initial_checkpoint is not None:
        learner.initialize_from_checkpoint(initial_checkpoint, scope=scenario["adaptation"]["load_scope"],
            observation_extension=params.get("pretrained_observation_extension"))
        console.print_info(f"Initialized PPO actor and critic from {initial_checkpoint}; fresh optimizer and schedule.")
    elif pretrained_actor_path is not None:
        learner.initialize_from_checkpoint(pretrained_actor_path, scope=scenario["adaptation"]["load_scope"],
            observation_extension=params.get("pretrained_actor_observation_extension"))
        console.print_info(f"Initialized MAPPO actors from checkpoint: {pretrained_actor_path}")
    tune_metadata = wandb_cfg.get("tune")
    expected_source = tune_metadata.get("source_sha256") if isinstance(tune_metadata, dict) else None
    if expected_source and (learner.source_checkpoint or {}).get("sha256") != expected_source:
        raise ValueError("Loaded PPO source differs from the frozen sweep SHA-256")
    counts = learner_parameter_counts(learner)
    if getattr(learner, "lora_config", None) is not None:
        trainable, total = counts["actor_trainable"], counts["actor_total"]
        console.print_info(f"LoRA mode={learner.lora_config['mode']} rank={learner.lora_config['rank']} "
                           f"actor_trainable={trainable}/{total}; centralized critic fully trainable")

    if distributed:
        distributed.attach(learner)
        resources.callback(distributed.detach, learner)
        seed_process((int(exp_cfg.get("seed") or 0) + distributed.rank) % 2**32)

    provenance = build_run_provenance(
        scenario,
        scenario_path=args.scenario,
        run_id=run_id,
        algorithm=algorithm,
        trainable_agents=trainable_ids,
        observation_contracts=spec.observation_contracts,
        devices=(learner.device, collection_device),
    )
    if learner.source_checkpoint:
        provenance["pretrained_source"] = learner.source_checkpoint
    if resume_payload:
        provenance["recovery_source"] = resume_payload["_identity"]
    if wandb_logger:
        provenance["wandb_run_id"] = wandb_logger.wandb_run_id
    provenance["network"] = learner.network_config
    provenance["parameter_counts"] = counts
    if getattr(learner, "lora_config", None) is not None:
        provenance["lora"] = {**learner.lora_contract, "target": "actor_head",
                              "target_modules": learner.actor.target_modules}
    if distributed:
        provenance["distributed"] = distributed.metadata()
        provenance["distributed"].update(global_workers=exp_cfg.get("parallel_envs", 1),
                                          workers_per_rank=workers, envs_per_worker=envs_per_worker)
    provenance["execution"] = {
        "requested_device": scenario.get("execution", {}).get("params", {}).get("device", params.get("device", "cpu")),
        "requested_collection_device": params.get("collection_device"),
        "collection_device": str(collection_device),
        "learner_device": str(learner.device),
        "policy_device": str(collection_device),
        "environment_device": "cpu",
        "rollout_storage_device": "cpu",
        "update_device": str(learner.device),
        "torch_threads": exp_cfg.get("torch_threads", 1),
        "parallel_envs": workers,
        "envs_per_worker": envs_per_worker,
        "collection_backend": collection_backend,
        "total_envs": total_envs,
        "measure_services": exp_cfg.get("measure_services", False),
        "measure_stages": exp_cfg.get("measure_stages", False),
        "measure_cuda_sync": exp_cfg.get("measure_cuda_sync", False),
        "profiling_detail_samples": exp_cfg.get("profiling_detail_samples", 0),
        "measure_mpc": exp_cfg.get("measure_mpc", False),
        "measure_hybrid": exp_cfg.get("measure_hybrid", False),
        "profiling_warmup_updates": exp_cfg.get("profiling_warmup_updates", 1),
        "compile_networks": dict(learner.compilation_config),
        "compilation_cuda_graphs": False,
    }
    provenance["algorithm_implementation"] = {
        "backend": "torchrl",
        "loss": "torchrl.ClipPPOLoss" if algorithm == "ppo" else "torchrl.MAPPOLoss",
        "advantage_estimator": "torchrl.GAE" if algorithm == "ppo" else "torchrl.MultiAgentGAE",
        "update_version": params.get("update_version"),
    }
    provenance["collector"] = {
        "implementation": "torchrl.collectors.Collector",
        "backend": "process" if process_collection and not centralized else "direct",
        "task_adapter": "native_envbase",
        "mode": "synchronous" if process_collection else "serial",
        "num_collectors": 1 if centralized else workers,
        "envs_per_worker": envs_per_worker,
        "num_envs": total_envs,
        "environment_batch": ("ParallelEnv(SerialEnv)" if envs_per_worker > 1 else "ParallelEnv")
                             if centralized else ("SerialEnv" if envs_per_worker > 1 else "unbatched"),
        "inference": "centralized" if centralized else ("worker" if process_collection else "local"),
        "frames_per_batch": learner.frames_per_batch,
        **({"rank_frames_per_batch": (learner.frames_per_batch + distributed.world_size - 1) // distributed.world_size,
            "sample_targets_are_global": True} if distributed else {}),
        "learning_sample_target": learner.n_steps,
        "learning_sample_unit": "active_learner_samples",
        "effective_frames_per_batch": ((learner.frames_per_batch + total_envs - 1)
                                       // total_envs * total_envs),
        "worker_seed_scheme": "(experiment.seed + environment_id) modulo 2**32",
        "environment_id_scheme": "worker_id * envs_per_worker + local_env_id",
        "sync": True,
        "reset_at_each_iter": False,
        "set_truncated": False,
        "use_buffers": not process_collection,
        "environment_use_buffers": centralized,
        "track_policy_version": process_collection,
        "no_cuda_sync": not exp_cfg.get("measure_cuda_sync", False),
        "num_sub_threads": exp_cfg.get("torch_threads", 1),
    }
    # TorchRL reads this at import time; set WEIGHT_SYNC_TIMEOUT in the launch
    # environment, never as a Hydra override after imports have already run.
    from torchrl._utils import WEIGHT_SYNC_TIMEOUT
    provenance["collector"]["initial_weight_sync_timeout_seconds"] = WEIGHT_SYNC_TIMEOUT
    provenance["collector"]["measure_startup"] = exp_cfg.get("measure_collector_startup", False)
    hooks = []
    if primary:
        csv_logger = CSVLogger(
            output_dir=output_dir,
            scenario_config=scenario,
            provenance=provenance,
            append=bool(resume_path),
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
                summary_every=int(os.environ.get("F110_SUMMARY_EVERY", "25")),
                lap_completion=compact_laps,
            ),
            CSVHook(csv_logger),
            CheckpointHook(learner, output_dir, provenance=provenance, configuration=scenario,
                           save_every_steps=scenario["checkpoint"]["every_steps"],
                           save_every_updates=scenario["checkpoint"]["every_updates"]),
        ]
        if wandb_logger:
            hooks.append(WandbHook(wandb_logger))

        if (params.get("_physics_contract") is not None
                and (scenario.get("logging") or {}).get("physics_episodes", True)):
            from training.hooks import PhysicsEpisodeHook
            hooks.append(PhysicsEpisodeHook(output_dir))

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
                console=console, wandb_logger=wandb_logger, configuration=scenario,
            ))
            console.print_info(f"Selection evaluation: {evaluator.episodes} fixed-seed races; "
                               f"completion={evaluator.completion}")
    trainer = create_trainer(algorithm, task, learner, hooks=hooks, render=render, run_id=run_id,
                             collection_device=str(collection_device),
                             parallel_envs=workers, envs_per_worker=envs_per_worker,
                             collection_backend=collection_backend, distributed=distributed,
                             scenario=scenario, scenario_dir=scenario_dir,
                             measure_services=exp_cfg.get("measure_services", False),
                             measure_stages=exp_cfg.get("measure_stages", False),
                             measure_cuda_sync=exp_cfg.get("measure_cuda_sync", False),
                             profiling_detail_samples=exp_cfg.get("profiling_detail_samples", 0),
                             measure_mpc=exp_cfg.get("measure_mpc", False),
                             measure_hybrid=exp_cfg.get("measure_hybrid", False),
                             profiling_warmup_updates=exp_cfg.get("profiling_warmup_updates", 1))
    trainer.initialization_seconds = time.perf_counter() - initialization_started
    if recovery_state is not None:
        trainer.restore_training_state(recovery_state)
    total_steps = exp_cfg.get("total_steps")
    episodes = 0 if total_steps is not None else int(exp_cfg.get("episodes", 1000))
    budget = f"{total_steps} joint environment decisions" if total_steps is not None else f"{episodes} episodes"
    console.print_info(f"Starting {algorithm.upper()} training for {budget}")
    options = {"total_steps": total_steps} if total_steps is not None else {}
    trainer.train(episodes, **options)
    if primary and (trainer.measure_services or trainer.measure_stages or trainer.measure_mpc or trainer.measure_hybrid
                    or trainer.measure_collector_startup):
        # Include end-of-run service work, especially latest/final checkpoint saves.
        csv_logger.save_summary({**trainer.checkpoint_progress(), **trainer.service_metrics(),
                                 **trainer.performance_metrics(),
                                 **({"mpc_profile": trainer.mpc_summary()} if trainer.measure_mpc else {}),
                                 **({"hybrid_profile": trainer.hybrid_summary()} if trainer.measure_hybrid else {})})
