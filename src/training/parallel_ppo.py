"""Grouped PPO collectors: several environments per CPU process, one learner.

Each environment runs the task collector's generator and stores CPU TensorDict
fragments with its own reset boundaries, seed and budget. The parent batches inference and freezes
weights until every live environment has delivered its next fragment.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path
from contextlib import ExitStack

import numpy as np

from training.worker_runtime import (
    worker_settings, close_workers, report_worker_error, receive_worker,
    worker_thread_limits, collector_scenario,
    initialize_worker, start_worker, worker_assignments,
)

from training.hooks import transition_record_hooks
from training.collector_scheduling import (
    CollectorEventSink, CollectorScheduler,
)
from training.ppo_collector import PPOCollector
from training.collector_hooks import WorkerHook
from training.torchrl_collectors import PPOCollectorState, deserialize_rollout, serialize_rollout


def _make_collector(scenario, directory, agent_id, env_id, quota, horizon,
                    run_id, sink, record, step_budget):
    from core.task_builder import create_race_task
    from training.runtime import seed_process

    scenario, seed = collector_scenario(scenario, env_id)
    seed_process(seed)
    task = create_race_task(scenario, scenario_dir=Path(directory))
    try:
        if task.possible_agents != (agent_id,):
            raise ValueError("PPO collector requires the configured policy agent")
        spec = task.spec
        policy = PPOCollectorState(horizon)
        trainer = PPOCollector(
            task, policy, hooks=[WorkerHook(sink, env_id, seed, record)],
            run_id=f'{run_id}_worker{env_id:03d}',
        )
        generator = trainer.iter_train(0 if step_budget else quota,
                                      total_steps=quota if step_budget else None)
        return task, generator, (spec.observation_dims[agent_id], spec.action_lows[agent_id], spec.action_highs[agent_id])
    except BaseException:
        task.close()
        raise


def _collect_worker(connection, scenario, directory, agent_id, assignments,
                    horizon, run_id, record, step_budget):
    initialize_worker()
    tasks, generators, pending = {}, {}, {}
    sink = CollectorEventSink()
    try:
        contracts = []
        for env_id, quota in assignments:
            tasks[env_id], generators[env_id], contract = _make_collector(
                scenario, directory, agent_id, env_id, quota, horizon, run_id,
                sink, record, step_budget)
            contracts.append(contract)
        connection.send(('ready', contracts))
        if connection.recv() != ('start', None):
            raise RuntimeError('PPO collector expected start after readiness')

        def advance(env_id, response=None):
            try:
                pending[env_id] = generators[env_id].send(response)
            except StopIteration:
                pending.pop(env_id, None)

        for env_id in generators:
            advance(env_id)
        while pending:
            paused = {}
            while set(pending) - paused.keys():
                requests = {}
                for env_id, (kind, payload) in pending.items():
                    if env_id in paused:
                        continue
                    if kind == 'rollout':
                        payload = serialize_rollout(payload)
                        paused[env_id] = payload
                    else:
                        requests[env_id] = (kind, payload)
                if requests:
                    connection.send(('requests', (requests, sink.take())))
                    for env_id, response in connection.recv().items():
                        advance(env_id, response)
            connection.send(('rollout', (paused, sink.take())))
            reply = connection.recv()
            for env_id in sorted(paused):
                advance(env_id, reply)
        connection.send(('done', sink.take()))
    except (BrokenPipeError, EOFError, ConnectionResetError):
        pass
    except BaseException:
        report_worker_error(connection)
    finally:
        for task in tasks.values():
            task.close()
        connection.close()


def train_parallel(trainer, scenario, directory, num_envs, n_episodes, *, total_steps=None):
    experiment = scenario.get('experiment', {})
    workers = min(num_envs, int(experiment.get('num_workers', num_envs)))
    agent = trainer.agent
    horizon = agent.n_steps // num_envs
    budget = total_steps if total_steps is not None else n_episodes
    if workers < 1 or horizon < 1 or agent.n_steps % num_envs or budget < num_envs:
        raise ValueError('Grouped PPO requires positive workers, n_steps divisible by num_envs, and budget >= num_envs')
    startup = worker_settings(scenario)
    scheduler = CollectorScheduler(experiment.get("collector_scheduling", "synchronous"),
                                   startup["worker_response_timeout_s"])
    record_hooks = transition_record_hooks(trainer.hooks)
    context = mp.get_context('spawn')
    connections, processes, process_by_worker = {}, [], {}
    waiting = {}
    completed = collected = updates = 0
    stopped = success = False
    started = time.perf_counter()
    trainer._set_training_progress(0, budget)
    resources = ExitStack()

    def receive(worker_id, starting=False):
        timeout = startup["worker_startup_timeout_s" if starting else "worker_response_timeout_s"]
        return receive_worker(
            connections[worker_id], process_by_worker[worker_id],
            label=f"PPO worker {worker_id}", timeout=timeout,
        )

    def process_events(items):
        nonlocal completed
        for kind, payload in items:
            if kind == 'transition':
                for hook in record_hooks:
                    hook.on_step(payload)
            elif kind == 'episode_start':
                for hook in trainer.hooks:
                    hook.on_episode_start(payload)
            elif kind == 'episode':
                if total_steps is None:
                    trainer._set_training_progress(completed + 1, budget)
                for hook in trainer.hooks:
                    hook.on_episode_end(completed, *payload)
                completed += 1
            else:
                raise RuntimeError(f'Unexpected PPO event: {kind}')

    def events(items):
        event_start = time.monotonic()
        try:
            process_events(items)
        finally:
            scheduler.exclude_parent_time(time.monotonic() - event_start)

    try:
        resources.enter_context(worker_thread_limits())
        batch_size = startup['worker_startup_batch_size']
        for start in range(0, workers, batch_size):
            batch = range(start, min(start + batch_size, workers))
            for worker_id in batch:
                assignments = worker_assignments(worker_id, workers, num_envs, budget)
                parent, process = start_worker(context, _collect_worker, (
                    scenario, str(directory), trainer.rl_agent_id, assignments,
                    horizon, trainer.run_id, bool(record_hooks), total_steps is not None),
                    name=f'ppo-collector-{worker_id}')
                connections[worker_id] = parent
                processes.append(process)
                process_by_worker[worker_id] = process
            for worker_id in batch:
                kind, contracts = receive(worker_id, starting=True)
                if kind != 'ready' or any(
                    dim != agent.obs_dim or not np.array_equal(low, agent.action_low)
                    or not np.array_equal(high, agent.action_high) for dim, low, high in contracts
                ):
                    raise ValueError(f'PPO worker {worker_id} observation/action contract mismatch')
            print(f'[PPO] Initialized {min(start + batch_size, workers)}/{workers} workers '
                  f'for {num_envs} environments', flush=True)
        startup_s = time.perf_counter() - started
        for connection in connections.values():
            connection.send(('start', None))
        round_start = time.perf_counter()
        while connections:
            requests = {}
            for worker_id in scheduler.workers(connections, waiting):
                if worker_id in waiting:
                    continue
                kind, payload = receive(worker_id)
                scheduler.received(worker_id)
                if kind == 'requests':
                    batch, items = payload
                    events(items)
                    requests.update({(worker_id, env_id): request for env_id, request in batch.items()})
                elif kind == 'rollout':
                    rollouts, items = payload
                    events(items)
                    waiting[worker_id] = rollouts
                elif kind == 'done':
                    events(payload)
                    connections.pop(worker_id).close()
                else:
                    raise RuntimeError(f'Unexpected PPO worker message: {kind}')
            if requests:
                responses = {}
                for kind in ('value', 'act'):
                    keys = sorted(key for key, request in requests.items() if request[0] == kind)
                    if not keys:
                        continue
                    observations = np.stack([requests[key][1] for key in keys])
                    if kind == 'value':
                        responses.update(zip(keys, map(float, agent.value_batch(observations))))
                    else:
                        output = agent.sample_batch(observations)
                        for row, key in enumerate(keys):
                            responses[key] = (output.actions[row], float(output.log_probs[row]), float(output.values[row]),
                                              output.raw_actions[row])
                if len(responses) != len(requests):
                    raise RuntimeError('Unknown PPO inference request')
                for worker_id in sorted({key[0] for key in requests}):
                    connections[worker_id].send({env_id: response for (worker, env_id), response
                                                 in responses.items() if worker == worker_id})
            if waiting and len(waiting) == len(connections):
                collection_s = time.perf_counter() - round_start
                rollouts = [rollout for worker in sorted(waiting)
                            for _, rollout in sorted(waiting[worker].items())]
                rollouts = [deserialize_rollout(rollout) for rollout in rollouts]
                steps = sum(rollout.numel() for rollout in rollouts)
                collected += steps
                trainer.collected_steps = collected
                if total_steps is not None:
                    trainer._set_training_progress(collected, budget)
                update_start = time.perf_counter()
                metrics = agent.update_rollouts(rollouts) if not trainer._should_stop() else {}
                update_s = time.perf_counter() - update_start
                did_update = bool(metrics)
                updates += did_update
                metrics.update({
                    'train/environment_steps': collected, 'train/updates': updates,
                    'perf/startup_seconds': startup_s, 'perf/collection_seconds': collection_s,
                    'perf/update_seconds': update_s,
                    'perf/collection_env_steps_per_second': steps / max(collection_s, 1e-9),
                    'perf/round_env_steps_per_second': steps / max(collection_s + update_s, 1e-9),
                    'perf/elapsed_seconds': time.perf_counter() - started,
                    'perf/end_to_end_env_steps_per_second': collected / max(time.perf_counter() - started, 1e-9),
                    'perf/num_workers': workers, 'perf/num_envs': num_envs,
                })
                if did_update:
                    for hook in trainer.hooks:
                        hook.on_update(metrics)
                stopped = trainer._should_stop()
                reply = metrics
                if stopped:
                    reply = {'metrics': metrics, 'collector_control': {'stop': True}}
                for worker_id in waiting:
                    connections[worker_id].send(reply)
                waiting.clear()
                scheduler.reset()
                round_start = time.perf_counter()
        if not stopped and total_steps is not None and collected != total_steps:
            raise RuntimeError(f'PPO collected {collected} of {total_steps} transitions')
        if not stopped and total_steps is None and completed != n_episodes:
            raise RuntimeError(f'PPO completed {completed} of {n_episodes} episodes')
        if not stopped:
            trainer._flush_pending_update()
        for hook in trainer.hooks:
            hook.on_training_end()
        success = True
    finally:
        try:
            close_workers(list(connections.values()), processes, failed=not success)
        finally:
            resources.close()
