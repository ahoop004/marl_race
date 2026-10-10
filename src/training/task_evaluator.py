"""Shared checkpoint-evaluation reporting around the task episode runner."""
import time

from adapters.rewards import RewardMapping
from metrics.racing_eval import aggregate_eval_episodes
from training.evaluation import run_evaluation_episode
from training.runtime import evaluation_mode


class TaskEvaluator:
    completion = "race"
    focal_summary = False

    def __init__(self, *, task, episodes, base_seed, protocol_name="selection",
                 reward_mapping=RewardMapping()):
        if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
            raise ValueError("Evaluation episodes must be a positive integer")
        self.task = task
        self.trainable_ids = list(task.possible_agents)
        self.episodes, self.base_seed = episodes, int(base_seed)
        self.protocol_name, self.reward_mapping = protocol_name, reward_mapping
        self.action_repeat = task.action_repeat
        self.progress_callback = None
        self._next_progress = 0.0
        if not task.episode_limits.is_bounded(
            task.physical_agents, task.possible_agents, completion=self.completion,
        ):
            raise ValueError("Checkpoint evaluation requires a finite horizon for its termination group")

    def set_progress_callback(self, callback):
        previous, self.progress_callback = self.progress_callback, callback
        return previous

    def _report_progress(self, facts, episode, steps, status="running"):
        if self.progress_callback is None:
            return
        now = time.monotonic()
        if status == "running" and now < self._next_progress:
            return
        self._next_progress = now + 1.0
        learners = [facts.agents[aid] for aid in self.trainable_ids]
        limits = self.task.episode_limits
        finishers = limits.finish_agents
        self.progress_callback(dict(episode=episode + 1, episodes=self.episodes,
            map=str(self.task.episode_metadata.map_id),
            status=status, steps=steps, max_steps=self.task.episode_limits.max_steps,
            sim_seconds=steps * self.task.timestep,
            laps=",".join(f'{a.agent_id}:{a.final_lap_count}/'
                          f'{limits.target_laps if a.agent_id in finishers else "unlimited"}' for a in learners),
            outcome=",".join(f'{a.agent_id}:{a.terminal_reason or "active"}' for a in learners)))

    def bind_agent(self, agent):
        self.agent = agent
        return self

    def _evaluate_episode(self, episode, protocol):
        result = run_evaluation_episode(
            self.task, self.agent.evaluation_actions, episode=episode, seed=self.base_seed + episode,
            reward_mapping=self.reward_mapping, completion=self.completion,
            render=getattr(self, "render", False), progress=self._report_progress,
            include_rewards=False,
        )
        physics = ({"seed": self.base_seed + episode, "map_bundle": result.map_id,
                    "physics": result.physics} if result.physics is not None else None)
        return result.facts, str(result.map_id or "unknown"), result.record, physics

    def _collect_episodes(self, protocol):
        return [self._evaluate_episode(episode, protocol) for episode in range(self.episodes)]

    def evaluate(self, agent=None):
        if agent is not None:
            self.bind_agent(agent)
        if not hasattr(self, "agent"):
            raise ValueError("Checkpoint evaluation requires a bound policy")
        protocol = dict(name=self.protocol_name, spawn_schedule="episode_index_v1",
            seeds=list(range(self.base_seed, self.base_seed + self.episodes)),
            max_steps=self.task.episode_limits.max_steps, timestep_s=self.task.timestep,
            target_laps=self.task.episode_limits.target_laps, action_repeat=self.action_repeat,
            completion=self.completion)
        by_map, physics, records, results = {}, [], [], []
        try:
            with evaluation_mode(self.agent):
                for result, map_id, record, physics_record in self._collect_episodes(protocol):
                    results.append(result)
                    by_map.setdefault(map_id, []).append(result)
                    records.append(record)
                    if physics_record is not None:
                        physics.append(physics_record)
        finally:
            if self.progress_callback is not None:
                self.progress_callback(None)
        kwargs = {"timestep": self.task.timestep}
        if self.focal_summary:
            kwargs["focal_agent_id"] = self.trainable_ids[0]
        summary = aggregate_eval_episodes(results, **kwargs)
        summary["per_map"] = {name: aggregate_eval_episodes(rows, **kwargs)
                              for name, rows in by_map.items()}
        summary["episode_results"], summary["evaluation_protocol"] = records, protocol
        for row in [summary, *summary["per_map"].values()]:
            # Selection is fact-based, although rewards follow the same task path.
            for key in ("mean_episode_reward", "per_agent_rewards_mean",
                        "per_agent_individual_rewards_mean", "per_agent_reward_components_mean"):
                row.pop(key, None)
        if self.focal_summary:
            summary["strict_clean_finish_count"] = sum(row["strict_clean_finish"] for row in records)
            for name, row in summary["per_map"].items():
                row["strict_clean_finish_count"] = sum(record["strict_clean_finish"]
                    for record in records if str(record["map_id"] or "unknown") == name)
            if physics:
                protocol["physics_episodes"] = physics
        elif physics:
            summary["physics_episodes"] = physics
        return summary

    def close(self):
        self.task.close()
