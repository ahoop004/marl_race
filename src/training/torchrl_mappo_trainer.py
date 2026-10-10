from adapters import RaceParallelEnv
from training.marl_trainer import MARLTrainer


class _TrainingParallelEnv(RaceParallelEnv):
    on_physics_step = None

    def _on_physics_step(self, substep):
        if self.on_physics_step is not None:
            self.on_physics_step(substep)


class TorchRLMAPPOTrainer(MARLTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parallel_env = _TrainingParallelEnv(self.task)

    def train_parallel(self, scenario, scenario_dir, num_envs, n_episodes=0, *, total_steps=None):
        from training.parallel_mappo import train_parallel
        return train_parallel(self, scenario, scenario_dir, num_envs, n_episodes, total_steps=total_steps)

    def _reset_task(self):
        self.parallel_env.reset()
        return self.parallel_env.snapshot

    def _step_task(self, actions, on_physics_step):
        self.parallel_env.on_physics_step = on_physics_step
        if self.parallel_env.agents:
            self.parallel_env.step(actions)
        else:
            self.parallel_env.advance_fixed_agents()
        return self.parallel_env.last_step
