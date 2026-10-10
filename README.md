# MARL racing

Reinforcement learning experiments for racing.

Install training and test dependencies with `python3 -m pip install -e '.[rl,dev]'`
(or `python3 -m pip install -r requirements.txt`). The `rl` extra includes
Gymnasium, PettingZoo, TorchRL and TensorDict; MAPPO requires the pinned TorchRL
0.14 series.

The PPO scenarios select `experiment.ppo_backend: torchrl`; the 2v2 MAPPO
scenarios select `experiment.mappo_backend: torchrl`. Serial PPO collection uses
`RaceGymEnv`, and MAPPO uses `RaceParallelEnv`. Both adapt the same `RaceTask`, which owns
observations, actions, rewards and race lifecycle rules. Fixed MPC opponents
remain inside the task and are not exposed as PettingZoo policy agents.
Parallel PPO workers still use the shared task collection loop and transport
TensorDict rollouts to the TorchRL learner.

For a local PPO run with smaller rollouts:

```bash
python3 run.py --scenario scenarios/ppo_lap_completion_pretrain.yaml \
  --num-envs 1 --num-workers 1 --rollout-steps-per-env 1024 --no-wandb
```

For parallel PPO collection, use `--num-envs 400 --num-workers 100
--rollout-steps-per-env 1024`. PPO `n_steps` counts pooled decisions; the flag
sets it to `num_envs * 1024`. The shipped PPO scenarios retain their existing
409600-decision rollout unless overridden. MAPPO `n_steps` counts joint race
decisions for serial updates, while `training_defaults.rollout_steps_per_env`
sets each parallel environment's horizon. MAPPO actor sample counts also depend
on how many learners remain active.

The scratch, pretrained, shared-actor and per-agent LoRA MAPPO scenarios inherit
the same completion task. To apply the parallel preset, create a YAML file in
`scenarios/` with these includes (paths are relative to the new YAML file):

```yaml
includes:
  - mappo_2v2_completion_scratch.yaml
  - ../configs/training/mappo_parallel.yaml
experiment:
  name: mappo_2v2_completion_scratch_parallel
```

Include the pretrained or LoRA scenario instead of scratch for those arms. Agent
`params` override `training_defaults`; scenario values override included files.
Use `--pretrained-actor` for MAPPO or `--checkpoint` for PPO to select a transfer
source. The three-lap PPO transfer scenario inherits the TorchRL backend too.

The old implementations are still available with `--ppo-backend torch` or
`--mappo-backend torch`. Switching implementations changes losses, advantage
estimation and minibatch ordering, so new configs record a TorchRL
`update_version` in run provenance. Before removing the old classes, extract
their shared network setup, checkpoint contracts, transfer/LoRA support and
trainer/collector utilities: the TorchRL implementations currently inherit or
import these. Standalone evaluation must also stop constructing the old agent
classes before they can be deleted. Keep the task, adapters, evaluation, logging and physics code;
they are shared by the refactor. Legacy vehicle dynamics are a separate removal
decision from the old PPO/MAPPO learning code.

