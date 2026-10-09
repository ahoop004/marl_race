# MARL racing

Reinforcement learning experiments for racing.

The shared racing task contract is defined in
[`src/tasks/contracts.py`](src/tasks/contracts.py). It keeps `RaceEnv` independent
of Gymnasium, PettingZoo, and TorchRL, and specifies the boundary for extracting
the duplicated task behavior from the existing PPO and MAPPO trainers.

Stage 1 defines typed snapshots, agent decisions, physical substeps, and the task
interface. The current trainers continue to run directly against `RaceEnv` until
the shared task implementation is extracted in stage 2.

Run the regression tests with `python3 -m pytest -q` after installing the
development dependencies from `pyproject.toml`.
