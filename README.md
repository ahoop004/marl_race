Marl racing

Reward and observation definitions live inline in the checked-in scenarios.
Both composers also accept user-supplied YAML files, including YAML includes.

Supported observations, in vector order: `lidar`, `frenet_vehicle_track`,
`frenet_neighbors`. The driving prefix is 158 values with 108 LiDAR beams and
20 track-preview points; the current team layout appends neighbors and vehicle
identities for 192 values. Ordering, normalization and identity encoding are
part of checkpoint compatibility.

Supported rewards: `progress_delta_bonus`, `lap_completion`, `timeout_penalty`,
`step_time_penalty`, `team_race_result`, `collision`, `opponent_crash_bonus`,
`team_support`. The asymmetric render scenario uses the last three rewards.
Unknown component keys fail explicitly, including disabled entries and former
reward aliases. The old reward/observation presets have been removed.

PPO actor transfer to MAPPO, including per-agent LoRA, remains supported.
The former `adapter_transfer` option requiring a designated-target observation
layout has been removed.

Run configuration checks with `python3 -m pytest`.
