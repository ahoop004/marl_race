"""Dashboard selection only; training and checkpoint selection retain all facts."""
from fnmatch import fnmatchcase

from training.evaluation_config import EVALUATION_STRATEGIES


AXES = {
    "episode": "episode/number",
    "train": "train/environment_steps",
    "perf": "train/environment_steps",
    "eval": "eval/environment_steps",
}

CORE = (
    "episode/reward", "episode/steps", "episode/lap_count", "episode/lap_time_s",
    "episode/completed", "episode/failed", "episode/timeout", "episode/net_progress_laps",
    "episode/reward/*", "episode/individual_reward/*",
    "episode/team/completion_rate", "episode/team/all_finished",
    "episode/team/failure_rate", "episode/team/timeout_rate",
    "episode/team/first_place", "episode/team/sweep", "episode/team/rank_score",
    "train/policy_loss", "train/value_loss", "train/entropy", "train/approx_kl",
    "train/*/policy_loss", "train/*/value_loss", "train/*/entropy", "train/*/approx_kl",
    "train/learning_rate", "train/*/learning_rate", "train/kl_early_stop",
    "perf/end_to_end_env_steps_per_second", "perf/update_seconds",
    "eval/completion_rate", "eval/learner_failure_rate", "eval/timeout_rate",
    "eval/mean_net_progress", "eval/mean_clean_finish_time_s", "eval/clean_finish_count",
    "eval/race_count", "eval/evaluation_seconds", "eval/map/*/clean_completion_rate",
)
RACING = (
    "eval/win_rate", "eval/team_both_finished_rate", "eval/team_first_place",
    "eval/team_sweep", "eval/team_rank_score", "eval/team_rank_penalty_score",
    "eval/mean_valid_lap_time_s", "eval/valid_laps", "eval/fastest_valid_lap_s",
)
LAP_COMPLETION = (
    "eval/mean_valid_lap_time_s", "eval/valid_laps", "eval/fastest_valid_lap_s",
    "eval/team_both_finished_rate",
)
class MetricPolicy:
    def __init__(self, config=None, scenario=None):
        self.config = config or {}
        scenario = scenario or {}
        profile = self.config.get("profile", "auto")
        if profile not in {"auto", "lap_completion", "racing", "debug"}:
            raise ValueError("wandb.logging.profile must be auto, lap_completion, racing, or debug")
        self.debug = profile == "debug"
        self.shared_reward = scenario.get("mappo", {}).get("reward_mode") == "team_shared"
        self.finite_training = scenario.get("environment", {}).get("episode_termination", {}).get("lap_completion", True)
        self.selection_strategy = scenario.get("evaluation", {}).get("selection_strategy")
        self.lap_completion = profile == "lap_completion" or (
            profile == "auto" and
            self.selection_strategy in EVALUATION_STRATEGIES)
        patterns = (*CORE, *(LAP_COMPLETION if self.lap_completion else RACING))
        self._names = set(patterns)
        self._patterns = tuple(pattern for pattern in patterns if "*" in pattern)

    def group_enabled(self, group):
        groups = self.config.get("groups", {})
        default = self.debug or group != "reward_components"
        return bool(groups.get(group, default))

    def accepts(self, key):
        namespace = key.split("/", 1)[0]
        group = {"episode": "train", "perf": "performance"}.get(namespace, namespace)
        if not self.group_enabled(group):
            return False
        components = "reward_component" in key
        if components and not self.group_enabled("reward_components"):
            return False
        allowlist = self.config.get("metrics")
        if allowlist is not None:
            rules = allowlist if isinstance(allowlist, dict) else dict.fromkeys(allowlist, True)
            matches = [bool(value) for pattern, value in rules.items() if fnmatchcase(key, pattern)]
            return bool(matches) and all(matches)
        if self.debug or components:
            return True
        if self.lap_completion and (key.startswith(("episode/reward/", "episode/individual_reward/"))
                or key in {"episode/team/first_place", "episode/team/sweep", "episode/team/rank_score"}):
            return False
        if self.shared_reward and key.startswith("episode/reward/"):
            return False
        if not self.finite_training and key in {
                "episode/completed", "episode/team/completion_rate", "episode/team/all_finished",
                "episode/team/first_place", "episode/team/sweep", "episode/team/rank_score"}:
            return False
        if self.selection_strategy == "lap_time" and key in {
                "eval/offtrack_error_m_s_per_lap", "eval/boundary_violation_lap_rate"}:
            return True
        return key in self._names or any(fnmatchcase(key, pattern) for pattern in self._patterns)

    def filter(self, metrics):
        axes = set(AXES.values())
        selected = {key: value for key, value in metrics.items()
                    if key not in axes and self.accepts(key)}
        # Keep the x-axis with an allowed measurement, even under an allowlist.
        # Disabled groups never generate axis-only W&B events.
        for key in tuple(selected):
            axis = AXES.get(key.split("/", 1)[0])
            if axis in metrics:
                selected[axis] = metrics[axis]
        return selected
