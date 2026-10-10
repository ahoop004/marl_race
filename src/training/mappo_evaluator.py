"""MAPPO checkpoint selection observes the complete physical race."""
from training.task_evaluator import TaskEvaluator


class DeterministicMAPPOEvaluator(TaskEvaluator):
    completion = "race"
