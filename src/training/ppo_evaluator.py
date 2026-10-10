"""PPO checkpoint selection ends when the exposed racer retires."""
from training.task_evaluator import TaskEvaluator


class DeterministicPPOEvaluator(TaskEvaluator):
    completion = "policy"
    focal_summary = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if len(self.trainable_ids) != 1:
            raise ValueError("PPO evaluation requires one policy agent")
