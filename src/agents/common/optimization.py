"""Optimization settings shared by on-policy learners and checkpoint policies."""
import numpy as np


class OnPolicyOptimizationSettings:
    def configure_optimization(self, params):
        self.lr = float(params.get("learning_rate", 3e-4))
        self.lr_schedule = str(params.get("lr_schedule", "constant"))
        if self.lr_schedule not in {"constant", "linear"}:
            raise ValueError("lr_schedule must be 'constant' or 'linear'")
        if self.lr_schedule == "linear" and "learning_rate_end" not in params:
            raise ValueError("Linear lr_schedule requires learning_rate_end")
        self.lr_end = float(params.get("learning_rate_end", self.lr))
        if not all(np.isfinite(rate) and rate > 0 for rate in (self.lr, self.lr_end)):
            raise ValueError("Learning rates must be finite and positive")
        if self.lr_schedule == "constant" and self.lr_end != self.lr:
            raise ValueError("A different learning_rate_end requires lr_schedule: linear")
        self.target_kl = params.get("target_kl")
        if self.target_kl is not None:
            self.target_kl = float(self.target_kl)
            if not np.isfinite(self.target_kl) or self.target_kl <= 0:
                raise ValueError("target_kl must be finite and positive")

    def set_training_progress(self, progress):
        if self.lr_schedule == "linear":
            fraction = float(np.clip(progress, 0.0, 1.0))
            rate = self.lr + fraction * (self.lr_end - self.lr)
            for group in self.optimizer.param_groups:
                group["lr"] = rate
