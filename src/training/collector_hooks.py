"""Forward worker events to parent-owned training hooks."""
from training.hooks import TrainingHook


class WorkerHook(TrainingHook):
    def __init__(self, connection, worker_id, seed, record_transitions):
        self.connection = connection
        self.worker_id = worker_id
        self.seed = seed
        self._record_transitions = record_transitions
        self.requires_transition_record = record_transitions

    def on_step(self, record):
        if not self._record_transitions:
            return
        from dataclasses import replace
        record = replace(record, info={**record.info, "worker_id": self.worker_id,
                                       "worker_seed": self.seed})
        self.connection.send(("transition", record))

    def on_episode_start(self, metadata):
        self.connection.send(('episode_start', metadata))

    def on_episode_end(self, episode, reward, info, metrics):
        info = {**info, "worker_id": self.worker_id, "worker_seed": self.seed,
                "worker_episode": episode}
        metrics = {**metrics, "worker_id": self.worker_id, "worker_seed": self.seed,
                   "worker_episode": episode}
        self.connection.send(("episode", (reward, info, metrics)))

