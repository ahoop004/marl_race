# Lazy import — avoids circular dependency when importing env.spaces or other
# submodules before ParallelEnv is fully initialized.

def __getattr__(name: str):
    if name == "Env":
        from env.RaceEnv import RaceEnv
        return RaceEnv
    raise AttributeError(f"module 'env' has no attribute {name!r}")
