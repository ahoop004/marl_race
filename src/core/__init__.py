"""Core infrastructure for the F110 training pipeline.

Load public exports on demand so configuration helpers do not initialize the
simulator or fixed-policy registry merely by importing the core package.
"""

__all__ = ["create_fixed_controller", "register_fixed_controller", "create_training_setup"]


def __getattr__(name: str):
    if name in {"create_fixed_controller", "register_fixed_controller"}:
        from core.agent_builder import create_fixed_controller, register_fixed_controller
        return {"create_fixed_controller": create_fixed_controller,
                "register_fixed_controller": register_fixed_controller}[name]
    if name == "create_training_setup":
        from core.setup import create_training_setup
        return create_training_setup
    raise AttributeError(f"module 'core' has no attribute {name!r}")
