"""Validate a runnable experiment without coupling task builders to learners."""
from core.scenario import validate_scenario
from training.configuration import validate_training_scenario


def validate_experiment_scenario(scenario):
    validate_scenario(scenario)
    validate_training_scenario(scenario)
