"""Validate a runnable experiment without coupling task builders to learners."""
from core.scenario import validate_scenario
from training.configuration import validate_training_scenario
from core.configuration import validate_configuration


def validate_experiment_scenario(scenario):
    validate_configuration(scenario)
    validate_scenario(scenario)
    validate_training_scenario(scenario)
