"""Matched configuration and allocation safety for the small HPC runner."""
import argparse
import runpy

import pytest

from core.configuration import PROJECT_ROOT
from core.scenario import load_and_expand_scenario
from training.algorithms import resolve_training_params


@pytest.fixture(scope="module")
def benchmark():
    return runpy.run_path(str(PROJECT_ROOT / "scripts" / "benchmark_collection.py"))


@pytest.mark.parametrize("algorithm", ["ppo", "mappo"])
@pytest.mark.parametrize("frames", [512, 1024, 2048])
def test_benchmark_frames_preserve_learning_and_episode_settings(benchmark, algorithm, frames, tmp_path):
    args = argparse.Namespace(algorithm=algorithm, total_steps=262144, torch_threads=1,
                              workers=8, envs_per_worker=2, learner_device="cuda")
    command = benchmark["training_command"](args, frames, tmp_path / "run")
    scenario = command[command.index("--scenario") + 1]
    overrides = [command[index + 1] for index, value in enumerate(command) if value == "--set"]
    baseline = load_and_expand_scenario(scenario)
    configured = load_and_expand_scenario(scenario, overrides=overrides)
    original = resolve_training_params(baseline["agents"]["car_0"], baseline)
    actual = resolve_training_params(configured["agents"]["car_0"], configured)
    assert actual.pop("frames_per_batch") == frames
    assert actual.pop("device") == "cuda"
    original.pop("device")
    actual.pop("collection_device")
    original.pop("collection_device", None)
    original.pop("frames_per_batch", None)
    assert actual == original
    assert configured["environment"] == baseline["environment"]
    assert configured["agents"] == baseline["agents"]
    assert configured["experiment"]["parallel_envs"] == 8
    assert configured["experiment"]["envs_per_worker"] == 2
    assert configured["evaluation"]["enabled"] is False
    assert configured["experiment"]["measure_stages"] is False


def test_placement_uses_only_allocated_cores_and_equal_sized_masks(benchmark):
    # Sparse allocation with SMT siblings, an uneven socket split and unknown IDs.
    topology = [dict(cpu=cpu, socket=socket, core=core) for cpu, socket, core in (
        (7, 0, 1), (11, 0, 1), (29, 0, 2), (37, 1, 1), (41, 1, 2), (45, 1, 3))]
    placement = dict(cpu_affinity=[row["cpu"] for row in topology], cpu_topology=topology)
    masks = benchmark["placement_masks"](placement, count=2, minimum=2)
    assert masks == {"inherited": None, "socket": [37, 41], "spread": [7, 37]}
    for mask in (masks["socket"], masks["spread"]):
        assert set(mask).issubset(placement["cpu_affinity"])
        cores = {(row["socket"], row["core"]) for row in topology if row["cpu"] in mask}
        assert len(cores) == len(mask) == 2
    for count in (1, 4):
        with pytest.raises(ValueError, match="allocated physical cores"):
            benchmark["placement_masks"](placement, count=count, minimum=2)
    topology[0]["core"] = None
    with pytest.raises(ValueError, match="topology"):
        benchmark["placement_masks"](placement, count=2, minimum=2)


@pytest.mark.parametrize("algorithm", ["ppo", "mappo"])
def test_inference_comparisons_preserve_workload(benchmark, algorithm, tmp_path):
    args = argparse.Namespace(algorithm=algorithm, total_steps=262144, torch_threads=1,
                              workers=8, envs_per_worker=2, learner_device="cuda", profile=False)
    configs = []
    for backend, device in (("process", "cpu"), ("centralized", "cpu"), ("centralized", "cuda")):
        command = benchmark["training_command"](args, 1024, tmp_path / device,
                                                 backend=backend, collection_device=device)
        scenario = command[command.index("--scenario") + 1]
        overrides = [command[index + 1] for index, value in enumerate(command) if value == "--set"]
        config = load_and_expand_scenario(scenario, overrides=overrides)
        from application.configuration import validate_experiment_scenario
        validate_experiment_scenario(config)
        assert config["execution"]["params"].pop("collection_device") == device
        assert config["experiment"].pop("collection_backend", "process") == backend
        config["training_defaults"].pop("collection_device")
        for agent in config["agents"].values():
            agent.get("params", {}).pop("collection_device", None)
        config.pop("_configuration", None)
        configs.append(config)
    assert configs[0] == configs[1] == configs[2]
