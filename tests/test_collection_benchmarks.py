"""Matched configuration and allocation safety for the small HPC runner."""
import argparse
import json
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
def test_benchmark_frames_preserve_fixed_learning_and_episode_settings(benchmark, algorithm, frames, tmp_path):
    args = argparse.Namespace(algorithm=algorithm, total_steps=262144, torch_threads=1,
                              workers=8, envs_per_worker=2, learner_device="cuda")
    command = benchmark["training_command"](args, frames, tmp_path / "run")
    scenario = command[command.index("--scenario") + 1]
    overrides = [command[index + 1] for index, value in enumerate(command) if value == "--set"]
    baseline = load_and_expand_scenario(scenario)
    configured = load_and_expand_scenario(scenario, overrides=overrides)
    original = resolve_training_params(baseline["agents"]["car_0"], baseline)
    # Benchmarks explicitly retain smaller update sizes than production runs.
    original.update(n_steps=4096 if algorithm == "ppo" else 2048,
                    n_epochs=4 if algorithm == "ppo" else 5, batch_size=512)
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


@pytest.mark.parametrize("algorithm,expected", [
    ("ppo", [(4, 2, 1024), (8, 1, 1024), (8, 2, 1024), (8, 4, 1024),
             (16, 2, 1024), (8, 2, 512), (8, 2, 2048), (8, 4, 2048)]),
    ("mappo", [(4, 2, 1024), (8, 2, 1024), (8, 4, 1024)]),
])
def test_final_matrix_uses_requested_layouts_and_baseline_checkout(benchmark, algorithm, expected, tmp_path):
    args = argparse.Namespace(matrix="final", algorithm=algorithm, layout=None,
                              workers=1, envs_per_worker=1, total_steps=262144,
                              torch_threads=1, learner_device="cuda", source_root=tmp_path / "baseline")
    candidates = benchmark["matrix_candidates"](args, {"inherited": None})
    assert [row[:3] for row in candidates] == expected
    for workers, envs, frames, _, mask, backend, device in candidates:
        assert (mask, backend, device) == (None, "process", "cpu")
        command = benchmark["training_command"](args, frames, tmp_path / "run",
            workers=workers, envs_per_worker=envs)
        assert command[1] == str(args.source_root / "run.py")
        assert command[command.index("--total-steps") + 1] == "262144"
        overrides = [command[index + 1] for index, value in enumerate(command) if value == "--set"]
        config = load_and_expand_scenario(command[command.index("--scenario") + 1], overrides=overrides)
        assert (config["experiment"]["parallel_envs"], config["experiment"]["envs_per_worker"]) == (workers, envs)
        assert config["algorithm"]["params"]["frames_per_batch"] == frames
        assert config["experiment"]["measure_stages"] is False
        assert config["experiment"]["measure_cuda_sync"] is False
    args.layout = [benchmark["parse_layout"]("8x2x2048")]
    assert [row[:3] for row in benchmark["matrix_candidates"](args, {})] == [(8, 2, 2048)]
    for value in ("8x2", "8x0x1024", "-1x2x512", "8x2xno"):
        with pytest.raises(argparse.ArgumentTypeError):
            benchmark["parse_layout"](value)


@pytest.mark.parametrize("legacy", [False, True])
def test_hpc_summary_handles_baseline_instrumentation_and_rejects_invalid_runs(benchmark, legacy, tmp_path):
    summary = {"environment_steps": 262144, "agent_steps": 262144, "optimizer_steps": 2048,
               "physics_steps": 262144, "completed_episodes": 30, "perf/worker_count": 2,
               "perf/environment_slots": 4, "perf/collection_device": "cpu", "perf/learner_device": "cuda",
               "perf/environment_steps_per_second": 500., "perf/learner_samples_per_second": 500.,
               "perf/physics_steps_per_second": 500.}
    snapshot = {"config": {"algorithm": {"params": {"frames_per_batch": 7}},
                            "experiment": {"total_steps": 262144}},
                "provenance": {"execution": {}, "git": {"commit": "baseline"}}}
    if not legacy:
        summary.update({"perf/collection_backend": "process", "perf/collection_batches": 32768,
                        "perf/last_collection_batch_frames": 8, "perf/process/cpu_affinity": [1, 3],
                        "perf/workers/0/cpu_affinity": [1, 3], "perf/workers/1/cpu_affinity": [1, 3],
                        "perf/workers/0/pid": 101, "perf/workers/1/pid": 102})
    (tmp_path / "config_snapshot.json").write_text(json.dumps(snapshot))
    def report(**kwargs):
        (tmp_path / "run_summary.json").write_text(json.dumps(summary))
        return benchmark["report_run"](tmp_path, 262144, None, [1, 3], learner_device="cuda", **kwargs)
    result = report()
    assert result["actual_batch_frames"] == 8
    assert result["batch_frames_measured"] is not legacy
    assert result["worker_affinity_verified"] == (None if legacy else True)
    assert result["actual_worker_processes"] == (None if legacy else 2)
    summary["environment_steps"] = 262152
    with pytest.raises(RuntimeError, match="invalid collected budget"):
        report()
    summary["environment_steps"] = 262144
    summary["perf/learner_device"] = "cpu"
    with pytest.raises(RuntimeError, match="learner_device"):
        report()
    summary["perf/learner_device"] = "cuda"
    with pytest.raises(RuntimeError, match="actor compilation"):
        report(compilation="both")
    summary.update({"perf/compilation/actor": True, "perf/compilation/critic": True,
                    "perf/compilation/backend": "aot_eager", "perf/compilation/mode": "default"})
    with pytest.raises(RuntimeError, match="compilation backend/mode"):
        report(compilation="both", compile_mode="default")
    summary["perf/compilation/backend"] = "inductor"
    report(compilation="both", compile_mode="default")
    if not legacy:
        summary["perf/learner_device"] = "cuda"
        summary["perf/workers/1/cpu_affinity"] = [1, 4]
        with pytest.raises(RuntimeError, match="workers did not inherit"):
            report()
        summary["perf/workers/1/cpu_affinity"] = [1, 3]
        summary["perf/workers/1/pid"] = 101
        with pytest.raises(RuntimeError, match="worker process count"):
            report()


@pytest.mark.parametrize("algorithm", ["ppo", "mappo"])
def test_compilation_matrix_changes_only_compilation_with_equal_warmup(benchmark, algorithm, tmp_path):
    args = argparse.Namespace(matrix="compilation", algorithm=algorithm, workers=8, envs_per_worker=2,
                              frames_per_batch=1024, total_steps=262144, torch_threads=1,
                              learner_device="cuda", collection_backend="process", collection_device="cpu",
                              compile_mode="default", compile_warmup_updates=3, profile=False)
    from application.configuration import validate_experiment_scenario
    from agents.common.compilation import resolve_compile_config
    configs = []
    for workers, envs, frames, name, mask, backend, device in benchmark["matrix_candidates"](args, {}):
        command = benchmark["training_command"](args, frames, tmp_path / name, workers=workers,
            envs_per_worker=envs, compilation=name, backend=backend, collection_device=device)
        overrides = [command[index + 1] for index, value in enumerate(command) if value == "--set"]
        config = load_and_expand_scenario(command[command.index("--scenario") + 1], overrides=overrides)
        validate_experiment_scenario(config)
        assert config["experiment"]["profiling_warmup_updates"] == 3
        assert config["experiment"]["measure_stages"] is False
        flag = config["execution"]["params"].pop("compile_networks")
        settings = resolve_compile_config(flag)
        assert settings["actor"] == (name in {"actor", "both"})
        assert settings["critic"] == (name in {"critic", "both"})
        config["training_defaults"].pop("compile_networks")
        for agent in config["agents"].values():
            agent.get("params", {}).pop("compile_networks", None)
        config.pop("_configuration", None)
        configs.append(config)
    assert len(configs) == 4 and all(config == configs[0] for config in configs)
