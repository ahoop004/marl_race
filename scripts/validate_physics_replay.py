"""Verify P2 task traces against an isolated committed source archive.

Imports physics/task code only from --source-root; uses the recorded configuration,
map assets and learner action stream. Does not measure throughput or edit Git.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    directory = args.results.resolve()
    sys.path.insert(0, str(source / "src"))
    os.environ.setdefault("PYGLET_HEADLESS", "true")
    from core.task_builder import create_race_task
    import physics.simulaton
    assert Path(physics.simulaton.__file__).resolve().is_relative_to(source)
    spec = importlib.util.spec_from_file_location("reference_p1", source / "scripts/benchmark_physics.py")
    p1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p1)
    report = json.loads((directory / "results.json").read_text())
    validated = {}
    for name, workload in report["workloads"].items():
        trace = np.load(directory / f"{name}_trace.npz")
        config = workload["config"]["scenario"]
        task = create_race_task(config, scenario_dir=Path(config["_configuration"]["source"]).parent)
        try:
            task.reset(seed=42, options=dict(poses=np.array(workload["config"]["poses"]),
                                            map_episode_index=2, spawn_episode_index=3))
            for index, actions in enumerate(trace["normalized_actions"]):
                output = task.step(dict(zip(task.possible_agents, actions)))
                actual = np.stack([car.physics_state for car in task.env.sim.agents])
                np.testing.assert_array_equal(actual, trace["states"][index])
                if p1.digest(output).hexdigest() != trace["task_digests"][index]:
                    raise AssertionError(f"{name}: task diverged at {index}")
            validated[name] = dict(steps=index + 1, exact_states=True, exact_task_outputs=True)
            print(f"{name}: {index+1} archived-source task outputs bit-identical", flush=True)
        finally:
            task.close()
    for name, guard in report["collision_guards"].items():
        sim, task, reset, advance, _ = p1.build_case(name, directory / "arena.yaml", source, 12)
        row = p1.run_rollout(sim, task, reset, advance, 12, capture=True)
        if row["digest"] != guard["detail"]["digest"]:
            raise AssertionError(f"{name}: archived contact guard diverged")
        validated[name] = dict(steps=12, exact_complete_digest=True)
        print(f"{name}: archived-source collision guard bit-identical", flush=True)
    (directory / "archived_source_parity.json").write_text(json.dumps(dict(
        source_revision=args.source_revision, source_root=str(source),
        source_sha256={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in sorted((source/'src').rglob('*.py'))},
        validated=validated), indent=2) + "\n")


if __name__ == "__main__":
    main()
