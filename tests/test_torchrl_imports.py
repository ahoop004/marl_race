"""PPO startup must not load optional legacy training stacks."""
from pathlib import Path
import subprocess
import sys

import pytest


pytest.importorskip("torchrl")


def test_native_ppo_does_not_import_external_training_adapters():
    # A fresh interpreter prevents earlier tests from hiding transitive imports.
    script = '''
import sys

blocked = {"gym", "stable_baselines3", "tensorflow"}
def guard(event, args):
    if event == "import" and args[0].split(".")[0] in blocked:
        raise AssertionError(f"Unexpected optional import: {args[0]}")
sys.addaudithook(guard)

from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))
from core.scenario import load_and_expand_scenario
from core.task_builder import create_race_task
from training.algorithms import create_learner, learner_params
from training.on_policy import OnPolicyTrainer as TorchRLPPOTrainer

directory = Path("scenarios")
scenario = load_and_expand_scenario(str(directory / "ppo_lap_completion_pretrain.yaml"))
scenario["environment"]["max_steps"] = 2
task = create_race_task(scenario, scenario_dir=directory)
try:
    spec = task.spec
    params = learner_params(scenario, spec, "ppo")
    params.update(device="cpu", hidden_dims=[8], n_steps=2, n_epochs=1, batch_size=2)
    learner = create_learner("ppo", spec, params)
    trainer = TorchRLPPOTrainer(task, learner)
    trainer.train(total_steps=2)
    assert trainer.collected_steps == 2
    assert not blocked.intersection(sys.modules)
    assert "adapters.gymnasium" not in sys.modules
    assert "adapters.pettingzoo" not in sys.modules
finally:
    task.close()
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
