"""Verify native TorchRL startup in a fresh interpreter."""
from pathlib import Path
import subprocess
import sys

def test_native_ppo_startup_uses_torchrl_environment_and_loss():
    # A fresh interpreter prevents earlier tests from hiding transitive imports.
    script = '''
import sys

from pathlib import Path
sys.path.insert(0, str(Path("src").resolve()))
from core.scenario import load_and_expand_scenario
from core.task_builder import create_race_task
from training.algorithms import create_learner, learner_params
from training.on_policy import OnPolicyTrainer as TorchRLPPOTrainer
from adapters import NativeRaceTorchRLEnv
from torchrl.envs import EnvBase
from torchrl.objectives import ClipPPOLoss

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
    assert isinstance(trainer.env, NativeRaceTorchRLEnv)
    assert isinstance(trainer.env, EnvBase)
    assert isinstance(learner.loss_module, ClipPPOLoss)
    trainer.train(total_steps=2)
    assert trainer.collected_steps == 2
finally:
    task.close()
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
