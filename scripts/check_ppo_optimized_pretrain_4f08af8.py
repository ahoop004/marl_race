"""Fast configuration/recovery preflight; no learner construction or compilation."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def check(config, gpu=False, resume=None):
    from core.scenario import load_and_expand_scenario
    from core.provenance import physics_contract
    baseline = load_and_expand_scenario(str(ROOT / 'scenarios/ppo_lap_completion_pretrain.yaml'))
    for key in ('algorithm', 'network', 'agents', 'controllers', 'reward', 'observation', 'maps', 'evaluation', 'checkpoint'):
        if config[key] != baseline[key]:
            raise ValueError(f'Unexpected modification to {key}')
    environment = deepcopy(config['environment'])
    assert environment['vehicle_params']['max_integration_step'] == .0025
    environment['vehicle_params']['max_integration_step'] = .001
    assert environment == baseline['environment'], 'Other environment settings changed'
    expected = dict(total_steps=120000000, seed=42, torch_threads=1, parallel_envs=16,
                    envs_per_worker=2, distributed=False, measure_stages=False,
                    measure_mpc=False, measure_hybrid=False)
    for key, value in expected.items():
        assert config['experiment'][key] == value, (key, config['experiment'][key])
    params = config['training_defaults']
    for key, value in dict(n_steps=409600, frames_per_batch=4096, n_epochs=10,
                           batch_size=1024, learning_rate=.001, lr_schedule='linear',
                           learning_rate_end=.0001, device='cuda', collection_device='cpu').items():
        assert params[key] == value, (key, params[key])
    assert config['environment']['max_steps'] == 80000
    assert config['environment']['timestep'] == .05
    assert config['environment']['action_repeat'] == 1
    assert config['wandb']['enabled'] and config['wandb']['mode'] == 'online'
    assert config['wandb']['validation']['integration_step_s'] == .0025
    # Same sizing expression as integrate_combined_slip and RaceCar.__init__.
    # No integration call or JIT compilation is needed for this check.
    import math
    assert math.ceil(config['environment']['timestep'] /
                     config['environment']['vehicle_params']['max_integration_step']) == 20
    contract = physics_contract(config['environment'])
    if resume:
        from agents.common.checkpoints import read_checkpoint
        checkpoint = read_checkpoint(resume)
        metadata = checkpoint['metadata']
        assert metadata['algorithm'] == 'ppo'
        assert metadata['contracts']['physics'] == contract
        for key in ('network', 'reward', 'observation', 'agents', 'maps', 'training_defaults'):
            assert metadata['configuration'][key] == config[key], f'Recovery mismatch: {key}'
        progress = metadata['progress']['environment_steps']
        stop = config['experiment'].get('stop_after_steps') or 120000000
        assert progress < stop <= 120000000, (progress, stop)
    if gpu:
        import torch
        assert torch.cuda.is_available(), 'CUDA unavailable inside allocated container'
        assert torch.cuda.device_count() == 1, 'Expected exactly one visible GPU'
        assert 'L40S' in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


if __name__ == '__main__':
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('resolved', type=Path)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--resume')
    args = parser.parse_args()
    print(check(yaml.safe_load(args.resolved.read_text()), args.gpu, args.resume))
