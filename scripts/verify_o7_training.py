"""Prepare and audit the bounded O7 training runs; never submits jobs."""
import argparse
import csv
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
EXPECTED_SHA = 'b17b6721a95893a4eaad09807003584c9aae1d0e'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_wandb(config):
    from loggers.wandb_logger import WandbLogger
    flat = WandbLogger._flatten_config(config)
    keys = [
        'environment/vehicle_params/max_integration_step',
        'environment/vehicle_params/model', 'environment/vehicle_params/model_version',
        'environment/vehicle_params/tire_model', 'environment/friction/train/mode',
        'environment/map_bundles_train', 'maps/train',
        'wandb/validation/git_sha', 'wandb/validation/integration_step_s',
        'wandb/validation/physics_contract_sha256', 'wandb/validation/script_sha256',
    ]
    return {key: flat[key] for key in keys}


def prepare(args):
    import yaml
    from core.scenario import load_and_expand_scenario
    from core.provenance import physics_contract
    config = yaml.safe_load(args.resolved.read_text())
    baseline = load_and_expand_scenario(str(args.scenario.resolve()))
    for key in ('algorithm', 'network', 'reward', 'agents', 'controllers', 'observation'):
        require(config[key] == baseline[key], f'Unexpected change to {key}')
    contract = physics_contract(config['environment'])
    require(contract['vehicle_params']['max_integration_step'] == .0025, 'Expected 2.5 ms')
    require(contract['timestep'] == .05, 'Decision timestep changed')
    normalized = deepcopy(contract)
    normalized['vehicle_params']['max_integration_step'] = .001
    require(normalized == physics_contract(baseline['environment']), 'Unexpected physics change')
    require(config['experiment']['total_steps'] == 8192, 'Expected bounded 8192-step budget')
    require(config['environment']['map_bundles_train'] == ['circle_map', 'Budapest_map'], 'Unexpected maps')
    require(config['wandb']['mode'] == 'online', 'Online W&B required for HPC validation')
    require(config['wandb']['validation']['git_sha'] == EXPECTED_SHA, 'Wrong logged Git SHA')
    # Constructing the real task resolves the dense trace allocation as well.
    from core.task_builder import create_race_task
    task = create_race_task(config, scenario_dir=args.scenario.resolve().parent)
    try:
        require(all(len(car._motion_trace) == 20 for car in task.env.sim.agents), 'Expected 20 trace segments')
    finally:
        task.close()
    print(digest(contract))


def verify_run(path, online):
    snapshot = json.loads((path / 'config_snapshot.json').read_text())
    config, provenance = snapshot['config'], snapshot['provenance']
    summary = json.loads((path / 'run_summary.json').read_text())
    from core.provenance import physics_contract
    contract = physics_contract(config['environment'])
    require(contract['vehicle_params']['max_integration_step'] == .0025, 'Wrong integration limit')
    require(provenance['git']['commit'] == EXPECTED_SHA, 'Wrong experiment revision')
    require(provenance['physics_contract'] == contract, 'Snapshot contract mismatch')
    require(digest(contract) == config['wandb']['validation']['physics_contract_sha256'], 'Contract hash mismatch')
    require(set(config['environment']['map_bundles_train']) <= set(provenance['map_protocols']), 'Missing map provenance')
    require((path / 'final.pt').is_file(), 'Missing final checkpoint')
    require(8192 <= summary['environment_steps'] <= 8192 + 4096, 'Unexpected smoke-test budget')
    require(summary['updates'] >= 2 and summary['optimizer_steps'] > 0, 'Insufficient optimizer updates')
    with (path / 'update_metrics.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) >= 2, 'Missing update metrics')
    ranges = {}
    for key in ('train/policy_loss', 'train/value_loss', 'train/entropy', 'train/approx_kl', 'train/action_std'):
        values = [float(row[key]) for row in rows]
        require(all(math.isfinite(value) for value in values), f'Nonfinite {key}')
        ranges[key] = [min(values), max(values)]
    for row in rows:
        for key, value in row.items():
            if value and key.startswith(('train/', 'perf/')):
                require(math.isfinite(float(value)), f'Nonfinite metric {key}')
    def collection_rate(selected):
        seconds = sum(float(row['perf/collection_seconds']) for row in selected)
        steps = sum(float(row['train/update_environment_steps']) for row in selected)
        require(seconds > 0, 'Missing collection timing')
        return steps / seconds
    episodes = [json.loads(line) for line in (path / 'physics_episodes.jsonl').read_text().splitlines()]
    require(bool(episodes), 'No episode physics records')
    for episode in episodes:
        require(math.isfinite(float(episode['physics']['mu'])), 'Nonfinite episode friction')
    keys = expected_wandb(config)
    require(keys['wandb/validation/git_sha'] == EXPECTED_SHA, 'Wrong W&B SHA')
    require(keys['wandb/validation/integration_step_s'] == .0025, 'Wrong W&B integration label')
    online_status = 'not checked'
    if online:
        import wandb
        remote = wandb.Api().run(f"{config['wandb']['entity']}/{config['wandb']['project']}/{provenance['wandb_run_id']}")
        for key, value in keys.items():
            require(remote.config.get(key) == value, f'W&B missing or mismatched {key}')
        online_status = remote.url
    result = dict(path=str(path), summary=summary, finite_update_metric_ranges=ranges,
                  collection_steps_per_second=collection_rate(rows),
                  collection_steps_per_second_after_first_update=collection_rate(rows[1:]),
                  physics_contract_sha256=digest(contract), physics_episode_count=len(episodes),
                  wandb_expected_config=keys, wandb_online_verification=online_status,
                  scope='Short smoke test: finite metrics and completed updates do not establish convergence.')
    (path / 'o7_validation.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--scenario', type=Path, required=True)
    prep.add_argument('--resolved', type=Path, required=True)
    check = sub.add_parser('verify')
    check.add_argument('runs', type=Path, nargs='+')
    check.add_argument('--wandb-api', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == EXPECTED_SHA, 'Wrong checkout')
        prepare(args)
    else:
        for path in args.runs:
            verify_run(path, args.wandb_api)


if __name__ == '__main__':
    main()
