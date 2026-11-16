"""Assign independent map-family PPO LoRA runs to Slurm GPU tasks."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import yaml

FAMILIES = ("straight", "sweeper", "hairpin", "chicane", "s_curve")
REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    rank = int(os.environ.get('SLURM_PROCID', '0'))
    size = int(os.environ.get('SLURM_NTASKS', '1'))
    if size != len(FAMILIES) or not 0 <= rank < size:
        parser.error('Exactly five Slurm tasks are required, one per family')
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 32:
        parser.error(f'Task {rank}: expected at least 32 allocated CPUs; affinity={affinity}')
    # Slurm binds each task to one GPU. Families have separate learners and
    # optimizers; no torchrun process group or gradient synchronization is used.
    environment = dict(os.environ)
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        environment.pop(key, None)
    failures = []
    for family in (FAMILIES[rank],):
        out = root / family
        out.mkdir(exist_ok=True)
        with (out / '.training.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            latest = out / 'latest.pt'
            frozen = out / 'launch_config.yaml'
            if not args.resume and (latest.exists() or frozen.exists()):
                raise RuntimeError(f'{out}: existing experiment; use --resume')
            if not frozen.exists():
                if args.resume and any(path.name != '.training.lock' for path in out.iterdir()):
                    raise RuntimeError(f'{out}: missing original launch_config.yaml; start a new RUN_ROOT')
                # Resolve once before W&B initialization or collection. Recovery,
                # including a failure before latest.pt exists, reuses this snapshot.
                # Retain the preset's explicit total_steps: null. --episodes removes
                # that key, letting execution defaults reappear on snapshot reload.
                resolve = [sys.executable, str(REPO / 'run.py'),
                    '--scenario', 'scenarios/ppo_generated_lora_20k.yaml',
                    '--seed', str(args.seed), '--output-dir', str(out),
                    '--run-id', f'{root.name}_{family}_s{args.seed}',
                    '--set', f'maps=generated_{family}', '--set', 'experiment.distributed=false',
                    '--set', f'wandb.group={root.name}',
                    '--set', f'wandb.name={root.name}_{family}_s{args.seed}',
                    '--set', f"++wandb.id='{uuid.uuid4().hex}'",
                    '--checkpoint', str(root / 'source.pt'), '--no-render', '--resolve-config']
                resolved = subprocess.run(resolve, cwd=REPO, env=environment,
                                          check=True, stdout=subprocess.PIPE, text=True)
                temporary = frozen.with_suffix('.tmp')
                temporary.write_text(resolved.stdout)
                temporary.replace(frozen)
            config = yaml.safe_load(frozen.read_text())
            seed = config['experiment']['seed']
            if config['agents']['car_0']['reward']['reward']['progress_delta_bonus'].get(
                    'collision_penalty_once_per_contact') is not True:
                raise RuntimeError(f'{out}: old collision reward semantics; start a new RUN_ROOT')
            if (config['experiment'].get('total_steps') is not None
                    or config['experiment'].get('episodes') != 20000):
                raise RuntimeError(f'{out}: expected the original 20000-episode budget')
            if (config['experiment']['parallel_envs'], config['experiment']['envs_per_worker']) != (24, 4):
                raise RuntimeError(f'{out}: incompatible collector layout; start a new RUN_ROOT')
            if config['adaptation']['checkpoint'] != str(root / 'source.pt'):
                raise RuntimeError(f'{out}: source snapshot path changed')
            if args.resume and (out / 'completed.json').is_file():
                summary = json.loads((out / 'run_summary.json').read_text())
                if int(summary['completed_episodes']) < 20000 or not (out / 'final.pt').is_file():
                    raise RuntimeError(f'{out}: completion marker has no completed training result')
                print(f'{family}: already completed; skipping', flush=True)
                continue
            command = [sys.executable, str(REPO / 'run.py'), '--scenario', str(frozen),
                '--output-dir', str(out), '--run-id', f'{root.name}_{family}_s{seed}', '--no-render']
            if args.resume and latest.exists():
                command += ['--resume', str(latest)]
            else:
                command += ['--checkpoint', str(root / 'source.pt')]
            record = dict(family=family, seed=seed, task=rank, tasks=size,
                node=os.environ.get('SLURMD_NODENAME'), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                cpu_affinity=affinity, configuration=str(frozen), source=str(root / 'source.pt'),
                command=command)
            attempt = os.environ.get('SLURM_JOB_ID', 'local')
            (out / f'command_{attempt}.json').write_text(json.dumps(record, indent=2) + '\n')
            print(f'Task {rank}/{size}: training {family}; output={out}', flush=True)
            with (out / 'train.log').open('a') as log:
                log.write(f'\nJob {attempt}: {"resume" if args.resume else "start"}; '
                          f'family={family}; seed={seed}; checkpoint={latest if latest.exists() else root / "source.pt"}\n')
                log.flush()
                result = subprocess.run(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                print(f'{family}: failed ({result.returncode}); see {out / "train.log"}', flush=True)
                failures.append(family)
                continue
            summary = json.loads((out / 'run_summary.json').read_text())
            completed = int(summary['completed_episodes'])
            if completed < 20000 or not (out / 'final.pt').is_file():
                raise RuntimeError(f'{family}: run ended before completing 20000 episodes')
            temporary = out / '.completed.json.tmp'
            temporary.write_text(json.dumps(dict(episodes=completed, seed=seed,
                checkpoint=str(out / 'final.pt')), indent=2) + '\n')
            temporary.replace(out / 'completed.json')
            print(f'{family}: completed {completed} episodes', flush=True)
    if failures:
        raise SystemExit('Failed families: ' + ', '.join(failures))


if __name__ == '__main__':
    main()
