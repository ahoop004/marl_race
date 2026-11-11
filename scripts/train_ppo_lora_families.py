"""Assign independent map-family PPO LoRA runs to Slurm GPU tasks."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

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
    if not 0 <= rank < size:
        parser.error('Invalid Slurm task rank/count')
    # Slurm binds each task to one GPU. Families have separate learners and
    # optimizers; no torchrun process group or gradient synchronization is used.
    environment = dict(os.environ)
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        environment.pop(key, None)
    failures = []
    for family in FAMILIES[rank::size]:
        out = root / family
        out.mkdir(exist_ok=True)
        with (out / '.training.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.resume and (out / 'completed.json').is_file():
                print(f'{family}: already completed; skipping', flush=True)
                continue
            latest = out / 'latest.pt'
            if latest.exists() and not args.resume:
                raise RuntimeError(f'{out}: existing checkpoint; use --resume')
            command = [sys.executable, str(REPO / 'run.py'),
                '--scenario', 'scenarios/ppo_generated_lora_20k.yaml',
                '--episodes', '20000', '--seed', str(args.seed), '--output-dir', str(out),
                '--run-id', f'{root.name}_{family}_s{args.seed}',
                '--set', f'maps=generated_{family}', '--set', 'experiment.distributed=false',
                '--set', f'wandb.group={root.name}',
                '--set', f'wandb.name={root.name}_{family}_s{args.seed}', '--no-render']
            if args.resume and latest.exists():
                command += ['--resume', str(latest)]
            else:
                command += ['--checkpoint', str(root / 'source.pt')]
            record = dict(family=family, seed=args.seed, task=rank, tasks=size,
                node=os.environ.get('SLURMD_NODENAME'), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                command=command)
            attempt = os.environ.get('SLURM_JOB_ID', 'local')
            (out / f'command_{attempt}.json').write_text(json.dumps(record, indent=2) + '\n')
            print(f'Task {rank}/{size}: training {family}; output={out}', flush=True)
            with (out / 'train.log').open('a') as log:
                result = subprocess.run(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                print(f'{family}: failed ({result.returncode}); see {out / "train.log"}', flush=True)
                failures.append(family)
                continue
            summary = json.loads((out / 'run_summary.json').read_text())
            completed = int(summary['completed_episodes'])
            if completed < 20000 or not (out / 'final.pt').is_file():
                raise RuntimeError(f'{family}: run ended before completing 20000 episodes')
            (out / 'completed.json').write_text(json.dumps(dict(episodes=completed, seed=args.seed,
                checkpoint=str(out / 'final.pt')), indent=2) + '\n')
            print(f'{family}: completed {completed} episodes', flush=True)
    if failures:
        raise SystemExit('Failed families: ' + ', '.join(failures))


if __name__ == '__main__':
    main()
