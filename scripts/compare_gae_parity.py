"""Fixed-trajectory audit of historical GAE and the current PPO handoff.

No environments, policy inference, optimizer steps, or training are run here.
The historical code is exported from its exact commit and imported in isolation.
Full arrays, masks, bootstrap values and accounting results are saved as JSON.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace

import numpy as np

COMMIT = "71a3e0f62d8a77dbcd93ff67c01f78dae8f9b9aa"
ROOT = Path(__file__).resolve().parents[1]


def sequence(rewards, values, next_values, *, term=(), trunc=(), episodes=None, chunks=None):
    n = len(rewards)
    return dict(rewards=rewards, values=values, next_values=next_values,
                terminated=[i in term for i in range(n)], truncated=[i in trunc for i in range(n)],
                episodes=episodes or [0]*n, chunks=chunks or [n])


def fixtures():
    base = ([1., .5, -.25, 2.], [.2, .4, .6, .8], [.4, .6, .8, 1.25])
    reset = ([1., .5, -2., 3., -.1, 2.], [.2, .4, .6, 10., 11., 12.],
             [.4, .6, 7., 11., 12., 13.])
    t = np.arange(800)
    long_values = (.2 + .1*np.sin(t*.05)).tolist()
    long_next = long_values[1:] + [.3]
    return {
        "continuing": [sequence(*base)],
        "collision": [sequence(*base, term=(3,))],
        "clean_completion": [sequence([1., .5, -.25, 4.], base[1], base[2], term=(3,))],
        "timeout_fragment_end": [sequence(*base, trunc=(3,))],
        "termination_then_reset": [sequence(*reset, term=(2,), episodes=[0,0,0,1,1,1], chunks=[2,4])],
        "truncation_then_reset": [sequence(*reset, trunc=(2,), episodes=[0,0,0,1,1,1], chunks=[3,3])],
        "nonterminal_rollout_cut": [sequence(*[row[:2] for row in base])],
        "consecutive_fragments": [sequence((.01+.002*np.cos(t*.03)).tolist(), long_values,
                                          long_next, chunks=[8]*100)],
        "independent_environments": [sequence(*base), sequence(*reset, trunc=(2,),
            episodes=[0,0,0,1,1,1], chunks=[1,2,3])],
    }


def historical(args):
    sys.path.insert(0, str(args.source / "src"))
    import torch
    from agents.common import compute_gae
    results = {}
    for name, streams in fixtures().items():
        advantages, targets = [], []
        for stream in streams:
            r, v, nv = [torch.tensor(stream[k], dtype=torch.float32) for k in
                        ("rewards", "values", "next_values")]
            terminal, truncated = [torch.tensor(stream[k], dtype=torch.float32) for k in
                                   ("terminated", "truncated")]
            finals = torch.where(truncated.bool(), nv, torch.full_like(nv, float("nan")))
            # Call the historical implementation directly, including its final-value path.
            adv, ret = compute_gae(r, v, terminal, truncated, float(nv[-1]), .99, .95, finals)
            advantages.extend(adv.tolist())
            targets.extend(ret.tolist())
        results[name] = dict(advantages=advantages, targets=targets)
    args.output.write_text(json.dumps(results))


def td_stream(stream, environment, device="cpu"):
    import torch
    from tensordict import TensorDict
    n = len(stream["rewards"])
    tensor = lambda values, dtype=torch.float32: torch.tensor(values, dtype=dtype, device=device).reshape(n, 1)
    terminated, truncated = tensor(stream["terminated"], torch.bool), tensor(stream["truncated"], torch.bool)
    done = terminated | truncated
    raw = torch.arange(n*2, dtype=torch.float32, device=device).reshape(n, 2)*.03
    data = TensorDict({
        "uid": torch.arange(n, device=device) + environment*10000,
        "environment_id": torch.full((n,1), environment, dtype=torch.int64, device=device),
        "environment_episode": tensor(stream["episodes"], torch.int64),
        "agents": TensorDict({"observation": tensor(stream["values"]).unsqueeze(-2),
                              "active": torch.ones(n,1,1,dtype=torch.bool,device=device)}, [n,1]),
        "state_value": tensor(stream["values"]),
        "raw_action": raw, "action": raw.tanh(),
        "raw_log_prob": -.5*raw.square().sum(-1)-float(np.log(2*np.pi)),
        "next": TensorDict({
            "uid": torch.arange(n,device=device)+environment*10000+1,
            "state_value": tensor(stream["next_values"]),
            "done": done.clone(), "terminated": terminated.clone(), "truncated": truncated,
            "agents": TensorDict({"observation": tensor(stream["next_values"]).unsqueeze(-2),
                "reward": tensor(stream["rewards"]).unsqueeze(-2), "active": (~done).unsqueeze(-2),
                "done": done.unsqueeze(-2), "terminated": terminated.unsqueeze(-2),
                "truncated": truncated.unsqueeze(-2)}, [n,1]),
            # Deliberately terminal on timeout, as in NativeRaceTorchRLEnv.
            "learning": TensorDict({"done": done, "terminated": done,
                                    "truncated": torch.zeros_like(done)}, [n]),
        }, [n]),
    }, [n], device=device)
    return data


def new_handoff(streams, device):
    import torch
    from tensordict.nn import TensorDictModule
    from torchrl.objectives.value import GAE
    from agents.torchrl_ppo import TorchRLPPOAgent
    from training.on_policy import OnPolicyTrainer
    pending, originals = {}, []
    for environment, stream in enumerate(streams):
        data = td_stream(stream, environment, device)
        originals.append(data)
        start = 0
        pending[environment] = []
        for length in stream["chunks"]:
            pending[environment].append(data[start:start+length].clone())
            start += length
        assert start == len(data)
    rollouts = OnPolicyTrainer._process_rollouts(pending)
    # Identity is a deterministic critic: supplied observations encode the fixed
    # predictions. This exercises TorchRL's real current/next value-network calls.
    value_network = TensorDictModule(torch.nn.Identity(), in_keys=["observation"], out_keys=["state_value"])
    learner = SimpleNamespace(device=torch.device(device), _pending_batches=[], _pending_steps=0,
        _pending_environment_steps=0, gae=GAE(gamma=.99, lmbda=.95, value_network=value_network,
                                             average_gae=False, auto_reset_env=False))
    for rollout in rollouts:
        TorchRLPPOAgent._append_rollout(learner, rollout)
    data = torch.cat(learner._pending_batches)
    original = torch.cat(originals)
    torch.testing.assert_close(data["state_value"],original["state_value"],rtol=0,atol=0)
    torch.testing.assert_close(data["next","state_value"],original["next","state_value"],rtol=0,atol=0)
    for key in ("uid", "raw_action", "raw_log_prob", ("next", "uid"),
                ("next", "agents", "observation"), ("next", "agents", "done")):
        torch.testing.assert_close(data[key], original[key], rtol=0, atol=0)
    assert learner._pending_steps == learner._pending_environment_steps == len(original)
    assert torch.unique(data["uid"]).numel() == len(data)
    return data, len(rollouts)


def error(old, new):
    delta = np.abs(np.asarray(old, dtype=np.float64)-np.asarray(new, dtype=np.float64))
    relative = delta/np.maximum(np.abs(old), 1e-12)
    divergent = np.flatnonzero(~np.isclose(old,new,atol=1e-6,rtol=1e-5))
    return dict(max_absolute=float(delta.max()), max_relative=float(relative.max()),
                first_divergence=None if not len(divergent) else int(divergent[0]))


def accounting():
    import torch
    from tensordict import TensorDict
    from training.on_policy import OnPolicyTrainer
    layout = SimpleNamespace(parallel_envs=64, envs_per_worker=8)
    pending = {}
    # Exact HPC layout, but integer-only fixtures: zero initialized environments.
    for fragment in range(100):
        tick = torch.arange(fragment*8, fragment*8+8).expand(64,8,8)
        env = torch.arange(512).reshape(64,8,1).expand(64,8,8)
        uid = env*800+tick
        active = torch.ones(64,8,8,1,1,dtype=torch.bool)
        data = TensorDict({"uid":uid, "environment_id":env.unsqueeze(-1),
            "environment_episode":(tick//200).unsqueeze(-1),
            "agents":TensorDict({"active":active},[64,8,8,1]),
            "next":TensorDict({"uid":uid+1,"terminal":tick%200==199},[64,8,8])},[64,8,8])
        for worker, local, stream in OnPolicyTrainer._process_streams(layout,data):
            pending.setdefault(worker*8+local,[]).append(stream)
    rollouts = OnPolicyTrainer._process_rollouts(pending)
    ids = torch.cat([row["uid"] for row in rollouts])
    torch.testing.assert_close(ids,torch.arange(409600),rtol=0,atol=0)
    assert all(row["environment_episode"].unique().numel()==1 for row in rollouts)
    terminals = sum(int(row["next","terminal"].sum()) for row in rollouts)
    return dict(environment_slots=512,initialized_environments=0,fragments_per_slot=100,
                decisions_per_slot=800,pooled_samples=len(ids),unique_samples=ids.unique().numel(),
                terminal_transitions=terminals,expected_terminal_transitions=512*(800//200),
                chronological=True,final_observation_pairing=True)


def main(args):
    sys.path.insert(0,str(ROOT/"src"))
    import torch
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    sha = subprocess.check_output(["git","-C",str(args.baseline_repo),"rev-parse",COMMIT],text=True).strip()
    with tempfile.TemporaryDirectory(prefix="gae-parity-") as directory:
        archive = args.output/"baseline.tar"
        with archive.open("wb") as out:
            subprocess.run(["git","-C",str(args.baseline_repo),"archive",sha],stdout=out,check=True)
        with tarfile.open(archive) as tar:
            tar.extractall(directory,filter="data")
        archive.unlink()
        subprocess.run([sys.executable,str(Path(__file__).resolve()),"--source",directory,
            "--output",str(args.output/"historical.json")],check=True,
            env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1"})
    old = json.loads((args.output/"historical.json").read_text())
    report = dict(baseline_commit=sha,current_commit=subprocess.check_output(
        ["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
        gamma=.99,gae_lambda=.95,atol=1e-6,rtol=1e-5,cases={})
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    report["devices"] = devices
    for name, streams in fixtures().items():
        row = dict(transitions=sum(len(s["rewards"]) for s in streams),inputs=streams,
                   historical=old[name],current={})
        for device in devices:
            data,count = new_handoff(streams,device)
            adv,ret = [data[k].flatten().cpu().tolist() for k in ("advantage","value_target")]
            old_adv = torch.tensor(old[name]["advantages"],dtype=torch.float32)
            new_adv = data["advantage"].flatten().cpu()
            normalized = lambda x: (x-x.mean())/(x.std(correction=0)+1e-8)
            row["current"][device] = dict(advantages=adv,targets=ret,reconstructed_sequences=count,
                advantage_error=error(old[name]["advantages"],adv),target_error=error(old[name]["targets"],ret),
                normalized_advantage_error=error(normalized(old_adv).tolist(),normalized(new_adv).tolist()),
                bootstrap_values=data["next","state_value"].flatten().cpu().tolist(),
                bootstrap_mask=(~data["next","terminated"]).flatten().cpu().tolist(),
                continuation_mask=(~data["next","done"]).flatten().cpu().tolist())
        report["cases"][name] = row
    report["accounting"] = accounting()
    (args.output/"report.json").write_text(json.dumps(report,indent=2))
    for name,row in report["cases"].items():
        print(name,row["transitions"],{dev:{k:v for k,v in result.items() if k.endswith("error")}
              for dev,result in row["current"].items()})
    print("accounting",report["accounting"])


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-repo",type=Path,default=ROOT.parent/"F110_MARL")
    parser.add_argument("--output",type=Path,default=Path("/tmp/f110-gae-parity"))
    parser.add_argument("--source",type=Path)
    args=parser.parse_args()
    historical(args) if args.source else main(args)
