"""Preloaded, independently routed frozen actors; no inference-time disk access."""
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch
from torch import nn

from agents.common.checkpoints import read_checkpoint, solo_evaluation_actor
from utils.torch_io import validate_checkpoint_compatibility

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AdapterBank(nn.Module):
    def __init__(self, manifest, *, spec=None, device="cpu", root=PROJECT_ROOT):
        super().__init__()
        started = time.perf_counter()
        entries = manifest["skills"]
        if (any(type(e["id"]) is not int for e in entries)
                or [e["id"] for e in entries] != list(range(len(entries)))
                or len({e["name"] for e in entries}) != len(entries)
                or not entries or entries[0]["name"] != "base"):
            raise ValueError("Skills require ordered contiguous IDs starting with base=0 and unique names")
        self.actors = nn.ModuleList()
        self.manifest = []
        base = None
        for entry in entries:
            path = Path(entry["checkpoint"])
            checkpoint = read_checkpoint(path if path.is_absolute() else Path(root) / path)
            if entry.get("sha256", checkpoint["_identity"]["sha256"]) != checkpoint["_identity"]["sha256"]:
                raise ValueError(f"{entry['name']}: checkpoint content hash changed")
            metadata = checkpoint["metadata"]
            if metadata["algorithm"] != "ppo":
                raise ValueError(f"{entry['name']}: expected a single-agent PPO checkpoint")
            if base is None:
                base = checkpoint
                if metadata["adaptation"]["lora"] is not None:
                    raise ValueError("Skill 0 must be the unadapted pretrained base")
                self.metadata = metadata
                self.agent_id, obs = next(iter(metadata["contracts"]["observations"].items()))
                contracts = metadata["contracts"]
                self.obs_dim = obs["dim"]
                self.action_dim = contracts["actions"]["dim"]
                stored_spec = SimpleNamespace(agent_ids=(self.agent_id,),
                    observation_dims={self.agent_id: self.obs_dim},
                    observation_contracts={self.agent_id: obs["contract"]},
                    action_lows={self.agent_id: contracts["actions"]["low"]},
                    action_highs={self.agent_id: contracts["actions"]["high"]})
            else:
                validate_checkpoint_compatibility(metadata, {
                    key: self.metadata[key] for key in ("network", "contracts", "structure", "implementation")})
                adaptation = metadata["adaptation"]
                source = adaptation["source_checkpoint"]
                if not adaptation["lora"] or not source or source["sha256"] != base["_identity"]["sha256"]:
                    raise ValueError(f"{entry['name']}: adapter source hash differs from configured base")
                # Exploration log_std can be trained; every deterministic base tensor must match.
                for name, value in base["models"]["actor"].items():
                    if name != "log_std" and (name not in checkpoint["models"]["actor"] or not torch.equal(value, checkpoint["models"]["actor"][name])):
                        raise ValueError(f"{entry['name']}: frozen base tensor differs: {name}")
                family = entry.get("family", entry["name"])
                split = metadata["provenance"].get("map_split", {})
                if not split.get("train") or not all(f"train_specialist_{family}_" in p for p in split["train"]):
                    raise ValueError(f"{entry['name']}: specialist training provenance does not match {family}")
                if not metadata["selection"] or not split.get("eval") or not all(
                        f"validation_specialist_{family}_" in p for p in split["eval"]):
                    raise ValueError(f"{entry['name']}: validation-selected specialist checkpoint required")
            params = {"network": metadata["network"], "device": device,
                      "_physics_contract": self.metadata["contracts"]["physics"],
                      "_action_contract": self.metadata["contracts"]["actions"]["contract"]}
            actor = solo_evaluation_actor(checkpoint, spec or stored_spec, params, self.agent_id).actor
            self.actors.append(actor.requires_grad_(False).eval())
            self.manifest.append({**entry, "sha256": checkpoint["_identity"]["sha256"]})
        self.device = torch.device(device)
        self.initialization_seconds = time.perf_counter() - started

    @torch.no_grad()
    def forward(self, observations, skills):
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        ids = torch.as_tensor(skills, device=self.device)
        if obs.ndim != 2 or obs.shape[1] != self.obs_dim or ids.shape != (len(obs),) or ids.dtype != torch.long:
            raise ValueError("Expected LL observations [batch, obs_dim] and int64 skills [batch]")
        if ((ids < 0) | (ids >= len(self.actors))).any():
            raise ValueError("Unknown skill ID")
        actions = obs.new_empty((len(obs), self.action_dim))
        for skill, actor in enumerate(self.actors):
            rows = torch.where(ids == skill)[0]
            if rows.numel():
                actions[rows] = actor.get_action(obs[rows], deterministic=True)[0]
        return actions

    def accounting(self):
        return dict(parameters=sum(p.numel() for p in self.parameters()),
                    trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    tensor_bytes=sum(t.numel() * t.element_size() for t in (*self.parameters(), *self.buffers())),
                    initialization_seconds=self.initialization_seconds)


# Weak ownership avoids retaining banks after all environments have closed.
# Each process has its own cache; mutable skill selection never lives here.
import json
import weakref
_BANKS = weakref.WeakValueDictionary()


def shared_adapter_bank(manifest, *, spec, device='cpu'):
    key = json.dumps({'manifest': manifest, 'device': str(device)}, sort_keys=True)
    bank = _BANKS.get(key)
    if bank is None:
        bank = AdapterBank(manifest, spec=spec, device=device)
        _BANKS[key] = bank
    else:
        aid = bank.agent_id
        validate_checkpoint_compatibility(bank.metadata['contracts']['observations'][aid],
            {'dim': spec.observation_dims[aid], 'contract': spec.observation_contracts[aid]})
        validate_checkpoint_compatibility(bank.metadata['contracts']['actions'],
            {'low': spec.action_lows[aid], 'high': spec.action_highs[aid]})
    return bank
