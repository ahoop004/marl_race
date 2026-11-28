"""Preloaded frozen driving skills for hard selection and residual composition."""
from itertools import accumulate
import json
from pathlib import Path
from types import SimpleNamespace
import time
import weakref

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from agents.common.checkpoints import read_checkpoint, solo_evaluation_actor, validate_solo_actor_checkpoint
from agents.common.networks import ResidualMLP
from utils.torch_io import validate_checkpoint_compatibility

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _FrozenBank(nn.Module):
    def _checkpoints(self, manifest, spec, device, root):
        entries = manifest["skills"]
        if (any(type(e["id"]) is not int for e in entries)
                or [e["id"] for e in entries] != list(range(len(entries)))
                or len({e["name"] for e in entries}) != len(entries)
                or not entries or entries[0]["name"] != "base"):
            raise ValueError("Skills require ordered contiguous IDs starting with base=0 and unique names")
        self.manifest = []
        self.device = torch.device(device)
        base_state = None
        for entry in entries:
            path = Path(entry["checkpoint"])
            checkpoint = read_checkpoint(path if path.is_absolute() else Path(root) / path)
            if entry.get("sha256", checkpoint["_identity"]["sha256"]) != checkpoint["_identity"]["sha256"]:
                raise ValueError(f"{entry['name']}: checkpoint content hash changed")
            metadata = checkpoint["metadata"]
            if metadata["algorithm"] != "ppo":
                raise ValueError(f"{entry['name']}: expected a single-agent PPO checkpoint")
            if base_state is None:
                base_state = checkpoint["models"]["actor"]
                base_hash = checkpoint["_identity"]["sha256"]
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
                if not adaptation["lora"] or not source or source["sha256"] != base_hash:
                    raise ValueError(f"{entry['name']}: adapter source hash differs from configured base")
                # Exploration log_std can be trained; every deterministic base tensor must match.
                for name, value in base_state.items():
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
            self.manifest.append({**entry, "sha256": checkpoint["_identity"]["sha256"]})
            yield checkpoint, spec or stored_spec, params


class AdapterBank(_FrozenBank):
    def __init__(self, manifest, *, spec=None, device="cpu", root=PROJECT_ROOT):
        super().__init__()
        started = time.perf_counter()
        self.actors = nn.ModuleList(
            solo_evaluation_actor(checkpoint, skill_spec, params, self.agent_id).actor
            for checkpoint, skill_spec, params in self._checkpoints(manifest, spec, device, root)
        )
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
_BANKS = weakref.WeakValueDictionary()


def _shared_bank(factory, manifest, spec, device):
    key = (factory, json.dumps(manifest, sort_keys=True), torch.device(device))
    bank = _BANKS.get(key)
    if bank is None:
        bank = factory(manifest, spec=spec, device=device)
        _BANKS[key] = bank
    else:
        aid = bank.agent_id
        if list(spec.agent_ids) != [aid]:
            raise ValueError("Solo evaluation requires the selected original vehicle identity")
        validate_checkpoint_compatibility(bank.metadata['contracts']['observations'][aid],
            {'dim': spec.observation_dims[aid], 'contract': spec.observation_contracts[aid]})
        validate_checkpoint_compatibility(bank.metadata['contracts']['actions'],
            {'low': spec.action_lows[aid], 'high': spec.action_highs[aid]})
    return bank


def shared_adapter_bank(manifest, *, spec, device='cpu'):
    return _shared_bank(AdapterBank, manifest, spec, device)


class ResidualBank(_FrozenBank):
    """One materialized frozen base and stacked LoRA factors."""
    def __init__(self, manifest, *, spec=None, device='cpu', root=PROJECT_ROOT):
        super().__init__()
        from agents.common.lora import LoRAActor, head_adapter_targets

        started = time.perf_counter()
        checkpoints = self._checkpoints(manifest, spec, device, root)
        checkpoint, skill_spec, params = next(checkpoints)
        self.base = solo_evaluation_actor(checkpoint, skill_spec, params, self.agent_id).actor
        targets = head_adapter_targets(self.base.net)
        if not targets:
            raise ValueError('Unsupported residual composition actor head')
        layer_factors = {name: [] for name in targets}
        for checkpoint, skill_spec, params in checkpoints:
            template, _, _, _, lora = validate_solo_actor_checkpoint(checkpoint, skill_spec, params, self.agent_id)
            if not isinstance(template, LoRAActor) or lora['mode'] != 'shared' or len(template.adapters) != 1:
                raise ValueError('Residual composition requires shared specialist LoRA actors')
            if set(template._adapter_keys) != set(targets):
                raise ValueError('Specialist adapted layer mappings differ')
            state = checkpoint["models"]["actor"]
            for name, key in template._adapter_keys.items():
                prefix = f'adapters.0.{key}'
                layer_factors[name].append((state[f'{prefix}.A'], state[f'{prefix}.B'], template.adapters[0][key].scale))
        if len(self.manifest) < 2:
            raise ValueError('Residual composition requires shared specialist LoRA actors')
        self.layer_contract = []
        self._factor_names = {}
        self._factor_slices = {}
        for index, (name, factors) in enumerate(layer_factors.items()):
            ranks = [a.shape[0] for a, _, _ in factors]
            scales = [scale for _, _, scale in factors]
            a_name, b_name, index_name = f'a_{index}', f'b_{index}', f'owners_{index}'
            self.register_buffer(a_name, torch.cat([a.detach() for a, _, _ in factors], dim=0).to(self.device))
            self.register_buffer(b_name, torch.cat([b.detach() * scale for _, b, scale in factors], dim=1).to(self.device))
            self.register_buffer(index_name, torch.repeat_interleave(
                torch.arange(len(factors), device=self.device),
                torch.tensor(ranks, device=self.device)))
            offsets = list(accumulate(ranks, initial=0))
            self._factor_slices[name] = [slice(start, end) for start, end in zip(offsets, offsets[1:])]
            self._factor_names[name] = (a_name, b_name, index_name)
            self.layer_contract.append({'layer': name, 'ranks': ranks,
                                        'alphas': [scale * rank for scale, rank in zip(scales, ranks)],
                                        'scales': scales})
        self.requires_grad_(False).eval()
        self.initialization_seconds = time.perf_counter() - started

    def _layer(self, name, layer, inputs, weights, diagnostics=None):
        output = layer(inputs)
        if name in self._factor_names:
            a, b, owners = (getattr(self, key) for key in self._factor_names[name])
            low_rank = F.linear(inputs, a) * weights.index_select(-1, owners)
            residual = F.linear(low_rank, b)
            if diagnostics is not None:
                norm = residual.norm(dim=-1)
                diagnostics[f'{name}/combined_norm'] = norm.mean()
                diagnostics[f'{name}/relative_norm'] = (norm / output.norm(dim=-1).clamp_min(1e-8)).mean()
                for entry, columns in zip(self.manifest[1:], self._factor_slices[name]):
                    value = F.linear(low_rank[..., columns], b[:, columns])
                    diagnostics[f'{name}/weighted_norm/{entry["name"]}'] = value.norm(dim=-1).mean()
            output = output + residual
        return output

    def _sequence(self, sequence, inputs, weights, prefix='', diagnostics=None):
        for name, layer in sequence.named_children():
            path = f'{prefix}.{name}' if prefix else name
            inputs = self._layer(path, layer, inputs, weights, diagnostics)
        return inputs

    @torch.no_grad()
    def forward(self, observations, coefficients, *, collect_influence=False):
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        weights = torch.as_tensor(coefficients, dtype=torch.float32, device=self.device)
        if obs.ndim != 2 or obs.shape[1] != self.obs_dim or weights.shape != (len(obs), len(self.manifest) - 1):
            raise ValueError('Expected LL observations and one coefficient per specialist and batch row')
        if not torch.isfinite(weights).all() or torch.any(weights.abs() > 1):
            raise ValueError('Residual coefficients must be finite and within [-1, 1]')
        x = self.base.encoder(obs)
        diagnostics = {} if collect_influence else None
        if isinstance(self.base.net, nn.Sequential):
            x = self._sequence(self.base.net, x, weights, diagnostics=diagnostics)
        elif isinstance(self.base.net, ResidualMLP):
            x = self._sequence(self.base.net.input_projection, x, weights, 'input_projection', diagnostics)
            for index, block in enumerate(self.base.net.blocks):
                x = x + self._sequence(block.branch, x, weights, f'blocks.{index}.branch', diagnostics)
            x = self.base.net.output(x)
        action = torch.tanh(x)
        if diagnostics is not None:
            values = torch.stack(tuple(diagnostics.values())).cpu().tolist()
            return action, dict(zip(diagnostics, values))
        return action

    def accounting(self):
        base = sum(p.numel() for p in self.base.parameters())
        residual = sum(getattr(self, key).numel() for names in self._factor_names.values() for key in names[:2])
        return dict(parameters=base + residual, frozen_base=base, frozen_residual=residual, trainable=0,
                    tensor_bytes=sum(t.numel() * t.element_size() for t in (*self.parameters(), *self.buffers())),
                    initialization_seconds=self.initialization_seconds)


def shared_residual_bank(manifest, *, spec, device='cpu'):
    return _shared_bank(ResidualBank, manifest, spec, device)
