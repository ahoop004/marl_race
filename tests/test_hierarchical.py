"""Focused HRL contract checks, including the available trained checkpoint bank."""
from pathlib import Path
import pytest
import torch
import yaml

from hierarchical.bank import AdapterBank, PROJECT_ROOT
from agents.common.checkpoints import read_checkpoint, solo_evaluation_actor


@pytest.fixture(scope="module")
def bank():
    torch.set_num_threads(1)
    with (PROJECT_ROOT / 'configs/hierarchical/bank.yaml').open() as f:
        return AdapterBank(yaml.safe_load(f))


def test_real_bank_parity_and_routing(bank, monkeypatch):
    obs = torch.randn(6, bank.obs_dim, generator=torch.Generator().manual_seed(13))
    ids = torch.arange(6)
    expected = torch.cat([a.get_action(obs[i:i+1], deterministic=True)[0] for i, a in enumerate(bank.actors)])
    assert torch.equal(bank(obs, ids), expected)
    before = {k: v.clone() for k, v in bank.state_dict().items()}
    monkeypatch.setattr('hierarchical.bank.read_checkpoint', lambda *a: pytest.fail('step-time checkpoint I/O'))
    for _ in range(5):
        bank(obs, ids.roll(1))
    assert all(torch.equal(v, before[k]) for k, v in bank.state_dict().items())
    assert not any(p.requires_grad for p in bank.parameters())
    assert not bank(obs, ids).requires_grad
    print(bank.accounting())


def test_changed_base_rejected(bank, monkeypatch):
    original = read_checkpoint
    def changed(path):
        c = original(path)
        if c['metadata']['adaptation']['lora']:
            c['metadata']['adaptation']['source_checkpoint']['sha256'] = '0' * 64
        return c
    monkeypatch.setattr('hierarchical.bank.read_checkpoint', changed)
    with pytest.raises(ValueError, match='source hash'):
        AdapterBank({'skills': [{k: v for k, v in e.items() if k != 'sha256'} for e in bank.manifest]})
