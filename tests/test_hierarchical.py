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


def test_selectors_and_independent_execution(bank):
    from hierarchical.selectors import SkillExecutor, FixedSelector, RandomSelector, GeometrySelector
    a, b = SkillExecutor(bank), SkillExecutor(bank)
    obs = torch.zeros(bank.obs_dim).numpy()
    assert (a.action(obs, 1) == bank.actors[1].get_action(torch.from_numpy(obs[None]), deterministic=True)[0][0].numpy(force=True)).all()
    b.action(obs, 4)
    assert (a.state.active, b.state.active) == (1, 4)
    a.action(obs, 1)
    assert a.state.switches == 0 and a.state.since_change == 2
    a.action(obs, 2)
    assert a.state.switches == 1 and a.state.since_change == 1
    a.reset()
    assert a.state.active == -1 and b.state.active == 4
    assert FixedSelector(2).select() == 2
    r = RandomSelector(6, seed=4)
    assert set(r.select() for _ in range(100)) == set(range(6))
    g = GeometrySelector(bank.manifest, anticipation=1)
    metadata = {'schema_version': '2.0.0', 'measured': {'length_m': 10}, 'maneuvers': [
        {'type': 'hairpin', 's_start_m': 8, 's_end_m': 2, 'wraps_seam': True,
         'requested': {'composition_role': 'primary'}}]}
    assert g.select(metadata=metadata, s=9.5) == 3
    metadata['maneuvers'][0]['type'] = 'corner'
    assert g.select(metadata=metadata, s=9.5) == 0


def test_macro_trajectory_and_specs(bank):
    import numpy as np
    from core.task_builder import create_race_task
    from core.scenario import load_and_expand_scenario
    from hierarchical.task import HierarchicalTask
    from adapters.native_torchrl import NativeRaceTorchRLEnv
    from torchrl.envs.utils import check_env_specs
    scenario = load_and_expand_scenario('scenarios/ppo_circle_pretrain_20k.yaml')
    scenario['environment']['max_steps'] = 7
    task = create_race_task(scenario)
    wrapper = HierarchicalTask(task, bank, {'interval': 5, 'gamma': .99})
    plain = create_race_task(scenario)
    try:
        initial = wrapper.reset(seed=42)
        plain.reset(seed=42)
        assert initial.observations['car_0'].shape == (bank.obs_dim + 9,)
        rewards = []
        state = plain.reset(seed=42)
        for _ in range(5):
            action = bank(state.observations['car_0'][None], torch.tensor([1]))[0].numpy()
            step = plain.step({'car_0': action})
            rewards.append(step.decisions['car_0'].individual_reward)
            state = step.after
        macro = wrapper.step({'car_0': np.array([1], dtype=np.int64)})
        assert macro.ll_decisions == 5 and macro.physics_steps == 5
        assert macro.decisions['car_0'].individual_reward == pytest.approx(sum(.99**i*r for i,r in enumerate(rewards)))
        assert np.array_equal(macro.after.observations['car_0'][:bank.obs_dim], state.observations['car_0'])
        macro = wrapper.step({'car_0': np.array([2], dtype=np.int64)})
        assert macro.ll_decisions == 2 and macro.decisions['car_0'].truncated
        assert wrapper.total_ll == 7 and wrapper.total_hl == 2
        env = NativeRaceTorchRLEnv(wrapper)
        check_env_specs(env)
    finally:
        task.close()
        plain.close()


@pytest.mark.parametrize('k', [1, 5, 10, 20])
def test_duration_gae(k):
    from hierarchical.policy import DurationGAE
    from tensordict import TensorDict
    class Value(torch.nn.Module):
        def forward(self, x):
            return x[..., 0]
    data = TensorDict({'observation': torch.tensor([[2.], [3.], [5.]]),
        'steps_to_next_obs': torch.full((3, 1), k),
        'next': {'observation': torch.tensor([[3.], [7.], [11.]]),
                 'reward': torch.ones(3, 1), 'done': torch.tensor([[False], [True], [True]]),
                 'terminated': torch.tensor([[False], [False], [True]])}}, [3])
    DurationGAE(Value(), .9, .8)(data)
    second = 1 + .9 ** k * 7 - 3
    assert data['advantage'][:, 0].tolist() == pytest.approx([
        1 + .9 ** k * 3 - 2 + (.9 * .8) ** k * second, second, 1 - 5], abs=1e-6)


def test_categorical_training_checkpoint_resume(bank, tmp_path):
    import numpy as np
    from core.task_builder import create_race_task
    from core.scenario import load_and_expand_scenario
    from hierarchical.task import HierarchicalTask
    from adapters.native_torchrl import NativeRaceTorchRLEnv
    from agents.torchrl_ppo import TorchRLPPOAgent
    from agents.common.checkpoints import capture_rng, RESUME_BOUNDARY, optimization_contract
    from torchrl.envs.utils import set_exploration_type, ExplorationType
    scenario = load_and_expand_scenario('scenarios/ppo_circle_pretrain_20k.yaml')
    scenario['environment']['max_steps'] = 11
    config = {'enabled': True, 'interval': 5, 'gamma': .99,
              'bank': {'skills': [{k:v for k,v in e.items() if k != 'sha256'} for e in bank.manifest]}}
    task = HierarchicalTask(create_race_task(scenario), bank, config)
    params = {'device': 'cpu', '_agent_ids': ['car_0'], 'hierarchical': config,
        'network': {'actor_hidden_dims': [128, 128], 'critic_hidden_dims': [128, 128]},
        'n_steps': 4, 'n_epochs': 1, 'batch_size': 4, '_observation_contract': task.spec.observation_contracts['car_0']}
    agent = TorchRLPPOAgent(task.dim, np.array([0]), np.array([5]), params)
    env = NativeRaceTorchRLEnv(task)
    before = {k:v.clone() for k,v in bank.state_dict().items()}
    try:
        with torch.no_grad():
            rollout = env.rollout(4, agent.collection_policy, break_when_any_done=False)
        metrics = agent.update(rollout)
        assert metrics and all(np.isfinite(v) for v in metrics.values())
        assert sum(p.grad.abs().sum() for p in agent.actor.parameters() if p.grad is not None) > 0
        assert sum(p.grad.abs().sum() for p in agent.critic.parameters() if p.grad is not None) > 0
        assert all(p.grad is None for p in bank.parameters())
        assert all(torch.equal(v,before[k]) for k,v in bank.state_dict().items())
        obs = rollout['agents', 'observation'][0, 0].numpy()
        assert agent.predict(obs).dtype == np.int64
        state = {'boundary': RESUME_BOUNDARY, 'optimizer': agent.optimizer.state_dict(),
            'optimization_contract': optimization_contract(agent), 'scheduler': None,
            'schedule_progress': 0., 'progress': {}, 'budget': {}, 'rng': capture_rng(),
            'hooks': {}, 'recovery_seed': 42}
        path = tmp_path / 'latest.pt'
        agent.save_checkpoint(path, training_state=state)
        restored = TorchRLPPOAgent(task.dim, np.array([0]), np.array([5]), params)
        restored.resume_training(path)
        assert np.array_equal(agent.predict(obs), restored.predict(obs))
        assert restored.update(rollout)
        print(metrics)
    finally:
        env.close()
