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


def test_residual_bank_materializes_only_base_and_preserves_one_hot_actions(bank, monkeypatch):
    from hierarchical.bank import ResidualBank

    def load_base_only(checkpoint, *args):
        assert checkpoint['metadata']['adaptation']['lora'] is None
        return solo_evaluation_actor(checkpoint, *args)

    monkeypatch.setattr('hierarchical.bank.solo_evaluation_actor', load_base_only)
    residual = ResidualBank({'skills': bank.manifest})
    obs = torch.randn(6, bank.obs_dim, generator=torch.Generator().manual_seed(13))
    weights = torch.eye(6)[:, 1:]
    actions, influence = residual(obs, weights, collect_influence=True)
    torch.testing.assert_close(actions, bank(obs, torch.arange(6)), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actions, residual(obs, weights))
    assert influence
    assert residual.accounting()['tensor_bytes'] < bank.accounting()['tensor_bytes']


def test_dirichlet_native_spec_and_deterministic_sampling():
    from adapters.native_torchrl import NativeRaceTorchRLEnv
    from core.scenario import load_and_expand_scenario
    from core.task_builder import create_race_task
    from torchrl.envs.utils import check_env_specs
    from training.algorithms import create_learner, learner_params

    scenario = load_and_expand_scenario('scenarios/ppo_hierarchical_composition.yaml')
    scenario['environment']['max_steps'] = 8
    task = create_race_task(scenario)
    env = NativeRaceTorchRLEnv(task)
    try:
        params = learner_params(scenario, task.spec, 'ppo')
        agent = create_learner('ppo', task.spec, {**params, 'device': 'cpu'})
        check_env_specs(env, seed=42)
        observation = task.reset(seed=42).observations['car_0']
        sample = agent.sample_batch(observation[None], deterministic=True)
        executed = agent.evaluation_actions(['car_0'], {'car_0': observation})
        torch.testing.assert_close(torch.from_numpy(sample.actions[0]),
                                   torch.from_numpy(executed['car_0'].coefficients))
        assert torch.from_numpy(sample.log_probs).isfinite().all()
        task.step(executed)
        assert task.metrics()['composition/influence_samples'] == 0
        task.influence_every = 1
        task.step(executed)
        assert task.metrics()['composition/influence_samples'] == 1
    finally:
        env.close()


def test_final_protocol_uses_expanded_seed_range_for_categorical_and_composition():
    from core.scenario import ScenarioError, load_and_expand_scenario
    from training.evaluation_config import resolve_evaluation_protocol

    protocols = []
    for scenario_name in ('ppo_hierarchical_continual', 'ppo_hierarchical_composition'):
        scenario = load_and_expand_scenario(f'scenarios/{scenario_name}.yaml')
        scenario['evaluation'].update(seed=110, episodes=1)
        scenario['evaluation']['final_test'].update(seed=106, episodes=1)
        with pytest.raises(ScenarioError, match='seed ranges must be disjoint'):
            resolve_evaluation_protocol(scenario, 'selection')
        scenario['evaluation']['final_test']['seed'] = 111
        protocols.append(resolve_evaluation_protocol(scenario, 'final'))

    assert protocols[0] == protocols[1]
    assert protocols[0]['episodes'] == 7
    assert protocols[0]['episodes_per_map'] == 1
    scenario['evaluation']['final_test'].pop('per_map')
    assert resolve_evaluation_protocol(scenario, 'final') == protocols[1]
    scenario['evaluation']['final_test']['seed'] = 2**32 - 6
    with pytest.raises(ScenarioError, match='exceed the NumPy seed range'):
        resolve_evaluation_protocol(scenario, 'final')


@pytest.mark.parametrize('hierarchy', [
    {'routing_type': 'composition'},
    {'metrics': {'composition/routing_entropy': .5}},
])
def test_hard_skill_report_rejects_composition_without_a_trajectory(tmp_path, hierarchy):
    import json
    from hierarchical.reporting import summarize

    path = tmp_path / 'evaluation_report.json'
    path.write_text(json.dumps({'episode_results': [{'race_record': {'hierarchical': hierarchy}}]}))
    with pytest.raises(ValueError, match='require categorical routing'):
        summarize([path])


def test_episode_hook_forwards_composition_metrics():
    from types import SimpleNamespace
    from training.hooks import WandbHook

    logged = []
    WandbHook(SimpleNamespace(log_metrics=logged.append)).on_episode_end(
        1, 0., {}, {'composition/routing_entropy': .5, 'train/environment_steps': 15})
    assert logged[0]['episode/composition/routing_entropy'] == .5
    assert logged[0]['episode/environment_steps'] == 15


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


def test_standalone_checkpoint_parity(bank):
    import numpy as np
    from agents.common.ppo_policy import PPOPolicy
    obs = torch.randn(3, bank.obs_dim, generator=torch.Generator().manual_seed(23))
    for skill, entry in enumerate(bank.manifest):
        checkpoint = read_checkpoint(PROJECT_ROOT / entry['checkpoint'])
        metadata = checkpoint['metadata']
        params = {'device': 'cpu', '_agent_ids': [bank.agent_id], 'network': metadata['network'],
            '_observation_contract': metadata['contracts']['observations'][bank.agent_id]['contract'],
            '_physics_contract': metadata['contracts']['physics'],
            '_action_contract': metadata['contracts']['actions']['contract'],
            'lora': metadata['adaptation']['lora'] and {key:metadata['adaptation']['lora'][key]
                for key in ('mode', 'rank', 'alpha', 'train_log_std')}}
        policy = PPOPolicy(bank.obs_dim, metadata['contracts']['actions']['low'], metadata['contracts']['actions']['high'], params)
        policy.load_for_evaluation(checkpoint)
        expected = np.stack([policy.predict(row.numpy()) for row in obs])
        assert np.allclose(bank(obs, torch.full((3,), skill)).numpy(), expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize('selector', ['fixed', 'random', 'geometry'])
def test_real_selector_evaluation(selector):
    import numpy as np
    from core.task_builder import create_race_task
    from core.scenario import load_and_expand_scenario
    from training.evaluation import run_evaluation_episode
    scenario = load_and_expand_scenario('scenarios/ppo_hierarchical.yaml')
    scenario['environment']['max_steps'] = scenario['evaluation']['max_steps'] = 7
    scenario['hierarchical']['record_trajectory'] = True
    scenario['hierarchical']['selector']['type'] = selector
    task = create_race_task(scenario, mode='eval')
    try:
        result = run_evaluation_episode(task, lambda ids, obs: {'car_0': np.array([0], dtype=np.int64)}, episode=0, seed=42)
        diagnostic = result.record['hierarchical']
        assert len(diagnostic['trajectory']) == 7
        assert diagnostic['metrics']['hierarchical/ll_decisions'] == 7
        assert diagnostic['metrics']['hierarchical/hl_decisions'] == 2
        assert task.annotations and task.annotations['schema_version'] == '2.0.0'
    finally:
        task.close()


def test_offline_reporting(tmp_path):
    import json
    from hierarchical.reporting import write_reports
    path = tmp_path / 'evaluation_report.json'
    trace = [{'timestamp': i*.05, 'x': float(i), 'y': 0., 'skill': i,
              'switched': bool(i), 'speed': 1., 'lateral_error': .1, 'command_jump': .2}
             for i in range(2)]
    record = {'map_id': 'validation_composite_01', 'seed': 42, 'duration_s': .1,
        'strict_clean_finish': False, 'mean_net_progress_laps': .1, 'any_learner_collision_dnf': False,
        'hierarchical': {'skills': [{'sha256': 'a'*64}], 'trajectory': trace, 'transition_counts': [[1]]}}
    path.write_text(json.dumps({'episode_results': [{'race_record': record}]}))
    result = write_reports([path], tmp_path / 'reports')
    assert result['matrix'] and result['segments'] and result['switches']
    assert (tmp_path / 'reports/trajectory_0_0.svg').exists()
    record['hierarchical']['skills'][0]['sha256'] = 'b'*64
    other = tmp_path / 'other.json'
    other.write_text(json.dumps({'episode_results': [{'race_record': record}]}))
    with pytest.raises(ValueError, match='different adapter banks'):
        write_reports([path, other], tmp_path / 'mixed')


def test_categorical_likelihood_and_entropy():
    from hierarchical.policy import SkillDistribution
    logits = torch.tensor([[[0., 1., -1.]], [[2., 0., 0.]]])
    dist = SkillDistribution(logits)
    sampled = dist.sample()
    assert sampled.shape == (2, 1) and sampled.dtype == torch.long
    reference = torch.distributions.Categorical(logits=logits.squeeze(-2))
    assert torch.equal(dist.mode.squeeze(-1), logits.squeeze(-2).argmax(-1))
    assert torch.allclose(dist.log_prob(sampled), reference.log_prob(sampled.squeeze(-1)))
    assert torch.allclose(dist.entropy(), reference.entropy())


def test_early_terminal_and_reset_ownership(bank, monkeypatch):
    import numpy as np
    from dataclasses import replace
    from core.task_builder import create_race_task
    from core.scenario import load_and_expand_scenario
    from hierarchical.task import HierarchicalTask
    scenario = load_and_expand_scenario('scenarios/ppo_circle_pretrain_20k.yaml')
    task = create_race_task(scenario)
    wrapper = HierarchicalTask(task, bank, {'interval': 20, 'gamma': .99})
    original = task.step
    def terminate(actions, **kwargs):
        step = original(actions, **kwargs)
        return replace(step, decisions={'car_0': replace(step.decisions['car_0'], terminated=True)},
            after=replace(step.after, episode_done=True, agents=(), observations={}))
    monkeypatch.setattr(task, 'step', terminate)
    try:
        wrapper.reset(seed=42)
        step = wrapper.step({'car_0': np.array([3], dtype=np.int64)})
        assert step.ll_decisions == 1 and step.decisions['car_0'].terminated
        assert len(step.decisions['car_0'].next_observation) == wrapper.dim
        wrapper.reset(seed=43)
        assert wrapper.executor.state.active == -1
        assert wrapper.executor.state.switches == 0
        assert wrapper.executor.last_action.tolist() == [0., 0.]
    finally:
        wrapper.close()


def test_dwell_returns_and_selector_reset_stream(bank):
    import numpy as np
    from core.task_builder import create_race_task
    from core.scenario import load_and_expand_scenario
    from hierarchical.task import HierarchicalTask
    scenario = load_and_expand_scenario('scenarios/ppo_circle_pretrain_20k.yaml')
    scenario['environment']['max_steps'] = 12
    scenario['environment']['action_repeat'] = 2
    task = HierarchicalTask(create_race_task(scenario), bank,
        {'interval': 2, 'gamma': .9, 'selector': {'type': 'random'}, 'dwell_time_scale_seconds': 1.0})
    try:
        start = task.reset(seed=42)
        dwell_index = bank.obs_dim + len(bank.actors)
        assert start.observations['car_0'][dwell_index] == 0
        first = task.selector.rng.integers(2**32, size=8)
        task.reset(seed=42)
        assert np.array_equal(first, task.selector.rng.integers(2**32, size=8))
        task.reset()
        assert not np.array_equal(first, task.selector.rng.integers(2**32, size=8))
        task.config['selector'] = {'type': 'learned'}
        task.reset(seed=43)
        for skill in (0, 0, 1):
            previous_elapsed = task.elapsed
            task.step({'car_0': np.array([skill], dtype=np.int64)})
            if skill == 1:
                assert task.dwell_seconds == pytest.approx(task.elapsed - previous_elapsed)
            else:
                assert task.dwell_seconds == pytest.approx(task.elapsed)
        # Identical fixed-skill trajectory under different macro partitions.
        outputs = []
        for interval in (1, 2, 5):
            task.config['selector'] = {'type': 'fixed', 'skill': 0}
            task.executor.interval = interval
            task.reset(seed=43)
            while not task.episode_done:
                result = task.step({'car_0': np.array([0], dtype=np.int64)})
                assert task.dwell_seconds == pytest.approx(task.elapsed)
                if result.after.observations:
                    assert result.after.observations['car_0'][dwell_index] == pytest.approx(np.tanh(task.elapsed))
            outputs.append(task.metrics())
        for name in ('return_ll_undiscounted', 'return_ll_globally_discounted'):
            assert [row['hierarchical/' + name] for row in outputs] == pytest.approx([outputs[0]['hierarchical/' + name]] * 3)
    finally:
        task.close()


def test_extended_preview_contract_and_invalid_scales(bank):
    import numpy as np
    from core.task_builder import create_race_task
    from core.scenario import load_and_expand_scenario
    from hierarchical.task import HierarchicalTask
    scenario = load_and_expand_scenario('scenarios/ppo_circle_pretrain_20k.yaml')
    task = create_race_task(scenario)
    try:
        with pytest.raises(ValueError, match='dwell_time_scale_seconds'):
            HierarchicalTask(task, bank, {'dwell_time_scale_seconds': 0})
        extended = HierarchicalTask(task, bank, {'preview': {'enabled': True, 'samples': 20, 'spacing_m': .6}})
        initial = extended.reset(seed=42)
        assert initial.observations['car_0'].shape == (bank.obs_dim + 9 + 40,)
        assert extended.spec.observation_contracts['car_0']['version'] == 2
        assert np.isfinite(initial.observations['car_0']).all()
    finally:
        task.close()
