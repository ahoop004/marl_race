"""Small continual sequences verify stage isolation and durable optimizer continuity."""
from copy import deepcopy
import json

import pytest
import torch

from agents.common.checkpoints import read_checkpoint
from application.checkpoints import validate_resume_experiment
from core.configuration import PROJECT_ROOT
from core.provenance import collect_map_protocols
from core.scenario import load_and_expand_scenario
from core.task_builder import create_race_task
from loggers.metric_policy import MetricPolicy
from training.algorithms import create_learner, create_trainer, learner_params
from training.continual import ContinualHook, continual_metrics
from training.hooks import CheckpointHook, TrainingHook


def test_continual_matrix_definitions_and_logging():
    matrix = [[.1, .2, .3], [.8, .4, .5], [.6, .9, .6], [.5, .7, 1.]]
    result = continual_metrics(matrix)
    assert result['average_performance'] == pytest.approx(2.2 / 3)
    assert result['forgetting_per_task'] == pytest.approx([.3, .2])
    assert result['forgetting_max'] == pytest.approx(.3)
    assert result['bwt'] == pytest.approx(-.25)
    assert result['fwt'] == pytest.approx((.4 - .2 + .6 - .3) / 2)
    assert continual_metrics([matrix[0]])['bwt'] is None
    policy = MetricPolicy(scenario={'continual': {'steps_per_task': 5}})
    assert policy.filter({'cl/task_index': 1, 'cl/global_steps': 4}) == {
        'cl/task_index': 1, 'cl/global_steps': 4}
    assert policy.accepts('eval/cl/hairpin/clean_completion_rate')


class Capture(TrainingHook):
    def __init__(self):
        self.rows = []
        self.optimizer_steps = []

    def on_update(self, metrics):
        self.rows.append((metrics['cl/task_family'], metrics['train/update_environment_steps']))
        self.optimizer_steps.append(max(int(v['step']) for v in self._trainer.agent.optimizer.state.values()))


@pytest.mark.parametrize('condition', ['full_finetune', 'ppo_lora'])
def test_small_sequence_recovery_and_collector_overshoot(tmp_path, condition):
    overrides = [f'adaptation={condition}', 'wandb.enabled=false', 'execution.params.device=cpu',
        'execution.params.collection_device=cpu', 'continual.steps_per_task=5',
        'continual.evaluate_every_steps=4', 'experiment.parallel_envs=1', 'experiment.envs_per_worker=2',
        'experiment.measure_stages=false', 'environment.max_steps=2', 'evaluation.episodes=1',
        'algorithm.params.frames_per_batch=4', 'algorithm.params.n_steps=4',
        'algorithm.params.n_epochs=1', 'algorithm.params.batch_size=4',
        'network.actor_hidden_dims=[8,8]', 'network.critic_hidden_dims=[8,8]']
    scenario = load_and_expand_scenario(PROJECT_ROOT / 'scenarios/ppo_continual.yaml', overrides=overrides)
    scenario['adaptation']['checkpoint'] = str(tmp_path / 'source.pt')
    output = tmp_path / 'run'
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def setup(config, resume=None):
        task = create_race_task(config, scenario_dir=PROJECT_ROOT / 'scenarios')
        params = learner_params(config, task.spec, 'ppo')
        learner = create_learner('ppo', task.spec, params)
        if resume is None:
            source_params = {**params, 'lora': None}
            source = create_learner('ppo', task.spec, source_params)
            source.save_checkpoint(tmp_path / 'source.pt')
            learner.initialize_from_checkpoint(tmp_path / 'source.pt', scope='actor_and_critic')
        else:
            state, _ = learner.resume_training(resume)
        checkpoint = CheckpointHook(learner, output, configuration=config, save_every_updates=1,
            provenance={'version': '2.0', 'map_protocols': collect_map_protocols(config['environment'])})
        continual = ContinualHook(config, PROJECT_ROOT / 'scenarios', task.spec, params, output)
        capture = Capture()
        training = create_trainer('ppo', task, learner, hooks=[continual, checkpoint, capture],
            scenario=config, scenario_dir=PROJECT_ROOT / 'scenarios', parallel_envs=1, envs_per_worker=2,
            collection_device='cpu')
        if resume:
            training.restore_training_state(state)
        return task, learner, training, continual, capture

    try:
        first_config = deepcopy(scenario)
        first_config['experiment']['stop_after_steps'] = 4
        task, learner, training, cl, capture = setup(first_config)
        optimizer_identity = id(learner.optimizer)
        try:
            cl.run(training)
            assert id(learner.optimizer) == optimizer_identity
            assert training.collected_steps == 4 and cl.index == 0
            assert len(cl.evaluations) == 1
            saved = read_checkpoint(output / 'latest.pt')
            before_actor = deepcopy(saved['models']['actor'])
            before_critic = deepcopy(saved['models']['critic'])
            before_steps = capture.optimizer_steps[-1]
        finally:
            task.close()
        validate_resume_experiment(saved['metadata'], scenario)
        changed = deepcopy(scenario)
        changed['continual']['families'].reverse()
        with pytest.raises(ValueError):
            validate_resume_experiment(saved['metadata'], changed)
        task, learner, training, cl, capture = setup(scenario, saved)
        try:
            for name, tensor in before_actor.items():
                torch.testing.assert_close(learner.actor.state_dict()[name], tensor, rtol=0, atol=0)
            for name, tensor in before_critic.items():
                torch.testing.assert_close(learner.critic.state_dict()[name], tensor, rtol=0, atol=0)
            optimizer_identity = id(learner.optimizer)
            cl.run(training)
            assert id(learner.optimizer) == optimizer_identity
            assert torch.get_num_threads() == 1
            assert cl.complete and len(cl.evaluations) == 6
            assert training.collected_steps == 40
            assert capture.optimizer_steps[0] > before_steps
            assert capture.optimizer_steps == sorted(set(capture.optimizer_steps))
            assert all(s['actual_steps'] == 8 and s['overshoot'] == 3 for s in cl.stages)
            assert [family for family, _ in capture.rows] == [
                'straight', 'sweeper', 'sweeper', 'hairpin', 'hairpin',
                'chicane', 'chicane', 's_curve', 's_curve']
            assert all(steps == 4 for _, steps in capture.rows)
            assert all('/validation_' in path and '/test_' not in path for path in cl.validation.values())
            final = read_checkpoint(output / 'latest.pt')
            assert final['training']['hooks']['ContinualHook']['task_index'] == 5
            assert final['training']['progress']['updates'] == 10
            results = json.loads((output / 'continual_results.json').read_text())
            assert results['complete'] and len(results['matrices']['clean_completion_rate']) == 6
            assert len(list(output.glob('task_*.pt'))) == 5
            if condition == 'ppo_lora':
                for name, value in before_actor.items():
                    if name.startswith(('net.', 'encoder.')):
                        torch.testing.assert_close(learner.actor.state_dict()[name], value, rtol=0, atol=0)
        finally:
            task.close()
    finally:
        torch.set_num_threads(old_threads)


def test_evaluation_records_nonterminal_contact_and_boundary_facts():
    import numpy as np
    from test_race_task import make_task
    from training.evaluation import run_evaluation_episode
    task, _, _ = make_task({2: ((), True, (), ('learner', 'opponent'))}, repeat=1)
    step = task.env.step
    def with_contact(actions):
        task.env.infos['learner'].update(collision=task.env.tick == 0,
            track_limits={'exceeded': task.env.tick == 0, 'offtrack_distance': 0.1})
        return step(actions)
    task.env.step = with_contact
    result = run_evaluation_episode(task, lambda ids, obs: {aid: np.zeros(2, dtype=np.float32) for aid in ids},
                                    episode=0, seed=42)
    assert result.record['collision_observed']
    assert result.record['boundary_violation_observed']
    assert not result.record['strict_clean_finish']
    assert result.record['agents']['learner']['terminal_reason'] != 'collision'
