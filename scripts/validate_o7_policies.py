"""Bounded checkpoint sensitivity study using the production evaluation runner.

Weights load under the original strict checkpoint contract. Study tasks retain
identical policy specifications; only integration limits and the explicitly
selected friction/initial-condition stress cases differ. No checkpoint is edited
and the ordinary evaluation/recovery compatibility rules remain unchanged.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('PYGLET_HEADLESS', 'true')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--maps', nargs='+', default=['circle_map', 'Budapest_map'])
    p.add_argument('--seeds', nargs='+', type=int, default=[42,43,44])
    p.add_argument('--max-steps', type=int, default=1200)
    p.add_argument('--cases', nargs='+', choices=['nominal','variable_friction','high_slip'],
                   default=['nominal','variable_friction','high_slip'])
    p.add_argument('--preflight-only', action='store_true')
    args=p.parse_args()
    if not 1 <= args.max_steps <= 5000 or not 1 <= len(args.seeds) <= 10:
        p.error('Validation requires 1–5000 steps/episode and 1–10 seeds')
    import torch
    torch.set_num_threads(1)
    from agents.common.checkpoints import read_checkpoint, evaluation_routing
    from core.scenario import load_and_expand_scenario
    from core.task_builder import create_race_task
    from core.provenance import physics_contract, build_run_provenance
    from training.algorithms import learner_params,create_learner,check_evaluation_spec
    from training.runtime import evaluation_mode
    from training.evaluation import run_evaluation_episode
    from metrics.racing_eval import aggregate_eval_episodes
    from adapters.rewards import RewardMapping
    from utils.torch_io import validate_checkpoint_compatibility
    from scripts.benchmark_physics_p2 import replace

    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    scenario_path=Path(args.scenario).resolve()
    checkpoint=read_checkpoint(args.checkpoint)
    base=load_and_expand_scenario(str(scenario_path))
    base['training_defaults']['device']='cpu'
    source_step=checkpoint['metadata']['contracts']['physics']['vehicle_params']['max_integration_step']
    base['environment']['vehicle_params']['max_integration_step']=source_step
    reference=create_race_task(base,scenario_dir=scenario_path.parent,mode='eval')
    try:
        algorithm=checkpoint['metadata']['algorithm'];spec=reference.spec
        params=learner_params(base,spec,algorithm);params['device']='cpu'
        if algorithm=='mappo':params.update(evaluation_routing(checkpoint))
        agent=create_learner(algorithm,spec,params,training=False)
        agent.load_for_evaluation(checkpoint)  # All original contracts remain strict.
        reward_mapping=RewardMapping(params.get('reward_mode','individual'),params.get('team_reward_reduction','mean'))
    finally:reference.close()
    report=dict(checkpoint=checkpoint['_identity'],checkpoint_steps=checkpoint['metadata']['progress'],
        source_contract=checkpoint['metadata']['contracts'],source_step=source_step,
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),algorithm=algorithm,
        maps=args.maps,seeds=args.seeds,max_steps=args.max_steps,episodes=[],failures=[],
        compatibility='Strict source-contract load passed. Every study task also passes check_evaluation_spec.',
        scope='Deterministic existing policy inference plus RaceTask evaluation; bounded sensitivity, not policy-quality certification.')
    def save():
        (output/'policy_validation.json').write_text(json.dumps(report,indent=2,allow_nan=False,default=str)+'\n')
    save()
    if args.preflight_only:
        print('Strict checkpoint load passed',flush=True);return
    for case in args.cases:
        for map_id in args.maps:
            paired_physics=None;initials={}
            for step in (.001,.0025):
                config=deepcopy(base)
                env=config['environment']
                env.update(max_steps=args.max_steps,map_bundles_eval=[map_id],map_pick='round_robin',map_cycle='per_episode',render=False)
                config['evaluation']['max_steps']=args.max_steps
                env['vehicle_params']['max_integration_step']=step
                if case=='variable_friction':
                    env['friction']['eval']={'mode':'grid','values':[.9,1.0489,1.1]}
                contract=physics_contract(env)
                normalized=deepcopy(contract);normalized['vehicle_params']['max_integration_step']=.001
                if paired_physics is None:paired_physics=normalized
                else:validate_checkpoint_compatibility(paired_physics,normalized)
                task=create_race_task(config,scenario_dir=scenario_path.parent,mode='eval')
                check_evaluation_spec(spec,task.spec)
                provenance=build_run_provenance(config,scenario_path=scenario_path,run_id=f'o7_{algorithm}_{case}_{map_id}_{step}',
                    algorithm=algorithm,trainable_agents=task.possible_agents,observation_contracts=task.spec.observation_contracts,devices=('cpu',))
                sim=task.env.sim
                with ExitStack() as stack:
                    stack.callback(task.close)
                    if case=='high_slip':
                        original=sim.reset
                        def high_slip_reset(*a,**kw):
                            original(*a,**kw)
                            for aid in task.possible_agents:
                                car=sim.agents[task.physical_agents.index(aid)]
                                car._physics.reset(pose=car.physics_state[:3],velocity=(6.,1.),yaw_rate=.5,
                                                   steering_angle=.2,wheel_speed=160.)
                                car._sync_physics_view()
                                car._motion_trace.fill(0.)
                                car._motion_trace[:,0]=car.physics_state[:6]
                                car._pose_trace[:]=car._motion_trace[:,:,:3]
                                car._motion_actuator_start[:]=car.physics_state[6:]
                            return sim.current_observation()
                        stack.enter_context(replace(sim,'reset',high_slip_reset))
                    facts=[]
                    for episode,seed in enumerate(args.seeds):
                        statistics=dict(env_seconds=0.,env_steps=0,boundary_agent_steps=0,collision_agent_steps=0)
                        original_step=task.env.step
                        def measured(actions):
                            started=perf_counter()
                            result=original_step(actions)
                            statistics['env_seconds']+=perf_counter()-started
                            statistics['env_steps']+=1
                            for info in result[4].values():
                                statistics['boundary_agent_steps']+=int(bool((info.get('track_limits') or {}).get('exceeded',False)))
                                statistics['collision_agent_steps']+=int(bool(info.get('collision_event',False)))
                            return result
                        initial={}
                        def actions(ids,obs):
                            if not initial:
                                state=np.stack([car.physics_state for car in sim.agents])
                                initial.update(states=state.tolist(),spawn=task.episode_metadata.spawn_configuration,
                                               mu=task.env._episode_physics,
                                               slips={aid:task.env.sim.agents[task.physical_agents.index(aid)].physics_diagnostics()['slip_ratio'].tolist() for aid in task.possible_agents})
                                if step==.001:initials[seed]=state
                                else:np.testing.assert_array_equal(state,initials[seed])
                            return agent.evaluation_actions(ids,obs)
                        started=perf_counter()
                        try:
                            with replace(task.env,'step',measured),evaluation_mode(agent):
                                result=run_evaluation_episode(task,actions,episode=episode,seed=seed,reward_mapping=reward_mapping)
                            elapsed=perf_counter()-started
                            facts.append(result.facts)
                            row=dict(case=case,map=map_id,seed=seed,max_integration_step=step,
                                initial=initial,record=result.record,physics=result.physics,env_steps=result.snapshot.physics_steps,
                                episode_wall_seconds=elapsed,evaluation_steps_per_second=result.snapshot.physics_steps/elapsed,
                                simulation_steps_per_second=statistics['env_steps']/statistics['env_seconds'],
                                statistics=statistics,resolved_trace_segments=len(sim.agents[0]._motion_trace))
                            report['episodes'].append(row)
                            print(algorithm,case,map_id,step,seed,result.snapshot.physics_steps,
                                  {aid:result.facts.agents[aid].terminal_reason for aid in task.possible_agents},flush=True)
                        except Exception as error:
                            report['failures'].append(dict(case=case,map=map_id,seed=seed,step=step,error=repr(error)))
                            save();raise
                        save()
                    summary=aggregate_eval_episodes(facts,focal_agent_id=task.possible_agents[0],
                        opponent_agent_id=task.fixed_policy_agents[0] if task.fixed_policy_agents else None,timestep=task.timestep)
                    report.setdefault('summaries',[]).append(dict(case=case,map=map_id,step=step,summary=summary,provenance=provenance))
                    save()
    save()


if __name__=='__main__':main()
