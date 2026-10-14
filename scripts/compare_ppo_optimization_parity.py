"""One fixed PPO minibatch per implementation; no environments or training run.

Exports the historical commit, copies corresponding parameters and Adam state,
and invokes both production update functions. Gradients are observed with hooks.
Artifacts default to /tmp/f110-ppo-optimization-parity.
"""
from __future__ import annotations

from copy import deepcopy
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "71a3e0f62d8a77dbcd93ff67c01f78dae8f9b9aa"
DECISIONS = [0,409600,819200,4096000,8192000,12288000,120000000]


def names(agent):
    return {f"{owner}.{name}": parameter for owner in ("actor","critic")
            for name,parameter in getattr(agent,owner).named_parameters()}


def outputs(agent, observation, raw):
    mean,scale=agent.actor(observation)
    log_prob,entropy=agent.actor.evaluate_actions(observation,raw.tanh(),raw)
    latent=torch.distributions.Normal(mean,scale).log_prob(raw).sum(-1)
    return {"mean":mean.detach(),"std":scale.expand_as(mean).detach(),
            "critic":agent.critic(observation).detach(),"latent_log_prob":latent.detach(),
            "squashed_log_prob":log_prob.detach(),"entropy":entropy.detach()}


def initialization(factory,observation,raw):
    result={}
    for seed in (42,43):
        torch.manual_seed(seed)
        agent=factory()
        state=names(agent)
        result[str(seed)]={"hashes":{name:hashlib.sha256(p.detach().numpy().tobytes()).hexdigest()
                                     for name,p in state.items()},
            "rng_hash":hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),
            "parameters":sum(p.numel() for p in state.values()),
            "outputs":outputs(agent,observation,raw)}
    return result


def capture_update(agent,invoke):
    parameters=names(agent)
    pre,post={},{}
    handles=[parameter.register_hook(lambda gradient,name=name: pre.update({name:gradient.detach().clone()}))
             for name,parameter in parameters.items()]
    step=agent.optimizer.step
    backward=torch.Tensor.backward
    combined_losses=[]
    def observed_backward(loss,*args,**kwargs):
        combined_losses.append(float(loss.detach()))
        return backward(loss,*args,**kwargs)
    def observed_step(*args,**kwargs):
        post.update({name:p.grad.detach().clone() for name,p in parameters.items() if p.grad is not None})
        return step(*args,**kwargs)
    agent.optimizer.step=observed_step
    torch.Tensor.backward=observed_backward
    try:
        metrics=invoke()
    finally:
        agent.optimizer.step=step
        torch.Tensor.backward=backward
        for handle in handles:handle.remove()
    norm=lambda data:float(torch.linalg.vector_norm(torch.stack([g.norm() for g in data.values()])))
    result={"metrics":metrics,"pre_gradients":pre,"post_gradients":post,
        "combined_losses":combined_losses,
        "gradient_norms":{phase:{"total":norm(values),**{owner:norm({k:v for k,v in values.items()
            if k.startswith(owner+'.')}) for owner in ("actor","critic")}}
            for phase,values in (("pre",pre),("post",post))},
        "parameters_after":{name:p.detach().clone() for name,p in parameters.items()},
        "adam_state":{name:deepcopy(agent.optimizer.state[p]) for name,p in parameters.items()},
        "missing_gradients":sorted(set(parameters)-set(pre)),
        "nonfinite_gradients":[name for name,value in pre.items() if not torch.isfinite(value).all()],
        "zero_gradients":[name for name,value in pre.items() if not value.any()]}
    result["clipping_coefficient"]=min(1.,agent.max_grad_norm/(result["gradient_norms"]["pre"]["total"]+1e-6))
    return result


def metadata(agent):
    parameters=names(agent)
    lookup={id(p):name for name,p in parameters.items()}
    groups=[]
    for group in agent.optimizer.param_groups:
        groups.append({**{k:v for k,v in group.items() if k!='params'},
                       "parameter_order":[lookup[id(p)] for p in group['params']]})
    return {"actor":str(agent.actor),"critic":str(agent.critic),"optimizer_groups":groups,
            "clip_parameter_order":[lookup[id(p)] for p in agent._optim_parameters],
            "unique_optimizer_parameters":len({id(p) for g in agent.optimizer.param_groups for p in g['params']}),
            "actor_critic_shared_parameters":bool(set(map(id,agent.actor.parameters()))&set(map(id,agent.critic.parameters())))}


def schedule(agent):
    result={}
    for count in DECISIONS:
        agent.set_training_progress(count/120000000)
        result[str(count)]=agent.optimizer.param_groups[0]['lr']
    agent.set_training_progress(0)
    return result


def historical(args):
    sys.path.insert(0,str(args.source/'src'))
    import yaml
    from agents.ppo import PPOAgent
    config=yaml.safe_load((args.source/'scenarios/ppo_lap_completion_pretrain.yaml').read_text())
    production={**config['training_defaults'],**config['agents']['car_0']['params']}
    params={**config['training_defaults'],**config['agents']['car_0']['params'],
            'device':'cpu','n_steps':8,'n_epochs':1,'batch_size':8}
    fixture=torch.load(args.output/'fixture.pt',weights_only=False)
    factory=lambda:PPOAgent(158,np.array([-1.,0.]),np.array([1.,5.]),params)
    initial=initialization(factory,fixture['observations'],fixture['raw_actions'])
    agent=factory()
    agent.actor.load_state_dict(fixture['actor'])
    agent.critic.load_state_dict(fixture['critic'])
    native_metadata=metadata(agent)
    rates=schedule(agent)
    diagnostic=outputs(agent,fixture['observations'],fixture['raw_actions'])
    # Both old/new policy predictions are shared explicitly; historical stable
    # Jacobian supplies the old squashed log probability of each latent sample.
    old_log_prob=fixture['old_latent_log_prob']-agent.actor._log_jacobian(fixture['raw_actions']).sum(-1)
    log_difference=diagnostic['squashed_log_prob']-old_log_prob
    diagnostic.update(log_difference=log_difference,ratio=log_difference.exp(),
        clip_fraction=((log_difference.exp()<.8)|(log_difference.exp()>1.2)).float().mean())
    entropy=agent.actor.evaluate_actions(fixture['observations'],fixture['raw_actions'].tanh(),fixture['raw_actions'])[1].mean()
    entropy_gradients={name:g.detach() for name,g in zip(names(agent),torch.autograd.grad(entropy,
        tuple(agent.actor.parameters()),allow_unused=True))}
    agent.optimizer.load_state_dict(fixture['optimizer'])
    order=[]
    handle=agent.actor.net.register_forward_pre_hook(lambda module,inputs:order.append(inputs[0].detach().clone()))
    torch.manual_seed(77)
    result=capture_update(agent,lambda:agent._update(fixture['observations'],fixture['raw_actions'].tanh(),
        old_log_prob,fixture['advantages'],fixture['targets'],fixture['raw_actions']))
    handle.remove()
    result.update(initialization=initial,metadata=native_metadata,schedule=rates,outputs=diagnostic,
                  entropy_gradients=entropy_gradients,minibatch_observations=order[0],
                  production_configuration={k:production.get(k) for k in ('n_steps','n_epochs','batch_size',
                    'pi_hidden_dims','vf_hidden_dims','activation','learning_rate','learning_rate_end','lr_schedule','target_kl')})
    torch.save(result,args.output/'historical.pt')


def errors(a,b):
    a,b=torch.as_tensor(a,dtype=torch.float64),torch.as_tensor(b,dtype=torch.float64)
    delta=(a-b).abs()
    return {'absolute':float(delta.max()),'relative':float((delta/a.abs().clamp_min(1e-12)).max()),
            'within_tolerance':bool(torch.isclose(a,b,atol=1e-6,rtol=1e-5).all())}


def tensor_errors(old,new):
    each={name:errors(value,new[name]) for name,value in old.items()}
    largest=max(each,key=lambda name:each[name]['absolute'])
    relative_name=max(each,key=lambda name:each[name]['relative'])
    left,right=old[relative_name].double().flatten(),new[relative_name].double().flatten()
    index=int(((left-right).abs()/left.abs().clamp_min(1e-12)).argmax())
    return {'largest_parameter':largest,'max_absolute':each[largest]['absolute'],
            'max_relative':max(row['relative'] for row in each.values()),
            'largest_relative_parameter':relative_name,'relative_error_values':[float(left[index]),float(right[index])],
            'within_tolerance':all(row['within_tolerance'] for row in each.values()),'per_parameter':each}


def sampler_probe():
    from tensordict import TensorDict
    from torchrl.data import TensorDictReplayBuffer,LazyTensorStorage,SamplerWithoutReplacement
    result={}
    for n,batch in ((8,4),(7,3)):
        replay=TensorDictReplayBuffer(storage=LazyTensorStorage(n),
            sampler=SamplerWithoutReplacement(drop_last=False),batch_size=batch)
        replay.extend(TensorDict({'index':torch.arange(n)},[n]))
        epochs=[]
        for _ in range(3):
            pieces=[replay.sample()['index'] for _ in range(math.ceil(n/batch))]
            indices=torch.cat(pieces)
            assert indices.sort().values.tolist()==list(range(n))
            epochs.append({'indices':indices.tolist(),'minibatch_sizes':[len(p) for p in pieces]})
        result[f'{n}/{batch}']=epochs
    return result


def resume_probe(agent,factory,directory):
    from agents.common.checkpoints import RESUME_BOUNDARY,capture_rng,optimization_contract
    progress=12288000/120000000
    agent.set_training_progress(progress)
    state={'boundary':RESUME_BOUNDARY,'optimizer':deepcopy(agent.optimizer.state_dict()),
        'scheduler':None,'optimization_contract':optimization_contract(agent),'schedule_progress':progress,
        'progress':{'environment_steps':12288000,'updates':30},
        'budget':{'total_steps':120000000,'n_episodes':None},'rng':capture_rng(),'hooks':{},'recovery_seed':42}
    path=directory/'resume_probe.pt'
    agent.save_checkpoint(path,training_state=state)
    recovered=factory()
    restored,_=recovered.resume_training(path)
    assert restored['schedule_progress']==progress
    assert recovered.optimizer.param_groups[0]['lr']==agent.optimizer.param_groups[0]['lr']
    for actual,expected in zip(recovered.optimizer.state.values(),agent.optimizer.state.values()):
        for key in expected:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
    return {'progress':progress,'learning_rate':recovered.optimizer.param_groups[0]['lr'],
            'adam_state_restored_exactly':True,'environments_initialized':0}


def main(args):
    sys.path.insert(0,str(ROOT/'src'))
    from core.scenario import load_and_expand_scenario
    from training.algorithms import resolve_training_params
    from agents.torchrl_ppo import TorchRLPPOAgent
    from agents.torchrl_updates import optimize_ppo
    from tensordict import TensorDict
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    scenario=load_and_expand_scenario('scenarios/ppo_lap_completion_pretrain.yaml')
    resolved=resolve_training_params(scenario['agents']['car_0'],scenario)
    params={**resolved,'device':'cpu','n_steps':8,'n_epochs':1,'batch_size':8}
    rng=np.random.default_rng(321)
    observation=torch.tensor(rng.uniform(-1,1,(8,158)),dtype=torch.float32)
    raw=torch.tensor([[-.4,.3],[.2,-.5],[1.5,-1.],[-1.8,1.6],[5.,-5.],[-5.,5.],[8.,-8.],[-8.,8.]])
    factory=lambda:TorchRLPPOAgent(158,np.array([-1.,0.]),np.array([1.,5.]),params)
    initial=initialization(factory,observation,raw)
    torch.manual_seed(42)
    agent=factory()
    native_metadata=metadata(agent)
    rates=schedule(agent)
    diagnostic=outputs(agent,observation,raw)
    mean,scale=agent.actor(observation)
    old_mean=mean.detach()+torch.tensor([.12,-.10])
    old_latent=torch.distributions.Normal(old_mean,scale.detach()).log_prob(raw).sum(-1).detach()
    # Use nonzero identical Adam moments to verify state mapping as well as a
    # fresh gradient. Exactly one real optimizer step is made on either side.
    for p in agent._optim_parameters:
        agent.optimizer.state[p]={'step':torch.tensor(7.),'exp_avg':torch.full_like(p,1e-4),
                                  'exp_avg_sq':torch.full_like(p,3e-6)}
    fixture={'actor':deepcopy(agent.actor.state_dict()),'critic':deepcopy(agent.critic.state_dict()),
        'optimizer':deepcopy(agent.optimizer.state_dict()),'observations':observation,'raw_actions':raw,
        'old_latent_log_prob':old_latent,'old_mean':old_mean,'old_std':scale.detach(),
        'advantages':torch.tensor([2.,-1.,.5,-2.,1.5,-.5,-1.5,1.]),
        'targets':torch.tensor([2.,-.5,1.,-2.,.25,1.5,-1.25,.75])}
    torch.save(fixture,args.output/'fixture.pt')
    entropy=agent.actor.evaluate_actions(observation,raw.tanh(),raw)[1].mean()
    entropy_gradients={name:g.detach() for name,g in zip(names(agent),torch.autograd.grad(entropy,
        tuple(agent.actor.parameters()),allow_unused=True))}
    data=TensorDict({'observation':observation,'raw_action':raw,'action':raw.tanh(),
        'raw_log_prob':old_latent,'advantage':fixture['advantages'][:,None],
        'value_target':fixture['targets'][:,None]},[8])
    losses=agent.loss_module(data.clone())
    distribution=agent.probabilistic_actor.get_dist(data.clone(False))
    log_difference=distribution.log_prob(raw).detach()-old_latent
    diagnostic.update(log_difference=log_difference,ratio=log_difference.exp(),
                      mean=distribution.mean.detach(),std=distribution.stddev.detach(),
                      latent_log_prob=distribution.log_prob(raw).detach(),
                      clip_fraction=losses['clip_fraction'].detach(),
                      builtin_kl=losses['kl_approx'].detach())
    order=[]
    handle=agent.loss_module.register_forward_pre_hook(lambda module,inputs:order.append(inputs[0]['observation'].clone()))
    torch.manual_seed(77)
    result=capture_update(agent,lambda:optimize_ppo(agent,data.clone()))
    handle.remove()
    result.update(initialization=initial,metadata=native_metadata,schedule=rates,outputs=diagnostic,
                  entropy_gradients=entropy_gradients,minibatch_observations=order[0])
    torch.save(result,args.output/'current.pt')
    with tempfile.TemporaryDirectory(prefix='ppo-parity-') as directory:
        archive=args.output/'baseline.tar'
        with archive.open('wb') as out:
            subprocess.run(['git','-C',str(args.baseline_repo),'archive',COMMIT],stdout=out,check=True)
        with tarfile.open(archive) as tar:tar.extractall(directory,filter='data')
        archive.unlink()
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--source',directory,
            '--output',str(args.output)],check=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    old=torch.load(args.output/'historical.pt',weights_only=False)
    report={'baseline_commit':COMMIT,'current_commit':subprocess.check_output(
        ['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),'device':'cpu','cuda_verified':False,
        'configuration':{'historical':old['production_configuration'],
                         'current':{k:resolved.get(k) for k in ('n_steps','n_epochs','batch_size','network','learning_rate',
            'learning_rate_end','lr_schedule','optimizer','optimizer_kwargs','target_kl')}},
        'initialization':{seed:{'weights_identical':old['initialization'][seed]['hashes']==initial[seed]['hashes'],
            'rng_consumption_identical':old['initialization'][seed]['rng_hash']==initial[seed]['rng_hash'],
            'parameter_count':initial[seed]['parameters'],'mean_error':errors(old['initialization'][seed]['outputs']['mean'],initial[seed]['outputs']['mean']),
            'current_mean_range':[float(initial[seed]['outputs']['mean'].min()),float(initial[seed]['outputs']['mean'].max())],
            'entropy_mean':float(initial[seed]['outputs']['entropy'].mean())} for seed in initial},
        'metadata':{'historical':old['metadata'],'current':native_metadata},
        'minibatch_order_identical':bool(torch.equal(old['minibatch_observations'],result['minibatch_observations'])),
        'distribution':{k:errors(old['outputs'][k],diagnostic[k]) for k in old['outputs']},
        'entropy_gradients':tensor_errors(old['entropy_gradients'],entropy_gradients),
        'losses':{},'metrics':{'historical':old['metrics'],'current':result['metrics']},
        'pre_gradients':tensor_errors(old['pre_gradients'],result['pre_gradients']),
        'post_gradients':tensor_errors(old['post_gradients'],result['post_gradients']),
        'gradient_norms':{'historical':old['gradient_norms'],'current':result['gradient_norms']},
        'clipping_coefficients':{'historical':old['clipping_coefficient'],'current':result['clipping_coefficient']},
        'parameters_after':tensor_errors(old['parameters_after'],result['parameters_after']),
        'adam_first_moment':tensor_errors({k:v['exp_avg'] for k,v in old['adam_state'].items()},
                                        {k:v['exp_avg'] for k,v in result['adam_state'].items()}),
        'adam_second_moment':tensor_errors({k:v['exp_avg_sq'] for k,v in old['adam_state'].items()},
                                         {k:v['exp_avg_sq'] for k,v in result['adam_state'].items()}),
        'adam_step_counters':{'historical':sorted({float(v['step']) for v in old['adam_state'].values()}),
                              'current':sorted({float(v['step']) for v in result['adam_state'].values()})},
        'gradient_health':{side:{k:value[k] for k in ('missing_gradients','nonfinite_gradients','zero_gradients')}
                           for side,value in (('historical',old),('current',result))},
        'schedule':{'historical':old['schedule'],'current':rates},'sampling':sampler_probe(),
        'resume':resume_probe(agent,factory,args.output)}
    for label,key,coefficient in (('policy','policy_loss',1),('unscaled_value','value_loss',1),
                                 ('weighted_value','value_loss',.5),('entropy','entropy',1)):
        a,b=[value['metrics']['train/'+key]*coefficient for value in (old,result)]
        report['losses'][label]={'historical':a,'current':b,**errors(a,b)}
    totals=[v['combined_losses'][0] for v in (old,result)]
    report['losses']['total']={'historical':totals[0],'current':totals[1],**errors(*totals)}
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('metadata','sampling')},indent=2)[:16000])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-repo',type=Path,default=ROOT.parent/'F110_MARL')
    parser.add_argument('--output',type=Path,default=Path('/tmp/f110-ppo-optimization-parity'))
    parser.add_argument('--source',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    historical(args) if args.source else main(args)
