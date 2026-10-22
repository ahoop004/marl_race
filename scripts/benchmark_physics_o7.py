"""O7 support for existing P1/P2 replays: archived sources and configured substeps.

Run each source/step in a separate process. Throughput uses existing unprofiled
replay loops; captures, diagnostics, compilation and reset stay outside timing.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import resource
import sys
from time import perf_counter

import numpy as np


def save_json(path, value):
    def clean(v):
        if isinstance(v, dict): return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (tuple, list)): return [clean(x) for x in v]
        if isinstance(v, np.ndarray): return clean(v.tolist())
        if isinstance(v, np.generic): return clean(v.item())
        if isinstance(v, float) and not np.isfinite(v): return None
        return v
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def task_from_trace(p2, inputs, map_id, step):
    from core.task_builder import create_race_task
    report = json.loads((inputs / 'results.json').read_text())
    config = deepcopy(report['workloads'][map_id]['config']['scenario'])
    config['environment']['vehicle_params']['max_integration_step'] = step
    task = create_race_task(config, scenario_dir=Path(config['_configuration']['source']).parent)
    poses = np.array(report['workloads'][map_id]['config']['poses'])
    def reset():
        return task.reset(seed=42, options=dict(poses=poses, map_episode_index=2, spawn_episode_index=3))
    trace = np.load(inputs / f'{map_id}_trace.npz')
    frames = [dict(controls=controls, normalized=dict(zip(task.possible_agents, normalized)),
                   env_actions=dict(zip(task.physical_agents, controls)), state=state,
                   physics_digest=str(physics_digest), task_digest=str(task_digest))
              for controls, normalized, state, physics_digest, task_digest in zip(
                  trace['physical_actions'], trace['normalized_actions'], trace['states'],
                  trace['physics_digests'], trace['task_digests'])]
    return task, reset, frames, trace['rhs_samples']


def capture_sim(sim, reset, advance, steps):
    from physics.dynamic_models import combined_slip_dynamics
    reset()
    states, diagnostics, contacts, indices, dense = [], [], [], [], []
    initial = np.stack([car.physics_state for car in sim.agents])
    from physics.collision_models import trace_state
    for index in range(steps):
        advance(index)
        states.append(np.stack([car.physics_state for car in sim.agents]))
        diagnostics.append(np.stack([combined_slip_dynamics(car._physics._chassis,
            car._physics._actuators._state, car._physics._dynamics_params)[1] for car in sim.agents]))
        contacts.append(sim.collision_times.copy())
        indices.append(sim.collision_idx.copy())
        # Compare dense paths at common times, not merely RK endpoints.
        dense.append(np.stack([[trace_state(car._motion_trace, f) for f in np.linspace(0., 1., 21)]
                               for car in sim.agents]))
    return dict(initial=initial, states=np.asarray(states), diagnostics=np.asarray(diagnostics),
                contacts=np.asarray(contacts), indices=np.asarray(indices), dense=np.asarray(dense))


def measure(args, p2, p1):
    from physics.profiling import SerialPhysicsProfile
    result = dict(source_root=str(args.source_root), max_step=args.max_step,
                  environment_dt=.05, repeats=args.repeats, maps={}, guards={})
    for map_id in ('circle_map', 'Budapest_map'):
        task, reset, frames, samples = task_from_trace(p2, args.inputs, map_id, args.max_step)
        try:
            started = perf_counter()
            rows = dict(runs={layer: [] for layer in p2.LAYERS}, detail={})
            for layer in p2.LAYERS:
                p2.replay(task, reset, frames, layer)  # warm each complete path
            rows['startup_warmup_seconds'] = perf_counter() - started
            reset()
            rows['resolved_max_step'] = task.env.params['max_integration_step']
            rows['trace_shape'] = list(task.env.sim.agents[0]._motion_trace.shape)
            for repeat in range(args.repeats):
                order = p2.LAYERS[repeat % 4:] + p2.LAYERS[:repeat % 4]
                for layer in order:
                    row = p2.replay(task, reset, frames, layer)
                    row['individual_actor_decisions_per_second'] = 2 * row['steps_per_second'] if layer.startswith('task') else None
                    rows['runs'][layer].append(row)
                    print(f'{map_id} {layer} repeat {repeat+1}: {row["steps_per_second"]:.1f} steps/s', flush=True)
            for layer in ('environment', 'task_replay', 'task_live'):
                # Warm the counter/profile specializations separately.
                p2.replay(task, reset, frames[:2], layer, profile=SerialPhysicsProfile())
                rows['detail'][layer] = p2.replay(task, reset, frames, layer, profile=SerialPhysicsProfile())
            selected = samples[np.linspace(0, len(samples)-1, min(256,len(samples)), dtype=int)]
            rows['solver_breakdown'] = p2.solver_breakdown(selected, task.env.sim.agents[0]._physics._dynamics_params,
                                                         repeats=5, batches=8)
            result['maps'][map_id] = rows
            save_json(args.output / 'performance.json', result)
        finally:
            task.close()
    arena = p1.synthetic_map(args.output)
    from physics.simulaton import Simulator
    original = Simulator.__init__
    def configured(self, params, *a, **kw):
        params = deepcopy(params); params['max_integration_step'] = args.max_step
        original(self, params, *a, **kw)
    with p2.replace(Simulator, '__init__', configured):
        for name in ('B1','B3','B4','B5'):
            steps = 200 if name in ('B1','B3') else 12
            sim, task, reset, advance, config = p1.build_case(name, arena, args.source_root, steps)
            p1.run_rollout(sim, task, reset, advance, steps)
            runs = [p1.run_rollout(sim, task, reset, advance, steps) for _ in range(args.repeats)]
            p1.run_rollout(sim, task, reset, advance, steps, SerialPhysicsProfile())
            detail = p1.run_rollout(sim, task, reset, advance, steps, SerialPhysicsProfile())
            result['guards'][name] = dict(runs=runs, detail=detail)
            capture = capture_sim(sim, reset, advance, steps)
            np.savez_compressed(args.output / f'{name}.npz', **capture)
            print(name, 'complete', flush=True)
    save_json(args.output / 'performance.json', result)


def accuracy(args, p2, p1):
    import yaml
    from physics.vehicle import CombinedSlipVehicle
    from physics.dynamic_models import combined_slip_dynamics
    from physics.collision_models import trace_state
    params = yaml.safe_load((args.source_root / 'configs/vehicle/combined_slip.yaml').read_text())
    params['max_integration_step'] = args.max_step
    config = {k:v for k,v in params.items() if k not in {'length','width','wheel_actuators'}}
    cases = {
        'acceleration': (200, (0.,0.), 0., 0., None, lambda i:(0.,160.)),
        'braking': (160, (8.,0.), 0., 0., None, lambda i:(0.,0.)),
        'constant_turn': (200, (4.,0.), 0., .08, None, lambda i:(.08,80.)),
        'changing_steering': (200, (4.,0.), 0., 0., None, lambda i:(.15*np.sin(i*.1),100.)),
        'high_slip': (150, (6.,1.), .5, .25, 160., lambda i:(.3*np.sin(i*.15),160.)),
        'low_speed': (100, (.05,0.), 0., 0., None, lambda i:(.02*np.sin(i*.1),1.5)),
        'reverse': (120, (-2.,0.), 0., 0., None, lambda i:(-.08*np.sin(i*.1),-80.)),
        'saturation': (120, (4.,.5), 0., .25, 160., lambda i:(.25,160.)),
    }
    if args.mode == 'stress':
        cases = {'low_speed_yaw': (120, (.1,.2), 2., .35, 2.,
                                  lambda i:(.35*np.cos(i*.3), 4.*np.sin(i*.2)))}
    result = dict(max_step=args.max_step, free_motion={}, contacts={}, maps={}, kernels={})
    for name, (steps,velocity,yaw,delta,wheel,command) in cases.items():
        car = CombinedSlipVehicle(config, params['wheel_actuators'])
        car.reset(velocity=velocity,yaw_rate=yaw,steering_angle=delta,wheel_speed=wheel)
        initial = car.state
        states, diagnostics, dense, failures = [], [], [], []
        trace = np.empty((int(np.ceil(.05/args.max_step)),4,6))
        for index in range(steps):
            car.command(*command(index))
            try: car.advance(.05,motion_trace=trace)
            except (ValueError, FloatingPointError) as error:
                failures.append(dict(step=index, time=index*.05, error=str(error))); break
            states.append(car.state)
            diagnostics.append(combined_slip_dynamics(car._chassis,car._actuators._state,car._dynamics_params)[1])
            dense.append([trace_state(trace,f) for f in np.linspace(0.,1.,21)])
        np.savez_compressed(args.output / f'{name}.npz', initial=initial, states=np.asarray(states),
                            diagnostics=np.asarray(diagnostics), dense=np.asarray(dense))
        result['free_motion'][name] = dict(steps=len(states), failures=failures, trace_shape=trace.shape)
        print(name,len(states),'steps',failures,flush=True)
    # Reuse the established collision/finish fixtures, with configured limits.
    sys.path.insert(0,str(args.source_root/'tests'))
    from test_swept_collision import make_sim, actions_for, finish_env
    contact_cases = {
        'wall': ([(2.,3.,0.)], [(20.,0.)]),
        'head_on': ([(2.55,3.,0.),(3.45,3.,np.pi)], [(20.,0.),(20.,0.)]),
        'crossing_contact': ([(2.5,3.,0.),(3.,2.5,np.pi/2)], [(20.,0.),(20.,0.)]),
        'near_miss': ([(1.9,3.,0.),(2.6,3.3101,0.)], [(20.,0.),(0.,0.)]),
    }
    if args.mode == 'stress':
        contact_cases = {
            'rotating_wall': ([(2.25,3.,np.pi/2)], [(0.,0.)]),
            'accelerating_wall': ([(5.,20.,0.)], [(0.,0.)]),
            'accelerating_pair': ([(5.,20.,0.),(35.,20.,np.pi)], [(0.,0.),(0.,0.)]),
        }
    for name,(poses,velocities) in contact_cases.items():
        directory=args.output/name; directory.mkdir(exist_ok=True)
        size=800 if name.startswith('accelerating_') else 120
        pixels=np.full((size,size),255,dtype=np.uint8)
        if name in ('wall', 'rotating_wall'):pixels[:,50]=0
        if name == 'accelerating_wall':pixels[:,700]=0
        sim=make_sim(directory,params,poses,velocities,pixels,wall_collision_response=False)
        if name == 'rotating_wall':
            sim.agents[0]._physics.reset(pose=poses[0], yaw_rate=2*np.pi/.05, wheel_speed=0.)
            sim.agents[0]._sync_physics_view()
        actions=actions_for(sim)
        steps=1
        if name.startswith('accelerating_'):
            actions[:,1]=160.;steps=120
        capture=capture_sim(sim,lambda:None,lambda i:sim.step(actions),steps)
        np.savez_compressed(args.output/f'{name}.npz',**capture)
        result['contacts'][name]=dict(times=sim.collision_times,indices=sim.collision_idx)
    if args.mode == 'stress':
        save_json(args.output/'stress.json', result)
        return
    for name,x in [('finish_before_contact',2.05),('contact_before_finish',2.3),('finish_at_contact',2.11)]:
        directory=args.output/name; directory.mkdir(exist_ok=True)
        env=finish_env(directory,params,finish_x=x)
        try:
            output=env.step({'car_0':np.array([0.,100.])})
            car=env.sim.agents[0]
            # Crossing time on the actual cubic path; report along with the
            # authoritative lifecycle outcome (which applies arming and limits).
            candidates=[]
            for part,segment in enumerate(car._pose_trace):
                coefficients=segment[:,0].copy();coefficients[0]-=np.float32(x)
                for root in np.polynomial.polynomial.polyroots(coefficients):
                    if abs(root.imag)<1e-9 and 0.<=root.real<=1.:
                        candidates.append((part+float(root.real))/len(car._pose_trace)*.05)
            result['contacts'][name]=dict(collision_times=env.sim.collision_times,
                terminal_reason=output[4]['car_0']['terminal_reason'], lap_count=int(env.lap_counts[0]),
                crossing_times_on_proposal=candidates)
        finally:env.close()
    # Fixed physical streams isolate integration from controller feedback.
    for map_id in ('circle_map','Budapest_map'):
        task,reset,frames,samples=task_from_trace(p2,args.inputs,map_id,args.max_step)
        try:
            capture=capture_sim(task.env.sim,reset,lambda i:task.env.sim.step(frames[i]['controls']),len(frames))
            np.savez_compressed(args.output/f'{map_id}_fixed.npz',**capture)
            # Live fixed opponents receive the same learner stream; differences
            # here are behavioral sensitivity and are not integration-only error.
            reset(); states=[]; commands=[]; modes=[]; first_error=None; lap_events=[]
            for index,frame in enumerate(frames):
                try: output=task.step(frame['normalized'])
                except Exception as error:
                    first_error=dict(step=index,error=repr(error));break
                states.append(np.stack([car.physics_state for car in task.env.sim.agents]))
                commands.append(np.stack([output.substeps[0].physical_actions[aid] for aid in task.physical_agents]))
                modes.append([getattr(c,'controller',c).mode for c in task.fixed_controllers.values()])
                for aid,record in task.env.lifecycle.records.items():
                    if record.lap_crossed:lap_events.append((index,aid,record.lap_count,record.finish_position))
                if task.episode_done:break
            np.savez_compressed(args.output/f'{map_id}_live.npz',states=states,commands=commands,modes=modes)
            result['maps'][map_id]=dict(fixed_steps=len(capture['states']),live_steps=len(states),
                live_failure=first_error,lap_events=lap_events)
            # Identical RHS queries and integration proposals across archives.
            rhs=[];axles=[];queries=[];proposals=[];proposal_traces=[];stage_samples=[]
            from physics.dynamic_models import integrate_combined_slip
            car=task.env.sim.agents[0];a=car._physics._actuators;p=car._physics._dynamics_params
            for sample in samples:
                force_trace=np.zeros((63,8))
                r,d=combined_slip_dynamics(sample[:6],sample[6:],p,force_samples=force_trace)
                rhs.append(r);axles.append(d);queries.append(force_trace)
            for sample in samples[::32]:
                trace=np.empty((int(np.ceil(.05/args.max_step)),4,6));stages=np.empty((4,8))
                state,act=integrate_combined_slip(sample[:6],sample[6:],sample[6:]+np.array([.01,3.]),
                    a._time_constants,a._rate_limits,p,.05,args.max_step,motion_trace=trace,rhs_samples=stages)
                proposals.append(np.r_[state,act]);proposal_traces.append(trace);stage_samples.append(stages)
            np.savez_compressed(args.output/f'{map_id}_kernels.npz',rhs=rhs,axles=axles,queries=queries,
                                proposals=proposals,proposal_traces=proposal_traces,stage_samples=stage_samples)
            print(map_id,'fixed/live/kernel capture complete',flush=True)
        finally:task.close()
    save_json(args.output/'accuracy.json',result)


def dense_parity(args, p2, p1):
    result = {}
    for map_id in ('circle_map', 'Budapest_map'):
        task,reset,frames,_=task_from_trace(p2,args.inputs,map_id,args.max_step)
        try:
            reset(); hashes=[]
            for frame in frames:
                task.env.sim.step(frame['controls'])
                hashes.append(p1.digest([car._motion_trace for car in task.env.sim.agents]).hexdigest())
            result[map_id]=hashes
        finally:task.close()
    save_json(args.output/'dense_trace_hashes.json',result)


def integration(args, p2, p1):
    from numba import njit
    from physics.dynamic_models import integrate_combined_slip
    result = {}
    for map_id in ('circle_map', 'Budapest_map'):
        task, reset, frames, samples = task_from_trace(p2,args.inputs,map_id,args.max_step)
        try:
            reset(); car=task.env.sim.agents[0]._physics; a=car._actuators
            selected=np.ascontiguousarray(samples[::32])
            @njit
            def batch(samples, constants, rates, params, max_step, batches):
                trace=np.empty((int(np.ceil(.05/max_step)),4,6))
                checksum=0.
                for _ in range(batches):
                    for sample in samples:
                        proposed, act=integrate_combined_slip(sample[:6],sample[6:],sample[6:]+np.array([.01,3.]),
                            constants,rates,params,.05,max_step,motion_trace=trace)
                        checksum+=proposed[3]+act[1]
                return checksum
            started=perf_counter()
            batch(selected,a._time_constants,a._rate_limits,car._dynamics_params,args.max_step,1)
            startup=perf_counter()-started; values=[]
            for _ in range(args.repeats):
                started=perf_counter()
                checksum=batch(selected,a._time_constants,a._rate_limits,car._dynamics_params,args.max_step,20)
                values.append((perf_counter()-started)/(20*len(selected)))
            result[map_id]=dict(seconds_per_vehicle_advance=values,median_vehicle_advances_per_second=1/float(np.median(values)),
                checksum=float(checksum),startup_seconds=startup,trace_segments=int(np.ceil(.05/args.max_step)),
                scope='compiled RK4 integration with dense trace and final validation; excludes Python vehicle wrapper, LiDAR, collision, env and task')
        finally:task.close()
    save_json(args.output/'integration.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-step',type=float,default=.001)
    parser.add_argument('--mode',choices=('accuracy','measure','stress','integration','dense'),required=True)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    for name in ('source_root','inputs','output'):setattr(args,name,getattr(args,name).resolve())
    args.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.source_root))
    sys.path.insert(0,str(args.source_root/'src'))
    os.environ.setdefault('PYGLET_HEADLESS','true')
    from scripts import benchmark_physics_p2 as p2
    from scripts import benchmark_physics as p1
    import physics.dynamic_models
    if not Path(physics.dynamic_models.__file__).resolve().is_relative_to(args.source_root):
        raise RuntimeError('Wrong physics source tree')
    started=perf_counter()
    {'accuracy':accuracy,'stress':accuracy,'measure':measure,'integration':integration,'dense':dense_parity}[args.mode](args,p2,p1)
    save_json(args.output/f'{args.mode}_process.json',dict(wall_seconds=perf_counter()-started,
        lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        python=sys.version,affinity=sorted(os.sched_getaffinity(0)),source_root=str(args.source_root),
        max_step=args.max_step,command=sys.argv))


if __name__=='__main__':main()
