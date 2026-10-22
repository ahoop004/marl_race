"""Deterministic F5 sweeps, including endpoint-free adversarial motion."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from physics.collision_models import (
    collision, get_vertices, trace_state, swept_contact, swept_map_contact, map_motion_trace,
)
from physics.simulaton import Simulator, ENV_COLLISION_IDX
from physics.vehicle import CombinedSlipVehicle
from env.state_buffer import TerminalAgentConfig, TerminalVehicleController
from env.types import AgentRaceStatus, TerminalReason
from test_map_lidar import write_map, load_map
from utils.map_loader import MapLoader

ROOT = Path(__file__).resolve().parents[1]
EXTENTS = np.array([.29, .29, .155, .155])


def linear_trace(start, end):
    trace = np.zeros((1, 4, 3))
    trace[0, 0] = start
    trace[0, 1] = np.asarray(end) - start
    return trace


@pytest.fixture(params=["default", "combined_slip"], ids=["legacy_st", "combined_slip_st"])
def params(request):
    return yaml.safe_load((ROOT / "configs/vehicle" / f"{request.param}.yaml").read_text())


@pytest.mark.parametrize("a0,a1,b0,b1", [
    ((-1., 0., 0.), (1., 0., 0.), (1., 0., np.pi), (-1., 0., np.pi)),
    ((-1., 0., 0.), (1., 0., 0.), (0., -1., np.pi / 2), (0., 1., np.pi / 2)),
    ((-1., 0., 0.), (1., 0., 0.), (0., 0., np.pi / 2), (0., 0., np.pi / 2)),
    ((-1., 0., 0.), (1., 0., 0.), (0., 0., 0.), (0., 0., 0.)),
])
def test_adversarial_pairs_have_clear_endpoints_but_continuous_contact(a0, a1, b0, b1):
    for a, b in ((a0, b0), (a1, b1)):
        assert not collision(get_vertices(np.array(a), .58, .31), get_vertices(np.array(b), .58, .31))
    a, b = linear_trace(a0, a1), linear_trace(b0, b1)
    time = swept_contact(a, EXTENTS, b, EXTENTS)
    assert 0. < time < 1.
    # Returned time is conservatively just before first touch, not after crossing.
    if a0[1] == b0[1] and a0[2] == b0[2]:
        assert time == pytest.approx(.21, abs=1e-6)


def test_rotation_full_turn_and_curved_translation_are_not_endpoint_interpolation():
    stationary = linear_trace((.40, 0., 0.), (.40, 0., 0.))
    small = np.full(4, .02)
    # All endpoint rectangles agree, but the long body sweeps the small obstacle.
    rotating = linear_trace((0., 0., np.pi / 2), (0., 0., 5 * np.pi / 2))
    long = np.array([.5, .5, .05, .05])
    assert np.isfinite(swept_contact(rotating, long, stationary, small))
    # A polynomial arc has identical endpoint positions and passes through y=1.
    arc = linear_trace((0., 0., 0.), (0., 0., 0.))
    arc[0, 1, 1], arc[0, 2, 1] = 4., -4.
    obstacle = linear_trace((0., 1., 0.), (0., 1., 0.))
    assert np.isfinite(swept_contact(arc, small, obstacle, small))


def test_interval_certification_catches_subsample_contact_and_rejects_close_miss():
    fast = linear_trace((-1000., 0., 0.), (1000., 0., 0.))
    obstacle = linear_trace((.1234567, 0., 0.), (.1234567, 0., 0.))
    tiny = np.full(4, 1e-5)
    # Contact interval lasts ~2e-8 of a step, away from ordinary sample times.
    time = swept_contact(fast, tiny, obstacle, tiny)
    expected = (1000. + .1234567 - 2e-5) / 2000.
    assert time == pytest.approx(expected, abs=2e-10)
    fast[0, 0, 1] = 2e-5 + 1e-5
    assert np.isinf(swept_contact(fast, tiny, obstacle, tiny))
    # A close pass with overlapping broad AABBs must still be a certified miss.
    a = linear_trace((-1., 0., 0.), (1., 0., 0.))
    b = linear_trace((0., .3101, 0.), (0., .3101, 0.))
    assert np.isinf(swept_contact(a, EXTENTS, b, EXTENTS))


@pytest.mark.parametrize("yaw", [0., .71])
def test_occupancy_sweep_includes_rotation_cells_and_map_exterior(yaw):
    occupancy = np.ones((100, 100), dtype=np.bool_)
    occupancy[:, 50] = False  # A single 5 cm cell-thick wall.
    trace = linear_trace((2., 2.5, 0.), (3., 2.5, 0.))
    assert swept_map_contact(trace, EXTENTS, occupancy, .05) == pytest.approx(.21, abs=1e-6)
    origin = np.array([1., -2., yaw])
    world = trace.copy()
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    world[:, :, :2] = trace[:, :, :2] @ rotation.T
    world[:, 0, :2] += origin[:2]
    world[:, 0, 2] += yaw
    transformed = map_motion_trace(world, origin)
    assert swept_map_contact(transformed, EXTENTS, occupancy, .05) == pytest.approx(.21, abs=1e-6)
    # Rear/side of a rotating body contacts a cell invisible to forward LiDAR.
    occupancy[:] = True
    occupancy[50, 58] = False
    rotating = linear_trace((2.5, 2.5, np.pi / 2), (2.5, 2.5, 3 * np.pi / 2))
    long = np.array([.5, .5, .05, .05])
    assert np.isfinite(swept_map_contact(rotating, long, occupancy, .05))
    occupancy[:] = True
    boundary = linear_trace((.7, 2., 0.), (-1., 2., 0.))
    assert swept_map_contact(boundary, EXTENTS, occupancy, .05) == pytest.approx((.7 - .29) / 1.7, abs=1e-6)


def make_sim(tmp_path, params, poses, velocities, pixels=None, **kwargs):
    pixels = np.full((120, 120), 255, dtype=np.uint8) if pixels is None else pixels
    path = write_map(tmp_path, pixels, resolution=.05)
    sim = Simulator(params, num_agents=len(poses), seed=42, time_step=.05, num_beams=17, **kwargs)
    sim.set_map(str(path), ".png", map_data=load_map(MapLoader(), path))
    sim.reset(np.asarray(poses, dtype=np.float64))
    for agent, velocity in zip(sim.agents, velocities):
        if agent.nonlinear:
            agent._physics.reset(pose=agent.state[[0, 1, 4]], velocity=velocity,
                                 wheel_speed=np.clip(velocity[0] / .05, -400., 160.))
            agent._sync_physics_view()
        else:
            agent.state[3] = velocity[0]
    return sim


def actions_for(sim):
    return np.array([[0., np.clip(agent.body_velocity[0] / .05, -400., 160.)
                     if agent.nonlinear else agent.state[3]] for agent in sim.agents])


@pytest.mark.parametrize("crossing", [False, True])
def test_simulator_stops_head_on_and_crossing_motion_between_clear_endpoints(tmp_path, params, crossing):
    poses = [(2.5, 3., 0.), (3., 2.5, np.pi / 2)] if crossing else [(2.55, 3., 0.), (3.45, 3., np.pi)]
    sim = make_sim(tmp_path, params, poses, [(20., 0.), (20., 0.)])
    obs = sim.step(actions_for(sim))
    np.testing.assert_array_equal(obs["collisions"], [1., 1.])
    np.testing.assert_array_equal(sim.collision_idx, [1, 0])
    assert np.all((sim.collision_times > 0.) & (sim.collision_times < .05))
    for car in sim.agents:
        np.testing.assert_allclose(car.body_velocity, (0., 0.))
    # The unconstrained proposals passed each other; the resolved bodies did not.
    proposals = [trace_state(a._pose_trace, 1.) for a in sim.agents]
    assert not collision(sim.agents[0].geometry.world_vertices(proposals[0]),
                         sim.agents[1].geometry.world_vertices(proposals[1]))
    if not crossing:
        assert sim.agents[0].state[0] < sim.agents[1].state[0]
    np.testing.assert_allclose(sim.collision_times[0], sim.collision_times[1], atol=1e-10)


@pytest.mark.parametrize("status", [AgentRaceStatus.CRASHED, AgentRaceStatus.FINISHED])
def test_terminal_stationary_vehicles_remain_obstacles_until_clearance(tmp_path, params, status):
    sim = make_sim(tmp_path, params, [(2., 3., 0.), (2.8, 3., 0.)], [(20., 0.), (0., 0.)])
    config = TerminalAgentConfig(finish_clearance_steps=0, crash_clearance_steps=2, remove_after_clearance=False)
    terminal = TerminalVehicleController(("car_0", "car_1"), config)
    terminal.capture("car_1", status=status, terminal_step=0, action=np.zeros(2),
                     vehicle_state=sim.agents[1].physics_state)
    actions = actions_for(sim)
    terminal.apply(actions, agent_index={"car_0": 0, "car_1": 1}, simulator=sim, step=0)
    obs = sim.step(actions)
    np.testing.assert_array_equal(obs["collisions"], [1., 1.])
    assert sim.agents[0].state[0] < 2.8
    assert sim.collidable_mask[1]
    terminal.config = TerminalAgentConfig(finish_clearance_steps=0, crash_clearance_steps=0,
                                         remove_after_clearance=True)
    terminal.apply(actions, agent_index={"car_0": 0, "car_1": 1}, simulator=sim, step=1)
    assert not sim.collidable_mask[1]
    sim.agents[0].reset(np.array([2., 3., 0.]))
    if sim.agents[0].nonlinear:
        sim.agents[0]._physics.reset(pose=(2., 3., 0.), velocity=(20., 0.), wheel_speed=160.)
        sim.agents[0]._sync_physics_view()
    else:
        sim.agents[0].state[3] = 20.
    obs = sim.step(actions_for(sim))
    np.testing.assert_array_equal(obs["collisions"], [0., 0.])
    assert sim.agents[0].state[0] > 2.8


@pytest.mark.parametrize("predictive_response", [True, False])
def test_high_speed_wall_contact_stops_before_thin_wall(tmp_path, params, predictive_response):
    pixels = np.full((120, 120), 255, dtype=np.uint8)
    pixels[:, 50] = 0  # x in [2.5, 2.55]
    sim = make_sim(tmp_path, params, [(2., 3., 0.)], [(20., 0.)], pixels,
                   wall_collision_response=predictive_response)
    obs = sim.step(actions_for(sim))
    car = sim.agents[0]
    assert obs["collisions"][0] == 1. and sim.collision_idx[0] == ENV_COLLISION_IDX
    assert 0. < sim.collision_times[0] < .05
    assert 2. < car.state[0] <= 2.5 - car.geometry.front + 1e-7
    proposal = trace_state(car._pose_trace, 1.)
    assert proposal[0] - car.geometry.rear > 2.55
    np.testing.assert_allclose(car.body_velocity, [0., 0.])


def test_multiple_contacts_recheck_stopped_vehicle_and_use_deterministic_primary_index(tmp_path, params):
    # Car 0 hits parked car 1; car 2 then hits car 0's stopped body this step.
    sim = make_sim(tmp_path, params, [(2., 3., 0.), (2.8, 3., 0.), (1.1, 3., 0.)],
                   [(20., 0.), (0., 0.), (40., 0.)])
    obs = sim.step(actions_for(sim))
    np.testing.assert_array_equal(obs["collisions"], [1., 1., 1.])
    assert sim.collision_idx[0] == 1 and sim.collision_idx[2] == 0
    assert sim.collision_times[0] <= sim.collision_times[2]
    assert sim.agents[2].state[0] < sim.agents[0].state[0] < sim.agents[1].state[0]


def test_nonlinear_contact_uses_exact_actuator_time_and_consistent_full_state(tmp_path):
    params = yaml.safe_load((ROOT / "configs/vehicle/combined_slip.yaml").read_text())
    pixels = np.full((120, 120), 255, dtype=np.uint8)
    pixels[:, 50] = 0
    sim = make_sim(tmp_path, params, [(2., 3., 0.)], [(20., 0.)], pixels)
    car = sim.agents[0]
    start = car._physics._actuators.state.copy()
    reference = np.array([.2, 160.])
    constants = car._physics._actuators._time_constants
    rates = car._physics._actuators._rate_limits
    sim.step(reference.reshape(1, 2))
    from physics.dynamic_models import sample_wheel_actuators
    expected = sample_wheel_actuators(start, reference, sim.collision_times[0], constants, rates)
    state = car.physics_state
    assert state[6] == pytest.approx(expected[0], abs=1e-14)
    assert state[6] < sample_wheel_actuators(start, reference, .05, constants, rates)[0]
    np.testing.assert_array_equal(state[3:6], np.zeros(3))
    assert state[7] == 0.
    np.testing.assert_allclose(car.control_reference, [state[6], 0.])
    np.testing.assert_allclose(car.state[[0, 1, 4, 2]], state[[0, 1, 2, 6]])
    car.freeze_motion()
    frozen = state.copy()
    sim.step(np.array([[0., 0.]]))
    np.testing.assert_array_equal(car.physics_state, frozen)


def test_recording_dense_trace_preserves_unconstrained_dynamics_bit_for_bit():
    params = yaml.safe_load((ROOT / "configs/vehicle/combined_slip.yaml").read_text())
    config = {k: v for k, v in params.items() if k not in {"length", "width", "wheel_actuators"}}
    reference = CombinedSlipVehicle(config, params["wheel_actuators"])
    recorded = CombinedSlipVehicle(config, params["wheel_actuators"])
    reference.reset(velocity=(2., .1), yaw_rate=.2)
    recorded.reset(velocity=(2., .1), yaw_rate=.2)
    trace = np.zeros((50, 4, 6))
    for steer, wheel in ((.1, 50.), (-.1, 40.)):
        reference.command(steer, wheel)
        recorded.command(steer, wheel)
        expected = reference.advance(.05)
        actual = recorded.advance(.05, motion_trace=trace)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(trace_state(trace, 1.), actual[:6], atol=1e-15)


def test_simultaneous_four_car_contacts_choose_stable_indices(tmp_path, params):
    poses = [(2., 3., 0.), (4., 3., np.pi), (3., 2., np.pi / 2), (3., 4., -np.pi / 2)]
    sim = make_sim(tmp_path, params, poses, [(20., 0.)] * 4)
    obs = sim.step(actions_for(sim))
    np.testing.assert_array_equal(obs["collisions"], np.ones(4))
    np.testing.assert_array_equal(sim.collision_idx, [2, 2, 0, 0])
    np.testing.assert_allclose(sim.collision_times, np.full(4, sim.collision_times[0]), atol=1e-9)


def test_close_parallel_motion_remains_collision_free(tmp_path, params):
    separation = params["width"] + .001
    sim = make_sim(tmp_path, params, [(2., 3., 0.), (2., 3. + separation, 0.)], [(20., 0.)] * 2)
    obs = sim.step(actions_for(sim))
    np.testing.assert_array_equal(obs["collisions"], [0., 0.])
    np.testing.assert_array_equal(sim.collision_idx, [-1, -1])
    assert np.isinf(sim.collision_times).all()
    assert sim.agents[0].state[0] > 2.9


def test_nonlinear_rotating_vehicle_hits_wall_with_clear_endpoints(tmp_path):
    params = yaml.safe_load((ROOT / "configs/vehicle/combined_slip.yaml").read_text())
    pixels = np.full((120, 120), 255, dtype=np.uint8)
    pixels[:, 50] = 0
    sim = make_sim(tmp_path, params, [(2.25, 3., np.pi / 2)], [(0., 0.)], pixels)
    car = sim.agents[0]
    car._physics.reset(pose=(2.25, 3., np.pi / 2), yaw_rate=2 * np.pi / .05, wheel_speed=0.)
    car._sync_physics_view()
    assert car.geometry.world_vertices(car.state[[0, 1, 4]])[:, 0].max() < 2.5
    obs = sim.step(np.zeros((1, 2)))
    assert obs["collisions"][0] == 1.
    assert sim.collision_idx[0] == ENV_COLLISION_IDX
    assert 0. < sim.collision_times[0] < .05
    assert car.geometry.world_vertices(car.state[[0, 1, 4]])[:, 0].max() <= 2.5 + 1e-7
    assert car.geometry.world_vertices(trace_state(car._pose_trace, 1.))[:, 0].max() < 2.5


def test_public_collision_reason_and_clearance_preserve_stopped_state(tmp_path, params):
    from env.RaceEnv import RaceEnv

    path = write_map(tmp_path, np.full((120, 120), 255, dtype=np.uint8), resolution=.05)
    env = RaceEnv(map=path.name, map_dir=str(tmp_path), vehicle_params=params,
                  n_agents=2, timestep=.05, lidar_beams=17, render_mode=None,
                  terminal_agents={"remove_after_clearance": True, "crash_clearance_steps": 0})
    try:
        poses = np.array([[2.55, 3., 0.], [3.45, 3., np.pi]])
        env.reset(seed=42, options={"poses": poses})
        for car, pose in zip(env.sim.agents, poses):
            if car.nonlinear:
                car._physics.reset(pose=pose, velocity=(20., 0.), wheel_speed=160.)
                car._sync_physics_view()
            else:
                car.state[3] = 20.
        if not env.sim.agents[0].nonlinear:
            # Curriculum speed locking must not restart a car after CCD stops it.
            env._locked_velocities = {aid: 20. for aid in env.possible_agents}
            env._lock_speed_steps = 2
        action = 160. if env.sim.agents[0].nonlinear else 20.
        _, _, terminated, _, infos = env.step({aid: np.array([0., action]) for aid in env.agents})
        assert all(terminated.values())
        for aid in env.possible_agents:
            assert infos[aid]["collision_event"]
            assert infos[aid]["terminal_reason"] == TerminalReason.COLLISION.value
            assert env.lifecycle.records[aid].status is AgentRaceStatus.CRASHED
        assert not env.sim.collidable_mask.any()
        for car in env.sim.agents:
            np.testing.assert_allclose(car.body_velocity, (0., 0.))
    finally:
        env.close()


def finish_env(tmp_path, params, *, finish_x=2.05, wall=True, **settings):
    from env.RaceEnv import RaceEnv
    pixels = np.full((120, 120), 255, dtype=np.uint8)
    if wall:
        pixels[:, 48] = 0  # x=2.4, reached by either configured footprint at 5 m/s.
    path = write_map(tmp_path, pixels, resolution=.05)
    env = RaceEnv(map=path.name, map_dir=str(tmp_path), vehicle_params=params,
                  n_agents=1, timestep=.05, lidar_beams=17, render_mode=None,
                  finish_line={"version": 1, "start": [finish_x, 1.], "end": [finish_x, 5.],
                               "direction": [1., 0.], "hysteresis": .01, "padding": 0.,
                               "trigger_speed": .5}, **settings)
    env.reset(seed=42, options={"poses": np.array([[2., 3., 0.]])})
    env.sim.set_agent_speed(0, 5.)
    env._update_state(env.sim.current_observation())
    return env


@pytest.mark.parametrize("finish_x,reason", [(2.05, TerminalReason.RACE_COMPLETE),
                                             (2.3, TerminalReason.COLLISION)])
def test_finish_and_contact_follow_resolved_motion_chronology(tmp_path, params, finish_x, reason):
    env = finish_env(tmp_path, params, finish_x=finish_x)
    try:
        action = 100. if env.sim.agents[0].nonlinear else 5.
        obs, _, terminated, _, infos = env.step({"car_0": np.array([0., action])})
        assert terminated["car_0"]
        assert infos["car_0"]["collision_event"]
        assert infos["car_0"]["terminal_reason"] == reason.value
        assert bool(obs["car_0"]["lap"][0]) == (reason is TerminalReason.RACE_COMPLETE)
    finally:
        env.close()


def test_zero_clearance_finish_updates_all_public_state_buffers(tmp_path, params):
    env = finish_env(tmp_path, params, finish_x=2.1, wall=False,
                     terminal_agents={"remove_after_clearance": True, "finish_clearance_steps": 0})
    try:
        env._agent_sensor_spec["car_0"] += ("acceleration",)
        bounds = env.observation_space("car_0").spaces["pose"]
        env._build_observation_spaces(bounds.low[0], bounds.high[0], bounds.low[1], bounds.high[1])
        action = 100. if env.sim.agents[0].nonlinear else 5.
        obs, _, terminated, _, infos = env.step({"car_0": np.array([0., action])})
        assert terminated["car_0"] and infos["car_0"]["race_completed"]
        car = env.sim.agents[0]
        assert car.body_velocity == (0., 0.)
        np.testing.assert_array_equal(obs["car_0"]["velocity"], [0., 0.])
        np.testing.assert_allclose(obs["car_0"]["acceleration"], [-100., 0.], atol=1e-5)
        np.testing.assert_array_equal(env.sim.current_observation()["linear_vels_x"], [0.])
        np.testing.assert_array_equal(env.get_agent_state("car_0").velocity, [0., 0.])
        assert obs["car_0"]["lap"][0] == 1.
        assert env.get_global_state().vector[3] == 0.
    finally:
        env.close()


def test_partial_reset_rebuilds_geometry_and_velocity_from_authoritative_state(tmp_path, params):
    sim = make_sim(tmp_path, params, [(1., 1., 0.), (4., 4., 0.)], [(0., 0.), (0., 0.)])
    car = sim.agents[0]
    # A public vehicle reset/state edit need not update the simulator's old buffers.
    car.reset(np.array([3., 3., .4]))
    car.set_longitudinal_speed(1.)
    sim.reset(np.array([[1., 1., 0.], [3.05, 3., .4]]), agent_indices=[1])
    np.testing.assert_allclose(sim.agent_poses[0], [3., 3., .4], atol=1e-7)
    np.testing.assert_allclose(sim._verts_buffer_f64[0], car.geometry.world_vertices(np.array([3., 3., .4])))
    assert sim.current_observation()["linear_vels_x"][0] == 1.
    np.testing.assert_array_equal(sim.collisions, [1., 1.])
    np.testing.assert_array_equal(sim.collision_times, [0., 0.])


def test_invalid_actions_and_states_are_rejected_before_any_car_advances(tmp_path, params):
    sim = make_sim(tmp_path, params, [(1., 1., 0.), (4., 4., 0.)], [(1., 0.), (1., 0.)])
    before = [a.physics_state.copy() for a in sim.agents]
    actions = actions_for(sim)
    actions[1, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        sim.step(actions)
    for car, state in zip(sim.agents, before):
        np.testing.assert_array_equal(car.physics_state, state)
    car = sim.agents[1]
    if car.nonlinear:
        car._physics._chassis[0] = np.nan
    else:
        car.state[0] = np.nan
    with pytest.raises(ValueError, match="states must be finite"):
        sim.step(actions_for(sim))
    np.testing.assert_array_equal(sim.agents[0].physics_state, before[0])


def test_physical_contact_with_termination_disabled_keeps_active_state_consistent(tmp_path, params):
    env = finish_env(tmp_path, params, finish_x=2.3, terminate_on_collision=False)
    try:
        action = 100. if env.sim.agents[0].nonlinear else 5.
        obs, _, terms, truncs, infos = env.step({"car_0": np.array([0., action])})
        assert not terms["car_0"] and not truncs["car_0"]
        assert infos["car_0"]["collision_event"] and infos["car_0"]["terminal_reason"] is None
        assert env.lifecycle.records["car_0"].is_active
        np.testing.assert_array_equal(obs["car_0"]["velocity"], [0., 0.])
        # Clearing contact clears the per-step flag; the historical latch remains.
        env.sim.agents[0].reset(np.array([1., 3., 0.]))
        obs, _, _, _, infos = env.step({"car_0": np.zeros(2)})
        assert not infos["car_0"]["collision_event"]
        assert env.get_agent_state("car_0").collision
        assert np.isinf(env.sim.collision_times[0]) and env.sim.collision_idx[0] == -1
    finally:
        env.close()


def test_invalid_trace_and_reset_inputs_leave_committed_states_unchanged(tmp_path, params):
    sim = make_sim(tmp_path, params, [(1., 1., 0.), (4., 4., 0.)], [(1., 0.), (1., 0.)])
    before = [car.physics_state.copy() for car in sim.agents]
    with pytest.raises(ValueError, match="finite"):
        sim.reset(np.array([[2., 2., 0.], [3., 3., 0.]]), velocities=np.array([1., np.inf]))
    for car, state in zip(sim.agents, before):
        np.testing.assert_array_equal(car.physics_state, state)
    car = sim.agents[0]
    with pytest.raises(ValueError, match="finite"):
        car.reset(np.array([np.nan, 0., 0.]))
    np.testing.assert_array_equal(car.physics_state, before[0])
    if car.nonlinear:
        with pytest.raises(ValueError, match="motion_trace"):
            car._physics.advance(.05, motion_trace=np.zeros((1, 4, 6)))
        np.testing.assert_array_equal(car.physics_state, before[0])


def test_initial_overlap_stops_attempted_motion_at_zero_time(tmp_path, params):
    sim = make_sim(tmp_path, params, [(3., 3., 0.), (3., 3., 0.)], [(1., 0.), (0., 0.)])
    before = np.stack([car.state[[0, 1, 4]] for car in sim.agents])
    obs = sim.step(actions_for(sim))
    np.testing.assert_array_equal(obs["collisions"], [1., 1.])
    np.testing.assert_array_equal(sim.collision_times, [0., 0.])
    np.testing.assert_allclose(sim.agent_poses, before)
    for car in sim.agents:
        np.testing.assert_allclose(car.body_velocity, [0., 0.])


@pytest.mark.parametrize("predictive_response", [False, True])
def test_predictive_ttc_event_is_distinct_from_physical_contact(tmp_path, params, predictive_response):
    from test_vehicle_geometry import WallScanner
    pixels = np.full((120, 120), 255, dtype=np.uint8)
    pixels[:, 48] = 0
    start_x = 2.4 - params["length"] / 2. - .008 - 2. * .05
    sim = make_sim(tmp_path, params, [(start_x, 3., 0.)], [(2., 0.)], pixels,
                   wall_collision_response=predictive_response)
    # Analytic ranges isolate the established 5 ms TTC from grid quantization
    # and noise; physical sweeps still use the real F4 occupancy mask.
    analytic = WallScanner(fov=4.7)
    analytic.normal, analytic.wall_point = np.array([1., 0.]), np.array([2.4, 0.])
    sim.scan_simulator.scan = analytic.scan
    obs = sim.step(actions_for(sim))
    car = sim.agents[0]
    assert obs["collisions"][0] == 1. and sim.collision_idx[0] == ENV_COLLISION_IDX
    assert sim.collision_times[0] == .05
    assert 2.4 - (car.state[0] + car.geometry.front) == pytest.approx(.008)
    assert car.body_velocity[0] == pytest.approx(0. if predictive_response else 2.)


def test_environment_rejects_nonfinite_physical_commands_before_recording_them(tmp_path, params):
    env = finish_env(tmp_path, params, wall=False)
    try:
        before = env.sim.agents[0].physics_state.copy()
        commands = env._last_control_commands.copy()
        with pytest.raises(ValueError, match="finite"):
            env.step({"car_0": np.array([np.nan, 1.])})
        np.testing.assert_array_equal(env.sim.agents[0].physics_state, before)
        np.testing.assert_array_equal(env._last_control_commands, commands)
        assert env.current_time == 0. and env._episode_step_count == 0
    finally:
        env.close()
