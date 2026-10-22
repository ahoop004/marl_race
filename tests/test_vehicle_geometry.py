"""Targeted F0–F2 checks; analytic walls avoid maps, noise and long rollouts."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from physics.collision_models import VehicleGeometry, collision, get_vertices
from physics.laser_models import check_ttc_jit, get_range
from physics.simulaton import ENV_COLLISION_IDX, Simulator
from physics.vehicle import RaceCar


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["default", "combined_slip"], ids=["legacy_st", "combined_slip_st"])
def params(request):
    return yaml.safe_load((ROOT / "configs/vehicle" / f"{request.param}.yaml").read_text())


class WallScanner:
    """Exact ranges to a plane, using the scanner's existing scan API."""

    def __init__(self, beams=17, fov=2 * np.pi):
        self.angles = np.linspace(-fov / 2, fov / 2, beams)
        self.wall_point = None
        self.normal = None
        self.pose = None

    def get_increment(self):
        return self.angles[1] - self.angles[0]

    def scan(self, pose, rng):
        self.pose = pose.copy()
        result = np.full(len(self.angles), 30.0)
        if self.wall_point is not None:
            angles = self.angles + pose[2]
            directions = np.column_stack((np.cos(angles), np.sin(angles)))
            denom = directions @ self.normal
            valid = denom > 1e-12
            result[valid] = ((self.wall_point - pose[:2]) @ self.normal) / denom[valid]
        return result


def make_car(params, offset=0.0):
    scanner = WallScanner()
    car = RaceCar(params, seed=7, num_beams=17, fov=2 * np.pi,
                  lidar_dist=offset, scan_simulator=scanner, integrator="RK4")
    return car, scanner


def rotation(heading):
    return np.array([[np.cos(heading), -np.sin(heading)],
                     [np.sin(heading), np.cos(heading)]])


@pytest.mark.parametrize("heading", [0.0, np.pi / 2, -np.pi / 2, 0.73])
def test_footprint_and_reference_frame(params, heading):
    geometry = VehicleGeometry.from_params(params, lidar_dist=0.06)
    length, width = params["length"], params["width"]
    local = np.array([[-length / 2, width / 2], [-length / 2, -width / 2],
                      [length / 2, -width / 2], [length / 2, width / 2]])
    pose = np.array([2.3, -1.7, heading])
    expected = local @ rotation(heading).T + pose[:2]
    np.testing.assert_allclose(geometry.footprint, local, atol=1e-15)
    np.testing.assert_allclose(geometry.world_vertices(pose), expected, atol=1e-15)
    np.testing.assert_array_equal(geometry.world_vertices(pose), get_vertices(pose, length, width))
    np.testing.assert_array_equal(geometry.axle_locations,
                                  [[params["lf"], 0.0], [-params["lr"], 0.0]])
    np.testing.assert_allclose(geometry.lidar_pose(pose),
                               [*(pose[:2] + rotation(heading) @ [0.06, 0.0]), heading])


@pytest.mark.parametrize("offset", [-0.07, 0.0, 0.09])
def test_front_rear_side_and_corner_ranges(params, offset):
    geometry = VehicleGeometry.from_params(params, offset)
    angles = np.array([0.0, np.pi, np.pi / 2, -np.pi / 2, np.pi / 4])
    entry, distances = geometry.ray_intersections(angles)
    front = params["length"] / 2 - offset
    rear = params["length"] / 2 + offset
    side = params["width"] / 2
    np.testing.assert_array_equal(entry, np.zeros(5))
    np.testing.assert_allclose(distances, [front, rear, side, side, min(front, side) * np.sqrt(2)])


def test_asymmetric_footprint_and_lateral_mount():
    geometry = VehicleGeometry(.4, .2, .18, .12, .15875, .17145,
                               lidar_x=.03, lidar_y=-.02)
    np.testing.assert_allclose(geometry.footprint,
                               [[-.2, .18], [-.2, -.12], [.4, -.12], [.4, .18]])
    _, distances = geometry.ray_intersections(np.array([0.0, np.pi, np.pi / 2, -np.pi / 2]))
    np.testing.assert_allclose(distances, [.37, .23, .20, .10])
    pose = np.array([1.0, 2.0, np.pi / 2])
    np.testing.assert_allclose(geometry.lidar_pose(pose), [1.02, 2.03, np.pi / 2])


@pytest.mark.parametrize("heading", [0.0, np.pi / 2, -1.19])
@pytest.mark.parametrize("offset", [-.07, .09])
def test_wall_ranges_agree_with_world_gjk_edges(params, heading, offset):
    geometry = VehicleGeometry.from_params(params, offset)
    pose = np.array([1.2, -2.1, heading])
    lidar_pose = geometry.lidar_pose(pose)
    vertices = geometry.world_vertices(pose)
    angles = np.linspace(-np.pi, np.pi, 65)
    _, ranges = geometry.ray_intersections(angles)
    expected = [min(get_range(lidar_pose, angle + heading, vertices[i], vertices[(i + 1) % 4])
                    for i in range(4)) for angle in angles]
    np.testing.assert_allclose(ranges, expected, atol=1e-14)


@pytest.mark.parametrize("heading", [0.0, np.pi / 2, .73])
@pytest.mark.parametrize("side", ["front", "rear", "left", "right"])
@pytest.mark.parametrize("gap", [-.01, .003, .01])
def test_stationary_wall_contact_and_gjk_agreement(params, heading, side, gap):
    car, scanner = make_car(params, offset=.07)
    pose = np.array([1.2, -2.1, heading])
    car.reset(pose)
    normals = {"front": [1., 0.], "rear": [-1., 0.], "left": [0., 1.], "right": [0., -1.]}
    normal = rotation(heading) @ normals[side]
    extent = getattr(car.geometry, side)
    scanner.normal = normal
    scanner.wall_point = pose[:2] + normal * (extent + gap)
    car.compute_scan()
    assert car.in_collision == (gap < .005)
    # A thin rectangle covers the wall half-plane locally. GJK detects actual
    # penetration; iTTC additionally preserves its existing 5 mm contact margin.
    wall_center = scanner.wall_point + .01 * normal
    wall_heading = np.arctan2(normal[1], normal[0])
    wall = get_vertices(np.array([*wall_center, wall_heading]), .02, 3.0)
    assert bool(collision(car.geometry.world_vertices(pose), wall)) == (gap < 0.0)
    np.testing.assert_allclose(scanner.pose[:2], pose[:2] + rotation(heading) @ [.07, 0.])


def test_external_mount_skips_missed_rays_and_obstacles_before_body():
    geometry = VehicleGeometry(.29, .29, .155, .155, .15875, .17145, lidar_x=.4)
    angles = np.array([0.0, np.pi, np.pi / 2])
    entry, exits = geometry.ray_intersections(angles)
    assert np.isnan(exits[[0, 2]]).all()
    np.testing.assert_allclose([entry[1], exits[1]], [.11, .69])
    # An obstacle before the footprint, or in a ray that misses it, is not contact.
    assert not check_ttc_jit(np.array([.01, .05, .01]), 0., angles, np.cos(angles), exits, .005, entry)
    assert check_ttc_jit(np.array([30., .68, 30.]), 0., angles, np.cos(angles), exits, .005, entry)


@pytest.mark.parametrize("angle,velocity", [(0., (2., 0.)), (np.pi, (-2., 0.)),
                                           (np.pi / 2, (0., 2.))])
def test_ttc_threshold_and_velocity_projection(params, angle, velocity):
    car, _ = make_car(params, offset=.07)
    if not car.nonlinear and velocity[1]:
        # Legacy TTC keeps its historical longitudinal-only velocity contract.
        pytest.skip("Legacy TTC projects longitudinal velocity only")
    if car.nonlinear:
        car._physics.reset(velocity=velocity)
        car._sync_physics_view()
    else:
        car.state[3] = velocity[0]
    beam = np.argmin(np.abs(car.scan_angles - angle))
    scan = np.full(car.num_beams, 30.)
    for clearance, expected in [(.009, True), (.011, False)]:
        scan[beam] = car.side_distances[beam] + clearance
        car.check_ttc(scan)
        assert car.in_collision == expected
    # Velocity away from this beam must not cause anticipatory contact.
    if car.nonlinear:
        car._physics.reset(velocity=tuple(-v for v in velocity))
        car._sync_physics_view()
    else:
        car.state[3] *= -1
    scan[beam] = car.side_distances[beam] + .009
    car.check_ttc(scan)
    assert not car.in_collision
    assert car.ttc_thresh == .005
    # The original six-argument helper API remains callable.
    assert not check_ttc_jit(np.full(3, 30.), 0., np.zeros(3), np.ones(3), np.ones(3), .005)


def test_lidar_mount_shared_by_scan_update_and_opponent_rays(params, monkeypatch):
    car, scanner = make_car(params, offset=.07)
    pose = np.array([1., 2., np.pi / 2])
    car.reset(pose)
    car.update_pose(0., 0.)
    np.testing.assert_allclose(scanner.pose, [1., 2.07, np.pi / 2])
    recorded = []
    monkeypatch.setattr("physics.vehicle.ray_cast", lambda pose, *args: recorded.append(pose.copy()))
    car.ray_cast_agents(np.zeros((2, 4, 2), dtype=np.float32), 0)
    np.testing.assert_allclose(recorded[0], scanner.pose)


def test_stationary_collision_event_interface(params):
    sim = Simulator(params, num_agents=1, seed=7, num_beams=17, lidar_dist=.07)
    scanner = WallScanner(fov=4.7)
    scanner.normal = np.array([1., 0.])
    scanner.wall_point = np.array([params["length"] / 2 - .01, 0.])
    sim.agents[0].scan_simulator = scanner
    sim.reset(np.zeros((1, 3)))
    obs = sim.step(np.zeros((1, 2)))
    np.testing.assert_array_equal(obs["collisions"], [1.])
    np.testing.assert_array_equal(sim.collision_idx, [ENV_COLLISION_IDX])
    np.testing.assert_array_equal(obs["linear_vels_x"], [0.])
    # Events remain per-step, clearing after contact is removed.
    scanner.wall_point = None
    obs = sim.step(np.zeros((1, 2)))
    np.testing.assert_array_equal(obs["collisions"], [0.])
    np.testing.assert_array_equal(sim.collision_idx, [-1])


def test_parameter_updates_preserve_collision_geometry_contract(params):
    sim = Simulator(params, num_agents=2, seed=7, num_beams=17, lidar_dist=.04)
    updated = {**params, "length": .6, "width": .3}
    if params.get("model") == "combined_slip_st":
        with pytest.raises(ValueError, match="Recreate the environment"):
            sim.update_params(updated, agent_idx=1)
        assert sim.agents[1].geometry.front == params["length"] / 2
        return
    sim.update_params(updated, agent_idx=1)
    for car in sim.agents:
        car.scan_simulator = WallScanner(fov=4.7)
    poses = np.array([[0., 0., 0.], [.4, 0., 0.]])
    sim.reset(poses)
    # Updated rear (.3) overlaps the other car's front (.16).
    np.testing.assert_array_equal(sim.collisions, [1., 1.])
    np.testing.assert_allclose(sim._verts_buffer_f64[1, :, 0], [.1, .1, .7, .7])
    geometry = sim.agents[1].geometry
    assert geometry.front == .3 and geometry.left == .15
    _, expected = geometry.ray_intersections(sim.agents[1].scan_angles)
    np.testing.assert_allclose(sim.agents[1].side_distances, expected)
    sim.step(np.zeros((2, 2)))
    np.testing.assert_array_equal(sim.collisions, [1., 1.])
    # A partial update keeps the established body size and the legacy dynamics
    # fallback behavior; it must not resurrect stale global GJK dimensions.
    sim.update_params({"mu": .8}, agent_idx=1)
    assert sim.agents[1].geometry.front == .3
    assert sim.agents[1].geometry.left == .15
    assert sim.agents[1]._mu == .8
    assert sim.agents[1].geometry.lf == sim.agents[1]._lf
