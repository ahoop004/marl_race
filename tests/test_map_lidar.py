"""Focused F3/F4/F6 tests on small, deterministic occupancy maps."""
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from env.map_schedule import MapScheduler
from physics.laser_models import ScanSimulator2D, distance_transform, trace_ray, xy_2_rc
from physics.simulaton import ENV_COLLISION_IDX, Simulator
from utils.map_loader import MapLoader
from wrappers.observations.ego import LidarComponent


ROOT = Path(__file__).resolve().parents[1]


def write_map(tmp_path, pixels, name="map", resolution=.1, origin=(0., 0., 0.)):
    # Deliberately use an image name that differs from the YAML stem.
    image = tmp_path / f"{name}_occupancy.png"
    Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(image)
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump({"image": image.name, "resolution": resolution,
                                    "origin": list(origin)}))
    return path


def load_map(loader, path, **settings):
    return loader.load({"map_dir": str(path.parent), "map_yaml": path.name,
                        "centerline_autoload": False, "walls_autoload": False, **settings})


def rotate(vector, yaw):
    x, y = vector
    return np.array([x * np.cos(yaw) - y * np.sin(yaw),
                     x * np.sin(yaw) + y * np.cos(yaw)])


@pytest.mark.parametrize("yaw", [0., np.pi / 2, .73])
@pytest.mark.parametrize("point", [(-.01, 1.), (3.01, 1.), (1., -.01), (1., 2.01),
                                   (-.01, -.01), (-.01, 2.01), (3.01, -.01), (3.01, 2.01)])
def test_outside_lookups_and_rays_are_occupied(yaw, point):
    origin = np.array([1.3, -2.1])
    world = rotate(point, yaw) + origin
    dt = np.full((4, 6), 99.)  # Last cell must never be used as an outside lookup.
    args = (*origin, np.cos(yaw), np.sin(yaw), 4, 6, .5)
    assert xy_2_rc(*world, *args) == (-1, -1)
    assert distance_transform(*world, *args, dt) == 0.
    # Outside is occupied even when a ray points back into the map.
    for angle in (0., 1.2, np.pi, -1.2):
        result = trace_ray(*world, 0., np.array([np.sin(angle)]),
                           np.array([np.cos(angle)]), .0001, *args, dt, 30.)
        assert result == 0.


@pytest.mark.parametrize("point", [(np.nan, 0.), (0., np.inf), (-np.inf, 0.)])
def test_nonfinite_coordinates_never_index_distance_transform(point):
    args = (0., 0., 1., 0., 4, 6, .5)
    assert xy_2_rc(*point, *args) == (-1, -1)
    assert distance_transform(*point, *args, np.full((4, 6), 99.)) == 0.


@pytest.mark.parametrize("yaw", [0., np.pi / 2, -.83])
@pytest.mark.parametrize("target", [(0., 1.1), (3., 1.1), (.7, 0.), (.7, 2.),
                                    (0., 0.), (0., 2.), (3., 0.), (3., 2.)])
def test_rays_stop_at_each_edge_and_corner(yaw, target):
    origin = np.array([1.3, -2.1])
    local = np.array([.7, 1.1])
    delta = np.array(target) - local
    expected = np.linalg.norm(delta)
    direction = rotate(delta / expected, yaw)
    world = rotate(local, yaw) + origin
    result = trace_ray(*world, 0., np.array([direction[1]]), np.array([direction[0]]),
                       .0001, *origin, np.cos(yaw), np.sin(yaw), 4, 6, .5,
                       np.full((4, 6), 99.), 30.)
    assert result == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("point,direction", [((0., 1.), (-1., 0.)), ((3., 1.), (1., 0.)),
                                             ((1., 0.), (0., -1.)), ((1., 2.), (0., 1.))])
def test_boundary_origin_outward_ray_is_zero(point, direction):
    assert trace_ray(*point, 0., np.array([direction[1]]), np.array([direction[0]]),
                     .0001, 0., 0., 1., 0., 4, 6, .5, np.full((4, 6), 99.), 30.) == 0.


@pytest.mark.parametrize("threshold,inverted", [(None, False), (127, False), (128, False),
                                                 (200, False), (128, True)])
@pytest.mark.parametrize("binary", [True, False])
def test_loader_and_scanner_share_threshold_inversion_and_vertical_flip(tmp_path, threshold, inverted, binary):
    pixels = np.array([[0, 127, 128, 199], [200, 201, 254, 255], [0, 255, 128, 200]], dtype=np.uint8)
    if binary:
        pixels = np.where(pixels >= 128, 255, 0).astype(np.uint8)
    path = write_map(tmp_path, pixels)
    data = load_map(MapLoader(), path, track_threshold=threshold, track_inverted=inverted)
    actual_threshold = threshold if threshold is not None else (200 if pixels.mean() > 127 else 127)
    expected = pixels >= actual_threshold
    if inverted:
        expected = ~expected
    scanner = ScanSimulator2D(5, 1.)
    scanner.set_map(str(path), ".png", map_data=data)
    np.testing.assert_array_equal(data.track_mask, expected)
    np.testing.assert_array_equal(scanner.map_img, expected[::-1])
    np.testing.assert_array_equal(scanner.dt == 0., ~expected[::-1])
    for row in range(3):
        for col in range(4):
            distance = distance_transform((col + .5) * .1, (row + .5) * .1,
                                          0., 0., 1., 0., 3, 4, .1, scanner.dt)
            assert (distance > 0.) == bool(expected[::-1][row, col])


def test_map_switches_reuse_only_matching_occupancy_cache(tmp_path, monkeypatch):
    from physics import laser_models

    pixels = np.array([[0, 255, 255], [0, 150, 255], [0, 255, 255]])
    path = write_map(tmp_path, pixels)
    loader = MapLoader()
    scanner = ScanSimulator2D(5, 1.)
    other = ScanSimulator2D(5, 1.)
    original_get_dt = laser_models.get_dt
    calls = []

    def counted_dt(*args):
        calls.append(1)
        return original_get_dt(*args)

    monkeypatch.setattr(laser_models, "get_dt", counted_dt)
    first = load_map(loader, path, track_threshold=128)
    scanner.set_map(str(path), ".png", map_data=first)
    first_dt = scanner.dt
    scanner.set_map(str(path), ".png", map_data=load_map(loader, path, track_threshold=128))
    assert scanner.dt is first_dt and len(calls) == 1
    high = load_map(loader, path, track_threshold=200)
    scanner.set_map(str(path), ".png", map_data=high)
    assert scanner.dt is not first_dt and scanner.dt[1, 1] == 0.
    # A concurrent evaluation scanner owns its selected map independently.
    other.set_map(str(path), ".png", map_data=first)
    assert other.dt is first_dt and other.dt[1, 1] > 0.
    inverted = load_map(loader, path, track_threshold=128, track_inverted=True)
    scanner.set_map(str(path), ".png", map_data=inverted)
    np.testing.assert_array_equal(scanner.map_img, ~first.track_mask[::-1])
    second_path = write_map(tmp_path, np.zeros((2, 4)), name="second")
    scanner.set_map(str(second_path), ".png", map_data=load_map(loader, second_path))
    assert scanner.dt.shape == (2, 4)
    scanner.set_map(str(path), ".png", map_data=first)
    assert scanner.dt is first_dt and len(calls) == 4
    # File changes at the same path invalidate the loader and scanner caches.
    Image.fromarray(np.zeros_like(pixels, dtype=np.uint8)).save(first.image_path)
    changed = load_map(loader, path, track_threshold=128)
    scanner.set_map(str(path), ".png", map_data=changed)
    assert scanner.dt is not first_dt and np.all(scanner.dt == 0.)
    assert other.dt is first_dt
    metadata = yaml.safe_load(path.read_text())
    metadata.update(resolution=.2, origin=[2., 3., .7])
    path.write_text(yaml.safe_dump(metadata))
    changed = load_map(loader, path, track_threshold=128)
    scanner.set_map(str(path), ".png", map_data=changed)
    assert scanner.map_resolution == .2
    np.testing.assert_allclose(scanner.origin, [2., 3., .7])


def test_standalone_scanner_reloads_same_path_and_resolves_yaml_image(tmp_path):
    path = write_map(tmp_path, np.full((4, 5), 255))
    scanner = ScanSimulator2D(5, 1.)
    scanner.set_map(str(path), ".png")
    assert scanner.map_img.all()
    Image.fromarray(np.zeros((4, 5), dtype=np.uint8)).save(path.with_name("map_occupancy.png"))
    scanner.set_map(str(path), ".png")
    assert not scanner.map_img.any()
    np.testing.assert_array_equal(scanner.scan(np.array([.2, .2, 0.]), None), np.zeros(5))


def test_scheduler_hot_switch_preserves_occupancy_settings(tmp_path):
    path = write_map(tmp_path, [[0, 150, 255], [0, 150, 255]])
    scheduler = MapScheduler({"map_dir": str(tmp_path), "track_threshold": 200,
                              "track_inverted": True}, rng=np.random.default_rng(7))
    data = scheduler.load_from_path(str(path), ".png")
    np.testing.assert_array_equal(data.track_mask, [[True, True, False], [True, True, False]])


def test_seeded_noise_is_applied_then_clipped_to_physical_range(tmp_path):
    path = write_map(tmp_path, np.full((20, 20), 255))
    scanner = ScanSimulator2D(17, 4.7, max_range=.3)
    scanner.set_map(str(path), ".png")
    pose = np.array([1., 1., 0.])
    clean = scanner.scan(pose, None)
    expected = np.clip(clean + np.random.default_rng(42).normal(0., .01, 17), 0., .3)
    actual = scanner.scan(pose, np.random.default_rng(42))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, scanner.scan(pose, np.random.default_rng(42)))
    assert np.isfinite(actual).all() and actual.min() >= 0. and actual.max() <= .3
    outside = np.array([-1., -1., 0.])
    noisy_zero = scanner.scan(outside, np.random.default_rng(42))
    np.testing.assert_array_equal(noisy_zero, np.clip(np.random.default_rng(42).normal(0., .01, 17), 0., .3))
    assert np.any(noisy_zero == 0.)
    with pytest.raises(ValueError, match="finite"):
        scanner.scan(np.array([np.nan, 0., 0.]), None)


@pytest.mark.parametrize("normalized", [False, True])
def test_policy_range_normalization_and_invalid_readings(normalized):
    raw = np.array([-1., 0., 1., 10., 30., np.nan, np.inf, -np.inf], dtype=np.float32)
    original = raw.copy()
    component = LidarComponent(len(raw), lidar_range=10., normalize=normalized)
    result = component.compute({"lidar": raw}, {})
    expected = [0., 0., .1, 1., 1., 0., 0., 0.] if normalized else [0., 0., 1., 10., 30., 0., 0., 0.]
    np.testing.assert_allclose(result, expected)
    np.testing.assert_array_equal(raw, original)
    assert np.isfinite(result).all()


@pytest.mark.parametrize("model", ["default", "combined_slip"])
def test_outside_map_stationary_contact_and_shared_scanner(tmp_path, model):
    params = yaml.safe_load((ROOT / "configs/vehicle" / f"{model}.yaml").read_text())
    path = write_map(tmp_path, np.full((20, 20), 255))
    loader = MapLoader()
    sim = Simulator(params, num_agents=2, seed=42, num_beams=17, lidar_dist=.07)
    data = load_map(loader, path)
    sim.set_map(str(path), ".png", map_data=data)
    assert sim.agents[0].scan_simulator is sim.agents[1].scan_simulator
    poses = np.array([[-1., -1., 0.], [1., 1., 0.]])
    sim.reset(poses)
    obs = sim.step(np.zeros((2, 2)))
    assert obs["collisions"][0] == 1.
    assert sim.collision_idx[0] == ENV_COLLISION_IDX
    assert np.isfinite(obs["scans"]).all()
    assert obs["scans"].min() >= 0. and obs["scans"].max() <= 30.
    initial_scans = sim.reset(poses)["scans"].copy()
    np.testing.assert_array_equal(sim.reset(poses)["scans"], initial_scans)
    changed = load_map(loader, path, track_inverted=True)
    sim.set_map(str(path), ".png", map_data=changed)
    assert not sim.scan_simulator.map_img.any()


@pytest.mark.parametrize("heading", [0., 1.13])
@pytest.mark.parametrize("angle,velocity", [(0., (2., 0.)), (np.pi, (-2., 0.)),
                                           (np.pi / 2, (0., 2.)), (-np.pi / 2, (0., -2.)),
                                           (np.pi / 4, (1., 1.)), (-3 * np.pi / 4, (-1., -1.))])
def test_nonlinear_ttc_projects_both_body_velocities_at_any_heading(heading, angle, velocity):
    from test_vehicle_geometry import make_car

    params = yaml.safe_load((ROOT / "configs/vehicle/combined_slip.yaml").read_text())
    car, _ = make_car(params, offset=.07)
    car._physics.reset(pose=(1., 2., heading), velocity=velocity)
    car._sync_physics_view()
    beam = int(np.argmin(np.abs(car.scan_angles - angle)))
    speed = velocity[0] * np.cos(angle) + velocity[1] * np.sin(angle)
    assert speed > 0.
    scan = np.full(car.num_beams, 30.)
    for clearance, contact in [(speed * .0049, True), (speed * .0051, False), (0., True)]:
        scan[beam] = car.side_distances[beam] + clearance
        car.check_ttc(scan)
        assert car.in_collision == contact
    car._physics.reset(pose=(1., 2., heading), velocity=(0., 0.))
    car._sync_physics_view()
    scan[beam] = car.side_distances[beam] + .0049
    car.check_ttc(scan)
    assert car.in_collision
    scan[beam] = car.side_distances[beam] + .0051
    car.check_ttc(scan)
    assert not car.in_collision
    assert car.ttc_thresh == .005


def test_scanner_sanitizes_invalid_ranges_before_returning_them(tmp_path, monkeypatch):
    path = write_map(tmp_path, np.full((4, 5), 255))
    scanner = ScanSimulator2D(5, 1.)
    scanner.set_map(str(path), ".png")
    monkeypatch.setattr("physics.laser_models.get_scan",
                        lambda *args: np.array([-1., np.nan, np.inf, -np.inf, 31.]))
    np.testing.assert_array_equal(scanner.scan(np.array([.2, .2, 0.]), None),
                                  [0., 0., 30., 0., 30.])


@pytest.mark.parametrize("preloaded", [False, True])
def test_environment_passes_processed_occupancy_through_initialization_and_hot_swap(tmp_path, preloaded):
    from env.RaceEnv import RaceEnv

    path = write_map(tmp_path, np.tile([0, 150, 255], (20, 4)))
    settings = {"track_threshold": 200, "track_inverted": True}
    data = load_map(MapLoader(), path, **settings) if preloaded else None
    env = RaceEnv(map=path.name, map_dir=str(tmp_path), map_data=data,
                  n_agents=1, lidar_beams=17, render_mode=None, **settings)
    try:
        expected = np.tile([True, True, False], (20, 4))
        np.testing.assert_array_equal(env._track_mask, expected)
        np.testing.assert_array_equal(env.sim.scan_simulator.map_img, expected[::-1])
        other = write_map(tmp_path, np.tile([0, 255, 150], (20, 4)), name="other")
        env.update_map(str(other), ".png")
        expected = np.tile([True, False, True], (20, 4))
        np.testing.assert_array_equal(env._track_mask, expected)
        np.testing.assert_array_equal(env.sim.scan_simulator.map_img, expected[::-1])
    finally:
        env.close()
