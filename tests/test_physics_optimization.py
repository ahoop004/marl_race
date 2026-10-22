"""Small O1--O3 guards; comprehensive replays and timing remain separate."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from physics.dynamic_models import first_order_actuator_step, sample_wheel_actuators
from physics.tire_models import MF61_KEYS, _mf61_contact_terms, mf61_tire_force
from physics.vehicle import CombinedSlipVehicle


@pytest.fixture
def vehicle():
    config = yaml.safe_load((Path(__file__).resolve().parents[1]
                             / "configs/vehicle/combined_slip.yaml").read_text())
    return CombinedSlipVehicle(
        {k: v for k, v in config.items() if k not in {"length", "width", "wheel_actuators"}},
        config["wheel_actuators"])


def test_mf61_alternate_coefficients_retain_reference_forces(vehicle):
    coefficients = dict(vehicle.params["front_tire"])
    coefficients.update(
        PEX2=.03, PEX3=.01, PEX4=.1, PKX2=.4, PKX3=.1,
        PHX1=.01, PHX2=.002, PVX1=.003, PVX2=.001, PEY2=.02, PEY3=.1,
        PHY1=.01, PHY2=.002, PVY1=.003, PVY2=.001,
        RBX2=.2, REX1=.1, REX2=.02, RHX1=.01,
        RBY2=.2, RBY3=.01, REY1=.1, REY2=.02, RHY1=.01, RHY2=.002,
        RVY1=.01, RVY2=.002, RVY4=.2, RVY5=.3, RVY6=.4)
    params = tuple(coefficients[key] for key in MF61_KEYS)
    # Captured from 263fa76 before optimization; exercise load dependence,
    # reverse/low speed, nonzero curvature, horizontal and vertical shifts.
    cases = [
        ((2., .1, 2.5, 19., 1.),
         (16.44304656897633, -3.6415307556169427, .25, .049958395721942765)),
        ((-2., -.2, -2.5, 5., .7),
         (-2.7712254273390236, 1.4016036711868924, -.25, -.09966865249116204)),
        ((.01, .02, .1, 30., 1.2),
         (30.54703872601947, -4.54044183500034, .18000000000000002, .039978687123290044)),
    ]
    for (u, v, rolling, load, mu), expected in cases:
        direct = mf61_tire_force(u, v, rolling, load, mu, params, .5)
        terms = _mf61_contact_terms(u, v, rolling, mu, params, .5)
        prepared = mf61_tire_force(u, v, rolling, load, mu, params, .5, terms)
        np.testing.assert_allclose(direct, expected, rtol=2e-14, atol=2e-14)
        np.testing.assert_array_equal(prepared, direct)
    with pytest.raises(ValueError, match="normal-load range"):
        mf61_tire_force(2., .1, 2.5, 1000., 1., params, .5)


def test_rk4_reference_state_counters_and_snapshot_ownership(vehicle):
    vehicle.reset(pose=(1., 2., .3), velocity=(2., .1), yaw_rate=.2,
                  steering_angle=.1, wheel_speed=50.)
    vehicle.command(-.15, 30.)
    vehicle.profile_counters = np.zeros(8, dtype=np.int64)
    before = vehicle.state
    trace = np.zeros((50, 4, 6))
    actual = vehicle.advance(.05, motion_trace=trace)
    # Same single-step reference as above, with all four RK stages and traces.
    expected = [1.1026275924660276, 2.037534198150097, .323336359879043,
                2.250479106719227, .07267981066214524, .5428437533906108,
                .0762093545089899, 44.33062621147578]
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_array_equal(vehicle.profile_counters, [3284, 201, 1240, 50, 151, 50, 0, 0])
    np.testing.assert_array_equal(before, [1., 2., .3, 2., .1, .2, .1, 50.])
    assert np.isfinite(trace).all()
    assert not np.shares_memory(actual, vehicle._chassis)
    assert not np.shares_memory(actual, vehicle._actuators._state)
    actual[:] = 0.
    np.testing.assert_allclose(vehicle.state, expected, rtol=2e-14, atol=2e-14)


def test_actuator_sampling_reuses_output_through_linear_and_exponential_regions():
    state, reference = np.array([0., 20.]), np.array([2., -3.])
    constants, rates = (.5, .15), ((-1., 2.), (-3., 4.))
    output = np.empty(2)
    counters = np.zeros(8, dtype=np.int64)
    for elapsed in (0., .05, .75, 1., 8.):
        expected = [first_order_actuator_step(state[i], reference[i], elapsed,
                                              constants[i], *rates[i]) for i in range(2)]
        actual = sample_wheel_actuators(state, reference, elapsed, constants, rates, counters, output)
        assert np.shares_memory(actual, output)
        np.testing.assert_array_equal(actual, expected)
    assert counters[4] == 5
    np.testing.assert_array_equal(state, [0., 20.])
    np.testing.assert_array_equal(reference, [2., -3.])
    snapshot = sample_wheel_actuators(state, reference, 1., constants, rates)
    assert not np.shares_memory(snapshot, state)
    assert not np.shares_memory(snapshot, reference)
