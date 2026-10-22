# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



"""
Prototype of Utility functions and GJK algorithm for Collision checks between vehicles
Originally from https://github.com/kroitor/gjk.c
Author: Hongrui Zheng
"""

from dataclasses import dataclass

import numpy as np
from numba import njit

# Version 1 used a wheelbase-sized wall footprint and ignored the LiDAR mount.
# Dynamics retain their independent model_version values.
GEOMETRY_COLLISION_VERSION = 2


@dataclass(frozen=True)
class VehicleGeometry:
    """Planar body bounds about the dynamics reference: x forward, y left.

    The nonlinear reference is the center of mass (axles at +lf and -lr).
    Legacy dynamic ST uses the same convention; its low-speed kinematic
    approximation is preserved. Neither config specifies body overhangs, so
    from_params preserves the existing GJK body centered on the state pose.
    Explicit extents also allow an asymmetric body without shifting the axles.
    LiDAR coordinates are relative to this reference, not the rear axle.
    """

    front: float
    rear: float
    left: float
    right: float
    lf: float
    lr: float
    lidar_x: float = 0.0
    lidar_y: float = 0.0

    def __post_init__(self):
        dimensions = (self.front, self.rear, self.left, self.right, self.lf, self.lr)
        if not all(np.isfinite(v) and v > 0 for v in dimensions):
            raise ValueError("Vehicle geometry extents and axle distances must be finite and positive")
        if not all(np.isfinite(v) for v in self.lidar_position):
            raise ValueError("LiDAR mounting position must be finite")

    @classmethod
    def from_params(cls, params, lidar_dist=0.0):
        return cls(params['length'] / 2.0, params['length'] / 2.0,
                   params['width'] / 2.0, params['width'] / 2.0,
                   params['lf'], params['lr'], lidar_x=lidar_dist)

    @property
    def axle_locations(self):
        return np.array([[self.lf, 0.0], [-self.lr, 0.0]])

    @property
    def lidar_position(self):
        return np.array([self.lidar_x, self.lidar_y])

    @property
    def footprint(self):
        return self.world_vertices(np.zeros(3))

    def world_vertices(self, pose):
        return get_vertices(pose, self.front + self.rear, self.left + self.right,
                            (self.front - self.rear) / 2.0,
                            (self.left - self.right) / 2.0)

    def lidar_pose(self, pose, out=None):
        """Transform the mounting point with the same matrix used by GJK."""
        if out is None:
            out = np.empty(3, dtype=np.float64)
        transform = get_trmtx(pose)
        out[:2] = transform[:2, :2] @ self.lidar_position + transform[:2, 3]
        out[2] = pose[2]
        return out

    def ray_intersections(self, angles):
        """Entry/exit ranges of body-frame LiDAR rays through the footprint.

        Entry is zero for an in-body sensor. Misses have NaN ranges. Keeping
        entry as well as exit also handles mounts outside the body: an obstacle
        between the sensor and body is not itself body contact.
        """
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        near = np.zeros(len(angles))
        far = np.full(len(angles), np.inf)
        bounds = ((-self.rear, self.front), (-self.right, self.left))
        for axis, (lower, upper) in enumerate(bounds):
            origin = self.lidar_position[axis]
            direction = directions[:, axis]
            parallel = np.abs(direction) < 1e-15
            axis_near = np.full(len(angles), -np.inf)
            axis_far = np.full(len(angles), np.inf)
            moving = ~parallel
            first = (lower - origin) / direction[moving]
            second = (upper - origin) / direction[moving]
            axis_near[moving] = np.minimum(first, second)
            axis_far[moving] = np.maximum(first, second)
            if origin < lower or origin > upper:
                axis_far[parallel] = -np.inf
            near = np.maximum(near, axis_near)
            far = np.minimum(far, axis_far)
        misses = far < near
        near[misses] = np.nan
        far[misses] = np.nan
        return near, far


@njit(cache=True)
def perpendicular(pt):
    """
    Return a 2-vector's perpendicular vector

    Args:
        pt (np.ndarray, (2,)): input vector

    Returns:
        pt (np.ndarray, (2,)): perpendicular vector
    """
    temp = pt[0]
    pt[0] = pt[1]
    pt[1] = -1*temp
    return pt


@njit(cache=True)
def tripleProduct(a, b, c):
    """
    Return triple product of three vectors

    Args:
        a, b, c (np.ndarray, (2,)): input vectors

    Returns:
        (np.ndarray, (2,)): triple product
    """
    ac = a.dot(c)
    bc = b.dot(c)
    return b*ac - a*bc


@njit(cache=True)
def avgPoint(vertices):
    """
    Return the average point of multiple vertices

    Args:
        vertices (np.ndarray, (n, 2)): the vertices we want to find avg on

    Returns:
        avg (np.ndarray, (2,)): average point of the vertices
    """
    return np.sum(vertices, axis=0)/vertices.shape[0]


@njit(cache=True)
def indexOfFurthestPoint(vertices, d):
    """
    Return the index of the vertex furthest away along a direction in the list of vertices

    Args:
        vertices (np.ndarray, (n, 2)): the vertices we want to find avg on

    Returns:
        idx (int): index of the furthest point
    """
    return np.argmax(vertices.dot(d))


@njit(cache=True)
def support(vertices1, vertices2, d):
    """
    Minkowski sum support function for GJK

    Args:
        vertices1 (np.ndarray, (n, 2)): vertices of the first body
        vertices2 (np.ndarray, (n, 2)): vertices of the second body
        d (np.ndarray, (2, )): direction to find the support along

    Returns:
        support (np.ndarray, (n, 2)): Minkowski sum
    """
    i = indexOfFurthestPoint(vertices1, d)
    j = indexOfFurthestPoint(vertices2, -d)
    return vertices1[i] - vertices2[j]


@njit(cache=True)
def collision(vertices1, vertices2):
    """
    GJK test to see whether two bodies overlap

    Args:
        vertices1 (np.ndarray, (n, 2)): vertices of the first body
        vertices2 (np.ndarray, (n, 2)): vertices of the second body

    Returns:
        overlap (boolean): True if two bodies collide
    """
    index = 0
    simplex = np.empty((3, 2))

    position1 = avgPoint(vertices1)
    position2 = avgPoint(vertices2)

    d = position1 - position2

    if d[0] == 0 and d[1] == 0:
        d[0] = 1.0

    a = support(vertices1, vertices2, d)
    simplex[index, :] = a

    if d.dot(a) <= 0:
        return False

    d = -a

    iter_count = 0
    while iter_count < 1e3:
        a = support(vertices1, vertices2, d)
        index += 1
        simplex[index, :] = a
        if d.dot(a) <= 0:
            return False

        ao = -a

        if index < 2:
            b = simplex[0, :]
            ab = b-a
            d = tripleProduct(ab, ao, ab)
            if np.linalg.norm(d) < 1e-10:
                d = perpendicular(ab)
            continue

        b = simplex[1, :]
        c = simplex[0, :]
        ab = b-a
        ac = c-a

        acperp = tripleProduct(ab, ac, ac)

        if acperp.dot(ao) >= 0:
            d = acperp
        else:
            abperp = tripleProduct(ac, ab, ab)
            if abperp.dot(ao) < 0:
                return True
            simplex[0, :] = simplex[1, :]
            d = abperp

        simplex[1, :] = simplex[2, :]
        index -= 1

        iter_count += 1
    return False

@njit(cache=True)
def collision_multiple(vertices):
    """
    Check pair-wise collisions for all provided vertices

    Args:
        vertices (np.ndarray (num_bodies, 4, 2)): all vertices for checking pair-wise collision

    Returns:
        collisions (np.ndarray (num_vertices, )): whether each body is in collision
        collision_idx (np.ndarray (num_vertices, )): which index of other body is each index's body is in collision, -1 if not in collision
    """
    collisions = np.zeros((vertices.shape[0], ))
    collision_idx = -1 * np.ones((vertices.shape[0], ))
    # looping over all pairs
    for i in range(vertices.shape[0]-1):
        for j in range(i+1, vertices.shape[0]):
            # check collision
            vi = np.ascontiguousarray(vertices[i, :, :])
            vj = np.ascontiguousarray(vertices[j, :, :])
            ij_collision = collision(vi, vj)
            # fill in results
            if ij_collision:
                collisions[i] = 1.
                collisions[j] = 1.
                collision_idx[i] = j
                collision_idx[j] = i

    return collisions, collision_idx

"""
Utility functions for getting vertices by pose and shape
"""

@njit(cache=True)
def get_trmtx(pose):
    x = pose[0]
    y = pose[1]
    th = pose[2]

    cos = np.cos(th)
    sin = np.sin(th)

    H = np.zeros((4, 4), dtype=np.float64)

    # assign scalars instead of slices
    H[0, 0] = cos
    H[0, 1] = -sin
    H[0, 3] = x

    H[1, 0] = sin
    H[1, 1] = cos
    H[1, 3] = y

    H[2, 2] = 1.0
    H[3, 3] = 1.0

    return H

@njit(cache=True)
def get_vertices(pose, length, width, center_x=0.0, center_y=0.0):
    """World corners, with optional body-frame center offsets from the pose."""
    H = get_trmtx(pose)

    hl = length / 2.0
    hw = width / 2.0

    # homogeneous corners
    corners = np.zeros((4, 4), dtype=np.float64)
    corners[0, 0] = center_x - hl; corners[0, 1] = center_y + hw; corners[0, 3] = 1.0
    corners[1, 0] = center_x - hl; corners[1, 1] = center_y - hw; corners[1, 3] = 1.0
    corners[2, 0] = center_x + hl; corners[2, 1] = center_y - hw; corners[2, 3] = 1.0
    corners[3, 0] = center_x + hl; corners[3, 1] = center_y + hw; corners[3, 3] = 1.0

    verts_h = corners @ H.T

    verts = np.zeros((4, 2), dtype=np.float64)
    for i in range(4):
        verts[i, 0] = verts_h[i, 0]
        verts[i, 1] = verts_h[i, 1]

    return verts


# Separate from geometry v2: continuous detection and stop-at-contact response.
SWEPT_COLLISION_VERSION = 1
CONTACT_DISTANCE_TOL = 1e-7  # metres; conservative numerical uncertainty, not TTC margin
CONTACT_TIME_TOL = 1e-10   # fraction of one environment step


@njit(cache=True)
def trace_state(trace, fraction):
    """Evaluate a piecewise integration polynomial, including unwrapped yaw."""
    scaled = min(1., max(0., fraction)) * len(trace)
    index = min(int(scaled), len(trace) - 1)
    t = scaled - index
    return ((trace[index, 3] * t + trace[index, 2]) * t + trace[index, 1]) * t + trace[index, 0]


@njit(cache=True)
def _footprint_radius(extents):
    return np.hypot(max(extents[0], extents[1]), max(extents[2], extents[3]))


@njit(cache=True)
def _trace_bounds(trace, extents):
    """Bernstein control-point hull bounds all translations, then adds body radius."""
    lower = np.full(2, np.inf)
    upper = np.full(2, -np.inf)
    for segment in trace:
        for axis in range(2):
            c0, c1, c2, c3 = segment[:, axis]
            controls = (c0, c0 + c1 / 3., c0 + 2. * c1 / 3. + c2 / 3., c0 + c1 + c2 + c3)
            for value in controls:
                lower[axis] = min(lower[axis], value)
                upper[axis] = max(upper[axis], value)
    radius = _footprint_radius(extents)
    return lower - radius, upper + radius


@njit(cache=True)
def _motion_bound(trace, extents):
    """Upper bound on any footprint point's speed per environment-step fraction.

    For c0+c1*t+c2*t²+c3*t³, translation speed is bounded by the
    triangle inequality on c1+2*c2*t+3*c3*t². Rotation adds R*|yaw'|.
    Multiplication by segment count converts local fraction to global fraction.
    """
    radius = _footprint_radius(extents)
    bound = 0.
    for segment in trace:
        translation, angular = 0., 0.
        for power in range(1, 4):
            translation += power * np.hypot(segment[power, 0], segment[power, 1])
            angular += power * abs(segment[power, 2])
        bound = max(bound, translation + radius * angular)
    return bound * len(trace)


@njit(cache=True)
def _relative_motion_bound(trace_a, ext_a, trace_b, ext_b):
    """Remove shared translation when both traces use the same time segments."""
    bound = 0.
    radius_a, radius_b = _footprint_radius(ext_a), _footprint_radius(ext_b)
    for index in range(len(trace_a)):
        speed = 0.
        for power in range(1, 4):
            delta = trace_b[index, power, :2] - trace_a[index, power, :2]
            speed += power * (np.hypot(delta[0], delta[1])
                              + radius_a * abs(trace_a[index, power, 2])
                              + radius_b * abs(trace_b[index, power, 2]))
        bound = max(bound, speed)
    return bound * len(trace_a)


@njit(cache=True)
def _trace_vertices(trace, extents, fraction):
    pose = trace_state(trace, fraction)
    return get_vertices(pose, extents[0] + extents[1], extents[2] + extents[3],
                        (extents[0] - extents[1]) / 2., (extents[2] - extents[3]) / 2.)


@njit(cache=True)
def _separation(trace_a, ext_a, trace_b, ext_b, fraction, stop_a, stop_b, map_width, map_height):
    a = _trace_vertices(trace_a, ext_a, min(fraction, stop_a))
    if map_width > 0.:
        # Distance from footprint to the occupied exterior of the map rectangle.
        return min(np.min(a[:, 0]), np.min(a[:, 1]),
                   map_width - np.max(a[:, 0]), map_height - np.max(a[:, 1]))
    b = _trace_vertices(trace_b, ext_b, min(fraction, stop_b))
    # Separating-axis gaps are lower bounds on Euclidean distance. SAT is exact
    # for rectangles, including grazing contact (GJK's legacy API tests overlap).
    gap = -np.inf
    for vertices in (a, b):
        for edge in range(2):
            delta = vertices[edge + 1] - vertices[edge]
            norm = np.hypot(delta[0], delta[1])
            nx, ny = -delta[1] / norm, delta[0] / norm
            amin, amax, bmin, bmax = np.inf, -np.inf, np.inf, -np.inf
            for corner in range(4):
                av = a[corner, 0] * nx + a[corner, 1] * ny
                bv = b[corner, 0] * nx + b[corner, 1] * ny
                amin, amax = min(amin, av), max(amax, av)
                bmin, bmax = min(bmin, bv), max(bmax, bv)
            gap = max(gap, bmin - amax, amin - bmax)
    return gap


@njit(cache=True)
def swept_contact(trace_a, ext_a, trace_b, ext_b, start=0., stop_a=np.inf,
                  stop_b=np.inf, map_width=0., map_height=0.):
    """Earliest conservative contact fraction, or infinity for a certified miss.

    An interval is discarded only when midpoint separation exceeds the maximum
    possible Hausdorff displacement over that entire interval. Otherwise bisect
    left-first. Thus even a transient/grazing collision between all query points
    cannot be discarded. Uncertainty <= 1e-7 m (or 1e-10 step fraction) is contact;
    the returned left endpoint prevents advancing through an unresolved interval.
    This is adaptive interval certification, not fixed sampling.
    """
    if map_width == 0.:
        lo_a, hi_a = _trace_bounds(trace_a, ext_a)
        lo_b, hi_b = _trace_bounds(trace_b, ext_b)
        if np.any(lo_a > hi_b + CONTACT_DISTANCE_TOL) or np.any(lo_b > hi_a + CONTACT_DISTANCE_TOL):
            return np.inf
    bound_a, bound_b = _motion_bound(trace_a, ext_a), _motion_bound(trace_b, ext_b)
    relative_bound = bound_a + bound_b
    if map_width == 0. and len(trace_a) == len(trace_b):
        relative_bound = min(relative_bound, _relative_motion_bound(trace_a, ext_a, trace_b, ext_b))
    if map_width > 0.:
        bound_b = 0.
    if _separation(trace_a, ext_a, trace_b, ext_b, start, stop_a, stop_b, map_width, map_height) <= CONTACT_DISTANCE_TOL:
        return start
    left_stack, right_stack = np.empty(64), np.empty(64)
    left_stack[0], right_stack[0], count = start, 1., 1
    while count:
        count -= 1
        left, right = left_stack[count], right_stack[count]
        middle = (left + right) / 2.
        bound = (bound_a if left < stop_a else 0.) + (bound_b if left < stop_b else 0.)
        if right <= stop_a and right <= stop_b:
            bound = min(bound, relative_bound)
        gap = _separation(trace_a, ext_a, trace_b, ext_b, middle, stop_a, stop_b, map_width, map_height)
        if gap > bound * (right - left) / 2. + CONTACT_DISTANCE_TOL:
            continue
        if bound * (right - left) <= CONTACT_DISTANCE_TOL or right - left <= CONTACT_TIME_TOL:
            return left
        # Push right first so left is examined first; depth <= 35 at time tolerance.
        left_stack[count], right_stack[count] = middle, right
        left_stack[count + 1], right_stack[count + 1] = left, middle
        count += 2
    return np.inf


@njit(cache=True)
def map_motion_trace(trace, origin):
    """Transform all polynomial coefficients into the F4 occupancy-grid frame."""
    result = trace.copy()
    c, s = np.cos(origin[2]), np.sin(origin[2])
    for index in range(len(trace)):
        for power in range(4):
            x, y = trace[index, power, 0], trace[index, power, 1]
            if power == 0:
                x, y = x - origin[0], y - origin[1]
            result[index, power, 0] = c * x + s * y
            result[index, power, 1] = -s * x + c * y
        result[index, 0, 2] -= origin[2]
    return result


@njit(cache=True)
def swept_map_contact(trace, extents, occupancy, resolution):
    """Sweep the entire body against occupied closed cells and map exterior.

    occupancy is the scanner's authoritative, bottom-up free-cell mask. A broad
    swept AABB limits cell enumeration; every candidate gets certified CCD.
    """
    height, width = occupancy.shape
    stationary = np.zeros((1, 4, 3))
    cell_extents = np.full(4, resolution / 2.)
    earliest = swept_contact(trace, extents, stationary, cell_extents,
                             map_width=width * resolution, map_height=height * resolution)
    lower, upper = _trace_bounds(trace, extents)
    c0 = max(0, int(np.floor(lower[0] / resolution)) - 1)
    c1 = min(width - 1, int(np.floor(upper[0] / resolution)) + 1)
    r0 = max(0, int(np.floor(lower[1] / resolution)) - 1)
    r1 = min(height - 1, int(np.floor(upper[1] / resolution)) + 1)
    for row in range(r0, r1 + 1):
        for col in range(c0, c1 + 1):
            if occupancy[row, col]:
                continue
            stationary[0, 0, 0] = (col + .5) * resolution
            stationary[0, 0, 1] = (row + .5) * resolution
            earliest = min(earliest, swept_contact(trace, extents, stationary, cell_extents))
            if earliest == 0.:
                return earliest
    return earliest
