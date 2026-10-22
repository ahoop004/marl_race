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
