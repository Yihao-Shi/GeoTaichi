import numpy as np
import pytest
import taichi as ti
from geotaichi import init
from src.mpm.engines.EngineKernel import enforce_particle_solid_sdf_collision, interpolate_solid_sdf
from src.mpm.structs.Particle import ParticleCloudIncompressible2D, ParticleCloudIncompressible3D
from examples.mpm.IncompressibleFluid.dam_break_visible_sdf_2d.obstacle import VERTICES


def signed_polygon_distance(points):
    distance = np.full(len(points), np.inf)
    inside = np.zeros(len(points), dtype=bool)
    for start, end in zip(VERTICES, np.roll(VERTICES, -1, axis=0)):
        edge = end - start
        t = np.clip(np.sum((points - start) * edge, axis=1) / np.dot(edge, edge), 0, 1)
        distance = np.minimum(distance, np.linalg.norm(points - start - t[:, None] * edge, axis=1))
        crossing = (start[1] > points[:, 1]) != (end[1] > points[:, 1])
        inside ^= crossing & (points[:, 0] < start[0] + (points[:, 1] - start[1]) * edge[0] / edge[1])
    return np.where(inside, -distance, distance)


@pytest.mark.parametrize("dimension", [2, pytest.param(3, marks=pytest.mark.isolated_dimension(3))])
def test_corner_collision_leaves_interpolated_solid_and_preserves_tangent(dimension):
    init(dim=dimension, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    shape = (182, 94) + ((6,) if dimension == 3 else ())
    spacing = np.array([0.005, 0.45 / 92] + ([0.005] if dimension == 3 else []))
    sdf = ti.field(float, shape=shape, offset=(-1,) * dimension)
    indices = np.indices(shape).reshape(dimension, -1).T - 1
    points = (indices + 0.5) * spacing
    sdf.from_numpy(signed_polygon_distance(points[:, :2]).reshape(shape))
    particles = (ParticleCloudIncompressible2D if dimension == 2 else ParticleCloudIncompressible3D).field(shape=1)
    particles.active[0] = 1
    particles.x[0] = [0.4452854096889496, 0.05459030345082283] + ([0.01] if dimension == 3 else [])
    particles.v[0] = [0.0, 0.0] + ([0.3] if dimension == 3 else [])
    velocities = [
        ti.field(float, shape=tuple(n + int(d == axis) for d, n in enumerate(shape)), offset=(-1,) * dimension)
        for axis in range(dimension)
    ]
    phi = ti.field(float, shape=())
    cnum, grid_size = ti.Vector(list(shape)), ti.Vector(list(spacing))

    @ti.kernel
    def sample():
        phi[None] = interpolate_solid_sdf(particles[0].x, 1, cnum, grid_size, sdf)

    sample()
    assert phi[None] < -0.0007
    enforce_particle_solid_sdf_collision(
        1,
        1,
        ti.Vector(list(shape)),
        ti.Vector(list(spacing)),
        sdf,
        velocities[0],
        velocities[1],
        velocities[-1],
        particles,
    )
    sample()
    assert phi[None] >= -1e-7
    assert particles.active[0] == 1
    if dimension == 3:
        assert particles.v[0][2] == pytest.approx(0.3)
