"""Interactive Semi-Lagrangian level-set/particle advection diagnostic."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--particle-side", type=int, default=100)
    args = parser.parse_args()

    import taichi as ti

    from src.levelset.SemiLagrangian import SemiLagrangian

    ti.init(
        arch=getattr(ti, args.arch),
        default_fp=ti.f32,
        offline_cache=False,
    )
    dt = 1.0e-4
    dx = 0.1
    inverse_dx = 1.0 / dx
    particles_per_cell = 2
    particle_side = args.particle_side
    particle_count = particle_side**2
    grid_size = args.grid_size

    position = ti.Vector.field(2, ti.f32, shape=particle_count)
    velocity = ti.Vector.field(2, ti.f32, shape=particle_count)
    visible = ti.Vector.field(3, ti.f32, shape=particle_count)
    grid_mass = ti.field(ti.f32, shape=(grid_size, grid_size))
    grid_velocity = ti.Vector.field(2, ti.f32, shape=(grid_size, grid_size))
    level_set = SemiLagrangian(2, (grid_size, grid_size), dx)

    @ti.kernel
    def initialize_particles():
        for particle in position:
            i = particle % particle_side
            j = particle // particle_side
            position[particle] = (
                ti.Vector([i, j]) + 0.5
            ) * dx / particles_per_cell + ti.Vector([3.0, 3.0])

    @ti.func
    def box_distance(point, lower, upper):
        value = ti.cast(0.0, ti.f32)
        if all(point > lower) and all(point < upper):
            value = ti.max(lower - point, point - upper).max()
        else:
            closest = ti.Vector.zero(ti.f32, 2)
            for axis in ti.static(range(2)):
                closest[axis] = ti.min(
                    ti.max(point[axis], lower[axis]), upper[axis]
                )
            value = (point - closest).norm()
        return value

    @ti.kernel
    def initialize_level_set():
        for index in ti.grouped(level_set.distance_field):
            level_set.distance_field[index] = box_distance(
                (index + 0.5) * dx,
                ti.Vector([3.0, 3.0]),
                ti.Vector([8.0, 8.0]),
            )

    @ti.kernel
    def particle_grid_step():
        for particle in position:
            base = (position[particle] * inverse_dx - 0.5).cast(int)
            fractional = position[particle] * inverse_dx - base.cast(float)
            weights = [
                0.5 * (1.5 - fractional) ** 2,
                0.75 - (fractional - 1.0) ** 2,
                0.5 * (fractional - 0.5) ** 2,
            ]
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                weight = weights[i][0] * weights[j][1]
                grid_velocity[base + offset] += weight * velocity[particle]
                grid_mass[base + offset] += weight

        for i, j in grid_mass:
            if grid_mass[i, j] > 0.0:
                grid_velocity[i, j] /= grid_mass[i, j]
                grid_velocity[i, j] += dt * ti.Vector([0.0, -9.8])

        for particle in position:
            base = (position[particle] * inverse_dx - 0.5).cast(int)
            fractional = position[particle] * inverse_dx - base.cast(float)
            weights = [
                0.5 * (1.5 - fractional) ** 2,
                0.75 - (fractional - 1.0) ** 2,
                0.5 * (fractional - 0.5) ** 2,
            ]
            new_velocity = ti.Vector.zero(ti.f32, 2)
            for i, j in ti.static(ti.ndrange(3, 3)):
                new_velocity += (
                    weights[i][0]
                    * weights[j][1]
                    * grid_velocity[base + ti.Vector([i, j])]
                )
            velocity[particle] = new_velocity
            position[particle] += dt * new_velocity
            visible[particle] = ti.Vector(
                [
                    position[particle][0] / (grid_size * dx),
                    position[particle][1] / (grid_size * dx),
                    0.0,
                ]
            )

    initialize_particles()
    initialize_level_set()
    window = ti.ui.Window("Level-set advection", (892, 892))
    while window.running:
        level_set.run(grid_velocity, dt)
        grid_velocity.fill(0)
        grid_mass.fill(0)
        particle_grid_step()
        canvas = window.get_canvas()
        canvas.set_background_color((0.0, 0.0, 0.0))
        canvas.contour(level_set.distance_field)
        canvas.circles(
            visible,
            radius=1.0 / (2.0 * particles_per_cell * grid_size),
            color=(1.0, 1.0, 1.0),
        )
        window.show()


if __name__ == "__main__":
    main()
