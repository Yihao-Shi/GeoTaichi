"""Construction helpers shared by the coupled CPT examples.

The helpers preserve the standard CPT dimensions, load, soil constants and
penetration speed. Both implicit IPC routes use the physical axisymmetric
meridian; the legacy explicit routes use a thin 3-D slice.
"""

from __future__ import annotations

import math

import numpy as np

from examples.mpm.Contact.CPT2D.cpt_reference import (
    ALPHA_PIC,
    BACKGROUND_DAMPING,
    COUPLED_DOMAIN,
    COUPLED_DP_MATERIAL,
    COUPLED_GRAVITY,
    COUPLED_INITIAL_STRESS,
    COUPLED_SOIL_SIZE,
    DOMAIN,
    FEMPM_EXPLICIT_CONTACT,
    GRID_SIZE,
    GRAVITY,
    IGAMPM_EXPLICIT_CONTACT,
    INITIAL_STRESS,
    IPC_CONTACT,
    MAPPING,
    PARTICLES_PER_CELL,
    PENETRATOR_MATERIAL,
    PILE_PROFILE,
    PILE_SPEED,
    SHAPE_FUNCTION,
    SOIL_ORIGIN,
    SOIL_SIZE,
    STABILIZATION,
    SURFACE_PRESSURE,
)


def validate_run_parameters(dt, simulation_time, save_interval, resolution_scale):
    values = {
        "dt": dt,
        "simulation_time": simulation_time,
        "save_interval": save_interval,
        "resolution_scale": resolution_scale,
    }
    for name, value in values.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")


def realized_grid_size(resolution_scale):
    return GRID_SIZE * float(resolution_scale)


def native_particle_capacity(grid_size):
    cells = np.ceil(np.asarray(COUPLED_SOIL_SIZE) / float(grid_size)).astype(int)
    particle_count = int(np.prod(cells)) * PARTICLES_PER_CELL**3
    ten_percent_headroom = (particle_count + 9) // 10
    return max(10_000, particle_count + ten_percent_headroom)


def dp_native_material():
    material = COUPLED_DP_MATERIAL
    return {
        "MaterialID": 1,
        "Density": material["density"],
        "YoungModulus": material["young_modulus"],
        "PoissonRatio": material["poisson_ratio"],
        "Friction": material["friction_angle"],
        "Dilation": material["dilation_angle"],
        "Cohesion": material["cohesion"],
        "Tensile": 0.0,
    }


def dp_direct_material():
    material = COUPLED_DP_MATERIAL
    return {
        "model": "DruckerPrager",
        "density": material["density"],
        "young_modulus": material["young_modulus"],
        "poisson_ratio": material["poisson_ratio"],
        "FrictionAngle": material["friction_angle"],
        "DilationAngle": material["dilation_angle"],
        "Cohesion": material["cohesion"],
        "dpType": "Circumscribed",
    }


def configure_native_explicit_mpm(mpm, output_path, dt, simulation_time, save_interval, resolution_scale=1.0):
    """Configure the native explicit 3-D thin-slice CPT soil."""
    validate_run_parameters(dt, simulation_time, save_interval, resolution_scale)
    grid_size = realized_grid_size(resolution_scale)
    surface_band = 0.5 * grid_size
    particle_capacity = native_particle_capacity(grid_size)

    mpm.set_configuration(
        domain=list(COUPLED_DOMAIN),
        gravity=list(COUPLED_GRAVITY),
        background_damping=BACKGROUND_DAMPING,
        alphaPIC=ALPHA_PIC,
        mapping=MAPPING,
        shape_function=SHAPE_FUNCTION,
        stabilize=STABILIZATION,
        configuration="ULMPM",
        solver_type="Explicit",
        material_type="Solid",
        visualize=False,
        log=True,
    )
    mpm.set_solver(
        {
            "Timestep": dt,
            "SimulationTime": simulation_time,
            "SaveInterval": save_interval,
            "SavePath": str(output_path),
        },
        log=True,
    )
    mpm.memory_allocate(
        {
            "max_material_number": 1,
            "max_particle_number": particle_capacity,
            "max_constraint_number": {
                "max_velocity_constraint": 400_000,
                "max_reflection_constraint": 200_000,
                "max_particle_traction_constraint": 400_000,
            },
        },
        log=True,
    )
    mpm.add_material(model="DruckerPrager", material=dp_native_material())
    mpm.add_element(
        {
            "ElementType": "R8N3D",
            "ElementSize": [grid_size] * 3,
        }
    )
    mpm.add_region(
        [
            {
                "Name": "cpt_soil",
                "Type": "Rectangle",
                "BoundingBoxPoint": [0.0, 0.0, 0.0],
                "BoundingBoxSize": list(COUPLED_SOIL_SIZE),
            },
            {
                "Name": "cpt_surface",
                "Type": "Rectangle",
                "BoundingBoxPoint": [
                    0.0,
                    0.0,
                    COUPLED_SOIL_SIZE[2] - surface_band,
                ],
                "BoundingBoxSize": [
                    COUPLED_SOIL_SIZE[0],
                    COUPLED_SOIL_SIZE[1],
                    surface_band,
                ],
            },
        ]
    )
    mpm.add_body(
        {
            "Template": {
                "RegionName": "cpt_soil",
                "nParticlesPerCell": PARTICLES_PER_CELL,
                "BodyID": 0,
                "MaterialID": 1,
                "ParticleStress": {"InternalStress": list(COUPLED_INITIAL_STRESS)},
                "Traction": [
                    {
                        "Pressure": [0.0, 0.0, -SURFACE_PRESSURE],
                        "RegionName": "cpt_surface",
                    }
                ],
                "InitialVelocity": [0.0, 0.0, 0.0],
                "FixVelocity": ["Free", "Free", "Free"],
            }
        }
    )
    x_max, y_max, z_max = COUPLED_DOMAIN
    mpm.add_boundary_condition(
        [
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0, 0.0, 0.0],
                "StartPoint": [0.0, 0.0, 0.0],
                "EndPoint": [x_max, y_max, 0.0],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0, None, None],
                "StartPoint": [0.0, 0.0, 0.0],
                "EndPoint": [0.0, y_max, z_max],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0, None, None],
                "StartPoint": [x_max, 0.0, 0.0],
                "EndPoint": [x_max, y_max, z_max],
            },
            {
                "BoundaryType": "ReflectionConstraint",
                "StartPoint": [0.0, 0.0, 0.0],
                "EndPoint": [x_max, 0.0, z_max],
                "Norm": [0.0, -1.0, 0.0],
            },
            {
                "BoundaryType": "ReflectionConstraint",
                "StartPoint": [0.0, y_max, 0.0],
                "EndPoint": [x_max, y_max, z_max],
                "Norm": [0.0, 1.0, 0.0],
            },
        ]
    )
    mpm.select_save_data(particle=True, grid=False, object=False)
    return grid_size


def _direct_particle_points(grid_size):
    # Keep the first radial point close to the standard r=0.001 pile edge even
    # when the implicit tangent uses a coarser grid.
    xs = np.arange(0.5 * GRID_SIZE, COUPLED_SOIL_SIZE[0], grid_size)
    ys = np.arange(0.5 * grid_size, COUPLED_SOIL_SIZE[1], grid_size)
    zs = np.arange(0.5 * grid_size, COUPLED_SOIL_SIZE[2], grid_size)
    return np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)


def _direct_boundary(grid_size):
    from src.mpm.boundaries.BoundaryCondition import DirichletBoundary

    grid_num = np.ceil(np.asarray(COUPLED_DOMAIN) / grid_size).astype(np.int32) + 1
    nx, ny, nz = map(int, grid_num)
    node = np.arange(nx * ny * nz, dtype=np.int32).reshape(nz, ny, nx)
    bottom = node[0, :, :].reshape(-1)
    x_sides = np.concatenate((node[:, :, 0].reshape(-1), node[:, :, -1].reshape(-1)))
    y_sides = np.concatenate((node[:, 0, :].reshape(-1), node[:, -1, :].reshape(-1)))
    entries = [
        list(3 * np.unique(np.concatenate((bottom, x_sides))) + 0),
        list(3 * np.unique(np.concatenate((bottom, y_sides))) + 1),
        list(3 * bottom + 2),
    ]
    values = [0.0] * sum(len(entry) for entry in entries)
    boundary = DirichletBoundary()
    boundary.append(entries, values)
    return boundary


def configure_direct_implicit_mpm(mpm, output_path, dt, simulation_time, save_interval, resolution_scale=4.0):
    """Configure the Direct implicit DP soil used by either IPC coupling."""
    validate_run_parameters(dt, simulation_time, save_interval, resolution_scale)
    grid_size = realized_grid_size(resolution_scale)
    points = _direct_particle_points(grid_size)

    mpm.set_configuration(
        dimension=3,
        mpm_backend="Direct",
        solver_type="Implicit",
        configuration="ULMPM",
        domain=list(COUPLED_DOMAIN),
        gravity=list(COUPLED_GRAVITY),
        background_damping=BACKGROUND_DAMPING,
        visualize=True,
        log=True,
    )
    body = mpm.create_body()
    body.add_particles(
        points,
        volume=grid_size**3,
        init_v=[0.0, 0.0, 0.0],
        name="cpt_soil",
        grid_size=grid_size,
        xmin=[0.0, 0.0, 0.0],
        xmax=list(COUPLED_DOMAIN),
    )
    mpm.add_body(body)
    mpm.memory_allocate({"max_particle_number": points.shape[0]}, log=False)
    mpm.add_boundary_condition(dirichlet=_direct_boundary(grid_size))
    mpm.add_material(**dp_direct_material())
    mpm.add_element({"ElementSize": grid_size, "ShapeFunction": "Linear"})
    step_count = int(math.ceil(simulation_time / dt))
    output_interval = max(1, int(round(save_interval / dt)))
    mpm.set_solver(
        {
            "dt": dt,
            "step": step_count,
            "interval": output_interval,
            "path": str(output_path),
            "newmark": [1.0, 0.5, 1.0],
            "residual": 5.0e-4,
            "max_iters": 35,
            "scale": 1.0,
            "ccd": True,
            "project_pd": True,
        }
    )
    return grid_size, points.shape[0]


def axisymmetric_particle_count(grid_size, particles_per_cell=PARTICLES_PER_CELL):
    cells = np.floor(
        np.asarray(SOIL_SIZE) / float(grid_size)
        + 8.0 * np.finfo(float).eps * np.maximum(1.0, np.asarray(SOIL_SIZE) / float(grid_size))
    ).astype(int)
    return int(np.prod(cells)) * int(particles_per_cell) ** 2


def axisymmetric_surface_pressure_particles(points, grid_size):
    """Return material-point IDs and reference annular areas for top pressure.

    Each point represents a reference annulus of radial width dx/ppc. The
    forces keep that reference area and follow the selected particle IDs;
    current shape functions transfer them to the grid at every step.
    """
    points = np.asarray(points, dtype=np.float64)
    particle_spacing = float(grid_size) / PARTICLES_PER_CELL
    top_height = SOIL_ORIGIN[1] + SOIL_SIZE[1] - 0.5 * particle_spacing
    particle_ids = np.flatnonzero(np.isclose(points[:, 1], top_height, rtol=0.0, atol=1.0e-12)).astype(np.int32)
    surface_area = 2.0 * np.pi * points[particle_ids, 0] * particle_spacing
    return particle_ids, surface_area


def axisymmetric_initial_deformation_gradient():
    """Elastic stretch whose Hencky Kirchhoff stress is the CPT preload."""
    material = COUPLED_DP_MATERIAL
    young = material["young_modulus"]
    poisson = material["poisson_ratio"]
    stress = np.asarray(INITIAL_STRESS[:3], dtype=np.float64)
    log_stretch = ((1.0 + poisson) * stress - poisson * np.sum(stress)) / young
    return np.diag(np.exp(log_stretch))


def _direct_axisymmetric_boundaries(grid_size):
    from src.mpm.boundaries.BoundaryCondition import DirichletBoundary

    ratios = np.asarray(DOMAIN) / float(grid_size)
    grid_num = np.floor(ratios + 8.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(ratios))).astype(np.int32) + 1
    nr, nz = map(int, grid_num)
    nodes = np.arange(nr * nz, dtype=np.int32).reshape(nz, nr)
    bottom = nodes[0, :]
    radial_fixed = np.unique(np.concatenate((bottom, nodes[:, 0], nodes[:, -1])))
    dirichlet = DirichletBoundary()
    dirichlet.append(
        [list(2 * radial_fixed), list(2 * bottom + 1)],
        [0.0] * (radial_fixed.size + bottom.size),
    )

    return dirichlet


def configure_direct_axisymmetric_mpm(
    mpm,
    output_path,
    dt,
    simulation_time,
    save_interval,
    resolution_scale=1.0,
):
    """Configure the Direct implicit axisymmetric DP soil for IPC coupling."""
    validate_run_parameters(dt, simulation_time, save_interval, resolution_scale)
    grid_size = realized_grid_size(resolution_scale)
    mpm.set_configuration(
        dimension=2,
        mpm_backend="Direct",
        solver_type="Implicit",
        configuration="ULMPM",
        domain=list(DOMAIN),
        axisymmetric=True,
        axis_offset=0.0,
        gravity=list(GRAVITY),
        background_damping=BACKGROUND_DAMPING,
        alphaPIC=ALPHA_PIC,
        visualize=True,
        log=True,
    )
    body = mpm.create_body()
    body.add_rectangle(
        SOIL_ORIGIN,
        SOIL_SIZE,
        grid_size,
        PARTICLES_PER_CELL,
        init_v=[0.0, 0.0],
        name="cpt_soil",
        grid_size=grid_size,
        xmin=[0.0, 0.0],
        xmax=list(DOMAIN),
    )
    particle_count = int(body.particle_counter)
    expected_count = axisymmetric_particle_count(grid_size)
    if particle_count != expected_count:
        raise RuntimeError(f"axisymmetric CPT generated {particle_count} particles; expected {expected_count}")
    mpm.add_body(body)
    mpm.memory_allocate({"max_particle_number": particle_count}, log=False)
    mpm.add_boundary_condition(dirichlet=_direct_axisymmetric_boundaries(grid_size))
    mpm.add_material(**dp_direct_material())
    mpm.add_element({"ElementSize": grid_size, "ShapeFunction": "Linear"})
    step_count = int(math.ceil(simulation_time / dt))
    output_interval = max(1, int(round(save_interval / dt)))
    mpm.set_solver(
        {
            "dt": dt,
            "step": step_count,
            "interval": output_interval,
            "path": str(output_path),
            "newmark": [1.0, 0.5, 1.0],
            "residual": 5.0e-4,
            "max_iters": 35,
            "scale": 1.0,
            "ccd": True,
            "project_pd": True,
        }
    )
    mpm.add_engine()
    particle_ids, surface_area = axisymmetric_surface_pressure_particles(body.bodies["cpt_soil"]["points"], grid_size)
    mpm.enginer.init_particle_pressure(particle_ids, [0.0, -SURFACE_PRESSURE], surface_area)
    return grid_size, particle_count


def penetrator_vertices():
    profile = np.asarray(PILE_PROFILE, dtype=np.float64)
    front = np.column_stack((profile[:, 0], np.zeros(4), profile[:, 1]))
    back = front.copy()
    back[:, 1] = COUPLED_SOIL_SIZE[1]
    return np.vstack((front, back))


def penetrator_tetrahedra():
    # Five positive-volume tetrahedra forming the extruded standard profile.
    return np.asarray(
        [
            [0, 3, 1, 4],
            [1, 3, 2, 6],
            [1, 3, 6, 4],
            [1, 5, 4, 6],
            [3, 6, 4, 7],
        ],
        dtype=np.int32,
    )


def configure_fem_penetrator(fem, implicit):
    from src.fem import DirichletBoundary, FEMMesh

    points = penetrator_vertices()
    fem.set_configuration(
        dimension=3,
        solver_type="Implicit" if implicit else "Explicit",
        backend="taichi",
    )
    fem.add_mesh(FEMMesh(points, penetrator_tetrahedra(), "TET4"))
    fem.add_material(
        model="NeoHookean",
        young_modulus=PENETRATOR_MATERIAL["young_modulus"],
        poisson_ratio=PENETRATOR_MATERIAL["poisson_ratio"],
        density=PENETRATOR_MATERIAL["density"],
    )
    top = np.flatnonzero(np.isclose(points[:, 2], np.max(points[:, 2])))

    def driven_displacement(time, coordinates):
        displacement = np.zeros_like(coordinates)
        displacement[:, 2] = -PILE_SPEED * float(time)
        return displacement

    fem.add_boundary_condition(dirichlet=DirichletBoundary().add(top, "all", driven_displacement))
    return points.shape[0]


def configure_fem_axisymmetric_penetrator(fem, penetration_start=0.0):
    from src.fem import DirichletBoundary, FEMMesh

    points, cells = axisymmetric_penetrator_mesh()
    fem.set_configuration(
        dimension=2,
        solver_type="Implicit",
        axisymmetric=True,
        axis_offset=0.0,
        backend="taichi",
    )
    fem.add_mesh(FEMMesh(points, cells, "TRI3"))
    fem.add_material(
        model="NeoHookean",
        young_modulus=PENETRATOR_MATERIAL["young_modulus"],
        poisson_ratio=PENETRATOR_MATERIAL["poisson_ratio"],
        density=PENETRATOR_MATERIAL["density"],
    )
    top = np.flatnonzero(np.isclose(points[:, 1], np.max(points[:, 1])))

    def driven_displacement(time, coordinates):
        displacement = np.zeros((coordinates.shape[0], 3), dtype=np.float64)
        displacement[:, 1] = -PILE_SPEED * max(float(time) - penetration_start, 0.0)
        return displacement

    fem.add_boundary_condition(dirichlet=DirichletBoundary().add(top, "all", driven_displacement))
    return top


def axisymmetric_penetrator_mesh(radial_nodes=4, axial_nodes=34):
    """Structured TRI3 mesh of the standard CPT meridian profile."""
    radial_nodes = int(radial_nodes)
    axial_nodes = int(axial_nodes)
    if radial_nodes < 2 or axial_nodes < 2:
        raise ValueError("axisymmetric penetrator mesh needs at least 2 nodes per direction")
    profile = np.asarray(PILE_PROFILE, dtype=np.float64)
    radii = np.linspace(profile[0, 0], profile[1, 0], radial_nodes)
    bottom = np.linspace(profile[0, 1], profile[1, 1], radial_nodes)
    points = np.asarray(
        [
            [radius, bottom[i] + eta * (profile[2, 1] - bottom[i])]
            for eta in np.linspace(0.0, 1.0, axial_nodes)
            for i, radius in enumerate(radii)
        ],
        dtype=np.float64,
    )
    cells = []
    for row in range(axial_nodes - 1):
        for column in range(radial_nodes - 1):
            lower_left = row * radial_nodes + column
            lower_right = lower_left + 1
            upper_left = lower_left + radial_nodes
            upper_right = upper_left + 1
            cells.extend(([lower_left, lower_right, upper_right], [lower_left, upper_right, upper_left]))
    return points, np.asarray(cells, dtype=np.int32)


def configure_iga_axisymmetric_penetrator(iga, dt, step_count, output_interval, output_path):
    """Configure a refined axisymmetric NURBS representation of the CPT pile."""
    from src.iga import DirichletBoundary, Primitives, Rectangle

    profile = np.asarray(PILE_PROFILE, dtype=np.float64)
    pile = Rectangle()
    pile.set_parameters(
        start_point=profile[0],
        size=[profile[1, 0] - profile[0, 0], profile[2, 1] - profile[0, 1]],
    )
    pile.generate_knot_u(degree=2, num_ctrlpts=3)
    pile.generate_knot_v(degree=2, num_ctrlpts=34)
    pile.generate_ctrlpts()
    pile.generate_weights()
    normalized_height = (pile.control_points[:, 1] - profile[0, 1]) / (profile[2, 1] - profile[0, 1])
    bottom_offset = (pile.control_points[:, 0] - profile[0, 0]) * (
        (profile[1, 1] - profile[0, 1]) / (profile[1, 0] - profile[0, 0])
    )
    pile.control_points[:, 1] += (1.0 - normalized_height) * bottom_offset

    primitives = Primitives()
    primitives.append(pile, "cpt_penetrator", init_v=[0.0, -PILE_SPEED])
    primitives.finialize()
    iga.set_configuration(
        dimension=2,
        solver_type="Implicit",
        axisymmetric=True,
        axis_offset=0.0,
    )
    iga.add_primitives(primitives)
    all_control_points = np.arange(pile.control_points.shape[0], dtype=np.int32)
    boundary = DirichletBoundary()
    boundary.append(
        [list(2 * all_control_points), list(2 * all_control_points + 1)],
        [0.0] * all_control_points.size + [-PILE_SPEED * dt] * all_control_points.size,
    )
    iga.add_boundary_condition(dirichlet=boundary)
    iga.add_element(degree=[2, 2])
    iga.add_material(
        young_modulus=PENETRATOR_MATERIAL["young_modulus"],
        poisson_ratio=PENETRATOR_MATERIAL["poisson_ratio"],
        density=PENETRATOR_MATERIAL["density"],
        gravity=list(GRAVITY),
    )
    iga.set_solver(
        dt=dt,
        step=step_count,
        interval=output_interval,
        path=str(output_path),
        newmark=[1.0, 0.5, 1.0],
        residual=5.0e-4,
        max_iters=35,
        assemble_type="HashTriplet",
        linear_solver="PCG",
        project_pd=True,
    )
    top = np.flatnonzero(np.isclose(pile.control_points[:, 1], profile[2, 1]))
    return top


def configure_iga_penetrator(iga, implicit, dt, step_count, output_interval, output_path):
    from src.iga import Cube, DirichletBoundary, Primitives

    profile = np.asarray(PILE_PROFILE, dtype=np.float64)
    x_min, z_min = profile[0]
    x_max = float(np.max(profile[:, 0]))
    z_max = float(np.max(profile[:, 1]))
    iga.set_configuration(dimension=3, solver_type="Implicit" if implicit else "Explicit")
    pile = Cube()
    pile.set_parameters(
        start_point=[x_min, 0.0, z_min],
        size=[x_max - x_min, COUPLED_SOIL_SIZE[1], z_max - z_min],
    )
    pile.generate_knot_u(degree=2, num_ctrlpts=3)
    pile.generate_knot_v(degree=2, num_ctrlpts=3)
    pile.generate_knot_w(degree=2, num_ctrlpts=4)
    pile.generate_ctrlpts()
    pile.generate_weights()
    normalized_height = (pile.control_points[:, 2] - z_min) / (z_max - z_min)
    bottom_offset = (
        (pile.control_points[:, 0] - x_min) * (profile[1, 1] - profile[0, 1]) / (profile[1, 0] - profile[0, 0])
    )
    pile.control_points[:, 2] += (1.0 - normalized_height) * bottom_offset

    primitives = Primitives()
    primitives.append(pile, "cpt_penetrator", init_v=[0.0, 0.0, -PILE_SPEED])
    primitives.finialize()
    iga.add_primitives(primitives)
    all_control_points = np.arange(pile.control_points.shape[0], dtype=np.int32)
    fixed_dofs = [list(3 * all_control_points), list(3 * all_control_points + 1)]
    boundary = DirichletBoundary()
    boundary.append(fixed_dofs, [0.0] * (2 * all_control_points.size))
    iga.add_boundary_condition(dirichlet=boundary)
    iga.add_element(degree=[2, 2, 2])
    iga.add_material(
        young_modulus=PENETRATOR_MATERIAL["young_modulus"],
        poisson_ratio=PENETRATOR_MATERIAL["poisson_ratio"],
        density=PENETRATOR_MATERIAL["density"],
        gravity=list(COUPLED_GRAVITY),
    )
    iga.set_solver(
        dt=dt,
        step=step_count,
        interval=output_interval,
        path=str(output_path),
        newmark=[1.0, 0.5, 1.0],
        residual=5.0e-4,
        max_iters=35,
        assemble_type="HashTriplet",
        linear_solver="PCG",
        project_pd=True,
    )
    return pile.control_points.shape[0]


def explicit_contact_parameters(family):
    if str(family).lower() == "fempm":
        return dict(FEMPM_EXPLICIT_CONTACT)
    if str(family).lower() == "igampm":
        return dict(IGAMPM_EXPLICIT_CONTACT)
    raise ValueError("family must be FEMPM or IGAMPM")


def ipc_contact_parameters(grid_size):
    scale = float(grid_size) / GRID_SIZE
    return {
        "dhat": scale * IPC_CONTACT["dhat"],
        "dmin": scale * IPC_CONTACT["dmin"],
        "kappa": IPC_CONTACT["kappa"],
        "friction_coefficient": IPC_CONTACT["friction_coefficient"],
        "epsv": IPC_CONTACT["epsv"],
    }
