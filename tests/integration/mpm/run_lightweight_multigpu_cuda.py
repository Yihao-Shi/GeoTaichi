"""Run once in Python, then with torchrun against the single-GPU result."""

import argparse
import os
from pathlib import Path

import numpy as np


def build_model(output, owned, args):
    standalone, dimension, axisymmetric = args.standalone, args.dimension, args.axisymmetric
    import geotaichi as gt

    fluid = args.stabilize == "Displacement F-Bar Method"

    if standalone:
        model = mpm = gt.MPM(log=False)
    else:
        model = gt.MPDEM(log=False)
        mpm = model.mpm
        model.set_configuration(domain=[2.0] * 3, gravity=[0.0] * 3, visualize=False, log=False)
    mpm.set_configuration(
        dimension=dimension,
        domain=[2.0] * dimension,
        mode=args.mode,
        material_type="Fluid" if fluid else "Solid",
        axisymmetric=axisymmetric,
        configuration=args.configuration,
        solver_type="Explicit",
        mapping=args.mapping,
        shape_function=args.shape_function,
        velocity_projection=args.projection,
        stabilize=args.stabilize,
        gauss_number=2 if args.gauss else 0,
        pressure_smoothing=args.pressure_smoothing,
        moving_least_square=args.mls,
        gravity=[0.0] * dimension,
        boundary_direction_detection=not standalone,
        log=False,
    )
    dt = 1e-6
    solver = dict(Timestep=dt, SimulationTime=4 * dt, SaveInterval=1.0, SavePath=str(output))
    if args.checkpoint or args.restart_dir:
        solver["SaveInterval"] = 2 * dt
    elif standalone and args.mode == "Lightweight":
        # The fastest shard determines the first step through the CFL check.
        solver.update(CFL=3.2e-5, AdaptiveStep=1)
    if not standalone:
        model.dem.set_configuration(scheme="LSDEM", search="LinkedCell", gravity=[0.0] * 3, log=False)
    model.set_solver(solver, log=False)
    mpm.memory_allocate(
        dict(
            max_material_number=1,
            max_particle_number=len(owned),
            verlet_distance_multiplier=0.4,
            max_constraint_number={},
        ),
        log=False,
    )
    if fluid:
        mpm.add_material("Newtonian", dict(MaterialID=1, Density=1000.0, Modulus=1e4, Viscosity=1e-3))
    else:
        mpm.add_material("LinearElastic", dict(MaterialID=1, Density=1000.0, YoungModulus=1.0e4, PoissonRatio=0.3))
    element_type = "Q4N2D" if dimension == 2 else "R8N3D"
    if args.contact == "DEMContact":
        mpm.add_contact(args.contact, contact=dict(materialID=1, stiffness=[1e4, 1e3], friction=0.3))
    elif args.contact:
        mpm.add_contact(args.contact, friction=0.3)
    mpm.add_element(element=dict(ElementType=element_type, ElementSize=[0.1] * dimension))
    mpm.add_region(
        region=dict(
            Name="points",
            Type="Rectangle2D" if dimension == 2 else "Rectangle",
            BoundingBoxPoint=[0.5, 0.45, 0.45][:dimension],
            BoundingBoxSize=[0.1] * dimension,
        )
    )
    for local_index, global_index in enumerate(owned if args.restart_dir is None else []):
        body_id = global_index % 2 if args.contact in ("MPMContact", "GeoContact") else 0
        rigid = args.contact == "GeoContact" and body_id == 1
        mpm.add_body(
            body=dict(
                Template=dict(
                    RegionName="points",
                    nParticlesPerCell=1,
                    BodyID=body_id,
                    MaterialID=1,
                    RigidBody=rigid,
                )
            )
        )
        particle = mpm.scene.particle[local_index]
        particle.x = [0.51 + global_index * 0.005, 0.48 + global_index * 0.004, 0.52][:dimension]
        particle.v = [0.1 if axisymmetric else 0.0, (global_index - 3.5) * 0.02, 0.0][:dimension]
        if args.mode == "Normal":
            x = 0.52 + 0.018 * body_id
            particle.x = [x, 0.48 + 0.003 * (global_index // 2), 0.52][:dimension]
            axial_velocity = (global_index - 3.5) * (-0.02 if fluid else 0.02)
            particle.v = [0.3 if body_id == 0 else 0.0, axial_velocity, 0.0][:dimension]
            if not rigid and not fluid:
                if args.configuration == "TLMPM":
                    particle.stress = np.diag([-200.0 * global_index, 0.0, 0.0])
                else:
                    particle.stress = [-200.0 * global_index, 0.0, 0.0, 0.0, 0.0, 0.0]
            if args.gauss:
                particle.x = [0.525 + 0.05 * ((global_index >> i) & 1) for i in range(dimension)]
                particle.vol = 0.1**dimension / 8
                particle.m = 1000.0 * particle.vol
        if axisymmetric:
            hoop_stress = 3000.0 + 100.0 * global_index if args.mode == "Normal" else 3000.0
            particle.stress = [0.0, 0.0, hoop_stress, 0.0, 0.0, 0.0]
    if args.contact == "DEMContact":
        mpm.add_polygons(
            body=dict(
                Vertices=np.array([[0.3, 0.3], [0.9, 0.3], [0.9, 0.49], [0.3, 0.49]]),
                InitialVelocity=[0.0, 0.1],
            )
        )
    mpm.add_boundary_condition()
    mpm.select_save_data(particle=bool(args.checkpoint or args.restart_dir), grid=False, object=False)
    if args.restart_dir:
        mpm.read_restart(1, str(args.restart_dir / output.name))
    if not standalone:
        add_lsdem(model, output, save_contact=bool(args.checkpoint or args.restart_dir))
        if args.restart_dir:
            model.dem.sims.set_is_continue(True)
            model.dem.sims.current_print = 1
            model.dem.sims.current_time = mpm.sims.current_time
            model.read_restart(1, str(args.restart_dir / output.name), ppcontact=True)
    return model


def add_lsdem(model, output, save_contact=False):
    import geotaichi as gt
    import trimesh

    dem = model.dem
    dem.memory_allocate(
        dict(
            max_material_number=1,
            max_rigid_body_number=1,
            levelset_grid_number=60000,
            surface_node_number=500,
            body_coordination_number=0,
            wall_coordination_number=0,
            point_coordination_number=[0, 0],
        ),
        log=False,
    )
    model.memory_allocate(
        dict(body_coordination_number=1, wall_coordination_number=0, compaction_ratio=[1.0, 0.1]), log=False
    )
    dem.add_attribute(0, dict(Density=1000.0, ForceLocalDamping=0.0, TorqueLocalDamping=0.0))
    output.mkdir(parents=True, exist_ok=True)
    mesh = output / "sphere.obj"
    trimesh.creation.icosphere(subdivisions=1, radius=0.25).export(mesh)
    dem.add_template(dict(Name="sphere", Object=gt.polyhedron(file=str(mesh)).grids(space=0.025, extent=8)))
    dem.create_body(
        dict(
            GenerateType="Create",
            BodyType="RigidBody",
            Template=dict(
                Name="sphere",
                GroupID=0,
                MaterialID=0,
                BodyPoint=[0.8, 0.5, 0.5],
                ScaleFactor=1.0,
                BodyOrientation="constant",
            ),
        )
    )
    dem.choose_contact_model(None, None)
    dem.select_save_data(surface=False)
    model.choose_contact_model("Linear Model")
    model.add_property(
        0,
        1,
        dict(
            NormalStiffness=1.0e5,
            TangentialStiffness=1.0e5,
            Friction=0.25,
            NormalViscousDamping=0.0,
            TangentialViscousDamping=0.0,
        ),
    )
    model.select_save_data(particle_particle_contact=save_contact)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--standalone", action="store_true")
    parser.add_argument("--dimension", type=int, choices=(2, 3), default=3)
    parser.add_argument("--axisymmetric", action="store_true")
    parser.add_argument("--mode", choices=("Lightweight", "Normal"), default="Lightweight")
    parser.add_argument("--mapping", choices=("USL", "USF", "MUSL"), default="USL")
    parser.add_argument("--configuration", choices=("ULMPM", "TLMPM"), default="ULMPM")
    parser.add_argument("--projection", choices=("Taylor", "Affine"))
    parser.add_argument("--shape-function", default="Linear")
    parser.add_argument("--stabilize", choices=("B-Bar Method", "F-Bar Method", "Displacement F-Bar Method"))
    parser.add_argument("--gauss", action="store_true")
    parser.add_argument("--pressure-smoothing", action="store_true")
    parser.add_argument("--mls", action="store_true")
    parser.add_argument("--contact", choices=("MPMContact", "GeoContact", "DEMContact"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compare-with", type=Path)
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument("--checkpoint", action="store_true")
    restart.add_argument("--restart-dir", type=Path)
    args = parser.parse_args()
    if args.dimension == 2 and not args.standalone:
        parser.error("2D requires --standalone")
    if args.axisymmetric and args.dimension != 2:
        parser.error("--axisymmetric requires --dimension 2")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    assert 1 <= world <= 8
    os.environ["GEOTAICHI_REAL_DTYPE"] = "float64"

    import geotaichi as gt
    import taichi as ti
    from src.utils import GlobalVariable

    if world > 1:
        gt.init(dim=args.dimension, arch="gpu", distributed=True, log=False, offline_cache=False)
        import torch.distributed as dist
    else:
        GlobalVariable.DIMENSION = args.dimension
        ti.init(arch=ti.cuda, default_fp=ti.f64, enable_fallback=False, offline_cache=False)
    try:
        owned = np.arange(rank, 8, world)
        output = args.output_dir / f"rank-{rank:02d}"
        model = build_model(output, owned, args)
        mpm = model if args.standalone else model.mpm
        particle = mpm.scene.particle
        rigid = None if args.standalone else model.dem.scene.rigid
        mass = particle.m.to_numpy()[: len(owned)]
        initial_momentum = (mass[:, None] * particle.v.to_numpy()[: len(owned)]).sum(axis=0)
        initial_radius = particle.x.to_numpy()[: len(owned), 0]
        particle_fields = ("x", "v", "stress", "vol")
        if rigid is not None:
            particle_fields += ("external_force",)
        trajectory = []

        @gt.python_callback
        def record_step():
            state = {name: getattr(particle, name).to_numpy()[: len(owned)] for name in particle_fields}
            state["next_dt"] = model.sims.delta
            if args.contact:
                state["grid_contact"] = mpm.scene.node.contact_force.to_numpy()
            if args.stabilize in ("F-Bar Method", "Displacement F-Bar Method"):
                state["grid_jacobian"] = mpm.scene.node.jacobian.to_numpy()
            if args.pressure_smoothing or args.stabilize == "Displacement F-Bar Method":
                pressure = mpm.scene.node.pressure.to_numpy()
                if args.stabilize != "Displacement F-Bar Method" and mpm.sims.current_step % 20 != 0:
                    pressure.fill(0)  # The smoothing pass did not write this scratch field this step.
                state["grid_pressure"] = pressure
            if args.gauss:
                state["grid_gauss_active"] = mpm.scene.element.cell.active.to_numpy()
            if rigid is not None:
                state.update(
                    {
                        f"rigid_{name}": getattr(rigid, name).to_numpy()[0]
                        for name in ("mass_center", "q", "v", "w", "contact_force", "contact_torque")
                    }
                )
            trajectory.append(state)

        if rigid is not None and (args.checkpoint or args.restart_dir):
            # Preserve the generated rigid geometry, as for a prescribed excavator body.
            model.add_essentials()
            model.enginer.dengine.integration = lambda *unused: None
        start_step = 2 if args.restart_dir else 0
        assert model.sims.current_step == start_step
        model.run(function=record_step)
        assert model.sims.current_step == start_step + len(trajectory) >= 4
        if args.checkpoint or args.restart_dir:
            assert model.sims.current_step == 4
        np.testing.assert_allclose(model.sims.current_time, model.sims.time, rtol=1e-12, atol=0.0)
        if args.standalone and args.mode == "Lightweight" and not (args.checkpoint or args.restart_dir):
            assert 0.0 < model.sims.init_delta < 1e-6
        state = {name: np.stack([step[name] for step in trajectory]) for name in trajectory[0]}
        impulse = (mass[None, :, None] * state["v"]).sum(axis=1) - initial_momentum
        if args.axisymmetric and args.mode == "Lightweight":
            radii = np.vstack((initial_radius, state["x"][:-1, :, 0]))
            hoop_force = -(state["vol"] * state["stress"][:, :, 2] / radii).sum(axis=1)
            step_dt = np.r_[model.sims.init_delta, state["next_dt"][:-1]]
            step_dt[-1] = model.sims.time - step_dt[:-1].sum()
            hoop_impulse = (hoop_force * step_dt).cumsum()
            assert np.all(hoop_impulse < 0)
            impulse[:, 0] -= hoop_impulse
        if world > 1:
            import torch

            total = torch.tensor(impulse, device="cuda")
            dist.all_reduce(total)
            impulse = total.cpu().numpy()
        if rigid is not None:
            assert np.linalg.norm(state["rigid_contact_force"][-1]) > 0.0
            assert np.linalg.norm(state["rigid_contact_torque"][-1]) > 0.0
            impulse += rigid[0].m * state["rigid_v"]
        if (args.standalone or not (args.checkpoint or args.restart_dir)) and (
            (not args.contact and not args.axisymmetric) or args.mode == "Lightweight"
        ):
            np.testing.assert_allclose(impulse, 0.0, atol=2e-9, rtol=0.0)
        for name in ("grid_contact", "grid_jacobian", "grid_pressure"):
            if name in state:
                assert np.any(state[name] != 0), name
        if args.gauss:
            assert np.count_nonzero(state["grid_gauss_active"][0]) == 1
        assert all(np.isfinite(value).all() for value in state.values())
        if args.checkpoint and rigid is not None:
            with np.load(output / "DEMPMcontacts" / "DEMPMContactPP000001.npz") as saved:
                assert np.any(saved["oldTangentialOverlap"] != 0)
        if args.restart_dir:
            with np.load(args.restart_dir / output.name / "state.npz") as reference:
                for name, value in state.items():
                    np.testing.assert_allclose(value, reference[name][start_step:], rtol=2e-7, atol=2e-8, err_msg=name)
            assert (output / "particles" / "MPMParticle000002.npz").is_file()
        if args.compare_with:
            with np.load(args.compare_with) as reference:
                for name, value in state.items():
                    replicated = name.startswith(("rigid_", "grid_")) or name == "next_dt"
                    expected = reference[name] if replicated else reference[name][:, owned]
                    atol = 0.0 if name == "next_dt" else 2e-8
                    np.testing.assert_allclose(value, expected, rtol=2e-7, atol=atol, err_msg=name)
        np.savez(output / "state.npz", **state)
        print(f"CUDA rank {rank}/{world}: {len(trajectory)} steps validated")
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
