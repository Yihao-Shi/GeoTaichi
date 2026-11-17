import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import taichi as ti

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geotaichi import MPM, init


def build_case(args):
    init(
        dim=args.dim,
        arch=args.arch,
        default_fp=args.fp,
        default_ip="int32",
        device_memory_GB=args.device_memory_gb,
        offline_cache=True,
        log=False,
    )

    save_path = Path(args.save_path).resolve()
    if save_path.exists():
        shutil.rmtree(save_path)

    sparse_grid = False
    if args.sparse:
        sparse_grid = {
            "Enabled": True,
            "Backend": "BlockScan",
            "BlockSize": args.block_size,
            "CapacityFactor": args.capacity_factor,
            "MaxActiveBlocks": args.max_active_blocks,
        }

    domain = ti.Vector([args.domain, args.domain, args.domain]) if args.dim == 3 else ti.Vector([args.domain, args.domain])
    gravity = [0.0, 0.0, -9.8] if args.dim == 3 else [0.0, -9.8]
    element_size = ti.Vector([args.dx, args.dx, args.dx]) if args.dim == 3 else ti.Vector([args.dx, args.dx])
    region_point = ti.Vector([0.25, 0.25, 0.25]) if args.dim == 3 else ti.Vector([0.25, 0.25])
    region_size = ti.Vector([args.body, args.body, args.body]) if args.dim == 3 else ti.Vector([args.body, args.body])
    initial_velocity = ti.Vector([0.0, 0.0, 0.0]) if args.dim == 3 else ti.Vector([0.0, 0.0])
    fix_velocity = ["Free", "Free", "Free"] if args.dim == 3 else ["Free", "Free"]
    boundary_end = [args.domain, args.domain, 0.0] if args.dim == 3 else [args.domain, 0.0]
    element_type = "R8N3D" if args.dim == 3 else "Q4N2D"
    particle_traction_method = "Virtual" if args.virtual_traction else "Stable"

    mpm = MPM(log=False)
    mpm.set_configuration(
        log=False,
        domain=domain,
        configuration=args.configuration,
        gravity=gravity,
        background_damping=0.0,
        alphaPIC=0.0,
        mapping="USL",
        stabilize=None,
        shape_function=args.shape_function,
        velocity_projection=args.velocity_projection,
        sparse_grid=sparse_grid,
        neighbor_detection=args.neighbor_detection,
        free_surface_detection=args.free_surface_detection,
        particle_traction_method=particle_traction_method,
    )
    mpm.set_solver(
        log=False,
        solver={
            "Timestep": args.dt,
            "SimulationTime": args.dt * args.steps,
            "SaveInterval": args.dt * (args.steps + 10),
            "SavePath": str(save_path),
        },
    )
    mpm.memory_allocate(
        log=False,
        memory={
            "max_material_number": 1,
            "max_particle_number": args.max_particles,
            "max_constraint_number": {
                "max_velocity_constraint": args.max_constraints,
                "max_particle_traction_constraint": args.max_particle_traction_constraints,
            },
        },
    )
    mpm.add_material(
        model="LinearElastic",
        material={
            "MaterialID": 1,
            "Density": 2650.0,
            "YoungModulus": 2.0e7,
            "PoissonRatio": 0.3,
        },
    )
    element = {
        "ElementType": element_type,
        "ElementSize": element_size,
    }
    if args.adaptive:
        element["AdaptiveGrid"] = {
            "MaxLevel": args.adaptive_max_level,
            "RefineInterval": args.adaptive_refine_interval,
            "RefineThreshold": args.adaptive_refine_threshold,
            "RefineCriterion": args.adaptive_refine_criterion,
            "RefineRatio": args.adaptive_refine_ratio,
            "RefineParticles": args.adaptive_refine_particles,
            "HangingConstraintMode": args.adaptive_hanging_mode,
            "ParticleSplitBatch": args.adaptive_particle_split_batch,
        }
    mpm.add_element(element=element)
    mpm.add_region(
        region={
            "Name": "bench_region",
            "Type": "Rectangle" if args.dim == 3 else "Rectangle2D",
            "BoundingBoxPoint": region_point,
            "BoundingBoxSize": region_size,
        },
    )
    mpm.add_body(
        body={
            "Template": {
                "RegionName": "bench_region",
                "nParticlesPerCell": args.ppc,
                "BodyID": 0,
                "MaterialID": 1,
                "InitialVelocity": initial_velocity,
                "FixVelocity": fix_velocity,
            },
        },
    )
    mpm.add_boundary_condition(
        boundary=[
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0 for _ in range(args.dim)],
                "StartPoint": [0.0 for _ in range(args.dim)],
                "EndPoint": boundary_end,
            },
        ],
    )
    if args.particle_traction:
        traction = [0.0 for _ in range(args.dim)]
        traction[-1] = args.particle_traction_pressure
        mpm.add_particle_traction({"Pressure": traction})
    if args.virtual_traction:
        stress = [-args.virtual_confining_pressure for _ in range(3)] + [0.0, 0.0, 0.0]
        force = [0.0 for _ in range(args.dim)]
        mpm.add_virtual_stress_field(
            field={
                "ConfiningPressure": stress,
                "VirtualForce": force,
            }
        )
    mpm.select_save_data(particle=False, grid=False, object=False)
    return mpm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="gpu")
    parser.add_argument("--dim", type=int, default=3, choices=(2, 3))
    parser.add_argument("--fp", default="float32", choices=("float32", "float64"))
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--configuration", default="ULMPM", choices=("ULMPM", "TLMPM"))
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--adaptive-max-level", type=int, default=1)
    parser.add_argument("--adaptive-refine-interval", type=int, default=1)
    parser.add_argument("--adaptive-refine-threshold", type=float, default=1.0e-9)
    parser.add_argument("--adaptive-refine-criterion", default="EquivalentStress")
    parser.add_argument("--adaptive-refine-ratio", type=float, default=0.25)
    parser.add_argument("--adaptive-refine-particles", action="store_true")
    parser.add_argument("--adaptive-hanging-mode", default="Penalty")
    parser.add_argument("--adaptive-particle-split-batch", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--max-active-blocks", type=int, default=0)
    parser.add_argument("--domain", type=float, default=6.0)
    parser.add_argument("--body", type=float, default=1.0)
    parser.add_argument("--dx", type=float, default=0.05)
    parser.add_argument("--ppc", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=2.0e-5)
    parser.add_argument("--max-particles", type=int, default=100000)
    parser.add_argument("--max-constraints", type=int, default=20000)
    parser.add_argument("--max-particle-traction-constraints", type=int, default=0)
    parser.add_argument("--shape-function", default="Linear")
    parser.add_argument("--velocity-projection", default="PIC/FLIP")
    parser.add_argument("--particle-traction", action="store_true")
    parser.add_argument("--particle-traction-pressure", type=float, default=-1000.0)
    parser.add_argument("--virtual-traction", action="store_true")
    parser.add_argument("--virtual-confining-pressure", type=float, default=1000.0)
    parser.add_argument("--neighbor-detection", action="store_true")
    parser.add_argument("--free-surface-detection", action="store_true")
    parser.add_argument("--device-memory-gb", type=float, default=4.0)
    parser.add_argument("--save-path", default="/tmp/geotaichi_sparse_benchmark")
    args = parser.parse_args()

    setup_start = time.perf_counter()
    mpm = build_case(args)
    setup_seconds = time.perf_counter() - setup_start

    run_start = time.perf_counter()
    mpm.run(gravity_field=True)
    ti.sync()
    run_seconds = time.perf_counter() - run_start

    sparse_info = {}
    if mpm.scene.sparse_grid is not None:
        sparse_info = mpm.scene.sparse_grid.describe(mpm.scene.node_slot_bytes)
    elif args.sparse and getattr(mpm.scene.element, "adaptive", False):
        dense_node_slots = int(mpm.scene.element.logical_grid_sum)
        if getattr(mpm.scene.element, "bridging_domain", False):
            dense_node_slots += int(mpm.scene.element.node_map.coarse_node_count)
        sparse_info = {
            "backend": "AdaptiveNodeMap",
            "dense_node_slots": dense_node_slots,
            "allocated_node_slots": int(mpm.scene.element.gridSum),
            "allocated_node_slot_ratio": int(mpm.scene.element.gridSum) / max(1, dense_node_slots),
        }

    adaptive_info = {}
    if getattr(mpm.scene.element, "adaptive", False):
        element = mpm.scene.element
        logical_node_slots = int(getattr(element, "logical_grid_sum", 0))
        compact_node_slots = int(getattr(element, "gridSum", 0))
        coarse_cells = int(getattr(element, "cellSum", 0))
        refined_cells = int(getattr(element, "last_reported_refined_cells", 0))
        leaf_count_by_level = {}
        if getattr(element, "refined_cell", None) is not None:
            refined_cell = element.refined_cell.to_numpy()
            for level in range(int(getattr(element, "max_level", 1)) + 1):
                leaf_count_by_level[str(level)] = int((refined_cell == level).sum())
        adaptive_info = {
            "max_level": int(getattr(element, "max_level", 1)),
            "logical_node_slots": logical_node_slots,
            "compact_node_slots": compact_node_slots,
            "node_capacity_ratio": compact_node_slots / max(1, logical_node_slots),
            "coarse_cells": coarse_cells,
            "refined_cells": refined_cells,
            "leaf_count_by_level": leaf_count_by_level,
            "refined_cell_ratio": refined_cells / max(1, coarse_cells),
            "max_refined_cells": int(getattr(element, "max_refined_cells", 0)),
            "max_refined_extra_cells": int(getattr(element, "max_refined_extra_cells", 0)),
            "refine_ratio": float(getattr(element, "max_refined_ratio", 0.0)),
            "remaining_unrefined_particles": int(getattr(element, "remaining_unrefined_particles", -1)),
        }

    result = {
        "sparse": args.sparse,
        "arch": args.arch,
        "dim": args.dim,
        "fp": args.fp,
        "shape_function": args.shape_function,
        "velocity_projection": args.velocity_projection,
        "configuration": args.configuration,
        "adaptive": args.adaptive,
        "particle_traction": args.particle_traction,
        "virtual_traction": args.virtual_traction,
        "neighbor_detection": bool(mpm.sims.neighbor_detection),
        "free_surface_detection": bool(mpm.sims.free_surface_detection),
        "steps": args.steps,
        "domain": args.domain,
        "body": args.body,
        "dx": args.dx,
        "ppc": args.ppc,
        "particles": int(mpm.scene.particleNum[0]),
        "grid_sum": int(mpm.scene.element.gridSum),
        "setup_seconds": setup_seconds,
        "run_seconds": run_seconds,
        "total_seconds": setup_seconds + run_seconds,
        "sparse_info": sparse_info,
        "adaptive_info": adaptive_info,
    }
    print("SPARSE_BENCHMARK_RESULT " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
