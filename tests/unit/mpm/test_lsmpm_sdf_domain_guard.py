import sys

import numpy as np


def test_soft_levelset_domain_margin_detects_deformed_body_escape(
    taichi_runtime,
):
    ti = taichi_runtime
    from src.dem.structs.BaseStruct import BoundingBox
    from src.mpm.soft_particle.SoftBodyKernel import (
        min_soft_levelset_domain_margin_,
    )
    from src.mpm.soft_particle.Structs import SoftBody

    soft = SoftBody.field(shape=1)
    box = BoundingBox.field(shape=1)

    @ti.kernel
    def set_bounds(
        shape_min_x: float,
        shape_max_x: float,
    ):
        soft[0].bodyID = 0
        box[0].grid_space = 0.25
        box[0].xmin = ti.Vector([-1.0, -1.0, -1.0])
        box[0].xmax = ti.Vector([1.0, 1.0, 1.0])
        box[0].shape_min = ti.Vector([shape_min_x, -0.75, -0.50])
        box[0].shape_max = ti.Vector([shape_max_x, 0.75, 0.50])

    set_bounds(-0.75, 0.75)
    np.testing.assert_allclose(
        min_soft_levelset_domain_margin_(1, soft, box),
        1.0,
        atol=1.0e-12,
    )

    set_bounds(-1.125, 0.75)
    np.testing.assert_allclose(
        min_soft_levelset_domain_margin_(1, soft, box),
        -0.5,
        atol=1.0e-12,
    )


def test_soft_levelset_domain_tolerance_configuration(taichi_runtime):
    from src.mpm.Simulation import Simulation

    simulation = Simulation()
    assert simulation.soft_levelset_domain_check
    simulation.set_soft_levelset_reinitialization(
        domain_check=False,
        domain_tolerance_cells=2.5e-6,
    )
    assert not simulation.soft_levelset_domain_check
    assert simulation.soft_levelset_domain_tolerance_cells == 2.5e-6


def test_soft_levelset_advection_configuration_defaults_to_semi_lagrangian(
    taichi_runtime,
):
    from src.dem.Simulation import Simulation as DEMSimulation
    from src.mpm.Simulation import Simulation

    simulation = Simulation()
    assert simulation.soft_levelset_advection_scheme == "SemiLagrangian"
    assert simulation.soft_levelset_advection_cfl == 1.0
    dem_simulation = DEMSimulation()
    assert dem_simulation.soft_levelset_advection_scheme == "SemiLagrangian"
    assert dem_simulation.soft_levelset_advection_cfl == 1.0

    simulation.set_soft_levelset_reinitialization(
        advection_scheme="weno5",
    )
    assert simulation.soft_levelset_advection_scheme == "WENO5"
    assert simulation.soft_levelset_advection_cfl == 0.20

    simulation.set_soft_levelset_reinitialization(
        advection_scheme="semi_lagrangian",
    )
    assert simulation.soft_levelset_advection_scheme == "SemiLagrangian"
    assert simulation.soft_levelset_advection_cfl == 1.0


def test_triaxial_levelset_extent_includes_verlet_and_deformation_reserve():
    from research.LSMPM.scripts.run_v5_mixture_triaxial import (
        levelset_extent_contract,
    )

    assert levelset_extent_contract(0.15, 1, 2) == (3, 1)
    assert levelset_extent_contract(0.10, 1, 2) == (4, 2)


def test_triaxial_formal_preset_requests_exactly_500_particles(monkeypatch):
    from research.LSMPM.scripts.run_v5_mixture_triaxial import (
        apply_preset,
        parse_args,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_v5_mixture_triaxial.py", "--preset", "paper", "--evidence-role", "formal"],
    )
    args = apply_preset(parse_args())
    assert args.body_count == 500
    assert args.minimum_body_count == 500
    assert args.side_count == 8
    assert args.packing == "random"


def test_triaxial_random_packing_honors_non_cubic_body_count():
    from research.LSMPM.scripts.validation_common import (
        make_random_nonoverlap_packing,
    )

    centers, radii, _ = make_random_nonoverlap_packing(
        500,
        0.025,
        np.asarray([0.10, 0.10, 0.10]),
        initial_solid_fraction=0.30,
        polydispersity=0.10,
        seed=20260712,
    )
    assert centers.shape == (500, 3)
    assert radii.shape == (500,)


def test_v5_capacity_uses_production_sphere_topology():
    from research.LSMPM.scripts.build_v5_capacity_protocol import (
        production_sphere_topology,
    )

    topology = production_sphere_topology(0.15, 5, 1)
    assert topology["levelset_nodes_per_body"] == 15625
    assert topology["logical_nodes_per_soft_body"] == 9261
    assert topology["compact_nodes_per_soft_body"] == 4913
    assert topology["levelset_verlet_padding_cells"] == 1
    assert topology["levelset_deformation_padding_cells"] == 4
