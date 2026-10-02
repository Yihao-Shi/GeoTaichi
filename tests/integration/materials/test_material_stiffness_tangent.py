import numpy as np
import taichi as ti


import src.utils.GlobalVariable as GlobalVariable
from src.mpm.MaterialManager import MaterialHandle
from src.mpm.structs import ParticleCloud


class _Sims:
    def __init__(self):
        self.max_material_num = 2
        self.material_type = "Solid"
        self.contact_detection = None
        self.max_particle_num = 8
        self.configuration = "ULMPM"
        self.solver_type = "Implicit"
        self.random_field = False
        self.stress_integration = "ReturnMapping"


@ti.func
def _engineering_gradient(comp: int, eps: float):
    velocity_gradient = ti.Matrix.zero(float, 3, 3)
    if comp == 0:
        velocity_gradient[0, 0] = eps
    elif comp == 1:
        velocity_gradient[1, 1] = eps
    elif comp == 2:
        velocity_gradient[2, 2] = eps
    elif comp == 3:
        velocity_gradient[0, 1] = eps
    elif comp == 4:
        velocity_gradient[1, 2] = eps
    elif comp == 5:
        velocity_gradient[0, 2] = eps
    return velocity_gradient


@ti.kernel
def _initialize_particles(particle: ti.template(), stress: ti.types.vector(6, float), total: int):
    for np in range(total):
        particle[np].particleID = np
        particle[np].active = ti.u8(1)
        particle[np].bodyID = ti.u8(0)
        particle[np].materialID = ti.u8(1)
        particle[np].m = 1.0
        particle[np].vol = 1.0
        particle[np].stress = stress
        particle[np].velocity_gradient = ti.Matrix.zero(float, 3, 3)


@ti.kernel
def _evaluate_tangent_columns(matProps: ti.template(), stateVars: ti.template(), particle: ti.template(), dt: ti.template(),
                              eps: float, tangent: ti.template(), finite_difference: ti.template()):
    zero_gradient = ti.Matrix.zero(float, 3, 3)
    base_stress = matProps.ComputeStress(0, particle[0].stress, zero_gradient, stateVars, dt)
    tangent[None] = matProps.compute_stiffness_tensor(0, base_stress, stateVars)
    for comp in range(6):
        velocity_gradient = _engineering_gradient(comp, eps)
        stress = matProps.ComputeStress(comp + 1, particle[comp + 1].stress, velocity_gradient, stateVars, dt)
        for row in ti.static(range(6)):
            finite_difference[None][row, comp] = (stress[row] - base_stress[row]) / eps


@ti.kernel
def _evaluate_preloaded_tangent_columns(matProps: ti.template(), stateVars: ti.template(), particle: ti.template(), dt: ti.template(),
                                        preload: ti.types.matrix(3, 3, float), eps: float,
                                        tangent: ti.template(), finite_difference: ti.template()):
    zero_gradient = ti.Matrix.zero(float, 3, 3)
    for np in range(7):
        particle[np].stress = matProps.ComputeStress(np, particle[np].stress, preload, stateVars, dt)

    base_stress = matProps.ComputeStress(0, particle[0].stress, zero_gradient, stateVars, dt)
    tangent[None] = matProps.compute_stiffness_tensor(0, base_stress, stateVars)
    for comp in range(6):
        velocity_gradient = _engineering_gradient(comp, eps)
        stress = matProps.ComputeStress(comp + 1, particle[comp + 1].stress, velocity_gradient, stateVars, dt)
        for row in ti.static(range(6)):
            finite_difference[None][row, comp] = (stress[row] - base_stress[row]) / eps


def _make_material(model, material):
    sims = _Sims()
    material_handle = MaterialHandle(sims)
    material_handle.setup(sims, None, model, material)
    material_handle.activate_state_variables(sims)
    particle = ParticleCloud.field(shape=sims.max_particle_num)
    return sims, material_handle, particle


def _relative_error(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0e-30)


def test_linear_elastic_stiffness_matches_stress_update_difference():
    GlobalVariable.DIMENSION = 3
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1, offline_cache=False)
    _, material, particle = _make_material(
        "LinearElastic",
        {
            "MaterialID": 1,
            "Density": 1800.0,
            "YoungModulus": 2.0e5,
            "PoissonRatio": 0.3,
        },
    )
    stress = ti.Vector([-1200.0, -1000.0, -900.0, 10.0, -5.0, 7.0])
    _initialize_particles(particle, stress, 7)
    material.state_vars_initialize(1, 0, 7, particle)

    dt = ti.field(float, shape=())
    dt[None] = 1.0
    tangent = ti.Matrix.field(6, 6, float, shape=())
    finite_difference = ti.Matrix.field(6, 6, float, shape=())
    _evaluate_tangent_columns(material.matProps[1], material.stateVars, particle, dt, 1.0e-4, tangent, finite_difference)

    tangent_np = tangent.to_numpy()[()]
    fd_np = finite_difference.to_numpy()[()]
    rel = _relative_error(tangent_np, fd_np)
    print(f"LinearElastic tangent relative error vs finite difference: {rel:.6e}")
    assert rel < 1.0e-3


def _assert_tangent_matches_finite_difference(model, material, stress, eps=1.0e-6, tolerance=1.0e-3):
    _, material_handle, particle = _make_material(model, material)
    _initialize_particles(particle, stress, 7)
    material_handle.state_vars_initialize(1, 0, 7, particle)

    dt = ti.field(float, shape=())
    dt[None] = 1.0
    tangent = ti.Matrix.field(6, 6, float, shape=())
    finite_difference = ti.Matrix.field(6, 6, float, shape=())
    _evaluate_tangent_columns(material_handle.matProps[1], material_handle.stateVars, particle, dt, eps, tangent, finite_difference)

    tangent_np = tangent.to_numpy()[()]
    fd_np = finite_difference.to_numpy()[()]
    rel = _relative_error(tangent_np, fd_np)
    print(f"{model} tangent relative error vs finite difference: {rel:.6e}")
    assert np.isfinite(rel)
    assert rel < tolerance


def test_plastic_material_elastic_branch_stiffness_matches_stress_update_difference():
    GlobalVariable.DIMENSION = 3
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1, offline_cache=False)
    stress = ti.Vector([-2000.0, -1800.0, -1600.0, 20.0, -10.0, 7.0])
    common = {
        "MaterialID": 1,
        "Density": 1800.0,
        "YoungModulus": 2.0e5,
        "PoissonRatio": 0.3,
    }
    cases = [
        (
            "ElasticPerfectlyPlastic",
            {
                **common,
                "YieldStress": 1.0e8,
            },
        ),
        (
            "MohrCoulomb",
            {
                **common,
                "Cohesion": 1.0e8,
                "Friction": 30.0,
                "Dilation": 0.0,
                "Tensile": 1.0e8,
            },
        ),
        (
            "DruckerPrager",
            {
                **common,
                "Cohesion": 1.0e8,
                "Friction": 30.0,
                "Dilation": 0.0,
                "Tensile": 1.0e8,
                "dpType": "MiddleCircumscribed",
            },
        ),
    ]
    for model, material in cases:
        _assert_tangent_matches_finite_difference(model, material, stress, eps=1.0e-6, tolerance=1.0e-3)


def test_modified_cam_clay_stiffness_matches_stress_update_difference():
    GlobalVariable.DIMENSION = 3
    ti.init(
        arch=ti.cpu,
        default_fp=ti.f64,
        cpu_max_num_threads=1,
        offline_cache=False,
    )
    _, material, particle = _make_material(
        "ModifiedCamClay",
        {
            "MaterialID": 1,
            "Density": 1800.0,
            "PoissonRatio": 0.3,
            "StressRatio": 1.2,
            "lambda": 0.12,
            "kappa": 0.03,
            "void_ratio_ref": 1.1,
            "pressure_ref": 1000.0,
            "ConsolidationPressure": 120000.0,
        },
    )
    stress = ti.Vector([-100000.0, -100000.0, -100000.0, 0.0, 0.0, 0.0])
    _initialize_particles(particle, stress, 7)
    material.state_vars_initialize(1, 0, 7, particle)

    dt = ti.field(float, shape=())
    dt[None] = 1.0
    tangent = ti.Matrix.field(6, 6, float, shape=())
    finite_difference = ti.Matrix.field(6, 6, float, shape=())
    preload = ti.Matrix([[-8.0e-4, 2.0e-4, 0.0], [0.0, 3.0e-4, 1.0e-4], [0.0, 0.0, 2.0e-4]])
    _evaluate_preloaded_tangent_columns(material.matProps[1], material.stateVars, particle, dt, preload, 1.0e-4, tangent, finite_difference)

    tangent_np = tangent.to_numpy()[()]
    fd_np = finite_difference.to_numpy()[()]
    rel = _relative_error(tangent_np, fd_np)
    assert np.isfinite(rel)
    print(f"ModifiedCamClay tangent relative error vs finite difference: {rel:.6e}")
    assert rel < 1.0e-2


if __name__ == "__main__":
    test_linear_elastic_stiffness_matches_stress_update_difference()
    test_plastic_material_elastic_branch_stiffness_matches_stress_update_difference()
    test_modified_cam_clay_stiffness_matches_stress_update_difference()
