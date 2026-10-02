import numpy as np
import taichi as ti

from geotaichi import init
from src.mpm.boundaries.BoundaryCore import apply_particle_traction_constraint_twophase
from src.mpm.boundaries.BoundaryStrcut import ParticleLoadTwoPhase2D
from src.mpm.engines.TwoPhaseDoubleLayerKernel import (
    FLUID_CELL,
    SOLID_CELL,
    AIR_CELL,
    MAC_SHAPE_LINEAR,
    PHASE_FLUID,
    PHASE_SOLID,
    _drag_accelerations_props,
    kernel_build_double_layer_fluid_sdf2d,
    kernel_classify_double_layer_fluid_cells2d,
    kernel_enforce_double_layer_solid_plane_nodes2d,
    kernel_enforce_double_layer_solid_plane_nodes3d,
    kernel_enforce_double_layer_solid_cell_faces2d,
    kernel_mac_p2g_double_layer2d,
    kernel_reset_double_layer_grid,
    kernel_normalize_double_layer_mac_fields2d,
    kernel_assemble_double_layer_pressure_A,
    kernel_assemble_double_layer_pressure_mg_A,
    kernel_assemble_double_layer_pressure_rhs,
    kernel_correct_double_layer_velocity2d,
    kernel_coarsen_double_layer_grid_type2d,
    kernel_correct_double_layer_solid_velocity_paper2d,
    kernel_delta_correct_double_layer_fluid2d,
    kernel_mark_double_layer_solid_cell_region2d,
    kernel_project_double_layer_pressure_to_solid_nodes2d,
    kernel_sample_double_layer_solid_pressure_from_nodes2d,
    kernel_update_double_layer_solid_state2d,
    kernel_update_double_layer_fluid_volume2d,
    kernel_volume_p2g_double_layer_fluid2d,
    _sample_cell_pressure_gfm_shape2d,
    _sample_mac_velocity,
    _sample_mac_velocity3d,
)
from src.mpm.structs.GridNode import NodeTwoPhase, NodeTwoPhase2D
from src.mpm.structs.Particle import ParticleCloudTwoPhase2D
from src.physics_model.consititutive_model.infinitesimal_strain.MohrCoulomb import MohrCoulombModel
from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import ElasticTensorMultiplyVector
from src.utils.TypeDefination import mat2x2, vec2f, vec3f

NX = 5
NY = 6
DX = 0.2
DY = 0.25
PHI = 0.4
RHO_S = 2650.0
RHO_F = 1000.0
DT_VALUE = 1.0e-4


@ti.dataclass
class _MohrCoulombState:
    epdstrain: float


@ti.dataclass
class _VolumeNode:
    porosity: float
    vol: float


@ti.dataclass
class _VolumeParticle:
    active: ti.u8
    materialID: ti.u8
    bodyID: ti.u8
    phase: ti.u8
    m: float
    mf: float
    vol: float
    porosity: float
    rad: float
    x: vec2f
    grad_E2: vec2f


@ti.kernel
def _setup_double_layer_volume_correction(
    node: ti.template(),
    particle: ti.template(),
    material_mapping: ti.template(),
    lnid: ti.template(),
    shape_fn: ti.template(),
    dshape_fn: ti.template(),
    node_size: ti.template(),
):
    for I in ti.grouped(node):
        node[I].porosity = 1.0
        node[I].vol = 0.0
    for p in range(2):
        material_mapping[p] = p
        particle[p].active = ti.u8(1)
        particle[p].materialID = ti.u8(1)
        particle[p].bodyID = ti.u8(0)
        particle[p].phase = ti.u8(PHASE_SOLID)
        if p == 0:
            particle[p].phase = ti.u8(PHASE_FLUID)
        particle[p].vol = 0.01
        particle[p].porosity = 0.39
        particle[p].x = vec2f(0.5, 0.5 + 0.1 * p)
        lnid[p] = 4
        shape_fn[p] = 1.0
        dshape_fn[p] = vec2f(10.0, 0.0)
        node_size[p] = 1


@ti.kernel
def _setup_solid_plane_node_case(node: ti.template(), normal: ti.types.vector(2, float)):
    tangent = vec2f(normal[1], -normal[0])
    for ng, nb in node:
        node[ng, nb]._grid_reset()
    node[2, 0].m = 1.0
    node[2, 0].ms = 1.0
    node[2, 0].momentum = -2.0 * normal + 3.0 * tangent
    node[2, 0].momentums = 2.0 * normal + 3.0 * tangent
    node[2, 0].force = -4.0 * normal - tangent
    node[2, 0].forces = 4.0 * normal - tangent
    node[6, 0].m = 1.0
    node[6, 0].ms = 1.0
    node[6, 0].momentum = 2.0 * normal
    node[6, 0].momentums = 2.0 * normal


@ti.kernel
def _setup_solid_plane_node_case3d(node: ti.template(), normal: ti.types.vector(3, float)):
    tangent = vec3f(0.0, 1.0, 0.0)
    for ng, nb in node:
        node[ng, nb]._grid_reset()
    node[13, 0].m = 1.0
    node[13, 0].ms = 1.0
    node[13, 0].momentum = -2.0 * normal + 3.0 * tangent
    node[13, 0].momentums = 2.0 * normal + 3.0 * tangent


@ti.kernel
def _setup_particle_traction_case(
    constraints: ti.template(),
    node: ti.template(),
    particle: ti.template(),
    lnid: ti.template(),
    shape_fn: ti.template(),
    node_size: ti.template(),
):
    constraints[0].pid = 0
    constraints[0].tractions = vec2f(0.0, -5000.0)
    constraints[0].tractionf = vec2f(0.0, 0.0)
    constraints[0].psize = vec2f(0.005, 0.005)
    particle[0].bodyID = ti.u8(0)
    particle[0].phase = ti.u8(1)
    particle[0].solid_velocity_gradient = mat2x2([[25.0, 0.0], [0.0, -10.0]])
    lnid[0] = 0
    shape_fn[0] = 1.0
    node_size[0] = 1
    node[0, 0]._grid_reset()


@ti.kernel
def _setup_fluid_only_stale_node(node: ti.template()):
    node[0, 0]._grid_reset()
    node[0, 0].m = 0.0
    node[0, 0].mf = 1.0
    node[0, 0].momentumf = vec2f(11.0, -7.0)
    node[0, 0].forcef = vec2f(3.0, 5.0)


@ti.kernel
def _setup_solid_state_case(
    node: ti.template(),
    particle: ti.template(),
    material_mapping: ti.template(),
    lnid: ti.template(),
    dshape_fn: ti.template(),
    node_size: ti.template(),
    porosity: float,
    velocity_x: float,
    initial_stress: float,
    fixed: int,
):
    node[0, 0]._grid_reset()
    node[0, 0].momentums = vec2f(velocity_x, 0.0)
    material_mapping[0] = 0
    particle[0].active = ti.u8(1)
    particle[0].materialID = ti.u8(1)
    particle[0].bodyID = ti.u8(0)
    particle[0].phase = ti.u8(PHASE_SOLID)
    particle[0].vol = 1.0
    particle[0].porosity = porosity
    particle[0].stress = ti.Vector([initial_stress, initial_stress, initial_stress, 0.0, 0.0, 0.0])
    particle[0].fix_v = ti.Vector([fixed, fixed], dt=ti.u8)
    lnid[0] = 0
    dshape_fn[0] = vec2f(1.0, 0.0)
    node_size[0] = 1


@ti.kernel
def _setup_node_pressure_sample_case(
    node: ti.template(),
    particle: ti.template(),
    material_mapping: ti.template(),
    lnid: ti.template(),
    shape_fn: ti.template(),
    node_size: ti.template(),
    cell_type: ti.template(),
    cell_pressure: ti.template(),
    position: ti.types.vector(2, float),
    gx: float,
    gy: float,
):
    for I in ti.grouped(cell_type):
        cell_type[I] = FLUID_CELL
        cell_pressure[I] = gx * (I[0] + 0.5) * DX + gy * (I[1] + 0.5) * DY
    for I in ti.grouped(node):
        node[I]._grid_reset()
        node[I].ms = 1.0
    material_mapping[0] = 0
    particle[0].active = ti.u8(1)
    particle[0].materialID = ti.u8(1)
    particle[0].bodyID = ti.u8(0)
    particle[0].phase = ti.u8(1)
    particle[0].x = position
    particle[0].pressure = 0.0
    base = ti.floor(position / ti.Vector([DX, DY])).cast(int)
    fx = position[0] / DX - ti.cast(base[0], float)
    fy = position[1] / DY - ti.cast(base[1], float)
    lnid[0] = base[0] + base[1] * (NX + 1)
    lnid[1] = base[0] + 1 + base[1] * (NX + 1)
    lnid[2] = base[0] + (base[1] + 1) * (NX + 1)
    lnid[3] = base[0] + 1 + (base[1] + 1) * (NX + 1)
    shape_fn[0] = (1.0 - fx) * (1.0 - fy)
    shape_fn[1] = fx * (1.0 - fy)
    shape_fn[2] = (1.0 - fx) * fy
    shape_fn[3] = fx * fy
    node_size[0] = 4


@ti.kernel
def _fill_pressure_fields(
    cell_type: ti.template(),
    face_porosity_x: ti.template(),
    face_porosity_y: ti.template(),
    cell_solid_density: ti.template(),
    cell_fluid_density: ti.template(),
    fluid_sdf: ti.template(),
):
    for I in ti.grouped(cell_type):
        cell_type[I] = FLUID_CELL
        cell_solid_density[I] = RHO_S
        cell_fluid_density[I] = RHO_F
        fluid_sdf[I] = -0.5 * ti.min(DX, DY)
    for I in ti.grouped(face_porosity_x):
        face_porosity_x[I] = PHI
    for I in ti.grouped(face_porosity_y):
        face_porosity_y[I] = PHI


@ti.kernel
def _fill_side_wall_pressure_fields(
    cell_type: ti.template(),
    face_porosity_x: ti.template(),
    face_porosity_y: ti.template(),
    cell_solid_density: ti.template(),
    cell_fluid_density: ti.template(),
    fluid_sdf: ti.template(),
):
    for I in ti.grouped(cell_type):
        if I[0] == 0 or I[0] == cell_type.shape[0] - 1:
            cell_type[I] = SOLID_CELL
        elif I[1] == cell_type.shape[1] - 1:
            cell_type[I] = AIR_CELL
        else:
            cell_type[I] = FLUID_CELL
        cell_solid_density[I] = RHO_S
        cell_fluid_density[I] = RHO_F
        fluid_sdf[I] = -0.5 * ti.min(DX, DY) if cell_type[I] == FLUID_CELL else 0.5 * ti.min(DX, DY)
    for I in ti.grouped(face_porosity_x):
        face_porosity_x[I] = PHI
    for I in ti.grouped(face_porosity_y):
        face_porosity_y[I] = PHI


@ti.kernel
def _fill_coarse_free_surface_pressure_fields(
    coarse_cell_type: ti.template(),
    face_porosity_x: ti.template(),
    face_porosity_y: ti.template(),
    cell_solid_density: ti.template(),
    cell_fluid_density: ti.template(),
    fluid_sdf: ti.template(),
):
    for I in ti.grouped(coarse_cell_type):
        coarse_cell_type[I] = SOLID_CELL
    coarse_cell_type[1, 1] = FLUID_CELL
    coarse_cell_type[2, 1] = AIR_CELL

    for I in ti.grouped(cell_solid_density):
        cell_solid_density[I] = RHO_S
        cell_fluid_density[I] = RHO_F
        fluid_sdf[I] = 0.15
    fluid_sdf[2, 2] = -0.05
    fluid_sdf[2, 3] = -0.05
    fluid_sdf[3, 2] = -0.05
    fluid_sdf[3, 3] = -0.05

    for I in ti.grouped(face_porosity_x):
        face_porosity_x[I] = PHI
    for I in ti.grouped(face_porosity_y):
        face_porosity_y[I] = PHI


@ti.kernel
def _fill_linear_pressure(
    cell_type: ti.template(),
    cell_pressure: ti.template(),
    fluid_sdf: ti.template(),
    fluid_mass_x: ti.template(),
    fluid_mass_y: ti.template(),
    face_fluid_density_x: ti.template(),
    face_fluid_density_y: ti.template(),
    gx: float,
    gy: float,
):
    for I in ti.grouped(cell_type):
        cell_type[I] = FLUID_CELL
        cell_pressure[I] = gx * (I[0] + 0.5) * DX + gy * (I[1] + 0.5) * DY
        fluid_sdf[I] = -0.5 * ti.min(DX, DY)
    for I in ti.grouped(fluid_mass_x):
        fluid_mass_x[I] = 1.0
        face_fluid_density_x[I] = RHO_F
    for I in ti.grouped(fluid_mass_y):
        fluid_mass_y[I] = 1.0
        face_fluid_density_y[I] = RHO_F


@ti.kernel
def _fill_free_surface_solid_pressure_case(
    cell_type: ti.template(),
    cell_pressure: ti.template(),
    fluid_sdf: ti.template(),
    fluid_mass_x: ti.template(),
    fluid_mass_y: ti.template(),
    face_fluid_density_x: ti.template(),
    face_fluid_density_y: ti.template(),
    p0: float,
):
    for I in ti.grouped(cell_type):
        if I[1] == 0:
            cell_type[I] = FLUID_CELL
            cell_pressure[I] = p0
            fluid_sdf[I] = -0.5 * ti.min(DX, DY)
        else:
            cell_type[I] = AIR_CELL
            cell_pressure[I] = 0.0
            fluid_sdf[I] = 0.5 * ti.min(DX, DY)
    for I in ti.grouped(fluid_mass_x):
        fluid_mass_x[I] = 0.0
        face_fluid_density_x[I] = RHO_F
    for I in ti.grouped(fluid_mass_y):
        fluid_mass_y[I] = 0.0
        face_fluid_density_y[I] = RHO_F


@ti.kernel
def _sample_bottom_corner_pressure_case(
    cell_type: ti.template(),
    cell_pressure: ti.template(),
    fluid_sdf: ti.template(),
    out: ti.template(),
    p0: float,
):
    for I in ti.grouped(cell_type):
        if I[0] == 0 and I[1] == 0:
            cell_type[I] = AIR_CELL
        elif I[0] == 0 or I[1] == 0:
            cell_type[I] = SOLID_CELL
        else:
            cell_type[I] = FLUID_CELL
        cell_pressure[I] = p0 if cell_type[I] == FLUID_CELL else 0.0
        fluid_sdf[I] = -0.5 * ti.min(DX, DY) if cell_type[I] == FLUID_CELL else 0.5 * ti.min(DX, DY)

    out[0] = _sample_cell_pressure_gfm_shape2d(
        ti.Vector([1.25 * DX, 1.25 * DY]),
        ti.Vector([DX, DY]),
        2,
        3,
        ti.Vector([DX, DY]),
        cell_type,
        cell_pressure,
        fluid_sdf,
    )


@ti.kernel
def _sample_wall_pressure_case(
    cell_type: ti.template(),
    cell_pressure: ti.template(),
    fluid_sdf: ti.template(),
    out: ti.template(),
    p0: float,
):
    for I in ti.grouped(cell_type):
        cell_type[I] = FLUID_CELL
        cell_pressure[I] = p0
        fluid_sdf[I] = -0.5 * ti.min(DX, DY)
    out[0] = _sample_cell_pressure_gfm_shape2d(
        ti.Vector([0.125 * DX, 0.125 * DY]),
        ti.Vector([0.125 * DX, 0.125 * DY]),
        2,
        3,
        ti.Vector([DX, DY]),
        cell_type,
        cell_pressure,
        fluid_sdf,
    )


@ti.kernel
def _sample_wall_tangential_velocity_case(velocity_x: ti.template(), velocity_y: ti.template(), out: ti.template()):
    velocity_x.fill(0.0)
    velocity_y.fill(2.0)
    out[0] = _sample_mac_velocity(
        ti.Vector([0.125 * DX, 2.25 * DY]),
        ti.Vector([0.125 * DX, 0.125 * DY]),
        2,
        3,
        ti.Vector([DX, DY]),
        velocity_x,
        velocity_y,
    )[1]


@ti.kernel
def _sample_wall_tangential_velocity_case3d(
    velocity_x: ti.template(), velocity_y: ti.template(), velocity_z: ti.template(), out: ti.template()
):
    velocity_x.fill(0.0)
    velocity_y.fill(2.0)
    velocity_z.fill(0.0)
    out[0] = _sample_mac_velocity3d(
        ti.Vector([0.125 * DX, 2.25 * DY, 0.375]),
        ti.Vector([0.125 * DX, 0.125 * DY, 0.0375]),
        2,
        3,
        ti.Vector([DX, DY, 0.3]),
        velocity_x,
        velocity_y,
        velocity_z,
    )[1]


@ti.kernel
def _fill_divergent_velocity(
    cell_type: ti.template(),
    face_porosity_x: ti.template(),
    face_porosity_y: ti.template(),
    fluid_velocity_x: ti.template(),
    fluid_velocity_y: ti.template(),
    solid_velocity_x: ti.template(),
    solid_velocity_y: ti.template(),
    rate: float,
):
    for I in ti.grouped(cell_type):
        cell_type[I] = FLUID_CELL
    for I in ti.grouped(face_porosity_x):
        face_porosity_x[I] = PHI
        fluid_velocity_x[I] = 0.0
        solid_velocity_x[I] = 0.0
    for I in ti.grouped(face_porosity_y):
        face_porosity_y[I] = PHI
        fluid_velocity_y[I] = rate * I[1] * DY
        solid_velocity_y[I] = 0.0


@ti.kernel
def _setup_boundary_face_porosity_case(
    fluid_mass_x: ti.template(),
    fluid_mass_y: ti.template(),
    fluid_velocity_x: ti.template(),
    fluid_velocity_y: ti.template(),
    solid_mass_x: ti.template(),
    solid_mass_y: ti.template(),
    face_porosity_x: ti.template(),
    face_porosity_y: ti.template(),
    cell_fluid_mass: ti.template(),
    cell_solid_mass: ti.template(),
):
    for I in ti.grouped(fluid_mass_x):
        fluid_mass_x[I] = 0.0
        fluid_velocity_x[I] = 0.0
        solid_mass_x[I] = 0.0
        face_porosity_x[I] = 0.0
    for I in ti.grouped(fluid_mass_y):
        fluid_mass_y[I] = 0.0
        fluid_velocity_y[I] = 0.0
        solid_mass_y[I] = 0.0
        face_porosity_y[I] = 0.0
    for I in ti.grouped(cell_fluid_mass):
        cell_fluid_mass[I] = 0.0
        cell_solid_mass[I] = 0.0

    cell_fluid_mass[1, 1] = 1.0
    cell_solid_mass[1, 1] = 2.0
    fluid_mass_y[1, 1] = 1.0
    fluid_mass_x[1, 1] = 1.0


@ti.kernel
def _setup_sparse_fluid_particle(particle: ti.template(), cal_length: ti.template()):
    particle[0].active = ti.u8(1)
    particle[0].materialID = ti.u8(1)
    particle[0].bodyID = ti.u8(0)
    particle[0].phase = ti.u8(PHASE_FLUID)
    particle[0].x = vec2f(DX, 1.5 * DY)
    particle[0].vol = DX * DY
    particle[0].mf = RHO_F * DX * DY
    particle[0].vf = vec2f(0.3, -0.1)
    particle[0].fluid_velocity_gradient = mat2x2([[0.0, 0.0], [0.0, 0.0]])
    cal_length[0] = vec2f(0.0, 0.0)


@ti.kernel
def _fill_face_velocity_and_density(
    face_porosity_x: ti.template(),
    face_porosity_y: ti.template(),
    fluid_velocity_x: ti.template(),
    fluid_velocity_y: ti.template(),
    solid_velocity_x: ti.template(),
    solid_velocity_y: ti.template(),
    fluid_mass_x: ti.template(),
    fluid_mass_y: ti.template(),
    face_fluid_density_x: ti.template(),
    face_fluid_density_y: ti.template(),
    rate: float,
):
    for I in ti.grouped(face_porosity_x):
        face_porosity_x[I] = PHI
        fluid_velocity_x[I] = 0.0
        solid_velocity_x[I] = 0.0
        fluid_mass_x[I] = 1.0
        face_fluid_density_x[I] = RHO_F
    for I in ti.grouped(face_porosity_y):
        face_porosity_y[I] = PHI
        fluid_velocity_y[I] = rate * I[1] * DY
        solid_velocity_y[I] = 0.0
        fluid_mass_y[I] = 1.0
        face_fluid_density_y[I] = RHO_F


@ti.kernel
def _compute_drag(
    relx: float,
    rely: float,
    porosity: float,
    permeability: float,
    drag_model: float,
    dt: ti.template(),
    out: ti.template(),
):
    solid_acc, fluid_acc = _drag_accelerations_props(
        ti.Vector([relx, rely]),
        porosity,
        RHO_S,
        RHO_F,
        1.0e-3,
        1.0e-2,
        permeability,
        9.8e3,
        drag_model,
        dt,
    )
    out[0] = solid_acc[0]
    out[1] = solid_acc[1]
    out[2] = fluid_acc[0]
    out[3] = fluid_acc[1]


@ti.kernel
def _compute_elastic_oedometer_response(
    young: float,
    poisson: float,
    strain: float,
    out: ti.template(),
):
    bulk = young / (3.0 * (1.0 - 2.0 * poisson))
    shear = young / (2.0 * (1.0 + poisson))
    stress_increment = ElasticTensorMultiplyVector(ti.Vector([0.0, strain, 0.0, 0.0, 0.0, 0.0]), bulk, shear)
    out[0] = stress_increment[1] / strain
    out[1] = stress_increment[0] / strain
    out[2] = stress_increment[2] / strain


@ti.kernel
def _setup_paper_solid_pressure_correction_case(
    node: ti.template(),
    particle: ti.template(),
    material_mapping: ti.template(),
    lnid: ti.template(),
    shape_fn: ti.template(),
    dshape_fn: ti.template(),
    node_size: ti.template(),
    rho: float,
    porosity: float,
    gx: float,
    gy: float,
):
    volume = 1.0
    material_mapping[0] = 0
    particle[0].active = ti.u8(1)
    particle[0].materialID = ti.u8(1)
    particle[0].bodyID = ti.u8(0)
    particle[0].phase = ti.u8(1)
    particle[0].vol = volume
    particle[0].porosity = porosity
    node_size[0] = 4
    for node_id in range(4):
        node[node_id, 0]._grid_reset()
        x = ti.cast(node_id % 2, float)
        y = ti.cast(node_id // 2, float)
        node[node_id, 0].ms = 0.25 * rho * (1.0 - porosity) * volume
        node[node_id, 0].pressure = 1234.0 + gx * x + gy * y
        lnid[node_id] = node_id
        shape_fn[node_id] = 0.25
        dshape_fn[node_id] = vec2f(x - 0.5, y - 0.5)


def test_pressure_matrix_reduces_to_uniform_laplacian():
    init(dim=2, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    dt = ti.field(float, shape=())
    dt[None] = DT_VALUE
    cell_type = ti.field(int, shape=(NX, NY))
    face_porosity_x = ti.field(float, shape=(NX + 1, NY))
    face_porosity_y = ti.field(float, shape=(NX, NY + 1))
    cell_solid_density = ti.field(float, shape=(NX, NY))
    cell_fluid_density = ti.field(float, shape=(NX, NY))
    fluid_sdf = ti.field(float, shape=(NX, NY))
    adiag = ti.field(float, shape=(NX, NY))
    ax = ti.Vector.field(2, float, shape=(NX, NY))

    _fill_pressure_fields(
        cell_type, face_porosity_x, face_porosity_y, cell_solid_density, cell_fluid_density, fluid_sdf
    )
    kernel_assemble_double_layer_pressure_A(
        ti.Vector([DX, DY]),
        dt,
        cell_type,
        face_porosity_x,
        face_porosity_y,
        cell_solid_density,
        cell_fluid_density,
        fluid_sdf,
        adiag,
        ax,
    )

    mobility = (1.0 - PHI) / RHO_S + PHI / RHO_F
    sx = DT_VALUE * mobility / (DX * DX)
    sy = DT_VALUE * mobility / (DY * DY)
    expected_diag = 2.0 * sx + 2.0 * sy + 1.0e-12
    adiag_np = adiag.to_numpy()
    ax_np = ax.to_numpy()
    assert np.allclose(adiag_np[1:-1, 1:-1], expected_diag, rtol=1.0e-5, atol=1.0e-12)
    assert np.allclose(ax_np[1:-1, 1:-1, 0], -sx, rtol=1.0e-5, atol=1.0e-12)
    assert np.allclose(ax_np[1:-1, 1:-1, 1], -sy, rtol=1.0e-5, atol=1.0e-12)


def test_double_layer_grid_reset_clears_fluid_only_nodes():
    node = NodeTwoPhase2D.field(shape=(1, 1))
    _setup_fluid_only_stale_node(node)
    kernel_reset_double_layer_grid(node)

    nodal = node.to_numpy()
    assert np.isclose(nodal["m"][0, 0], 0.0)
    assert np.isclose(nodal["mf"][0, 0], 0.0)
    assert np.allclose(nodal["momentumf"][0, 0], [0.0, 0.0])
    assert np.allclose(nodal["forcef"][0, 0], [0.0, 0.0])


def test_double_layer_fluid_volume_correction_preserves_mass_and_solid_position():
    init(dim=2, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    node = _VolumeNode.field(shape=(9, 1))
    particle = _VolumeParticle.field(shape=2)
    material_mapping = ti.field(int, shape=2)
    lnid = ti.field(int, shape=2)
    shape_fn = ti.field(float, shape=2)
    dshape_fn = ti.Vector.field(2, float, shape=2)
    node_size = ti.field(int, shape=2)
    _setup_double_layer_volume_correction(node, particle, material_mapping, lnid, shape_fn, dshape_fn, node_size)

    kernel_update_double_layer_fluid_volume2d(
        1, 0, 2, 1, 1000.0, node, particle, material_mapping, lnid, shape_fn, node_size
    )
    initialized = particle.to_numpy()
    assert np.isclose(initialized["mf"][0], 10.0)
    assert np.isclose(initialized["vol"][0], 0.01)

    node[4, 0].porosity = 0.5
    kernel_update_double_layer_fluid_volume2d(
        1, 0, 2, 0, 1000.0, node, particle, material_mapping, lnid, shape_fn, node_size
    )
    updated = particle.to_numpy()
    assert np.isclose(updated["mf"][0], 10.0)
    assert np.isclose(updated["vol"][0], 0.02)

    node.vol.fill(0.0)
    kernel_volume_p2g_double_layer_fluid2d(1, 2, node, particle, lnid, shape_fn, node_size)
    assert np.isclose(node.to_numpy()["vol"][4, 0], 0.02)
    before = particle.to_numpy()["x"].copy()
    kernel_delta_correct_double_layer_fluid2d(
        1,
        2,
        ti.Vector([1.0, 1.0]),
        ti.Vector([0.1, 0.1]),
        ti.Vector([3, 3]),
        node,
        particle,
        lnid,
        dshape_fn,
        node_size,
    )
    after = particle.to_numpy()["x"]
    assert after[0, 0] < before[0, 0]
    assert np.allclose(after[1], before[1])


def test_double_layer_solid_fluidizes_above_maximum_porosity_and_recovers_below_it():
    init(dim=2, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    dt = ti.field(float, shape=())
    dt[None] = DT_VALUE
    node = NodeTwoPhase2D.field(shape=(1, 1))
    particle = ParticleCloudTwoPhase2D.field(shape=1)
    material_mapping = ti.field(int, shape=1)
    lnid = ti.field(int, shape=1)
    dshape_fn = ti.Vector.field(2, float, shape=1)
    node_size = ti.field(int, shape=1)
    state_vars = _MohrCoulombState.field(shape=1)
    material = MohrCoulombModel(material_type="TwoPhaseDoubleLayer")
    material.initialize_coupling()
    material.model_initialize(
        {
            "YoungModulus": 3.0e6,
            "Cohesion": 1000.0,
            "Friction": 30.0,
            "Porosity": 0.38,
            "MaximumPorosity": 0.50,
            "FluidBulkModulus": 2.2e8,
            "Permeability": 1.0e-3,
        }
    )

    _setup_solid_state_case(
        node,
        particle,
        material_mapping,
        lnid,
        dshape_fn,
        node_size,
        0.60,
        0.0,
        -1000.0,
        0,
    )
    kernel_update_double_layer_solid_state2d(
        1,
        0,
        1,
        dt,
        material,
        state_vars,
        node,
        particle,
        material_mapping,
        lnid,
        dshape_fn,
        node_size,
    )
    assert np.allclose(particle.to_numpy()["stress"][0], 0.0)

    _setup_solid_state_case(
        node,
        particle,
        material_mapping,
        lnid,
        dshape_fn,
        node_size,
        0.49,
        -1.0,
        0.0,
        0,
    )
    kernel_update_double_layer_solid_state2d(
        1,
        0,
        1,
        dt,
        material,
        state_vars,
        node,
        particle,
        material_mapping,
        lnid,
        dshape_fn,
        node_size,
    )
    assert np.linalg.norm(particle.to_numpy()["stress"][0]) > 0.0

    _setup_solid_state_case(
        node,
        particle,
        material_mapping,
        lnid,
        dshape_fn,
        node_size,
        0.39,
        -1.0,
        -1000.0,
        1,
    )
    kernel_update_double_layer_solid_state2d(
        1,
        0,
        1,
        dt,
        material,
        state_vars,
        node,
        particle,
        material_mapping,
        lnid,
        dshape_fn,
        node_size,
    )
    rigid = particle.to_numpy()
    assert np.isclose(rigid["vol"][0], 1.0)
    assert np.isclose(rigid["porosity"][0], 0.39)
    assert np.allclose(rigid["stress"][0], [-1000.0, -1000.0, -1000.0, 0.0, 0.0, 0.0])


def test_mac_face_porosity_without_solid_support_tends_smoothly_to_one():
    init(dim=2, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    nx, ny = 3, 3
    fluid_mass_x = ti.field(float, shape=(nx + 1, ny))
    fluid_mass_y = ti.field(float, shape=(nx, ny + 1))
    fluid_velocity_x = ti.field(float, shape=(nx + 1, ny))
    fluid_velocity_y = ti.field(float, shape=(nx, ny + 1))
    fluid_velocity0_x = ti.field(float, shape=(nx + 1, ny))
    fluid_velocity0_y = ti.field(float, shape=(nx, ny + 1))
    solid_mass_x = ti.field(float, shape=(nx + 1, ny))
    solid_mass_y = ti.field(float, shape=(nx, ny + 1))
    solid_velocity_x = ti.field(float, shape=(nx + 1, ny))
    solid_velocity_y = ti.field(float, shape=(nx, ny + 1))
    face_porosity_x = ti.field(float, shape=(nx + 1, ny))
    face_porosity_y = ti.field(float, shape=(nx, ny + 1))
    cell_type = ti.field(int, shape=(nx, ny))
    cell_fluid_mass = ti.field(float, shape=(nx, ny))
    cell_solid_mass = ti.field(float, shape=(nx, ny))
    cell_porosity = ti.field(float, shape=(nx, ny))
    cell_solid_velocity = ti.Vector.field(2, float, shape=(nx, ny))
    cell_fluid_velocity = ti.Vector.field(2, float, shape=(nx, ny))

    _setup_boundary_face_porosity_case(
        fluid_mass_x,
        fluid_mass_y,
        fluid_velocity_x,
        fluid_velocity_y,
        solid_mass_x,
        solid_mass_y,
        face_porosity_x,
        face_porosity_y,
        cell_fluid_mass,
        cell_solid_mass,
    )
    kernel_normalize_double_layer_mac_fields2d(
        1.0e-12,
        fluid_mass_x,
        fluid_mass_y,
        fluid_velocity_x,
        fluid_velocity_y,
        fluid_velocity0_x,
        fluid_velocity0_y,
        solid_mass_x,
        solid_mass_y,
        solid_velocity_x,
        solid_velocity_y,
        face_porosity_x,
        face_porosity_y,
        cell_type,
        cell_fluid_mass,
        cell_solid_mass,
        cell_porosity,
        cell_solid_velocity,
        cell_fluid_velocity,
    )

    porosity_x = face_porosity_x.to_numpy()
    porosity_y = face_porosity_y.to_numpy()
    assert np.isclose(porosity_x[1, 1], 1.0)
    assert np.isclose(porosity_y[1, 1], 1.0)
    assert np.isclose(porosity_x[0, 0], 1.0)
    assert np.isclose(porosity_y[0, 0], 1.0)


def test_solid_cell_face_imposes_moving_wall_normal_velocity():
    init(dim=2, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    cell_type = ti.field(int, shape=(2, 1))
    fluid_velocity_x = ti.field(float, shape=(3, 1))
    fluid_velocity_y = ti.field(float, shape=(2, 2))
    fluid_acceleration_x = ti.field(float, shape=(3, 1))
    fluid_acceleration_y = ti.field(float, shape=(2, 2))
    solid_velocity_x = ti.field(float, shape=(3, 1))
    solid_velocity_y = ti.field(float, shape=(2, 2))
    cell_type[0, 0] = SOLID_CELL
    cell_type[1, 0] = FLUID_CELL
    fluid_velocity_x[1, 0] = -1.0
    fluid_acceleration_x[1, 0] = 2.0
    solid_velocity_x[1, 0] = 0.37

    kernel_enforce_double_layer_solid_cell_faces2d(
        cell_type,
        fluid_velocity_x,
        fluid_velocity_y,
        fluid_acceleration_x,
        fluid_acceleration_y,
        solid_velocity_x,
        solid_velocity_y,
    )
    assert np.isclose(fluid_velocity_x[1, 0], 0.37)
    assert np.isclose(solid_velocity_x[1, 0], 0.37)
    assert fluid_acceleration_x[1, 0] == 0.0


def test_cell_centered_transfer_keeps_both_sides_of_grid_crossing_in_pressure_domain():
    init(dim=2, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    nx, ny = 3, 3
    particle = ParticleCloudTwoPhase2D.field(shape=1)
    cal_length = ti.Vector.field(2, float, shape=1)
    fluid_mass_x = ti.field(float, shape=(nx + 1, ny))
    fluid_mass_y = ti.field(float, shape=(nx, ny + 1))
    fluid_velocity_x = ti.field(float, shape=(nx + 1, ny))
    fluid_velocity_y = ti.field(float, shape=(nx, ny + 1))
    fluid_velocity0_x = ti.field(float, shape=(nx + 1, ny))
    fluid_velocity0_y = ti.field(float, shape=(nx, ny + 1))
    solid_mass_x = ti.field(float, shape=(nx + 1, ny))
    solid_mass_y = ti.field(float, shape=(nx, ny + 1))
    solid_velocity_x = ti.field(float, shape=(nx + 1, ny))
    solid_velocity_y = ti.field(float, shape=(nx, ny + 1))
    face_porosity_x = ti.field(float, shape=(nx + 1, ny))
    face_porosity_y = ti.field(float, shape=(nx, ny + 1))
    cell_type = ti.field(int, shape=(nx, ny))
    cell_fluid_mass = ti.field(float, shape=(nx, ny))
    cell_solid_mass = ti.field(float, shape=(nx, ny))
    cell_porosity = ti.field(float, shape=(nx, ny))
    cell_solid_velocity = ti.Vector.field(2, float, shape=(nx, ny))
    cell_fluid_velocity = ti.Vector.field(2, float, shape=(nx, ny))

    _setup_sparse_fluid_particle(particle, cal_length)
    kernel_mac_p2g_double_layer2d(
        1,
        ti.Vector([DX, DY]),
        MAC_SHAPE_LINEAR,
        2,
        0,
        fluid_mass_x,
        fluid_mass_y,
        fluid_velocity_x,
        fluid_velocity_y,
        solid_mass_x,
        solid_mass_y,
        solid_velocity_x,
        solid_velocity_y,
        face_porosity_x,
        face_porosity_y,
        cell_fluid_mass,
        cell_solid_mass,
        cell_porosity,
        cell_solid_velocity,
        cell_fluid_velocity,
        particle,
        cal_length,
    )
    kernel_normalize_double_layer_mac_fields2d(
        1.0e-12,
        fluid_mass_x,
        fluid_mass_y,
        fluid_velocity_x,
        fluid_velocity_y,
        fluid_velocity0_x,
        fluid_velocity0_y,
        solid_mass_x,
        solid_mass_y,
        solid_velocity_x,
        solid_velocity_y,
        face_porosity_x,
        face_porosity_y,
        cell_type,
        cell_fluid_mass,
        cell_solid_mass,
        cell_porosity,
        cell_solid_velocity,
        cell_fluid_velocity,
    )

    mass = cell_fluid_mass.to_numpy()
    types = cell_type.to_numpy()
    assert mass[0, 1] > 0.0
    assert mass[1, 1] > 0.0
    assert types[0, 1] == FLUID_CELL
    assert types[1, 1] == FLUID_CELL


def test_double_layer_volume_fraction_classifies_holes_and_reconstructs_surface_distance():
    init(dim=2, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    shape = (4, 4)
    cell_volume = DX * DY
    particle = ParticleCloudTwoPhase2D.field(shape=1)
    cell_fluid_mass = ti.field(float, shape=shape)
    cell_fluid_density = ti.field(float, shape=shape)
    cell_porosity = ti.field(float, shape=shape)
    cell_type = ti.field(int, shape=shape)
    fluid_sdf = ti.field(float, shape=shape)
    cell_fluid_density.fill(RHO_F)
    cell_porosity.fill(1.0)

    mass = np.zeros(shape)
    mass[1, 1] = 0.9 * RHO_F * cell_volume
    mass[2, 1] = 0.2 * RHO_F * cell_volume
    cell_fluid_mass.from_numpy(mass)
    kernel_classify_double_layer_fluid_cells2d(
        0,
        ti.Vector([DX, DY]),
        particle,
        cell_fluid_mass,
        cell_fluid_density,
        cell_porosity,
        cell_type,
    )
    kernel_build_double_layer_fluid_sdf2d(
        ti.Vector([DX, DY]),
        cell_fluid_mass,
        cell_fluid_density,
        cell_porosity,
        cell_type,
        fluid_sdf,
    )
    types = cell_type.to_numpy()
    sdf = fluid_sdf.to_numpy()
    assert types[1, 1] == FLUID_CELL
    assert types[2, 1] == AIR_CELL
    assert np.isclose(sdf[1, 1], -0.4 * min(DX, DY))
    assert np.isclose(sdf[2, 1], 0.3 * min(DX, DY))

    cell_fluid_mass.fill(0.0)
    cell_fluid_mass[2, 0] = 0.9 * RHO_F * cell_volume
    cell_fluid_mass[2, 3] = 0.9 * RHO_F * cell_volume
    kernel_classify_double_layer_fluid_cells2d(
        0,
        ti.Vector([DX, DY]),
        particle,
        cell_fluid_mass,
        cell_fluid_density,
        cell_porosity,
        cell_type,
    )
    assert cell_type[2, 1] == FLUID_CELL
    assert cell_type[2, 2] == FLUID_CELL


def test_double_layer_multigrid_keeps_thin_fluid_support_on_coarse_level():
    init(dim=2, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    fine = ti.field(int, shape=(4, 4))
    coarse = ti.field(int, shape=(2, 2))
    fine.fill(AIR_CELL)
    fine[1, 1] = FLUID_CELL

    kernel_coarsen_double_layer_grid_type2d(fine, coarse)

    coarse_type = coarse.to_numpy()
    assert coarse_type[0, 0] == FLUID_CELL
    assert np.all(coarse_type[1:, :] == AIR_CELL)
    assert coarse_type[0, 1] == AIR_CELL


def test_solid_cell_region_does_not_mark_adjacent_fluid_cell_on_grid_line():
    cell_type = ti.field(int, shape=(NX, NY))
    cell_type.fill(FLUID_CELL)

    kernel_mark_double_layer_solid_cell_region2d(
        ti.Vector([0.02, 0.02]),
        ti.Vector([0.0, 0.0]),
        ti.Vector([0.06, 0.12]),
        cell_type,
    )

    cell_type_np = cell_type.to_numpy()
    assert np.all(cell_type_np[:3, :6] == SOLID_CELL)
    assert np.all(cell_type_np[3:, :6] == FLUID_CELL)


def test_solid_plane_node_constraint_removes_normal_motion_for_arbitrary_normal():
    init(dim=2, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    node = NodeTwoPhase2D.field(shape=(9, 1))
    normal = np.array([1.0, -1.0]) / np.sqrt(2.0)
    tangent = np.array([normal[1], -normal[0]])
    _setup_solid_plane_node_case(node, ti.Vector(normal.tolist()))

    kernel_enforce_double_layer_solid_plane_nodes2d(
        1.0e-12,
        ti.Vector([0.5, 0.5]),
        ti.Vector([3, 3]),
        ti.Vector([0.0, 0.0]),
        ti.Vector([1.0, 1.0]),
        ti.Vector([0.5, 0.5]),
        ti.Vector(normal.tolist()),
        node,
    )

    constrained = node.to_numpy()
    assert np.allclose(constrained["momentum"][2, 0], 3.0 * tangent)
    assert np.allclose(constrained["momentums"][2, 0], 3.0 * tangent)
    assert np.allclose(constrained["force"][2, 0], -tangent)
    assert np.allclose(constrained["forces"][2, 0], -tangent)
    assert np.allclose(constrained["momentums"][6, 0], 2.0 * normal)


def test_solid_plane_node_constraint_works_in_3d():
    init(dim=3, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    node = NodeTwoPhase.field(shape=(27, 1))
    normal = np.array([1.0, 0.0, -1.0]) / np.sqrt(2.0)
    _setup_solid_plane_node_case3d(node, ti.Vector(normal.tolist()))

    kernel_enforce_double_layer_solid_plane_nodes3d(
        1.0e-12,
        ti.Vector([0.5, 0.5, 0.5]),
        ti.Vector([3, 3, 3]),
        ti.Vector([0.0, 0.0, 0.0]),
        ti.Vector([1.0, 1.0, 1.0]),
        ti.Vector([0.5, 0.0, 0.5]),
        ti.Vector(normal.tolist()),
        node,
    )

    constrained = node.to_numpy()
    assert np.allclose(constrained["momentum"][13, 0], [0.0, 3.0, 0.0], atol=1.0e-7)
    assert np.allclose(constrained["momentums"][13, 0], [0.0, 3.0, 0.0], atol=1.0e-7)


def test_pressure_matrix_uses_neumann_for_solid_walls_and_dirichlet_for_air():
    dt = ti.field(float, shape=())
    dt[None] = DT_VALUE
    cell_type = ti.field(int, shape=(NX, NY))
    face_porosity_x = ti.field(float, shape=(NX + 1, NY))
    face_porosity_y = ti.field(float, shape=(NX, NY + 1))
    cell_solid_density = ti.field(float, shape=(NX, NY))
    cell_fluid_density = ti.field(float, shape=(NX, NY))
    fluid_sdf = ti.field(float, shape=(NX, NY))
    adiag = ti.field(float, shape=(NX, NY))
    ax = ti.Vector.field(2, float, shape=(NX, NY))

    _fill_side_wall_pressure_fields(
        cell_type, face_porosity_x, face_porosity_y, cell_solid_density, cell_fluid_density, fluid_sdf
    )
    kernel_assemble_double_layer_pressure_A(
        ti.Vector([DX, DY]),
        dt,
        cell_type,
        face_porosity_x,
        face_porosity_y,
        cell_solid_density,
        cell_fluid_density,
        fluid_sdf,
        adiag,
        ax,
    )

    mobility = (1.0 - PHI) / RHO_S + PHI / RHO_F
    sx = DT_VALUE * mobility / (DX * DX)
    sy = DT_VALUE * mobility / (DY * DY)
    adiag_np = adiag.to_numpy()
    ax_np = ax.to_numpy()
    left_fluid = (1, 2)
    top_fluid = (2, NY - 2)
    assert np.isclose(adiag_np[left_fluid], sx + 2.0 * sy + 1.0e-12, rtol=1.0e-5, atol=1.0e-12)
    assert np.isclose(ax_np[left_fluid][0], -sx, rtol=1.0e-5, atol=1.0e-12)
    assert np.isclose(adiag_np[top_fluid], 2.0 * sx + 3.0 * sy + 1.0e-12, rtol=1.0e-5, atol=1.0e-12)
    assert np.isclose(ax_np[top_fluid][1], 0.0, atol=1.0e-12)


def test_coarse_pressure_matrix_uses_gfm_theta_for_free_surface():
    dt = ti.field(float, shape=())
    dt[None] = DT_VALUE
    factor = 2
    coarse_cell_type = ti.field(int, shape=(3, 3))
    face_porosity_x = ti.field(float, shape=(NX + 1, NY))
    face_porosity_y = ti.field(float, shape=(NX, NY + 1))
    cell_solid_density = ti.field(float, shape=(NX, NY))
    cell_fluid_density = ti.field(float, shape=(NX, NY))
    fluid_sdf = ti.field(float, shape=(NX, NY))
    adiag = ti.field(float, shape=(3, 3))
    ax = ti.Vector.field(2, float, shape=(3, 3))

    _fill_coarse_free_surface_pressure_fields(
        coarse_cell_type,
        face_porosity_x,
        face_porosity_y,
        cell_solid_density,
        cell_fluid_density,
        fluid_sdf,
    )
    kernel_assemble_double_layer_pressure_mg_A(
        ti.Vector([DX, DY]),
        dt,
        factor,
        coarse_cell_type,
        face_porosity_x,
        face_porosity_y,
        cell_solid_density,
        cell_fluid_density,
        fluid_sdf,
        adiag,
        ax,
    )

    mobility = (1.0 - PHI) / RHO_S + PHI / RHO_F
    coarse_dx = DX * factor
    scale_x = DT_VALUE * mobility / (coarse_dx * coarse_dx)
    expected_theta = 0.25
    expected_diag = scale_x / expected_theta + 1.0e-12
    assert np.isclose(adiag.to_numpy()[1, 1], expected_diag, rtol=1.0e-5, atol=1.0e-12)
    assert np.allclose(ax.to_numpy()[1, 1], [0.0, 0.0], atol=1.0e-12)


def test_pressure_rhs_matches_weighted_divergence():
    rate = 0.125
    cell_type = ti.field(int, shape=(NX, NY))
    face_porosity_x = ti.field(float, shape=(NX + 1, NY))
    face_porosity_y = ti.field(float, shape=(NX, NY + 1))
    fluid_velocity_x = ti.field(float, shape=(NX + 1, NY))
    fluid_velocity_y = ti.field(float, shape=(NX, NY + 1))
    solid_velocity_x = ti.field(float, shape=(NX + 1, NY))
    solid_velocity_y = ti.field(float, shape=(NX, NY + 1))
    b = ti.field(float, shape=(NX, NY))

    _fill_divergent_velocity(
        cell_type,
        face_porosity_x,
        face_porosity_y,
        fluid_velocity_x,
        fluid_velocity_y,
        solid_velocity_x,
        solid_velocity_y,
        rate,
    )
    kernel_assemble_double_layer_pressure_rhs(
        ti.Vector([DX, DY]),
        cell_type,
        face_porosity_x,
        face_porosity_y,
        fluid_velocity_x,
        fluid_velocity_y,
        solid_velocity_x,
        solid_velocity_y,
        b,
    )
    assert np.allclose(b.to_numpy(), -PHI * rate, rtol=1.0e-6, atol=1.0e-7)


def _solve_pressure_numpy(cell_type, adiag, ax, rhs):
    fluid_cells = [tuple(idx) for idx in np.argwhere(cell_type == FLUID_CELL)]
    cell_to_row = {cell: row for row, cell in enumerate(fluid_cells)}
    matrix = np.zeros((len(fluid_cells), len(fluid_cells)), dtype=np.float64)
    vector = np.zeros(len(fluid_cells), dtype=np.float64)
    for row, cell in enumerate(fluid_cells):
        i, j = cell
        matrix[row, row] = adiag[i, j]
        vector[row] = rhs[i, j]
        for axis, offset in enumerate(((1, 0), (0, 1))):
            neighbor = (i + offset[0], j + offset[1])
            if neighbor in cell_to_row:
                matrix[row, cell_to_row[neighbor]] = ax[i, j, axis]
            neighbor = (i - offset[0], j - offset[1])
            if neighbor in cell_to_row:
                matrix[row, cell_to_row[neighbor]] = ax[neighbor[0], neighbor[1], axis]
    pressure = np.zeros_like(rhs)
    solution = np.linalg.solve(matrix, vector)
    for row, cell in enumerate(fluid_cells):
        pressure[cell] = solution[row]
    return pressure


def _weighted_divergence_norm(cell_type, fluid_velocity_x, fluid_velocity_y, solid_velocity_x, solid_velocity_y):
    residuals = []
    for i in range(cell_type.shape[0]):
        for j in range(cell_type.shape[1]):
            if cell_type[i, j] != FLUID_CELL:
                continue
            div_f = (fluid_velocity_x[i + 1, j] - fluid_velocity_x[i, j]) / DX
            div_f += (fluid_velocity_y[i, j + 1] - fluid_velocity_y[i, j]) / DY
            div_s = (solid_velocity_x[i + 1, j] - solid_velocity_x[i, j]) / DX
            div_s += (solid_velocity_y[i, j + 1] - solid_velocity_y[i, j]) / DY
            residuals.append((1.0 - PHI) * div_s + PHI * div_f)
    return np.linalg.norm(np.asarray(residuals, dtype=np.float64))


def test_pressure_projection_reduces_weighted_divergence_with_code_sign_convention():
    dt = ti.field(float, shape=())
    dt[None] = DT_VALUE
    rate = 0.125
    cell_type = ti.field(int, shape=(NX, NY))
    face_porosity_x = ti.field(float, shape=(NX + 1, NY))
    face_porosity_y = ti.field(float, shape=(NX, NY + 1))
    cell_solid_density = ti.field(float, shape=(NX, NY))
    cell_fluid_density = ti.field(float, shape=(NX, NY))
    fluid_sdf = ti.field(float, shape=(NX, NY))
    adiag = ti.field(float, shape=(NX, NY))
    ax = ti.Vector.field(2, float, shape=(NX, NY))
    b = ti.field(float, shape=(NX, NY))
    cell_pressure = ti.field(float, shape=(NX, NY))
    fluid_mass_x = ti.field(float, shape=(NX + 1, NY))
    fluid_mass_y = ti.field(float, shape=(NX, NY + 1))
    fluid_velocity_x = ti.field(float, shape=(NX + 1, NY))
    fluid_velocity_y = ti.field(float, shape=(NX, NY + 1))
    fluid_acceleration_x = ti.field(float, shape=(NX + 1, NY))
    fluid_acceleration_y = ti.field(float, shape=(NX, NY + 1))
    solid_velocity_x = ti.field(float, shape=(NX + 1, NY))
    solid_velocity_y = ti.field(float, shape=(NX, NY + 1))
    face_fluid_density_x = ti.field(float, shape=(NX + 1, NY))
    face_fluid_density_y = ti.field(float, shape=(NX, NY + 1))
    node_solid_density = ti.field(float, shape=(1, 1))
    node_type = ti.types.struct(
        ms=float,
        momentums=ti.types.vector(2, float),
        momentum=ti.types.vector(2, float),
        forces=ti.types.vector(2, float),
    )
    node = node_type.field(shape=(1, 1))

    _fill_side_wall_pressure_fields(
        cell_type,
        face_porosity_x,
        face_porosity_y,
        cell_solid_density,
        cell_fluid_density,
        fluid_sdf,
    )
    _fill_face_velocity_and_density(
        face_porosity_x,
        face_porosity_y,
        fluid_velocity_x,
        fluid_velocity_y,
        solid_velocity_x,
        solid_velocity_y,
        fluid_mass_x,
        fluid_mass_y,
        face_fluid_density_x,
        face_fluid_density_y,
        rate,
    )
    kernel_assemble_double_layer_pressure_rhs(
        ti.Vector([DX, DY]),
        cell_type,
        face_porosity_x,
        face_porosity_y,
        fluid_velocity_x,
        fluid_velocity_y,
        solid_velocity_x,
        solid_velocity_y,
        b,
    )
    kernel_assemble_double_layer_pressure_A(
        ti.Vector([DX, DY]),
        dt,
        cell_type,
        face_porosity_x,
        face_porosity_y,
        cell_solid_density,
        cell_fluid_density,
        fluid_sdf,
        adiag,
        ax,
    )
    pressure = _solve_pressure_numpy(cell_type.to_numpy(), adiag.to_numpy(), ax.to_numpy(), b.to_numpy())
    cell_pressure.from_numpy(pressure)
    before = _weighted_divergence_norm(
        cell_type.to_numpy(),
        fluid_velocity_x.to_numpy(),
        fluid_velocity_y.to_numpy(),
        solid_velocity_x.to_numpy(),
        solid_velocity_y.to_numpy(),
    )
    kernel_correct_double_layer_velocity2d(
        1.0e-12,
        ti.Vector([DX, DY]),
        dt,
        cell_type,
        cell_pressure,
        fluid_sdf,
        fluid_mass_x,
        fluid_mass_y,
        fluid_velocity_x,
        fluid_velocity_y,
        fluid_acceleration_x,
        fluid_acceleration_y,
        face_fluid_density_x,
        face_fluid_density_y,
    )
    after = _weighted_divergence_norm(
        cell_type.to_numpy(),
        fluid_velocity_x.to_numpy(),
        fluid_velocity_y.to_numpy(),
        solid_velocity_x.to_numpy(),
        solid_velocity_y.to_numpy(),
    )
    _fill_face_velocity_and_density(
        face_porosity_x,
        face_porosity_y,
        fluid_velocity_x,
        fluid_velocity_y,
        solid_velocity_x,
        solid_velocity_y,
        fluid_mass_x,
        fluid_mass_y,
        face_fluid_density_x,
        face_fluid_density_y,
        rate,
    )
    cell_pressure.from_numpy(-pressure)
    kernel_correct_double_layer_velocity2d(
        1.0e-12,
        ti.Vector([DX, DY]),
        dt,
        cell_type,
        cell_pressure,
        fluid_sdf,
        fluid_mass_x,
        fluid_mass_y,
        fluid_velocity_x,
        fluid_velocity_y,
        fluid_acceleration_x,
        fluid_acceleration_y,
        face_fluid_density_x,
        face_fluid_density_y,
    )
    after_wrong_sign = _weighted_divergence_norm(
        cell_type.to_numpy(),
        fluid_velocity_x.to_numpy(),
        fluid_velocity_y.to_numpy(),
        solid_velocity_x.to_numpy(),
        solid_velocity_y.to_numpy(),
    )
    assert after < before
    assert after < after_wrong_sign


def test_pressure_gradient_correction_matches_linear_pressure():
    dt = ti.field(float, shape=())
    dt[None] = DT_VALUE
    gx = 21.0
    gy = -13.0
    cell_type = ti.field(int, shape=(NX, NY))
    cell_pressure = ti.field(float, shape=(NX, NY))
    fluid_sdf = ti.field(float, shape=(NX, NY))
    fluid_mass_x = ti.field(float, shape=(NX + 1, NY))
    fluid_mass_y = ti.field(float, shape=(NX, NY + 1))
    fluid_velocity_x = ti.field(float, shape=(NX + 1, NY))
    fluid_velocity_y = ti.field(float, shape=(NX, NY + 1))
    fluid_acceleration_x = ti.field(float, shape=(NX + 1, NY))
    fluid_acceleration_y = ti.field(float, shape=(NX, NY + 1))
    face_fluid_density_x = ti.field(float, shape=(NX + 1, NY))
    face_fluid_density_y = ti.field(float, shape=(NX, NY + 1))
    node_solid_density = ti.field(float, shape=(1, 1))
    node_type = ti.types.struct(
        ms=float,
        momentums=ti.types.vector(2, float),
        momentum=ti.types.vector(2, float),
        forces=ti.types.vector(2, float),
    )
    node = node_type.field(shape=(1, 1))

    _fill_linear_pressure(
        cell_type,
        cell_pressure,
        fluid_sdf,
        fluid_mass_x,
        fluid_mass_y,
        face_fluid_density_x,
        face_fluid_density_y,
        gx,
        gy,
    )
    kernel_correct_double_layer_velocity2d(
        1.0e-12,
        ti.Vector([DX, DY]),
        dt,
        cell_type,
        cell_pressure,
        fluid_sdf,
        fluid_mass_x,
        fluid_mass_y,
        fluid_velocity_x,
        fluid_velocity_y,
        fluid_acceleration_x,
        fluid_acceleration_y,
        face_fluid_density_x,
        face_fluid_density_y,
    )

    vx = fluid_velocity_x.to_numpy()
    vy = fluid_velocity_y.to_numpy()
    assert np.allclose(vx[1:NX, :], -DT_VALUE * gx / RHO_F, rtol=1.0e-6, atol=1.0e-9)
    assert np.allclose(vy[:, 1:NY], -DT_VALUE * gy / RHO_F, rtol=1.0e-6, atol=1.0e-9)
    assert np.allclose(vx[0, :], 0.0)
    assert np.allclose(vx[NX, :], 0.0)
    assert np.allclose(vy[:, 0], 0.0)
    assert np.allclose(vy[:, NY], 0.0)


def test_paper_solid_pressure_correction_matches_linear_pressure_gradient():
    init(dim=2, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    dt = ti.field(float, shape=())
    dt[None] = DT_VALUE
    particle_count = 1
    gx = 17.0
    gy = -29.0
    node = NodeTwoPhase2D.field(shape=(4, 1))
    particle = ParticleCloudTwoPhase2D.field(shape=particle_count)
    material_mapping = ti.field(int, shape=particle_count)
    lnid = ti.field(int, shape=4)
    shape_fn = ti.field(float, shape=4)
    dshape_fn = ti.Vector.field(2, float, shape=4)
    node_size = ti.field(int, shape=particle_count)

    _setup_paper_solid_pressure_correction_case(
        node,
        particle,
        material_mapping,
        lnid,
        shape_fn,
        dshape_fn,
        node_size,
        RHO_S,
        PHI,
        gx,
        gy,
    )
    kernel_correct_double_layer_solid_velocity_paper2d(
        1,
        0,
        particle_count,
        1.0e-12,
        dt,
        node,
        particle,
        material_mapping,
        lnid,
        shape_fn,
        dshape_fn,
        node_size,
    )

    nodal = node.to_numpy()
    velocity = nodal["momentums"][:, 0]
    acceleration = nodal["forces"][:, 0]
    expected_acceleration = np.array([-gx / RHO_S, -gy / RHO_S])
    assert np.allclose(acceleration, expected_acceleration, rtol=1.0e-6, atol=1.0e-10)
    assert np.allclose(velocity, DT_VALUE * expected_acceleration, rtol=1.0e-6, atol=1.0e-10)


def test_fluid_pressure_correction_uses_free_surface_dirichlet_gradient():
    dt = ti.field(float, shape=())
    dt[None] = DT_VALUE
    p0 = 1234.0
    cell_type = ti.field(int, shape=(NX, NY))
    cell_pressure = ti.field(float, shape=(NX, NY))
    fluid_sdf = ti.field(float, shape=(NX, NY))
    fluid_mass_x = ti.field(float, shape=(NX + 1, NY))
    fluid_mass_y = ti.field(float, shape=(NX, NY + 1))
    fluid_velocity_x = ti.field(float, shape=(NX + 1, NY))
    fluid_velocity_y = ti.field(float, shape=(NX, NY + 1))
    fluid_acceleration_x = ti.field(float, shape=(NX + 1, NY))
    fluid_acceleration_y = ti.field(float, shape=(NX, NY + 1))
    face_fluid_density_x = ti.field(float, shape=(NX + 1, NY))
    face_fluid_density_y = ti.field(float, shape=(NX, NY + 1))

    _fill_free_surface_solid_pressure_case(
        cell_type,
        cell_pressure,
        fluid_sdf,
        fluid_mass_x,
        fluid_mass_y,
        face_fluid_density_x,
        face_fluid_density_y,
        p0,
    )
    fluid_mass_y.fill(1.0)
    kernel_correct_double_layer_velocity2d(
        1.0e-12,
        ti.Vector([DX, DY]),
        dt,
        cell_type,
        cell_pressure,
        fluid_sdf,
        fluid_mass_x,
        fluid_mass_y,
        fluid_velocity_x,
        fluid_velocity_y,
        fluid_acceleration_x,
        fluid_acceleration_y,
        face_fluid_density_x,
        face_fluid_density_y,
    )

    velocity = fluid_velocity_y.to_numpy()
    expected_grad_y = (0.0 - p0) / (0.5 * DY)
    expected_vy = -DT_VALUE * expected_grad_y / RHO_F
    assert np.allclose(velocity[:, 1], expected_vy, rtol=1.0e-6, atol=1.0e-10)


def test_solid_pressure_node_projection_preserves_interior_linear_pressure():
    node = NodeTwoPhase2D.field(shape=((NX + 1) * (NY + 1), 1))
    particle = ParticleCloudTwoPhase2D.field(shape=1)
    material_mapping = ti.field(int, shape=1)
    lnid = ti.field(int, shape=4)
    shape_fn = ti.field(float, shape=4)
    node_size = ti.field(int, shape=1)
    cell_type = ti.field(int, shape=(NX, NY))
    cell_pressure = ti.field(float, shape=(NX, NY))
    fluid_sdf = ti.field(float, shape=(NX, NY))
    cal_length = ti.Vector.field(2, float, shape=1)
    position = np.array([2.35 * DX, 2.40 * DY])
    gx = 17.0
    gy = -29.0

    _setup_node_pressure_sample_case(
        node,
        particle,
        material_mapping,
        lnid,
        shape_fn,
        node_size,
        cell_type,
        cell_pressure,
        ti.Vector(position),
        gx,
        gy,
    )
    fluid_sdf.fill(-0.5 * min(DX, DY))
    cal_length[0] = ti.Vector([DX, DY])
    kernel_project_double_layer_pressure_to_solid_nodes2d(
        1.0e-12,
        ti.Vector([NX + 1, NY + 1]),
        ti.Vector([DX, DY]),
        MAC_SHAPE_LINEAR,
        2,
        cell_type,
        cell_pressure,
        fluid_sdf,
        node,
        cal_length,
    )
    kernel_sample_double_layer_solid_pressure_from_nodes2d(
        1,
        0,
        1,
        1.0e-12,
        particle,
        material_mapping,
        node,
        lnid,
        shape_fn,
        node_size,
    )

    expected = gx * position[0] + gy * position[1]
    assert np.isclose(particle.to_numpy()["pressure"][0], expected, rtol=1.0e-6, atol=1.0e-10)


def test_pressure_sampling_ignores_diagonal_air_at_solid_corner():
    cell_type = ti.field(int, shape=(NX, NY))
    cell_pressure = ti.field(float, shape=(NX, NY))
    fluid_sdf = ti.field(float, shape=(NX, NY))
    out = ti.field(float, shape=1)
    p0 = 2468.0

    _sample_bottom_corner_pressure_case(cell_type, cell_pressure, fluid_sdf, out, p0)

    assert np.isclose(out.to_numpy()[0], p0, rtol=1.0e-6, atol=1.0e-9)


def test_pressure_sampling_preserves_constant_pressure_inside_wall_half_cell():
    init(dim=2, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    cell_type = ti.field(int, shape=(NX, NY))
    cell_pressure = ti.field(float, shape=(NX, NY))
    fluid_sdf = ti.field(float, shape=(NX, NY))
    out = ti.field(float, shape=1)
    p0 = 2468.0

    _sample_wall_pressure_case(cell_type, cell_pressure, fluid_sdf, out, p0)

    assert np.isclose(out.to_numpy()[0], p0, rtol=1.0e-6, atol=1.0e-9)


def test_mac_sampling_preserves_tangential_velocity_inside_wall_half_cell():
    init(dim=2, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    velocity_x = ti.field(float, shape=(NX + 1, NY))
    velocity_y = ti.field(float, shape=(NX, NY + 1))
    out = ti.field(float, shape=1)

    _sample_wall_tangential_velocity_case(velocity_x, velocity_y, out)

    assert np.isclose(out.to_numpy()[0], 2.0, rtol=1.0e-6, atol=1.0e-9)


def test_mac_sampling_preserves_3d_tangential_velocity_inside_wall_half_cell():
    init(dim=3, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    velocity_x = ti.field(float, shape=(NX + 1, NY, 4))
    velocity_y = ti.field(float, shape=(NX, NY + 1, 4))
    velocity_z = ti.field(float, shape=(NX, NY, 5))
    out = ti.field(float, shape=1)

    _sample_wall_tangential_velocity_case3d(velocity_x, velocity_y, velocity_z, out)

    assert np.isclose(out.to_numpy()[0], 2.0, rtol=1.0e-6, atol=1.0e-9)


def test_darcy_drag_matches_paper_coefficient_for_small_timestep():
    dt = ti.field(float, shape=())
    dt[None] = 1.0e-12
    out = ti.field(float, shape=4)
    permeability = 1.0e-5
    _compute_drag(1.0e4, -2.0e3, PHI, permeability, 1.0, dt, out)
    solid_acc = out.to_numpy()[:2]
    fluid_acc = out.to_numpy()[2:]
    rel = np.array([1.0e4, -2.0e3])
    coeff = PHI * PHI * 9.8e3 / permeability
    expected_solid_acc = coeff * rel / (RHO_S * (1.0 - PHI))
    expected_fluid_acc = -coeff * rel / (RHO_F * PHI)
    momentum_residual = RHO_S * (1.0 - PHI) * solid_acc + RHO_F * PHI * fluid_acc
    relative_work = np.dot(rel, fluid_acc - solid_acc)
    assert np.allclose(solid_acc, expected_solid_acc, rtol=5.0e-7, atol=1.0e-7)
    assert np.allclose(fluid_acc, expected_fluid_acc, rtol=5.0e-7, atol=1.0e-7)
    assert np.linalg.norm(momentum_residual) < 1.0e-6 * max(1.0, np.linalg.norm(RHO_F * PHI * fluid_acc))
    assert relative_work < 0.0


def test_darcy_drag_matches_paper_consolidation_coefficient_for_small_timestep():
    dt = ti.field(float, shape=())
    dt[None] = 1.0e-12
    out = ti.field(float, shape=4)
    rel = np.array([1.0e-5, -2.0e-6])
    _compute_drag(float(rel[0]), float(rel[1]), PHI, 1.0e-3, 1.0, dt, out)
    solid_acc = out.to_numpy()[:2]
    fluid_acc = out.to_numpy()[2:]

    coeff = PHI * PHI * 9.8e3 / 1.0e-3
    expected_solid_acc = coeff * rel / (RHO_S * (1.0 - PHI))
    expected_fluid_acc = -coeff * rel / (RHO_F * PHI)
    assert np.allclose(solid_acc, expected_solid_acc, rtol=5.0e-7, atol=1.0e-12)
    assert np.allclose(fluid_acc, expected_fluid_acc, rtol=5.0e-7, atol=1.0e-12)


def test_stiff_drag_decays_relative_velocity_without_reversal():
    dt = ti.field(float, shape=())
    dt[None] = DT_VALUE
    out = ti.field(float, shape=4)
    rel = np.array([1.0e4, -2.0e3])
    _compute_drag(float(rel[0]), float(rel[1]), PHI, 1.0e-5, 1.0, dt, out)
    solid_acc = out.to_numpy()[:2]
    fluid_acc = out.to_numpy()[2:]
    updated_rel = rel + DT_VALUE * (fluid_acc - solid_acc)
    coeff = PHI * PHI * 9.8e3 / 1.0e-5
    decay_rate = coeff * (1.0 / (RHO_S * (1.0 - PHI)) + 1.0 / (RHO_F * PHI))
    expected_rel = rel / (1.0 + decay_rate * DT_VALUE)
    momentum_residual = RHO_S * (1.0 - PHI) * solid_acc + RHO_F * PHI * fluid_acc
    assert np.allclose(updated_rel, expected_rel, rtol=2.0e-5, atol=1.0e-5)
    assert np.linalg.norm(momentum_residual) < 1.0e-6 * max(1.0, np.linalg.norm(RHO_F * PHI * fluid_acc))


def test_drag_vanishes_without_solid_phase():
    dt = ti.field(float, shape=())
    dt[None] = DT_VALUE
    out = ti.field(float, shape=4)
    _compute_drag(1.0, -2.0, 1.0, 1.0e-5, 1.0, dt, out)
    assert np.allclose(out.to_numpy(), 0.0)


def test_beetstra_drag_matches_equation_17_for_small_timestep():
    init(dim=2, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    dt = ti.field(float, shape=())
    dt[None] = 1.0e-12
    out = ti.field(float, shape=4)
    porosity = 0.39
    diameter = 1.0e-2
    rel = np.array([0.2, -0.05])
    _compute_drag(float(rel[0]), float(rel[1]), porosity, 1.0, 2.0, dt, out)

    solid_fraction = 1.0 - porosity
    reynolds = RHO_F * diameter * np.linalg.norm(rel) / 1.0e-3
    f0 = 10.0 * solid_fraction / porosity**2 + porosity**2 * (1.0 + 1.5 * np.sqrt(solid_fraction))
    fre = 0.413 * reynolds / (24.0 * porosity**2)
    fre *= (1.0 / porosity + 3.0 * solid_fraction * porosity + 8.4 * reynolds**-0.343) / (
        1.0 + 10.0 ** (3.0 * solid_fraction) * reynolds ** (-(1.0 + 4.0 * solid_fraction) / 2.0)
    )
    coeff = porosity * solid_fraction * 18.0e-3 * (f0 + fre) / diameter**2
    expected_solid = coeff * rel / (RHO_S * solid_fraction)
    expected_fluid = -coeff * rel / (RHO_F * porosity)
    result = out.to_numpy()
    assert np.allclose(result[:2], expected_solid, rtol=5.0e-7, atol=1.0e-8)
    assert np.allclose(result[2:], expected_fluid, rtol=5.0e-7, atol=1.0e-8)


def test_elastic_oedometer_modulus_matches_terzaghi_compressibility():
    young = 1.0e8
    poisson = 0.3
    strain = -1.0e-6
    out = ti.field(float, shape=3)
    _compute_elastic_oedometer_response(young, poisson, strain, out)

    constrained_modulus = young * (1.0 - poisson) / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    lateral_modulus = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    moduli = out.to_numpy()
    assert np.isclose(moduli[0], constrained_modulus, rtol=5.0e-7)
    assert np.isclose(moduli[1], lateral_modulus, rtol=5.0e-7)
    assert np.isclose(moduli[2], lateral_modulus, rtol=5.0e-7)


def test_twophase_particle_traction_can_keep_reference_area_fixed():
    dt = ti.field(float, shape=())
    dt[None] = DT_VALUE
    constraints = ParticleLoadTwoPhase2D.field(shape=1)
    node = NodeTwoPhase2D.field(shape=(1, 1))
    particle = ParticleCloudTwoPhase2D.field(shape=1)
    lnid = ti.field(int, shape=1)
    shape_fn = ti.field(float, shape=1)
    node_size = ti.field(int, shape=1)

    _setup_particle_traction_case(constraints, node, particle, lnid, shape_fn, node_size)
    apply_particle_traction_constraint_twophase(
        1,
        1,
        constraints,
        dt,
        node,
        particle,
        lnid,
        shape_fn,
        node_size,
        0,
    )
    fixed_area = constraints.to_numpy()["psize"][0]
    assert np.allclose(fixed_area, [0.005, 0.005])

    _setup_particle_traction_case(constraints, node, particle, lnid, shape_fn, node_size)
    apply_particle_traction_constraint_twophase(
        1,
        1,
        constraints,
        dt,
        node,
        particle,
        lnid,
        shape_fn,
        node_size,
        1,
    )
    updated_area = constraints.to_numpy()["psize"][0]
    assert not np.allclose(updated_area, [0.005, 0.005])


if __name__ == "__main__":
    test_pressure_matrix_reduces_to_uniform_laplacian()
    test_double_layer_grid_reset_clears_fluid_only_nodes()
    test_double_layer_fluid_volume_correction_preserves_mass_and_solid_position()
    test_mac_face_porosity_without_solid_support_tends_smoothly_to_one()
    test_cell_centered_transfer_keeps_both_sides_of_grid_crossing_in_pressure_domain()
    test_double_layer_multigrid_keeps_thin_fluid_support_on_coarse_level()
    test_solid_cell_region_does_not_mark_adjacent_fluid_cell_on_grid_line()
    test_pressure_matrix_uses_neumann_for_solid_walls_and_dirichlet_for_air()
    test_coarse_pressure_matrix_uses_gfm_theta_for_free_surface()
    test_pressure_rhs_matches_weighted_divergence()
    test_pressure_projection_reduces_weighted_divergence_with_code_sign_convention()
    test_pressure_gradient_correction_matches_linear_pressure()
    test_paper_solid_pressure_correction_matches_linear_pressure_gradient()
    test_fluid_pressure_correction_uses_free_surface_dirichlet_gradient()
    test_solid_pressure_node_projection_preserves_interior_linear_pressure()
    test_pressure_sampling_ignores_diagonal_air_at_solid_corner()
    test_darcy_drag_matches_paper_coefficient_without_limiter()
    test_darcy_drag_matches_paper_consolidation_coefficient_when_unlimited()
    test_drag_vanishes_without_solid_phase()
    test_elastic_oedometer_modulus_matches_terzaghi_compressibility()
    test_twophase_particle_traction_can_keep_reference_area_fixed()
