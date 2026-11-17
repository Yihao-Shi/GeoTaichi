"""Taichi kernels for the u-p two-phase MPM engine."""

import taichi as ti
from src.utils.constants import ZEROVEC2F
from src.utils.TypeDefinition import vec2f


@ti.kernel
def kernel_recover_darcy_fluid_velocity_2d(
    total_nodes: int,
    start_index: int,
    end_index: int,
    material_mapping: ti.template(),
    mat_prop: ti.template(),
    beta: float,
    gravity: ti.types.vector(3, float),
    particle: ti.template(),
    node: ti.template(),
    ln_id: ti.template(),
    dshapefn: ti.template(),
    node_size: ti.template(),
):
    # The u-p pressure RHS uses quasi-static Darcy flow, without fluid inertia:
    # q = n*(vf-vs) = -k/gamma_f * (grad(p_new) - rho_f*g).
    # Recover this dependent variable; do not advance a second momentum equation.
    for i in range(start_index, end_index):
        np = material_mapping[i]
        if int(particle[np].active) == 1 and int(particle[np].materialID) > 0:
            offset = np * total_nodes
            body_id = int(particle[np].bodyID)
            gradient = ZEROVEC2F
            for ln in range(offset, offset + int(node_size[np])):
                ng = ln_id[ln]
                pressure = beta * node[ng, body_id].pressure + node[ng, body_id].dpressure
                gradient += dshapefn[ln] * pressure
            driving_gradient = gradient - mat_prop.fluid_density * vec2f([gravity[0], gravity[1]])
            flux = -mat_prop.permeability / mat_prop.fluid_unit_weight * driving_gradient
            particle[np].vf = particle[np].vs + flux / particle[np].porosity
