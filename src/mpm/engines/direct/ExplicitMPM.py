import taichi as ti
import numpy as np

import src.mpm.config as config
from src.mpm.engines.direct.MPMSolver import MPMSolver
from src.mpm.utils import RodriguesRotationMatrix, mat3x3, vec3f, vectorize_id
from src.utils.linalg import no_operation


@ti.data_oriented
class ExplicitMPM(MPMSolver):
    def __init__(self, bodies, dirichlet=None, neumann=None, name="case", configuration="UL", **kwargs):
        self.configuration = configuration
        self.handle_material(**kwargs)
        super().__init__(bodies, dirichlet, neumann, name, solver="Explicit", **kwargs)
        self.apply_neumann_step = self.apply_neumann if self.neumann.num > 0 else no_operation
        self.apply_dirichlet_step = self.apply_dirichlet if self.dirichlet.num > 0 else no_operation
        if self.is_axisymmetric:
            raise ValueError(
                "Direct explicit MPM does not implement no-swirl "
                "axisymmetric kinematics; use Direct implicit ULMPM or the "
                "standard axisymmetric MPM backend"
            )
        stateVariable = self.material.get_state_vars()
        if "deformation_gradient" in stateVariable:
            stateVariable.update({"deformation_gradient": ti.types.matrix(config.DIM, config.DIM, ti.f64)})
        self.stateVars = ti.Struct.field(stateVariable, shape=self.n_particles)
        if kwargs.get("volumetric_smooth", False):
            self.gvol = ti.field(dtype=ti.f64, shape=self.total_grid_num)
            self.gjacobian = ti.field(dtype=ti.f64, shape=self.total_grid_num)

    def handle_material(self, **kwargs):
        constitutive_model = kwargs.get("material", "LinearElastic")
        if constitutive_model == "HenckyElastic":
            from src.physics_model.consititutive_model.finite_strain.HenckyElastic import HenckyElasticModel

            self.material_type = 0
            self.material = HenckyElasticModel(
                material_type="Solid", configuration=self.configuration, solver_type="Explicit"
            )
        elif constitutive_model == "NeoHookean":
            from src.physics_model.consititutive_model.finite_strain.NeoHookean import NeoHookeanModel

            self.material_type = 0
            self.material = NeoHookeanModel(
                material_type="Solid", configuration=self.configuration, solver_type="Explicit"
            )
        elif constitutive_model == "MooneyRivlin":
            from src.physics_model.consititutive_model.finite_strain.MooneyRivlin import MooneyRivlin

            self.material_type = 0
            self.material = MooneyRivlin(
                material_type="Solid", configuration=self.configuration, solver_type="Explicit"
            )
        elif constitutive_model == "Gent":
            from src.physics_model.consititutive_model.finite_strain.Gent import Gent

            self.material_type = 0
            self.material = Gent(material_type="Solid", configuration=self.configuration, solver_type="Explicit")
        elif constitutive_model == "Hydrogel":
            from src.physics_model.consititutive_model.finite_strain.Hydrogel import Hydrogel

            self.material_type = 0
            self.material = Hydrogel(material_type="Solid", configuration=self.configuration, solver_type="Explicit")
        elif constitutive_model == "LinearElastic":
            from src.physics_model.consititutive_model.infinitesimal_strain.LinearElastic import LinearElasticModel

            self.material_type = 1
            self.material = LinearElasticModel(
                material_type="Solid", configuration=self.configuration, solver_type="Explicit"
            )
        elif constitutive_model == "ElasticPerfectlyPlastic":
            from src.physics_model.consititutive_model.infinitesimal_strain.ElasticPerfectlyPlastic import (
                ElasticPerfectlyPlasticModel,
            )

            stress_integration = kwargs.get("stress_integration", "ReturnMapping")
            self.material_type = 1
            self.material = ElasticPerfectlyPlasticModel(
                material_type="Solid",
                configuration=self.configuration,
                solver_type="Explicit",
                stress_integration=stress_integration,
            )
        elif constitutive_model == "MohrCoulomb":
            from src.physics_model.consititutive_model.infinitesimal_strain.MohrCoulomb import MohrCoulombModel

            stress_integration = kwargs.get("stress_integration", "ReturnMapping")
            self.material_type = 1
            self.material = MohrCoulombModel(
                material_type="Solid",
                configuration=self.configuration,
                solver_type="Explicit",
                stress_integration=stress_integration,
            )
        elif constitutive_model == "StateDependentMohrCoulomb":
            from src.physics_model.consititutive_model.infinitesimal_strain.StateDependentMohrCoulomb import (
                StateDependentMohrCoulombModel,
            )

            stress_integration = kwargs.get("stress_integration", "ReturnMapping")
            self.material_type = 1
            self.material = StateDependentMohrCoulombModel(
                material_type="Solid",
                configuration=self.configuration,
                solver_type="Explicit",
                stress_integration=stress_integration,
            )
        elif constitutive_model == "DruckerPrager":
            from src.physics_model.consititutive_model.infinitesimal_strain.DruckerPrager import DruckerPragerModel

            stress_integration = kwargs.get("stress_integration", "ReturnMapping")
            self.material_type = 1
            self.material = DruckerPragerModel(
                material_type="Solid",
                configuration=self.configuration,
                solver_type="Explicit",
                stress_integration=stress_integration,
            )
        elif constitutive_model == "ModifiedCamClay":
            from src.physics_model.consititutive_model.infinitesimal_strain.ModifiedCamClay import ModifiedCamClayModel

            stress_integration = kwargs.get("stress_integration", "ReturnMapping")
            self.material_type = 1
            self.material = ModifiedCamClayModel(
                material_type="Solid",
                configuration=self.configuration,
                solver_type="Explicit",
                stress_integration=stress_integration,
            )
        elif constitutive_model == "Newtonian":
            from src.physics_model.consititutive_model.strain_rate.Newtonian import NewtonianModel

            self.material_type = 2
            self.material = NewtonianModel(
                material_type="Solid", configuration=self.configuration, solver_type="Explicit"
            )
        elif constitutive_model == "Bingham":
            from src.physics_model.consititutive_model.strain_rate.Bingham import BinghamModel

            self.material_type = 2
            self.material = BinghamModel(
                material_type="Solid", configuration=self.configuration, solver_type="Explicit"
            )
        else:
            raise ValueError(f"Constitutive Model: {constitutive_model} error!")
        self.material.model_initialize(self._material_parameters(kwargs))
        self.material.print_message(0)

    def _material_parameters(self, kwargs):
        material_parameters = dict(kwargs.get("material_parameters") or {})
        aliases = {
            "Density": ("density", "Density"),
            "YoungModulus": ("young_modulus", "YoungModulus", "ElasticModulus"),
            "PoissonRatio": ("poisson_ratio", "PoissonRatio"),
        }
        for target, keys in aliases.items():
            if target in material_parameters:
                continue
            for key in keys:
                if key in kwargs:
                    material_parameters[target] = kwargs[key]
                    break
        return material_parameters

    def initial_gravity(self, dist):
        poisson = self.material.poisson
        k0 = np.repeat(poisson / (1.0 - poisson), self.n_particles)

        @ti.kernel
        def set_initial_gravity(
            gravity: ti.types.vector(3, ti.f64), k0: ti.types.ndarray(), distance: ti.types.ndarray()
        ):
            for i in range(self.particleNum[0]):
                direction = gravity.normalized()
                gamma = self.material.density * distance[i] * gravity.norm()
                initial_gravity_stress = mat3x3([k0[i] * gamma, 0.0, 0.0], [0.0, k0[i] * gamma, 0.0], [0.0, 0.0, gamma])
                rotation_matrix = RodriguesRotationMatrix(-direction, vec3f(0.0, 0.0, 1.0))
                gravity_field = rotation_matrix.transpose() @ initial_gravity_stress @ rotation_matrix
                self.particle[i].stress += gravity_field

        grav = self.gravity
        if config.DIM == 2:
            grav = [*grav, 0.0]
        set_initial_gravity(grav, k0, dist)

    @ti.kernel
    def initial_material(self):
        for i in range(self.particleNum[0]):
            self.material._initialize_vars(i, self.particle, self.stateVars)

    @ti.kernel
    def force_p2g(self, gravity: ti.types.vector(config.DIM, ti.f64)):
        for i in range(self.particleNum[0]):
            p_vol = self.particle[i].vol0
            pmass = self.particle[i].m
            p_stress = self.particle[i].stress
            external_force = pmass * gravity
            internal_force = -p_vol * p_stress
            for j in range(self.offset[i]):
                nodeID = self.LnID[i, j]
                shape_grad = self.dshape[i, j]
                BTsigma = ti.Vector.zero(ti.f64, config.DIM)
                if ti.static(config.DIM == 2):
                    BTsigma = ti.Vector(
                        [
                            internal_force[0, 0] * shape_grad[0] + internal_force[0, 1] * shape_grad[1],
                            internal_force[1, 0] * shape_grad[0] + internal_force[1, 1] * shape_grad[1],
                        ]
                    )
                else:
                    BTsigma = internal_force @ shape_grad
                force = BTsigma + external_force * self.shape[i, j]
                self.grid[nodeID].a += force

    @ti.kernel
    def volumertic_smooth_p2g(self):
        for i in range(self.particleNum[0]):
            p_vol = self.particle[i].vol
            for j in range(self.offset[i]):
                nodeID = self.LnID[i, j]
                shape_fn = self.shape[i, j]
                self.gvol[nodeID] += shape_fn * p_vol

    @ti.kernel
    def traction_p2g(self):
        for i in range(self.tractionNum[0]):
            pid = self.traction[i].particleID
            traction = self.particle_traction_force(i)
            for j in range(self.offset[pid]):
                nodeID = self.LnID[pid, j]
                shape_fn = self.shape[pid, j]
                force = traction * shape_fn
                self.grid[nodeID].a += force

    @ti.kernel
    def grid_kinematics(self, damping: ti.f64):
        dt = self.TIdt[None]
        for i in self.grid:
            if self.grid[i].m > self.val_lim:
                self.grid[i].a += (
                    -damping * self.grid[i].v / self.grid[i].v.norm() * self.grid[i].a.norm()
                    if self.grid[i].v.norm() > self.val_lim
                    else ti.Vector.zero(ti.f64, config.DIM)
                )
                self.grid[i].a /= self.grid[i].m
                self.grid[i].v += self.grid[i].a * dt

    @ti.kernel
    def advent_particles(self, coeffPIC: float):
        dt = self.TIdt[None]
        for i in range(self.particleNum[0]):
            bodyID = self.particle[i].bodyID
            goffset = self.body[bodyID].goffset
            grid_num = self.body[bodyID].grid_num
            grid_size = self.body[bodyID].grid_size
            xmin = self.body[bodyID].xmin

            p_pos = self.particle[i].x
            acc = ti.Vector.zero(ti.f64, config.DIM)
            vel = ti.Vector.zero(ti.f64, config.DIM)
            Bp = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
            Dp = ti.Matrix.zero(ti.f64, config.DIM, config.DIM)
            v0 = self.particle[i].v
            for j in range(self.offset[i]):
                grid_id = self.LnID[i, j]
                shape_fn = self.shape[i, j]
                grid_v = self.grid[grid_id].v
                acc += shape_fn * self.grid[grid_id].a
                vel += shape_fn * grid_v
                if ti.static(self.velocity_proj):
                    dpos = xmin + grid_size * ti.Vector(vectorize_id(grid_id - goffset, grid_num)) - p_pos
                    Bp += shape_fn * grid_v.outer_product(dpos)
                    Dp += shape_fn * dpos.outer_product(dpos)
            v1 = (1.0 - coeffPIC) * (v0 + acc * dt) + coeffPIC * vel
            self.particle[i].v = v1
            self.particle[i].x += vel * dt
            if ti.static(self.velocity_proj):
                self.gradv[i] = Bp @ Dp.inverse()

    @ti.kernel
    def apply_dirichlet(self):
        for i in self.grid:
            if self.grid[i].m > self.val_lim:
                for d in ti.static(range(config.DIM)):
                    if self.dirichlet.node[config.DIM * i + d] == 1:
                        self.grid[i].v[d] = self.dirichlet.value[config.DIM * i + d]
                        self.grid[i].a[d] = 0.0

    @ti.kernel
    def apply_neumann(self):
        for i in self.neumann.node:
            node_dof = self.neumann.node[i]
            node_id = int(node_dof // config.DIM)
            d = int(node_dof % config.DIM)
            if self.grid[node_id].m > self.val_lim:
                self.grid[node_id].a[d] += self.neumann.value[i] / self.grid[node_id].m

    def calculate_von_mises(self):
        particle_num = self.particleNum.to_numpy()[0]
        stress = self.particle.stress.to_numpy()[:particle_num]
        I = np.repeat(np.eye(3)[None, :, :], particle_num, axis=0)
        trace_sigma = np.einsum("...ii", stress) / 3.0
        s = stress - np.einsum("i,ijk->ijk", trace_sigma, I)
        J2 = 0.5 * np.einsum("ijk,ijk->i", s, s)
        return np.sqrt(3.0 * J2)
