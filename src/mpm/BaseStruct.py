import taichi as ti

from src.utils.constants import ZEROVEC2f, ZEROVEC3f, Threshold, DELTA, ZEROVEC6f
from src.utils.ScalarFunction import sign
from src.utils.TypeDefination import vec2f, vec3f, vec2u8, vec3u8, vec6f, mat2x2, mat3x3
from src.utils.VectorFunction import Zero2OneVector, SquareLen, MeanValue, vsign


@ti.dataclass
class ParticleCloud2D:      # memory usage: 108B
    bodyID: ti.u8
    materialID: ti.u8
    active: ti.u8
    m: float
    vol: float
    psize: vec2f
    x: vec2f
    v: vec2f
    traction: vec2f
    strain: vec6f
    stress: vec6f
    velocity_gradient: mat2x2
    fix_v: vec2u8
    unfix_v: vec2u8

    @ti.func
    def _restart(self, bodyID, materialID, active, mass, position, velocity, volume, traction, strain, stress, psize, velocity_gradient, fix_v):
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.active = ti.u8(active)
        self.m = float(mass)
        self.x = float(position)
        self.v = float(velocity)
        self.vol = float(volume)
        self.traction = float(traction) * float(volume) * vec3f(1./psize[2], 1./psize[1], 1./psize[0])
        self.strain = float(strain)
        self.stress = float(stress)
        self.psize = float(psize)
        self.velocity_gradient = float(velocity_gradient)
        self.fix_v = ti.cast(fix_v, ti.u8)
        self.unfix_v = ti.cast(Zero2OneVector(fix_v), ti.u8)

    @ti.func
    def _set_essential(self, bodyID, materialID, density, particle_volume, psize, position, init_v, fix_v):
        self.active = ti.u8(1)
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.vol = float(particle_volume)
        self.psize = float(psize)
        self.m = float(particle_volume * density)
        self.x = float(position)
        self.v = init_v
        self.fix_v = fix_v
        self.unfix_v = Zero2OneVector(fix_v)

    @ti.func
    def _add_gravity_field(self, k0, gamma):
        self.stress[1] += float(gamma)
        self.stress[0] += float(k0 * gamma)
        self.stress[2] += float(k0 * gamma)

    @ti.func
    def _set_particle_traction(self, traction):
        self.traction += 0.5 * float(traction) * self.vol * vec2f(1./self.psize[2], 1./self.psize[1])

    @ti.func
    def _compute_external_force(self, gravity):
        return self.m * gravity + self.traction
    
    @ti.func
    def _compute_internal_force(self):
        return -self.vol * self.stress 
    
    @ti.func
    def _update_particle_state(self, dt, alpha, vPIC, vFLIP):
        v0 = self.v
        flag1 = int(self.fix_v)
        flag2 = int(self.unfix_v)
        self.v = (alpha * vPIC + (1 - alpha) * (vFLIP + v0)) * flag2 + v0 * flag1
        self.x += vPIC * dt[None] * flag2 + v0 * dt[None] * flag1

    @ti.func
    def _update_rigid_body(self, dt):
        self.x += self.v * dt[None]
    
    @ti.func
    def _compute_particle_velocity(self, xg):
        return self.v - self.velocity_gradient @ (self.x - xg)
    

@ti.dataclass
class ParticleCloud:      # memory usage: 108B
    bodyID: ti.u8
    materialID: ti.u8
    active: ti.u8
    m: float
    vol: float
    psize: vec3f
    x: vec3f
    v: vec3f
    traction: vec3f
    strain: vec6f
    stress: vec6f
    velocity_gradient: mat3x3
    fix_v: vec3u8
    unfix_v: vec3u8

    @ti.func
    def _restart(self, bodyID, materialID, active, mass, position, velocity, volume, traction, strain, stress, psize, velocity_gradient, fix_v):
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.active = ti.u8(active)
        self.m = float(mass)
        self.x = float(position)
        self.v = float(velocity)
        self.vol = float(volume)
        self.traction = float(traction) * float(volume) * vec3f(1./psize[2], 1./psize[1], 1./psize[0])
        self.strain = float(strain)
        self.stress = float(stress)
        self.psize = float(psize)
        self.velocity_gradient = float(velocity_gradient)
        self.fix_v = ti.cast(fix_v, ti.u8)
        self.unfix_v = ti.cast(Zero2OneVector(fix_v), ti.u8)

    @ti.func
    def _set_essential(self, bodyID, materialID, density, particle_volume, psize, position, init_v, fix_v):
        self.active = ti.u8(1)
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.vol = float(particle_volume)
        self.psize = float(psize)
        self.m = float(particle_volume * density)
        self.x = float(position)
        self.v = init_v
        self.fix_v = fix_v
        self.unfix_v = Zero2OneVector(fix_v)

    @ti.func
    def _add_gravity_field(self, k0, gamma):
        self.stress[2] += float(gamma)
        self.stress[0] += float(k0 * gamma)
        self.stress[1] += float(k0 * gamma)

    @ti.func
    def _set_particle_traction(self, traction):
        self.traction += 0.5 * float(traction) * self.vol * vec3f(1./self.psize[2], 1./self.psize[1], 1./self.psize[0])

    @ti.func
    def _compute_external_force(self, gravity):
        return self.m * gravity + self.traction
    
    @ti.func
    def _compute_internal_force(self):
        return -self.vol * self.stress 
    
    @ti.func
    def _update_particle_state(self, dt, alpha, vPIC, vFLIP):
        v0 = self.v
        flag1 = int(self.fix_v)
        flag2 = int(self.unfix_v)
        self.v = (alpha * vPIC + (1 - alpha) * (vFLIP + v0)) * flag2 + v0 * flag1
        self.x += vPIC * dt[None] * flag2 + v0 * dt[None] * flag1

    @ti.func
    def _update_rigid_body(self, dt):
        self.x += self.v * dt[None]
    
    @ti.func
    def _compute_particle_velocity(self, xg):
        return self.v - self.velocity_gradient @ (self.x - xg)
    
    @ti.func
    def _calc_psize_cp(self, dt):
        deformation_gradient_rate = DELTA + dt[None] * self.velocity_gradient
        self.psize[0] *= deformation_gradient_rate[0, 0] 
        self.psize[1] *= deformation_gradient_rate[1, 1]
        self.psize[2] *= deformation_gradient_rate[2, 2] 

    @ti.func
    def _calc_psize_r(self, dt):
        deformation_gradient_rate = DELTA + dt[None] * self.velocity_gradient
        self.psize[0] *= ti.sqrt(deformation_gradient_rate[0, 0] ** 2 + deformation_gradient_rate[1, 0] ** 2 + deformation_gradient_rate[0, 2] ** 2)
        self.psize[1] *= ti.sqrt(deformation_gradient_rate[0, 1] ** 2 + deformation_gradient_rate[1, 1] ** 2 + deformation_gradient_rate[1, 2] ** 2)
        self.psize[2] *= ti.sqrt(deformation_gradient_rate[0, 2] ** 2 + deformation_gradient_rate[1, 2] ** 2 + deformation_gradient_rate[2, 2] ** 2)


@ti.dataclass
class ParticleFBar:
    jacobian: float
    djacobian: float

    @ti.func
    def initialize(self):
        self.jacobian = 1.
        self.djacobian = 0.


@ti.dataclass
class ParticleCoupling:      # memory usage: 108B
    bodyID: ti.u8
    materialID: ti.u8
    active: ti.u8
    free_surface: ti.u8
    m: float
    vol: float
    rad: float
    mass_density: float
    psize: vec3f
    normal: vec3f
    x: vec3f
    verletDisp: vec3f
    v: vec3f
    contact_force: vec3f
    traction: vec3f
    strain: vec6f
    stress: vec6f
    velocity_gradient: mat3x3
    fix_v: vec3u8
    unfix_v: vec3u8

    @ti.func
    def _restart(self, bodyID, materialID, active, free_surface, normal, mass, position, velocity, volume, 
                 traction, strain, stress, psize, velocity_gradient, fix_v):
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.active = ti.u8(active)
        self.free_surface = ti.u8(free_surface)
        self.normal = float(normal)
        self.m = float(mass)
        self.x = float(position)
        self.v = float(velocity)
        self.vol = float(volume)
        self.traction = float(traction) * float(volume) * vec3f(1./psize[2], 1./psize[1], 1./psize[0])
        self.strain = strain
        self.stress = float(stress)
        self.psize = float(psize)
        self.velocity_gradient = float(velocity_gradient)
        self.fix_v = ti.cast(fix_v, ti.u8)
        self.unfix_v = ti.cast(Zero2OneVector(fix_v), ti.u8)

    @ti.func
    def _set_essential(self, bodyID, materialID, density, particle_volume, psize, position, init_v, fix_v):
        self.active = ti.u8(1)
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.vol = float(particle_volume)
        self.psize = float(psize)
        self.rad = 0.5 * float((particle_volume) ** (1./3.))
        self.m = float(particle_volume * density)
        self.x = float(position)
        self.v = init_v
        self.fix_v = fix_v
        self.unfix_v = Zero2OneVector(fix_v)

    @ti.func
    def _add_gravity_field(self, k0, gamma):
        self.stress[2] += float(gamma)
        self.stress[0] += float(k0 * gamma)
        self.stress[1] += float(k0 * gamma)

    @ti.func
    def _set_particle_traction(self, traction):
        self.traction += 0.5 * float(traction) * self.vol * vec3f(1./self.psize[2], 1./self.psize[1], 1./self.psize[0])

    @ti.func
    def _reset_contact_force(self):
        self.contact_force = ZEROVEC3f

    @ti.func
    def _reset_mass_density(self):
        self.mass_density = 0.

    @ti.func
    def _compute_external_force(self, gravity):
        return self.m * gravity + self.contact_force + self.traction 
    
    @ti.func
    def _compute_internal_force(self):
        return -self.vol * self.stress 

    @ti.func
    def _update_particle_state(self, dt, alpha, vPIC, vFLIP):
        v0 = self.v
        flag1 = int(self.fix_v)
        flag2 = int(self.unfix_v)
        self.v = (alpha * vPIC + (1 - alpha) * (vFLIP + v0)) * flag2 + v0 * flag1
        deltax = vPIC * dt[None] * flag2 + v0 * dt[None] * flag1
        self.x += deltax
        self.verletDisp += deltax

    @ti.func
    def _update_rigid_body(self, dt):
        deltax = self.v * dt[None]
        self.x += deltax
        self.verletDisp += deltax
    
    @ti.func
    def _compute_particle_velocity(self, xg):
        return self.v - self.velocity_gradient @ (self.x - xg)
    
    @ti.func
    def _update_contact_interaction(self, cforce):
        self.contact_force += cforce
    
    @ti.func
    def _calc_psize_cp(self, dt):
        deformation_gradient_rate = DELTA + dt[None] * self.velocity_gradient
        self.psize[0] *= deformation_gradient_rate[0, 0] 
        self.psize[1] *= deformation_gradient_rate[1, 1]
        self.psize[2] *= deformation_gradient_rate[2, 2] 

    @ti.func
    def _calc_psize_r(self, dt):
        deformation_gradient_rate = DELTA + dt[None] * self.velocity_gradient
        self.psize[0] *= ti.sqrt(deformation_gradient_rate[0, 0] ** 2 + deformation_gradient_rate[1, 0] ** 2 + deformation_gradient_rate[0, 2] ** 2)
        self.psize[1] *= ti.sqrt(deformation_gradient_rate[0, 1] ** 2 + deformation_gradient_rate[1, 1] ** 2 + deformation_gradient_rate[1, 2] ** 2)
        self.psize[2] *= ti.sqrt(deformation_gradient_rate[0, 2] ** 2 + deformation_gradient_rate[1, 2] ** 2 + deformation_gradient_rate[2, 2] ** 2)

@ti.dataclass
class ContactNodes:
    m: float
    contact_pos: vec3f
    force: vec3f
    momentum: vec3f
    contact_force: vec3f
    grad_domain: vec3f

    @ti.func
    def _tlgrid_reset(self):
        self.momentum = ZEROVEC3f
        self.force = ZEROVEC3f
        self.contact_force = ZEROVEC3f
        self.grad_domain = ZEROVEC3f
        self.contact_pos = ZEROVEC3f

    @ti.func
    def _grid_reset(self):
        self.m = 0.
        self.momentum = ZEROVEC3f
        self.force = ZEROVEC3f
        self.contact_force = ZEROVEC3f
        self.grad_domain = ZEROVEC3f
        self.contact_pos = ZEROVEC3f

    @ti.func
    def _set_dofs(self, rowth):
        pass

    @ti.func
    def _update_nodal_mass(self, m):
        self.m += m

    @ti.func
    def _update_nodal_momentum(self, momentum):
        self.momentum += momentum

    @ti.func
    def _compute_nodal_velocity(self):
        self.momentum /= self.m

    @ti.func
    def _update_nodal_force(self, force):
        self.force += force

    @ti.func
    def _update_external_force(self, external_force):
        self.force += external_force

    @ti.func
    def _update_internal_force(self, internal_force):
        self.force += internal_force

    @ti.func
    def _compute_nodal_kinematic(self, damp, dt):
        unbalanced_force = self.force / self.m
        force = unbalanced_force - damp * unbalanced_force.norm() * vsign(self.momentum)
        self.momentum += force * dt[None]
        self.force = force 

    @ti.func
    def _update_nodal_kinematic(self):
        self.force /= self.m
        self.momentum /= self.m

    @ti.func
    def _recorrect_nodal_kinematic(self, dt):
        contact_acceleration = self.contact_force / self.m
        self.force += contact_acceleration
        self.momentum += contact_acceleration * dt[None]

    @ti.func
    def _update_contact_force(self, force):
        self.contact_force += force

    @ti.func
    def _contact_force_assemble(self, dt):
        contact_acceleration =  self.contact_force / self.m
        self.force += contact_acceleration
        self.momentum += contact_acceleration * dt[None]

    @ti.func
    def _update_nodal_grad_domain(self, grad_domain):
        self.grad_domain += grad_domain

    @ti.func
    def _update_nodal_contact_pos(self, contact_pos):
        self.contact_pos += contact_pos

    @ti.func
    def contact_velocity_constraint(self, fix_v, unfix_v):
        self.contact_force = unfix_v * self.contact_force + fix_v * ZEROVEC3f

    @ti.func
    def contact_reflection_constraint(self, norm1, norm2, norm3):
        pre_accelerate = self.contact_force
        if SquareLen(norm1) > Threshold and pre_accelerate.dot(norm1) > 0:
            pre_accelerate -= pre_accelerate.dot(norm1) * norm1
        if SquareLen(norm2) > Threshold and pre_accelerate.dot(norm2) > 0:
            pre_accelerate -= pre_accelerate.dot(norm2) * norm2
        if SquareLen(norm3) > Threshold and pre_accelerate.dot(norm3) > 0:
            pre_accelerate -= pre_accelerate.dot(norm3) * norm3
        self.contact_force = pre_accelerate
        
    @ti.func
    def velocity_constraint(self, fix_v, unfix_v, prescribed_velocity):
        self.momentum = unfix_v * self.momentum + fix_v * prescribed_velocity 
        self.force = unfix_v * self.force + fix_v * ZEROVEC3f

    @ti.func
    def rigid_body_velocity_constraint(self, fix_v, unfix_v):
        self.grad_domain = unfix_v * self.grad_domain + fix_v * ZEROVEC3f

    @ti.func
    def reflection_constraint(self, norm1, norm2, norm3):
        pre_velocity, pre_accelerate = self.momentum, self.force
        if SquareLen(norm1) > Threshold and pre_velocity.dot(norm1) > 0:
            pre_velocity -= pre_velocity.dot(norm1) * norm1
            pre_accelerate -= pre_accelerate.dot(norm1) * norm1
        if SquareLen(norm2) > Threshold and pre_velocity.dot(norm2) > 0:
            pre_velocity -= pre_velocity.dot(norm2) * norm2
            pre_accelerate -= pre_accelerate.dot(norm2) * norm2
        if SquareLen(norm3) > Threshold and pre_velocity.dot(norm3) > 0:
            pre_velocity -= pre_velocity.dot(norm3) * norm3
            pre_accelerate -= pre_accelerate.dot(norm3) * norm3
        
        self.momentum = pre_velocity
        self.force = pre_accelerate

    @ti.func
    def rigid_body_reflection_constraint(self, norm1, norm2, norm3):
        pre_gradient = self.grad_domain
        if SquareLen(norm1) > Threshold and pre_gradient.dot(norm1) > 0:
            pre_gradient -= pre_gradient.dot(norm1) * norm1
        if SquareLen(norm2) > Threshold and pre_gradient.dot(norm2) > 0:
            pre_gradient -= pre_gradient.dot(norm2) * norm2
        if SquareLen(norm3) > Threshold and pre_gradient.dot(norm3) > 0:
            pre_gradient -= pre_gradient.dot(norm3) * norm3
        self.grad_domain = pre_gradient

    @ti.func
    def friction_constraint(self, mu, norm, dt):
        velocity = self.momentum
        acceleration = self.force / self.m

        '''acc_n = acceleration.dot(norm)
        acc_t = acceleration - acc_n * norm
        vel_n = velocity.dot(norm)
        vel_t = velocity - vel_n * norm
        if acc_n > 0.:
            if vel_t.norm() != 0.:
                vel_net = vel_t + acc_t * dt[None]
                vel_net_t = vel_net.norm()
                vel_friction = mu * ti.abs(acc_n) * dt[None]

                if vel_net_t <= vel_friction:
                    acc_t -= vel_t / dt[None]
                else:
                    acc_t -= mu * ti.abs(acc_n) * (vel_net / vel_net_t)
            else:
                if acc_t.norm() <= mu * ti.abs(acc_n):
                    acc_t = ZEROVEC3f
                else:
                    acc_tt = acc_t.norm() - mu * ti.abs(acc_n)
                    acc_t -= mu * ti.abs(acc_n) * acc_t / acc_tt
            self.force = acc_t * self.m'''

        vel_n = velocity.dot(norm)
        if vel_n > Threshold:
            v0 = velocity - acceleration * dt[None]
            vt0 = v0 - v0.dot(norm) * norm
            vn1 = vel_n * norm
            vt1 = velocity - vn1
            t = vt1.normalized()
            ft = -sign(vt0.dot(t)) * mu * self.force.dot(norm) * t
            vt2 = vt1 + ft * dt[None]

            if vt1.dot(vt2) > Threshold:
                self.force = self.force.dot(t) * t + ft
                self.momentum = vt2
            else:
                self.force = ZEROVEC3f
                self.momentum = ZEROVEC3f

    @ti.func
    def rigid_friction_constraint(self, norm):
        pre_gradient = self.grad_domain
        if SquareLen(norm) > Threshold and pre_gradient.dot(norm) > 0:
            pre_gradient -= pre_gradient.dot(norm) * norm
        self.grad_domain = pre_gradient


@ti.dataclass
class ContactNodes2D:
    m: float
    contact_pos: vec2f
    force: vec2f
    momentum: vec2f
    contact_force: vec2f
    grad_domain: vec2f

    @ti.func
    def _tlgrid_reset(self):
        self.momentum = ZEROVEC2f
        self.force = ZEROVEC2f
        self.contact_force = ZEROVEC2f
        self.grad_domain = ZEROVEC2f
        self.contact_pos = ZEROVEC2f

    @ti.func
    def _grid_reset(self):
        self.m = 0.
        self.momentum = ZEROVEC2f
        self.force = ZEROVEC2f
        self.contact_force = ZEROVEC2f
        self.grad_domain = ZEROVEC2f
        self.contact_pos = ZEROVEC2f

    @ti.func
    def _set_dofs(self, rowth):
        pass

    @ti.func
    def _update_nodal_mass(self, m):
        self.m += m

    @ti.func
    def _update_nodal_momentum(self, momentum):
        self.momentum += momentum

    @ti.func
    def _compute_nodal_velocity(self):
        self.momentum /= self.m

    @ti.func
    def _update_nodal_force(self, force):
        self.force += force

    @ti.func
    def _update_external_force(self, external_force):
        self.force += external_force

    @ti.func
    def _update_internal_force(self, internal_force):
        self.force += internal_force

    @ti.func
    def _compute_nodal_kinematic(self, damp, dt):
        unbalanced_force = self.force / self.m
        force = unbalanced_force - damp * unbalanced_force.norm() * vsign(self.momentum)
        self.momentum += force * dt[None]
        self.force = force 

    @ti.func
    def _update_nodal_kinematic(self):
        self.force /= self.m
        self.momentum /= self.m

    @ti.func
    def _recorrect_nodal_kinematic(self, dt):
        contact_acceleration = self.contact_force / self.m
        self.force += contact_acceleration
        self.momentum += contact_acceleration * dt[None]

    @ti.func
    def _update_contact_force(self, force):
        self.contact_force += force

    @ti.func
    def _contact_force_assemble(self, dt):
        contact_acceleration =  self.contact_force / self.m
        self.force += contact_acceleration
        self.momentum += contact_acceleration * dt[None]

    @ti.func
    def _update_nodal_grad_domain(self, grad_domain):
        self.grad_domain += grad_domain

    @ti.func
    def _update_nodal_contact_pos(self, contact_pos):
        self.contact_pos += contact_pos

    @ti.func
    def contact_velocity_constraint(self, fix_v, unfix_v):
        self.contact_force = unfix_v * self.contact_force + fix_v * ZEROVEC3f

    @ti.func
    def contact_reflection_constraint(self, norm1, norm2):
        pre_accelerate = self.contact_force
        if SquareLen(norm1) > Threshold and pre_accelerate.dot(norm1) > 0:
            pre_accelerate -= pre_accelerate.dot(norm1) * norm1
        if SquareLen(norm2) > Threshold and pre_accelerate.dot(norm2) > 0:
            pre_accelerate -= pre_accelerate.dot(norm2) * norm2
        self.contact_force = pre_accelerate
        
    @ti.func
    def velocity_constraint(self, fix_v, unfix_v, prescribed_velocity):
        self.momentum = unfix_v * self.momentum + fix_v * prescribed_velocity 
        self.force = unfix_v * self.force + fix_v * ZEROVEC2f

    @ti.func
    def rigid_body_velocity_constraint(self, fix_v, unfix_v):
        self.grad_domain = unfix_v * self.grad_domain + fix_v * ZEROVEC2f

    @ti.func
    def reflection_constraint(self, norm1, norm2):
        pre_velocity, pre_accelerate = self.momentum, self.force
        if SquareLen(norm1) > Threshold and pre_velocity.dot(norm1) > 0:
            pre_velocity -= pre_velocity.dot(norm1) * norm1
            pre_accelerate -= pre_accelerate.dot(norm1) * norm1
        if SquareLen(norm2) > Threshold and pre_velocity.dot(norm2) > 0:
            pre_velocity -= pre_velocity.dot(norm2) * norm2
            pre_accelerate -= pre_accelerate.dot(norm2) * norm2
        
        self.momentum = pre_velocity
        self.force = pre_accelerate

    @ti.func
    def rigid_body_reflection_constraint(self, norm1, norm2, norm3):
        pre_gradient = self.grad_domain
        if SquareLen(norm1) > Threshold and pre_gradient.dot(norm1) > 0:
            pre_gradient -= pre_gradient.dot(norm1) * norm1
        if SquareLen(norm2) > Threshold and pre_gradient.dot(norm2) > 0:
            pre_gradient -= pre_gradient.dot(norm2) * norm2
        self.grad_domain = pre_gradient

    @ti.func
    def friction_constraint(self, mu, norm, dt):
        velocity = self.momentum
        acceleration = self.force / self.m

        vel_n = velocity.dot(norm)
        if vel_n > Threshold:
            v0 = velocity - acceleration * dt[None]
            vt0 = v0 - v0.dot(norm) * norm
            vn1 = vel_n * norm
            vt1 = velocity - vn1
            t = vt1.normalized()
            ft = -sign(vt0.dot(t)) * mu * self.force.dot(norm) * t
            vt2 = vt1 + ft * dt[None]

            if vt1.dot(vt2) > Threshold:
                self.force = self.force.dot(t) * t + ft
                self.momentum = vt2
            else:
                self.force = ZEROVEC2f
                self.momentum = ZEROVEC2f

    @ti.func
    def rigid_friction_constraint(self, norm):
        pre_gradient = self.grad_domain
        if SquareLen(norm) > Threshold and pre_gradient.dot(norm) > 0:
            pre_gradient -= pre_gradient.dot(norm) * norm
        self.grad_domain = pre_gradient


@ti.dataclass
class ExtraNode:
    jacobian: float
    vol: float
    pressure: float

    @ti.func
    def _grid_reset(self):
        self.vol = 0.
        self.jacobian = 0.
        self.pressure = 0.

    @ti.func
    def _update_nodal_pressure(self, pressure):
        self.pressure += pressure

    @ti.func
    def _update_nodal_jacobian(self, jacobian):
        self.jacobian += jacobian

    @ti.func
    def _update_nodal_volume(self, volume):
        self.vol += volume


@ti.dataclass
class Nodes:
    m: float
    force: vec3f
    momentum: vec3f

    @ti.func
    def _tlgrid_reset(self):
        self.momentum = ZEROVEC3f
        self.force = ZEROVEC3f
    
    @ti.func
    def _grid_reset(self):
        self.m = 0.
        self.momentum = ZEROVEC3f
        self.force = ZEROVEC3f

    @ti.func
    def _set_dofs(self, rowth):
        pass

    @ti.func
    def _update_nodal_mass(self, m):
        self.m += m

    @ti.func
    def _update_nodal_momentum(self, momentum):
        self.momentum += momentum

    @ti.func
    def _compute_nodal_velocity(self):
        self.momentum /= self.m

    @ti.func
    def _update_nodal_force(self, force):
        self.force += force

    @ti.func
    def _update_external_force(self, external_force):
        self.force += external_force

    @ti.func
    def _update_internal_force(self, internal_force):
        self.force += internal_force

    @ti.func
    def _compute_nodal_kinematic(self, damp, dt):
        unbalanced_force = self.force / self.m
        force = unbalanced_force - damp * unbalanced_force.norm() * vsign(self.momentum)
        self.momentum += force * dt[None]
        self.force = force 

    @ti.func
    def _update_nodal_kinematic(self):
        self.force /= self.m
        self.momentum /= self.m

    @ti.func
    def velocity_constraint(self, fix_v, unfix_v, prescribed_velocity):
        self.momentum = unfix_v * self.momentum + fix_v * prescribed_velocity 
        self.force = unfix_v * self.force + fix_v * ZEROVEC3f

    @ti.func
    def rigid_body_velocity_constraint(self, fix_v, unfix_v):
        pass

    @ti.func
    def reflection_constraint(self, norm1, norm2, norm3):
        pre_velocity, pre_accelerate = self.momentum, self.force
        if SquareLen(norm1) > Threshold and pre_velocity.dot(norm1) > 0:
            pre_velocity -= pre_velocity.dot(norm1) * norm1
            pre_accelerate -= pre_accelerate.dot(norm1) * norm1
        if SquareLen(norm2) > Threshold and pre_velocity.dot(norm2) > 0:
            pre_velocity -= pre_velocity.dot(norm2) * norm2
            pre_accelerate -= pre_accelerate.dot(norm2) * norm2
        if SquareLen(norm3) > Threshold and pre_velocity.dot(norm3) > 0:
            pre_velocity -= pre_velocity.dot(norm3) * norm3
            pre_accelerate -= pre_accelerate.dot(norm3) * norm3
        
        self.momentum = pre_velocity
        self.force = pre_accelerate

    @ti.func
    def rigid_body_reflection_constraint(self, norm1, norm2, norm3):
        pass

    @ti.func
    def friction_constraint(self, mu, norm, dt):
        velocity = self.momentum
        acceleration = self.force / self.m

        '''acc_n = acceleration.dot(norm)
        acc_t = acceleration - acc_n * norm
        vel_n = velocity.dot(norm)
        vel_t = velocity - vel_n * norm
        if acc_n > 0.:
            if vel_t.norm() != 0.:
                vel_net = vel_t + acc_t * dt[None]
                vel_net_t = vel_net.norm()
                vel_friction = mu * ti.abs(acc_n) * dt[None]

                if vel_net_t <= vel_friction:
                    acc_t -= vel_t / dt[None]
                else:
                    acc_t -= mu * ti.abs(acc_n) * (vel_net / vel_net_t)
            else:
                if acc_t.norm() <= mu * ti.abs(acc_n):
                    acc_t = ZEROVEC3f
                else:
                    acc_tt = acc_t.norm() - mu * ti.abs(acc_n)
                    acc_t -= mu * ti.abs(acc_n) * acc_t / acc_tt
            self.force = acc_t * self.m'''
        
        vel_n = velocity.dot(norm)
        if vel_n > Threshold:
            v0 = velocity - acceleration * dt[None]
            vt0 = v0 - v0.dot(norm) * norm
            vn1 = vel_n * norm
            vt1 = velocity - vn1
            t = vt1.normalized()
            ft = -sign(vt0.dot(t)) * mu * self.force.dot(norm) * t
            vt2 = vt1 + ft * dt[None]

            if vt1.dot(vt2) > Threshold:
                self.force = self.force.dot(t) * t + ft
                self.momentum = vt2
            else:
                self.force = ZEROVEC3f
                self.momentum = ZEROVEC3f

    @ti.func
    def rigid_friction_constraint(self, norm):
        pass


@ti.dataclass
class Nodes2D:
    m: float
    force: vec2f
    momentum: vec2f

    @ti.func
    def _tlgrid_reset(self):
        self.momentum = ZEROVEC2f
        self.force = ZEROVEC2f
    
    @ti.func
    def _grid_reset(self):
        self.m = 0.
        self.momentum = ZEROVEC2f
        self.force = ZEROVEC2f

    @ti.func
    def _set_dofs(self, rowth):
        pass

    @ti.func
    def _update_nodal_mass(self, m):
        self.m += m

    @ti.func
    def _update_nodal_momentum(self, momentum):
        self.momentum += momentum

    @ti.func
    def _compute_nodal_velocity(self):
        self.momentum /= self.m

    @ti.func
    def _update_nodal_force(self, force):
        self.force += force

    @ti.func
    def _update_external_force(self, external_force):
        self.force += external_force

    @ti.func
    def _update_internal_force(self, internal_force):
        self.force += internal_force

    @ti.func
    def _compute_nodal_kinematic(self, damp, dt):
        unbalanced_force = self.force / self.m
        force = unbalanced_force - damp * unbalanced_force.norm() * vsign(self.momentum)
        self.momentum += force * dt[None]
        self.force = force 

    @ti.func
    def _update_nodal_kinematic(self):
        self.force /= self.m
        self.momentum /= self.m

    @ti.func
    def velocity_constraint(self, fix_v, unfix_v, prescribed_velocity):
        self.momentum = unfix_v * self.momentum + fix_v * prescribed_velocity 
        self.force = unfix_v * self.force + fix_v * ZEROVEC2f

    @ti.func
    def rigid_body_velocity_constraint(self, fix_v, unfix_v):
        pass

    @ti.func
    def reflection_constraint(self, norm1, norm2):
        pre_velocity, pre_accelerate = self.momentum, self.force
        if SquareLen(norm1) > Threshold and pre_velocity.dot(norm1) > 0:
            pre_velocity -= pre_velocity.dot(norm1) * norm1
            pre_accelerate -= pre_accelerate.dot(norm1) * norm1
        if SquareLen(norm2) > Threshold and pre_velocity.dot(norm2) > 0:
            pre_velocity -= pre_velocity.dot(norm2) * norm2
            pre_accelerate -= pre_accelerate.dot(norm2) * norm2
        
        self.momentum = pre_velocity
        self.force = pre_accelerate

    @ti.func
    def rigid_body_reflection_constraint(self, norm1, norm2, norm3):
        pass

    @ti.func
    def friction_constraint(self, mu, norm, dt):
        velocity = self.momentum
        acceleration = self.force / self.m

        vel_n = velocity.dot(norm)
        if vel_n > Threshold:
            v0 = velocity - acceleration * dt[None]
            vt0 = v0 - v0.dot(norm) * norm
            vn1 = vel_n * norm
            vt1 = velocity - vn1
            t = vt1.normalized()
            ft = -sign(vt0.dot(t)) * mu * self.force.dot(norm) * t
            vt2 = vt1 + ft * dt[None]

            if vt1.dot(vt2) > Threshold:
                self.force = self.force.dot(t) * t + ft
                self.momentum = vt2
            else:
                self.force = ZEROVEC2f
                self.momentum = ZEROVEC2f

    @ti.func
    def rigid_friction_constraint(self, norm):
        pass


@ti.dataclass
class ImplicitParticleCoupling:
    bodyID: ti.u8
    materialID: ti.u8
    active: ti.u8
    free_surface: ti.u8
    m: float
    vol: float
    rad: float
    mass_density: float
    psize: vec3f
    normal: vec3f
    x: vec3f
    a: vec3f
    verletDisp: vec3f
    v: vec3f
    contact_force: vec3f
    traction: vec3f
    strain: vec6f
    stress: vec6f
    velocity_gradient: mat3x3
    fix_v: vec3u8
    unfix_v: vec3u8

    @ti.func
    def _restart(self, bodyID, materialID, active, free_surface, normal, mass, position, velocity, volume, 
                 traction, strain, stress, psize, velocity_gradient, fix_v):
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.active = ti.u8(active)
        self.free_surface = ti.u8(free_surface)
        self.normal = float(normal)
        self.m = float(mass)
        self.x = float(position)
        self.v = float(velocity)
        self.vol = float(volume)
        self.traction = float(traction) * float(volume) * vec3f(1./psize[2], 1./psize[1], 1./psize[0])
        self.strain = float(strain)
        self.stress = float(stress)
        self.psize = float(psize)
        self.velocity_gradient = float(velocity_gradient)
        self.fix_v = ti.cast(fix_v, ti.u8)
        self.unfix_v = ti.cast(Zero2OneVector(fix_v), ti.u8)

    @ti.func
    def _set_essential(self, bodyID, materialID, density, particle_volume, psize, position, init_v, fix_v):
        self.active = ti.u8(1)
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.vol = float(particle_volume)
        self.psize = float(psize)
        self.rad = 0.5 * float((particle_volume) ** (1./3.))
        self.m = float(particle_volume * density)
        self.x = float(position)
        self.v = init_v
        self.fix_v = fix_v
        self.unfix_v = Zero2OneVector(fix_v)

    @ti.func
    def _add_gravity_field(self, k0, gamma):
        self.stress[2] += float(gamma)
        self.stress[0] += float(k0 * gamma)
        self.stress[1] += float(k0 * gamma)

    @ti.func
    def _set_particle_traction(self, traction):
        self.traction += 0.5 * float(traction) * self.vol * vec3f(1./self.psize[2], 1./self.psize[1], 1./self.psize[0])

    @ti.func
    def _reset_contact_force(self):
        self.contact_force = ZEROVEC3f

    @ti.func
    def _reset_mass_density(self):
        self.mass_density = 0.

    @ti.func
    def _compute_external_force(self, gravity):
        return self.m * gravity + self.contact_force + self.traction 
    
    @ti.func
    def _compute_internal_force(self):
        return -self.vol * self.stress 

    @ti.func
    def _update_particle_state(self, dt, alpha, vPIC, acc, disp):
        v0 = self.v
        flag1 = int(self.fix_v)
        flag2 = int(self.unfix_v)
        vFLIP = 0.5 * (acc + self.a) * dt[None]
        self.a = acc
        self.v = (alpha * vPIC + (1 - alpha) * (vFLIP + v0)) * flag2 + v0 * flag1
        self.x += disp * flag2 + v0 * dt[None] * flag1

    @ti.func
    def _update_rigid_body(self, dt):
        deltax = self.v * dt[None]
        self.x += deltax
        self.verletDisp += deltax
    
    @ti.func
    def _compute_particle_velocity(self, xg):
        return self.v - self.velocity_gradient @ (self.x - xg)
    
    @ti.func
    def _update_contact_interaction(self, cforce):
        self.contact_force += cforce
    
    @ti.func
    def _calc_psize_cp(self, deformation_gradient_rate):
        self.psize[0] *= deformation_gradient_rate[0, 0] 
        self.psize[1] *= deformation_gradient_rate[1, 1]
        self.psize[2] *= deformation_gradient_rate[2, 2] 

    @ti.func
    def _calc_psize_r(self, deformation_gradient_rate):
        self.psize[0] *= ti.sqrt(deformation_gradient_rate[0, 0] ** 2 + deformation_gradient_rate[1, 0] ** 2 + deformation_gradient_rate[0, 2] ** 2)
        self.psize[1] *= ti.sqrt(deformation_gradient_rate[0, 1] ** 2 + deformation_gradient_rate[1, 1] ** 2 + deformation_gradient_rate[1, 2] ** 2)
        self.psize[2] *= ti.sqrt(deformation_gradient_rate[0, 2] ** 2 + deformation_gradient_rate[1, 2] ** 2 + deformation_gradient_rate[2, 2] ** 2) 


@ti.dataclass
class ImplicitParticle:
    bodyID: ti.u8
    materialID: ti.u8
    active: ti.u8
    m: float
    vol: float
    vol0 : float
    psize: vec3f
    x: vec3f
    v: vec3f
    a: vec3f
    traction: vec3f
    strain: vec6f
    stress: vec6f
    stress0: vec6f
    velocity_gradient: mat3x3
    fix_v: vec3u8
    unfix_v: vec3u8

    @ti.func
    def _restart(self, bodyID, materialID, active, mass, position, velocity, volume, traction, strain, stress, psize, velocity_gradient, fix_v):
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.active = ti.u8(active)
        self.m = float(mass)
        self.x = float(position)
        self.v = float(velocity)
        self.vol = float(volume)
        self.vol0 = float(volume)
        self.traction = float(traction) * float(volume) * vec3f(1./psize[2], 1./psize[1], 1./psize[0])
        self.strain = float(strain)
        self.stress = float(stress)
        self.stress0 = float(stress)
        self.psize = float(psize)
        self.velocity_gradient = float(velocity_gradient)
        self.fix_v = ti.cast(fix_v, ti.u8)
        self.unfix_v = ti.cast(Zero2OneVector(fix_v), ti.u8)

    @ti.func
    def _set_essential(self, bodyID, materialID, density, particle_volume, psize, position, init_v, fix_v):
        self.active = ti.u8(1)
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.vol = float(particle_volume)
        self.vol0 = float(particle_volume)
        self.psize = float(psize)
        self.m = float(particle_volume * density)
        self.x = float(position)
        self.v = init_v
        self.fix_v = fix_v
        self.unfix_v = Zero2OneVector(fix_v)

    @ti.func
    def _add_gravity_field(self, k0, gamma):
        self.stress[2] += float(gamma)
        self.stress[0] += float(k0 * gamma)
        self.stress[1] += float(k0 * gamma)
        self.stress0 = self.stress

    @ti.func
    def _set_particle_traction(self, traction):
        self.traction += 0.5 * float(traction) * self.vol * vec3f(1./self.psize[2], 1./self.psize[1], 1./self.psize[0])

    @ti.func
    def _compute_external_force(self, gravity):
        return self.m * gravity + self.traction 
    
    @ti.func
    def _compute_internal_force(self):
        return -self.vol * self.stress 
    
    @ti.func
    def _update_particle_state(self, dt, alpha, vPIC, acc, disp):
        v0 = self.v
        flag1 = int(self.fix_v)
        flag2 = int(self.unfix_v)
        vFLIP = 0.5 * (acc + self.a) * dt[None]
        self.a = acc
        self.v = (alpha * vPIC + (1 - alpha) * (vFLIP + v0)) * flag2 + v0 * flag1
        self.x += disp * flag2 + v0 * dt[None] * flag1

    @ti.func
    def _update_rigid_body(self, dt):
        self.x += self.v * dt[None]
    
    @ti.func
    def _compute_particle_velocity(self, xg):
        return self.v - self.velocity_gradient @ (self.x - xg)
    
    @ti.func
    def _calc_psize_cp(self, deformation_gradient_rate):
        self.psize[0] *= deformation_gradient_rate[0, 0] 
        self.psize[1] *= deformation_gradient_rate[1, 1]
        self.psize[2] *= deformation_gradient_rate[2, 2] 

    @ti.func
    def _calc_psize_r(self, deformation_gradient_rate):
        self.psize[0] *= ti.sqrt(deformation_gradient_rate[0, 0] ** 2 + deformation_gradient_rate[1, 0] ** 2 + deformation_gradient_rate[0, 2] ** 2)
        self.psize[1] *= ti.sqrt(deformation_gradient_rate[0, 1] ** 2 + deformation_gradient_rate[1, 1] ** 2 + deformation_gradient_rate[1, 2] ** 2)
        self.psize[2] *= ti.sqrt(deformation_gradient_rate[0, 2] ** 2 + deformation_gradient_rate[1, 2] ** 2 + deformation_gradient_rate[2, 2] ** 2)


@ti.dataclass
class ImplicitNodes:
    dof: int
    m: float
    inertia: vec3f
    ext_force: vec3f
    int_force: vec3f
    momentum: vec3f
    displacement: vec3f

    @ti.func
    def _grid_reset(self):
        self.m = 0.
        self.momentum = ZEROVEC3f
        self.ext_force = ZEROVEC3f
        self.inertia = ZEROVEC3f
        self.displacement = ZEROVEC3f
        self.dof = -1

    @ti.func
    def _reset_internal_force(self):
        self.int_force = ZEROVEC3f

    @ti.func
    def _update_nodal_mass(self, m):
        self.m += m

    @ti.func
    def _update_nodal_momentum(self, momentum):
        self.momentum += momentum

    @ti.func
    def _update_nodal_acceleration(self, inertia):
        self.inertia += inertia

    @ti.func
    def _compute_nodal_velocity(self):
        self.momentum /= self.m

    @ti.func
    def _compute_nodal_acceleration(self):
        self.inertia /= self.m

    @ti.func
    def _update_external_force(self, external_force):
        self.ext_force += external_force

    @ti.func
    def _update_internal_force(self, internal_force):
        self.int_force += internal_force

    @ti.func
    def _compute_nodal_kinematic(self, damp, dt):
        pass

    @ti.func
    def _update_nodal_kinematic_newmark(self, beta, gamma, dt):
        previous_velocity = self.momentum
        previous_acceleration = self.inertia
        self.momentum = gamma / beta / dt[None] * self.displacement - (gamma / beta - 1.) * previous_velocity - 0.5 * dt[None] * (gamma / beta - 2.) * previous_acceleration
        self.inertia = 1. / beta / dt[None] / dt[None] * self.displacement - 1. / beta / dt[None] * previous_velocity - (0.5 / beta - 1.) * previous_acceleration

    @ti.func
    def _update_nodal_disp(self, disp):
        self.displacement += disp

    @ti.func
    def _set_dofs(self, rowth):
        self.dof = int(3 * rowth)


@ti.dataclass
class HashTable:
    current: int
    count: int


@ti.dataclass
class HexahedronCell:
    active: ti.u8
    volume: float

    @ti.func
    def _reset(self):
        self.volume = 0.

    @ti.func
    def _update_cell_volume(self, volume):
        self.volume += volume


@ti.dataclass
class HexahedronGuassCell:
    stress: vec6f
    vol: float

    @ti.func
    def _reset(self):
        if self.vol > 0.:
            self.stress = ZEROVEC6f
            self.vol = 0.



