import taichi as ti

from src.utils.constants import ZEROVEC2f, ZEROVEC3f
from src.utils.ScalarFunction import sign, sgn
from src.utils.TypeDefination import vec2f, vec3f, mat3x2


@ti.dataclass
class ContactNodes:
    m: float
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

    @ti.func
    def _grid_reset(self):
        self.m = 0.
        self.momentum = ZEROVEC3f
        self.force = ZEROVEC3f
        self.contact_force = ZEROVEC3f
        self.grad_domain = ZEROVEC3f

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
        unbalanced_force = self.force 
        velocity = self.momentum
        for d in ti.static(range(3)):
            if velocity[d] * unbalanced_force[d] > 0.: 
                unbalanced_force[d] -= damp * ti.abs(unbalanced_force[d]) * sgn(velocity[d])
        acceleration = unbalanced_force / self.m
        self.momentum += acceleration * dt[None]
        self.force = acceleration 

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
    def contact_velocity_constraint(self, dirs):
        self.contact_force[dirs] = 0.

    @ti.func
    def contact_reflection_constraint(self, dirs, signs):
        pre_velocity = self.momentum[dirs] 
        if pre_velocity * signs > 0:
            self.contact_force[abs(dirs)] = 0.
        
    @ti.func
    def velocity_constraint(self, dirs, prescribed_velocity):
        self.momentum[dirs] = prescribed_velocity 
        self.force[dirs] = 0.

    @ti.func
    def rigid_body_velocity_constraint(self, dirs):
        self.grad_domain[dirs] = 0.

    @ti.func
    def reflection_constraint(self, dirs, signs):
        pre_velocity = self.momentum[dirs]
        if pre_velocity * signs > 0:
            self.momentum[dirs] = 0.
            self.force[dirs] = 0.

    @ti.func
    def rigid_body_reflection_constraint(self, dirs, signs):
        pre_gradient = self.grad_domain[dirs]
        if pre_gradient * signs > 0:
            self.grad_domain[dirs] = 0.

    @ti.func
    def friction_constraint(self, mu, dirs_n, signs, dt):
        velocity = self.momentum
        acceleration = self.force - velocity * dt[None]
        dir = mat3x2([1, 2], [0, 2], [0, 1])
        dir_t0 = dir[dirs_n, 0]
        dir_t1 = dir[dirs_n, 1]

        acc_n = acceleration[dirs_n]
        acc_t = ti.sqrt(acceleration[dir_t0] * acceleration[dir_t0] + acceleration[dir_t1] * acceleration[dir_t1])
        vel_t = ti.sqrt(velocity[dir_t0] * velocity[dir_t0] + velocity[dir_t1] * velocity[dir_t1])
        if acc_n * signs > 0.0:                                                                         # dynamic friction
            if vel_t != 0.0:
                vel_net = vec2f(0., 0.)
                vel_net[0] = velocity[dir_t0] + acceleration[dir_t0] * dt[None]                        # friction is applied opposite to the vel_net
                vel_net[1] = velocity[dir_t1] + acceleration[dir_t1] * dt[None]
                vel_net_t = ti.sqrt(vel_net[0] * vel_net[0] + vel_net[1] * vel_net[1])
                vel_fricion = mu * abs(acc_n) * dt[None]

                if vel_net_t <= vel_fricion:
                    acceleration[dir_t0] = -velocity[dir_t0] / dt[None]
                    acceleration[dir_t1] = -velocity[dir_t1] / dt[None]
                else:
                    acceleration[dir_t0] -= mu * abs[acc_n] * (vel_net[0] / vel_net_t)
                    acceleration[dir_t1] -= mu * abs[acc_n] * (vel_net[1] / vel_net_t)
            else:                                                                                     # static friction
                if acc_t <= mu * abs(acc_n):                                                          # since acc_t is positive
                    acceleration[dir_t0] = 0
                    acceleration[dir_t1] = 0
                else:
                    acc_t -= mu * abs(acc_n)
                    acceleration[dir_t0] -= mu * abs(acc_n) * (acceleration[dir_t0] / acc_t)
                    acceleration[dir_t1] -= mu * abs(acc_n) * (acceleration[dir_t1] / acc_t)
            self.momentum += acceleration * dt[None]
            self.force = acceleration
            
    @ti.func
    def rigid_friction_constraint(self, dirs, signs):
        pre_gradient = self.grad_domain[dirs]
        if pre_gradient * signs > 0:
            self.grad_domain[dirs] = 0.


@ti.dataclass
class ContactNodes2D:
    m: float
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

    @ti.func
    def _grid_reset(self):
        self.m = 0.
        self.momentum = ZEROVEC2f
        self.force = ZEROVEC2f
        self.contact_force = ZEROVEC2f
        self.grad_domain = ZEROVEC2f

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
        unbalanced_force = self.force 
        velocity = self.momentum
        for d in ti.static(range(2)):
            if velocity[d] * unbalanced_force[d] > 0.: 
                unbalanced_force[d] -= damp * ti.abs(unbalanced_force[d]) * sgn(velocity[d])
        acceleration = unbalanced_force / self.m
        self.momentum += acceleration * dt[None]
        self.force = acceleration 

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
    def contact_velocity_constraint(self, dirs):
        self.contact_force[dirs] = 0.

    @ti.func
    def contact_reflection_constraint(self, dirs, signs):
        pre_velocity = self.momentum[dirs] 
        if pre_velocity * signs > 0:
            self.contact_force[abs(dirs)] = 0.
        
    @ti.func
    def velocity_constraint(self, dirs, prescribed_velocity):
        self.momentum[dirs] = prescribed_velocity 
        self.force[dirs] = 0.

    @ti.func
    def rigid_body_velocity_constraint(self, dirs):
        self.grad_domain[dirs] = 0.

    @ti.func
    def reflection_constraint(self, dirs, signs):
        pre_velocity = self.momentum[dirs]
        if pre_velocity * signs > 0:
            self.momentum[dirs] = 0.
            self.force[dirs] = 0.

    @ti.func
    def rigid_body_reflection_constraint(self, dirs, signs):
        pre_gradient = self.grad_domain[dirs]
        if pre_gradient * signs > 0:
            self.grad_domain[dirs] = 0.

    @ti.func
    def friction_constraint(self, mu, dirs_n, signs, dt):
        velocity = self.momentum
        acceleration = self.force - velocity * dt[None]
        dir_t = 1 - dirs_n

        acc_n = acceleration[dirs_n]
        acc_t = acceleration[dir_t]
        vel_t = velocity[dir_t]
        if acc_n * signs > 0.0:                                                                         # dynamic friction
            if vel_t != 0.0:
                vel_net = velocity[dir_t] + acc_t * dt[None]                                           # friction is applied opposite to the vel_net
                vel_frictional = dt[None] * mu * abs(acc_n)
                if abs(vel_net) <= vel_frictional:
                    acc_t = -vel_t / dt[None]
                else:
                    acc_t -= sign(vel_net) * mu * abs(acc_n)
            else:                                                                                     # static friction
                if abs(acc_t) <= mu * abs(acc_n):                                                     # since acc_t is positive
                    acc_t = 0.
                else:
                    acc_t -= sign(acc_t) * mu * abs(acc_n)
            self.momentum += vec2f(acc_n, acc_t) * dt[None]
            self.force[dir_t] = acc_t
            
    @ti.func
    def rigid_friction_constraint(self, dirs, signs):
        pre_gradient = self.grad_domain[dirs]
        if pre_gradient * signs > 0:
            self.grad_domain[dirs] = 0.


@ti.dataclass
class NodeTwoPhase2D:
    m: float
    ms: float
    mf: float
    force: vec2f
    forces: vec2f
    forcef: vec2f
    momentum: vec2f
    momentums: vec2f
    momentumf: vec2f

    @ti.func
    def _tlgrid_reset(self):
        self.momentum = ZEROVEC2f
        self.momentums = ZEROVEC2f
        self.momentumf = ZEROVEC2f
        self.force = ZEROVEC2f
        self.forces = ZEROVEC2f
        self.forcef = ZEROVEC2f
    
    @ti.func
    def _grid_reset(self):
        self.m = 0.
        self.ms = 0.
        self.mf = 0.
        self.momentum = ZEROVEC2f
        self.momentums = ZEROVEC2f
        self.momentumf = ZEROVEC2f
        self.force = ZEROVEC2f
        self.forces = ZEROVEC2f
        self.forcef = ZEROVEC2f

    @ti.func
    def _update_nodal_mass(self, m, ms, mf):
        self.m += m
        self.ms += ms
        self.mf += mf

    @ti.func
    def _update_nodal_momentum(self, momentum, momentums, momentumf):
        self.momentum += momentum
        self.momentums += momentums
        self.momentumf += momentumf

    @ti.func   # notef
    def _compute_nodal_velocity(self, cutoff):
        if self.m > cutoff:
            self.momentum /= self.m
        if self.ms > cutoff:
            self.momentums /= self.ms
        if self.mf > cutoff:  
            self.momentumf /= self.mf

    @ti.func
    def _update_nodal_force(self, force, forcef):
        self.force += force
        self.forcef += forcef

    @ti.func
    def _update_external_force(self, external_force, external_forcef):
        self.force += external_force
        self.forcef += external_forcef

    @ti.func
    def _update_internal_force(self, internal_force, internal_forcef):
        self.force += internal_force
        self.forcef += internal_forcef

    @ti.func  # note
    def _compute_nodal_kinematic(self, damp, dt):
        unbalanced_force = self.force / self.m
        velocity = self.momentum
        for d in ti.static(range(2)):
            if velocity[d] * unbalanced_force[d] > 0.: 
                unbalanced_force[d] -= damp * ti.abs(unbalanced_force[d]) * sgn(velocity[d])
        self.momentum += unbalanced_force * dt[None]
        self.force = unbalanced_force

    @ti.func
    def _compute_nodal_kinematic_fluid(self, damp, dt):
        unbalanced_force = self.forcef / self.mf
        velocity = self.momentumf
        for d in ti.static(range(2)):
            if velocity[d] * unbalanced_force[d] > 0.: 
                unbalanced_force[d] -= damp * ti.abs(unbalanced_force[d]) * sgn(velocity[d])
        self.momentumf += unbalanced_force * dt[None]   # velocity
        self.forcef = unbalanced_force   # acceleration

    @ti.func
    def _compute_nodal_kinematic_solid(self, damp, dt):
        unbalanced_force = (self.force - self.forcef * self.mf) / self.ms
        forces = unbalanced_force # - damp * unbalanced_force.norm() * vsign(self.momentum)
        self.momentums += forces * dt[None]
        self.forces = forces

    @ti.func
    def _update_nodal_kinematic(self):
        self.force /= self.m
        self.forces /= self.ms
        self.forcef /= self.mf
        self.momentum /= self.m
        self.momentums /= self.ms
        self.momentumf /= self.mf

    @ti.func
    def velocity_constraint(self, dirs, prescribed_velocity):
        self.momentum[dirs] = prescribed_velocity 
        self.momentums[dirs] = prescribed_velocity 
        self.momentumf[dirs] = prescribed_velocity 
        self.force[dirs] = 0.
        self.forces[dirs] = 0.
        self.forcef[dirs] = 0.

    @ti.func
    def rigid_body_velocity_constraint(self, dirs):
        pass

    @ti.func
    def reflection_constraint(self, dirs, signs):
        pre_velocity = self.momentum[dirs]
        if pre_velocity * signs > 0:
            self.momentum[dirs] = 0.
            self.force[dirs] = 0.

    @ti.func
    def rigid_body_reflection_constraint(self, dirs, signs):
        pass

    # @ti.func
    # def friction_constraint(self, mu, norm, dt):
    #     velocity = self.momentum
    #     acceleration = self.force / self.m

    #     vel_n = velocity.dot(norm)
    #     if vel_n > Threshold:
    #         v0 = velocity - acceleration * dt[None]
    #         vt0 = v0 - v0.dot(norm) * norm
    #         vn1 = vel_n * norm
    #         vt1 = velocity - vn1
    #         t = vt1.normalized()
    #         ft = -sign(vt0.dot(t)) * mu * self.force.dot(norm) * t
    #         vt2 = vt1 + ft * dt[None]

    #         if vt1.dot(vt2) > Threshold:
    #             self.force = self.force.dot(t) * t + ft
    #             self.momentum = vt2
    #         else:
    #             self.force = ZEROVEC2f
    #             self.momentum = ZEROVEC2f

    @ti.func
    def rigid_friction_constraint(self, dirs, signs):
        pass


@ti.dataclass
class ExtraNode:
    vol: float

    @ti.func
    def _grid_reset(self):
        self.vol = 0.

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
        unbalanced_force = self.force 
        velocity = self.momentum
        for d in ti.static(range(3)):
            if velocity[d] * unbalanced_force[d] > 0.: 
                unbalanced_force[d] -= damp * ti.abs(unbalanced_force[d]) * sgn(velocity[d])
        acceleration = unbalanced_force / self.m
        self.momentum += acceleration * dt[None]
        self.force = acceleration 

    @ti.func
    def _update_nodal_kinematic(self):
        self.force /= self.m
        self.momentum /= self.m

    @ti.func
    def contact_velocity_constraint(self, dirs):
        pass

    @ti.func
    def contact_reflection_constraint(self, dirs, signs):
        pass
        
    @ti.func
    def velocity_constraint(self, dirs, prescribed_velocity):
        self.momentum[dirs] = prescribed_velocity 
        self.force[dirs] = 0.

    @ti.func
    def rigid_body_velocity_constraint(self, dirs):
        pass

    @ti.func
    def reflection_constraint(self, dirs, signs):
        pre_velocity = self.momentum[dirs]
        if pre_velocity * signs > 0:
            self.momentum[dirs] = 0.
            self.force[dirs] = 0.

    @ti.func
    def rigid_body_reflection_constraint(self, dirs, signs):
        pass

    @ti.func
    def friction_constraint(self, mu, dirs_n, signs, dt):
        velocity = self.momentum
        acceleration = self.force - velocity * dt[None]
        dir = mat3x2([1, 2], [0, 2], [0, 1])
        dir_t0 = dir[dirs_n, 0]
        dir_t1 = dir[dirs_n, 1]

        acc_n = acceleration[dirs_n]
        acc_t = ti.sqrt(acceleration[dir_t0] * acceleration[dir_t0] + acceleration[dir_t1] * acceleration[dir_t1])
        vel_t = ti.sqrt(velocity[dir_t0] * velocity[dir_t0] + velocity[dir_t1] * velocity[dir_t1])
        if acc_n * signs > 0.0:                                                                         # dynamic friction
            if vel_t != 0.0:
                vel_net = vec2f(0., 0.)
                vel_net[0] = velocity[dir_t0] + acceleration[dir_t0] * dt[None]                        # friction is applied opposite to the vel_net
                vel_net[1] = velocity[dir_t1] + acceleration[dir_t1] * dt[None]
                vel_net_t = ti.sqrt(vel_net[0] * vel_net[0] + vel_net[1] * vel_net[1])
                vel_fricion = mu * abs(acc_n) * dt[None]

                if vel_net_t <= vel_fricion:
                    acceleration[dir_t0] = -velocity[dir_t0] / dt[None]
                    acceleration[dir_t1] = -velocity[dir_t1] / dt[None]
                else:
                    acceleration[dir_t0] -= mu * abs[acc_n] * (vel_net[0] / vel_net_t)
                    acceleration[dir_t1] -= mu * abs[acc_n] * (vel_net[1] / vel_net_t)
            else:                                                                                     # static friction
                if acc_t <= mu * abs(acc_n):                                                          # since acc_t is positive
                    acceleration[dir_t0] = 0
                    acceleration[dir_t1] = 0
                else:
                    acc_t -= mu * abs(acc_n)
                    acceleration[dir_t0] -= mu * abs(acc_n) * (acceleration[dir_t0] / acc_t)
                    acceleration[dir_t1] -= mu * abs(acc_n) * (acceleration[dir_t1] / acc_t)
            self.momentum += acceleration * dt[None]
            self.force = acceleration
            
    @ti.func
    def rigid_friction_constraint(self, dirs, signs):
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
        unbalanced_force = self.force 
        velocity = self.momentum
        for d in ti.static(range(2)):
            if velocity[d] * unbalanced_force[d] > 0.: 
                unbalanced_force[d] -= damp * ti.abs(unbalanced_force[d]) * sgn(velocity[d])
        acceleration = unbalanced_force / self.m
        self.momentum += acceleration * dt[None]
        self.force = acceleration 

    @ti.func
    def _update_nodal_kinematic(self):
        self.force /= self.m
        self.momentum /= self.m

    @ti.func
    def contact_velocity_constraint(self, dirs):
        pass

    @ti.func
    def contact_reflection_constraint(self, dirs, signs):
        pass
        
    @ti.func
    def velocity_constraint(self, dirs, prescribed_velocity):
        self.momentum[dirs] = prescribed_velocity 
        self.force[dirs] = 0.

    @ti.func
    def rigid_body_velocity_constraint(self, dirs):
        pass

    @ti.func
    def reflection_constraint(self, dirs, signs):
        pre_velocity = self.momentum[dirs]
        if pre_velocity * signs > 0:
            self.momentum[dirs] = 0.
            self.force[dirs] = 0.

    @ti.func
    def rigid_body_reflection_constraint(self, dirs, signs):
        pass

    @ti.func
    def friction_constraint(self, mu, dirs_n, signs, dt):
        velocity = self.momentum
        acceleration = self.force - velocity * dt[None]
        dir_t = 1 - dirs_n

        acc_n = acceleration[dirs_n]
        acc_t = acceleration[dir_t]
        vel_t = velocity[dir_t]
        if acc_n * signs > 0.0:                                                                        # dynamic friction
            if vel_t != 0.0:
                vel_net = velocity[dir_t] + acc_t * dt[None]                                           # friction is applied opposite to the vel_net
                vel_frictional = dt[None] * mu * abs(acc_n)
                if abs(vel_net) <= vel_frictional:
                    acc_t = -vel_t / dt[None]
                else:
                    acc_t -= sign(vel_net) * mu * abs(acc_n)
            else:                                                                                     # static friction
                if abs(acc_t) <= mu * abs(acc_n):                                                     # since acc_t is positive
                    acc_t = 0.
                else:
                    acc_t -= sign(acc_t) * mu * abs(acc_n)
            self.momentum += vec2f(acc_n, acc_t) * dt[None]
            self.force[dir_t] = acc_t
            
    @ti.func
    def rigid_friction_constraint(self, dirs, signs):
        pass

@ti.dataclass
class IncompressibleNodes2D:
    m: float
    force: vec2f
    momentum: vec2f
    vbar: vec2f
    
    @ti.func
    def _grid_reset(self):
        self.m = 0.
        self.momentum = ZEROVEC2f
        self.vbar = ZEROVEC2f
        self.force = ZEROVEC2f

    @ti.func
    def _update_nodal_mass(self, m):
        self.m += m

    @ti.func
    def _update_nodal_momentum(self, momentum):
        self.momentum += momentum

    @ti.func
    def _compute_nodal_velocity(self, gravity, dt):
        self.vbar = self.momentum
        acceleration = self.force / self.m + vec2f(gravity[0], gravity[1])
        self.momentum += acceleration * dt[None]

    @ti.func
    def _compute_nodal_acceleration(self, dt):
        self.force = self.momentum - self.vbar

    @ti.func
    def velocity_constraint(self, dirs, prescribed_velocity):
        self.momentum[dirs] = prescribed_velocity 
        self.force[dirs] = 0.

    @ti.func
    def rigid_body_velocity_constraint(self, dirs):
        pass

    @ti.func
    def reflection_constraint(self, dirs, signs):
        pre_velocity = self.momentum[dirs]
        if pre_velocity * signs > 0:
            self.momentum[dirs] = 0.
            self.force[dirs] = 0.

    @ti.func
    def rigid_body_reflection_constraint(self, dirs, signs):
        pass

@ti.dataclass
class IncompressibleNodes3D:
    m: float
    force: vec3f
    momentum: vec3f
    vbar: vec3f
    
    @ti.func
    def _grid_reset(self):
        self.m = 0.
        self.vbar = ZEROVEC3f
        self.momentum = ZEROVEC3f
        self.force = ZEROVEC3f

    @ti.func
    def _update_nodal_mass(self, m):
        self.m += m

    @ti.func
    def _update_nodal_momentum(self, momentum):
        self.momentum += momentum

    @ti.func
    def _compute_nodal_velocity(self, gravity, dt):
        self.vbar = self.momentum
        acceleration = self.force / self.m + gravity
        self.momentum += acceleration * dt[None]

    @ti.func
    def _compute_nodal_acceleration(self, dt):
        self.force = self.momentum - self.force

    @ti.func
    def velocity_constraint(self, dirs, prescribed_velocity):
        self.momentum[dirs] = prescribed_velocity 
        self.force[dirs] = 0.

    @ti.func
    def rigid_body_velocity_constraint(self, dirs):
        pass

    @ti.func
    def reflection_constraint(self, dirs, signs):
        pre_velocity = self.momentum[dirs]
        if pre_velocity * signs > 0:
            self.momentum[dirs] = 0.
            self.force[dirs] = 0.

    @ti.func
    def rigid_body_reflection_constraint(self, dirs, signs):
        pass

@ti.dataclass
class ImplicitNodes:
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


@ti.dataclass
class ImplicitNodes2D:
    m: float
    inertia: vec2f
    ext_force: vec2f
    int_force: vec2f
    momentum: vec2f
    displacement: vec2f

    @ti.func
    def _grid_reset(self):
        self.m = 0.
        self.momentum = ZEROVEC2f
        self.ext_force = ZEROVEC2f
        self.inertia = ZEROVEC2f
        self.displacement = ZEROVEC2f

    @ti.func
    def _reset_internal_force(self):
        self.int_force = ZEROVEC2f

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