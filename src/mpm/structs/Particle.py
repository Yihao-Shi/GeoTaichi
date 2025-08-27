import taichi as ti

from src.utils.constants import ZEROVEC3f, ZEROVEC6f
from src.utils.TypeDefination import vec2f, vec3f, vec2u8, vec3u8, vec6f, mat2x2, mat3x3
from src.utils.BitFunction import Zero2OneVector
import src.utils.GlobalVariable as GlobalVariable


@ti.dataclass
class ParticleCloud2D:      # memory usage: 108B
    particleID: int
    bodyID: ti.u8
    materialID: ti.u8
    active: ti.u8
    m: float
    vol: float
    x: vec2f
    v: vec2f
    stress: vec6f
    velocity_gradient: mat2x2
    fix_v: vec2u8

    @ti.func
    def _restart(self, bodyID, materialID, active, mass, position, velocity, volume, stress, velocity_gradient, fix_v):
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.active = ti.u8(active)
        self.m = float(mass)
        self.x = float(position)
        self.v = float(velocity)
        self.vol = float(volume)
        self.stress = float(stress)
        self.velocity_gradient = float(velocity_gradient)
        self.fix_v = ti.cast(fix_v, ti.u8)

    @ti.func
    def _set_essential(self, particleID, bodyID, materialID, density, particle_volume, position, init_v, fix_v):
        self.particleID = particleID
        self.active = ti.u8(1)
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.vol = float(particle_volume)
        self.m = float(particle_volume * density)
        self.x = float(position)
        self.v = init_v
        self.fix_v = fix_v

    @ti.func
    def _add_gravity_field(self, gamma):
        if ti.static(self.stress.n == 6):
            self.stress[0] += float(gamma[0, 0])
            self.stress[1] += float(gamma[1, 1])
            self.stress[2] += float(gamma[2, 2])
            self.stress[3] += 0.5 * float(gamma[0, 1] + gamma[1, 0])
            self.stress[4] += 0.5 * float(gamma[1, 2] + gamma[2, 1])
            self.stress[5] += 0.5 * float(gamma[0, 2] + gamma[2, 0])
        elif ti.static(self.stress.n == 3):
            self.stress += float(gamma)

    @ti.func
    def _compute_external_force(self, gravity):
        return self.m * vec2f(gravity[0], gravity[1]) 
    
    @ti.func
    def _compute_internal_force(self):
        return -self.vol * self.stress
    
    @ti.func
    def _update_particle_state(self, dt, alpha, vPIC, vFLIP):
        v0 = self.v
        flag1 = int(self.fix_v)
        flag2 = Zero2OneVector(flag1)
        self.v = (alpha * vPIC + (1 - alpha) * (vFLIP * dt[None] + v0)) * flag2 + v0 * flag1
        self.x += vPIC * dt[None] * flag2 + v0 * dt[None] * flag1
        if ti.static(GlobalVariable.MPMXPBC):
            self.x[0] -= ti.floor(self.x[0] / GlobalVariable.MPMXSIZE) * GlobalVariable.MPMXSIZE
        if ti.static(GlobalVariable.MPMYPBC):
            self.x[1] -= ti.floor(self.x[1] / GlobalVariable.MPMYSIZE) * GlobalVariable.MPMYSIZE

    @ti.func
    def _update_stress(self, stress):
        if ti.static(self.stress.n == 6):
            self.stress += stress
        elif ti.static(self.stress.n == 3):
            self.stress += mat3x3([stress[0], stress[3], stress[5]],
                                  [stress[3], stress[1], stress[4]],
                                  [stress[5], stress[4], stress[2]])
            
    @ti.func
    def _get_mean_stress(self):
        pressure = 0.
        if ti.static(self.stress.n == 6):
            pressure = 1./3. * (self.stress[0] + self.stress[1] + self.stress[2])
        elif ti.static(self.stress.n == 3):
            pressure = 1./3. * (self.stress[0, 0] + self.stress[1, 1] + self.stress[2, 2])
        return pressure

    @ti.func
    def _update_rigid_body(self, dt):
        self.x += self.v * dt[None]
    
    @ti.func
    def _compute_particle_velocity(self, xg):
        return self.v - self.velocity_gradient @ (self.x - xg)

@ti.dataclass
class ParticleCloudIncompressible2D:      # memory usage: 108B
    particleID: int
    bodyID: ti.u8
    materialID: ti.u8
    active: ti.u8
    m: float
    vol: float
    x: vec2f
    v: vec2f
    pressure: float
    velocity_gradient: mat2x2
    fix_v: vec2u8
    xvelocity_gradient: vec2f
    yvelocity_gradient: vec2f

    @ti.func
    def _restart(self, bodyID, materialID, active, mass, position, velocity, volume, stress, velocity_gradient, fix_v):
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.active = ti.u8(active)
        self.m = float(mass)
        self.x = float(position)
        self.v = float(velocity)
        self.vol = float(volume)
        self.pressure = float(stress)
        self.velocity_gradient = float(velocity_gradient)
        self.fix_v = ti.cast(fix_v, ti.u8)

    @ti.func
    def _set_essential(self, particleID, bodyID, materialID, density, particle_volume, position, init_v, fix_v):
        self.particleID = particleID
        self.active = ti.u8(1)
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.vol = float(particle_volume)
        self.m = float(particle_volume * density)
        self.x = float(position)
        self.v = init_v
        self.fix_v = fix_v

    @ti.func
    def _add_gravity_field(self, gamma):
        self.pressure += 1./3. * float(gamma[0, 0] + gamma[1, 1] + gamma[2, 2])

    @ti.func
    def _compute_external_force(self, gravity):
        return self.m * vec2f(gravity[0], gravity[1]) 
    
    @ti.func
    def _compute_internal_force(self):
        pass
    
    @ti.func
    def _update_particle_state(self, dt, alpha, vPIC, vFLIP):
        v0 = self.v
        self.v = alpha * vPIC + (1 - alpha) * (vFLIP + v0)
        self.x += vPIC * dt[None]
        if ti.static(GlobalVariable.MPMXPBC):
            self.x[0] -= ti.floor(self.x[0] / GlobalVariable.MPMXSIZE) * GlobalVariable.MPMXSIZE
        if ti.static(GlobalVariable.MPMYPBC):
            self.x[1] -= ti.floor(self.x[1] / GlobalVariable.MPMYSIZE) * GlobalVariable.MPMYSIZE

    @ti.func
    def _update_stress(self, stress):
        self.pressure += 1./3. * (stress[0] + stress[1] + stress[2])

    @ti.func
    def _get_mean_stress(self):
        return self.pressure

    @ti.func
    def _update_rigid_body(self, dt):
        pass
    
    @ti.func
    def _compute_particle_velocity(self, xg):
        return self.v - self.velocity_gradient @ (self.x - xg)

@ti.dataclass
class ParticleCloudTwoPhase2D:      # memory usage: 108B
    particleID: int
    bodyID: ti.u8
    materialID: ti.u8
    active: ti.u8
    m: float
    ms: float    # soild phase
    mf: float    # fluid phase
    vol: float
    porosity: float
    x: vec2f
    v: vec2f
    vs: vec2f
    vf: vec2f
    stress: vec6f
    pressure: float
    permeability: float
    fluid_velocity_gradient: mat2x2
    solid_velocity_gradient: mat2x2
    fix_v: vec2u8

    @ti.func
    def _restart(self, bodyID, materialID, active, mass, mass_s, mass_f, position, velocity,velocity_s, velocity_f, 
                 volume, porosity, stress, pressure, permeability, fluid_velocity_gradient, solid_velocity_gradient, fix_v):
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.active = ti.u8(active)
        self.m = float(mass)
        self.ms = float(mass_s)
        self.mf = float(mass_f)
        self.x = float(position)
        self.v = float(velocity)
        self.vs = float(velocity_s)
        self.vf = float(velocity_f)
        self.vol = float(volume)
        self.porosity = float(porosity)
        self.stress = float(stress)
        self.pressure = float(pressure)
        self.permeability = float(permeability)
        self.fluid_velocity_gradient = float(fluid_velocity_gradient)
        self.solid_velocity_gradient = float(solid_velocity_gradient)
        self.fix_v = ti.cast(fix_v, ti.u8)

    @ti.func
    def _set_essential(self, particleID, bodyID, materialID, densitys, densityf, porosity, particle_volume, position, init_v, fix_v, permeability):
        self.particleID = particleID
        self.active = ti.u8(1)
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.vol = float(particle_volume)
        self.porosity = float(porosity)
        self.ms = float(particle_volume * densitys * (1.0 - porosity))
        self.mf = float(particle_volume * densityf * porosity)
        self.m = self.ms + self.mf
        self.x = float(position)
        self.v = init_v
        self.vs = init_v
        self.vf = init_v
        self.permeability = float(permeability)
        self.fix_v = fix_v

    @ti.func
    def _add_gravity_field(self, gamma):
        if ti.static(self.stress.n == 6):
            self.stress[0] += float(gamma[0, 0])
            self.stress[1] += float(gamma[1, 1])
            self.stress[2] += float(gamma[2, 2])
            self.stress[3] += 0.5 * float(gamma[0, 1] + gamma[1, 0])
            self.stress[4] += 0.5 * float(gamma[1, 2] + gamma[2, 1])
            self.stress[5] += 0.5 * float(gamma[0, 2] + gamma[2, 0])
        elif ti.static(self.stress.n == 3):
            self.stress += float(gamma)

    @ti.func
    def _update_stress(self, stress):
        if ti.static(self.stress.n == 6):
            self.stress += stress
        elif ti.static(self.stress.n == 3):
            self.stress += mat3x3([stress[0], stress[3], stress[5]],
                                  [stress[3], stress[1], stress[4]],
                                  [stress[5], stress[4], stress[2]])
            
    @ti.func
    def _get_mean_stress(self):
        pressure = 0.
        if ti.static(self.stress.n == 6):
            pressure = 1./3. * (self.stress[0] + self.stress[1] + self.stress[2])
        elif ti.static(self.stress.n == 3):
            pressure = 1./3. * (self.stress[0, 0] + self.stress[1, 1] + self.stress[2, 2])
        return pressure

    @ti.func   # total, fluid?
    def _compute_external_force(self, gravity):
        return self.m * vec2f(gravity[0], gravity[1]), self.m * vec2f(gravity[0], gravity[1])
    
    @ti.func
    def _compute_drag_force(self):
        return -self.porosity * self.porosity * 9.8 * 1000 * self.vol * (self.vf - self.vs) / self.permeability
    
    @ti.func  # total internal force, fluid internal force
    def _compute_internal_force(self):
        fluid_pressure =  ZEROVEC6f
        fluid_pressure[0] = self.pressure
        fluid_pressure[1] = self.pressure
        fluid_pressure[2] = self.pressure
        return -self.vol * (self.stress - fluid_pressure), self.vol * self.porosity * fluid_pressure
    
    @ti.func
    def _update_particle_state(self, dt, alpha, vPIC, vFLIP, vPICs, vFLIPs, vPICf, vFLIPf):
        v0 = self.v
        v0s = self.vs
        v0f = self.vf
        flag1 = int(self.fix_v)
        flag2 = Zero2OneVector(flag1)
        self.v = (alpha * vPIC + (1 - alpha) * (vFLIP + v0)) * flag2 + v0 * flag1
        self.vs = (alpha * vPICs + (1 - alpha) * (vFLIPs + v0s)) * flag2 + v0s * flag1
        self.vf = (alpha * vPICf + (1 - alpha) * (vFLIPf + v0f)) * flag2 + v0f * flag1
        self.x += vPIC * dt[None] * flag2 + v0 * dt[None] * flag1
        if ti.static(GlobalVariable.MPMXPBC):
            self.x[0] -= ti.floor(self.x[0] / GlobalVariable.MPMXSIZE) * GlobalVariable.MPMXSIZE
        if ti.static(GlobalVariable.MPMYPBC):
            self.x[1] -= ti.floor(self.x[1] / GlobalVariable.MPMYSIZE) * GlobalVariable.MPMYSIZE

    @ti.func
    def _update_rigid_body(self, dt):
        self.x += self.v * dt[None]
    
    @ti.func
    def _compute_particle_velocity(self, xg):
        return self.v - self.solid_velocity_gradient @ (self.x - xg)
    

@ti.dataclass
class ParticleCloud2DAxisy:  # memory usage: 108B
    particleID: int
    bodyID: ti.u8
    materialID: ti.u8
    active: ti.u8
    m: float
    vol: float
    x: vec2f
    v: vec2f
    stress: vec6f
    velocity_gradient: mat3x3
    fix_v: vec2u8

    @ti.func
    def _restart(self, bodyID, materialID, active, mass, position, velocity, volume, stress, velocity_gradient, fix_v):
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.active = ti.u8(active)
        self.m = float(mass)
        self.x = float(position)
        self.v = float(velocity)
        self.vol = float(volume)
        self.stress = float(stress)
        self.velocity_gradient = float(velocity_gradient)
        self.fix_v = ti.cast(fix_v, ti.u8)

    @ti.func
    def _set_essential(self, particleID, bodyID, materialID, density, particle_volume, position, init_v, fix_v):
        self.particleID = particleID
        self.active = ti.u8(1)
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.vol = float(position[0] * particle_volume)
        self.m = float(position[0] * particle_volume * density)
        self.x = float(position)
        self.v = init_v
        self.fix_v = fix_v

    @ti.func
    def _add_gravity_field(self, gamma):
        if ti.static(self.stress.n == 6):
            self.stress[0] += float(gamma[0, 0])
            self.stress[1] += float(gamma[1, 1])
            self.stress[2] += float(gamma[2, 2])
            self.stress[3] += 0.5 * float(gamma[0, 1] + gamma[1, 0])
            self.stress[4] += 0.5 * float(gamma[1, 2] + gamma[2, 1])
            self.stress[5] += 0.5 * float(gamma[0, 2] + gamma[2, 0])
        elif ti.static(self.stress.n == 3):
            self.stress += float(gamma)

    @ti.func
    def _compute_external_force(self, gravity):
        return self.m * vec2f(gravity[0], gravity[1])

    @ti.func
    def _compute_internal_force(self):
        return -self.vol * self.stress

    @ti.func
    def _update_particle_state(self, dt, alpha, vPIC, vFLIP):
        v0 = self.v
        flag1 = int(self.fix_v)
        flag2 = Zero2OneVector(flag1)
        self.v = (alpha * vPIC + (1 - alpha) * (vFLIP * dt[None] + v0)) * flag2 + v0 * flag1
        self.x += vPIC * dt[None] * flag2 + v0 * dt[None] * flag1
        if ti.static(GlobalVariable.MPMXPBC):
            self.x[0] -= ti.floor(self.x[0] / GlobalVariable.MPMXSIZE) * GlobalVariable.MPMXSIZE
        if ti.static(GlobalVariable.MPMYPBC):
            self.x[1] -= ti.floor(self.x[1] / GlobalVariable.MPMYSIZE) * GlobalVariable.MPMYSIZE

    @ti.func
    def _update_stress(self, stress):
        if ti.static(self.stress.n == 6):
            self.stress += stress
        elif ti.static(self.stress.n == 3):
            self.stress += mat3x3([stress[0], stress[3], stress[5]],
                                  [stress[3], stress[1], stress[4]],
                                  [stress[5], stress[4], stress[2]])
            
    @ti.func
    def _get_mean_stress(self):
        pressure = 0.
        if ti.static(self.stress.n == 6):
            pressure = 1./3. * (self.stress[0] + self.stress[1] + self.stress[2])
        elif ti.static(self.stress.n == 3):
            pressure = 1./3. * (self.stress[0, 0] + self.stress[1, 1] + self.stress[2, 2])
        return pressure

    @ti.func
    def _update_rigid_body(self, dt):
        self.x += self.v * dt[None]

    @ti.func
    def _compute_particle_velocity(self, xg):
        return self.v - self.velocity_gradient @ (self.x - xg)

@ti.dataclass
class LargeScaleParticle:
    particleID: int
    vol: float
    x: vec3f
    v: vec3f
    stress: vec6f

    @ti.func
    def _restart(self, bodyID, materialID, active, mass, position, velocity, volume, stress, velocity_gradient, fix_v):
        self.x = float(position)
        self.v = float(velocity)
        self.vol = float(volume)
        self.stress = float(stress)

    @ti.func
    def _set_essential(self, particleID, bodyID, materialID, density, particle_volume, position, init_v, fix_v):
        self.particleID = particleID
        self.m = float(particle_volume * density)
        self.x = float(position)
        self.v = float(init_v)
        self.vol = float(particle_volume)

    @ti.func
    def _add_gravity_field(self, gamma):
        if ti.static(self.stress.n == 6):
            self.stress[0] += float(gamma[0, 0])
            self.stress[1] += float(gamma[1, 1])
            self.stress[2] += float(gamma[2, 2])
            self.stress[3] += 0.5 * float(gamma[0, 1] + gamma[1, 0])
            self.stress[4] += 0.5 * float(gamma[1, 2] + gamma[2, 1])
            self.stress[5] += 0.5 * float(gamma[0, 2] + gamma[2, 0])
        elif ti.static(self.stress.n == 3):
            self.stress += float(gamma)

    @ti.func
    def _update_stress(self, stress):
        if ti.static(self.stress.n == 6):
            self.stress += stress
        elif ti.static(self.stress.n == 3):
            self.stress += mat3x3([stress[0], stress[3], stress[5]],
                                  [stress[3], stress[1], stress[4]],
                                  [stress[5], stress[4], stress[2]])
            
    @ti.func
    def _get_mean_stress(self):
        pressure = 0.
        if ti.static(self.stress.n == 6):
            pressure = 1./3. * (self.stress[0] + self.stress[1] + self.stress[2])
        elif ti.static(self.stress.n == 3):
            pressure = 1./3. * (self.stress[0, 0] + self.stress[1, 1] + self.stress[2, 2])
        return pressure

@ti.dataclass
class ParticleCloud:      
    particleID: int
    bodyID: ti.u8
    materialID: ti.u8
    active: ti.u8
    m: float
    vol: float
    x: vec3f
    v: vec3f
    stress: vec6f
    velocity_gradient: mat3x3
    fix_v: vec3u8

    @ti.func
    def _restart(self, bodyID, materialID, active, mass, position, velocity, volume, stress, velocity_gradient, fix_v):
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.active = ti.u8(active)
        self.m = float(mass)
        self.x = float(position)
        self.v = float(velocity)
        self.vol = float(volume)
        self.stress = float(stress)
        self.velocity_gradient = float(velocity_gradient)
        self.fix_v = ti.cast(fix_v, ti.u8)

    @ti.func
    def _set_essential(self, particleID, bodyID, materialID, density, particle_volume, position, init_v, fix_v):
        self.particleID = particleID
        self.active = ti.u8(1)
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.vol = float(particle_volume)
        self.m = float(particle_volume * density)
        self.x = float(position)
        self.v = init_v
        self.fix_v = fix_v

    @ti.func
    def _add_gravity_field(self, gamma):
        if ti.static(self.stress.n == 6):
            self.stress[0] += float(gamma[0, 0])
            self.stress[1] += float(gamma[1, 1])
            self.stress[2] += float(gamma[2, 2])
            self.stress[3] += 0.5 * float(gamma[0, 1] + gamma[1, 0])
            self.stress[4] += 0.5 * float(gamma[1, 2] + gamma[2, 1])
            self.stress[5] += 0.5 * float(gamma[0, 2] + gamma[2, 0])
        elif ti.static(self.stress.n == 3):
            self.stress += float(gamma)

    @ti.func
    def _compute_external_force(self, gravity):
        return self.m * gravity 
    
    @ti.func
    def _compute_internal_force(self):
        return -self.vol * self.stress 
    
    @ti.func
    def _update_particle_state(self, dt, alpha, vPIC, vFLIP):
        v0 = self.v
        flag1 = int(self.fix_v)
        flag2 = Zero2OneVector(flag1)
        self.v = (alpha * vPIC + (1 - alpha) * (vFLIP * dt[None] + v0)) * flag2 + v0 * flag1
        self.x += vPIC * dt[None] * flag2 + v0 * dt[None] * flag1
        if ti.static(GlobalVariable.MPMXPBC):
            self.x[0] -= ti.floor(self.x[0] / GlobalVariable.MPMXSIZE) * GlobalVariable.MPMXSIZE
        if ti.static(GlobalVariable.MPMYPBC):
            self.x[1] -= ti.floor(self.x[1] / GlobalVariable.MPMYSIZE) * GlobalVariable.MPMYSIZE
        if ti.static(GlobalVariable.MPMXPBC):
            self.x[2] -= ti.floor(self.x[2] / GlobalVariable.MPMZSIZE) * GlobalVariable.MPMZSIZE

    @ti.func
    def _update_stress(self, stress):
        if ti.static(self.stress.n == 6):
            self.stress += stress
        elif ti.static(self.stress.n == 3):
            self.stress += mat3x3([stress[0], stress[3], stress[5]],
                                  [stress[3], stress[1], stress[4]],
                                  [stress[5], stress[4], stress[2]])
            
    @ti.func
    def _get_mean_stress(self):
        pressure = 0.
        if ti.static(self.stress.n == 6):
            pressure = 1./3. * (self.stress[0] + self.stress[1] + self.stress[2])
        elif ti.static(self.stress.n == 3):
            pressure = 1./3. * (self.stress[0, 0] + self.stress[1, 1] + self.stress[2, 2])
        return pressure

    @ti.func
    def _update_rigid_body(self, dt):
        self.x += self.v * dt[None]
    
    @ti.func
    def _compute_particle_velocity(self, xg):
        return self.v - self.velocity_gradient @ (self.x - xg)


@ti.dataclass
class ParticleCoupling:      # memory usage: 108B
    particleID: int
    bodyID: ti.u8
    materialID: ti.u8
    active: ti.u8
    coupling: ti.u8
    m: float
    vol: float
    rad: float
    x: vec3f
    verletDisp: vec3f
    v: vec3f
    external_force: vec3f
    stress: vec6f
    velocity_gradient: mat3x3
    fix_v: vec3u8

    @ti.func
    def _restart(self, bodyID, materialID, active, mass, position, velocity, volume, stress, velocity_gradient, fix_v):
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.active = ti.u8(active)
        self.m = float(mass)
        self.x = float(position)
        self.v = float(velocity)
        self.vol = float(volume)
        self.stress = float(stress)
        self.velocity_gradient = float(velocity_gradient)
        self.fix_v = ti.cast(fix_v, ti.u8)

    @ti.func
    def _set_essential(self, particleID, bodyID, materialID, density, particle_volume, position, init_v, fix_v):
        self.particleID = particleID
        self.active = ti.u8(1)
        self.coupling = ti.u8(1)
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.vol = float(particle_volume)
        self.rad = 0.5 * float((particle_volume) ** (1./3.))
        self.m = float(particle_volume * density)
        self.x = float(position)
        self.v = init_v
        self.fix_v = fix_v

    @ti.func
    def _add_gravity_field(self, gamma):
        if ti.static(self.stress.n == 6):
            self.stress[0] += float(gamma[0, 0])
            self.stress[1] += float(gamma[1, 1])
            self.stress[2] += float(gamma[2, 2])
            self.stress[3] += 0.5 * float(gamma[0, 1] + gamma[1, 0])
            self.stress[4] += 0.5 * float(gamma[1, 2] + gamma[2, 1])
            self.stress[5] += 0.5 * float(gamma[0, 2] + gamma[2, 0])
        elif ti.static(self.stress.n == 3):
            self.stress += float(gamma)

    @ti.func
    def _reset_contact_force(self):
        self.external_force = ZEROVEC3f

    @ti.func
    def _reset_mass_density(self):
        self.mass_density = 0.

    @ti.func
    def _compute_external_force(self, gravity):
        return self.m * gravity + self.external_force
    
    @ti.func
    def _compute_internal_force(self):
        return -self.vol * self.stress 

    @ti.func
    def _update_particle_state(self, dt, alpha, vPIC, vFLIP):
        v0 = self.v
        flag1 = int(self.fix_v)
        flag2 = Zero2OneVector(flag1)
        self.v = (alpha * vPIC + (1 - alpha) * (vFLIP * dt[None] + v0)) * flag2 + v0 * flag1
        deltax = vPIC * dt[None] * flag2 + v0 * dt[None] * flag1
        self.x += deltax
        if ti.static(GlobalVariable.MPMXPBC):
            self.x[0] -= ti.floor(self.x[0] / GlobalVariable.MPMXSIZE) * GlobalVariable.MPMXSIZE
        if ti.static(GlobalVariable.MPMYPBC):
            self.x[1] -= ti.floor(self.x[1] / GlobalVariable.MPMYSIZE) * GlobalVariable.MPMYSIZE
        if ti.static(GlobalVariable.MPMXPBC):
            self.x[2] -= ti.floor(self.x[2] / GlobalVariable.MPMZSIZE) * GlobalVariable.MPMZSIZE
        if int(self.coupling) == 1:
            self.verletDisp += deltax
    
    @ti.func
    def _update_stress(self, stress):
        if ti.static(self.stress.n == 6):
            self.stress += stress
        elif ti.static(self.stress.n == 3):
            self.stress += mat3x3([stress[0], stress[3], stress[5]],
                                  [stress[3], stress[1], stress[4]],
                                  [stress[5], stress[4], stress[2]])
            
    @ti.func
    def _get_mean_stress(self):
        pressure = 0.
        if ti.static(self.stress.n == 6):
            pressure = 1./3. * (self.stress[0] + self.stress[1] + self.stress[2])
        elif ti.static(self.stress.n == 3):
            pressure = 1./3. * (self.stress[0, 0] + self.stress[1, 1] + self.stress[2, 2])
        return pressure

    @ti.func
    def _update_rigid_body(self, dt):
        deltax = self.v * dt[None]
        self.x += deltax
        if int(self.coupling) == 1:
            self.verletDisp += deltax
    
    @ti.func
    def _compute_particle_velocity(self, xg):
        return self.v - self.velocity_gradient @ (self.x - xg)
    
    @ti.func
    def _update_contact_interaction(self, cforce, ctorque):
        self.external_force += cforce
    
    @ti.func
    def _get_position(self): return self.x

    @ti.func
    def _get_contact_radius(self, gapn): return self.rad + gapn

    @ti.func
    def _get_radius(self): return self.rad

    @ti.func
    def _get_mass(self): return self.m

    @ti.func
    def _get_velocity(self): return self.v

    @ti.func
    def _get_angular_velocity(self): return ZEROVEC3f

@ti.dataclass
class ImplicitParticleCoupling:
    particleID: int
    bodyID: ti.u8
    materialID: ti.u8
    active: ti.u8
    coupling: ti.u8
    m: float
    vol: float
    rad: float
    x: vec3f
    a: vec3f
    verletDisp: vec3f
    v: vec3f
    external_force: vec3f
    stress: vec6f
    velocity_gradient: mat2x2
    fix_v: vec3u8

    @ti.func
    def _restart(self, bodyID, materialID, active, mass, position, velocity, volume, stress, velocity_gradient, fix_v):
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.active = ti.u8(active)
        self.m = float(mass)
        self.x = float(position)
        self.v = float(velocity)
        self.vol = float(volume)
        self.stress = float(stress)
        self.velocity_gradient = float(velocity_gradient)
        self.fix_v = ti.cast(fix_v, ti.u8)

    @ti.func
    def _set_essential(self, particleID, bodyID, materialID, density, particle_volume, position, init_v, fix_v):
        self.particleID = particleID
        self.active = ti.u8(1)
        self.coupling = ti.u8(1)
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.vol = float(particle_volume)
        self.rad = 0.5 * float((particle_volume) ** (1./3.))
        self.m = float(particle_volume * density)
        self.x = float(position)
        self.v = init_v
        self.fix_v = fix_v

    @ti.func
    def _add_gravity_field(self, gamma):
        if ti.static(self.stress.n == 6):
            self.stress[0] += float(gamma[0, 0])
            self.stress[1] += float(gamma[1, 1])
            self.stress[2] += float(gamma[2, 2])
            self.stress[3] += 0.5 * float(gamma[0, 1] + gamma[1, 0])
            self.stress[4] += 0.5 * float(gamma[1, 2] + gamma[2, 1])
            self.stress[5] += 0.5 * float(gamma[0, 2] + gamma[2, 0])
        elif ti.static(self.stress.n == 3):
            self.stress += float(gamma)

    @ti.func
    def _update_stress(self, stress):
        if ti.static(self.stress.n == 6):
            self.stress += stress
        elif ti.static(self.stress.n == 3):
            self.stress += mat3x3([stress[0], stress[3], stress[5]],
                                  [stress[3], stress[1], stress[4]],
                                  [stress[5], stress[4], stress[2]])
            
    @ti.func
    def _get_mean_stress(self):
        pressure = 0.
        if ti.static(self.stress.n == 6):
            pressure = 1./3. * (self.stress[0] + self.stress[1] + self.stress[2])
        elif ti.static(self.stress.n == 3):
            pressure = 1./3. * (self.stress[0, 0] + self.stress[1, 1] + self.stress[2, 2])
        return pressure

    @ti.func
    def _reset_contact_force(self):
        self.external_force = ZEROVEC3f

    @ti.func
    def _reset_mass_density(self):
        self.mass_density = 0.

    @ti.func
    def _compute_external_force(self, gravity):
        return self.m * gravity + self.external_force 
    
    @ti.func
    def _compute_internal_force(self):
        return -self.vol * self.stress 

    @ti.func
    def _update_particle_state(self, dt, alpha, vPIC, acc, disp):
        v0 = self.v
        flag1 = int(self.fix_v)
        flag2 = Zero2OneVector(flag1)
        vFLIP = 0.5 * (acc + self.a) * dt[None]
        self.a = acc
        self.v = (alpha * vPIC + (1 - alpha) * (vFLIP + v0)) * flag2 + v0 * flag1
        deltax = disp * flag2 + v0 * dt[None] * flag1
        self.x += deltax
        if ti.static(GlobalVariable.MPMXPBC):
            self.x[0] -= ti.floor(self.x[0] / GlobalVariable.MPMXSIZE) * GlobalVariable.MPMXSIZE
        if ti.static(GlobalVariable.MPMYPBC):
            self.x[1] -= ti.floor(self.x[1] / GlobalVariable.MPMYSIZE) * GlobalVariable.MPMYSIZE
        if ti.static(GlobalVariable.MPMXPBC):
            self.x[2] -= ti.floor(self.x[2] / GlobalVariable.MPMZSIZE) * GlobalVariable.MPMZSIZE
        if int(self.coupling) == 1:
            self.verletDisp += deltax

    @ti.func
    def _update_rigid_body(self, dt):
        deltax = self.v * dt[None]
        self.x += deltax
        if int(self.coupling) == 1:
            self.verletDisp += deltax
    
    @ti.func
    def _compute_particle_velocity(self, xg):
        return self.v - self.velocity_gradient @ (self.x - xg)
    
    @ti.func
    def _update_contact_interaction(self, cforce, ctorque):
        self.external_force += cforce
    
    @ti.func
    def _get_position(self): return self.x

    @ti.func
    def _get_contact_radius(self, gapn): return self.rad + gapn

    @ti.func
    def _get_radius(self): return self.rad

    @ti.func
    def _get_mass(self): return self.m

    @ti.func
    def _get_velocity(self): return self.v

    @ti.func
    def _get_angular_velocity(self): return ZEROVEC3f


@ti.dataclass
class ImplicitParticle:
    particleID: int
    bodyID: ti.u8
    materialID: ti.u8
    active: ti.u8
    m: float
    vol: float
    vol0 : float
    x: vec3f
    v: vec3f
    a: vec3f
    stress: vec6f
    stress0: vec6f
    velocity_gradient: mat3x3
    fix_v: vec3u8

    @ti.func
    def _restart(self, bodyID, materialID, active, mass, position, velocity, volume, stress, velocity_gradient, fix_v):
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.active = ti.u8(active)
        self.m = float(mass)
        self.x = float(position)
        self.v = float(velocity)
        self.vol = float(volume)
        self.vol0 = float(volume)
        self.stress = float(stress)
        self.stress0 = float(stress)
        self.velocity_gradient = float(velocity_gradient)
        self.fix_v = ti.cast(fix_v, ti.u8)

    @ti.func
    def _set_essential(self, particleID, bodyID, materialID, density, particle_volume, position, init_v, fix_v):
        self.particleID = particleID
        self.active = ti.u8(1)
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.vol = float(particle_volume)
        self.vol0 = float(particle_volume)
        self.m = float(particle_volume * density)
        self.x = float(position)
        self.v = init_v
        self.fix_v = fix_v

    @ti.func
    def _add_gravity_field(self, gamma):
        if ti.static(self.stress.n == 6):
            self.stress[0] += float(gamma[0, 0])
            self.stress[1] += float(gamma[1, 1])
            self.stress[2] += float(gamma[2, 2])
            self.stress[3] += 0.5 * float(gamma[0, 1] + gamma[1, 0])
            self.stress[4] += 0.5 * float(gamma[1, 2] + gamma[2, 1])
            self.stress[5] += 0.5 * float(gamma[0, 2] + gamma[2, 0])
        elif ti.static(self.stress.n == 3):
            self.stress += float(gamma)
        self.stress0 = self.stress

    @ti.func
    def _update_stress(self, stress):
        if ti.static(self.stress.n == 6):
            self.stress += stress
        elif ti.static(self.stress.n == 3):
            self.stress += mat3x3([stress[0], stress[3], stress[5]],
                                  [stress[3], stress[1], stress[4]],
                                  [stress[5], stress[4], stress[2]])
            
    @ti.func
    def _get_mean_stress(self):
        pressure = 0.
        if ti.static(self.stress.n == 6):
            pressure = 1./3. * (self.stress[0] + self.stress[1] + self.stress[2])
        elif ti.static(self.stress.n == 3):
            pressure = 1./3. * (self.stress[0, 0] + self.stress[1, 1] + self.stress[2, 2])
        return pressure

    @ti.func
    def _compute_external_force(self, gravity):
        return self.m * gravity 
    
    @ti.func
    def _compute_internal_force(self):
        return -self.vol * self.stress 
    
    @ti.func
    def _update_particle_state(self, dt, alpha, vPIC, acc, disp):
        v0 = self.v
        flag1 = int(self.fix_v)
        flag2 = Zero2OneVector(flag1)
        vFLIP = 0.5 * (acc + self.a) * dt[None]
        self.a = acc
        self.v = (alpha * vPIC + (1 - alpha) * (vFLIP + v0)) * flag2 + v0 * flag1
        self.x += disp * flag2 + v0 * dt[None] * flag1
        if ti.static(GlobalVariable.MPMXPBC):
            self.x[0] -= ti.floor(self.x[0] / GlobalVariable.MPMXSIZE) * GlobalVariable.MPMXSIZE
        if ti.static(GlobalVariable.MPMYPBC):
            self.x[1] -= ti.floor(self.x[1] / GlobalVariable.MPMYSIZE) * GlobalVariable.MPMYSIZE
        if ti.static(GlobalVariable.MPMXPBC):
            self.x[2] -= ti.floor(self.x[2] / GlobalVariable.MPMZSIZE) * GlobalVariable.MPMZSIZE

    @ti.func
    def _update_rigid_body(self, dt):
        self.x += self.v * dt[None]
    
    @ti.func
    def _compute_particle_velocity(self, xg):
        return self.v - self.velocity_gradient @ (self.x - xg)


@ti.dataclass
class ImplicitParticle2D:
    particleID: int
    bodyID: ti.u8
    materialID: ti.u8
    active: ti.u8
    m: float
    vol: float
    vol0 : float
    x: vec2f
    v: vec2f
    a: vec2f
    stress: vec6f
    stress0: vec6f
    velocity_gradient: mat2x2
    fix_v: vec2u8

    @ti.func
    def _restart(self, bodyID, materialID, active, mass, position, velocity, volume, stress, velocity_gradient, fix_v):
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.active = ti.u8(active)
        self.m = float(mass)
        self.x = float(position)
        self.v = float(velocity)
        self.vol = float(volume)
        self.vol0 = float(volume)
        self.stress = float(stress)
        self.stress0 = float(stress)
        self.velocity_gradient = float(velocity_gradient)
        self.fix_v = ti.cast(fix_v, ti.u8)

    @ti.func
    def _set_essential(self, particleID, bodyID, materialID, density, particle_volume, position, init_v, fix_v):
        self.particleID = particleID
        self.active = ti.u8(1)
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.vol = float(particle_volume)
        self.vol0 = float(particle_volume)
        self.m = float(particle_volume * density)
        self.x = float(position)
        self.v = init_v
        self.fix_v = fix_v

    @ti.func
    def _add_gravity_field(self, gamma):
        if ti.static(self.stress.n == 6):
            self.stress[0] += float(gamma[0, 0])
            self.stress[1] += float(gamma[1, 1])
            self.stress[2] += float(gamma[2, 2])
            self.stress[3] += 0.5 * float(gamma[0, 1] + gamma[1, 0])
            self.stress[4] += 0.5 * float(gamma[1, 2] + gamma[2, 1])
            self.stress[5] += 0.5 * float(gamma[0, 2] + gamma[2, 0])
        elif ti.static(self.stress.n == 3):
            self.stress += float(gamma)
        self.stress0 = self.stress

    @ti.func
    def _update_stress(self, stress):
        if ti.static(self.stress.n == 6):
            self.stress += stress
        elif ti.static(self.stress.n == 3):
            self.stress += mat3x3([stress[0], stress[3], stress[5]],
                                  [stress[3], stress[1], stress[4]],
                                  [stress[5], stress[4], stress[2]])
        self.stress0 = self.stress

    @ti.func
    def _get_mean_stress(self):
        pressure = 0.
        if ti.static(self.stress.n == 6):
            pressure = 1./3. * (self.stress[0] + self.stress[1] + self.stress[2])
        elif ti.static(self.stress.n == 3):
            pressure = 1./3. * (self.stress[0, 0] + self.stress[1, 1] + self.stress[2, 2])
        return pressure

    @ti.func
    def _set_particle_traction(self, traction):
        self.traction += float(traction)

    @ti.func
    def _compute_external_force(self, gravity):
        return self.m * vec2f(gravity[0], gravity[1]) 
    
    @ti.func
    def _compute_internal_force(self):
        return -self.vol * self.stress 
    
    @ti.func
    def _update_particle_state(self, dt, alpha, vPIC, acc, disp):
        v0 = self.v
        flag1 = int(self.fix_v)
        flag2 = Zero2OneVector(flag1)
        vFLIP = 0.5 * (acc + self.a) * dt[None]
        self.a = acc
        self.v = (alpha * vPIC + (1 - alpha) * (vFLIP + v0)) * flag2 + v0 * flag1
        self.x += disp * flag2 + v0 * dt[None] * flag1
        if ti.static(GlobalVariable.MPMXPBC):
            self.x[0] -= ti.floor(self.x[0] / GlobalVariable.MPMXSIZE) * GlobalVariable.MPMXSIZE
        if ti.static(GlobalVariable.MPMYPBC):
            self.x[1] -= ti.floor(self.x[1] / GlobalVariable.MPMYSIZE) * GlobalVariable.MPMYSIZE

    @ti.func
    def _update_rigid_body(self, dt):
        self.x += self.v * dt[None]

@ti.dataclass
class ParticleCPDI:
    r0: vec3f
    r1: vec3f
    r2: vec3f

    @ti.func
    def _set(self, r0, r1, r2):
        self.r0 = r0
        self.r1 = r1
        self.r2 = r2

    @ti.func
    def _update(self, deformation_gradient):
        self.r0 = deformation_gradient @ self.r0
        self.r1 = deformation_gradient @ self.r1
        self.r2 = deformation_gradient @ self.r2

@ti.dataclass
class ParticleCloudIncompressible3D:      # memory usage: 108B
    particleID: int
    bodyID: ti.u8
    materialID: ti.u8
    active: ti.u8
    m: float
    vol: float
    x: vec3f
    v: vec3f
    pressure: float
    velocity_gradient: mat3x3
    fix_v: vec3u8

    @ti.func
    def _restart(self, bodyID, materialID, active, mass, position, velocity, volume, stress, velocity_gradient, fix_v):
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.active = ti.u8(active)
        self.m = float(mass)
        self.x = float(position)
        self.v = float(velocity)
        self.vol = float(volume)
        self.pressure = float(stress)
        self.velocity_gradient = float(velocity_gradient)
        self.fix_v = ti.cast(fix_v, ti.u8)

    @ti.func
    def _set_essential(self, particleID, bodyID, materialID, density, particle_volume, position, init_v, fix_v):
        self.particleID = particleID
        self.active = ti.u8(1)
        self.bodyID = ti.u8(bodyID)
        self.materialID = ti.u8(materialID)
        self.vol = float(particle_volume)
        self.m = float(particle_volume * density)
        self.x = float(position)
        self.v = init_v
        self.fix_v = fix_v

    @ti.func
    def _add_gravity_field(self, gamma):
        self.pressure += 1./3. * float(gamma[0, 0] + gamma[1, 1] + gamma[2, 2])

    @ti.func
    def _compute_external_force(self, gravity):
        return self.m * gravity
    
    @ti.func
    def _compute_internal_force(self):
        pass
    
    @ti.func
    def _update_particle_state(self, dt, alpha, vPIC, vFLIP):
        v0 = self.v
        self.v = alpha * vPIC + (1 - alpha) * (vFLIP + v0)
        self.x += vPIC * dt[None]
        if ti.static(GlobalVariable.MPMXPBC):
            self.x[0] -= ti.floor(self.x[0] / GlobalVariable.MPMXSIZE) * GlobalVariable.MPMXSIZE
        if ti.static(GlobalVariable.MPMYPBC):
            self.x[1] -= ti.floor(self.x[1] / GlobalVariable.MPMYSIZE) * GlobalVariable.MPMYSIZE
        if ti.static(GlobalVariable.MPMXPBC):
            self.x[2] -= ti.floor(self.x[2] / GlobalVariable.MPMZSIZE) * GlobalVariable.MPMZSIZE

    @ti.func
    def _update_stress(self, stress):
        self.pressure += 1./3. * (stress[0] + stress[1] + stress[2])

    @ti.func
    def _get_mean_stress(self):
        return self.pressure

    @ti.func
    def _update_rigid_body(self, dt):
        pass
    
    @ti.func
    def _compute_particle_velocity(self, xg):
        return self.v - self.velocity_gradient @ (self.x - xg)