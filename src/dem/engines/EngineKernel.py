import taichi as ti

from src.utils.constants import ZEROVEC3f
from src.utils.Quaternion import SetDQ, SetToRotate, UpdateQAccurate
from src.utils.ScalarFunction import PairingMapping, sgn
from src.utils.TypeDefination import vec3f
from src.utils.VectorFunction import SquaredLength, Squared, Normalize
from src.utils import GlobalVariable


@ti.kernel
def particle_force_reset_(particleNum: int, particle: ti.template()):
    for np in range(particleNum):
        particle[np].contact_force = vec3f([0., 0., 0.])
        particle[np].contact_torque = vec3f([0., 0., 0.])
        if ti.static(GlobalVariable.TRACKENERGY):
            particle[np].elastic_energy = 0.

@ti.kernel
def wall_force_reset_(wallNum: int, wall: ti.template()):
    for nw in range(wallNum):
        wall[nw]._reset()

@ti.func
def cundall_damping_energy(fdamp, tdamp, velocity, angular_velocity, force, ctorque, dt):
    return -ti.abs(velocity).dot(ti.abs(force)) * fdamp * dt[None] - ti.abs(angular_velocity).dot(ti.abs(ctorque)) * tdamp * dt[None]

@ti.func
def cundall_damp1st(damp, force, vel):
    force[0] *= 1. - damp * sgn(force[0] * vel[0])
    force[1] *= 1. - damp * sgn(force[1] * vel[1])
    force[2] *= 1. - damp * sgn(force[2] * vel[2])
    return force

@ti.func
def cundall_damp2nd(damp, dt, force, vel, accel):
    force[0] *= 1. - damp * sgn(force[0] * (vel[0] + 0.5 * dt[None] * accel[0]))
    force[1] *= 1. - damp * sgn(force[1] * (vel[1] + 0.5 * dt[None] * accel[1]))
    force[2] *= 1. - damp * sgn(force[2] * (vel[2] + 0.5 * dt[None] * accel[2]))
    return force

@ti.func
def w_dot(w, torque, inertia, inv_inertia):
    return vec3f((torque[0] + w[1] * w[2] * (inertia[1] - inertia[2])) * inv_inertia[0],
                 (torque[1] + w[2] * w[0] * (inertia[2] - inertia[0])) * inv_inertia[1],
                 (torque[2] + w[0] * w[1] * (inertia[0] - inertia[1])) * inv_inertia[2])

@ti.kernel
def move_spheres_euler_(bodyNum: int, dt: ti.template(), sphere: ti.template(), particle: ti.template(), material: ti.template(), gravity: ti.types.vector(3, float)):
    for nsphere in range(bodyNum):
        np = sphere[nsphere].sphereIndex
        materialID = int(particle[np].materialID)
        fdamp = material[materialID].fdamp
        tdamp = material[materialID].tdamp
        
        cforce, ctorque = particle[np].contact_force, particle[np].contact_torque
        # particle_num = particle_particle[nsphere + 1] - particle_particle[nsphere]
        # for j in range(nsphere * potential_particle_num, nsphere * potential_particle_num + particle_num):
        #     nc = particle_contact[j]
        #     cforce += cplistPP[nc].cnforce + cplistPP[nc].csforce
        #     ctorque += cplistPP[nc].torque
        # for j in range((nsphere + 1)*potential_particle_num - iparticle_particle[nsphere + 1], (nsphere + 1) * potential_particle_num):
        #     nc = particle_contact[j]
        #     cforce -= cplistPP[nc].cnforce + cplistPP[nc].csforce
         #    ctorque += cplistPP[nc].torque
    
        # wall_num = particle_wall[nsphere + 1] - particle_wall[nsphere]
        # for j in range(nsphere * potential_wall_num, nsphere * potential_wall_num + wall_num):
        #     nc = wall_contact[j]
        #     cforce += cplistPW[nc].cnforce + cplistPW[nc].csforce
        #     ctorque += cplistPW[nc].torque

        mass, is_fix = particle[np].m, sphere[nsphere].fix_v
        old_vel, old_disp = particle[np].v, particle[np].verletDisp
        force = cundall_damp1st(fdamp, cforce + gravity * mass, old_vel)
        
        av = force / mass * int(is_fix)
        vel = old_vel + dt[None] * av
        delta_x = dt[None] * vel

        particle[np].v = vel
        particle[np].x += delta_x
        particle[np].verletDisp = old_disp + delta_x

        inv_i, is_fix = sphere[nsphere].inv_I, sphere[nsphere].fix_w
        old_omega, old_q = particle[np].w, sphere[nsphere].q

        torque = cundall_damp1st(tdamp, ctorque, old_omega)
        aw = torque * inv_i * int(is_fix)
        omega = old_omega + dt[None] * aw
        dq = dt[None] * SetDQ(old_q, omega)
        q = old_q + dq

        particle[np].w = omega
        sphere[nsphere].q = Normalize(q)

        if ti.static(GlobalVariable.TRACKENERGY):
            particle[np].damp_energy += cundall_damping_energy(fdamp, tdamp, old_vel, old_omega, cforce + gravity * mass, ctorque, dt)

@ti.kernel
def move_clumps_euler_(bodyNum: int, dt: ti.template(), clump: ti.template(), particle: ti.template(), material: ti.template(), gravity: ti.types.vector(3, float)):    
    for nclump in range(bodyNum):
        pebb_beg, pebb_end = clump[nclump].startIndex, clump[nclump].endIndex
        materialID = int(particle[pebb_beg].materialID)
        fdamp = material[materialID].fdamp
        tdamp = material[materialID].tdamp
        
        mass = clump[nclump].m
        old_vel, old_pos = clump[nclump].v, clump[nclump].mass_center

        cforce, ctorque = ZEROVEC3f, ZEROVEC3f
        for np in range(pebb_beg, pebb_end + 1):
            contact_force = particle[np].contact_force
            cforce += contact_force
            ctorque += particle[np].contact_torque + (particle[np].x - old_pos).cross(contact_force)
            # particle_num = particle_particle[np + 1] - particle_particle[np]
            # for j in range(np * potential_particle_num, np * potential_particle_num + particle_num):
            #     nc = particle_contact[j]
            #     contact_force = cplistPP[nc].cnforce + cplistPP[nc].csforce
            #     cforce += contact_force
            #     ctorque += cplistPP[nc].torque + contact_force.cross(old_pos - particle[np].x)
            # for j in range((np + 1)*potential_particle_num - iparticle_particle[np + 1], (np + 1) * potential_particle_num):
            #     nc = particle_contact[j]
            #     contact_force = cplistPP[nc].cnforce + cplistPP[nc].csforce
            #     cforce -= contact_force
            #     ctorque += cplistPP[nc].torque - contact_force.cross(old_pos - particle[np].x)

            # wall_num = particle_wall[np + 1] - particle_wall[np]
            # for j in range(np * potential_wall_num, np * potential_wall_num + wall_num):
            #     nc = wall_contact[j]
            #     contact_force = cplistPW[nc].cnforce + cplistPW[nc].csforce
            #     cforce += cplistPW[nc].cnforce + cplistPW[nc].csforce
            #     ctorque += cplistPW[nc].torque + contact_force.cross(old_pos - particle[np].x)
        force = cundall_damp1st(fdamp, cforce + gravity * mass , old_vel)
        av = force / mass
        vel = old_vel + dt[None] * av
        pos = old_pos + dt[None] * vel

        clump[nclump].v = vel
        clump[nclump].mass_center = pos

        inv_i = clump[nclump].inv_I
        old_omega, old_q = clump[nclump].w, clump[nclump].q

        torque = cundall_damp1st(tdamp, ctorque, old_omega)
        rotation_matrix = SetToRotate(old_q)
        torque_local = rotation_matrix.transpose() @ torque
        omega_local = rotation_matrix.transpose() @ old_omega
        aw_local = inv_i * (torque_local - omega_local.cross(1. / inv_i * omega_local))
        aw = rotation_matrix @ aw_local 
        omega = old_omega + aw * dt[None]

        # see Langston et al. (2004) Distinct element modelling of non-spherical frictionless particle flow.
        dq = SetDQ(old_q, rotation_matrix.transpose() @ omega) * dt[None]                  # SetDQ(old_q, old_omega)
        q = Normalize(old_q + dq)

        clump[nclump].w = omega
        clump[nclump].q = q
        
        rotation_matrix1 = SetToRotate(q)
        for np in range(pebb_beg, pebb_end + 1):
            old_pebble_pos = particle[np].x
            pebble_pos = pos + rotation_matrix1 @ (rotation_matrix.transpose() @ (old_pebble_pos - old_pos))
            particle[np].v = vel + omega.cross(pebble_pos - pos)
            particle[np].w = omega
            particle[np].x = pebble_pos
            particle[np].verletDisp += pebble_pos - old_pebble_pos

        if ti.static(GlobalVariable.TRACKENERGY):
            damp_energy = cundall_damping_energy(fdamp, tdamp, old_vel, old_omega, cforce + gravity * mass, ctorque, dt)
            for np in range(pebb_beg, pebb_end + 1):
                particle[np].damp_energy += damp_energy / (pebb_end - pebb_beg + 1)

@ti.kernel
def move_level_set_euler_(bodyNum: int, dt: ti.template(), sphere: ti.template(), rigid: ti.template(), material: ti.template(), gravity: ti.types.vector(3, float)):
    for np in range(bodyNum):
        materialID = int(rigid[np].materialID)
        fdamp = material[materialID].fdamp
        tdamp = material[materialID].tdamp
        
        cforce, ctorque = rigid[np].contact_force, rigid[np].contact_torque
        old_center, old_x = rigid[np].mass_center, sphere[np].x

        mass, is_fix = rigid[np].m, rigid[np].is_fix
        old_vel = rigid[np].v
        force = cundall_damp1st(fdamp, cforce + gravity * mass , old_vel)
        
        av = force / mass * int(is_fix)
        vel = old_vel + dt[None] * av
        delta_x = dt[None] * vel
        mass_center = old_center + delta_x

        rigid[np].v = vel
        rigid[np].mass_center = mass_center

        inv_i = rigid[np].inv_I
        old_omega, old_q = rigid[np].w, rigid[np].q

        torque = cundall_damp1st(tdamp, ctorque, old_omega)
        rotation_matrix = SetToRotate(old_q)
        torque_local = rotation_matrix.transpose() @ torque
        omega_local = rotation_matrix.transpose() @ old_omega
        aw_local = inv_i * (torque_local - omega_local.cross(1. / inv_i * omega_local))
        aw = rotation_matrix @ aw_local * int(is_fix)
        omega = old_omega + aw * dt[None]

        # see Langston et al. (2004) Distinct element modelling of non-spherical frictionless particle flow.
        dq = SetDQ(old_q, rotation_matrix.transpose() @ omega) * dt[None]                  # SetDQ(old_q, old_omega)
        q = Normalize(old_q + dq)
        rigid[np].w = omega
        rigid[np].q = q

        rotation_matrix1 = SetToRotate(q)
        sphere[np]._move(mass_center + rotation_matrix1 @ (rotation_matrix.transpose() @ (old_x - old_center)) - old_x)

        if ti.static(GlobalVariable.TRACKENERGY):
            rigid[np].damp_energy += cundall_damping_energy(fdamp, tdamp, old_vel, old_omega, cforce + gravity * mass, ctorque, dt)

@ti.kernel
def move_spheres_verlet_(bodyNum: int, dt: ti.template(), sphere: ti.template(), particle: ti.template(), material: ti.template(), gravity: ti.types.vector(3, float)):
    for nsphere in range(bodyNum):
        np = sphere[nsphere].sphereIndex
        materialID = int(particle[np].materialID)
        fdamp = material[materialID].fdamp
        tdamp = material[materialID].tdamp
        
        cforce, ctorque = particle[np].contact_force, particle[np].contact_torque
        # particle_num = particle_particle[nsphere + 1] - particle_particle[nsphere]
        # for j in range(nsphere * potential_particle_num, nsphere * potential_particle_num + particle_num):
        #     nc = particle_contact[j]
        #     cforce += cplistPP[nc].cnforce + cplistPP[nc].csforce
        #     ctorque += cplistPP[nc].torque
        # for j in range((nsphere + 1)*potential_particle_num - iparticle_particle[nsphere + 1], (nsphere + 1) * potential_particle_num):
        #     nc = particle_contact[j]
        #     cforce -= cplistPP[nc].cnforce + cplistPP[nc].csforce
         #    ctorque += cplistPP[nc].torque
    
        # wall_num = particle_wall[nsphere + 1] - particle_wall[nsphere]
        # for j in range(nsphere * potential_wall_num, nsphere * potential_wall_num + wall_num):
        #     nc = wall_contact[j]
        #     cforce += cplistPW[nc].cnforce + cplistPW[nc].csforce
        #     ctorque += cplistPW[nc].torque

        mass, is_fix = particle[np].m, sphere[nsphere].fix_v
        old_av, old_vel, old_disp = sphere[nsphere].a, particle[np].v, particle[np].verletDisp
        vel_half = old_vel + 0.5 * dt[None] * old_av
        force = cundall_damp1st(fdamp, cforce + gravity * mass , vel_half)

        delta_x = dt[None] * vel_half 
        av = force / mass * int(is_fix)
        vel = vel_half + 0.5 * av * dt[None]
        
        sphere[nsphere].a = av
        particle[np].v = vel
        particle[np].x += delta_x
        particle[np].verletDisp = old_disp + delta_x
        
        # see Rozmanov and Kusalik (2010) Robust rotational-velocity-Verlet integration methods. Phys. Rev. E
        inv_i, is_fix = sphere[nsphere].inv_I, sphere[nsphere].fix_w
        old_angmoment, old_omega, old_q = sphere[nsphere].angmoment, particle[np].w, sphere[nsphere].q

        torque = cundall_damp1st(tdamp, ctorque, old_omega + 0.5 * ctorque * inv_i * dt[None])
        angmoment = (old_angmoment + 0.5 * torque * dt[None]) * int(is_fix)
        omega = angmoment * inv_i
        dq = dt[None] * SetDQ(old_q, omega)
        half_q = old_q + 0.5 * dq
        angmoment_half = (old_angmoment + torque * dt[None]) * int(is_fix)
        omega_half = angmoment_half * inv_i
        dq_half = dt[None] * SetDQ(half_q, omega_half)
        q = old_q + dq_half

        particle[np].w = omega_half
        sphere[nsphere].angmoment = angmoment_half 
        sphere[nsphere].q = Normalize(q)

        if ti.static(GlobalVariable.TRACKENERGY):
            particle[np].damp_energy += cundall_damping_energy(fdamp, tdamp, old_vel, old_omega, cforce + gravity * mass, ctorque, dt)

@ti.kernel
def move_clumps_verlet_(bodyNum: int, dt: ti.template(), clump: ti.template(), particle: ti.template(), material: ti.template(), gravity: ti.types.vector(3, float)):    
    for nclump in range(bodyNum):
        pebb_beg, pebb_end = clump[nclump].startIndex, clump[nclump].endIndex
        materialID = int(particle[pebb_beg].materialID)
        fdamp = material[materialID].fdamp
        tdamp = material[materialID].tdamp
        
        mass = clump[nclump].m
        old_av, old_vel, old_pos = clump[nclump].a, clump[nclump].v, clump[nclump].mass_center

        cforce, ctorque = ZEROVEC3f, ZEROVEC3f
        for np in range(pebb_beg, pebb_end + 1):
            contact_force = particle[np].contact_force
            cforce += contact_force
            ctorque += particle[np].contact_torque + (particle[np].x - old_pos).cross(contact_force)
            # particle_num = particle_particle[np + 1] - particle_particle[np]
            # for j in range(np * potential_particle_num, np * potential_particle_num + particle_num):
            #     nc = particle_contact[j]
            #     contact_force = cplistPP[nc].cnforce + cplistPP[nc].csforce
            #     cforce += contact_force
            #     ctorque += cplistPP[nc].torque + contact_force.cross(old_pos - particle[np].x)
            # for j in range((np + 1)*potential_particle_num - iparticle_particle[np + 1], (np + 1) * potential_particle_num):
            #     nc = particle_contact[j]
            #     contact_force = cplistPP[nc].cnforce + cplistPP[nc].csforce
            #     cforce -= contact_force
            #     ctorque += cplistPP[nc].torque - contact_force.cross(old_pos - particle[np].x)

            # wall_num = particle_wall[np + 1] - particle_wall[np]
            # for j in range(np * potential_wall_num, np * potential_wall_num + wall_num):
            #     nc = wall_contact[j]
            #     contact_force = cplistPW[nc].cnforce + cplistPW[nc].csforce
            #     cforce += cplistPW[nc].cnforce + cplistPW[nc].csforce
            #     ctorque += cplistPW[nc].torque + contact_force.cross(old_pos - particle[np].x)
        
        vel_half = old_vel + 0.5 * dt[None] * old_av
        force = cundall_damp1st(fdamp, cforce + gravity * mass , vel_half)
        pos = old_pos + dt[None] * vel_half 
        av = force / mass 
        vel = vel_half + 0.5 * av * dt[None]

        clump[nclump].a = av
        clump[nclump].v = vel
        clump[nclump].mass_center = pos

        inv_i = clump[nclump].inv_I
        i = 1. / inv_i
        old_omega, old_q = clump[nclump].w, clump[nclump].q

        torque = cundall_damp1st(tdamp, ctorque, old_omega)
        rotation_matrix = SetToRotate(old_q)

        torque_local = rotation_matrix.transpose() @ torque
        omega_local = rotation_matrix.transpose() @ old_omega
        K1 = dt[None] * w_dot(omega_local, torque_local, i, inv_i)
        K2 = dt[None] * w_dot(omega_local + K1, torque_local, i, inv_i)
        K3 = dt[None] * w_dot(omega_local + 0.25 * (K1 + K2), torque_local, i, inv_i)
        omega_local += (K1 + K2 + 4. * K3) / 6.
        omega = rotation_matrix @ omega_local
        # see Langston et al. (2004) Distinct element modelling of non-spherical frictionless particle flow.
        dq = SetDQ(old_q, omega_local) * dt[None]                 
        q = Normalize(old_q + dq)

        clump[nclump].angmoment = omega * i
        clump[nclump].w = omega
        clump[nclump].q = q
        
        rotation_matrix1 = SetToRotate(q)
        for np in range(pebb_beg, pebb_end + 1):
            old_pebble_pos = particle[np].x
            pebble_pos = pos + rotation_matrix1 @ (rotation_matrix.transpose() @ (old_pebble_pos - old_pos))
            particle[np].v = vel + omega.cross(pebble_pos - pos)
            particle[np].w = omega
            particle[np].x = pebble_pos
            particle[np].verletDisp += pebble_pos - old_pebble_pos

@ti.kernel
def move_level_set_verlet_(bodyNum: int, dt: ti.template(), sphere: ti.template(), rigid: ti.template(), material: ti.template(), gravity: ti.types.vector(3, float)):    
    for np in range(bodyNum):
        materialID = int(rigid[np].materialID)
        fdamp = material[materialID].fdamp
        tdamp = material[materialID].tdamp
        
        cforce, ctorque = rigid[np].contact_force, rigid[np].contact_torque
        old_center, old_x = rigid[np].mass_center, sphere[np].x

        mass, is_fix = rigid[np].m, rigid[np].is_fix
        old_av, old_vel = rigid[np].a, rigid[np].v
        
        vel_half = old_vel + 0.5 * dt[None] * old_av
        force = cundall_damp1st(fdamp, cforce + gravity * mass , vel_half)
        delta_x = dt[None] * vel_half 
        av = force / mass * int(is_fix)
        vel = vel_half + 0.5 * av * dt[None]
        mass_center = old_center + delta_x

        rigid[np].v = vel
        rigid[np].a = av
        rigid[np].mass_center = mass_center

        inv_i = rigid[np].inv_I
        i = 1. / inv_i
        old_omega, old_q = rigid[np].w, rigid[np].q

        torque = cundall_damp1st(tdamp, ctorque, old_omega)
        rotation_matrix = SetToRotate(old_q)

        torque_local = rotation_matrix.transpose() @ torque
        omega_local = rotation_matrix.transpose() @ old_omega
        K1 = dt[None] * w_dot(omega_local, torque_local, i, inv_i) * int(is_fix)
        K2 = dt[None] * w_dot(omega_local + K1, torque_local, i, inv_i) * int(is_fix)
        K3 = dt[None] * w_dot(omega_local + 0.25 * (K1 + K2), torque_local, i, inv_i) * int(is_fix)
        omega_local += (K1 + K2 + 4. * K3) / 6.
        omega = rotation_matrix @ omega_local
        # see Langston et al. (2004) Distinct element modelling of non-spherical frictionless particle flow.
        dq = SetDQ(old_q, omega_local) * dt[None]                 
        q = Normalize(old_q + dq)

        rigid[np].angmoment = omega * i
        rigid[np].w = omega
        rigid[np].q = q

        rotation_matrix1 = SetToRotate(q)
        sphere[np]._move(mass_center + rotation_matrix1 @ (rotation_matrix.transpose() @ (old_x - old_center)) - old_x)

        if ti.static(GlobalVariable.TRACKENERGY):
            rigid[np].damp_energy += cundall_damping_energy(fdamp, tdamp, old_vel, old_omega, cforce + gravity * mass, ctorque, dt)

@ti.kernel
def move_clumps_predictor_corrector_(bodyNum: int, dt: ti.template(), clump: ti.template(), particle: ti.template(), material: ti.template(), gravity: ti.types.vector(3, float)):    
    for nclump in range(bodyNum):
        pebb_beg, pebb_end = clump[nclump].startIndex, clump[nclump].endIndex
        materialID = int(particle[pebb_beg].materialID)
        fdamp = material[materialID].fdamp
        tdamp = material[materialID].tdamp
        
        mass = clump[nclump].m
        old_av, old_vel, old_pos = clump[nclump].a, clump[nclump].v, clump[nclump].mass_center

        cforce, ctorque = ZEROVEC3f, ZEROVEC3f
        for np in range(pebb_beg, pebb_end + 1):
            contact_force = particle[np].contact_force
            cforce += contact_force
            ctorque += particle[np].contact_torque + (particle[np].x - old_pos).cross(contact_force)
        
        vel_half = old_vel + 0.5 * dt[None] * old_av
        force = cundall_damp1st(fdamp, cforce + gravity * mass , vel_half)
        pos = old_pos + dt[None] * vel_half 
        av = force / mass 
        vel = vel_half + 0.5 * av * dt[None]

        clump[nclump].a = av
        clump[nclump].v = vel
        clump[nclump].mass_center = pos

        inv_i = clump[nclump].inv_I
        i = 1. / inv_i
        old_omega, old_q, old_alpha = clump[nclump].w, clump[nclump].q, clump[nclump].angmoment

        torque = cundall_damp1st(tdamp, ctorque, old_omega)
        rotation_matrix = SetToRotate(old_q)

        torque_local = rotation_matrix.transpose() @ torque
        omega_local = rotation_matrix.transpose() @ old_omega

        temp_alpha0, temp_alpha = vec3f(0, 0, 0), old_alpha
        while SquaredLength(temp_alpha0, temp_alpha) > 1e-5:
            temp_alpha0 = temp_alpha
            temp_omega = omega_local + 0.5 * temp_alpha * dt[None]
            temp_alpha = vec3f(torque_local[0] + temp_omega[1] * temp_omega[2] * (i[1] - i[2]) * i[0],
                               torque_local[1] + temp_omega[2] * temp_omega[0] * (i[2] - i[0]) * i[1],
                               torque_local[2] + temp_omega[0] * temp_omega[1] * (i[0] - i[1]) * i[2])
        omega_local += temp_alpha * dt[None]
        omega = rotation_matrix @ omega_local

        # see Langston et al. (2004) Distinct element modelling of non-spherical frictionless particle flow.
        q = Normalize(UpdateQAccurate(old_q, omega_local, dt))                 

        clump[nclump].angmoment = temp_alpha
        clump[nclump].w = omega
        clump[nclump].q = q
        
        rotation_matrix1 = SetToRotate(q)
        for np in range(pebb_beg, pebb_end + 1):
            old_pebble_pos = particle[np].x
            pebble_pos = pos + rotation_matrix1 @ (rotation_matrix.transpose() @ (old_pebble_pos - old_pos))
            particle[np].v = vel + omega.cross(pebble_pos - pos)
            particle[np].w = omega
            particle[np].x = pebble_pos
            particle[np].verletDisp += pebble_pos - old_pebble_pos

@ti.kernel
def move_walls_euler_(wallNum: int, dt: ti.template(), wall: ti.template()):
    # ti.block_local(dt)
    for nw in range(wallNum):
        dx = wall[nw].v * dt[None]
        wall[nw]._move(dx)

@ti.kernel
def get_contact_stiffness(max_material_num: int, particleNum: int, particle: ti.template(), wall: ti.template(), surfaceProps: ti.template(), cplist: ti.template(), particle_wall: ti.template()):
    total_contact_num = particle_wall[particleNum]
    for nc in range(total_contact_num):
        end1, end2 = cplist[nc].endID1, cplist[nc].endID2
        matID1, matID2 = particle[end1].materialID, wall[end2].materialID
        materialID = PairingMapping(matID1, matID2, max_material_num)
        if Squared(cplist[nc].cnforce) > 0.:
            equivalent_stiffness = surfaceProps[materialID]._get_equivalent_stiffness(end1, end2, particle, wall)
            wall[end2]._update_contact_stiffness(equivalent_stiffness)
            wall[end2]._update_contact_interaction(-(cplist[nc].cnforce + cplist[nc].csforce))

@ti.kernel
def get_gain(dt: ti.template(), servoNum: int, servo: ti.template(), wall: ti.template()):
    ti.loop_config(parallelize=16, block_dim=16)
    for nservo in range(servoNum):
        if int(servo[nservo].active) == 1:
            servo[nservo].calculate_gains(dt, wall)

@ti.kernel
def servo(servoNum: int, wall: ti.template(), servo: ti.template()):
    ti.loop_config(parallelize=16, block_dim=16)
    for nservo in range(servoNum):
        if int(servo[nservo].active) == 1:
            velocity = servo[nservo].calculate_velocity(wall)
            for nwall in range(servo[nservo].startIndex, servo[nservo].endIndex):
                wall[nwall].v = velocity

@ti.func
def engine_update_box_size(servo, wall):
    down_wall_position = servo[0].get_geometry_center(wall)
    up_wall_position = servo[1].get_geometry_center(wall)
    left_wall_position = servo[2].get_geometry_center(wall)
    right_wall_position = servo[3].get_geometry_center(wall)
    front_wall_position = servo[4].get_geometry_center(wall)
    back_wall_position = servo[5].get_geometry_center(wall)

    width = right_wall_position[0] - left_wall_position[0]
    depth = back_wall_position[1] - front_wall_position[1]
    height = up_wall_position[2] - down_wall_position[2]

    return width, depth, height

@ti.kernel
def engine_sole_consol_ss(dt: ti.template(), servo: ti.template(), wall: ti.template()):
    width, depth, height = engine_update_box_size(servo, wall)
    
    fdown = -servo[0].get_geometry_force(wall)[2] 
    fup = servo[1].get_geometry_force(wall)[2] 
    fleft = -servo[2].get_geometry_force(wall)[0] 
    fright = servo[3].get_geometry_force(wall)[0] 
    ffront = -servo[4].get_geometry_force(wall)[1] 
    fback = servo[5].get_geometry_force(wall)[1] 
    
    servo[0].update_area(width * depth)
    servo[1].update_area(width * depth)
    servo[2].update_area(height * depth)
    servo[3].update_area(height * depth)
    servo[4].update_area(width * height)
    servo[5].update_area(width * height)

    servo[0].update_current_force(fdown)
    servo[1].update_current_force(fup)
    servo[2].update_current_force(fleft)
    servo[3].update_current_force(fright)
    servo[4].update_current_force(ffront)
    servo[5].update_current_force(fback)

    down_stiffness = servo[0].get_geometry_stiffness(wall)
    up_stiffness = servo[1].get_geometry_stiffness(wall)
    left_stiffness = servo[2].get_geometry_stiffness(wall)
    right_stiffness = servo[3].get_geometry_stiffness(wall)
    front_stiffness = servo[4].get_geometry_stiffness(wall)
    back_stiffness = servo[5].get_geometry_stiffness(wall)

    xstiffness = 0.5 * (left_stiffness + right_stiffness)
    ystiffness = 0.5 * (front_stiffness + back_stiffness)
    zstiffness = 0.5 * (down_stiffness + up_stiffness)
    
    servo[0].calculate_sole_gains(dt, zstiffness)
    servo[1].calculate_sole_gains(dt, zstiffness)
    servo[2].calculate_sole_gains(dt, xstiffness)
    servo[3].calculate_sole_gains(dt, xstiffness)
    servo[4].calculate_sole_gains(dt, ystiffness)
    servo[5].calculate_sole_gains(dt, ystiffness)

@ti.kernel
def engine_consol_ss(dt: ti.template(), servo: ti.template(), wall: ti.template()):
    width, depth, height = engine_update_box_size(servo, wall)
    
    zforce = 0.5 * (servo[1].get_geometry_force(wall)[2] - servo[0].get_geometry_force(wall)[2]) 
    xforce = 0.5 * (servo[3].get_geometry_force(wall)[0] - servo[2].get_geometry_force(wall)[0]) 
    yforce = 0.5 * (servo[6].get_geometry_force(wall)[1] - servo[5].get_geometry_force(wall)[1]) 

    servo[0].update_area(width * depth)
    servo[1].update_area(width * depth)
    servo[2].update_area(height * depth)
    servo[3].update_area(height * depth)
    servo[4].update_area(width * height)
    servo[5].update_area(width * height)

    servo[0].update_current_force(zforce)
    servo[1].update_current_force(zforce)
    servo[2].update_current_force(xforce)
    servo[3].update_current_force(xforce)
    servo[4].update_current_force(yforce)
    servo[5].update_current_force(yforce)

    servo[0].calculate_gains(dt, wall)
    servo[1].calculate_gains(dt, wall)
    servo[2].calculate_gains(dt, wall)
    servo[3].calculate_gains(dt, wall)
    servo[4].calculate_gains(dt, wall)
    servo[5].calculate_gains(dt, wall)

@ti.kernel
def engine_sole_shear_ss(dt: ti.template(), servo: ti.template(), wall: ti.template()):
    width, depth, height = engine_update_box_size(servo, wall)
    
    fleft = -servo[2].get_geometry_force(wall)[0] 
    fright = servo[3].get_geometry_force(wall)[0] 
    ffront = -servo[4].get_geometry_force(wall)[1] 
    fback = servo[5].get_geometry_force(wall)[1] 
    
    servo[2].update_area(height * depth)
    servo[3].update_area(height * depth)
    servo[4].update_area(width * height)
    servo[5].update_area(width * height)

    servo[2].update_current_force(fleft)
    servo[3].update_current_force(fright)
    servo[4].update_current_force(ffront)
    servo[5].update_current_force(fback)

    left_stiffness = servo[2].get_geometry_stiffness(wall)
    right_stiffness = servo[3].get_geometry_stiffness(wall)
    front_stiffness = servo[4].get_geometry_stiffness(wall)
    back_stiffness = servo[5].get_geometry_stiffness(wall)

    xstiffness = 0.5 * (left_stiffness + right_stiffness)
    ystiffness = 0.5 * (front_stiffness + back_stiffness)
    
    servo[2].calculate_sole_gains(dt, xstiffness)
    servo[3].calculate_sole_gains(dt, xstiffness)
    servo[4].calculate_sole_gains(dt, ystiffness)
    servo[5].calculate_sole_gains(dt, ystiffness)

@ti.kernel
def engine_shear_ss(dt: ti.template(), servo: ti.template(), wall: ti.template()):
    width, depth, height = engine_update_box_size(servo, wall)
    
    xforce = 0.5 * (servo[3].get_geometry_force(wall)[0] - servo[2].get_geometry_force(wall)[0]) 
    yforce = 0.5 * (servo[6].get_geometry_force(wall)[1] - servo[5].get_geometry_force(wall)[1]) 

    servo[2].update_area(height * depth)
    servo[3].update_area(height * depth)
    servo[4].update_area(width * height)
    servo[5].update_area(width * height)

    servo[2].update_current_force(xforce)
    servo[3].update_current_force(xforce)
    servo[4].update_current_force(yforce)
    servo[5].update_current_force(yforce)

    servo[2].calculate_gains(dt, wall)
    servo[3].calculate_gains(dt, wall)
    servo[4].calculate_gains(dt, wall)
    servo[5].calculate_gains(dt, wall)

@ti.kernel
def conso(dt: ti.template(), servo: ti.template(), wall: ti.template()):
    ti.loop_config(parallelize=16, block_dim=16)
    for nservo in range(6):
        velocity = servo[nservo].calculate_velocity(wall)
        servo[nservo].move(velocity * dt[None], wall)

@ti.kernel
def drained(time: float, start_time: float, current_time: float, final_time: float, velocity: float, dt: ti.template(), servo: ti.template(), wall: ti.template()):
    vel = velocity
    if current_time < time:
        vel = ((current_time - start_time) / (final_time - start_time)) * velocity
    
    delta = vel * dt[None]
    servo[0].move(vec3f([0., 0., 0.5 * delta]), wall)
    servo[1].move(-vec3f([0., 0., 0.5 * delta]), wall)

    ti.loop_config(parallelize=16, block_dim=16)
    for nservo in range(2, 6):
        velocity = servo[nservo].calculate_velocity(wall)
        servo[nservo].move(velocity * dt[None], wall)

@ti.kernel
def undrained_shear(initial_volume: float, time: float, start_time: float, current_time: float, final_time: float, velocity: float, dt: ti.template(), servo: ti.template(), wall: ti.template()):
    width, depth, height = engine_update_box_size(servo, wall)
    vel = velocity
    if current_time < time:
        vel = ((current_time - start_time) / (final_time - start_time)) * velocity

    delta = vel * dt[None]
    servo[0].move(vec3f(0., 0., 0.5 * delta), wall)
    servo[1].move(-vec3f(0., 0., 0.5 * delta), wall)
    
    S = initial_volume / height
    delta_d = 0.5 * (S / width - depth)
    delta_w = S / (depth + delta_d) - width

    servo[2].move(vec3f(0.5 * delta_w, 0, 0), wall)
    servo[3].move(vec3f(-0.5 * delta_w, 0, 0), wall)
    servo[4].move(vec3f(0, 0.5 * delta_d, 0), wall)
    servo[5].move(vec3f(0, -0.5 * delta_d, 0), wall)

@ti.kernel
def calculate_total_contact_force(particleNum: int, object_object: ti.template(), cplist: ti.template()):
    contactF = 0.
    count = 0
    total_contact_num = object_object[particleNum]
    for nc in range(total_contact_num):
        if Squared(cplist[nc].cnforce) > 0.:
            contactF += (cplist[nc].cnforce + cplist[nc].csforce).norm()
            count += 1
    return contactF, count

@ti.kernel
def calculate_sphere_total_unbalance_force(gravity: ti.types.vector(3, float), sphereNum: int, sphere: ti.template(), particle: ti.template()):
    currF = 0.
    for nsphere in range(sphereNum):
        np = sphere[nsphere].sphereIndex
        currF += (particle[np].contact_force + particle[np].m * gravity).norm()
    return currF

@ti.kernel
def calculate_clump_total_unbalance_force(gravity: ti.types.vector(3, float), clumpNum: int, clump: ti.template(), particle: ti.template()):
    currF = 0.
    for nclump in range(clumpNum):
        contact_force = ZEROVEC3f
        for np in range(clump[nclump].startIndex, clump[nclump].endIndex):
            contact_force += particle[np].contact_force
        currF += (contact_force + clump[nclump].m * gravity).norm()
    return currF

@ti.kernel
def calculate_sphere_maximum_unbalance_force(gravity: ti.types.vector(3, float), sphereNum: int, sphere: ti.template(), particle: ti.template()):
    currF = 0.
    for nsphere in range(sphereNum):
        np = sphere[nsphere].sphereIndex
        tempF = (particle[np].contact_force + particle[np].m * gravity).norm()
        if tempF > currF:
            currF = tempF
    return currF

@ti.func
def calculate_clump_maximum_unbalance_force(gravity: ti.types.vector(3, float), clumpNum: int, clump: ti.template(), particle: ti.template()):
    currF = 0.
    for nclump in range(clumpNum):
        contact_force = ZEROVEC3f
        for np in range(clump[nclump].startIndex, clump[nclump].endIndex):
            contact_force += particle[np].contact_force
        tempF = (contact_force + clump[nclump].m * gravity).norm()
        if tempF > currF:
            currF = tempF
    return currF
