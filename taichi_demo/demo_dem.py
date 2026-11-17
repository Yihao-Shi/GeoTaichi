import taichi as ti
import numpy as np
import math

end_frame = 1000

# Particle size
num_particles = 72000
particle_size = 0.2
radius = particle_size * 0.5
density = 3000 # kg/m3
mass = 4/3*math.pi*ti.pow(radius, 3)*density
inv_mass = 1/mass
poisson = 0.3
youngs = 1e8
I_moment = 2/5*mass*ti.pow(radius,2)

# Grid size, cell size
cell_size = particle_size
size = 64
grid_size = [size, size, size]
per_grid = 2 # max number of particles in a cell
max_contact = 26 # max contacts per particle

# contact overlap tolerances
particle_tolerance = -1e-5
ground_tolerance = -1e-5

# interaction parameters
equiv_radius = radius/2
equiv_youngs = youngs/(2*(1-ti.pow(poisson,2)))
eff_mass = mass/2
k_G = youngs/(4*(2-poisson)*(1+poisson)) # Effective shear modulus
coeff_restitution = 0.3
k_f = 0.8 # Normal friction
k_roll = 0.8 # Rolling friction

dt = 5e-5 # timestep
#print("Timestep: ", dt, equiv_youngs, equiv_radius)

# Calculate contact forces
@ti.func
def contact_forces(n, v_rel, n_norm):
    vn = n.dot(v_rel)
    kn_calc = 0.9375*eff_mass*ti.pow(vn,2)/(ti.sqrt(equiv_radius)*equiv_youngs)
    kn_spring = 1.06667 * ti.sqrt(equiv_radius)*equiv_youngs*ti.pow(kn_calc,0.2)  # spring constant
    if vn > 0.0:
        kn_spring = 0.0
    k_damping = ti.sqrt((4*eff_mass*kn_spring)/(1 + ti.pow(math.pi/ti.log(coeff_restitution),2)))
    jn = n_norm*kn_spring #spring
    jd = vn*k_damping #dashpot

    # normal contact forces
    fn = jn+jd

    # tangential velocity
    vt = v_rel - n*vn
    vt_norm = vt.norm()
    ft = ti.types.vector(3, float)((0,0,0))
    f_t = ti.types.vector(3, float)((0,0,0))

    if vt_norm > 0.0:
        overlap_t = vt*dt # tangential overlap

        # tangential spring and damping coefficients
        kn_calc_t = 0.9375*eff_mass*ti.pow(vt_norm,2)/(ti.sqrt(equiv_radius)*equiv_youngs)
        kn_spring_t = 1.06667 * ti.sqrt(equiv_radius)*equiv_youngs*ti.pow(kn_calc_t,0.2)  # spring constant
        k_damping_t = ti.sqrt((4*eff_mass*kn_spring_t)/(1 + ti.pow(math.pi/ti.log(coeff_restitution),2)))
        f_damping_t = k_damping_t*vt
        f_t = kn_spring_t*overlap_t + f_damping_t

        # Coulomb limit condition
        coulomb_thres = k_f*fn
        if(f_t.norm() > coulomb_thres):
            overlap_t = -(coulomb_thres * f_t.normalized() - f_damping_t) / kn_spring_t
            ft = (kn_spring_t * overlap_t) + f_damping_t
        else:
            ft = f_t
    forces_normal = n*fn
    forces = -forces_normal - ft
    return forces, forces_normal, ft

# Calculate tangential torque
@ti.func
def calc_torque(dist_normal, radius, F_contact):
    torque = (radius*dist_normal).cross(F_contact)
    return torque

# Calculate rolling friction torque
@ti.func
def calc_rolling_torque(k_roll, eff_radius, F_n, rel_omega, v_omega):
    torque = k_roll*eff_radius*F_n.norm()*rel_omega*v_omega
    return -torque

# Hash grid function
@ti.func
def hashFunction(particlePos):
    factor = 1 / cell_size
    I = (particlePos * factor).cast(ti.int32)
    return I

# Build hash grid
@ti.kernel
def build_hash_grid(particles_local : ti.template(), hash_grid_local: ti.template()):
    for i in particles_local:
        I = hashFunction(particles_local[i].p)
        num_in_cell = hash_grid_local[I].num
        for num in ti.static(range(per_grid)):
            if num == num_in_cell:
                hash_grid_local[I].indices[num] = i
                hash_grid_local[I].num += 1

# Set gravity
def set_gravity(g):
    assert isinstance(g, (tuple, list))
    assert len(g) == 3
    gravity[None] = g

# Function to create a bunch of particles in a grid
def particle_grid(dim_x, dim_y, dim_z, lower, radius, rand):
    points = np.meshgrid(np.linspace(0, dim_x, dim_x), np.linspace(0, dim_y, dim_y), np.linspace(0, dim_z, dim_z))
    points_t = np.array((points[0], points[1], points[2])).T*radius*2.0 + np.array(lower)
    points_t = points_t + np.random.rand(*points_t.shape)*radius*rand
    
    return points_t.reshape((-1, 3))

# Loop through the particles and add forces
@ti.kernel
def calculateForces(particlesLocal : ti.template(), hash_grid_local: ti.template(), frame: ti.i32):
    for I in particlesLocal:
        base = hashFunction(particlesLocal[I].p) # target cell
        num_in_cell = hash_grid_local[base].num
        # Reset forces to zero
        particlesLocal[I].forces = ti.types.vector(3, ti.f32)([0,0,0])
        particlesLocal[I].torque = ti.types.vector(3, ti.f32)([0,0,0])
        localForce = ti.types.vector(3, ti.f32)([0,0,0])
        totalContact = 0  # keep track of total contact
        # Loop through the particles inside
        if num_in_cell > 1:
            for num in ti.static(range(per_grid)):
                neighParticle = hash_grid_local[base].indices[num]
                if I != neighParticle and neighParticle != -1 and totalContact <= max_contact:
                    dist = particlesLocal[I].p - particlesLocal[neighParticle].p
                    contact_dist = dist.norm() - particle_size

                    if contact_dist < particle_tolerance:
                        rot1 = particlesLocal[I].omega
                        rot2 = particlesLocal[neighParticle].omega
                        # relative velocity
                        v = particlesLocal[I].v - particlesLocal[neighParticle].v + (radius*rot1 + radius*rot2).cross(dist.normalized())
                        localForce, F_n, F_t = contact_forces(dist/dist.norm(), v, contact_dist)
                        particlesLocal[I].forces = particlesLocal[I].forces + localForce
                        totalContact += 1

                        torque = calc_torque(dist/dist.norm(), radius, -F_t)
                        particlesLocal[I].torque = particlesLocal[I].torque + torque
                        if num > 2:
                            print("local")
                         # rolling friction torque
                        rel_omega = rot1-rot2
                        if rel_omega.norm() > 0.0:
                            v_omega = rot1.cross(radius*dist.normalized()) - rot2.cross(-radius*dist.normalized())
                            torque_friction = calc_rolling_torque(k_roll, equiv_radius, -F_n, rel_omega/rel_omega.norm(), v_omega.norm())
                            total_torque = torque + torque_friction
                            particlesLocal[I].torque = particlesLocal[I].torque + total_torque
    
        # Continue neighbour search
        for offset in ti.grouped(ti.ndrange(*((3, ) * 3))):
            # Particle contacts
            neigh = base + offset - 1 # neighbouring cells
            num_in_cell = hash_grid_local[neigh].num
            if num_in_cell > 0:
                for num in ti.static(range(per_grid)):
                    neighParticle = hash_grid_local[neigh].indices[num]
                    if I != neighParticle and neighParticle != -1 and totalContact <= max_contact:
                        dist = particlesLocal[I].p - particlesLocal[neighParticle].p
                        contact_dist =  dist.norm() - particle_size
                        
                        if contact_dist < particle_tolerance:
                            rot1 = particlesLocal[I].omega
                            rot2 = particlesLocal[neighParticle].omega
                            # relative velocity
                            v = particlesLocal[I].v - particlesLocal[neighParticle].v + (radius*rot1 + radius*rot2).cross(dist.normalized())
                            localForce, F_n, F_t = contact_forces(dist/dist.norm(), v, contact_dist)
                            particlesLocal[I].forces = particlesLocal[I].forces + localForce
                            totalContact += 1
                            # tangential torque
                            torque = calc_torque(dist/dist.norm(), radius, -F_t)
                            # rolling friction torque
                            rel_omega = rot1-rot2
                            if rel_omega.norm() > 0.0:
                                v_omega = rot1.cross(radius*dist.normalized()) - rot2.cross(-radius*dist.normalized())
                                torque_friction = calc_rolling_torque(k_roll, equiv_radius, -F_n, rel_omega/rel_omega.norm(), v_omega.norm())
                                total_torque = torque + torque_friction
                                particlesLocal[I].torque = particlesLocal[I].torque + total_torque

        # Ground contact
        ground_dist = particlesLocal[I].p - ground[None]
        ground_contact_dist = ground_dist.dot(ground[None])
        rot1 = particlesLocal[I].omega

        if(ground_contact_dist < ground_tolerance):
            v = particlesLocal[I].v + (radius*rot1).cross(ground[None].normalized())
            localForce, F_n, F_t = contact_forces(ground[None].normalized(), particlesLocal[I].v, ground_contact_dist)
            particlesLocal[I].forces = particlesLocal[I].forces + localForce
            torque = calc_torque(ground_dist/ground_dist.norm(), radius, -F_t)

            rel_omega = rot1
            if rel_omega.norm() > 0.0:
                v_omega = rot1.cross(-radius*ground_dist.normalized())
                torque_friction = calc_rolling_torque(k_roll, equiv_radius, -F_n, rel_omega/rel_omega.norm(), v_omega.norm())
                total_torque = torque + torque_friction
                particlesLocal[I].torque = particlesLocal[I].torque + total_torque

# Simple Euler integration
@ti.kernel
def integrate(particlesLocal: ti.template(), dt: ti.f32):
    # Done calculating forces, integrate over time
    for I in particlesLocal:
        # update positions
        acceleration = (particlesLocal[I].forces)*inv_mass + gravity[None]
        vel_new = particlesLocal[I].v + acceleration*dt
        particlesLocal[I].p = particlesLocal[I].p + vel_new*dt
        particlesLocal[I].v = vel_new

        # update rotation
        particlesLocal[I].omega = particlesLocal[I].omega + particlesLocal[I].torque*dt/I_moment

##############################################################################
# Initialise taichi
ti.init(arch=ti.gpu, device_memory_GB=4.0)

# gravity
gravity = ti.Vector.field(3, dtype=ti.f32, shape=())
set_gravity((0, -9.81, 0))

# a particle structure containing all the information
particleData = ti.Struct.field({"p": ti.types.vector(3, ti.f32),
                                "v": ti.types.vector(3, ti.f32),
                                "forces": ti.types.vector(3, ti.f32),
                                "omega": ti.types.vector(3,ti.f32),
                                "torque": ti.types.vector(3,ti.f32)}, shape=(num_particles,))

# create a grid of particles
points = particle_grid(20, 60, 20, (0.6, 0.4, 0.6), particle_size*0.5, 0.2)

# initialise
particleData.p.from_numpy(points)
particleData.v.fill((0,0,0))
particleData.forces.fill((0,0,0))
particleData.omega.fill((0,0,0))
particleData.torque.fill((0,0,0))

# setup ground
ground = ti.Vector.field(3, dtype=ti.f32, shape=())
ground[None] = [0, 0.3, 0]

# initialise hash grid
hash_grid = ti.Struct.field({"num": ti.i32, 
                            "indices": ti.types.vector(8, ti.i32)},
                            shape=(grid_size[0], grid_size[1], grid_size[2]))

# GUI - Taichi built-in GGUI
res = (512, 512)
window = ti.ui.Window("DEM 3D", res)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(3, 6, 17)
camera.lookat(3, 4, 1.5)
camera.fov(55)

# Render the scene
def render():
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    #cene.ambient_light((0, 0, 0))
    scene.particles(particleData.p, radius=radius, color=(0.5, 0, 0))

    scene.point_light(pos=(5, 1, 3), color=(1, 1, 1))
    scene.point_light(pos=(5, 1, -3), color=(1, 1, 1))
    scene.point_light(pos=(0, 5, 6), color=(1, 1, 1))
    scene.point_light(pos=(-5, 2, 6), color=(1, 1, 1))
    scene.point_light(pos=(-5, 2, -3), color=(1, 1, 1))
    canvas.scene(scene)
    window.show()

# Main loop
while window.running:
    # step through simulation    
    for frame in range(end_frame):
        hash_grid.indices.fill(-1)
        hash_grid.num.fill(0)
        build_hash_grid(particleData, hash_grid)
        calculateForces(particleData, hash_grid, frame)
        integrate(particleData, dt)
        render()

