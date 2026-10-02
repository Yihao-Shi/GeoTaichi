import taichi as ti

from src.utils.TypeDefination import vec3f


# ========================== Constitutive Model Utility ========================== #
@ti.func
def calculate_vorticity_increment2D(velocity_gradient, dt):
    return calculate_vorticity_rate2D(velocity_gradient) * dt[None]

@ti.func
def calculate_vorticity_increment(velocity_gradient, dt):
    return calculate_vorticity_rate(velocity_gradient) * dt[None]

@ti.func
def calculate_vorticity_rate2D(velocity_gradient):
    return 0.5 * vec3f(velocity_gradient[1, 0] - velocity_gradient[0, 1], 0., 0.)

@ti.func
def calculate_vorticity_rate(velocity_gradient):
    return 0.5 * vec3f(velocity_gradient[1, 0] - velocity_gradient[0, 1],
                       velocity_gradient[2, 1] - velocity_gradient[1, 2],
                       velocity_gradient[0, 2] - velocity_gradient[2, 0])

@ti.func
def get_angular_velocity(velocity_gradient):
    spin_tensor = calculate_vorticity_rate(velocity_gradient)
    return vec3f(spin_tensor[1], spin_tensor[2], spin_tensor[0])

@ti.func
def SphericalTensor(tensor):
    return (tensor[0] + tensor[1] + tensor[2]) / 3.

# ========================== Constitutive Model Kernel ========================== #
@ti.kernel
def kernel_initial_state_variables(to_beg: int, to_end: int, particle: ti.template(), matProps: ti.template(), stateVars: ti.template()):
    for np in range(to_beg, to_end):
        matProps._initialize_vars(np, particle, stateVars)

@ti.kernel
def compute_stiffness_matrix(stiffness_matrix: ti.template(), start_index: int, end_index: int, particle: ti.template(), materialID: ti.template(), matProps: ti.template(), stateVars: ti.template()):
    for i in range(start_index, end_index):
        np = materialID[i]
        stress = particle[np].stress
        stiffness_matrix[np] = matProps.compute_stiffness_tensor(np, stress, stateVars)

@ti.kernel
def compute_elastic_stiffness_matrix(stiffness_matrix: ti.template(), start_index: int, end_index: int, particle: ti.template(), materialID: ti.template(), matProps: ti.template(), stateVars: ti.template()):
    for i in range(start_index, end_index):
        np = materialID[i]
        stress = particle[np].stress
        stiffness_matrix[np] = matProps.compute_elastic_tensor(np, stress, stateVars)
