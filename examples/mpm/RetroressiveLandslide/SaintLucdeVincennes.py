import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(dim=2, arch='gpu', kernel_profiler=False, debug=False)

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([320.0, 30.0]),
                      is_2DAxisy=False,
                      background_damping=0.1,
                      gravity=ti.Vector([0., -10.0]),
                      alphaPIC=0.00,
                      mapping="USF",
                      shape_function="GIMP",
                      stress_integration="ReturnMapping",
                      solver_type="Explicit",
                      )

mpm.set_solver(solver={
    "Timestep": 2.5e-4,
    "SimulationTime": 26,
    "SaveInterval": 0.5,
    "SavePath": 'SaintLucdeVincennes'
})

mpm.memory_allocate(memory={
    "max_material_number": 1,
    "max_particle_number": 8.12e5,
    "max_constraint_number": {
        "max_velocity_constraint": 83000
    }
})

# mpm.add_material(model="DruckerPrager",
#                 material={
#                     "MaterialID": 1,
#                     "MaterialFile": args.input
#                 })
mpm.add_material(model="DruckerPrager",
                 material={
                     "dpType": "MiddleCircumscribed",
                     "MaterialID": 1,
                     "Density": 1700.,
                     "YoungModulus": 10000000.0,
                     "PoissonRatio": 0.45,
                     "Cohesion": 20000.0,
                     "Friction": 0.0,
                     "k0": 0.8
                 })

mpm.add_material(model="DruckerPrager",
                 material={
                     "SoftType": "Exponential",
                     "dpType": "MiddleCircumscribed",
                     "alpha":       0.945,
                     "beta":       80,
                     "MaterialID": 2,
                     "Density": 1700.,
                     "YoungModulus": 13290000.0,
                     "PoissonRatio": 0.45,
                     "Cohesion": 55000.0,
                     "Friction": 0.0,
                     "ResidualCohesion": 1375.0,
                     "ResidualFriction": 0.0,
                     "ResidualDilation": 0.0,
                     "PlasticDevStrain": 0.001,
                     "ResidualPlasticDevStrain": 1.95,
                     "k0": 0.8
                 })

mpm.add_material(model="DruckerPrager",
                 material={
                     "SoftType": "Exponential",
                     "dpType": "MiddleCircumscribed",
                     "alpha":       0.945,
                     "beta":       2.9,
                     "SoftenParameter": 1.0,
                     "MaterialID": 3,
                     "Density": 1700.,
                     "YoungModulus": 13290000.0,
                     "PoissonRatio": 0.45,
                     "Cohesion": 70000.0,
                     "Friction": 0.0,
                     "ResidualCohesion": 1375.0,
                     "ResidualFriction": 0.0,
                     "ResidualDilation": 0.0,
                     "PlasticDevStrain": 0.001,
                     "ResidualPlasticDevStrain": 1.95,
                     "k0": 0.8
                 })


mpm.add_element(element={
    "ElementType": "Q4N2D",
    "ElementSize": ti.Vector([0.5, 0.5])
})


def get_gravity(points):
    import numpy as np
    x = points[:, 0]
    y = points[:, 1]

    # 初始化 terrain_y
    terrain_y = np.zeros_like(x)

    # 逐段条件赋值
    mask1 = (0 <= x) & (x <= 110)
    terrain_y[mask1] = 22.0

    mask2 = (110 < x) & (x <= 135)
    terrain_y[mask2] = (770. - 2. * x[mask2]) / 25.

    mask3 = (135 < x) & (x <= 185)
    terrain_y[mask3] = (1345. - 7. * x[mask3]) / 20.

    mask4 = (185 < x) & (x <= 200)
    terrain_y[mask4] = (2. * x[mask4] - 355.) / 6.

    mask5 = (200 < x) & (x <= 320)
    terrain_y[mask5] = 7.5

    # 竖向距离
    return terrain_y - y
    #return np.zeros_like(points[:,2])


"""
B3层
"""
def region_func1(new_position, new_radius=0.0):
    """
    输入: new_position = (x, y)
    输出: 点是否在分段函数下方 (True/False)
    """
    x, y = new_position
    temp = False

    if 0 <= x <= 175:
        # CD: y = 5.0
        terrain_y = 5.0
        temp = (y <= terrain_y)

    elif 175 < x <= 185:
        # DE: y = -0.25x + 48.75
        terrain_y = -0.25 * x + 48.75
        temp = (y <= terrain_y and y <= 5.0)

    elif 185 < x <= 200:
        # EF: y = (1/3)x - 59.166...
        terrain_y = (x / 3.0) - 59.1666667
        temp = (y <= terrain_y and y <= 7.5)

    elif 200 < x <= 320:
        # FG: y = 7.5
        terrain_y = 7.5
        temp = (y <= terrain_y)

    return temp


def volume_func():
    # return the area of our target region
    return 100.


mpm.add_region(region=[
    {
        "Name": "region1",
        "Type": "UserDefined",
        "BoundingBoxPoint": ti.Vector([0., 0.]),
        "BoundingBoxSize": ti.Vector([320., 30.]),
        "RegionVolume": volume_func,
        "RegionFunction": region_func1
    }
])
mpm.add_body(body={
    "Template": [
        {
            "RegionName": "region1",
            "nParticlesPerCell": 2,
            "BodyID": 0,
            "MaterialID": 3,
            "InitialVelocity": ti.Vector([0, 0]),
            "FixVelocity": ["Free", "Free"]

        }
    ]
})

"""
S3层
"""
def region_func2(new_position, new_radius=0.0):
    """
    输入: new_position = (x, y)
    输出: 点是否在 HI-ID 分段函数下方 (True/False)
    约束: y ∈ [5, 20], x ≤ 175
    """
    x, y = new_position
    temp = False

    if 0 <= x < 134:
        # HI 段: y = 20
        terrain_y = 20.5
        temp = (y <= terrain_y and 5.0 < y <= 25.0)

    elif 134 < x <= 175:
        # ID 段: y = -0.375x + 70.625
        terrain_y = -0.375 * x + 70.625
        temp = (y <= terrain_y and 5.0 < y <= 25.0)

    return temp


def volume_func():
    # return the area of our target region
    return 100.


mpm.add_region(region=[
    {
        "Name": "region2",
        "Type": "UserDefined",
        "BoundingBoxPoint": ti.Vector([0., 0.]),
        "BoundingBoxSize": ti.Vector([320., 30.]),
        "RegionVolume": volume_func,
        "RegionFunction": region_func2
    }
])
mpm.add_body(body={
    "Template": [
        {
            "RegionName": "region2",
            "nParticlesPerCell": 2,
            "BodyID": 0,
            "MaterialID": 2,
            "InitialVelocity": ti.Vector([0, 0]),
            "FixVelocity": ["Free", "Free"]

        }
    ]
})

"""
c3层
"""
def region_func3(new_position, new_radius=0.0):
    """
    输入: new_position = (x, y)
    输出: 点是否位于 JK, KI 定义的分段曲线下方 (True/False)
    约束: y ∈ (20.5, 22], x ≤ 110
    """
    x, y = new_position
    temp = False

    if 0 <= x <= 110:
        # JK 段: y = 22
        terrain_y = 22.0
        temp = (y <= terrain_y and 20.5 < y <= 22.0)

    elif 110 < x <= 135:
        # KI 段: y = -0.06x + 28.6 (但约束 x ≤ 110，所以不会触发)
        terrain_y = -0.06 * x + 28.6
        temp = (y <= terrain_y and 20.5 < y <= 22.0)

    return temp


def volume_func():
    # return the area of our target region
    return 100.


mpm.add_region(region=[
    {
        "Name": "region3",
        "Type": "UserDefined",
        "BoundingBoxPoint": ti.Vector([0., 0.]),
        "BoundingBoxSize": ti.Vector([320., 30.]),
        "RegionVolume": volume_func,
        "RegionFunction": region_func3
    }
])
mpm.add_body(body={
    "Template": [
        {
            "RegionName": "region3",
            "nParticlesPerCell": 2,
            "BodyID": 0,
            "MaterialID": 1,
            "InitialVelocity": ti.Vector([0, 0]),
            "FixVelocity": ["Free", "Free"]

        }
    ]
})


mpm.add_boundary_condition(boundary=[
    {
        "BoundaryType": "VelocityConstraint",
        "Velocity": [0., 0],
        "StartPoint": [0., 0.],
        "EndPoint": [320.0, 0.]
    },

    {
        "BoundaryType": "VelocityConstraint",
        "Velocity": [0., None],
        "StartPoint": [0., 0.],
        "EndPoint": [0., 30.0]
    },

    {
        "BoundaryType": "VelocityConstraint",
        "Velocity": [0., None],
        "StartPoint": [320., 0.],
        "EndPoint": [320., 30.0]
    },

])

mpm.select_save_data()

mpm.run(gravity_field=get_gravity)

# ti.profiler.print_kernel_profiler_info()

mpm.postprocessing()
