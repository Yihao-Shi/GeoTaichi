import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(dim=2, arch='gpu', kernel_profiler=False, debug=False)

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([275.0, 21.75]),
                      is_2DAxisy=False,
                      background_damping=0.1,
                      gravity=ti.Vector([0., -10.0]),
                      alphaPIC=0.002,
                      mapping="USF",
                      shape_function="GIMP",
                      #velocity_projection='Affine',
                      random_field=False,
                      stress_integration="ReturnMapping",
                      solver_type="Explicit",
                      )

mpm.set_solver(solver={
    "Timestep": 4e-4,
    "SimulationTime": 56,
    "SaveInterval": 0.5,
    "SavePath": 'SainteMonique'
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
                     "SoftenParameter": 5,
                     "MaterialID": 1,
                     "Density": 1800.,
                     "YoungModulus": 10000000.0,
                     "PoissonRatio": 0.45,
                     "Cohesion": 40000.0,
                     "Friction": 0.0,
                     "k0": 0.5
                 })

mpm.add_material(model="DruckerPrager",
                 material={
                     "SoftType": "Exponential",
                     "dpType": "MiddleCircumscribed",
                     "alpha":       0.95,
                     "beta":       15,
                     "MaterialID": 2,
                     "Density": 1700.,
                     "YoungModulus": 11745000.0,
                     "PoissonRatio": 0.45,
                     "Cohesion": 40500.0,
                     "Friction": 0.0,
                     "Dilation": 0.0,
                     "ResidualCohesion": 1500.0,
                     "ResidualFriction": 0.0,
                     "ResidualDilation": 0.0,
                     "PlasticDevStrain": 0.0,
                     "ResidualPlasticDevStrain": 1.8,
                     "k0": 0.5
                 })

mpm.add_material(model="DruckerPrager",
                 material={
                     "SoftType": "Linear",
                     "dpType": "MiddleCircumscribed",
                     "alpha":       0.95,
                      "beta":       12,
                     "MaterialID": 3,
                     "Density": 1700.,
                     "YoungModulus": 11745000.0,
                     "PoissonRatio": 0.45,
                     "Cohesion": 55000.0,
                     "Friction": 0.0,
                     "Dilation": 0.0,
                     "ResidualCohesion": 2000.0,
                     "ResidualFriction": 0.0,
                     "ResidualDilation": 0.,
                     "PlasticDevStrain": 0.0,
                     "ResidualPlasticDevStrain": 1.8,
                     "k0": 0.5
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
    # LM段: 水平线段，y=21.75
    mask1 = (0 <= x) & (x <= 135)
    terrain_y[mask1] = 21.75

    # MF段: 从M(135,21.75)到F(141,19.75)
    mask2 = (135 < x) & (x <= 141)
    # 斜率k = (19.75-21.75)/(141-135) = -2/6 = -1/3
    terrain_y[mask2] = 21.75 - (1/3) * (x[mask2] - 135)

    # FG段: 从F(141,19.75)到G(150,18.5)
    mask3 = (141 < x) & (x <= 150)
    # 斜率k = (18.5-19.75)/(150-141) = -1.25/9 ≈ -0.1389
    terrain_y[mask3] = 19.75 - 0.1389 * (x[mask3] - 141)

    # GH段: 从G(150,18.5)到H(183,5.0)
    mask4 = (150 < x) & (x <= 183)
    # 斜率k = (5.0-18.5)/(183-150) = -13.5/33 ≈ -0.4091
    terrain_y[mask4] = 18.5 - 0.4091 * (x[mask4] - 150)

    # HI段: 水平线段，y=5.0
    mask5 = (183 < x) & (x <= 218)
    terrain_y[mask5] = 5.0

    # IJ段: 从I(218,5.0)到J(250,16.0)
    mask6 = (218 < x) & (x <= 250)
    # 斜率k = (16.0-5.0)/(250-218) = 11/32 = 0.34375
    terrain_y[mask6] = 5.0 + 0.34375 * (x[mask6] - 218)

    # JK段: 水平线段，y=16.0
    mask7 = (250 < x) & (x <= 275)
    terrain_y[mask7] = 16.0

    # 竖向距离
    return terrain_y - y
    #return np.zeros_like(points[:,2])


"""
B1层
"""
def region_func1(new_position, new_radius=0.0):
    """
    输入: new_position = (x, y)
    输出: 点是否在线段CD下方 (True/False)
    """
    x, y = new_position
    temp = False

    if 0.0 <= x <= 275.0:
        # CD: y = 5.0
        terrain_y = 5.0
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
        "BoundingBoxSize": ti.Vector([275., 21.75]),
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
S1层
"""
def region_func2(new_position, new_radius=0.0):
    """
    输入: new_position = (x, y)
    输出: 点是否在分段函数下方 (True/False)
    """
    x, y = new_position
    temp = False

    # 左侧区域：EF、FG、GH
    if 0 <= x <= 183 and 5 < y < 19.75:
        if 0 <= x <= 141:
            # EF: y = 19.75 (水平线段)
            terrain_y = 19.75
            temp = (y <= terrain_y)
        elif 141 < x <= 150:
            # FG: 从F(141,19.75)到G(150,18.5)
            # 斜率 = (18.5-19.75)/(150-141) = -1.25/9 ≈ -0.1389
            terrain_y = 19.75 - 0.1389 * (x - 141)
            temp = (y <= terrain_y)
        elif 150 < x <= 183:
            # GH: 从G(150,18.5)到H(183,5.0)
            # 斜率 = (5.0-18.5)/(183-150) = -13.5/33 ≈ -0.4091
            terrain_y = 18.5 - 0.4091 * (x - 150)
            temp = (y <= terrain_y)

    # 右侧区域：IJ、JK
    elif 218 <= x <= 275 and 5 < y < 16:
        if 218 <= x <= 250:
            # IJ: 从I(218,5.0)到J(250,16.0)
            # 斜率 = (16.0-5.0)/(250-218) = 11/32 = 0.34375
            terrain_y = 5.0 + 0.34375 * (x - 218)
            temp = (y <= terrain_y)
        elif 250 < x <= 275:
            # JK: y = 16.0 (水平线段)
            terrain_y = 16.0
            temp = (y <= terrain_y)

    return temp


def volume_func():
    # return the area of our target region
    return 100.


mpm.add_region(region=[
    {
        "Name": "region2",
        "Type": "UserDefined",
        "BoundingBoxPoint": ti.Vector([0., 0.]),
        "BoundingBoxSize": ti.Vector([275., 21.75]),
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
c1层
"""
def region_func3(new_position, new_radius=0.0):
    """
    输入: new_position = (x, y)
    输出: 点是否位于 LM, MF 定义的分段曲线下方 (True/False)
    约束: y ∈ (19.75, 21.75), x ≤ 141
    """
    x, y = new_position
    temp = False

    if 0 <= x <= 135:
        # LM 段: y = 21.75
        terrain_y = 21.75
        temp = (y <= terrain_y and 19.75 < y < 21.75)

    elif 135 < x <= 141:
        # MF 段: 从M(135, 21.75)到F(141, 19.75)
        # 斜率k = (19.75 - 21.75)/(141 - 135) = -2/6 = -1/3
        terrain_y = 21.75 - (1/3) * (x - 135)
        temp = (y <= terrain_y and 19.75 < y < 21.75)

    return temp


def volume_func():
    # return the area of our target region
    return 100.


mpm.add_region(region=[
    {
        "Name": "region3",
        "Type": "UserDefined",
        "BoundingBoxPoint": ti.Vector([0., 0.]),
        "BoundingBoxSize": ti.Vector([275., 21.75]),
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
        "EndPoint": [275.0, 0.]
    },

    {
        "BoundaryType": "VelocityConstraint",
        "Velocity": [0., None],
        "StartPoint": [0., 0.],
        "EndPoint": [0., 21.75]
    },

    {
        "BoundaryType": "VelocityConstraint",
        "Velocity": [0., None],
        "StartPoint": [275., 0.],
        "EndPoint": [275., 21.75]
    },

])

mpm.select_save_data()

mpm.run(gravity_field=get_gravity)

# ti.profiler.print_kernel_profiler_info()

mpm.postprocessing()
