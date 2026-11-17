import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from geotaichi import *

init(device_memory_GB=5.5, debug=False)

vert_pressure = 96000
hori_pressure = 40000

dempm = DEMPM()

dempm.set_configuration(
    domain=[144.0, 46.0, 75.0],
    gravity=[0.0, 0.0, -9.8],
    coupling_scheme="MPDEM",
    particle_interaction=False,
    wall_interaction=True,
    enable_shell=True,
)

dempm.mpm.set_configuration(
    background_damping=0.2,
    #   mode="Lightweight",
    alphaPIC=0.001,
    mapping="USF",
    shape_function="GIMP",
)

dempm.dem.set_configuration(engine="SymplecticEuler", search="LinkedCell")

dempm.set_solver({"Timestep": 1e-5, "SimulationTime": 1, "SaveInterval": 0.02, "SavePath": "1_centrifuge"})

dempm.dem.memory_allocate(
    memory={
        "max_material_number": 1,
        "max_particle_number": 0,
        "max_sphere_number": 0,
        "max_patch_number": 20632,
        "verlet_distance_multiplier": 0.01,
        "body_coordination_number": 0,
        "wall_coordination_number": 0,
        "compaction_ratio": [1.0, 1.0],
    },
    log=True,
)

dempm.mpm.memory_allocate(
    memory={
        "max_material_number": 7,
        "max_particle_number": 508605,
        "verlet_distance_multiplier": 0.1,
        "max_constraint_number": {
            "max_velocity_constraint": 12434,
            "max_particle_traction_constraint": 400615,
            "max_friction_constraint": 10,
        },
    }
)

dempm.memory_allocate(
    memory={"body_coordination_number": 0, "wall_coordination_number": 160, "compaction_ratio": [1.0, 0.15]}
)


dempm.dem.add_attribute(
    materialID=0, attribute={"Density": 26500, "ForceLocalDamping": 0.15, "TorqueLocalDamping": 0.05}
)

dempm.dem.choose_contact_model(particle_particle_contact_model=None, particle_wall_contact_model=None)

dempm.dem.add_wall(
    body={
        "WallType": "Patch",
        "WallID": 0,
        "WallFile": f"{ROOT}/assets/mesh/MPDEM/bucket.stl",
        "Translation": [0.0, 0.0, 0.0],
        "Counterclockwise": None,
        "MaterialID": 0,
        "Visualize": False,
        "ShellOffset": 0.016,
        "Density": 2650,
        "ExternalForce": [0.0, 0.0, 0.0],
        "AppliedPoint": [72, 23, 66.5],
    }
)

dempm.dem.select_save_data(particle=False, wall=True)

dempm.mpm.add_material(
    model="DruckerPrager",
    material={
        "MaterialID": 1,
        "Density": 1173.0,
        "YoungModulus": 3.0e6,
        "poissonRatio": 0.2,  # equivalent=0.4950
        "Cohesion": 12000,
        "Friction": 15,
        "Dilation": 0.0,
    },
)  # 海侧碎石桩

dempm.mpm.add_material(
    model="DruckerPrager",
    material={
        "MaterialID": 2,
        "Density": 1100.0,
        "YoungModulus": 2.5e6,
        "poissonRatio": 0.20,
        "Cohesion": 11000,
        "Friction": 15,
        "Dilation": 0.0,
    },
)  # 陆侧淤泥质粘土

dempm.mpm.add_material(
    model="DruckerPrager",
    material={
        "MaterialID": 3,
        "Density": 1558.0,
        "YoungModulus": 5.0e6,
        "poissonRatio": 0.20,
        "Cohesion": 33000,
        "Friction": 17,
        "Dilation": 0.0,
    },
)  # 粉质粘土

dempm.mpm.add_material(
    model="DruckerPrager",
    material={
        "MaterialID": 4,
        "Density": 1865.0,
        "YoungModulus": 2.0e7,
        "poissonRatio": 0.20,
        "Cohesion": 8000,
        "Friction": 35,
        "Dilation": 0.0,
    },
)  # 砂土

dempm.mpm.add_material(
    model="DruckerPrager",
    material={
        "MaterialID": 5,
        "Density": 1865.0,
        "YoungModulus": 2.0e8,
        "poissonRatio": 0.20,
        "Cohesion": 150000,
        "Friction": 20,
        "Dilation": 0.0,
    },
)  # 回填土

dempm.mpm.add_element(element={"ElementType": "R8N3D", "ElementSize": ti.Vector([2.0, 2.0, 2.0])})


@ti.pyfunc
def common_region(
    pos,
    z_min,
    z_max,
):
    eps = 1e-12
    x, y, z = pos[0], pos[1], pos[2]

    is_inside = 0
    if z_min + eps < z < z_max - eps:
        d_lower2 = (x - 72.0) ** 2 + (y - 11.0) ** 2
        d_upper2 = (x - 72.0) ** 2 + (y - 35.0) ** 2
        d_middle2 = (x - 72.0) ** 2 + (y - 23.0) ** 2

        inside_lower = d_lower2 < 11.0**2 - eps
        inside_upper = d_upper2 < 11.0**2 - eps
        inside_middle = d_middle2 < 9.0**2 - eps

        if inside_lower or inside_upper or inside_middle:
            is_inside = 1
    return is_inside


def region7(pos, rad):
    return common_region(pos, z_min=61.0, z_max=73.0)


def region5(pos, rad=0):
    z_min = 60.0
    z_max = 61.0
    return not common_region(pos, z_min=z_min, z_max=z_max) and z_min < pos[2] < z_max


def region6(pos, rad=0):
    z_min = 60.0
    z_max = 61.0
    return not common_region(pos, z_min=z_min, z_max=z_max) and z_min < pos[2] < z_max


def region8(pos, rad=0):
    return common_region(pos, z_min=72.0, z_max=73.0)


dempm.mpm.add_region(
    region=[
        {
            "Name": "region1",
            "Type": "Rectangle",
            "BoundingBoxPoint": ti.Vector([0.0, 0.0, 0.0]),
            "BoundingBoxSize": ti.Vector([144.0, 46.0, 5.0]),
        },
        {
            "Name": "region2",
            "Type": "Rectangle",
            "BoundingBoxPoint": ti.Vector([0.0, 0.0, 5.0]),
            "BoundingBoxSize": ti.Vector([144.0, 46.0, 26.0]),
        },
        {
            "Name": "region3",
            "Type": "Rectangle",
            "BoundingBoxPoint": ti.Vector([0.0, 0.0, 31.0]),
            "BoundingBoxSize": ti.Vector([72.0, 46.0, 30.0]),
        },
        {
            "Name": "region4",
            "Type": "Rectangle",
            "BoundingBoxPoint": ti.Vector([72.0, 0.0, 31.0]),
            "BoundingBoxSize": ti.Vector([72.0, 46.0, 30.0]),
        },
        {
            "Name": "region5",
            "Type": "UserDefined",
            "BoundingBoxPoint": ti.Vector([0.0, 0.0, 60.0]),
            "BoundingBoxSize": ti.Vector([72.0, 46.0, 1.0]),
            "RegionFunction": region5,
        },
        {
            "Name": "region6",
            "Type": "UserDefined",
            "BoundingBoxPoint": ti.Vector([72.0, 0.0, 60.0]),
            "BoundingBoxSize": ti.Vector([72.0, 46.0, 1.0]),
            "RegionFunction": region6,
        },
        {
            "Name": "region7",
            "Type": "UserDefined",
            "BoundingBoxPoint": ti.Vector([61.0, 0.0, 61.0]),
            "BoundingBoxSize": ti.Vector([22.0, 46.0, 12]),
            "RegionFunction": region7,
        },
        {
            "Name": "region8",
            "Type": "UserDefined",
            "BoundingBoxPoint": ti.Vector([61.0, 0.0, 72.0]),
            "BoundingBoxSize": ti.Vector([22.0, 46.0, 1.0]),
            "RegionFunction": region8,
        },
    ]
)

dempm.mpm.add_body(
    body={
        "Template": [
            {
                "RegionName": "region1",
                "nParticlesPerCell": 2,
                "BodyID": 0,
                "MaterialID": 4,
                # "ParticleStress": {
                #                        "InternalStress": ti.Vector([-vert_pressure, -vert_pressure, -vert_pressure, 0., 0., 0.]),
                #                        "PorePressure": 0.
                #                   },
                "InitialVelocity": ti.Vector([0.0, 0.0, 0.0]),
                "FixVelocity": ["Free", "Free", "Free"],
            },
            {
                "RegionName": "region2",
                "nParticlesPerCell": 2,
                "BodyID": 0,
                "MaterialID": 3,
                # "ParticleStress": {
                #                        "InternalStress": ti.Vector([-vert_pressure, -vert_pressure, -vert_pressure, 0., 0., 0.]),
                #                        "PorePressure": 0.
                #                   },
                "InitialVelocity": ti.Vector([0.0, 0.0, 0.0]),
                "FixVelocity": ["Free", "Free", "Free"],
            },
            {
                "RegionName": "region3",
                "nParticlesPerCell": 2,
                "BodyID": 0,
                "MaterialID": 1,
                # "ParticleStress": {
                #                        "InternalStress": ti.Vector([-vert_pressure, -vert_pressure, -vert_pressure, 0., 0., 0.]),
                #                        "PorePressure": 0.
                #                   },
                # "Traction": [{"Pressure": ti.Vector([0, 0., -vert_pressure]),
                #              "FluidPressure": ti.Vector([0, 0., 0.]),
                #              "RegionName": "region5"}],
                "InitialVelocity": ti.Vector([0.0, 0.0, 0.0]),
                "FixVelocity": ["Free", "Free", "Free"],
            },
            {
                "RegionName": "region4",
                "nParticlesPerCell": 2,
                "BodyID": 0,
                "MaterialID": 2,
                # "ParticleStress": {
                #                        "InternalStress": ti.Vector([-vert_pressure, -vert_pressure, -vert_pressure, 0., 0., 0.]),
                #                        "PorePressure": 0.
                #                  },
                # "Traction": [{"Pressure": ti.Vector([0, 0., -vert_pressure]),
                #              "FluidPressure": ti.Vector([0, 0., 0.]),
                #              "RegionName": "region6"}],
                "InitialVelocity": ti.Vector([0.0, 0.0, 0.0]),
                "FixVelocity": ["Free", "Free", "Free"],
            },
            {
                "RegionName": "region7",
                "nParticlesPerCell": 2,
                "BodyID": 0,
                "MaterialID": 5,
                # "ParticleStress": {
                #                        "InternalStress": ti.Vector([-vert_pressure, -vert_pressure, -vert_pressure, 0., 0., 0.]),
                #                        "PorePressure": 0.
                #                  },
                # "Traction": [{"Pressure": ti.Vector([0, 0., -vert_pressure]),
                #              "FluidPressure": ti.Vector([0, 0., 0.]),
                #              "RegionName": "region8"}],
                "InitialVelocity": ti.Vector([0.0, 0.0, 0.0]),
                "FixVelocity": ["Free", "Free", "Free"],
            },
        ]
    }
)

dempm.mpm.add_boundary_condition(
    boundary=[
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, 0.0, 0.0],
            "StartPoint": [0.0, 0.0, 0.0],
            "EndPoint": [144.0, 46.0, 0.0],
        },
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, None, None],
            "StartPoint": [0.0, 0.0, 0.0],
            "EndPoint": [0.0, 46.0, 73.0],
        },
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [0.0, None, None],
            "StartPoint": [144.0, 0.0, 0.0],
            "EndPoint": [144.0, 46.0, 73.0],
        },
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [None, 0.0, None],
            "StartPoint": [0.0, 0.0, 0.0],
            "EndPoint": [144.0, 0.0, 73.0],
        },
        {
            "BoundaryType": "VelocityConstraint",
            "Velocity": [None, 0.0, None],
            "StartPoint": [0.0, 46.0, 0.0],
            "EndPoint": [144.0, 46.0, 73.0],
        },
    ]
)

dempm.mpm.select_save_data()

dempm.add_body(adaptive_boundary_radius=True)

dempm.choose_contact_model(particle_particle_contact_model=None, particle_wall_contact_model="Linear Model")

dempm.add_property(
    DEMmaterial=0,
    MPMmaterial=1,
    property={
        "NormalStiffness": 1e7,
        "TangentialStiffness": 1e7,
        "Friction": 0.5,
        "NormalViscousDamping": 0.5,
        "TangentialViscousDamping": 0.5,
    },
    dType="particle-wall",
)

dempm.add_property(
    DEMmaterial=0,
    MPMmaterial=2,
    property={
        "NormalStiffness": 1e7,
        "TangentialStiffness": 1e7,
        "Friction": 0.5,
        "NormalViscousDamping": 0.5,
        "TangentialViscousDamping": 0.5,
    },
    dType="particle-wall",
)

dempm.add_property(
    DEMmaterial=0,
    MPMmaterial=3,
    property={
        "NormalStiffness": 1e7,
        "TangentialStiffness": 1e7,
        "Friction": 0.5,
        "NormalViscousDamping": 0.5,
        "TangentialViscousDamping": 0.5,
    },
    dType="particle-wall",
)

dempm.add_property(
    DEMmaterial=0,
    MPMmaterial=4,
    property={
        "NormalStiffness": 1e7,
        "TangentialStiffness": 1e7,
        "Friction": 0.5,
        "NormalViscousDamping": 0.5,
        "TangentialViscousDamping": 0.5,
    },
    dType="particle-wall",
)

dempm.add_property(
    DEMmaterial=0,
    MPMmaterial=5,
    property={
        "NormalStiffness": 1e7,
        "TangentialStiffness": 1e7,
        "Friction": 0.5,
        "NormalViscousDamping": 0.5,
        "TangentialViscousDamping": 0.5,
    },
    dType="particle-wall",
)


def get_gravity(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    eps = 1e-12

    d_lower2 = (x - 72.0) ** 2 + (y - 11.0) ** 2
    d_upper2 = (x - 72.0) ** 2 + (y - 35.0) ** 2
    d_middle2 = (x - 72.0) ** 2 + (y - 23.0) ** 2

    inside_lower = d_lower2 < 11.0**2 - eps
    inside_upper = d_upper2 < 11.0**2 - eps
    inside_middle = d_middle2 < 9.0**2 - eps

    inside_cylinder = inside_lower | inside_upper | inside_middle

    surface_z = 61.0 + 0.0 * z
    surface_z = ti.select(inside_cylinder, 72.0, surface_z)

    return surface_z - z


dempm.run(mpm_gravity_field=get_gravity)

dempm.dem.scene.geometry.modify(
    0,
    {"Density": 2650, "ExternalForce": [5.8524e2 * hori_pressure, 0.0, 0.0], "AppliedPoint": [72, 23, 66.5]},
    dempm.dem.scene.wall,
)
dempm.mpm.add_boundary_condition({""})
dempm.modify_parameters(SimulationTime=2)
dempm.run()

dempm.mpm.postprocessing()

# ti.profiler.print_kernel_profiler_info()
