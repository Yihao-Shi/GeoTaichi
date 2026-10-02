from geotaichi import *

init(dim=2, arch='gpu', kernel_profiler=False, debug=False)

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([100.0, 21.0]),
                      is_2DAxisy=False,
                      background_damping=0.0,
                      gravity=ti.Vector([0., -10.0]),
                      alphaPIC=0.0,
                      mapping="USL",
                      shape_function="GIMP",
                      random_field=False,
                      stress_integration="SubStepping",
                      solver_type="Explicit"
                      )

mpm.set_solver(solver={
    "Timestep":       1.0e-4,
    "SimulationTime": 8,
    "SaveInterval":   0.05,
    "SavePath":       'SoftDP'
})

mpm.memory_allocate(memory={
    "max_material_number": 1,
    "max_particle_number": 8.12e4,
    "max_constraint_number": {
        "max_velocity_constraint": 83000
    }
})

#mpm.add_material(model="DruckerPrager",
#                 material={
#                     "MaterialID": 1,
#                     "MaterialFile": args.input
#                 })
mpm.add_material(model="DruckerPrager",
                 material={
                               "SoftType":             "Linear",
                               "dpType":               "MiddleCircumscribed",
                               "SoftenParameter":       5,
                               "MaterialID":           1,
                               "Density":              2000.,
                               "YoungModulus":         1000000.0,
                               "PoissonRatio":         0.33,
                               "Cohesion":             20000.0,
                               "Friction":             1.0,
                               "Dilation":             0.,
                               "Residualcohesion":     4000.0,
                               "ResidualFriction":     1.0,
                               "ResidualDilation":     0.,
                               "PlasticDevStrain":     0.0,
                               "ResidualPlasticDevStrain":  0.1
                 })


mpm.add_element(element={
    "ElementType": "Q4N2D",
    "ElementSize": ti.Vector([0.2, 0.2])
})


def get_gravity(points):
    import numpy as np
    return np.where(points[:, 0] < 20.0, 5 - points[:, 1], 25.0 - points[:, 0] - points[:, 1])

def region_func(new_position, new_radius=0.):
    temp = 0
    if (new_position[0]<20.0):
        temp = 1
    else :
        temp = 25-new_position[0]-new_position[1]>0.0
    return temp
    
def volume_func():
    # return the area of our target region
    return 100.
    
mpm.add_region(region=[
                            {
                                "Name":                     "region2",
                                "Type":                     "UserDefined",
                                "BoundingBoxPoint":         ti.Vector([0., 0.]),
                                "BoundingBoxSize":          ti.Vector([25., 5.]),
                                "RegionVolume":             volume_func,
                                "RegionFunction":           region_func
                            }
                      ])
mpm.add_body(body={
                        "Template": [
                                            {
                                                "RegionName":               "region2",
                                                "nParticlesPerCell":        2,
                                                "BodyID":                   0,
                                                "MaterialID":               1,
                                                "InitialVelocity":          ti.Vector([0, 0]),
                                                "FixVelocity":              ["Free",  "Free"]

                                            }
                        ]
                  })
                  
mpm.add_boundary_condition(boundary=[
    {
        "BoundaryType": "VelocityConstraint",
        "Velocity": [0., 0],
        "StartPoint": [0., 0.],
        "EndPoint": [100.0, 0.]
    },

    {
        "BoundaryType": "VelocityConstraint",
        "Velocity": [0., None],
        "StartPoint": [0., 0.],
        "EndPoint": [0., 6.0]
    },

    {
        "BoundaryType": "VelocityConstraint",
        "Velocity": [0., None],
        "StartPoint": [100., 0.],
        "EndPoint": [100., 6.0]
    },

])

mpm.select_save_data()

mpm.run(gravity_field=get_gravity)

# ti.profiler.print_kernel_profiler_info()

mpm.postprocessing(read_path='SoftDP')
