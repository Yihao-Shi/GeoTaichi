import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init()

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([10, 11]),
                      dimension="2-Dimension",
                      background_damping=0.0,
                      gravity=ti.Vector([0., 0]),
                      alphaPIC=0.005,
                      mapping="USL",
                      shape_function="GIMP",
                      stabilize="B-Bar Method",
                      gauss_number=2
                      )

mpm.set_solver(solver={
                           "Timestep":                   1e-5,
                           "SimulationTime":             5,
                           "SaveInterval":               0.25,
                           "SavePath":                   'TrescaLargeStrain'
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    250000,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":     134474,
                                                               "max_reflection_constraint":   0
                                                          }
                            })
                            
mpm.add_material(model="MohrCoulomb",
                 material={
                               "MaterialID":           1,
                               "Density":              1000.,
                               "YoungModulus":         1e5,
                               "PoissonRatio":        0.49,
                               "Cohesion":             1000,
                               "Friction":             0.,
                               "Dilation":             0.
                 })

mpm.add_element(element={
                             "ElementType":               "Q4N2D",
                             "ElementSize":               ti.Vector([0.05, 0.05]),
                             "Contact":   {
                                               "ContactDetection":                "GeoContact",
                                               "Friction":                        0.5,
                                               "CutOff":                          1.,
                                               "Penalty":                         [0.,10.]
                                          }
                        })


mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([0., 0.]),
                            "BoundingBoxSize": ti.Vector([10., 10.]),
                            
                      },
                      
                      {
                            "Name": "region2",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([0., 10.]),
                            "BoundingBoxSize": ti.Vector([0.5, 1]),
                            
                      },
                      ])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":ti.Vector([0., 0.]),
                                       "FixVelocity":    ["Free", "Free"]
                                       
                                   },
                                   
                                   {
                                       "RegionName":         "region2",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             1,
                                       "RigidBody":          True,
                                       "Density":            1500,
                                       "ParticleStress":     {},
                                       "Traction":           {},
                                       "InitialVelocity":    ti.Vector([0., -0.02]),
                                       "FixVelocity":        ["Fix", "Fix"]
                                       
                                   }]
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0.],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [10., 0.],
                                             "NLevel":         0
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [0., 11.]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None],
                                             "StartPoint":     [10., 0.],
                                             "EndPoint":       [10., 11.]
                                        },
                                    ])

def stepwise():
    ramp = mpm.sims.time
    vel0 = 0.1
    deltat = mpm.sims.CurrentTime[None]
    vel = 0.
    for i in range(mpm.scene.particleNum[0]):
        if mpm.scene.particle[i].bodyID==1:
            mpm.scene.particle[i].v=[0., -(deltat / ramp) * vel0]
    mpm.sims.CurrentTime[None] += mpm.sims.dt[None]

mpm.select_save_data(grid=True)

mpm.run(function=stepwise)

mpm.modify_parameters(SimulationTime=50, SaveInterval=1)

mpm.run()

mpm.postprocessing(read_path='TrescaLargeStrain', write_background_grid=True)
