import sys
sys.path.append('/home/eleven/work/GeoTaichi')

import taichi as ti
ti.init(arch=ti.cpu, default_fp=ti.f64, default_ip=ti.i32, debug=False)

from src.mpm.mainMPM import MPM

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([0.55, 0.2, 0.11]), 
                      background_damping=0.02, 
                      alphaPIC=0.005, 
                      mapping="USL", 
                      shape_function="GIMP",
                      simulation_type="MultiBody")

mpm.set_solver(solver={
                           "Timestep":                   1e-5,
                           "SimulationTime":             0.6,
                           "SaveInterval":               0.01
                      })

mpm.add_region(region={
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.005, 0.005, 0.005]),
                            "BoundingBoxSize": ti.Vector([0.2, 0.05, 0.1]),
                            "zdirection": ti.Vector([0., 0., 1.])
                      })

mpm.add_multibody_template(template={
                                        "Name":               "Template1",
                                        "Resolution":         0.1,
                                        "nParticlesPerCell":  2,
                                        "RegionName":         "region1"
                                   })


