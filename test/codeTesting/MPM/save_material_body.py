import sys
sys.path.append('/home/eleven/work/GeoTaichi')

import taichi as ti
ti.init(default_fp=ti.f64, arch=ti.cpu, debug=True)

from src.mpm.mainMPM import MPM

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([10., 10., 10.]))

mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.1, 0.1, 0.1])
                        })

mpm.add_grid(grid={
                       "GridNumber":                      1,
                       "ContactDetection":                False
                  })

mpm.add_region(region={
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.,0.,0.]),
                            "BoundingBoxSize": ti.Vector([7.0,7.0,7.0]),
                            "zdirection": ti.Vector([1.,1.,1.])
                      })

mpm.add_body(body={    
                       "WriteFile":                          True,
                       "Template": {
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2
                                   }
                   })