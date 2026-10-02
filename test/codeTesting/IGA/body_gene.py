import taichi as ti
ti.init(default_fp=ti.f64, debug=True)

from src.iga.mainIGA import IGA

iga = IGA()

iga.set_configuration(domain=[10, 10, 10])

iga.set_solver(solver={"Timestep": 1e-2,
                       "SimulationTime": 10})

iga.memory_allocate(memory={ 
                            "max_material_number": 1,
                            "max_point_number": [5, 2, 2],
                            "max_degree_number":  [1, 1, 1],
                            "max_gauss_number": [2, 2, 2]
                            }, log=True)

iga.add_patch(patch={
                       "BodyID":     0,
                       "MaterialID": 0,
                       "BasicShape": "Rectangle",
                       "NodeNumber": [5, 2, 2],
                       "Degree":     [1, 1, 1],
                       "Refinement": [0, 0, 0],
                       "Dimensions":{
                                     "StartPoint": [2, 0, 2],
                                     "Size": [4, 2, 2]
                                    },
                       "Resolution": [10, 4, 4],
                       "Visualize": False
                   })

iga.select_save_data()

iga.run()

