import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(arch='gpu', log=False, debug=True)

lsdem = DEM()

lsdem.set_configuration(domain=ti.Vector([10.,5.,5.]),
                        scheme="LSDEM",
                        gravity=ti.Vector([0., 0., -9.8]))

                  
'''c = capped_cylinder([0., 0., -1.], [0., 0., 1.], 0.25)                   
lsdem.add_template(template={
                                "Name":               "Template1",
                                "Object":              polysuperellipsoid(xrad1=1.5, yrad1=1.5, zrad1=1.5, epsilon_e=1.5, epsilon_n=1.5)

,
                                "SurfaceResolution":   200000,
                                "GridSpace":          0.1,
                                "WriteFile":          True}, 
                   types="LevelSet") '''

lsdem.add_template(template={
                                "Name":               "Template1",
                                "Object":              polyhedron(file=f'{ROOT}/assets/mesh/LSDEM/sand.stl')

,
                                "SurfaceResolution":   200000,
                                "GridSpace":          5,
                                "WriteFile":          True}, 
                   types="LevelSet")
