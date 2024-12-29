import sys
sys.path.append('/home/eleven/work/GeoTaichi')

from geotaichi import *

init(device_memory_GB=3, debug=False)

f = polyhedron(file='/home/eleven/work/pyLevelSet/assets/sand.stl').grids(space=5)
f.save('sand.stl')
f.visualize()

sf = sphere(1) & box(1.5)
cy = cylinder(0.5)
sf -= cy.orient([1, 0, 0]) | cy.orient([0, 1, 0]) | cy.orient([0, 0, 1])
sf.grids(space=0.1).reset(False).save('sdf.stl', samples=90002)
sf.visualize()

f = polysuperellipsoid(xrad1=0.5, yrad1=0.25, zrad1=0.75, xrad2=0.25, yrad2=0.75, zrad2=0.5, epsilon_e=1.5, epsilon_n=1.5).grids(space=0.05)
f.save('ellipsoid.stl', samples=2502, sparse=False)
f.visualize()

f = polysuperquadrics(xrad1=0.5, yrad1=2.5, zrad1=1.7, xrad2=1.0, yrad2=0.5, zrad2=0.5, epsilon_x=0.5, epsilon_y=1.5, epsilon_z=1.2).grids(space=0.05)
f.save('quadrics.stl', samples=10002, sparse=False)
f.visualize()