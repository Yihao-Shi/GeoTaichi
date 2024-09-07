import sys
sys.path.append('/home/eleven/work/GeoTaichi_release')

import taichi as ti
ti.init(default_fp=ti.f32, arch=ti.cpu, debug=True)

from src.dem.BaseStruct import FacetFamily

# must be anticlockwise
facet = FacetFamily.field(shape=4)
facet[0].add_wall_geometry(wallID=0,vertice1=ti.Vector([0., 0., 1.1]),
                           vertice2=ti.Vector([15., 0., 1.1]),
                           vertice3=ti.Vector([15., 6., 1.1]),
                           norm=ti.Vector([0., 0., 1.]),
                           init_v=ti.Vector([0., 0., 0.]))
facet[1].add_wall_geometry(wallID=0,vertice1=ti.Vector([0., 0., 1.1]),
                           vertice2=ti.Vector([15., 6., 1.1]),
                           vertice3=ti.Vector([0., 6., 1.1]),
                           norm=ti.Vector([0., 0., 1.]),
                           init_v=ti.Vector([0., 0., 0.]))

@ti.kernel
def area_fraction(facet: ti.template(), position: ti.types.vector(3, float), radius: float):
    distance1 = facet[0]._get_norm_distance(position)
    distance2 = facet[1]._get_norm_distance(position)
    area1 = facet[0].processCircleShape(position, radius, distance1)
    area2 = facet[1].processCircleShape(position, radius, distance2)
    print(f"Area fraction in facet 0 is {area1}")
    print(f"Area fraction in facet 1 is {area2}")
area_fraction(facet, position=ti.Vector([2., 3., 2.6]), radius=1.6)
