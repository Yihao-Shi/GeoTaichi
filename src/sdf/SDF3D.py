from src.sdf.SDF2D import SDF2D

from src.sdf.SDF import SDF
from src.sdf.ease import linear
import src.sdf.utils_3d as utils_3d
import src.sdf.utils_2d_3d as utils_2d_3d
import src.sdf.MultiSDF as MultiSDF


class SDF3D(SDF):
    def __init__(self, ray=False):
        super().__init__(ray)
        self.is_simple_shape = 3

    def union(self, other, smooth=0.0):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = MultiSDF.union(self, other, k=smooth)
        return new_sdf

    def difference(self, other, smooth=0.0):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = MultiSDF.difference(self, other, k=smooth)
        return new_sdf

    def intersection(self, other, smooth=0.0):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = MultiSDF.intersection(self, other, k=smooth)
        return new_sdf

    def blend(self, other, smooth=0.0):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = MultiSDF.blend(self, other, k=smooth)
        return new_sdf

    def negate(
        self,
    ):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = MultiSDF.negate(self)
        return new_sdf

    def dilate(self, radius):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = MultiSDF.dilate(self, radius)
        return new_sdf

    def erode(self, radius):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = MultiSDF.erode(self, radius)
        return new_sdf

    def shell(self, thickness):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = MultiSDF.shell(self, thickness)
        return new_sdf

    def repeat(self, spacing, count=None, padding=0):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = MultiSDF.repeat(self, spacing, count, padding)
        return new_sdf

    def translate(self, offset):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils_3d.translate(self, offset)
        return new_sdf

    def scale(self, factor):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils_3d.scale(self, factor)
        return new_sdf

    def rotate(self, angle, vector=(0.0, 0.0, 1.0)):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils_3d.rotate(self, angle, vector)
        return new_sdf

    def orient(self, axis):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils_3d.orient(self, axis)
        return new_sdf

    def circular_array(self, count, offset=0):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils_3d.circular_array(self, count, offset)
        return new_sdf

    def elongate(self, size):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils_3d.elongate(self, size)
        return new_sdf

    def twist(self, parameter):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils_3d.twist(self, parameter)
        return new_sdf

    def bend(self, parameter):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils_3d.bend(self, parameter)
        return new_sdf

    def bend_linear(self, p0, p1, v, functions=linear):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils_3d.bend_linear(self, p0, p1, v, functions)
        return new_sdf

    def bend_radial(self, r0, r1, dz, functions=linear):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils_3d.bend_radial(self, r0, r1, dz, functions)
        return new_sdf

    def transition_linear(self, objects, p0=(0.0, 0.0, -1.0), p1=(0.0, 0.0, 1.0), functions=linear):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils_3d.transition_linear(self, objects, p0, p1, functions)
        return new_sdf

    def transition_radial(self, objects, r0=0.0, r1=1.0, functions=linear):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils_3d.transition_radial(self, objects, r0, r1, functions)
        return new_sdf

    def wrap_around(self, x0, x1, r=None, functions=linear):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils_3d.wrap_around(self, x0, x1, r, functions)
        return new_sdf

    def slice(self, slab):
        """
        slab: define a slab objects
        """
        new_sdf = SDF2D()
        new_sdf.calculate_distance = utils_2d_3d.slice(self, slab)
        return new_sdf
