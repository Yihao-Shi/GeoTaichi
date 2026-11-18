from src.utils.errors import SimulationStateError

from src.sdf.SDF import SDF
from src.sdf.ease import linear
import src.sdf.utils_2d as utils_2d
import src.sdf.utils_2d_3d as utils_2d_3d
import src.sdf.MultiSDF as MultiSDF


class SDF2D(SDF):
    def __init__(self, ray=False):
        super().__init__(ray)
        self.is_simple_shape = -1

    @property
    def volume(self):
        raise SimulationStateError("2D shape do not have volume attribute")

    def generate(self, step, samples, workers, batch_size, verbose=True, sparse=True):
        raise SimulationStateError("2D shape do not have this attribute")

    def save(self, path, step, samples, workers, batch_size, verbose=True, sparse=True):
        raise SimulationStateError("2D shape do not have this attribute")

    def show_slice(self, show_abs=False, w=1024, h=1024, x=None, y=None, z=None):
        raise SimulationStateError("2D shape do not have this attribute")

    def union(self, other, smooth=0.0):
        new_sdf = SDF2D()
        new_sdf.calculate_distance = MultiSDF.union(self, other, k=smooth)
        return new_sdf

    def difference(self, other, smooth=0.0):
        new_sdf = SDF2D()
        new_sdf.calculate_distance = MultiSDF.difference(self, other, k=smooth)
        return new_sdf

    def intersection(self, other, smooth=0.0):
        new_sdf = SDF2D()
        new_sdf.calculate_distance = MultiSDF.intersection(self, other, k=smooth)
        return new_sdf

    def blend(self, other, smooth=0.0):
        new_sdf = SDF2D()
        new_sdf.calculate_distance = MultiSDF.blend(self, other, k=smooth)
        return new_sdf

    def negate(
        self,
    ):
        new_sdf = SDF2D()
        new_sdf.calculate_distance = MultiSDF.negate(self)
        return new_sdf

    def dilate(self, radius):
        new_sdf = SDF2D()
        new_sdf.calculate_distance = MultiSDF.dilate(self, radius)
        return new_sdf

    def erode(self, radius):
        new_sdf = SDF2D()
        new_sdf.calculate_distance = MultiSDF.erode(self, radius)
        return new_sdf

    def shell(self, thickness):
        new_sdf = SDF2D()
        new_sdf.calculate_distance = MultiSDF.shell(self, thickness)
        return new_sdf

    def repeat(self, spacing, count=None, padding=0):
        new_sdf = SDF2D()
        new_sdf.calculate_distance = MultiSDF.repeat(self, spacing, count, padding)
        return new_sdf

    def translate(self, offset):
        new_sdf = SDF2D()
        new_sdf.calculate_distance = utils_2d.translate(self, offset)
        return new_sdf

    def scale(self, factor):
        new_sdf = SDF2D()
        new_sdf.calculate_distance = utils_2d.scale(self, factor)
        return new_sdf

    def rotate(self, angle):
        new_sdf = SDF2D()
        new_sdf.calculate_distance = utils_2d.rotate(self, angle)
        return new_sdf

    def circular_array(self, count):
        new_sdf = SDF2D()
        new_sdf.calculate_distance = utils_2d.circular_array(self, count)
        return new_sdf

    def extrude(self, height):
        from src.sdf.SDF3D import SDF3D

        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils_2d_3d.extrude(self, height)
        return new_sdf

    def extrude_to(self, other, height, functions=linear):
        from src.sdf.SDF3D import SDF3D

        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils_2d_3d.extrude_to(self, other, height, functions)
        return new_sdf

    def revolve(self, count):
        from src.sdf.SDF3D import SDF3D

        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils_2d_3d.revolve(self, count)
        return new_sdf
