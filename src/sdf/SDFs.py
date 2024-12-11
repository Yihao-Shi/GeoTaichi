from src.sdf.BasicShape import SDF
from src.sdf.ease import linear
import src.sdf.utils2D as utils2D
import src.sdf.utils3D as utils3D
import src.sdf.utils23D as utils23D
import src.sdf.MultiSDF as MultiSDF


class SDF2D(SDF):
    def __init__(self, ray=False):
        super().__init__(ray)
        self.is_simple_shape = -1

    @property
    def volume(self):
        raise RuntimeError("2D shape do not have volume attribute")
    
    def generate(self, step, samples, workers, batch_size, verbose=True, sparse=True):
        raise RuntimeError("2D shape do not have this attribute")
        
    def save(self, path, step, samples, workers, batch_size, verbose=True, sparse=True):
        raise RuntimeError("2D shape do not have this attribute")
        
    def show_slice(self, show_abs=False, w=1024, h=1024, x=None, y=None, z=None):
        raise RuntimeError("2D shape do not have this attribute")

    def union(self, other, smooth=0.):
        new_sdf = SDF2D()
        new_sdf.calculate_distance = MultiSDF.union(self, other, k=smooth)
        return new_sdf
    
    def difference(self, other, smooth=0.):
        new_sdf = SDF2D()
        new_sdf.calculate_distance = MultiSDF.difference(self, other, k=smooth)
        return new_sdf
    
    def intersection(self, other, smooth=0.):
        new_sdf = SDF2D()
        new_sdf.calculate_distance = MultiSDF.intersection(self, other, k=smooth)
        return new_sdf
    
    def blend(self, other, smooth=0.):
        new_sdf = SDF2D()
        new_sdf.calculate_distance = MultiSDF.blend(self, other, k=smooth)
        return new_sdf
    
    def negate(self, ):
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
        new_sdf.calculate_distance = utils2D.translate(self, offset)
        return new_sdf
    
    def scale(self, factor):
        new_sdf = SDF2D()
        new_sdf.calculate_distance = utils2D.scale(self, factor)
        return new_sdf
    
    def rotate(self, angle):
        new_sdf = SDF2D()
        new_sdf.calculate_distance = utils2D.rotate(self, angle)
        return new_sdf
    
    def circular_array(self, count):
        new_sdf = SDF2D()
        new_sdf.calculate_distance = utils2D.circular_array(self, count)
        return new_sdf
    
    def extrude(self, height):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils23D.extrude(self, height)
        return new_sdf
    
    def extrude_to(self, other, height, functions=linear):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils23D.extrude_to(self, other, height, functions)
        return new_sdf
    
    def revolve(self, count):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils23D.revolve(self, count)
        return new_sdf
    

class SDF3D(SDF):
    def __init__(self, ray=False):
        super().__init__(ray)
        self.is_simple_shape = 3
    
    def union(self, other, smooth=0.):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = MultiSDF.union(self, other, k=smooth)
        return new_sdf
    
    def difference(self, other, smooth=0.):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = MultiSDF.difference(self, other, k=smooth)
        return new_sdf
    
    def intersection(self, other, smooth=0.):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = MultiSDF.intersection(self, other, k=smooth)
        return new_sdf
    
    def blend(self, other, smooth=0.):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = MultiSDF.blend(self, other, k=smooth)
        return new_sdf
    
    def negate(self, ):
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
        new_sdf.calculate_distance = utils3D.translate(self, offset)
        return new_sdf
    
    def scale(self, factor):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils3D.scale(self, factor)
        return new_sdf
    
    def rotate(self, angle, vector=(0., 0., 1.)):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils3D.rotate(self, angle, vector)
        return new_sdf
    
    def orient(self, axis):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils3D.orient(self, axis)
        return new_sdf
    
    def circular_array(self, count, offset=0):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils3D.circular_array(self, count, offset)
        return new_sdf
    
    def elongate(self, size):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils3D.elongate(self, size)
        return new_sdf
    
    def twist(self, parameter):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils3D.twist(self, parameter)
        return new_sdf
    
    def bend(self, parameter):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils3D.bend(self, parameter)
        return new_sdf
    
    def bend_linear(self, p0, p1, v, functions=linear):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils3D.bend_linear(self, p0, p1, v, functions)
        return new_sdf
    
    def bend_radial(self, r0, r1, dz, functions=linear):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils3D.bend_radial(self, r0, r1, dz, functions)
        return new_sdf
    
    def transition_linear(self, objects, p0=(0., 0., -1.), p1=(0., 0., 1.), functions=linear):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils3D.transition_linear(self, objects, p0, p1, functions)
        return new_sdf
    
    def transition_radial(self, objects, r0=0., r1=1., functions=linear):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils3D.transition_radial(self, objects, r0, r1, functions)
        return new_sdf
    
    def wrap_around(self, x0, x1, r=None, functions=linear):
        new_sdf = SDF3D()
        new_sdf.calculate_distance = utils3D.wrap_around(self, x0, x1, r, functions)
        return new_sdf
    
    def slice(self, slab):
        '''
        slab: define a slab objects
        '''
        new_sdf = SDF2D()
        new_sdf.calculate_distance = utils23D.slice(self, slab)
        return new_sdf