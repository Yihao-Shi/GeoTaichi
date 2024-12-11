import numpy as np
import taichi as ti

from src.mpm.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.utils.ObjectIO import DictIO

class MPMContact(object):
    def __init__(self, contact):
        friction = DictIO.GetAlternative(contact, "Friction", 0.)
        self._friction = friction
        self.polygon_vertices = None

    @property
    def friction(self):
        return self._friction
    
    @friction.setter
    def friction(self, friction):
        self._friction = friction


class GeoContact(MPMContact):
    def __init__(self, contact):
        super().__init__(contact)
        penalty = DictIO.GetAlternative(contact, "Penalty", [1., 2.])
        cut_off = DictIO.GetAlternative(contact, "CutOff", 0.8)
        self._cut_off = cut_off
        if isinstance(penalty, (tuple, list, np.ndarray)):
            if len(list(penalty)) != 2:
                raise RuntimeError("The dimension of Keyword:: /Penalty/ should be 2")
            self._alpha = penalty[0]
            self._beta = penalty[1]
        if self._alpha < 0. or self._alpha > 1. or self._beta < 2.:
            raise RuntimeError("Keyword:: 0 <= /Penalty[0]/ <= 1 and /Penalty[1]/ >= 2")

    @property
    def cut_off(self):
        return self._cut_off
    
    @cut_off.setter
    def cut_off(self, cut_off):
        self._cut_off = cut_off
        
    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha

    @property
    def beta(self):
        return self._beta
    
    @beta.setter
    def beta(self, beta):
        self._beta = beta


class DEMContact(MPMContact):
    def __init__(self, contact, material: ConstitutiveModelBase):
        super().__init__(contact)
        if "Friction" in contact:
            stiffness = DictIO.GetAlternative(contact, "Stiffness", [1e6, 1e6])
            friction = DictIO.GetAlternative(contact, "Friction", 0.)
            self._friction = friction
            if isinstance(stiffness, (tuple, list, np.ndarray)):
                if len(list(stiffness)) != 2:
                    raise RuntimeError("The dimension of Keyword:: /Stiffness/ should be 2")
                self._kn = stiffness[0]
                self._kt = stiffness[1]

            for materialID in range(material.matProps.shape[0]):
                material.matProps[materialID].add_contact_parameter(friction, stiffness[0], stiffness[1])
        self._velocity = ti.Vector([0., 0.])

    def generate_vertice_field(self, dim, size):
        self.polygon_vertices = ti.Vector.field(dim, float, shape=size)

    @property
    def velocity(self):
        return self._velocity
    
    @velocity.setter
    def velocity(self, velocity):
        self._velocity = velocity

    @property
    def kn(self):
        return self._kn
    
    @kn.setter
    def kn(self, kn):
        self._kn = kn

    @property
    def kt(self):
        return self._kt
    
    @kt.setter
    def kt(self, kt):
        self._kt = kt