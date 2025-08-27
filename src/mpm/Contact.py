import numpy as np
import taichi as ti

from src.utils.linalg import read_dict_list
from src.utils.ObjectIO import DictIO


class ContactBase(object):
    def __init__(self, contact_phys):
        self._name = None
        self.contact_phys = None
        self.polygon_vertices = None
        self._friction = DictIO.GetAlternative(contact_phys, "friction", 0.)

    def get_parameters(self, *arg, **kwargs):
        pass
    
    def print_contact_message(self, *arg, **kwargs):
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    @property
    def friction(self):
        return self._friction
    
    @friction.setter
    def friction(self, friction):
        self._friction = friction


class MPMContact(ContactBase):
    def __init__(self, contact_phys):
        super().__init__(contact_phys)
        self._name = "MPMContact"
    
    def print_contact_message(self):
        print("Friction Coefficient =", self.friction, '\n')


class GeoContact(ContactBase):
    def __init__(self, contact_phys):
        super().__init__(contact_phys)
        self._name = "GeoContact"
        penalty = DictIO.GetAlternative(contact_phys, "penalty", [1., 2.])
        cut_off = DictIO.GetAlternative(contact_phys, "cut_off", 1.5)
        if isinstance(penalty, (tuple, list, np.ndarray)):
            if len(list(penalty)) != 2:
                raise RuntimeError("The dimension of Keyword:: /Penalty/ should be 2")
        if penalty[0] < 0. or penalty[0] > 1. or penalty[1] < 2.:
            raise RuntimeError("Keyword:: 0 <= /Penalty[0]/ <= 1 and /Penalty[1]/ >= 2")
        self._alpha = penalty[0]
        self._beta = penalty[1]
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

    @property
    def cut_off(self):
        return self._cut_off
    
    @cut_off.setter
    def cut_off(self, cut_off):
        self._cut_off = cut_off

    def print_contact_message(self):
        print("Penalty Parameter = ", self.alpha, self.beta)
        print("Friction Coefficient =", self.friction)
        print("Cut-off Distance Multiplier =", self.cut_off, '\n')


class DEMContact(ContactBase):
    def __init__(self, contact_phys: dict):
        self._name = "DEMContact"
        self.polygon_vertices = None
        self._velocity = ti.Vector([0., 0.])
        self.contact_phys = []
        if len(contact_phys) == 1:
            first_value = next(iter(contact_phys.values()))
            self.contact_phys.append([first_value] if isinstance(first_value, dict) else first_value)
        else:
            if all(isinstance(value, dict) for value in contact_phys.values()):
                for value in contact_phys.values():
                    self.contact_phys.append([value])
            else:
                self.contact_phys.append(contact_phys)

    def get_parameters(self, contact_parameter):
        stiffness = DictIO.GetAlternative(contact_parameter, "stiffness", [1e6, 1e6])
        friction = DictIO.GetAlternative(contact_parameter, "friction", 0.)
        if len(list(stiffness)) != 2:
            raise RuntimeError("The dimension of Keyword:: /stiffness/ should be 2")
        return {'kn': stiffness[0], 'kt': stiffness[1], "friction":friction}

    def generate_vertice_field(self, dim, size):
        self.polygon_vertices = ti.Vector.field(dim, float, shape=size)

    @property
    def velocity(self):
        return self._velocity
    
    @velocity.setter
    def velocity(self, velocity):
        self._velocity = velocity

    def print_contact_message(self):
        read_dict_list(self.contact_phys, self.print_single_contact_message)

    def print_single_contact_message(self, phys):
        print("Material ID: ", DictIO.GetEssential(phys, 'materialID'))
        cphys = self.get_parameters(phys)
        print("Stiffness Parameter = ", cphys['kn'], cphys['kt'])
        print("Friction Coefficient =", cphys['friction'], '\n')
            

        