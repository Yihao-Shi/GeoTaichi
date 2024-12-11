import taichi as ti

from src.dem.contact.ContactKernel import *
from src.dem.contact.ContactModelBase import ContactModelBase
from src.dem.SceneManager import myScene
from src.utils.constants import PI, ZEROVEC3f
from src.utils.ObjectIO import DictIO
from src.utils.VectorFunction import Normalize, Squared


class HertzMindlinModel(ContactModelBase):
    def __init__(self, sims) -> None:
        super().__init__(sims)
        self.surfaceProps = HertzMindlinSurfaceProperty.field(shape=self.sims.max_material_num * self.sims.max_material_num)
        self.null_model = False
        self.model_type = 1

    def calcu_critical_timestep(self, scene: myScene):
        radius = scene.find_particle_min_radius(self.sims.scheme)
        density = scene.find_min_density()
        modulus, Possion = self.find_max_mparas()
        return PI * radius * ti.sqrt(density / modulus) / (0.1631 * Possion + 0.8766)

    def find_max_mparas(self):
        maxmodulus, maxpossion = 0., 0.
        for materialID1 in range(self.sims.max_material_num):
            for materialID2 in range(self.sims.max_material_num):
                componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
                if self.surfaceProps[componousID].ShearModulus > 0.:
                    Possion = (4 * self.surfaceProps[componousID].ShearModulus - self.surfaceProps[componousID].YoungModulus) / \
                              (2 * self.surfaceProps[componousID].ShearModulus - self.surfaceProps[componousID].YoungModulus)
                    modulus = 2 * self.surfaceProps[componousID].ShearModulus * (2 - Possion)
                    maxpossion = ti.max(maxpossion, Possion)
                    maxmodulus = ti.max(maxpossion, modulus)
        return maxmodulus, maxpossion
    
    def add_surface_property(self, materialID1, materialID2, property):
        modulus = DictIO.GetEssential(property, 'ShearModulus')
        possion = DictIO.GetEssential(property, 'Possion')
        ShearModulus = 0.5 * modulus / (2. - possion)
        YoungModulus = (4. * ShearModulus - 2. * ShearModulus * possion) / (1. - possion)
        mus = DictIO.GetEssential(property, 'StaticFriction', 'Friction')
        mud = DictIO.GetEssential(property, 'DynamicFriction', 'Friction')
        restitution = DictIO.GetEssential(property, 'Restitution')
        componousID = 0
        if restitution < 1e-16:
            restitution = 0.
        else:
            restitution = -ti.log(restitution) / ti.sqrt(PI * PI + ti.log(restitution) * ti.log(restitution))
        if materialID1 == materialID2:
            componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(YoungModulus, ShearModulus, mus, mud, restitution)
        else:
            componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
            self.surfaceProps[componousID].add_surface_property(YoungModulus, ShearModulus, mus, mud, restitution)
            componousID = self.get_componousID(self.sims.max_material_num, materialID2, materialID1)
            self.surfaceProps[componousID].add_surface_property(YoungModulus, ShearModulus, mus, mud, restitution)
        return componousID

    
    def update_property(self, componousID, property_name, value, override):
        factor = 0
        if not override:
            factor = 1

        if property_name == "ShearModulus":
            E, G = self.surfaceProps[componousID].YoungModulus, self.surfaceProps[componousID].ShearModulus
            possion = (4. * G - E) / (2. * G - E)
            modulus = 2. * G * (2. - possion)
            modulus = factor * modulus + value
            ShearModulus = 0.5 * modulus / (2. - possion)
            YoungModulus = (4. * ShearModulus - 2. * ShearModulus * possion) / (1. - possion)
            self.surfaceProps[componousID].YoungModulus = YoungModulus
            self.surfaceProps[componousID].ShearModulus = ShearModulus
        elif property_name == "Possion":
            E, G = self.surfaceProps[componousID].YoungModulus, self.surfaceProps[componousID].ShearModulus
            possion = (4. * G - E) / (2. * G - E)
            modulus = 2. * G * (2. - possion)
            possion = factor * possion + value
            ShearModulus = 0.5 * modulus / (2. - possion)
            YoungModulus = (4. * ShearModulus - 2. * ShearModulus * possion) / (1. - possion)
            self.surfaceProps[componousID].YoungModulus = YoungModulus
            self.surfaceProps[componousID].ShearModulus = ShearModulus
        elif property_name == "Friction":
            self.surfaceProps[componousID].mu = factor * self.surfaceProps[componousID].mu + value
        elif property_name == "Restitution":
            self.surfaceProps[componousID].restitution = factor * self.surfaceProps[componousID].restitution + value


@ti.dataclass
class HertzMindlinSurfaceProperty:
    YoungModulus: float
    ShearModulus: float
    restitution: float
    mus: float
    mud: float
    ncut: float

    def add_surface_property(self, YoungModulus, ShearModulus, mus, mud, restitution):
        self.YoungModulus = YoungModulus
        self.ShearModulus = ShearModulus
        self.mus = mus
        self.mud = mud
        self.restitution = restitution
        self.ncut = 0.

    def print_surface_info(self, matID1, matID2):
        print(" Surface Properties Information ".center(71, '-'))
        print('Contact model: Hertz contact Model')
        print(f'MaterialID{matID1} < --- > MaterialID{matID2}')
        print('Effecitive Youngs Modulus: = ', self.YoungModulus)
        print('Effecitive Shear Modulus: = ', self.ShearModulus)
        print('Static friction coefficient = ', self.mus)
        print('Dynamic friction coefficient = ', self.mud)
        print('Restitution = ', self.restitution, '\n')

    @ti.func
    def _get_equivalent_stiffness(self, end1, end2, particle, wall):
        pos1, pos2 = particle[end1].x, wall[end2]._get_center()
        particle_rad, norm = particle[end1].rad, wall[end2].norm
        distance = (pos1 - pos2).dot(norm)
        gapn = distance - particle_rad
        contactAreaRadius = ti.sqrt(-gapn * particle_rad)
        fraction = ti.abs(wall[end2].processCircleShape(pos1, particle_rad, distance))
        return 2 * fraction * self.YoungModulus * contactAreaRadius
    
    @ti.func
    def _elastic_normal_energy(self, kn, normal_force):
        return 0.5 * normal_force * normal_force / kn 
    
    @ti.func
    def _viscous_normal_energy_rate(self, normal_damping_force, vn):
        return normal_damping_force * vn
    
    @ti.func
    def _elastic_tangential_energy(self, ks, tangOverTemp):
        return 0.5 * Squared(tangOverTemp) * ks
    
    @ti.func
    def _viscous_tangential_energy(self, tangential_damping_force, vs):
        return tangential_damping_force.dot(vs)
    
    @ti.func
    def _friction_energy(self, fric_ds, tangential_force):
        return fric_ds.dot(tangential_force)

    @ti.func
    def _normal_force(self, kn, restitution, m_eff, gapn, vn):
        normal_contact_force = -2./3. * kn * gapn 
        normal_damping_force = -1.8257 * restitution * vn * ti.sqrt(kn * m_eff) 
        norm_elastic, norm_viscous_rate = 0., 0.
        if ti.static(GlobalVariable.TRACKENERGY):
            norm_elastic = self._elastic_normal_energy(kn, normal_contact_force)
            norm_viscous_rate = self._viscous_normal_energy_rate(normal_damping_force, vn)
        return normal_contact_force + normal_damping_force, norm_elastic, norm_viscous_rate
    
    @ti.func
    def _tangential_force(self, ks, restitution, m_eff, normal_force, vs, norm, tangOverlapOld, dt):
        tangOverlapRot = tangOverlapOld - tangOverlapOld.dot(norm) * norm
        tangOverTemp = vs * dt[None] + tangOverlapOld.norm() * Normalize(tangOverlapRot)
        trial_ft = -ks * tangOverTemp
        
        cf = ti.abs(normal_force)
        tangential_force = ZEROVEC3f
        tang_elastic, tang_viscous_rate, friction_energy = 0., 0., 0.
        if trial_ft.norm() > self.mus * cf:
            tangential_force = self.mud * cf * trial_ft.normalized()
            tangOverTemp = -tangential_force / ks
            if ti.static(GlobalVariable.TRACKENERGY):
                tang_elastic = self._elastic_tangential_energy(ks, tangOverTemp)
                friction_energy = self._friction_energy(vs * dt[None], tangential_force)
        else:
            tang_damping_force = -1.8257 * restitution * vs * ti.sqrt(ks * m_eff)
            tangential_force = trial_ft + tang_damping_force
            if ti.static(GlobalVariable.TRACKENERGY):
                tang_elastic = self._elastic_tangential_energy(ks, tangOverTemp)
                tang_viscous_rate = self._viscous_tangential_energy(tang_damping_force, vs)
        return tangential_force, tangOverTemp, tang_elastic, tang_viscous_rate, friction_energy
    
    @ti.func
    def _force_assemble(self, m_eff, rad_eff, gapn, coeff, norm, v_rel, tangOverlapOld, dt):
        contactAreaRadius = ti.sqrt(-gapn * rad_eff)
        effective_E, effective_G = self.YoungModulus, self.ShearModulus
        restitution = self.restitution
        kn = 2 * effective_E * contactAreaRadius * coeff
        ks = 8 * effective_G * contactAreaRadius * coeff

        vn = v_rel.dot(norm)
        vs = v_rel - vn * norm
        
        normal_force, norm_elastic, norm_viscous_rate = self._normal_force(kn, restitution, m_eff, gapn, vn)
        tangential_force, tangOverTemp, tang_elastic, tang_viscous_rate, friction_energy = self._tangential_force(ks, restitution, m_eff, normal_force, vs, norm, tangOverlapOld, dt)
        return normal_force * norm, tangential_force, tangOverTemp, norm_elastic + tang_elastic, (norm_viscous_rate + tang_viscous_rate) * dt[None], friction_energy

