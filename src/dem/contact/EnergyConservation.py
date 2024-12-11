import taichi as ti

from src.dem.contact.ContactKernel import *
from src.dem.contact.ContactModelBase import ContactModelBase
from src.dem.SceneManager import myScene
from src.utils.constants import ZEROVEC3f
from src.utils.ObjectIO import DictIO
from src.utils.VectorFunction import Normalize, Squared


class EnergyConservation(ContactModelBase):
    def __init__(self, sims, types) -> None:
        super().__init__(sims)
        if types == "Penalty":
            self.types = 1
            self.surfaceProps = PenaltyProperty.field(shape=self.sims.max_material_num * self.sims.max_material_num)
        elif types == "Barrier":
            self.types = 2
            self.surfaceProps = BarrierProperty.field(shape=self.sims.max_material_num * self.sims.max_material_num)

        self.null_model = False
        self.model_type = 0

    def calcu_critical_timestep(self, scene: myScene):
        mass = scene.find_particle_min_mass(self.sims.scheme)
        stiffness = self.find_max_stiffness(scene)
        return ti.sqrt(mass / stiffness)

    def find_max_stiffness(self, scene: myScene):
        maxstiff = 0.
        if self.types == 1:
            for materialID1 in range(self.sims.max_material_num):
                for materialID2 in range(self.sims.max_material_num):
                    componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
                    if self.surfaceProps[componousID].kn > 0.:
                        maxstiff = ti.max(ti.max(maxstiff, self.surfaceProps[componousID].kn), self.surfaceProps[componousID].ks)
        elif self.types == 2:
            for materialID1 in range(self.sims.max_material_num):
                for materialID2 in range(self.sims.max_material_num):
                    componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
                    if self.surfaceProps[componousID].kappa > 0.:
                        kappa = self.surfaceProps[componousID].kappa
                        ratio = kernel_get_min_ratio(componousID, int(scene.particleNum[0]), self.surfaceProps, scene.rigid)
                        kn = -kappa * (2. * ti.log(ratio) + ((ratio - 1) * (3 * ratio + 1)) / ratio ** 2)
                        maxstiff = ti.max(maxstiff, kn)
        return maxstiff
    
    def find_max_penetration(self):
        max_penetration = 0.
        if self.types == 2:
            for materialID1 in range(self.sims.max_material_num):
                for materialID2 in range(self.sims.max_material_num):
                    componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
                    if self.surfaceProps[componousID].kappa > 0.:
                        max_penetration = ti.max(max_penetration, self.surfaceProps[componousID].ncut)
        return max_penetration
    
    def add_surface_property(self, materialID1, materialID2, property):
        if self.types == 1:
            kn = DictIO.GetEssential(property, 'NormalStiffness')
            ks = DictIO.GetEssential(property, 'TangentialStiffness')
            mu = DictIO.GetEssential(property, 'Friction')
            theta = DictIO.GetAlternative(property, 'FreeParameter', 2.5)
            ndratio = DictIO.GetEssential(property, 'NormalViscousDamping')
            sdratio = DictIO.GetEssential(property, 'TangentialViscousDamping')
            componousID = 0
            if materialID1 == materialID2:
                componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
                self.surfaceProps[componousID].add_surface_property(kn, ks, theta, mu, ndratio, sdratio)
            else:
                componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
                self.surfaceProps[componousID].add_surface_property(kn, ks, theta, mu, ndratio, sdratio)
                componousID = self.get_componousID(self.sims.max_material_num, materialID2, materialID1)
                self.surfaceProps[componousID].add_surface_property(kn, ks, theta, mu, ndratio, sdratio)
            return componousID
        elif self.types == 2:
            kappa = DictIO.GetEssential(property, 'Stiffness')
            ncut = DictIO.GetEssential(property, 'NormalCutOff')
            ratio = DictIO.GetAlternative(property, 'StiffnessRatio', 1.)
            mu = DictIO.GetEssential(property, 'Friction')
            ndratio = DictIO.GetEssential(property, 'NormalViscousDamping')
            sdratio = DictIO.GetEssential(property, 'TangentialViscousDamping')
            componousID = 0
            if materialID1 == materialID2:
                componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
                self.surfaceProps[componousID].add_surface_property(kappa, ncut, ratio, mu, ndratio, sdratio)
            else:
                componousID = self.get_componousID(self.sims.max_material_num, materialID1, materialID2)
                self.surfaceProps[componousID].add_surface_property(kappa, ncut, ratio, mu, ndratio, sdratio)
                componousID = self.get_componousID(self.sims.max_material_num, materialID2, materialID1)
                self.surfaceProps[componousID].add_surface_property(kappa, ncut, ratio, mu, ndratio, sdratio)
            return componousID
    
    def update_property(self, componousID, property_name, value, override):
        factor = 0
        if not override:
            factor = 1

        if property_name == "NormalStiffness":
            self.surfaceProps[componousID].kn = factor * self.surfaceProps[componousID].kn + value
        elif property_name == "TangentialStiffness":
            self.surfaceProps[componousID].ks = factor * self.surfaceProps[componousID].ks + value
        elif property_name == "Friction":
            self.surfaceProps[componousID].mu = factor * self.surfaceProps[componousID].mu + value
        elif property_name == "NormalViscousDamping":
            self.surfaceProps[componousID].ndratio = factor * self.surfaceProps[componousID].ndratio + value
        elif property_name == "TangentialViscousDamping":
            self.surfaceProps[componousID].sdratio = factor * self.surfaceProps[componousID].sdratio + value
    


@ti.dataclass
class PenaltyProperty:
    kn: float
    ks: float
    mu: float
    ncut: float
    theta: float
    ndratio: float
    sdratio: float

    def add_surface_property(self, kn, ks, theta, mu, ndratio, sdratio):
        self.kn = kn
        self.ks = ks
        self.theta = theta
        self.mu = mu
        self.ndratio = ndratio
        self.sdratio = sdratio
        self.ncut = 0.

    def print_surface_info(self, matID1, matID2):
        print(" Surface Properties Information ".center(71, '-'))
        print('Contact model: Energy Conservation Contact Model')
        print(f'MaterialID{matID1} < --- > MaterialID{matID2}')
        print('Contact normal stiffness: = ', self.kn)
        print('Contact tangential stiffness: = ', self.ks)
        print('Free parameter: = ', self.theta)
        print('Friction coefficient = ', self.mu)
        print('Viscous damping coefficient = ', self.ndratio)
        print('Viscous damping coefficient = ', self.sdratio, '\n')

    @ti.func
    def _get_equivalent_stiffness(self, end1, end2, particle, wall):
        pos1, pos2 = particle[end1].x, wall[end2]._get_center()
        particle_rad, norm = particle[end1].rad, wall[end2].norm
        distance = (pos1 - pos2).dot(norm)
        fraction = ti.abs(wall[end2].processCircleShape(pos1, particle_rad, distance))
        return fraction * self.kn
    
    @ti.func
    def _elastic_normal_energy(self, kn, gapn):
        return 1./self.theta * kn * (-gapn) ** self.theta
    
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
    def _normal_force(self, kn, ndratio, m_eff, gapn, vn):
        normal_contact_force = kn * (-gapn) ** (self.theta - 1)
        normal_damping_force = -2 * ndratio * ti.sqrt(m_eff * kn) * vn
        norm_elastic, norm_viscous_rate = 0., 0.
        if ti.static(GlobalVariable.TRACKENERGY):
            norm_elastic = self._elastic_normal_energy(kn, gapn)
            norm_viscous_rate = self._viscous_normal_energy_rate(normal_damping_force, vn)
        return normal_contact_force + normal_damping_force, norm_elastic, norm_viscous_rate
    
    @ti.func
    def _tangential_force(self, ks, sdratio, m_eff, vs, normal_force, norm, tangOverlapOld, dt):
        tangOverlapRot = tangOverlapOld - tangOverlapOld.dot(norm) * norm
        tangOverTemp = vs * dt[None] + tangOverlapOld.norm() * Normalize(tangOverlapRot)
        trial_ft = -ks * tangOverTemp
        
        fric = self.mu * ti.abs(normal_force)
        tangential_force = ZEROVEC3f
        tang_elastic, tang_viscous_rate, friction_energy = 0., 0., 0.
        if trial_ft.norm() > fric:
            tangential_force = fric * trial_ft.normalized()
            tangOverTemp = -tangential_force / ks
            if ti.static(GlobalVariable.TRACKENERGY):
                tang_elastic = self._elastic_tangential_energy(ks, tangOverTemp)
                friction_energy = self._friction_energy(vs * dt[None], tangential_force)
        else:
            tang_damping_force = -2 * sdratio * ti.sqrt(m_eff * ks) * vs
            tangential_force = trial_ft + tang_damping_force
            if ti.static(GlobalVariable.TRACKENERGY):
                tang_elastic = self._elastic_tangential_energy(ks, tangOverTemp)
                tang_viscous_rate = self._viscous_tangential_energy(tang_damping_force, vs)
        return tangential_force, tangOverTemp, tang_elastic, tang_viscous_rate, friction_energy
    
    @ti.func
    def _force_assemble(self, m_eff, rad_eff, gapn, coeff, dgdx, v_rel, tangOverlapOld, dt):
        coeff *= dgdx.norm()
        kn, ks = self.kn * coeff, self.ks * coeff
        ndratio, sdratio = self.ndratio, self.sdratio
        norm = dgdx.normalized(Threshold)
        vn = v_rel.dot(norm) 
        vs = v_rel - vn * norm

        normal_force, norm_elastic, norm_viscous_rate = self._normal_force(kn, ndratio, m_eff, gapn, vn)
        tangential_force, tangOverTemp, tang_elastic, tang_viscous_rate, friction_energy = self._tangential_force(ks, sdratio, m_eff, vs, normal_force, norm, tangOverlapOld, dt)
        return normal_force * norm, tangential_force, tangOverTemp, norm_elastic + tang_elastic, (norm_viscous_rate + tang_viscous_rate) * dt[None], friction_energy
    

@ti.dataclass
class BarrierProperty:
    kappa: float
    mu: float
    ncut: float
    ratio: float
    ndratio: float
    sdratio: float

    def add_surface_property(self, kappa, ncut, ratio, mu, ndratio, sdratio):
        self.kappa = kappa
        self.ncut = ncut
        self.ratio = ratio
        self.mu = mu
        self.ndratio = ndratio
        self.sdratio = sdratio

    def print_surface_info(self, matID1, matID2):
        print(" Surface Properties Information ".center(71, '-'))
        print('Contact model: Barrier Contact Model')
        print(f'MaterialID{matID1} < --- > MaterialID{matID2}')
        print('Scalar parameter: = ', self.kappa)
        print('Maximum value of gap distance: = ', self.ncut)
        print('Normal to tangental stiffness ratio: = ', self.ratio)
        print('Friction coefficient = ', self.mu)
        print('Viscous damping coefficient = ', self.ndratio)
        print('Viscous damping coefficient = ', self.sdratio, '\n')

    @ti.func
    def _get_equivalent_stiffness(self, end1, end2, particle, wall):
        pos1, pos2 = particle[end1].x, wall[end2]._get_center()
        particle_rad, norm = particle[end1].rad, wall[end2].norm
        distance = (pos1 - pos2).dot(norm)
        gapn = distance - particle_rad
        fraction = ti.abs(wall[end2].processCircleShape(pos1, particle_rad, distance))
        return fraction * (-2 * self.kappa * ti.log(-gapn / self.ncut) - self.kappa * (-gapn - self.ncut) * (3. * -gapn + self.ncut) / (gapn * gapn))
    
    @ti.func
    def _elastic_normal_energy(self, kn, eta, d_cap):
        return -kn * (eta - d_cap) * (eta - d_cap) * ti.log(eta / d_cap)
    
    @ti.func
    def _viscous_normal_energy_rate(self, normal_damping_force, vn):
        return normal_damping_force * vn
    
    @ti.func
    def _elastic_tangential_energy(self, xs, tangential_force):
        return xs.dot(tangential_force)
    
    @ti.func
    def _viscous_tangential_energy(self, tangential_damping_force, vs):
        return tangential_damping_force.dot(vs)
    
    @ti.func
    def _friction_energy(self, fric_ds, tangential_force):
        return fric_ds.dot(tangential_force)

    @ti.func
    def _normal_force(self, kn, ndratio, m_eff, eta, d_cap, vn, coeff):
        normal_contact_force = self.kappa * coeff * (eta - d_cap) * (2. * ti.log(eta / d_cap) - d_cap / eta + 1)
        normal_damping_force = -2 * ndratio * ti.sqrt(m_eff * kn) * vn
        norm_elastic, norm_viscous_rate = 0., 0.
        if ti.static(GlobalVariable.TRACKENERGY):
            norm_elastic = self._elastic_normal_energy(self.kappa * coeff, eta, d_cap)
            norm_viscous_rate = self._viscous_normal_energy_rate(normal_damping_force, vn)
        return normal_contact_force + normal_damping_force, norm_elastic, norm_viscous_rate
    
    @ti.func
    def _tangential_force(self, coeff, sdratio, m_eff, vs, normal_force, norm, tangOverlapOld, dt):
        m = 1.
        cut_off = self.mu * ti.abs(normal_force) / (self.ratio * self.kappa * coeff)
        tangOverlapRot = tangOverlapOld - tangOverlapOld.dot(norm) * norm
        tangOverTemp = vs * dt[None] + tangOverlapOld.norm() * Normalize(tangOverlapRot)
        overlap_tang = tangOverTemp.norm()
        overlap_dir = Normalize(tangOverTemp)
        tangential_force = -self.mu * ti.abs(normal_force) * overlap_dir
        tang_elastic, tang_viscous_rate, friction_energy = 0., 0., 0.
        if overlap_tang < cut_off:
            m = overlap_tang * (2. * cut_off - overlap_tang) / (cut_off * cut_off)
            tangential_force *= m
            eqkt = tangential_force.norm() / overlap_tang if overlap_tang != 0. else 0.
            tang_damping_force = -2 * sdratio * ti.sqrt(m_eff * eqkt) * vs
            if ti.static(GlobalVariable.TRACKENERGY):
                tang_elastic = self._elastic_tangential_energy(tangOverTemp, tangential_force)
                tang_viscous_rate = self._viscous_tangential_energy(tang_damping_force, vs)
            tangential_force += tang_damping_force
        else:
            tangOverNew = cut_off * overlap_dir
            if ti.static(GlobalVariable.TRACKENERGY):
                tang_elastic = self._elastic_tangential_energy(tangOverNew, tangential_force)
                friction_energy = self._friction_energy(tangOverTemp - tangOverNew, tangential_force)
            tangOverTemp = tangOverNew
        return tangential_force, tangOverTemp, tang_elastic, tang_viscous_rate, friction_energy

    @ti.func
    def _force_assemble(self, m_eff, rad_eff, gapn, coeff, dgdx, v_rel, tangOverlapOld, dt):
        coeff = coeff * dgdx.norm()
        eta = gapn + self.ncut
        d_cap = 2. * self.ncut
        
        kn = -2 * coeff * self.kappa * ti.log(eta / d_cap) - self.kappa * (eta - d_cap) * (3. * eta + d_cap) / (eta * eta)
        ndratio, sdratio = self.ndratio, self.sdratio
        norm = dgdx.normalized(Threshold)
        vn = v_rel.dot(norm)
        vs = v_rel - vn * norm

        normal_force, norm_elastic, norm_viscous_rate = self._normal_force(kn, ndratio, m_eff, eta, d_cap, vn, coeff)
        tangential_force, tangOverTemp, tang_elastic, tang_viscous_rate, friction_energy = self._tangential_force(coeff, sdratio, m_eff, vs, normal_force, norm, tangOverlapOld, dt)
        return normal_force * norm, tangential_force, tangOverTemp, norm_elastic + tang_elastic, (norm_viscous_rate + tang_viscous_rate) * dt[None], friction_energy

