import taichi as ti

from src.utils.constants import ZEROVEC3f, Threshold
import src.utils.GlobalVariable as GlobalVariable
from src.utils.VectorFunction import Squared, Normalize


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
    