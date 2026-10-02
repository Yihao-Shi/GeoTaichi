import taichi as ti

from src.utils.constants import Threshold
import src.utils.GlobalVariable as GlobalVariable
from src.utils.VectorFunction import Normalize

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