import taichi as ti

from src.utils.constants import ZEROVEC3f
from src.utils.VectorFunction import Normalize


@ti.dataclass
class JiangRollingSurfaceProperty:
    YoungModulus: float
    stiffness_ratio: float
    shape_factor: float
    crush_factor: float
    mu: float
    ndratio: float
    sdratio: float
    ncut: float

    def add_surface_property(self, YoungModulus, stiffness_ratio, mu, shape_factor, crush_factor, ndratio, sdratio):
        self.YoungModulus = YoungModulus
        self.stiffness_ratio = stiffness_ratio
        self.mu = mu
        self.shape_factor = shape_factor
        self.crush_factor = crush_factor
        self.ndratio = ndratio
        self.sdratio = sdratio
        self.ncut = 0.

    def print_surface_info(self, matID1, matID2):
        print(" Surface Properties Information ".center(71, '-'))
        print('Contact model: Linear Rolling Resistance Contact Model')
        print(f'MaterialID{matID1} < --- > MaterialID{matID2}')
        print('Youngs Modulus: = ', self.YoungModulus)
        print('Stiffness Ratio: = ', self.stiffness_ratio)
        print('Shape Factor = ', self.shape_factor)
        print('Crush Factor = ', self.crush_factor)
        print('Friction coefficient = ', self.mu)
        print('Viscous damping coefficient = ', self.ndratio)
        print('Viscous damping coefficient = ', self.sdratio, '\n')

    @ti.func
    def _get_equivalent_stiffness(self, end1, end2, particle, wall):
        pos1, pos2 = particle[end1].x, wall[end2]._get_center()
        particle_rad, norm = particle[end1].rad, wall[end2].norm
        distance = (pos1 - pos2).dot(norm)
        fraction = ti.abs(wall[end2].processCircleShape(pos1, particle_rad, distance))
        return 2 * fraction * particle_rad * self.YoungModulus
    
    @ti.func
    def _normal_force(self, kn, ndratio, gapn, vn):
        normal_contact_force = -kn * gapn 
        normal_damping_force = -ndratio * vn
        return normal_contact_force + normal_damping_force
    
    @ti.func
    def _tangential_force(self, ks, miu, sdratio, normal_force, vs, norm, tangOverlapOld, dt):
        tangOverlapRot = tangOverlapOld - tangOverlapOld.dot(norm) * norm
        tangOverTemp = vs * dt[None] + tangOverlapOld.norm() * Normalize(tangOverlapRot)
        trial_ft = -ks * tangOverTemp
        tang_damping_force = -sdratio * vs
        
        fric = miu * ti.abs(normal_force)
        tangential_force = ZEROVEC3f
        if trial_ft.norm() > fric:
            tangential_force = fric * trial_ft.normalized()
            tangOverTemp = -tangential_force / ks
        else:
            tangential_force = trial_ft + tang_damping_force
        return tangential_force, tangOverTemp
    
    @ti.func
    def _rolling_force(self, kr, rmiu, rdratio, normal_force, wr, norm, tangRollingOld, dt):
        tangRollingRot = tangRollingOld - tangRollingOld.dot(norm) * norm
        tangRollingTemp = wr * dt[None] + tangRollingOld.norm() * Normalize(tangRollingRot)
        trial_fr = -kr * tangRollingTemp
        rolling_damping_force = -rdratio * wr
        
        fricRoll = rmiu * ti.abs(normal_force)
        rolling_momentum = ZEROVEC3f
        if trial_fr.norm() > fricRoll:
            rolling_momentum = fricRoll * trial_fr.normalized()
            tangRollingTemp = -rolling_momentum / kr
        else:
            rolling_momentum = trial_fr + rolling_damping_force
        return rolling_momentum, tangRollingTemp

    @ti.func
    def _twisting_force(self, kt, tmiu, tdratio, normal_force, wt, norm, tangTwistingOld, dt):
        tangTwistingTemp = wt * dt[None] + tangTwistingOld.norm() * Normalize(norm)
        trial_ft = -kt * tangTwistingTemp
        twisting_damping_force = -tdratio * wt
        
        fricTwist = tmiu * ti.abs(normal_force)
        twisting_momentum = ZEROVEC3f
        if trial_ft.norm() > fricTwist:
            twisting_momentum = fricTwist * trial_ft.normalized()
            tangTwistingTemp = -twisting_momentum / kt
        else:
            twisting_momentum = trial_ft + twisting_damping_force
        return twisting_momentum, tangTwistingTemp

    @ti.func
    def _force_assemble(self, m_eff, rad_eff, gapn, coeff, param, norm, v_rel, w_rel, wr_rel, tangOverlapOld, tangRollingOld, tangTwistingOld, dt):
        YoungModulus, stiffness_ratio = self.YoungModulus, self.stiffness_ratio
        shape_factor, crush_factor = self.shape_factor, self.crush_factor
        ndratio = self.ndratio * 2 * ti.sqrt(m_eff * kn)
        sdratio = self.sdratio * 2 * ti.sqrt(m_eff * ks)
        miu = self.mu
        kn = 2 * rad_eff * YoungModulus * coeff
        ks = kn * stiffness_ratio

        RBar = shape_factor * rad_eff
        SquareR = RBar * RBar
        kr = 0.25 * kn * SquareR
        kt = 0.5 * ks * SquareR
        rdratio = 0.25 * ndratio * SquareR
        tdratio = 0.5 * sdratio * SquareR
        rmiu = 0.25 * RBar * crush_factor
        tmiu = 0.65 * RBar * miu

        vn = v_rel.dot(norm) 
        vs = v_rel - vn * norm
        wt = (w_rel).dot(norm) * norm
        wr = w_rel - wt 

        normal_force = self._normal_force(kn, ndratio, gapn, vn)
        tangential_force, tangOverTemp = self._tangential_force(ks, miu, sdratio, normal_force, vs, norm, tangOverlapOld, dt)
        rolling_momentum, tangRollingTemp = self._rolling_force(kr, rmiu, rdratio, normal_force, wr, norm, tangRollingOld, dt)
        twisting_momentum, tangTwistingTemp = self._twisting_force(kt, tmiu, tdratio, normal_force, wt, norm, tangTwistingOld, dt)
        return normal_force * norm, tangential_force, rolling_momentum + twisting_momentum, tangOverTemp, tangRollingTemp, tangTwistingTemp
