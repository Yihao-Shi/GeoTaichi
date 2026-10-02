import taichi as ti

from src.utils.constants import ZEROVEC3f, PI
import src.utils.GlobalVariable as GlobalVariable
from src.utils.VectorFunction import Normalize


@ti.dataclass
class LinearRollingSurfaceProperty:
    kn: float
    ks: float
    kr: float
    kt: float
    emod: float
    kratio: float
    mu: float
    rmu: float
    tmu: float
    ndratio: float
    sdratio: float
    rdratio: float
    tdratio: float
    ncut: float

    def add_surface_property(self, kn, ks, kr, kt, emod, kratio, mu, rmu, tmu, ndratio, sdratio, rdratio, tdratio):
        self.kn = kn
        self.ks = ks
        self.kr = kr
        self.kt = kt
        self.emod = emod
        self.kratio = kratio
        self.mu = mu
        self.rmu = rmu
        self.tmu = tmu
        self.ndratio = ndratio
        self.sdratio = sdratio
        self.rdratio = rdratio
        self.tdratio = tdratio
        self.ncut = 0.

    def print_surface_info(self, matID1, matID2):
        print(" Surface Properties Information ".center(71, '-'))
        print('Contact model: Linear Contact Model')
        print(f'MaterialID{matID1} < --- > MaterialID{matID2}')
        if GlobalVariable.ADAPTIVESTIFF:
            print('Effective modulus: = ', self.emod)
            print('Normal-to-shear stiffness ratio: = ', self.kratio)
        else:
            print('Contact normal stiffness: = ', self.kn)
            print('Contact tangential stiffness: = ', self.ks)
            print('Contact rolling stiffness: = ', self.kr)
            print('Contact twisting stiffness: = ', self.kt)
        print('Friction coefficient = ', self.mu)
        print('Rolling friction coefficient = ', self.rmu)
        print('Twisting friction coefficient = ', self.tmu)
        print('Viscous damping coefficient = ', self.ndratio)
        print('Viscous damping coefficient = ', self.sdratio)
        print('Viscous damping coefficient = ', self.rdratio)
        print('Viscous damping coefficient = ', self.tdratio, '\n')

    @ti.func
    def _get_equivalent_stiffness(self, end1, end2, particle, wall):
        pos1, pos2 = particle[end1].x, wall[end2]._get_center()
        particle_rad, norm = particle[end1].rad, wall[end2].norm
        distance = (pos1 - pos2).dot(norm)
        fraction = ti.abs(wall[end2].processCircleShape(pos1, particle_rad, distance))
        kn = self.kn
        if ti.static(GlobalVariable.ADAPTIVESTIFF):
            kn = PI * particle_rad * self.emod
        return fraction * kn
    
    @ti.func
    def _normal_force(self, kn, ndratio, m_eff, gapn, vn):
        normal_contact_force = -kn * gapn 
        normal_damping_force = -2 * ndratio * ti.sqrt(m_eff * kn) * vn
        return normal_contact_force + normal_damping_force
    
    @ti.func
    def _tangential_force(self, ks, miu, sdratio, m_eff, normal_force, vs, norm, tangOverlapOld, dt):
        tangOverlapRot = tangOverlapOld - tangOverlapOld.dot(norm) * norm
        tangOverTemp = vs * dt[None] + tangOverlapOld.norm() * Normalize(tangOverlapRot)
        trial_ft = -ks * tangOverTemp
        tang_damping_force = -2 * sdratio * ti.sqrt(m_eff * ks) * vs
        
        fric = miu * ti.abs(normal_force)
        tangential_force = ZEROVEC3f
        if trial_ft.norm() > fric:
            tangential_force = fric * trial_ft.normalized()
            tangOverTemp = -tangential_force / ks
        else:
            tangential_force = trial_ft + tang_damping_force
        return tangential_force, tangOverTemp
    
    @ti.func
    def _rolling_force(self, kr, rmiu, rdratio, m_eff, rad_eff, normal_force, vr, norm, tangRollingOld, dt):
        tangRollingRot = tangRollingOld - tangRollingOld.dot(norm) * norm
        tangRollingTemp = vr * dt[None] + tangRollingOld.norm() * Normalize(tangRollingRot)
        trial_fr = -kr * tangRollingTemp
        rolling_damping_force = -2 * rdratio * ti.sqrt(m_eff * kr) * vr
        
        fricRoll = rmiu * ti.abs(normal_force)
        rolling_force = ZEROVEC3f
        if trial_fr.norm() > fricRoll:
            rolling_force = fricRoll * trial_fr.normalized()
            tangRollingTemp = -rolling_force / kr
        else:
            rolling_force = trial_fr + rolling_damping_force
        rolling_momentum = rad_eff * norm.cross(rolling_force)
        return rolling_momentum, tangRollingTemp

    @ti.func
    def _twisting_force(self, kt, tmiu, tdratio, m_eff, rad_eff, normal_force, vt, norm, tangTwistingOld, dt):
        tangTwistingTemp = vt * dt[None] + tangTwistingOld.norm() * Normalize(norm)
        trial_ft = -kt * tangTwistingTemp
        twisting_damping_force = -2 * tdratio * ti.sqrt(m_eff * kt) * vt
        
        fricTwist = tmiu * ti.abs(normal_force)
        twisting_force = ZEROVEC3f
        if trial_ft.norm() > fricTwist:
            twisting_force = fricTwist * trial_ft.normalized()
            tangTwistingTemp = -twisting_force / kt
        else:
            twisting_force = trial_ft + twisting_damping_force
        twisting_momentum = rad_eff * twisting_force
        return twisting_momentum, tangTwistingTemp
    
    @ti.func
    def _get_stiffness(self, coeff, param, rad_eff):
        if ti.static(GlobalVariable.ADAPTIVESTIFF):
            kn = PI * param * self.emod
            ks = kn / self.kratio
            kr = ks * rad_eff
            kt = 2. * kr
            return kn, ks, kr, kt
        else:
            return self.kn * coeff, self.ks * coeff, self.kr * coeff, self.kt * coeff
    
    @ti.func
    def _force_assemble(self, m_eff, rad_eff, gapn, coeff, param, norm, v_rel, w_rel, wr_rel, tangOverlapOld, tangRollingOld, tangTwistingOld, dt):
        kn, ks, kr, kt = self._get_stiffness(coeff, param, rad_eff)
        
        ndratio, sdratio = self.ndratio, self.sdratio
        miu = self.mu
        rdratio, tdratio = self.rdratio, self.tdratio
        rmiu, tmiu = self.rmu, self.tmu

        vn = v_rel.dot(norm) 
        vs = v_rel - vn * norm
        
        normal_force = self._normal_force(kn, ndratio, m_eff, gapn, vn)
        tangential_force, tangOverTemp = self._tangential_force(ks, miu, sdratio, m_eff, normal_force, vs, norm, tangOverlapOld, dt)
        resisting_momentum, tangRollingTemp, tangTwistingTemp = ZEROVEC3f, ZEROVEC3f, ZEROVEC3f
        vt = rad_eff * (w_rel).dot(norm) * norm
        vr = -rad_eff * wr_rel
        rolling_momentum, tangRollingTemp = self._rolling_force(kr, rmiu, rdratio, m_eff, rad_eff, normal_force, vr, norm, tangRollingOld, dt)
        twisting_momentum, tangTwistingTemp = self._twisting_force(kt, tmiu, tdratio, m_eff, rad_eff, normal_force, vt, norm, tangTwistingOld, dt)
        resisting_momentum = rolling_momentum + twisting_momentum
        return normal_force * norm, tangential_force, resisting_momentum, tangOverTemp, tangRollingTemp, tangTwistingTemp