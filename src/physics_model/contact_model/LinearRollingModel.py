import taichi as ti

from src.utils.constants import ZEROVEC3f, PI
import src.utils.GlobalVariable as GlobalVariable
from src.utils.VectorFunction import Normalize, Squared


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
    elastic_energy: float
    friction_energy: float
    damp_energy: float

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
        self.ncut = 0.0

    def print_surface_info(self, matID1, matID2):
        print(" Surface Properties Information ".center(71, "-"))
        print("Contact model: Linear Contact Model")
        print(f"MaterialID{matID1} < --- > MaterialID{matID2}")
        if GlobalVariable.ADAPTIVESTIFF:
            print("Effective modulus: = ", self.emod)
            print("Normal-to-shear stiffness ratio: = ", self.kratio)
        else:
            print("Contact normal stiffness: = ", self.kn)
            print("Contact tangential stiffness: = ", self.ks)
            print("Contact rolling stiffness: = ", self.kr)
            print("Contact twisting stiffness: = ", self.kt)
        print("Friction coefficient = ", self.mu)
        print("Rolling friction coefficient = ", self.rmu)
        print("Twisting friction coefficient = ", self.tmu)
        print("Viscous damping coefficient = ", self.ndratio)
        print("Viscous damping coefficient = ", self.sdratio)
        print("Viscous damping coefficient = ", self.rdratio)
        print("Viscous damping coefficient = ", self.tdratio, "\n")

    @ti.func
    def _get_equivalent_stiffness(self, end1, end2, particle, wall):
        pos1, pos2 = particle[end1].x, wall[end2]._get_center()
        particle_rad, norm = particle[end1].rad, wall[end2]._get_norm(pos1)
        distance = (pos1 - pos2).dot(norm)
        fraction = ti.abs(wall[end2].processCircleShape(pos1, particle_rad, distance))
        kn = self.kn
        if ti.static(GlobalVariable.ADAPTIVESTIFF):
            kn = PI * particle_rad * self.emod
        return fraction * kn

    @ti.func
    def _get_ls_equivalent_stiffness(self, parameter, end1, end2, rigid, particle, wall):
        if ti.static(GlobalVariable.ADAPTIVESTIFF):
            particle_rad = rigid[end1].equi_r
            kn = PI * particle_rad * self.emod * parameter
            return kn
        else:
            kn = self.kn * parameter
            return kn

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
    def _normal_force(self, kn, ndratio, m_eff, gapn, vn):
        normal_contact_force = -kn * gapn
        normal_damping_force = -2 * ndratio * ti.sqrt(m_eff * kn) * vn
        norm_elastic, norm_viscous_rate = 0.0, 0.0
        normal_force = normal_contact_force + normal_damping_force
        if normal_force < 0.0:
            normal_contact_force = 0.0
            normal_damping_force = 0.0
            normal_force = 0.0
        if ti.static(GlobalVariable.TRACKENERGY):
            norm_elastic = self._elastic_normal_energy(kn, normal_contact_force)
            norm_viscous_rate = self._viscous_normal_energy_rate(normal_damping_force, vn)
        return normal_force, norm_elastic, norm_viscous_rate

    @ti.func
    def _tangential_force(self, ks, miu, sdratio, m_eff, normal_force, vs, norm, tangOverlapOld, dt):
        tangOverlapRot = tangOverlapOld - tangOverlapOld.dot(norm) * norm
        tangOverTemp = vs * dt[None] + tangOverlapOld.norm() * Normalize(tangOverlapRot)
        tangOverlapTrial = tangOverTemp
        trial_ft = -ks * tangOverTemp

        fric = miu * ti.abs(normal_force)
        tangential_force = ZEROVEC3f
        tang_elastic, tang_viscous_rate, friction_energy = 0.0, 0.0, 0.0
        if trial_ft.norm() > fric:
            tangential_force = fric * trial_ft.normalized()
            tangOverTemp = -tangential_force / ks
            if ti.static(GlobalVariable.TRACKENERGY):
                tang_elastic = self._elastic_tangential_energy(ks, tangOverTemp)
                friction_energy = self._friction_energy(tangOverlapTrial - tangOverTemp, tangential_force)
        else:
            tang_damping_force = -2 * sdratio * ti.sqrt(m_eff * ks) * vs
            tangential_force = trial_ft + tang_damping_force
            if ti.static(GlobalVariable.TRACKENERGY):
                tang_elastic = self._elastic_tangential_energy(ks, tangOverTemp)
                tang_viscous_rate = self._viscous_tangential_energy(tang_damping_force, vs)
        return tangential_force, tangOverTemp, tang_elastic, tang_viscous_rate, friction_energy

    @ti.func
    def _rolling_force(self, kr, rmiu, rdratio, m_eff, rad_eff, normal_force, vr, norm, tangRollingOld, dt):
        tangRollingRot = tangRollingOld - tangRollingOld.dot(norm) * norm
        tangRollingTemp = vr * dt[None] + tangRollingOld.norm() * Normalize(tangRollingRot)
        tangRollingTrial = tangRollingTemp
        trial_fr = -kr * tangRollingTemp

        fricRoll = rmiu * ti.abs(normal_force)
        rolling_force = ZEROVEC3f
        rolling_elastic, rolling_viscous_rate, rolling_friction_energy = 0.0, 0.0, 0.0
        if trial_fr.norm() > fricRoll:
            rolling_force = fricRoll * trial_fr.normalized()
            tangRollingTemp = -rolling_force / kr
            if ti.static(GlobalVariable.TRACKENERGY):
                rolling_elastic = self._elastic_tangential_energy(kr, tangRollingTemp)
                rolling_friction_energy = self._friction_energy(tangRollingTrial - tangRollingTemp, rolling_force)
        else:
            rolling_damping_force = -2 * rdratio * ti.sqrt(m_eff * kr) * vr
            rolling_force = trial_fr + rolling_damping_force
            if ti.static(GlobalVariable.TRACKENERGY):
                rolling_elastic = self._elastic_tangential_energy(kr, tangRollingTemp)
                rolling_viscous_rate = self._viscous_tangential_energy(rolling_damping_force, vr)
        rolling_momentum = rad_eff * norm.cross(rolling_force)
        return rolling_momentum, tangRollingTemp, rolling_elastic, rolling_viscous_rate, rolling_friction_energy

    @ti.func
    def _twisting_force(self, kt, tmiu, tdratio, m_eff, rad_eff, normal_force, vt, norm, tangTwistingOld, dt):
        tangTwistingTemp = vt * dt[None] + tangTwistingOld.norm() * Normalize(norm)
        tangTwistingTrial = tangTwistingTemp
        trial_ft = -kt * tangTwistingTemp

        fricTwist = tmiu * ti.abs(normal_force)
        twisting_force = ZEROVEC3f
        twisting_elastic, twisting_viscous_rate, twisting_friction_energy = 0.0, 0.0, 0.0
        if trial_ft.norm() > fricTwist:
            twisting_force = fricTwist * trial_ft.normalized()
            tangTwistingTemp = -twisting_force / kt
            if ti.static(GlobalVariable.TRACKENERGY):
                twisting_elastic = self._elastic_tangential_energy(kt, tangTwistingTemp)
                twisting_friction_energy = self._friction_energy(tangTwistingTrial - tangTwistingTemp, twisting_force)
        else:
            twisting_damping_force = -2 * tdratio * ti.sqrt(m_eff * kt) * vt
            twisting_force = trial_ft + twisting_damping_force
            if ti.static(GlobalVariable.TRACKENERGY):
                twisting_elastic = self._elastic_tangential_energy(kt, tangTwistingTemp)
                twisting_viscous_rate = self._viscous_tangential_energy(twisting_damping_force, vt)
        twisting_momentum = rad_eff * twisting_force
        return twisting_momentum, tangTwistingTemp, twisting_elastic, twisting_viscous_rate, twisting_friction_energy

    @ti.func
    def _get_stiffness(self, coeff, param, rad_eff):
        if ti.static(GlobalVariable.ADAPTIVESTIFF):
            kn = PI * param * self.emod
            ks = kn / self.kratio
            kr = ks * rad_eff * rad_eff
            kt = 0.5 * kr
            return kn, ks, kr, kt
        else:
            return self.kn * coeff, self.ks * coeff, self.kr * coeff, self.kt * coeff

    @ti.func
    def _force_assemble(
        self,
        m_eff,
        rad_eff,
        gapn,
        coeff,
        param,
        norm,
        v_rel,
        w_rel,
        wr_rel,
        tangOverlapOld,
        tangRollingOld,
        tangTwistingOld,
        dt,
    ):
        kn, ks, kr, kt = self._get_stiffness(coeff, param, rad_eff)

        ndratio, sdratio = self.ndratio, self.sdratio
        miu = self.mu
        rdratio, tdratio = self.rdratio, self.tdratio
        rmiu, tmiu = self.rmu, self.tmu

        vn = v_rel.dot(norm)
        vs = v_rel - vn * norm

        normal_force, norm_elastic, norm_viscous_rate = self._normal_force(kn, ndratio, m_eff, gapn, vn)
        tangential_force, tangOverTemp, tang_elastic, tang_viscous_rate, friction_energy = self._tangential_force(
            ks, miu, sdratio, m_eff, normal_force, vs, norm, tangOverlapOld, dt
        )
        resisting_momentum, tangRollingTemp, tangTwistingTemp = ZEROVEC3f, ZEROVEC3f, ZEROVEC3f
        vt = rad_eff * (w_rel).dot(norm) * norm
        vr = -rad_eff * wr_rel
        rolling_momentum, tangRollingTemp, rolling_elastic, rolling_viscous_rate, rolling_friction_energy = (
            self._rolling_force(kr, rmiu, rdratio, m_eff, rad_eff, normal_force, vr, norm, tangRollingOld, dt)
        )
        twisting_momentum, tangTwistingTemp, twisting_elastic, twisting_viscous_rate, twisting_friction_energy = (
            self._twisting_force(kt, tmiu, tdratio, m_eff, rad_eff, normal_force, vt, norm, tangTwistingOld, dt)
        )
        resisting_momentum = rolling_momentum + twisting_momentum

        if ti.static(GlobalVariable.TRACKENERGY):
            self.elastic_energy += norm_elastic + tang_elastic + rolling_elastic + twisting_elastic
            if dt[None] > 0.0:
                self.friction_energy += friction_energy + rolling_friction_energy + twisting_friction_energy
                self.damp_energy += (
                    norm_viscous_rate + tang_viscous_rate + rolling_viscous_rate + twisting_viscous_rate
                ) * dt[None]
        if dt[None] == 0.0:
            tangOverTemp, tangRollingTemp, tangTwistingTemp = tangOverlapOld, tangRollingOld, tangTwistingOld
        return (
            normal_force * norm,
            tangential_force,
            resisting_momentum,
            tangOverTemp,
            tangRollingTemp,
            tangTwistingTemp,
        )
