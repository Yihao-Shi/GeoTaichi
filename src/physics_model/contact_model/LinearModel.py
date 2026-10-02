import taichi as ti

from src.utils.constants import ZEROVEC3f, PI
import src.utils.GlobalVariable as GlobalVariable
from src.utils.TypeDefination import real
from src.utils.VectorFunction import Normalize, Squared


@ti.dataclass
class LinearSurfaceProperty:
    kn: real
    ks: real
    emod: real
    kratio: real
    mus: real
    mud: real
    rmu: real
    ndratio: real
    sdratio: real
    ncut: real
    elastic_energy: real
    friction_energy: real
    damp_energy: real

    def add_surface_property(self, kn, ks, emod, kratio, mus, mud, rmu, ndratio, sdratio):
        self.kn = kn
        self.ks = ks
        self.emod = emod
        self.kratio = kratio
        self.mus = mus
        self.mud = mud
        self.rmu = rmu
        self.ndratio = ndratio
        self.sdratio = sdratio
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
        print("Static friction coefficient = ", self.mus)
        print("Dynamic friction coefficient = ", self.mud)
        if self.rmu > 0.0:
            print("Rolling friction coefficient = ", self.rmu)
        print("Viscous damping coefficient = ", self.ndratio)
        print("Viscous damping coefficient = ", self.sdratio, "\n")

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
        normal_contact_force = kn * (-gapn)
        normal_damping_force = -2 * ndratio * ti.sqrt(m_eff * kn) * vn
        norm_elastic, norm_viscous_rate = 0.0, 0.0
        normal_force = normal_contact_force + normal_damping_force
        if normal_force < 0.0:
            # A unilateral Kelvin--Voigt contact cannot carry tension.  Keep
            # the compressed spring in the energy decomposition and cap the
            # dashpot at the equal-and-opposite value, so the total force is
            # still exactly zero.  During subsequent unloading the decrease
            # of spring energy is then recorded as viscous work instead of
            # disappearing when the tensile trial force is clipped.
            normal_damping_force = -normal_contact_force
            normal_force = 0.0
        if ti.static(GlobalVariable.TRACKENERGY):
            norm_elastic = self._elastic_normal_energy(kn, normal_contact_force)
            norm_viscous_rate = self._viscous_normal_energy_rate(normal_damping_force, vn)
        return normal_force, norm_elastic, norm_viscous_rate

    @ti.func
    def _tangential_force(self, ks, sdratio, m_eff, normal_force, vs, norm, tangOverlapOld, dt):
        tangOverlapRot = tangOverlapOld - tangOverlapOld.dot(norm) * norm
        tangOverTemp = vs * dt[None] + tangOverlapOld.norm() * Normalize(tangOverlapRot)
        tangOverlapTrial = tangOverTemp
        trial_ft = -ks * tangOverTemp

        cf = ti.abs(normal_force)
        tangential_force = ZEROVEC3f
        tang_elastic, tang_viscous_rate, friction_energy = 0.0, 0.0, 0.0
        if trial_ft.norm() > self.mus * cf:
            tangential_force = self.mud * cf * trial_ft.normalized()
            tangOverTemp = -tangential_force / ks
            if ti.static(GlobalVariable.TRACKENERGY):
                tang_elastic = self._elastic_tangential_energy(ks, tangOverTemp)
                plastic_slip = tangOverlapTrial - tangOverTemp
                friction_energy = self._friction_energy(plastic_slip, tangential_force)
        else:
            tang_damping_force = -2 * sdratio * ti.sqrt(m_eff * ks) * vs
            tangential_force = trial_ft + tang_damping_force
            if ti.static(GlobalVariable.TRACKENERGY):
                tang_elastic = self._elastic_tangential_energy(ks, tangOverTemp)
                tang_viscous_rate = self._viscous_tangential_energy(tang_damping_force, vs)
        return tangential_force, tangOverTemp, tang_elastic, tang_viscous_rate, friction_energy

    @ti.func
    def _resisting_momentum(self, rad_eff, normal_force, w_rel):
        return -Normalize(w_rel) * self.rmu * ti.abs(normal_force) * rad_eff

    @ti.func
    def _get_stiffness(self, coeff, param):
        if ti.static(GlobalVariable.ADAPTIVESTIFF):
            kn = PI * param * self.emod * coeff
            ks = kn / self.kratio
            return kn, ks
        else:
            return self.kn * coeff, self.ks * coeff

    @ti.func
    def _force_assemble(self, m_eff, rad_eff, gapn, coeff, param, norm, v_rel, w_rel, tangOverlapOld, dt):
        kn, ks = self._get_stiffness(coeff, param)

        ndratio, sdratio = self.ndratio, self.sdratio
        vn = v_rel.dot(norm)
        vs = v_rel - vn * norm

        normal_force, norm_elastic, norm_viscous_rate = self._normal_force(kn, ndratio, m_eff, gapn, vn)
        tangential_force, tangOverTemp, tang_elastic, tang_viscous_rate, friction_energy = self._tangential_force(
            ks, sdratio, m_eff, normal_force, vs, norm, tangOverlapOld, dt
        )
        resisting_momentum = ZEROVEC3f
        if ti.static(GlobalVariable.CONSTANTORQUEMODEL):
            resisting_momentum = self._resisting_momentum(rad_eff, normal_force, w_rel)
        if ti.static(GlobalVariable.TRACKENERGY):
            self.elastic_energy += norm_elastic + tang_elastic
            if dt[None] > 0.0:
                self.friction_energy += friction_energy
                self.damp_energy += (norm_viscous_rate + tang_viscous_rate) * dt[None]
        # A zero-duration force probe may project/cap the trial spring, but
        # must not commit that return mapping to contact history.
        if dt[None] == 0.0:
            tangOverTemp = tangOverlapOld
        return normal_force * norm, tangential_force, resisting_momentum, tangOverTemp
