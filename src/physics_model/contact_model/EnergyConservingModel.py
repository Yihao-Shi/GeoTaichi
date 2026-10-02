import taichi as ti

from src.utils.constants import ZEROVEC3f, Threshold
from src.utils.TypeDefination import real
import src.utils.GlobalVariable as GlobalVariable
from src.utils.VectorFunction import Squared, Normalize


@ti.dataclass
class PenaltyProperty:
    kn: real
    ks: real
    mu: real
    ncut: real
    theta: real
    ndratio: real
    sdratio: real
    elastic_energy: real
    friction_energy: real
    damp_energy: real

    def add_surface_property(self, kn, ks, theta, mu, ndratio, sdratio):
        self.kn = kn
        self.ks = ks
        self.theta = theta
        self.mu = mu
        self.ndratio = ndratio
        self.sdratio = sdratio
        self.ncut = 0.0

    def print_surface_info(self, matID1, matID2):
        print(" Surface Properties Information ".center(71, "-"))
        print("Contact model: Energy Conservation Contact Model")
        print(f"MaterialID{matID1} < --- > MaterialID{matID2}")
        print("Contact normal stiffness: = ", self.kn)
        print("Contact tangential stiffness: = ", self.ks)
        print("Free parameter: = ", self.theta)
        print("Friction coefficient = ", self.mu)
        print("Viscous damping coefficient = ", self.ndratio)
        print("Viscous damping coefficient = ", self.sdratio, "\n")

    @ti.func
    def _get_equivalent_stiffness(self, end1, end2, particle, wall):
        pos1, pos2 = particle[end1].x, wall[end2]._get_center()
        particle_rad, norm = particle[end1].rad, wall[end2]._get_norm(pos1)
        distance = (pos1 - pos2).dot(norm)
        fraction = ti.abs(wall[end2].processCircleShape(pos1, particle_rad, distance))
        return fraction * self.kn

    @ti.func
    def _get_ls_equivalent_stiffness(self, parameter, end1, end2, rigid, particle, wall):
        pass

    @ti.func
    def _elastic_normal_energy(self, kn, gapn):
        return 1.0 / self.theta * kn * (-gapn) ** self.theta

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
        norm_elastic, norm_viscous_rate = 0.0, 0.0
        normal_force = normal_contact_force + normal_damping_force
        if normal_force < 0.0:
            normal_damping_force = -normal_contact_force
            normal_force = 0.0
        if ti.static(GlobalVariable.TRACKENERGY):
            norm_elastic = self._elastic_normal_energy(kn, gapn)
            norm_viscous_rate = self._viscous_normal_energy_rate(normal_damping_force, vn)
        return normal_force, norm_elastic, norm_viscous_rate

    @ti.func
    def _tangential_force(self, ks, sdratio, m_eff, vs, normal_force, norm, tangOverlapOld, dt):
        tangOverlapRot = tangOverlapOld - tangOverlapOld.dot(norm) * norm
        tangOverTemp = vs * dt[None] + tangOverlapOld.norm() * Normalize(tangOverlapRot)
        tangOverlapTrial = tangOverTemp
        trial_ft = -ks * tangOverTemp

        fric = self.mu * ti.abs(normal_force)
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
    def _force_assemble(self, m_eff, rad_eff, gapn, coeff, dgdx, v_rel, tangOverlapOld, dt):
        kn, ks = self.kn * coeff, self.ks * coeff
        ndratio, sdratio = self.ndratio, self.sdratio
        gradient_norm = dgdx.norm()
        norm = dgdx.normalized(Threshold)
        normal_velocity = v_rel.dot(norm)
        gap_rate = v_rel.dot(dgdx)
        vs = v_rel - normal_velocity * norm

        normal_force, norm_elastic, norm_viscous_rate = self._normal_force(kn, ndratio, m_eff, gapn, gap_rate)
        normal_force_magnitude = normal_force * gradient_norm
        tangential_force, tangOverTemp, tang_elastic, tang_viscous_rate, friction_energy = self._tangential_force(
            ks, sdratio, m_eff, vs, normal_force_magnitude, norm, tangOverlapOld, dt
        )
        if ti.static(GlobalVariable.TRACKENERGY):
            self.elastic_energy += norm_elastic + tang_elastic
            if dt[None] > 0.0:
                self.friction_energy += friction_energy
                self.damp_energy += (norm_viscous_rate + tang_viscous_rate) * dt[None]
        if dt[None] == 0.0:
            tangOverTemp = tangOverlapOld
        return normal_force * dgdx, tangential_force, tangOverTemp

    @ti.func
    def _force_assemble_work_conjugate(
        self,
        m_eff,
        rad_eff,
        penetration,
        next_penetration,
        coeff,
        dgdx,
        v_rel,
        tangOverlapOld,
        dt,
    ):
        """Discrete-gradient penalty force for an advected soft SDF.

        ``penetration`` is a contact-history variable advanced with the same
        gap rate and trace velocities used to apply contact work.  Consequently
        the normal elastic force satisfies, to roundoff,

            F_n . v_rel * dt + U(delta_next) - U(delta) = 0.

        This avoids creating energy when an independently advected SDF lags the
        physical surface during soft--soft unloading.
        """
        kn, ks = self.kn * coeff, self.ks * coeff
        gradient_norm = dgdx.norm()
        norm = dgdx.normalized(Threshold)
        normal_velocity = v_rel.dot(norm)
        gap_rate = v_rel.dot(dgdx)
        vs = v_rel - normal_velocity * norm

        current_energy = kn / self.theta * ti.pow(ti.max(penetration, 0.0), self.theta)
        next_energy = kn / self.theta * ti.pow(ti.max(next_penetration, 0.0), self.theta)
        work_displacement = gap_rate * dt[None]
        normal_contact_force = kn * ti.pow(ti.max(penetration, 0.0), self.theta - 1.0)
        if ti.abs(work_displacement) > Threshold:
            normal_contact_force = -(next_energy - current_energy) / work_displacement

        normal_damping_force = -2.0 * self.ndratio * ti.sqrt(m_eff * kn) * gap_rate
        normal_force = normal_contact_force + normal_damping_force
        if normal_force < 0.0:
            normal_damping_force = -normal_contact_force
            normal_force = 0.0

        tangential_force, tangOverTemp, tang_elastic, tang_viscous_rate, friction_energy = self._tangential_force(
            ks,
            self.sdratio,
            m_eff,
            vs,
            normal_force * gradient_norm,
            norm,
            tangOverlapOld,
            dt,
        )
        if ti.static(GlobalVariable.TRACKENERGY):
            self.elastic_energy += current_energy + tang_elastic
            self.friction_energy += friction_energy
            self.damp_energy += (
                self._viscous_normal_energy_rate(normal_damping_force, gap_rate) + tang_viscous_rate
            ) * dt[None]
        return normal_force * dgdx, tangential_force, tangOverTemp
