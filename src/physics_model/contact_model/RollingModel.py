import taichi as ti

from src.utils.constants import ZEROVEC3F
import src.utils.GlobalVariable as GlobalVariable
from src.utils.VectorFunction import normalize, squared


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
    elastic_energy: float
    friction_energy: float
    damp_energy: float

    def add_surface_property(self, YoungModulus, stiffness_ratio, mu, shape_factor, crush_factor, ndratio, sdratio):
        self.YoungModulus = YoungModulus
        self.stiffness_ratio = stiffness_ratio
        self.mu = mu
        self.shape_factor = shape_factor
        self.crush_factor = crush_factor
        self.ndratio = ndratio
        self.sdratio = sdratio
        self.ncut = 0.0

    def print_surface_info(self, matID1, matID2):
        print(" Surface Properties Information ".center(71, "-"))
        print("Contact model: Linear Rolling Resistance Contact Model")
        print(f"MaterialID{matID1} < --- > MaterialID{matID2}")
        print("Youngs Modulus: = ", self.YoungModulus)
        print("Stiffness Ratio: = ", self.stiffness_ratio)
        print("Shape Factor = ", self.shape_factor)
        print("Crush Factor = ", self.crush_factor)
        print("Friction coefficient = ", self.mu)
        print("Viscous damping coefficient = ", self.ndratio)
        print("Viscous damping coefficient = ", self.sdratio, "\n")

    @ti.func
    def _get_equivalent_stiffness(self, end1, end2, particle, wall):
        pos1, pos2 = particle[end1].x, wall[end2]._get_center()
        particle_rad, norm = particle[end1].rad, wall[end2]._get_norm(pos1)
        distance = (pos1 - pos2).dot(norm)
        fraction = ti.abs(wall[end2].process_circle_shape(pos1, particle_rad, distance))
        return 2.0 * fraction * particle_rad * self.YoungModulus

    @ti.func
    def _get_ls_equivalent_stiffness(self, parameter, end1, end2, rigid, particle, wall):
        particle_rad = rigid[end1].equi_r
        kn = 2.0 * particle_rad * self.YoungModulus * parameter
        return kn

    @ti.func
    def _elastic_normal_energy(self, kn, normal_force):
        return 0.5 * normal_force * normal_force / kn

    @ti.func
    def _viscous_normal_energy_rate(self, normal_damping_force, vn):
        return normal_damping_force * vn

    @ti.func
    def _elastic_tangential_energy(self, ks, tang_over_temp):
        return 0.5 * squared(tang_over_temp) * ks

    @ti.func
    def _viscous_tangential_energy(self, tangential_damping_force, vs):
        return tangential_damping_force.dot(vs)

    @ti.func
    def _friction_energy(self, fric_ds, tangential_force):
        return fric_ds.dot(tangential_force)

    @ti.func
    def _normal_force(self, kn, ndratio, gapn, vn):
        normal_contact_force = -kn * gapn
        normal_damping_force = -ndratio * vn
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
    def _tangential_force(self, ks, miu, sdratio, normal_force, vs, norm, tang_overlap_old, dt):
        tang_overlap_rot = tang_overlap_old - tang_overlap_old.dot(norm) * norm
        tang_over_temp = vs * dt[None] + tang_overlap_old.norm() * normalize(tang_overlap_rot)
        tang_overlap_trial = tang_over_temp
        trial_ft = -ks * tang_over_temp

        fric = miu * ti.abs(normal_force)
        tangential_force = ZEROVEC3F
        tang_elastic, tang_viscous_rate, friction_energy = 0.0, 0.0, 0.0
        if trial_ft.norm() > fric:
            tangential_force = fric * trial_ft.normalized()
            tang_over_temp = -tangential_force / ks
            if ti.static(GlobalVariable.TRACKENERGY):
                tang_elastic = self._elastic_tangential_energy(ks, tang_over_temp)
                friction_energy = self._friction_energy(tang_overlap_trial - tang_over_temp, tangential_force)
        else:
            tang_damping_force = -sdratio * vs
            tangential_force = trial_ft + tang_damping_force
            if ti.static(GlobalVariable.TRACKENERGY):
                tang_elastic = self._elastic_tangential_energy(ks, tang_over_temp)
                tang_viscous_rate = self._viscous_tangential_energy(tang_damping_force, vs)
        return tangential_force, tang_over_temp, tang_elastic, tang_viscous_rate, friction_energy

    @ti.func
    def _rolling_force(self, kr, rmiu, rdratio, normal_force, wr, norm, tang_rolling_old, dt):
        tang_rolling_rot = tang_rolling_old - tang_rolling_old.dot(norm) * norm
        tang_rolling_temp = wr * dt[None] + tang_rolling_old.norm() * normalize(tang_rolling_rot)
        tang_rolling_trial = tang_rolling_temp
        trial_fr = -kr * tang_rolling_temp

        fric_roll = rmiu * ti.abs(normal_force)
        rolling_momentum = ZEROVEC3F
        rolling_elastic, rolling_viscous_rate, rolling_friction_energy = 0.0, 0.0, 0.0
        if trial_fr.norm() > fric_roll:
            rolling_momentum = fric_roll * trial_fr.normalized()
            tang_rolling_temp = -rolling_momentum / kr
            if ti.static(GlobalVariable.TRACKENERGY):
                rolling_elastic = self._elastic_tangential_energy(kr, tang_rolling_temp)
                rolling_friction_energy = self._friction_energy(
                    tang_rolling_trial - tang_rolling_temp, rolling_momentum
                )
        else:
            rolling_damping_force = -rdratio * wr
            rolling_momentum = trial_fr + rolling_damping_force
            if ti.static(GlobalVariable.TRACKENERGY):
                rolling_elastic = self._elastic_tangential_energy(kr, tang_rolling_temp)
                rolling_viscous_rate = self._viscous_tangential_energy(rolling_damping_force, wr)
        return rolling_momentum, tang_rolling_temp, rolling_elastic, rolling_viscous_rate, rolling_friction_energy

    @ti.func
    def _twisting_force(self, kt, tmiu, tdratio, normal_force, wt, norm, tang_twisting_old, dt):
        tang_twisting_temp = wt * dt[None] + tang_twisting_old.norm() * normalize(norm)
        tang_twisting_trial = tang_twisting_temp
        trial_ft = -kt * tang_twisting_temp

        fric_twist = tmiu * ti.abs(normal_force)
        twisting_momentum = ZEROVEC3F
        twisting_elastic, twisting_viscous_rate, twisting_friction_energy = 0.0, 0.0, 0.0
        if trial_ft.norm() > fric_twist:
            twisting_momentum = fric_twist * trial_ft.normalized()
            tang_twisting_temp = -twisting_momentum / kt
            if ti.static(GlobalVariable.TRACKENERGY):
                twisting_elastic = self._elastic_tangential_energy(kt, tang_twisting_temp)
                twisting_friction_energy = self._friction_energy(
                    tang_twisting_trial - tang_twisting_temp, twisting_momentum
                )
        else:
            twisting_damping_force = -tdratio * wt
            twisting_momentum = trial_ft + twisting_damping_force
            if ti.static(GlobalVariable.TRACKENERGY):
                twisting_elastic = self._elastic_tangential_energy(kt, tang_twisting_temp)
                twisting_viscous_rate = self._viscous_tangential_energy(twisting_damping_force, wt)
        return twisting_momentum, tang_twisting_temp, twisting_elastic, twisting_viscous_rate, twisting_friction_energy

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
        tang_overlap_old,
        tang_rolling_old,
        tang_twisting_old,
        dt,
    ):
        young_modulus, stiffness_ratio = self.YoungModulus, self.stiffness_ratio
        shape_factor, crush_factor = self.shape_factor, self.crush_factor
        miu = self.mu
        kn = 2 * rad_eff * young_modulus * coeff
        ks = kn * stiffness_ratio
        ndratio = self.ndratio * 2 * ti.sqrt(m_eff * kn)
        sdratio = self.sdratio * 2 * ti.sqrt(m_eff * ks)

        RBar = shape_factor * rad_eff
        square_r = RBar * RBar
        kr = 0.25 * kn * square_r
        kt = 0.5 * ks * square_r
        rdratio = 0.25 * ndratio * square_r
        tdratio = 0.5 * sdratio * square_r
        rmiu = 0.25 * RBar * crush_factor
        tmiu = 0.65 * RBar * miu

        vn = v_rel.dot(norm)
        vs = v_rel - vn * norm
        wt = (w_rel).dot(norm) * norm
        wr = w_rel - wt

        normal_force, norm_elastic, norm_viscous_rate = self._normal_force(kn, ndratio, gapn, vn)
        tangential_force, tang_over_temp, tang_elastic, tang_viscous_rate, friction_energy = self._tangential_force(
            ks, miu, sdratio, normal_force, vs, norm, tang_overlap_old, dt
        )
        rolling_momentum, tang_rolling_temp, rolling_elastic, rolling_viscous_rate, rolling_friction_energy = (
            self._rolling_force(kr, rmiu, rdratio, normal_force, wr, norm, tang_rolling_old, dt)
        )
        twisting_momentum, tang_twisting_temp, twisting_elastic, twisting_viscous_rate, twisting_friction_energy = (
            self._twisting_force(kt, tmiu, tdratio, normal_force, wt, norm, tang_twisting_old, dt)
        )
        if ti.static(GlobalVariable.TRACKENERGY):
            self.elastic_energy += norm_elastic + tang_elastic + rolling_elastic + twisting_elastic
            if dt[None] > 0.0:
                self.friction_energy += friction_energy + rolling_friction_energy + twisting_friction_energy
                self.damp_energy += (
                    norm_viscous_rate + tang_viscous_rate + rolling_viscous_rate + twisting_viscous_rate
                ) * dt[None]
        if dt[None] == 0.0:
            tang_over_temp, tang_rolling_temp, tang_twisting_temp = (
                tang_overlap_old,
                tang_rolling_old,
                tang_twisting_old,
            )
        return (
            normal_force * norm,
            tangential_force,
            rolling_momentum + twisting_momentum,
            tang_over_temp,
            tang_rolling_temp,
            tang_twisting_temp,
        )
