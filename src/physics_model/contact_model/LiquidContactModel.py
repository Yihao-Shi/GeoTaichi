import taichi as ti


@ti.dataclass
class LiquidSurfaceProperty:
    kn: float
    ndratio: float
    ncut: float
    scale: float

    def add_surface_property(self, kn, ndratio, scale):
        self.kn = kn
        self.ndratio = ndratio
        self.ncut = 0.
        self.scale = scale

    def print_surface_info(self, matID1, matID2):
        print(" Surface Properties Information ".center(71, '-'))
        print('Contact model: Fluid-Particle Model')
        print(f'MaterialID{matID1} < --- > MaterialID{matID2}')
        print('Contact normal stiffness: = ', self.kn)
        print('Viscous damping coefficient = ', self.ndratio)
        print('Non-slip fraction = ', self.scale)
        print('\n')

    @ti.func
    def _fluid_force_assemble(self, m_eff, gapn, coeff, norm, v_rel, dt):
        kn, ndratio = self.kn, self.ndratio
        vn = v_rel.dot(norm) 
        vs = v_rel - v_rel.dot(norm) * norm

        normal_contact_force = -kn * coeff * gapn 
        normal_damping_force = -2 * ndratio * ti.sqrt(m_eff * kn * coeff) * vn
        normal_force = (normal_contact_force + normal_damping_force) * norm
        tangential_force = -vs / dt[None] * m_eff * self.scale
        return normal_force, tangential_force