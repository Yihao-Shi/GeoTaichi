import taichi as ti

from src.utils.constants import PI
from src.utils.ObjectIO import DictIO


@ti.data_oriented
class DragForce:
    def __init__(self, drag_model):
        drag_force_model = DictIO.GetAlternative(drag_model, "DragForceModel", "AbrahamModel")
        drag_law = DictIO.GetAlternative(drag_model, "DragLaw", "Quadratic")
        particle_reynold = DictIO.GetAlternative(drag_model, "ParticleReynold", None)

        if drag_force_model == "AbrahamModel":
            self.drag_coefficient_model = self.AbrahamModel
        elif drag_force_model == "EmpiricalModel":
            self.drag_coefficient_model = self.EmpiricalModel
        elif drag_force_model == "BrownLawlerModel":
            self.drag_coefficient_model = self.BrownLawlerModel
        elif drag_force_model == "SchillerNaumannModel":
            self.drag_coefficient_model = self.SchillerNaumannModel
        else:
            valid_list = ["AbrahamModel", "EmpiricalModel", "BrownLawlerModel", "SchillerNaumannModel"]
            raise RuntimeError(f"Drag force model is invalid! Only the following options are valid: {valid_list}")

        if drag_law == "Linear":
            self.drag_law = self.linear_drag_law
        elif drag_law == "Quadratic":
            self.drag_law = self.quadratic_drag_law
        else:
            valid_list = ["Linear", "Quadratic"]
            raise RuntimeError(f"Drag law is invalid! Only the following options are valid: {valid_list}")

        self.particle_reynold = ti.field(float, shape=())
        self.is_constant_reynold = False
        if particle_reynold is not None:
            if float(particle_reynold) <= 0.0:
                raise ValueError("ParticleReynold must be positive when specified")
            self.particle_reynold[None] = float(particle_reynold)
            self.is_constant_reynold = True

    @ti.func
    def GidaspowModel(
        self,
        fluid_volume_fraction,
        fluid_density,
        fluid_viscosity,
        particle_radius,
        particle_reynold,
        relative_velocity,
    ):
        beta = 0.0
        relative_velocity_norm = relative_velocity.norm()
        solid_volume_fraction = 1.0 - fluid_volume_fraction
        particle_diameter = 2.0 * particle_radius
        if fluid_volume_fraction <= 0.8:
            beta = (
                150
                * solid_volume_fraction
                * solid_volume_fraction
                * fluid_viscosity
                / (fluid_volume_fraction * particle_diameter * particle_diameter)
                + 1.75 * solid_volume_fraction * fluid_density / particle_diameter * relative_velocity_norm
            )
        else:
            Cd = self.SchillerNaumannModel(particle_reynold)
            beta = (
                0.75
                * Cd
                * fluid_density
                * relative_velocity_norm
                * solid_volume_fraction
                / particle_diameter
                * fluid_volume_fraction ** (-1.65)
            )
        return beta

    @ti.func
    def SchillerNaumannModel(self, particle_reynold):
        Cd0 = 0.44
        if particle_reynold < 1000.0:
            Cd0 = 24.0 / particle_reynold * (1.0 + 0.15 * particle_reynold**0.687)
        return Cd0

    @ti.func
    def BrownLawlerModel(self, particle_reynold):
        Cd0 = 24.0 / particle_reynold * (1.0 + 0.15 * particle_reynold**0.681) + 0.407 / (1.0 + 8710 / particle_reynold)
        return Cd0

    @ti.func
    def EmpiricalModel(self, particle_reynold):
        temp_val = 0.63 + 4.8 / ti.sqrt(particle_reynold)
        Cd0 = temp_val * temp_val
        return Cd0

    @ti.func
    def AbrahamModel(self, particle_reynold):
        temp_val = 1.0 + 9.06 / ti.sqrt(particle_reynold)
        Cd0 = 24.0 / (9.06 * 9.06) * temp_val * temp_val
        return Cd0

    @ti.func
    def drag_coefficient(self, particle_reynold):
        particle_reynold = ti.max(particle_reynold, 1e-12)
        Cd0 = 0.0
        if particle_reynold <= 1.0:
            Cd0 = 24.0 / particle_reynold
        elif particle_reynold < 1000.0:
            Cd0 = self.drag_coefficient_model(particle_reynold)
        else:
            Cd0 = 0.44
        return Cd0

    @ti.func
    def linear_drag_law(
        self, fluid_volume_fraction, fluid_density, fluid_viscosity, particle_radius, relative_velocity
    ):
        drag_force = 0.0 * relative_velocity
        solid_volume_fraction = 1.0 - fluid_volume_fraction
        if relative_velocity.norm() > 1e-12:
            particle_reynold = self.compute_particle_reynold(
                fluid_volume_fraction, fluid_density, fluid_viscosity, particle_radius, relative_velocity
            )
            particle_volume = 4.0 / 3.0 * PI * particle_radius * particle_radius * particle_radius
            if fluid_volume_fraction > 0.8:
                # Cancel (1-epsilon) analytically: the isolated-sphere limit
                # epsilon=1 has finite drag, not the indeterminate beta/0.
                Cd = self.SchillerNaumannModel(particle_reynold)
                coefficient = (
                    0.75
                    * Cd
                    * fluid_density
                    * relative_velocity.norm()
                    / (2.0 * particle_radius)
                    * fluid_volume_fraction ** (-1.65)
                )
                drag_force = particle_volume * coefficient * relative_velocity
            else:
                beta = self.GidaspowModel(
                    fluid_volume_fraction,
                    fluid_density,
                    fluid_viscosity,
                    particle_radius,
                    particle_reynold,
                    relative_velocity,
                )
                drag_force = particle_volume / solid_volume_fraction * beta * relative_velocity
        return drag_force

    @ti.func
    def quadratic_drag_law(
        self, fluid_volume_fraction, fluid_density, fluid_viscosity, particle_radius, relative_velocity
    ):
        drag_force = 0.0 * relative_velocity
        if relative_velocity.norm() > 1e-12:
            particle_reynold = ti.max(
                self.compute_particle_reynold(
                    fluid_volume_fraction,
                    fluid_density,
                    fluid_viscosity,
                    particle_radius,
                    relative_velocity,
                ),
                1e-12,
            )
            log_reynold = ti.log(particle_reynold) / ti.log(10.0)
            kappa = 3.7 - 0.65 * ti.exp(-0.5 * (1.5 - log_reynold) ** 2)
            drag_coeff = self.drag_coefficient(particle_reynold)
            drag_force = (
                0.5
                * PI
                * drag_coeff
                * fluid_density
                * particle_radius
                * particle_radius
                * fluid_volume_fraction ** (2.0 - kappa)
                * relative_velocity.norm()
                * relative_velocity
            )
        return drag_force

    @ti.func
    def compute_particle_reynold(
        self, fluid_volume_fraction, fluid_density, fluid_viscosity, particle_radius, relative_velocity
    ):
        if ti.static(self.is_constant_reynold):
            return self.particle_reynold[None]
        else:
            return (
                fluid_volume_fraction
                * 2.0
                * fluid_density
                * particle_radius
                * relative_velocity.norm()
                / ti.max(fluid_viscosity, 1e-12)
            )
