import math

import src.igampm.config as config
from src.utils.TimeTicker import Timer


class Simulation:
    def __init__(self):
        self.dimension = config.get_dimension()
        self.coupling_scheme = "IGAMPM"
        self.contact_model = "IPC"
        self.activate_friction = False
        self.is_axisymmetric = False
        self.axis_offset = 0.0
        self.axis_configuration_explicit = False
        self._configuration_frozen = False
        self.timer = Timer()

    def freeze_configuration(self):
        """Prevent storage-defining coupling options from changing after build."""
        self._configuration_frozen = True

    def set_configuration(
        self,
        dimension=None,
        coupling_scheme="IGAMPM",
        contact_model="IPC",
        activate_friction=False,
        axisymmetric=False,
        axis_offset=0.0,
    ):
        if self._configuration_frozen:
            raise RuntimeError(
                "IGA-MPM configuration cannot be changed after build(); " "construct a new coupling engine instead"
            )
        if dimension is not None:
            config.set_dimension(dimension)
            self.dimension = config.get_dimension()
        self.coupling_scheme = str(coupling_scheme)
        self.contact_model = config.normalize_contact_model(contact_model)
        self.activate_friction = bool(activate_friction)
        self.is_axisymmetric = bool(axisymmetric)
        self.axis_offset = float(axis_offset)
        self.axis_configuration_explicit = True
        self.validate_configuration()

    def validate_configuration(self):
        if self.coupling_scheme not in ("IGAMPM", "IGA-MPM"):
            raise ValueError("IGA-MPM coupling_scheme must be 'IGAMPM' or 'IGA-MPM'")
        config.normalize_contact_model(self.contact_model)
        if self.is_axisymmetric and self.dimension != 2:
            raise ValueError("axisymmetric IGA-MPM requires dimension=2")
        if not math.isfinite(self.axis_offset):
            raise ValueError("IGA-MPM axis_offset must be finite")
