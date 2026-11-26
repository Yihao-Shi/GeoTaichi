from hashlib import sha256
from pathlib import Path

import taichi as ti

from src.mpm.DistributedMPM import DistributedMPM
from src.utils.errors import ConfigurationError


@ti.kernel
def exchange_rigid_reactions(rigid: ti.template(), values: ti.types.ndarray(ndim=2), pack: ti.template()):
    for body in range(values.shape[0]):
        for d in ti.static(range(3)):
            if ti.static(pack):
                values[body, d] = rigid[body].contact_force[d]
                values[body, d + 3] = rigid[body].contact_torque[d]
            else:
                rigid[body].contact_force[d] = values[body, d]
                rigid[body].contact_torque[d] = values[body, d + 3]


class DistributedLSDEM(DistributedMPM):
    def __init__(self, model):
        msims, dsims, sims = model.mpm.sims, model.dem.sims, model.sims
        dscene = model.dem.scene
        if msims.dimension != 3:
            raise ConfigurationError("distributed MPM–LSDEM coupling requires dimension=3")
        if not (
            sims.coupling_scheme == "MPDEM"
            and sims.particle_interaction
            and not sims.wall_interaction
            and msims.coupling == "Lagrangian"
            and msims.configuration == "ULMPM"
            and msims.material_type == "Solid"
            and not msims.contact_detection
            and dsims.scheme == "LSDEM"
        ):
            raise ConfigurationError(
                "distributed MPDEM requires Lagrangian solid ULMPM–LSDEM body contact without MPM grid contact"
            )
        if (
            dsims.particle_particle_contact_model
            or dsims.particle_wall_contact_model
            or int(dscene.wallNum[0]) > 0
            or sims.dem_timestep != sims.delta
        ):
            raise ConfigurationError(
                "distributed MPDEM requires LSDEM without walls, internal DEM contacts or subcycling"
            )
        if model.generator.myGenerator or model.dem.generator.myGenerator:
            raise ConfigurationError("distributed MPDEM requires pre-generated bodies")
        rigid_count = int(dscene.particleNum[0])
        if rigid_count <= 0:
            raise ConfigurationError("each distributed MPDEM rank requires replicated LSDEM bodies")

        rigid = dscene.rigid
        rigid_state = sha256()
        for name in ("m", "mass_center", "v", "w", "q", "inv_I", "is_fix", "materialID"):
            rigid_state.update(getattr(rigid, name).to_numpy()[:rigid_count].tobytes())
        super().__init__(
            model.mpm,
            clock=sims,
            coupled_settings=(
                rigid_count,
                str(rigid.contact_force.dtype),
                rigid_state.digest(),
                tuple(dsims.gravity),
                tuple(dsims.boundary),
                dsims.engine,
            ),
        )
        if self.restart_index is not None:
            history = sims.history_contact_path
            shard = Path(model.mpm._restart_checkpoint[1]).resolve()
            if (
                history.get("file_number") != self.restart_index
                or not history.get("ppcontact")
                or Path(history["ppcontact"]).resolve().parent != shard
            ):
                raise ConfigurationError(
                    "distributed MPDEM restart requires coupling contact history from the same shard"
                )
        self.rigid = rigid
        self.reactions = self._buffer((rigid_count, 6), rigid.contact_force.dtype)

    def reduce_reactions(self):
        exchange_rigid_reactions(self.rigid, self.reactions, True)
        self._sum(self.reactions)
        exchange_rigid_reactions(self.rigid, self.reactions, False)
