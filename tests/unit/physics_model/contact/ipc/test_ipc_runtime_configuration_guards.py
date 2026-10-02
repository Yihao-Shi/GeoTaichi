from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ipc, pytest.mark.contact]

from src.dem.Simulation import Simulation
from src.dem.engines.AffineBodyEngine import AffineBodyEngine
from src.dem.mainDEM import DEM
from src.fempm.contact import IPCModel
from src.fempm.mainFEMPM import FEMPM
from src.mpm.mainMPM import MPM
from src.mpdem.engines.SoftAffineIPCEngine import SoftAffineIPCEngine


def test_affine_ipc_parameter_setter_is_frozen_after_initialization():
    sims = object.__new__(Simulation)
    sims._affine_body_parameters_frozen = False
    sims.affine_friction_mode = "lagged"
    original_mode = sims.affine_friction_mode
    sims.freeze_affine_body_parameters()

    with pytest.raises(RuntimeError, match="cannot be changed after engine"):
        sims.set_affine_body_parameters(friction_mode="fully_implicit")

    assert sims.affine_friction_mode == original_mode


def test_lsmpm_soft_rigid_route_is_frozen_after_engine_selection():
    sims = object.__new__(Simulation)
    sims._lsmpm_soft_rigid_contact_frozen = False
    sims.lsmpm_soft_rigid_contact = "DEM"
    original_route = sims.lsmpm_soft_rigid_contact
    sims.freeze_lsmpm_soft_rigid_contact()

    with pytest.raises(RuntimeError, match="cannot be changed after engine"):
        sims.set_lsmpm_soft_rigid_contact("IPC")

    assert sims.lsmpm_soft_rigid_contact == original_route


def test_dem_add_engine_freezes_lsmpm_affine_route():
    calls = []
    sims = SimpleNamespace(
        scheme="LSMPM",
        freeze_lsmpm_soft_rigid_contact=lambda: calls.append("freeze"),
    )
    engine = SimpleNamespace(
        choose_engine=lambda *args: calls.append("choose"),
        set_servo_mechanism=lambda *args: calls.append("servo"),
    )
    coupling = SimpleNamespace(
        sims=sims,
        scene=SimpleNamespace(affine_bodies=[object()]),
        enginer=engine,
    )

    DEM.add_engine(coupling, callback=None)

    assert calls == ["choose", "freeze", "servo"]


def test_affine_engine_initialization_freezes_operator_parameters():
    calls = []
    engine = AffineBodyEngine()
    engine.state = object()
    engine.operator = object()
    sims = SimpleNamespace(
        search="LinkedCell",
        enable_step_retry=False,
        step_retry_max_retries=0,
        step_retry_reduction=0.5,
        step_retry_minimum_timestep=0.0,
        freeze_affine_body_parameters=lambda: calls.append("affine"),
    )

    engine.initialize(sims, SimpleNamespace(affine_bodies=[object()]))

    assert calls == ["affine"]


def test_soft_affine_engine_initialization_freezes_parameters_and_route():
    calls = []
    engine = SoftAffineIPCEngine()
    engine.soft_material = object()
    engine.operator = SimpleNamespace(friction_mode="fully_implicit")
    sims = SimpleNamespace(
        affine_friction_mode="fully_implicit",
        affine_friction_iterations=1,
        affine_assemble_type="HashTriplet",
        enable_step_retry=False,
        step_retry_max_retries=0,
        step_retry_reduction=0.5,
        step_retry_minimum_timestep=0.0,
        freeze_affine_body_parameters=lambda: calls.append("affine"),
        freeze_lsmpm_soft_rigid_contact=lambda: calls.append("route"),
    )
    scene = SimpleNamespace(softNum=[1], affine_bodies=[object()])

    engine.initialize(sims, scene)

    assert calls == ["affine", "route"]


def test_public_direct_mpm_rejects_multibody_ipc():
    mpm = object.__new__(MPM)
    mpm.direct_bodies = SimpleNamespace(bodies={"first": object(), "second": object()})
    mpm.sims = SimpleNamespace(ipc_contact=True)

    with pytest.raises(NotImplementedError, match="self/multibody IPC"):
        mpm._build_direct_engine()


def test_public_fempm_rejects_soft_particle_mpm_cloth_ipc():
    coupling = object.__new__(FEMPM)
    coupling.sims = SimpleNamespace(delta=1.0e-3)
    coupling._memory = {}
    coupling.contactor = SimpleNamespace(model=object.__new__(IPCModel))
    coupling.mpm = SimpleNamespace(sims=SimpleNamespace(dimension=3, soft_particle=True))
    coupling.fem = SimpleNamespace(scene=SimpleNamespace(material=SimpleNamespace(is_cloth=True)))

    with pytest.raises(NotImplementedError, match="soft-particle MPM--cloth IPC"):
        coupling.add_essentials()
