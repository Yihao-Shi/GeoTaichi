import ast
import inspect
import textwrap
from types import SimpleNamespace

from src.fedem.Engine import Engine as FEDEMEngine
from src.fedem.FEDEMBase import Solver as FEDEMSolver
from src.fedem.ContactManager import ContactManager as FEDEMContactManager
from src.dem.engines.ExplicitEngine import ExplicitEngine as DEMExplicitEngine
from src.fempm.Engine import Engine as FEMPMEngine
from src.fempm.FEMPMBase import Solver as FEMPMSolver
from src.fempm.ContactManager import ContactManager as FEMPMContactManager
from src.fem.engines.ClassicalAssembler import ClassicalAssembler
from src.mpdem.Engine import Engine as MPDEMEngine
from src.mpdem.fluid_dynamics.IncompressibleSemiResolved import (
    IncompressibleDEMSphereCoupling,
)
from src.mpdem.fluid_dynamics.IncompressibleCoupling import (
    IncompressibleLSDEMCoupling,
    kernel_accumulate_lsdem_volume_fraction_ibm_force,
    kernel_update_lsdem_cell_ibm_fields,
)
from src.mpm.engines.TLExplicitEngine import TLExplicitEngine
from src.mpm.engines.Engine import Engine as MPMEngine
from src.mpm.engines.IncompressibleEngine import IncompressibleEngine
from src.mpm.engines.ULImplicitEngine import ImplicitEngine as MPMImplicitEngine
from src.mpm.engines.ULSemiImplicitTwoPhaseEngine import ULSemiImplicitTwoPhaseEngine
from src.mpm.engines.NewtonIteration import MomentumConservation
from src.mpm.engines.direct.ExplicitTLMPM import ExplicitTLMPM
from src.mpm.engines.direct.ExplicitULMPM import ExplicitULMPM
from src.mpm.engines.direct.ImplicitTLMPM import ImplicitTLMPM
from src.mpm.engines.direct.ImplicitULMPM import ImplicitULMPM
from src.mpm.engines.direct.StaticTwoPhaseULMPM import StaticTwoPhaseULMPM
import src.mpm.soft_particle.DEMPMBridge as DEMPBridge
from src.igampm.engines.CoupledEngine import Engine as IGAMPMEngine
from src.igampm.engines.ExplicitEngine import ExplicitEngine as ExplicitIGAMPMEngine


def _calls(function, name):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
        for node in ast.walk(tree)
    )


def test_coupled_explicit_integrators_use_bound_fem_operations():
    assert not _calls(FEDEMEngine._integrate_dynamic_fem, "getattr")
    assert not _calls(FEMPMEngine.integration, "getattr")
    assert "assemble_explicit_internal_force" in inspect.getsource(FEDEMEngine._integrate_dynamic_fem)
    assert "assemble_explicit_internal_force" in inspect.getsource(FEMPMEngine.integration)
    assert "getattr" not in inspect.getsource(ClassicalAssembler.assemble_force_device)


def test_fedem_neighbor_mode_is_bound_at_construction():
    state = SimpleNamespace(reduce_kinetic_energy=lambda: None)
    fem_engine = SimpleNamespace(
        is_fully_constrained_static=False,
        track_energy=False,
        state=state,
    )
    fem = SimpleNamespace(engine=fem_engine)
    dem = SimpleNamespace(
        sims=SimpleNamespace(scheme="DEM"),
        enginer=object(),
        contactor=SimpleNamespace(neighbor=object()),
    )
    engine = FEDEMEngine(SimpleNamespace(), fem, dem, SimpleNamespace(), object())

    assert engine.update_dem_verlet_tables.__func__ is FEDEMEngine._update_dem_verlet_tables
    assert "scheme ==" not in inspect.getsource(FEDEMEngine.update_verlet_tables)


def test_stage_profiler_is_bound_once():
    events = []

    class _Profiler:
        def measure(self, name, function, *args, **kwargs):
            events.append(name)
            return function(*args, **kwargs)

    engine = object.__new__(FEDEMEngine)
    engine.stage_call = engine._direct_stage_call
    assert engine.stage_call("plain", lambda value: value + 1, 2) == 3

    engine.bind_stage_profiler(_Profiler())
    assert engine.stage_call("profiled", lambda value: value + 1, 2) == 3
    assert events == ["profiled"]


def test_coupled_callback_adapters_have_no_python_version_fallback():
    assert "dict_values" not in inspect.getsource(FEDEMSolver)
    assert "dict_values" not in inspect.getsource(FEMPMSolver)


def test_configured_contact_model_is_not_redispatched_each_step():
    assert "isinstance" not in inspect.getsource(FEDEMContactManager.resolve)
    assert "isinstance" not in inspect.getsource(FEMPMContactManager.resolve)
    assert "resolve_contact_step" in inspect.getsource(FEMPMContactManager.resolve)
    assert "_resolve_linear_levelset_contact_with_energy" in inspect.getsource(
        FEDEMContactManager._bind_runtime_functions
    )


def test_mpdem_optional_fluid_operations_are_bound_before_the_hot_loop():
    class _Coupler:
        def pre_compute(self):
            return "prepared"

        def accumulate_pressure_force(self, _sims, _scene):
            return "accumulated"

        def transform_translational_load(self):
            return "transformed"

    engine = object.__new__(MPDEMEngine)
    engine.incompressible_coupler = _Coupler()
    engine.dengine = SimpleNamespace()
    engine.sims = SimpleNamespace(coupling_scheme="MPDEM", particle_interaction=True)
    engine.system_resolve = lambda: "cross contact"
    engine._bind_incompressible_coupler()

    assert engine.precompute_incompressible() == "prepared"
    assert engine.accumulate_pressure_force(None, None) == "accumulated"
    assert engine.dengine.transform_translational_load() == "transformed"
    assert engine.resolve_cross_contact() == "cross contact"
    assert "incompressible_coupler is not None" not in inspect.getsource(MPDEMEngine.integration)
    assert "incompressible_coupler is not None" not in inspect.getsource(MPDEMEngine.lsintegration)
    assert "coupling_scheme" not in inspect.getsource(MPDEMEngine.incompressible_dem_sphere_integration)
    assert "coupling_scheme" not in inspect.getsource(MPDEMEngine.incompressible_lsdem_integration)
    assert "accumulate_pressure_force" in inspect.getsource(MPDEMEngine._incompressible_dem_integration)
    assert "cell_solid_fraction is None" not in inspect.getsource(IncompressibleDEMSphereCoupling.pre_compute)
    assert "configure_runtime_functions" in inspect.getsource(IncompressibleDEMSphereCoupling.attach)


def test_moving_lsdem_uses_volume_fraction_ibm_not_cut_cells():
    choose_source = inspect.getsource(MPDEMEngine.choose_incompressible_lsdem_coupling)
    update_source = inspect.getsource(IncompressibleLSDEMCoupling.update_solid_cells)
    force_source = inspect.getsource(IncompressibleLSDEMCoupling.accumulate_pressure_force)

    assert "set_solid_sdf_cut_cell" not in choose_source
    assert "solid_sdf" not in update_source
    assert "cell.type" not in update_source
    assert "kernel_accumulate_lsdem_pressure_force" not in force_source
    assert "kernel_accumulate_lsdem_volume_fraction_ibm_force" in force_source


def test_lsdem_ibm_parallelizes_over_fluid_cells():
    update_source = inspect.getsource(kernel_update_lsdem_cell_ibm_fields)
    force_source = inspect.getsource(kernel_accumulate_lsdem_volume_fraction_ibm_force)

    assert update_source.index("for I in ti.grouped(solid_fraction)") < update_source.index(
        "for body in range(rigid_num)"
    )
    assert force_source.index("for I in ti.grouped(ti.ndrange") < force_source.index("for body in range(rigid_num)")


def test_cut_cell_wall_and_ibm_field_callbacks_are_independent():
    pcg_source = inspect.getsource(IncompressibleEngine.fdm_discretization_pcg)
    mgpcg_source = inspect.getsource(IncompressibleEngine.fdm_discretization_mgpcg)
    attach_source = inspect.getsource(IncompressibleLSDEMCoupling.attach)

    for source in (pcg_source, mgpcg_source):
        assert "update_external_cut_cell_boundary" in source
        assert "update_external_ibm_fields" in source
    assert "update_external_ibm_fields" in attach_source
    assert "update_external_cut_cell_boundary" not in attach_source


def test_lsmpm_soft_engine_capability_is_bound_before_explicit_steps():
    bridge_step = inspect.getsource(DEMPBridge.euler_lsmpm_integration)
    dem_step = inspect.getsource(DEMExplicitEngine.euler_lsmpm_integration)
    soft_step = inspect.getsource(DEMPBridge.SoftParticleExplicitEngineMixin.euler_lsmpm_integration)

    assert "hasattr" not in bridge_step
    assert "MethodType" not in bridge_step
    assert "_soft_particle_backend" not in dem_step
    assert "soft_particle_backend" in dem_step
    assert "bind_explicit_engine" in inspect.getsource(DEMExplicitEngine.choose_engine)
    assert soft_step.index("if int(scene.softNum[0]) <= 0") < soft_step.index("if self.soft_particle_material is None")
    assert "soft_levelset_advection_scheme" not in soft_step
    assert "advance_soft_levelset_transport" in soft_step


def test_tlmpm_adaptive_transfer_is_bound_before_the_hot_loop():
    assert "getattr" not in inspect.getsource(TLExplicitEngine.compute_nodal_kinematics)
    assert "getattr" not in inspect.getsource(TLExplicitEngine.reset_reference_grid_message)

    engine = object.__new__(TLExplicitEngine)
    interpolation = lambda *_args: None
    engine.calculate_interpolation = interpolation
    adaptive_scene = SimpleNamespace(element=SimpleNamespace(adaptive=True), sparse_grid=None)
    engine._bind_adaptive_runtime_functions(adaptive_scene)
    assert engine.update_adaptive_interpolation is interpolation
    assert engine.compute_nodal_kinematic.__func__ is TLExplicitEngine.compute_adaptive_nodal_kinematics
    assert engine.reset_grid_messages.__func__ is TLExplicitEngine.reset_adaptive_grid_message

    reference_scene = SimpleNamespace(element=SimpleNamespace(adaptive=False), sparse_grid=None)
    engine._bind_adaptive_runtime_functions(reference_scene)
    assert engine.compute_nodal_kinematic.__func__ is TLExplicitEngine.compute_nodal_kinematics
    assert engine.reset_grid_messages.__func__ is TLExplicitEngine.reset_reference_grid_message


def test_direct_mpm_boundary_modes_are_bound_before_substeps():
    for engine_type in (
        ExplicitTLMPM,
        ExplicitULMPM,
        ImplicitTLMPM,
        ImplicitULMPM,
    ):
        source = inspect.getsource(engine_type.substep)
        assert "if self.compute_traction" not in source
        assert "if self.neumann.num" not in source
        assert "if self.dirichlet.num" not in source
        assert "assemble_traction_step" in source

    static_source = inspect.getsource(StaticTwoPhaseULMPM.solve_current_step)
    assert "if self.neumann.num" not in static_source
    assert "if self.linear_solver" not in static_source
    assert "if self.line_search" not in static_source
    assert "if self.require_both_convergence_checks" not in static_source
    assert "assemble_neumann_step" in static_source
    assert "apply_dirichlet_step" in static_source
    assert "solve_linear_increment_step" in static_source
    assert "accept_newton_increment_step" in static_source
    assert "if self.linear_solver" not in inspect.getsource(StaticTwoPhaseULMPM.solve_linear_system)
    assert "method ==" not in inspect.getsource(StaticTwoPhaseULMPM.solve_reduced_iterative)


def test_static_twophase_bound_newton_policies_preserve_acceptance_logic():
    solver = object.__new__(StaticTwoPhaseULMPM)
    solver.tol = 1.0e-3
    assert solver._both_newton_checks_converged(5.0e-4, 2.0e-4, 1.0e-3)
    assert not solver._both_newton_checks_converged(2.0e-3, 2.0e-4, 1.0e-3)
    assert solver._either_newton_check_converged(2.0e-3, 2.0e-4, 1.0e-3)

    solver.line_search_max_backtrack = 2
    solver.line_search_beta = 0.5
    solver.dt = 0.1
    solver.active_dof = 1
    solver.rhs = object()
    solver.rhs_inf_field = {None: 0.0}
    solver.hash_matrix = SimpleNamespace(reset_system=lambda: None)
    solver.backup_current_solution = lambda: None
    solver.zero_vector = lambda *_args: None
    solver.assemble_system = lambda *_args: None
    solver.assemble_neumann_step = lambda: None
    solver.apply_dirichlet_step = lambda: None
    solver.current_step_state_is_finite_and_positive = lambda: True
    trial = {"alpha": None}
    solver.apply_increment_with_alpha = lambda alpha: trial.__setitem__("alpha", alpha)

    def reduce_rhs():
        solver.rhs_inf_field[None] = 8.0 * trial["alpha"]

    solver.reduce_rhs_inf = reduce_rhs
    residual, accepted_rhs, accepted_alpha = solver._accept_line_search_increment(object(), 10.0, 2.0)
    assert accepted_alpha == 1.0
    assert accepted_rhs == 8.0
    assert residual == 2.0


def test_mpm_dense_and_sparse_taylor_p2g_are_separate_bound_paths():
    assert "sparse_grid" not in inspect.getsource(MPMEngine.compute_nodal_kinematics_taylor)
    assert "kernel_mass_momentum_taylor_p2g_sparse" in inspect.getsource(
        MPMEngine.compute_nodal_kinematics_taylor_sparse
    )
    assert "solid_sdf is not None" not in inspect.getsource(IncompressibleEngine.enforce_particle_domain_boundary)
    assert "use_elastic_matrix_free_tangent" not in inspect.getsource(MPMImplicitEngine.compute_stress_strain)
    assert "use_elastic_matrix_free_tangent" not in inspect.getsource(MPMImplicitEngine.compute_stress_strain_2D)
    fused_p2g = inspect.getsource(IncompressibleEngine.compute_nodal_kinematics)
    assert "cell_volumefrac" in fused_p2g
    assert "classify_fluid_domain" in fused_p2g
    assert "identify_fluid_domain" not in inspect.getsource(IncompressibleEngine.fdm_discretization_pcg)
    assert "identify_fluid_domain" not in inspect.getsource(IncompressibleEngine.fdm_discretization_mgpcg)
    for function in (
        MomentumConservation.calculate_reaction_forces_penalty,
        MomentumConservation.calculate_reaction_forces_matrix_free_2D,
        MomentumConservation.calculate_reaction_forces_matrix_free,
        MomentumConservation.calculate_reaction_forces_assembled_2D,
        MomentumConservation.calculate_reaction_forces_assembled,
    ):
        assert "sparse_grid is not None" not in inspect.getsource(function)
    poisson_step = inspect.getsource(ULSemiImplicitTwoPhaseEngine.compute_Poisson_equation_2D)
    assert "assemble_type" not in poisson_step
    assert "solve_poisson_system" in poisson_step


def test_igampm_mpm_mode_is_bound_before_the_implicit_hot_path():
    assert "hasattr" not in inspect.getsource(IGAMPMEngine._material_feasible_step_cuda)
    assert "total_lagrangian_mpm" not in inspect.getsource(IGAMPMEngine._prepare_implicit_ipc_step)
    assert "prepare_mpm_step" in inspect.getsource(IGAMPMEngine._prepare_implicit_ipc_step)
    assert "solve_configured_implicit_system" in inspect.getsource(IGAMPMEngine._implicit_ipc_substep_once)
    assert "solve_configured_implicit_system" in inspect.getsource(IGAMPMEngine.__init__)
    assert "track_energy and" not in inspect.getsource(IGAMPMEngine.implicit_ipc_substep)
    explicit_force = inspect.getsource(IGAMPMEngine._assemble_explicit_mpm_forces)
    assert "hasattr" not in explicit_force
    assert "explicit_mpm_velocity_p2g" in explicit_force
    assert "if self.mpm.compute_traction" not in explicit_force
    assert "compute_mpm_traction" in explicit_force


def test_explicit_igampm_energy_sampling_does_not_replay_contact():
    source = inspect.getsource(ExplicitIGAMPMEngine.step)
    assert source.count("self.contact.resolve") == 1
    assert "add_contact_energy_record" in source
    assert "record_history" in source
