import sys
from functools import partial
from types import MethodType

from src.mpm.Simulation import Simulation as MPMSimulation
from src.mpm.engines.SoftParticleEngine import SoftParticleExplicitEngineMixin
from src.mpm.generator.SoftParticleCreatorMixin import SoftParticleCreatorMixin
from src.mpm.generator.SoftParticleGeneratorMixin import SoftParticleGeneratorMixin
from src.mpm.soft_particle.ContactKernel import (
    gather_soft_contact_trace_force,
    kernel_lsmpm_l_sparticle_l_sparticle_force_assemble,
    kernel_lsmpm_l_sparticle_wall_force_assemble,
    normalize_soft_contact_trace,
    project_current_soft_surface_to_contact_trace,
    reset_soft_contact_trace,
)
from src.mpm.soft_particle.NeighborKernel import (
    board_search_lsmpm_lsparticle_lsparticle_linked_cell,
    board_search_lsmpm_lsparticle_wall_linked_cell,
    rebuild_lsmpm_contact_nodes,
)
from src.mpm.soft_particle.SceneFields import (
    activate_soft_body,
    activate_soft_bounding_box,
    activate_soft_bounding_sphere,
    activate_soft_levelset_grid,
    activate_soft_material_points,
    check_material_point_number,
    check_soft_body_number,
    level_body_count,
    levelset_template_grid_count,
)
from src.mpm.soft_particle.SoftBodyKernel import (
    advect_soft_levelset,
    advect_soft_levelset_weno5,
    soft_material_point_force_reset,
    soft_surface_force_reset,
)
from src.utils.linalg import no_operation
from src.mpm.soft_particle.ReferenceMap import advect_soft_levelset_reference_map


def install_dem_soft_particle_backend(dem):
    backend = sys.modules[__name__]
    dem.sims.soft_particle_backend = backend
    dem.scene.soft_particle_backend = backend
    dem.generator.soft_particle_backend = backend
    dem.generator.bodyCreator.soft_particle_backend = backend
    return backend


def bind_explicit_engine(engine, sims, scene):
    engine.initialize_soft_levelset_transport = no_operation
    engine.advance_soft_levelset_transport = no_operation
    engine.audit_soft_levelset_domain_step = no_operation
    engine.validate_soft_levelset_departure = no_operation
    engine.correct_soft_levelset_volume_when_due = no_operation
    engine.reinitialize_soft_levelset_if_needed = MethodType(
        SoftParticleExplicitEngineMixin.skip_soft_levelset_reinitialization,
        engine,
    )
    if not sims.soft_levelset_transport:
        return

    engine.advance_soft_levelset_transport = MethodType(
        SoftParticleExplicitEngineMixin.advance_soft_levelset_transport,
        engine,
    )
    if sims.soft_levelset_domain_check:
        engine.audit_soft_levelset_domain_step = MethodType(
            SoftParticleExplicitEngineMixin.audit_soft_levelset_domain,
            engine,
        )
        engine.validate_soft_levelset_departure = MethodType(
            SoftParticleExplicitEngineMixin.validate_soft_levelset_departure,
            engine,
        )
    if sims.soft_levelset_reinitialization:
        engine.reinitialize_soft_levelset_if_needed = MethodType(
            SoftParticleExplicitEngineMixin.reinitialize_soft_levelset_if_needed,
            engine,
        )
    if sims.soft_levelset_volume_correction:
        engine.initialize_soft_levelset_transport = MethodType(
            SoftParticleExplicitEngineMixin.initialize_soft_levelset_volume_reference,
            engine,
        )
        engine.correct_soft_levelset_volume_when_due = MethodType(
            SoftParticleExplicitEngineMixin.correct_soft_levelset_volume_when_due,
            engine,
        )
    if sims.soft_levelset_advection_scheme == "ReferenceMap":
        if scene.soft_levelset_inverse_derivative is None:
            import taichi as ti

            scene.soft_levelset_inverse_derivative = ti.Matrix.field(3, 3, float, shape=scene.rigid_grid.shape[0])
        engine.initialize_soft_levelset_transport = MethodType(
            SoftParticleExplicitEngineMixin.initialize_soft_levelset_reference_map,
            engine,
        )
        engine.advect_soft_levelset = partial(
            advect_soft_levelset_reference_map,
            reference=scene.soft_levelset_initial_sdf,
            derivative=scene.soft_levelset_inverse_derivative,
        )
        return
    engine.advect_soft_levelset = (
        partial(
            advect_soft_levelset_weno5,
            maximum_cfl=sims.soft_levelset_advection_cfl,
        )
        if sims.soft_levelset_advection_scheme == "WENO5"
        else advect_soft_levelset
    )


initialize_soft_particle_options = MPMSimulation.initialize_soft_particle_options
set_soft_body_num = MPMSimulation.set_soft_body_num
set_material_point_num = MPMSimulation.set_material_point_num
set_soft_grid_num = MPMSimulation.set_soft_grid_num
set_soft_velocity_constraint_num = MPMSimulation.set_soft_velocity_constraint_num
set_soft_template_support_num = MPMSimulation.set_soft_template_support_num
set_soft_shape_function = MPMSimulation.set_soft_shape_function
set_soft_grid_storage = MPMSimulation.set_soft_grid_storage
set_soft_grid_type = MPMSimulation.set_soft_grid_type
set_soft_mechanical_grid_spacing_ratio = MPMSimulation.set_soft_mechanical_grid_spacing_ratio
set_soft_levelset_reinitialization = MPMSimulation.set_soft_levelset_reinitialization
set_levelset_contact_list_size = MPMSimulation.set_levelset_contact_list_size
soft_particle_contact_list_length = MPMSimulation.soft_particle_contact_list_length
validate_soft_particle_configuration = MPMSimulation.validate_soft_particle_configuration


def create_soft_body(creator, sims, scene, template):
    if not hasattr(creator, "create_template_soft_body"):
        creator.create_template_soft_body = MethodType(SoftParticleCreatorMixin.create_template_soft_body, creator)
    if not hasattr(creator, "sample_soft_material_points"):
        creator.sample_soft_material_points = MethodType(SoftParticleCreatorMixin.sample_soft_material_points, creator)
    return SoftParticleCreatorMixin.create_soft_body(creator, sims, scene, template)


def _bind_generator_methods(generator):
    method_names = (
        "sample_soft_material_points",
        "generate_template_soft_body",
        "lattice_template_soft_body",
        "insert_soft_levelset",
    )
    for name in method_names:
        if not hasattr(generator, name):
            setattr(generator, name, MethodType(getattr(SoftParticleGeneratorMixin, name), generator))


def generate_soft_bodys(generator, scene):
    _bind_generator_methods(generator)
    return SoftParticleGeneratorMixin.generate_soft_bodys(generator, scene)


def lattice_soft_bodys(generator, scene):
    _bind_generator_methods(generator)
    return SoftParticleGeneratorMixin.lattice_soft_bodys(generator, scene)


def add_soft_levelsets_to_scene(generator, scene):
    _bind_generator_methods(generator)
    return SoftParticleGeneratorMixin.add_soft_levelsets_to_scene(generator, scene)


def insert_soft_body_batch(generator, scene, template, body_count):
    _bind_generator_methods(generator)
    return generator.insert_soft_levelset(scene, template, 0, body_count, body_count)


euler_lsmpm_integration = SoftParticleExplicitEngineMixin.euler_lsmpm_integration
ensure_soft_grid_reference_mass = SoftParticleExplicitEngineMixin.ensure_soft_grid_reference_mass
soft_surface_force_reset = soft_surface_force_reset
soft_material_point_force_reset = soft_material_point_force_reset


def update_verlet_table_lsparticle_lsparticle_scenod_layer(neighbor, scene):
    rebuild_lsmpm_contact_nodes(
        int(scene.particleNum[0]),
        scene.rigid,
        scene.soft,
        scene.soft_surface_point_id,
        scene.ls_contact_body,
        scene.ls_contact_kind,
        scene.ls_contact_ref,
        scene.ls_contact_body_start,
        scene.ls_contact_body_end,
        scene.ls_contact_count,
    )
    scene.lsContactNodeNum[0] = int(scene.ls_contact_count[None])
    board_search_lsmpm_lsparticle_lsparticle_linked_cell(
        int(scene.particleNum[0]),
        neighbor.sims.point_particle_coordination_number,
        neighbor.sims.point_verlet_distance,
        neighbor.pplist,
        neighbor.potential_list_point_particle,
        neighbor.particle_particle,
        neighbor.lsparticle_lsparticle,
        scene.rigid,
        scene.box,
        scene.vertice,
        scene.rigid_grid,
        scene.soft_point,
        scene.ls_contact_kind,
        scene.ls_contact_ref,
        scene.ls_contact_body_start,
        scene.ls_contact_body_end,
        scene.ls_contact_count,
    )


def update_verlet_table_lsparticle_wall_scenod_layer(neighbor, scene):
    rebuild_lsmpm_contact_nodes(
        int(scene.particleNum[0]),
        scene.rigid,
        scene.soft,
        scene.soft_surface_point_id,
        scene.ls_contact_body,
        scene.ls_contact_kind,
        scene.ls_contact_ref,
        scene.ls_contact_body_start,
        scene.ls_contact_body_end,
        scene.ls_contact_count,
    )
    scene.lsContactNodeNum[0] = int(scene.ls_contact_count[None])
    board_search_lsmpm_lsparticle_wall_linked_cell(
        int(scene.particleNum[0]),
        neighbor.sims.point_wall_coordination_number,
        neighbor.sims.point_verlet_distance,
        neighbor.pwlist,
        neighbor.potential_list_point_wall,
        neighbor.particle_wall,
        neighbor.lsparticle_wall,
        scene.wall,
        scene.rigid,
        scene.vertice,
        scene.box,
        scene.soft_point,
        scene.ls_contact_kind,
        scene.ls_contact_ref,
        scene.ls_contact_body_start,
        scene.ls_contact_body_end,
    )


def tackle_lsparticle_lsparticle_contact(model, sims, scene, pcontact):
    reset_soft_contact_trace(
        int(scene.softNum[0]),
        int(scene.softMaxLogicalGridNum[0]),
        scene.soft,
        scene.soft_levelset_velocity,
        scene.soft_levelset_projection_weight,
        scene.soft_contact_trace_force,
        scene.soft_contact_trace_uncovered,
    )
    project_current_soft_surface_to_contact_trace(
        int(scene.surfaceNum[0]),
        scene.surface,
        scene.rigid,
        scene.box,
        scene.soft,
        scene.vertice,
        scene.soft_levelset_velocity,
        scene.soft_levelset_projection_weight,
    )
    normalize_soft_contact_trace(
        int(scene.softNum[0]),
        int(scene.softMaxLogicalGridNum[0]),
        scene.soft,
        scene.soft_levelset_velocity,
        scene.soft_levelset_projection_weight,
    )
    kernel_lsmpm_l_sparticle_l_sparticle_force_assemble(
        int(scene.lsContactNodeNum[0]),
        sims.dt,
        sims.max_material_num,
        model.surfaceProps,
        scene.rigid,
        scene.rigid_grid,
        scene.vertice,
        scene.box,
        scene.soft,
        scene.soft_levelset_velocity,
        scene.soft_levelset_projection_weight,
        scene.soft_contact_trace_force,
        scene.soft_contact_trace_uncovered,
        scene.ls_contact_body,
        scene.ls_contact_kind,
        scene.ls_contact_ref,
        model.cplist,
        pcontact.hist_lsparticle_lsparticle,
        model.model_type,
    )
    gather_soft_contact_trace_force(
        int(scene.surfaceNum[0]),
        scene.surface,
        scene.rigid,
        scene.box,
        scene.soft,
        scene.vertice,
        scene.soft_contact_trace_force,
    )


def tackle_lsparticle_wall_contact(model, sims, scene, pcontact):
    kernel_lsmpm_l_sparticle_wall_force_assemble(
        int(scene.lsContactNodeNum[0]),
        sims.dt,
        sims.max_material_num,
        model.surfaceProps,
        scene.rigid,
        scene.vertice,
        scene.box,
        scene.wall,
        scene.soft,
        scene.soft_point,
        scene.soft_surface_point_id,
        scene.ls_contact_body,
        scene.ls_contact_kind,
        scene.ls_contact_ref,
        model.cplist,
        pcontact.hist_lsparticle_wall,
        model.model_type,
    )
