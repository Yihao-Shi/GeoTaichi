"""LSMPM soft-particle integration and level-set maintenance."""

import math

import numpy as np


class SoftParticleExplicitEngineMixin(object):
    def ensure_soft_grid_reference_mass(self, scene):
        """Refresh cached TLMPM nodal mass when the soft-body set changes."""
        soft_num = int(scene.softNum[0])
        grid_num = int(scene.softGridNum[0])
        point_num = int(scene.softPointNum[0])
        # Field identity distinguishes a rebuilt scene even when its counts
        # happen to match the previous one handled by this engine.
        signature = (id(scene.soft_grid), soft_num, grid_num, point_num)
        if getattr(self, "soft_grid_reference_mass_signature", None) == signature:
            return
        if soft_num <= 0 or grid_num <= 0 or point_num <= 0:
            self.soft_grid_reference_mass_signature = signature
            return

        from src.mpm.soft_particle.SoftBodyKernel import (
            precompute_soft_grid_reference_mass_,
        )

        precompute_soft_grid_reference_mass_(
            soft_num,
            grid_num,
            point_num,
            scene.soft,
            scene.soft_point,
            scene.soft_grid,
            scene.soft_shape_node,
            scene.soft_shape,
            scene.soft_shape_count,
            scene.rigid,
        )
        self.soft_grid_reference_mass_signature = signature

    def ensure_soft_particle_constitutive_state(self, scene):
        """Initialize stress once, then reuse each step's terminal state."""
        point_num = int(scene.softPointNum[0])
        signature = (
            id(scene.soft_point),
            point_num,
            id(self.soft_particle_material.matProps),
        )
        if getattr(self, "soft_constitutive_state_signature", None) == signature:
            return
        if point_num <= 0:
            self.soft_constitutive_state_signature = signature
            return

        from src.mpm.soft_particle.SoftBodyKernel import (
            initialize_soft_particle_constitutive_state_,
        )

        initialize_soft_particle_constitutive_state_(
            point_num,
            scene.soft_point,
            self.soft_particle_material.matProps,
            self.soft_particle_material.stateVars,
        )
        self.soft_constitutive_state_signature = signature

    def audit_soft_levelset_domain(self, sims, scene):
        from src.mpm.soft_particle.SoftBodyKernel import (
            min_soft_levelset_domain_margin_,
        )

        margin = float(min_soft_levelset_domain_margin_(int(scene.softNum[0]), scene.soft, scene.box))
        sims.soft_levelset_domain_audit_count += 1
        sims.soft_levelset_domain_min_margin_cells = min(sims.soft_levelset_domain_min_margin_cells, margin)
        tolerance = sims.soft_levelset_domain_tolerance_cells
        if not np.isfinite(margin) or margin < -tolerance:
            raise RuntimeError(
                "LSMPM soft-particle deformation exceeded the fixed SDF "
                f"domain by {max(-margin, 0.0):.6e} grid cells at step "
                f"{int(sims.current_step)}. Increase the template level-set "
                "extent or its deformation-padding cells."
            )

    def reinitialize_soft_levelset_if_needed(self, sims, scene):
        if sims.current_step % sims.soft_levelset_reinit_check_interval != 0:
            return False

        from src.mpm.soft_particle.SoftBodyKernel import (
            monitor_soft_levelset_signed_distance_error_,
            reinitialize_soft_levelset_step_,
            store_soft_levelset_reinit_reference_,
        )

        monitor_soft_levelset_signed_distance_error_(
            int(scene.softNum[0]),
            int(scene.softMaxLogicalGridNum[0]),
            sims.soft_levelset_reinit_monitor_band,
            scene.soft,
            scene.rigid_grid,
            scene.box,
            scene.soft_levelset_grad_error,
        )
        sims.soft_levelset_max_grad_error = float(scene.soft_levelset_grad_error[None])
        if sims.soft_levelset_max_grad_error <= sims.soft_levelset_reinit_grad_threshold:
            return False
        grad_error_before = sims.soft_levelset_max_grad_error

        store_soft_levelset_reinit_reference_(
            int(scene.softNum[0]), int(scene.softMaxLogicalGridNum[0]), scene.soft, scene.rigid_grid
        )
        iterations = sims.soft_levelset_reinit_iterations
        if iterations <= 0:
            iterations = max(1, int(math.ceil(sims.soft_levelset_reinit_band / sims.soft_levelset_reinit_cfl)))
        for _ in range(iterations):
            reinitialize_soft_levelset_step_(
                int(scene.softNum[0]),
                int(scene.softMaxLogicalGridNum[0]),
                sims.soft_levelset_reinit_band,
                sims.soft_levelset_reinit_cfl,
                scene.soft,
                scene.rigid_grid,
                scene.box,
            )
        monitor_soft_levelset_signed_distance_error_(
            int(scene.softNum[0]),
            int(scene.softMaxLogicalGridNum[0]),
            sims.soft_levelset_reinit_monitor_band,
            scene.soft,
            scene.rigid_grid,
            scene.box,
            scene.soft_levelset_grad_error,
        )
        sims.soft_levelset_max_grad_error = float(scene.soft_levelset_grad_error[None])
        sims.soft_levelset_reinitialization_count += 1
        sims.soft_levelset_last_reinitialization_step = int(sims.current_step)
        sims.soft_levelset_last_grad_error_before = float(grad_error_before)
        sims.soft_levelset_last_grad_error_after = sims.soft_levelset_max_grad_error
        sims.soft_levelset_reinitialization_events.append(
            {
                "count": sims.soft_levelset_reinitialization_count,
                "step": int(sims.current_step),
                "grad_error_before": sims.soft_levelset_last_grad_error_before,
                "grad_error_after": sims.soft_levelset_last_grad_error_after,
            }
        )
        return True

    def skip_soft_levelset_reinitialization(self, sims, scene):
        return False

    def initialize_soft_levelset_volume_reference(self, sims, scene):
        soft_num = int(scene.softNum[0])
        if sims.soft_levelset_volume_reference_count >= soft_num:
            return
        from src.mpm.soft_particle.LevelSet import (
            accumulate_soft_levelset_volume_integrals_,
            accumulate_soft_material_volume_,
            initialize_soft_levelset_volume_reference_,
            reset_soft_levelset_material_volume_,
            reset_soft_levelset_volume_integrals_,
        )

        soft_start = int(sims.soft_levelset_volume_reference_count)
        point_num = int(scene.softPointNum[0])
        reset_soft_levelset_material_volume_(soft_num, scene.soft_levelset_material_volume)
        accumulate_soft_material_volume_(
            soft_num,
            point_num,
            scene.soft_point,
            scene.rigid,
            scene.soft_levelset_material_volume,
        )
        reset_soft_levelset_volume_integrals_(
            soft_num,
            scene.soft_levelset_current_volume,
            scene.soft_levelset_interface_measure,
        )
        accumulate_soft_levelset_volume_integrals_(
            soft_num,
            int(scene.softMaxLogicalGridNum[0]),
            sims.soft_levelset_volume_epsilon_cells,
            scene.soft,
            scene.rigid_grid,
            scene.box,
            scene.soft_levelset_current_volume,
            scene.soft_levelset_interface_measure,
        )
        initialize_soft_levelset_volume_reference_(
            soft_start,
            soft_num,
            scene.soft_levelset_reference_sdf_volume,
            scene.soft_levelset_reference_material_volume,
            scene.soft_levelset_material_volume,
            scene.soft_levelset_current_volume,
            scene.soft_levelset_target_volume,
            scene.soft_levelset_volume_shift,
            scene.soft_levelset_volume_cumulative_shift,
            scene.soft_levelset_volume_error,
            scene.soft_levelset_volume_max_error,
        )
        sims.soft_levelset_volume_reference_count = soft_num
        sims.soft_levelset_volume_reference_initialized = True
        sims.soft_levelset_volume_max_error = 0.0

    def correct_soft_levelset_volume(self, sims, scene):
        from src.mpm.soft_particle.LevelSet import (
            accumulate_soft_levelset_volume_integrals_,
            accumulate_soft_material_volume_,
            apply_soft_levelset_volume_shift_,
            compute_soft_levelset_safeguarded_shift_,
            compute_soft_levelset_volume_shift_,
            initialize_soft_levelset_volume_bracket_,
            prepare_soft_levelset_target_volume_,
            reset_soft_levelset_material_volume_,
            reset_soft_levelset_volume_integrals_,
        )

        soft_num = int(scene.softNum[0])
        point_num = int(scene.softPointNum[0])
        reset_soft_levelset_material_volume_(soft_num, scene.soft_levelset_material_volume)
        accumulate_soft_material_volume_(
            soft_num,
            point_num,
            scene.soft_point,
            scene.rigid,
            scene.soft_levelset_material_volume,
        )
        prepare_soft_levelset_target_volume_(
            soft_num,
            scene.soft_levelset_reference_sdf_volume,
            scene.soft_levelset_reference_material_volume,
            scene.soft_levelset_material_volume,
            scene.soft_levelset_target_volume,
        )

        reset_soft_levelset_volume_integrals_(
            soft_num,
            scene.soft_levelset_current_volume,
            scene.soft_levelset_interface_measure,
        )
        accumulate_soft_levelset_volume_integrals_(
            soft_num,
            int(scene.softMaxLogicalGridNum[0]),
            sims.soft_levelset_volume_epsilon_cells,
            scene.soft,
            scene.rigid_grid,
            scene.box,
            scene.soft_levelset_current_volume,
            scene.soft_levelset_interface_measure,
        )
        initialize_soft_levelset_volume_bracket_(
            soft_num,
            sims.soft_levelset_volume_tolerance,
            sims.soft_levelset_volume_max_shift_cells,
            scene.soft,
            scene.box,
            scene.soft_levelset_target_volume,
            scene.soft_levelset_current_volume,
            scene.soft_levelset_volume_lower_shift,
            scene.soft_levelset_volume_upper_shift,
            scene.soft_levelset_volume_trial_shift,
            scene.soft_levelset_volume_shift,
        )
        apply_soft_levelset_volume_shift_(
            soft_num,
            int(scene.softMaxLogicalGridNum[0]),
            scene.soft,
            scene.rigid_grid,
            scene.soft_levelset_volume_shift,
            scene.soft_levelset_volume_cumulative_shift,
        )

        for _ in range(sims.soft_levelset_volume_iterations):
            reset_soft_levelset_volume_integrals_(
                soft_num,
                scene.soft_levelset_current_volume,
                scene.soft_levelset_interface_measure,
            )
            accumulate_soft_levelset_volume_integrals_(
                soft_num,
                int(scene.softMaxLogicalGridNum[0]),
                sims.soft_levelset_volume_epsilon_cells,
                scene.soft,
                scene.rigid_grid,
                scene.box,
                scene.soft_levelset_current_volume,
                scene.soft_levelset_interface_measure,
            )
            compute_soft_levelset_safeguarded_shift_(
                soft_num,
                sims.soft_levelset_volume_tolerance,
                scene.soft_levelset_target_volume,
                scene.soft_levelset_current_volume,
                scene.soft_levelset_interface_measure,
                scene.soft_levelset_volume_lower_shift,
                scene.soft_levelset_volume_upper_shift,
                scene.soft_levelset_volume_trial_shift,
                scene.soft_levelset_volume_shift,
                scene.soft_levelset_volume_error,
                scene.soft_levelset_volume_max_error,
            )
            max_error = float(scene.soft_levelset_volume_max_error[None])
            if max_error <= sims.soft_levelset_volume_tolerance:
                break
            apply_soft_levelset_volume_shift_(
                soft_num,
                int(scene.softMaxLogicalGridNum[0]),
                scene.soft,
                scene.rigid_grid,
                scene.soft_levelset_volume_shift,
                scene.soft_levelset_volume_cumulative_shift,
            )

        reset_soft_levelset_volume_integrals_(
            soft_num,
            scene.soft_levelset_current_volume,
            scene.soft_levelset_interface_measure,
        )
        accumulate_soft_levelset_volume_integrals_(
            soft_num,
            int(scene.softMaxLogicalGridNum[0]),
            sims.soft_levelset_volume_epsilon_cells,
            scene.soft,
            scene.rigid_grid,
            scene.box,
            scene.soft_levelset_current_volume,
            scene.soft_levelset_interface_measure,
        )
        compute_soft_levelset_volume_shift_(
            soft_num,
            sims.soft_levelset_volume_tolerance,
            sims.soft_levelset_volume_max_shift_cells,
            scene.soft,
            scene.box,
            scene.soft_levelset_target_volume,
            scene.soft_levelset_current_volume,
            scene.soft_levelset_interface_measure,
            scene.soft_levelset_volume_shift,
            scene.soft_levelset_volume_error,
            scene.soft_levelset_volume_max_error,
        )
        sims.soft_levelset_volume_max_error = float(scene.soft_levelset_volume_max_error[None])
        sims.soft_levelset_volume_correction_count += 1

    def validate_soft_levelset_departure(self, sims, departure_excess):
        if not np.isfinite(departure_excess) or departure_excess > sims.soft_levelset_domain_tolerance_cells:
            raise RuntimeError(
                "LSMPM SDF transport exceeded the fixed SDF domain by "
                f"{float(departure_excess):.6e} grid cells at step "
                f"{int(sims.current_step)}. Increase the template level-set "
                "extent or reduce the time step."
            )

    def correct_soft_levelset_volume_when_due(self, sims, scene, volume_correction_due, reinitialized):
        if volume_correction_due or reinitialized:
            SoftParticleExplicitEngineMixin.correct_soft_levelset_volume(self, sims, scene)

    def advance_soft_levelset_transport(self, sims, scene):
        self.audit_soft_levelset_domain_step(sims, scene)
        sims.soft_levelset_advection_elapsed += float(sims.dt[None])
        sims.soft_levelset_advection_steps += 1
        volume_correction_due = False
        if sims.soft_levelset_advection_steps >= sims.soft_levelset_advection_interval:
            advection_arguments = (
                int(scene.softNum[0]),
                int(scene.softMaxLogicalGridNum[0]),
                int(scene.softPointNum[0]),
                int(sims.soft_shape_function_type),
                float(sims.soft_levelset_projection_monitor_band),
                int(sims.soft_levelset_projection_extension_iterations),
                sims.soft_levelset_advection_elapsed,
                scene.soft,
                scene.rigid_grid,
                scene.soft_point,
                scene.soft_grid,
                scene.soft_shape_node,
                scene.soft_dshape,
                scene.soft_shape_count,
                scene.soft_levelset_velocity,
                scene.soft_levelset_projection_weight,
                scene.soft_levelset_projection_band_nodes,
                scene.soft_levelset_projection_uncovered_nodes,
                scene.soft_levelset_projection_max_support_loss,
                scene.rigid,
                scene.box,
            )
            departure_excess = self.advect_soft_levelset(*advection_arguments)
            projection_band_nodes = int(scene.soft_levelset_projection_band_nodes[None])
            projection_uncovered_nodes = int(scene.soft_levelset_projection_uncovered_nodes[None])
            projection_uncovered_fraction = (
                projection_uncovered_nodes / projection_band_nodes if projection_band_nodes > 0 else 0.0
            )
            sims.soft_levelset_projection_band_nodes = projection_band_nodes
            sims.soft_levelset_projection_uncovered_nodes = projection_uncovered_nodes
            sims.soft_levelset_projection_max_uncovered_fraction = max(
                sims.soft_levelset_projection_max_uncovered_fraction,
                projection_uncovered_fraction,
            )
            sims.soft_levelset_projection_max_support_loss = max(
                sims.soft_levelset_projection_max_support_loss,
                float(scene.soft_levelset_projection_max_support_loss[None]),
            )
            sims.soft_levelset_domain_max_departure_excess_cells = max(
                sims.soft_levelset_domain_max_departure_excess_cells,
                float(departure_excess),
            )
            self.validate_soft_levelset_departure(sims, departure_excess)
            sims.soft_levelset_advection_elapsed = 0.0
            sims.soft_levelset_advection_steps = 0
            sims.soft_levelset_volume_update_count += 1
            volume_correction_due = sims.soft_levelset_volume_update_count % sims.soft_levelset_volume_interval == 0
        reinitialized = self.reinitialize_soft_levelset_if_needed(sims, scene)
        self.correct_soft_levelset_volume_when_due(sims, scene, volume_correction_due, reinitialized)

    def euler_lsmpm_integration(self, sims, scene):
        from src.mpm.MaterialManager import SoftParticleMaterialManager
        from src.mpm.soft_particle.SoftBodyKernel import (
            apply_soft_body_translation_constraint_,
            finalize_soft_body_bounds_,
            finalize_soft_body_rotation_,
            prepare_soft_body_surface_frame_,
            reduce_soft_body_kinematics_,
            reduce_soft_body_surface_frame_,
            remap_soft_grid_velocity_,
            reset_soft_body_kinematic_reduction_,
            reset_soft_grid_step_,
            soft_body_force_p2g_,
            soft_body_g2p_,
            soft_surface_force_p2g_,
            track_soft_surface_points_,
            update_soft_grid_kinematic_,
            update_soft_particle_stress_,
        )

        sims.timer.begin("LSMPM Rigid integration")
        self.euler_level_set_integration(sims, scene)
        sims.timer.end("LSMPM Rigid integration")

        sims.timer.begin("LSMPM Setup")
        soft_num = int(scene.softNum[0])
        if soft_num <= 0:
            sims.timer.end("LSMPM Setup")
            return
        grid_num = int(scene.softGridNum[0])
        fixed_grid_num = int(scene.softVelocityConstraintNum[0])
        point_num = int(scene.softPointNum[0])
        surface_num = int(scene.surfaceNum[0])
        SoftParticleExplicitEngineMixin.ensure_soft_grid_reference_mass(self, scene)
        self.initialize_soft_levelset_transport(sims, scene)

        # Staged generation may start with no soft body. Build the material
        # table exactly once, when the first active soft body supplies the
        # authoritative material IDs.
        if self.soft_particle_material is None:
            self.soft_particle_material = SoftParticleMaterialManager()
            self.soft_particle_material.setup(scene, sims)
        SoftParticleExplicitEngineMixin.ensure_soft_particle_constitutive_state(self, scene)
        sims.timer.end("LSMPM Setup")

        sims.timer.begin("LSMPM Grid reset")
        reset_soft_grid_step_(grid_num, scene.soft_grid)
        sims.timer.end("LSMPM Grid reset")

        sims.timer.begin("LSMPM Force P2G")
        soft_body_force_p2g_(
            point_num,
            scene.soft,
            scene.soft_point,
            scene.soft_grid,
            scene.soft_shape_node,
            scene.soft_shape,
            scene.soft_dshape,
            scene.soft_shape_count,
            scene.rigid,
            sims.gravity,
        )
        sims.timer.end("LSMPM Force P2G")

        sims.timer.begin("LSMPM Surface force P2G")
        soft_surface_force_p2g_(
            surface_num,
            scene.soft,
            scene.soft_grid,
            scene.surface_shape_node,
            scene.surface_shape,
            scene.surface_shape_count,
            scene.vertice,
            scene.surface,
            scene.rigid,
        )
        sims.timer.end("LSMPM Surface force P2G")

        sims.timer.begin("LSMPM Grid kinematic")
        update_soft_grid_kinematic_(
            soft_num,
            grid_num,
            fixed_grid_num,
            sims.dt,
            scene.soft,
            scene.soft_grid,
            scene.soft_grid_owner,
            scene.soft_grid_local,
            scene.soft_velocity_constraint,
            scene.box,
            scene.material,
        )
        sims.timer.end("LSMPM Grid kinematic")

        sims.timer.begin("LSMPM G2P")
        soft_body_g2p_(
            point_num,
            sims.soft_pic_fraction,
            sims.dt,
            scene.soft,
            scene.soft_point,
            scene.soft_grid,
            scene.soft_shape_node,
            scene.soft_shape,
            scene.soft_shape_count,
            scene.rigid,
        )
        sims.timer.end("LSMPM G2P")

        sims.timer.begin("LSMPM Body reduction")
        reset_soft_body_kinematic_reduction_(soft_num, scene.soft)
        reduce_soft_body_kinematics_(
            point_num,
            scene.soft,
            scene.soft_point,
            scene.rigid,
        )
        sims.timer.end("LSMPM Body reduction")

        sims.timer.begin("LSMPM Constraint correction")
        apply_soft_body_translation_constraint_(
            point_num,
            scene.soft,
            scene.soft_point,
            scene.rigid,
        )
        sims.timer.end("LSMPM Constraint correction")

        sims.timer.begin("LSMPM Velocity remap P2G")
        remap_soft_grid_velocity_(
            grid_num,
            fixed_grid_num,
            point_num,
            scene.soft,
            scene.soft_point,
            scene.soft_grid,
            scene.soft_velocity_constraint,
            scene.soft_shape_node,
            scene.soft_shape,
            scene.soft_shape_count,
            scene.rigid,
        )
        sims.timer.end("LSMPM Velocity remap P2G")

        sims.timer.begin("LSMPM Stress update")
        update_soft_particle_stress_(
            point_num,
            sims.dt,
            scene.soft,
            scene.soft_point,
            scene.soft_grid,
            scene.soft_shape_node,
            scene.soft_dshape,
            scene.soft_shape_count,
            scene.rigid,
            self.soft_particle_material.matProps,
            self.soft_particle_material.stateVars,
        )
        sims.timer.end("LSMPM Stress update")

        sims.timer.begin("LSMPM Surface frame reduction")
        prepare_soft_body_surface_frame_(
            soft_num,
            scene.soft,
            scene.rigid,
            scene.box,
        )
        reduce_soft_body_surface_frame_(
            point_num,
            scene.soft,
            scene.soft_point,
            scene.rigid,
        )
        sims.timer.end("LSMPM Surface frame reduction")

        sims.timer.begin("LSMPM Surface rotation")
        finalize_soft_body_rotation_(
            soft_num,
            sims.dt,
            scene.soft,
            scene.rigid,
        )
        sims.timer.end("LSMPM Surface rotation")

        sims.timer.begin("LSMPM Surface point tracking")
        track_soft_surface_points_(
            surface_num,
            sims.dt,
            scene.soft,
            scene.soft_grid,
            scene.surface_shape_node,
            scene.surface_shape,
            scene.surface_shape_count,
            scene.vertice,
            scene.surface,
            scene.rigid,
            scene.box,
        )
        sims.timer.end("LSMPM Surface point tracking")

        sims.timer.begin("LSMPM Bounding volume finalize")
        finalize_soft_body_bounds_(
            soft_num,
            scene.soft,
            scene.rigid,
            scene.box,
            scene.particle,
        )
        sims.timer.end("LSMPM Bounding volume finalize")

        sims.timer.begin("LSMPM SDF transport")
        self.advance_soft_levelset_transport(sims, scene)
        sims.timer.end("LSMPM SDF transport")


__all__ = ["SoftParticleExplicitEngineMixin"]
