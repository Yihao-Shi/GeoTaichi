#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_bin="${GT_LOCAL_PYTHON:-/opt/anaconda3/envs/geotaichi/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    python_bin="python"
fi
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

run_pytest() {
    "$python_bin" -m pytest -q "$@"
}

run_runtime_contract() {
    run_pytest \
        tests/unit/utils/test_solver_runtime.py \
        tests/unit/tools/test_audit_postpaper_solvers.py \
        tests/unit/fem/test_runtime_function_binding.py \
        tests/unit/dem/test_solver_preintegration_callback.py \
        tests/unit/dem/test_simulation_runtime_binding.py \
        tests/unit/examples/test_cpt_reference.py \
        tests/unit/coupling/test_explicit_runtime_binding.py \
        tests/unit/coupling/test_solver_factory_routing.py \
        tests/unit/coupling/test_surface_patch_ownership.py \
        tests/unit/coupling/test_all_solver_step_retry.py \
        tests/unit/coupling/test_implicit_step_retry.py
    "$python_bin" tools/audit_postpaper_solvers.py
}

run_explicit_contact() {
    run_pytest \
        tests/unit/fem/test_fem_soft_particle.py \
        tests/unit/fem/test_fem_contact.py \
        tests/unit/fedem/test_levelset_broadphase.py \
        tests/unit/fedem/test_soft_particle_contact_history.py \
        tests/integration/fem/test_fem_contact_solver.py \
        tests/integration/fedem/test_explicit_surface_coupling.py \
        tests/integration/fempm/test_explicit_fempm_surface.py
}

run_implicit_coupling() {
    run_pytest \
        tests/unit/coupling/test_fem_affine_matrix_capacity.py \
        tests/integration/fem/test_fem_solvers.py \
        tests/integration/fedem/test_affine_ipc_coupling.py \
        tests/integration/fempm/test_ipc.py \
        tests/integration/fempm/test_ipc_2d_axisymmetric.py
}

run_iga_igampm() {
    run_pytest \
        tests/unit/iga/solver/test_iga_backend.py \
        tests/unit/iga/solver/test_igampm_implicit_lifecycle.py \
        tests/unit/iga/solver/test_igampm_output_schedule.py \
        tests/unit/iga/solver/test_igampm_coo_assembly.py \
        tests/integration/igampm
}

run_restart_capacity() {
    run_pytest \
        tests/unit/fedem/test_checkpoint_capacity.py \
        tests/unit/fedem/test_funnel_wall_geometry.py \
        tests/unit/fedem/test_mixed_particle_mesh.py \
        tests/unit/fedem/test_soft_particle_contact_history.py \
        tests/unit/coupling/test_fem_affine_matrix_capacity.py \
        tests/unit/iga/solver/test_igampm_output_schedule.py
}

run_solver_integrations() {
    run_pytest \
        tests/integration/dem \
        tests/integration/fem \
        tests/integration/fedem \
        tests/integration/fempm \
        tests/integration/igampm \
        tests/integration/mpm/test_direct_mpm_backend.py \
        tests/integration/mpm/test_solid_engine_kernel_dispatch.py
}

run_cpt_smoke() (
    local smoke_root
    smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/geotaichi-cpt-smoke.XXXXXX")"
    trap 'rm -rf "$smoke_root"' EXIT
    "$python_bin" examples/fempm/cpt_dp.py \
        --contact explicit --arch "${GT_LOCAL_ARCH:-cpu}" \
        --dt 1.0e-5 --time 1.0e-5 --save-interval 1.0e-5 \
        --resolution-scale 8 --output-dir "$smoke_root/fempm-explicit"
    "$python_bin" examples/igampm/cpt_dp.py \
        --contact explicit --arch "${GT_LOCAL_ARCH:-cpu}" \
        --dt 1.0e-5 --time 1.0e-5 --save-interval 1.0e-5 \
        --resolution-scale 8 --output-dir "$smoke_root/igampm-explicit"
    "$python_bin" examples/fempm/cpt_dp.py \
        --contact ipc --arch "${GT_LOCAL_ARCH:-cpu}" \
        --dt 5.0e-4 --time 5.0e-4 --save-interval 5.0e-4 \
        --resolution-scale 8 --output-dir "$smoke_root/fempm-ipc"
    "$python_bin" examples/igampm/cpt_dp.py \
        --contact ipc --arch "${GT_LOCAL_ARCH:-cpu}" \
        --dt 5.0e-4 --time 5.0e-4 --save-interval 5.0e-4 \
        --resolution-scale 8 --output-dir "$smoke_root/igampm-ipc"
)

run_axisymmetric_smoke() (
    local smoke_root
    smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/geotaichi-axisymmetric-smoke.XXXXXX")"
    trap 'rm -rf "$smoke_root"' EXIT
    export MPLCONFIGDIR="$smoke_root/.mplconfig"
    mkdir -p "$MPLCONFIGDIR"
    "$python_bin" examples/fem/axisymmetric_annulus.py \
        --solver explicit --arch "${GT_LOCAL_ARCH:-cpu}" \
        --radial-divisions 2 --axial-divisions 2 --steps 1 --dt 1.0e-5 \
        --output-interval 1 --output-dir "$smoke_root/fem-explicit"
    "$python_bin" examples/fem/axisymmetric_annulus.py \
        --solver implicit --arch "${GT_LOCAL_ARCH:-cpu}" \
        --radial-divisions 2 --axial-divisions 2 --steps 1 --dt 1.0e-3 \
        --output-interval 1 --output-dir "$smoke_root/fem-implicit"
    "$python_bin" examples/iga/axisymmetric_annulus.py \
        --solver explicit --arch "${GT_LOCAL_ARCH:-cpu}" \
        --radial-control-points 3 --axial-control-points 3 \
        --steps 1 --dt 1.0e-5 --output-interval 1 \
        --output-dir "$smoke_root/iga-explicit"
    "$python_bin" examples/iga/axisymmetric_annulus.py \
        --solver implicit --arch "${GT_LOCAL_ARCH:-cpu}" \
        --radial-control-points 3 --axial-control-points 3 \
        --steps 1 --dt 1.0e-3 --output-interval 1 \
        --output-dir "$smoke_root/iga-implicit"
    "$python_bin" examples/mpm/AxisyExample/axisymmetric_annulus.py \
        --solver explicit --arch "${GT_LOCAL_ARCH:-cpu}" \
        --cell-size 0.2 --particles-per-cell 1 \
        --steps 1 --dt 1.0e-5 --output-interval 1 \
        --output-dir "$smoke_root/mpm-explicit"
    "$python_bin" examples/mpm/AxisyExample/axisymmetric_annulus.py \
        --solver implicit --arch "${GT_LOCAL_ARCH:-cpu}" \
        --cell-size 0.2 --particles-per-cell 1 \
        --steps 1 --dt 1.0e-3 --output-interval 1 \
        --output-dir "$smoke_root/mpm-implicit"
)

run_abd_smoke() (
    local smoke_root
    smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/geotaichi-abd-smoke.XXXXXX")"
    trap 'rm -rf "$smoke_root"' EXIT
    export MPLCONFIGDIR="$smoke_root/.mplconfig"
    mkdir -p "$MPLCONFIGDIR"
    "$python_bin" examples/mpdem/AffineBody/MPMAffineTriaxial/mpm_affine_triaxial.py \
        --arch "${GT_LOCAL_ARCH:-cpu}" \
        --soft-count 1 --affine-count 1 --particles-per-cell 1 \
        --steps 1 --dt 1.0e-4 --output-interval 1 \
        --output-dir "$smoke_root/mpm-triaxial"
    "$python_bin" examples/mpdem/AffineBody/MPMAffineDeposition/mpm_affine_deposition.py \
        --arch "${GT_LOCAL_ARCH:-cpu}" \
        --soft-count 1 --affine-count 1 --particles-per-cell 1 \
        --steps 1 --dt 1.0e-4 --output-interval 1 \
        --output-dir "$smoke_root/mpm-deposition"
    "$python_bin" examples/fedem/FEMAffineTriaxial/fem_affine_triaxial.py \
        --arch "${GT_LOCAL_ARCH:-cpu}" \
        --fem-divisions 1 \
        --steps 1 --dt 1.0e-4 --output-interval 1 \
        --output-dir "$smoke_root/fem-triaxial"
    "$python_bin" examples/fedem/FEMAffineDeposition/fem_affine_deposition.py \
        --arch "${GT_LOCAL_ARCH:-cpu}" \
        --particle-count 1 --fem-divisions 2 \
        --steps 1 --dt 1.0e-4 --output-interval 1 \
        --output-dir "$smoke_root/fem-deposition"
)

run_group() {
    case "$1" in
        runtime-contract) run_runtime_contract ;;
        explicit-contact) run_explicit_contact ;;
        implicit-coupling) run_implicit_coupling ;;
        iga-igampm) run_iga_igampm ;;
        restart-capacity) run_restart_capacity ;;
        solver-integrations) run_solver_integrations ;;
        cpt-smoke) run_cpt_smoke ;;
        axisymmetric-smoke) run_axisymmetric_smoke ;;
        abd-smoke) run_abd_smoke ;;
        all)
            run_runtime_contract
            run_explicit_contact
            run_implicit_coupling
            run_iga_igampm
            run_restart_capacity
            run_solver_integrations
            run_cpt_smoke
            run_axisymmetric_smoke
            run_abd_smoke
            ;;
        *)
            local choices
            choices='{runtime-contract|explicit-contact|implicit-coupling|'
            choices+='iga-igampm|restart-capacity|solver-integrations|'
            choices+='cpt-smoke|axisymmetric-smoke|abd-smoke|all}'
            printf 'usage: %s %s\n' "$0" "$choices" >&2
            return 2
            ;;
    esac
}

run_group "${1:-all}"
