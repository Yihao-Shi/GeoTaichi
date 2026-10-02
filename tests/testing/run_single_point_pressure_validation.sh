#!/usr/bin/env bash
# Run from the repository root; gate full collapses on pressure regressions.
set -euo pipefail
test -f src/mpm/engines/ULSemiImplicitTwoPhaseEngine.py
validation_root="${1:?Supply an absolute validation output directory}"
case_python="${GT_POSTPAPER_PYTHON:-python3}"
mkdir -p "$validation_root"/{tmp,cache,mpl}
export TMPDIR="$validation_root/tmp" MPLCONFIGDIR="$validation_root/mpl"
export TI_OFFLINE_CACHE_FILE_PATH="$validation_root/cache" PYTHONDONTWRITEBYTECODE=1
export GEOTAICHI_REAL_DTYPE=float64 GEOTAICHI_TEST_ARCH=gpu PYTHONPATH="$PWD"
trap 'printf "%s\n" "$?" > "$validation_root/exit_code"' EXIT

"$case_python" -m pytest -q -p no:cacheprovider --taichi-arch=cpu \
    tests/unit/utils/test_time_ticker.py \
    tests/unit/test_compile_step_clock.py \
    tests/unit/mpm/test_twophase_single_layer_pressure_operator.py \
    tests/integration/mpm/test_twophase_single_layer_semiimplicit.py \
    tests/integration/mpm/test_twophase_single_layer_3d.py \
    tests/unit/linear_solver/test_matrix_free_krylov_contract.py \
    tests/unit/physics_model/materials/test_elastoplastic_models.py \
    --basetemp="$validation_root/pytest" > "$validation_root/regression.log" 2>&1
printf 'PASS\n' > "$validation_root/regression.status"

for formulation in up uvp; do
    mkdir -p "$validation_root/formal_$formulation"
    export GT_SATURATED_COLUMN_SAVE_PATH="$validation_root/formal_$formulation/OutputData"
    export GT_SATURATED_COLUMN_TIME=0.5 GT_SATURATED_COLUMN_SAVE_INTERVAL=0.005
    if [[ "$formulation" == up ]]; then
        export GT_SATURATED_COLUMN_SOLVER_TYPE=SemiImplicit_u_p
        export GT_SATURATED_COLUMN_PRESSURE_SOLVER=PCG GT_SATURATED_COLUMN_PRESSURE_BETA=1
    else
        export GT_SATURATED_COLUMN_SOLVER_TYPE=SemiImplicit
        export GT_SATURATED_COLUMN_PRESSURE_SOLVER=MGPCG GT_SATURATED_COLUMN_PRESSURE_BETA=0
    fi
    "$case_python" -u examples/mmpm/ColumnCollapse/SaturatedSoilColumnCollapseSemiImplicit2D.py \
        > "$validation_root/formal_$formulation/run.log" 2>&1
    printf '0\n' > "$validation_root/formal_$formulation/exit_code"
done
