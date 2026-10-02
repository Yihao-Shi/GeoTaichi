#!/usr/bin/env bash
#
# Offline validation gate for the IPC/contact changes.
#
# The default mode runs only the checks that still need a final, uninterrupted
# pass: the complete IPC partition (including all 446 bundled friction-data
# cases), the required merge gate, and repository hygiene checks.
#
# Use --targeted to repeat the already-green SoftParticle/MPM, bounded-static-
# expansion, and assembly regressions.  Use --full instead to run the complete
# correctness suite once, without repeating the smaller partitions.

set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPOSITORY_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
readonly PARTITION_RUNNER="$REPOSITORY_ROOT/tests/testing/run_partition.py"
readonly EXPECTED_IPC_DATA_COMMIT="c7eba549d9a80d15569a013c473f0aff104ac44a"
readonly EXPECTED_IPC_DATA_CASES=446
readonly EXPECTED_IPC_ARCHIVE_SHA256="1b3ddd33327957d759564a3aeb3e44ea80e3e4cead49e12dc534761e80570d38"
readonly EXPECTED_IPC_CONTENTS_SHA256="72f21aeba625e7d2fb083d5edd726b20abfa25bd325479cbea3d0d1477e8c73b"

PYTHON_EXECUTABLE="${GEOTAICHI_PYTHON:-}"
IPC_DATA_ROOT="${IPC_TOOLKIT_TEST_DATA:-$REPOSITORY_ROOT/tests/data/ipc_toolkit}"
LOG_DIRECTORY=""
RUN_TARGETED=0
RUN_FULL=0
CLEAN_CACHE=0
EXTRA_PYTEST_ARGUMENTS=()

usage() {
    cat <<'USAGE'
Usage:
  tests/testing/run_offline_validation.sh [options] [-- pytest-args...]

Modes:
  (default)       Run bundled-data preflight, ipc, required, and hygiene.
  --targeted      Also repeat SoftParticle/MPM, static-loop, and assembly tests.
  --full          Replace ipc/required/targeted runs with the complete correctness
                  partition. This is intentionally not combined with --targeted.

Options:
  --python PATH   Python executable from the GeoTaichi environment.
                  Default: $GEOTAICHI_PYTHON, active Conda Python, or python.
  --ipc-data DIR  IPC reference-data package root.
                  Default: $IPC_TOOLKIT_TEST_DATA or
                  tests/data/ipc_toolkit
  --log-dir DIR   Output directory for per-step logs and summary.tsv.
                  Default: a timestamped directory below $TMPDIR (or /tmp).
  --clean-cache   After successful tests, remove generated Python/pytest caches
                  only below maintained source roots.
  -h, --help      Show this help.

Everything following -- is forwarded to every pytest invocation. The script
never downloads data or dependencies and excludes tests marked requires_network.

Examples:
  tests/testing/run_offline_validation.sh
  tests/testing/run_offline_validation.sh --targeted -- -vv
  tests/testing/run_offline_validation.sh --full \
      --python /path/to/geotaichi/bin/python
USAGE
}

while (($#)); do
    case "$1" in
        --python)
            [[ $# -ge 2 ]] || {
                printf 'error: --python requires a path\n' >&2
                exit 2
            }
            PYTHON_EXECUTABLE="$2"
            shift 2
            ;;
        --ipc-data)
            [[ $# -ge 2 ]] || {
                printf 'error: --ipc-data requires a directory\n' >&2
                exit 2
            }
            IPC_DATA_ROOT="$2"
            shift 2
            ;;
        --log-dir)
            [[ $# -ge 2 ]] || {
                printf 'error: --log-dir requires a directory\n' >&2
                exit 2
            }
            LOG_DIRECTORY="$2"
            shift 2
            ;;
        --targeted)
            RUN_TARGETED=1
            shift
            ;;
        --full)
            RUN_FULL=1
            shift
            ;;
        --clean-cache)
            CLEAN_CACHE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_PYTEST_ARGUMENTS=("$@")
            break
            ;;
        *)
            printf 'error: unknown option: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ((RUN_FULL && RUN_TARGETED)); then
    printf 'error: --full and --targeted are mutually exclusive\n' >&2
    exit 2
fi

if [[ -z "$PYTHON_EXECUTABLE" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
        PYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_EXECUTABLE="$(command -v python)"
    else
        printf 'error: no Python found; pass --python PATH\n' >&2
        exit 2
    fi
fi
if [[ "$PYTHON_EXECUTABLE" != */* ]]; then
    RESOLVED_PYTHON="$(command -v "$PYTHON_EXECUTABLE" 2>/dev/null || true)"
    if [[ -z "$RESOLVED_PYTHON" ]]; then
        printf 'error: Python command was not found: %s\n' \
            "$PYTHON_EXECUTABLE" >&2
        exit 2
    fi
    PYTHON_EXECUTABLE="$RESOLVED_PYTHON"
fi

if [[ -z "$LOG_DIRECTORY" ]]; then
    readonly LOG_BASE="${TMPDIR:-/tmp}"
    LOG_DIRECTORY="$LOG_BASE/geotaichi-offline-validation-$(date '+%Y%m%d-%H%M%S')"
fi

mkdir -p "$LOG_DIRECTORY"
LOG_DIRECTORY="$(cd "$LOG_DIRECTORY" && pwd -P)"

# These variables make common optional clients deterministic and offline.  The
# selected pytest expressions additionally exclude every requires_network test.
export IPC_TOOLKIT_TEST_DATA="$IPC_DATA_ROOT"
export PIP_NO_INDEX=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

PLAN_KEYS=()
PLAN_DESCRIPTIONS=()
PLAN_RESULTS=()
PLAN_DURATIONS=()
FINAL_SUMMARY_WRITTEN=0

add_plan() {
    local index="${#PLAN_KEYS[@]}"
    PLAN_KEYS[$index]="$1"
    PLAN_DESCRIPTIONS[$index]="$2"
    PLAN_RESULTS[$index]="PENDING"
    PLAN_DURATIONS[$index]="-"
}

add_plan "preflight" "Python environment and bundled IPC data integrity"
if ((RUN_FULL)); then
    add_plan "all" "complete offline correctness suite"
else
    if ((RUN_TARGETED)); then
        add_plan "softparticle-mpm" "SoftParticle and MPM focused regressions"
        add_plan "bounded-static" "bounded static expansion and runtime-loop equivalence"
        add_plan "assembly" "all matrix-free, hash-triplet, and COO assembly paths"
    fi
    add_plan "ipc" "complete IPC partition, including the 446-case bundled fixture"
    add_plan "required" "minimal merge-gate contracts"
fi
if ((CLEAN_CACHE)); then
    add_plan "clean-cache" "remove generated caches from explicit maintained roots"
fi
add_plan "hygiene" "diff whitespace check and cache inventory"

write_summary() {
    local original_status=$?
    local index
    local result
    local overall_status="$original_status"

    if ((FINAL_SUMMARY_WRITTEN)); then
        return 0
    fi
    FINAL_SUMMARY_WRITTEN=1
    set +e

    {
        printf 'step\tresult\tduration_seconds\tdescription\n'
        for ((index = 0; index < ${#PLAN_KEYS[@]}; ++index)); do
            printf '%s\t%s\t%s\t%s\n' \
                "${PLAN_KEYS[$index]}" \
                "${PLAN_RESULTS[$index]}" \
                "${PLAN_DURATIONS[$index]}" \
                "${PLAN_DESCRIPTIONS[$index]}"
        done
    } >"$LOG_DIRECTORY/summary.tsv"

    printf '\nOffline validation summary\n'
    printf '  logs: %s\n' "$LOG_DIRECTORY"
    for ((index = 0; index < ${#PLAN_KEYS[@]}; ++index)); do
        result="${PLAN_RESULTS[$index]}"
        printf '  %-20s %-8s %ss\n' \
            "${PLAN_KEYS[$index]}" "$result" "${PLAN_DURATIONS[$index]}"
        if [[ "$result" != "PASS" ]]; then
            overall_status=1
        fi
    done
    if ((overall_status == 0)); then
        printf '  overall: PASS\n'
    else
        printf '  overall: FAIL (exit %d)\n' "$overall_status"
    fi
    return 0
}
trap write_summary EXIT

run_step() {
    local index="$1"
    shift
    local key="${PLAN_KEYS[$index]}"
    local log_file="$LOG_DIRECTORY/$(printf '%02d' "$((index + 1))")-$key.log"
    local started_at=$SECONDS
    local command_status

    PLAN_RESULTS[$index]="RUNNING"
    printf '\n[%s] START %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$key"
    printf '  %s\n' "${PLAN_DESCRIPTIONS[$index]}"
    printf '  log: %s\n' "$log_file"

    set +e
    (
        # ``run_step`` temporarily disables errexit in the parent only so it
        # can capture and summarize the failing status.  Re-enable the strict
        # shell contract here; otherwise an intermediate failure inside a
        # function (for example the Python import check or ``git diff
        # --check``) could be hidden by a later successful ``printf``.
        set -Eeuo pipefail
        printf '+'
        printf ' %q' "$@"
        printf '\n'
        "$@"
    ) 2>&1 | tee "$log_file"
    command_status=${PIPESTATUS[0]}
    set -e

    PLAN_DURATIONS[$index]="$((SECONDS - started_at))"
    if ((command_status == 0)); then
        PLAN_RESULTS[$index]="PASS"
        printf '[%s] PASS %s (%ss)\n' \
            "$(date '+%Y-%m-%d %H:%M:%S')" \
            "$key" "${PLAN_DURATIONS[$index]}"
        return 0
    fi

    PLAN_RESULTS[$index]="FAIL"
    printf '[%s] FAIL %s (exit %d, %ss)\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" \
        "$key" "$command_status" "${PLAN_DURATIONS[$index]}" >&2
    return "$command_status"
}

validate_preflight() {
    local data_report

    [[ -x "$PYTHON_EXECUTABLE" ]] || {
        printf 'Python is not executable: %s\n' "$PYTHON_EXECUTABLE" >&2
        return 2
    }
    [[ -f "$PARTITION_RUNNER" ]] || {
        printf 'partition runner is missing: %s\n' "$PARTITION_RUNNER" >&2
        return 2
    }
    "$PYTHON_EXECUTABLE" -c \
        'import numpy, pytest, taichi; print("python/pytest/numpy/taichi imports: OK")'

    data_report="$(
        "$PYTHON_EXECUTABLE" - \
            "$IPC_DATA_ROOT" \
            "$EXPECTED_IPC_DATA_COMMIT" \
            "$EXPECTED_IPC_DATA_CASES" \
            "$EXPECTED_IPC_ARCHIVE_SHA256" \
            "$EXPECTED_IPC_CONTENTS_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile

root = Path(sys.argv[1]).expanduser().resolve()
expected_commit = sys.argv[2]
expected_count = int(sys.argv[3])
expected_archive_digest = sys.argv[4]
expected_contents_digest = sys.argv[5]
metadata_path = root / "SOURCE.json"
license_path = root / "LICENSE"

if not metadata_path.is_file():
    raise SystemExit(f"bundled IPC provenance is missing: {metadata_path}")
if not license_path.is_file():
    raise SystemExit(f"bundled IPC license is missing: {license_path}")
with metadata_path.open(encoding="utf-8") as stream:
    metadata = json.load(stream)
if metadata.get("data_commit") != expected_commit:
    raise SystemExit(
        "unexpected IPC data commit: "
        f"{metadata.get('data_commit')} (expected {expected_commit})"
    )
if int(metadata.get("case_count", -1)) != expected_count:
    raise SystemExit(
        "unexpected IPC case count in SOURCE.json: "
        f"{metadata.get('case_count')} (expected {expected_count})"
    )

archive_path = root / metadata["archive"]
if not archive_path.is_file():
    raise SystemExit(f"bundled IPC archive is missing: {archive_path}")
archive_digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
if archive_digest != expected_archive_digest:
    raise SystemExit(
        "bundled IPC archive checksum mismatch: "
        f"{archive_digest} != {expected_archive_digest}"
    )

case_name = re.compile(r"friction_data_([0-9]+)\.json")
payloads = {}
with tarfile.open(archive_path, mode="r:gz") as archive:
    for member in archive:
        if not member.isfile():
            continue
        name = PurePosixPath(member.name).name
        match = case_name.fullmatch(name)
        if match is None:
            raise SystemExit(f"unexpected IPC archive member: {member.name}")
        case_id = int(match.group(1))
        if case_id in payloads:
            raise SystemExit(f"duplicate IPC friction case: {case_id}")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise SystemExit(f"cannot read IPC archive member: {member.name}")
        payloads[case_id] = (name, extracted.read())

expected_ids = set(range(expected_count))
actual_ids = set(payloads)
if actual_ids != expected_ids:
    raise SystemExit(
        "bundled IPC case IDs are not continuous: "
        f"missing={sorted(expected_ids - actual_ids)}, "
        f"extra={sorted(actual_ids - expected_ids)}"
    )

contents_digest = hashlib.sha256()
for case_id in range(expected_count):
    name, payload = payloads[case_id]
    contents_digest.update(name.encode("utf-8"))
    contents_digest.update(b"\0")
    contents_digest.update(payload)
    contents_digest.update(b"\0")
if contents_digest.hexdigest() != expected_contents_digest:
    raise SystemExit(
        "bundled IPC payload checksum mismatch: "
        f"{contents_digest.hexdigest()} != {expected_contents_digest}"
    )

print(f"bundled IPC data package: {root}")
print(f"bundled IPC data commit: {expected_commit}")
print(f"bundled IPC archive: {archive_path}")
print(f"bundled IPC friction cases: {expected_count} (continuous 0..445)")
print(f"bundled IPC archive SHA-256: {archive_digest}")
print(f"bundled IPC contents SHA-256: {contents_digest.hexdigest()}")
PY
    )"

    printf 'repository: %s\n' "$REPOSITORY_ROOT"
    printf 'python: %s\n' "$PYTHON_EXECUTABLE"
    printf '%s\n' "$data_report"
    printf 'network access: disabled by test selection; no download step exists\n'
}

clean_generated_caches() {
    local relative_root
    local absolute_root
    local maintained_roots=(
        "src"
        "tests"
        "examples"
        "tools"
        "images"
        "geotaichi"
    )

    for relative_root in "${maintained_roots[@]}"; do
        absolute_root="$REPOSITORY_ROOT/$relative_root"
        [[ -d "$absolute_root" ]] || continue
        find "$absolute_root" -type d -name '__pycache__' -prune \
            -exec rm -rf '{}' '+'
        find "$absolute_root" -type f \
            \( -name '*.pyc' -o -name '*.pyo' -o -name '._*' \) -delete
    done
    if [[ -d "$REPOSITORY_ROOT/.pytest_cache" ]]; then
        rm -rf "$REPOSITORY_ROOT/.pytest_cache"
    fi
}

check_repository_hygiene() {
    local cache_directories=0
    local bytecode_files=0
    local appledouble_files=0
    local relative_root
    local absolute_root
    local maintained_roots=(
        "src"
        "tests"
        "examples"
        "tools"
        "images"
        "geotaichi"
    )

    git -C "$REPOSITORY_ROOT" diff --check
    for relative_root in "${maintained_roots[@]}"; do
        absolute_root="$REPOSITORY_ROOT/$relative_root"
        [[ -d "$absolute_root" ]] || continue
        cache_directories="$(
            (
                printf '%s\n' "$cache_directories"
                find "$absolute_root" -type d -name '__pycache__' -print |
                    wc -l | tr -d '[:space:]'
            ) | awk '{sum += $1} END {print sum + 0}'
        )"
        bytecode_files="$(
            (
                printf '%s\n' "$bytecode_files"
                find "$absolute_root" -type f \
                    \( -name '*.pyc' -o -name '*.pyo' \) -print |
                    wc -l | tr -d '[:space:]'
            ) | awk '{sum += $1} END {print sum + 0}'
        )"
        appledouble_files="$(
            (
                printf '%s\n' "$appledouble_files"
                find "$absolute_root" -type f -name '._*' -print |
                    wc -l | tr -d '[:space:]'
            ) | awk '{sum += $1} END {print sum + 0}'
        )"
    done

    printf 'git diff --check: OK\n'
    printf 'generated cache inventory under maintained roots:\n'
    printf '  __pycache__ directories: %s\n' "$cache_directories"
    printf '  .pyc/.pyo files: %s\n' "$bytecode_files"
    printf '  AppleDouble files: %s\n' "$appledouble_files"
    if [[ -d "$REPOSITORY_ROOT/.pytest_cache" ]]; then
        printf '  .pytest_cache: present\n'
    else
        printf '  .pytest_cache: absent\n'
    fi
    if ((CLEAN_CACHE == 0)); then
        printf 'cache cleanup was not requested; use --clean-cache to remove it\n'
    fi
}

cd "$REPOSITORY_ROOT"
step_index=0
run_step "$step_index" validate_preflight
step_index=$((step_index + 1))

if ((RUN_FULL)); then
    run_step "$step_index" \
        "$PYTHON_EXECUTABLE" "$PARTITION_RUNNER" all -q \
        -m "not requires_network" \
        ${EXTRA_PYTEST_ARGUMENTS[@]+"${EXTRA_PYTEST_ARGUMENTS[@]}"}
    step_index=$((step_index + 1))
else
    if ((RUN_TARGETED)); then
        run_step "$step_index" \
            "$PYTHON_EXECUTABLE" -m pytest \
            tests/unit/mpm \
            -q -m "not requires_network" \
            ${EXTRA_PYTEST_ARGUMENTS[@]+"${EXTRA_PYTEST_ARGUMENTS[@]}"}
        step_index=$((step_index + 1))

        run_step "$step_index" \
            "$PYTHON_EXECUTABLE" -m pytest \
            tests/unit/testing/test_suite_hygiene.py::test_implicit_and_contact_static_expansion_is_bounded_by_three_by_three \
            tests/unit/testing/test_suite_hygiene.py::test_taichi_static_expansion_ast_oracle_handles_nested_loops_and_comprehensions \
            tests/unit/testing/test_suite_hygiene.py::test_taichi_static_expansion_ast_oracle_ignores_compile_time_if_and_unknown_bounds \
            tests/unit/linear_solver/test_runtime_dense_block_pipeline.py \
            tests/unit/physics_model/materials/test_finite_strain_runtime_kernels.py \
            tests/integration/mpm/test_direct_mpm_compile_hotspots.py \
            -q -m "not requires_network" \
            ${EXTRA_PYTEST_ARGUMENTS[@]+"${EXTRA_PYTEST_ARGUMENTS[@]}"}
        step_index=$((step_index + 1))

        run_step "$step_index" \
            "$PYTHON_EXECUTABLE" "$PARTITION_RUNNER" assembly -q \
            -m "not requires_network" \
            ${EXTRA_PYTEST_ARGUMENTS[@]+"${EXTRA_PYTEST_ARGUMENTS[@]}"}
        step_index=$((step_index + 1))
    fi

    run_step "$step_index" \
        "$PYTHON_EXECUTABLE" "$PARTITION_RUNNER" ipc -q \
        -m "not requires_network" \
        ${EXTRA_PYTEST_ARGUMENTS[@]+"${EXTRA_PYTEST_ARGUMENTS[@]}"}
    step_index=$((step_index + 1))

    run_step "$step_index" \
        "$PYTHON_EXECUTABLE" "$PARTITION_RUNNER" required -q \
        -m "not requires_network" \
        ${EXTRA_PYTEST_ARGUMENTS[@]+"${EXTRA_PYTEST_ARGUMENTS[@]}"}
    step_index=$((step_index + 1))
fi

if ((CLEAN_CACHE)); then
    run_step "$step_index" clean_generated_caches
    step_index=$((step_index + 1))
fi

run_step "$step_index" check_repository_hygiene
