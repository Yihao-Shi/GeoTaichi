"""Structural contracts that keep the maintained test suite reproducible."""

import ast
from pathlib import Path
import subprocess

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.required
COLLECTED_ROOTS = tuple(
    REPOSITORY_ROOT / "tests" / layer
    for layer in (
        "unit",
        "integration",
        "verification",
        "regression",
        "benchmarks",
    )
)
DEPRECATED_TEST_ROOTS = (
    "codeTesting",
    "pythonTesting",
    "demTesting",
    "debug",
    "sympy_generate",
)
IMPORT_TIME_CALLS = {
    "init",
    "ti.init",
    "geotaichi.init",
    "open",
    "os.mkdir",
    "os.makedirs",
    "shutil.rmtree",
    "np.save",
    "np.savez",
    "np.savetxt",
    "numpy.save",
    "numpy.savez",
    "numpy.savetxt",
    "ti.GUI",
    "ti.ui.Window",
}
IMPORT_TIME_OUTPUT_METHODS = {"write_text", "write_bytes"}
SYS_PATH_MUTATORS = {
    "sys.path.append",
    "sys.path.extend",
    "sys.path.insert",
}
MACHINE_PATH_PREFIXES = (
    "/home/",
    "/Users/",
    "/Volumes/",
    "C:\\",
    "D:\\",
    "E:\\",
)
MAINTAINED_SOURCE_ROOTS = ("tests", "examples", "tools", "images")
EDITOR_RECOVERY_SUFFIXES = (".swp", ".swo", ".swn")
EDITOR_RECOVERY_PREFIXES = (".goutputstream-",)
MAX_TAICHI_STATIC_EXPANSION = 9
TAICHI_STATIC_EXPANSION_ROOTS = (
    "src/linear_solver",
    "src/mpm/soft_particle/IPCMPM.py",
    "src/physics_model/contact_model/ipc",
    "src/iga",
    "src/dem/engines/AffineBodyOperator.py",
    "src/dem/engines/AffineDiffIPC.py",
    "src/mpdem/engines/SoftAffineIPCOperator.py",
    "src/physics_model/consititutive_model/finite_strain",
)


def _test_modules():
    for root in COLLECTED_ROOTS:
        yield from sorted(root.rglob("test_*.py"))


def _qualified_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


class _ImportTimeCallVisitor(ast.NodeVisitor):
    """Visit statements executed while importing a module.

    A class body executes immediately when Python creates the class, so its
    assignments and expressions belong to the import-time surface.  Function
    bodies and lambda bodies do not execute on definition, but decorators,
    defaults, annotations, bases, and nested class bodies do.
    """

    def __init__(self):
        self.calls = []

    def visit_FunctionDef(self, node):
        self._visit_function_definition_surface(node)
        return None

    def visit_AsyncFunctionDef(self, node):
        self._visit_function_definition_surface(node)
        return None

    def visit_ClassDef(self, node):
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        for statement in node.body:
            self.visit(statement)
        return None

    def visit_Lambda(self, node):
        self._visit_arguments_definition_surface(node.args)
        return None

    def _visit_function_definition_surface(self, node):
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._visit_arguments_definition_surface(node.args)
        if node.returns is not None:
            self.visit(node.returns)

    def _visit_arguments_definition_surface(self, arguments):
        positional = (*arguments.posonlyargs, *arguments.args)
        for argument in (*positional, *arguments.kwonlyargs):
            if argument.annotation is not None:
                self.visit(argument.annotation)
        if arguments.vararg is not None and arguments.vararg.annotation is not None:
            self.visit(arguments.vararg.annotation)
        if arguments.kwarg is not None and arguments.kwarg.annotation is not None:
            self.visit(arguments.kwarg.annotation)
        for default in arguments.defaults:
            self.visit(default)
        for default in arguments.kw_defaults:
            if default is not None:
                self.visit(default)

    def visit_If(self, node):
        if _is_main_guard(node.test):
            for statement in node.orelse:
                self.visit(statement)
            return None
        self.generic_visit(node)

    def visit_Call(self, node):
        method_name = node.func.attr if isinstance(node.func, ast.Attribute) else ""
        self.calls.append((_qualified_name(node.func), method_name, node.lineno))
        self.generic_visit(node)


def _is_main_guard(node):
    if (
        not isinstance(node, ast.Compare)
        or len(node.ops) != 1
        or not isinstance(node.ops[0], ast.Eq)
        or len(node.comparators) != 1
    ):
        return False
    operands = (node.left, node.comparators[0])
    return any(
        isinstance(name, ast.Name)
        and name.id == "__name__"
        and isinstance(value, ast.Constant)
        and value.value == "__main__"
        for name, value in (operands, operands[::-1])
    )


def _is_import_time_side_effect(call_name, method_name):
    is_taichi_field = call_name.startswith("ti.") and call_name.endswith(".field")
    is_output_write = method_name in IMPORT_TIME_OUTPUT_METHODS
    is_runtime_run = method_name == "run" and not call_name.startswith("pytest.")
    return call_name in IMPORT_TIME_CALLS or is_taichi_field or is_output_write or is_runtime_run


def _import_time_side_effects(source, filename="<ast-oracle>"):
    tree = ast.parse(source, filename=filename)
    visitor = _ImportTimeCallVisitor()
    visitor.visit(tree)
    return [
        (call_name, line_number)
        for call_name, method_name, line_number in visitor.calls
        if _is_import_time_side_effect(call_name, method_name)
    ]


def _shared_tempdir_calls(tree):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _qualified_name(node.func) in {"gettempdir", "tempfile.gettempdir"}
    ]


def _is_generated_artifact(relative):
    filename = relative.name
    return (
        filename == ".DS_Store"
        or filename.startswith("._")
        or filename.startswith(EDITOR_RECOVERY_PREFIXES)
        or filename.endswith((".pyc", ".pyo", *EDITOR_RECOVERY_SUFFIXES))
        or "__pycache__" in relative.parts
    )


def _static_integer(node):
    """Evaluate only source-level integer bounds with an unambiguous maximum.

    ``config.DIM`` is the one project template for which the maintained source
    contract provides a global upper bound (three).  Shape-dependent template
    members such as ``matrix.n`` deliberately remain unknown so this hygiene
    check cannot reject a valid small specialization.
    """

    if isinstance(node, ast.Constant):
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value
        return None
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "config"
        and node.attr == "DIM"
    ):
        return 3
    if isinstance(node, ast.UnaryOp):
        value = _static_integer(node.operand)
        if value is None:
            return None
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        return None
    if isinstance(node, ast.BinOp):
        left = _static_integer(node.left)
        right = _static_integer(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv) and right != 0:
            return left // right
    return None


def _range_iteration_count(call):
    if not isinstance(call, ast.Call) or _qualified_name(call.func) != "range":
        return None
    if call.keywords or not 1 <= len(call.args) <= 3:
        return None
    values = [_static_integer(argument) for argument in call.args]
    if any(value is None for value in values):
        return None
    try:
        return len(range(*values))
    except (TypeError, ValueError):
        return None


def _ndrange_axis_count(node):
    scalar = _static_integer(node)
    if scalar is not None:
        return max(scalar, 0)
    if not isinstance(node, (ast.Tuple, ast.List)):
        return None
    values = [_static_integer(element) for element in node.elts]
    if any(value is None for value in values) or not 1 <= len(values) <= 3:
        return None
    try:
        return len(range(*values))
    except (TypeError, ValueError):
        return None


def _static_iteration_count(node):
    """Return a known ``ti.static`` loop trip count, otherwise ``None``."""

    if (
        not isinstance(node, ast.Call)
        or _qualified_name(node.func) != "ti.static"
        or len(node.args) != 1
        or node.keywords
    ):
        return None
    iterator = node.args[0]
    if (
        isinstance(iterator, ast.Call)
        and _qualified_name(iterator.func) == "ti.grouped"
        and len(iterator.args) == 1
        and not iterator.keywords
    ):
        iterator = iterator.args[0]
    count = _range_iteration_count(iterator)
    if count is not None:
        return count
    if not isinstance(iterator, ast.Call) or _qualified_name(iterator.func) != "ti.ndrange" or iterator.keywords:
        return None
    count = 1
    for axis in iterator.args:
        axis_count = _ndrange_axis_count(axis)
        if axis_count is None:
            return None
        count *= axis_count
    return count


def _is_taichi_static_iterator(node):
    return isinstance(node, ast.Call) and _qualified_name(node.func) == "ti.static"


class _TaichiStaticExpansionVisitor(ast.NodeVisitor):
    """Find explicitly bounded nested ``ti.static`` expansion over nine."""

    def __init__(self):
        self._expansion = 1
        self.violations = []

    def _enter_static_iterator(self, iterator, line_number):
        count = _static_iteration_count(iterator)
        previous = self._expansion
        if count is None and _is_taichi_static_iterator(iterator):
            # An unknown template bound must not produce a false positive, and
            # it also terminates the explicitly known nesting chain.
            self._expansion = 1
        elif count is not None:
            self._expansion *= count
            if self._expansion > MAX_TAICHI_STATIC_EXPANSION:
                self.violations.append((line_number, self._expansion))
        return previous

    def visit_For(self, node):
        self.visit(node.iter)
        previous = self._enter_static_iterator(node.iter, node.lineno)
        for statement in node.body:
            self.visit(statement)
        self._expansion = previous
        for statement in node.orelse:
            self.visit(statement)
        return None

    def _visit_comprehension(self, node, value_nodes):
        previous = self._expansion
        for generator in node.generators:
            self.visit(generator.iter)
            self._enter_static_iterator(generator.iter, generator.target.lineno)
            for condition in generator.ifs:
                self.visit(condition)
        for value in value_nodes:
            self.visit(value)
        self._expansion = previous

    def visit_ListComp(self, node):
        self._visit_comprehension(node, (node.elt,))
        return None

    def visit_SetComp(self, node):
        self._visit_comprehension(node, (node.elt,))
        return None

    def visit_GeneratorExp(self, node):
        self._visit_comprehension(node, (node.elt,))
        return None

    def visit_DictComp(self, node):
        self._visit_comprehension(node, (node.key, node.value))
        return None


def _taichi_static_expansion_violations(source, filename="<ast-oracle>"):
    tree = ast.parse(source, filename=filename)
    visitor = _TaichiStaticExpansionVisitor()
    visitor.visit(tree)
    return visitor.violations


def _taichi_static_expansion_modules():
    for relative in TAICHI_STATIC_EXPANSION_ROOTS:
        path = REPOSITORY_ROOT / relative
        candidates = (path,) if path.is_file() else path.rglob("*.py")
        for candidate in sorted(candidates):
            if candidate.name.startswith("._"):
                continue
            yield candidate


def test_only_maintained_test_roots_remain():
    tests_root = REPOSITORY_ROOT / "tests"
    leftovers = [name for name in DEPRECATED_TEST_ROOTS if (tests_root / name).exists()]
    assert leftovers == []


def test_generated_files_are_never_version_controlled():
    """Reject repository artifacts without rejecting runtime-created caches.

    Python creates untracked ``__pycache__`` directories during a normal test
    run, and macOS may create ignored AppleDouble sidecars.  Inspecting the Git
    index keeps those ordinary runtime files from making the test self-failing
    while still preventing any such artifact from being committed.
    """

    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(REPOSITORY_ROOT),
                "ls-files",
                "-z",
                "--",
                *MAINTAINED_SOURCE_ROOTS,
            ],
            check=False,
            capture_output=True,
        )
    except FileNotFoundError:
        pytest.skip("git is unavailable; tracked-artifact hygiene needs a worktree")
    if completed.returncode != 0:
        pytest.skip("tracked-artifact hygiene needs a Git worktree")

    violations = []
    for raw_path in completed.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = Path(raw_path.decode("utf-8", errors="surrogateescape"))
        # Treat a tracked path deleted in the working tree as the prospective
        # repository state.  A clean checkout cannot have this mismatch, while
        # this keeps the hygiene test runnable before the deletion is committed.
        if not (REPOSITORY_ROOT / relative).exists():
            continue
        if _is_generated_artifact(relative):
            violations.append(relative.as_posix())
    assert violations == []


@pytest.mark.parametrize(
    "relative",
    (
        "examples/case/.scene.py.swp",
        "examples/case/.scene.py.swo",
        "examples/case/.scene.py.swn",
        "examples/case/.goutputstream-ABC123",
    ),
)
def test_generated_artifact_classifier_covers_editor_recovery_files(
    relative,
):
    assert _is_generated_artifact(Path(relative))


def test_every_test_module_exposes_a_pytest_node():
    empty_modules = []
    for path in _test_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        has_node = any(
            (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"))
            or (isinstance(node, ast.ClassDef) and node.name.startswith("Test"))
            for node in tree.body
        )
        if not has_node:
            empty_modules.append(path.relative_to(REPOSITORY_ROOT).as_posix())
    assert empty_modules == []


def test_test_modules_have_no_import_time_runtime_or_output_side_effects():
    violations = []
    for path in _test_modules():
        source = path.read_text(encoding="utf-8")
        for call_name, line_number in _import_time_side_effects(source, filename=str(path)):
            relative = path.relative_to(REPOSITORY_ROOT).as_posix()
            violations.append(f"{relative}:{line_number}: {call_name}")
    assert violations == []


def test_test_modules_use_isolated_temporary_output_paths():
    violations = []
    for path in _test_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in _shared_tempdir_calls(tree):
            relative = path.relative_to(REPOSITORY_ROOT).as_posix()
            violations.append(f"{relative}:{node.lineno}: tempfile.gettempdir")
    assert violations == []


def test_test_modules_do_not_mutate_sys_path():
    violations = []
    for path in _test_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _qualified_name(node.func) in SYS_PATH_MUTATORS:
                relative = path.relative_to(REPOSITORY_ROOT).as_posix()
                violations.append(f"{relative}:{node.lineno}: " f"{_qualified_name(node.func)}")
    assert violations == []


def test_import_time_ast_oracle_checks_all_executed_definition_surfaces():
    source = """
class ImportTimeBody:
    output = Path("artifact.txt")
    output.write_text("created too early")
    Path("artifact.bin").write_bytes(b"created too early")
    solver.run()

    def method(self):
        Path("method.txt").write_bytes(b"not import time")
        solver.run()

    class NestedDefinition:
        Path("nested.txt").write_text("also import time")
        solver.run()

@decorator_factory.run()
def module_function(value=default_factory.run()):
    solver.run()
"""
    assert _import_time_side_effects(source) == [
        ("output.write_text", 4),
        ("write_bytes", 5),
        ("solver.run", 6),
        ("write_text", 13),
        ("solver.run", 14),
        ("decorator_factory.run", 16),
        ("default_factory.run", 17),
    ]


def test_import_time_ast_oracle_reports_runtime_decorators_and_ignores_main_guard():
    source = """
@pytest.mark.parametrize("value", [1])
@runner.run()
def test_value(value):
    solver.run()

pytestmark = pytest.mark.usefixtures("runtime")

if __name__ == "__main__":
    solver.run()
"""
    assert _import_time_side_effects(source) == [("runner.run", 3)]


def test_shared_tempdir_ast_oracle_rejects_fixed_process_paths():
    source = """
def bad_output_path(case):
    first = os.path.join(tempfile.gettempdir(), "fixed-output")
    second = Path(tempfile.gettempdir()) / f"case-{case}"
    return first, second

def isolated_output_path(tmp_path):
    return tmp_path / "output"
"""
    tree = ast.parse(source)
    assert [node.lineno for node in _shared_tempdir_calls(tree)] == [3, 4]


def test_test_modules_do_not_embed_machine_specific_paths():
    violations = []
    for path in _test_modules():
        if path.resolve() == Path(__file__).resolve():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if node.value.startswith(MACHINE_PATH_PREFIXES):
                relative = path.relative_to(REPOSITORY_ROOT).as_posix()
                violations.append(f"{relative}:{node.lineno}: {node.value!r}")
    assert violations == []


def test_randomized_tests_use_explicit_local_seeds():
    global_random_calls = {
        "np.random.rand",
        "np.random.randn",
        "np.random.random",
        "np.random.uniform",
        "np.random.normal",
        "numpy.random.rand",
        "numpy.random.randn",
        "numpy.random.random",
        "numpy.random.uniform",
        "numpy.random.normal",
        "random.random",
        "random.uniform",
        "random.randrange",
        "random.randint",
    }
    violations = []
    for path in _test_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _qualified_name(node.func)
            unseeded_generator = (
                call_name
                in {
                    "np.random.default_rng",
                    "numpy.random.default_rng",
                }
                and not node.args
                and not node.keywords
            )
            if call_name in global_random_calls or unseeded_generator:
                relative = path.relative_to(REPOSITORY_ROOT).as_posix()
                violations.append(f"{relative}:{node.lineno}: {call_name}")
    assert violations == []


def test_xfail_is_not_used_to_mask_implementation_defects():
    violations = []
    for path in _test_modules():
        if path.resolve() == Path(__file__).resolve():
            continue
        source = path.read_text(encoding="utf-8")
        if "pytest.mark.xfail" in source or "pytest.xfail(" in source:
            violations.append(path.relative_to(REPOSITORY_ROOT).as_posix())
    assert violations == []


def test_implicit_and_contact_static_expansion_is_bounded_by_three_by_three():
    """Keep compile-time loop cloning at or below nine source instances.

    Runtime ``range``/``while`` loops still execute on the Taichi device.  This
    contract targets only explicit ``ti.static`` unrolling, whose nested trip
    counts multiply compiler IR size and previously made the implicit/contact
    kernels prohibitively slow to compile.
    """

    violations = []
    for path in _taichi_static_expansion_modules():
        source = path.read_text(encoding="utf-8")
        for line_number, expansion in _taichi_static_expansion_violations(source, filename=str(path)):
            relative = path.relative_to(REPOSITORY_ROOT).as_posix()
            violations.append(
                f"{relative}:{line_number}: explicit ti.static expansion "
                f"{expansion} > {MAX_TAICHI_STATIC_EXPANSION}"
            )
    assert violations == []


def test_taichi_static_expansion_ast_oracle_handles_nested_loops_and_comprehensions():
    source = """
def allowed(matrix):
    for row in ti.static(range(config.DIM)):
        for column in ti.static(range(config.DIM)):
            matrix[row, column] = 0.0
    for component in ti.static(range(vector.n)):
        matrix[component, component] = 1.0

def rejected(matrix):
    for axis in ti.static(range(config.DIM)):
        for row, column in ti.static(ti.ndrange(2, 2)):
            matrix[row, column] += axis
    values = [
        [row + column for column in ti.static(range(4))]
        for row in ti.static(range(3))
    ]
    for component in ti.static(range(10)):
        matrix[component, component] = values[0][0]
    for axis in ti.static(range(3)):
        for support in range(num_supports):
            for component in ti.static(range(4)):
                matrix[support, component] += axis
"""
    assert _taichi_static_expansion_violations(source) == [
        (11, 12),
        (14, 12),
        (17, 10),
        (21, 12),
    ]


def test_taichi_static_expansion_ast_oracle_ignores_compile_time_if_and_unknown_bounds():
    source = """
def bounded_by_template(matrix):
    if ti.static(matrix.n == 3):
        for row in ti.static(range(3)):
            for column in ti.static(range(3)):
                matrix[row, column] = 0.0
    for support in ti.static(range(shape_values.n)):
        for component in ti.static(range(4)):
            matrix[support, component] = 0.0
"""
    assert _taichi_static_expansion_violations(source) == []
