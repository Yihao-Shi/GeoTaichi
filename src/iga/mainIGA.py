from src.iga.config import set_dimension
from src.iga.GenerateManager import IGAGenerateManager
from src.iga.MaterialManager import IGAMaterialManager
from src.iga.SceneManager import IGAScene
from src.iga.Simulation import IGASimulation
from src.utils.SolverDiagnostics import SolverDiagnosticsMixin
from src.utils.SolverConsole import (
    print_material_info,
    print_simulation_start,
    print_solver_section,
    runtime_architecture,
)
from src.utils.StepRetry import StepRetryPolicy


class IGA(SolverDiagnosticsMixin):
    def __init__(self, title="IsoGeometric Analysis Engine", log=True):
        self.title = title
        self.log = log
        self.sims = IGASimulation()
        self.scene = IGAScene()
        self.generator = IGAGenerateManager()
        self.material_manager = IGAMaterialManager()
        self.solver_type = "Implicit"
        self.dimension = None
        self.primitives = None
        self.rest_shape = None
        self.dirichlet = None
        self.neumann = None
        self.material_kwargs = {}
        self.solver_kwargs = {}
        self.element_kwargs = {}
        self.engine = None
        if self.log:
            print("# =================================================================== #")
            print("#", "GeoTaichi -- IsoGeometric Analysis Engine".center(67), "#")
            print("#", str(title).center(67), "#")
            print("# =================================================================== #", "\n")

    def set_configuration(self, dimension=3, solver_type="Implicit", log=True, **kwargs):
        set_dimension(dimension)
        self.dimension = int(dimension)
        self.solver_type = str(solver_type)
        if self.solver_type not in ("Explicit", "Implicit"):
            raise ValueError("IGA solver_type must be 'Explicit' or 'Implicit'")
        self.sims.set_configuration(dimension=dimension, solver_type=solver_type, **kwargs)
        kwargs.setdefault("axisymmetric", self.sims.is_axisymmetric)
        kwargs.setdefault("axis_offset", self.sims.axis_offset)
        self.solver_kwargs.update(kwargs)
        if log:
            self.print_basic_simulation_info()
            print()

    def add_primitives(self, primitives, rest_shape=None):
        self.primitives = primitives
        self.rest_shape = rest_shape
        self.scene.add_primitives(primitives, rest_shape=rest_shape)

    def add_boundary_condition(self, dirichlet=None, neumann=None):
        self.dirichlet = dirichlet
        self.neumann = neumann
        self.scene.add_boundary_condition(dirichlet, neumann)

    def add_material(self, **kwargs):
        self.material_manager.add_material(**kwargs)
        self.scene.add_material(**kwargs)
        self.material_kwargs.update(kwargs)
        print_material_info(
            "Neo-Hookean",
            0,
            [
                ("Density", kwargs.get("density", kwargs.get("Density", 2650))),
                (
                    "Young Modulus",
                    kwargs.get("young_modulus", kwargs.get("YoungModulus")),
                ),
                (
                    "Poisson Ratio",
                    kwargs.get("poisson_ratio", kwargs.get("PoissonRatio", 0.3)),
                ),
            ],
            solver_name="IGA",
        )

    def add_element(self, degree):
        self.scene.add_element(degree)
        self.element_kwargs["degree"] = degree

    def set_solver(self, log=True, **kwargs):
        retry_policy = StepRetryPolicy(
            enabled=kwargs.get("enable_step_retry", False),
            maximum_retries=kwargs.get("step_retry_max_retries", 2),
            reduction=kwargs.get("step_retry_reduction", 0.5),
            minimum_timestep=kwargs.get("step_retry_minimum_timestep", 0.0),
        )
        if retry_policy.enabled and self.solver_type != "Implicit":
            raise ValueError("IGA step retry is available only for solver_type='Implicit'")
        kwargs.update(
            enable_step_retry=retry_policy.enabled,
            step_retry_max_retries=retry_policy.maximum_retries,
            step_retry_reduction=retry_policy.reduction,
            step_retry_minimum_timestep=retry_policy.minimum_timestep,
        )
        self.solver_kwargs.update(kwargs)
        if log:
            self.print_solver_info()
            print()

    def print_basic_simulation_info(self):
        print_solver_section(
            "IGA",
            "Basic Configuration",
            [
                ("Simulation Type", runtime_architecture()),
                ("Dimension", self.sims.dimension),
                ("Solver Type", self.solver_type),
                ("Axisymmetric", self.sims.is_axisymmetric),
                ("Axis Offset", self.sims.axis_offset),
            ],
        )

    def print_solver_info(self):
        print_solver_section(
            "IGA",
            "Solver Information",
            [
                ("Time Step", self.solver_kwargs.get("dt", 1.0e-2)),
                ("Requested Steps", self.solver_kwargs.get("step", 100)),
                ("Output Interval (steps)", self.solver_kwargs.get("interval", 1)),
                ("Save Path", self.solver_kwargs.get("path", "IGAData_case")),
                (
                    "Assembly Type",
                    self.solver_kwargs.get("assemble_type", self.solver_kwargs.get("assembly", "Hash")),
                ),
                (
                    "Linear Solver",
                    self.solver_kwargs.get(
                        "linear_solver",
                        "PCG" if self.solver_type == "Implicit" else None,
                    ),
                ),
            ],
        )

    def print_memory_info(self):
        engine = self.engine
        patch = None if engine is None else engine.patch
        primitive = None if patch is None else patch.primitive
        total_elements = None
        if patch is not None:
            total_elements = int(patch.total_num_element[-1])
        print_solver_section(
            "IGA",
            "Memory Information",
            [
                ("Control Points", None if primitive is None else primitive.num_ctrlpts),
                ("Elements", total_elements),
                ("Degrees of Freedom", None if engine is None else engine.degree_of_freedom),
                ("Matrix Nonzeros", None if engine is None else getattr(engine, "total_nnz", None)),
            ],
        )

    def print_neighbor_search_info(self):
        return

    def build(self):
        if self.primitives is None:
            raise RuntimeError("IGA primitives have not been set")

        kwargs = {}
        kwargs.update(self.material_kwargs)
        kwargs.update(self.solver_kwargs)
        kwargs.update(self.element_kwargs)
        if self.rest_shape is not None and "rest_shape" not in kwargs:
            kwargs["rest_shape"] = self.rest_shape

        if self.solver_type == "Implicit":
            from src.iga.engines import ImplicitIGA

            self.engine = ImplicitIGA(
                primitives=self.primitives,
                dirichlet=self.dirichlet,
                neumann=self.neumann,
                **kwargs,
            )
        else:
            from src.iga.engines import ExplicitIGA

            self.engine = ExplicitIGA(
                primitives=self.primitives,
                dirichlet=self.dirichlet,
                neumann=self.neumann,
                **kwargs,
            )
        self.engine.timer = self.sims.timer
        if self.log:
            self.print_memory_info()
            self.print_neighbor_search_info()
            print()
        return self.engine

    def run(self, *args, **kwargs):
        if self.engine is None:
            self.build()
        print_simulation_start("IGA")
        if kwargs.pop("visualize", False):
            from src.visualization.solver_adapters import run_iga_gui

            return run_iga_gui(
                self.engine,
                resolution=kwargs.pop("visualize_resolution", 16),
                verbose=kwargs.pop("verbose", True),
                postprocessing=kwargs.pop("postprocessing", ()),
            )
        if not hasattr(self.engine, "run"):
            raise NotImplementedError(f"{type(self.engine).__name__} does not provide run()")
        return self.engine.run(*args, **kwargs)
