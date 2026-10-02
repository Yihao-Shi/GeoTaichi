"""Public FEM facade following the MPM/IGA construction workflow."""

from __future__ import annotations

import numpy as np

from src.fem.MaterialManager import FEMMaterialManager
from src.fem.SceneManager import FEMScene
from src.fem.Simulation import FEMSimulation
from src.fem.boundaries import DirichletBoundary, NeumannBoundary
from src.fem.contact import FEMContact
from src.fem.generator import FEMGenerateManager, FEMMesh
from src.fem.soft_particle import FEMSoftParticleContactModel
from src.utils.SolverDiagnostics import SolverDiagnosticsMixin
from src.utils.SolverConsole import (
    print_material_info,
    print_simulation_start,
    print_solver_section,
    runtime_architecture,
)
from src.utils.StepRetry import StepRetryPolicy


def _key(mapping, *names, default=None):
    normalized = {
        str(name).replace("_", "").replace("-", "").replace(" ", "").lower(): value for name, value in mapping.items()
    }
    for name in names:
        candidate = str(name).replace("_", "").replace("-", "").replace(" ", "").lower()
        if candidate in normalized:
            return normalized[candidate]
    return default


def _canonical_parameters(parameters):
    aliases = {
        "size": "size",
        "divisions": "divisions",
        "elementtype": "element_type",
        "celltype": "cell_type",
        "origin": "origin",
        "center": "center",
        "plane": "plane",
        "name": "name",
        "radius": "radius",
        "height": "height",
        "length": "length",
        "width": "width",
        "nx": "nx",
        "ny": "ny",
        "nz": "nz",
        "radialdivisions": "radial_divisions",
        "circumferentialdivisions": "circumferential_divisions",
        "angulardivisions": "circumferential_divisions",
        "heightdivisions": "height_divisions",
        "file": "file",
        "filename": "filename",
        "path": "path",
        "restshape": "rest_shape",
    }
    canonical = {}
    for name, value in parameters.items():
        normalized = str(name).replace("_", "").replace("-", "").replace(" ", "").lower()
        canonical[aliases.get(normalized, name)] = value
    return canonical


def _taichi_is_initialized():
    try:
        import taichi as ti

        return ti.lang.impl.get_runtime().prog is not None
    except (ImportError, AttributeError):
        return False


class FEM(SolverDiagnosticsMixin):
    def __init__(self, title="Finite Element Analysis Engine", log=True):
        self.title = title
        self.log = bool(log)
        if self.log:
            print("# =================================================================== #")
            print("#", "GeoTaichi -- Finite Element Method Engine".center(67), "#")
            print("#", str(title).center(67), "#")
            print("# =================================================================== #", "\n")
        self.sims = FEMSimulation()
        self.scene = FEMScene()
        self.generator = FEMGenerateManager()
        self.material_manager = FEMMaterialManager()
        self.solver_kwargs = {}
        self.engine = None
        self.enginer = None

    def set_configuration(self, dimension=3, solver_type="Explicit", log=True, **kwargs):
        self.sims.set_configuration(dimension, solver_type, **kwargs)
        self.solver_kwargs.update(kwargs)
        if log:
            self.print_basic_simulation_info()
            print()

    def create_mesh(self, geometry=None, **kwargs):
        kwargs = _canonical_parameters(kwargs)
        rest_shape = kwargs.pop("rest_shape", None)

        def apply_rest_shape(generated):
            if rest_shape is not None:
                generated.set_rest_shape(
                    rest_shape,
                    update_material_coordinates=generated.is_membrane,
                )
            return generated

        if isinstance(geometry, FEMMesh):
            return apply_rest_shape(geometry)
        if geometry is None:
            geometry = kwargs.pop("type", kwargs.pop("primitive", None))
        if geometry is None:
            path = kwargs.pop("file", kwargs.pop("filename", kwargs.pop("path", None)))
            if path is None:
                raise ValueError("create_mesh requires a primitive geometry or input file")
            return apply_rest_shape(self.generator.read(path, **kwargs))
        normalized = str(geometry).strip().replace("_", "").replace("-", "").lower()
        if normalized in ("box", "cube", "cuboid", "rectangle3d"):
            if "cell_type" in kwargs and "element_type" not in kwargs:
                kwargs["element_type"] = kwargs.pop("cell_type")
            if "size" not in kwargs and all(key in kwargs for key in ("length", "width", "height")):
                kwargs["size"] = [kwargs.pop("length"), kwargs.pop("width"), kwargs.pop("height")]
            if "divisions" not in kwargs and all(key in kwargs for key in ("nx", "ny", "nz")):
                kwargs["divisions"] = [kwargs.pop("nx"), kwargs.pop("ny"), kwargs.pop("nz")]
            return apply_rest_shape(self.generator.create_box(**kwargs))
        if normalized in ("rectangle", "rect", "plane"):
            requested = kwargs.pop("element_type", kwargs.pop("cell_type", "TRI3"))
            from src.fem.generator import normalize_cell_type

            if normalize_cell_type(requested) != "triangle":
                raise ValueError("rectangle membrane generation requires element_type='TRI3'")
            if "size" not in kwargs and all(key in kwargs for key in ("length", "width")):
                kwargs["size"] = [kwargs.pop("length"), kwargs.pop("width")]
            if "divisions" not in kwargs and all(key in kwargs for key in ("nx", "ny")):
                kwargs["divisions"] = [kwargs.pop("nx"), kwargs.pop("ny")]
            return apply_rest_shape(self.generator.create_rectangle(**kwargs))
        if normalized in ("circle", "disk", "disc"):
            requested = kwargs.pop("element_type", kwargs.pop("cell_type", "TRI3"))
            from src.fem.generator import normalize_cell_type

            if normalize_cell_type(requested) != "triangle":
                raise ValueError("circle membrane generation requires element_type='TRI3'")
            return apply_rest_shape(self.generator.create_circle(**kwargs))
        if normalized in ("cylinder", "cylindrical"):
            requested = kwargs.pop("element_type", kwargs.pop("cell_type", "TET4"))
            from src.fem.generator import normalize_cell_type

            if normalize_cell_type(requested) != "tetra":
                raise ValueError("cylinder volume generation currently requires element_type='TET4'")
            return apply_rest_shape(self.generator.create_cylinder(**kwargs))
        if normalized in ("file", "mesh", "obj", "gmsh"):
            path = kwargs.pop("file", kwargs.pop("filename", kwargs.pop("path", None)))
            if path is None:
                raise ValueError("file mesh creation requires file/filename/path")
            return apply_rest_shape(self.generator.read(path, **kwargs))
        raise ValueError(f"Unknown FEM primitive geometry {geometry!r}")

    def add_mesh(self, mesh=None, **kwargs):
        kwargs = _canonical_parameters(kwargs)
        rest_shape = kwargs.pop("rest_shape", None)
        if isinstance(mesh, FEMMesh):
            generated = mesh
        elif isinstance(mesh, str):
            parameters = dict(kwargs)
            if "element_type" in parameters and "cell_type" not in parameters:
                parameters["cell_type"] = parameters.pop("element_type")
            generated = self.generator.read(mesh, **parameters)
        elif isinstance(mesh, dict):
            geometry = _key(mesh, "geometry", "type", "primitive")
            filename = _key(mesh, "file", "filename", "path")
            parameters = dict(mesh)
            for candidate in list(parameters):
                normalized = str(candidate).replace("_", "").replace("-", "").lower()
                if normalized in ("geometry", "type", "primitive", "file", "filename", "path"):
                    parameters.pop(candidate)
            parameters.update(kwargs)
            parameters = _canonical_parameters(parameters)
            rest_shape = parameters.pop("rest_shape", rest_shape)
            if filename is not None:
                if "element_type" in parameters and "cell_type" not in parameters:
                    parameters["cell_type"] = parameters.pop("element_type")
                generated = self.generator.read(filename, **parameters)
            else:
                generated = self.create_mesh(geometry, **parameters)
        elif mesh is None:
            generated = self.create_mesh(**kwargs)
        else:
            raise TypeError("FEM.add_mesh expects FEMMesh, filename, or a mesh dictionary")
        if rest_shape is not None:
            generated.set_rest_shape(
                rest_shape,
                update_material_coordinates=generated.is_membrane,
            )
        self.scene.add_mesh(generated)
        self.engine = None
        self.enginer = None
        return generated

    add_body = add_mesh

    def add_soft_particle(self, mesh=None, **kwargs):
        """Append one compatible volume mesh as an independent soft particle."""
        kwargs = _canonical_parameters(kwargs)
        rest_shape = kwargs.pop("rest_shape", None)
        if isinstance(mesh, FEMMesh):
            generated = mesh
        elif isinstance(mesh, dict):
            geometry = _key(mesh, "geometry", "type", "primitive")
            filename = _key(mesh, "file", "filename", "path")
            parameters = dict(mesh)
            for candidate in list(parameters):
                normalized = str(candidate).replace("_", "").replace("-", "").lower()
                if normalized in (
                    "geometry",
                    "type",
                    "primitive",
                    "file",
                    "filename",
                    "path",
                ):
                    parameters.pop(candidate)
            parameters.update(kwargs)
            parameters = _canonical_parameters(parameters)
            rest_shape = parameters.pop("rest_shape", rest_shape)
            if filename is not None:
                if "element_type" in parameters and "cell_type" not in parameters:
                    parameters["cell_type"] = parameters.pop("element_type")
                generated = self.generator.read(filename, **parameters)
            else:
                generated = self.create_mesh(geometry, **parameters)
        elif isinstance(mesh, str):
            if "element_type" in kwargs and "cell_type" not in kwargs:
                kwargs["cell_type"] = kwargs.pop("element_type")
            generated = self.generator.read(mesh, **kwargs)
        elif mesh is None:
            generated = self.create_mesh(**kwargs)
        else:
            raise TypeError("FEM.add_soft_particle expects FEMMesh, filename, or mesh dictionary")
        if rest_shape is not None:
            generated.set_rest_shape(rest_shape)
        if not generated.is_volume:
            raise ValueError("FEM soft particles require TET4 or HEX8 volume meshes")
        if self.scene.mesh is None:
            self.scene.add_mesh(generated)
        else:
            self.scene.add_mesh(FEMMesh.concatenate((self.scene.mesh, generated)))
        self.engine = None
        self.enginer = None
        return generated

    def add_material(self, model="StVK", material=None, **kwargs):
        if isinstance(model, dict):
            parameters = dict(model)
            parameters.update(kwargs)
            model = _key(parameters, "model", "type", default="StVK")
            material = _key(parameters, "material", default=material)
            for candidate in list(parameters):
                if str(candidate).replace("_", "").lower() in ("model", "type", "material"):
                    parameters.pop(candidate)
            kwargs = parameters
        created = self.material_manager.add_material(model, material, **kwargs)
        self.scene.add_material(created)
        self.engine = None
        self.enginer = None
        if hasattr(created, "print_message"):
            created.console_solver_name = "FEM"
            created.print_message(0)
        else:
            print_material_info(
                created,
                0,
                [
                    ("Density", getattr(created, "density", None)),
                    ("Thickness", getattr(created, "thickness", None)),
                ],
                solver_name="FEM",
            )
        return created

    def _boundary_from_dict(self, boundary):
        entries = boundary if isinstance(boundary, (list, tuple)) else [boundary]
        dirichlet = self.scene.dirichlet or DirichletBoundary()
        neumann = self.scene.neumann or NeumannBoundary()
        for entry in entries:
            kind = str(_key(entry, "type", "boundary_type", default="Dirichlet"))
            normalized = kind.replace("_", "").replace("-", "").replace(" ", "").lower()
            nodes = _key(entry, "nodes", "node_ids", "node_set")
            value = _key(entry, "value", "force", "traction", "pressure", default=0.0)
            if normalized in ("dirichlet", "displacement", "fixed", "fix"):
                dirichlet.add(nodes, _key(entry, "components", "component", default="all"), value)
            elif normalized in ("neumann", "nodal", "nodalforce", "force"):
                neumann.add_nodal_force(nodes, value, total=bool(_key(entry, "total", default=False)))
            elif normalized in ("traction", "surfacetraction"):
                neumann.add_traction(value, selector=_key(entry, "selector"))
            elif normalized in ("pressure", "surfacepressure"):
                neumann.add_pressure(value, selector=_key(entry, "selector"))
            elif normalized in ("edge", "edgetraction"):
                neumann.add_edge_traction(value, selector=_key(entry, "selector"))
            else:
                raise ValueError(f"Unsupported FEM boundary type {kind!r}")
        return dirichlet, neumann

    def add_boundary_condition(self, boundary=None, dirichlet=None, neumann=None):
        if boundary is not None:
            parsed_dirichlet, parsed_neumann = self._boundary_from_dict(boundary)
            dirichlet = parsed_dirichlet if dirichlet is None else dirichlet
            neumann = parsed_neumann if neumann is None else neumann
        self.scene.add_boundary_condition(dirichlet, neumann)
        self.engine = None
        self.enginer = None

    def add_contact(self, model="IPC", **kwargs):
        """Attach ``BarrierIPC`` or augmented-Lagrangian ``SemiIPC`` contact."""
        contact = FEMContact.create(model, **kwargs)
        self.scene.add_contact(contact)
        self.engine = None
        self.enginer = None
        return contact

    def add_soft_particle_contact(self, model="Linear", **kwargs):
        """Enable explicit DEM-style PT/EE contact between FEM bodies."""
        contact = FEMSoftParticleContactModel(model, **kwargs)
        self.scene.add_soft_particle_contact(contact)
        self.engine = None
        self.enginer = None
        return contact

    def add_soft_particle_property(self, body_id1, body_id2, property=None, **kwargs):
        """Set one explicit FEM soft-particle body-pair contact law."""
        if self.scene.soft_particle_contact is None:
            raise RuntimeError("call FEM.add_soft_particle_contact(...) before adding pair properties")
        created = self.scene.soft_particle_contact.add_property(body_id1, body_id2, property, **kwargs)
        self.engine = None
        self.enginer = None
        return created

    def add_contact_property(self, body_id1, body_id2, property=None, **kwargs):
        """Set IPC parameters for one unordered FEM body pair."""
        if self.scene.contact is None:
            raise RuntimeError("call FEM.add_contact('IPC', ...) before adding pair properties")
        created = self.scene.contact.add_property(body_id1, body_id2, property, **kwargs)
        self.engine = None
        self.enginer = None
        return created

    def set_bending_model(self, model="Quadratic"):
        """Select ``Quadratic``, ``Dihedral`` or ``None`` cloth bending."""
        from src.fem.cloth.ClothEnergy import normalize_bending_model

        self.solver_kwargs["bending_model"] = normalize_bending_model(model)
        self.engine = None
        self.enginer = None

    def add_cloth_energy(self, energy, **parameters):
        """Add a garment stitch, target spring, or frozen-frame SDF spring."""
        if isinstance(energy, dict):
            specification = dict(energy)
            specification.update(parameters)
        else:
            specification = {"type": energy, **parameters}
        self.scene.add_cloth_energy(specification)
        self.engine = None
        self.enginer = None
        return specification

    def add_stitch(self, stitches, stiffness, ratios=None):
        """Tie one cloth node to an interpolated point on another edge."""
        parameters = {
            "stitches": stitches,
            "stiffness": stiffness,
        }
        if ratios is not None:
            parameters["ratios"] = ratios
        return self.add_cloth_energy("GarmentStitch", **parameters)

    def add_spring(self, nodes, stiffness, targets=None):
        """Add isotropic node-to-target springs."""
        parameters = {"nodes": nodes, "stiffness": stiffness}
        if targets is not None:
            parameters["targets"] = targets
        return self.add_cloth_energy("Spring", **parameters)

    def add_sdf(
        self,
        nodes="all",
        stiffness=1.0,
        dhat=1.0,
        *,
        sdf=None,
        targets=None,
        normals=None,
    ):
        """Add C2 one-sided spring-SDF potential.

        ``targets`` and ``normals`` may be supplied directly.  Alternatively,
        an existing GeoTaichi SDF object is sampled during FEM preprocessing.
        The resulting target/normal frame is then evaluated entirely on the
        Taichi device.
        """
        parameters = {
            "nodes": nodes,
            "stiffness": stiffness,
            "dhat": dhat,
        }
        if sdf is not None:
            parameters["sdf"] = sdf
        if targets is not None:
            parameters["targets"] = targets
        if normals is not None:
            parameters["normals"] = normals
        return self.add_cloth_energy("SDF", **parameters)

    def add_element(self, element=None, **kwargs):
        """Compatibility hook; the element formulation is selected by mesh cells."""
        if element is not None and self.scene.mesh is not None:
            requested = element if isinstance(element, str) else _key(element, "type", "element_type")
            if requested is not None:
                from src.fem.generator import normalize_cell_type

                if normalize_cell_type(requested) != self.scene.mesh.cell_type:
                    raise ValueError("requested FEM element does not match the current mesh cell type")
        self.solver_kwargs.update(kwargs)

    def set_solver(self, solver=None, log=True, **kwargs):
        if solver is not None:
            if not isinstance(solver, dict):
                raise TypeError("FEM.set_solver positional argument must be a dictionary")
            kwargs = {**solver, **kwargs}
        aliases = {
            "Timestep": "dt",
            "SimulationTime": "simulation_time",
            "SaveInterval": "output_interval",
            "SavePath": "path",
            "max_iteration_number": "max_iterations",
        }
        for source, target in aliases.items():
            value = _key(kwargs, source)
            if value is not None and target not in kwargs:
                kwargs[target] = value
        retry_policy = StepRetryPolicy(
            enabled=_key(kwargs, "enable_step_retry", default=False),
            maximum_retries=_key(kwargs, "step_retry_max_retries", default=2),
            reduction=_key(kwargs, "step_retry_reduction", default=0.5),
            minimum_timestep=_key(kwargs, "step_retry_minimum_timestep", default=0.0),
        )
        kwargs.update(
            enable_step_retry=retry_policy.enabled,
            step_retry_max_retries=retry_policy.maximum_retries,
            step_retry_reduction=retry_policy.reduction,
            step_retry_minimum_timestep=retry_policy.minimum_timestep,
        )
        if retry_policy.enabled and self.sims.solver_type != "Implicit":
            raise ValueError("FEM step retry is available only for solver_type='Implicit'")
        self.solver_kwargs.update(kwargs)
        self.engine = None
        self.enginer = None
        if log:
            self.print_solver_info()
            print()

    def print_basic_simulation_info(self):
        print_solver_section(
            "FEM",
            "Basic Configuration",
            [
                ("Simulation Type", runtime_architecture()),
                ("Dimension", self.sims.dimension),
                ("Solver Type", self.sims.solver_type),
                ("Axisymmetric", self.sims.is_axisymmetric),
                ("Axis Offset", self.sims.axis_offset),
            ],
        )

    def print_solver_info(self):
        print_solver_section(
            "FEM",
            "Solver Information",
            [
                ("Simulation Time", _key(self.solver_kwargs, "simulation_time", "time")),
                ("Time Step", _key(self.solver_kwargs, "dt", "time_step", default="Automatic")),
                ("Requested Steps", _key(self.solver_kwargs, "steps", "step")),
                ("Output Interval (steps)", _key(self.solver_kwargs, "output_interval", "interval", default=1)),
                ("Save Path", _key(self.solver_kwargs, "path", "save_path")),
                ("Assembly Type", _key(self.solver_kwargs, "assemble_type", "assembly", default="MatrixFree")),
                ("Linear Solver", _key(self.solver_kwargs, "linear_solver")),
            ],
        )

    def print_memory_info(self):
        mesh = self.scene.mesh
        engine = self.engine
        print_solver_section(
            "FEM",
            "Memory Information",
            [
                ("Mesh Nodes", None if mesh is None else mesh.number_of_nodes),
                ("Mesh Elements", None if mesh is None else mesh.number_of_cells),
                ("Degrees of Freedom", None if engine is None else engine.degree_of_freedom),
                ("Element Type", None if mesh is None else mesh.cell_type),
                ("Contact Enabled", self.scene.contact is not None),
                ("Soft-particle Contact Enabled", self.scene.soft_particle_contact is not None),
            ],
        )

    def print_neighbor_search_info(self):
        contact = self.scene.contact
        soft_contact = self.scene.soft_particle_contact
        if contact is None and soft_contact is None:
            return
        entries = [("Search Usage", "Contact broad phase")]
        if contact is not None:
            entries.append(("FEM Contact Broad Phase", contact.broad_phase))
        if soft_contact is not None:
            entries.extend(
                [
                    ("Soft-particle Broad Phase", soft_contact.search),
                    (
                        "Soft-particle Verlet Multiplier",
                        soft_contact.verlet_distance_multiplier,
                    ),
                    (
                        "Soft-particle Verlet Distance",
                        soft_contact.verlet_distance,
                    ),
                ]
            )
        print_solver_section(
            "FEM",
            "Neighbor Search Information",
            entries,
        )

    def build(self):
        if self.scene.mesh is None:
            raise RuntimeError("FEM mesh has not been set")
        if self.scene.material is None:
            raise RuntimeError("FEM material has not been set")
        if self.scene.mesh.is_volume and self.sims.dimension != 3:
            raise ValueError("TET4/HEX8 volume FEM requires dimension=3")
        if self.sims.dimension == 2:
            out_of_plane_extent = float(self.scene.mesh.points[:, 2].max() - self.scene.mesh.points[:, 2].min())
            in_plane_scale = max(
                float(np.ptp(self.scene.mesh.points[:, 0])),
                float(np.ptp(self.scene.mesh.points[:, 1])),
                1.0,
            )
            if out_of_plane_extent > 1.0e-12 * in_plane_scale:
                raise ValueError("two-dimensional FEM coordinates must lie in the stored " "x-y / (r,z) plane")
        is_cloth = bool(getattr(self.scene.material, "is_cloth", False))
        is_elastoplastic = bool(getattr(self.scene.material, "is_fem_elastoplastic", False))
        kwargs = dict(self.sims.kwargs)
        kwargs.update(self.solver_kwargs)
        kwargs.setdefault("dimension", self.sims.dimension)
        kwargs.setdefault("axisymmetric", self.sims.is_axisymmetric)
        kwargs.setdefault("axis_offset", self.sims.axis_offset)
        requested_backend = kwargs.pop("assembly_backend", kwargs.pop("backend", "auto"))
        requested_backend = str(requested_backend).strip().replace("_", "").replace("-", "").lower()
        if requested_backend in ("cpu", "numpy", "scipy"):
            raise ValueError(
                "FEM no longer provides a NumPy runtime backend; initialize "
                "Taichi and use backend='taichi'. Scipy remains available only "
                "through linear_solver='Scipy'."
            )
        if requested_backend not in ("auto", "taichi", "device", "gpu"):
            raise ValueError("FEM backend must be 'auto' or 'taichi'")
        if not _taichi_is_initialized():
            raise RuntimeError("FEM requires initialized Taichi; call geotaichi.init(...) " "before FEM.build()")
        backend = "taichi"
        if is_cloth and not self.scene.mesh.is_membrane:
            raise ValueError("cloth constitutive models require a TRI3 surface mesh")
        if is_cloth and self.sims.dimension != 3:
            raise ValueError("cloth FEM uses embedded 3D positions and requires dimension=3")
        if is_elastoplastic and self.scene.mesh.cell_type != "hexahedron":
            raise ValueError("FEM elastoplastic constitutive models currently require HEX8")
        if is_elastoplastic and self.sims.solver_type != "Explicit":
            raise ValueError("HEX8 elastoplastic FEM currently requires solver_type='Explicit'")
        if is_elastoplastic and self.scene.contact is not None:
            raise ValueError("HEX8 elastoplastic FEM does not support IPC/AL contact")
        if self.scene.soft_particle_contact is not None:
            if self.sims.solver_type != "Explicit":
                raise ValueError(
                    "DEM-style FEM soft-particle contact requires solver_type='Explicit'; "
                    "use IPC for implicit soft-particle contact"
                )
            if not self.scene.mesh.is_volume:
                raise ValueError("FEM soft-particle contact requires a TET4 or HEX8 mesh")
            kwargs["soft_particle_contact"] = self.scene.soft_particle_contact
        if self.scene.cloth_energies and not is_cloth:
            raise ValueError(
                "garment stitch, SDF, and spring energies require a cloth " "constitutive model on a TRI3 surface mesh"
            )
        if self.scene.contact is not None and self.sims.solver_type != "Implicit":
            raise ValueError("FEM IPC and augmented-Lagrangian contact currently require solver_type='Implicit'")
        if (
            kwargs.get("enable_step_retry", False)
            and self.scene.contact is not None
            and self.scene.contact.model == "AugmentedLagrangian"
        ):
            raise ValueError(
                "FEM step retry does not support augmented-Lagrangian contact "
                "because its multipliers advance inside Newton iterations"
            )
        engine_class = None
        if is_cloth:
            if requested_backend in ("cpu", "numpy", "scipy"):
                raise ValueError("cloth FEM currently requires backend='taichi'")
            from src.fem.engines.ClothFEM import (
                ClothExplicitFEM,
                ClothImplicitFEM,
            )

            engine_class = ClothExplicitFEM if self.sims.solver_type == "Explicit" else ClothImplicitFEM
        else:
            from src.fem.engines.ClassicalFEM import (
                ClassicalExplicitFEM,
                ClassicalImplicitFEM,
            )

            engine_class = ClassicalExplicitFEM if self.sims.solver_type == "Explicit" else ClassicalImplicitFEM
        if self.scene.contact is not None:
            kwargs["contact"] = self.scene.contact
        if self.scene.cloth_energies:
            kwargs["cloth_energies"] = list(self.scene.cloth_energies)
        self.engine = engine_class(
            self.scene.mesh,
            self.scene.material,
            self.scene.dirichlet,
            self.scene.neumann,
            **kwargs,
        )
        # Boundary topology and every configured transient frame are uploaded
        # once after the concrete integrator has finalized dt/step count.  No
        # boundary callable or NumPy array is touched by a production substep.
        self.engine.initialize_device_boundaries()
        self.engine.timer = self.sims.timer
        self.enginer = self.engine
        if self.log:
            self.print_memory_info()
            self.print_neighbor_search_info()
            print()
        return self.engine

    def run(self, *args, **kwargs):
        if self.engine is None:
            self.build()
        print_simulation_start("FEM")
        return self.engine.run(*args, **kwargs)

    def differentiable(self, steps=None):
        """Create a fixed-step device trajectory adjoint for implicit FEM."""
        if self.engine is None:
            self.build()
        from src.fem.engines.DifferentiableFEM import DifferentiableFEM

        return DifferentiableFEM(self.engine, steps=steps)


__all__ = ["FEM"]
