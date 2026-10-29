# Find the code that owns a result

Use this map to choose a starting point, then follow the actual call and the
selected solver branch. It is not a promise that every facade supports every
material, contact law, dimension, or backend.

The public package [geotaichi](../geotaichi/__init__.py) re-exports constructors
from [src](../src/__init__.py). A typical route is:

```text
public constructor / method
  -> configuration and compatibility checks
  -> scene, material, geometry, and capacity construction
  -> selected engine / solver
  -> Taichi kernels and accepted state
  -> recorder, restart, or client result
```

## Solver entry points

| Requested behavior | Start here | Follow into |
| --- | --- | --- |
| MPM configuration and dispatch | [mainMPM.py](../src/mpm/mainMPM.py) | [Simulation](../src/mpm/Simulation.py), [SceneManager](../src/mpm/SceneManager.py), [engines](../src/mpm/engines/) |
| Direct MPM solve | [MPMSolver](../src/mpm/engines/direct/MPMSolver.py) | [direct MPM implementation](../src/mpm/engines/direct/) and [soft-particle operators](../src/mpm/soft_particle/) |
| DEM bodies, walls, and contact | [mainDEM.py](../src/dem/mainDEM.py) | [SceneManager](../src/dem/SceneManager.py), [contact models](../src/dem/contact/), [engines](../src/dem/engines/) |
| FEM volume or membrane/cloth | [mainFEM.py](../src/fem/mainFEM.py) | [material selection](../src/fem/MaterialManager.py), [FEMSolver](../src/fem/engines/FEMSolver.py), [contact](../src/fem/contact/) |
| IGA solve | [mainIGA.py](../src/iga/mainIGA.py) | [material selection](../src/iga/MaterialManager.py), [IGASolver](../src/iga/engines/IGASolver.py), [patch](../src/iga/elements/Patch.py) |
| DEM--MPM or fluid--particle exchange | [mainDEMPM.py](../src/mpdem/mainDEMPM.py) | [coupled engines](../src/mpdem/engines/) and [MPDEM guide](../src/mpdem/README.md); `DEMPM` and `MPDEM` enter this family |
| FEM--DEM coupling | [mainFEDEM.py](../src/fedem/mainFEDEM.py) | [explicit engine](../src/fedem/Engine.py), [affine IPC engine](../src/fedem/AffineIPCEngine.py), [FEDEM guide](../src/fedem/README.md) |
| FEM--MPM coupling | [mainFEMPM.py](../src/fempm/mainFEMPM.py) | [explicit engine](../src/fempm/Engine.py), [implicit engine](../src/fempm/ImplicitEngine.py), [FEMPM guide](../src/fempm/README.md) |
| IGA--MPM coupling | [mainIGAMPM.py](../src/igampm/mainIGAMPM.py) | [contact manager](../src/igampm/ContactManager.py), [engines](../src/igampm/engines/), [IGA--MPM guide](../src/igampm/README.md) |

Find the method that consumes the option, not only its public spelling. In
particular, check the chosen native/direct MPM route and explicit/implicit
coupling route before transferring an implementation from a neighboring solver.

## Shared code and evidence

| Area | Implementation owner | Useful evidence |
| --- | --- | --- |
| Constitutive response and history | [constitutive models](../src/physics_model/consititutive_model/), plus the consuming solver's MaterialManager | [material coverage](../tests/unit/physics_model/materials/README.md); small-strain and finite-strain contracts differ |
| Contact law and closest features | [shared contact models](../src/physics_model/contact_model/), [contact detection](../src/contact_detection/) | [contact guide](../src/physics_model/contact_model/README.md) and the nearest solver-specific contact test |
| Sparse assembly and linear solves | [linear_solver](../src/linear_solver/) | [assembly equivalence tests](../tests/unit/linear_solver/test_assembly_backends.py) |
| NURBS basis and geometry | [nurbs](../src/nurbs/) and [IGA patch](../src/iga/elements/Patch.py) | [basis/refinement tests](../tests/unit/iga/geometry/test_nurbs_basis_and_refinement.py), [VTU round trips](../tests/unit/iga/io/test_patch_vtu_output.py) |
| Precision, field transfer, diagnostics | [TypeDefination.py](../src/utils/TypeDefination.py), [FieldIO.py](../src/utils/FieldIO.py), [SolverDiagnostics.py](../src/utils/SolverDiagnostics.py) | [test runtime isolation](../tests/README.md#runtime-and-dimension-isolation), [prefix-transfer tests](../tests/unit/utils/test_field_io_prefix.py) |
| MPM output and restart | [Recorder.py](../src/mpm/Recorder.py), [restart entry](../src/mpm/mainMPM.py), [BodyReader](../src/mpm/generator/LoadFromFile.py) | [two-phase restart tests](../tests/unit/mpm/test_two_phase_restart.py) |
| Coupled FEM--DEM checkpoint | [Checkpoint.py](../src/fedem/Checkpoint.py) | [capacity/state restore tests](../tests/unit/fedem/test_checkpoint_capacity.py) |
| Model resources and task execution | [shared resources](geotaichi-model-builder/assets/), [MCP core](geotaichi-mcp/src/geotaichi_mcp/core/), [execution](geotaichi-mcp/src/geotaichi_mcp/execution/) | [MCP guide](geotaichi-mcp/README.md); numerical loops use [RuntimeHook](../src/utils/RuntimeHook.py) |
| Blender exchange | [manifest builder](../blender/core/manifest.py), [job client](../blender/core/job_client.py), [result import](../blender/results.py) | [Blender guide](../blender/README.md), shared SolverJob/SceneManifest contracts |

For an unfamiliar model or key, use the existing capability lookup rather than
expanding this map into a second API catalogue:

```bash
python agent/geotaichi-model-builder/scripts/query_capabilities.py browse
python agent/geotaichi-model-builder/scripts/query_capabilities.py query "incompressible pressure"
```

Read [WORKFLOWS.md](WORKFLOWS.md) once the owning path is known. Public API and
theory details remain in the owning module and the existing model-builder
references; this map only connects those sources.
