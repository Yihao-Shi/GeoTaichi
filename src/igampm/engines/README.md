# Nonlinear Solution Algorithms for Coupled Isogeometric–Material Point Mechanics

This package separates explicit DEM-law stepping from the composed implicit
IPC engine. Public engines are exported directly from this package.

| Module | Responsibility |
| --- | --- |
| `ExplicitEngine.py` | Explicit IGA/MPM force exchange, CFL gate, and synchronized advancement |
| `CoupledEngine.py` | Engine composition, configuration validation, and persistent Taichi fields |
| `ContactEngine.py` | Point--NURBS closest-point kernels, barrier blocks, and contact energy |
| `ImplicitEngine.py` | Coupled COO/HashTriplet assembly, point--NURBS ACCD, material CCD, Newton, and Armijo search |
| `FrictionEngine.py` | Lagged and fully implicit friction assembly and nonlinear orchestration |
| `FullyImplicitFriction.py` | Reusable exact Taichi friction derivative functions |
| `TimeIntegration.py` | Host validation of scalar time-integration coefficients |
| `_Common.py` | Imports shared by the responsibility mixins; it owns no solver state |

All production per-contact calculations and reductions execute in Taichi.
IPC queries rebuild current positive-weight control-hull AABBs. Far inactive
pairs keep conservative distance lower bounds for ACCD; near pairs and retained
SemiIPC multipliers use the complete closest-point query. ACCD skips moving
geometry queries only when its relative-motion bound certifies the whole
segment. Contact buffers retain all particle--surface pairs, and diagnostics
for inactive pairs can report a distance lower bound.
Explicit contact rebuilds NURBS bounds, projects points, updates history, and
scatters DEM-law forces through device kernels. In implicit IPC, every
point--NURBS ACCD pair owns its complete bounded `while` and
virtual moving-geometry closest-point queries inside a kernel. Boundaries with
the same degree signature share a specialization; Python only dispatches those
basis groups and reads the final scalar minimum. Python coordinates the
remaining solver-level Newton and friction iterations and reads scalar
diagnostics. The first three-dimensional compilation is comparatively
expensive because Taichi inlines the span-multistart projected-Newton closest
query inside the ACCD loop; later launches reuse the compiled specialization.
Direct ULMPM plastic materials reuse the owning MPM residual/tangent kernels;
this package adds only the coupled acceptance snapshot for material-owned
history. Elastic MPM therefore allocates no plastic rollback field.
An explicitly selected SciPy linear solve is the only permitted numerical
host boundary; NumPy reference laws belong under `tests/helpers`.

The explicit, barrier, friction, and implicit capabilities are validated and
bound when the coupled engine is built. Hot loops call the selected capability
directly; they do not repeatedly probe mixins with `hasattr`. Contact rebuild,
CCD feasibility, nonlinear convergence, and accepted-state decisions remain
dynamic because they depend on the current physical state.

The historical internal `*_cuda` method names are retained temporarily for
test and downstream compatibility. They dispatch Taichi kernels and are not
an alternative NumPy backend; availability is determined by Taichi device
fields rather than by reimplementing the algorithm in Python.
