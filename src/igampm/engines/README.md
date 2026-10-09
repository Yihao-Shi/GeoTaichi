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
IPC queries rebuild current positive-weight control-hull AABBs. A bulk
screen stores near candidates per surface; projection, barrier assembly, and
friction initialization traverse those candidates. The complete pair table
still stores conservative distance bounds for inactive pairs and rollback.
Explicit contact rebuilds NURBS bounds, projects points, updates history, and
scatters DEM-law forces through device kernels.

For CCD, bounds enclose both current control points and their proposed
endpoints. Each swept particle box queries a refitted global surface BVH;
3D candidates also traverse the surface's swept knot-span BVH. Clearance and
a floating-point guard expand overlap tests. Remaining pairs pass through the
relative-motion certificate and then the existing moving NURBS ACCD query.
Only compact CCD candidates advance and contribute to the global minimum;
noncandidate entries of `contact_accd_toc` are not initialized or meaningful.
One kernel advances one ACCD iteration. Python dispatches degree groups and
reads the unfinished-pair count between iterations; pair geometry remains on
the device. This avoids Taichi's expensive compilation of the closest-point
search inside an additional kernel-level ACCD loop.
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
