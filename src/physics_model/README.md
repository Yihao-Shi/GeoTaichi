# Constitutive and Contact Mechanics: Shared Material and Interface Models

`src/physics_model` is the shared physics-law layer for GeoTaichi solvers. It
contains continuum constitutive responses and pair/contact potentials, while
solver modules own discretization, topology, state integration, and global
assembly.

Constitutive implementations use the correctly spelled `constitutive_model`
package. Low-level imports must use this path; the former misspelled package
has no forwarding module. Public `geotaichi` solver constructors are unchanged.

## Package layout

| Path | Responsibility |
| --- | --- |
| [`constitutive_model/`](constitutive_model/README.md) | Shared constitutive theory and material-model implementations |
| `constitutive_model/finite_strain/` | Hyperelastic and multiplicative finite-strain plastic models |
| `constitutive_model/cloth_model/` | Cloth ARAP and Neo-Hookean membranes, quadratic and dihedral bending laws |
| `constitutive_model/infinitesimal_strain/` | Elasticity, plasticity, critical-state, and granular models |
| `constitutive_model/strain_rate/` | Newtonian and Bingham fluid responses |
| [`contact_model/`](contact_model/README.md) | Shared DEM and IPC contact theory and implementations |
| `contact_model/ipc/` | IPC barrier, friction, closest-point derivatives, mollifiers, measures, PSD projection, and NURBS/affine contact helpers |

## Constitutive families

Finite-strain models include St. Venant-Kirchhoff, compressible Neo-Hookean,
Hencky elasticity, Mooney-Rivlin, Gent, hydrogel, cloth ARAP/Neo-Hookean,
classical associated Drucker--Prager, associated von Mises, and finite-strain
Modified Cam--Clay responses. The
plastic models use multiplicative Hencky return mapping and expose indexed
energy, first Piola stress, analytic tangent, elastic projection, and device
state-commit functions. Drucker--Prager and von Mises are parallel material
classes over a shared internal Hencky-plasticity base. Drucker--Prager is
currently perfect plastic and
associated; von Mises optionally uses linear isotropic hardening.
All finite-strain plastic classes require a 3-by-3 deformation gradient.
Direct Cartesian 2D solvers provide it through plane-strain embedding, while
Direct axisymmetric solvers provide the full no-swirl map.

Infinitesimal-strain and elastoplastic models include linear elasticity,
Drucker-Prager, Mohr-Coulomb variants, Modified Cam-Clay, NorSand, SANISAND,
elastic-perfectly-plastic, granular, softening, and rate-dependent responses.
State-variable layout and stress integration are selected by the MPM material
manager.

Rate-based fluid models include Newtonian and Bingham behavior.

## Solver-level use

Users normally select a model through a solver facade:

```python
fem.add_material(
    "NeoHookean",
    young_modulus=1.0e6,
    poisson_ratio=0.3,
    density=1000.0,
)

mpm.add_material(
    model="DruckerPrager",
    material={
        "MaterialID": 1,
        "Density": 2500.0,
        "YoungModulus": 1.0e6,
        "PoissonRatio": 0.3,
        "Friction": 30.0,
        "Cohesion": 0.0,
        "Dilation": 0.0,
    },
)
```

Direct low-level use is intended for material development and unit tests:

```python
from src.physics_model.constitutive_model.finite_strain.NeoHookean import NeoHookeanModel

model = NeoHookeanModel(
    material_type="Solid", configuration="TL", solver_type="Implicit"
)
model.model_initialize(
    {"Density": 1000.0, "YoungModulus": 1.0e6, "PoissonRatio": 0.3}
)
```

## IPC building blocks

`contact_model/ipc` separates scalar barrier/friction laws, closest-point
geometry, distance derivatives, mollifiers, geometric measures, local matrix
assembly, affine/level-set mappings, and NURBS contact. Reuse these shared
functions when adding a new IPC consumer so coefficients, friction smoothing,
and PSD projection remain consistent across FEM, MPM, DEM, and IGA-MPM.
The NURBS geometry helpers also accept virtual linear control-point motion
`P_i(alpha) = P_i0 + alpha*dP_i`; point--NURBS ACCD uses this path to rebuild
closest points inside per-contact Taichi loops without mutating shared geometry.

Near-parallel edge classification and closest coordinates use cross-product
identities instead of subtracting nearly equal Gram products. This keeps finite
segment gaps consistent across ABD, FEM, FEM--ABD, friction frames, and CCD/ACCD;
an intersection of the supporting lines outside the segments is not contact.

## Runtime conventions

Production stress updates and contact evaluations are `@ti.func` or
`@ti.kernel` operations. Python/NumPy reference responses belong under `tests/`;
production model changes should expose device numerical implementations.
Output adapters download device results rather than recomputing constitutive
responses on the host. Python owns configuration validation and preprocessing.
Native MPM keeps raw restart history separate from visualization invariants;
`ConstitutiveBase.reload_state_variables` validates field names, particle counts,
components and finite values before uploading any material history.
A constitutive implementation should keep energy, stress, and tangent
formulas mutually consistent and document any regularization or PSD projection
that changes the Newton path.
Shared failure types are defined in `src/utils/errors.py`. Import
`InvertedElementError` directly from there when handling invalid element maps;
constitutive modules do not define or re-export their own copy.

## Tests

Material invariants, finite-strain tangents, elastoplastic integration, and
fluid models are tested under `tests/unit/physics_model/materials/`. IPC laws
and geometry are tested under `tests/unit/physics_model/contact/` and by each
owning solver's contact tests.
