# GeoTaichi Compatibility Reference

## Contents

1. General rule
2. MPM
3. DEM and LSMPM
4. Coupling
5. FEM
6. IGA/IGAMPM
7. Backends and precision
8. Aliases and fallbacks

## 1. General rule

Treat this file as a checklist, not authority. Confirm every selected
combination in current `validate_configuration` and the consuming engine.
Reject unsupported combinations before generating a long script.

## 2. MPM

- Incompressible fluid FDM requires the supported implicit fluid and staggered
  element route.
- Semi-implicit two-phase branches are FEM-style and are not interchangeable
  with incompressible FDM pressure projection.
- Adaptive explicit penalty/bridging and implicit strong shape constraints are
  different methods.
- Native MPM and the direct backend expose different body/contact lifecycle
  methods.
- `MatrixFree`, `COO`, and `HashTriplet` have different boundary and symmetry
  assumptions.
- Strong Dirichlet behavior is not automatically available in every
  matrix-free branch.
- A sparse grid or direct IPC flag does not imply compatibility with every
  material and solver family.
- Direct `MPM(mode="SoftParticle").run()` is currently incomplete/rejected;
  use the DEM/MPDEM LSMPM route.

## 3. DEM and LSMPM

- Scheme selects storage, generator, neighbor/contact, solver, and recorder
  behavior; do not mix settings from DEM, LSDEM, LSMPM, poly-superquadric, and
  AffineBody branches.
- LSMPM requires 3D TLMPM soft-body assumptions in its current implementation.
- Tetrahedral soft-grid support forces the linear basis and four-node support.
- Legacy soft-grid `Sparse`/`BlockSparse`/`FixedSparse` names normalize to
  compact contiguous support storage.
- WENO5 and Semi-Lagrangian are LSMPM level-set transport choices; generic
  level-set WENO order settings are a separate implementation.
- Mixed soft--AffineBody IPC has dedicated implicit assembly/backend limits.
- Contact property completeness and point coordination are mandatory; LSMPM
  capacity overflow must fail rather than drop contacts.
- Energy-conserving contact requires `FreeParameter >= 2`.

## 4. Coupling

- `DEMPM()` may create its coupled sub-solvers, while `DEMPM(dem, mpm)` may
  receive existing ones; both must be configured with coupling enabled before
  allocation/build/run.
- Coupling support is scheme-specific; an individually valid sub-solver pair
  may still be invalid together.
- Implicit coupling support does not mean arbitrary implicit solid MPM contact.
- CFDEM requires its explicit coupling route and supported fluid/particle
  combination.
- Incompressible DEM-sphere and LSDEM volume-fraction IBM coupling require a
  3D implicit fluid FDM child. The LSDEM body does not become cut-cell
  geometry; cut cells in that combination are reserved for fixed walls.
- Sparse background-grid support is limited to validated coupling branches.
- LSMPM DEM-owned routing requires its specific scene state; routing queries do
  not enable the mode.
- `FEDEM` accepts three scheme-specific routes. `DEM` and `LSDEM` require
  explicit FEM, use `Linear`/`HertzMindlin`, and do not combine with FEM
  IPC/AL. The FEM side remains a volume or membrane mesh with a TRI3 contact
  boundary; it does not acquire an SDF. LSDEM samples only the rigid child's
  existing SDF. `AffineBody` requires implicit elastic FEM and coupled IPC,
  with direct PT/EE geometry, analytic PSD-projected Hessians, CCD/ACCD,
  Armijo, and `COO`/`HashTriplet`. All routes support `LinkedCell` and `BVH`.
  Affine mixed contact supports pair-local lagged IPC friction and coupled
  fixed-point refreshes; fully implicit mixed friction is unavailable. Affine
  self-friction must remain lagged with one friction iteration. Explicit
  classical volume/TRI3, cloth
  TRI3, and HEX8 elastoplastic FEM remain valid penalty-contact surfaces.
- `FEMPM` has a 3D DEM-style `Linear`/`HertzMindlin` branch that
  requires explicit particle/grid MPM with `coupling="Lagrangian"` plus
  explicit FEM. `IPC` requires Direct implicit ULMPM plus implicit elastic FEM;
  the MPM material may be Neo-Hookean, associated finite-strain
  Drucker--Prager, or associated finite-strain von Mises. It uses monolithic
  `COO` or `HashTriplet`, PSD projection, CCD, and line search. Both contact
  branches support `LinkedCell` and `BVH`; coupled IPC does not combine with
  standalone FEM/MPM IPC or FEM elastoplasticity. Direct TLMPM finite-strain
  plasticity is not implemented. Its implicit IPC branch supports 3D,
  Cartesian 2D, and no-swirl axisymmetric 2D. The latter two use point--edge
  contact. Cartesian 2D DP/von Mises uses a 3D plane-strain material map and
  rejects `plane_strain=False`; axisymmetry requires a shared `axis_offset`, positive reference
  radii, `F_theta_theta=r/R`, and `2*pi*R` reference measures.

## 5. FEM

- FEM requires initialized Taichi; Scipy is available only as an explicitly
  selected linear solver, not as a NumPy runtime backend.
- Volume FEM requires 3D TET4/HEX8. Cloth models require embedded-3D TRI3 and
  automatically select the cloth membrane deformation map.
- Standalone explicit FEM soft-particle contact requires disconnected
  TET4/HEX8 bodies plus `Linear` or `HertzMindlin`; it supports `LinkedCell`
  and `BVH`, uses a Verlet skin, and requires positive `ContactThickness` for
  edge--edge response. FEM itself owns no SDF.
- Linear TET4 uses the direct edge-matrix map `F=Ds inverse(Dm)`, equivalent
  to the standard reference-gradient sum. Its `F` is constant within each
  tetrahedron; HEX8 retains quadrature-dependent deformation gradients.
- FEM contact currently requires the implicit solver and line search.
- HEX8 elastoplastic FEM uses shared incremental solid models, eight Gauss
  points, and explicit updated-Lagrangian state updates. It deliberately
  rejects FEM IPC/AL.
- Pair-specific `add_contact_property` is IPC-only; once pair properties are
  present, unlisted body pairs are inactive.
- PCG requires `project_pd=True`; use BiCGSTAB for an unprojected tangent.
- `broad_phase="LinkedCell"` or `"BVH"` selects the spatial AABB backend only.
  Both pass through common topology/stitch exclusion, exact-distance or swept
  CCD culling, prefix compaction, and contact assembly.
- IPC rebuilds current and swept candidates without a Verlet multiplier.
  Augmented Lagrangian contact permits trial penetration and is not CCD-clipped.
- Rigid LSDEM--FEM contact belongs to explicit `FEDEM`: current FEM boundary
  nodes query only the rigid child's SDF and the list rebuilds every step.
  Elastic volume FEM--AffineBody contact belongs to implicit `FEDEM` IPC.

## 6. IGA/IGAMPM

- Pure IGA and IGAMPM have distinct ownership.
- IGAMPM accepts two solver-family branches. Implicit/implicit selects
  monolithic IPC. Explicit/explicit selects the shared DEM `Linear` or
  `HertzMindlin` law for finite-radius MPM point--NURBS contact. Mixed pairs
  are rejected.
- Explicit IGAMPM is currently 3D Cartesian, requires the native MPM backend
  with `coupling="Lagrangian"`, and must be constructed before MPM particle
  allocation. Add every active `MPMmaterial`/`IGAbody` property before build.
  Rolling friction is rejected because IGA control points have no rotational
  DOFs. The explicit CFL gate covers MPM waves and contact stiffness; IGA has
  no independent spectral stability estimator.
- The MPM child may use Direct implicit ULMPM `NeoHookean`, associated
  finite-strain `DruckerPrager`, or associated finite-strain `VonMises`.
  Plastic MPM keeps the IGA child elastic and is not supported by Direct
  TLMPM. Accepted plastic history and `F0` share one device transaction.
  Cartesian 2D plasticity is a 3D plane-strain embedding, not an intrinsic-2D
  return mapping; axisymmetric plasticity uses the 3D no-swirl map.
- Freeze contact configuration before build only after the intended mode is
  final.
- Contact parameters and, for IPC, the COO/HashTriplet assembly choice are
  frozen after the IGAMPM engine is built.
- IGA and IGAMPM support Cartesian 2D/3D and no-swirl axisymmetric 2D.
  Axisymmetric IGA and Direct MPM children must share `axis_offset`; component
  zero is radius, both constitutive blocks use a 3D map with
  `F_theta_theta=r/R`, and reference volume/contact-ring measures include
  `2*pi*R`.

## 7. Backends and precision

- Set `GEOTAICHI_REAL_DTYPE` before importing GeoTaichi and pass the matching
  `default_fp` to `init` where required by examples.
- Apple Silicon may map `arch="cpu"` to Metal; report the effective backend.
- CUDA-only kernels, shared memory, atomics, sparse structures, or memory
  behavior require CUDA validation.
- A CPU/Metal orchestration smoke run does not prove CUDA kernel correctness.

## 8. Aliases and fallbacks

Use canonical names in new models. Record aliases only after source confirms
normalization. A fallback is allowed when it preserves the requested physics
and is documented in the handoff. Do not use a different constitutive model,
contact law, precision, or solver family as an implementation fallback without
user approval.
