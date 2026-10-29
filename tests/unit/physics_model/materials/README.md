# Material unit-test matrix

Keep constitutive-law oracles in this directory and solver material-entry tests
under `unit/fem`, `unit/iga`, and `unit/mpm`. Shared valid parameter sets live in
`tests/helpers/material_cases.py`. All these tests carry the `materials` marker
and are selected by the materials partition. Tests use CPU/f64; dimension-bound
solver tests run in isolated 2D or 3D interpreters.

The matrix follows the production material factories. A model explicitly
rejected by a solver is excluded from that solver's positive tests. Aliases
are not separate constitutive laws. Rigid bodies and the `UserDefined` extension
template are not additional deformable built-in laws.

## Solver entry points

| Entry point | Supported models exercised | Tests and checks |
| --- | --- | --- |
| FEM elastic/surface | StVK (including the LinearElastic alias), NeoHookean, ClothARAP, ClothNeoHookean | `unit/fem/test_fem_materials.py`: factory selection, rigid maps, directional tangent finite differences |
| FEM explicit HEX8 | ElasticPerfectlyPlastic, MohrCoulomb, DruckerPrager, StateDependentMohrCoulomb, ModifiedCamClay, GranularMaterial, SanisandMS, NorSand | `unit/fem/test_hex8_elastoplastic.py`: construction, state initialization, affine loading at all eight Gauss points, internal-force balance |
| IGA explicit/implicit | NeoHookean | `unit/iga/test_iga_materials.py`: actual solver construction in 2D/3D, zero and nonzero deformation, energy/PK1 analytical oracle, tangent finite differences |
| Native explicit MPM | LinearElastic; NeoHookean, HenckyElastic, MooneyRivlin, Gent, Hydrogel; all eight FEM incremental models above; Newtonian, Bingham | `unit/mpm/test_material_models.py`: production material manager, state schema, zero and repeated loading, independent particle history, volume update, finite state in 2D/3D |
| Direct explicit MPM | Native models above except GranularMaterial, SanisandMS and NorSand | Same test through `ExplicitMPM.handle_material`; finite-strain 2D stress must agree with the embedded 3D plane-strain response, including the out-of-plane component |
| Direct implicit MPM | NeoHookean; finite-strain DruckerPrager, VonMises, ModifiedCamClay, StateDependentDruckerPrager | `unit/mpm/test_finite_material_adapters.py`: actual UL solver construction, physical stress/tangent finite differences and accepted-state stress consistency |
| Soft-particle MPM | NeoHookean, HenckyElastic, MooneyRivlin, Gent, Hydrogel | `unit/mpm/test_finite_material_adapters.py`: production manager, energy/stress/tangent finite differences and state commit; `test_lsmpm_multimaterial.py` covers the NeoHookean material table |
| Static/dynamic two-phase Direct MPM | linearElastic, neoHookean, druckerPrager | `unit/mpm/test_static_twophase_jacobian_reference.py`: local constitutive and global assembly Jacobians; DP covered in 3D |

IGA currently constructs NeoHookean directly. FEM incremental models are
supported only by explicit 3D HEX8; Direct implicit StateDependentDruckerPrager
requires UL; mixed soft-particle materials currently require NeoHookean.
These tests do not imply support outside those production restrictions.

## Shared constitutive-law oracles

| Family | Coverage |
| --- | --- |
| LinearElastic, StVK | parameters, stress, tangent and invariants |
| NeoHookean, MooneyRivlin, Gent, HenckyElastic, Hydrogel | energy/PK1 consistency, analytic tangent versus finite differences, repeated-stretch Hencky limit, invalid deformation/extensibility, sound speed |
| ElasticPerfectlyPlastic, MohrCoulomb, DruckerPrager | validation, state, zero/elastic/plastic response, return surfaces, unloading/history and elastic tangent |
| Small-strain history consistency | `test_small_strain_consistency.py`: tensor-invariant directional derivative, current local MC/Mises strength, signed MC interior distance, analytical shear curves through residual strength, MCC bond/subloading chain |
| Finite-strain DruckerPrager, VonMises, ModifiedCamClay, StateDependentDruckerPrager | return mapping, physical tangents, accepted history and model-specific hardening/lagged-state checks |
| SanisandMS | state, constitutive response and model-specific history checks |
| Newtonian, Bingham | state, pressure, viscosity/yield branches, mixed Voigt response and critical rate |
| Explicit mu(I): DruckerPrager, MohrCoulomb, ModifiedCamClay, GranularMaterial | pressure derivatives, invalid parameters, zero-pressure/regularization limits, rate/yield consistency, softening derivatives, history and Direct explicit kernel |

The full per-model contract in `tests/README.md` remains the target. Entry-point
loading tests do not prove every plastic branch or every solver configuration.
In particular, infinitesimal ModifiedCamClay, StateDependentMohrCoulomb and
NorSand have solver loading coverage, not a complete standalone branch/tangent
contract. Random-field and user-defined extensions require their own cases.
The existing Bingham regression records the current normal-component final
yield check, which clears a pure-shear-only state.

Run all material-marked tests with:

```bash
GEOTAICHI_FORCE_CPU=1 python tests/testing/run_partition.py materials
```

For the local `[full test]` commit gate, the `commit-materials` partition selects
all shared law-oracle files and the solver entry checks listed above, without
collecting the whole repository. The hook selects it when the staged diff
affects material code/tests or shared dependencies; see
[local commit checks](../../../README.md#local-commit-checks). Run it explicitly with:

```bash
GEOTAICHI_FORCE_CPU=1 python tests/testing/run_partition.py commit-materials -q --taichi-arch cpu
```

Run only the solver material entry tests with:

```bash
GEOTAICHI_FORCE_CPU=1 python -m pytest \
  tests/unit/fem/test_fem_materials.py \
  tests/unit/fem/test_hex8_elastoplastic.py \
  tests/unit/iga/test_iga_materials.py \
  tests/unit/mpm/test_material_models.py \
  tests/unit/mpm/test_finite_material_adapters.py \
  tests/unit/mpm/test_static_twophase_jacobian_reference.py
```
