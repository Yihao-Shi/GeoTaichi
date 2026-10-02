# Material unit-test matrix

This directory follows the material-test contract documented in
`tests/README.md`: initialization, state schema, zero/elastic/plastic branches,
history, and tangent finite differences are separate test nodes.

| Model family | Current contract coverage | Known production gaps |
| --- | --- | --- |
| Linear elastic | parameters, state, 2-D/3-D stress, stiffness, invariants | none exposed by this suite |
| Neo-Hookean | keyword initialization, energy, PK1, tangent FD, sound speed | none exposed by this suite |
| Mooney-Rivlin, Gent, Hencky, Hydrogel | initialization, energy/PK1 consistency, analytic tangent against an FD oracle, repeated-stretch Hencky limit, invalid deformation/extensibility, sound speed | none exposed by this suite |
| Newtonian, Bingham | state, pressure, viscosity/yield branches, mixed Voigt response, critical rate | Bingham pressure/history conventions need dedicated regression coverage |
| Elastic-perfectly-plastic, Mohr-Coulomb, Drucker-Prager | parameters, state, zero/elastic/plastic branches, history, elastic tangent FD | strict `xfail` nodes record validation, EPP return mapping, and Drucker-Prager branch-state defects |

The following production models still need the same full contract and must not
be considered unit-covered merely because an example script exists:

- Modified Cam Clay
- State-dependent Mohr-Coulomb
- NorSand
- SANISAND-MS
- rate-dependent and granular wrappers
- softening, random-field, and user-defined material adapters

The Bingham shear law performs its final yield check using only the first
three normal components. Consequently, the regression test expects a
pure-shear-only state to be cleared; this behavior should be reviewed
separately.

Run this partition with:

```bash
python tests/testing/run_partition.py materials
```
