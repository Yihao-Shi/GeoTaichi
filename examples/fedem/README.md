# FEM–DEM, FEM–LSDEM, and FEM–Affine Body Dynamics Contact Examples

These examples cover deformable grains, rigid level-set grains, membranes, and affine bodies. The [FEM–DEM coupling theory log](../../src/fedem/README.md#fedem-coupling-theory-log) gives the contact geometry, force exchange, and coupled IPC equations; each case below links the formulation to a concrete simulation.

Each case family has its own directory containing its entry point, local
support files, assets, and output.  Entry points construct the geometry,
materials, boundary conditions and contact models, advance the calculation,
and write their own results.  The text/image mesh-settling study is
intentionally not included.  All quantities use SI units.

Each case family owns its script, assets, and `OutputData` directory:

- `HertzContact/hertz_contact.py`: one FEM soft sphere against a fixed LSDEM plane.
- `HertzContact/hertz_mesh_study.py`: the four locally refined meshes used for the Hertz
  pressure and force--contact-radius comparisons.
- `ShapeCollision/shape_collision.py`: sphere--sphere, sphere--cube, and cube--cube impact
  for FEM--FEM and FEM--LSDEM contact.
- `ShapeCollision/collision_timestep_convergence.py`: separated-state energy differences for
  the same six impacts at four time steps.
- `InclinedPlaneFriction/inclined_plane_friction.py`: FEM sphere and cube motion on a 45-degree
  LSDEM plane for friction coefficients 0, 0.2, and 0.4.
- `MixedFunnel/mixed_funnel.py`: 400 deformable and 400 rigid grains settling behind a
  gate and discharging into a separate catcher.
- `IsotropicCompaction/isotropic_compaction.py`: 125-particle all-soft and 50% soft packings under
  six-wall isotropic compression.
- `ExplicitLevelSetSoftParticle/`, `ExplicitSphereMembrane/`,
  `ImplicitAffineIPCMembrane/`, and `ImplicitAffineIPCSoftParticle/`: compact
  explicit or implicit coupling examples.
- `FEMAffineDeposition/`, `FEMAffineTriaxial/`, and
  `FEMAffineMixedTriaxial25/`: AffineBody deposition and compression examples.

Run a lightweight model check before a production calculation:

```bash
python examples/fedem/HertzContact/hertz_mesh_study.py --preflight
python examples/fedem/ShapeCollision/shape_collision.py --preflight
python examples/fedem/ShapeCollision/collision_timestep_convergence.py --preflight
python examples/fedem/InclinedPlaneFriction/inclined_plane_friction.py --preflight --friction 0.2
python examples/fedem/MixedFunnel/mixed_funnel.py --preflight
python examples/fedem/IsotropicCompaction/isotropic_compaction.py --preflight --soft-percent 50
```

Remove `--preflight` to execute a complete physical case.  To reproduce the
friction matrix, run `InclinedPlaneFriction/inclined_plane_friction.py` for friction coefficients
0, 0.2, and 0.4.  To reproduce the compression comparison, run
`IsotropicCompaction/isotropic_compaction.py` with `--soft-percent 100` and 50.  The example
selects the corresponding paper loading velocities of 0.02 and 0.005 m/s.
Results are written below the selected case's `OutputData` directory.  Each
production example also writes `config.json`, `metrics.json`, histories, native
VTU/NPZ output, and timing metadata.
