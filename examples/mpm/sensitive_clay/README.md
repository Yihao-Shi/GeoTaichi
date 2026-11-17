# Sensitive clay: uniform 2D DP/MC comparison

The current validation uses section 4.3 of Zhang et al., *GPU-accelerated
smoothed particle finite element method for large deformation analysis in
geomechanics*, Computers and Geotechnics 129 (2021), 103856, pp.10–12,
Figs.15–18. It runs **2D plane strain with no adaptive grid**. The paper's
x–z section maps to computational x–y with unit out-of-plane thickness.

Physical parameters remain E=1 MPa, nu=0.33, density=2000 kg/m³, peak/residual
cohesion=20/4 kPa and dc/d(equivalent plastic strain)=-50 kPa. The initial
section has a 20 m plateau, height 5 m, 45° face and toe at x=25 m.
The paper uses Tresca; this requested DP control uses `DruckerPrager` with
`dpType="Inscribed"` and zero friction/dilation. Then sqrt(J2)=c, so pure-shear
strength is exactly 20→4 kPa. The general DP and Tresca surfaces still differ.
DP accumulates `sqrt(2/3 * dplastic_strain:dplastic_strain)` in `epdstrain`.
For zero-dilation shear this equals the paper's deviatoric invariant, so the
residual endpoint is **0.32**. Hydrostatic tensile-cap flow contributes to
native DP history; the paper defines its invariant from deviatoric strain.
The old MC conversion 0.32/sqrt(2) must not be carried into this DP model.
`--material mc` selects the corrected native `MohrCoulomb` model on exactly
the same grid, geometry, timestep and boundaries. With zero friction its
shear surface is Tresca, and its native residual strain endpoint is
0.32/sqrt(2). Its smoothed deviatoric flow potential is retained; the recorded
scalar `strain` already uses the paper's normalization. `dp` remains the default.

The original bottom x=0..25 m is fixed, the left boundary is a horizontal
roller, and ground beyond the toe uses Coulomb friction, mu=0.3. The rigid
wall projection follows the impulse construction in paper §2.4, Eqs.(13)–(15).
Elastic dynamic relaxation first generates initial stress, using PIC=1 and
local damping 0.7 until maximum speed <0.002 m/s (maximum relaxation 10 s).
Relaxed particle state transfers directly between device fields, velocity
resets to zero and plastic history starts at zero. Failure then uses FLIP,
zero background damping and gravity 9.81 m/s². The paper does not state the
gravity magnitude or all relaxation controls; these are explicit assumptions.

The fixed grid is the original coarse 0.2 m spacing, with 2 particles per
axis, capacity 20,000, dt=0.0002 s and duration 10 s. Neither initialization
nor failure configures `AdaptiveGrid`; there is
no CLI switch to enable it. The timestep is checked against the native CFL
limit. The paper's SPFEM node count and adaptive timestep factor are not MPM
inputs. Local softening has no spatial regularisation, as in the paper.

Run from the repository root with the geotaichi conda environment:

```bash
conda run -n geotaichi python examples/mpm/sensitive_clay/sensitive_clay.py
conda run -n geotaichi python examples/mpm/sensitive_clay/sensitive_clay.py --material mc
conda run -n geotaichi python examples/mpm/sensitive_clay/draw/evaluate_sensitive_clay.py \
  examples/mpm/sensitive_clay/OutputData/paper_4_3_dp_uniform_h02_consistent_mu03
conda run -n geotaichi python examples/mpm/sensitive_clay/draw/compare_paper.py \
  examples/mpm/sensitive_clay/OutputData/paper_4_3_dp_uniform_h02_consistent_mu03
pvpython examples/mpm/sensitive_clay/draw/render_paraview.py \
  examples/mpm/sensitive_clay/OutputData/paper_4_3_dp_uniform_h02_consistent_mu03 --stride 10 --fps 10
```

`--duration` selects a short smoke run; paper comparison requires 10 s.
`--friction 0.1 --output <new-path>` selects the paper's low-friction contrast.
`--dt` and `--cell-size` allow uniform numerical checks within the alignment and
capacity guards. Use a new `--output` for each run. Outputs default to this
example's own `OutputData/` directory; existing snapshots are not overwritten.
Simulation, evaluation and rendering are separate scripts. Comparison plots
use strain 0–2 like Fig.17; ParaView MP4/GIF use the requested fixed **0–1**
range, with larger values only saturated in the display.

The native DP substepping path updates softened strength at the current local
plastic history. Signed negative yield values inside the elastic domain are
preserved so the elastic/plastic intersection search does not treat an
interior point as already yielding. Material-point checks compare the entire
elastic/linear-softening/residual shear curve with its analytical solution.
The classical closed-form DP return still freezes strength during its step;
this validation explicitly uses `stress_integration="SubStepping"`.

```bash
conda run -n geotaichi python -m pytest \
  tests/unit/physics_model/materials/test_dp_softening_consistency.py \
  tests/unit/mpm/test_grid_wall_friction.py \
  tests/unit/mpm/test_sensitive_clay_evaluation.py -q
```

The evaluator writes `evaluation.json`, `validation-evidence.json`,
`paper_comparison.json` and a ParaView PVD. The paper reports runout/retrogression
12.0/13.6 m for mu=0.3 and 21.0/18.4 m for mu=0.1 at 10 s. Runout is current
front edge minus original toe x=25; retrogression is original crest x=20 minus
the head-scarp top, measured by the upper envelope with height-drop thresholds
0.25/0.5/0.75 m. The threshold sensitivity and signed distance errors are
reported independently of numerical sanity checks. Passing mass/finite-state
checks does not establish paper agreement or spatial/time convergence.
The intended comparison is approximate agreement in the main failure stages,
block positions, runout/retrogression distances and friction trend; matching
individual pixels or every small shear band is not the acceptance criterion.

## Retained results

`OutputData/` retains exactly two main cases, both with ground friction mu=0.3:

- `paper_4_3_dp_uniform_h02_consistent_mu03`: corrected DP result.
- `paper_4_3_mc_uniform_h02_consistent_mu03`: corrected MC result on the same grid.

Low-friction controls and superseded MC/adaptive trials were removed. The
cleanup record is `OutputData/cleanup.json`. Final particle snapshots, the
ParaView PVD and its `paper_vtks`, evaluation reports and finished animations
are retained. Duplicate native VTKs and temporary render frames are removed
after postprocessing completes.

The 0.2 m uniform DP result at 10 s gives runout/retrogression
**14.41/11.60 m**, compared with the paper's 12.0/13.6 m. It has 11,225
particles, zero mass drift and no refinement; CPU/f64 wall time including
initialization was about 4.05 minutes. This is a measured 10 s state, without
a claim of a fully resting deposit or mesh convergence. Each retained case
contains its own `evaluation.json` and `comparison.md`. The corrected MC
result gives **13.64/7.00 m**: runout is within 13.7% of the paper, while
retrogression remains 48.5% smaller. Its numerical checks pass, but this does
not establish agreement in the progressive failure process. The MC run took
about 5.60 minutes including initialization; both ParaView animations are complete.
See [the numerical comparison and artifacts](OutputData/paper_4_3_dp_uniform_comparison.md).
