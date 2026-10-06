# Two-dimensional lid-driven cavity

The unit-square fluid domain has a lid speed of 1, density 1 and dynamic viscosity 0.01, giving Re=100. All four physical walls are no-slip. The lid is at y=1, with stationary corner/side-wall normal velocities. The example uses the ordinary incompressible MPM FDM engine: particle/grid transfer, explicit viscous prediction, incompressible pressure projection and particle update.

```sh
GEOTAICHI_ARCH=gpu GEOTAICHI_LID_2D_DT=0.004 GEOTAICHI_LID_2D_TIME=30 \
  python examples/mpm/IncompressibleFluid/lid_driven_cavity_2d/lid_driven_cavity_2d.py
python examples/mpm/IncompressibleFluid/lid_driven_cavity_2d/draw/compare_lid_driven_cavity_2d.py \
  examples/mpm/IncompressibleFluid/OutputData/lid_driven_cavity_2d
```

Default spacing is 1/64 and PPC is 3 per direction. The example selects float64 before importing the particle structs. `GEOTAICHI_LID_2D_DX`, `GEOTAICHI_LID_2D_DT`, `GEOTAICHI_LID_2D_TIME`, `GEOTAICHI_LID_2D_SAVE_INTERVAL` and `GEOTAICHI_LID_2D_SAVE_PATH` configure refinement studies. For explicit diffusion use νΔt(1/Δx²+1/Δy²)≤1/2, in addition to the advective CFL restriction.

`ghia_re100.csv` transcribes tables I/II of [Ghia et al. (1982)](https://doi.org/10.1016/0021-9991(82)90058-4): horizontal velocity on x=0.5 and vertical velocity on y=0.5. This is a high-accuracy **numerical Navier–Stokes reference**, not a closed-form analytic solution. The comparison reads saved MAC velocities and reports centerline errors normalized by lid speed, last-snapshot change, divergence, AIR-cell counts and particle conservation. A transient snapshot is not a steady benchmark pass.

The viscosity/no-slip kernel also has analytic startup/steady Couette checks in `tests/unit/mpm/test_incompressible_viscosity.py`. `fluid_wall_no_slip` is opt-in and requires `solid_sdf_cut_cell=True`; existing slip configurations keep their boundary convention.
