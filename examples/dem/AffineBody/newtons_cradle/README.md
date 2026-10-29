# ABD Newton's cradle

Five equal ABD balls hang from independent world-fixed revolute hinges. The
first ball starts at 20 degrees and is released from rest. All pairs use the
original triangle PT/EE IPC, BVH, CCD, and device Newton/PCG solve. There is no
velocity reset, added damping, or separate impact solver.

Each closed sphere has 2562 vertices and 5120 triangles, radius 0.05 m,
suspension length 0.5 m, density 7800 kg/m³, and initial surface gap 2.2 mm.
ABD rigidity is 1e9 Pa; pivot position/axis penalties are 1e7. Contact activation
distance is 2 mm and barrier stiffness is 2e16. Friction and physical damping
are zero. These finite penalties are a demonstration, not calibrated steel.

The main README shows the first five impacts, over 0–3.605 s, from the completed
12 s CUDA/f64 run with eight measured returns: average-acceleration Newmark
`[0.5, 0.25, 0.5]` and `dt=1e-4 s`. The image SHA-256 is
`7057505b14a6e8f5a2986984255cbde6233ee7bcb310f2949e3fd605f1b2ee10`.
These are the example defaults. `--newmark 1 0.5 1` remains an explicit
BackwardEuler comparison through the same solver path.

The validated rigidity remains 1 GPa. Increasing the orthogonality penalty to
2 TPa requires separate temporal energy validation: tightening Newton/PCG alone
did not remove energy growth at `dt=1e-4 s`. A short `dt=5e-5 s` comparison
passed the first-impact checks; it does not establish acceptance over 12 s.

```bash
python examples/dem/AffineBody/newtons_cradle/newtons_cradle.py \
  --arch cpu --duration 0.002 --save-interval 0.001 --smoke \
  --output-dir /tmp/abd-cradle-smoke

python examples/dem/AffineBody/newtons_cradle/newtons_cradle.py \
  --arch gpu --duration 12 --output-dir /tmp/abd-cradle-formal
```

The CUDA/f64 production run targets eight visible returns over 12 s; seven or
eight meets the requested duration. The reference period is about 1.43 s.
Count actual returns, not elapsed time alone. Newton permits up to 1000
iterations with unchanged velocity tolerance `1e-7` and PCG tolerance `1e-14`.
There is no hidden timestep retry or tolerance relaxation.

The [model contract](model-contract.json) records these diagnostics:

- Reference horizontal transfer: striker and middle-ball speeds below 5% of
  the incoming reference speed after the outgoing ball advances 0.01 m.
  Vertical vibration is excluded from this motion criterion by user request.
- At least seven measured returns to the raised side, with excursions above
  0.01 m and a half-period separation to exclude small fast wobble.
- Pivot drift at most 0.2 mm and plane drift at most 10 micrometers.
- No energy above the initial budget beyond a relative 1e-6 allowance: kinetic
  plus gravity in saved frames, and complete conservative energy when geometry
  certifies zero IPC energy. Dissipation is permitted.

Finite rigidity and hinge penalties produce vertical vibration even before
impact; ignoring it in the motion criterion does not remove it from kinetic
energy. CCD and implicit convergence do not alone guarantee nonlinear energy
conservation. The checks do not assert monotonic energy at every substep.

A failed physical check preserves results and marks `passed=false`. In the
remote production queue, acceptance failure or solver failure is recorded and
the next independent case still runs. An unconverged numerical step is never
accepted to keep a case moving.

Saved NPZ/VTU and `trajectory.csv` contain native motion and velocity history.
`GraphicAffineBody*.vtu` and `PendulumRods*.vtu` show balls and suspension arms.
`validation_summary.json` reports measured returns, horizontal residuals and
energy checks. The 12 s acceptance does not establish temporal or mesh convergence.
