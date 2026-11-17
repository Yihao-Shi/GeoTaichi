# Newton cloth regressions

These scripts mirror the Newton cloth examples:

- `newton_cloth_bending.py`: initially curved cloth, dihedral bending, gravity,
  ground and self BarrierIPC.
- `newton_cloth_hanging.py`: planar cloth with its left strip fixed, gravity,
  ground and self BarrierIPC; its default 0.1 m spacing gives a 6.3×3.1 m
  64×32 sheet that can reach the ground from the 4 m start height.
- `implicit_cloth_twist.py`: opposite edge rotations with PT/EE self-contact;
  its default 300 steps at 60 Hz cover 5 seconds.
- `newton_cloth_rollers.py`: rolled cloth with a rotating inner seam and two
  static cylindrical SDF-spring frames. The latter is an explicit GeoTaichi
  approximation because the current FEM cloth contact API does not have moving
  rigid-cylinder primitives.

Example:

```bash
python examples/fem/cloth/newton_cloth_bending/newton_cloth_bending.py --arch=gpu
python examples/fem/cloth/newton_cloth_hanging/newton_cloth_hanging.py --arch=gpu
python examples/fem/cloth/implicit_cloth_twist/implicit_cloth_twist.py --arch=gpu
python examples/fem/cloth/newton_cloth_rollers/newton_cloth_rollers.py --arch=gpu
```

The IPC examples use `--contact-model=BarrierIPC`.

The bending and hanging defaults run 100 steps at 60 Hz (1.6667 seconds).
The static-SDF rollers approximation now uses the same 100-frame/60 Hz
physical duration; twist explicitly runs 300 steps (5 seconds).
The defaults use local PSD membrane/bending/contact blocks with device PCG,
250 Newton iterations, a `1e-2` correction-velocity tolerance, and a relative
Krylov tolerance.  This is the robust
full-time configuration; the exact indefinite tangent remains available with
`--no-project-pd --no-project-bending-pd --linear-solver=BiCGSTAB` for
diagnostics.  Bending and hanging map their thickness-aware plate stiffness to
an effective hinge modulus near 10, matching the corresponding Newton setup.

The checked-in `OutputData/newton_cloth_*_3step/vtks/` folders are only old
smoke evidence and must not be treated as full-duration validation.
