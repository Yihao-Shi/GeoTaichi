# 15° screw-pile installation with ABD and Direct ULMPM

A 13 m screw pile follows seven clockwise turns while advancing 14.35 m into
a 16 × 16 × 16 m bed. The [explicit case](../../LevelSet/ScrewPile/README.md)
uses the same full mesh, 15° inclination, initial tip position, pitch and
70.5 s schedule. Four compliant spherical anchors track a smooth prescribed
motion after a 0.5 s hold.

| Profile | Pile triangles | Soil points | Grid (m) | Nominal step (s) |
| --- | ---: | ---: | ---: | ---: |
| `low` | 3,000 | 196 | 1.0 | 0.020 |
| `full` | 14,012 | 149,940 (42 × 42 × 85) | 0.5 | 0.008 |

```sh
python -u examples/mpdem/AffineBody/ScrewPile/screw_pile.py \
  --resolution low --arch cpu --stop-time 0.02 --output-dir /tmp/abd-pile-pilot
python -u examples/mpdem/AffineBody/ScrewPile/screw_pile.py \
  --resolution full --arch gpu --output-dir /tmp/abd-pile-full
```

The initial K₀ overburden stress is initialized by inverting the finite-strain
elastic Hencky law. The box uses a rough fixed base and normal-only side grid
constraints. Use `model-contract.json` for the full geometry, nominal soil
and steel parameters, and BarrierIPC settings.

`accepted-steps.csv` records advance, signed turns, axial drive force, drive
torque about the head, tracking error and contact gaps. Loads are measured
through the anchor springs and include pile weight and inertia. This is a
prescribed-motion case with illustrative material properties. Use
`run-config.json` to identify the backend and initial state and
`summary.json` to check whether the full installation completed. Confirm
contact capacity, nonlinear convergence and refinement on the target backend
before using the loads quantitatively.
