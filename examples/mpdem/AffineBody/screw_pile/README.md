# 15° screw-pile installation

`screw_pile.py` couples a 13 m ABD screw pile to a 5.6 × 5.6 × 3.2 m dry
Drucker–Prager MPM soil bed with frictional BarrierIPC. The bed has open sides
and a fixed frictional floor. The pile starts with a 20 mm mesh clearance
above the nominal soil surface. `inputs/` contains the original mesh and
inclination data from the geometry demo, and a decimated mesh for the pilot.
Both meshes retain five right-handed flights and a 2.05 m pitch.

| Profile | Pile triangles | Soil particles | Grid (m) | Nominal step (s) |
| --- | ---: | ---: | ---: | ---: |
| `low` | 3,000 | 196 | 0.8 | 0.020 |
| `full` | 14,012 | 18,032 | 0.25 | 0.008 |

```sh
python -u examples/mpdem/AffineBody/screw_pile/screw_pile.py \
  --resolution low --arch cpu --output-dir /tmp/screw-pile-low
python -u examples/mpdem/AffineBody/screw_pile/screw_pile.py \
  --resolution full --arch gpu --output-dir /tmp/screw-pile-full
```

After 0.5 s of settling, the drive advances along the pile axis, 15° from
vertical, and rotates clockwise as viewed from the head. One turn takes
10 s with a smooth start and stop. The mean speed is 6 rpm and the peak is
9 rpm. The axial advance per turn equals one pitch (`AR = 1`), giving 2.05 m
of commanded axial advance and about 1.98 m of vertical descent. Four
finite-stiffness world anchors track the pose at each attempted step.

`accepted-steps.csv` records the commanded and actual advance and rotation,
positive-in-insertion axial drive force, and positive-in-spin drive torque
about the head. Force and torque come from the anchor springs and include
pile weight and inertia in the drive balance. To study a machine whose
penetration rate responds to load, use its available axial thrust and motor
torque or torque–speed curve as controls. The pose-controlled case records
those loads as outputs. The soil is initially stress-free and the mesh uses
solid steel-like inertia. These loads are illustrative until the initial
stress, soil law, and pile inertia are calibrated.

`--stop-time` supports shorter probes. `summary.json` marks the full
installation only after reaching 10.5 s. `run-config.json` records the
effective backend, precision, and solver inputs. `failure.json` records the
last recorded time and the solver error if a run aborts. A mixed gap of `inf` in the
CSV denotes an empty contact diagnostic. The 196-point profile is for
checking the drive and coupling. The full profile gives finer soil sampling
and retains all authored surface details. Check capacity and convergence on
the target backend before using installation loads quantitatively.
