# Excavation and truck loading in dry sand

Each `dry-sand/` directory holds the 800,000-point H200 run. Position
snapshots are saved every 0.1 s in `results/snapshots/`. The five native
checkpoints and contact histories are in `results/`, and the `vtu/`
collection contains the soil stresses and rigid surfaces from those five
checkpoints. The left-front videos use the repaired excavator linkage and
render at 1920 × 1440.

| Case | End time (s) | Bucket release (s) | Step (s) | Points in the bed at the end |
| --- | ---: | ---: | ---: | ---: |
| `explicit_cone` | 16.8 | 1.8 | 0.0001 | 3,973 |
| `explicit_trench` | 16.1 | 1.1 | 0.00005 | 17,652 |

Both use GIMP at 0.125 m, K₀ overburden initialization, a rough base,
`alphaPIC=0`, zero cohesion, and a zero mean tensile cutoff. The 24° cone
uses Halton sampling. The flat trench uses a 100 × 200 × 40 lattice and
normal-only side-wall velocity constraints. The bucket holds its start
pose until the measured settling criterion is satisfied. The cone uses grid
damping of 0.6 during this hold. `--dig-damping` controls grid damping
from bucket release through the end of the run and defaults to zero. The
bucket motion is slowed by a factor of 1.75 and interpolated with a clamped
cubic spline.

Run from this directory with a fresh output path.

```sh
python -u explicit_cone/excavator_cone_cycle.py /tmp/cone-dry-sand \
  --arch gpu --settle-damping 0.6
python -u explicit_cone/excavator_cone_cycle.py /tmp/cone-dry-sand-damped \
  --arch gpu --settle-damping 0.6 --dig-damping 0.25
python -u explicit_cone/excavator_cone_cycle.py /tmp/trench-dry-sand \
  --arch gpu --trench explicit_trench/inputs/trench.json --dt 5e-5
```

To resume, use the same case options, `--resume-from /path/to/results`, and
`--resume-index INDEX`. `export_vtu.py RUN OUTPUT` exports the native
checkpoints and validates the VTUs by reading them back.

For video replay set `case` to `cone` or `trench` and set `frames` to an
absolute output path. The video repeats the 10 source frames per playback
second to encode 30 fps.

```sh
assets="explicit_${case}/inputs"
blender -b "$assets/excavator-${case}.blend" --python-exit-code 1 \
  --python explicit_cone/render_cone.py -- \
  "explicit_${case}/dry-sand/results" "$frames" "$assets/trajectory.csv" --view left-front
ffmpeg -framerate 10 -i "$frames/%04d.png" -c:v libx264 -preset fast \
  -crf 17 -pix_fmt yuv420p -r 30 -movflags +faststart "excavator-${case}.mp4"
```
