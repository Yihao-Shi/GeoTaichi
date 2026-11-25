# 15° screw-pile installation with explicit MPM and LSDEM

The 13 m pile turns clockwise seven times and advances 14.35 m along its
axis into a 16 × 16 × 16 m bed. All profiles use Lightweight transfers
without BlockScan, a regular particle lattice, K₀ overburden stress, and
normal constraints at the base and sides. The Drucker–Prager soil uses
250 Pa cohesion and `alphaPIC=0.1`. All profiles use GIMP, a fixed
`1e-4 s` step, and an initial 0.5 s hold. The `h200` profile completes
the same seven turns and 14.35 m travel in 18 s instead of 70.5 s by
driving the pile four times faster. It is a separate loading rate,
not a rate-equivalent reproduction of the slower run.

| Profile | Particles | Grid spacing (m) | Maximum initial points per cell |
| --- | ---: | ---: | ---: |
| `full` | 327,680 | 0.25 | 2 |
| `dense` | 3,193,344 | 0.20 | 8 |
| `h200` | 201,230,056 | 16 / 293 | 8 |

```sh
python -u examples/mpdem/LevelSet/screw_pile/screw_pile_explicit.py \
  --resolution h200 --arch gpu --positions-only \
  --output-dir /dev/shm/screw-pile-h200
```

On an H200, 200 steps with 327,680 points and `--initial-overlap 0.12`
gave a final pile force of 6,398.705 N in all three modes.

| MPM mode | Peak GPU memory (MiB) | Warm step (ms) |
| --- | ---: | ---: |
| Normal, dense grid | 1,429 | 2.67 |
| Lightweight, dense grid | 1,109 | 2.34 |
| Lightweight, BlockScan grid | 1,109 | 3.33 |

Earlier 343-million-point contact probes on CUDA peaked at about 128,700 MiB. With
`--initial-overlap 0.04 --stop-time 0.006 --monitor-only`, both `1e-4 s`
and `2e-4 s` finished, but the larger step changed the final pile force
from 223 N to 97 N and the peak particle speed at that time from 1.47 m/s
to 1.94 m/s. This is a stationary contact transient, not a validation of
installation at a larger step. `3e-4 s` was above the initial contact
timestep limit for that finer grid.

The command above writes 37 NPZ frames (every 0.5 s, including both
endpoints), about 42 GiB total. Each contains every particle position,
quantized for rendering only (maximum error below 0.25 mm per axis).
It does not write VTU. The renderer reads NPZ directly and still supports
legacy NPY frames.

`latest.npz` is a separate FP64 checkpoint: particle positions, velocities,
stress, volumes, plastic state, contact history and the completed step.
It is saved after the first step, every 0.25 simulation seconds thereafter,
and at the final step. A complete new file atomically replaces the old one;
budget space for both during that write. Re-run the same command with
`--resume` to continue an interrupted run. Resume rebuilds the contact
search and resets its Verlet displacement; it is not a bitwise replay of
the old search schedule.

Use `--checkpoint /persistent/path/latest.npz` on both commands to place
the checkpoint on a persistent volume. Outputs kept in `/dev/shm` only
survive process exits, not machine restarts. Store the output directory
on persistent storage as well to retain the full replay after reboot.

`dense/results/` and `dense/vtu/` contain the earlier H200 run on the
8 × 8 × 16 m bed with 798,336 points. `dense/vtu/screw_pile.pvd` opens the
seven native save times including soil stress and rigid geometry.
`python export_pile_snapshots.py dense/results NEW_DIRECTORY` exports all
saved replay frames.

The 1520 × 1640 renderer draws every saved point, colors the sand by depth,
and uses two depth-tested layers for 22% opacity in the camera-facing half.
Set `frames` to an absolute output path and run from this directory.

```sh
blender -b --python-exit-code 1 --python render_pile_preview.py -- dense/results "$frames"
ffmpeg -framerate 10 -i "$frames/section/%04d.png" \
  -framerate 10 -i "$frames/full/%04d.png" \
  -filter_complex '[0:v][1:v]blend=all_expr=0.78*A+0.22*B' \
  -c:v libx264 -preset fast -crf 17 -pix_fmt yuv420p -r 30 -movflags +faststart screw-pile.mp4
```
