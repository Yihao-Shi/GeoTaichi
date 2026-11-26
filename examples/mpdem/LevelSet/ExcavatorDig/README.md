# Excavation and truck loading in dry sand

Each `dry-sand/` directory holds the 800,000-point H200 run. Position
snapshots are saved every 0.1 s in `results/snapshots/`. The five native
checkpoints and contact histories are in `results/`, and the `vtu/`
collection contains the soil stresses and rigid surfaces from those five
checkpoints. The left-front videos render at 1920 × 1440. The replay
poses a single stick and a short bucket four-bar around the recorded bucket
trajectory and draws sand at the recorded material-point centers. The visible
truck bed uses the same wall and floor geometry as the contact mesh in the VTUs.
The visible bucket shell uses the VTU contact mesh. Five decorative teeth
sit on the outside of its continuous cutting edge and extend 4 cm past it.
The teeth are added only for replay.

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
cubic spline. `--sdf-spacing` sets the rigid contact grid spacing in metres
for both the bucket and the truck bed. It defaults to 0.04 m. The contact
grid must resolve the wall thickness, and the linear penalty law permits
finite overlap even with a refined SDF.

Run from this directory with a fresh output path.

```sh
python -u explicit_cone/excavator_cone_cycle.py /tmp/cone-dry-sand \
  --arch gpu --settle-damping 0.6
python -u explicit_cone/excavator_cone_cycle.py /tmp/cone-dry-sand-damped \
  --arch gpu --settle-damping 0.6 --dig-damping 0.25
python -u explicit_cone/excavator_cone_cycle.py /tmp/trench-dry-sand \
  --arch gpu --trench explicit_trench/inputs/trench.json --dt 5e-5
```

For a single 200-million-point cone across eight CUDA GPUs, run from this
same directory with PyTorch CUDA/NCCL installed.

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc-per-node=8 \
  explicit_cone/excavator_cone_cycle.py /path/to/cone-200m \
  --arch gpu --particles 200000000 --dt 8e-5 --settle-damping 0.6 --dig-damping 0.25
```

The fixed step of 80 μs is below the 200-million-point case’s initial
CFL limit of about 85 μs. Lightweight MPM keeps 25 million particles and a full
grid on each GPU.
Each rank seeds a distinct slice of one scrambled Sobol sequence. The global
settling check uses the pooled RMS speed and the fraction of all points below
the p95 speed limit. The progress CSV reports the rank-local p95 speed.
Each `rank-NN/` directory holds its own particle and contact output. The
run-level `latest-checkpoint.json` is updated after all ranks have written
the same checkpoint, at 0 s, 0.01 s, every 2 s thereafter, and at the end.
Allow about 1.6 TB for a run that uses the entire settling window, including
the full-resolution replay snapshots.

To resume, use the same launcher, rank count and case options. Pass a new output
path and `--resume-from /path/to/results --resume-index INDEX`. Read `INDEX` from
`latest-checkpoint.json` for the latest complete distributed checkpoint. Each
rank reads the corresponding `rank-NN/` directory. The script restores the
particle and friction state and reconstructs the bucket and truck from the
prescribed trajectory. Old checkpoints use the saved time and original fixed
step to recover the step count, and the rank’s progress CSV and snapshots to
recover the settling state and loaded cohort. For the fourfold contact
stiffness trial, include `--contact-stiffness-scale 4` with the original
`--dt 4e-5` and `--sdf-spacing 0.01`.

The single-process run accepts the same resume flags. `export_vtu.py RUN OUTPUT`
exports the native checkpoints and validates the VTUs by reading them back.

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

The realistic damped-cone replay uses Poly Haven's [Quarry 02](https://polyhaven.com/a/quarry_02)
and [Gravel Ground 01](https://polyhaven.com/a/gravel_ground_01) ([CC0](https://polyhaven.com/license)).
The machine and lighting references are the [Volvo EC230](https://www.volvoce.com/en-us/products/excavators/ec230/),
[Volvo quick couplers](https://www.volvoce.com/en-us/products/excavator-attachments/excavator-quick-couplers/),
and the [CM Labs excavator simulator](https://cm-labs.com/en/simulation-curriculum/intellia-excavator-simulation-curriculum/).

```sh
case_dir="explicit_cone/dry-sand-damped"
blender -b explicit_cone/inputs/excavator-cone.blend --python-exit-code 1 \
  --python explicit_cone/render_cone.py -- \
  "$case_dir/results" "$frames" "$case_dir/results/provenance/trajectory.csv" \
  --view left-front --textures "$case_dir/render-textures"
```

Use an empty frames directory for a new render. Append `--check-linkage` to
check the rendered stick, linkage, bucket and truck meshes at each snapshot.
