# Developer tools

This directory contains explicit developer workflows that are useful but are
not correctness tests and are never run during pytest collection.

- `diagnostics/`: targeted numerical probes and viewers. A diagnostic must
  expose a callable entry point, use deterministic inputs where possible, and
  perform work only under `if __name__ == "__main__"`.
- `derivations/`: symbolic/reference derivations used to check or generate
  formulas. Install the optional dependencies with
  `pip install -e '.[gen]'`.
- `benchmarks/`: standalone timing and memory studies. Correctness belongs in
  `tests/`; benchmark output belongs in a caller-selected directory outside
  the repository.
  `benchmarks/igampm/benchmark_wavy_contact.py` compares span candidate screening
  and CCD motion-bound screening on a saved wavy-plate checkpoint. It alternates
  warmed baseline/optimized kernels, checks active-distance and motion-screen
  equivalence, and reports kernel timings rather than whole-step speedups.
  `benchmarks/fempm/benchmark_fixed_assembly.py` compares raw reduction with
  permanent-slot FEM material assembly, including matrix equivalence, alternating
  warmed samples, and explicit CPU/CUDA selection.
  `benchmarks/igampm/benchmark_fixed_assembly.py` times cached fixed-slot IGA
  solid assembly and checks it against the raw matrix; run it in separate source
  snapshots for a kernel comparison.
  `benchmarks/linear_solver/benchmark_pcg_fusion.py` compares split and fused
  HashTriplet PCG launches on identical SPD block systems, with independent true
  residual checks, alternating samples, and CPU/CUDA/Metal selection.
- `postprocessing/`: parameterized command-line transforms and data summaries
  for existing simulation output; input and output paths must be explicit.

Do not add a script here merely to avoid writing an assertion. If a small,
deterministic oracle exists, put it under `tests/unit`; keep the tool only when
interactive output, symbolic code generation, or performance instrumentation
is itself useful.
