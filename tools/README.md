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
- `postprocessing/`: parameterized command-line transforms and data summaries
  for existing simulation output; input and output paths must be explicit.

Do not add a script here merely to avoid writing an assertion. If a small,
deterministic oracle exists, put it under `tests/unit`; keep the tool only when
interactive output, symbolic code generation, or performance instrumentation
is itself useful.
