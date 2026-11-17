# Physics-Grounded Agent Validation Loop

## Purpose

The model agent is a bounded engineering loop, not a one-shot script generator.
It must preserve the user's physical problem while moving one candidate through
capability discovery, static review, execution, evidence scoring, and repair.
The canonical scoring rules are `physics-validation-rubric.json`.

## State machine

```text
natural-language intent
  -> capability query and source trace
  -> completed model contract
  -> candidate script
  -> static inspection
  -> reduced execution
  -> physical evidence score
  -> production execution
  -> validated handoff
```

`geotaichi_review_model` returns the current state:

- `repair_static_model`: fix lifecycle/API errors before execution;
- `resolve_model_contract`: resolve placeholders or physical choices;
- `collect_physics_evidence`: run the reduced case and measure missing checks;
- `diagnose_physics_failure`: repair the first failed invariant or hard gate;
- `run_production_validation`: repeat unchanged checks at production settings;
- `prepare_handoff`: freeze artifacts and evidence.

Use `geotaichi_score_physics` when only contract/evidence scoring is required.
Without MCP, run:

```bash
python agent/geotaichi-model-builder/scripts/score_physics.py \
  model-contract.json --evidence validation-evidence.json
```

## Evidence contract

The validation evidence JSON records effective execution facts and physical
checks. A pass requires an artifact, command, or derivation; an unexplained
boolean does not count as evidence.

Expected values and tolerances are selected in the model contract before the
run. `validation.invariants` declares `kind`, `expectation`, `tolerance`, and
`basis` for every scored invariant. The evidence must repeat those criteria
exactly; the scorer rejects an undeclared check or any post-run criterion
change.

```json
{
  "schema_version": 1,
  "task_status": "completed",
  "finite_state": true,
  "capacity_overflow": false,
  "solver_converged": "not_applicable",
  "timestep_consistent": true,
  "command": "python reduced_model.py",
  "backend": "cuda",
  "precision": "float64",
  "production_parameters": false,
  "checks": [
    {
      "name": "peak normal force",
      "kind": "contract_observable",
      "observed": 10.2,
      "expected": 10.0,
      "tolerance": {"relative": 0.05},
      "evidence": "output/contact-force.csv"
    }
  ]
}
```

Numeric checks use `observed`, a scalar `expected` or two-value expected range,
and a non-negative scalar tolerance or `{absolute, relative}` tolerance. A
non-numeric check uses `status=pass|fail|skip|not_run` plus evidence. Mark an
exploratory check `required=false`; required failures are hard gates.

SolverJob workers look for a public `validation_evidence` dictionary in the
entry-script namespace, then `output/validation-evidence.json`. When a staged
model contract exists, the terminal task writes `physics-score.json` even if
evidence is incomplete. Missing evidence is reported as
`insufficient_evidence`, not confused with physical failure.

## Bounded repair

Use at most three repair iterations for one candidate. In each iteration:

1. preserve the exact contract, evidence inputs, and first causal failure;
2. classify the failure as schema/API, capacity, units/scale, timestep,
   convergence, backend, or physical invariant;
3. make the smallest source-backed repair in that class;
4. rerun static inspection and the same reduced evidence checks;
5. compare the new score and retain both candidate records.

Stop and request a physical choice when the repair would change material law,
friction, loading, drainage, geometry, units, or coordinate assumption. Never
hide NaN/non-convergence, drop entities on overflow, or widen an acceptance
tolerance without a derivation. A lower score with new evidence is still more
informative than a high score obtained by deleting a check.

## FEM soft-particle--rigid LSDEM profile

For `fedem`, the rubric requires all of the following evidence families:

1. the contract observable, such as Hertz peak force, rebound, or displacement;
2. bounded signed contact gap or normalized penetration;
3. at least one exchange invariant: rigid-SDF nodal action/reaction, total
   momentum, energy/work balance, force/moment balance, or a coupled benchmark;
4. finite state, capacity integrity, timestep consistency, and recorded runtime.

Exercise `LinkedCell` and `BVH` in separate capability tests when claiming both.
Do not average their results into one check: search equivalence, contact physics,
and performance are separate claims.
