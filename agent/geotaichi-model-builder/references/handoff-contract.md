# Model Handoff Contract

## Contents

1. Status
2. Required fields
3. Success shape
4. Blocked shape

## 1. Status

Use exactly one:

- `validated`: all requested gates, including the required physical observable,
  passed at the intended production settings;
- `partially_validated`: implementation is complete but production/backend/
  scale validation remains;
- `blocked`: a required physical choice, authority, unavailable backend, or
  repeated external failure prevents meaningful progress.

Do not use `validated` for syntax-only or smoke-only results.

## 2. Required fields

- `status`
- `summary`
- `artifacts`
- `model_contract`
- `assumptions`
- `unresolved`
- `evidence_ledger`
- `validation`
- `physics_score`
- `runtime`
- `warnings`
- `next_action`

Keep machine-readable facts separate from narrative. Avoid duplicating full
logs in the handoff; link the artifact and quote only the causal line.

## 3. Success shape

```json
{
  "status": "validated",
  "summary": "What model was built and what result passed",
  "artifacts": {
    "entry_script": "relative/path.py",
    "contract": "relative/contract.json",
    "outputs": ["path"]
  },
  "model_contract": {"module": "mpm", "dimension": 2},
  "assumptions": [],
  "unresolved": [],
  "evidence_ledger": [
    {"feature": "...", "level": "verified-runtime", "source": "..."}
  ],
  "validation": [
    {"gate": "syntax", "status": "pass", "command": "..."},
    {"gate": "numerical", "status": "pass", "observable": "..."}
  ],
  "physics_score": {"score": 100.0, "decision": "accept"},
  "runtime": {"backend": "cuda", "precision": "float64"},
  "warnings": [],
  "next_action": null
}
```

## 4. Blocked shape

```json
{
  "status": "blocked",
  "summary": "Concrete blocker",
  "artifacts": {},
  "model_contract": {},
  "assumptions": [],
  "unresolved": ["Physical choice or missing authority"],
  "evidence_ledger": [],
  "validation": [
    {"gate": "configuration", "status": "fail", "first_error": "..."}
  ],
  "physics_score": {"score": 0.0, "decision": "insufficient_evidence"},
  "runtime": {"backend": "unavailable", "precision": "float64"},
  "warnings": [],
  "next_action": "One specific user or environment action"
}
```

When returning prose to a user, lead with outcome, list the few important
artifacts, state validation truthfully, and make the next action explicit.
