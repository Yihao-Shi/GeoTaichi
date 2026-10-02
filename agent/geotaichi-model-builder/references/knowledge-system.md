# GeoTaichi Agent Knowledge System

## Contents

1. Purpose
2. Knowledge layers
3. Browse and query protocol
4. Evidence and confidence
5. Capability record shape
6. Maintenance

## 1. Purpose

Organize GeoTaichi knowledge so an agent can discover supported behavior
without loading the whole repository or guessing from a related method. The
design separates known-path navigation from keyword search, then requires a
source trace before a capability is used.

## 2. Knowledge layers

Use three distinct layers:

1. **Facade/API layer**: constructors, public methods, signatures, lifecycle,
   and dictionary keys from `geotaichi/` and `src/*/main*.py`.
2. **Reference layer**: material/contact models, accepted properties,
   compatibility rules, theory, capacities, output, and restart semantics.
3. **Execution layer**: syntax/configuration checks, reduced runs, production
   tasks, monitoring, interruption, and result validation.

The canonical knowledge content lives in `agent/geotaichi-model-builder`.
`agent/geotaichi-mcp/src/geotaichi_mcp/knowledge` exposes browse/query and
inspection without owning a second corpus, while the sibling `execution`
package owns only process and task lifecycle.

Do not substitute one layer for another. A theory section does not prove that a
facade exposes the switch; a public key does not prove that the selected
backend implements it; a successful run does not prove the physics.

## 3. Browse and query protocol

Use `query_capabilities.py browse` like directory navigation:

- no path: list facade categories;
- `<module>`: show module overview;
- `<module>/methods`: list public methods;
- `<module>/methods/<name>`: show one signature and source;
- `<module>/keys`: list configuration keys;
- `<module>/keys/<name>`: show every consuming source location;
- `<module>/examples`: list nearby examples.

Use `query_capabilities.py query <terms>` like search:

- exact canonical names rank first;
- mechanics phrases, plurals, hyphens, and a bounded Chinese/English synonym
  vocabulary are normalized before token matching;
- category routing terms rank coupled solver owners even when the indexed API
  method name does not repeat the user's physical wording;
- token and substring matches rank next, while fuzzy name matching is limited
  to short queries;
- results return paths that should be browsed for full detail.

Query results include `matched_terms` and `match_reason`. These explain routing
but are not capability evidence; browse and source-trace the returned path.

If query returns no entries, broaden the physical term, search a related owner,
or inspect the helper API index. Do not invent a key from the user's wording.

Recommended search sequence:

```text
physical concept
  -> query capability index
  -> browse candidate facade/key
  -> inspect nearby examples
  -> trace facade -> validation -> consuming engine/model
  -> inspect maintained tests
  -> record evidence ledger
```

## 4. Evidence and confidence

Assign one evidence level to every non-default choice:

| Level | Required evidence | Permitted use |
|---|---|---|
| `verified-runtime` | Current source trace plus successful relevant run | Production candidate, subject to scale/backend limits |
| `verified-source` | Current source trace and maintained focused test | Generate and smoke-test |
| `documented` | Helper docs or maintained example only | Discovery candidate; source trace required before final script |
| `assumed` | Inference or user wording only | Contract placeholder; never silently emit as final API |

Confidence is about evidence, not familiarity. A common numerical method still
requires proof that this repository's selected branch exposes and implements
it.

## 5. Capability record shape

`build_capability_index.py` generates `capability-index.json` with:

- category name and facade class;
- facade source path;
- public method name, signature, and line;
- configuration key and consuming source locations, including both `DictIO`
  readers and explicit memory-mapping `get` calls;
- nearby examples;
- related helper documents and reference files;
- quick-reference maps from method/key names to browse paths.

The generated category set includes `mpm`, `dem`, `mpdem`, `cfdem`, `fem`,
`fedem`, `fempm`, `iga`, and `igampm`. FEM contact capability records must distinguish the
user-selected linked-cell/BVH spatial backend from the common collision-culling
stage that performs exclusions, exact-distance or CCD pruning, and compaction.

The index is a locator, not the source of truth. Regenerate it from the current
working tree and use the recorded paths to inspect source.

## 6. Maintenance

After a public API change:

1. regenerate `references/capability-index.json`;
2. run `audit_api_docs.py`;
3. update the relevant workflow/compatibility reference;
4. update `docs/helper` API and linked theory;
5. add or update a maintained test and example when user-visible;
6. run `scripts/self_test.py` and the helper LaTeX build.

Keep canonical names in generated scripts. Record accepted legacy aliases in
compatibility/reference text without making them the default style.
