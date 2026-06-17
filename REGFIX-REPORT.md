# Regression fix: `likelihoodof(relabel'd, record)` scoring

Branch `fix-likelihoodof-relabel-record` (off `main`, includes PR #38).

## The regression

PR #38 ("type relabel transparently so iid accepts a relabel'd measure")
added two coupled changes:

- `inferRelabel` in `typeinfer.ts` (relabel is type-kind-transparent), and
- an `isMeasureLikeArg` / `MEASURE_PRODUCING` guard in `lift.ts`
  `inlineRelabel`, so that `relabel(<measure>, [...])` is KEPT as a `relabel`
  node instead of being rewritten to `record(<axis> = <measure>)`.

That fixed `iid(relabel(M), n)`, but BROKE the single-observation,
record-keyed likelihood:

```
gauss_x = relabel(Normal(mu=mu, sigma=sigma), ["x"])
obs     = likelihoodof(gauss_x, record(x = 1.27))
```

Before #38, `inlineRelabel` rewrote the body to `record(x = Normal(…))`, so
the record observation `{x: 1.27}` walked field-by-field: the `record` density
handler consumed field `x` (the scalar `1.27`) and handed that scalar to the
Normal leaf. After #38 the body stays `relabel(Normal(…), ["x"])`; the density
walk's relabel-peel (`density.ts walkAcc`) discarded the axis labels and
recursed on the inner Normal with the WHOLE record object, which Normal's
scalar leaf cannot consume.

## Root cause (pinned)

`density.ts walkAcc`, relabel branch:

```js
if (op === 'relabel' && …) return walkAcc(ir.args[0], value, …);
```

passes `value` straight through. With `value = {x: 1.27}` and inner
`Normal`, the walk reaches `walkLeaf → consumeScalar({x:1.27})` →
**"density: cannot consume scalar from value of type object"** (density.ts:142).

That worker error was then MASKED: `matScore`'s `sendWorker` resolves with the
worker's `{type:'error', …}` reply (it does not reject on it), and
`applyReduce` dereferenced `reply.samples.length` →
**"Cannot read properties of undefined (reading 'length')"** (mat-density.ts:132).
The cryptic message is exactly why the original break slipped through.

## The fix (minimal, two parts)

### 1. `density.ts walkAcc` relabel branch — project the record by axis labels

The relabel labels name which record field(s) carry M's observation. Two obs
shapes reach the peel:

- **positional value** (scalar / vector / array) — e.g. the per-copy footprint
  inside `iid(relabel(M), n)`. Labels are inert; recurse on M verbatim
  (unchanged — keeps the iid path working).
- **record** keyed by the variate name(s) — the single-observation
  `likelihoodof(relabel(M, ["x"]), record(x = v))` form. Project the named
  axes off the record (label order, the same consume/rest protocol
  `walkJointFieldsOrPositional` uses) and recurse on M with that projection,
  returning the residual record as the rest.

Two new local helpers: `isRecordValue` (a plain record obs vs a positional
value/Value/typed-array) and `relabelAxisNames` (the static string labels from
the lowerer's `vector("x", …)` names arg; null for a dynamic relabel, which
then stays label-agnostic and passes through).

This is the SAME semantics the old `inlineRelabel` record-rewrite produced,
moved into the density walk where the kept `relabel` node now lives — so it is
density-transparent exactly as spec §04 requires.

### 2. `mat-density.ts applyReduce` — surface worker errors loudly

Reject (throw) on a `{type:'error'}` worker reply with the real message instead
of dereferencing a missing `samples`. This un-masks future density failures
that previously appeared as the cryptic `reading 'length'`.

## Why both forms now work

- `iid(relabel(M), n)`: the obs flowing into the relabel peel is a positional
  vector/array, not a record → `isRecordValue` is false → the peel passes the
  value through unchanged, exactly as before. (test/iid-relabel.test.ts green.)
- `likelihoodof(relabel(M), record(x=v))`: the obs is a record →
  `relabelAxisNames` yields `["x"]`, the peel consumes field `x` (the scalar),
  recurses on M with the scalar, and returns the empty rest. Density-transparent
  = plain Normal log-pdf of `v`.

## Repro, before / after

```
cd flatppl-testsuite/tests/conversion
N=/opt/homebrew/opt/node@24/bin/node
$N score_js.cjs gaussian.flatppl obs 'record(mu = 0.0, sigma = 1.0)' \
   --engine .../wt-js-regfix/packages/engine
```

- BEFORE: `score_js: Cannot read properties of undefined (reading 'length')`
- AFTER:  `-1.7253885332046728`  (closed-form / pre-#38 ROOT value `-1.7253885332`)

(The `cartprod() requires at least two arguments` diagnostic is pre-existing,
on a binding off the scored measure's path; it does not affect the density.)

## Acceptance gate results

**A. Repro** — `gaussian.flatppl obs` → `-1.7253885332046728`. PASS.

**B. Rosetta golden harness** (`repro_hs3_js.cjs`, vs ROOT oracle):

```
gaussian  obs @ mu=0,sigma=1.0          -1.7253885332   PASS
gaussian  obs @ mu=0.5,sigma=1.0        -1.2153885332   PASS
gaussian  obs @ mu=1.27,sigma=1.0       -0.9189385332   PASS
product   likelihood @ default         -13.9458491571   PASS
product   likelihood @ mu1=0.5         -12.6373303953   PASS
product   Δ(default→mu1=0.5)             1.3085187618   PASS
histfact  Δ(pt0→pt1) vs ROOT            -3.3237121290   PASS
histfact  Δ(pt0→pt2) vs ROOT            -4.7802014001   PASS
histfact  Δ(pt0→pt3) vs ROOT           -14.8676854784   PASS
histfact  Δ(pt0→pt4) vs ROOT            -2.9179700669   PASS
```

All PASS rows pass.

**C. iid-relabel** — `test/iid-relabel.test.ts`: 4 pass, 0 fail.

**D. Full engine suite** — `node --test 'test/*.test.ts'`:
tests 3585, pass 3583, fail 0, skipped 2 (pre-existing `[OUT-OF-SCOPE]`
markers, unrelated to this change).

**E. Regression test** — `test/likelihoodof-relabel-record.test.ts` (new):
scores `logdensityof(likelihoodof(relabel(Normal(mu,sigma),["x"]),
record(x = v)), record(mu=m, sigma=s))` against an INDEPENDENT closed-form
Normal log-pdf oracle, 4 cases (incl. the HS3 `v=1.27, mu=0, sigma=1` point).
4 pass, 0 fail.

## Files changed

- `packages/engine/density.ts` — relabel-peel record projection + 2 helpers.
- `packages/engine/mat-density.ts` — `applyReduce` rejects on worker error reply.
- `packages/engine/test/likelihoodof-relabel-record.test.ts` — new regression test.
