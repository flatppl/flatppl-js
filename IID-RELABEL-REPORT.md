# iid accepts a relabel'd measure — fix report

Worktree: `/Users/bcox/Code/flatppl/wt-js-iid-relabel` (branch `iid-accepts-relabel`).
Engine: `packages/engine`. Node 24 (`/opt/homebrew/opt/node@24/bin/node`).

## Bug

The measure-algebra op `iid` rejected a `relabel`'d measure:

```
iid: arg 1 expects a measure, got deferred
```

Repro (diagnostics): `iid(relabel(Normal(mu=m,sigma=s),["x"]), n)` reported the
error above, while the plain `iid(Normal(...), n)` reported `[]`.

## Spec basis (conformant)

`flatppl-design/docs/04-design.md` "Interface adaptation": output-side renaming
`relabel(M, names)` "lifts directly to sets, functions, measures, and kernels"
and "for measures it is equivalent to `pushfwd(fn(relabel(_, names)), M)`". So
`relabel(M)` IS a measure and `iid(relabel(M), n)` must type-check.
`06-measure-algebra.md` "Independent composition" takes a measure. Density
evaluation already treats relabel as transparent (`density.ts` walkAcc peels it;
`mat-density.ts asScalarFactor` peels relabel/record), so inference was the
inconsistent layer.

## Root cause (two layers)

1. **typeinfer.ts** had NO type rule for `relabel`, so a `relabel(...)` call
   inferred as `T.deferred()`. `inferIid`'s `T.isMeasure` guard then failed.

2. **lift.ts `inlineRelabel`** was *over-eager*. Its single-name scalar-wrap
   branch (`relabel(<anything>, [name]) → record(name = arg)`) fired even when
   the arg was a MEASURE. So `relabel(Normal(...), ["x"])` was rewritten to
   `record(x = Normal(...))`. With the typeinfer rule alone, the type-check
   passed but the runtime density walker then saw a named-field joint and tried
   to consume field `x` from each scalar observation:
   `density: cannot consume named field 'x' from non-record value`. This second
   layer had to be fixed too, otherwise the relabel'd form type-checked but
   could not be scored.

## The fix

### typeinfer.ts — `inferRelabel` (new), wired into the call switch

Added `case 'relabel': return write(inferRelabel(expr, scopes), expr);` next to
`pushfwd` (the spec equivalence partner). `inferRelabel` is kind-transparent:

- **measure → measure** (return the base measure type verbatim — labels only,
  shape/domain/sampleShape unchanged; mirrors `inferLocscale`/`inferPushfwd`'s
  measure handling and is at least as permissive as `pushfwd`).
- **function → function, kernel → kernel** (callable lifts).
- **failed → failed**, **deferred/any → deferred** (propagate; never fabricate a
  measure from an unresolved base).
- **value → record** (spec §04 lines 482-507). When the names arg is a literal
  `vector(<string lit>, …)` it builds a *field-typed* record so downstream
  `get_field` sees real fields:
  - array base → one field per name, each the array element type;
  - record base → rename existing field types by position (arity-checked);
  - scalar base → single-field record carrying the scalar type.
  When names aren't a static literal vector, it DEFERS (permissive — matches the
  pre-rule behaviour) rather than fabricating an empty/wrong record.

Helper `literalStringVector(arg)` reads the lowered `vector(<string lit>, …)`
names form, returning `null` for any non-literal element.

### lift.ts — `inlineRelabel` measure guard (new `isMeasureLikeArg`)

Added an early `if (isMeasureLikeArg(argExpr)) return astArg;` so a relabel whose
first arg denotes a measure (or kernel) is LEFT as a `relabel(...)` node and the
density walkers peel it transparently. Only VALUE-shaped args keep the existing
array/record/scalar → `record(...)` rewrite. `isMeasureLikeArg` classifies via
`builtins.MEASURE_PRODUCING` (a direct call to a measure op / distribution) and
follows Identifier bindings' RHS (peeling nested relabel). Newly imported
`MEASURE_PRODUCING` from `./builtins.ts`.

## Value-level array→record handling (no regression)

Value relabel still becomes a record. Top-level binding RHS value relabels (e.g.
`v = relabel([1.0,2.0],["a","b"])`) reach typeinfer as a raw `relabel` node (not
rewritten by `inlineRelabel`, which only fires in inline arg positions). Before
the fix these typed as `deferred` (so `v.a` was permissively accepted); the
naive first cut of `inferRelabel` returned an empty record `T.record({})`, which
REGRESSED `v.a` to `get_field: 'a' is not a field of record`. The field-typed
record construction above fixes that. Verified: `v.a` (array), `v.a` (scalar),
and `v.q` (record rename) all type-check clean.

## TDD RED/GREEN

New focused test: `packages/engine/test/iid-relabel.test.ts`.

- RED (before fix): the named-binding and inline type-check cases reported
  `['iid: arg 1 expects a measure, got deferred']`; the numeric-equivalence test
  threw `density: cannot consume named field 'x' from non-record value`. Baseline
  plain case passed.
- GREEN (after fix): all 4 tests pass.

## Numeric equivalence (relabel is density-transparent)

Scored `logdensityof(likelihoodof(iid(relabel(Normal(mu=m,sigma=s),["x"]), n),
obs), record(m=0.4,s=1.3))` vs the same with a plain `Normal` (no relabel),
obs `[0.0, 1.0]`:

```
plain ld   = -2.5164517491904816
relabel ld = -2.5164517491904816
|diff|     = 0   (exactly; well under the 1e-12 bound)
```

The named-binding and inline-relabel forms both equal the plain form. The fix
admits the relabel'd form without changing densities.

## Other measure-algebra ops confirmed accepting relabel

Verified clean (no diagnostics) on `rm = relabel(Normal(mu=m,sigma=s),["x"])`:

- `truncate(rm, interval(-2.0, 2.0))`
- `normalize(truncate(rm, interval(-2.0, 2.0)))`
- `joint(rm, Normal(...))`

These all share the `T.isMeasure` guard pattern that `iid` uses, so the
measure-transparent rule unblocks them uniformly.

## Full-suite tally (Node 24)

`node --test 'test/*.test.ts'`:

```
tests 3581   pass 3579   fail 0   skipped 2   todo 0   cancelled 0
```

The 2 skipped are pre-existing. No regressions in existing relabel / value-level
or measure-algebra tests.

## Concerns

- The two-layer nature: the prompt anticipated only the typeinfer layer; the
  `inlineRelabel` over-eager scalar-wrap was a second, runtime-only bug that the
  numeric-equivalence test surfaced. Both are now fixed. Anyone touching
  `inlineRelabel` should keep the measure guard.
- `isMeasureLikeArg` keys off `builtins.MEASURE_PRODUCING`; if a new
  measure-producing op is added but not registered there, a relabel directly
  over it could again be mis-rewritten to a record. Low risk — `MEASURE_PRODUCING`
  is the canonical registry — but worth noting.
- `inferRelabel`'s value branch defers (rather than erroring) on dynamic
  (non-literal) names; this preserves the prior permissive behaviour and is
  consistent with how other ops defer when shapes aren't statically resolvable.
