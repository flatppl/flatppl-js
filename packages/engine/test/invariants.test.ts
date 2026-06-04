'use strict';

// Cross-file invariant tests.
//
// Several catalogs in the engine must agree across multiple files; without
// these tests, drift produces silent runtime failures (the canonical example
// being the historical `==` → `eq` lowering bug, where `eq` existed in
// `lower.js BIN_OP_MAP` but was missing from every downstream catalog).
//
// Each block below pins one invariant, citing the rationale. When you change
// one of the catalogs, the test that fails tells you which other catalog(s)
// to update.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const builtins   = require('../builtins.ts');
const types      = require('../types.ts');
const lower      = require('../lower.ts');
const orchestrator = require('../orchestrator.ts');
const sampler    = require('../sampler.ts');

// ---------------------------------------------------------------------
// 1. SAMPLEABLE_DISTRIBUTIONS ↔ sampler.REGISTRY
//
// The orchestrator's chain builder admits a distribution iff it's in
// SAMPLEABLE_DISTRIBUTIONS; the worker dispatches by REGISTRY entry. They
// MUST agree — otherwise the orchestrator will hand off a distribution the
// worker doesn't know how to sample, or refuse one the worker actually
// supports.
// ---------------------------------------------------------------------

test('invariant: orchestrator.SAMPLEABLE_DISTRIBUTIONS ⊆ sampler.REGISTRY', () => {
  for (const name of orchestrator.SAMPLEABLE_DISTRIBUTIONS) {
    assert.ok(sampler.isKnownDistribution(name),
      `SAMPLEABLE_DISTRIBUTIONS lists '${name}' but sampler.REGISTRY has no entry`);
  }
});

test('invariant: sampler.REGISTRY ⊆ orchestrator.SAMPLEABLE_DISTRIBUTIONS', () => {
  for (const name of sampler.listDistributions()) {
    assert.ok(orchestrator.SAMPLEABLE_DISTRIBUTIONS.has(name),
      `sampler.REGISTRY has '${name}' but SAMPLEABLE_DISTRIBUTIONS doesn't list it`);
  }
});

// ---------------------------------------------------------------------
// 2. DISCRETE_DISTRIBUTIONS ⊆ SAMPLEABLE_DISTRIBUTIONS
//
// The discrete subset must be sample-able. A "discrete-only" entry would
// dispatch through nothing.
// ---------------------------------------------------------------------

test('invariant: orchestrator.DISCRETE_DISTRIBUTIONS ⊆ SAMPLEABLE_DISTRIBUTIONS', () => {
  for (const name of orchestrator.DISCRETE_DISTRIBUTIONS) {
    assert.ok(orchestrator.SAMPLEABLE_DISTRIBUTIONS.has(name),
      `DISCRETE_DISTRIBUTIONS lists '${name}' but SAMPLEABLE_DISTRIBUTIONS doesn't`);
  }
});

test('invariant: REGISTRY discrete flag matches DISCRETE_DISTRIBUTIONS', () => {
  for (const name of sampler.listDistributions()) {
    const entry = sampler._internal.REGISTRY[name];
    const inSet = orchestrator.DISCRETE_DISTRIBUTIONS.has(name);
    assert.equal(!!entry.discrete, inSet,
      `${name}: REGISTRY.discrete=${entry.discrete}, DISCRETE_DISTRIBUTIONS.has=${inSet}`);
  }
});

// ---------------------------------------------------------------------
// 3. orchestrator.EVALUABLE_OPS ↔ sampler.ARITH_OPS (plus the special
//    handful evaluateCall handles inline)
//
// EVALUABLE_OPS is the static gate the orchestrator uses to decide if a
// binding's IR can be evaluated by the worker's evaluateExpr. Drifting
// the two means either the orchestrator refuses something the worker
// could compute, or admits something the worker will throw on.
// ---------------------------------------------------------------------

// The worker's evaluateCall handles a few non-ARITH_OPS ops inline (see
// sampler.evaluateCall): tuple, tuple_get, get_field, record, rnginit,
// rngstate, rand. EVALUABLE_OPS lists those + ARITH_OPS keys.
const SAMPLER_INLINE_EVALUABLE = new Set([
  'tuple', 'tuple_get', 'get_field', 'record',
  // get / get0: unified element/subset/axis-slice access (spec §07),
  // dispatched by evaluateCall via a dedicated recursive case (not the
  // positional-spread ARITH_OPS shape).
  'get', 'get0',
  'rnginit', 'rngstate', 'rand',
  // Shape functions (spec §07 Approximation functions). Dispatched
  // by sampler.evaluateCall via dedicated cases because they take
  // kwargs (coefficients / edges / values + x) that ARITH_OPS's
  // positional-spread form doesn't handle.
  'polynomial', 'bernstein', 'stepwise',
  // Binning (spec §07). bincounts also kwargs-shaped.
  'bincounts', 'selectbins',
  // Higher-order ops (filter, reduce, scan, broadcast) are dispatched
  // via dedicated cases in evaluateCall — each evaluates a referenced
  // function's body per element rather than fitting the positional-
  // spread ARITH_OPS shape.
  'filter', 'reduce', 'scan', 'broadcast',
  // Multi-axis aggregation (spec §04 §sec:aggregate). Dispatched by
  // evaluateCall via a dedicated case that walks the axis-name iteration
  // and reduces with one of the seven order-invariant reductions.
  'aggregate',
  // FlatPDL measure-eval primitives (spec §07 §sec:measure-eval-prims).
  // Dispatched in evaluateCall: builtin_logdensityof / builtin_touniform
  // / builtin_fromuniform / builtin_tonormal / builtin_fromnormal go to
  // density-prims.ts; builtin_sample synthesises a measure IR and walks
  // it via the measure walker (sampler.walk).
  'builtin_logdensityof', 'builtin_sample',
  'builtin_touniform', 'builtin_fromuniform',
  'builtin_tonormal',  'builtin_fromnormal',
]);

test('invariant: EVALUABLE_OPS ⊆ ARITH_OPS ∪ SAMPLER_INLINE_EVALUABLE', () => {
  const arithKeys = new Set(Object.keys(sampler._internal.ARITH_OPS));
  for (const op of orchestrator.EVALUABLE_OPS) {
    const ok = arithKeys.has(op) || SAMPLER_INLINE_EVALUABLE.has(op);
    assert.ok(ok,
      `EVALUABLE_OPS lists '${op}' but neither sampler.ARITH_OPS nor ` +
      `evaluateCall's inline handlers cover it`);
  }
});

// `vector` is in ARITH_OPS (the sampler can compute `[a, b, c]`) but
// deliberately NOT in EVALUABLE_OPS — per the comment in orchestrator.js,
// stochastic-element arrays like `[mu, 1.0]` must NOT classify as evaluable;
// the kind:'array' / kind:'tuple' derivations own that path.
const EVALUABLE_OPS_EXEMPTIONS = new Set([
  'vector',
  // `cat` deliberately NOT in EVALUABLE_OPS — same reasoning as
  // `vector`: cat of stochastic refs would produce per-atom vectors,
  // which the scalar-per-atom worker can't handle. cat lives in
  // ARITH_OPS so it works inside fn bodies and fixed-phase pre-eval.
  'cat',
]);

test('invariant: every ARITH_OPS key is in EVALUABLE_OPS (or exempt)', () => {
  for (const op of Object.keys(sampler._internal.ARITH_OPS)) {
    if (EVALUABLE_OPS_EXEMPTIONS.has(op)) continue;
    assert.ok(orchestrator.EVALUABLE_OPS.has(op),
      `sampler.ARITH_OPS has '${op}' but EVALUABLE_OPS doesn't list it`);
  }
});

// ---------------------------------------------------------------------
// 4. Operator desugaring → known op names
//
// Every BIN_OP_MAP / UN_OP_MAP target name MUST be either a known built-in
// (so the analyzer / classifier recognises it) AND have a type signature
// (so type inference works) AND either be evaluable or be a comparison
// op the typeinfer COMPARISON_OPS set recognises. The historical `==`
// → `eq` bug was exactly this invariant breaking silently.
// ---------------------------------------------------------------------

test('invariant: every BIN_OP_MAP target has a type signature', () => {
  const T = require('../types.ts');
  for (const op of Object.values(lower._internal ? lower._internal.BIN_OP_MAP : lower.BIN_OP_MAP)) {
    if (op === 'in') continue; // 'in' is a set-membership op, deliberately untyped today
    const sig = T.signatureOf(op);
    assert.ok(sig,
      `BIN_OP_MAP emits '${op}' but types.signatureOf has no entry`);
  }
});

// ---------------------------------------------------------------------
// 5. SIGNATURE_FACTORIES distribution kwargs match REGISTRY params
//
// For every distribution that's both type-checked AND sampleable, the
// kwarg names declared in types.js MUST be a superset of the params the
// runtime registry expects (with the runtime's `aliases` map allowed to
// translate). Otherwise a binding that typechecks won't sample.
// ---------------------------------------------------------------------

test('invariant: REGISTRY params (incl. aliases) covered by types.js kwargs', () => {
  for (const name of sampler.listDistributions()) {
    const sig = types.signatureOf(name);
    if (!sig) continue;   // distribution untyped today; separate gap
    const sigKwargs = new Set(Object.keys(sig.kwargs || {}));
    const entry = sampler._internal.REGISTRY[name];
    const aliases = entry.aliases || {};
    for (const param of entry.params) {
      // The type system kwarg should match either the spec name OR an
      // alias the runtime accepts.
      const matches = sigKwargs.has(param)
        || (aliases[param] && sigKwargs.has(aliases[param]))
        || Object.entries(aliases).some(([k, v]) => v === param && sigKwargs.has(k));
      assert.ok(matches,
        `Distribution '${name}': runtime expects param '${param}' but type ` +
        `system kwargs are {${[...sigKwargs].join(', ')}}; aliases = ` +
        JSON.stringify(aliases));
    }
  }
});

// ---------------------------------------------------------------------
// 6. UN_OP_MAP targets have type signatures
//
// Symmetric to block 4 (BIN_OP_MAP). Block 4's header names both maps
// but the test only iterated BIN_OP_MAP, leaving unary desugaring
// (`-x` → `neg`, `+x` → `pos`) unguarded against the same `==`→`eq`
// class of silent drift. Pin it.
// ---------------------------------------------------------------------

test('invariant: every UN_OP_MAP target has a type signature', () => {
  const T = require('../types.ts');
  const UN = lower._internal ? lower._internal.UN_OP_MAP : lower.UN_OP_MAP;
  for (const op of Object.values(UN)) {
    assert.ok(T.signatureOf(op),
      `UN_OP_MAP emits '${op}' but types.signatureOf has no entry`);
  }
});

// ---------------------------------------------------------------------
// 7. Sampleable-distribution sets anchor to the canonical name registry
//
// The orchestrator's sampleable / discrete sets must name only real
// builtin distributions, else the orchestrator admits a constructor
// the parser/analyzer never recognised.
//
// (Deliberately NOT asserting MEASURE_OP_CLASSIFIERS ⊆
// builtins.MEASURE_OPS, nor ⊆ sampler.walk's MEASURE_OP_WALKERS: the
// classifier map is intentionally cross-cut — it also keys structural
// ops (record/tuple/draw/broadcast) that aren't measure-algebra ops,
// and only the trace-sampled subset has a walker. Both would be false
// invariants; the audit confirmed the measure-op catalogs do not form
// clean subset relations by design.)
// ---------------------------------------------------------------------

test('invariant: SAMPLEABLE_DISTRIBUTIONS ⊆ builtins.DISTRIBUTIONS', () => {
  for (const name of orchestrator.SAMPLEABLE_DISTRIBUTIONS) {
    assert.ok(builtins.DISTRIBUTIONS.has(name),
      `SAMPLEABLE_DISTRIBUTIONS lists '${name}' but builtins.DISTRIBUTIONS ` +
      `doesn't — parser/analyzer won't recognise the constructor name`);
  }
});

test('invariant: DISCRETE_DISTRIBUTIONS ⊆ builtins.DISTRIBUTIONS', () => {
  for (const name of orchestrator.DISCRETE_DISTRIBUTIONS) {
    assert.ok(builtins.DISTRIBUTIONS.has(name),
      `DISCRETE_DISTRIBUTIONS lists '${name}' but builtins.DISTRIBUTIONS ` +
      `doesn't`);
  }
});


// ---------------------------------------------------------------------
// 8. Every measure-algebra op's classifier handles inline subexpressions
//
// Each measure-algebra op has metadata scattered across several
// registries: a classifier in derivations.ts (often requiring
// self-ref args), an entry in lift.ts argSignature (declaring which
// arg positions need anon-lifting), the measure-producing set,
// signatures in types.ts. Drift between any pair fails silently: the
// classifier returns null and the binding gets no derivation — the
// viewer then surfaces "Not plottable for <name>" with no clue
// what's missing.
//
// The bridge fix until the OpDecl migration lands (engine-concepts
// §18 → measure-algebra phase): for each op below, build a binding
// with the op applied to inline subexpressions and assert the
// resulting binding has a derivation. Catches missing argSignature
// entries, missing inlineXxxLift handlers, missing classifier paths.
//
// One concrete bug caught by this invariant: flatppl-js commit
// 868dd1f fixed `bayesupdate` + `likelihoodof` missing from
// argSignature, which silently broke `restrict` expansion.
// ---------------------------------------------------------------------

test('invariant: every measure-algebra op classifies under inline subexpressions', () => {
  const { processSource } = require('..');
  // Each fixture has exactly one binding 'b' whose RHS exercises the
  // op with INLINE subexpression args (no explicit anon bindings).
  // After lift + buildDerivations, b must have a derivation entry.
  const fixtures: Array<{op: string, src: string}> = [
    { op: 'weighted',     src: 'b = weighted(0.5, Normal(0, 1))' },
    { op: 'logweighted',  src: 'b = logweighted(-1.0, Normal(0, 1))' },
    { op: 'normalize',    src: 'b = normalize(weighted(0.5, Normal(0, 1)))' },
    { op: 'superpose',    src: 'b = superpose(Normal(0, 1), Normal(1, 1))' },
    { op: 'iid',          src: 'b = iid(Normal(0, 1), 3)' },
    { op: 'joint',        src: 'b = joint(x = Normal(0, 1), y = Exponential(1))' },
    { op: 'truncate',     src: 'b = truncate(Normal(0, 1), interval(0.0, 1.0))' },
    { op: 'pushfwd',      src: 'b = pushfwd(fn(_ + 1.0), Normal(0, 1))' },
    { op: 'jointchain',   src: 'b = jointchain(Normal(0, 1), fn(Normal(_, 1.0)))' },
    { op: 'kchain',       src: 'b = kchain(Normal(0, 1), fn(Normal(_, 1.0)))' },
    { op: 'bayesupdate',  src: 'b = bayesupdate(likelihoodof(fn(Normal(_, 1.0)), 0.5), Normal(0, 1))' },
  ];
  const failures: string[] = [];
  for (const f of fixtures) {
    const r = processSource(f.src);
    const errs = r.diagnostics.filter((d: any) => d.severity === 'error');
    if (errs.length > 0) {
      failures.push(`${f.op}: parse/analyse error — ${errs.map((d: any) => d.message).join('; ')}`);
      continue;
    }
    const lifted = orchestrator.liftInlineSubexpressions(r.bindings);
    const built = orchestrator.buildDerivations(lifted);
    if (!built.derivations.b) {
      failures.push(`${f.op}: no derivation (silent failure — missing argSignature entry, lift handler, or classifier path?)`);
    }
  }
  assert.deepEqual(failures, [],
    'measure-algebra op classification under inline subexpressions:\n  - ' + failures.join('\n  - '));
});

// ---------------------------------------------------------------------
// 9. Likelihood bindings are tagged 'likelihood' (consumed, not materialised)
//
// likelihoodof returns a likelihood object, NOT a measure (spec §06).
// Likelihood bindings don't get measure-algebra derivations; they're
// consumed by bayesupdate / densityof / logdensityof. The
// classifyStatement → 'likelihood' tag is what downstream consumers
// dispatch on; ensure that's still wired.
// ---------------------------------------------------------------------

test('invariant: likelihoodof bindings classify as type=likelihood', () => {
  const { processSource } = require('..');
  const r = processSource('L = likelihoodof(fn(Normal(_, 1.0)), 0.5)');
  assert.deepEqual(r.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const L = r.bindings.get('L');
  assert.equal(L?.type, 'likelihood',
    'likelihoodof bindings must have binding.type === "likelihood" ' +
    '(the consumer-side tag bayesupdate / densityof / logdensityof read)');
});

