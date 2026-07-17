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
const materialiser = require('../materialiser.ts');

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
    // Density-only entries (spec §09 ContinuedPoisson) have no sampler and are
    // deliberately absent from SAMPLEABLE_DISTRIBUTIONS — exempt them.
    if (sampler._internal.REGISTRY[name].densityOnly) continue;
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
  // checked(value, condition): spec §07 value-preserving assertion;
  // dispatched inline in evaluateCall (carries a `condition` kwarg the
  // positional-spread ARITH_OPS shape doesn't handle).
  'checked',
  // get / get0: unified element/subset/axis-slice access (spec §07),
  // dispatched by evaluateCall via a dedicated recursive case (not the
  // positional-spread ARITH_OPS shape).
  'get', 'get0',
  'rnginit', 'rngstate', 'rand',
  // rand_succ(state): composite-rand successor (split lane 1), synthesised
  // by the lift rand_succ rewrite; dispatched inline in evaluateCall like
  // rand/rngstate (value-typed rngstate, not ARITH_OPS).
  'rand_succ',
  // Shape functions (spec §07 Approximation functions). Dispatched
  // by sampler.evaluateCall via dedicated cases because they take
  // kwargs (coefficients / edges / values + x) that ARITH_OPS's
  // positional-spread form doesn't handle.
  'polynomial', 'bernstein', 'stepwise',
  // Binning (spec §07). bincounts also kwargs-shaped.
  'bincounts', 'selectbins',
  // Set membership (spec §07 `x in S`). Dispatched by evaluateCall via a
  // dedicated case: its second arg is a SET descriptor (interval / named set),
  // not a scalar, so it doesn't fit the positional-spread ARITH_OPS shape.
  'in',
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

// ---------------------------------------------------------------------
// Scalar-primitive arity — ONE source of truth (engine-concepts §18)
//
// `ops.SCALAR_PRIM_ARITY` is the canonical name→arity for the elementwise
// scalar primitives. The broadcasted-variant table (BCAST_TABLE), the
// batched evaluator (ARITH_OPS_N) and the fused-loop compiler all derive
// from it. The BCAST registration asserts per-entry arity at load; this
// pins the SET (an op added to the canonical map but missing a variant)
// and the REAL/COMPLEX partition + the consumer dedup.
// ---------------------------------------------------------------------

test('invariant: every ops.SCALAR_PRIM_ARITY op has a broadcast variant', () => {
  const ops = require('../ops.ts');
  require('../ops-declarations.ts');   // triggers _ensureBroadcastedRegistered
  // Scalar prims are variant-only (no full OpDecl), so query hasVariantFor
  // directly rather than enumerating listDeclared (full decls only).
  for (const op of Object.keys(ops.SCALAR_PRIM_ARITY)) {
    assert.ok(ops.hasVariantFor(op, 'broadcast'),
      `SCALAR_PRIM_ARITY lists '${op}' but ops has no broadcast variant for it ` +
      `(BCAST_TABLE drift)`);
  }
});

test('invariant: REAL ⊎ COMPLEX scalar prims partition SCALAR_PRIM_ARITY', () => {
  const ops = require('../ops.ts');
  const full = Object.keys(ops.SCALAR_PRIM_ARITY);
  const real = new Set(Object.keys(ops.REAL_SCALAR_PRIM_ARITY));
  const complex = new Set(Object.keys(ops.COMPLEX_SCALAR_PRIMS));
  assert.equal(real.size + complex.size, full.length, 'real + complex count = full');
  for (const op of full) {
    assert.ok(real.has(op) !== complex.has(op),
      `'${op}' must be in exactly one of REAL / COMPLEX scalar prims`);
    assert.equal((ops.SCALAR_PRIM_ARITY as any)[op],
      real.has(op) ? (ops.REAL_SCALAR_PRIM_ARITY as any)[op]
                   : (ops.COMPLEX_SCALAR_PRIMS as any)[op],
      `arity of '${op}' agrees across the full + subset maps`);
  }
});

test('invariant: batched + compile consumers share the canonical REAL map', () => {
  const ops = require('../ops.ts');
  const compile = require('../sampler-eval-compile.ts');
  // Same object reference — the dedup, not a copy that could drift.
  assert.equal(compile._COMPILE_ARITY, ops.REAL_SCALAR_PRIM_ARITY);
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
// 9. Measure-op / distribution catalog coverage — every classified
//    derivation kind has a materialiser KIND_HANDLERS entry (or is
//    reroute-exempt).
//
// The measure-op metadata is split across four catalogs (derivations
// MEASURE_OP_CLASSIFIERS / sampler MEASURE_OP_WALKERS / density OP_HANDLERS
// / materialiser KIND_HANDLERS) which drift SILENTLY: a new distribution or
// measure-op whose classifier lands but whose materialiser handler is
// forgotten throws "no handler for kind X" only when that op is first
// materialised — far from the wiring mistake. (The earlier subset-invariant
// attempt was declined because the catalogs are intentionally cross-cut by
// op-name vs derivation-kind — block 7's note.) These pin the one clean
// correspondence: classifier-emitted KIND → materialiser handler.
//
// `jointchain` is the sole union kind WITHOUT a KIND_HANDLERS entry — it
// and `kchain` route through clmRerouteStage → matClm before kind dispatch
// (materialiser.ts ~1054). Listed exempt.
// ---------------------------------------------------------------------

// The derivation kinds — KEEP IN SYNC with engine-types.d.ts `Derivation`
// union (the authoritative catalog). A new union member must be added here
// AND given a KIND_HANDLERS entry (or a KIND_HANDLER_EXEMPT entry); block 9a
// fails loudly if a handler names a kind missing from this set.
const ALL_DERIVATION_KINDS = new Set([
  'alias', 'array', 'tuple', 'record', 'sample', 'evaluate',
  'weighted', 'normalize', 'superpose', 'iid', 'randsample', 'jointchain',
  'truncate', 'pushfwd', 'bayesupdate', 'logdensityof', 'likelihood_density',
  'joint_likelihood_density', 'posterior_density',
  'totalmass', 'broadcast_logdensity', 'select', 'kernelbroadcast',
  'mvnormal', 'dirichlet', 'multinomial', 'wishart', 'inversewishart',
  'lkjcholesky', 'lkj', 'binnedpoissonprocess', 'poissonprocess',
]);
// Kinds handled OUTSIDE KIND_HANDLERS (clmRerouteStage → matClm).
const KIND_HANDLER_EXEMPT = new Set(['jointchain']);

test('invariant 9a: every KIND_HANDLERS key is a known derivation kind', () => {
  for (const kind of Object.keys(materialiser.KIND_HANDLERS)) {
    assert.ok(ALL_DERIVATION_KINDS.has(kind),
      `KIND_HANDLERS has a handler for '${kind}', which is not in ` +
      `ALL_DERIVATION_KINDS — add it (and engine-types.d.ts Derivation), or fix the typo`);
  }
});

test('invariant 9b: every derivation kind has a KIND_HANDLERS entry (or is reroute-exempt)', () => {
  for (const kind of ALL_DERIVATION_KINDS) {
    if (KIND_HANDLER_EXEMPT.has(kind)) continue;
    assert.ok(Object.prototype.hasOwnProperty.call(materialiser.KIND_HANDLERS, kind),
      `derivation kind '${kind}' has no materialiser KIND_HANDLERS entry — the ` +
      `classifier can emit it but the materialiser would throw "no handler for kind"`);
  }
});

test('invariant 9c: every distribution constructor classifies to a handled kind', () => {
  const { processSource } = require('..');
  // Measure-binding form `b = Dist(...)` ⇒ b's derivation kind IS the dist's
  // (the draw form `b ~ Dist` aliases b to an anon carrying the kind). Robust
  // to lifting: MvNormal lowers to `pushfwd`, also a handled kind.
  const dists = [
    'Normal(0.0, 1.0)',
    'Dirichlet(alpha = [1.0, 1.0, 1.0])',
    'Multinomial(n = 5, p = [0.2, 0.3, 0.5])',
    'Wishart(nu = 3.0, scale = [[1.0, 0.0], [0.0, 1.0]])',
    'InverseWishart(nu = 3.0, scale = [[1.0, 0.0], [0.0, 1.0]])',
    'LKJ(n = 2, eta = 1.0)',
    'LKJCholesky(n = 2, eta = 1.0)',
    'MvNormal(mu = [0.0, 0.0], cov = [[1.0, 0.0], [0.0, 1.0]])',
    'BinnedPoissonProcess(rates = [1.0, 2.0, 3.0])',
    'PoissonProcess(intensity = weighted(5.0, Normal(0.0, 1.0)))',
  ];
  const failures: string[] = [];
  for (const call of dists) {
    const r = processSource(`b = ${call}`);
    const errs = r.diagnostics.filter((d: any) => d.severity === 'error');
    if (errs.length) { failures.push(`${call}: parse/analyse — ${errs.map((d: any) => d.message).join('; ')}`); continue; }
    const built = orchestrator.buildDerivations(orchestrator.liftInlineSubexpressions(r.bindings));
    const d = built.derivations.b;
    if (!d) { failures.push(`${call}: no derivation`); continue; }
    if (!Object.prototype.hasOwnProperty.call(materialiser.KIND_HANDLERS, d.kind)
        && !KIND_HANDLER_EXEMPT.has(d.kind)) {
      failures.push(`${call}: classifies to kind '${d.kind}' with no KIND_HANDLERS entry`);
    }
  }
  assert.deepEqual(failures, [],
    'distribution → handled-kind coverage:\n  - ' + failures.join('\n  - '));
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

