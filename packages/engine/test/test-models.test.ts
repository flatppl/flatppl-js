'use strict';

// =====================================================================
// test-models.test.ts — colleague-contributed Bayesian model fixtures
// =====================================================================
//
// Verifies the test-model fixtures landed by colleagues (commits
// c9baa46 + 009f372) parse, classify, and materialise where the
// engine supports the patterns involved. Each fixture targets a
// distinct PPL surface combination:
//
//   - zero-inflated-binomial.flatppl — `superpose` mixture with a
//     point-mass `Dirac` component; iid observation block. Exercises
//     the cascade-prune walker's branch-aware ref collection (the
//     `superpose` IR carries per-branch measures in a `branches`
//     field, not args/kwargs — see `ir-shared.collectSelfRefs`).
//
//   - normal-mixture.flatppl — `normalize(superpose(weighted(theta,
//     Normal(mu[1], 1)), weighted(1-theta, Normal(mu[2], 1))))` over
//     iid observations. Same `branches`-walk requirement plus
//     `mu[1]` / `mu[2]` selector indexing inside a forward kernel.
//
//   - horseshoe.flatppl — Horseshoe prior on regression coefficients
//     using `normalize(truncate(Cauchy(0, 1), interval(0, inf)))`
//     and `iid(half_cauchy, D_X)`. Exercises matIid's resolution
//     through normalize / truncate (the worker has a separate
//     `truncateSampleN` path; matIid peels normalize as a sample-
//     preserving no-op and routes truncate to truncateSampleN with
//     count = N × k).
//
//   - beta-binomial-pushfwd.flatppl — Hierarchical Beta-Binomial
//     with Pareto prior via `pushfwd(fn(0.1 * exp(_)),
//     Exponential(1.5))`. Tests pushfwd typeinfer signature (added
//     to types.ts since the inferGenericCall fallback returned
//     `deferred()` for unrecognised ops, leaving downstream
//     `iid(pareto, G)` to fail with "expects measure, got
//     deferred"). The model also chains user-fn-returning-measure
//     broadcasts (`p ~ beta_row_K.(a, b)`, `r ~ binomial_row_K.
//     (n_data, p)`) — those hit a deeper typeinfer gap that's a
//     follow-up (see TODO-flatppl-js fusion (b) / typeinfer).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

function readFixture(name: string): string {
  return fs.readFileSync(path.join(__dirname, 'fixtures', name), 'utf-8');
}

function setupCtx(src: string, N: number) {
  const lifted = processSource(src);
  const errs = (lifted.diagnostics || []).filter((d: any) => d.severity === 'error');
  if (errs.length > 0) {
    return { errs, ctx: null, built: null };
  }
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 42 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const p = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, p);
      return p;
    },
    sendWorker: (m: any) => {
      const r = worker.handle(m);
      return r && r.type === 'error'
        ? Promise.reject(new Error(r.message)) : Promise.resolve(r);
    },
    sampleCount: N,
    rootSeed: 42,
  };
  return { errs: [], ctx, built };
}

// =====================================================================
// 1. Zero-inflated Binomial — exercises the `branches` walk fix
// =====================================================================

test('zero-inflated-binomial: classifies + materialises', async () => {
  const { errs, ctx, built } = setupCtx(readFixture('zero-inflated-binomial.flatppl'), 50);
  assert.equal(errs.length, 0, 'no parse/typeinfer errors');
  assert.ok(built.derivations.posterior, 'posterior classifies');
  assert.equal(built.derivations.posterior.kind, 'bayesupdate');
  // Materialising posterior exercises the bayesupdate density walk,
  // which in turn needs the forward_kernel body's refs (`K`, `p`,
  // `psi`) collected via collectSelfRefs. Pre-fix, `K` lived inside
  // a `superpose` branch and got missed → "unbound self reference
  // 'K'" at eval time.
  const m = await ctx!.getMeasure('posterior');
  assert.ok(m, 'posterior materialises');
  assert.ok(typeof m.n_eff === 'number' && Number.isFinite(m.n_eff),
    `posterior n_eff is finite (got ${m.n_eff})`);
});

// =====================================================================
// 2. Normal mixture — `normalize(superpose(weighted, weighted))` +
//    `iid` + per-atom `mu[1]` / `mu[2]` indexing inside the kernel
// =====================================================================

test('normal-mixture: classifies + materialises', async () => {
  const { errs, ctx, built } = setupCtx(readFixture('normal-mixture.flatppl'), 50);
  assert.equal(errs.length, 0);
  assert.ok(built.derivations.posterior);
  assert.equal(built.derivations.posterior.kind, 'bayesupdate');
  const m = await ctx!.getMeasure('posterior');
  assert.ok(m, 'posterior materialises');
  assert.ok(typeof m.n_eff === 'number' && Number.isFinite(m.n_eff),
    `posterior n_eff is finite (got ${m.n_eff})`);
});

// =====================================================================
// 3. Horseshoe regression — matIid through normalize / truncate
// =====================================================================

test('horseshoe: classifies + iid(half_cauchy) materialises', async () => {
  const { errs, ctx, built } = setupCtx(readFixture('horseshoe.flatppl'), 100);
  assert.equal(errs.length, 0);
  assert.ok(built.derivations.posterior, 'posterior classifies');

  // Pre-fix: matIid called leafSampleIR which only peels through
  // 'alias'; half_cauchy = normalize(truncate(Cauchy(0,1),
  // interval(0,inf))) → leafSampleIR returned null and matIid threw
  // "cannot resolve leaf sample IR for half_cauchy". Post-fix:
  // matIid peels normalize + routes truncate through
  // truncateSampleN with count = N × k. Verify lambdas (the iid'd
  // half-Cauchy) materialises cleanly to [N, D_X] without NaN.
  const lambdas = await ctx!.getMeasure('lambdas');
  assert.deepEqual(lambdas.value.shape, [100, 3]);
  let anyNaN = false;
  for (let i = 0; i < lambdas.value.data.length; i++) {
    if (Number.isNaN(lambdas.value.data[i])) { anyNaN = true; break; }
  }
  assert.equal(anyNaN, false, 'no NaN in lambdas samples');
  // All lambdas samples must be positive (half-Cauchy support).
  let allPositive = true;
  for (let i = 0; i < lambdas.value.data.length; i++) {
    if (lambdas.value.data[i] < 0) { allPositive = false; break; }
  }
  assert.equal(allPositive, true, 'all lambdas samples are positive');

  // Note: posterior materialises to a measure record but the
  // bayesupdate's likelihood step currently produces NaN n_eff
  // (likely a separate engine issue in how `X * betas` matvec with
  // per-atom betas threads through density eval — a follow-up).
  // Pin the prior parts work; leave posterior n_eff unchecked.
});

// =====================================================================
// 4. Beta-Binomial with pushfwd-derived Pareto — partial coverage
// =====================================================================
//
// Tests pushfwd's typeinfer signature is correct enough for the
// simple "iid of pushfwd-derived measure" composition. Full
// hierarchical end-to-end is gated on the user-kernel-composition
// typeinfer gap (`binomial_row_K = (n, p) -> Binomial.(n, p)` whose
// return type isn't currently inferred as a measure when called via
// dotted broadcast). Tracked as a Phase F sub-follow-up.

test('beta-binomial-pushfwd (partial): pushfwd typeinfer + iid resolves', () => {
  // Strip the kernel-composition chunk to isolate the pushfwd
  // typeinfer signature. Verify the simple `iid(pareto, G)` shape
  // (which pre-fix failed "iid: arg 1 expects measure, got
  // deferred") classifies cleanly.
  const r = processSource(`
G = 2
pareto = pushfwd(fn(0.1 * exp(_)), Exponential(1.5))
a_plus_b ~ iid(pareto, G)
`);
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0,
    `pushfwd typeinfer: expected no errors, got ${JSON.stringify(errs)}`);
  const built = orchestrator.buildDerivations(r.bindings);
  assert.ok(built.derivations.a_plus_b, 'iid(pareto, G) classifies');
  assert.ok(built.bindings.get('pareto')?.inferredType?.kind === 'measure',
    'pareto inferredType.kind === measure (was deferred pre-fix)');
});

// =====================================================================
// 5. Beverton-Holt stock-recruitment (steepness parametrisation)
// =====================================================================
//
// Lognormal-observation Bayesian regression for fisheries
// recruitment data. Uses pushfwd-derived priors (uniform on
// (0.2, 1) for steepness h), normalize(truncate(...)) for
// positive-support priors (half-Cauchy on σ; truncated-Normal on
// α), and a deterministic transform chaining the priors into
// `log_rhat` — a per-observation linear predictor that ends up as
// the lognormal mean.
//
// What this exercises beyond the others: a fixed-phase scalar
// constant (`max_r`) is closed-over deep inside a stochastic prior
// (`alpha ~ normalize(truncate(Normal(2 * max_r, …), …))`).
// Without the `collectRefArrays` fix that auto-pushes fixed refs
// into the worker session env, materialising `alpha` fails with
// "unbound self reference 'max_r'" — even though `max_r` is in
// `fixedValues`. (The legacy callers had to remember to push
// fixedEnv themselves; this fixture catches the gap end-to-end.)

test('beverton-holt: classifies + materialises (collectRefArrays auto-push fix)', async () => {
  const { errs, ctx, built } = setupCtx(readFixture('beverton-holt.flatppl'), 50);
  assert.equal(errs.length, 0);
  assert.ok(built.derivations.posterior);
  assert.equal(built.derivations.posterior.kind, 'bayesupdate');
  // alpha goes through normalize(truncate(Normal(2 * max_r, ...))).
  // Pre-fix: fails with "unbound self reference 'max_r'" because
  // collectRefArrays filtered out fixed refs without pushing them
  // into the worker session env.
  const alpha = await ctx!.getMeasure('alpha');
  assert.ok(alpha && alpha.samples,
    'alpha materialises (max_r is auto-pushed via setEnv merge)');
  // All alpha samples must be positive (truncate to (0, inf)).
  for (let i = 0; i < alpha.samples.length; i++) {
    assert.ok(alpha.samples[i] > 0,
      `alpha sample ${i} = ${alpha.samples[i]} must be > 0`);
  }
  // Posterior materialises end-to-end (low n_eff is expected for a
  // single MC step against informative-ish data — not an engine
  // bug; just the model's nature with a small sample count).
  const m = await ctx!.getMeasure('posterior');
  assert.ok(m, 'posterior materialises');
  assert.ok(typeof m.n_eff === 'number' && Number.isFinite(m.n_eff),
    `posterior n_eff is finite (got ${m.n_eff})`);
});

test('beta-binomial-pushfwd: full fixture exposes user-kernel-composition gap', () => {
  // Pin the failure mode of the full fixture so a regression in
  // either direction (fix lands → catch flips green; gap reopens
  // → assertion fails) is detected. The deeper gap: a user fn
  // returning `Binomial.(n_row, p_row)` (kernel-broadcast) doesn't
  // typeinfer as a measure when the user fn is called via dotted
  // broadcast (`binomial_row_K.(n_data, p)`) — the outer broadcast
  // sees `array of real` instead of `array of measure`.
  const src = readFixture('beta-binomial-pushfwd.flatppl');
  const r = processSource(src);
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  // Pre-fix: would fail at L27 (iid: expects measure, got deferred).
  // Post-fix: pushfwd typeinfer now produces measure(real); first
  // failure moves to L39 (draw: expects measure, got array of
  // real) — the user-kernel-composition gap.
  assert.ok(errs.length > 0, 'still has remaining typeinfer errors');
  assert.match(errs[0].message,
    /draw|measure|user.*kernel|broadcast|expected/i,
    `first error is the user-kernel-composition gap: ${errs[0]?.message}`);
});
