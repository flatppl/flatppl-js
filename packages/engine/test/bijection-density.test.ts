'use strict';

// Tests for `bijection(f, f_inv, logvolume)` + first-class
// `pushfwd(f, M)` density evaluation.
//
// Bijection is a wrapper/annotation around a function: semantically
// it IS `f` (calls to `b(x)` evaluate `f(x)`); the inverse and log-
// volume slots are metadata that density paths consult. Sampling
// goes through matPushfwd unchanged whether f is a plain function or
// a bijection. Density of `pushfwd(b, M)` uses the bijection
// metadata to evaluate
//
//   log p_{b*M}(y) = log p_M(f_inv(y)) − logvolume(f_inv(y))
//
// Test coverage:
//   1. LogNormal closed form: pushfwd(bijection(fn(exp(_)), fn(log(_)),
//      fn(_)), Normal(0,1)) density at y matches LogNormal logpdf.
//   2. Linear affine bijection (Jacobian = log|2| = log 2):
//      pushfwd(bijection(fn(2*_+1), fn((_-1)/2), log(2)), Normal(0,1))
//      density at y matches N_logpdf((y-1)/2, 0, 1) − log 2.
//   3. Non-bijection pushfwd in density mode raises a clear error.
//   4. Calling a bijection binding from value position evaluates the
//      forward function (the transparency property).
//   5. Sampling through a bijection works (regression — same as
//      plain function pushfwd).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const SAMPLE_COUNT = 1024;
const ROOT_SEED    = 0xB1737CFC;

function makeCtx(source: any, opts: any) {
  opts = opts || {};
  const lifted = processSource(source);
  const built  = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: ROOT_SEED });
  const cache = new Map();
  const ctx = {
    derivations: built.derivations,
    bindings:    built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure:  (name: any) => {
      if (cache.has(name)) return cache.get(name);
      const p = materialiser.materialiseMeasure(name, ctx);
      cache.set(name, p);
      return p;
    },
    sendWorker:  (msg: any) => {
      const reply = worker.handle(msg);
      if (reply && reply.type === 'error') return Promise.reject(new Error(reply.message));
      return Promise.resolve(reply);
    },
    sampleCount: opts.sampleCount != null ? opts.sampleCount : SAMPLE_COUNT,
    rootSeed:    ROOT_SEED,
  };
  return ctx;
}

const lnLogpdf = require('@stdlib/stats-base-dists-lognormal-logpdf');
const nLogpdf = require('@stdlib/stats-base-dists-normal-logpdf');

// =====================================================================
// LogNormal closed-form check
// =====================================================================

test('bijection: pushfwd(exp/log/identity, Normal(0,1)) density ≡ LogNormal(0,1)', async () => {
  // Spec equivalence: LogNormal(mu, sigma) = pushfwd(exp, Normal(mu, sigma)).
  // exp's bijection annotation: forward = exp, inverse = log, logvolume(x) = x
  // (because log|d/dx exp(x)| = log(exp(x)) = x).
  const ctx = makeCtx(`
M = Normal(mu = 0.0, sigma = 1.0)
b = bijection(fn(exp(_)), fn(log(_)), fn(_))
LN = pushfwd(b, M)
lp = logdensityof(LN, 2.0)
`);
  const lp = await ctx.getMeasure('lp');
  // logdensityof returns a per-atom Float64Array; for an atom-indep
  // expansion every entry equals the analytic value.
  const expected = lnLogpdf(2.0, 0, 1);
  for (let i = 0; i < lp.samples.length; i++) {
    assert.ok(Math.abs(lp.samples[i] - expected) < 1e-10,
      'atom ' + i + ': got ' + lp.samples[i] + ', expected ' + expected);
  }
});

test('bijection: pushfwd(exp-bijection, Normal) density matches LogNormal at multiple points', async () => {
  // Check across the support — y = 0.5, 1.0, 1.5, 3.0.
  for (const yVal of [0.5, 1.0, 1.5, 3.0]) {
    const ctx = makeCtx(`
M = Normal(mu = 0.0, sigma = 1.0)
b = bijection(fn(exp(_)), fn(log(_)), fn(_))
LN = pushfwd(b, M)
lp = logdensityof(LN, ${yVal})
`);
    const lp = await ctx.getMeasure('lp');
    const expected = lnLogpdf(yVal, 0, 1);
    assert.ok(Math.abs(lp.samples[0] - expected) < 1e-10,
      `y=${yVal}: got ${lp.samples[0]}, expected ${expected}`);
  }
});

// =====================================================================
// Affine bijection — non-trivial Jacobian
// =====================================================================

test('bijection: pushfwd(2*_+1, _) on Normal(0,1) density matches Normal(1, 2) − log 2', async () => {
  // Forward f(x) = 2x + 1; inverse f_inv(y) = (y - 1)/2;
  // logvolume = log|f'| = log 2 (constant — volume-doubling).
  // Pushforward density at y: log p_N((y-1)/2; 0, 1) − log 2.
  // (Equivalent to Normal(1, 2): same density up to the constant
  // shift.)
  const ctx = makeCtx(`
M = Normal(mu = 0.0, sigma = 1.0)
b = bijection(fn(2.0 * _ + 1.0), fn((_ - 1.0) / 2.0), fn(log(2.0)))
T = pushfwd(b, M)
lp = logdensityof(T, 3.0)
`);
  const lp = await ctx.getMeasure('lp');
  // At y = 3.0: f_inv(y) = 1.0; base logpdf = N(1; 0, 1); subtract log 2.
  const expected = nLogpdf(1.0, 0, 1) - Math.log(2);
  assert.ok(Math.abs(lp.samples[0] - expected) < 1e-10,
    'got ' + lp.samples[0] + ', expected ' + expected);
});

test('bijection: pushfwd(2*_+1) with SCALAR logvolume slot accepts a constant', async () => {
  // logvolume can be a literal scalar per spec §06 ("`logvolume` may
  // be a function or a scalar (`0` for volume-preserving maps)").
  // We pass the constant log(2) ≈ 0.6931 directly.
  const ctx = makeCtx(`
M = Normal(mu = 0.0, sigma = 1.0)
b = bijection(fn(2.0 * _ + 1.0), fn((_ - 1.0) / 2.0), 0.6931471805599453)
T = pushfwd(b, M)
lp = logdensityof(T, 3.0)
`);
  const lp = await ctx.getMeasure('lp');
  const expected = nLogpdf(1.0, 0, 1) - Math.log(2);
  assert.ok(Math.abs(lp.samples[0] - expected) < 1e-10,
    'got ' + lp.samples[0] + ', expected ' + expected);
});

// =====================================================================
// Non-bijection pushfwd in density mode: clear error
// =====================================================================

test('density: pushfwd of a plain (non-bijection) function raises a clear error', async () => {
  // Sampling works for any f; density requires bijection metadata to
  // know the inverse + Jacobian. We expect a descriptive error.
  const ctx = makeCtx(`
M = Normal(mu = 0.0, sigma = 1.0)
T = pushfwd(fn(exp(_)), M)
lp = logdensityof(T, 1.5)
`);
  await assert.rejects(
    ctx.getMeasure('lp'),
    /requires a bijection annotation/);
});

// =====================================================================
// Sampling regression: pushfwd through a bijection samples like
// pushfwd through the plain forward function (transparency).
// =====================================================================

test('bijection: pushfwd through bijection samples like pushfwd through plain f', async () => {
  // Both `pushfwd(b, M)` and `pushfwd(fn(exp(_)), M)` should produce
  // samples drawn from LogNormal. Compare empirical means.
  const ctxA = makeCtx(`
M = Normal(mu = 0.0, sigma = 1.0)
b = bijection(fn(exp(_)), fn(log(_)), fn(_))
LN_a = pushfwd(b, M)
`, { sampleCount: 4096 });
  const ctxB = makeCtx(`
M = Normal(mu = 0.0, sigma = 1.0)
LN_b = pushfwd(fn(exp(_)), M)
`, { sampleCount: 4096 });
  const a = await ctxA.getMeasure('LN_a');
  const b = await ctxB.getMeasure('LN_b');
  // All positive (exp of reals).
  for (let i = 0; i < a.samples.length; i++) {
    assert.ok(a.samples[i] > 0, 'bijection-pushed sample non-positive: ' + a.samples[i]);
  }
  // Means within MC tolerance of each other and of analytic E[exp(N(0,1))] = √e.
  const meanA = a.samples.reduce((s: any, v: any) => s + v, 0) / a.samples.length;
  const meanB = b.samples.reduce((s: any, v: any) => s + v, 0) / b.samples.length;
  assert.ok(Math.abs(meanA - Math.exp(0.5)) < 0.1,
    'bijection-path mean off: got ' + meanA);
  assert.ok(Math.abs(meanB - Math.exp(0.5)) < 0.1,
    'plain-fn-path mean off: got ' + meanB);
});

// =====================================================================
// Bijection call transparency in value position
// =====================================================================

test('bijection: calling a bijection binding in value position evaluates the forward function', async () => {
  // `b(x)` should evaluate as `f(x)`. The inlineOnce branch for
  // bijection bindings rewrites the call to use the forward function;
  // here we observe it through a downstream binding.
  const ctx = makeCtx(`
b = bijection(fn(exp(_)), fn(log(_)), fn(_))
x = 1.5
y = b(x)
`);
  const y = await ctx.getMeasure('y');
  assert.ok(Math.abs(y.samples[0] - Math.exp(1.5)) < 1e-12,
    'b(1.5) should equal exp(1.5), got ' + y.samples[0]);
});
