'use strict';

// §04 always-splat enforcement. Spec §04 "Calling conventions":
//
//   "A sole positional record or table therefore always splats: whether
//    its field or column names match the callable's argument names
//    decides only whether the call is valid, never whether the splat
//    occurs. Passing a record or table as one ordinary argument requires
//    the keyword spelling, as in `f(pars = record(...))`."
//
// and, in the same bullet, "A call with field or column names that do not
// match the callable's argument names is a static error."
//
// Before enforcement the engine splatted only when the record's fields
// covered the callee's inputs and otherwise bound the whole record to
// input 0 with no name check — the name-conditioned reading §04 rules
// out. That fallback made `generator(pars)` against a one-input
// `generator` score a value, and made `Poisson(record(zzz = 0.5))` score
// NaN with no error at all.
//
// INDEPENDENT ORACLE — closed form, NOT an engine output:
//   Normal(mu = 1.1, sigma = 0.2) logpdf at x = 1.1 (the mean) is
//     -log(0.2 · √(2π)) = -log(0.2) - ½log(2π)
//                       = 1.6094379124341003 - 0.9189385332046727
//                       = 0.6904993792294276

const test = require('node:test');
const assert = require('node:assert');
const ENG = '../';
const { processSource, orchestrator, materialiser } = require(ENG + 'index.ts');
const { createWorkerHandler } = require(ENG + 'worker.ts');

const NORMAL_AT_MEAN = 0.6904993792294276;   // closed form; NOT an engine output

// A kernel with ONE input that is itself a record. Under always-splat the
// only valid call spelling is `generator(pars = <record>)`.
function transportSrc(callSpelling: string) {
  return `sigma = 0.2
pars = elementof(cartprod(a = reals, mu = reals))
x ~ Normal(pars.mu, sigma)
generator = kernelof(x, pars = pars)
gp = record(a = 0.1, mu = 1.1)
m = ${callSpelling}
ld = logdensityof(m, 1.1)`;
}

function errorsOf(src: string) {
  return processSource(src).diagnostics
    .filter((d: any) => d.severity === 'error')
    .map((d: any) => d.message);
}

function buildCtx(src: string, N: number, seed: number) {
  const proc = processSource(src);
  const built = orchestrator.buildDerivations(proc.bindings);
  const w = createWorkerHandler(); w.handle({ type: 'init', seed });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(), sampleCount: N,
    rootKey: seed, rootSeed: seed, marginalizationCount: 32,
    moduleRegistry: proc.loweredModule && proc.loweredModule.moduleRegistry,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx); cache.set(n, m); return m;
    },
    sendWorker: (m: any) => Promise.resolve(w.handle(m)),
  };
  return ctx;
}

test('always-splat: a sole positional record against a one-input callable is a §04 error', () => {
  const errs = errorsOf(transportSrc('generator(gp)'));
  const splat = errs.filter((m: string) => /no argument named/.test(m));
  assert.equal(splat.length, 1, 'expected one §04 splat-name error; got: ' + errs.join(' | '));
  // Both fields are reported, and the message names the keyword fix.
  assert.match(splat[0], /"a", "mu"/);
  assert.match(splat[0], /generator\(pars = …\)/);
});

test('always-splat: the invalid positional spelling yields NO value (fallback is gone)', async () => {
  // The whole-record positional fallback used to score this model
  // successfully, which is what hid the spec violation.
  const ctx = buildCtx(transportSrc('generator(gp)'), 1, 1);
  await assert.rejects(() => Promise.resolve(ctx.getMeasure('ld')), /no derivation/);
});

test('always-splat: the keyword spelling scores, matching the closed form', async () => {
  const src = transportSrc('generator(pars = gp)');
  assert.deepEqual(errorsOf(src), [], 'keyword spelling must be diagnostic-free');
  const ctx = buildCtx(src, 1, 1);
  const got = (await ctx.getMeasure('ld')).samples[0];
  assert.ok(Math.abs(got - NORMAL_AT_MEAN) < 1e-12,
    `logdensity ${got} ≈ closed form ${NORMAL_AT_MEAN}`);
});

test('always-splat: an inline record literal splats into a two-input kernel', async () => {
  // The other splat source at the call site: `record(...)` written inline
  // rather than referenced through a binding. Same closed-form oracle —
  // splatting a = 0.1 (unused by the body) and mu = 1.1.
  const src = `sigma = 0.2
a = elementof(reals)
mu = elementof(reals)
x ~ Normal(mu, sigma)
k = kernelof(x, a = a, mu = mu)
m = k(record(a = 0.1, mu = 1.1))
ld = logdensityof(m, 1.1)`;
  assert.deepEqual(errorsOf(src), []);
  const got = (await buildCtx(src, 1, 1).getMeasure('ld')).samples[0];
  assert.ok(Math.abs(got - NORMAL_AT_MEAN) < 1e-12,
    `logdensity ${got} ≈ closed form ${NORMAL_AT_MEAN}`);
});

test('always-splat: a record whose fields match the inputs still splats', () => {
  // The splat itself is unchanged for a matching record — enforcement
  // removes the *precondition*, not the feature.
  assert.deepEqual(errorsOf(`f = (a, b) -> a + b
r = record(a = 1.0, b = 2.0)
y = f(r)`), []);
});

test('always-splat: a surplus field is a §04 error, not a whole-record bind', () => {
  // Previously the surplus `c` failed the cover test, so the record bound
  // positionally to `a` and the user saw only a misleading
  // 'missing argument "b"'.
  const errs = errorsOf(`f = (a, b) -> a + b
r = record(a = 1.0, b = 2.0, c = 3.0)
y = f(r)`);
  const splat = errs.filter((m: string) => /no argument named/.test(m));
  assert.equal(splat.length, 1, 'got: ' + errs.join(' | '));
  assert.match(splat[0], /"c"/);
});

test('always-splat: a keyword-bound record is an ordinary value, never splatted', () => {
  // §04 excludes it explicitly, so a record bound by keyword to a
  // one-input callable must NOT be name-checked against its fields.
  assert.deepEqual(errorsOf(transportSrc('generator(pars = gp)')), []);
});

test('always-splat: a record alongside another argument is an ordinary value', () => {
  // §04 excludes this too — the record is not the call's SOLE argument,
  // so `restrict(M, x)`'s observation record is never splatted.
  assert.deepEqual(errorsOf(`mu = joint(a = Normal(0.0, 1.0), b = Normal(0.0, 1.0))
x = record(a = 0.5)
nu = restrict(mu, x)`), []);
});
