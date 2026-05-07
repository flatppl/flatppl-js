'use strict';

// Tests for engine/traceeval.js — the unified generative + scoring
// walker. The tests exercise both modes (sampling, scoring) on the
// same IRs and check that the four `tally` modes produce the
// expected log-density accumulation, that observed/unobserved
// dispatch works, and that all supported structural ops (joint,
// record, iid, weighted, logweighted) compose correctly.
//
// Reading note: this is the foundation primitive that bayesupdate
// and likelihoodof will lower onto, so the tests double as the
// behavioural spec — if you're looking for "what does logdensityof
// do for IR shape X", check the cases below.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const traceeval = require('../traceeval');
const samplerLib = require('../sampler');
const rng = require('../rng');

// Helpers — minimal IR builders so the tests read like specs.
const lit  = (v) => ({ kind: 'lit', value: v });
const ref  = (n) => ({ kind: 'ref', ns: 'self', name: n });
const dist = (op, kwargs) => {
  const out = {};
  for (const [k, v] of Object.entries(kwargs)) out[k] = (typeof v === 'object' && v !== null && v.kind) ? v : lit(v);
  return { kind: 'call', op, kwargs: out };
};
const joint = (fields) => ({
  kind: 'call', op: 'joint',
  fields: Object.entries(fields).map(([name, value]) => ({ name, value })),
});
const iid       = (M, n) => ({ kind: 'call', op: 'iid',         args: [M, lit(n)] });
const weighted  = (w, M) => ({ kind: 'call', op: 'weighted',    args: [(typeof w === 'object' && w.kind) ? w : lit(w), M] });
const logweight = (g, M) => ({ kind: 'call', op: 'logweighted', args: [(typeof g === 'object' && g.kind) ? g : lit(g), M] });

const STD_NORMAL = dist('Normal', { mu: 0, sigma: 1 });
const EXP1       = dist('Exponential', { rate: 1 });

// =====================================================================
// Sampling mode (observed=undefined) — unchanged from rand() semantics.
// =====================================================================

test('walk: sampling with tally=none produces samples, logp stays 0', () => {
  const r = traceeval.walk(rng.stateFromKey(7), STD_NORMAL, {}, undefined, { tally: 'none' });
  assert.equal(typeof r.value, 'number');
  assert.equal(Number.isFinite(r.value), true);
  assert.equal(r.logp, 0);
});

test('walk: sampling with tally=all also accumulates logpdf at the drawn value', () => {
  const r = traceeval.walk(rng.stateFromKey(7), STD_NORMAL, {}, undefined, { tally: 'all' });
  // logp should equal stdlib logpdf at the drawn x, exactly.
  const logpdf = require('@stdlib/stats-base-dists-normal-logpdf');
  assert.equal(r.logp, logpdf(r.value, 0, 1));
});

test('walk: sampling reproducibility — same state + same IR → same value & logp', () => {
  const a = traceeval.walk(rng.stateFromKey(11), STD_NORMAL, {}, undefined, { tally: 'all' });
  const b = traceeval.walk(rng.stateFromKey(11), STD_NORMAL, {}, undefined, { tally: 'all' });
  assert.equal(a.value, b.value);
  assert.equal(a.logp, b.logp);
});

// =====================================================================
// Scoring mode (observed=value) — no RNG advance, logp = logpdf.
// =====================================================================

test('walk: scoring a leaf at an observed value gives logpdf(x | params)', () => {
  const x = 1.5;
  const seed = rng.stateFromKey(99);
  const r = traceeval.walk(seed, STD_NORMAL, {}, x, { tally: 'all' });
  assert.equal(r.value, x);
  const logpdf = require('@stdlib/stats-base-dists-normal-logpdf');
  assert.equal(r.logp, logpdf(x, 0, 1));
  // RNG state untouched when nothing is sampled.
  assert.deepEqual(r.state, seed);
});

test('walk: scoring outside support yields -Infinity (Exponential at x<0)', () => {
  const r = traceeval.walk(rng.stateFromKey(1), EXP1, {}, -2, { tally: 'all' });
  assert.equal(r.logp, -Infinity);
});

test('walk: tally=none with an observed value still echoes value but logp=0', () => {
  // 'none' means "don't bother computing the density"; useful when the
  // caller only wants to verify the trace shape.
  const r = traceeval.walk(rng.stateFromKey(1), STD_NORMAL, {}, 0.5, { tally: 'none' });
  assert.equal(r.value, 0.5);
  assert.equal(r.logp, 0);
});

// =====================================================================
// Param resolution from env — refs in kwargs.
// =====================================================================

test('walk: refs in distribution params resolve from env', () => {
  const M = dist('Normal', { mu: ref('mu'), sigma: ref('sigma') });
  const r = traceeval.walk(rng.stateFromKey(1), M, { mu: 5, sigma: 0.001 }, undefined, { tally: 'all' });
  assert.ok(Math.abs(r.value - 5) < 0.01, 'tight sigma should center around mu');
  // Density at the drawn value should be high (close to mode).
  assert.ok(r.logp > 0); // peak of N(5, 0.001) has logpdf > 0.
});

// =====================================================================
// joint / record — field-wise recursion.
// =====================================================================

test('walk: joint scoring sums per-field logpdfs (product reference)', () => {
  const M = joint({ a: STD_NORMAL, b: EXP1 });
  const obs = { a: 0.0, b: 0.5 };
  const r = traceeval.walk(rng.stateFromKey(1), M, {}, obs, { tally: 'all' });
  assert.deepEqual(r.value, obs);
  const lpA = require('@stdlib/stats-base-dists-normal-logpdf')(0.0, 0, 1);
  const lpB = require('@stdlib/stats-base-dists-exponential-logpdf')(0.5, 1);
  assert.equal(r.logp, lpA + lpB);
});

test('walk: joint with one field clamped, one sampled (tally=clamped)', () => {
  // Mode used by likelihoodof: latents free, observation clamped.
  const M = joint({ latent: STD_NORMAL, obs: dist('Normal', { mu: ref('latent_drawn'), sigma: 1 }) });
  // We can't easily hand-clamp latent's drawn value, so for this test
  // just clamp obs and let latent sample. With tally='clamped' only
  // obs's logpdf goes into the tally.
  const Msimple = joint({ latent: STD_NORMAL, obs: dist('Normal', { mu: 0, sigma: 1 }) });
  const r = traceeval.walk(rng.stateFromKey(3), Msimple, {}, { obs: 0.7 }, { tally: 'clamped' });
  assert.equal(r.value.obs, 0.7);
  // latent was sampled — its value is finite, but its logpdf is NOT
  // in the tally because tally='clamped' only counts observed sites.
  assert.equal(Number.isFinite(r.value.latent), true);
  const lpObs = require('@stdlib/stats-base-dists-normal-logpdf')(0.7, 0, 1);
  assert.equal(r.logp, lpObs);
});

test('walk: joint sampling with all unobserved fills every field', () => {
  const M = joint({ a: STD_NORMAL, b: EXP1 });
  const r = traceeval.walk(rng.stateFromKey(5), M, {}, undefined, { tally: 'none' });
  assert.equal(typeof r.value.a, 'number');
  assert.equal(typeof r.value.b, 'number');
  assert.ok(r.value.b >= 0, 'Exponential is non-negative');
});

// =====================================================================
// iid — n shared-param draws.
// =====================================================================

test('walk: iid sampling produces n values', () => {
  const M = iid(STD_NORMAL, 5);
  const r = traceeval.walk(rng.stateFromKey(2), M, {}, undefined, { tally: 'none' });
  assert.equal(r.value.length, 5);
  for (const v of r.value) assert.equal(Number.isFinite(v), true);
});

test('walk: iid scoring sums n logpdfs', () => {
  const M = iid(STD_NORMAL, 3);
  const obs = [0.0, 1.0, -1.0];
  const r = traceeval.walk(rng.stateFromKey(2), M, {}, obs, { tally: 'all' });
  assert.deepEqual(r.value, obs);
  const lp = require('@stdlib/stats-base-dists-normal-logpdf');
  assert.equal(r.logp, lp(0, 0, 1) + lp(1, 0, 1) + lp(-1, 0, 1));
});

test('walk: iid observed length must match n', () => {
  const M = iid(STD_NORMAL, 3);
  assert.throws(() => traceeval.walk(rng.stateFromKey(1), M, {}, [0, 1], { tally: 'all' }),
    /observed length 2 does not match count 3/);
});

// =====================================================================
// weighted / logweighted — additive log-shifts.
// =====================================================================

test('walk: weighted(0.5, M) adds log(0.5) to the tally', () => {
  const M = weighted(0.5, STD_NORMAL);
  const r = traceeval.walk(rng.stateFromKey(1), M, {}, 0.0, { tally: 'all' });
  const lpBase = require('@stdlib/stats-base-dists-normal-logpdf')(0, 0, 1);
  assert.equal(r.logp, Math.log(0.5) + lpBase);
});

test('walk: weighted(0, M) yields -Infinity (zero-mass measure)', () => {
  const M = weighted(0, STD_NORMAL);
  const r = traceeval.walk(rng.stateFromKey(1), M, {}, 0.0, { tally: 'all' });
  assert.equal(r.logp, -Infinity);
});

test('walk: weighted with negative weight throws', () => {
  const M = weighted(-1, STD_NORMAL);
  assert.throws(() => traceeval.walk(rng.stateFromKey(1), M, {}, 0.0, { tally: 'all' }),
    /weight must be non-negative/);
});

test('walk: logweighted(g, M) adds g directly', () => {
  const M = logweight(-2.5, STD_NORMAL);
  const r = traceeval.walk(rng.stateFromKey(1), M, {}, 0.0, { tally: 'all' });
  const lpBase = require('@stdlib/stats-base-dists-normal-logpdf')(0, 0, 1);
  assert.equal(r.logp, -2.5 + lpBase);
});

test('walk: logweighted(-Infinity, M) gives -Infinity (sentinel for zero mass)', () => {
  const M = logweight({ kind: 'const', name: 'inf' }, STD_NORMAL);
  // Build using neg(inf). inf is an evaluable const in evaluateExpr.
  const Mneg = logweight({ kind: 'call', op: 'neg', args: [{ kind: 'const', name: 'inf' }] }, STD_NORMAL);
  const r = traceeval.walk(rng.stateFromKey(1), Mneg, {}, 0.0, { tally: 'all' });
  assert.equal(r.logp, -Infinity);
});

// =====================================================================
// Composition — bayesupdate-style: outer logweighted of a likelihood
// applied to a clamped observation and free latent.
// =====================================================================

test('walk: scoring joint(latent, iid obs) with obs clamped is the per-atom likelihood', () => {
  // model: latent ~ N(0,1); obs ~ iid Normal(mu=latent, sigma=1) of length 4
  // For a fixed env-supplied latent value, the likelihood at obs is
  //   sum_j logpdf_Normal(obs[j] | latent, 1)
  const M = joint({
    latent: STD_NORMAL,
    obs: iid(dist('Normal', { mu: ref('theta'), sigma: 1 }), 4),
  });
  const obsArr = [1.0, 0.5, 1.5, -0.5];
  const env = { theta: 1.0 };
  const r = traceeval.walk(rng.stateFromKey(1), M, env,
    { obs: obsArr }, { tally: 'clamped' });
  const lp = require('@stdlib/stats-base-dists-normal-logpdf');
  const expected = obsArr.reduce((s, x) => s + lp(x, env.theta, 1), 0);
  assert.equal(r.logp, expected);
});

// =====================================================================
// Self-ref to other measure bindings via resolveMeasureRef.
// =====================================================================

test('walk: resolveMeasureRef dereferences self-refs in measure positions', () => {
  // joint with a field that points at a "named" measure binding.
  const namedM = STD_NORMAL;
  const M = joint({ a: ref('mybinding') });
  const r = traceeval.walk(rng.stateFromKey(1), M, {}, { a: 0 }, {
    tally: 'all',
    resolveMeasureRef: (name) => name === 'mybinding' ? namedM : null,
  });
  const lp = require('@stdlib/stats-base-dists-normal-logpdf');
  assert.equal(r.logp, lp(0, 0, 1));
});

test('walk: self-ref without resolveMeasureRef raises a clear error', () => {
  const M = joint({ a: ref('not_inlined') });
  assert.throws(() => traceeval.walk(rng.stateFromKey(1), M, {}, { a: 0 }, { tally: 'all' }),
    /no resolveMeasureRef was supplied/);
});

// =====================================================================
// Error surface.
// =====================================================================

test('walk: unknown op gives a clear "not a measure expression" error', () => {
  const bogus = { kind: 'call', op: 'totalmass', args: [STD_NORMAL] };
  assert.throws(() => traceeval.walk(rng.stateFromKey(1), bogus, {}, 0, { tally: 'all' }),
    /not a measure expression we can sample or score/);
});

test('walk: invalid tally option is rejected up front', () => {
  assert.throws(() => traceeval.walk(rng.stateFromKey(1), STD_NORMAL, {}, 0, { tally: 'wat' }),
    /opts.tally must be/);
});
