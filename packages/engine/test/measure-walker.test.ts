'use strict';

// Tests for the in-module measure walker (sampler.walk; was traceeval.ts,
// folded into sampler.ts in §11). Density evaluation lives in
// density.ts (see test/density.test.ts); the cases below cover only the
// sampling primitive: leaf draws, env-resolved distribution params,
// joint / record / iid structural recursion, weighted / logweighted
// refusal, lawof / draw unwrapping, and resolveMeasureRef
// dereferencing. All assertions are self-relative (same-seed walk-vs-walk
// identity) or distributional, so they hold regardless of which batched
// leaf realisation the endpoint uses.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler = require('../sampler.ts');
const rng = require('../rng.ts');

// Helpers — minimal IR builders so the tests read like specs.
const lit  = (v: any) => ({ kind: 'lit', value: v });
const ref  = (n: any) => ({ kind: 'ref', ns: 'self', name: n });
const dist = (op: any, kwargs: any) => {
  const out: any = {};
  for (const [k, v] of Object.entries(kwargs)) out[k] = (typeof v === 'object' && v !== null && (v as any).kind) ? v : lit(v);
  return { kind: 'call', op, kwargs: out };
};
const joint = (fields: any) => ({
  kind: 'call', op: 'joint',
  fields: Object.entries(fields).map(([name, value]) => ({ name, value })),
});
const iid       = (M: any, n: any) => ({ kind: 'call', op: 'iid',         args: [M, lit(n)] });
const weighted  = (w: any, M: any) => ({ kind: 'call', op: 'weighted',    args: [(typeof w === 'object' && w.kind) ? w : lit(w), M] });
const logweight = (g: any, M: any) => ({ kind: 'call', op: 'logweighted', args: [(typeof g === 'object' && g.kind) ? g : lit(g), M] });

const STD_NORMAL = dist('Normal', { mu: 0, sigma: 1 });
const EXP1       = dist('Exponential', { rate: 1 });

// =====================================================================
// Leaf sampling
// =====================================================================

test('walk: leaf produces a finite numeric sample', () => {
  const r = sampler.walk(rng.stateFromKey(7), STD_NORMAL, {});
  assert.equal(typeof r.value, 'number');
  assert.equal(Number.isFinite(r.value), true);
});

test('walk: leaf reproducibility — same state + same IR → same value', () => {
  const a = sampler.walk(rng.stateFromKey(11), STD_NORMAL, {});
  const b = sampler.walk(rng.stateFromKey(11), STD_NORMAL, {});
  assert.equal(a.value, b.value);
});

test('walk: refs in distribution params resolve from env', () => {
  const M = dist('Normal', { mu: ref('mu'), sigma: ref('sigma') });
  const r = sampler.walk(rng.stateFromKey(1), M, { mu: 5, sigma: 0.001 });
  assert.ok(Math.abs(r.value - 5) < 0.01, 'tight sigma should center around mu');
});

// =====================================================================
// joint / record — field-wise recursion
// =====================================================================

test('walk: joint sampling fills every field with a finite numeric draw', () => {
  const M = joint({ a: STD_NORMAL, b: EXP1 });
  const r = sampler.walk(rng.stateFromKey(5), M, {});
  assert.equal(typeof r.value.a, 'number');
  assert.equal(typeof r.value.b, 'number');
  assert.ok(r.value.b >= 0, 'Exponential is non-negative');
});

test('walk: positional joint sampling produces an array of per-component draws', () => {
  const M = { kind: 'call', op: 'joint', args: [STD_NORMAL, EXP1] };
  const r = sampler.walk(rng.stateFromKey(5), M, {});
  assert.equal(Array.isArray(r.value), true);
  assert.equal(r.value.length, 2);
  assert.equal(typeof r.value[0], 'number');
  assert.ok(r.value[1] >= 0, 'second component (Exponential) non-negative');
});

// =====================================================================
// iid — n shared-param draws
// =====================================================================

test('walk: iid sampling produces n values', () => {
  const M = iid(STD_NORMAL, 5);
  const r = sampler.walk(rng.stateFromKey(2), M, {});
  assert.equal(r.value.length, 5);
  for (const v of r.value) assert.equal(Number.isFinite(v), true);
});

test('walk: iid count may reference an env binding', () => {
  const M = { kind: 'call', op: 'iid', args: [STD_NORMAL, ref('n')] };
  const r = sampler.walk(rng.stateFromKey(2), M, { n: 4 });
  assert.equal(r.value.length, 4);
});

// =====================================================================
// weighted / logweighted — rand refuses (spec §07 sec:random, §13):
// a weighted measure is not a probability measure unless renormalized,
// so sampling its base and dropping the weight is a wrong answer, not a
// legal shortcut.
// =====================================================================

test('walk: weighted refuses to sample rather than dropping the weight', () => {
  const M = weighted(0.5, STD_NORMAL);
  assert.throws(
    () => sampler.walk(rng.stateFromKey(1), M, {}),
    /weighted.*cannot be sampled|cannot be sampled.*weighted/,
  );
});

test('walk: logweighted refuses to sample rather than dropping the weight', () => {
  const M = logweight(-2.5, STD_NORMAL);
  assert.throws(
    () => sampler.walk(rng.stateFromKey(1), M, {}),
    /logweighted.*cannot be sampled|cannot be sampled.*logweighted/,
  );
});

// normalize — spec §06 "given a measure M with finite total mass
// Z = totalmass(M) > 0, returns the probability measure M / Z", and §07
// sec:random makes `rand` legal on any probability measure. `normalize` now
// has a MEASURE_OP_WALKERS entry: it samples the IDENTITY case (base already
// %normalized) directly, and refuses any other base with an honest
// diagnostic — known normalization is not the same as samplable, since
// sampling M/Z in general needs reweighting (importance/rejection) this
// sampler does not implement. normalize(weighted(...)) renormalizes back to
// a probability measure, so per §07 `rand` in principle allows it, but
// mass known ≠ samplable: this stays refused, now for that honest reason
// rather than the generic "not a measure we can sample" wall normalize used
// to hit before it had any MEASURE_OP_WALKERS entry at all.
test('walk: normalize(weighted(...)) refuses via the honest normalize diagnostic', () => {
  const M = { kind: 'call', op: 'normalize', args: [weighted(0.5, STD_NORMAL)] };
  assert.throws(
    () => sampler.walk(rng.stateFromKey(1), M, {}),
    /'normalize' cannot be sampled here.*requires reweighting/,
  );
});

test('walk: normalize(logweighted(...)) refuses via the honest normalize diagnostic (same wall)', () => {
  const M = { kind: 'call', op: 'normalize', args: [logweight(-1.0, STD_NORMAL)] };
  assert.throws(
    () => sampler.walk(rng.stateFromKey(1), M, {}),
    /'normalize' cannot be sampled here.*requires reweighting/,
  );
});

// Identity case: normalize(M) where M is already a probability measure is a
// no-op (Z = 1) — sampling M directly IS sampling normalize(M). Moment
// check against the closed-form oracle (Standard Normal: mean 0, var 1)
// rather than pinning specific draws, since the walker legitimately
// delegates to the leaf sampler underneath.
test('walk: normalize(iid(Normal(0,1), N)) samples the identity case, moments match the closed-form oracle', () => {
  const M = { kind: 'call', op: 'normalize', args: [iid(STD_NORMAL, 20000)] };
  const r = sampler.walk(rng.stateFromKey(42), M, {});
  const xs = r.value;
  const n = xs.length;
  const mean = xs.reduce((a: number, b: number) => a + b, 0) / n;
  const varr = xs.reduce((a: number, b: number) => a + (b - mean) * (b - mean), 0) / n;
  assert.ok(Math.abs(mean) < 0.05, `mean ${mean} should be ~0 (Standard Normal)`);
  assert.ok(Math.abs(varr - 1) < 0.05, `var ${varr} should be ~1 (Standard Normal)`);
});

// A single leaf normalize(Normal(0,1)) also identity-samples (no iid
// wrapper) — the plain leaf branch of the identity check.
test('walk: normalize(Normal(0,1)) (bare leaf) samples the identity case', () => {
  const M = { kind: 'call', op: 'normalize', args: [STD_NORMAL] };
  const r = sampler.walk(rng.stateFromKey(1), M, {});
  assert.equal(typeof r.value, 'number');
  assert.equal(Number.isFinite(r.value), true);
});

// Nested normalize(normalize(M)) is trivially identity too — normalize's
// own output is always normalized once it type-checked.
test('walk: normalize(normalize(Normal(0,1))) samples the identity case (nested normalize)', () => {
  const M = { kind: 'call', op: 'normalize', args: [{ kind: 'call', op: 'normalize', args: [STD_NORMAL] }] };
  const r = sampler.walk(rng.stateFromKey(1), M, {});
  assert.equal(typeof r.value, 'number');
  assert.equal(Number.isFinite(r.value), true);
});

// A base the walker cannot PROVE is already normalized (e.g. an
// unrecognised op) refuses via the same honest diagnostic — the identity
// check is a conservative allow-list, not "refuse only weighted/logweighted".
test('walk: normalize(<unprovable base>) refuses via the honest diagnostic too', () => {
  const M = { kind: 'call', op: 'normalize', args: [{ kind: 'call', op: 'superpose', args: [STD_NORMAL, EXP1] }] };
  assert.throws(
    () => sampler.walk(rng.stateFromKey(1), M, {}),
    /'normalize' cannot be sampled here.*requires reweighting/,
  );
});

// =====================================================================
// lawof / draw — identity wrappers per spec §06
// =====================================================================

test('walk: lawof(M) samples as M (identity)', () => {
  const M = { kind: 'call', op: 'lawof', args: [STD_NORMAL] };
  const r = sampler.walk(rng.stateFromKey(1), M, {});
  const baseline = sampler.walk(rng.stateFromKey(1), STD_NORMAL, {});
  assert.equal(r.value, baseline.value);
});

test('walk: draw(M) at measure position unwraps to M', () => {
  // The orchestrator usually canonicalises draw out; this branch is the
  // safety net for inline forms the walker sees pre-canonicalisation.
  const M = { kind: 'call', op: 'draw', args: [STD_NORMAL] };
  const r = sampler.walk(rng.stateFromKey(1), M, {});
  const baseline = sampler.walk(rng.stateFromKey(1), STD_NORMAL, {});
  assert.equal(r.value, baseline.value);
});

// =====================================================================
// Self-ref to other measure bindings via resolveMeasureRef
// =====================================================================

test('walk: resolveMeasureRef dereferences self-refs in measure positions', () => {
  const namedM = STD_NORMAL;
  const M = joint({ a: ref('mybinding') });
  const r = sampler.walk(rng.stateFromKey(1), M, {}, {
    resolveMeasureRef: (name: any) => name === 'mybinding' ? namedM : null,
  });
  assert.equal(typeof r.value.a, 'number');
});

test('walk: self-ref without resolveMeasureRef raises a clear error', () => {
  const M = joint({ a: ref('not_inlined') });
  assert.throws(() => sampler.walk(rng.stateFromKey(1), M, {}),
    /no resolveMeasureRef was supplied/);
});

// =====================================================================
// Error surface
// =====================================================================

test('walk: unknown op gives a clear "not a measure expression" error', () => {
  const bogus = { kind: 'call', op: 'totalmass', args: [STD_NORMAL] };
  assert.throws(() => sampler.walk(rng.stateFromKey(1), bogus, {}),
    /not a measure expression we can sample/);
});

// =====================================================================
// Single leaf endpoint (§11): a scalar leaf draw and the first
// element of a 1-element iid draw of the same leaf are bit-for-bit equal
// — both route through the one batched leaf endpoint (sampleLeafN), with
// no separate ziggurat scalar realisation. Pins the leaf-collapse.
// =====================================================================

test('walk: rand(state, Normal) === rand(state, iid(Normal, 1))[0] (one leaf endpoint)', () => {
  const scalar = sampler.walk(rng.stateFromKey(21), STD_NORMAL, {});
  const single = sampler.walk(rng.stateFromKey(21), iid(STD_NORMAL, 1), {});
  assert.equal(Array.isArray(single.value), true);
  assert.equal(single.value.length, 1);
  assert.equal(scalar.value, single.value[0],
    'scalar leaf draw and 1-element iid leaf draw share the single batched endpoint');
});
