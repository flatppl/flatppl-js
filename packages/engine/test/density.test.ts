'use strict';

// =====================================================================
// density.js — log-density-with-consume/rest primitive.
// =====================================================================
//
// These tests exercise the per-IR-kind dispatch directly via
// hand-built IRs. End-to-end source-to-density coverage is the
// materialiser's job (matLogdensityof tests); here we pin the
// primitive's numeric correctness and its empty-rest invariant.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const density = require('../density.ts');

// Convenience IR constructors — keeps the asserts focused on the
// density math rather than IR plumbing.
function lit(v: any)        { return { kind: 'lit', value: v }; }
function refSelf(name: any) { return { kind: 'ref', ns: 'self', name }; }
function Normal(mu: any, sigma: any) {
  return { kind: 'call', op: 'Normal',
    kwargs: { mu: lit(mu), sigma: lit(sigma) } };
}
function Exponential(rate: any) {
  return { kind: 'call', op: 'Exponential', kwargs: { rate: lit(rate) } };
}
function callOp(op: any, args: any, fields?: any) {
  const ir: any = { kind: 'call', op };
  if (args)   ir.args   = args;
  if (fields) ir.fields = fields;
  return ir;
}

const LOG_TWO_PI = Math.log(2 * Math.PI);
const STD_NORMAL_LOGP_AT_ZERO = -0.5 * LOG_TWO_PI;

// =====================================================================
// Scalar leaf — Normal, Exponential
// =====================================================================

test('density: standard Normal at 0 matches -log(sqrt(2π))', () => {
  const logp = density.logDensity(Normal(0, 1), 0, {});
  assert.ok(Math.abs(logp - STD_NORMAL_LOGP_AT_ZERO) < 1e-12);
});

test('density: Normal(2, 3) at 5 matches stdlib logpdf form', () => {
  // logpdf = -log(σ√(2π)) - (x-μ)²/(2σ²)
  const mu = 2, sigma = 3, x = 5;
  const expected = -Math.log(sigma) - 0.5 * LOG_TWO_PI
    - (x - mu) * (x - mu) / (2 * sigma * sigma);
  const logp = density.logDensity(Normal(mu, sigma), x, {});
  assert.ok(Math.abs(logp - expected) < 1e-12);
});

test('density: scalar leaf consumes one entry from a vector head', () => {
  const r = density.logDensityConsume(Normal(0, 1), [0.0, 1.0, 2.0], {});
  assert.equal(r.rest && r.rest.length, 2);
  assert.equal(r.rest[0], 1.0);
});

test('density: scalar leaf with bare number consumes fully', () => {
  const r = density.logDensityConsume(Normal(0, 1), 0.0, {});
  assert.equal(r.rest, null);
});

// =====================================================================
// weighted / logweighted
// =====================================================================

test('density: weighted(0.5, Normal) at 0 adds log(0.5)', () => {
  const ir = callOp('weighted', [lit(0.5), Normal(0, 1)]);
  const logp = density.logDensity(ir, 0, {});
  assert.ok(Math.abs(logp - (STD_NORMAL_LOGP_AT_ZERO + Math.log(0.5))) < 1e-12);
});

test('density: weighted(0, M) → -Infinity', () => {
  const ir = callOp('weighted', [lit(0), Normal(0, 1)]);
  const logp = density.logDensity(ir, 0, {});
  assert.equal(logp, -Infinity);
});

test('density: logweighted(-1, Normal) at 0 adds -1 directly (no log call)', () => {
  const ir = callOp('logweighted', [lit(-1), Normal(0, 1)]);
  const logp = density.logDensity(ir, 0, {});
  assert.ok(Math.abs(logp - (STD_NORMAL_LOGP_AT_ZERO - 1)) < 1e-12);
});

// =====================================================================
// truncate(M, S) — indicator over S
// =====================================================================

test('density: truncate(Normal, posreals) at +0.5 keeps base density', () => {
  const ir = callOp('truncate', [Normal(0, 1), { kind: 'const', name: 'posreals' }]);
  const opts = { parseSet: (setIR: any) => ({ kind: 'posreals' }) };
  const logp = density.logDensity(ir, 0.5, {}, opts);
  const expected = -0.5 * LOG_TWO_PI - 0.5 * 0.5 / 2;
  assert.ok(Math.abs(logp - expected) < 1e-12);
});

test('density: truncate(Normal, posreals) at -0.5 returns -Infinity', () => {
  const ir = callOp('truncate', [Normal(0, 1), { kind: 'const', name: 'posreals' }]);
  const opts = { parseSet: (setIR: any) => ({ kind: 'posreals' }) };
  const logp = density.logDensity(ir, -0.5, {}, opts);
  assert.equal(logp, -Infinity);
});

// =====================================================================
// record / kwarg-joint
// =====================================================================

test('density: joint(a=N(0,1), b=N(0,1)) at {a:0,b:1} sums field logps', () => {
  const ir = callOp('joint', null, [
    { name: 'a', value: Normal(0, 1) },
    { name: 'b', value: Normal(0, 1) },
  ]);
  const logp = density.logDensity(ir, { a: 0, b: 1 }, {});
  const expected = STD_NORMAL_LOGP_AT_ZERO + (-0.5 * LOG_TWO_PI - 0.5);
  assert.ok(Math.abs(logp - expected) < 1e-12);
});

test('density: MIXED continuous-discrete joint multiplies Lebesgue × Counting refs', () => {
  // Spec §06: a product measure combines its components' reference
  // measures multiplicatively. A joint of a continuous (Lebesgue-ref)
  // and a discrete (Counting-ref) component is well-formed — its
  // density w.r.t. Lebesgue⊗Counting is the product (sum in log-space)
  // of the per-component densities. (Contrast superpose, where mixing
  // continuous⊕discrete is correctly refused — no common reference.)
  // Oracle: independent closed forms, not the engine's own output.
  const Poisson = (rate: any) =>
    ({ kind: 'call', op: 'Poisson', kwargs: { rate: lit(rate) } });
  const logfact = (n: any) => { let s = 0; for (let i = 2; i <= n; i++) s += Math.log(i); return s; };
  const lnNormal = (x: any, m: any, s: any) =>
    -0.5 * LOG_TWO_PI - Math.log(s) - 0.5 * ((x - m) / s) ** 2;
  const lnPoisson = (k: any, r: any) => k * Math.log(r) - r - logfact(k);

  const ir = callOp('joint', null, [
    { name: 'c', value: Normal(0, 1) },
    { name: 'k', value: Poisson(3) },
  ]);
  for (const [c, k] of [[0.5, 2], [0, 0], [-2, 1], [1.5, 5]]) {
    const logp = density.logDensity(ir, { c, k }, {});
    const oracle = lnNormal(c, 0, 1) + lnPoisson(k, 3);
    assert.ok(Math.abs(logp - oracle) < 1e-12,
      `joint density at (c=${c},k=${k}): engine ${logp} vs oracle ${oracle}`);
  }
});

test('density: kwarg-joint with missing field throws', () => {
  // Consuming 'a' empties the record; the next iteration can't find
  // 'b' because there's nothing left to consume from. Either error
  // shape is a clear shape-mismatch signal.
  const ir = callOp('joint', null, [
    { name: 'a', value: Normal(0, 1) },
    { name: 'b', value: Normal(0, 1) },
  ]);
  assert.throws(() => density.logDensity(ir, { a: 0 }, {}),
    /missing field 'b'|non-record value|exhausted/);
});

test('density: kwarg-joint with extra field surfaces as leftover rest', () => {
  const ir = callOp('joint', null, [
    { name: 'a', value: Normal(0, 1) },
  ]);
  assert.throws(() => density.logDensity(ir, { a: 0, extra: 99 }, {}),
    /unconsumed leftover/);
});

// =====================================================================
// Positional joint — consume in declared order
// =====================================================================

test('density: positional joint(N, N) at [0, 1] = sum of components', () => {
  const ir = callOp('joint', [Normal(0, 1), Normal(0, 1)]);
  const logp = density.logDensity(ir, [0, 1], {});
  const expected = STD_NORMAL_LOGP_AT_ZERO + (-0.5 * LOG_TWO_PI - 0.5);
  assert.ok(Math.abs(logp - expected) < 1e-12);
});

test('density: positional joint with mixed leaves consumes correctly', () => {
  // joint(Normal, Exponential) at [0.0, 2.0]
  const ir = callOp('joint', [Normal(0, 1), Exponential(1)]);
  const logp = density.logDensity(ir, [0.0, 2.0], {});
  // logpdf_Normal(0;0,1) + logpdf_Exp(2;1) = -log√(2π) + (log 1 − 2)
  const expected = STD_NORMAL_LOGP_AT_ZERO + (0 - 2);
  assert.ok(Math.abs(logp - expected) < 1e-12);
});

test('density: positional joint with leftover vector entries throws', () => {
  const ir = callOp('joint', [Normal(0, 1), Normal(0, 1)]);
  assert.throws(() => density.logDensity(ir, [0, 1, 999], {}),
    /unconsumed leftover/);
});

// =====================================================================
// iid(M, n) — n copies of M's footprint
// =====================================================================

test('density: iid(Normal, 3) at [0, 0, 0] = 3 × logp(0)', () => {
  const ir = callOp('iid', [Normal(0, 1), lit(3)]);
  const logp = density.logDensity(ir, [0, 0, 0], {});
  assert.ok(Math.abs(logp - 3 * STD_NORMAL_LOGP_AT_ZERO) < 1e-12);
});

test('density: iid count mismatch surfaces as leftover', () => {
  const ir = callOp('iid', [Normal(0, 1), lit(3)]);
  assert.throws(() => density.logDensity(ir, [0, 0], {}),
    /exhausted/);
  assert.throws(() => density.logDensity(ir, [0, 0, 0, 0], {}),
    /unconsumed leftover/);
});

// =====================================================================
// Composition — iid + joint, weighted + iid
// =====================================================================

test('density: joint(iid(N, 2), N) at [0, 0, 1] consumes 2 then 1', () => {
  const ir = callOp('joint', [
    callOp('iid', [Normal(0, 1), lit(2)]),
    Normal(0, 1),
  ]);
  const logp = density.logDensity(ir, [0, 0, 1], {});
  const expected = 2 * STD_NORMAL_LOGP_AT_ZERO
    + (-0.5 * LOG_TWO_PI - 0.5);
  assert.ok(Math.abs(logp - expected) < 1e-12);
});

test('density: weighted(c, iid(N, n)) propagates log(c) once', () => {
  const ir = callOp('weighted',
    [lit(0.25), callOp('iid', [Normal(0, 1), lit(2)])]);
  const logp = density.logDensity(ir, [0, 0], {});
  const expected = 2 * STD_NORMAL_LOGP_AT_ZERO + Math.log(0.25);
  assert.ok(Math.abs(logp - expected) < 1e-12);
});

// =====================================================================
// select — discrete-selector mixture (engine-concepts §11)
//   log p(x) = logsumexp_k ( logw_k + log p_{branch_k}(x) )
// =====================================================================

function sel(branches: any, logweights: any) {
  const ir: any = { kind: 'call', op: 'select', branches };
  ir.logweights = logweights || null;
  return ir;
}
function nLogp(x: any, mu: any, sigma: any) {
  return -Math.log(sigma) - 0.5 * LOG_TWO_PI
    - (x - mu) * (x - mu) / (2 * sigma * sigma);
}

test('density: select(null weights) = raw measure addition log Σ p_k (superpose)', () => {
  // superpose semantics: ν = M0 + M1 ⇒ p(x) = p0(x) + p1(x).
  const ir = sel([Normal(0, 1), Normal(4, 1)], null);
  const x = 1.0;
  const expected = Math.log(Math.exp(nLogp(x, 0, 1)) + Math.exp(nLogp(x, 4, 1)));
  const logp = density.logDensity(ir, x, {});
  assert.ok(Math.abs(logp - expected) < 1e-12, `got ${logp}, exp ${expected}`);
});

test('density: select with explicit logweights = log Σ w_k p_k (mixture)', () => {
  // 0.25·N(0,1) + 0.75·N(5,2) — the canonical 2-Gaussian mixture.
  const w0 = 0.25, w1 = 0.75, x = 1.3;
  const ir = sel([Normal(0, 1), Normal(5, 2)],
    [lit(Math.log(w0)), lit(Math.log(w1))]);
  const expected = Math.log(
    w0 * Math.exp(nLogp(x, 0, 1)) + w1 * Math.exp(nLogp(x, 5, 2)));
  const logp = density.logDensity(ir, x, {});
  assert.ok(Math.abs(logp - expected) < 1e-12, `got ${logp}, exp ${expected}`);
});

test('density: select equals weighted+superpose composition', () => {
  // select([N0,N1], [log a, log b]) ≡ logsumexp of weighted branches.
  const a = 2, b = 3, x = 0.7;
  const viaSelect = density.logDensity(
    sel([Normal(0, 1), Normal(2, 0.5)], [lit(Math.log(a)), lit(Math.log(b))]),
    x, {});
  const viaWeighted = density.logDensity(
    sel([callOp('weighted', [lit(a), Normal(0, 1)]),
      callOp('weighted', [lit(b), Normal(2, 0.5)])], null),
    x, {});
  assert.ok(Math.abs(viaSelect - viaWeighted) < 1e-12,
    `select=${viaSelect}, weighted+superpose=${viaWeighted}`);
});

test('density: single-branch select equals that branch', () => {
  const x = 1.7;
  const one = density.logDensity(sel([Normal(0.5, 2)], null), x, {});
  const bare = density.logDensity(Normal(0.5, 2), x, {});
  assert.ok(Math.abs(one - bare) < 1e-12);
});

test('density: select drops a -Infinity (zero-mass) branch', () => {
  // weighted(0, ·) collapses that branch to -Inf; logsumexp ignores it
  // ⇒ result equals the surviving branch's logp exactly.
  const x = 0.3;
  const ir = sel([callOp('weighted', [lit(0), Normal(0, 1)]), Normal(0, 1)], null);
  const logp = density.logDensity(ir, x, {});
  assert.ok(Math.abs(logp - nLogp(x, 0, 1)) < 1e-12, `got ${logp}`);
});

test('density: select all-zero-mass branches → -Infinity', () => {
  const ir = sel([callOp('weighted', [lit(0), Normal(0, 1)]),
    callOp('weighted', [lit(0), Normal(3, 1)])], null);
  assert.equal(density.logDensity(ir, 0.0, {}), -Infinity);
});

test('density: select logsumexp is numerically stable for far-apart branches', () => {
  // Branch logps differ by ~5000; naive exp() would overflow/underflow.
  // Stable logsumexp ⇒ result == max + log1p(exp(min-max)) ≈ max.
  const x = 0.0;
  const ir = sel([Normal(0, 1), Normal(100, 1)], null);
  const lpNear = nLogp(x, 0, 1);          // ≈ -0.919
  const lpFar  = nLogp(x, 100, 1);        // ≈ -5000.9
  const expected = Math.log(Math.exp(lpNear) + Math.exp(lpFar)); // == lpNear effectively
  const logp = density.logDensity(ir, x, {});
  assert.ok(Number.isFinite(logp) && Math.abs(logp - expected) < 1e-9
    && Math.abs(logp - lpNear) < 1e-9, `got ${logp}, exp ${expected}`);
});

test('density: select branch footprint mismatch throws', () => {
  // A scalar leaf vs a 2-component positional joint don't share a
  // variate space — must be rejected loudly.
  const ir = sel([Normal(0, 1), callOp('joint', [Normal(0, 1), Normal(0, 1)])],
    null);
  assert.throws(() => density.logDensity(ir, [0, 0], {}),
    /different observation footprints|share one variate space|leftover|exhausted/);
});

test('density: select consumes one scalar (rest threads to sibling)', () => {
  // positional joint( select(N,N), N ) at [x0, x1]: the select consumes
  // exactly one scalar so the trailing Normal scores x1.
  const ir = callOp('joint', [sel([Normal(0, 1), Normal(1, 1)], null), Normal(0, 1)]);
  const x0 = 0.2, x1 = -0.4;
  const selLogp = Math.log(Math.exp(nLogp(x0, 0, 1)) + Math.exp(nLogp(x0, 1, 1)));
  const expected = selLogp + nLogp(x1, 0, 1);
  const logp = density.logDensity(ir, [x0, x1], {});
  assert.ok(Math.abs(logp - expected) < 1e-12, `got ${logp}, exp ${expected}`);
});

// =====================================================================
// Measure refs — resolveMeasureRef callback
// =====================================================================

test('density: measure ref dispatch via resolveMeasureRef opt', () => {
  const ir = callOp('joint', [refSelf('Mref'), Normal(0, 1)]);
  const opts = {
    resolveMeasureRef: (name: any) => name === 'Mref' ? Normal(0, 1) : null,
  };
  const logp = density.logDensity(ir, [0, 0], {}, opts);
  assert.ok(Math.abs(logp - 2 * STD_NORMAL_LOGP_AT_ZERO) < 1e-12);
});

test('density: missing resolveMeasureRef opt with ref throws clearly', () => {
  const ir = callOp('joint', [refSelf('Mref'), Normal(0, 1)]);
  assert.throws(() => density.logDensity(ir, [0, 0], {}),
    /without resolveMeasureRef/);
});
