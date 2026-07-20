'use strict';

// =====================================================================
// builtin_logdensityof — FlatPDL primitive (spec §07 §sec:measure-eval-prims).
// =====================================================================
//
// Two-level coverage:
//  1. JS helpers `builtinLogdensityof` / `builtinLogdensityofPositional` —
//     direct dispatch into the per-kernel formulae (the cross-engine ABI).
//  2. Surface evaluation through processSource: a fixed-phase binding
//     `lp = builtin_logdensityof(Normal, (mu=0, sigma=1), 0)` lowers to
//     a kernel-name string lit + record kwargs + scalar variate; the
//     orchestrator evaluates it via sampler.evaluateCall.
//
// The numeric oracles are the same closed-form formulae the per-kernel
// `walk*` functions in density.ts use today — verifying that the
// primitive computes the SAME numbers as the existing leaf walker is
// the conformance contract on the refactor that routes density.ts
// through this primitive (engine-concepts §13.6).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const densityPrims = require('../density-prims.ts');
const engine = require('../index.ts');

// =====================================================================
// Univariate kernels via the helpers
// =====================================================================

const LOG_2PI = Math.log(2 * Math.PI);
const STD_NORMAL_LOGP_AT_ZERO = -0.5 * LOG_2PI;

test('builtinLogdensityof: standard Normal at 0', () => {
  const lp = densityPrims.builtinLogdensityof('Normal', { mu: 0, sigma: 1 }, 0);
  assert.ok(Math.abs(lp - STD_NORMAL_LOGP_AT_ZERO) < 1e-12);
});

test('builtinLogdensityof: Normal(2, 3) at 5 matches stdlib closed form', () => {
  const mu = 2, sigma = 3, x = 5;
  const expected = -Math.log(sigma) - 0.5 * LOG_2PI
    - (x - mu) * (x - mu) / (2 * sigma * sigma);
  const lp = densityPrims.builtinLogdensityof('Normal', { mu, sigma }, x);
  assert.ok(Math.abs(lp - expected) < 1e-12);
});

test('builtinLogdensityof: Exponential(rate=2) at 0.5', () => {
  const lp = densityPrims.builtinLogdensityof('Exponential', { rate: 2.0 }, 0.5);
  // log(λ) - λ x
  const expected = Math.log(2.0) - 2.0 * 0.5;
  assert.ok(Math.abs(lp - expected) < 1e-12);
});

test('builtinLogdensityof: Bernoulli(0.3) at 1', () => {
  const lp = densityPrims.builtinLogdensityof('Bernoulli', { p: 0.3 }, 1);
  assert.ok(Math.abs(lp - Math.log(0.3)) < 1e-12);
});

test('builtinLogdensityof: Uniform({lo:0, hi:2}) density is -log(2) on support', () => {
  const lp = densityPrims.builtinLogdensityof('Uniform', { support: { lo: 0, hi: 2 } }, 1);
  assert.ok(Math.abs(lp - (-Math.log(2))) < 1e-12);
});

test('builtinLogdensityof: Uniform with [lo,hi]-array support also works', () => {
  const lp = densityPrims.builtinLogdensityof('Uniform', { support: [0, 2] }, 1);
  assert.ok(Math.abs(lp - (-Math.log(2))) < 1e-12);
});

test('builtinLogdensityofPositional: matches REGISTRY.logpdfFn directly', () => {
  // The positional form skips the param-extraction step; should match
  // the record form numerically.
  const a = densityPrims.builtinLogdensityof('Normal', { mu: 1.5, sigma: 2.5 }, 3.0);
  const b = densityPrims.builtinLogdensityofPositional('Normal', [1.5, 2.5], 3.0);
  assert.ok(Math.abs(a - b) < 1e-15);
});

test('builtinLogdensityof: bare-scalar kernel_input == record form (Poisson)', () => {
  // spec §07: kernel_input is "a valid kernel input value" for
  // kernel(kernel_input) — a positional application Poisson(rate) is
  // valid, so a bare scalar is spec-legal, not just a record.
  const rate = 2.5, k = 3;
  const expected = k * Math.log(rate) - rate - Math.log(6); // ln(3!) = ln 6
  const viaRecord = densityPrims.builtinLogdensityof('Poisson', { rate }, k);
  const viaBare = densityPrims.builtinLogdensityof('Poisson', rate, k);
  assert.ok(Math.abs(viaRecord - expected) < 1e-12);
  assert.ok(Math.abs(viaBare - expected) < 1e-12);
});

test('builtinLogdensityof: bare positional list == record form (Normal)', () => {
  const viaRecord = densityPrims.builtinLogdensityof('Normal', { mu: 0.0, sigma: 1.0 }, 0.0);
  const viaPositional = densityPrims.builtinLogdensityof('Normal', [0.0, 1.0], 0.0);
  assert.ok(Math.abs(viaRecord - viaPositional) < 1e-15);
});

test('builtinLogdensityof: unknown kernel name throws', () => {
  assert.throws(() => densityPrims.builtinLogdensityof('NotARealKernel', {}, 0),
    /unknown kernel/);
});

test('builtinLogdensityof: missing kwarg throws', () => {
  assert.throws(() => densityPrims.builtinLogdensityof('Normal', { mu: 0 }, 0),
    /missing param 'sigma'/);
});

// =====================================================================
// Multivariate kernels via the helpers
// =====================================================================

test('builtinLogdensityof: MvNormal standard 2-D at origin', () => {
  // log p = -½ n log(2π) - ½ log|cov| - ½ (x-μ)ᵀ cov⁻¹ (x-μ)
  //       = -log(2π)  (n=2, cov=I, x=0, μ=0)
  const mu = [0, 0];
  const cov = { shape: [2, 2], data: new Float64Array([1, 0, 0, 1]) };
  const x = { shape: [2], data: new Float64Array([0, 0]) };
  const lp = densityPrims.builtinLogdensityof('MvNormal', { mu, cov }, x);
  assert.ok(Math.abs(lp - (-LOG_2PI)) < 1e-12);
});

test('builtinLogdensityof: Dirichlet(α=[1,1,1]) on simplex centre', () => {
  // Uniform on stdsimplex(3): log p = log Γ(3) - 3·log Γ(1) = log 2
  const lp = densityPrims.builtinLogdensityof('Dirichlet',
    { alpha: [1.0, 1.0, 1.0] },
    { shape: [3], data: new Float64Array([1/3, 1/3, 1/3]) });
  assert.ok(Math.abs(lp - Math.log(2)) < 1e-12);
});

test('builtinLogdensityof: Multinomial(n=4, p=[.5,.3,.2]) at [2,1,1]', () => {
  // log Γ(5) - log Γ(3) - log Γ(2) - log Γ(2)
  //  + 2 log .5 + 1 log .3 + 1 log .2
  const stdlibGammaln = require('@stdlib/math-base-special-gammaln');
  const expected = stdlibGammaln(5) - stdlibGammaln(3)
    - stdlibGammaln(2) - stdlibGammaln(2)
    + 2 * Math.log(0.5) + Math.log(0.3) + Math.log(0.2);
  const lp = densityPrims.builtinLogdensityof('Multinomial',
    { n: 4, p: [0.5, 0.3, 0.2] },
    { shape: [3], data: new Float64Array([2, 1, 1]) });
  assert.ok(Math.abs(lp - expected) < 1e-12);
});

test('builtinLogdensityof: LKJ/LKJCholesky density divides by c_n(eta)', () => {
  // Oracle: density = det(C)^(eta-1) / c_n(eta), confirmed against Distributions.jl
  // and numpyro. At eta=1 (n=2) the density is uniform over rho in (-1,1) => 1/2 =>
  // log(1/2) = -0.6931 (NOT +0.6931 — the pre-fix value multiplied by c_n). See
  // flatppl-design#43.
  const C2 = (off: any) => ({ shape: [2, 2], data: new Float64Array([1, off, off, 1]) });
  const C3 = { shape: [3, 3], data: new Float64Array([1, 0.3, 0.2, 0.3, 1, 0.1, 0.2, 0.1, 1]) };
  const lkj = (n: any, eta: any, x: any) => densityPrims.builtinLogdensityof('LKJ', { n, eta }, x);
  assert.ok(Math.abs(lkj(2, 1.0, C2(0.4)) - (-0.6931471805599453)) < 1e-9, 'LKJ n=2 eta=1 (uniform)');
  assert.ok(Math.abs(lkj(2, 2.0, C2(0.4)) - (-0.46203545959655856)) < 1e-9, 'LKJ n=2 eta=2');
  assert.ok(Math.abs(lkj(3, 2.0, C3) - (-0.7524491932002857)) < 1e-9, 'LKJ n=3 eta=2');
  // LKJCholesky of C2(0.4): L = [[1,0],[0.4, sqrt(1-0.16)]]
  const s = Math.sqrt(1 - 0.16);
  const Lfac = { shape: [2, 2], data: new Float64Array([1, 0, 0.4, s]) };
  const lpc = densityPrims.builtinLogdensityof('LKJCholesky', { n: 2, eta: 2.0 }, Lfac);
  assert.ok(Math.abs(lpc - (-0.46203545959655856)) < 1e-9, 'LKJCholesky n=2 eta=2 matches LKJ');
});

// =====================================================================
// Surface evaluation through processSource (fixed-phase binding)
// =====================================================================

test('surface: builtin_logdensityof(Normal, record(...), 0) ≡ -½log(2π)', () => {
  const src = `
flatppl_compat = "0.1"

lp = builtin_logdensityof(Normal, record(mu = 0.0, sigma = 1.0), 0.0)
`;
  const r = engine.processSource(src);
  // The binding should be fixed-phase. We pull its evaluated value
  // through the orchestrator's fixed-phase pre-eval.
  const orchestrator = require('../orchestrator.ts');
  const derivs = orchestrator.buildDerivations(r.bindings, r.loweredModule);
  const lp = derivs.fixedValues.get('lp');
  assert.ok(typeof lp === 'number',
    `lp should be a number, got ${JSON.stringify(lp)}`);
  assert.ok(Math.abs(lp - STD_NORMAL_LOGP_AT_ZERO) < 1e-12);
});

// =====================================================================
// Rank-d (nested) kernel-broadcast density — the determiniser's
// flattened nested-iid lowering.
// =====================================================================
//
// `logdensityof(iid(iid(Normal(0,1),2),3), x)` (spec §06: recursing
// through nesting → Σ over ALL leaves) determinises to a SINGLE
// `sum(broadcast(builtin_logdensityof, K, params, obs))` — no
// `functionof`/`get0` unroll (flatppl-rust `crates/determinizer`
// `lower_iid`'s flatten path). For a d≥2-nested iid, params/obs are
// rank-d bracket literals (`mu = [[0.0]]` against a rank-2 `[3,2]`
// obs); those literals carry an outerRank tag (spec §03 "vectors of
// vectors are not matrices"), which the generic per-cell broadcast
// loop would otherwise mis-read as "pass this block whole per cell"
// instead of "every bracket level is an iid axis". Both cases below
// score the SAME FlatPDL surface shape flatppl-rust actually emits
// (verified against a real `flatppl determinize` run); the rank-1
// case pins the pre-existing (unaffected) path, the rank-2/rank-3
// cases pin the fix. Oracles: independent scipy/hand closed-form
// Σ norm.logpdf(x, 0, 1) over the flattened observations.
test('surface: rank-1 (simple iid) kernel-broadcast density — unaffected baseline', () => {
  const src = `
flatppl_compat = "0.1"

data = [0.5, -0.3, 1.2]
lp = sum(builtin_logdensityof.(Normal, broadcast(record, mu = [0.0], sigma = [1.0]), data))
`;
  const r = engine.processSource(src);
  const orchestrator = require('../orchestrator.ts');
  const derivs = orchestrator.buildDerivations(r.bindings, r.loweredModule);
  const lp = derivs.fixedValues.get('lp');
  assert.ok(typeof lp === 'number', `lp should be a number, got ${JSON.stringify(lp)}`);
  // scipy: sum(norm.logpdf(v,0,1) for v in [0.5,-0.3,1.2]) = -3.646815599614018
  assert.ok(Math.abs(lp - (-3.646815599614018)) < 1e-9, `got ${lp}`);
});

test('surface: rank-2 (twice-nested iid) kernel-broadcast density', () => {
  const src = `
flatppl_compat = "0.1"

data = [[0.5, -0.3], [1.2, 0.1], [-0.7, 0.9]]
lp = sum(builtin_logdensityof.(Normal, broadcast(record, mu = [[0.0]], sigma = [[1.0]]), data))
`;
  const r = engine.processSource(src);
  const orchestrator = require('../orchestrator.ts');
  const derivs = orchestrator.buildDerivations(r.bindings, r.loweredModule);
  const lp = derivs.fixedValues.get('lp');
  assert.ok(typeof lp === 'number', `lp should be a number, got ${JSON.stringify(lp)}`);
  // scipy: sum(norm.logpdf(v,0,1) for row in data for v in row)
  //      = -7.058631199228036
  assert.ok(Math.abs(lp - (-7.058631199228036)) < 1e-9, `got ${lp}`);
});

test('surface: rank-3 (thrice-nested iid) kernel-broadcast density', () => {
  const src = `
flatppl_compat = "0.1"

data = [[[0.5, -0.3], [1.2, 0.1]], [[-0.7, 0.9], [0.2, -1.1]]]
lp = sum(builtin_logdensityof.(Normal, broadcast(record, mu = [[[0.0]]], sigma = [[[1.0]]]), data))
`;
  const r = engine.processSource(src);
  const orchestrator = require('../orchestrator.ts');
  const derivs = orchestrator.buildDerivations(r.bindings, r.loweredModule);
  const lp = derivs.fixedValues.get('lp');
  assert.ok(typeof lp === 'number', `lp should be a number, got ${JSON.stringify(lp)}`);
  // hand closed-form: sum(-0.5*log(2*pi) - 0.5*v*v for v in the 8 leaves)
  //      = -9.521508265637381
  assert.ok(Math.abs(lp - (-9.521508265637381)) < 1e-9, `got ${lp}`);
});

test('surface: lowerer rejects non-distribution kernel arg, records lowerError', () => {
  const src = `
flatppl_compat = "0.1"

lp = builtin_logdensityof(not_a_kernel, record(mu = 0.0, sigma = 1.0), 0.0)
`;
  // pir.lowerToModule catches lowerer exceptions and stores them on the
  // resulting IR's `lowerError` field (analyzer keeps processing). We
  // check that an error mentioning the bad kernel arg made it there.
  const r = engine.processSource(src);
  const lpBinding = r.loweredModule.bindings.get('lp');
  const err = lpBinding && lpBinding.rhs && lpBinding.rhs.lowerError;
  assert.ok(err && /builtin_logdensityof/.test(err),
    `expected lowerError mentioning builtin_logdensityof, got: ${err}`);
});
