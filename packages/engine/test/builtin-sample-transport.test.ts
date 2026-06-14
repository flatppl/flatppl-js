'use strict';

// =====================================================================
// FlatPDL primitives: builtin_sample + the four canonical transports
// (spec §07 §sec:measure-eval-prims).
// =====================================================================
//
// Coverage:
//
//  1. `builtin_sample(rngstate, kernel, kernel_input, [n, m, ...])` —
//     synthesises a kernel call (optionally wrapped in iid) and threads
//     RNG state through the measure walker (sampler.walk). The cross-engine ABI: same kernel
//     + same input + same root state ⇒ same samples. We check
//     equivalence with `rand(state, Kernel(kw))` and confirm
//     state-threading produces independent successive draws.
//
//  2. `builtin_touniform` / `builtin_fromuniform` /
//     `builtin_tonormal` / `builtin_fromnormal` — checked via:
//       a. round-trips on continuous univariate kernels (Normal,
//          Exponential, Beta).
//       b. consistency relations from spec §07:
//            touniform ≡ invprobit ∘ tonormal
//            tonormal  ≡ probit   ∘ touniform   (+ the two inverses)
//       c. MvNormal direct form: tonormal(x) = L⁻¹(x − μ),
//          fromnormal(z) = μ + L·z.
//       d. discrete kernels (Bernoulli, Poisson) → static-error refusal.
//       e. multivariate-but-not-MvNormal (Dirichlet, Wishart) →
//          "not yet implemented" refusal.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const densityPrims = require('../density-prims.ts');
const sampler = require('../sampler.ts');
const engine = require('../index.ts');

// Convenience IR builders for the helper-level tests below.
function lit(v: any)        { return { kind: 'lit', value: v }; }
function refSelf(name: any) { return { kind: 'ref', ns: 'self', name }; }
function recordIR(fields: Record<string, any>) {
  const fs: any[] = [];
  for (const k in fields) fs.push({ name: k, value: fields[k] });
  return { kind: 'call', op: 'record', fields: fs };
}

// =====================================================================
// builtin_sample — surface evaluation
// =====================================================================

test('builtin_sample(Normal, record(...), n) matches rand+iid path bit-for-bit', () => {
  // Use processSource so the parser + lowerer + evaluator path is
  // exercised end-to-end. We seed an rngstate with rnginit, draw via
  // builtin_sample, and compare to the equivalent rand(... iid ...).
  const src = `
flatppl_compat = "0.1"

state = rnginit([0, 1, 2, 3])
xs1, _ = builtin_sample(state, Normal, record(mu = 0.0, sigma = 1.0), 8)
xs2, _ = rand(state, iid(Normal(mu = 0.0, sigma = 1.0), 8))
`;
  const r = engine.processSource(src);
  assert.deepEqual(r.diagnostics.filter((d: any) =>
    d.severity === 'error'), []);
  const orchestrator = require('../orchestrator.ts');
  const derivs = orchestrator.buildDerivations(r.bindings, r.loweredModule);
  const xs1 = derivs.fixedValues.get('xs1');
  const xs2 = derivs.fixedValues.get('xs2');
  assert.ok(Array.isArray(xs1) || (xs1 && xs1.data),
    'xs1 should be a vector (got ' + JSON.stringify(xs1) + ')');
  const xs1Arr = Array.isArray(xs1) ? xs1 : Array.from(xs1.data);
  const xs2Arr = Array.isArray(xs2) ? xs2 : Array.from(xs2.data);
  for (let i = 0; i < xs1Arr.length; i++) {
    assert.ok(Math.abs(xs1Arr[i] - xs2Arr[i]) < 1e-15,
      `atom ${i}: builtin_sample=${xs1Arr[i]} vs rand+iid=${xs2Arr[i]}`);
  }
});

test('builtin_sample accepts a record-typed BINDING as kernel_input (not just inline)', () => {
  // Spec §07: kernel_input is a record matching the kernel's kwarg
  // interface — nothing restricts it to a syntactic `record(...)`
  // literal. A ref to a record-typed binding resolves to the same kwargs
  // and must draw bit-for-bit identically to the inline form.
  const src = `
flatppl_compat = "0.1"

state = rnginit([0, 1, 2, 3])
pars = record(mu = 0.0, sigma = 1.0)
xs1, _ = builtin_sample(state, Normal, pars, 8)
xs2, _ = rand(state, iid(Normal(mu = 0.0, sigma = 1.0), 8))
`;
  const r = engine.processSource(src);
  assert.deepEqual(r.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const orchestrator = require('../orchestrator.ts');
  const derivs = orchestrator.buildDerivations(r.bindings, r.loweredModule);
  const xs1 = derivs.fixedValues.get('xs1');
  const xs2 = derivs.fixedValues.get('xs2');
  const xs1Arr = Array.isArray(xs1) ? xs1 : Array.from(xs1.data);
  const xs2Arr = Array.isArray(xs2) ? xs2 : Array.from(xs2.data);
  assert.equal(xs1Arr.length, 8);
  for (let i = 0; i < xs1Arr.length; i++) {
    assert.ok(Math.abs(xs1Arr[i] - xs2Arr[i]) < 1e-15,
      `atom ${i}: record-binding=${xs1Arr[i]} vs inline-iid=${xs2Arr[i]}`);
  }
});

test('builtin_sample (no dims) returns a scalar + threaded state', () => {
  const src = `
flatppl_compat = "0.1"

s0 = rnginit([4, 5, 6, 7])
x, s1 = builtin_sample(s0, Normal, record(mu = 0.0, sigma = 1.0))
y, s2 = builtin_sample(s1, Normal, record(mu = 0.0, sigma = 1.0))
`;
  const r = engine.processSource(src);
  assert.deepEqual(r.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const orchestrator = require('../orchestrator.ts');
  const derivs = orchestrator.buildDerivations(r.bindings, r.loweredModule);
  const x = derivs.fixedValues.get('x');
  const y = derivs.fixedValues.get('y');
  assert.ok(typeof x === 'number', 'x should be a scalar');
  assert.ok(typeof y === 'number', 'y should be a scalar');
  assert.notEqual(x, y, 'two consecutive draws under threaded state should differ');
});

test('builtin_sample with multi-dim trailing args produces a rank-≥2 Value', () => {
  const src = `
flatppl_compat = "0.1"

s0 = rnginit([1, 2, 3, 4])
A, _ = builtin_sample(s0, Normal, record(mu = 0.0, sigma = 1.0), 3, 4)
`;
  const r = engine.processSource(src);
  const orchestrator = require('../orchestrator.ts');
  const derivs = orchestrator.buildDerivations(r.bindings, r.loweredModule);
  const A = derivs.fixedValues.get('A');
  assert.ok(A && A.shape, 'A should be a shape-explicit Value');
  assert.deepEqual(A.shape, [3, 4]);
  assert.equal(A.data.length, 12);
});

// =====================================================================
// Transports — univariate continuous round-trips and consistency
// =====================================================================

test('touniform / fromuniform round-trip: Normal', () => {
  const ki = { mu: 1.5, sigma: 2.5 };
  for (const x of [-2.0, -0.3, 0.0, 1.0, 5.7]) {
    const u = densityPrims.builtinTouniform('Normal', ki, x);
    const xr = densityPrims.builtinFromuniform('Normal', ki, u);
    assert.ok(Math.abs(xr - x) < 1e-9,
      `Normal touniform/fromuniform round-trip at ${x}: got ${xr}`);
  }
});

test('touniform / fromuniform round-trip: Exponential', () => {
  const ki = { rate: 0.7 };
  for (const x of [0.1, 1.0, 3.0, 10.0]) {
    const u = densityPrims.builtinTouniform('Exponential', ki, x);
    const xr = densityPrims.builtinFromuniform('Exponential', ki, u);
    assert.ok(Math.abs(xr - x) < 1e-9,
      `Exponential round-trip at ${x}: got ${xr}`);
  }
});

test('touniform / fromuniform round-trip: Beta', () => {
  const ki = { alpha: 2.0, beta: 5.0 };
  for (const x of [0.1, 0.3, 0.5, 0.7, 0.9]) {
    const u = densityPrims.builtinTouniform('Beta', ki, x);
    const xr = densityPrims.builtinFromuniform('Beta', ki, u);
    assert.ok(Math.abs(xr - x) < 1e-9,
      `Beta round-trip at ${x}: got ${xr}`);
  }
});

test('tonormal / fromnormal round-trip: Normal', () => {
  const ki = { mu: 1.5, sigma: 2.5 };
  for (const x of [-2.0, -0.3, 0.0, 1.0, 5.7]) {
    const z = densityPrims.builtinTonormal('Normal', ki, x);
    const xr = densityPrims.builtinFromnormal('Normal', ki, z);
    assert.ok(Math.abs(xr - x) < 1e-9,
      `Normal tonormal/fromnormal round-trip at ${x}: got ${xr}`);
  }
});

test('consistency: tonormal ≡ probit ∘ touniform (Normal)', () => {
  const ki = { mu: 0.0, sigma: 1.0 };
  const { _normalQuantile, _normalCdf } = densityPrims._internal;
  for (const x of [-1.0, 0.0, 0.5, 2.0]) {
    const z1 = densityPrims.builtinTonormal('Normal', ki, x);
    const z2 = _normalQuantile(densityPrims.builtinTouniform('Normal', ki, x));
    assert.ok(Math.abs(z1 - z2) < 1e-9,
      `Normal consistency at x=${x}: tonormal=${z1}, probit∘touniform=${z2}`);
    // Also touniform ≡ invprobit ∘ tonormal.
    const u1 = densityPrims.builtinTouniform('Normal', ki, x);
    const u2 = _normalCdf(densityPrims.builtinTonormal('Normal', ki, x));
    assert.ok(Math.abs(u1 - u2) < 1e-9);
  }
});

test('Normal touniform is the standard CDF (Φ((x-μ)/σ))', () => {
  const u = densityPrims.builtinTouniform('Normal',
    { mu: 0.0, sigma: 1.0 }, 0.0);
  assert.ok(Math.abs(u - 0.5) < 1e-12, `Φ(0)=½ expected, got ${u}`);
});

// =====================================================================
// Transports — MvNormal multivariate (direct form)
// =====================================================================

test('MvNormal tonormal / fromnormal: round-trip', () => {
  const mu = [1.0, -2.0];
  const cov = { shape: [2, 2], data: new Float64Array([2.0, 0.3, 0.3, 1.5]) };
  const ki = { mu, cov };
  const x = { shape: [2], data: new Float64Array([0.5, -1.0]) };
  const z = densityPrims.builtinTonormal('MvNormal', ki, x);
  assert.deepEqual(z.shape, [2]);
  const xr = densityPrims.builtinFromnormal('MvNormal', ki, z);
  for (let i = 0; i < 2; i++) {
    assert.ok(Math.abs(xr.data[i] - x.data[i]) < 1e-12,
      `coord ${i}: got ${xr.data[i]}, expected ${x.data[i]}`);
  }
});

test('MvNormal touniform / fromuniform: round-trip', () => {
  const mu = [0.0, 0.0];
  const cov = { shape: [2, 2], data: new Float64Array([1.0, 0.0, 0.0, 1.0]) };
  const ki = { mu, cov };
  const x = { shape: [2], data: new Float64Array([0.7, -0.3]) };
  const u = densityPrims.builtinTouniform('MvNormal', ki, x);
  const xr = densityPrims.builtinFromuniform('MvNormal', ki, u);
  for (let i = 0; i < 2; i++) {
    assert.ok(Math.abs(xr.data[i] - x.data[i]) < 1e-9);
  }
});

test('MvNormal tonormal: identity cov, zero mu ⇒ identity map', () => {
  const ki = { mu: [0.0, 0.0],
    cov: { shape: [2, 2], data: new Float64Array([1, 0, 0, 1]) } };
  const x = { shape: [2], data: new Float64Array([1.3, -0.7]) };
  const z = densityPrims.builtinTonormal('MvNormal', ki, x);
  assert.ok(Math.abs(z.data[0] - 1.3) < 1e-12);
  assert.ok(Math.abs(z.data[1] - (-0.7)) < 1e-12);
});

// =====================================================================
// Refusals
// =====================================================================

test('discrete kernels refuse all four transports (spec §07)', () => {
  for (const fn of ['Touniform', 'Fromuniform', 'Tonormal', 'Fromnormal']) {
    const f = densityPrims['builtin' + fn];
    for (const k of ['Bernoulli', 'Poisson']) {
      assert.throws(() => f(k, { p: 0.5, rate: 1.0 }, 0),
        new RegExp(`'${k}' is discrete`),
        `${fn} on ${k} should refuse`);
    }
  }
});

test('multivariate-non-MvNormal kernels report not-yet-implemented', () => {
  for (const fn of ['Touniform', 'Fromuniform', 'Tonormal', 'Fromnormal']) {
    const f = densityPrims['builtin' + fn];
    assert.throws(() => f('Dirichlet', { alpha: [1, 1, 1] },
      { shape: [3], data: new Float64Array([0.3, 0.3, 0.4]) }),
      /not yet implemented/);
    assert.throws(() => f('LKJ', { n: 2, eta: 1.0 },
      { shape: [2, 2], data: new Float64Array([1, 0, 0, 1]) }),
      /not yet implemented/);
  }
});

// =====================================================================
// Surface evaluation through processSource for the transports
// =====================================================================

test('surface: builtin_touniform(Normal, ..., 0.0) ≡ 0.5', () => {
  const src = `
flatppl_compat = "0.1"

u = builtin_touniform(Normal, record(mu = 0.0, sigma = 1.0), 0.0)
`;
  const r = engine.processSource(src);
  assert.deepEqual(r.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const orchestrator = require('../orchestrator.ts');
  const derivs = orchestrator.buildDerivations(r.bindings, r.loweredModule);
  const u = derivs.fixedValues.get('u');
  assert.ok(Math.abs(u - 0.5) < 1e-12, `Φ(0)=½ expected, got ${u}`);
});
