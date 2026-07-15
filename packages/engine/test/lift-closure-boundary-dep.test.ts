'use strict';

// Buffy #265 — reproduces the lift.ts fn-inline closure-duplication bug.
//
// `computeClosure` (lift.ts ~2616-2647), when inlining a user-function call,
// unconditionally COPIES every reachable stochastic/parameterized free
// variable a fn body closes over (any var that is not the fn's own
// parameter and not fixed-phase) into a fresh, independent synthetic
// binding subgraph — even when that variable has no dependency on the call
// argument. The inlined body then references a SECOND, uncorrelated draw of
// the same prior (e.g. `mu` in the cases below), which a scoring point's
// boundary override (`kernelof`'s `mu = mu`) never patches, since the
// override only reaches the ORIGINAL `mu` binding. Result: a silent,
// large, finite wrong density for any model that calls a user-defined
// function whose body closes over a stochastic outer variable that is not
// one of the function's own parameters.
//
// Both cases below assert fn-form == direct-form (the correct, post-fix
// target, independently oracle-verified via Distributions.jl — see Buffy
// #265). Pre-fix they FAIL: the fn-form silently mis-scores.
//
// A third case is a must-stay-green PASSTHROUGH guard: a fn whose body uses
// only its own formal parameter (substituted with the call argument) and a
// fixed-phase constant (shared, never copied). It verifies that inlining
// keeps fixed-phase constants shared and substitutes formal parameters
// correctly — the paths `computeClosure`'s rewrite must leave untouched. It
// is NOT a "must-copy" case: for this model the closure copy-set is empty
// (the only body names are the boundary `par` and the fixed `c`).
//
// The must-COPY coverage lives in cases 4/5's DIRECT arms: `ydir` / `mixdir`
// reference the stochastic `mu` kernelof boundary, so their lifted anons ARE
// copied and boundary-substituted. If a fix broke boundary-dependent
// copying, the direct-forms would stop hitting their oracle targets and
// those assertions would fail — see the per-case notes below.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { ctxFor } = require('./_ctx-factory.ts');

const TOL = 1e-9;

async function scoreOf(src: string) {
  const { ctx } = ctxFor(src, 1);
  const m = await ctx.getMeasure('__score__');
  const s: Float64Array | null = m.samples ?? (m.value && m.value.data) ?? null;
  if (!s || s.length === 0) {
    throw new Error('scoreOf: __score__ produced no data (measure shape unexpected)');
  }
  return s[0];
}

// ---------------------------------------------------------------------
// Case 5 (minimal, no mixture) — isolates the closure-duplication defect
// from any normalize/mixture machinery. `yfn` closes over the stochastic
// `mu` (not one of its own parameters); the direct binding `ydir` uses the
// same `mu` verbatim.
//
// The DIRECT arm here doubles as must-COPY coverage: `ydir = Normal(mu[1],1)`
// references the `mu` kernelof boundary, so its lifted anons are copied and
// boundary-substituted. If boundary-dependent copying broke, this arm would
// stop equalling the oracle target and the assertion below would fail.
// ---------------------------------------------------------------------

const CASE5_TARGET = -6.1896926660233635;

const CASE5_DIRECT = `
N = 5
y_data = [0.1, -0.2, 0.3, 1.0, -1.0]
mu ~ iid(Normal(0.0, 5.0), 2)
ydir = Normal(mu[1], 1.0)
y ~ iid(ydir, N)
forward_kernel = kernelof(record(y = y), mu = mu)
L = likelihoodof(forward_kernel, record(y = y_data))
__score__ = logdensityof(L, record(mu = [0.5, -0.5]))
`;

const CASE5_FN = `
N = 5
y_data = [0.1, -0.2, 0.3, 1.0, -1.0]
mu ~ iid(Normal(0.0, 5.0), 2)
yfn(dummy) = Normal(mu[1] + dummy, 1.0)
y ~ iid(yfn(0.0), N)
forward_kernel = kernelof(record(y = y), mu = mu)
L = likelihoodof(forward_kernel, record(y = y_data))
__score__ = logdensityof(L, record(mu = [0.5, -0.5]))
`;

test('#265 case 5 (minimal): fn-form closing over stochastic mu must equal direct-form', async () => {
  const dir = await scoreOf(CASE5_DIRECT);
  const fn = await scoreOf(CASE5_FN);

  assert.ok(
    Math.abs(dir - CASE5_TARGET) <= TOL,
    `direct-form ${dir} does not match the oracle-verified target ${CASE5_TARGET}`
  );
  // Pre-fix this FAILS: computeClosure duplicates `mu`'s whole subgraph into
  // a fresh, independent draw that the mu=[0.5,-0.5] override never patches,
  // yielding a large, finite wrong density (documented pre-fix value:
  // -37.2544880877822).
  assert.ok(
    Math.abs(fn - dir) <= TOL,
    `fn-form (${fn}) must equal direct-form (${dir}) — lift.ts closure-duplication bug (#265)`
  );
});

// ---------------------------------------------------------------------
// Case 4 (mixture behind a fn) — same defect, surfacing through a
// normalize(superpose(...)) mixture wrapped in a named function whose body
// closes over the stochastic `mu` (the shape that originally surfaced the
// bug via a BAT.jl model port). `theta` is itself passed as an explicit fn
// argument (both weights), so only `mu` is the boundary-independent
// closure capture under test. As in case 5, the DIRECT arm (`mixdir`
// referencing the `mu` boundary) doubles as must-COPY coverage.
// ---------------------------------------------------------------------

const CASE4_TARGET = -6.032014419862954;

const CASE4_DIRECT = `
N = 5
y_data = [0.1, -0.2, 0.3, 1.0, -1.0]
theta ~ Beta(2, 2)
mu ~ iid(Normal(0, 5), 2)
mixdir = normalize(superpose(weighted(theta, Normal(mu[1], 1)), weighted(theta, Normal(mu[2], 1))))
y ~ iid(mixdir, N)
forward_kernel = kernelof(record(y = y), theta = theta, mu = mu)
L = likelihoodof(forward_kernel, record(y = y_data))
__score__ = logdensityof(L, record(theta = 0.5, mu = [0.5, -0.5]))
`;

const CASE4_FN = `
N = 5
y_data = [0.1, -0.2, 0.3, 1.0, -1.0]
theta ~ Beta(2, 2)
mu ~ iid(Normal(0, 5), 2)
mixfn(wB, wS) = normalize(superpose(weighted(wB, Normal(mu[1], 1)), weighted(wS, Normal(mu[2], 1))))
y ~ iid(mixfn(theta, theta), N)
forward_kernel = kernelof(record(y = y), theta = theta, mu = mu)
L = likelihoodof(forward_kernel, record(y = y_data))
__score__ = logdensityof(L, record(theta = 0.5, mu = [0.5, -0.5]))
`;

test('#265 case 4 (mixture behind a fn): fn-form closing over stochastic mu must equal direct-form', async () => {
  const dir = await scoreOf(CASE4_DIRECT);
  const fn = await scoreOf(CASE4_FN);

  assert.ok(
    Math.abs(dir - CASE4_TARGET) <= TOL,
    `direct-form ${dir} does not match the oracle-verified (theta-independent) target ${CASE4_TARGET}`
  );
  // Pre-fix this FAILS the same way as case 5 (documented pre-fix value:
  // -69.14816301508702), confirming the defect is not normalize-specific.
  assert.ok(
    Math.abs(fn - dir) <= TOL,
    `fn-form (${fn}) must equal direct-form (${dir}) — lift.ts closure-duplication bug (#265)`
  );
});

// ---------------------------------------------------------------------
// Must-stay-green PASSTHROUGH guard (the `packages/web/demo/feature-test1.flatppl`
// shape). `f_a`'s body uses only its own formal parameter `par` (substituted
// with the call argument `beta1`) and the fixed-phase constant `c` (shared,
// never copied). For this model the closure copy-set is EMPTY — the only
// body names are the boundary `par` and the fixed `c`, so no copy path
// fires. It is therefore NOT a must-copy case; it verifies that inlining
// keeps fixed-phase constants shared and substitutes formal parameters
// correctly (both paths the computeClosure rewrite must leave untouched).
// Must score identically to the manually-inlined form before and after the
// fix. (Must-copy coverage is in cases 4/5's DIRECT arms — see above.)
// ---------------------------------------------------------------------

const GUARD_FN = `
c = 2.5
beta1 ~ Normal(0.0, 1.0)
f_a(par) = c * par
a = f_a(beta1)
M = Normal(a, 1.0)
y_obs = 0.3
forward_kernel = kernelof(record(y = M), beta1 = beta1)
L = likelihoodof(forward_kernel, record(y = y_obs))
__score__ = logdensityof(L, record(beta1 = 0.7))
`;

const GUARD_INLINED = `
c = 2.5
beta1 ~ Normal(0.0, 1.0)
aInlined = c * beta1
M = Normal(aInlined, 1.0)
y_obs = 0.3
forward_kernel = kernelof(record(y = M), beta1 = beta1)
L = likelihoodof(forward_kernel, record(y = y_obs))
__score__ = logdensityof(L, record(beta1 = 0.7))
`;

test('#265 guard: passthrough inline (f_a(par) = c * par) shares fixed c + substitutes par correctly (must not regress)', async () => {
  const fn = await scoreOf(GUARD_FN);
  const inlined = await scoreOf(GUARD_INLINED);

  assert.ok(
    Math.abs(fn - inlined) <= TOL,
    `fn-form (${fn}) must equal its manually-inlined form (${inlined}) — fixed-share + parameter-substitution passthrough, must not regress`
  );
});
