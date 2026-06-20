'use strict';

// =====================================================================
// optimizer-polish.test.ts
// =====================================================================
//
// The local polish: batched central finite-difference gradient + a
// parallel line search (one batch of step candidates), iterated a few
// times to snap onto a smooth peak, then a finite-difference Hessian at
// the optimum → the Laplace covariance Σ = (−H)⁻¹. Cheap and near-exact
// on closed-form objectives; each gradient / Hessian / line search is one
// `evalCloud` batch. Tested against quadratics with known curvature.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { fdGradient, fdHessian, laplaceCovariance, polish } = require('../optimizer/polish.ts');

test('fdGradient matches the analytic gradient of a quadratic', async () => {
  // f(z) = -(2(z0-1)² + 3(z1+2)²); ∇f = (-4(z0-1), -6(z1+2))
  const f = async (cloud) => cloud.map(([a, b]) => -(2 * (a - 1) ** 2 + 3 * (b + 2) ** 2));
  const g = await fdGradient(f, [0.5, 0.0]);
  assert.ok(Math.abs(g[0] - 2) < 1e-3, `g0=${g[0]}`);   // -4(0.5-1) = 2
  assert.ok(Math.abs(g[1] + 12) < 1e-3, `g1=${g[1]}`);  // -6(0+2) = -12
});

test('fdHessian matches the analytic Hessian (incl. off-diagonal)', async () => {
  // f = -2a² - 3b² - ab; H = [[-4,-1],[-1,-6]]
  const f = async (cloud) => cloud.map(([a, b]) => -(2 * a * a + 3 * b * b + a * b));
  const H = await fdHessian(f, [0.3, -0.4]);
  assert.ok(Math.abs(H[0][0] + 4) < 1e-2, `H00=${H[0][0]}`);
  assert.ok(Math.abs(H[1][1] + 6) < 1e-2, `H11=${H[1][1]}`);
  assert.ok(Math.abs(H[0][1] + 1) < 1e-2, `H01=${H[0][1]}`);
  assert.ok(Math.abs(H[0][1] - H[1][0]) < 1e-12, 'symmetric');
});

test('laplaceCovariance inverts the negative Hessian when concave', () => {
  const { covariance, ok } = laplaceCovariance([[-4, 0], [0, -6]]);
  assert.ok(ok);
  assert.ok(Math.abs(covariance[0][0] - 0.25) < 1e-9, `${covariance[0][0]}`);
  assert.ok(Math.abs(covariance[1][1] - 1 / 6) < 1e-9, `${covariance[1][1]}`);
});

test('laplaceCovariance refuses a non-concave Hessian (not a maximum)', () => {
  const { ok } = laplaceCovariance([[4, 0], [0, -6]]); // indefinite → saddle
  assert.equal(ok, false);
});

test('polish climbs to the optimum and returns the Laplace covariance', async () => {
  // f(z) = -(z0² + 4 z1²)/2; max at 0; H = diag(-1,-4); Σ = diag(1, 1/4)
  const f = async (cloud) => cloud.map(([a, b]) => -(a * a + 4 * b * b) / 2);
  const r = await polish(f, [0.5, 0.5]);
  assert.ok(Math.abs(r.z[0]) < 1e-3 && Math.abs(r.z[1]) < 1e-3, `z=[${r.z}]`);
  assert.ok(r.ok, 'curvature ok');
  assert.ok(Math.abs(r.covariance[0][0] - 1) < 1e-1, `cov00=${r.covariance[0][0]}`);
  assert.ok(Math.abs(r.covariance[1][1] - 0.25) < 1e-1, `cov11=${r.covariance[1][1]}`);
  assert.ok(r.value > -1e-4, `value=${r.value}`);
});
