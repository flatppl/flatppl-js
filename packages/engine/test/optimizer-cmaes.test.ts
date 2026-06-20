'use strict';

// =====================================================================
// optimizer-cmaes.test.ts
// =====================================================================
//
// Behavioural tests for the CMA-ES core (engine-concepts: the §22-style
// "propose a cloud, reweight, adapt" rhythm, here as the optimizer). The
// core is pure and dependency-injected: it takes an async `evalCloud`
// (a batch of points → their fitness, MAXIMISED) and a seedable RNG, so
// it is tested against closed-form objectives with NO worker. Standard
// ES benchmarks: sphere (trivial), an ill-conditioned quadratic (forces
// covariance adaptation), and 2-D Rosenbrock (the classic curved valley).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { cmaes } = require('../optimizer/cmaes.ts');

test('cmaes maximises a concave quadratic to its centre', async () => {
  const c = [2, -1];
  const evalCloud = async (cloud: number[][]) =>
    cloud.map((x) => -((x[0] - c[0]) ** 2 + (x[1] - c[1]) ** 2));
  const res = await cmaes({ evalCloud, x0: [0, 0], sigma0: 1, opts: { seed: 1 } });
  assert.ok(Math.abs(res.x[0] - 2) < 1e-3, `x0=${res.x[0]}`);
  assert.ok(Math.abs(res.x[1] + 1) < 1e-3, `x1=${res.x[1]}`);
  assert.ok(res.value > -1e-5, `value=${res.value}`);
  assert.ok(res.generations > 0 && res.evals > 0, 'reports work done');
});

test('cmaes solves an ill-conditioned quadratic (covariance adaptation)', async () => {
  // condition number 1000 between the two axes — isotropic σ alone stalls;
  // the rank-µ/rank-one C update is what makes this converge.
  const evalCloud = async (cloud: number[][]) =>
    cloud.map((x) => -(1000 * (x[0] - 3) ** 2 + (x[1] - 4) ** 2));
  const res = await cmaes({
    evalCloud, x0: [0, 0], sigma0: 1, opts: { seed: 2, maxGenerations: 500 },
  });
  assert.ok(Math.abs(res.x[0] - 3) < 1e-2, `x0=${res.x[0]}`);
  assert.ok(Math.abs(res.x[1] - 4) < 1e-2, `x1=${res.x[1]}`);
});

test('cmaes finds the 2-D Rosenbrock optimum at (1,1)', async () => {
  const evalCloud = async (cloud: number[][]) =>
    cloud.map(([a, b]) => -(100 * (b - a * a) ** 2 + (1 - a) ** 2));
  const res = await cmaes({
    evalCloud, x0: [-1, 1], sigma0: 0.5, opts: { seed: 3, maxGenerations: 800 },
  });
  assert.ok(Math.abs(res.x[0] - 1) < 1e-2 && Math.abs(res.x[1] - 1) < 1e-2,
    `x=[${res.x}]`);
});

test('cmaes is reproducible for a fixed seed', async () => {
  const evalCloud = async (cloud: number[][]) => cloud.map((x) => -(x[0] ** 2 + x[1] ** 2));
  const a = await cmaes({ evalCloud, x0: [3, 3], sigma0: 1, opts: { seed: 7 } });
  const b = await cmaes({ evalCloud, x0: [3, 3], sigma0: 1, opts: { seed: 7 } });
  assert.equal(a.x[0], b.x[0]);
  assert.equal(a.x[1], b.x[1]);
});

test('cmaes treats NaN / -Inf fitness as worst (stays in the feasible region)', async () => {
  // A quadratic peak at (1,1), but the half-plane x0<0 is "invalid" (-Inf).
  const evalCloud = async (cloud: number[][]) => cloud.map((x) =>
    x[0] < 0 ? -Infinity : -((x[0] - 1) ** 2 + (x[1] - 1) ** 2));
  const res = await cmaes({ evalCloud, x0: [0.5, 0.5], sigma0: 0.5, opts: { seed: 4 } });
  assert.ok(res.x[0] >= 0, `stayed feasible: x0=${res.x[0]}`);
  assert.ok(Math.abs(res.x[0] - 1) < 1e-2 && Math.abs(res.x[1] - 1) < 1e-2, `x=[${res.x}]`);
});
