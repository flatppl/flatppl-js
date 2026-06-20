'use strict';

// =====================================================================
// optimizer.test.ts — the `optimize` orchestration + modular seam
// =====================================================================
//
// `optimize` ties coords + the chosen optimizer (CMA-ES) + the FD polish
// into a `ModeFit` (mode in original parameter space, Laplace covariance,
// diagnostics). It evaluates the objective in original x-space (the viewer
// hands it an `evalCloud` posting to the worker); domains/scales come from
// value-sets and plot ranges. The optimizer is selected from a registry so
// new optimizers can be added later. Tested with closed-form objectives.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { optimize, registerOptimizer } = require('../optimizer/optimize.ts');

test('optimize finds the interior maximum over a box domain, with a Laplace covariance', async () => {
  const evalCloud = async (cloud) => cloud.map(([x]) => -((x - 2) ** 2));
  const fit = await optimize({
    evalCloud, x0: [0], domains: [{ kind: 'interval', lo: -5, hi: 5 }], scales: [5],
    opts: { seed: 1 },
  });
  assert.ok(Math.abs(fit.mode[0] - 2) < 1e-2, `mode=${fit.mode[0]}`);
  assert.ok(fit.value > -1e-3, `value=${fit.value}`);
  assert.equal(fit.curvatureSource, 'fd-hessian');
  assert.ok(fit.covariance && fit.covariance[0][0] > 0, 'has a covariance');
  assert.equal(fit.boundaryActive[0], false);
  assert.ok(fit.nEvals > 0 && fit.nBatches > 0, 'reports work');
});

test('optimize reaches a boundary maximum and flags it active', async () => {
  const evalCloud = async (cloud) => cloud.map(([x]) => x); // increasing → max at hi
  const fit = await optimize({
    evalCloud, x0: [0.2], domains: [{ kind: 'interval', lo: 0, hi: 1 }], scales: [1],
    opts: { seed: 2 },
  });
  assert.ok(Math.abs(fit.mode[0] - 1) < 1e-2, `mode=${fit.mode[0]}`);
  assert.equal(fit.boundaryActive[0], true);
});

test('optimize over posreals finds a positive-domain maximum', async () => {
  const evalCloud = async (cloud) => cloud.map(([x]) => -((x - 5) ** 2));
  const fit = await optimize({
    evalCloud, x0: [1], domains: [{ kind: 'posreals' }], scales: [5], opts: { seed: 3 },
  });
  assert.ok(Math.abs(fit.mode[0] - 5) < 5e-2, `mode=${fit.mode[0]}`);
  assert.ok(fit.mode[0] > 0);
});

test('multi-start finds the global max of a bimodal objective from a bad start', async () => {
  // tall bump at +3 (value 2), short bump at -3 (value 1); start at the short one.
  const evalCloud = async (cloud) => cloud.map(([x]) =>
    Math.max(2.0 - 0.5 * (x - 3) ** 2, 1.0 - 0.5 * (x + 3) ** 2));
  const fit = await optimize({
    evalCloud, x0: [-3], domains: [{ kind: 'interval', lo: -6, hi: 6 }], scales: [3],
    opts: { seed: 5, starts: 6 },
  });
  assert.ok(Math.abs(fit.mode[0] - 3) < 2e-1, `mode=${fit.mode[0]} (should find the +3 bump)`);
  assert.ok(fit.value > 1.9, `value=${fit.value}`);
});

test('registerOptimizer makes a custom optimizer selectable via opts.optimizer', async () => {
  let seen = null;
  registerOptimizer('stub', async (zspec) => {
    seen = zspec.x0.slice();
    return { x: zspec.x0.slice(), value: 0, evals: 1, generations: 1, reason: 'stub' };
  });
  const evalCloud = async (cloud) => cloud.map(() => 0);
  const fit = await optimize({
    evalCloud, x0: [1], domains: [{ kind: 'real' }], scales: [1],
    opts: { optimizer: 'stub', polish: false },
  });
  assert.ok(seen, 'custom optimizer was invoked');
  assert.equal(fit.optimizer, 'stub');
  assert.equal(fit.curvatureSource, 'none'); // stub returns no covariance
});
