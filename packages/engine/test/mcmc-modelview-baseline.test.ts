'use strict';
// mcmc-modelview-baseline.test.ts — Gate A for the async vector-aware ModelView.
//
// For each of the 6 target models:
//   1. Build ctx via ctxFor.
//   2. Locate the bayesupdate derivation.
//   3. Call buildModelViewFromCtx (async).
//   4. Assert mv.dim matches expected.
//   5. Assert mv.logPosteriorConstrained(goldenPoint) ≈ priorGolden + likGolden (1e-9).
//
// logPosteriorConstrained accepts the SCORER format — i.e. {mu, tau, theta: Float64Array}.
// The goldenPoints here are exactly the same as in mcmc-scorer-baseline.test.ts.

const { test } = require('node:test');
const assert   = require('node:assert/strict');
const fs       = require('node:fs');
const path     = require('node:path');

const { ctxFor }                = require('./density/regression-baseline.test.ts');
const { buildModelViewFromCtx } = require('../model-view.ts');

const TOL = 1e-9;

interface ModelCase {
  file:        string;
  expectedDim: number;
  point:       Record<string, any>;
  priorGolden: number;
  likGolden:   number;
}

const CASES: ModelCase[] = [
  {
    file:        'eight-schools.flatppl',
    expectedDim: 10,    // mu(1) + tau(1) + theta(8) = 10
    point: {
      mu:    0.0,
      tau:   1.0,
      theta: Float64Array.from([1, 2, 3, 4, 5, 6, 7, 8]),
    },
    priorGolden: -113.98012604215299,
    likGolden:   -30.1834262038121,
  },
  {
    file:        'gamma-reparam.flatppl',
    expectedDim: 2,     // mu(1) + sigma(1) = 2
    point: { mu: 1.5, sigma: 0.5 },
    priorGolden: -4.671988734306883,
    likGolden:   -7.328956763223636,
  },
  {
    file:        'hierarchical-logistic.flatppl',
    expectedDim: 6,     // mu_a(1) + sigma_a(1) + a(3) + b(1) = 6
    point: {
      mu_a:    0.0,
      sigma_a: 1.0,
      a:       Float64Array.from([0.1, 0.2, 0.3]),
      b:       1.0,
    },
    priorGolden: -6.786023939166052,
    likGolden:   -1.9521867070175452,
  },
  {
    file:        'partial-pooling.flatppl',
    expectedDim: 10,    // phi(1) + kappa(1) + theta(8) = 10
    point: {
      phi:   0.3,
      kappa: 30.0,
      theta: Float64Array.from([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
    },
    priorGolden: 8.322039156851243,
    likGolden:   -18.7409785620807,
  },
  {
    file:        'rasch-1pl.flatppl',
    expectedDim: 9,     // theta(4) + b(5) = 9
    point: {
      theta: Float64Array.from([0.5, 0.0, -0.5, 1.0]),
      b:     Float64Array.from([0.0, 0.5, 1.0, -0.5, -1.0]),
    },
    priorGolden: -12.808521660704425,
    likGolden:   -16.38860389408287,
  },
  {
    file:        'poisson-glm-link.flatppl',
    expectedDim: 2,     // intercept(1) + slope(1) = 2
    point: { intercept: 0.0, slope: 1.0 },
    priorGolden: -2.3378770664093453,
    likGolden:   -6.262979602751885,
  },
];

const FIXTURE_DIR = path.join(__dirname, 'fixtures/baseline');

for (const c of CASES) {
  test(`mcmc-modelview: ${c.file} dim=${c.expectedDim} logPosteriorConstrained matches golden`, async () => {
    // 1. Build ctx
    const src = fs.readFileSync(path.join(FIXTURE_DIR, c.file), 'utf8');
    const { ctx } = ctxFor(src, 1);

    // 2. Find the bayesupdate derivation
    const derivations: Record<string, any> = ctx.derivations;
    let posteriorDeriv: any = null;
    for (const [, v] of Object.entries(derivations)) {
      if (v && (v as any).kind === 'bayesupdate') {
        posteriorDeriv = v;
        break;
      }
    }
    assert.ok(posteriorDeriv != null,
      `${c.file}: no bayesupdate derivation found`);

    // 3. Async setup
    const mv = await buildModelViewFromCtx(ctx, posteriorDeriv);

    // 4. Assert dim
    assert.strictEqual(mv.dim, c.expectedDim,
      `${c.file}: expected dim ${c.expectedDim}, got ${mv.dim}`);

    // 5. Assert logPosteriorConstrained — accepts scorer format (Float64Array vectors)
    const goldenSum = c.priorGolden + c.likGolden;
    const got = mv.logPosteriorConstrained(c.point);
    assert.ok(
      Math.abs(got - goldenSum) <= TOL,
      `${c.file} logPosteriorConstrained: got ${got}, golden ${goldenSum} (Δ ${Math.abs(got - goldenSum)})`,
    );
  });
}
