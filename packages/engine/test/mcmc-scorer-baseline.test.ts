'use strict';
// mcmc-scorer-baseline.test.ts — RED→GREEN gate for the SYNC posterior-
// numerator scorer (mcmc-density.ts).
//
// For each of the 6 target models the test:
//   1. Builds a materialiser ctx via ctxFor (imported from regression-baseline.test.ts).
//   2. Locates the 'posterior' bayesupdate derivation.
//   3. Calls buildLogPi (async setup, SYNC score).
//   4. Evaluates priorOf / likOf / logPi at the model's golden @pt point.
//   5. Asserts each to the golden to 1e-9.
//
// The golden numbers come from regression-baseline.json (Distributions.jl
// validated).  The points are hard-coded from the JSON score strings —
// exactly the same θ the async path scored.

const { test } = require('node:test');
const assert   = require('node:assert/strict');
const fs       = require('node:fs');
const path     = require('node:path');

const { ctxFor }     = require('./density/regression-baseline.test.ts');
const { buildLogPi } = require('../mcmc-density.ts');

const TOL = 1e-9;

const GOLDEN = JSON.parse(
  fs.readFileSync(path.join(__dirname, 'density/regression-baseline.json'), 'utf8'),
);

// Per-model golden points and expected values.  Extracted verbatim from the
// score strings in regression-baseline.json.
interface ModelCase {
  file: string;
  /** Plain JS record matching the score string's record(...) argument. */
  point: Record<string, any>;
  /** golden density:prior@pt */
  priorGolden: number;
  /** golden density:likelihood@pt */
  likGolden: number;
}

const CASES: ModelCase[] = [
  {
    file: 'eight-schools.flatppl',
    point: {
      mu:    0.0,
      tau:   1.0,
      theta: Float64Array.from([1, 2, 3, 4, 5, 6, 7, 8]),
    },
    priorGolden: -113.98012604215299,
    likGolden:   -30.1834262038121,
  },
  {
    file: 'gamma-reparam.flatppl',
    point: { mu: 1.5, sigma: 0.5 },
    priorGolden: -4.671988734306883,
    likGolden:   -7.328956763223636,
  },
  {
    file: 'hierarchical-logistic.flatppl',
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
    file: 'partial-pooling.flatppl',
    point: {
      phi:   0.3,
      kappa: 30.0,
      theta: Float64Array.from([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]),
    },
    priorGolden: 8.322039156851243,
    likGolden:   -18.7409785620807,
  },
  {
    file: 'rasch-1pl.flatppl',
    point: {
      theta: Float64Array.from([0.5, 0.0, -0.5, 1.0]),
      b:     Float64Array.from([0.0, 0.5, 1.0, -0.5, -1.0]),
    },
    priorGolden: -12.808521660704425,
    likGolden:   -16.38860389408287,
  },
  {
    file: 'poisson-glm-link.flatppl',
    point: { intercept: 0.0, slope: 1.0 },
    priorGolden: -2.3378770664093453,
    likGolden:   -6.262979602751885,
  },
];

const FIXTURE_DIR = path.join(__dirname, 'fixtures/baseline');

for (const c of CASES) {
  test(`mcmc-scorer: ${c.file} sync logPi matches golden to 1e-9`, async () => {
    // ── 1. Build ctx ──────────────────────────────────────────────────────
    const src = fs.readFileSync(path.join(FIXTURE_DIR, c.file), 'utf8');
    const { ctx } = ctxFor(src, 1);

    // ── 2. Find the bayesupdate derivation ────────────────────────────────
    const derivations: Record<string, any> = ctx.derivations;
    let posteriorDeriv: any = null;
    for (const [, v] of Object.entries(derivations)) {
      if (v && (v as any).kind === 'bayesupdate') {
        posteriorDeriv = v;
        break;
      }
    }
    assert.ok(
      posteriorDeriv != null,
      `${c.file}: no bayesupdate derivation found in ctx.derivations`,
    );

    // ── 3. Async setup ────────────────────────────────────────────────────
    const { logPi, priorOf, likOf } = await buildLogPi(ctx, posteriorDeriv);

    // ── 4. Sync evaluation at the golden point ────────────────────────────
    const gotPrior = priorOf(c.point);
    const gotLik   = likOf(c.point);
    const gotLogPi = logPi(c.point);

    const goldenSum = c.priorGolden + c.likGolden;

    // ── 5. Assert to 1e-9 ─────────────────────────────────────────────────
    assert.ok(
      Math.abs(gotPrior - c.priorGolden) <= TOL,
      `${c.file} priorOf: got ${gotPrior}, golden ${c.priorGolden} (Δ ${Math.abs(gotPrior - c.priorGolden)})`,
    );
    assert.ok(
      Math.abs(gotLik - c.likGolden) <= TOL,
      `${c.file} likOf: got ${gotLik}, golden ${c.likGolden} (Δ ${Math.abs(gotLik - c.likGolden)})`,
    );
    assert.ok(
      Math.abs(gotLogPi - goldenSum) <= TOL,
      `${c.file} logPi: got ${gotLogPi}, golden_sum ${goldenSum} (Δ ${Math.abs(gotLogPi - goldenSum)})`,
    );
  });
}
