'use strict';
// optimizer-peaked-mle.test.ts — the optimizer must find a sharply-peaked MLE
// from a poor start, and never return worse than the pivot it was handed.
//
// Regression for two fixes: (1) cmaes seeds its incumbent with x0 evaluated
// (never returns below the start), (2) optimize() multi-starts by default. The
// motivating case is the linear-regression likelihood: tight data + small σ make
// a narrow ridge a single CMA run disperses off (logL drops ~530 nats for a β
// error of 1.3), so the bare single-start optimizer landed far from the MLE.

const { test } = require('node:test');
const assert   = require('node:assert/strict');
const { optimize } = require('../optimizer/index.ts');

// log-likelihood of y ~ Normal(alpha + beta*x, sigma) over the linreg fixture data.
const X = [1.1, 1.5, 1.3, 1.4], Y = [3.2, 4.1, 3.4, 3.9];
function logL(p: number[]): number {
  const [a, b, s] = p; if (s <= 0) return -1e12;
  let t = 0; for (let i = 0; i < X.length; i++) { const d = Y[i] - (a + b * X[i]); t += -Math.log(s) - 0.5 * (d * d) / (s * s); }
  return t;
}
const domains = [{ kind: 'interval', lo: -1, hi: 2 }, { kind: 'interval', lo: 0, hi: 4 }, { kind: 'interval', lo: 0.05, hi: 0.5 }];
const scales = [3, 4, 0.45];
const MLE = [0.546, 2.343, 0.112];                 // OLS fit
const MLE_VAL = logL(MLE);

test('optimizer finds the sharply-peaked linreg MLE from a poor start', async () => {
  for (const x0 of [[0.5, 2.0, 0.275], [2.0, 0.0, 0.5]]) {   // domain centre + a corner
    const r = await optimize({ evalCloud: (pts: number[][]) => pts.map(logL), x0, domains, scales, opts: { seed: 1 } });
    assert.ok(Math.abs(r.mode[0] - MLE[0]) < 0.1, `alpha ${r.mode[0]} ≈ ${MLE[0]}`);
    assert.ok(Math.abs(r.mode[1] - MLE[1]) < 0.1, `beta ${r.mode[1]} ≈ ${MLE[1]}`);
    assert.ok(Math.abs(r.mode[2] - MLE[2]) < 0.05, `sigma ${r.mode[2]} ≈ ${MLE[2]}`);
    assert.ok(r.value >= MLE_VAL - 0.05, `value ${r.value} ≈ MLE ${MLE_VAL}`);
  }
});

test('optimizer never returns worse than the pivot it was given', async () => {
  // Start AT the MLE: the incumbent-seed fix guarantees the result is ≥ f(x0),
  // even though the sampled cloud disperses off the narrow peak.
  const r = await optimize({ evalCloud: (pts: number[][]) => pts.map(logL), x0: MLE, domains, scales, opts: { seed: 1 } });
  assert.ok(r.value >= MLE_VAL - 1e-9, `value ${r.value} >= pivot ${MLE_VAL}`);
});
