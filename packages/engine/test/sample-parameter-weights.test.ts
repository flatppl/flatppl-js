'use strict';
// A DRAW AT A WEIGHTED PARAMETER ENSEMBLE carries the parameter's weights.
//
// THE DEFECT. `matSample` drew one kernel sample per parameter atom and handed
// the ensemble back UNWEIGHTED. Where the parameter measure represents its law
// by REWEIGHTING proposal positions — `normalize(weighted(f, Q))`, whose atoms
// sit at Q's positions and carry f/Z in `logWeights` — the drop replaces the
// parameter measure by its proposal, so the draw is from a different measure.
// `matEvaluate` propagates (via `propagateLogWeights`); the sample path did not.
//
// THE ORACLE, closed form and exact. `normalize(weighted(x -> exp(x),
// Normal(0, 1)))` is exactly Normal(1, 1): e^x φ(x) ∝ exp(−(x−1)²/2). So with
//   theta ~ that,  y ~ Normal(mu = theta, sigma = 1)
// the joint is Gaussian with
//   E[theta] = 1, E[y] = 1, Var[theta] = 1, Var[y] = 2, cov(theta, y) = 1.
// The failing hypothesis has its own closed form — y's ensemble reads
// Normal(0, 1)'s mean, E[y] = 0, a full sigma low — and a DOUBLE count of
// theta's weight has a third, E[theta] = 2 (the tilt squared). Every witness
// below excludes both, so a green test cannot mean "the number moved somewhere
// plausible".
//
// Spec §06 (normative), "The measure monad": bind is
// $(\nu \mathbin{\texttt{>>=}} \kappa)(B) = \int_X \kappa(x)(B)\, d\nu(x)$, and
// §06 `kchain` / `jointchain` give `theta ~ M; y ~ K(theta)` as exactly that
// bind ("equivalence with stochastic nodes"). The integral is against $\nu$ —
// the parameter measure — not against the proposal its atoms were drawn from.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const H = 'flatppl_compat = "0.1"\n';
const N = 400000;

// Weighted mean / variance / covariance of a sampled ensemble read under
// `lw` (null = uniform), plus the ESS the weights leave.
function stats(lw: Float64Array | null, cols: Float64Array[], n: number) {
  let mx = -Infinity;
  if (lw) for (let i = 0; i < n; i++) if (lw[i] > mx) mx = lw[i];
  const w = new Float64Array(n);
  let tot = 0; let sq = 0;
  const mean = cols.map(() => 0);
  for (let i = 0; i < n; i++) {
    const wi = lw ? Math.exp(lw[i] - mx) : 1;
    w[i] = wi; tot += wi; sq += wi * wi;
    for (let c = 0; c < cols.length; c++) mean[c] += wi * cols[c][i];
  }
  for (let c = 0; c < cols.length; c++) mean[c] /= tot;
  const cov: number[][] = cols.map(() => cols.map(() => 0));
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < cols.length; a++) {
      for (let b = 0; b < cols.length; b++) {
        cov[a][b] += w[i] * (cols[a][i] - mean[a]) * (cols[b][i] - mean[b]);
      }
    }
  }
  for (let a = 0; a < cols.length; a++) {
    for (let b = 0; b < cols.length; b++) cov[a][b] /= tot;
  }
  return { mean, cov, n_eff: tot * tot / sq };
}

const WITNESS = H
  + 'tilt = x -> exp(x)\n'
  + 'tm = normalize(weighted(tilt, Normal(mu = 0.0, sigma = 1.0)))\n'
  + 'theta ~ tm\n'
  + 'y ~ Normal(mu = theta, sigma = 1.0)\n';

test('a draw at a weighted parameter ensemble reproduces the joint moments',
  async () => {
    const { proc, ctx } = ctxFor(WITNESS, N);
    assert.equal(
      proc.diagnostics.filter((d: any) => d.severity === 'error').length, 0);
    const th = await ctx.getMeasure('theta');
    const y = await ctx.getMeasure('y');
    assert.ok(y.logWeights, "y's ensemble must carry theta's importance weights "
      + '— without them it is a draw at Normal(0, 1), not at the tilted law');
    const s = stats(y.logWeights, [th.samples, y.samples], N);
    // Bands are ~4 standard errors at n_eff ≈ 0.36 N (the tilt's own ESS).
    assert.ok(Math.abs(s.mean[0] - 1) < 0.03,
      `E[theta] = ${s.mean[0]}, oracle 1 (0 = weights dropped, 2 = double count)`);
    assert.ok(Math.abs(s.mean[1] - 1) < 0.03,
      `E[y] = ${s.mean[1]}, oracle 1 (0 = weights dropped, 2 = double count)`);
    assert.ok(Math.abs(s.cov[0][0] - 1) < 0.04, `Var[theta] = ${s.cov[0][0]}, oracle 1`);
    assert.ok(Math.abs(s.cov[1][1] - 2) < 0.08, `Var[y] = ${s.cov[1][1]}, oracle 2`);
    // The covariance is what a mean-only check misses: reading y under theta's
    // weights while y carries none still gives E[y] ≈ 1, but the two ensembles
    // are then separate objects and no consumer sees the dependence.
    assert.ok(Math.abs(s.cov[0][1] - 1) < 0.05,
      `cov(theta, y) = ${s.cov[0][1]}, oracle 1 (= Var[theta])`);
  });

test('the draw introduces no weighting event of its own', async () => {
  // The kernel sample at atom i is conditionally independent given theta_i, so
  // the atom's weight IS theta's — `propagateLogWeights` returns the shared
  // reference. Reference identity is load-bearing: it is how the engine's
  // independence dedupe recognises the stream downstream (empirical.ts).
  const { ctx } = ctxFor(WITNESS, 4000);
  const th = await ctx.getMeasure('theta');
  const y = await ctx.getMeasure('y');
  assert.equal(y.logWeights, th.logWeights,
    'a second array here means the draw counted the stream again');
  // `normalize` leaves the weights summing to one, so the drawn probability
  // measure still reports mass 0 in log space.
  assert.ok(Math.abs(y.logTotalmass) < 1e-9, `logTotalmass ${y.logTotalmass}`);
  assert.ok(Math.abs(y.n_eff - th.n_eff) < 1e-9,
    `n_eff ${y.n_eff} must inherit theta's ${th.n_eff}, not report a confident N`);
  assert.ok(y.n_eff < 0.6 * 4000, `n_eff ${y.n_eff} is too high for this tilt`);
});

test('an UNWEIGHTED parameter leaves the draw unweighted', async () => {
  const { ctx } = ctxFor(H
    + 'theta ~ Normal(mu = 0.0, sigma = 1.0)\n'
    + 'y ~ Normal(mu = theta, sigma = 1.0)\n', 4000);
  const y = await ctx.getMeasure('y');
  assert.equal(y.logWeights, null,
    'an all-equal weight array here would break the dedupe contract');
  assert.equal(y.logTotalmass, 0);
  assert.equal(y.n_eff, 4000);
});

test('two independent weighted parameters combine as a joint IS weight',
  async () => {
    // a is tilted to Normal(1, 1), b to Normal(0.5, 1) — two DISTINCT weighting
    // events, so §06's product measure makes y's atom weight their product
    // (the sum in log space). sigma is 1 exactly, so b enters y's law not at
    // all; it enters y's ENSEMBLE because y is a draw at b's atom, and reading
    // b's marginal off y's weights must still recover its own tilt.
    const { proc, ctx } = ctxFor(H
      + 't1 = x -> exp(x)\n'
      + 't2 = x -> exp(0.5 * x)\n'
      + 'ma = normalize(weighted(t1, Normal(mu = 0.0, sigma = 1.0)))\n'
      + 'mb = normalize(weighted(t2, Normal(mu = 0.0, sigma = 1.0)))\n'
      + 'a ~ ma\n'
      + 'b ~ mb\n'
      + 'sb = 1.0 + 0.0 * b\n'
      + 'y ~ Normal(mu = a, sigma = sb)\n', N);
    assert.equal(
      proc.diagnostics.filter((d: any) => d.severity === 'error').length, 0);
    const a = await ctx.getMeasure('a');
    const sb = await ctx.getMeasure('sb');
    const y = await ctx.getMeasure('y');
    assert.ok(y.logWeights, 'y must carry both parameter streams');
    assert.notEqual(y.logWeights, a.logWeights);
    assert.notEqual(y.logWeights, sb.logWeights);
    for (let i = 0; i < 64; i++) {
      assert.ok(Math.abs(y.logWeights[i] - (a.logWeights[i] + sb.logWeights[i]))
        < 1e-12, `atom ${i}: y's weight is not the sum of the two streams`);
    }
    const b = await ctx.getMeasure('b');
    const s = stats(y.logWeights, [a.samples, b.samples, y.samples], N);
    assert.ok(Math.abs(s.mean[0] - 1) < 0.03, `E[a] = ${s.mean[0]}, oracle 1`);
    assert.ok(Math.abs(s.mean[1] - 0.5) < 0.03, `E[b] = ${s.mean[1]}, oracle 0.5`);
    assert.ok(Math.abs(s.mean[2] - 1) < 0.03, `E[y] = ${s.mean[2]}, oracle 1`);
  });

test('iid counts a weighted parameter ancestor ONCE across the repeat axis',
  async () => {
    // The repeat axis tiles theta across an atom's k inner draws, so theta's
    // weight is ONE event for the atom, not one per coordinate. Summing it k
    // times raises it to the power k: at k = 2 the tilt exp(theta) would become
    // exp(2 theta) and E[theta] would read 2 instead of 1.
    //
    // The inner measure carries NO tilt of its own here (`weighted(1.0, …)` is
    // mass 1 and variate-independent), so theta's stream is the only one in
    // play and the oracle is the untouched Normal(1, 1) marginal. A tilted
    // inner measure adds the per-coordinate residue Ẑ(theta) the pooled
    // divisor still leaves behind — a separate, tracked defect — which would
    // confound this reading.
    const { proc, ctx } = ctxFor(H
      + 'tilt = x -> exp(x)\n'
      + 'tm = normalize(weighted(tilt, Normal(mu = 0.0, sigma = 1.0)))\n'
      + 'theta ~ tm\n'
      + 'm = normalize(weighted(1.0, Normal(mu = theta, sigma = 1.0)))\n'
      + 'y ~ iid(m, 2)\n', N);
    assert.equal(
      proc.diagnostics.filter((d: any) => d.severity === 'error').length, 0);
    const th = await ctx.getMeasure('theta');
    const y = await ctx.getMeasure('y');
    assert.ok(y.logWeights, "the iid fold must carry theta's weights");
    assert.deepEqual(y.dims, [2]);
    const s = stats(y.logWeights, [th.samples], N);
    assert.ok(Math.abs(s.mean[0] - 1) < 0.03,
      `E[theta] = ${s.mean[0]}, oracle 1 — 2 would mean the shared stream was `
      + 'summed once per coordinate');
    // Both coordinates are Normal(theta, 1) at the SAME theta, so the oracle
    // marginals are Normal(1, 2) with cov(y1, y2) = Var[theta] = 1.
    const c0 = new Float64Array(N); const c1 = new Float64Array(N);
    for (let i = 0; i < N; i++) { c0[i] = y.samples[i * 2]; c1[i] = y.samples[i * 2 + 1]; }
    const t = stats(y.logWeights, [c0, c1], N);
    assert.ok(Math.abs(t.mean[0] - 1) < 0.04, `E[y1] = ${t.mean[0]}, oracle 1`);
    assert.ok(Math.abs(t.mean[1] - 1) < 0.04, `E[y2] = ${t.mean[1]}, oracle 1`);
    assert.ok(Math.abs(t.cov[0][0] - 2) < 0.09, `Var[y1] = ${t.cov[0][0]}, oracle 2`);
    assert.ok(Math.abs(t.cov[0][1] - 1) < 0.06,
      `cov(y1, y2) = ${t.cov[0][1]}, oracle 1 (= Var[theta])`);
  });
