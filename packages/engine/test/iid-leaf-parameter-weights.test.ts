'use strict';
// AN iid OVER A RESOLVED LEAF carries its weighted parameter's importance
// weights.
//
// THE DEFECT. `matIid` resolves `iid(Normal(mu = theta, sigma = 1.0), 3)` to a
// sample leaf and batches it in one worker round-trip (`sampleN` with
// `repeat = k`). Both leaf branches — plain and truncated — rebuilt the output
// with `arrayMeasure(samples, dims, null)`, so a weighted parameter ancestor's
// `logWeights` were dropped. This is the path the spelling above ACTUALLY
// takes; the composite fallback's fold never sees it.
//
// THE ORACLE, closed form and exact. `normalize(weighted(x -> exp(x),
// Normal(0, 1)))` is exactly Normal(1, 1): e^x φ(x) ∝ exp(−(x−1)²/2). With
//   theta ~ that,  y ~ iid(Normal(mu = theta, sigma = 1.0), k)
// every coordinate is Normal(theta, 1) at the SAME theta, so
//   E[theta] = 1, E[y_i] = 1, Var[y_i] = 2,
//   cov(theta, y_i) = Var[theta] = 1, cov(y_i, y_j) = Var[theta] = 1.
// Both failure modes have their own closed form and every witness excludes
// them. Dropping the weights reads Normal(0, 1)'s mean, E[y_i] = 0. Counting
// theta's ONE shared weighting event once per coordinate raises the tilt to
// exp(k·x), which is exactly Normal(k, 1), so E[theta] reads k.
//
// The cross-coordinate covariance is the witness a mean-only check misses: it
// is what separates the shared-parameter tilt from a per-position re-draw,
// which would give cov(y_i, y_j) = 0.
//
// Spec §06 (normative). `iid(M, size)` is "the product measure
// $M^{\otimes N}$". §06 "The measure monad" gives bind as
// $(\nu \mathbin{\texttt{>>=}} \kappa)(B) = \int_X \kappa(x)(B)\, d\nu(x)$, and
// §06 `kchain`'s "Equivalence with stochastic nodes" gives `theta ~ M;
// y ~ K(theta)` as exactly that bind. Each coordinate integrates against the
// parameter MEASURE, not against the proposal its atoms were drawn from.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const H = 'flatppl_compat = "0.1"\n';
const N = 400000;

const TILT = 'tilt = x -> exp(x)\n'
  + 'tm = normalize(weighted(tilt, Normal(mu = 0.0, sigma = 1.0)))\n'
  + 'theta ~ tm\n';

// Weighted mean / covariance of a sampled ensemble read under `lw`
// (null = uniform), plus the ESS the weights leave.
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

// Split an iid measure's atom-major [n, k] buffer into k per-coordinate columns.
function coords(m: any, k: number, n: number) {
  const out: Float64Array[] = [];
  for (let j = 0; j < k; j++) {
    const c = new Float64Array(n);
    for (let i = 0; i < n; i++) c[i] = m.samples[i * k + j];
    out.push(c);
  }
  return out;
}

// The joint moments of `theta ~ Normal(1, 1); y ~ iid(Normal(theta, 1), k)`,
// asserted per coordinate and across coordinate pairs. Bands are ~4 standard
// errors at n_eff ≈ 0.36 N (the tilt's own ESS).
async function assertSharedThetaJoint(src: string, k: number) {
  const { proc, ctx } = ctxFor(src, N);
  assert.equal(
    proc.diagnostics.filter((d: any) => d.severity === 'error').length, 0);
  const th = await ctx.getMeasure('theta');
  const y = await ctx.getMeasure('y');
  assert.ok(y.logWeights, "the leaf iid must carry theta's importance weights "
    + '— without them every coordinate is a draw at Normal(0, 1)');
  assert.deepEqual(y.dims, [k]);
  const s = stats(y.logWeights, [th.samples].concat(coords(y, k, N)), N);
  assert.ok(Math.abs(s.mean[0] - 1) < 0.03,
    `E[theta] = ${s.mean[0]}, oracle 1 (${k} = the shared weight counted once `
    + 'per coordinate)');
  for (let j = 0; j < k; j++) {
    assert.ok(Math.abs(s.mean[1 + j] - 1) < 0.04,
      `E[y${j}] = ${s.mean[1 + j]}, oracle 1 (0 = weights dropped)`);
    assert.ok(Math.abs(s.cov[1 + j][1 + j] - 2) < 0.09,
      `Var[y${j}] = ${s.cov[1 + j][1 + j]}, oracle 2`);
    assert.ok(Math.abs(s.cov[0][1 + j] - 1) < 0.06,
      `cov(theta, y${j}) = ${s.cov[0][1 + j]}, oracle 1 (= Var[theta])`);
  }
  for (let a = 0; a < k; a++) {
    for (let b = a + 1; b < k; b++) {
      // Zero here would mean the coordinates saw independent thetas rather
      // than one shared draw reweighted.
      assert.ok(Math.abs(s.cov[1 + a][1 + b] - 1) < 0.06,
        `cov(y${a}, y${b}) = ${s.cov[1 + a][1 + b]}, oracle 1 (= Var[theta], `
        + '0 = a per-position re-draw)');
    }
  }
}

test('a leaf iid at a weighted parameter reproduces the joint moments',
  async () => {
    await assertSharedThetaJoint(
      H + TILT + 'y ~ iid(Normal(mu = theta, sigma = 1.0), 3)\n', 3);
  });

test('a NAMED inner measure that peels to a leaf goes the same route',
  async () => {
    // `inner = Normal(...)` binds the measure to a name, which `_resolveIidLeaf`
    // peels through to the same sample leaf. Same fast path, same drop.
    await assertSharedThetaJoint(
      H + TILT
      + 'inner = Normal(mu = theta, sigma = 1.0)\n'
      + 'y ~ iid(inner, 3)\n', 3);
  });

test('the shared parameter weight enters an atom once, not once per coordinate',
  async () => {
    // The parameter is pinned per atom, so one theta draw serves the atom's k
    // inner positions: ONE weighting event. `propagateLogWeights` returns
    // theta's stream itself, and that reference identity is load-bearing — it
    // is how the engine's independence dedupe recognises the stream downstream.
    const { ctx } = ctxFor(
      H + TILT + 'y ~ iid(Normal(mu = theta, sigma = 1.0), 3)\n', 4000);
    const th = await ctx.getMeasure('theta');
    const y = await ctx.getMeasure('y');
    // `assert.ok` on the comparison, not `assert.equal`: a failing `equal`
    // renders both operands, and inspecting an N-atom Float64Array to build
    // the diff takes minutes and can take the process down with it.
    assert.ok(y.logWeights === th.logWeights,
      'a second array here means the fold summed the stream over the block');
    // `normalize` leaves theta's weights summing to one, so the product
    // measure still reports mass 0 in log space.
    assert.ok(Math.abs(y.logTotalmass) < 1e-9, `logTotalmass ${y.logTotalmass}`);
    assert.ok(Math.abs(y.n_eff - th.n_eff) < 1e-9,
      `n_eff ${y.n_eff} must inherit theta's ${th.n_eff}, not report a `
      + 'confident N');
    assert.ok(y.n_eff < 0.6 * 4000, `n_eff ${y.n_eff} is too high for this tilt`);
  });

test('an UNWEIGHTED parameter leaves the leaf iid unweighted', async () => {
  const { ctx } = ctxFor(H
    + 'theta ~ Normal(mu = 0.0, sigma = 1.0)\n'
    + 'y ~ iid(Normal(mu = theta, sigma = 1.0), 3)\n', 4000);
  const y = await ctx.getMeasure('y');
  assert.equal(y.logWeights, null,
    'an all-equal weight array here would break the dedupe contract');
  assert.equal(y.logTotalmass, 0);
  assert.equal(y.n_eff, 4000);
});

test('two weighted parameters of a leaf iid combine as a joint IS weight',
  async () => {
    // a is tilted to Normal(1, 1), b to Normal(0.5, 1) — two DISTINCT weighting
    // events, so §06's product measure makes the atom weight their product (the
    // sum in log space). sigma is 1 exactly, so b enters y's law not at all; it
    // enters y's ENSEMBLE because y is a draw at b's atom, and reading b's
    // marginal off y's weights must still recover its own tilt.
    const { proc, ctx } = ctxFor(H
      + 't1 = x -> exp(x)\n'
      + 't2 = x -> exp(0.5 * x)\n'
      + 'ma = normalize(weighted(t1, Normal(mu = 0.0, sigma = 1.0)))\n'
      + 'mb = normalize(weighted(t2, Normal(mu = 0.0, sigma = 1.0)))\n'
      + 'a ~ ma\n'
      + 'b ~ mb\n'
      + 'sb = 1.0 + 0.0 * b\n'
      + 'y ~ iid(Normal(mu = a, sigma = sb), 2)\n', N);
    assert.equal(
      proc.diagnostics.filter((d: any) => d.severity === 'error').length, 0);
    const a = await ctx.getMeasure('a');
    const b = await ctx.getMeasure('b');
    const sb = await ctx.getMeasure('sb');
    const y = await ctx.getMeasure('y');
    assert.ok(y.logWeights, 'y must carry both parameter streams');
    assert.notEqual(y.logWeights, a.logWeights);
    assert.notEqual(y.logWeights, sb.logWeights);
    for (let i = 0; i < 64; i++) {
      assert.ok(Math.abs(y.logWeights[i] - (a.logWeights[i] + sb.logWeights[i]))
        < 1e-12, `atom ${i}: y's weight is not the sum of the two streams`);
    }
    const c = coords(y, 2, N);
    const s = stats(y.logWeights, [a.samples, b.samples, c[0], c[1]], N);
    assert.ok(Math.abs(s.mean[0] - 1) < 0.03, `E[a] = ${s.mean[0]}, oracle 1`);
    assert.ok(Math.abs(s.mean[1] - 0.5) < 0.03, `E[b] = ${s.mean[1]}, oracle 0.5`);
    assert.ok(Math.abs(s.mean[2] - 1) < 0.04, `E[y0] = ${s.mean[2]}, oracle 1`);
    assert.ok(Math.abs(s.mean[3] - 1) < 0.04, `E[y1] = ${s.mean[3]}, oracle 1`);
  });

test('a TRUNCATED leaf iid carries the weights too', async () => {
  // The truncated branch is the sibling `arrayMeasure(…, null)` site, and it
  // routes through `truncateSampleN` rather than `sampleN`.
  //
  // ORACLE. theta ~ Normal(1, 1) exactly, and each coordinate is
  // TruncNormal(theta, 1) on [-3, 3] at that shared theta. Numerical
  // integration over theta (scipy quad, cross-checked by a 4e6-draw
  // inverse-CDF Monte Carlo) gives
  //   E[y_i] = 0.86262, Var[y_i] = 1.48588, cov(y_i, y_j) = 0.67340.
  // Dropping the weights reads theta ~ Normal(0, 1): E[y_i] = 0 by symmetry.
  // Counting the shared weight per coordinate reads theta ~ Normal(3, 1):
  // E[y_i] = 2.09684. Both are far outside the bands below.
  //
  // The interval is wide on purpose: the branch draws by REJECTION against a
  // 1000-draw budget, and a narrow window at a far-tail theta exhausts it and
  // returns NaN atoms. The finiteness assertion below pins that.
  const { proc, ctx } = ctxFor(H + TILT
    + 'ty = normalize(truncate(Normal(mu = theta, sigma = 1.0), '
    + 'interval(-3.0, 3.0)))\n'
    + 'y ~ iid(ty, 3)\n', N);
  assert.equal(
    proc.diagnostics.filter((d: any) => d.severity === 'error').length, 0);
  const th = await ctx.getMeasure('theta');
  const y = await ctx.getMeasure('y');
  assert.ok(y.logWeights === th.logWeights,
    "the truncated leaf must carry theta's stream, by reference");
  assert.ok(y.n_eff < 0.6 * N, `n_eff ${y.n_eff} is too high for this tilt`);
  let firstBad = -1;
  for (let i = 0; i < y.samples.length; i++) {
    if (!Number.isFinite(y.samples[i])) { firstBad = i; break; }
  }
  assert.equal(firstBad, -1,
    `atom ${firstBad} is not finite — the rejection budget was exhausted`);
  const s = stats(y.logWeights, coords(y, 3, N), N);
  for (let j = 0; j < 3; j++) {
    assert.ok(Math.abs(s.mean[j] - 0.86262) < 0.04,
      `E[y${j}] = ${s.mean[j]}, oracle 0.86262 (0 = weights dropped, `
      + '2.09684 = counted once per coordinate)');
    assert.ok(Math.abs(s.cov[j][j] - 1.48588) < 0.07,
      `Var[y${j}] = ${s.cov[j][j]}, oracle 1.48588`);
  }
  for (let a = 0; a < 3; a++) {
    for (let b = a + 1; b < 3; b++) {
      assert.ok(Math.abs(s.cov[a][b] - 0.67340) < 0.05,
        `cov(y${a}, y${b}) = ${s.cov[a][b]}, oracle 0.67340 (0 = a `
        + 'per-position re-draw)');
    }
  }
});
