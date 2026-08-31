'use strict';
// EVERY mat-broadcast OUTPUT carries its weighted parameter's importance
// weights.
//
// THE DEFECT. Every executor in `mat-broadcast.ts` rebuilt its output with
// `arrayMeasure(out, axes, null)`, and the file referenced `logWeights`
// nowhere. Each one resolves its per-position parameters through
// `prepareDensityRefs`, which hands back only the per-atom POSITIONS — so a
// parameter measure that represents its law by REWEIGHTING those positions had
// the rest of its law dropped at the broadcast. Seven output sites, six
// executors:
//
//   1. the general per-cell path (a builtin distribution head over an N-D grid)
//   2. the iid-bodied composite batch-flatten, inner axis D > 1
//   3. the same, D == 1
//   4. the generative-bodied composite
//   5. the joint-bodied composite
//   6. the jointchain-bodied composite
//   7. the nested-broadcast batch-flatten
//
// The two MvNormal folds — `_executeMvNormalBroadcast` and
// `_executeNestedBroadcastVectorFold` — keep an unconditional
// `logWeights: null`: every input there has to resolve through
// `resolveIRToValue`, which an atom-dependent expression cannot, so no
// weighted parent reaches them.
//
// THE ORACLE, closed form and exact. `normalize(weighted(x -> exp(x),
// Normal(0, 1)))` is exactly Normal(1, 1): e^x φ(x) ∝ exp(−(x−1)²/2). So
// E[theta] = Var[theta] = 1, and each site's per-position law follows below.
// Both failure modes have their own closed form and every witness excludes
// them. Dropping the weights reads Normal(0, 1), E[theta] = 0. Counting
// theta's ONE shared weighting event once per position raises the tilt to
// exp(P·x) = Normal(P, 1), so E[theta] reads P.
//
// The CROSS-POSITION covariance is the witness a mean-only check misses: the
// positions of an atom are independent GIVEN theta and share one draw of it,
// so the covariance carries theta's whole variance. A per-position re-draw
// reads 0 there.
//
// Spec §04 sec:broadcasting (normative): `broadcast(kernel, ...)` returns "an
// **array-valued measure**: the independent product measure of the kernel
// applications at each array position". §06 "The measure monad" gives bind as
// $(\nu \mathbin{\texttt{>>=}} \kappa)(B) = \int_X \kappa(x)(B)\, d\nu(x)$, and
// §06 `kchain`'s "Equivalence with stochastic nodes" gives `theta ~ M;
// y ~ K(theta)` as exactly that bind. Every position integrates against the
// parameter MEASURE, not against the proposal its atoms were drawn from.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const H = 'flatppl_compat = "0.1"\n';
const N = 200000;

const TILT = 'tilt = x -> exp(x)\n'
  + 'tm = normalize(weighted(tilt, Normal(mu = 0.0, sigma = 1.0)))\n'
  + 'theta ~ tm\n';

// Weighted mean / covariance of a sampled ensemble read under `lw`
// (null = uniform).
function stats(lw: Float64Array | null, cols: Float64Array[], n: number) {
  let mx = -Infinity;
  if (lw) for (let i = 0; i < n; i++) if (lw[i] > mx) mx = lw[i];
  const w = new Float64Array(n);
  let tot = 0;
  const mean = cols.map(() => 0);
  for (let i = 0; i < n; i++) {
    const wi = lw ? Math.exp(lw[i] - mx) : 1;
    w[i] = wi; tot += wi;
    for (let c = 0; c < cols.length; c++) mean[c] += wi * cols[c][i];
  }
  for (let c = 0; c < cols.length; c++) mean[c] /= tot;
  const cov: number[][] = cols.map(() => cols.map(() => 0));
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < cols.length; a++) {
      for (let b = a; b < cols.length; b++) {
        cov[a][b] += w[i] * (cols[a][i] - mean[a]) * (cols[b][i] - mean[b]);
      }
    }
  }
  for (let a = 0; a < cols.length; a++) {
    for (let b = a; b < cols.length; b++) { cov[a][b] /= tot; cov[b][a] = cov[a][b]; }
  }
  return { mean, cov };
}

// Split an atom-major [n, P] buffer into P per-position columns. P is the
// product of the trailing axes, so a [n, K, D] output reads as K·D positions
// in row-major order — the layout every executor here writes.
function positions(m: any, n: number) {
  const data = (m.value && m.value.data) || m.samples;
  const P = data.length / n;
  const out: Float64Array[] = [];
  for (let j = 0; j < P; j++) {
    const c = new Float64Array(n);
    for (let i = 0; i < n; i++) c[i] = data[i * P + j];
    out.push(c);
  }
  return out;
}

// Per-position first and second moments of `theta ~ Normal(1, 1)` shared
// across an atom's positions.
type Oracle = {
  // E[position p].
  mean: number[];
  // Var[position p].
  variance: number[];
  // cov(position a, position b) for a != b. Var[theta] = 1 unless a site
  // couples two positions further (the jointchain's steps do).
  crossCov: (a: number, b: number) => number;
};

// The witness every fixed site shares. Reads the output's OWN ensemble — under
// its own `logWeights`, which is the whole point — and checks it against the
// closed form, then excludes both failure modes.
async function assertSharedTheta(
  label: string, src: string, oracle: Oracle, expectedPositions: number,
) {
  const { proc, ctx } = ctxFor(src, N);
  assert.equal(
    proc.diagnostics.filter((d: any) => d.severity === 'error').length, 0,
    label + ': ' + proc.diagnostics.map((d: any) => d.message).join('; '));
  const th = await ctx.getMeasure('theta');
  const y = await ctx.getMeasure('y');
  assert.ok(y.logWeights, label + ": the broadcast must carry theta's "
    + 'importance weights — without them every position is a draw at '
    + 'Normal(0, 1)');
  // Reference identity, not just equal values: `propagateLogWeights` returns a
  // single-stream parent's array ITSELF, and that identity is how the engine's
  // independence dedupe recognises the stream downstream. A second array here
  // means the fold summed the shared event over the positions.
  //
  // `assert.ok` on the comparison, not `assert.equal`: a failing `equal`
  // renders both operands, and inspecting an N-atom Float64Array to build the
  // diff takes minutes and can take the process down with it.
  assert.ok(y.logWeights === th.logWeights,
    label + ': theta is the only weighted parent, so its stream must be '
    + 'passed forward by reference');
  // `normalize` leaves theta's weights summing to one, so the product measure
  // still reports mass 0 in log space.
  assert.ok(Math.abs(y.logTotalmass) < 1e-9,
    label + ': logTotalmass ' + y.logTotalmass);
  assert.ok(Math.abs(y.n_eff - th.n_eff) < 1e-9,
    label + ': n_eff ' + y.n_eff + " must inherit theta's " + th.n_eff
    + ', not report a confident N');

  const pos = positions(y, N);
  assert.equal(pos.length, expectedPositions,
    label + ': position count');
  for (const c of pos) {
    for (let i = 0; i < 64; i++) {
      assert.ok(Number.isFinite(c[i]), label + ': atom ' + i + ' is not finite');
    }
  }
  const s = stats(y.logWeights, [th.samples].concat(pos), N);
  assert.ok(Math.abs(s.mean[0] - 1) < 0.03,
    label + `: E[theta] = ${s.mean[0]}, oracle 1 (0 = weights dropped, `
    + `${expectedPositions} = the shared weight counted once per position)`);
  for (let p = 0; p < pos.length; p++) {
    assert.ok(Math.abs(s.mean[1 + p] - oracle.mean[p]) < 0.04,
      label + `: E[y${p}] = ${s.mean[1 + p]}, oracle ${oracle.mean[p]}`);
    assert.ok(Math.abs(s.cov[1 + p][1 + p] - oracle.variance[p]) < 0.12,
      label + `: Var[y${p}] = ${s.cov[1 + p][1 + p]}, oracle `
      + oracle.variance[p]);
    assert.ok(Math.abs(s.cov[0][1 + p] - 1) < 0.06,
      label + `: cov(theta, y${p}) = ${s.cov[0][1 + p]}, oracle 1 `
      + '(= Var[theta])');
  }
  for (let a = 0; a < pos.length; a++) {
    for (let b = a + 1; b < pos.length; b++) {
      const o = oracle.crossCov(a, b);
      assert.ok(Math.abs(s.cov[1 + a][1 + b] - o) < 0.08,
        label + `: cov(y${a}, y${b}) = ${s.cov[1 + a][1 + b]}, oracle ${o} `
        + '(0 = a per-position re-draw rather than one shared theta)');
    }
  }

  // No double count. The same ensemble read under theta's weights SQUARED is a
  // draw at Normal(2, 1) — what summing the shared event over the positions
  // would produce — and the bands above must reject it.
  const dbl = new Float64Array(N);
  for (let i = 0; i < N; i++) dbl[i] = 2 * y.logWeights[i];
  const sd = stats(dbl, [th.samples], N);
  assert.ok(Math.abs(sd.mean[0] - 1) > 0.5,
    label + `: a doubled tilt reads E[theta] = ${sd.mean[0]}, which the `
    + 'oracle band must reject — it does not, so this witness proves nothing');
}

// Shared preamble: two broadcast cells at offsets 0 and 1 off theta.
const OFFS = 'offs = [0.0, 1.0]\n';

// -- 1. the general per-cell path (builtin distribution head) ---------------
//
// y[i, j] ~ Normal(theta_i + offs[j], 1). The fast-path registry declines an
// atom-dependent parameter, so this is the Ktot-worker-call general path.
test('a broadcast over a builtin distribution carries the parameter weights',
  async () => {
    await assertSharedTheta('general', H + TILT + OFFS
      + 'y ~ broadcast(Normal, mu = theta .+ offs, sigma = 1.0)\n',
      {
        mean: [1, 2],
        variance: [2, 2],
        crossCov: () => 1,
      }, 2);
  });

// The same path's degenerate all-scalar case: no collection argument, so the
// resolved axis vector is EMPTY and the output is one cell per atom (`dims` is
// [1] so `arrayMeasure` still builds a Value). y[i] ~ Normal(theta_i, 1), one
// position, so there is no cross-position covariance to read — the mean and
// cov(theta, y) carry the witness alone.
test('an all-scalar broadcast (no collection argument) carries them', async () => {
  await assertSharedTheta('all-scalar', H + TILT
    + 'y ~ broadcast(Normal, mu = theta, sigma = 1.0)\n',
    {
      mean: [1],
      variance: [2],
      crossCov: () => 1,
    }, 1);
});

// -- 2. the iid-bodied composite batch-flatten, D > 1 ----------------------
//
// y[i, j, r] ~ Normal(theta_i + offs[j], 1), 2 cells × 2 inner iid draws. All
// four positions of an atom share theta, so every cross-position covariance is
// Var[theta] — the inner axis is no more a re-draw of theta than the cell axis.
test('an iid-bodied broadcast carries the parameter weights across both axes',
  async () => {
    await assertSharedTheta('iid-composite D=2', H + TILT + OFFS
      + 'krow = m -> iid(Normal(mu = m, sigma = 1.0), 2)\n'
      + 'y ~ krow.(theta .+ offs)\n',
      {
        mean: [1, 1, 2, 2],
        variance: [2, 2, 2, 2],
        crossCov: () => 1,
      }, 4);
  });

// -- 3. the same executor's D == 1 branch ----------------------------------
test('the iid-bodied D = 1 branch carries them too', async () => {
  await assertSharedTheta('iid-composite D=1', H + TILT + OFFS
    + 'krow1 = m -> iid(Normal(mu = m, sigma = 1.0), 1)\n'
    + 'y ~ krow1.(theta .+ offs)\n',
    {
      mean: [1, 2],
      variance: [2, 2],
      crossCov: () => 1,
    }, 2);
});

// -- 4. the generative-bodied composite ------------------------------------
//
// y[i, j] = theta_i + offs[j] + u_ij with u = 2·Uniform(0, 1) drawn FRESH per
// (atom, cell): E[u] = 1, Var[u] = 1/3. So E[y_j] = 2 + offs[j] and
// Var[y_j] = Var[theta] + 1/3. The internal draw carries no weight of its own,
// which is why the cross-cell covariance is still Var[theta] exactly.
test('a generative-bodied broadcast carries the parameter weights',
  async () => {
    await assertSharedTheta('generative', H + TILT
      + 'xs = theta .+ [0.0, 1.0]\n'
      + 'x = elementof(reals)\n'
      + 'dz = 2.0 * draw(Uniform(interval(0.0, 1.0)))\n'
      + 'gy = x + dz\n'
      + 'gen = kernelof(gy, x = x)\n'
      + 'y ~ gen.(xs)\n',
      {
        mean: [2, 3],
        variance: [1 + 1 / 3, 1 + 1 / 3],
        crossCov: () => 1,
      }, 2);
  });

// -- 5. the joint-bodied composite ----------------------------------------
//
// Per cell j the kernel draws a 2-component joint variate, both components
// Normal(theta_i + offs[j], 1). The components are the variate's STRUCTURE, not
// a second weighting event: cov(a_j, b_j) = Var[theta] like every other pair.
test('a joint-bodied broadcast carries the parameter weights', async () => {
  await assertSharedTheta('joint', H + TILT + OFFS
    + 'obs = functionof(joint(a = Normal(mu = am, sigma = 1.0), '
    + 'b = Normal(mu = am, sigma = 1.0)), am = am)\n'
    + 'y = broadcast(obs, am = theta .+ offs)\n',
    {
      mean: [1, 1, 2, 2],
      variance: [2, 2, 2, 2],
      crossCov: () => 1,
    }, 4);
});

// -- 6. the jointchain-bodied composite ------------------------------------
//
// Per cell j: x0 = theta_i + offs[j] + e0, x1 = x0 + e1. So Var[x0] = 2,
// Var[x1] = 3 and cov(x0_j, x1_j) = Var[x0_j] = 2 WITHIN a cell — the chain
// couples the steps beyond the shared theta. ACROSS cells the only coupling is
// theta, so those covariances are 1. Positions are (cell, step) row-major:
// 0 = (0, x0), 1 = (0, x1), 2 = (1, x0), 3 = (1, x1).
test('a jointchain-bodied broadcast carries the parameter weights',
  async () => {
    await assertSharedTheta('jointchain', H + TILT + OFFS
      + 'step_prev = functionof(Normal(mu = _a_, sigma = 1.0), a = _a_)\n'
      + 'gc = functionof(jointchain(Normal(mu = _m_, sigma = 1.0), step_prev), '
      + 'm = _m_)\n'
      + 'y = broadcast(gc, m = theta .+ offs)\n',
      {
        mean: [1, 1, 2, 2],
        variance: [2, 3, 2, 3],
        // Same cell (0,1) and (2,3): the chain's own coupling, cov = Var[x0].
        crossCov: (a: number, b: number) => ((a >> 1) === (b >> 1) ? 2 : 1),
      }, 4);
  });

// -- 7. the nested-broadcast batch-flatten --------------------------------
//
// y[i, p, v] ~ Normal(theta_i + inner_offs[v], sigmas[p]). theta arrives as a
// closed-over per-atom ref inside the INNER broadcast's mu, which is the route
// that keeps both axis-ladder sizes resolvable. sigmas differ across the outer
// axis, so Var[y_{p,v}] = Var[theta] + sigmas[p]² separates the two.
// Positions are (outer p, inner v) row-major.
test('a nested broadcast carries the parameter weights', async () => {
  await assertSharedTheta('nested', H + TILT
    + 'inner_offs = [0.0, 1.0]\n'
    + 'sigmas = [1.0, 2.0]\n'
    + 'ok = functionof(Normal(mu = _mu_, sigma = _sigma_), '
    + 'mu = _mu_, sigma = _sigma_)\n'
    + 'shifted = inner_offs .+ theta\n'
    + 'pk = functionof(broadcast(ok, mu = shifted, sigma = _sigma_g_), '
    + 'sigma_g = _sigma_g_)\n'
    + 'y = broadcast(pk, sigma_g = sigmas)\n',
    {
      mean: [1, 2, 1, 2],
      variance: [2, 2, 5, 5],
      crossCov: () => 1,
    }, 4);
});

// -- the unweighted control ------------------------------------------------
//
// An all-equal weight array where `null` belongs would break
// `propagateLogWeights`'s reference-identity dedupe for every shape that was
// already right, so absence is the assertion, not "weights that happen to be
// flat". One case per fixed site.
test('an UNWEIGHTED parameter leaves every broadcast output unweighted',
  async () => {
    const PRIOR = H + 'theta ~ Normal(mu = 0.0, sigma = 1.0)\n' + OFFS;
    const CASES: [string, string][] = [
      ['general',
        'y ~ broadcast(Normal, mu = theta .+ offs, sigma = 1.0)\n'],
      ['iid-composite D=2',
        'krow = m -> iid(Normal(mu = m, sigma = 1.0), 2)\n'
        + 'y ~ krow.(theta .+ offs)\n'],
      ['iid-composite D=1',
        'krow1 = m -> iid(Normal(mu = m, sigma = 1.0), 1)\n'
        + 'y ~ krow1.(theta .+ offs)\n'],
      ['generative',
        'x = elementof(reals)\n'
        + 'dz = 2.0 * draw(Uniform(interval(0.0, 1.0)))\n'
        + 'gy = x + dz\n'
        + 'gen = kernelof(gy, x = x)\n'
        + 'y ~ gen.(theta .+ offs)\n'],
      ['joint',
        'obs = functionof(joint(a = Normal(mu = am, sigma = 1.0), '
        + 'b = Normal(mu = am, sigma = 1.0)), am = am)\n'
        + 'y = broadcast(obs, am = theta .+ offs)\n'],
      ['jointchain',
        'step_prev = functionof(Normal(mu = _a_, sigma = 1.0), a = _a_)\n'
        + 'gc = functionof(jointchain(Normal(mu = _m_, sigma = 1.0), '
        + 'step_prev), m = _m_)\n'
        + 'y = broadcast(gc, m = theta .+ offs)\n'],
      ['nested',
        'sigmas = [1.0, 2.0]\n'
        + 'ok = functionof(Normal(mu = _mu_, sigma = _sigma_), '
        + 'mu = _mu_, sigma = _sigma_)\n'
        + 'shifted = offs .+ theta\n'
        + 'pk = functionof(broadcast(ok, mu = shifted, sigma = _sigma_g_), '
        + 'sigma_g = _sigma_g_)\n'
        + 'y = broadcast(pk, sigma_g = sigmas)\n'],
    ];
    for (const [label, tail] of CASES) {
      const { ctx } = ctxFor(PRIOR + tail, 4000);
      const y = await ctx.getMeasure('y');
      assert.equal(y.logWeights, null, label
        + ': an all-equal weight array here would break the dedupe contract');
      assert.equal(y.logTotalmass, 0, label + ': logTotalmass');
      assert.equal(y.n_eff, 4000, label + ': n_eff');
    }
  });

// -- two weighted parameters ----------------------------------------------
//
// a is tilted to Normal(1, 1) and b to Normal(0.5, 1) — two DISTINCT weighting
// events, so the atom's weight is their product (the sum in log space). sigma
// is exactly 1, so b enters y's law not at all; it enters y's ENSEMBLE because
// y is a draw at b's atom, and reading b's marginal off y's weights must still
// recover its own tilt.
test('two weighted parameters of a broadcast combine as a joint IS weight',
  async () => {
    const { proc, ctx } = ctxFor(H
      + 't1 = x -> exp(x)\n'
      + 't2 = x -> exp(0.5 * x)\n'
      + 'ma = normalize(weighted(t1, Normal(mu = 0.0, sigma = 1.0)))\n'
      + 'mb = normalize(weighted(t2, Normal(mu = 0.0, sigma = 1.0)))\n'
      + 'a ~ ma\n'
      + 'b ~ mb\n'
      + 'sb = 1.0 + 0.0 * b\n'
      + OFFS
      + 'y ~ broadcast(Normal, mu = a .+ offs, sigma = sb)\n', N);
    assert.equal(
      proc.diagnostics.filter((d: any) => d.severity === 'error').length, 0);
    const a = await ctx.getMeasure('a');
    const b = await ctx.getMeasure('b');
    const sb = await ctx.getMeasure('sb');
    const y = await ctx.getMeasure('y');
    assert.ok(y.logWeights, 'y must carry both parameter streams');
    assert.ok(y.logWeights !== a.logWeights);
    assert.ok(y.logWeights !== sb.logWeights);
    for (let i = 0; i < 64; i++) {
      assert.ok(Math.abs(y.logWeights[i] - (a.logWeights[i] + sb.logWeights[i]))
        < 1e-12, `atom ${i}: y's weight is not the sum of the two streams`);
    }
    const pos = positions(y, N);
    const s = stats(y.logWeights, [a.samples, b.samples].concat(pos), N);
    assert.ok(Math.abs(s.mean[0] - 1) < 0.03, `E[a] = ${s.mean[0]}, oracle 1`);
    assert.ok(Math.abs(s.mean[1] - 0.5) < 0.03,
      `E[b] = ${s.mean[1]}, oracle 0.5`);
    assert.ok(Math.abs(s.mean[2] - 1) < 0.04, `E[y0] = ${s.mean[2]}, oracle 1`);
    assert.ok(Math.abs(s.mean[3] - 2) < 0.04, `E[y1] = ${s.mean[3]}, oracle 2`);
  });
