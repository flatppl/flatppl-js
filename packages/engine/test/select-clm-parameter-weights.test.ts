'use strict';
// A select's gather and a CLM body's fed output carry their weighted parent's
// importance weights.
//
// THE DEFECT, two sites.
//
// `matSelect` closed its gather with
// `scalarMeasureN(out, { logWeights: null, logTotalmass: 0, n_eff: N })`, and
// reduced both the selector and every branch to `.samples` on the way in
// (`empirical.materialiseUniform(m).samples` — a no-op read for a measure that
// already carries weights, so the weights were simply discarded). An INLINE
// branch also drew through the worker without collecting the parameter measures
// `collectRefArrays` resolved, so `matSample`'s parameter-weight fold had
// nothing to fold.
//
// `matClm`'s MARGINAL/FED branch walks its body with the boundary parents bound
// into `ctx._extraRefArrays` as per-atom POSITION columns (`clm.feedInputs` →
// `measureToRefValue`). `collectRefArrays` consults that overlay BEFORE
// `getMeasure`, so the body's own draws never see the parent measures and the
// weight channel had nowhere to enter — `clm.feedInputs` returned no such
// channel at all.
//
// `matClm`'s RETAIN branch (reduce = null, no history, not fed) is clean by
// construction and is not covered here: it returns `materialiseMeasureIR(
// ir.body, ctx)` with no feed, so its output IS the body's own measure and the
// joint walk's `propagateLogWeights` has already folded the fields' weights.
// Nothing is rebuilt, so nothing can be dropped.
//
// `matSelect`'s SYNTHESIZED-selector source (`d.synthWeights`, constant branch
// weights with no named selector) shares the fixed gather, so it is fixed with
// the rest, but it has no witness here: no FlatPPL source spelling reaches it.
// `superpose` always resolves to a named binding handled by `matSuperpose`
// (checked by name, inline, as a `joint` field, as a `kchain` base, in a
// `functionof` body, and via `ksuperpose`), and both `select`-IR emitters in
// `derivations.ts` carry `logweights` / `selectorName` rather than the
// `weighted(…)` branch args the IR bridge reads constant weights from. The
// documented remaining route is the viewer's kernel-plot select.
//
// THE ORACLE, closed form and exact. `normalize(weighted(x -> exp(x),
// Normal(0, 1)))` is exactly Normal(1, 1): e^x φ(x) ∝ exp(−(x−1)²/2). So
// E[theta] = Var[theta] = 1 and each case's law follows below; every band was
// re-derived against an independent 4e6-draw Monte Carlo. Dropping the weights
// reads Normal(0, 1) instead, E[theta] = 0.
//
// The CROSS-POSITION covariance is the witness a mean-only check misses: an
// atom's positions are independent GIVEN the parent and share one draw of it,
// so the covariance carries the parent's whole variance. A per-position re-draw
// reads 0 there.
//
// Spec §07 sec:functions makes `ifelse(cond, a, b)` return "`a` if `cond` is
// true, `b` otherwise", so `c ~ Bernoulli(p); y ~ ifelse(c, a, b)` is §06 "The
// measure monad"'s bind,
// $(\nu \mathbin{\texttt{>>=}} \kappa)(B) = \int_X \kappa(x)(B)\, d\nu(x)$,
// over the selector measure. §06 `kchain`'s "Equivalence with stochastic nodes"
// reads `y = kchain(M, K)` as `theta ~ M; y ~ K(theta)` — the same bind. Both
// integrate against the parameter MEASURE, not against the proposal its atoms
// were drawn from.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const H = 'flatppl_compat = "0.1"\n';
const N = 200000;

const TILT = 'tilt = x -> exp(x)\n'
  + 'tm = normalize(weighted(tilt, Normal(mu = 0.0, sigma = 1.0)))\n'
  + 'theta ~ tm\n';
// The same shape with no tilt, for the unweighted controls.
const FLAT = 'tm = Normal(mu = 0.0, sigma = 1.0)\n'
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

// Every per-atom column a measure carries, in order: a record's fields, a
// tuple's elements, and an [n, P] buffer's P trailing positions. A CLM body
// output takes all three shapes (scalar kernel, joint-bodied kernel,
// iid-bodied kernel, retained history joint).
function columns(m: any, n: number): Float64Array[] {
  if (m.elems) {
    return m.elems.reduce(
      (acc: Float64Array[], e: any) => acc.concat(columns(e, n)), []);
  }
  if (m.fields) {
    return Object.keys(m.fields).reduce(
      (acc: Float64Array[], k: string) => acc.concat(columns(m.fields[k], n)), []);
  }
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

type Oracle = {
  // E[position p] and Var[position p].
  mean: number[];
  variance: number[];
  // cov(theta, position p) — Var[theta] = 1 wherever the tilt reaches the
  // position through the mean alone.
  thetaCov: number[];
  // cov(position a, position b) for a != b.
  crossCov: (a: number, b: number) => number;
};

function build(label: string, src: string) {
  const { proc, ctx } = ctxFor(src, N);
  assert.equal(
    proc.diagnostics.filter((d: any) => d.severity === 'error').length, 0,
    label + ': ' + proc.diagnostics.map((d: any) => d.message).join('; '));
  return ctx;
}

// The witness every fixed site shares. Reads the output's OWN ensemble — under
// its own `logWeights`, which is the whole point — against the closed form,
// then excludes both failure modes.
async function assertCarriesTilt(
  label: string, src: string, oracle: Oracle,
) {
  const ctx = build(label, H + TILT + src);
  const th = await ctx.getMeasure('theta');
  const y = await ctx.getMeasure('y');
  assert.ok(y.logWeights, label + ": the output must carry theta's importance "
    + 'weights — without them every position is a draw at Normal(0, 1)');
  // Reference identity, not just equal values: `propagateLogWeights` returns a
  // single-stream parent's array ITSELF, and that identity is how the engine's
  // independence dedupe recognises the stream downstream. A second array here
  // means the fold summed the one shared event more than once.
  //
  // `assert.ok` on the comparison, not `assert.equal`: a failing `equal`
  // renders both operands, and inspecting an N-atom Float64Array to build the
  // diff takes minutes and can take the process down with it.
  assert.ok(y.logWeights === th.logWeights,
    label + ': theta is the only weighted parent, so its stream must be '
    + 'passed forward by reference');
  // `normalize` leaves theta's weights summing to one, so the output still
  // reports mass 0 in log space.
  assert.ok(Math.abs(y.logTotalmass) < 1e-9,
    label + ': logTotalmass ' + y.logTotalmass);
  assert.ok(Math.abs(y.n_eff - th.n_eff) < 1e-9,
    label + ': n_eff ' + y.n_eff + " must inherit theta's " + th.n_eff
    + ', not report a confident N');

  const pos = columns(y, N);
  assert.equal(pos.length, oracle.mean.length, label + ': position count');
  for (const c of pos) {
    for (let i = 0; i < 64; i++) {
      assert.ok(Number.isFinite(c[i]), label + ': atom ' + i + ' is not finite');
    }
  }
  const s = stats(y.logWeights, [th.samples].concat(pos), N);
  assert.ok(Math.abs(s.mean[0] - 1) < 0.03,
    label + `: E[theta] = ${s.mean[0]}, oracle 1 (0 = weights dropped)`);
  for (let p = 0; p < pos.length; p++) {
    assert.ok(Math.abs(s.mean[1 + p] - oracle.mean[p]) < 0.06,
      label + `: E[y${p}] = ${s.mean[1 + p]}, oracle ${oracle.mean[p]}`);
    assert.ok(Math.abs(s.cov[1 + p][1 + p] - oracle.variance[p]) < 0.5,
      label + `: Var[y${p}] = ${s.cov[1 + p][1 + p]}, oracle `
      + oracle.variance[p]);
    assert.ok(Math.abs(s.cov[0][1 + p] - oracle.thetaCov[p]) < 0.07,
      label + `: cov(theta, y${p}) = ${s.cov[0][1 + p]}, oracle `
      + oracle.thetaCov[p]);
  }
  for (let a = 0; a < pos.length; a++) {
    for (let b = a + 1; b < pos.length; b++) {
      const o = oracle.crossCov(a, b);
      assert.ok(Math.abs(s.cov[1 + a][1 + b] - o) < 0.1,
        label + `: cov(y${a}, y${b}) = ${s.cov[1 + a][1 + b]}, oracle ${o} `
        + '(0 = a per-position re-draw rather than one shared theta)');
    }
  }

  // No double count. The same ensemble read under theta's weights SQUARED is a
  // draw at Normal(2, 1) — what summing the one shared event twice would
  // produce — and the band above must reject it.
  const dbl = new Float64Array(N);
  for (let i = 0; i < N; i++) dbl[i] = 2 * y.logWeights[i];
  const sd = stats(dbl, [th.samples], N);
  assert.ok(Math.abs(sd.mean[0] - 1) > 0.5,
    label + `: a doubled tilt reads E[theta] = ${sd.mean[0]}, which the `
    + 'oracle band must reject — it does not, so this witness proves nothing');
}

// The unweighted control: with no tilt anywhere, `logWeights` must stay ABSENT
// rather than become an all-equal array — the shape the dedupe relies on.
async function assertNoWeights(label: string, src: string) {
  const ctx = build(label, H + FLAT + src);
  const y = await ctx.getMeasure('y');
  assert.equal(y.logWeights, null,
    label + ': an unweighted model must leave logWeights absent');
  assert.equal(y.n_eff, N, label + ': n_eff ' + y.n_eff);
  assert.equal(y.logTotalmass, 0, label + ': logTotalmass ' + y.logTotalmass);
}

// =====================================================================
// matSelect — the gather
// =====================================================================

// Shared branch pair: both branches parameterised by theta, 10 apart. Branch 0
// is the TRUE branch (matSelect's `sel ? 0 : 1` gather), so with
// `c ~ Bernoulli(0.5)` the mixture is theta + 10·Bernoulli(0.5) + Normal(0, 1):
// E[y] = 1 + 5 = 6, Var[y] = Var[theta] + 25 + 1 = 27, and cov(theta, y)
// = Var[theta] = 1.
const NAMED_BRANCHES = 'a = Normal(mu = theta, sigma = 1.0)\n'
  + 'b = Normal(mu = theta + 10.0, sigma = 1.0)\n';
const MIX_ORACLE: Oracle = {
  mean: [6], variance: [27], thetaCov: [1], crossCov: () => 0,
};

test('an ifelse over named measure branches carries the parameter weights',
  async () => {
    await assertCarriesTilt('named branches', NAMED_BRANCHES
      + 'c ~ Bernoulli(p = 0.5)\n'
      + 'y ~ ifelse(c, a, b)\n', MIX_ORACLE);
  });

// The INLINE-branch source: each branch is drawn by the worker rather than read
// off a named binding, so the weights can only come from the parameter measures
// `collectRefArrays` resolved.
test('an ifelse over inline distribution branches carries them', async () => {
  await assertCarriesTilt('inline branches',
    'c ~ Bernoulli(p = 0.5)\n'
    + 'y ~ ifelse(c, Normal(mu = theta, sigma = 1.0), '
    + 'Normal(mu = theta + 10.0, sigma = 1.0))\n', MIX_ORACLE);
});

// A NON-closed-form condition (`u > 0`): the selector is deterministic given
// the ensemble, so `classifyIfelse` emits per-atom indicator branch weights
// instead of a Bernoulli p. u is independent of theta, so P(true) = 1/2 and the
// law is the same mixture.
test('an ifelse over a non-closed-form condition carries them', async () => {
  await assertCarriesTilt('indicator selector', NAMED_BRANCHES
    + 'u ~ Normal(mu = 0.0, sigma = 1.0)\n'
    + 'c = u > 0.0\n'
    + 'y ~ ifelse(c, a, b)\n', MIX_ORACLE);
});

// Only ONE branch is tilted. The b-branch atoms are independent of theta, so
// the tilt leaves their contribution unchanged in expectation, and
// E[y] = 0.5·E[theta] + 0.5·10 = 5.5 — halfway between the oracle and the
// dropped-weights 5.0, which is exactly what the mean-only reading of a
// half-weighted mixture must land on. Var[y] = 52 − 5.5² = 21.75 and
// cov(theta, y) = 0.5·E[theta²] + 5·E[theta] − 5.5 = 0.5.
test('an ifelse with a single weighted branch carries that branch\'s weights',
  async () => {
    await assertCarriesTilt('one tilted branch',
      'a = Normal(mu = theta, sigma = 1.0)\n'
      + 'b = Normal(mu = 10.0, sigma = 1.0)\n'
      + 'c ~ Bernoulli(p = 0.5)\n'
      + 'y ~ ifelse(c, a, b)\n',
      { mean: [5.5], variance: [21.75], thetaCov: [0.5], crossCov: () => 0 });
  });

// The SELECTOR is the weighted parent, not the branches: `c ~ Bernoulli(p =
// invlogit(theta))` carries theta's stream through `matEvaluate` and
// `matSample`. Both branches are unweighted constants 10 apart, so the whole
// tilt reaches y through the selection probability alone.
//
// Oracle: with theta ~ Normal(1, 1), E[invlogit(theta)] = 0.696735 by
// quadrature, so E[y] = 10·(1 − 0.696735) = 3.03265,
// Var[y] = (101 − 100·0.696735) − 3.03265² = 22.1295, and
// cov(theta, y) = −10·cov(theta, invlogit(theta)) = −1.77943. Dropping the
// weights reads theta ~ Normal(0, 1), where E[invlogit] = 1/2 by symmetry and
// E[y] = 5 — so the mean alone separates the two.
test('an ifelse whose SELECTOR is the weighted parent carries its weights',
  async () => {
    await assertCarriesTilt('weighted selector',
      'pv = 1.0 / (1.0 + exp(0.0 - theta))\n'
      + 'c ~ Bernoulli(p = pv)\n'
      + 'a = Normal(mu = 0.0, sigma = 1.0)\n'
      + 'b = Normal(mu = 10.0, sigma = 1.0)\n'
      + 'y ~ ifelse(c, a, b)\n',
      {
        mean: [3.032653], variance: [22.129547], thetaCov: [-1.779434],
        crossCov: () => 0,
      });
  });

// A select as a `joint` FIELD: the enclosing record's own
// `propagateLogWeights` has to find the stream on the field's measure, so the
// gather's output is what carries it out. Field v is an independent
// Normal(0, 1) — E 0, Var 1, and cov(theta, v) = cov(u, v) = 0.
test('a select inside a joint carries the weights out to the record', async () => {
  await assertCarriesTilt('select in a joint', NAMED_BRANCHES
    + 'c ~ Bernoulli(p = 0.5)\n'
    + 'sel = ifelse(c, a, b)\n'
    + 'y = joint(u = sel, v = Normal(mu = 0.0, sigma = 1.0))\n',
    { mean: [6, 0], variance: [27, 1], thetaCov: [1, 0], crossCov: () => 0 });
});

test('an unweighted select leaves logWeights absent', async () => {
  await assertNoWeights('select control', NAMED_BRANCHES
    + 'c ~ Bernoulli(p = 0.5)\n'
    + 'y ~ ifelse(c, a, b)\n');
});

// =====================================================================
// matClm — the fed body output
// =====================================================================

const K1 = 'mu1 = elementof(reals)\n'
  + 'K1 = functionof(Normal(mu = mu1, sigma = 1.0), mu1 = mu1)\n';
// A tilted base drawn once more, so a chain over it has Var 2 at the base.
const BASE = 'base2 = Normal(mu = theta, sigma = 1.0)\n';

// The tilted measure IS the chain base: the boundary feed binds tm's atoms and
// the final kernel draws at them. y = theta + Normal(0, 1), so E[y] = 1,
// Var[y] = Var[theta] + 1 = 2, cov(theta, y) = Var[theta] = 1.
test('a marginal kchain over a weighted base carries its weights', async () => {
  await assertCarriesTilt('kchain(tm, K)', K1 + 'y = kchain(tm, K1)\n',
    { mean: [1], variance: [2], thetaCov: [1], crossCov: () => 0 });
});

// The base is itself a draw at theta, so its weights arrive already folded by
// `matSample` and the feed must not lose them a second time.
// y = theta + Normal(0, 1) + Normal(0, 1): Var[y] = 1 + 1 + 1 = 3.
test('a marginal kchain over a theta-parameterised base carries them',
  async () => {
    await assertCarriesTilt('kchain(base2, K)', BASE + K1
      + 'y = kchain(base2, K1)\n',
      { mean: [1], variance: [3], thetaCov: [1], crossCov: () => 0 });
  });

// A RECORD base: the boundary feeds the named `%local` param columns per field
// rather than one scalar column. y = base2 + rb + Normal(0, 1), so
// Var[y] = 2 + 1 + 1 = 4.
test('a record-base marginal kchain carries them', async () => {
  await assertCarriesTilt('record-base kchain', BASE
    + 'jb = joint(ra = base2, rb = Normal(mu = 0.0, sigma = 1.0))\n'
    + 'ra = elementof(reals)\n'
    + 'rb = elementof(reals)\n'
    + 'Kr = functionof(Normal(mu = ra + rb, sigma = 1.0), ra = ra, rb = rb)\n'
    + 'y = kchain(jb, Kr)\n',
    { mean: [1], variance: [4], thetaCov: [1], crossCov: () => 0 });
});

// An IID-bodied final kernel: two inner positions per atom, both drawn at the
// SAME fed base atom. Var[y_p] = Var[base2] + 1 = 3, and the cross-position
// covariance is Var[base2] = 2 — the inner axis is no more a re-draw of the
// base than of theta, so a 0 there would mean a per-position re-draw.
test('a marginal kchain with an iid-bodied kernel carries them across the '
  + 'inner axis', async () => {
    await assertCarriesTilt('iid-bodied kchain', BASE
      + 'mu1 = elementof(reals)\n'
      + 'Ki = functionof(iid(Normal(mu = mu1, sigma = 1.0), 2), mu1 = mu1)\n'
      + 'y = kchain(base2, Ki)\n',
      { mean: [1, 1], variance: [3, 3], thetaCov: [1, 1], crossCov: () => 2 });
  });

// A JOINT-bodied final kernel: the output is a record, and the stream has to
// reach its top level (the composite-measure invariant keeps the leaves' own
// `logWeights` stripped). Both components draw at the same fed atom, so the
// cross-component covariance is Var[base2] = 2 like the iid case.
test('a marginal kchain with a joint-bodied kernel carries them', async () => {
  await assertCarriesTilt('joint-bodied kchain', BASE
    + 'mu1 = elementof(reals)\n'
    + 'Kj = functionof(joint(p = Normal(mu = mu1, sigma = 1.0), '
    + 'q = Normal(mu = mu1, sigma = 1.0)), mu1 = mu1)\n'
    + 'y = kchain(base2, Kj)\n',
    { mean: [1, 1], variance: [3, 3], thetaCov: [1, 1], crossCov: () => 2 });
});

// A jointchain whose shared latent forces the fed branch: the output is the
// retained TUPLE [base2, kernel draw]. Element 0 is base2 itself (Var 2) and
// element 1 draws at it (Var 3); cov(base2, y_1) = Var[base2] = 2.
test('a fed jointchain tuple output carries them', async () => {
  await assertCarriesTilt('jointchain tuple', BASE + K1
    + 'y = jointchain(base2, K1)\n',
    { mean: [1, 1], variance: [2, 3], thetaCov: [1, 1], crossCov: () => 2 });
});

test('an unweighted marginal kchain leaves logWeights absent', async () => {
  await assertNoWeights('kchain control', K1 + 'y = kchain(tm, K1)\n');
});
