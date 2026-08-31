'use strict';
// A chain step that DECLARES its boundary input is checked against the type
// the chain feeds it.
//
// THE DEFECT. Two `kchain` shapes sampled NaN for every atom, silently:
//
//   1. A 3-step marginal chain over scalar-declared kernels,
//      `kchain(base, K1, K2)` — the `marginalHistoryBody` path.
//   2. A positional-`joint` base, `kchain(joint(base), K1)` — the
//      `measureToParamValue` whole-variate bind.
//
// ONE cause. §06 dependent composition lowers `kchain(M1, K2, K3)` to
// `a ~ M1; b ~ K2(a); c ~ K3([a, b])`, so the step is fed the `cat` of every
// variate to its left, and §06 `joint` makes a positional joint's variate "the
// `cat` of the component variates" — "all scalars (yielding a vector)". Both
// shapes therefore feed a step an ARRAY where the step declares a real, which
// reaches `Normal`'s `mu` as a vector and goes NaN.
//
// Nothing type-checked it. `functionof(body, mu = mu)` records the boundary in
// `paramSources` and drops it from `kwargs`, so `inferReification` finds no
// boundary expression and publishes the input as `any` — which unifies with
// anything, so the chain's boundary matcher could not reject the fed array.
// The body re-check (`checkChainStepBodies`) could not reach the spelling
// either: an identifier-bound boundary's body refs are `self`-namespaced
// module names, and `inferRef` resolves those through `loweredModule.bindings`
// rather than through the pushed scope, so the body was re-checked against the
// boundary NODE's own type (`real`) instead of the fed one.
//
// The PLACEHOLDER spelling of the same two models was already a located error,
// and §06 gives both spellings one lowering — so the silent half was the
// non-conformant one. Refusal, not a corrected sample, is the fix: the model
// asks a real-valued input to receive a vector, and §06 names the construct
// that means the previous state alone (`markovchain`) and the one that names a
// joint's components (the keyword form / `relabel`).
//
// THE ORACLES are closed-form and hand-derived, then cross-checked against an
// independent 4e6-draw Julia Monte Carlo. With `theta ~ Normal(1, 1)` (which
// `normalize(weighted(x -> exp(x), Normal(0, 1)))` is exactly, since
// e^x φ(x) ∝ exp(−(x−1)²/2)) and `base = Normal(mu = theta, sigma = 1)`:
//
//   3-step, final kernel `sum(v)`: y = s0 + s1 + eps = 2·theta + 2·e0 + e1 +
//   eps, so E 2, Var 4+4+1+1 = 10, cov(theta, y) = 2·Var[theta] = 2, and two
//   iid-bodied positions have cov = Var(2·theta + 2·e0 + e1) = 9.
//
//   positional-joint base, one component: y = s0 + eps = theta + e0 + eps, so
//   E 1, Var 3, cov(theta, y) = 1, and two iid-bodied positions have
//   cov = Var(theta + e0) = 2.
//
//   positional-joint base, two components: y = s0 + n + eps, so Var 4.
//
// The CROSS-POSITION covariance is the discriminator a moment-only check
// misses: an atom's positions are drawn at one shared fed history, so the
// covariance carries that history's whole variance. A per-position re-draw of
// the history reads 0 there.
//
// These are the two shapes the #238 sweep left UN-VERIFIED for the weight
// question, because no moment witness is possible while every atom is NaN.
// The weighted rows below close it: both feed paths carry a weighted parent's
// importance stream, by reference, once.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('..');
const { ctxFor } = require('./_ctx-factory.ts');

const H = 'flatppl_compat = "0.1"\n';
const N = 200000;

const TILT = 'tilt = x -> exp(x)\n'
  + 'tm = normalize(weighted(tilt, Normal(mu = 0.0, sigma = 1.0)))\n'
  + 'theta ~ tm\n';
// The same shape with no tilt, for the unweighted controls.
const FLAT = 'tm = Normal(mu = 0.0, sigma = 1.0)\n'
  + 'theta ~ tm\n';
const BASE = 'base2 = Normal(mu = theta, sigma = 1.0)\n';
// A scalar-declared step kernel: legal at a boundary fed one variate.
const K1 = 'mu1 = elementof(reals)\n'
  + 'K1 = functionof(Normal(mu = mu1, sigma = 1.0), mu1 = mu1)\n';

function errorsOf(src: string) {
  return processSource(src).diagnostics.filter((d: any) => d.severity === 'error');
}

// Every per-atom column a measure carries, in order: a record's fields, a
// tuple's elements, and an [n, P] buffer's P trailing positions.
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

type Oracle = {
  mean: number[];
  variance: number[];
  // cov(theta, position p).
  thetaCov: number[];
  // cov(position a, position b) for a != b.
  crossCov: number;
};

function build(label: string, src: string) {
  const { proc, ctx } = ctxFor(src, N);
  assert.equal(
    proc.diagnostics.filter((d: any) => d.severity === 'error').length, 0,
    label + ': ' + proc.diagnostics.map((d: any) => d.message).join('; '));
  return ctx;
}

// Read the output's own ensemble against the closed form, and assert every
// atom is finite — the whole point, since both shapes were entirely NaN.
async function assertMoments(
  label: string, head: string, src: string, oracle: Oracle, tilted: boolean,
) {
  const ctx = build(label, H + head + src);
  const th = await ctx.getMeasure('theta');
  const y = await ctx.getMeasure('y');
  const pos = columns(y, N);
  assert.equal(pos.length, oracle.mean.length, label + ': position count');
  for (const c of pos) {
    for (let i = 0; i < N; i++) {
      assert.ok(Number.isFinite(c[i]), label + ': atom ' + i + ' is not finite');
    }
  }
  if (tilted) {
    assert.ok(y.logWeights, label + ": the output must carry theta's importance "
      + 'weights — without them every position is a draw at Normal(0, 1)');
    // Reference identity, not just equal values: `propagateLogWeights` returns
    // a single-stream parent's array ITSELF, and that identity is how the
    // engine's independence dedupe recognises the stream downstream. A second
    // array here means the fold summed the one shared event more than once.
    //
    // `assert.ok` on the comparison, not `assert.equal`: a failing `equal`
    // renders both operands, and inspecting an N-atom Float64Array to build
    // the diff takes minutes and can take the process down with it.
    assert.ok(y.logWeights === th.logWeights,
      label + ': theta is the only weighted parent, so its stream must be '
      + 'passed forward by reference');
    // `normalize` leaves theta's weights summing to one, so the output still
    // reports mass 0 in log space.
    assert.ok(Math.abs(y.logTotalmass) < 1e-9,
      label + ': logTotalmass ' + y.logTotalmass);
  } else {
    assert.equal(y.logWeights, null,
      label + ': an unweighted model must leave logWeights absent');
  }

  const s = stats(y.logWeights, [th.samples].concat(pos), N);
  const thOracle = tilted ? 1 : 0;
  assert.ok(Math.abs(s.mean[0] - thOracle) < 0.03,
    label + `: E[theta] = ${s.mean[0]}, oracle ${thOracle}`);
  for (let p = 0; p < pos.length; p++) {
    assert.ok(Math.abs(s.mean[1 + p] - oracle.mean[p]) < 0.08,
      label + `: E[y${p}] = ${s.mean[1 + p]}, oracle ${oracle.mean[p]}`);
    assert.ok(Math.abs(s.cov[1 + p][1 + p] - oracle.variance[p]) < 0.5,
      label + `: Var[y${p}] = ${s.cov[1 + p][1 + p]}, oracle `
      + oracle.variance[p]);
    assert.ok(Math.abs(s.cov[0][1 + p] - oracle.thetaCov[p]) < 0.09,
      label + `: cov(theta, y${p}) = ${s.cov[0][1 + p]}, oracle `
      + oracle.thetaCov[p]);
  }
  for (let a = 0; a < pos.length; a++) {
    for (let b = a + 1; b < pos.length; b++) {
      assert.ok(Math.abs(s.cov[1 + a][1 + b] - oracle.crossCov) < 0.35,
        label + `: cov(y${a}, y${b}) = ${s.cov[1 + a][1 + b]}, oracle `
        + oracle.crossCov
        + ' (0 = a per-position re-draw rather than one shared history)');
    }
  }

  if (!tilted) return;
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

// =====================================================================
// The refusal: a declared boundary is checked against the fed type
// =====================================================================

// Shape 1. Both steps declare a real, so step 2 is fed the `cat` of the two
// variates to its left and there is no correct draw. §06 names the construct
// that means the previous state alone.
test('a 3-step marginal kchain of scalar-declared kernels is a located error',
  () => {
    const errs = errorsOf(H + FLAT + BASE + K1
      + 'mu2 = elementof(reals)\n'
      + 'K2 = functionof(Normal(mu = mu2, sigma = 1.0), mu2 = mu2)\n'
      + 'y = kchain(base2, K1, K2)\n');
    assert.equal(errs.length, 1, errs.map((d: any) => d.message).join(' | '));
    const m = errs[0].message;
    assert.match(m, /step boundary 1 → 2/);
    assert.match(m, /array of real \(length 2\)/);
    assert.match(m, /input "mu2" of type real/);
    assert.match(m, /spec §06 dependent composition/);
    assert.match(m, /markovchain\(kernel, init, n\)/);
    assert.ok(errs[0].loc, 'the error must be located');
  });

// Shape 2. §06 `joint`: the positional form's variate is the `cat` of the
// component variates, "all scalars (yielding a vector)" — so a one-component
// positional joint feeds a length-1 vector, not the scalar.
test('a positional-joint base against a scalar-declared kernel is a located '
  + 'error', () => {
    const errs = errorsOf(H + FLAT + BASE + K1
      + 'jp = joint(base2)\n'
      + 'y = kchain(jp, K1)\n');
    assert.equal(errs.length, 1, errs.map((d: any) => d.message).join(' | '));
    const m = errs[0].message;
    assert.match(m, /step boundary 0 → 1/);
    assert.match(m, /array of real \(length 1\)/);
    assert.match(m, /input "mu1" of type real/);
    assert.match(m, /positional `joint`'s variate is the `cat`/);
    assert.match(m, /relabel/);
    assert.ok(errs[0].loc, 'the error must be located');
  });

// The placeholder spelling of the same two models. Already refused before this
// change, and §06 gives both spellings one lowering — equality of the two
// verdicts is the assertion.
test('the placeholder spelling of both shapes stays a located error', () => {
  const three = errorsOf(H + FLAT + BASE
    + 'y = kchain(base2, fn(Normal(mu = _, sigma = 1.0)), '
    + 'fn(Normal(mu = _, sigma = 1.0)))\n');
  assert.equal(three.length, 1, three.map((d: any) => d.message).join(' | '));
  assert.match(three[0].message, /array of real \(length 2\)/);
  assert.match(three[0].message, /markovchain\(kernel, init, n\)/);

  const jb = errorsOf(H + FLAT + BASE
    + 'jp = joint(base2)\n'
    + 'y = kchain(jp, fn(Normal(mu = _, sigma = 1.0)))\n');
  assert.equal(jb.length, 1, jb.map((d: any) => d.message).join(' | '));
  assert.match(jb[0].message, /array of real \(length 1\)/);
});

// =====================================================================
// No false positive: what must still type
// =====================================================================

// A 2-step chain feeds ONE variate, which binds whole — the #238 shapes, every
// one a declared boundary.
test('a declared scalar boundary fed one scalar variate still types', () => {
  assert.deepEqual(
    errorsOf(H + FLAT + BASE + K1 + 'y = kchain(base2, K1)\n')
      .map((d: any) => d.message), []);
  assert.deepEqual(
    errorsOf(H + FLAT + BASE + K1 + 'y = jointchain(base2, K1)\n')
      .map((d: any) => d.message), []);
});

// A RECORD fed to a lone declared input is a SEPARATE surface and is
// deliberately not touched. The engine's chain runtime unwraps a single-field
// record into the input whatever the field is named — measured: this model
// samples Var[y] = 2 against its closed form — which no reading of §04's
// splat-by-field-name predicts. Enforcing the declared type there would refuse
// a program that works today. Carded in flatppl-dev/TODO-flatppl-js.md.
test('a record fed to a lone declared input still types (name mismatch and '
  + 'all)', () => {
    assert.deepEqual(
      errorsOf(H + 'prior = joint(theta = Normal(mu = 0.0, sigma = 1.0))\n'
        + 'mu = elementof(reals)\n'
        + 'K = functionof(Normal(mu = mu, sigma = 1.0), mu = mu)\n'
        + 'y = jointchain(prior, K)\n').map((d: any) => d.message), []);
  });

// A `relabel`d measure keeps its UN-relabelled variate type (`inferRelabel`'s
// measure arm is labels-only, by design), so an array-typed variate to the
// left of a step may really be the record §06's `relabel` produces. The
// declared-input check is withheld once a relabel-rooted step is to the left —
// refusing on a known under-approximation would reject a program the spec
// allows.
test('a relabel-rooted base withholds the declared-input check', () => {
  const RB = H
    + 'jd = joint(Normal(mu = 0.0, sigma = 1.0), Beta(alpha = 1.0, beta = 1.0))\n'
    + 'mu = elementof(reals)\n'
    + 'K = functionof(Normal(mu = mu, sigma = 1.0), mu = mu)\n';
  // Through a binding ref.
  assert.deepEqual(
    errorsOf(RB + 'rb = relabel(jd, ["a", "b"])\n' + 'y = kchain(rb, K)\n')
      .map((d: any) => d.message), []);
  // Written inline as the step. Same withholding — the reason is the op, not
  // the spelling.
  assert.deepEqual(
    errorsOf(RB + 'y = kchain(relabel(jd, ["a", "b"]), K)\n')
      .map((d: any) => d.message), []);
});

// A step naming a binding that does not exist. The declared-input walk reads
// the module bindings, so it has to tolerate a step whose own name failed to
// resolve — the undefined-name error is the one the reader needs, not a
// boundary cascade on top of it.
test('a chain step naming an undefined binding reports only that', () => {
  const errs = errorsOf(H + FLAT + BASE + K1 + 'y = kchain(nope, K1)\n');
  assert.ok(errs.length >= 1);
  assert.match(errs[0].message, /nope/);
  for (const d of errs) {
    assert.doesNotMatch(d.message, /step boundary/,
      'an unresolved step name must not also produce a boundary mismatch');
  }
});

// A step whose two boundaries type differently: `reals` is concrete, so it
// reaches the boundary check, while `anything` constrains nothing and its
// input keeps the `any` the reification published. The fill-in must key on the
// input NAME, so the second input is left alone rather than taking the first
// one's type off a position.
test('a step mixing a concrete and an unconstrained boundary fills in only the '
  + 'concrete one', () => {
    const errs = errorsOf(H + FLAT + BASE + K1
      + 'p = elementof(reals)\n'
      + 'q = elementof(anything)\n'
      + 'K2 = functionof(Normal(mu = p + q, sigma = 1.0), p = p, q = q)\n'
      + 'y = kchain(base2, K1, K2)\n');
    assert.equal(errs.length, 1, errs.map((d: any) => d.message).join(' | '));
    // Two inputs at a fed cat: §04's splat needs a record, and the cat of two
    // scalars is a vector. The declared type is not what refuses it, so the
    // message is the multi-input one and carries the same §06 route.
    assert.match(errs[0].message, /multi-input step boundary requires a record/);
    assert.match(errs[0].message, /markovchain\(kernel, init, n\)/);
  });

// A NON-array boundary mismatch appends no §06 route: the plain shape mismatch
// is the whole story, and naming `markovchain` there would send the reader to
// a construct that does not apply.
test('a record boundary mismatch carries no cat-feed route', () => {
  const errs = errorsOf(H
    + 'jr = joint(a = Normal(mu = 0.0, sigma = 1.0), '
    + 'b = Normal(mu = 0.0, sigma = 1.0))\n'
    + 'p = elementof(reals)\n'
    + 'q = elementof(reals)\n'
    + 'K = functionof(Normal(mu = p + q, sigma = 1.0), p = p, q = q)\n'
    + 'y = kchain(jr, K)\n');
  assert.equal(errs.length, 1, errs.map((d: any) => d.message).join(' | '));
  assert.match(errs[0].message, /auto-splat at step boundary/);
  assert.doesNotMatch(errs[0].message, /markovchain/);
  assert.doesNotMatch(errs[0].message, /positional `joint`/);
});

// =====================================================================
// The well-formed direction, and the weight question #238 left open
// =====================================================================

// The 3-step marginal chain whose final kernel consumes the `cat` §06 feeds
// it. This is the `marginalHistoryBody` path (refArrays base, s1, s0) — the
// first shape's own code path, reached by the spelling that types.
const V2 = 'v2 = elementof(cartpow(reals, 2))\n';
const CHAIN3 = BASE + K1 + V2
  + 'K2 = functionof(Normal(mu = sum(v2), sigma = 1.0), v2 = v2)\n'
  + 'y = kchain(base2, K1, K2)\n';
// The same, with an iid-bodied final kernel: two positions drawn at the SAME
// fed history, so the cross-position covariance witnesses the shared draw.
const CHAIN3_IID = BASE + K1 + V2
  + 'K2i = functionof(iid(Normal(mu = sum(v2), sigma = 1.0), 2), v2 = v2)\n'
  + 'y = kchain(base2, K1, K2i)\n';

// The positional-joint base, with a kernel whose declared input is the fed
// vector. This is the `measureToParamValue` whole-variate bind (refArrays jp)
// — the second shape's own code path.
const V1 = 'v1 = elementof(cartpow(reals, 1))\n';
const JBASE = BASE + V1
  + 'Kv = functionof(Normal(mu = sum(v1), sigma = 1.0), v1 = v1)\n'
  + 'jp = joint(base2)\ny = kchain(jp, Kv)\n';
const JBASE_IID = BASE + V1
  + 'Kvi = functionof(iid(Normal(mu = sum(v1), sigma = 1.0), 2), v1 = v1)\n'
  + 'jp = joint(base2)\ny = kchain(jp, Kvi)\n';
const JBASE2 = BASE + V2
  + 'Kv2 = functionof(Normal(mu = sum(v2), sigma = 1.0), v2 = v2)\n'
  + 'jp2 = joint(base2, Normal(mu = 0.0, sigma = 1.0))\ny = kchain(jp2, Kv2)\n';

const O_CHAIN3: Oracle = {
  mean: [2], variance: [10], thetaCov: [2], crossCov: 0,
};
const O_CHAIN3_IID: Oracle = {
  mean: [2, 2], variance: [10, 10], thetaCov: [2, 2], crossCov: 9,
};
const O_JBASE: Oracle = { mean: [1], variance: [3], thetaCov: [1], crossCov: 0 };
const O_JBASE_IID: Oracle = {
  mean: [1, 1], variance: [3, 3], thetaCov: [1, 1], crossCov: 2,
};
const O_JBASE2: Oracle = { mean: [1], variance: [4], thetaCov: [1], crossCov: 0 };

// The unweighted controls come first: they are the closed-form calibration of
// the two code paths, independent of the weight channel.
test('the 3-step marginal chain samples its closed form (unweighted)',
  async () => {
    await assertMoments('chain3 flat', FLAT, CHAIN3,
      { mean: [0], variance: [10], thetaCov: [2], crossCov: 0 }, false);
  });

test('the 3-step chain shares one history across an iid-bodied kernel '
  + '(unweighted)', async () => {
    await assertMoments('chain3 iid flat', FLAT, CHAIN3_IID,
      { mean: [0, 0], variance: [10, 10], thetaCov: [2, 2], crossCov: 9 },
      false);
  });

test('the positional-joint base samples its closed form (unweighted)',
  async () => {
    await assertMoments('jbase flat', FLAT, JBASE,
      { mean: [0], variance: [3], thetaCov: [1], crossCov: 0 }, false);
    await assertMoments('jbase2 flat', FLAT, JBASE2,
      { mean: [0], variance: [4], thetaCov: [1], crossCov: 0 }, false);
  });

test('the positional-joint base shares one variate across an iid-bodied '
  + 'kernel (unweighted)', async () => {
    await assertMoments('jbase iid flat', FLAT, JBASE_IID,
      { mean: [0, 0], variance: [3, 3], thetaCov: [1, 1], crossCov: 2 }, false);
  });

// The weight question. #238 fixed `matClm`'s fed-parent overlay but could not
// witness it on either of these shapes, because every atom was NaN. Both feed
// paths carry the stream, by reference, once.
test('the 3-step marginal chain carries a weighted parent\'s stream',
  async () => {
    await assertMoments('chain3 tilted', TILT, CHAIN3, O_CHAIN3, true);
  });

test('the 3-step chain carries it across an iid-bodied kernel', async () => {
  await assertMoments('chain3 iid tilted', TILT, CHAIN3_IID, O_CHAIN3_IID, true);
});

test('the positional-joint base carries a weighted parent\'s stream',
  async () => {
    await assertMoments('jbase tilted', TILT, JBASE, O_JBASE, true);
    await assertMoments('jbase2 tilted', TILT, JBASE2, O_JBASE2, true);
  });

test('the positional-joint base carries it across an iid-bodied kernel',
  async () => {
    await assertMoments('jbase iid tilted', TILT, JBASE_IID, O_JBASE_IID, true);
  });
