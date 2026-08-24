'use strict';

// =====================================================================
// ksuperpose-density.test.ts — spec §06 `ksuperpose` mixture density
// =====================================================================
//
// §06 density rule:
//   logdensityof(ksuperpose(κ, w)(θ), x)
//     = logsumexp_i(log wᵢ + logdensityof(κ(θᵢ), x))
// "so a zero weight contributes −∞ and drops out. All components come
// from one kernel and so share one reference measure — the mixture's."
//
// §06 mass: "the result has total mass Σᵢ wᵢ·totalmass(κ(θᵢ)) — Σᵢ wᵢ for
// a Markov `kernel`", so `normalize` divides by Σᵢ wᵢ.
//
// ORACLE — Distributions.jl (independent of this engine and of the spec
// lowering), computed before any engine output was read:
//
//   using Distributions
//   w = [0.3, 1.2]; mus = [-1.0, 2.0]; sig = [1.0, 0.5]
//   lse(v) = (m = maximum(v); m + log(sum(exp.(v .- m))))
//   f(x)  = lse([log(w[i]) + logpdf(Normal(mus[i], sig[i]), x) for i in 1:2])
//   f(x) - log(sum(w))      # normalized
//
// | case                              | x    | oracle           |
// |-----------------------------------|------|------------------|
// | two-Normal, normalized            |  0.5 |  -3.411415107516 |
// | two-Normal, normalized            | -1.0 |  -2.528376323799 |
// | two-Normal, normalized            |  2.0 |  -0.447547242639 |
// | two-Normal, normalized            |  5.0 | -18.331151868303 |
// | two-Normal, UNnormalized          |  0.5 |  -3.005949999408 |
// | shared scalar sigma = 1.0, norm.  |  0.5 |  -2.043938533205 |
// | singular mu = [1.0], normalized   |  0.5 |  -0.781833651554 |
// | one zero weight w = [0.0, 1.2]    |  0.5 |  -4.543469795851 |
// |   (UNnormalized; equals the       |      |                  |
// |    closed form log(1.2) +         |      |                  |
// |    logpdf(N(2, 0.5), 0.5) exactly)|      |                  |
//
// log Z = log 1.5 = 0.405465108108. Two independent cross-checks on the
// arithmetic itself rather than on the engine: the two x = 0.5 rows differ
// by exactly log 1.5, and the zero-weight row equals the single surviving
// component's closed form with no logsumexp involved.
//
// The Dirac rows are closed-form by hand: with p = [0.2, 0.8] summing to
// one, normalize is a no-op and the density at the label is log pₖ, i.e.
// log 0.8 = -0.223143551314 and log 0.2 = -1.609437912434.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const TOL = 1e-9;

// Score a single `logdensityof` binding named `__score__`.
async function score(src: string): Promise<number> {
  const { ctx } = ctxFor(src, 1);
  const m = await ctx.getMeasure('__score__');
  const s: Float64Array | null = m.samples ?? (m.value && m.value.data) ?? null;
  if (!s || s.length === 0) throw new Error('score: __score__ produced no data');
  return s[0];
}

function diagnosticsOf(src: string): any[] {
  return require('..').processSource(src).diagnostics
    .filter((d: any) => d.severity === 'error');
}

const PARAMS = `
weights = [0.3, 1.2]
means = [-1.0, 2.0]
sigmas = [1.0, 0.5]
`;

// =====================================================================
// §06 density rule
// =====================================================================

test('§06: the normalized two-Normal mixture matches Distributions.jl at '
  + 'four points', async () => {
  const oracle: [number, number][] = [
    [0.5, -3.411415107516122],
    [-1.0, -2.528376323798943],
    [2.0, -0.4475472426391285],
    [5.0, -18.331151868302555],
  ];
  for (const [x, want] of oracle) {
    const got = await score(PARAMS
      + 'mix = normalize(ksuperpose(Normal, weights)(mu = means, sigma = sigmas))\n'
      + `__score__ = logdensityof(mix, ${x})\n`);
    assert.ok(Math.abs(got - want) <= TOL,
      `x = ${x}: got ${got}, oracle ${want}`);
  }
});

test('§06: the UNnormalized mixture is the normalized one plus log Z, with '
  + 'Z = Σᵢ wᵢ for a Markov component', async () => {
  const unnorm = await score(PARAMS
    + 'mix = ksuperpose(Normal, weights)(mu = means, sigma = sigmas)\n'
    + '__score__ = logdensityof(mix, 0.5)\n');
  assert.ok(Math.abs(unnorm - (-3.005949999407958)) <= TOL,
    `unnormalized: got ${unnorm}, oracle -3.005949999407958`);
  // The Markov-component mass specialization, checked as an identity on the
  // two numbers rather than as a second oracle lookup.
  const norm = -3.411415107516122;
  assert.ok(Math.abs((unnorm - norm) - Math.log(1.5)) <= TOL,
    `unnormalized − normalized = ${unnorm - norm}, log Z = ${Math.log(1.5)}`);
});

test('§06: a non-collection family argument is held constant across the '
  + 'components (shared scalar sigma)', async () => {
  const got = await score(PARAMS
    + 'mix = normalize(ksuperpose(Normal, weights)(mu = means, sigma = 1.0))\n'
    + '__score__ = logdensityof(mix, 0.5)\n');
  assert.ok(Math.abs(got - (-2.0439385332046727)) <= TOL,
    `got ${got}, oracle -2.0439385332046727`);
});

test('§06: a singular (size-one) family collection is expanded by repetition',
  async () => {
    const got = await score(PARAMS
      + 'mu1 = [1.0]\n'
      + 'mix = normalize(ksuperpose(Normal, weights)(mu = mu1, sigma = sigmas))\n'
      + '__score__ = logdensityof(mix, 0.5)\n');
    assert.ok(Math.abs(got - (-0.7818336515543383)) <= TOL,
      `got ${got}, oracle -0.7818336515543383`);
  });

test('§06: a zero weight drops out EXACTLY — the mixture density equals the '
  + 'surviving component\'s closed form, with no logsumexp residue', async () => {
  const got = await score(
    'w0 = [0.0, 1.2]\n'
    + 'means = [-1.0, 2.0]\n'
    + 'sigmas = [1.0, 0.5]\n'
    + 'mix = ksuperpose(Normal, w0)(mu = means, sigma = sigmas)\n'
    + '__score__ = logdensityof(mix, 0.5)\n');
  // log(1.2) + logpdf(Normal(2.0, 0.5), 0.5), by hand — no mixture involved.
  const closed = Math.log(1.2)
    + (-Math.log(0.5) - 0.5 * Math.log(2 * Math.PI) - 0.5 * ((0.5 - 2.0) / 0.5) ** 2);
  assert.ok(Math.abs(got - closed) <= TOL, `got ${got}, closed form ${closed}`);
  assert.ok(Math.abs(got - (-4.543469795850773)) <= TOL,
    `got ${got}, oracle -4.543469795850773`);
});

test('§06: all-zero weights are the zero measure — log-density −∞ everywhere',
  async () => {
    for (const x of [0.5, -1.0, 2.0]) {
      const got = await score(
        'wz = [0.0, 0.0]\n'
        + 'means = [-1.0, 2.0]\n'
        + 'sigmas = [1.0, 0.5]\n'
        + 'mix = ksuperpose(Normal, wz)(mu = means, sigma = sigmas)\n'
        + `__score__ = logdensityof(mix, ${x})\n`);
      assert.equal(got, -Infinity, `x = ${x}: got ${got}, want -Infinity`);
    }
  });

test('§08: a categorical over arbitrary labels is a Dirac superposition, '
  + 'density log pₖ at the label', async () => {
  const SRC = 'p = [0.2, 0.8]\nlabels = [0.0, 1.5]\n'
    + 'cat = normalize(ksuperpose(Dirac, p)(value = labels))\n';
  const hi = await score(SRC + '__score__ = logdensityof(cat, 1.5)\n');
  assert.ok(Math.abs(hi - Math.log(0.8)) <= TOL,
    `at label 1.5: got ${hi}, closed form ${Math.log(0.8)}`);
  const lo = await score(SRC + '__score__ = logdensityof(cat, 0.0)\n');
  assert.ok(Math.abs(lo - Math.log(0.2)) <= TOL,
    `at label 0.0: got ${lo}, closed form ${Math.log(0.2)}`);
});

// =====================================================================
// The lift is a kernel, and the two spellings agree
// =====================================================================

test('§06: the mixture agrees with the variadic superpose spelling §06 gives '
  + 'for the same measure', async () => {
  const lifted = await score(PARAMS
    + 'mix = normalize(ksuperpose(Normal, weights)(mu = means, sigma = sigmas))\n'
    + '__score__ = logdensityof(mix, 0.5)\n');
  const variadic = await score(PARAMS
    + 'mix = normalize(superpose(weighted(weights[1], Normal(means[1], sigmas[1])), '
    + 'weighted(weights[2], Normal(means[2], sigmas[2]))))\n'
    + '__score__ = logdensityof(mix, 0.5)\n');
  assert.ok(Math.abs(lifted - variadic) <= TOL,
    `lift ${lifted} == variadic ${variadic}`);
});

test('§04 aliasing: the two-step spelling (name the lift, then apply it) is '
  + 'the same program as the inline one', async () => {
  const inline = await score(PARAMS
    + 'mix = normalize(ksuperpose(Normal, weights)(mu = means, sigma = sigmas))\n'
    + '__score__ = logdensityof(mix, 0.5)\n');
  const named = await score(PARAMS
    + 'lift = ksuperpose(Normal, weights)\n'
    + 'mix = normalize(lift(mu = means, sigma = sigmas))\n'
    + '__score__ = logdensityof(mix, 0.5)\n');
  assert.ok(Math.abs(named - inline) <= TOL,
    `named lift ${named} == inline ${inline}`);
});

test('§06: the family may be passed positionally (§05 lets a distribution '
  + 'take its parameters positionally)', async () => {
  const kw = await score(PARAMS
    + 'mix = normalize(ksuperpose(Normal, weights)(mu = means, sigma = sigmas))\n'
    + '__score__ = logdensityof(mix, 0.5)\n');
  const pos = await score(PARAMS
    + 'mix = normalize(ksuperpose(Normal, weights)(means, sigmas))\n'
    + '__score__ = logdensityof(mix, 0.5)\n');
  assert.ok(Math.abs(pos - kw) <= TOL, `positional ${pos} == keyword ${kw}`);
});

test('§06: the component may be a REIFIED kernel, not only a bare constructor '
  + '(§04 makes a reification of a measure a Markov kernel)', async () => {
  // K binds sigma = 1.0 inside the reification, so this is the shared-scalar-
  // sigma measure by another route and lands on the same oracle row — an
  // independent check that the reification path reads the same family.
  const named = await score(PARAMS
    + 'K = m -> Normal(mu = m, sigma = 1.0)\n'
    + 'mix = normalize(ksuperpose(K, weights)(means))\n'
    + '__score__ = logdensityof(mix, 0.5)\n');
  assert.ok(Math.abs(named - (-2.0439385332046727)) <= TOL,
    `named reification: got ${named}, oracle -2.0439385332046727`);
  const inline = await score(PARAMS
    + 'mix = normalize(ksuperpose(m -> Normal(mu = m, sigma = 1.0), weights)(means))\n'
    + '__score__ = logdensityof(mix, 0.5)\n');
  assert.ok(Math.abs(inline - (-2.0439385332046727)) <= TOL,
    `inline reification: got ${inline}, oracle -2.0439385332046727`);
});

test('§06: the weights and the family may be written INLINE as array literals '
  + '— their length is read off the AST', async () => {
  const got = await score(
    'mix = normalize(ksuperpose(Normal, [0.3, 1.2])(mu = [-1.0, 2.0], '
    + 'sigma = [1.0, 0.5]))\n'
    + '__score__ = logdensityof(mix, 0.5)\n');
  assert.ok(Math.abs(got - (-3.411415107516122)) <= TOL,
    `got ${got}, oracle -3.411415107516122`);
});

test('N may be pinned by a size-N FAMILY argument when the weight vector\'s '
  + 'own length does not resolve (§06 gives each collection "size N or one")',
async () => {
  // `l1unit(raw)` is an inline expression, so it has no name to read a type
  // off; `means` pins N = 2, and l1unit normalizes [0.3, 1.2] to
  // [0.2, 0.8], giving Z = 1 and the mixture density directly.
  const got = await score(
    'raw = [0.3, 1.2]\nmeans = [-1.0, 2.0]\nsigmas = [1.0, 0.5]\n'
    + 'mix = ksuperpose(Normal, l1unit(raw))(mu = means, sigma = sigmas)\n'
    + '__score__ = logdensityof(mix, 0.5)\n');
  // Same as the normalized two-Normal row: l1unit(w) = w / Σw, so the
  // density is logsumexp(log(wᵢ/Z) + logpdf) = the normalized oracle.
  assert.ok(Math.abs(got - (-3.411415107516122)) <= TOL,
    `got ${got}, oracle -3.411415107516122`);
});

test('§06: a nested array LITERAL family is a static error, caught on the AST '
  + 'without a type', () => {
  const ds = diagnosticsOf(PARAMS
    + 'mix = ksuperpose(Normal, weights)(mu = [[1.0, 2.0], [3.0, 4.0]], '
    + 'sigma = sigmas)\n');
  assert.ok(ds.some((d: any) => /single axis/.test(d.message)),
    `want a one-axis error, got ${JSON.stringify(ds.map((d: any) => d.message))}`);
});

test('a family argument that is an unnamed expression is refused, naming the '
  + 'remedy — its collection-or-scalar status cannot be read', () => {
  const ds = diagnosticsOf(PARAMS
    + 'mix = ksuperpose(Normal, weights)(mu = means + 1.0, sigma = sigmas)\n');
  assert.ok(ds.some((d: any) => /cannot tell whether family argument `mu`/.test(d.message)
      && /Bind it to a name/.test(d.message)),
    `want a classify refusal, got ${JSON.stringify(ds.map((d: any) => d.message))}`);
});

test('the weights may be LATENT — the rewrite indexes the weight EXPRESSION, '
  + 'so a mixture whose weights are inferred still normalizes per-θ',
async () => {
  // theta and 1-theta sum to one, so Z = 1 and normalize is a no-op; the
  // score is then the plain two-component mixture density at theta = 0.5,
  // which is logsumexp(log .5 + logpdf(N(-1,1), .5), log .5 + logpdf(N(2,.5), .5)).
  const src = 'means = [-1.0, 2.0]\nsigmas = [1.0, 0.5]\n'
    + 'theta = 0.5\n'
    + 'w = [theta, 1.0 - theta]\n'
    + 'mix = normalize(ksuperpose(Normal, w)(mu = means, sigma = sigmas))\n'
    + '__score__ = logdensityof(mix, 0.5)\n';
  const got = await score(src);
  const lp1 = Math.log(0.5) + (-0.5 * Math.log(2 * Math.PI) - 0.5 * (1.5) ** 2);
  const lp2 = Math.log(0.5)
    + (-Math.log(0.5) - 0.5 * Math.log(2 * Math.PI) - 0.5 * ((0.5 - 2.0) / 0.5) ** 2);
  const m = Math.max(lp1, lp2);
  const want = m + Math.log(Math.exp(lp1 - m) + Math.exp(lp2 - m));
  assert.ok(Math.abs(got - want) <= TOL, `got ${got}, closed form ${want}`);
});

// =====================================================================
// Static errors and refusals
// =====================================================================

test('§06: more than one family axis is a static error', () => {
  const ds = diagnosticsOf(PARAMS
    + 'grid = [[1.0, 2.0], [3.0, 4.0]]\n'
    + 'mix = ksuperpose(Normal, weights)(mu = grid, sigma = sigmas)\n');
  assert.ok(ds.some((d: any) => /single axis/.test(d.message)),
    `want a one-axis error, got ${JSON.stringify(ds.map((d: any) => d.message))}`);
  assert.ok(ds.some((d: any) => d.loc), 'the error is located');
});

test('§06: a family collection whose size is neither N nor one is a static '
  + 'error', () => {
  const ds = diagnosticsOf(PARAMS
    + 'three = [1.0, 2.0, 3.0]\n'
    + 'mix = ksuperpose(Normal, weights)(mu = three, sigma = sigmas)\n');
  assert.ok(ds.some((d: any) => /size 3/.test(d.message) && /N = 2/.test(d.message)),
    `want a size error, got ${JSON.stringify(ds.map((d: any) => d.message))}`);
});

test('§06: `weights` must be a one-axis vector — a scalar has no components',
  () => {
    const ds = diagnosticsOf(
      'means = [-1.0, 2.0]\nsigmas = [1.0, 0.5]\n'
      + 'mix = ksuperpose(Normal, 0.5)(mu = means, sigma = sigmas)\n');
    assert.ok(ds.some((d: any) => /one-axis vector/.test(d.message)),
      `want a weights error, got ${JSON.stringify(ds.map((d: any) => d.message))}`);
  });

test('ksuperpose takes exactly two positional arguments', () => {
  const ds = diagnosticsOf(PARAMS
    + 'mix = ksuperpose(Normal)(mu = means, sigma = sigmas)\n');
  assert.ok(ds.some((d: any) => /ksuperpose expects 2/.test(d.message)),
    `want an arity error, got ${JSON.stringify(ds.map((d: any) => d.message))}`);
});

test('a TABLE parameter family is refused with the keyword-vector remedy, not '
  + 'silently mis-scored', () => {
  const ds = diagnosticsOf(
    'weights = [0.3, 1.2]\n'
    + 'fam = table(mu = [-1.0, 2.0], sigma = [1.0, 0.5])\n'
    + 'mix = ksuperpose(Normal, weights)(fam)\n');
  assert.ok(ds.some((d: any) => /TABLE parameter family/.test(d.message)
      && /keyword vectors/.test(d.message)),
    `want a table refusal, got ${JSON.stringify(ds.map((d: any) => d.message))}`);
});

test('a component count that is not statically known is refused, naming the '
  + 'remedy — §06 allows a runtime N that this engine does not expand', () => {
  const ds = diagnosticsOf(
    'w = external(cartpow(nonnegreals, 2))\n'
    + 'mix = ksuperpose(Normal, w)(mu = 0.0, sigma = 1.0)\n');
  assert.ok(ds.some((d: any) => /not statically\s+known/.test(d.message)),
    `want an N refusal, got ${JSON.stringify(ds.map((d: any) => d.message))}`);
});
