'use strict';

// Moment conformance for the InverseGamma sampler, against closed-form
// oracles, plus the #591 seed-option regression.
//
// Oracle for InverseGamma(shape=a, scale=b), a > 2:
//   mean = b / (a-1)
//   var  = b^2 / ((a-1)^2 (a-2))
//
// #591: `randInverseGamma.factory`'s trailing-options detection recognised
// only a `prng` key, so a `{ seed }`-only options object fell back to `{}`
// and the inner `randGamma.factory` ran off `Math.random` — the caller's
// seed had no effect on the STATIC form, and the PARAMETRIC form threw
// (a `{ seed }` object mis-parsed as the numeric `shape` argument, since
// `args[0] === opts` failed with `opts` a freshly-built `{}`). This is the
// same idiom `randBetaFixed` carried (fixed for Beta in #547/#591's parent
// commit) — here it additionally discarded the seed after recognising it,
// since the resolved `prng` variable never read `seed` at all.
//
// The engine's own callers (`sampler.ts` rand/makeSampler/etc.) always pass
// `{ prng }` — a Philox bridge, never `{ seed }` — so this bug never
// affected an engine-internal draw stream; it affects only code that calls
// `randFn.factory` directly with `{ seed }`, as the tests below do.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler = require('../sampler.ts');
const rng = require('../rng.ts');

const N = 200_000;
const TOL_SIGMA = 5;

function synthLoc() {
  return { start: { line: -1, col: -1 }, end: { line: -1, col: -1 }, synthetic: true };
}

function invGammaIR(shape: number, scale: number) {
  return {
    kind: 'call',
    op:   'InverseGamma',
    kwargs: {
      shape: { kind: 'lit', value: shape, loc: synthLoc() },
      scale: { kind: 'lit', value: scale, loc: synthLoc() },
    },
    loc: synthLoc(),
  };
}

// mean requires a > 1, variance a > 2, and the fourth central moment (needed
// for the sample variance's own standard error) requires a > 4 — so every
// test case below picks shape a > 4, comfortably inside all three.
//
// Excess kurtosis of InverseGamma(a, b): (30a - 66) / ((a-3)(a-4)), a > 4.
// mu4 = (kurtosis + 3) * var^2; SE(sample variance) = sqrt((mu4 - var^2)/n).
function oracle(a: number, b: number, n: number) {
  const mean = b / (a - 1);
  const variance = (b * b) / ((a - 1) * (a - 1) * (a - 2));
  const kurtosisExcess = (30 * a - 66) / ((a - 3) * (a - 4));
  const mu4 = (kurtosisExcess + 3) * variance * variance;
  return {
    mean,
    variance,
    seMean: Math.sqrt(variance / n),
    seVar:  Math.sqrt((mu4 - variance * variance) / n),
  };
}

function draws(shape: number, scale: number, seed: number) {
  const s = sampler.makeSampler(rng.seedFromBytes([seed, 11, 0]), invGammaIR(shape, scale), {});
  const xs = new Float64Array(N);
  for (let i = 0; i < N; i++) xs[i] = s.draw();
  return xs;
}

function sampleMoments(xs: Float64Array) {
  let sum = 0;
  for (let i = 0; i < xs.length; i++) sum += xs[i];
  const mean = sum / xs.length;
  let c2 = 0;
  for (let i = 0; i < xs.length; i++) { const d = xs[i] - mean; c2 += d * d; }
  return { mean, variance: c2 / xs.length };
}

function assertMoments(xs: Float64Array, a: number, b: number, label: string) {
  const o = oracle(a, b, xs.length);
  const m = sampleMoments(xs);
  const zMean = (m.mean - o.mean) / o.seMean;
  const zVar = (m.variance - o.variance) / o.seVar;
  assert.ok(Math.abs(zMean) < TOL_SIGMA,
    `InverseGamma(${a},${b}) ${label}: mean ${m.mean} vs oracle ${o.mean} — ${zMean.toFixed(1)}σ`);
  assert.ok(Math.abs(zVar) < TOL_SIGMA,
    `InverseGamma(${a},${b}) ${label}: var ${m.variance} vs oracle ${o.variance} — ${zVar.toFixed(1)}σ`);
}

// a > 4 throughout, so mean, variance, and the sample-variance SE all
// exist in closed form.
for (const [shape, scale] of [[5, 4], [6, 2], [8, 5], [10, 3]] as [number, number][]) {
  test(`InverseGamma(${shape}, ${scale}): mean and variance within ${TOL_SIGMA}σ of closed form`, () => {
    for (const seed of [1, 2, 3]) {
      assertMoments(draws(shape, scale, seed), shape, scale, `seed ${seed}`);
    }
  });
}

test('InverseGamma draws are strictly positive', () => {
  for (const [a, b] of [[5, 4], [3, 2]] as [number, number][]) {
    const xs = draws(a, b, 11);
    for (let i = 0; i < xs.length; i++) assert.ok(xs[i] > 0, `InverseGamma(${a},${b}) drew ${xs[i]}`);
  }
}
);

// =====================================================================
// #591: randFn.factory {seed} regression
// =====================================================================

const INVGAMMA_RAND_FN = sampler._internal.REGISTRY.InverseGamma.randFn;

// Pre-fix, measured directly against origin/main's randInverseGamma (see
// wave-591-report for the reproduction script): a {seed}-only STATIC call
// discarded the seed entirely — `factory(3, 2, {seed: 12345})` called twice
// gave 0.5149260453559843 then 1.557928874359768, since the resolved `prng`
// fell through to `Math.random` and the two draws consumed the SAME global
// stream sequentially rather than each starting from the given seed. A
// {seed}-only PARAMETRIC call didn't even discard silently — it threw
// (`invalid argument... Value: NaN`), because the undetected {seed} object
// mis-routed into the static branch as the numeric `shape` argument.
test('#591: the STATIC form honours a {seed}-only options object — repeat '
  + 'factories with one seed agree', () => {
  const a = INVGAMMA_RAND_FN.factory(3, 2, { seed: 12345 });
  const b = INVGAMMA_RAND_FN.factory(3, 2, { seed: 12345 });
  const da = a(), db = b();
  assert.equal(da, db,
    `static form: ${da} != ${db} — the {seed} opts object was dropped`);
  assert.ok(da > 0, `draw ${da} should be positive`);

  // A different seed must give a different draw.
  const c = INVGAMMA_RAND_FN.factory(3, 2, { seed: 54321 })();
  assert.notEqual(c, da, 'a different seed gives a different draw');
});

// _sharedPrng's third arm — neither `prng` nor `seed` given at all — is
// dead from every engine-internal caller (sampler.ts always supplies
// `{ prng }`), but the STATIC form's bare call (no trailing options object
// at all) is still a supported call shape — it was before this fix too,
// via the old inline `opts.prng || Math.random` — and must still draw off
// Math.random rather than throw. (The PARAMETRIC form has no analogous
// case: its one-argument detection requires that argument to already
// carry `prng` or `seed` to be recognised as `opts` at all.)
test('InverseGamma.randFn: a bare STATIC factory call with no options object at all still draws', () => {
  const draw = INVGAMMA_RAND_FN.factory(3, 2)();
  assert.ok(draw > 0, `draw ${draw} should be positive`);
});

test('#591: the PARAMETRIC form recognises and honours a {seed}-only options object', () => {
  const pa = INVGAMMA_RAND_FN.factory({ seed: 555 });
  const pb = INVGAMMA_RAND_FN.factory({ seed: 555 });
  assert.equal(typeof pa, 'function');
  const da = pa(3, 2), db = pb(3, 2);
  assert.equal(da, db,
    `parametric form: ${da} != ${db} — the {seed} opts object was dropped`);
  assert.ok(da > 0, `draw ${da} should be positive`);

  const dc = INVGAMMA_RAND_FN.factory({ seed: 777 })(3, 2);
  assert.notEqual(dc, da, 'a different seed gives a different draw');

  // Static and parametric forms agree on the same seed and params — the
  // two forms now read the option the same way (parity with the Beta fix).
  const stat = INVGAMMA_RAND_FN.factory(3, 2, { seed: 4242 })();
  const para = INVGAMMA_RAND_FN.factory({ seed: 4242 })(3, 2);
  assert.equal(stat, para,
    `static ${stat} != parametric ${para} on the same seed`);
});

// A/B proof that the fix leaves the engine-internal draw stream unmoved:
// makeSampler always builds `factoryOpts` from a Philox `{ prng }` bridge
// (sampler.ts), never from `{ seed }`, and the `prng` key was already read
// correctly before this fix — only the `{ seed }`-only path was broken.
test('A/B: InverseGamma(3, 2) draws are bit-identical to the pre-fix implementation', () => {
  // Captured from origin/main (pre-#591) via makeSampler(seed=917) — the
  // `{ prng }` path this exercises was never affected by the bug.
  const PRE_FIX_FIRST5 = [
    1.238815976751559, 1.9727514031913418, 0.3677182567404411,
    1.3708268261608938, 0.8341935895280711,
  ];
  const first5 = Array.from(draws(3, 2, 917)).slice(0, 5);
  assert.deepEqual(first5, PRE_FIX_FIRST5);
});
