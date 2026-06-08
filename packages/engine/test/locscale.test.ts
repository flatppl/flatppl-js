'use strict';

// Tests for `locscale(m, shift, scale)` — affine (location-scale)
// pushforward sugar (spec §06 sec:locscale):
//
//   locscale(m, shift, scale)  ≡  pushfwd(fn(scale * _ + shift), m)
//
// It is desugared at lift time (lift.inlineLocscaleLift) into a
// first-class affine pushforward, so there is NO new derivation /
// density / materialiser handler. These tests pin the two equivalences
// the spec calls out:
//   1. Sampling: `locscale(Normal(0,1), mu, sigma)` agrees with
//      `Normal(mu, sigma)` (histogram / moment agreement).
//   2. Density: logdensityof through the desugared pushfwd matches the
//      closed-form Normal(mu, sigma) logpdf.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { makeCtxFactory } = require('./_measure-helpers.ts');

const SAMPLE_COUNT = 20000;
const ROOT_SEED    = 0x10C5CA1E;

const makeCtx = makeCtxFactory({ sampleCount: SAMPLE_COUNT, rootSeed: ROOT_SEED });

function meanStd(arr: any) {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  const m = s / arr.length;
  let v = 0;
  for (let i = 0; i < arr.length; i++) v += (arr[i] - m) * (arr[i] - m);
  return { mean: m, std: Math.sqrt(v / arr.length) };
}

const MU = 3.0;
const SIGMA = 2.0;

test('locscale: lowers to a pushfwd (no locscale node survives lift)', () => {
  const { built } = makeCtx(`
    base = Normal(mu = 0, sigma = 1)
    y = locscale(base, ${MU}, ${SIGMA})
  `);
  const yb = built.bindings.get('y');
  assert.ok(yb, 'binding y exists');
  // After lift the binding for y must be a pushfwd, never a locscale.
  const ir = yb.ir || (yb.node && yb.node.value);
  const seen = JSON.stringify(ir);
  assert.ok(seen.indexOf('locscale') === -1,
    'locscale must be desugared away at lift time');
  // A derivation must classify (pushfwd), proving downstream wiring.
  const der = built.derivations['y'];
  assert.ok(der, 'y has a derivation');
  assert.equal(der.kind, 'pushfwd', 'y derives as a pushfwd');
});

test('locscale(Normal(0,1), mu, sigma) samples ≡ Normal(mu, sigma)', async () => {
  const { ctx } = makeCtx(`
    base = Normal(mu = 0, sigma = 1)
    y = locscale(base, ${MU}, ${SIGMA})
  `);
  const m = await ctx.getMeasure('y');
  const samples = m.samples;
  assert.ok(samples && samples.length > 1000, 'got samples');
  const { mean, std } = meanStd(samples);
  // 20k samples → SE(mean) ≈ sigma/sqrt(N) ≈ 0.014; allow generous slack.
  assert.ok(Math.abs(mean - MU) < 0.1,
    `sample mean ${mean} should be near ${MU}`);
  assert.ok(Math.abs(std - SIGMA) < 0.1,
    `sample std ${std} should be near ${SIGMA}`);
});

test('locscale density matches Normal(mu, sigma) logpdf at several points', async () => {
  // Density of y = locscale(Normal(0,1), mu, sigma) at a point must
  // equal the closed-form Normal(mu, sigma) logpdf. We score via the
  // surface `logdensityof(y, point)` idiom (mirrors
  // test/bijection-density.test.ts) — exercising the bijection-metadata
  // branch of density.walkPushfwd that the locscale desugaring targets.

  // Closed-form Normal(mu,sigma) logpdf.
  const normLogpdf = (x: number) =>
    -0.5 * Math.log(2 * Math.PI) - Math.log(SIGMA)
    - 0.5 * ((x - MU) / SIGMA) ** 2;

  for (const yVal of [MU - 2, MU - 1, MU, MU + 1, MU + 2]) {
    const { ctx } = makeCtx(`
      base = Normal(mu = 0, sigma = 1)
      y = locscale(base, ${MU}, ${SIGMA})
      lp = logdensityof(y, ${yVal})
    `);
    const lp = await ctx.getMeasure('lp');
    const got = lp.samples[0];
    const want = normLogpdf(yVal);
    assert.ok(Math.abs(got - want) < 1e-9,
      `logpdf at ${yVal}: got ${got}, want ${want}`);
  }
});
