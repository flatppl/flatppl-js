'use strict';

// §06: locscale(D, shift, scale) is the pushforward under x -> scale * x + shift,
// and truncate(M, S) has total mass M(S). So normalize(truncate(locscale(D, m, s),
// I)) divides by D((I - m) / s), whose bounds swap for a negative scale.
//
// ORACLES are closed forms. StudentT(3) has density 2 / (pi sqrt(3)) (1 + t^2 / 3)^-2
// and F(0) = 1/2. Exponential(1) is asymmetric, so it detects a wrong orientation.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

async function value(src: string, N = 1): Promise<number[]> {
  const { ctx } = ctxFor(src, N);
  const m = await ctx.getMeasure('lp');
  return Array.from(m.value ? m.value.data : m.samples);
}

function near(got: number[], want: number, tol = 1e-12) {
  assert.ok(got.length > 0);
  got.forEach((g) => assert.ok(Math.abs(g - want) < tol, `got ${g}, want ${want}`));
}

const logT3 = (t: number) => Math.log(2 / (Math.PI * Math.sqrt(3))) - 2 * Math.log1p(t * t / 3);
const logN = (x: number, s: number) => -0.5 * Math.log(2 * Math.PI) - Math.log(s) - 0.5 * (x / s) ** 2;

test('half-t prior: normalize(truncate(locscale(StudentT(3), 0, 10), interval(0, inf)))', async () => {
  const want = logT3(0.15) - Math.log(10) + Math.log(2);
  near(await value('lp = logdensityof(normalize(truncate(locscale(StudentT(3), 0, 10), interval(0, inf))), 1.5)\n'), want);
  const drawn = 'sigma ~ normalize(truncate(locscale(StudentT(3), 0, 10), interval(0, inf)))\n'
    + 'lp = logdensityof(lawof(sigma), 1.5)\n';
  near(await value(drawn), want);
  near(await value(drawn, 3), want);
});

test('a positive scale maps the interval forward', async () => {
  // y = 1 + 2x on [2, 5] keeps x in [0.5, 2]; y = 3 is x = 1.
  const want = -1 - Math.log(2) - Math.log(Math.exp(-0.5) - Math.exp(-2));
  near(await value('lp = logdensityof(normalize(truncate(locscale(Exponential(1), 1, 2), interval(2, 5))), 3.0)\n'), want);
});

test('a negative scale swaps the preimage bounds', async () => {
  // y = 1 - 2x >= -3 keeps x <= 2; y = 0 is x = 0.5.
  const want = -0.5 - Math.log(2) - Math.log(1 - Math.exp(-2));
  near(await value('lp = logdensityof(normalize(truncate(locscale(Exponential(1), 1, -2), interval(-3, inf))), 0.0)\n'), want);
});

test('a run-time scale selects the orientation by its sign at every atom', async () => {
  const src = [
    's ~ Normal(0, 3)',
    'y ~ normalize(truncate(locscale(Exponential(1), 1, s), interval(-3, inf)))',
    'lp = logdensityof(lawof(record(s = s, y = y)), record(s = -2.0, y = 0.0))',
  ].join('\n') + '\n';
  const want = logN(-2, 3) - 0.5 - Math.log(2) - Math.log(1 - Math.exp(-2));
  near(await value(src), want);
  near(await value(src, 3), want);
});
