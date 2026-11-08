// `superpose` of record-valued measures (`superpose(prior, prior)` over a
// `lawof(record(…))`). The superpose materialiser read every component's flat
// `samples`, which a record measure does not have, and threw an internal
// "Cannot read properties of undefined (reading 'length')". Spec §06:
// superpose(M1, M2)(A) = M1(A) + M2(A).
// Oracles: the total mass (sum of component masses), the density (sum of the
// component densities), and the mixture moments of the denoted measure.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const lnN = (x: number, m: number, s: number) => -0.5 * Math.log(2 * Math.PI * s * s) - (x - m) ** 2 / (2 * s * s);
const N = 40000;

function weightedMoments(samples: Float64Array, lw: Float64Array, stride = 1, offset = 0) {
  let W = 0, S = 0, Q = 0;
  for (let i = 0; i < lw.length; i++) {
    const w = Math.exp(lw[i]), x = samples[i * stride + offset];
    W += w; S += w * x; Q += w * x * x;
  }
  return { mean: S / W, variance: Q / W - (S / W) ** 2 };
}

test('superpose(prior, prior) of a record prior has mass 2, twice its density, and its law', async () => {
  const src = 'mu ~ Normal(1.0, 2.0)\nv ~ iid(Normal(0.0, 1.0), 2)\n'
    + 'prior = lawof(record(mu = mu, v = v))\nM = superpose(prior, prior)\n'
    + 'lp = logdensityof(M, record(mu = 0.3, v = [0.1, -0.2]))\n';
  const { ctx } = makeMatCtx(src, { sampleCount: N });
  const m = await ctx.getMeasure('M');
  assert.ok(Math.abs(m.logTotalmass - Math.log(2)) < 1e-12, `log mass ${m.logTotalmass}`);
  const mu = weightedMoments(m.fields.mu.samples, m.logWeights);
  // SE at N = 40000: mean 0.01, variance 0.03.
  assert.ok(Math.abs(mu.mean - 1) < 0.05 && Math.abs(mu.variance - 4) < 0.15, JSON.stringify(mu));
  assert.deepEqual(m.fields.v.value.shape, [N, 2]);
  const v1 = weightedMoments(m.fields.v.value.data, m.logWeights, 2, 1);
  assert.ok(Math.abs(v1.mean) < 0.03 && Math.abs(v1.variance - 1) < 0.05, JSON.stringify(v1));
  const lp = await ctx.getMeasure('lp');
  const got = lp.value ? lp.value.data[0] : lp.samples[0];
  const want = Math.log(2) + lnN(0.3, 1, 2) + lnN(0.1, 0, 1) + lnN(-0.2, 0, 1);
  assert.ok(Math.abs(got - want) < 1e-12, `${got} vs ${want}`);
});

test('a weighted record superposition selects components by their mass', async () => {
  // 3·N(0, 1) + N(4, 1) over a record field: mass 4, mixture mean 1,
  // variance 1 + (3·0² + 1·4²)/4 − 1² = 4.
  const src = 'a ~ Normal(0.0, 1.0)\nb ~ Normal(4.0, 1.0)\n'
    + 'A = lawof(record(x = a))\nB = lawof(record(x = b))\nM = superpose(weighted(3.0, A), B)\n';
  const { ctx } = makeMatCtx(src, { sampleCount: N });
  const m = await ctx.getMeasure('M');
  assert.ok(Math.abs(m.logTotalmass - Math.log(4)) < 1e-10, `log mass ${m.logTotalmass}`);
  const x = weightedMoments(m.fields.x.samples, m.logWeights);
  // SE at N = 40000: mean 0.01, variance 0.04.
  assert.ok(Math.abs(x.mean - 1) < 0.05 && Math.abs(x.variance - 4) < 0.2, JSON.stringify(x));
});
