'use strict';

// truncate(<discrete base>, S): §06 gives ν(A) = M(A ∩ S), and §03 makes
// `interval(lo, hi)` closed, so an atom on either end of S belongs to ν. The
// worker's CDF route used F(hi) − F(lo), which is P(lo < X ≤ hi) and drops the
// atom at an integer lo: the mass, the samples and every normalizer built on
// that mass were wrong. Oracles below sum the Poisson pmf over the atoms in S.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const { buildLogPi } = require('../mcmc-density.ts');

function logPmf(k: number, rate: number): number {
  let logFactorial = 0;
  for (let j = 2; j <= k; j++) logFactorial += Math.log(j);
  return k * Math.log(rate) - rate - logFactorial;
}

// Mass of Poisson(rate) on the integers in [lo, hi].
function mass(rate: number, lo: number, hi: number): number {
  let s = 0;
  for (let k = Math.max(0, Math.ceil(lo)); k <= hi; k++) s += Math.exp(logPmf(k, rate));
  return s;
}

async function value(src: string, name: string, sampleCount = 1): Promise<Float64Array> {
  const { ctx } = makeMatCtx(src, { sampleCount });
  const m = await ctx.getMeasure(name);
  return m.samples ?? m.value.data;
}

const close = (actual: number, expected: number, label: string) =>
  assert.ok(Math.abs(actual - expected) < 1e-12 * Math.max(1, Math.abs(expected)),
    `${label}: ${actual} vs ${expected}`);

test('totalmass of a truncated Poisson keeps the atoms on both closed ends', async () => {
  for (const [lo, hi] of [[0, 5], [2, 4], [0, 0], [3, 3], [0.5, 4.5], [-1, 5]]) {
    const z = await value(`z = totalmass(truncate(Poisson(3.0), interval(${lo}, ${hi})))`, 'z');
    close(z[0], mass(3, lo, hi), `interval(${lo}, ${hi})`);
  }
  close((await value('z = totalmass(truncate(Poisson(3.0), nonnegreals))', 'z'))[0], 1, 'nonnegreals');
  // posreals is (0, inf], so the atom at 0 is out.
  close((await value('z = totalmass(truncate(Poisson(3.0), posreals))', 'z'))[0],
    -Math.expm1(-3), 'posreals');
});

test('normalize(truncate(Poisson, S)) scores pmf / mass at both ends', async () => {
  for (const [lo, hi] of [[0, 5], [2, 4], [3, 3]]) {
    for (const k of [lo, hi]) {
      const l = await value(`m = normalize(truncate(Poisson(3.0), interval(${lo}, ${hi})))
l = logdensityof(m, ${k})`, 'l');
      close(l[0], logPmf(k, 3) - Math.log(mass(3, lo, hi)), `interval(${lo}, ${hi}) at ${k}`);
    }
  }
  const l = await value('l = logdensityof(normalize(truncate(Poisson(3.0), posreals)), 1)', 'l');
  close(l[0], logPmf(1, 3) - Math.log(-Math.expm1(-3)), 'posreals at 1');
  assert.equal((await value('l = logdensityof(normalize(truncate(Poisson(3.0), posreals)), 0)', 'l'))[0],
    -Infinity, 'posreals excludes 0');
});

test('the kernel and broadcast paths resolve the same discrete normalizer', async () => {
  const kernel = await value(`mu ~ Uniform(1.0, 5.0)
y ~ normalize(truncate(Poisson(mu), interval(0, 5)))
K = kernelof(record(y = y), mu = mu)
L = likelihoodof(K, record(y = 0))
l = logdensityof(L, 3.0)`, 'l');
  close(kernel[0], logPmf(0, 3) - Math.log(mass(3, 0, 5)), 'kernelof');
  const broadcast = await value(`energy = p -> normalize(truncate(Poisson(p), interval(2, 5)))
l = logdensityof(energy.([3.0, 4.0]), [2, 5])`, 'l');
  close(broadcast[0], logPmf(2, 3) - Math.log(mass(3, 2, 5)) + logPmf(5, 4) - Math.log(mass(4, 2, 5)),
    'broadcast');
});

test('MCMC likelihood re-resolves a parameter-dependent discrete normalizer', async () => {
  const { ctx } = makeMatCtx(`r ~ Uniform(1.0, 5.0)
y ~ iid(normalize(truncate(Poisson(r), interval(2, 4))), 2)
K = kernelof(record(y = y), r = r)
L = likelihoodof(K, record(y = [2, 4]))
P = bayesupdate(L, lawof(r))`, { sampleCount: 16 });
  const derivation = Object.values(ctx.derivations).find((d: any) => d.kind === 'bayesupdate');
  const { likOf } = await buildLogPi(ctx, derivation);
  for (const r of [1.5, 3.5]) {
    close(likOf({ r }), logPmf(2, r) + logPmf(4, r) - 2 * Math.log(mass(r, 2, 4)), `r = ${r}`);
  }
});

test('samples of a truncated Poisson reach both closed ends at the pmf rate', async () => {
  const n = 20000;
  for (const [lo, hi] of [[0, 5], [2, 4]]) {
    const xs = Array.from(await value(`x ~ normalize(truncate(Poisson(3.0), interval(${lo}, ${hi})))`, 'x', n));
    assert.equal(Math.min(...xs), lo);
    assert.equal(Math.max(...xs), hi);
    for (const k of [lo, hi]) {
      const p = Math.exp(logPmf(k, 3)) / mass(3, lo, hi);
      const freq = xs.filter((x) => x === k).length / n;
      assert.ok(Math.abs(freq - p) < 5 * Math.sqrt(p * (1 - p) / n),
        `interval(${lo}, ${hi}): P(${k}) ${freq} vs ${p}`);
    }
  }
  const point = await value('x ~ normalize(truncate(Poisson(3.0), interval(0, 0)))', 'x', 100);
  assert.ok(Array.from(point).every((x) => x === 0), 'interval(0, 0) holds only the atom at 0');
  const positive = await value('x ~ normalize(truncate(Poisson(3.0), posreals))', 'x', 2000);
  assert.ok(Array.from(positive).every((x) => x >= 1), 'posreals excludes the atom at 0');
});

test('a truncation set holding no atom has zero mass and normalize refuses it', async () => {
  const src = 'm = normalize(truncate(Poisson(3.0), interval(0.2, 0.8)))';
  close((await value('z = totalmass(truncate(Poisson(3.0), interval(0.2, 0.8)))', 'z'))[0], 0, 'no atom');
  await assert.rejects(() => value(`${src}\nl = logdensityof(m, 0)`, 'l'), /mass over the truncation set is 0|Z = 0/);
});

test('filtering a superposed base keeps both closed ends', async () => {
  // A weighted parent is filtered: an atom outside S keeps weight -inf.
  const { ctx } = makeMatCtx(
    'x ~ normalize(truncate(superpose(weighted(0.5, Poisson(3.0)), weighted(0.5, Poisson(3.0))), interval(2, 4)))',
    { sampleCount: 4000 });
  const m = await ctx.getMeasure('x');
  const kept = Array.from(m.samples as Float64Array).filter((_x, i) => m.logWeights[i] > -Infinity);
  assert.ok(kept.includes(2) && kept.includes(4) && kept.every((x) => x >= 2 && x <= 4));
});

test('a reversed interval refuses to sample', async () => {
  await assert.rejects(() => value('x ~ normalize(truncate(Poisson(3.0), interval(5, 2)))', 'x', 10),
    /degenerate set bounds/);
});
