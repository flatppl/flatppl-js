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

// Atoms a filtered measure keeps: an atom outside S carries weight -inf.
async function ctx2(src: string): Promise<number[]> {
  const { ctx } = makeMatCtx(src, { sampleCount: 4000 });
  const m = await ctx.getMeasure('x');
  return Array.from(m.samples as Float64Array).filter((_x, i) => m.logWeights[i] > -Infinity);
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
  const open = await ctx2('x ~ normalize(truncate(superpose(weighted(0.5, Poisson(3.0)), weighted(0.5, Poisson(3.0))), posreals))');
  assert.ok(open.includes(1) && open.every((x) => x >= 1), 'posreals drops the atom at 0');
});

test('a reversed interval refuses to sample', async () => {
  await assert.rejects(() => value('x ~ normalize(truncate(Poisson(3.0), interval(5, 2)))', 'x', 10),
    /degenerate set bounds/);
});

test('reals, unitinterval and a lower-unbounded interval count the atoms they hold', async () => {
  close((await value('z = totalmass(truncate(Poisson(3.0), reals))', 'z'))[0], 1, 'reals');
  close((await value('z = totalmass(truncate(Poisson(3.0), interval(-inf, 2)))', 'z'))[0], mass(3, 0, 2), '(-inf, 2]');
  // Binomial(3, 0.4) keeps the atoms 0 and 1 of the closed unit interval.
  close((await value('z = totalmass(truncate(Binomial(3, 0.4), unitinterval))', 'z'))[0], 0.216 + 0.432, 'unitinterval');
  const reals = await value(`energy = p -> normalize(truncate(Poisson(p), reals))
l = logdensityof(energy.([3.0, 4.0]), [2, 5])`, 'l');
  close(reals[0], logPmf(2, 3) + logPmf(5, 4), 'per-row reals');
  const unit = await value(`energy = p -> normalize(truncate(Binomial(3, p), unitinterval))
l = logdensityof(energy.([0.4, 0.5]), [0, 1])`, 'l');
  close(unit[0], Math.log(0.216 / 0.648) + Math.log(0.375 / 0.5), 'per-row unitinterval');
  // §03 booleans has no numeric bounds here: the per-row normalizer refuses.
  await assert.rejects(() => value(`energy = p -> normalize(truncate(Poisson(p), booleans))
l = logdensityof(energy.([3.0, 4.0]), [0, 1])`, 'l'), /cannot resolve truncated normalizer/);
});

test('a truncate normalizer defers to the per-call path for a known scalar law only', () => {
  const { canDeferTruncateNormalizer } = require('../normalize-mass.ts');
  const lit = (value: number) => ({ kind: 'lit', value });
  const set = { kind: 'call', op: 'interval', args: [lit(0), lit(5)] };
  const truncate = (base: any) => ({ kind: 'call', op: 'truncate', args: [base, set] });
  const poisson = { kind: 'call', op: 'Poisson', args: [lit(3)] };
  assert.equal(canDeferTruncateNormalizer(truncate(poisson)), true, 'a discrete law');
  assert.equal(canDeferTruncateNormalizer(truncate({ kind: 'call', op: 'Normal', args: [lit(0), lit(1)] })), true, 'a CDF row');
  assert.equal(canDeferTruncateNormalizer(truncate({ kind: 'call', op: 'VonMises', args: [lit(0), lit(1)] })), false, 'no CDF row');
  assert.equal(canDeferTruncateNormalizer(truncate({ kind: 'call', op: 'weighted', args: [lit(2), poisson] })), false, 'not a law');
});

test('an integer set keeps the atoms of its range and no continuous mass', async () => {
  // posintegers keeps 1, 2, ...; nonnegintegers and integers keep every atom.
  for (const [set, first] of [['posintegers', 1], ['nonnegintegers', 0], ['integers', 0]] as const) {
    const Z = first === 1 ? -Math.expm1(-3) : 1;
    close((await value(`z = totalmass(truncate(Poisson(3.0), ${set}))`, 'z'))[0], Z, `${set} totalmass`);
    close((await value(`l = logdensityof(normalize(truncate(Poisson(3.0), ${set})), 2)`, 'l'))[0],
      logPmf(2, 3) - Math.log(Z), `${set} at 2`);
    const atZero = (await value(`l = logdensityof(normalize(truncate(Poisson(3.0), ${set})), 0)`, 'l'))[0];
    if (first === 1) assert.equal(atZero, -Infinity, `${set} excludes 0`);
    else close(atZero, logPmf(0, 3), `${set} at 0`);
    const Z4 = first === 1 ? -Math.expm1(-4) : 1;
    close((await value(`energy = p -> normalize(truncate(Poisson(p), ${set}))
l = logdensityof(energy.([3.0, 4.0]), [2, 5])`, 'l'))[0],
    logPmf(2, 3) - Math.log(Z) + logPmf(5, 4) - Math.log(Z4), `${set} per row`);
    const xs = Array.from(await value(`x ~ normalize(truncate(Poisson(3.0), ${set}))`, 'x', 2000));
    assert.ok(xs.includes(first) && xs.every((x) => Number.isInteger(x) && x >= first), `${set} draws`);
    const kept = await ctx2(`x ~ normalize(truncate(superpose(weighted(0.5, Poisson(3.0)), weighted(0.5, Poisson(3.0))), ${set}))`);
    assert.ok(kept.includes(first) && kept.every((x) => x >= first), `${set} filter`);
  }
  // A continuous law puts no mass on the integers, so its normalizer is undefined.
  close((await value('z = totalmass(truncate(Normal(0.0, 1.0), posintegers))', 'z'))[0], 0, 'Normal totalmass');
  await assert.rejects(() => value('l = logdensityof(normalize(truncate(Normal(0.0, 1.0), nonnegintegers)), 1.0)', 'l'),
    /Z = 0/);
  await assert.rejects(() => value(`energy = m -> normalize(truncate(Normal(m, 1.0), integers))
l = logdensityof(energy.([0.0, 1.0]), [1.0, 2.0])`, 'l'), /no mass on an integer set/);
});
