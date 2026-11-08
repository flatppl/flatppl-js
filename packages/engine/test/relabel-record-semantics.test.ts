'use strict';
// §04 "Interface adaptation": relabel assigns field names. A scalar becomes a
// one-field record, an array becomes a record of its elements, and the rule
// lifts to sets, functions, measures and kernels, with
// relabel(M, names) ≡ pushfwd(fn(relabel(_, names)), M).
//
// §06 `likelihoodof`: the observed data is one variate of the kernel's
// measures, so it must have the variates' type. likelihoodof checks it with
// the same unify rule logdensityof applies to its point.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const lnN = (x: number, m: number, s: number) => -0.5 * Math.log(2 * Math.PI) - Math.log(s) - 0.5 * ((x - m) / s) ** 2;
const errors = (src: string): string[] =>
  processSource(src).diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message);

async function score(src: string, name = 'ld'): Promise<number> {
  const { ctx } = makeMatCtx(src, { sampleCount: 1 });
  const m = await ctx.getMeasure(name);
  return (m.value ? Array.from(m.value.data) : m.samples)[0] as number;
}
function close(got: number, want: number, label: string) {
  assert.ok(Math.abs(got - want) < 1e-12, `${label}: got ${got}, want ${want}`);
}

const MU = 'mu = elementof(reals)\n';
const REFUSED = [
  ['a scalar kernel at a length-1 array', MU + 'L = likelihoodof(functionof(Normal(mu, 1.0), mu = mu), [2.0])\n', 'array of real (length 1)', 'real'],
  ['a scalar kernel at a length-2 array', MU + 'L = likelihoodof(functionof(Normal(mu, 1.0), mu = mu), [2.0, 3.0])\n', 'array of real (length 2)', 'real'],
  ['iid of 3 at a length-2 array', MU + 'L = likelihoodof(functionof(iid(Normal(mu, 1.0), 3), mu = mu), [2.0, 3.0])\n', 'array of real (length 2)', 'array of real (length 3)'],
  ['iid of 2 at a scalar', MU + 'L = likelihoodof(functionof(iid(Normal(mu, 1.0), 2), mu = mu), 2.0)\n', 'real', 'array of real (length 2)'],
  ['a scalar kernel at a record', MU + 'L = likelihoodof(functionof(Normal(mu, 1.0), mu = mu), record(x = 2.0))\n', 'record with fields x: real', 'real'],
  ['a record kernel at a length-1 field', MU + 'y = draw(Normal(mu, 1.0))\nL = likelihoodof(kernelof(record(y = y), mu = mu), record(y = [2.0]))\n',
    'record with fields y: array of real (length 1)', 'record with fields y: real'],
  ['iid of a relabelled measure at a plain vector', 'm = elementof(reals)\ns = elementof(posreals)\ng = relabel(Normal(mu = m, sigma = s), ["x"])\nL = likelihoodof(iid(g, 2), [0.0, 1.0])\n',
    'array of real (length 2)', 'table with columns x: real (2 rows)'],
] as const;

test('likelihoodof refuses observed data that is not a variate of the kernel', () => {
  for (const [label, src, observed, variate] of REFUSED) {
    const found = errors(src);
    assert.equal(found.length, 1, `${label}: ${JSON.stringify(found)}`);
    for (const needle of ['`likelihoodof`: the observed data is ' + observed,
      'variates of type ' + variate, 'explicit `iid`', 'spec §06 `likelihoodof`']) {
      assert.ok(found[0].includes(needle), `${label}: "${needle}" not in ${found[0]}`);
    }
  }
});

test('likelihoodof accepts observed data of the variate type and scores it', async () => {
  const at = 'ld = logdensityof(L, record(mu = 0.5))\n';
  close(await score(MU + 'L = likelihoodof(functionof(Normal(mu, 1.0), mu = mu), 2.0)\n' + at), lnN(2, 0.5, 1), 'scalar');
  close(await score(MU + 'L = likelihoodof(functionof(iid(Normal(mu, 1.0), 2), mu = mu), [2.0, 3.0])\n' + at),
    lnN(2, 0.5, 1) + lnN(3, 0.5, 1), 'iid');
  close(await score(MU + 'y = draw(Normal(mu, 1.0))\nL = likelihoodof(kernelof(record(y = y), mu = mu), record(y = 2.0))\n' + at),
    lnN(2, 0.5, 1), 'record');
});

test('an iid of a relabelled measure scores at a table', async () => {
  const want = lnN(0.0, 0.4, 1.3) + lnN(1.0, 0.4, 1.3);
  const g = 'g = relabel(Normal(mu = 0.4, sigma = 1.3), ["x"])\n';
  close(await score(g + 'm = iid(g, 2)\nld = logdensityof(m, table(x = [0.0, 1.0]))\n'), want, 'named');
  close(await score('ld = logdensityof(iid(relabel(Normal(mu = 0.4, sigma = 1.3), ["x"]), 2), table(x = [0.0, 1.0]))\n'), want, 'inline');
});

test('the §04 relabel of an iid scores at its record', async () => {
  close(await score('m = relabel(iid(Normal(mu = 0.4, sigma = 1.3), 2), ["a", "b"])\nld = logdensityof(m, record(a = 0.0, b = 1.0))\n'),
    lnN(0.0, 0.4, 1.3) + lnN(1.0, 0.4, 1.3), 'relabel(iid)');
});

test('a relabelled set has record members', async () => {
  close(await score('S = relabel(cartpow(reals, 2), ["x", "y"])\np = elementof(S)\nM = Normal(mu = p.x, sigma = 1.0)\n'
    + 'ld = logdensityof(likelihoodof(functionof(M), 0.3), record(p = record(x = 0.1, y = 2.0)))\n'), lnN(0.3, 0.1, 1.0), 'cartpow');
  close(await score('p = elementof(relabel(reals, ["x"]))\nM = Normal(mu = p.x, sigma = 1.0)\n'
    + 'ld = logdensityof(likelihoodof(functionof(M), 0.3), record(p = record(x = 0.1)))\n'), lnN(0.3, 0.1, 1.0), 'scalar set');
  close(await score('p = elementof(relabel(cartprod(a = reals, b = posreals), ["x", "y"]))\nM = Normal(mu = p.x, sigma = p.y)\n'
    + 'ld = logdensityof(likelihoodof(functionof(M), 0.3), record(p = record(x = 0.1, y = 2.0)))\n'), lnN(0.3, 0.1, 2.0), 'record set');
});

test('a relabel whose names do not match the components refuses before scoring', () => {
  // A measure with no derivation of its own reaches logdensityof through its
  // structural lowering; a name count that cannot match refuses statically
  // instead of failing in the density walk.
  for (const [src, names, base, count] of [
    ['m = relabel(iid(Normal(0.0, 1.0), 2), ["a", "b", "c"])\nld = logdensityof(m, record(a = 0.0, b = 1.0, c = 2.0))\n', '3 names', 'array of real (length 2)', '2 components'],
    ['m = relabel(iid(Normal(0.0, 1.0), 2), ["a"])\nld = logdensityof(m, record(a = 0.0))\n', '1 name', 'array of real (length 2)', '2 components'],
    ['v = relabel([1.0, 2.0, 3.0], ["x", "y"])\n', '2 names', 'array of real (length 3)', '3 components'],
    ['v = relabel(record(a = 1.0, b = 2.0), ["x"])\n', '1 name', 'record with fields a: real, b: real', '2 components'],
  ] as Array<[string, string, string, string]>) {
    const found = errors(src);
    assert.ok(found.some((m) => m.startsWith(`relabel: ${names} for ${base}, which has ${count}`)
      && m.includes('spec §04 "Interface adaptation"')), `${src}: ${JSON.stringify(found)}`);
  }
});

test('a point lies in the §08 domain: integer data for Bernoulli, not real data for Poisson', async () => {
  // §08: Bernoulli has domain integers and support booleans; §03 booleans ⊂
  // integers. Poisson's domain is integers, so a real-typed count refuses.
  assert.deepEqual(errors('lp = logdensityof(iid(Bernoulli(0.3), 2), [1, 0])\n'), []);
  close(await score('lp = logdensityof(iid(Bernoulli(0.3), 2), [1, 0])\n', 'lp'), Math.log(0.3) + Math.log(0.7), 'Bernoulli [1, 0]');
  assert.equal(await score('lp = logdensityof(Bernoulli(0.3), 2)\n', 'lp'), -Infinity);
  close(await score('p = elementof(interval(0.0, 1.0))\nL = likelihoodof(functionof(iid(Bernoulli(p), 3), p = p), [1, 0, 1])\nlp = logdensityof(L, record(p = 0.3))\n', 'lp'),
    2 * Math.log(0.3) + Math.log(0.7), 'Bernoulli likelihood');
  assert.deepEqual(errors('lp = logdensityof(iid(Poisson(3.0), 2), [50.0, 52.0])\n'),
    ['logdensityof: arg 2 expects array of integer (length 2), got array of real (length 2)']);
  // likelihoodof checks shape and structure only until the pyhf / HS3
  // converters stop observing a Poisson at real-typed counts, so on element
  // kinds it differs from logdensityof for now (TODO-flatppl-js).
  assert.deepEqual(errors('mu = elementof(posreals)\nL = likelihoodof(functionof(iid(Poisson(mu), 2), mu = mu), [50.0, 52.0])\n'), []);
});
