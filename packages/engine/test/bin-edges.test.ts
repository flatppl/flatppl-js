'use strict';

// Spec §07 `bincounts`: "Each edge vector must be strictly increasing", and §06
// `BinnedPoissonProcess` takes its bins in that format. Written edges are refused
// statically. Computed edges throw when they are evaluated.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const RULE = /bin edges must be strictly increasing/;
const MODEL = 'x = [0.5, 1.5]\nI = weighted(2.0, Normal(1.0, 1.0))\n';

function errors(src: string) {
  return processSource(src).diagnostics.filter((d: any) => d.severity === 'error');
}

test('written edges that do not strictly increase are refused', () => {
  for (const call of [
    'bincounts([0.0, 2.0, 1.0], x)',
    'bincounts([0.0, 1.0, 1.0], x)',
    'bincounts(record(a = [0.0, 1.0], b = [1.0, 0.0]), record(a = x, b = x))',
    'bincounts(linspace(2.0, 0.0, 3), x)',
    'bincounts(extlinspace(2.0, 0.0, 3), x)',
    'BinnedPoissonProcess([0.0, 2.0, 1.0], I)',
    'BinnedPoissonProcess([1.0, -inf], I)',
  ]) {
    const ds = errors(`${MODEL}n = ${call}\n`);
    assert.ok(ds.some((d: any) => RULE.test(d.message) && d.loc), call + ': ' + ds.map((d: any) => d.message));
  }
  const ds = errors(`${MODEL}e = [0.0, 0.0]\nn = bincounts(e, x)\n`);
  assert.ok(ds.some((d: any) => RULE.test(d.message)), 'through a binding');
});

test('strictly increasing written edges are accepted', () => {
  const ds = errors(`${MODEL}a = bincounts([0.0, 1.0, inf], x)\nb = bincounts(linspace(0.0, 2.0, 3), x)\n`
    + 'c = BinnedPoissonProcess(extlinspace(0.0, 2.0, 3), I)\n');
  assert.deepEqual(ds.map((d: any) => d.message), []);
});

test('computed edges that do not strictly increase throw at evaluation', async () => {
  const { ctx } = makeMatCtx('e = cumsum([0.0, 1.0, -2.0])\nc = bincounts(e, [0.5, 1.5])\n');
  await assert.rejects(ctx.getMeasure('c'), RULE);
  const binned = makeMatCtx('e = cumsum([0.0, 1.0, -2.0])\nI = weighted(2.0, Normal(1.0, 1.0))\n'
    + 'lp = logdensityof(BinnedPoissonProcess(e, I), [1, 0])\n');
  await assert.rejects(binned.ctx.getMeasure('lp'), RULE);
});
