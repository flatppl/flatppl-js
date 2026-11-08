'use strict';
// §09: a standard module's distribution, `hepphys.ContinuedPoisson(rate)`, is a
// measure constructor. A plain call bound to a name, or inline in logdensityof,
// failed to evaluate ("standard module … has no binding 'ContinuedPoisson'")
// while the broadcast spelling the §12 profile tables use already scored. The
// module entry's distribution set is the one source lower, the analyzer,
// typeinfer and the derivations read.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const H = 'hepphys = standard_module("particle-physics", "0.1")\n';

async function score(src: string): Promise<number> {
  const { ctx } = makeMatCtx(src, { sampleCount: 1 });
  const m = await ctx.getMeasure('lp');
  return (m.value ? Array.from(m.value.data) : m.samples)[0] as number;
}
function close(got: number, want: number, label: string) {
  assert.ok(Math.abs(got - want) < 1e-12, `${label}: got ${got}, want ${want}`);
}

// §09 ContinuedPoisson(rate) at x: x log(rate) − rate − lgamma(x + 1).
// lgamma(3.5) = log(15 √π / 8).
const CP = 2.5 * Math.log(3.0) - 3.0 - Math.log(15 * Math.sqrt(Math.PI) / 8);

test('a module distribution scores as a plain binding, inline and broadcast', async () => {
  close(await score(H + 'M = hepphys.ContinuedPoisson(3.0)\nlp = logdensityof(M, 2.5)\n'), CP, 'plain binding');
  close(await score(H + 'M = hepphys.ContinuedPoisson(rate = 3.0)\nlp = logdensityof(M, 2.5)\n'), CP, 'keyword');
  close(await score(H + 'lp = logdensityof(hepphys.ContinuedPoisson(3.0), 2.5)\n'), CP, 'inline');
  close(await score(H + 'lp = logdensityof(hepphys.ContinuedPoisson.([3.0]), [2.5])\n'), CP, 'broadcast');
});

test('a second module distribution agrees with scipy', async () => {
  // scipy 1.18: crystalball(beta = 1.5, m = 3.0).logpdf(0.5).
  close(await score(H + 'M = hepphys.CrystalBall(0.0, 1.0, 1.5, 3.0)\nlp = logdensityof(M, 0.5)\n'), -1.1047611536741122, 'CrystalBall');
});

test('a module distribution binding is a measure and a module function stays a value', async () => {
  const r = processSource(H + 'M = hepphys.ContinuedPoisson(3.0)\nf = hepphys.kallen(1.0, 0.25, 0.25)\n');
  assert.equal(r.bindings.get('M').inferredType.kind, 'measure');
  const { ctx } = makeMatCtx(H + 'f = hepphys.kallen(1.0, 0.25, 0.25)\n', { sampleCount: 1 });
  assert.equal((await ctx.getMeasure('f')).value.data[0], 0);
});
