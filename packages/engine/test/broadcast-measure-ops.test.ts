'use strict';
// §04 "Broadcasting": `f.(args)` lowers to `broadcast(f, args)`; `broadcast(
// kernel, ...)` returns a single product measure, not an array of measures; and
// measures, sets and scalars are "not iterated over but held constant", so with
// no collection argument "broadcast behaves like a single function or kernel
// call". So `truncate.(Normal.(mu, 1.0), S)` is the single call truncate(<product
// measure>, S), not one truncation per element; the per-element program is a
// kernel broadcast, `broadcast(fn(truncate(Normal(_, 1.0), S)), mu)`.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const errors = (src: string): string[] =>
  processSource(src).diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message);

async function score(src: string): Promise<number> {
  const { ctx } = makeMatCtx(src, { sampleCount: 1 });
  const m = await ctx.getMeasure('lp');
  return (m.value ? Array.from(m.value.data) : m.samples)[0] as number;
}

const MU = 'mu = [0.0, 1.0]\n';
// scipy 1.18: sum over mu in {0, 1} of truncnorm(-1 - mu, 2 - mu, loc = mu).logpdf(0.5).
const TRUNCNORM_SUM = -1.68754447776042;
const lnN = (x: number, m: number) => -0.5 * Math.log(2 * Math.PI) - 0.5 * (x - m) ** 2;

test('an inline kernel broadcast of a normalized truncation scores per element', async () => {
  const got = await score(MU + 'lp = logdensityof(broadcast(fn(normalize(truncate(Normal(_, 1.0), interval(-1.0, 2.0)))), mu), [0.5, 0.5])\n');
  assert.ok(Math.abs(got - TRUNCNORM_SUM) < 1e-12, `${got}`);
  const named = await score(MU + 'K(m) = normalize(truncate(Normal(m, 1.0), interval(-1.0, 2.0)))\nlp = logdensityof(K.(mu), [0.5, 0.5])\n');
  assert.ok(Math.abs(named - TRUNCNORM_SUM) < 1e-12, `named ${named}`);
});

test('a dotted truncate over the array-valued measure is one call and refuses with the reading', () => {
  for (const src of [
    MU + 'lp = logdensityof(truncate.(Normal.(mu, 1.0), interval(-1.0, 2.0)), [0.5, 0.5])\n',
    MU + 'lp = logdensityof(normalize.(truncate.(Normal.(mu, 1.0), interval(-1.0, 2.0))), [0.5, 0.5])\n',
  ]) {
    const errs = errors(src);
    assert.equal(errs.length, 1, JSON.stringify(errs));
    assert.match(errs[0], /^truncate\(M, S\): the region S is a set of scalar values but M has a array variate/);
    assert.ok(errs[0].includes('`truncate.(…)` has no collection operand, so broadcast holds its measures and sets '
      + 'constant and this is one call on the whole array-valued measure (spec §04 "Broadcasting")'), errs[0]);
    assert.ok(errs[0].includes('broadcast a kernel: `broadcast(fn(truncate(Normal(_, 1.0), …)), mu)`'), errs[0]);
  }
});

test('a valid dotted single call evaluates as that call', async () => {
  const dotted = await score(MU + 'lp = logdensityof(normalize.(Normal.(mu, 1.0)), [0.5, 0.5])\n');
  const plain = await score(MU + 'lp = logdensityof(normalize(Normal.(mu, 1.0)), [0.5, 0.5])\n');
  assert.equal(dotted, plain);
  assert.ok(Math.abs(dotted - (lnN(0.5, 0) + lnN(0.5, 1))) < 1e-12, `${dotted}`);
});

test('a dotted measure op with a collection operand keeps its broadcast reading', () => {
  // weighted.(w, M) iterates w: a kernel broadcast, not a single call.
  assert.deepEqual(errors('w = [1.0, 2.0]\nM = Normal(0.0, 1.0)\nlp = logdensityof(weighted.(w, M), [0.5, 0.5])\n'), []);
});

test('a value-valued lambda broadcast stays inline', async () => {
  const { ctx } = makeMatCtx('a = [1.0, 2.0]\nc = broadcast(fn(_ + 1.0), a)\n', { sampleCount: 1 });
  const m = await ctx.getMeasure('c');
  assert.deepEqual(Array.from(m.value.data), [2, 3]);
});

test('an operand of unknown type keeps the broadcast reading', () => {
  // Inside a callback the operand is a placeholder of unknown type; it may be an
  // array, so the no-collection single-call reading must not apply.
  const { orchestrator } = require('../index.ts');
  const lifted = orchestrator.buildDerivations(processSource('g = fn(weighted.(_, Normal(0.0, 1.0)))\n').bindings).bindings;
  const ir = JSON.stringify(lifted.get('g').ir);
  assert.ok(ir.includes('"op":"broadcast"') || ir.includes('broadcast'), ir.slice(0, 300));
  assert.ok(!/"op":"weighted","args":\[\{"kind":"ref","ns":"%local"/.test(ir), 'rewritten to a single weighted call: ' + ir.slice(0, 300));
});

test('logdensityof over a callback operand keeps iterating it', async () => {
  // §04: `logdensityof.(p, x)` iterates x; the zero-sum model scores per cell.
  const { ctx } = makeMatCtx(`
s = elementof(posreals)
N = Normal(0, s)
M = logweighted((x, p) -> sum(logdensityof.(p, x)), Lebesgue(zerosum(3)), N)
score = logdensityof(M, [-1, 0, 1])
f = functionof(score, scale = s)
a = f(2)
`, { sampleCount: 1 });
  const a = (await ctx.getMeasure('a')).samples[0];
  const want = -3 * Math.log(2 * Math.sqrt(2 * Math.PI)) - 1 / 4;
  assert.ok(Math.abs(a - want) < 1e-12, `${a} vs ${want}`);
});
