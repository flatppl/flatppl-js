'use strict';
// §04 "Broadcasting": collection operands are iterated, measures and other
// non-collection operands held constant. So `weighted.(w, M)` with w = [1, 2]
// is the kernel broadcast whose kernel at position i is weighted(w_i, M): the
// product of weighted(1, M) and weighted(2, M). It typed but failed at
// evaluation ("broadcast: first arg must be a function").

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator } = require('../index.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const lnN = (x: number) => -0.5 * Math.log(2 * Math.PI) - 0.5 * x * x;
const M = 'M = Normal(0.0, 1.0)\n';

async function value(src: string, name: string, N = 1): Promise<Float64Array> {
  const { ctx } = makeMatCtx(src, { sampleCount: N });
  const m = await ctx.getMeasure(name);
  return m.value ? m.value.data : m.samples;
}

test('a weighted kernel broadcast over a weight array scores the product', async () => {
  // log 1 + log N(0.5) + log 2 + log N(0.5)
  const want = Math.log(2) + 2 * lnN(0.5);
  const w = (await value('w = [1.0, 2.0]\n' + M + 'lp = logdensityof(weighted.(w, M), [0.5, 0.5])\n', 'lp'))[0];
  assert.ok(Math.abs(w - want) < 1e-12, `weighted ${w}`);
  const lw = (await value('lw = [0.0, log(2.0)]\n' + M + 'lp = logdensityof(logweighted.(lw, M), [0.5, 0.5])\n', 'lp'))[0];
  assert.ok(Math.abs(lw - want) < 1e-12, `logweighted ${lw}`);
});

test('its total mass is the product of the weights', async () => {
  // totalmass(weighted(1, M)) * totalmass(weighted(2, M)) = 1 * 2.
  const tm = (await value('w = [1.0, 2.0]\n' + M + 'tm = totalmass(weighted.(w, M))\n', 'tm'))[0];
  assert.ok(Math.abs(tm - 2) < 1e-12, `${tm}`);
});

test('its draws come from M in each cell', async () => {
  const N = 4000;
  const y = await value('w = [1.0, 2.0]\n' + M + 'y = draw(weighted.(w, M))\n', 'y', N);
  assert.equal(y.length, 2 * N);
  for (let c = 0; c < 2; c++) {
    let s = 0, s2 = 0;
    for (let i = 0; i < N; i++) { const v = y[2 * i + c]; s += v; s2 += v * v; }
    const mean = s / N, sd = Math.sqrt(s2 / N - mean * mean);
    assert.ok(Math.abs(mean) < 6 / Math.sqrt(N), `cell ${c} mean ${mean}`);
    assert.ok(Math.abs(sd - 1) < 0.08, `cell ${c} sd ${sd}`);
  }
});

test('with no collection operand the call keeps the single-call reading', () => {
  const errs = processSource('mu = [0.0, 1.0]\nlp = logdensityof(truncate.(Normal.(mu, 1.0), interval(-1.0, 2.0)), [0.5, 0.5])\n')
    .diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message);
  assert.equal(errs.length, 1);
  assert.ok(errs[0].includes('has no collection operand'), errs[0]);
});

test('an operand of unknown type leaves the broadcast as it is', () => {
  const lifted = orchestrator.buildDerivations(processSource('g = fn(weighted.(_, Normal(0.0, 1.0)))\n').bindings).bindings;
  const synthetic = Array.from(lifted.values()).filter((b: any) => b.synthetic
    && b.node?.value?.callee?.name === 'functionof');
  assert.deepEqual(synthetic.map((b: any) => b.name), []);
});
