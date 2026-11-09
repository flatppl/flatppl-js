'use strict';
// A measure-valued ifelse with an inline per-atom condition, `ifelse(x > 0.0,
// A, B)` with x a draw, had no derivation while the named spelling
// `c = x > 0.0; ifelse(c, A, B)` scored. §07: ifelse returns A where the
// condition holds and B elsewhere, so atom i's measure is the branch its x_i
// selects. Lift now hoists the condition, giving the named spelling.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator } = require('../index.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const lnN = (x: number, m: number, s: number) => -0.5 * Math.log(2 * Math.PI) - Math.log(s) - 0.5 * ((x - m) / s) ** 2;
const X = 'x = draw(Normal(0.0, 1.0))\n';

// The comparisons lift hoisted into their own bindings.
function hoistedComparisons(src: string): string[] {
  const lifted = orchestrator.buildDerivations(processSource(src).bindings).bindings;
  return Array.from(lifted.entries())
    .filter(([, b]: any) => b.synthetic && b.node && b.node.value && b.node.value.type === 'BinaryExpr')
    .map(([n]: any) => n);
}

async function columns(src: string, names: string[], N: number): Promise<Record<string, Float64Array>> {
  const { ctx } = makeMatCtx(src, { sampleCount: N });
  const out: Record<string, Float64Array> = {};
  for (const n of names) {
    const m = await ctx.getMeasure(n);
    out[n] = m.samples || m.value.data;
  }
  return out;
}

test('an inline condition scores the branch each atom selects, as the named spelling does', async () => {
  const N = 64;
  const hi = lnN(4, 5, 1), lo = lnN(4, -5, 1);
  const inline = await columns(X + 'm = ifelse(x > 0.0, Normal(5.0, 1.0), Normal(-5.0, 1.0))\nlp = logdensityof(m, 4.0)\n', ['x', 'lp'], N);
  const named = await columns(X + 'c = x > 0.0\nm = ifelse(c, Normal(5.0, 1.0), Normal(-5.0, 1.0))\nlp = logdensityof(m, 4.0)\n', ['x', 'lp'], N);
  for (let i = 0; i < N; i++) {
    const want = inline.x[i] > 0 ? hi : lo;
    assert.ok(Math.abs(inline.lp[i] - want) < 1e-12, `atom ${i}: ${inline.lp[i]} vs ${want}`);
    assert.equal(inline.lp[i], named.lp[i], `atom ${i}: inline equals named`);
  }
});

test('an inline condition samples the branch each atom selects', async () => {
  const N = 2000;
  const c = await columns(X + 'm = ifelse(x > 0.0, Normal(5.0, 1.0), Normal(-5.0, 1.0))\ny = draw(m)\n', ['x', 'y'], N);
  let agree = 0;
  for (let i = 0; i < N; i++) if ((c.x[i] > 0) === (c.y[i] > 0)) agree++;
  assert.equal(agree, N);
});

test('an inline condition over inline record branches selects per atom', async () => {
  const N = 32;
  const c = await columns(X + 'm = ifelse(x > 0.0, joint(mu = Normal(5.0, 1.0)), joint(mu = Normal(-5.0, 1.0)))\nlp = logdensityof(m, record(mu = 4.0))\n', ['x', 'lp'], N);
  for (let i = 0; i < N; i++) {
    const want = c.x[i] > 0 ? lnN(4, 5, 1) : lnN(4, -5, 1);
    assert.ok(Math.abs(c.lp[i] - want) < 1e-12, `atom ${i}: ${c.lp[i]} vs ${want}`);
  }
});

test('a value-valued ifelse keeps its condition in place', async () => {
  const src = X + 'y = ifelse(x > 0.0, x + 1.0, x - 1.0)\n';
  assert.deepEqual(hoistedComparisons(src), [], 'no hoisted condition');
  assert.equal(hoistedComparisons(X + 'm = ifelse(x > 0.0, Normal(5.0, 1.0), Normal(-5.0, 1.0))\n').length, 1, 'the measure-valued spelling hoists');
  const c = await columns(src, ['x', 'y'], 64);
  for (let i = 0; i < 64; i++) assert.equal(c.y[i], c.x[i] > 0 ? c.x[i] + 1 : c.x[i] - 1);
});

test('a fixed inline condition still selects its measure outright', async () => {
  const c = await columns('m = ifelse(1.0 > 0.0, Normal(0.0, 2.0), Normal(0.0, 1.0))\nlp = logdensityof(m, 0.7)\n', ['lp'], 1);
  assert.ok(Math.abs(c.lp[0] - lnN(0.7, 0, 2)) < 1e-12, `${c.lp[0]}`);
});

test('a condition holding a hole is not hoisted', () => {
  assert.deepEqual(hoistedComparisons('K = fn(ifelse(_ > 0.0, Normal(5.0, 1.0), Normal(-5.0, 1.0)))\n'), []);
});
