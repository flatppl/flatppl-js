'use strict';

// `rand(iid(M, n), state)` of a record-valued M. §06 and §03 make the iid of a
// record measure a table with one row per draw, and inference types it so,
// but the sampler's iid walk returned an array of records. `r.a` then failed
// ("Cannot read properties of undefined"), and so did a density at `r`. The
// materialiser's composite draw refused the record case outright. Both now
// build the table. M has a ~ Normal(1, 2) and b ~ Exponential(3), independent.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const N = 20000;
const SRC = `
s = rnginit(7)
M = lawof(record(a = draw(Normal(1.0, 2.0)), b = draw(Exponential(3.0))))
r, s2 = rand(iid(M, ${N}), s)
`;

function columnsOf(m: any): { a: number[], b: number[] } {
  // A table measure carries one field measure per column.
  const col = (name: string) => Array.from(m.fields[name].value.data) as number[];
  return { a: col('a'), b: col('b') };
}

function checkMoments({ a, b }: { a: number[], b: number[] }) {
  assert.equal(a.length, N);
  assert.equal(b.length, N);
  const mean = (x: number[]) => x.reduce((s, v) => s + v, 0) / x.length;
  const ma = mean(a), mb = mean(b);
  const va = mean(a.map(v => (v - ma) ** 2)), vb = mean(b.map(v => (v - mb) ** 2));
  const cov = mean(a.map((v, i) => (v - ma) * (b[i] - mb)));
  // Standard errors at N = 20000: mean(a) 0.014, var(a) 0.040, mean(b) 0.0024,
  // var(b) 0.0021; five of them each.
  assert.ok(Math.abs(ma - 1) < 0.07, `mean a ${ma}`);
  assert.ok(Math.abs(va - 4) < 0.2, `var a ${va}`);
  assert.ok(Math.abs(mb - 1 / 3) < 0.012, `mean b ${mb}`);
  assert.ok(Math.abs(vb - 1 / 9) < 0.011, `var b ${vb}`);
  // Independent fields: correlation within five standard errors of 0.
  const corr = cov / Math.sqrt(va * vb);
  assert.ok(Math.abs(corr) < 5 / Math.sqrt(N), `corr ${corr}`);
}

test('rand of an iid of a record measure is a table with the fields\' laws', async () => {
  const { ctx } = makeMatCtx(SRC, { sampleCount: 4 });
  checkMoments(columnsOf(await ctx.getMeasure('r')));
});

test('the materialiser\'s composite draw builds the same table', async () => {
  // Force the derivation path by hiding the fixed value.
  const { ctx } = makeMatCtx(SRC, { sampleCount: 4 });
  const has = ctx.fixedValues.has.bind(ctx.fixedValues);
  ctx.fixedValues.has = (name: string) => name !== 'r' && has(name);
  checkMoments(columnsOf(await ctx.getMeasure('r')));
});

test('the table reads by column and row, and scores under its law', async () => {
  const src = `
s = rnginit(7)
M = lawof(record(a = draw(Normal(1.0, 2.0)), b = draw(Exponential(3.0))))
r, s2 = rand(iid(M, 4), s)
a2 = r.a[2]
row = r[2]
lp = logdensityof(iid(M, 4), r)
`;
  const { ctx, built } = makeMatCtx(src, { sampleCount: 1 });
  const r = built.fixedValues.get('r');
  assert.equal(r.__table__, true);
  assert.equal(r.nrows, 4);
  assert.equal(built.fixedValues.get('a2'), r.columns.a[1]);
  assert.deepEqual(built.fixedValues.get('row'), { a: r.columns.a[1], b: r.columns.b[1] });
  // Closed form: Σ log N(a_i; 1, 2) + Σ log Exp(b_i; 3).
  let expect = 0;
  for (let i = 0; i < 4; i++) {
    expect += -0.5 * Math.log(2 * Math.PI) - Math.log(2) - 0.5 * ((r.columns.a[i] - 1) / 2) ** 2;
    expect += Math.log(3) - 3 * r.columns.b[i];
  }
  const lp = (await ctx.getMeasure('lp')).samples[0];
  assert.ok(Math.abs(lp - expect) < 1e-12, `${lp} vs ${expect}`);
});
