'use strict';
// iid over a record law that §06 `normalize` holds in importance weights
// samples one k-row table by sampling importance resampling: a pool of
// weighted rows grows until its effective sample size reaches ten per row,
// then k rows are picked in proportion to their weights.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

// The weight x·y on the unit box: x and y are independent, each with density
// 2t on [0, 1], so mean 2/3 and variance 1/2 − 4/9 = 1/18.
const SOURCE = (n: number) => `
P = normalize(weighted((x, y) -> x * y, Lebesgue(cartprod(x = interval(0.0, 1.0), y = interval(0.0, 1.0)))))
T ~ iid(P, ${n})
`;

function moments(xs: ArrayLike<number>) {
  let mean = 0, variance = 0;
  for (let i = 0; i < xs.length; i++) mean += xs[i] / xs.length;
  for (let i = 0; i < xs.length; i++) variance += (xs[i] - mean) ** 2 / xs.length;
  return { mean, variance };
}

async function table(n: number, opts: any = {}) {
  const { ctx } = makeMatCtx(SOURCE(n), { sampleCount: 1, rootSeed: 2026 });
  Object.assign(ctx, opts);
  return ctx.getMeasure('T');
}

test('the rows of an iid table over a weighted record law have the normalized moments', async () => {
  const n = 2000;
  const t = await table(n);
  assert.equal(t.nrows, n);
  assert.ok(t.sir.ess >= 10 * n, `pool ESS ${t.sir.ess}`);
  assert.equal(t.n_eff, t.sir.ess);
  for (const field of ['x', 'y']) {
    const { mean, variance } = moments(t.columns[field].data);
    assert.ok(Math.abs(mean - 2 / 3) < 5 * Math.sqrt(1 / 18 / n), `${field} mean ${mean}`);
    assert.ok(Math.abs(variance - 1 / 18) < 5 * Math.sqrt(2 / n) / 18, `${field} variance ${variance}`);
  }
  // x and y are independent under the normalized law.
  const x = t.columns.x.data, y = t.columns.y.data;
  let cov = 0;
  for (let i = 0; i < n; i++) cov += (x[i] - 2 / 3) * (y[i] - 2 / 3) / n;
  assert.ok(Math.abs(cov) < 5 / 18 / Math.sqrt(n), `cov ${cov}`);
});

test('few rows repeat when the pool ESS is ten per row', async () => {
  // A pick repeats an earlier one with probability at most i/ESS, so the
  // expected repeated fraction is at most n/(2 ESS) ≤ 1/20.
  const n = 500;
  const t = await table(n);
  const seen = new Set<string>();
  let repeated = 0;
  for (let i = 0; i < n; i++) {
    const key = t.columns.x.data[i] + ',' + t.columns.y.data[i];
    if (seen.has(key)) repeated++;
    seen.add(key);
  }
  assert.ok(repeated / n <= n / (2 * t.sir.ess) + 0.03, `repeated ${repeated} of ${n}, ESS ${t.sir.ess}`);
});

test('a pool stopped at its row cap warns with its ESS and still samples', async () => {
  const n = 200;
  const warnings: string[] = [];
  const original = console.warn;
  console.warn = (msg: string) => { warnings.push(String(msg)); };
  let t: any;
  try {
    t = await table(n, { iidSirMaxRows: 2 * n });
  } finally {
    console.warn = original;
  }
  assert.equal(t.nrows, n);
  assert.ok(t.sir.poolRows <= 2 * n, `pool ${t.sir.poolRows}`);
  assert.ok(t.n_eff < 10 * n);
  const w = warnings.find((m) => m.includes('of "P"'));
  assert.ok(w, 'no warning: ' + JSON.stringify(warnings));
  assert.ok(w!.includes('effective sample size') && w!.includes(`resamples ${n} rows`)
    && w!.includes(`K = ${t.sir.K}`), w);
});

test('a pool grown by appended chunks keeps the normalized moments', async () => {
  // exp(20x) on the unit box: the pool's ESS is about a tenth of its rows,
  // so it grows over several appended chunks. Under the normalized law x has
  // mean 1/(1 − e⁻²⁰) − 1/20 and variance 1/400 − e²⁰/(e²⁰ − 1)², and y
  // stays uniform.
  const n = 1000;
  const src = `
P = normalize(weighted((x, y) -> exp(20.0 * x), Lebesgue(cartprod(x = interval(0.0, 1.0), y = interval(0.0, 1.0)))))
T ~ iid(P, ${n})
`;
  const { ctx } = makeMatCtx(src, { sampleCount: 1, rootSeed: 2027 });
  const t = await ctx.getMeasure('T');
  assert.ok(t.sir.K >= 8, `K ${t.sir.K}`);
  assert.ok(t.sir.ess >= 10 * n, `ESS ${t.sir.ess}`);
  const a = 20;
  const mx = 1 / (1 - Math.exp(-a)) - 1 / a;
  const vx = 1 / (a * a) - Math.exp(a) / (Math.exp(a) - 1) ** 2;
  const x = moments(t.columns.x.data), y = moments(t.columns.y.data);
  assert.ok(Math.abs(x.mean - mx) < 5 * Math.sqrt(vx / n), `x mean ${x.mean}, oracle ${mx}`);
  assert.ok(Math.abs(x.variance - vx) < 5 * vx * Math.sqrt(8 / n), `x variance ${x.variance}, oracle ${vx}`);
  assert.ok(Math.abs(y.mean - 0.5) < 5 * Math.sqrt(1 / 12 / n), `y mean ${y.mean}`);
  assert.ok(Math.abs(y.variance - 1 / 12) < 5 * Math.sqrt(2 / n) / 12, `y variance ${y.variance}`);
});

test('an ensemble of record tables is refused before any row is drawn', async () => {
  // At a viewer-sized atom count the k rows per atom would be materialised
  // before the refusal; it now comes first.
  const { ctx } = makeMatCtx(SOURCE(100000), { sampleCount: 50000, rootSeed: 2026 });
  let calls = 0;
  const send = ctx.sendWorker;
  ctx.sendWorker = (msg: any) => { calls++; return send.call(ctx, msg); };
  const t0 = Date.now();
  await assert.rejects(() => ctx.getMeasure('T'), /an ensemble of tables/);
  assert.equal(calls, 0);
  assert.ok(Date.now() - t0 < 2000, `${Date.now() - t0} ms`);
});

test('a pushforward of a weighted record law resamples by the raw weights under it', async () => {
  // The law's weights are the normalize's, carried row for row through the
  // map. s = x + y with x, y independent of density 2t: mean 4/3, var 1/9.
  const n = 2000;
  const src = `
P = normalize(weighted((x, y) -> x * y, Lebesgue(cartprod(x = interval(0.0, 1.0), y = interval(0.0, 1.0)))))
Q = pushfwd(r -> record(s = r.x + r.y, d = r.x - r.y), P)
T ~ iid(Q, ${n})
`;
  const { ctx } = makeMatCtx(src, { sampleCount: 1, rootSeed: 2028 });
  const t = await ctx.getMeasure('T');
  assert.ok(t.sir.ess >= 10 * n, `ESS ${t.sir.ess}`);
  const s = moments(t.columns.s.data), dd = moments(t.columns.d.data);
  assert.ok(Math.abs(s.mean - 4 / 3) < 5 * Math.sqrt(1 / 9 / n), `s mean ${s.mean}`);
  assert.ok(Math.abs(s.variance - 1 / 9) < 5 * Math.sqrt(2 / n) / 9, `s variance ${s.variance}`);
  assert.ok(Math.abs(dd.mean) < 5 * Math.sqrt(1 / 9 / n), `d mean ${dd.mean}`);
});
