'use strict';
// G1 sampling: iid over a record measure must produce a TABLE the density path
// scores — sampling and density stay in agreement (measure-algebra-audit scar
// zone). The round-trip below samples the table, then scores those SAME draws
// through the density path and checks the score equals an independent hand
// closed-form normal-logpdf sum over the drawn values (never engine-vs-engine).
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

// Independent reference: log N(μ,σ) pdf = −½ln(2π) − ln σ − ½((v−μ)/σ)².
function normLogpdf(v: number, mu: number, sigma: number): number {
  const z = (v - mu) / sigma;
  return -0.5 * Math.log(2 * Math.PI) - Math.log(sigma) - 0.5 * z * z;
}
function mean(a: any): number { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; }
// Full-precision float literal so the rebuilt table is bit-identical to the draw.
function lit(v: number): string { return Number.isInteger(v) ? v.toFixed(1) : v.toString(); }

async function sampleTable(src: string, name: string, N: number): Promise<any> {
  const { ctx } = ctxFor('flatppl_compat = "0.1"\n' + src, N);
  return ctx.getMeasure(name);
}
async function score(src: string, name: string): Promise<number> {
  const { ctx } = ctxFor('flatppl_compat = "0.1"\n' + src, 1);
  const mm = await ctx.getMeasure(name);
  return mm.value ? mm.value.data[0] : mm.samples[0];
}

test('#231/G1: iid over a record measure samples a table (shape + moments)', async () => {
  const s = await sampleTable(`
g = joint(x = Normal(mu = 0.0, sigma = 1.0), y = Normal(mu = 5.0, sigma = 1.0))
s = iid(g, 64)
`, 's', 1);
  assert.equal(s.__table__, true);
  assert.equal(s.nrows, 64);
  assert.deepEqual(Object.keys(s.columns).sort(), ['x', 'y']);
  const xs = s.columns.x.data, ys = s.columns.y.data;
  assert.equal(xs.length, 64);
  // 64 draws, σ/√64 = 0.125; a 6σ band is 0.75.
  assert.ok(Math.abs(mean(xs) - 0.0) < 0.75, `mean x ${mean(xs)}`);
  assert.ok(Math.abs(mean(ys) - 5.0) < 0.75, `mean y ${mean(ys)}`);
});

test('#231/G1: sampled table round-trips through the density path (== hand closed form)', async () => {
  // Sample a small table, then score those SAME draws via the density path and
  // compare to the independent normal-logpdf sum over the drawn values.
  const s = await sampleTable(`
g = joint(x = Normal(mu = 0.0, sigma = 1.0), y = Normal(mu = 5.0, sigma = 1.0))
s = iid(g, 6)
`, 's', 1);
  const xs: number[] = Array.from(s.columns.x.data), ys: number[] = Array.from(s.columns.y.data);
  const oracle = xs.reduce((a: number, v: number) => a + normLogpdf(v, 0, 1), 0)
    + ys.reduce((a: number, v: number) => a + normLogpdf(v, 5, 1), 0);
  const xLits = xs.map(lit).join(', '), yLits = ys.map(lit).join(', ');
  const got = await score(`
g = joint(x = Normal(mu = 0.0, sigma = 1.0), y = Normal(mu = 5.0, sigma = 1.0))
data = table(x = [${xLits}], y = [${yLits}])
d6 = iid(g, 6)
__score__ = logdensityof(d6, data)
`, '__score__');
  assert.ok(Math.abs(got - oracle) < 1e-9,
    `density of sampled table ${got} != hand closed form ${oracle}`);
});

test('#231/G1: a VECTOR-valued record field samples with the right column shape + round-trips', async () => {
  // A field whose per-row value is itself a vector (here v ~ iid(Normal,2)) must
  // carry its inner element dims in the sampled column ([k, 2]) — a scalar-shaped
  // [k] column silently corrupts it (the density path reads v as a 2-vector).
  const s = await sampleTable(`
g = joint(v = iid(Normal(mu = 0.0, sigma = 1.0), 2), y = Normal(mu = 5.0, sigma = 1.0))
s = iid(g, 4)
`, 's', 1);
  assert.deepEqual(s.columns.v.shape, [4, 2]);
  assert.equal(s.columns.v.data.length, 8);
  assert.deepEqual(s.columns.y.shape, [4]);
  // Round-trip: score those SAME draws via the density path == hand closed form.
  const v: number[] = Array.from(s.columns.v.data), ys: number[] = Array.from(s.columns.y.data);
  const oracle = v.reduce((a: number, x: number) => a + normLogpdf(x, 0, 1), 0)
    + ys.reduce((a: number, x: number) => a + normLogpdf(x, 5, 1), 0);
  const rows: string[] = [];
  for (let i = 0; i < 4; i++) rows.push(`[${lit(v[2 * i])}, ${lit(v[2 * i + 1])}]`);
  const got = await score(`
g = joint(v = iid(Normal(mu = 0.0, sigma = 1.0), 2), y = Normal(mu = 5.0, sigma = 1.0))
data = table(v = [${rows.join(', ')}], y = [${ys.map(lit).join(', ')}])
d4 = iid(g, 4)
__score__ = logdensityof(d4, data)
`, '__score__');
  assert.ok(Math.abs(got - oracle) < 1e-9, `vector-field density ${got} != ${oracle}`);
});

test('#231/G1: sampling record-iid at >1 atoms (ensemble of tables) refuses loudly', async () => {
  // An N-atom ensemble of k-row tables has no single {__table__,columns,nrows}
  // representation; refuse rather than emit a table lying about its row count.
  await assert.rejects(async () => sampleTable(`
g = joint(x = Normal(mu = 0.0, sigma = 1.0), y = Normal(mu = 5.0, sigma = 1.0))
s = iid(g, 6)
`, 's', 4), /ensemble of tables|>1 atoms/);
});
