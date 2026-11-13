'use strict';
// §04 "Calling conventions": a callable with two or more required inputs,
// called with a lone record, receives the record's fields as keyword
// arguments. `weighted(f, M)`, `logweighted(f, M)` and `pushfwd(f, M)` call
// `f(x)` with the variate x (§04 "Explicit callback parameters"), so a record
// variate splats into a several-input f. Found through amplitude models whose
// `rate_fn = functionof(rate, a, b, c, d, e)` weights Lebesgue on
// `cartprod(a = …, …)`; they were refused with "callback receives 1 whole
// inputs, but declares 5".
//
// Oracles, closed form, with x ~ U(0, 1) and y ~ U(0, 2) on the box:
//   f = x·y has total mass ∫x dx · ∫y dy = 1/2 · 2 = 1, and f(0.5, 1.5) = 0.75.
//   Under normalize(f · Lebesgue) the density factorises as 2x · y/2, so
//   E[x] = 2/3, E[y] = 4/3, Var x = 1/18, Var y = 2/9, and x + y has mean 2
//   and variance 5/18.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('..');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const N = 20000;
const BOX = 'x = elementof(interval(0.0, 1.0))\ny = elementof(interval(0.0, 2.0))\n'
  + 'S = cartprod(x = interval(0.0, 1.0), y = interval(0.0, 2.0))\nB = Lebesgue(support = S)\n';
const NAMED = BOX + 'xy = x * y\nf = functionof(xy, x, y)\ns = x + y\ng = functionof(s, x, y)\n';

const errors = (src: string) => (processSource(src).diagnostics || [])
  .filter((d: any) => d.severity === 'error').map((d: any) => d.message);

async function scalar(src: string, name: string, n = 1) {
  const { ctx } = makeMatCtx(src, { sampleCount: n, rootSeed: 612 });
  const m = await ctx.getMeasure(name);
  return (m.value ? m.value.data : m.samples)[0];
}

// The weighted mean of a per-atom statistic and its standard error from the
// effective sample size of the atom weights.
function weightedMean(values: ArrayLike<number>, logWeights: ArrayLike<number> | null) {
  const n = values.length;
  const w = new Float64Array(n);
  let max = -Infinity;
  for (let i = 0; i < n; i++) max = Math.max(max, logWeights ? logWeights[i] : 0);
  let W = 0, W2 = 0, s = 0;
  for (let i = 0; i < n; i++) {
    w[i] = Math.exp((logWeights ? logWeights[i] : 0) - max);
    W += w[i]; W2 += w[i] * w[i]; s += w[i] * values[i];
  }
  return { mean: s / W, ess: (W * W) / W2 };
}

test('a record variate splats into a several-input weight or pushforward callback', () => {
  for (const src of [
    NAMED + 'M = weighted(f, B)\nQ = pushfwd(g, normalize(M))\n',
    NAMED + 'M = logweighted(f, B)\n',
    BOX + 'M = weighted((x, y) -> x * y, B)\nQ = pushfwd((x, y) -> x + y, normalize(M))\n',
  ]) assert.deepEqual(errors(src), [], src);
});

test('a one-input callback still receives the record whole', () => {
  assert.deepEqual(errors(BOX + 'M = weighted(r -> r.x * r.y, B)\n'), []);
});

test('a field the callback does not declare is a static error', () => {
  const errs = errors(BOX + 'M = weighted((x, z) -> x * z, B)\n');
  assert.ok(errs.some((m: string) => /weighted callback.*no argument named "y"/.test(m)), errs.join('; '));
});

test('a scalar variate is still refused for a several-input callback', () => {
  for (const op of ['weighted', 'pushfwd']) {
    const errs = errors(`M = Uniform(interval(0.0, 1.0))\nm = ${op}((x, y) -> x * y, M)\n`);
    assert.ok(errs.some((m: string) => new RegExp(`${op}: callback receives 1 whole inputs, but declares 2`).test(m)), errs.join('; '));
  }
});

test('the splatted weight scores f at the point, for named and inline callbacks', async () => {
  for (const weight of ['f', '(x, y) -> x * y']) {
    const lp = await scalar(BOX + 'xy = x * y\nf = functionof(xy, x, y)\n'
      + `M = weighted(${weight}, B)\nlp = logdensityof(M, record(x = 0.5, y = 1.5))\n`, 'lp');
    assert.ok(Math.abs(lp - Math.log(0.75)) < 1e-12, `${weight}: ${lp}`);
  }
  const llp = await scalar(NAMED + 'M = logweighted(f, B)\nlp = logdensityof(M, record(x = 0.5, y = 1.5))\n', 'lp');
  assert.ok(Math.abs(llp - 0.75) < 1e-12, `logweighted: ${llp}`);
});

test('the splatted weight has total mass 1', async () => {
  // Uniform proposal on the box of area 2: Var(2·xy) = 4·(4/9 − 1/4) = 7/9.
  const tm = await scalar(NAMED + 'M = weighted(f, B)\ntm = totalmass(M)\n', 'tm', N);
  assert.ok(Math.abs(tm - 1) < 6 * Math.sqrt(7 / 9 / N), `${tm}`);
});

test('the normalized measure and its pushforward have the closed-form moments', async () => {
  const { ctx } = makeMatCtx(NAMED + 'M = weighted(f, B)\nP = normalize(M)\nQ = pushfwd(g, P)\n',
    { sampleCount: N, rootSeed: 612 });
  const p = await ctx.getMeasure('P');
  assert.deepEqual(Object.keys(p.fields).sort(), ['x', 'y']);
  const ex = weightedMean(p.fields.x.samples, p.logWeights);
  const ey = weightedMean(p.fields.y.samples, p.logWeights);
  assert.ok(Math.abs(ex.mean - 2 / 3) < 6 * Math.sqrt(1 / 18 / ex.ess), `E[x] ${ex.mean}`);
  assert.ok(Math.abs(ey.mean - 4 / 3) < 6 * Math.sqrt(2 / 9 / ey.ess), `E[y] ${ey.mean}`);
  const q = await ctx.getMeasure('Q');
  const eq = weightedMean(q.samples || q.value.data, q.logWeights);
  assert.ok(Math.abs(eq.mean - 2) < 6 * Math.sqrt(5 / 18 / eq.ess), `E[x + y] ${eq.mean}`);
});
