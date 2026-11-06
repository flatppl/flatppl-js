'use strict';

// §06 logweighted takes a function of the variate. Its body may call the §07
// higher-order operations scan, reduce and filter, whose first argument is a
// callable, not a value the derivation must resolve.
//
// ORACLE: log N(a; 0, 1) plus the closed-form callback value at a = 0.3.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const A = 0.3;
const logN = (x: number) => -0.5 * Math.log(2 * Math.PI) - 0.5 * x * x;

async function value(body: string, N = 1): Promise<number[]> {
  const src = `a ~ Normal(0, 1)\n${body}\nP = logweighted(log_target, lawof(record(a = a)))\n`
    + `lp = logdensityof(P, record(a = ${A}))\n`;
  const { ctx } = ctxFor(src, N);
  const m = await ctx.getMeasure('lp');
  return Array.from(m.value ? m.value.data : m.samples);
}

function near(got: number[], want: number) {
  assert.ok(got.length > 0);
  got.forEach((g) => assert.ok(Math.abs(g - want) < 1e-12, `got ${g}, want ${want}`));
}

test('scan in a logweighted callback', async () => {
  // scan(step, 0, [1, 2], a)[2] = 1 a + 2 a.
  const body = 'step(acc, v, p) = acc + v * p\nlog_target(x) = scan(step, 0.0, [1.0, 2.0], x.a)[2]';
  near(await value(body), logN(A) + 3 * A);
  near(await value(body, 3), logN(A) + 3 * A);
});

test('reduce in a logweighted callback', async () => {
  near(await value('add2(acc, v) = acc + v\nlog_target(x) = reduce(add2, [1.0, 2.0] .* x.a)'), logN(A) + 3 * A);
});

test('filter in a logweighted callback', async () => {
  near(await value('pos(v) = v > 1.0\nlog_target(x) = sum(filter(pos, [1.0, 2.0, 3.0])) * x.a'), logN(A) + 5 * A);
});

test('a user function called and indexed inside an aggregate body', async () => {
  // shifted(X) sums exp(X[t, k] - max_k X[t, k]) with X = a [[1, 2], [3, 0.5]].
  const body = [
    'row_max(X) = aggregate(maximum, [.t], X[.t, .k])',
    'shifted(X) = aggregate(sum, [.t], exp(X[.t, .k] - row_max(X)[.t]))',
    'log_target(x) = sum(shifted(x.a .* rowstack([[1.0, 2.0], [3.0, 0.5]])))',
  ].join('\n');
  const X = [[A, 2 * A], [3 * A, 0.5 * A]];
  const want = X.reduce((s, r) => s + r.reduce((t, v) => t + Math.exp(v - Math.max(...r)), 0), 0);
  near(await value(body), logN(A) + want);
});

test('a user function applied to aggregate cells', async () => {
  // sum over t, k of (y[t] - a m[k])^2 with y = [1, 2], m = [1, 3].
  const body = 'sq(y, m) = (y - m)^2\n'
    + 'log_target(x) = sum(aggregate(sum, [.t, .k], sq([1.0, 2.0][.t], (x.a .* [1.0, 3.0])[.k])))';
  const want = [1, 2].reduce((s, y) => s + [1, 3].reduce((t, m) => t + (y - A * m) ** 2, 0), 0);
  near(await value(body), logN(A) + want);
});

test('ifelse and comparisons in an aggregate body', async () => {
  // Count of pairs with v[i] >= w[s], for v = [1, 2, 3] and w = [1.5, 2.5]: 3.
  const body = 'log_target(x) = x.a * sum(aggregate(sum, [.i, .s], '
    + 'ifelse([1.0, 2.0, 3.0][.i] >= [1.5, 2.5][.s], 1.0, 0.0)))';
  near(await value(body), logN(A) + 3 * A);
});
