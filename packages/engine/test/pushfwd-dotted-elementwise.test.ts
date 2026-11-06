'use strict';

// §06 case 1: a known bijection needs no annotation, and §04 makes a dotted
// operator or f.(x) a broadcast, so a broadcast of a scalar bijection inverts
// cell by cell. The values are from scipy 1.18.0 (stats.norm(1, 2),
// stats.norm(0, 1)): log p_Y(y) = Σ log p_X(f⁻¹(y)_i) − log|det J|.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const BASE = 'iid(Normal(1.0, 2.0), 3)';
const COEFFS = 'a = [1.0, 2.0, -3.0]\nb = [0.5, 0.0, 1.0]\n';

async function score(src: string) {
  const { proc, ctx } = ctxFor(COEFFS + src, 1);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message), []);
  const lp = await ctx.getMeasure('lp');
  return lp.samples[0];
}

function close(got: number, want: number) {
  assert.ok(Math.abs(got - want) < 1e-12 * Math.max(1, Math.abs(want)), `${got} != ${want}`);
}

const CASES: Array<[string, string, string, number]> = [
  // Σ log N(y_i / a_i) − Σ log|a_i|
  ['a vector scale', 'fn(_ .* a)', '[3.0, 1.0, -2.0]', -7.173155499410798],
  ['a vector scale on the left', 'fn(a .* _)', '[3.0, 1.0, -2.0]', -7.173155499410798],
  ['a vector divisor', 'fn(_ ./ a)', '[3.0, 1.0, -2.0]', -6.794497672065798],
  ['a vector shift', 'fn(_ .+ a)', '[3.0, 1.0, -2.0]', -5.461257141293855],
  ['an elementwise affine map', 'fn(_ .* a .+ b)', '[3.0, 1.0, -2.0]', -6.940516610521909],
  // Σ (log N(log y_i) − log y_i)
  ['a dotted exp', 'fn(exp.(_))', '[3.0, 1.0, 2.0]', -6.766001990056243],
  ['a dotted exp in a lambda', 'x -> exp.(x)', '[3.0, 1.0, 2.0]', -6.766001990056243],
  // A scalar op after a broadcast contributes its log|2| once per cell.
  ['a scalar scale over a dotted scale', 'fn(2.0 * (_ .* a))', '[3.0, 1.0, -2.0]', -8.8645762077573],
];

for (const [name, f, y, want] of CASES) {
  test(`pushfwd scores ${name} over an iid base without an annotation`, async () => {
    close(await score(`lp = logdensityof(pushfwd(${f}, ${BASE}), ${y})\n`), want);
  });
}

test('a dotted log over a positive iid base scores its cells', async () => {
  // log.(x) of LogNormal(0, 1) cells is Normal(0, 1): Σ log N(y_i).
  close(await score('lp = logdensityof(pushfwd(fn(log.(_)), iid(LogNormal(0.0, 1.0), 3)), [0.3, -1.0, 2.0])\n'),
    -5.301815599614018);
});

test('a dotted log over a real iid base is a static domain error', () => {
  const { proc } = ctxFor(`lp = logdensityof(pushfwd(fn(log.(_)), ${BASE}), [3.0, 1.0, 2.0])\n`, 1);
  const errors = proc.diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message);
  assert.equal(errors.length, 1);
  assert.match(errors[0], /pushfwd: forward function uses 'log'/);
});

test('a dotted record field of an iid ancestor scores as a pushforward', async () => {
  close(await score(`x ~ ${BASE}\ny = exp.(x)\nlp = logdensityof(lawof(record(y = y)), record(y = [3.0, 1.0, 2.0]))\n`),
    -6.766001990056243);
});

test('a dotted log record field refuses an iid ancestor with real support', async () => {
  const { ctx } = ctxFor(`x ~ ${BASE}\ny = log.(x)\nlp = logdensityof(lawof(record(y = y)), record(y = [3.0, 1.0, 2.0]))\n`, 1);
  await assert.rejects(ctx.getMeasure('lp'), /record field 'y' uses 'log'/);
});

test('a dotted map over a scalar base is refused with the undotted spelling', async () => {
  const { ctx } = ctxFor('lp = logdensityof(pushfwd(fn(_ .* 2.0), Normal(1.0, 2.0)), 3.0)\n', 1);
  await assert.rejects(ctx.getMeasure('lp'), /needs an undotted forward map/);
});
