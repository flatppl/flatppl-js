'use strict';

// Spec §07 gives every head of collection-domain-heads.ts a collection as its first argument
// (`sum`: "real/complex arrays"). The static gates refuse a record or scalar whose type is known,
// but an input of unknown type (`k(aa) = ...`, `fn(...)`) hides it, and the evaluator then folded the
// argument over nothing: `sum` gave 0, `prod` 1, `lany` false, `lall` true, `mean`, `var`, `maximum`
// and `logsumexp` null, the norms 0, the scans and `transpose` an empty array. A labelled
// `jointchain(aa = ..., bb = k)` with `k(aa) = Normal(sum(aa), 1.0)` therefore scored and sampled as if
// `sum(record(aa = x))` were 0. The evaluator now refuses it, located and tagged `modelRefusal`, on the
// sampler and the density paths, for the whole table.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');
const sampler = require('../sampler.ts');

const logN = (x: number, m: number, s = 1) => -0.5 * Math.log(2 * Math.PI) - Math.log(s) - 0.5 * ((x - m) / s) ** 2;

async function value(src: string, name: string, n = 1): Promise<any> {
  const { ctx } = ctxFor(src, n);
  return ctx.getMeasure(name);
}
const scalarOf = (m: any) => (m.value ? m.value.data : m.samples)[0];
const errors = (src: string) => processSource(src).diagnostics.filter((d: any) => d.severity === 'error');

// A one-input step whose input has no declared type, so nothing static sees the record it is fed.
// `body` turns the head's result into the step's mean.
const BODIES: Array<[string, string]> = [
  ['sum', 'sum(aa)'], ['prod', 'prod(aa)'], ['mean', 'mean(aa)'], ['var', 'var(aa)'], ['std', 'std(aa)'],
  ['maximum', 'maximum(aa)'], ['minimum', 'minimum(aa)'], ['median', 'median(aa)'],
  ['lengthof', 'lengthof(aa)'], ['l1norm', 'l1norm(aa)'], ['l2norm', 'l2norm(aa)'],
  ['linfnorm', 'linfnorm(aa)'], ['logsumexp', 'logsumexp(aa)'],
  ['lany', 'ifelse(lany(aa), 1.0, 0.0)'], ['lall', 'ifelse(lall(aa), 1.0, 0.0)'],
  ['cumsum', 'sum(cumsum(aa))'], ['transpose', 'sum(transpose(aa))'], ['sizeof', 'sum(sizeof(aa))'],
];
const chain = (body: string) => `k(aa) = Normal(${body}, 1.0)\nA = jointchain(aa = Normal(0.0, 1.0), bb = k)\n`;

for (const [head, body] of BODIES) {
  test(`${head} of a record fed to an untyped step input is refused, scored and sampled`, async () => {
    const want = new RegExp(`${head} at \\d+:\\d+: spec §07 ".+" gives \`${head}\` the domain .*argument 1 is a record with fields aa`);
    const src = chain(body);
    assert.deepEqual(errors(src), [], 'the static check cannot see it');
    await assert.rejects(value(src + 'lp = logdensityof(A, record(aa = 0.5, bb = 0.5))\n', 'lp'), want);
    await assert.rejects(value(src + 'y ~ A\n', 'y', 4), want);
  });
}

test('a scalar fed to an untyped step input is refused the same way', async () => {
  const src = 'k(a) = Normal(sum(a), 1.0)\nA = jointchain(Normal(0.0, 1.0), k)\n';
  assert.deepEqual(errors(src), []);
  const want = /sum at \d+:\d+: spec §07 "Reductions" gives `sum` the domain real\/complex arrays, but argument 1 is a scalar/;
  await assert.rejects(value(src + 'lp = logdensityof(A, [0.5, 0.5])\n', 'lp'), want);
  await assert.rejects(value(src + 'y ~ A\n', 'y', 4), want);
});

test('the refusal is tagged modelRefusal and located at the call, for the evaluator itself', () => {
  const rec = { kind: 'call', op: 'record', fields: [{ name: 'a', value: { kind: 'lit', value: 1.5 } }] };
  const loc = { start: { line: 4, col: 2 }, end: { line: 4, col: 12 } };
  for (const op of ['sum', 'prod', 'lany', 'lall', 'mean', 'logsumexp', 'cumsum', 'transpose', 'lengthof']) {
    let caught: any = null;
    try { sampler.evaluateExpr({ kind: 'call', op, args: [rec], loc }, {}); } catch (e) { caught = e; }
    assert.ok(caught, op);
    assert.equal(caught.modelRefusal, true, op);
    assert.match(caught.message, new RegExp(`^${op} at 5:3: spec §07`), op);
  }
});

// The valid array cases keep their hand-computed values.
test('arrays still reduce to their hand-computed values', async () => {
  const cases: Array<[string, number]> = [
    ['sum([1.0, 2.0, 3.0])', 6], ['prod([1.0, 2.0, 3.0, 4.0])', 24], ['mean([1.0, 2.0, 3.0])', 2],
    ['maximum([1.0, 3.0, 2.0])', 3], ['minimum([4.0, 3.0, 5.0])', 3], ['median([5.0, 1.0, 3.0])', 3],
    ['var([1.0, 2.0, 3.0, 4.0])', 5 / 3], ['l1norm([3.0, -4.0])', 7], ['l2norm([3.0, 4.0])', 5],
    ['linfnorm([3.0, -4.0])', 4], ['logsumexp([0.0, 0.0])', Math.LN2], ['lengthof([1.0, 2.0, 3.0])', 3],
    ['sum(cumsum([1.0, 2.0, 3.0]))', 10], ['ifelse(lany([false, true]), 1.0, 0.0)', 1],
    ['ifelse(lall([false, true]), 1.0, 0.0)', 0], ['sum(transpose([1.0, 2.0, 3.0]))', 6],
  ];
  for (const [expr, want] of cases) {
    const got = scalarOf(await value(`r = ${expr}\n`, 'r'));
    assert.ok(Math.abs(got - want) < 1e-12, `${expr}: ${got} vs ${want}`);
  }
  // A per-cell nested reading is still legal: `sum` of each inner vector.
  const nested = await value('vv = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]\nr = fn(sum(_) + 1.0).(vv)\n', 'r');
  assert.deepEqual(Array.from(nested.value.data), [7, 16]);
  // And a step that does take the record reads its field: aa ~ N(0, 1), bb | aa ~ N(aa, 1).
  const ok = 'k(r) = Normal(r.aa, 1.0)\nA = jointchain(aa = Normal(0.0, 1.0), bb = k)\n'
    + 'lp = logdensityof(A, record(aa = 0.5, bb = 0.25))\n';
  assert.ok(Math.abs(scalarOf(await value(ok, 'lp')) - (logN(0.5, 0) + logN(0.25, 0.5))) < 1e-12);
});

// ---- rank-0 Value scalars and the further collection slots ---------------------------------

const valueLib = require('../value.ts');
const ev = (op: string, args: any[], env: any = {}) => sampler.evaluateExpr({ kind: 'call', op, args }, env);
const lit = (value: any) => ({ kind: 'lit', value });
const ref = (name: string) => ({ kind: 'ref', ns: 'self', name });
const vec = (...xs: number[]) => ({ kind: 'call', op: 'vector', args: xs.map(lit) });
const mat = { kind: 'call', op: 'rowstack', args: [{ kind: 'call', op: 'vector', args: [vec(2, 0), vec(0, 3)] }] };
const rec = { kind: 'call', op: 'record', fields: [{ name: 'a', value: lit(1.5) }] };

test('a rank-0 Value scalar is refused exactly as a bare scalar is', () => {
  const zero = valueLib.asValue(2.5);
  assert.deepEqual(zero.shape, [], 'a rank-0 Value');
  for (const op of ['sum', 'prod', 'mean', 'l2norm', 'lany', 'cumsum', 'transpose', 'lengthof']) {
    assert.throws(() => ev(op, [ref('s')], { s: zero }),
      (e: any) => e.modelRefusal === true && new RegExp(`^${op}: spec §07 .*argument 1 is a scalar`).test(e.message), op);
  }
});

test('every §07 collection slot is judged, not only the first', () => {
  // quadform: "square A, vector x"; linsolve: "square A, vector b"; cross: "vectors".
  const cases: Array<[string, any[], number, string]> = [
    ['quadform', [mat, lit(2.5)], 2, 'a scalar'],
    ['quadform', [mat, rec], 2, 'a record with fields a'],
    ['linsolve', [mat, lit(2.5)], 2, 'a scalar'],
    ['linsolve', [mat, rec], 2, 'a record with fields a'],
    ['cross', [vec(1, 0, 0), lit(2.5)], 2, 'a scalar'],
    ['cross', [vec(1, 0, 0), rec], 2, 'a record with fields a'],
    ['quadform', [lit(2.5), vec(1, 2)], 1, 'a scalar'],
  ];
  for (const [op, args, argument, what] of cases) {
    assert.throws(() => ev(op, args), (e: any) => e.modelRefusal === true
      && new RegExp(`^${op}: spec §07 .*argument ${argument} is ${what}`).test(e.message), `${op} ${what}`);
  }
  // The refusal also reaches the second slot through an untyped step input, scored and sampled.
  const src = 'M = rowstack([[2.0, 0.0], [0.0, 3.0]])\nk(aa) = Normal(quadform(M, aa), 1.0)\n'
    + 'A = jointchain(aa = Normal(0.0, 1.0), bb = k)\n';
  assert.deepEqual(errors(src), []);
  const want = /quadform at \d+:\d+: spec §07 "Linear algebra" gives `quadform` the domain .*argument 2 is a record with fields aa/;
  return Promise.all([
    assert.rejects(value(src + 'lp = logdensityof(A, record(aa = 0.5, bb = 0.5))\n', 'lp'), want),
    assert.rejects(value(src + 'y ~ A\n', 'y', 4), want),
  ]);
});

test('the arguments §07 gives a scalar domain are left alone, and valid collections keep their values', () => {
  // quantile's level, tile's repeat count and bandedmat's integer are scalars by their Domains cells.
  assert.equal(ev('quantile', [vec(1, 2, 3, 4, 5), lit(0.5)]), 3);
  assert.deepEqual(Array.from(ev('tile', [vec(1, 2), lit(2)]).data), [1, 2, 1, 2]);
  // quadform(A, x) = x' A x with A = diag(2, 3), x = [1, 2]: 2 + 12.
  assert.equal(ev('quadform', [mat, vec(1, 2)]), 14);
  // linsolve(A, b): x = [3, 4] / diag(2, 3) = [1.5, 4/3].
  const x = Array.from(ev('linsolve', [mat, vec(3, 4)]).data) as number[];
  assert.ok(Math.abs(x[0] - 1.5) < 1e-14 && Math.abs(x[1] - 4 / 3) < 1e-14);
  // cross([1, 0, 0], [0, 1, 0]) = [0, 0, 1].
  assert.deepEqual(Array.from(ev('cross', [vec(1, 0, 0), vec(0, 1, 0)]).data), [0, 0, 1]);
});
