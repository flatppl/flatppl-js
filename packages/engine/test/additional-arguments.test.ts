'use strict';

// Spec §04 "Additional arguments" (flatppl-design #216): an operation that supplies the
// leading arguments of a callable passes the call's remaining arguments to the callable's
// remaining inputs, positionally or by name, whole. A sole additional record or table
// passes whole if one positional input remains, otherwise its fields or columns bind the
// remaining inputs by name. No input may be left unbound or bound twice.
//
// ORACLES are closed forms. weighted(w, Normal(0, 1), ...) has the log density
// log w(x, ...) + log N(x) at x; logweighted adds the weight; the pushforward of
// Normal(0, 1) through y = p exp(x) has log N(log(y / p)) - log y.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');

const logN = (x: number) => -0.5 * Math.log(2 * Math.PI) - 0.5 * x * x;

async function value(src: string, name: string): Promise<number[]> {
  const { ctx } = ctxFor(src, 1);
  const m = await ctx.getMeasure(name);
  return Array.from(m.value ? m.value.data : m.samples);
}

function errors(src: string): string[] {
  return processSource(src).diagnostics.filter((d: any) => d.severity === 'error')
    .map((d: any) => d.message);
}

function near(got: number[], want: number[], tol = 1e-12) {
  assert.equal(got.length, want.length);
  want.forEach((w, i) => assert.ok(Math.abs(got[i] - w) < tol, `got ${got[i]}, want ${w}`));
}

const W = 'w(x, p, q) = p * x * x + q\n';
const AT = 0.5;
const WANT = Math.log(3 * AT * AT + 1) + logN(AT);

test('weighted: two additional arguments bind positionally, by name, or from a record', async () => {
  for (const extras of ['3.0, 1.0', 'p = 3.0, q = 1.0', 'q = 1.0, p = 3.0', 'record(p = 3.0, q = 1.0)']) {
    near(await value(`${W}m = weighted(w, Normal(0, 1), ${extras})\nlp = logdensityof(m, ${AT})\n`, 'lp'), [WANT]);
  }
});

test('weighted: a sole record passes whole when one positional input remains', async () => {
  // w(x, p) = x * p.a with p = record(a = 2): the weight at 0.5 is 1.
  near(await value('w(x, p) = x * p.a\nm = weighted(w, Normal(0, 1), record(a = 2.0))\nlp = logdensityof(m, 0.5)\n', 'lp'),
    [logN(0.5)]);
});

test('weighted: a sole table splats its columns by name', async () => {
  const src = 'w(x, p, q) = p[1] * x * x + q[1]\nt = table(p = [3.0], q = [1.0])\n'
    + 'm = weighted(w, Normal(0, 1), t)\nlp = logdensityof(m, 0.5)\n';
  near(await value(src, 'lp'), [WANT]);
});

test('logweighted: the callback gets the same additional arguments', async () => {
  const lw = 'lw(x, p, q) = p * x + q\n';
  const want = 2.0 * AT + 1.0 + logN(AT);
  for (const extras of ['2.0, 1.0', 'q = 1.0, p = 2.0', 'record(p = 2.0, q = 1.0)']) {
    near(await value(`${lw}m = logweighted(lw, Normal(0, 1), ${extras})\nlp = logdensityof(m, ${AT})\n`, 'lp'), [want]);
  }
});

test('an unbound, unknown or doubly bound input is a static error', () => {
  assert.ok(errors(`${W}m = weighted(w, Normal(0, 1), p = 3.0)\n`).some(m => /input "q" is not bound/.test(m)));
  assert.ok(errors(`${W}m = weighted(w, Normal(0, 1), p = 3.0, q = 1.0, r = 2.0)\n`).some(m => /no remaining input named "r"/.test(m)));
  assert.ok(errors(`${W}m = weighted(w, Normal(0, 1), x = 3.0, p = 1.0, q = 1.0)\n`).some(m => /no remaining input named "x"/.test(m)));
  assert.ok(errors(`${W}m = weighted(w, Normal(0, 1), 3.0)\n`).some(m => /callback receives 2 whole inputs, but declares 3/.test(m)));
  assert.ok(errors(`${W}m = weighted(w, Normal(0, 1), 3.0, q = 1.0)\n`).length > 0);
});

test('the leading inputs of weighted stay positional', () => {
  assert.ok(errors(`${W}m = weighted(weight = w, base = Normal(0, 1), p = 3.0, q = 1.0)\n`)
    .some(m => /never by keyword argument/.test(m)));
});

test('pushfwd: additional arguments reach the function, its inverse and its log-volume', async () => {
  const bij = 'f(x, p) = p * exp(x)\nfi(y, p) = log(y / p)\nlv(x, p) = x + log(p)\nb = bijection(f, fi, lv)\n';
  const want = logN(Math.log(3 / 2)) - Math.log(3);
  for (const extras of ['2.0', 'p = 2.0']) {
    const src = `${bij}m = pushfwd(b, Normal(0, 1), ${extras})\nlp = logdensityof(m, 3.0)\n`;
    assert.deepEqual(errors(src), []);
    near(await value(src, 'lp'), [want]);
  }
  near(await value('g(x, p) = p * exp(x)\nm = pushfwd(g, Normal(0, 1), p = 2.0)\nlp = logdensityof(m, 3.0)\n', 'lp'), [want]);
});

test('reduce: additional arguments follow the two inputs and stay fixed in every call', async () => {
  const f = 'f(a, b, p, q) = a + b * p + q\n';
  // ((1 + 2*2 + 1) + 3*2 + 1) = 13
  for (const extras of ['2.0, 1.0', 'p = 2.0, q = 1.0', 'record(p = 2.0, q = 1.0)']) {
    assert.deepEqual(await value(`${f}r = reduce(f, [1.0, 2.0, 3.0], ${extras})\n`, 'r'), [13]);
  }
  assert.deepEqual(await value('r = reduce(add, [10, 20] .* [2, 3])\n', 'r'), [80]);
  assert.deepEqual(await value('r = reduce(add, [7.0])\n', 'r'), [7]);
  assert.ok(errors(`${f}r = reduce(f, [1.0, 2.0], p = 2.0)\n`).some(m => /input "q" is not bound/.test(m)));
});
