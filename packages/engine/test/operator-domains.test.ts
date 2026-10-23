'use strict';

// Spec §07 "Operator-equivalent functions": every arithmetic and comparison head has a Domains
// cell, and an operand outside it is a located static error. The infix operator, the named
// function and the per-cell call of a `broadcast` over the head share one statement of the table
// (operator-domains.ts). Expected values are the closed-form arithmetic of the operands.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');

const PRE = 's = 2.0\nt = 3.0\nn = 2\nb = true\nz = 1.0 + 2.0 * im\nstr = "a"\n'
  + 'v = [1.0, 2.0]\nw = [3.0, 4.0]\nu = [1.0, 2.0, 3.0]\niv = [1, 2]\n'
  + 'zv = [1.0 + 2.0 * im, 2.0 + 0.0 * im]\n'
  + 'm = rowstack([[1.0, 2.0], [3.0, 4.0]])\nm2 = rowstack([[1.0, 2.0], [3.0, 4.0]])\n'
  + 'tv = transpose(v)\ntw = transpose(w)\nr1 = record(a = 1.0)\n';

function errors(expr: string): Array<{ message: string; loc: any }> {
  return processSource(PRE + 'r = ' + expr + '\n').diagnostics.filter((d: any) => d.severity === 'error');
}

async function value(expr: string): Promise<number[]> {
  const { ctx } = ctxFor(PRE + 'r = ' + expr + '\n', 1);
  const m = await ctx.getMeasure('r');
  return Array.from(m.value ? m.value.data : m.samples);
}

function refused(exprs: string[], pattern: RegExp) {
  for (const expr of exprs) {
    const found = errors(expr);
    assert.ok(found.length > 0 && found.some(d => pattern.test(d.message)),
      `${expr}: ${JSON.stringify(found.map(d => d.message))}`);
    assert.ok(found.every(d => d.loc), `${expr}: every refusal is located`);
  }
}

function accepted(exprs: string[]) {
  for (const expr of exprs) assert.deepEqual(errors(expr).map(d => d.message), [], expr);
}

test('add and sub take scalars or arrays of the same shape', async () => {
  refused(['v + 1.0', '1.0 + v', 'v - s', 's - v', 'v + u', 'v - u', 'm + v', 'str + str', 'str - str'],
    /add|sub/);
  accepted(['s + t', 'n + n', 'b + b', 'z + s', 'v + w', 'm + m2', 'iv + iv', 'v - w', 's - t']);
  assert.deepEqual(await value('v + w'), [4, 6]);
  assert.deepEqual(await value('w - v'), [2, 2]);
  assert.deepEqual(await value('v .+ s'), [3, 4]);
});

test('mul takes only the §07 pairs', async () => {
  refused(['v * w', 'v * m', 'm * tv', 'tv * tw', 'str * s', 'm * u'], /mul/);
  accepted(['s * t', 'z * s', 'm * m2', 'm * v', 's * m', 's * v', 'tv * v', 'v * tv', 'tv * m']);
  // transposed-vector–vector is the inner product 1*3 + 2*4.
  assert.deepEqual(await value('tv * w'), [11]);
  // vector–transposed-vector is the outer product [[3, 4], [6, 8]].
  assert.deepEqual(await value('v * tw'), [3, 4, 6, 8]);
  // transposed-vector–matrix: [1 2] [[1 2] [3 4]] = [7 10].
  assert.deepEqual(await value('tv * m'), [7, 10]);
  assert.deepEqual(await value('m * v'), [5, 11]);
  assert.deepEqual(await value('s * v'), [2, 4]);
});

test('divide takes a scalar divisor, real or complex', async () => {
  refused(['s / v', 'v / w', 'm / m2', 'str / s'], /divide/);
  accepted(['s / t', 'z / s', 'v / s', 'm / s', 'tv / s', 's / z']);
  assert.deepEqual(await value('w / s'), [1.5, 2]);
  // (1 + 2i) / 2 = 0.5 + 1i.
  assert.deepEqual(await value('real(z / s)'), [0.5]);
  assert.deepEqual(await value('imag(z / s)'), [1]);
});

test('neg and pow take their §07 domains', () => {
  refused(['-str', '-r1'], /neg/);
  refused(['v ^ s', 's ^ v', 'm ^ s', 'str ^ s'], /pow/);
  accepted(['-s', '-n', '-z', '-v', '-m', 's ^ t', 'z ^ s', 'n ^ n']);
});

test('equal and unequal take integers, booleans and strings only', async () => {
  const domain = /integers, booleans and strings.*integer\(x\).*floor\(x\).*ceil\(x\).*round\(x\)/;
  refused(['1.5 == 2.5', 's == t', 's != t', 'n == s', 'z == z', 'z != s', 'equal(s, t)', 'unequal(1.5, 2.5)'], domain);
  refused(['str == n', 'b == str'], /compares a string only with a string/);
  accepted(['n == n', 'b == b', 'str == str', 'n != n', 'str != str', 'b != b', 'b == n',
    'floor(s) == floor(t)', 'ceil(s) != round(t)', 'integer(s) == n']);
  assert.deepEqual(await value('floor(1.5) == floor(1.9)'), [1]);
  assert.deepEqual(await value('n != 3'), [1]);
});

test('lt, le, gt and ge take reals only', async () => {
  const domain = /takes reals, but an operand has type (complex|string)/;
  refused(['z < s', 's >= z', 'z > z', '(1.0 + 2.0 * im) < 1.0', 'str < str', 'lt(z, s)', 'le(str, str)'], domain);
  accepted(['s < t', 'n <= n', 'b < b', 'n > s', 's >= t']);
  assert.deepEqual(await value('s < t'), [1]);
  assert.deepEqual(await value('t <= s'), [0]);
});

test('a broadcast applies the same domain to each cell', async () => {
  const equalDomain = /integers, booleans and strings/;
  refused(['v .== w', 'equal.(v, w)', 'v .!= s', 'unequal.(v, 1.5)', 'broadcast(equal, v, w)',
    'broadcast((a, c) -> a == c, v, w)'], equalDomain);
  refused(['zv .< w', 'lt.(zv, w)', 'gt.(["a"], ["b"])'], /takes reals/);
  accepted(['iv .== iv', 'equal.(iv, 2)', 'floor.(v) .== floor.(w)', 'v .< w', 'lt.(v, s)',
    'v .+ w', 'add.(v, w)', 'mul.(v, w)', 'divide.(v, w)', 'neg.(v)', 'pow.(v, s)', 'm .* m2',
    'broadcast(add, v, w)', 'broadcast((a, c) -> a + c, v, w)']);
  assert.deepEqual(await value('iv .== [2, 2]'), [0, 1]);
  assert.deepEqual(await value('mul.(v, w)'), [3, 8]);
  assert.deepEqual(await value('v .< [2.0, 1.0]'), [1, 0]);
});

test('an operand whose type is not resolved is not judged', () => {
  // A callback input is untyped until a call supplies it: `x + y` is judged at the call.
  const src = 'f(x, y) = x + y\ng(x, y) = x == y\nh(x) = 2.0 * x + [1.0, 2.0]\n'
    + 'r = f([1.0, 2.0], [3.0, 4.0])\n';
  assert.deepEqual(processSource(src).diagnostics.filter((d: any) => d.severity === 'error')
    .map((d: any) => d.message), []);
  const bad = processSource('f(x, y) = x + y\nr = f([1.0, 2.0], 1.0)\n').diagnostics
    .filter((d: any) => d.severity === 'error');
  assert.ok(bad.some((d: any) => /add/.test(d.message)), JSON.stringify(bad));
});
