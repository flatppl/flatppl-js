'use strict';

// A dependency is typed standalone, so a shape behind one of its inputs is
// `%dynamic` there. `load_module(..., n = 3)` fixes that input, and the loader
// now types `m.x` under the substitution: a wrong-length point or operand is a
// static error rather than a run-time failure ("value exhausted", or a read of
// `undefined`).
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('..');
const T = require('../types.ts');
const { ctxFor } = require('./_ctx-factory.ts');

const LIB = 'n = external(posintegers)\nx = zeros(n)\nM = iid(Normal(0.0, 1.0), n)\n';
const opts = { bundle: { sources: { 'lib.flatppl': LIB } } };
const compile = (body: string) => processSource('m = load_module("lib.flatppl", n = 3)\n' + body, opts);
const errors = (r: any) => r.diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message);
const typeOf = (r: any, name: string) => r.loweredModule.bindings.get(name).inferredType;

test('a loaded binding takes the shape its substituted input fixes', () => {
  const r = compile('y = m.x\nM = m.M\n');
  assert.deepEqual(errors(r), []);
  assert.ok(T.equal(typeOf(r, 'y'), T.array(1, [3], T.REAL)), JSON.stringify(typeOf(r, 'y')));
  assert.deepEqual(typeOf(r, 'M').domain.shape, [3]);
});

test('a wrong-length point or operand against a substituted shape is a static error', () => {
  assert.deepEqual(errors(compile('lp = logdensityof(m.M, [0.1, 0.2])\n')),
    ['logdensityof: arg 2 expects array of real (length 3), got array of real (length 2)']);
  assert.match(errors(compile('z = m.x + [1.0, 2.0]\n')).join('\n'),
    /^add: operand types array of real \(length 3\) and array of real \(length 2\) are not numerically compatible/);
});

test('a substitution through a loader binding folds, and the right length scores', async () => {
  const src = 'k = 3\nm = load_module("lib.flatppl", n = k)\nlp = logdensityof(m.M, [0.1, 0.2, 0.3])\n';
  const r = processSource(src, opts);
  assert.deepEqual(errors(r), []);
  // Closed form: 3 log N(·; 0, 1) at 0.1, 0.2, 0.3.
  const want = -1.5 * Math.log(2 * Math.PI) - 0.5 * (0.01 + 0.04 + 0.09);
  const { ctx } = ctxFor(src, 1, opts);
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - want) < 1e-12, `${lp.samples[0]} vs ${want}`);
});
