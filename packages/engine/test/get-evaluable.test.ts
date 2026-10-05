'use strict';

// #3 consolidation: `get` / `get0` (spec §07 unified access) is now
// implemented IN the single deterministic-evaluator authority
// (sampler.evaluateCall), not special-cased in callers. Previously a
// fixed-phase expression containing indexing dead-ended with
// "evaluateExpr: call op 'get' not evaluable". These tests pin the
// value-level semantics and the end-to-end fixed-phase path.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler = require('../sampler.ts');
const { processSource, orchestrator } = require('..');

const lit = (v: any) => ({ kind: 'lit', value: v });
const get = (op: any, c: any, ...sel: any[]) => ({ kind: 'call', op, args: [lit(c), ...sel] });
const ev = (ir: any) => sampler.evaluateExpr(ir, {});

test('get: 1-based element, multi-dim, all-slice, subset, record', () => {
  assert.equal(ev(get('get', [10, 20, 30], lit(2))), 20);
  assert.equal(ev(get('get', [[1, 2], [3, 4]], lit(2), lit(1))), 3);
  assert.deepEqual(
    ev(get('get', [[1, 2], [3, 4]], { kind: 'const', name: 'all' }, lit(1))),
    [1, 3], 'M[:,1] → column 1 (1-based)');
  assert.deepEqual(
    ev(get('get', [10, 20, 30],
      { kind: 'call', op: 'vector', args: [lit(1), lit(3)] })),
    [10, 30], 'subset v[[1,3]]');
  assert.equal(ev(get('get', { a: 1, b: 2 }, lit('a'))), 1);
  assert.deepEqual(
    ev(get('get', { a: 1, b: 2, c: 3 },
      { kind: 'call', op: 'vector', args: [lit('a'), lit('c')] })),
    { a: 1, c: 3 }, 'record subset → sub-record');
});

test('get0: 0-based element', () => {
  assert.equal(ev(get('get0', [10, 20, 30], lit(0))), 10);
  assert.equal(ev(get('get0', [10, 20, 30], lit(2))), 30);
});

test('empty tensor selection retains all surviving dimensions', () => {
  const result = processSource('source = elementof(cartpow(reals, [2, 1, 3]))\n'
    + 'ix = fill(1, get([0], 1))\n'
    + 'middle = source[:, ix, :]\nfirst = source[ix, :, :]\n'
    + 'last = source[:, :, ix]\ndropped = source[ix, 1, :]\n'
    + 'outputs = aggregate(sum, [.s, .b], middle[.s, .m, .b])');
  assert.deepEqual(result.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const built = orchestrator.buildDerivations(result.bindings);
  const env: any = { ...Object.fromEntries(built.fixedValues),
    source: { shape: [2, 1, 3], data: new Float64Array(6).fill(1) } };
  for (const [name, shape] of [['middle', [2, 0, 3]], ['first', [0, 1, 3]],
    ['last', [2, 1, 0]], ['dropped', [0, 3]]] as const) {
    env[name] = sampler.evaluateExpr(built.bindings.get(name).ir, env);
    assert.deepEqual(env[name].shape, shape);
    assert.equal(env[name].data.length, 0);
  }
  const sums = sampler.evaluateExpr(built.bindings.get('outputs').ir, env);
  assert.deepEqual(sums.shape, [2, 3]);
  assert.deepEqual(Array.from(sums.data), [0, 0, 0, 0, 0, 0]);
});

test('get: out-of-bounds and bad-target throw with a clear message', () => {
  assert.throws(() => ev(get('get', [1, 2], lit(5))),
    /get index 5 out of bounds for length 2/);
  assert.throws(() => ev(get('get', 3.0, lit(1))),
    /get index target is not an array/);
});

test('get: fixed-phase indexing now pre-evaluates (no dead end)', () => {
  const { fixedValues, diagnostics } = orchestrator.buildDerivations(
    processSource('b = [10.0, 20.0, 30.0]\nc = b[2]\nd = b[2] + 1.0\n').bindings);
  assert.equal(fixedValues.get('c'), 20);
  assert.equal(fixedValues.get('d'), 21);
  // The #1 fixed-phase-dead-end diagnostic must no longer fire — the
  // gap it was surfacing is closed.
  assert.deepEqual(diagnostics, []);
});

test('get: distribution param with indexing classifies (no dead end)', () => {
  // `tau[1]` inside leaf-distribution params previously threw
  // "call op 'get' not evaluable"; now tau is fixed-phase and the
  // param IR is evaluable, so g classifies as a sample derivation.
  const { derivations, diagnostics } = orchestrator.buildDerivations(
    processSource(
      'bkg = [50.0, 52.0]\ndbkg = [3.0, 7.0]\n' +
      'tau = (bkg ./ dbkg) .^ 2\n' +
      'g = draw(Gamma(shape = tau[1] + 1.0, rate = tau[1]))\n').bindings);
  assert.ok(derivations['g'], 'g has a derivation');
  assert.deepEqual(diagnostics, [], 'no fixed-phase dead end');
});
