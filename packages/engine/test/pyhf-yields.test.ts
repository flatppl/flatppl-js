'use strict';

// Public module calls, portable definitions and independent yield arithmetic.
const test = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator } = require('../index.ts');
const { evaluateExpr } = require('../sampler.ts');
const V = require('../value.ts');

const reference = 'flatppl_compat = "0.1"\n'
  + 'sample_yields(nominal, shifts, factors) =\n'
  + '  (nominal .+ aggregate(sum, [.s, .b], shifts[.s, .a, .b] / 1.0)) .*\n'
  + '  aggregate(prod, [.s, .b], factors[.s, .m, .b] / 1.0)\n'
  + 'expected_counts(samples) = aggregate(sum, [.b], samples[.s, .b] / 1.0)\n';

function evaluator(body: string, portable = false) {
  const module = portable ? 'load_module("reference.flatppl")' : 'standard_module("pyhf_helpers", "0.1")';
  const result = processSource(`flatppl_compat = "0.1"\npyhf = ${module}\n${body}`,
    { bundle: { sources: { 'reference.flatppl': reference } } });
  assert.deepEqual(result.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const built = orchestrator.buildDerivations(result.linkedBindings,
    { moduleRegistry: result.linkedModuleRegistry });
  return (inputs: Record<string, unknown> = {}) => evaluateExpr(built.bindings.get('outputs').ir, {
    ...Object.fromEntries(built.fixedValues), ...inputs,
    __moduleRegistry: result.linkedModuleRegistry,
    __resolveFnBody: (name: string) => {
      const ir = built.bindings.get(name)?.ir;
      return ir?.op === 'functionof' ? ir : null;
    },
  });
}

test('yield helpers preserve singleton batches, keywords and zero factors', () => {
  const body = 'n = elementof(cartpow(cartpow(reals, [1, 2]), 1))\n'
    + 'a = elementof(cartpow(cartpow(reals, [1, 2, 2]), 1))\n'
    + 'm = elementof(cartpow(cartpow(reals, [1, 3, 2]), 2))\n'
    + 'outputs = pyhf.expected_counts.(pyhf.sample_yields.(nominal = n, factors = m, shifts = a))';
  const inputs = {
    n: { shape: [1, 1, 2], outerRank: 1, data: Float64Array.of(10, 20) },
    a: { shape: [1, 1, 2, 2], outerRank: 1, data: Float64Array.of(1, 2, 3, 4) },
    m: { shape: [2, 1, 3, 2], outerRank: 1, data: Float64Array.of(0, 2, 3, 0, 5, 0, 1, 1, 1, 1, 1, 1) },
  };
  for (const portable of [false, true]) {
    const actual = evaluator(body, portable)(inputs);
    if (V.isValue(actual)) {
      assert.deepEqual(actual.shape, [2, 2]);
      assert.equal(V.outerRankOf(actual), 1);
      assert.deepEqual(Array.from(actual.data), [0, 0, 14, 26]);
    } else {
      assert.equal(actual.length, 2);
      assert.deepEqual(actual.map((row: unknown) => {
        const value = V.asValue(row);
        assert.deepEqual(value.shape, [2]);
        return Array.from(value.data);
      }), [[0, 0], [14, 26]]);
    }
  }
});

test('yield helpers retain empty identities and logical matrix views', () => {
  for (const portable of [false, true]) {
    const expected = evaluator('n = rowstack([[1, 2, 3], [4, 5, 6]])\n'
      + 'outputs = pyhf.expected_counts(transpose(n))', portable)();
    assert.deepEqual(Array.from(V.asValue(expected).data), [6, 15]);
    for (const [samples, additive, factors] of [[2, 1, 2], [2, 0, 2], [2, 1, 0], [2, 0, 0], [0, 1, 2]]) {
      const s = samples ? ':' : 'ix', a = additive ? ':' : 'ix', m = factors ? ':' : 'ix';
      const evaluate = evaluator('n = fill(10.0, [2, 3])\n'
        + 'a = fill(1.0, [2, 1, 3])\nm = fill(2.0, [2, 2, 3])\n'
        + 'ix = fill(1, get([0], 1))\n'
        + `outputs = pyhf.expected_counts(pyhf.sample_yields(record(nominal = n[${s}, :], `
        + `factors = m[${s}, ${m}, :], shifts = a[${s}, ${a}, :])))`, portable);
      let actual;
      assert.doesNotThrow(() => { actual = evaluate(); },
        `portable=${portable}, samples=${samples}, additive=${additive}, factors=${factors}`);
      const value = V.asValue(actual);
      assert.deepEqual(value.shape, [3]);
      assert.deepEqual(Array.from(value.data), Array(3).fill(samples * (10 + additive) * 2 ** factors));
    }
  }
});

test('yield helpers reject mismatched model cells rather than broadcast them', () => {
  const evaluate = evaluator('n = elementof(cartpow(reals, [2, 3]))\n'
    + 'a = elementof(cartpow(reals, [2, 1, 3]))\nm = elementof(cartpow(reals, [2, 2, 3]))\n'
    + 'outputs = pyhf.sample_yields(n, a, m)');
  const n = V.withShape(new Float64Array(6), [2, 3]);
  const a = V.withShape(new Float64Array(6), [2, 1, 3]);
  const m = V.withShape(new Float64Array(12), [2, 2, 3]);
  assert.throws(() => evaluate({ n: V.vector([1, 2, 3]), a, m }), /real rank-2 array/);
  assert.throws(() => evaluate({ n, a, m: V.withShape(new Float64Array(4), [2, 2, 1]) }), /extents must agree/);
  assert.throws(() => evaluate({ n: { ...n, outerRank: 1 }, a, m }), /real rank-2 array/);
  assert.throws(() => evaluate({ n: V.complexValue(n.data, n.data, n.shape), a, m }), /real rank-2 array/);
});
