'use strict';

// Exercise pyhf helpers through public parsing/linking and ordinary evaluation.
// Values are frozen from pyhf code4, not another FlatPPL implementation.
const test = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator } = require('../index.ts');
const { evaluateExpr } = require('../sampler.ts');

test('pyhf normsys call forms preserve the portable reference values', () => {
  const reference = 'hep = standard_module("particle-physics", "0.1")\n'
    + 'normsys_factor(lo, hi, alpha) = hep.interp_poly6_exp(lo, 1.0, hi, alpha)';
  const result = processSource('pyhf = standard_module("pyhf_helpers", "0.1")\n'
    + 'reference = load_module("reference.flatppl")\n'
    + 'lo = elementof(posreals)\nhi = elementof(posreals)\nalpha = elementof(reals)\n'
    + 'outputs = [pyhf.normsys_factor(lo, hi, alpha),\n'
    + '  pyhf.normsys_factor(lo, alpha = alpha, hi = hi),\n'
    + '  pyhf.normsys_factor(record(alpha = alpha, hi = hi, lo = lo)),\n'
    + '  reference.normsys_factor(lo, hi, alpha)]',
  { bundle: { sources: { 'reference.flatppl': reference } } });
  assert.deepEqual(result.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const built = orchestrator.buildDerivations(result.linkedBindings,
    { moduleRegistry: result.linkedModuleRegistry });
  const ir = built.bindings.get('outputs').ir;
  const alphas = [-2, -0.5, 0.7, 1.5];
  const expected = [0.7744, 0.937444710705783, 1.102564077845851, 1.2332376088978148];
  for (let i = 0; i < alphas.length; i++) {
    const actual = evaluateExpr(ir, {
      lo: 0.88, hi: 1.15, alpha: alphas[i],
      __moduleRegistry: result.linkedModuleRegistry,
      __resolveFnBody: (name: string) => {
        const body = built.bindings.get(name)?.ir;
        return body?.op === 'functionof' ? body : null;
      },
    });
    assert.equal(actual.data.length, 4);
    for (const value of actual.data) {
      assert.ok(Math.abs(value - expected[i]) < 1e-12, `${value} != ${expected[i]}`);
    }
  }
});

test('pyhf histosys broadcasts additive shifts without adding the nominal', () => {
  const result = processSource('pyhf = standard_module("pyhf_helpers", "0.1")\n'
    + 'alpha = [-5.0, -1.0, -0.4, 0.0, 0.4, 1.0, 5.0]\n'
    + 'outputs = pyhf.histosys_shift.(5.0, 10.0, 30.0, alpha)');
  assert.deepEqual(result.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const built = orchestrator.buildDerivations(result.linkedBindings,
    { moduleRegistry: result.linkedModuleRegistry });
  const actual = built.fixedValues.get('outputs');
  const expected = [-25, -5, -2.97848, 0, 7.02152, 20, 100];
  assert.equal(actual.data.length, expected.length);
  expected.forEach((value, i) => assert.ok(Math.abs(actual.data[i] - value) < 1e-12));
});
