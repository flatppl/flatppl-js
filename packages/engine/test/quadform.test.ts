'use strict';

// =====================================================================
// quadform.test.ts — quadform(A, x) = xᵀ A x (spec §07)
// =====================================================================

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator } = require('../index.ts');

test('quadform: xᵀ I x = ‖x‖²', () => {
  const src = `
A = rowstack([[1.0, 0.0], [0.0, 1.0]])
x = [3.0, 4.0]
q = quadform(A, x)
`;
  const ctx = processSource(src);
  const built = orchestrator.buildDerivations(ctx.bindings);
  // 3² + 4² = 25
  assert.equal(built.fixedValues.get('q'), 25);
});

test('quadform: xᵀ A x for non-identity A', () => {
  const src = `
A = rowstack([[2.0, 1.0], [1.0, 3.0]])
x = [1.0, 2.0]
q = quadform(A, x)
`;
  const ctx = processSource(src);
  const built = orchestrator.buildDerivations(ctx.bindings);
  // xᵀ A x = [1, 2] · [[2,1],[1,3]] · [1,2]ᵀ
  //        = [1, 2] · [4, 7]ᵀ
  //        = 4 + 14 = 18
  assert.equal(built.fixedValues.get('q'), 18);
});

test('quadform: parses + classifies as evaluable', () => {
  const src = `
A = rowstack([[1.0, 0.5], [0.5, 1.0]])
x = [1.0, 1.0]
q = quadform(A, x)
`;
  const ctx = processSource(src);
  const errs = (ctx.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, 'no diagnostics');
  const built = orchestrator.buildDerivations(ctx.bindings);
  // xᵀ A x = [1, 1] · [1.5, 1.5]ᵀ = 3
  assert.ok(Math.abs(built.fixedValues.get('q') - 3) < 1e-12);
});
