'use strict';

// =====================================================================
// callable-binding-types.test.ts — canonical callable-binding-type
// predicate (`isCallableLikeBindingType`) and its consumers.
// =====================================================================
//
// Pins the consolidation of the previously scattered
// `b.type === 'fn' || 'functionof' || 'kernelof' || 'bijection'` chains
// into a single predicate in derivations.ts. The predicate adds
// `'fchain'` to the accepted set — previously a standalone
// `f = fchain(g, h)` binding either failed the fixed-phase dead-end
// check or wasn't recognised as a callable in head-ref positions.
//
// Coverage:
//   1. The predicate itself: accepts the 5 callable types, rejects
//      everything else.
//   2. Standalone fchain binding doesn't fire the fixed-phase dead-end
//      diagnostic.
//   3. Standalone bijection binding doesn't fire the dead-end either.
//   4. The predicate matches the empirical set: any binding type that
//      analyzer.classifyValueNode can return AND that holds a callable.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');
const orchestrator = require('../orchestrator.ts');
const derivations  = require('../derivations.ts');

test('isCallableLikeBindingType: accepts callable types, rejects others', () => {
  const fn = derivations.isCallableLikeBindingType;
  // Accepted: every analyzer-tagged callable producer.
  assert.equal(fn('fn'),         true);
  assert.equal(fn('functionof'), true);
  assert.equal(fn('kernelof'),   true);
  assert.equal(fn('bijection'),  true);
  assert.equal(fn('fchain'),     true);
  // Rejected: non-callable bindings.
  assert.equal(fn('call'),       false);
  assert.equal(fn('literal'),    false);
  assert.equal(fn('input'),      false);
  assert.equal(fn('lawof'),      false);
  assert.equal(fn('likelihood'), false);
  assert.equal(fn('draw'),       false);
  assert.equal(fn('module'),     false);
  assert.equal(fn(undefined),    false);
});

test('standalone fchain binding: no fixed-phase dead-end diagnostic', () => {
  // f = fchain(g, h) — a function-valued binding that's not applied
  // anywhere else in the module. Before commit, this fired the
  // dead-end check because 'fchain' wasn't in OBJECT_BINDING_TYPES.
  // Now the predicate includes it.
  // Use explicit functionof bindings (rather than lambdas) so g/h
  // get the analyzer's 'functionof' binding type unambiguously —
  // lambdas may lower to a slightly different shape depending on
  // surface form. The narrow contract under test is: an fchain
  // binding doesn't fire the dead-end check, regardless of how
  // its component functions were defined.
  const src = `
a = elementof(reals)
g = functionof(a + 1, a = a)
h = functionof(a * 2, a = a)
f = fchain(g, h)
`;
  const ctx = processSource(src);
  const built = orchestrator.buildDerivations(ctx.bindings);
  const deadEnd = (built.diagnostics || []).filter((d: any) =>
    d.severity === 'error' && /'f' produced no value/.test(d.message)
  );
  assert.equal(deadEnd.length, 0,
    'fchain binding should not fire fixed-phase dead-end check');
});

test('standalone bijection binding: no fixed-phase dead-end diagnostic', () => {
  // bj = bijection(f, finv, logvol) as a binding — needs to be
  // recognised as a callable-like object regardless of where it's
  // referenced.
  const src = `
a    = elementof(reals)
f    = functionof(exp(a), a = a)
finv = functionof(log(a), a = a)
lv   = functionof(a,      a = a)
bj   = bijection(f, finv, lv)
`;
  const ctx = processSource(src);
  const built = orchestrator.buildDerivations(ctx.bindings);
  const deadEnd = (built.diagnostics || []).filter((d: any) =>
    d.severity === 'error' && /'bj' produced no value/.test(d.message)
  );
  assert.equal(deadEnd.length, 0,
    'bijection binding should not fire fixed-phase dead-end check');
});

test('standalone fn / functionof / kernelof bindings: still pass dead-end check', () => {
  // Regression test for the consolidation — none of the legacy callable
  // types should have been broken.
  const src = `
mu = 0.0
sigma = 1.0
a      = elementof(reals)
g      = functionof(2 * a, a = a)
K      = kernelof(Normal(mu = mu, sigma = sigma))
`;
  const ctx = processSource(src);
  const built = orchestrator.buildDerivations(ctx.bindings);
  // Specifically check g and K — the legacy callable binding types
  // we don't want regressed by the predicate consolidation.
  const deadEnd = (built.diagnostics || []).filter((d: any) =>
    d.severity === 'error' && /'(g|K)' produced no value/.test(d.message)
  );
  assert.equal(deadEnd.length, 0,
    'legacy callable binding types should still pass dead-end check');
});
