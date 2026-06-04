'use strict';

// =====================================================================
// Demand-driven fixed-value resolver (engine-concepts §17.4, stage 3).
// =====================================================================
//
// FixedValues replaces the former eager `while (progress)` pre-eval loop
// in buildDerivations with a lazy, memoised, cycle-guarded resolver. The
// unit tests below inject mock deps to pin the contract in isolation
// (laziness, memoisation, phase gate, cycle handling, iterate-forces-all,
// no-negative-cache); the integration test pins build-time laziness on a
// real module.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { FixedValues } = require('../fixed-values.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const { processSource, orchestrator } = require('..');

// Build a FixedValues over a tiny mock binding graph. Each binding is
// `{ phase, ir }` where `ir` is `{ refs: [names], val: (env) => value }`.
// `collectSelfRefs` reads `ir.refs`; `evaluateExpr` calls `ir.val(env)`
// and counts invocations so tests can assert what was computed.
function makeFV(bindingsObj: any) {
  const bindings = new Map(Object.entries(bindingsObj));
  let evalCalls = 0;
  const evaluated: string[] = [];
  const deps = {
    bindings,
    derivations: {},
    resolveMeasureRef: () => null,
    isMeasureBinding: () => false,
    samplerLib: {
      evaluateExpr: (ir: any, env: any) => { evalCalls++; if (ir._name) evaluated.push(ir._name); return ir.val(env); },
    },
    traceeval: { walk: () => { throw new Error('traceeval should not run in these tests'); } },
    expandMeasureIR: () => null,
    collectSelfRefs: (ir: any) => new Set((ir && ir.refs) || []),
    lowerExpr: (x: any) => x,
  };
  return { fv: new FixedValues(deps), evalCalls: () => evalCalls, evaluated };
}
const b = (phase: any, refs: any[], val: any, name?: string) =>
  ({ phase, ir: { refs, val, _name: name } });

test('FixedValues: nothing is computed until demanded; get memoises', () => {
  const { fv, evalCalls } = makeFV({
    a: b('fixed', [], () => 5),
    unused: b('fixed', [], () => 99),
  });
  assert.equal(evalCalls(), 0, 'construction computes nothing');
  assert.equal(fv.get('a'), 5);
  assert.equal(evalCalls(), 1, 'one demand → one compute');
  assert.equal(fv.get('a'), 5);
  assert.equal(evalCalls(), 1, 'second demand is memoised (no recompute)');
  // `unused` was never asked for → never computed (we deliberately never
  // call has/get on it; calling has() WOULD resolve it).
  assert.equal(evalCalls(), 1, 'a never-demanded binding is never computed');
});

test('FixedValues: phase gate rejects non-fixed bindings without computing', () => {
  const { fv, evalCalls } = makeFV({
    p: b('parameterized', [], () => 1),
    s: b('stochastic', [], () => 2),
    anon: b(null, [], () => 7),      // lift-anon (phase == null) IS resolvable
  });
  assert.equal(fv.has('p'), false);
  assert.equal(fv.get('p'), undefined);
  assert.equal(fv.has('s'), false);
  assert.equal(evalCalls(), 0, 'phase gate rejects before _compute');
  assert.equal(fv.get('anon'), 7, 'phase==null lift-anon resolves');
});

test('FixedValues: resolves a value-ref chain on demand', () => {
  const { fv } = makeFV({
    x: b('fixed', [], () => 3),
    y: b('fixed', ['x'], (env: any) => env.x + 1),
  });
  assert.equal(fv.get('y'), 4, 'y resolves x first, then computes');
});

test('FixedValues: a self-cycle yields UNRESOLVED, not an infinite loop', () => {
  const { fv } = makeFV({ c: b('fixed', ['c'], () => 1) });
  assert.equal(fv.has('c'), false);
  assert.equal(fv.get('c'), undefined);
});

test('FixedValues: a binding whose evaluator throws is UNRESOLVED (no negative cache)', () => {
  let throwIt = true;
  const bindings = new Map<string, any>([
    ['t', { phase: 'fixed', ir: { refs: [] } }],
  ]);
  const fv = new FixedValues({
    bindings, derivations: {},
    resolveMeasureRef: () => null, isMeasureBinding: () => false,
    samplerLib: { evaluateExpr: () => { if (throwIt) throw new Error('boom'); return 42; } },
    traceeval: { walk: () => { throw new Error('no'); } },
    expandMeasureIR: () => null, collectSelfRefs: () => new Set(), lowerExpr: (x: any) => x,
  });
  assert.equal(fv.get('t'), undefined, 'throw → UNRESOLVED');
  // No negative cache: a later demand re-attempts and now succeeds.
  throwIt = false;
  assert.equal(fv.get('t'), 42, 'retried on next demand (no negative cache)');
});

test('FixedValues: iteration force-resolves every fixed-phase binding', () => {
  const { fv, evalCalls } = makeFV({
    a: b('fixed', [], () => 1),
    b2: b('fixed', [], () => 2),
    p: b('parameterized', [], () => 3),   // excluded by phase
  });
  assert.equal(evalCalls(), 0);
  const keys = [...fv].map(([k]: any) => k).sort();
  assert.deepEqual(keys, ['a', 'b2'], 'only fixed-phase bindings populate the cache');
  assert.equal(evalCalls(), 2, 'iterate forced both fixed bindings (not the parameterized one)');
  assert.equal(fv.size, 2);
});

test('build-time laziness: a fixed binding not needed for shape/classification is not pre-computed', () => {
  // `extra` is a fixed scalar with an `evaluate` derivation, referenced
  // by nothing and not needed for any shape — so buildDerivations must
  // NOT resolve it (no eager sweep). It resolves only on first demand.
  const src = `
flatppl_compat = "0.1"
extra = 2.0 + 3.0
x ~ Normal(mu = 0, sigma = 1)
`;
  const { built } = makeMatCtx(src, { sampleCount: 16 });
  const fv = built.fixedValues;
  assert.ok(fv._cache instanceof Map, 'fixedValues is the lazy resolver (has a _cache)');
  assert.equal(fv._cache.has('extra'), false,
    'extra is NOT computed during buildDerivations (demand-driven)');
  assert.equal(fv.get('extra'), 5, 'resolves correctly on first demand');
  assert.equal(fv._cache.has('extra'), true, 'and is cached thereafter');
});

test('cascade-prune faithfulness: an unevaluable fixed binding is diagnosed AND its dependent is pruned', () => {
  // `bad` is fixed-phase but references an undefined function — neither
  // classifiable nor evaluable (an engine-gap dead end). Two guarantees,
  // both preserved under the lazy resolver:
  //   (1) the dead-end diagnostic fires (loud, not silent), and
  //   (2) the dependent `y` is cascade-pruned, NOT left to fail at
  //       materialise — resolvable() force-resolves the underived fixed
  //       binding (the bounded dead-end set) to confirm it has no value,
  //       exactly as the old eager prune did. (Laziness is preserved for
  //       fixed bindings that DO have a derivation — those never resolve
  //       in the prune sweep; verified by the build-time-laziness test.)
  const src = `flatppl_compat = "0.1"\nbad = no_such_fn(3)\ny = bad + 1\n`;
  const r = processSource(src);
  const built = orchestrator.buildDerivations(r.bindings);
  const deadEnd = (built.diagnostics || []).find((dd: any) =>
    dd.severity === 'error' && /Fixed-phase binding 'bad' produced no value/.test(dd.message));
  assert.ok(deadEnd, 'unevaluable fixed binding gets the dead-end error diagnostic');
  assert.ok(!built.derivations.y,
    'dependent of an unevaluable fixed binding is cascade-pruned (faithful to the old prune)');
});
