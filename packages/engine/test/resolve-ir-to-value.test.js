'use strict';

// Contract characterization for orchestrator.resolveIRToValue.
//
// This pins the resolver's observable behaviour BEFORE the planned
// "one deterministic-evaluation authority" consolidation (refactor #3),
// so that consolidation is provably behaviour-preserving on the static
// fast paths and the general delegate path. Two contracts:
//
//   - Static fast paths (lit / neg / vector / record / ref) return
//     exact-shaped plain JS (number | array | object) with no sampler
//     dependency.
//   - Anything else delegates to the deterministic evaluator and is
//     normalized back to plain JS (NOT a shape-tagged Value) so every
//     caller (obsIR clamp, MvNormal params, kernel-broadcast params)
//     sees the documented contract. Genuine errors (missing ref,
//     cycle) still throw with a naming message.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator } = require('..');
const R = orchestrator.resolveIRToValue;

function ctxFor(src) {
  const built = orchestrator.buildDerivations(processSource(src).bindings);
  return { bindings: built.bindings, fixedValues: built.fixedValues };
}

test('resolveIRToValue: static fast paths return plain JS', () => {
  const { bindings: B, fixedValues: FV } = ctxFor('z = 0.0\n');
  assert.equal(R({ kind: 'lit', value: 5 }, B, FV), 5);
  assert.equal(
    R({ kind: 'call', op: 'neg', args: [{ kind: 'lit', value: 3 }] }, B, FV), -3);
  assert.deepEqual(
    R({ kind: 'call', op: 'vector',
        args: [{ kind: 'lit', value: 1 }, { kind: 'lit', value: 2 }] }, B, FV),
    [1, 2]);
  assert.deepEqual(
    R({ kind: 'call', op: 'record',
        fields: [{ name: 'x', value: { kind: 'lit', value: 7 } }] }, B, FV),
    { x: 7 });
});

test('resolveIRToValue: ref resolves via fixedValues then binding IR', () => {
  const { bindings: B, fixedValues: FV } = ctxFor('b = [1.0, 2.0, 3.0]\n');
  assert.deepEqual(R({ kind: 'ref', ns: 'self', name: 'b' }, B, FV), [1, 2, 3]);
});

test('resolveIRToValue: general expr delegates and returns plain JS (not a Value)', () => {
  // broadcast(add, b, 1) — the kernel-broadcast param shape that
  // previously threw "unsupported op 'broadcast'". Must come back as a
  // plain JS array, not a shape-tagged Value, per the resolver's
  // documented output contract.
  const { bindings: B, fixedValues: FV } = ctxFor('b = [1.0, 2.0, 3.0]\n');
  const out = R({
    kind: 'call', op: 'broadcast',
    args: [
      { kind: 'call', op: 'functionof', params: ['_a_'], paramKwargs: ['a'],
        paramSources: [{ kind: 'placeholder', name: '_a_' }],
        body: { kind: 'call', op: 'add',
          args: [{ kind: 'ref', ns: '%local', name: '_a_' },
                 { kind: 'lit', value: 1 }] } },
      { kind: 'ref', ns: 'self', name: 'b' },
    ],
  }, B, FV);
  assert.ok(Array.isArray(out), 'plain JS array, not a Value');
  assert.deepEqual(out, [2, 3, 4]);
});

test('resolveIRToValue: missing ref throws a naming error', () => {
  const { bindings: B, fixedValues: FV } = ctxFor('z = 0.0\n');
  assert.throws(
    () => R({ kind: 'ref', ns: 'self', name: 'nope' }, B, FV),
    /resolveIRToValue: no IR for 'nope'/);
});

test('resolveIRToValue: dependency cycle throws the cycle message', () => {
  // Hand-built circular binding map (the pre-eval would normally break
  // this; resolveIRToValue must still guard structurally).
  const B = new Map([
    ['a', { ir: { kind: 'ref', ns: 'self', name: 'b' } }],
    ['b', { ir: { kind: 'ref', ns: 'self', name: 'a' } }],
  ]);
  assert.throws(
    () => R({ kind: 'ref', ns: 'self', name: 'a' }, B, new Map()),
    /resolveIRToValue: cycle through/);
});
