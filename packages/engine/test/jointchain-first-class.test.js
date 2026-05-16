'use strict';

// jointchain/kchain first-class derivation kind — STEP 1 (classifier +
// explicit step structure, behind the off-by-default
// JOINTCHAIN_STATE.firstClass flag). Dual-path migration: with the
// flag off the legacy inlineChainOps rewrite is unchanged (zero
// behaviour change — the rest of the suite proves that); with it on,
// classifyJointchain builds the explicit step structure. Materialiser
// is a step-2 stub, so these tests assert CLASSIFICATION only.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator } = require('..');
const { buildDerivations } = orchestrator;
const { JOINTCHAIN_STATE } = orchestrator._internal;

// Run `fn` with the first-class flag forced on, always restoring it
// (test isolation — every other suite must see the default off path).
function withFirstClass(fn) {
  const prev = JOINTCHAIN_STATE.firstClass;
  JOINTCHAIN_STATE.firstClass = true;
  try { return fn(); } finally { JOINTCHAIN_STATE.firstClass = prev; }
}

const KDEF =
  'M = Normal(mu = 0.0, sigma = 1.0)\n' +
  'K = functionof(Normal(mu = x, sigma = 1.0), x = x)\n' +
  'K2 = functionof(Normal(mu = y, sigma = 1.0), y = y)\n';

test('flag OFF (default): jointchain still goes through legacy rewrite', () => {
  // No kind:'jointchain' is produced; inlineChainOps rewrites to a
  // record/tuple/joint shape exactly as before. (Zero-behaviour-change
  // guarantee for step 1.)
  const { derivations } = buildDerivations(
    processSource(KDEF + 'jc = jointchain(a = M, b = K)\n').bindings);
  const d = derivations['jc'];
  assert.ok(!d || d.kind !== 'jointchain',
    'with the flag off, no first-class jointchain kind may appear');
  assert.equal(JOINTCHAIN_STATE.firstClass, false, 'flag stays off by default');
});

test('flag ON: positional 2-arg jointchain → explicit step structure', () => {
  withFirstClass(() => {
    const { derivations } = buildDerivations(
      processSource(KDEF + 'jc = jointchain(M, K)\n').bindings);
    const d = derivations['jc'];
    assert.ok(d, 'jc classified');
    assert.equal(d.kind, 'jointchain');
    assert.equal(d.marginalize, false);
    assert.equal(d.labels, null);
    assert.deepEqual(d.steps, [
      { var: 's0', role: 'base', ref: 'M', kernel: false },
      { var: 's1', role: 'kernel', ref: 'K', inputs: ['s0'] },
    ]);
  });
});

test('flag ON: kchain sets marginalize:true', () => {
  withFirstClass(() => {
    const { derivations } = buildDerivations(
      processSource(KDEF + 'kc = kchain(M, K)\n').bindings);
    const d = derivations['kc'];
    assert.equal(d.kind, 'jointchain');
    assert.equal(d.marginalize, true, 'kchain marginalizes intermediates');
  });
});

test('flag ON: kwarg form carries labels (record-shaped)', () => {
  withFirstClass(() => {
    const { derivations } = buildDerivations(
      processSource(KDEF + 'jc = jointchain(p = M, q = K)\n').bindings);
    const d = derivations['jc'];
    assert.equal(d.kind, 'jointchain');
    assert.deepEqual(d.labels, ['p', 'q']);
    assert.deepEqual(d.steps.map((s) => s.var), ['p', 'q']);
    assert.deepEqual(d.steps[1].inputs, ['p']);
  });
});

test('flag ON: N-ary positional → left-assoc step inputs accumulate', () => {
  withFirstClass(() => {
    const { derivations } = buildDerivations(
      processSource(KDEF + 'jc = jointchain(M, K, K2)\n').bindings);
    const d = derivations['jc'];
    assert.equal(d.kind, 'jointchain');
    assert.equal(d.steps.length, 3);
    assert.deepEqual(d.steps[2],
      { var: 's2', role: 'kernel', ref: 'K2', inputs: ['s0', 's1'] });
  });
});

test('flag ON: kernel-first (step-0 ref is itself a kernel)', () => {
  withFirstClass(() => {
    const { derivations } = buildDerivations(
      processSource(KDEF + 'jc = jointchain(K, K2)\n').bindings);
    const d = derivations['jc'];
    assert.equal(d.kind, 'jointchain');
    assert.equal(d.steps[0].role, 'base');
    assert.equal(d.steps[0].kernel, true, 'kernel-first base flagged');
  });
});

test('classifyJointchain: non-kernel later arg → null (parity fallback)', () => {
  // K_i must be a kernel; a measure in kernel position isn't covered
  // by the first-class path → null so the dual-path falls back.
  const cj = orchestrator._internal.classifyJointchain;
  const rhsIR = { kind: 'call', op: 'jointchain',
    args: [{ kind: 'ref', ns: 'self', name: 'M' },
           { kind: 'ref', ns: 'self', name: 'M' }] };
  const ast = { type: 'CallExpr',
    callee: { type: 'Identifier', name: 'jointchain' },
    args: [{ type: 'Identifier', name: 'M' },
           { type: 'Identifier', name: 'M' }] };
  const bindings = processSource(KDEF).bindings;
  assert.equal(cj(rhsIR, ast, bindings), null);
});
