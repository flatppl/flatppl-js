'use strict';

// =====================================================================
// kernel-broadcast-shape.test.ts — P7 shared kernel-broadcast classifier
// =====================================================================
//
// Pins the contract added by P7 of the broadcast/aggregate/batching
// consolidation (TODO-flatppl-js.md "In-flight P1-P9"):
//
//   - `kernel-broadcast-shape.detectIidKernelBinding(name, bindings,
//     fixedValues?)` is the SINGLE structural recogniser shared by
//     classify-time (derivations.ts:classifyKernelBroadcast) AND
//     runtime (mat-broadcast.ts:matKernelBroadcast).
//   - Recognises `functionof(lawof(iid(<BuiltinDist>, n)), kw)` shape
//     post-lowering (with one-level anon-deref through the lift
//     pass).
//   - With fixedValues, ref-to-fixed-integer n is admissible (runtime
//     use); without, literal n is preferred but ref-form is permitted
//     for classify-time yes/no gate.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const kbShape = require('../kernel-broadcast-shape.ts');

function refr(name: string) { return { kind: 'ref', ns: 'self', name }; }
function lit(v: any) { return { kind: 'lit', value: v }; }

function makeKernelBinding(distOp: string, distKwargs: any, nArg: any): any {
  return {
    ir: {
      kind: 'call', op: 'functionof',
      params: ['mu'],
      paramKwargs: ['mu'],
      body: {
        kind: 'call', op: 'lawof',
        args: [{
          kind: 'call', op: 'iid',
          args: [{ kind: 'call', op: distOp, kwargs: distKwargs }, nArg],
        }],
      },
    },
  };
}

// =====================================================================
// 1. detectIidKernelBinding — literal n
// =====================================================================

test('detectIidKernelBinding: literal n + builtin dist → descriptor', () => {
  const bindings = new Map([['K', makeKernelBinding('Normal',
    { mu: refr('mu'), sigma: lit(1) }, lit(4))]]);
  const d = kbShape.detectIidKernelBinding('K', bindings);
  assert.ok(d, 'descriptor returned');
  assert.equal(d.distOp, 'Normal');
  assert.equal(d.n, 4);
  assert.deepEqual(d.params, ['mu']);
  assert.deepEqual(d.distParams, ['mu', 'sigma']);
});

// =====================================================================
// 2. Ref-form n: classify-time vs runtime
// =====================================================================

test('detectIidKernelBinding: ref-n without fixedValues → classify-time admissible (n=NaN)', () => {
  const bindings = new Map([['K', makeKernelBinding('Normal',
    { mu: refr('mu'), sigma: lit(1) }, refr('N_per_group'))]]);
  // Classify-time call (no fixedValues). The classifier should
  // return a descriptor so isIidCompositeKernelBinding returns true.
  const d = kbShape.detectIidKernelBinding('K', bindings);
  assert.ok(d, 'classify-time accepts ref-form n');
  assert.ok(Number.isNaN(d.n),
    'ref-form n at classify-time is signalled as NaN (caller ignores n)');
});

test('detectIidKernelBinding: ref-n with fixedValues lookup → resolves to integer', () => {
  const bindings = new Map([['K', makeKernelBinding('Normal',
    { mu: refr('mu'), sigma: lit(1) }, refr('N_per_group'))]]);
  const fixedValues = new Map([['N_per_group', 7]]);
  const d = kbShape.detectIidKernelBinding('K', bindings, fixedValues);
  assert.ok(d);
  assert.equal(d.n, 7, 'fixedValues resolves the ref to a concrete integer');
});

test('detectIidKernelBinding: ref-n with fixedValues but binding missing → null', () => {
  const bindings = new Map([['K', makeKernelBinding('Normal',
    { mu: refr('mu'), sigma: lit(1) }, refr('N_missing'))]]);
  const fixedValues = new Map();   // empty
  assert.equal(kbShape.detectIidKernelBinding('K', bindings, fixedValues), null,
    'unresolved ref-n with fixedValues → null (runtime needs concrete n)');
});

// =====================================================================
// 3. Anon-deref through lift's hoisting
// =====================================================================

test('detectIidKernelBinding: inner dist as anon ref → deref through binding map', () => {
  const distCall = { kind: 'call', op: 'Normal', kwargs: {
    mu: refr('mu'), sigma: lit(1),
  }};
  const bindings = new Map([
    ['__anon0', { ir: distCall }],
    ['K', makeKernelBinding('Normal', null, lit(4))],  // placeholder
  ]);
  // Manually adjust: replace inner with ref to __anon0.
  (bindings.get('K') as any).ir.body.args[0].args[0] = refr('__anon0');
  const d = kbShape.detectIidKernelBinding('K', bindings);
  assert.ok(d);
  assert.equal(d.distOp, 'Normal');
});

// =====================================================================
// 4. Non-matching shapes return null
// =====================================================================

test('detectIidKernelBinding: missing binding → null', () => {
  assert.equal(kbShape.detectIidKernelBinding('missing', new Map()), null);
});

test('detectIidKernelBinding: non-functionof binding → null', () => {
  const bindings = new Map([['K', { ir: { kind: 'lit', value: 1 } }]]);
  assert.equal(kbShape.detectIidKernelBinding('K', bindings), null);
});

test('detectIidKernelBinding: functionof but body not lawof(iid) → null', () => {
  const bindings = new Map([['K', { ir: {
    kind: 'call', op: 'functionof', params: ['mu'],
    body: { kind: 'call', op: 'Normal', kwargs: { mu: refr('mu') } },
  } }]]);
  assert.equal(kbShape.detectIidKernelBinding('K', bindings), null);
});

test('detectIidKernelBinding: inner dist not in SAMPLEABLE → null', () => {
  const bindings = new Map([['K', makeKernelBinding('NotARealDist',
    { x: lit(0) }, lit(4))]]);
  assert.equal(kbShape.detectIidKernelBinding('K', bindings), null,
    'unknown inner distribution rejected');
});

// =====================================================================
// 5. isIidCompositeKernelBinding — convenience yes/no
// =====================================================================

test('isIidCompositeKernelBinding: returns true for valid composite', () => {
  const bindings = new Map([['K', makeKernelBinding('Normal',
    { mu: refr('mu'), sigma: lit(1) }, lit(4))]]);
  assert.equal(kbShape.isIidCompositeKernelBinding('K', bindings), true);
});

test('isIidCompositeKernelBinding: returns false for non-composite', () => {
  assert.equal(kbShape.isIidCompositeKernelBinding('missing', new Map()),
    false);
});
