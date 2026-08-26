'use strict';

// =====================================================================
// An aggregate over an INLINE container whose elements vary per atom
// =====================================================================
//
// `_alignedTensorFromGet` (sampler-aggregate.ts) recognises an atom-batched
// container only when it is a bare `self` ref named in the atom-batched set.
// Every other container is treated as atom-independent: evaluated ONCE under the
// batched env and broadcast with stride 0. That is right for a literal container
// and wrong for one built out of per-atom values — the shape a reified weight
// body produces (`[t, t^2][.i]`).
//
// Two distinct wrong outcomes, both pinned below:
//   - an element that is a bare batched ref silently yielded atom 0's value for
//     every atom (a WRONG NUMBER, no error);
//   - an element that is a computed expression yielded NaN.
//
// `_evalN` now routes that shape to `_perAtomFallback`, whose per-atom
// `evaluateExpr` is the reference implementation for an aggregate.
//
// ORACLE. Closed form by hand, since a second evaluator on the same build is
// not an oracle. With t = 0.6 and t = 0.5 over two atoms:
//   sum([t, 2])      = t + 2        → 2.6, 2.5
//   sum([t, t^2])    = t + t²       → 0.96, 0.75
//   sum([1, 2])      = 3            → 3, 3      (atom-independent, unchanged)
// The [1, 2] and bare-ref-container cases are the controls: they took the
// batched lifter before and must still take it.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { createWorkerHandler } = require('../worker.ts');
const { batchedScalar } = require('../value.ts');

const SUM = { kind: 'ref', ns: 'self', name: 'sum' };
const T = { kind: 'ref', ns: 'self', name: 'T' };
const vec = (args: any[]) => ({ kind: 'call', op: 'vector', args });
const sumOver = (container: any) => ({
  kind: 'call', op: 'aggregate',
  args: [SUM, vec([]),
    { kind: 'call', op: 'get', args: [container, { kind: 'axis', name: 'i' }] }],
});

function evalN(ir: any, refArrays: any, count: number) {
  const wk = createWorkerHandler();
  wk.handle({ type: 'init', seed: 7 });
  const r = wk.handle({ type: 'evaluateN', ir, count, refArrays });
  assert.equal(r.type, 'samples', 'evaluateN failed: ' + (r && r.message));
  return Array.from(r.samples) as number[];
}

const TWO_ATOMS = { T: batchedScalar(Float64Array.from([0.6, 0.5])) };

function assertClose(got: number[], want: number[], label: string) {
  assert.equal(got.length, want.length, label + ': length');
  for (let i = 0; i < want.length; i++) {
    assert.ok(Math.abs(got[i] - want[i]) < 1e-12,
      label + ': atom ' + i + ' got ' + got[i] + ', want ' + want[i]);
  }
}

test('an inline container holding a bare per-atom ref reduces per atom', () => {
  // sum([t, 2]) = t + 2. Previously [2.6, 2.6] — atom 0 broadcast to both.
  assertClose(evalN(sumOver(vec([T, { kind: 'lit', value: 2 }])), TWO_ATOMS, 2),
    [2.6, 2.5], 'sum([t, 2])');
});

test('an inline container holding a computed per-atom element reduces per atom', () => {
  // sum([t, t²]) = t + t². Previously NaN.
  const ir = sumOver(vec([T,
    { kind: 'call', op: 'pow', args: [T, { kind: 'lit', value: 2 }] }]));
  assertClose(evalN(ir, TWO_ATOMS, 2), [0.96, 0.75], 'sum([t, t²])');
});

test('an atom-independent literal container still reduces to a constant', () => {
  // The control: no per-atom name under the container, so the batched lifter
  // keeps handling it.
  assertClose(evalN(sumOver(vec([{ kind: 'lit', value: 1 }, { kind: 'lit', value: 2 }])),
    TWO_ATOMS, 2), [3, 3], 'sum([1, 2])');
});

test('a bare atom-batched ref container still reduces on the batched path', () => {
  // The other control: a `[N, k]` Value named by a bare ref is exactly what the
  // batched lifter detects, and must be untouched.
  const ir = sumOver({ kind: 'ref', ns: 'self', name: 'V' });
  const refs = {
    V: { shape: [2, 2], data: Float64Array.from([0.6, 0.36, 0.5, 0.25]), outerRank: 1 },
  };
  assertClose(evalN(ir, refs, 2), [0.96, 0.75], 'sum(V[.i])');
});
