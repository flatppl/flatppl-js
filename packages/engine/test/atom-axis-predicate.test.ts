'use strict';

// =====================================================================
// atom-axis-predicate.test.ts — P3 canonical isAtomBatched / outerRank
// =====================================================================
//
// Pins the contract added by P3 of the broadcast / aggregate /
// batching consolidation (TODO-flatppl-js.md "In-flight P1-P9"):
//
//   - `value.isAtomBatched(v, N)` is the SINGLE canonical predicate
//     answering "does this Value carry an atom axis at position 0?"
//   - `value.atomShape(v, N)` returns the per-atom shape suffix
//     (or null when not atom-batched).
//   - `outerRank` (engine-concepts §2.1) is populated by matIid
//     producers — every iid output Value carries `outerRank=1`
//     (the iid sample axis is logically distinct from the atom
//     axis, per spec §13 Pyro sampleShape semantics).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const valueLib = require('../value.ts');
const { processSource, orchestrator, materialiser } = require('../index.ts');

// =====================================================================
// 1. isAtomBatched — Float64Array form
// =====================================================================

test('isAtomBatched: Float64Array of length N → true (batched scalar)', () => {
  const N = 10;
  const v = new Float64Array(N);
  assert.equal(valueLib.isAtomBatched(v, N), true);
  // atomShape for a Float64Array(N) is the empty per-atom shape.
  assert.deepEqual(valueLib.atomShape(v, N), []);
});

test('isAtomBatched: Float64Array of wrong length → false', () => {
  assert.equal(valueLib.isAtomBatched(new Float64Array(7), 10), false);
});

// =====================================================================
// 2. isAtomBatched — Value form
// =====================================================================

test('isAtomBatched: Value with shape=[N] → true', () => {
  const N = 8;
  const v = { shape: [N], data: new Float64Array(N) };
  assert.equal(valueLib.isAtomBatched(v, N), true);
  assert.deepEqual(valueLib.atomShape(v, N), []);
});

test('isAtomBatched: Value with shape=[N, k] → true with inner=[k]', () => {
  const N = 5, k = 3;
  const v = { shape: [N, k], data: new Float64Array(N * k) };
  assert.equal(valueLib.isAtomBatched(v, N), true);
  assert.deepEqual(valueLib.atomShape(v, N), [k]);
});

test('isAtomBatched: Value with shape=[m, n] where m !== N → false', () => {
  const N = 10;
  const v = { shape: [3, 4], data: new Float64Array(12) };
  assert.equal(valueLib.isAtomBatched(v, N), false);
  assert.equal(valueLib.atomShape(v, N), null);
});

test('isAtomBatched: rank-0 Value (scalar) → false', () => {
  const v = { shape: [], data: new Float64Array([42]) };
  assert.equal(valueLib.isAtomBatched(v, 10), false);
});

// =====================================================================
// 3. outerRank gates isAtomBatched
// =====================================================================

test('isAtomBatched: outerRank=0 (per-atom matrix) → still treats shape[0]=N as atom-batched', () => {
  // A per-atom matrix has shape=[N, m, n] with outerRank absent
  // (or 0). The §2.1 contract puts the atom axis at position 0
  // regardless; outerRank only distinguishes the intrinsic structure
  // of the per-atom slice.
  const N = 4;
  const v = { shape: [N, 3, 3], data: new Float64Array(N * 9) };
  assert.equal(valueLib.isAtomBatched(v, N), true);
});

test('isAtomBatched: outerRank=1, shape=[N, k] (iid output) → true', () => {
  // matIid produces this shape: N outer atoms × k inner samples.
  const N = 6, k = 4;
  const v = { shape: [N, k], data: new Float64Array(N * k), outerRank: 1 };
  assert.equal(valueLib.isAtomBatched(v, N), true);
  assert.deepEqual(valueLib.atomShape(v, N), [k]);
});

test('isAtomBatched: outerRank=0 — explicit 0 → false (no atom axis)', () => {
  // outerRank=0 means "every axis is inner / cell"; no atom axis.
  // Only fires when something was explicitly tagged outerRank=0,
  // which is unusual but the canonical predicate respects it.
  const N = 4;
  const v = { shape: [N, 3], data: new Float64Array(N * 3), outerRank: 0 };
  assert.equal(valueLib.isAtomBatched(v, N), false);
});

// =====================================================================
// 4. matIid producer sets outerRank=1
// =====================================================================

async function materialiseTo(src: string, name: string, sampleCount: number) {
  const { bindings } = processSource(src);
  const built = orchestrator.buildDerivations(bindings);
  // Minimal worker stub: routes sampleN to a deterministic
  // stand-in (the test only checks shape/tag, not numerics).
  const sendWorker = (msg: any): Promise<any> => {
    if (msg.type === 'sampleN') {
      const N = msg.count, k = msg.repeat || 1;
      const samples = new Float64Array(N * k);
      for (let i = 0; i < samples.length; i++) samples[i] = i + 1;
      return Promise.resolve({ samples });
    }
    if (msg.type === 'setEnv') return Promise.resolve();
    throw new Error('test stub: unhandled worker message ' + msg.type);
  };
  const ctx: any = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues,
    moduleRegistry: built.moduleRegistry,
    rootSeed: 1,
    sampleCount,
    sendWorker,
    _measureCache: new Map(),
  };
  ctx.getMeasure = (nm: string) => {
    if (ctx._measureCache.has(nm)) return ctx._measureCache.get(nm);
    const p = materialiser.materialiseMeasure(nm, ctx);
    ctx._measureCache.set(nm, p);
    return p;
  };
  return ctx.getMeasure(name);
}

test('matIid: iid(Normal, 4) over N=6 → output Value has outerRank=1', async () => {
  const m = await materialiseTo(`
mu = elementof(reals)
sigma = elementof(posreals)
M ~ iid(Normal(mu = 0.0, sigma = 1.0), 4)
`, 'M', 6);
  assert.ok(m.value, 'iid output carries a Value');
  assert.deepEqual(m.value.shape, [6, 4]);
  assert.equal(m.value.outerRank, 1,
    'iid output Value carries outerRank=1');
  // The canonical predicate sees it as atom-batched with inner=[4].
  assert.equal(valueLib.isAtomBatched(m.value, 6), true);
  assert.deepEqual(valueLib.atomShape(m.value, 6), [4]);
});
