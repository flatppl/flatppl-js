'use strict';

// =====================================================================
// kernel-broadcast-handlers.test.ts — Phase 1.1 registry surface
// =====================================================================
//
// Pins the KERNEL_BROADCAST_FAST_PATHS registry surface introduced in
// Phase 1.1 of the broadcast staged plan (TODO-flatppl-js.md):
//
//   - `registerKernelBroadcastFastPath(distOp, handler)` registers a
//     handler keyed by distOp; multiple per distOp allowed.
//   - `tryKernelBroadcastFastPath(ctx)` walks registered handlers for
//     `ctx.d.distOp`, invokes the first whose `match` returns true,
//     returns its execute promise; returns null when nothing matches.
//   - Built-in Normal handler is registered at module load time.
//
// Behavioural equivalence with the pre-refactor Normal hot path is
// covered by the existing `kernel-broadcast.test.ts` /
// `broadcast-semantics.test.ts` / `dot-broadcast.test.ts` suites —
// those would break if the registry dispatched differently from the
// inline check.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const handlers = require('../kernel-broadcast-handlers.ts');

test('built-in Normal handler is registered at module load', () => {
  const listed = handlers._testListKernelBroadcastFastPaths();
  const normalEntry = listed.find((e: any) => e.distOp === 'Normal');
  assert.ok(normalEntry, 'expected Normal entry in the registry');
  assert.equal(normalEntry.count, 1,
    'expected exactly one Normal handler registered at module load');
  assert.match(normalEntry.labels[0], /atom-indep/,
    'expected the Normal handler label to mention its atom-indep gate');
});

test('tryKernelBroadcastFastPath returns null for an unknown distOp', () => {
  const result = handlers.tryKernelBroadcastFastPath({
    d: { distOp: 'NonexistentDistribution-' + Date.now() },
    ctx: {}, name: '_test', K: 1, N: 1, paramVals: {}, anyAtomDep: false,
  });
  assert.equal(result, null,
    'unknown distOp must return null so caller falls through');
});

test('tryKernelBroadcastFastPath returns null when no registered Normal handler matches', () => {
  // The built-in Normal handler requires !anyAtomDep AND both mu and
  // sigma present. Force a mismatch via anyAtomDep=true to confirm
  // the dispatcher returns null (rather than calling execute).
  const result = handlers.tryKernelBroadcastFastPath({
    d: { distOp: 'Normal' },
    ctx: {}, name: '_test', K: 1, N: 1,
    paramVals: { mu: 0, sigma: 1 }, anyAtomDep: true,
  });
  assert.equal(result, null,
    'Normal handler should not match when anyAtomDep is true');
});

test('register + dispatch: first-match-wins ordering', () => {
  // Register two handlers under a synthetic distOp and confirm the
  // first one's match wins, second is never consulted.
  const testDistOp = '__TestDistOp_' + Date.now();
  let firstMatchCalled = 0, firstExecuteCalled = 0;
  let secondMatchCalled = 0, secondExecuteCalled = 0;

  handlers.registerKernelBroadcastFastPath(testDistOp, {
    label: 'first',
    match: () => { firstMatchCalled++; return true; },
    execute: () => { firstExecuteCalled++; return Promise.resolve('first-result'); },
  });
  handlers.registerKernelBroadcastFastPath(testDistOp, {
    label: 'second',
    match: () => { secondMatchCalled++; return true; },
    execute: () => { secondExecuteCalled++; return Promise.resolve('second-result'); },
  });

  const result = handlers.tryKernelBroadcastFastPath({
    d: { distOp: testDistOp },
    ctx: {}, name: '_test', K: 1, N: 1, paramVals: {}, anyAtomDep: false,
  });

  return Promise.resolve(result).then((value: any) => {
    assert.equal(value, 'first-result', 'first handler should win');
    assert.equal(firstMatchCalled, 1, 'first match consulted once');
    assert.equal(firstExecuteCalled, 1, 'first execute called once');
    assert.equal(secondMatchCalled, 0, 'second match not consulted');
    assert.equal(secondExecuteCalled, 0, 'second execute not called');
  });
});

test('register + dispatch: falls through when earlier handler does not match', () => {
  // Earlier handler refuses (match=false); later handler accepts.
  const testDistOp = '__TestDistOpFallthrough_' + Date.now();
  let firstExecuted = false, secondExecuted = false;

  handlers.registerKernelBroadcastFastPath(testDistOp, {
    label: 'strict (declines)',
    match: () => false,
    execute: () => { firstExecuted = true; return Promise.resolve('first'); },
  });
  handlers.registerKernelBroadcastFastPath(testDistOp, {
    label: 'lenient',
    match: () => true,
    execute: () => { secondExecuted = true; return Promise.resolve('second'); },
  });

  const result = handlers.tryKernelBroadcastFastPath({
    d: { distOp: testDistOp },
    ctx: {}, name: '_test', K: 1, N: 1, paramVals: {}, anyAtomDep: false,
  });

  return Promise.resolve(result).then((value: any) => {
    assert.equal(value, 'second',
      'second handler should win when first declines');
    assert.equal(firstExecuted, false,
      'first execute must not be called when its match returned false');
    assert.equal(secondExecuted, true,
      'second execute should run');
  });
});
