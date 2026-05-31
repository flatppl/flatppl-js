'use strict';

// =====================================================================
// composite-body-recognizers.test.ts — Phase 1.2 recognizer surface
// =====================================================================
//
// Pins the COMPOSITE_BODY_RECOGNIZERS registry surface introduced in
// Phase 1.2 of the broadcast staged plan (TODO-flatppl-js.md):
//
//   - `registerCompositeBodyRecognizer(fn)` appends to a single
//     ordered list — first-match-wins.
//   - `tryRecognizeCompositeBody(d, ctx)` walks the list and returns
//     the first non-null CompositeBody, or null when nothing matches.
//   - Built-in iid recognizer is registered at module load time;
//     wraps `kernel-broadcast-shape.detectIidKernelBinding` and tags
//     the result with `kind: 'iid'`.
//
// Behavioural equivalence with the pre-refactor `_detectIidKernelBody`
// is covered by the existing `kernel-broadcast.test.ts` /
// `fusion-b-kernel-broadcast.test.ts` suites — those would break if
// the recognizer dispatch produced a different shape than the inline
// helper did.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const recognizers = require('../composite-body-recognizers.ts');

test('built-in iid recognizer is registered at module load', () => {
  // The iid recognizer is registered as a side effect of requiring
  // the module. There's at least one entry; the precise count grows
  // as Phase 4 recognizers land.
  const count = recognizers._testRegisteredCompositeBodyRecognizerCount();
  assert.ok(count >= 1, 'expected at least one recognizer registered');
});

test('tryRecognizeCompositeBody returns null when no recognizer matches', () => {
  // No bindings + a synthetic distOp → no recognizer can resolve a
  // CompositeBody.
  const result = recognizers.tryRecognizeCompositeBody(
    { distOp: 'NoSuchKernel-' + Date.now() },
    { bindings: new Map(), fixedValues: new Map() },
  );
  assert.equal(result, null,
    'no matching recognizer must return null');
});

test('tryRecognizeCompositeBody returns null when ctx.bindings is missing', () => {
  // The iid recognizer guards against missing ctx.bindings (it would
  // otherwise NPE on bindings.has). Verify the dispatcher tolerates
  // this even though it's the responsibility of individual recognizers.
  const result = recognizers.tryRecognizeCompositeBody(
    { distOp: 'AnyOp' },
    { /* bindings missing */ },
  );
  assert.equal(result, null,
    'absent bindings must produce null, not throw');
});

// The registry is module-global, so test-registered recognizers persist
// across tests. Each register test uses a unique synthetic distOp tag
// so its recognizers only fire when explicitly invoked with that tag —
// keeping the dispatch behaviour observable while leaving other tests
// undisturbed.

test('register + dispatch: first-match-wins ordering', () => {
  const tag = '__first_wins_' + Date.now();
  let firstCalls = 0, secondCalls = 0;
  const firstResult: any = { kind: 'iid', binding: { _label: 'first' },
    params: [], paramKwargs: [], distOp: tag, distParams: [],
    distKwargs: {}, n: 1 };
  const secondResult: any = { kind: 'iid', binding: { _label: 'second' },
    params: [], paramKwargs: [], distOp: tag, distParams: [],
    distKwargs: {}, n: 1 };

  recognizers.registerCompositeBodyRecognizer((d: any) => {
    if (!d || d.distOp !== tag) return null;
    firstCalls++; return firstResult;
  });
  recognizers.registerCompositeBodyRecognizer((d: any) => {
    if (!d || d.distOp !== tag) return null;
    secondCalls++; return secondResult;
  });

  const out = recognizers.tryRecognizeCompositeBody({ distOp: tag }, {});
  assert.strictEqual(out, firstResult, 'first recognizer must win');
  assert.equal(firstCalls, 1, 'first recognizer consulted once');
  assert.equal(secondCalls, 0, 'second recognizer not consulted');
});

test('register + dispatch: later recognizer wins when earlier declines', () => {
  const tag = '__fall_through_' + Date.now();
  let strictCalls = 0, lenientCalls = 0;
  const lenientResult: any = { kind: 'iid', binding: { _label: 'lenient' },
    params: [], paramKwargs: [], distOp: tag, distParams: [],
    distKwargs: {}, n: 1 };

  recognizers.registerCompositeBodyRecognizer((d: any) => {
    if (!d || d.distOp !== tag) return null;
    strictCalls++; return null;  // declines despite seeing the tag
  });
  recognizers.registerCompositeBodyRecognizer((d: any) => {
    if (!d || d.distOp !== tag) return null;
    lenientCalls++; return lenientResult;
  });

  const out = recognizers.tryRecognizeCompositeBody({ distOp: tag }, {});
  assert.strictEqual(out, lenientResult,
    'lenient recognizer should win when strict declines');
  assert.equal(strictCalls, 1, 'strict consulted once');
  assert.equal(lenientCalls, 1, 'lenient consulted once');
});
