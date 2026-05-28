'use strict';

// =====================================================================
// ops-variants-typed.test.ts — typed-arg dispatch (P1 ↔ P2 ↔ P3a)
// =====================================================================
//
// Pins the unified variant model that lets a single registry dispatch
// on EITHER value-domain args (Value with shape/tag/struct/dtype)
// OR measure-domain args (measure type with sample/batch/event shape
// from P2) OR IR-context (axisStack from P3a). Engine-concepts §18.11
// extended.
//
// What's pinned:
//   1. ArgInfo constructors wrap raw values / measure types /
//      callables uniformly.
//   2. Matcher dispatches on `kind` discriminator: value-kind
//      variants ignore measure-typed args and vice versa.
//   3. Measure-arg matching consumes P2's sampleShape / batchShape /
//      eventShape with '*' wildcards.
//   4. axisStack constraints match P3a IR-context annotations.
//   5. Specificity ordering across the new dimensions (kind / measure
//      shape / axisStack) places categorical filters above per-shape
//      refinements.
//   6. Legacy value-only callers (raw Values into dispatch / dispatchVariant)
//      keep working unchanged.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const ops = require('../ops.ts');
const valueLib = require('../value.ts');

// Trigger op + variant registrations.
require('../ops-declarations.ts');

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

function vec(arr: number[]): any {
  const data = new Float64Array(arr.length);
  for (let i = 0; i < arr.length; i++) data[i] = arr[i];
  return { shape: [arr.length], data };
}

function makeMeasureType(opts: any) {
  const t: any = { kind: 'measure', domain: opts.domain || { kind: 'scalar', prim: 'real' } };
  if (opts.sampleShape !== undefined) t.sampleShape = opts.sampleShape;
  if (opts.batchShape  !== undefined) t.batchShape  = opts.batchShape;
  if (opts.eventShape  !== undefined) t.eventShape  = opts.eventShape;
  return t;
}

// =====================================================================
// 1. ArgInfo constructors
// =====================================================================

test('argInfoFromValue: wraps a Value with kind=value', () => {
  const v = vec([1, 2, 3]);
  const info = ops.argInfoFromValue(v);
  assert.equal(info.kind, 'value');
  assert.strictEqual(info.value, v);
  assert.equal(info.axisStack, undefined);
});

test('argInfoFromValue: optional axisStack carries through', () => {
  const v = vec([1]);
  const stk = [{ source: 'iid', size: 5 }];
  const info = ops.argInfoFromValue(v, { axisStack: stk });
  assert.deepEqual(info.axisStack, stk);
});

test('argInfoFromMeasure: wraps a measure type with kind=measure', () => {
  const mt = makeMeasureType({ eventShape: [], sampleShape: [10] });
  const info = ops.argInfoFromMeasure(mt);
  assert.equal(info.kind, 'measure');
  assert.strictEqual(info.measureType, mt);
});

test('argInfoFromKernel / argInfoFromFunction', () => {
  const k = ops.argInfoFromKernel({ kind: 'kernel' });
  assert.equal(k.kind, 'kernel');
  const f = ops.argInfoFromFunction({ kind: 'function' });
  assert.equal(f.kind, 'function');
});

// =====================================================================
// 2. Kind-discriminated matching
// =====================================================================

test('matchArgPattern: kind=value variant rejects measure-kind ArgInfo', () => {
  const valuePattern = { kind: 'value', rank: 1 };
  const measureInfo = ops.argInfoFromMeasure(makeMeasureType({ eventShape: [] }));
  assert.equal(ops._matchArgPattern(valuePattern, measureInfo), false);
});

test('matchArgPattern: kind=measure variant matches measure-kind ArgInfo', () => {
  const measurePattern = { kind: 'measure' };
  const measureInfo = ops.argInfoFromMeasure(makeMeasureType({ eventShape: [] }));
  assert.equal(ops._matchArgPattern(measurePattern, measureInfo), true);
});

test('matchArgPattern: raw value implicitly treated as kind=value', () => {
  const valuePattern = { rank: 1 };
  const v = vec([1, 2, 3]);
  // Existing API: raw Value passed in is auto-wrapped.
  assert.equal(ops._matchArgPattern(valuePattern, v), true);
  // Explicit ArgInfo: same result.
  assert.equal(ops._matchArgPattern(valuePattern, ops.argInfoFromValue(v)), true);
});

// =====================================================================
// 3. Measure-arg shape matching (P2 integration)
// =====================================================================

test('matchArgPattern: sampleShape exact match', () => {
  const mt = makeMeasureType({ sampleShape: [10], eventShape: [] });
  const info = ops.argInfoFromMeasure(mt);
  assert.equal(ops._matchArgPattern({ kind: 'measure', sampleShape: [10] }, info), true);
  assert.equal(ops._matchArgPattern({ kind: 'measure', sampleShape: [11] }, info), false);
  assert.equal(ops._matchArgPattern({ kind: 'measure', sampleShape: [10, 5] }, info), false);
});

test('matchArgPattern: sampleShape with wildcard', () => {
  const mt = makeMeasureType({ sampleShape: [10] });
  const info = ops.argInfoFromMeasure(mt);
  assert.equal(ops._matchArgPattern({ kind: 'measure', sampleShape: ['*'] }, info), true);
  // Length still has to match.
  assert.equal(ops._matchArgPattern({ kind: 'measure', sampleShape: ['*', '*'] }, info), false);
});

test('matchArgPattern: batchShape distinguishes kernel-broadcast from iid', () => {
  const iidMt   = makeMeasureType({ sampleShape: [4], batchShape: [],  eventShape: [] });
  const bcastMt = makeMeasureType({ sampleShape: [],  batchShape: [4], eventShape: [] });
  // Pattern matches only batch-shape=[4] kernel-broadcast.
  const p = { kind: 'measure', batchShape: [4] };
  assert.equal(ops._matchArgPattern(p, ops.argInfoFromMeasure(iidMt)),   false);
  assert.equal(ops._matchArgPattern(p, ops.argInfoFromMeasure(bcastMt)), true);
});

test('matchArgPattern: eventShape matches multivariate event dims', () => {
  // Hypothetical MvNormal-shaped measure with eventShape=[3].
  const mt = makeMeasureType({
    domain: { kind: 'array', rank: 1, shape: [3], elem: { kind: 'scalar', prim: 'real' } },
    sampleShape: [], batchShape: [], eventShape: [3],
  });
  const info = ops.argInfoFromMeasure(mt);
  assert.equal(ops._matchArgPattern({ kind: 'measure', eventShape: [3] }, info), true);
  assert.equal(ops._matchArgPattern({ kind: 'measure', eventShape: [4] }, info), false);
});

// =====================================================================
// 4. axisStack matching (P3a integration)
// =====================================================================

test('matchArgPattern: axisStack source constraint matches IR context', () => {
  const v = vec([1, 2, 3]);
  const stk = [{ source: 'iid', size: 5 }];
  const info = ops.argInfoFromValue(v, { axisStack: stk });
  assert.equal(ops._matchArgPattern({ axisStack: [{ source: 'iid' }] }, info), true);
  assert.equal(ops._matchArgPattern({ axisStack: [{ source: 'broadcast' }] }, info), false);
});

test('matchArgPattern: axisStack length must match', () => {
  const v = vec([1]);
  const stk = [{ source: 'iid', size: 5 }];
  const info = ops.argInfoFromValue(v, { axisStack: stk });
  // Pattern with 2 entries doesn't match a 1-entry stack.
  assert.equal(ops._matchArgPattern({
    axisStack: [{ source: 'iid' }, { source: 'kernel_broadcast' }],
  }, info), false);
});

test('matchArgPattern: axisStack wildcard source admits any', () => {
  const v = vec([1]);
  const stk = [{ source: 'kernel_broadcast', size: 50 }];
  const info = ops.argInfoFromValue(v, { axisStack: stk });
  assert.equal(ops._matchArgPattern({ axisStack: [{ source: '*' }] }, info), true);
  // Omitting source entirely also matches.
  assert.equal(ops._matchArgPattern({ axisStack: [{}] }, info), true);
});

// =====================================================================
// 5. Specificity ordering with the new dimensions
// =====================================================================

test('specificityScore: kind=measure adds a categorical filter', () => {
  const value   = { argPatterns: [{ rank: 1 }], impl: () => null };
  const measure = { argPatterns: [{ kind: 'measure' }], impl: () => null };
  // kind=measure (5) > rank=1 (1).
  assert.ok(ops._specificityScore(measure) > ops._specificityScore(value));
});

test('specificityScore: measure shape dims add to specificity', () => {
  const general = { argPatterns: [{ kind: 'measure' }], impl: () => null };
  const refined = { argPatterns: [{ kind: 'measure', sampleShape: [10] }], impl: () => null };
  assert.ok(ops._specificityScore(refined) > ops._specificityScore(general));
});

test('specificityScore: axisStack source weighted like tag (~per-arg-axis filter)', () => {
  const general = { argPatterns: [{}], impl: () => null };
  const axisCtx = { argPatterns: [{ axisStack: [{ source: 'kernel_broadcast' }] }], impl: () => null };
  assert.ok(ops._specificityScore(axisCtx) > ops._specificityScore(general));
});

test('pickVariant: kind=measure beats kind=value when arg is a measure', () => {
  const valuePat   = { argPatterns: [{ kind: 'value' }],   impl: () => 'value' };
  const measurePat = { argPatterns: [{ kind: 'measure' }], impl: () => 'measure' };
  const info = ops.argInfoFromMeasure(makeMeasureType({ eventShape: [] }));
  // Only measurePat matches (kind discriminator filters out valuePat).
  assert.equal(ops._pickVariant([valuePat, measurePat], [info], {}), measurePat);
});

// =====================================================================
// 6. Backward compatibility — raw Values into existing entry points
// =====================================================================

test('dispatch: raw Value still routes via auto-wrap (no change for legacy callers)', () => {
  // Use the existing broadcasted-mul variant — registered with raw
  // value-kind patterns; routing through dispatch with raw Values
  // must still work identically.
  const A = vec([1, 2, 3]);
  const B = vec([4, 5, 6]);
  const r = ops.dispatchVariant('mul', [A, B], { wrappingOp: 'broadcast' });
  assert.deepEqual(Array.from(r.data), [4, 10, 18]);
});

test('dispatchTyped: ArgInfo path matches dispatchVariant on equivalent value args', () => {
  // Verify the typed entry point produces the same result on
  // value-kind ArgInfos as dispatchVariant does on raw Values.
  const A = vec([1, 2, 3]);
  const B = vec([4, 5, 6]);
  const rRaw   = ops.dispatchVariant('mul', [A, B], { wrappingOp: 'broadcast' });
  const rTyped = ops.dispatchTyped('mul',
    [ops.argInfoFromValue(A), ops.argInfoFromValue(B)],
    { wrappingOp: 'broadcast' });
  assert.deepEqual(Array.from(rTyped.data), Array.from(rRaw.data));
});

// =====================================================================
// 7. End-to-end: register a measure-arg variant, dispatch it
// =====================================================================

test('end-to-end: register kind=measure variant matching iid sampleShape=[*]', () => {
  // Define a fictional op "iidLogpdfBatched" with one variant that
  // matches when the first arg is a measure with sampleShape=['*']
  // (i.e. some iid replication, any length).
  let invokedWith: any = null;
  ops.registerVariant('iidLogpdfBatched', {
    argPatterns: [{ kind: 'measure', sampleShape: ['*'], eventShape: [] }],
    impl: (args: any) => {
      invokedWith = args;
      return 'iid-batched';
    },
    label: 'iidLogpdfBatched(measure with iid replication)',
  });

  // Measure ArgInfo: `iid(Normal, 10)`-shaped measure type.
  const mt = makeMeasureType({ sampleShape: [10], batchShape: [], eventShape: [] });
  const info = ops.argInfoFromMeasure(mt);
  const r = ops.dispatchTyped('iidLogpdfBatched', [info]);
  assert.equal(r, 'iid-batched');
  assert.ok(invokedWith);
  assert.equal(invokedWith[0].kind, 'measure');

  // Same op, NO matching variant for a kernel-broadcast measure:
  const bcastMt = makeMeasureType({ sampleShape: [], batchShape: [4], eventShape: [] });
  const bcastInfo = ops.argInfoFromMeasure(bcastMt);
  const r2 = ops.dispatchTyped('iidLogpdfBatched', [bcastInfo]);
  assert.equal(r2, null, 'kernel-broadcast measure does NOT match iid-sample pattern');
});
