'use strict';

// =====================================================================
// Demand-driven fixed-value boundary (engine-concepts §17.4).
// =====================================================================
//
// fixed-eval has TWO failure modes, not one:
//
//   undefined    — a value isn't statically knowable (an input depends
//                  on external / elementof / draw / a measure / a
//                  non-fixed binding). A shape that needs it is
//                  legitimately %dynamic — NOT an error.
//   UNSUPPORTED  — every input resolved (so the computation is
//                  fixed-phase and computable in principle) but the op
//                  isn't implemented in simple-eval mode. A shape that
//                  needs it raises a clear "could not compute <name>"
//                  error rather than silently degrading to %dynamic.
//
// These tests pin the distinction (resolver mechanism) and the
// typeinfer wiring (UNSUPPORTED at a shape position → clear error).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const fixedEval = require('../fixed-eval.ts');
const typeinfer = require('../typeinfer.ts');
const engine = require('../index.ts');

const lit = (v: any) => ({ kind: 'lit', value: v });
const refSelf = (name: any) => ({ kind: 'ref', ns: 'self', name });

test('fixed-eval: arithmetic over literals resolves', () => {
  const resolve = fixedEval.makeResolver({});
  assert.equal(
    resolve({ kind: 'call', op: 'add', args: [lit(2), lit(3)] }), 5);
});

test('fixed-eval: unsupported op with resolved inputs → UNSUPPORTED sentinel', () => {
  const resolve = fixedEval.makeResolver({});
  const r = resolve({ kind: 'call', op: '__no_such_op__', args: [lit(1)] });
  assert.equal(r, fixedEval.UNSUPPORTED, 'expected the UNSUPPORTED sentinel');
  // The sentinel is carried ON the resolver so callers recognise it
  // by identity without importing fixed-eval.
  assert.equal(r, (resolve as any).UNSUPPORTED);
});

test('fixed-eval: an unresolvable INPUT → undefined (DYNAMIC), not UNSUPPORTED', () => {
  // add(<unknown ref>, 1): the ref has no binding/value → not statically
  // knowable → undefined. This is the legitimately-%dynamic case, which
  // must stay distinct from the op-gap (UNSUPPORTED) case.
  const resolve = fixedEval.makeResolver({});
  const r = resolve({ kind: 'call', op: 'add', args: [refSelf('missing'), lit(1)] });
  assert.equal(r, undefined);
  assert.notEqual(r, fixedEval.UNSUPPORTED);
});

test('typeinfer: UNSUPPORTED at a shape position → clear "could not compute" error', () => {
  // `fill(0.0, n)` resolves its shape from `n`. With a resolver that
  // reports `n` as UNSUPPORTED (an op-gap), typeinfer must raise the
  // clear error naming the binding — not silently produce %dynamic.
  const src = `
flatppl_compat = "0.1"
n = 5
xs = fill(0.0, n)
`;
  const r = engine.processSource(src);
  // Mock resolver: report the size ref `n` as UNSUPPORTED, resolve
  // nothing else. (.UNSUPPORTED carried on the fn, as fixed-eval does.)
  const UNSUP = Symbol('mock-unsupported');
  const mock: any = (ir: any) =>
    (ir && ir.kind === 'ref' && ir.name === 'n') ? UNSUP : undefined;
  mock.UNSUPPORTED = UNSUP;
  const diags = typeinfer.inferTypes(r.loweredModule, { resolveFixed: mock });
  const errs = diags.filter((d: any) => d.severity === 'error');
  const hit = errs.find((d: any) =>
    /could not compute the value of 'n'/.test(d.message)
    && /not supported in fixed-phase/.test(d.message));
  assert.ok(hit,
    'expected a clear "could not compute the value of n" error, got: '
    + JSON.stringify(errs.map((d: any) => d.message)));
});

test('typeinfer: a resolver returning undefined for the size stays %dynamic (no error)', () => {
  // The legitimately-dynamic counterpart: resolver can't fold `n`
  // (returns undefined). The shape is %dynamic and NO error is raised.
  const src = `
flatppl_compat = "0.1"
n = 5
xs = fill(0.0, n)
`;
  const r = engine.processSource(src);
  const mock: any = () => undefined;
  mock.UNSUPPORTED = Symbol('unused');
  const diags = typeinfer.inferTypes(r.loweredModule, { resolveFixed: mock });
  const errs = diags.filter((d: any) =>
    d.severity === 'error' && /could not compute/.test(d.message));
  assert.equal(errs.length, 0, 'no "could not compute" error for a dynamic size');
});
