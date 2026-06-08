'use strict';

// Conformance regression for `mod` floor-modulo (spec §07) inside the
// VECTORISED AGGREGATE body path.
//
//   mod(a, b) = a − b·floor(a/b),  domain integers, b ≠ 0.
//
// sampler-aggregate.ts maintains its own scalar op table (`_AGG_BIN`)
// that feeds the broadcast-elementwise sweep used when an aggregate
// body contains arithmetic over indexed arrays (the `_liftAggregateExpr`
// dispatch). That table previously used JS `%` (truncated remainder,
// sign of the dividend) for `mod`, so e.g. `mod(-7, 3)` inside an
// aggregate returned -1 instead of the canonical 2.
//
// This file pins the aggregate body path specifically (NOT value-ops
// .modElem, which is covered by test/mod-conformance.test.ts) and
// asserts byte-for-byte agreement with value-ops.floorMod over a sweep
// of negative-dividend and mixed-sign cases, under BOTH the aggregate
// optimisation on (specialiser dispatch) and off (the broadcast-reduce
// reference loop that walks `_AGG_BIN` directly).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');
const sampler = require('../sampler.ts');
const lowerMod = require('../lower.ts');
const valueOps = require('../value-ops.ts');
const perfConfig = require('../perf-config.ts');

// Canonical reference: value-ops exports the engine's THE-definition
// floorMod. We assert the aggregate path equals it exactly.
const floorMod = valueOps.floorMod;

// Integer floor-division reference (spec §07): div(a, b) = floor(a/b),
// rounding toward −∞ (the separate `divide` op is real division). This
// is what the aggregate `_AGG_BIN.div` entry must compute.
const floorDiv = (a: number, b: number) => Math.floor(a / b);

// Lower `C = aggregate(...)` from source to IR and evaluate it through
// the sampler with `A`/`B` injected via env (mirrors aggregate.test.ts's
// evalAggregateRHS). The aggregate body `op(A[.i], B[.i])` has a single
// output axis and no reduce axis, so each output element is exactly
// `op(A[i], B[i])` evaluated through the `_AGG_BIN` dispatch.
function evalBinAggregate(op: string, A: number[], B: number[], optOn: boolean): number[] {
  const src = `C = aggregate(sum, [.i], ${op}(A[.i], B[.i]))`;
  const ctx = processSource(src);
  const errs = ctx.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0,
    `source must parse cleanly: ${errs.map((d: any) => d.message).join('; ')}`);
  const b = ctx.bindings.get('C');
  assert.ok(b, 'binding C not found');
  const ir = lowerMod.lowerExpr(b.node.value);
  const prev = perfConfig.setOptimization('aggregate', optOn);
  try {
    const got = sampler.evaluateExpr(ir, { A, B });
    return Array.from(got, (x: number) => x);
  } finally {
    perfConfig.setOptimization('aggregate', prev);
  }
}

// Spot cases the team-lead brief calls out explicitly. Each was wrong
// under the old JS `%` table (shown in the comments).
const DIVIDENDS = [-7, 7, -7, 8, 0, -1];
const DIVISORS = [3, -3, -3, 3, 5, 3];

for (const optOn of [false, true]) {
  const tag = optOn ? 'opt:aggregate=on' : 'opt:aggregate=off';

  test(`aggregate body mod matches floor-modulo on the brief's cases [${tag}]`, () => {
    const got = evalBinAggregate('mod', DIVIDENDS, DIVISORS, optOn);
    // mod(-7, 3) == 2  (JS `%` would give -1)
    assert.equal(got[0], 2);
    // mod(7, -3) == -2 (JS `%` would give 1)
    assert.equal(got[1], -2);
    // mod(-7, -3) == -1 (shares the sign of the divisor)
    assert.equal(got[2], -1);
    // mod(8, 3) == 2 (agrees with `%`)
    assert.equal(got[3], 2);
    // mod(0, 5) == 0
    assert.equal(got[4], 0);
    // mod(-1, 3) == 2 (JS `%` would give -1)
    assert.equal(got[5], 2);
  });

  test(`aggregate body mod equals value-ops floorMod over a sweep [${tag}]`, () => {
    const A: number[] = [];
    const B: number[] = [];
    const expected: number[] = [];
    for (let a = -12; a <= 12; a++) {
      for (const b of [1, 2, 3, 4, 5, -1, -2, -3, -4, -5]) {
        A.push(a);
        B.push(b);
        expected.push(floorMod(a, b));
      }
    }
    const got = evalBinAggregate('mod', A, B, optOn);
    assert.deepEqual(got, expected,
      'aggregate `mod` body must match value-ops.floorMod elementwise');
  });

  test(`aggregate body div matches floor-division on the brief's cases [${tag}]`, () => {
    // div(a, b) = floor(a/b); JS `/` would give the un-floored quotient.
    const A = [-7, 7, -7, 8, 0, -1];
    const B = [3, -3, -3, 3, 5, 3];
    const got = evalBinAggregate('div', A, B, optOn);
    // div(-7, 3) == -3 (JS `/` gives -2.33…)
    assert.equal(got[0], -3);
    // div(7, -3) == -3 (JS `/` gives -2.33…)
    assert.equal(got[1], -3);
    // div(-7, -3) == 2 (JS `/` gives 2.33…)
    assert.equal(got[2], 2);
    // div(8, 3) == 2
    assert.equal(got[3], 2);
    // div(0, 5) == 0
    assert.equal(got[4], 0);
    // div(-1, 3) == -1 (JS `/` gives -0.33…)
    assert.equal(got[5], -1);
  });

  test(`aggregate body div equals floor-division over a sweep [${tag}]`, () => {
    const A: number[] = [];
    const B: number[] = [];
    const expected: number[] = [];
    for (let a = -12; a <= 12; a++) {
      for (const b of [1, 2, 3, 4, 5, -1, -2, -3, -4, -5]) {
        A.push(a);
        B.push(b);
        expected.push(floorDiv(a, b));
      }
    }
    const got = evalBinAggregate('div', A, B, optOn);
    assert.deepEqual(got, expected,
      'aggregate `div` body must match floor-division elementwise');
  });
}
