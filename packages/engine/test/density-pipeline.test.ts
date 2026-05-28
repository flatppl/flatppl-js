'use strict';

// =====================================================================
// density-pipeline.test.ts — P3b density-walker pipeline
// =====================================================================
//
// Mirrors materialiser-pipeline.test.ts for the density side. Pins
// the composable-stage pipeline added by P3b (engine-concepts §18.11
// / §20.10.5 item 5).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const density = require('../density.ts');

function freshPipeline() {
  density._resetDensityPipeline();
}

// =====================================================================
// 1. Default pipeline (empty stages) runs densityCoreStage directly
// =====================================================================

test('default pipeline: empty stages → densityCoreStage runs', () => {
  freshPipeline();
  // Register a stage that short-circuits with a fake result (we
  // don't need to drive a real density evaluation — just confirm
  // _runDensityPipeline reaches the stage we register).
  let stageRan = false;
  density.registerDensityStage(function (
    _ir: any, _value: any, _refArrays: any, count: any, _opts: any, _next: any
  ) {
    stageRan = true;
    return { logps: new Float64Array(count), rest: null };
  });
  const r = density._runDensityPipeline(
    { kind: 'call', op: 'Normal' }, 1.0, {}, 2, {});
  assert.equal(stageRan, true);
  assert.equal(r.logps.length, 2);
});

// =====================================================================
// 2. Pre-dispatch stage forwards via next()
// =====================================================================

test('pipeline: pre-dispatch stage observes call + forwards', () => {
  freshPipeline();
  const calls: string[] = [];
  density.registerDensityStage(function (
    ir: any, value: any, refArrays: any, count: any, opts: any, next: any
  ) {
    calls.push('outer-pre');
    const r = next(ir, value, refArrays, count, opts);
    calls.push('outer-post');
    return r;
  });
  // Terminal short-circuit stage.
  density.registerDensityStage(function (
    _ir: any, _value: any, _refArrays: any, count: any, _opts: any, _next: any
  ) {
    calls.push('terminal');
    return { logps: new Float64Array(count), rest: null };
  });
  density._runDensityPipeline({ kind: 'call', op: 'Normal' }, 1.0, {}, 3, {});
  assert.deepEqual(calls, ['outer-pre', 'terminal', 'outer-post']);
});

// =====================================================================
// 3. Short-circuit cache stage example (demonstrates the use case)
// =====================================================================

test('density caching stage: skips next() on cache hit', () => {
  freshPipeline();
  // Trivial cache keyed by (ir.op, value, count). For real use the
  // key would include IR-shape hashing; this stub demonstrates the
  // pipeline placement.
  const cache: Record<string, any> = {};
  let coreCalls = 0;
  density.registerDensityStage(function (
    ir: any, value: any, refArrays: any, count: any, opts: any, next: any
  ) {
    const key = JSON.stringify({ op: ir.op, value, count });
    if (key in cache) return cache[key];
    const r = next(ir, value, refArrays, count, opts);
    cache[key] = r;
    return r;
  });
  density.registerDensityStage(function (
    _ir: any, _value: any, _refArrays: any, count: any, _opts: any, _next: any
  ) {
    coreCalls++;
    return { logps: new Float64Array(count).fill(0.5), rest: null };
  });
  const ir = { kind: 'call', op: 'Normal' };
  const r1 = density._runDensityPipeline(ir, 1.0, {}, 4, {});
  const r2 = density._runDensityPipeline(ir, 1.0, {}, 4, {});  // cache hit
  assert.equal(coreCalls, 1, 'core only runs once; second call hits cache');
  assert.strictEqual(r1, r2, 'cached result is the same object');
});

// =====================================================================
// 4. Tracing stage on the density side (analogous to materialiser's)
// =====================================================================

test('density tracing stage: records per-call duration via custom now()', () => {
  freshPipeline();
  const buffer: any[] = [];
  let t = 0;
  density.registerDensityStage(function (
    ir: any, value: any, refArrays: any, count: any, opts: any, next: any
  ) {
    const t0 = (t += 5);
    const r = next(ir, value, refArrays, count, opts);
    const t1 = (t += 11);
    buffer.push({ op: ir.op, durationMs: t1 - t0, ok: true });
    return r;
  });
  density.registerDensityStage(function (
    _ir: any, _value: any, _refArrays: any, count: any, _opts: any, _next: any
  ) {
    return { logps: new Float64Array(count), rest: null };
  });
  density._runDensityPipeline({ kind: 'call', op: 'Normal' }, 1.0, {}, 2, {});
  density._runDensityPipeline({ kind: 'call', op: 'Poisson' }, 3, {}, 2, {});
  assert.equal(buffer.length, 2);
  assert.equal(buffer[0].op, 'Normal');
  assert.equal(buffer[1].op, 'Poisson');
  assert.equal(buffer[0].durationMs, 11);
});

// =====================================================================
// 5. _resetDensityPipeline clears stages
// =====================================================================

test('_resetDensityPipeline clears registered stages', () => {
  freshPipeline();
  let counter = 0;
  density.registerDensityStage(function (
    ir: any, value: any, refArrays: any, count: any, opts: any, next: any
  ) {
    counter++;
    return next(ir, value, refArrays, count, opts);
  });
  density._resetDensityPipeline();
  density.registerDensityStage(function (
    _ir: any, _value: any, _refArrays: any, count: any, _opts: any, _next: any
  ) {
    return { logps: new Float64Array(count), rest: null };
  });
  density._runDensityPipeline({ kind: 'call', op: 'Normal' }, 1.0, {}, 1, {});
  assert.equal(counter, 0, 'reset removed pre-reset stage');
});
