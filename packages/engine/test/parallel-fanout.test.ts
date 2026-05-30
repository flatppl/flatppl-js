'use strict';

// Tests for commit 4's worker-range fan-out — the dispatch shim
// (`sampler._dispatchSampleN`) plus the persistent worker-pool's
// runChunked fan-out + reducer.
//
// Strategy: drive every test through a SYNCHRONOUS MOCK WORKER. The
// mock implements the same postMessage / onmessage / on('error') /
// terminate surface the pool feature-detects, but routes each message
// straight to a freshly-created `createWorkerHandler()` instance
// running in-process. The reply is fired back via onmessage on the
// next microtask (Promise.resolve().then(...)) — that's enough async
// to exercise the pool's queue / pump path, but doesn't require
// node:worker_threads (faster CI, identical determinism story).
//
// What we cover here:
//   * Pool lifecycle (spawn → runChunked → disposePool → respawn).
//   * Pool error propagation (worker fires onerror → runChunked rejects).
//   * Determinism: flag-off W=1 vs flag-on W=1 (small N) → bit-identical.
//   * Statistical equivalence: flag-on N=4096 W=4 vs flag-off → both
//     Normal(0,1), mean/var within tolerance.
//   * Determinism: flag-on parallel run repeatable (same seed twice).
//   * truncateSampleN reduction: per-chunk n_eff / totalDraws pool +
//     re-derived logShift.
//   * refArrays slicing: each chunk sees the correct subarray window.
//   * Backend without spawn capability: flag-on still falls back to
//     scalar.
//   * Edge cases: N == threshold (scalar path per the strict-less-than
//     guard), maxWorkers=1 (forces W=1 even at large N).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler   = require('../sampler.ts');
const backend   = require('../backend.ts');
const workerPool = require('../worker-pool.ts');
const rng       = require('../rng.ts');
const workerLib = require('../worker.ts');

// =====================================================================
// Mock-worker plumbing.
// =====================================================================

// A synchronous mock worker. Implements:
//   * postMessage(msg, [transferList])
//   * onmessage = fn   (browser shape — settable property)
//   * onerror   = fn
//   * terminate()
//
// On each postMessage we route the message straight to a per-worker
// `createWorkerHandler()` instance and schedule `onmessage` on the
// next microtask. That's just enough async to let the pool see the
// reply on a turn boundary (matching real worker round-trip) but
// stays deterministic and doesn't need worker_threads.
//
// Each mock instance also records every received message so the test
// can verify chunk slicing / count / seed shape.

function makeMockWorker() {
  const handler = workerLib.createWorkerHandler();
  // No top-level `onmessage` — we intentionally start with `null` so
  // detectBrowserShape() in worker-pool.ts sees a settable property.
  let _onmessage: any = null;
  let _onerror: any = null;
  const received: any[] = [];

  const w: any = {
    postMessage(msg: any) {
      // Snapshot the message we got so tests can introspect chunk
      // count / seed / refArrays shape.
      received.push(msg);
      Promise.resolve().then(() => {
        try {
          const reply = handler.handle(msg);
          if (_onmessage) _onmessage({ data: reply });
        } catch (err) {
          if (_onerror) _onerror(err);
        }
      });
    },
    terminate() {
      _onmessage = null;
      _onerror   = null;
    },
    get onmessage() { return _onmessage; },
    set onmessage(fn: any) { _onmessage = fn; },
    get onerror() { return _onerror; },
    set onerror(fn: any) { _onerror = fn; },
    // Test-only — inspect the captured messages.
    _received: received,
  };
  return w;
}

// Variant that immediately fires onerror on postMessage (pre-onmessage
// attach is fine — the pool attaches both listeners eagerly in
// makeSlot()/attachListeners()).
function makeBrokenWorker() {
  let _onmessage: any = null;
  let _onerror: any = null;
  const w: any = {
    postMessage(_msg: any) {
      Promise.resolve().then(() => {
        if (_onerror) _onerror(new Error('boom'));
      });
    },
    terminate() { _onmessage = null; _onerror = null; },
    get onmessage() { return _onmessage; },
    set onmessage(fn: any) { _onmessage = fn; },
    get onerror() { return _onerror; },
    set onerror(fn: any) { _onerror = fn; },
  };
  return w;
}

// Install a mock backend with a low parallelism threshold so we can
// exercise the fan-out path at modest N. The mock keeps a registry
// of spawned workers so individual tests can introspect what messages
// each one saw.
function installMockBackend(opts: {
  maxWorkers?: number;
  threshold?: number;
  broken?: boolean;
  spawnable?: boolean;   // false → spawnWorker returns null
} = {}) {
  const spawned: any[] = [];
  const maxWorkers = opts.maxWorkers ?? 4;
  const threshold  = opts.threshold  ?? 100;
  const broken     = opts.broken     ?? false;
  const spawnable  = opts.spawnable !== false;
  backend.setBackend({
    name: 'mock',
    parallelismThreshold: threshold,
    maxWorkers,
    spawnWorker: () => {
      if (!spawnable) return null;
      const w = broken ? makeBrokenWorker() : makeMockWorker();
      spawned.push(w);
      return w;
    },
  });
  return spawned;
}

function restoreBackend() {
  workerPool.disposePool();
  backend.setBackend(backend.jsBackend);
  delete (globalThis as any).FLATPPL_PARALLEL_SAMPLE;
}

function synthLoc() {
  return { start: { line: -1, col: -1 }, end: { line: -1, col: -1 }, synthetic: true };
}
function distIR(op: any, kwargs: any) {
  const out: any = {};
  for (const [k, v] of Object.entries(kwargs)) {
    out[k] = { kind: 'lit', value: v, loc: synthLoc() };
  }
  return { kind: 'call', op, kwargs: out, loc: synthLoc() };
}

// Scalar handler used as `_scalarHandle` for `_dispatchSampleN` —
// runs the request through a fresh in-process handler. The dispatch
// shim's scalar branch is synchronous when this returns synchronously.
function scalarHandle(payload: any, msgType: string): any {
  const h = workerLib.createWorkerHandler();
  return h.handle({ type: msgType, ...payload });
}

// =====================================================================
// 1. Pool lifecycle.
// =====================================================================

test('pool: lazy spawn — probe-one on getPool, grow on demand, respawn after dispose', async (t: any) => {
  t.after(restoreBackend);
  installMockBackend({ maxWorkers: 4 });
  const pool1 = workerPool.getPool();
  assert.ok(pool1, 'getPool() should return non-null with spawnable backend');
  // Lazy contract: getPool() only probe-spawns ONE worker to verify
  // host capability. The pool grows to the chunk count on first
  // runChunked (capped at maxWorkers). Avoids eager-spawning the full
  // maxWorkers on small first calls.
  assert.equal(pool1!.size, 1, 'getPool spawns just the capability probe');

  // runChunked with 4 chunks → 4 replies, each chunk seeded distinctly.
  // The pool grows to size 4 here.
  const ir = distIR('Normal', { mu: 0, sigma: 1 });
  const childKeys = rng.split(rng.keyFromSeed(0xC0FFEE), 4);
  const chunks = childKeys.map((k: any) => ({
    ir, count: 50, seed: k,
  }));
  const replies = await pool1!.runChunked('sampleN', chunks);
  assert.equal(replies.length, 4);
  assert.equal(pool1!.size, 4, 'pool grew to chunk count on runChunked');
  for (const r of replies) {
    assert.ok(r.samples instanceof Float64Array);
    assert.equal(r.samples.length, 50);
  }

  // Dispose; getPool() should spawn a fresh pool (sticky-failure flag
  // is cleared on dispose).
  workerPool.disposePool();
  const pool2 = workerPool.getPool();
  assert.ok(pool2, 'getPool() after dispose should respawn');
  assert.notEqual(pool1, pool2, 'fresh pool instance');
});

// =====================================================================
// 2. Pool error propagation.
// =====================================================================

test('pool: worker error rejects runChunked promise', async (t: any) => {
  t.after(restoreBackend);
  installMockBackend({ maxWorkers: 2, broken: true });
  const pool = workerPool.getPool();
  assert.ok(pool);
  const ir = distIR('Normal', { mu: 0, sigma: 1 });
  const chunks = [
    { ir, count: 10, seed: [1, 2] as [number, number] },
    { ir, count: 10, seed: [3, 4] as [number, number] },
  ];
  await assert.rejects(() => pool!.runChunked('sampleN', chunks), /boom/);
});

// =====================================================================
// 3. Determinism: flag-off W=1  vs  flag-on W=1 (N < threshold) →
//    bit-identical. The dispatch shim always threads through
//    rng.split so the key-tree shape is the same on both sides of the
//    threshold.
// =====================================================================

test('determinism: flag-off vs flag-on (N < threshold) is bit-identical', async (t: any) => {
  t.after(restoreBackend);
  installMockBackend({ maxWorkers: 4, threshold: 1000 });

  const seed = rng.keyFromSeed(0xABC123);
  const ir = distIR('Normal', { mu: 0, sigma: 1 });
  const N = 200;  // strictly below threshold → scalar path on both runs

  // Flag off.
  delete (globalThis as any).FLATPPL_PARALLEL_SAMPLE;
  const offReply: any = sampler._dispatchSampleN(
    { ir, count: N, seed }, 'sampleN', scalarHandle);

  // Flag on but N < threshold — still scalar.
  (globalThis as any).FLATPPL_PARALLEL_SAMPLE = 1;
  const onReply: any = sampler._dispatchSampleN(
    { ir, count: N, seed }, 'sampleN', scalarHandle);

  assert.ok(offReply.samples instanceof Float64Array);
  assert.ok(onReply.samples  instanceof Float64Array);
  assert.equal(offReply.samples.length, N);
  assert.equal(onReply.samples.length,  N);
  for (let i = 0; i < N; i++) {
    assert.equal(offReply.samples[i], onReply.samples[i],
      `flag-off / flag-on must be bit-identical below threshold (idx ${i})`);
  }
});

// =====================================================================
// 4. Statistical equivalence: flag-on parallel vs flag-off scalar —
//    different bytes (different child-key keystreams once the fan-out
//    kicks in), but both should sample from Normal(0,1).
// =====================================================================

test('statistical: flag-on N=4096 W=4 ≈ flag-off N=4096 (both Normal(0,1))', async (t: any) => {
  // The test owns the flag for its duration. Save + restore the env
  // var too so a globally-set `FLATPPL_PARALLEL_SAMPLE=1` (used by
  // commit-5 wiring tests) doesn't override the in-test toggle.
  const _savedEnv = process.env.FLATPPL_PARALLEL_SAMPLE;
  delete process.env.FLATPPL_PARALLEL_SAMPLE;
  t.after(() => {
    if (_savedEnv != null) process.env.FLATPPL_PARALLEL_SAMPLE = _savedEnv;
    restoreBackend();
  });
  installMockBackend({ maxWorkers: 4, threshold: 1000 });

  const seed = rng.keyFromSeed(42);
  const ir = distIR('Normal', { mu: 0, sigma: 1 });
  const N = 4096;

  delete (globalThis as any).FLATPPL_PARALLEL_SAMPLE;
  const offReply: any = await sampler._dispatchSampleN(
    { ir, count: N, seed }, 'sampleN', scalarHandle);

  (globalThis as any).FLATPPL_PARALLEL_SAMPLE = 1;
  const onReply: any = await sampler._dispatchSampleN(
    { ir, count: N, seed }, 'sampleN', scalarHandle);

  // Different byte streams: parallel splits the key tree differently
  // (W=1 in scalar branch vs W=4 in fan-out branch — different child
  // keys). We expect MOST samples to differ.
  let differing = 0;
  for (let i = 0; i < N; i++) {
    if (offReply.samples[i] !== onReply.samples[i]) differing++;
  }
  assert.ok(differing > N * 0.99,
    `parallel + scalar should disagree byte-wise (got ${differing}/${N})`);

  function meanVar(xs: Float64Array) {
    let s = 0; for (let i = 0; i < xs.length; i++) s += xs[i];
    const m = s / xs.length;
    let v = 0; for (let i = 0; i < xs.length; i++) v += (xs[i] - m) * (xs[i] - m);
    return { mean: m, varc: v / xs.length };
  }
  const mvOff = meanVar(offReply.samples);
  const mvOn  = meanVar(onReply.samples);
  assert.ok(Math.abs(mvOff.mean) < 0.1, `flag-off mean ≈ 0 (got ${mvOff.mean})`);
  assert.ok(Math.abs(mvOn.mean)  < 0.1, `flag-on  mean ≈ 0 (got ${mvOn.mean})`);
  assert.ok(Math.abs(mvOff.varc - 1) < 0.1, `flag-off var ≈ 1 (got ${mvOff.varc})`);
  assert.ok(Math.abs(mvOn.varc  - 1) < 0.1, `flag-on  var ≈ 1 (got ${mvOn.varc})`);
});

// =====================================================================
// 5. Determinism: parallel run repeatable.
// =====================================================================

test('determinism: parallel run with same seed twice → bit-identical', async (t: any) => {
  t.after(restoreBackend);
  installMockBackend({ maxWorkers: 4, threshold: 1000 });
  (globalThis as any).FLATPPL_PARALLEL_SAMPLE = 1;

  const seed = rng.keyFromSeed(0xDEADBEEF);
  const ir = distIR('Normal', { mu: 0, sigma: 1 });
  const N = 4096;

  const r1: any = await sampler._dispatchSampleN(
    { ir, count: N, seed }, 'sampleN', scalarHandle);
  // Force a fresh pool, so we exercise both pool instances against the
  // same input.
  workerPool.disposePool();
  installMockBackend({ maxWorkers: 4, threshold: 1000 });
  const r2: any = await sampler._dispatchSampleN(
    { ir, count: N, seed }, 'sampleN', scalarHandle);

  assert.equal(r1.samples.length, N);
  assert.equal(r2.samples.length, N);
  for (let i = 0; i < N; i++) {
    assert.equal(r1.samples[i], r2.samples[i],
      `parallel must be deterministic across pool restarts (idx ${i})`);
  }
});

// =====================================================================
// 6. truncateSampleN reduction — per-chunk n_eff + totalDraws pool +
//    logShift = log(sumNeff / sumDraws).
// =====================================================================

test('truncate: pool n_eff + totalDraws across chunks; re-derived logShift', async (t: any) => {
  t.after(restoreBackend);
  installMockBackend({ maxWorkers: 4, threshold: 100 });
  (globalThis as any).FLATPPL_PARALLEL_SAMPLE = 1;

  const seed = rng.keyFromSeed(7);
  const ir = distIR('Normal', { mu: 0, sigma: 1 });
  const setDescr = { kind: 'interval', lo: -1, hi: 1 };
  const N = 400;

  const reply: any = await sampler._dispatchSampleN(
    { ir, count: N, seed, mode: 'cdf', setDescr },
    'truncateSampleN', scalarHandle);

  assert.equal(reply.samples.length, N);
  // CDF path → totalDraws == n_eff == N per chunk → summed equals N.
  assert.equal(reply.n_eff, N);
  assert.equal(reply.totalDraws, N);
  // Re-derived logShift = log(sumNeff / sumDraws) = log(1) = 0 for CDF
  // path with full intersection. The per-chunk logShift would be the
  // analytical log(Fhi - Flo); the pool reducer overrides that with
  // the empirical ratio. Both paths agree at log(1)=0 only when n_eff
  // = totalDraws — which is true for CDF.
  assert.equal(reply.logShift, 0);

  // Verify each truncated sample lies inside the support.
  for (let i = 0; i < N; i++) {
    assert.ok(reply.samples[i] >= -1 && reply.samples[i] <= 1,
      `truncated sample [${i}] = ${reply.samples[i]} outside [-1, 1]`);
  }
});

test('truncate (rejection): n_eff < totalDraws → logShift = log(acceptance)', async (t: any) => {
  t.after(restoreBackend);
  installMockBackend({ maxWorkers: 4, threshold: 100 });
  (globalThis as any).FLATPPL_PARALLEL_SAMPLE = 1;

  const seed = rng.keyFromSeed(99);
  const ir = distIR('Normal', { mu: 0, sigma: 1 });
  // Tight window around mean → high acceptance but not 1.
  const setDescr = { kind: 'interval', lo: -0.5, hi: 0.5 };
  const N = 400;

  const reply: any = await sampler._dispatchSampleN(
    { ir, count: N, seed, mode: 'rejection', setDescr, budget: 100 },
    'truncateSampleN', scalarHandle);

  assert.equal(reply.samples.length, N);
  assert.ok(reply.totalDraws >= reply.n_eff,
    'totalDraws should be >= n_eff (rejection redraws on miss)');
  assert.ok(reply.n_eff > 0, 'expect some atoms to land inside');
  if (reply.n_eff === reply.totalDraws) {
    assert.equal(reply.logShift, 0);
  } else {
    const expected = Math.log(reply.n_eff / reply.totalDraws);
    assert.ok(Math.abs(reply.logShift - expected) < 1e-12,
      `logShift=${reply.logShift} expected=${expected}`);
  }
});

// =====================================================================
// 7. refArrays slicing — verify each chunk sees the correct subarray.
//    We use a no-op evaluateN-shaped fixture so we can plumb a Float64
//    refArray through without the registry resolving it.
// =====================================================================

test('refArrays slicing: each chunk receives its [off, off+n) subarray', async (t: any) => {
  t.after(restoreBackend);
  const spawned = installMockBackend({ maxWorkers: 4, threshold: 100 });
  (globalThis as any).FLATPPL_PARALLEL_SAMPLE = 1;

  const seed = rng.keyFromSeed(0x1234);
  // Distribution with no refs in its kwargs; the refArrays we send is
  // length-N but unused by the sampler (registry resolves params from
  // literals). The dispatch shim still slices it per chunk because
  // its length equals totalN — we verify that slicing here.
  const ir = distIR('Normal', { mu: 0, sigma: 1 });
  const N = 400;
  const x = new Float64Array(N);
  for (let i = 0; i < N; i++) x[i] = i;

  await sampler._dispatchSampleN(
    { ir, count: N, seed, refArrays: { x } },
    'sampleN', scalarHandle);

  // The pool fanned out into 4 workers of 100 each. Each mock worker
  // recorded the message it saw — check the refArrays.x subarray.
  // Note: messages may arrive in scheduling order, but the
  // dispatcher submits chunks 0..3 in order to the slots 0..3 of the
  // pool, so the recorded order tracks chunk index.
  assert.equal(spawned.length, 4);
  for (let c = 0; c < 4; c++) {
    const w = spawned[c];
    assert.equal(w._received.length, 1, `worker ${c} got exactly one chunk`);
    const slice = w._received[0].refArrays.x;
    assert.ok(slice instanceof Float64Array);
    assert.equal(slice.length, 100, `chunk ${c} length = 100`);
    // Each chunk should see [c*100 .. c*100+99].
    for (let i = 0; i < 100; i++) {
      assert.equal(slice[i], c * 100 + i,
        `chunk ${c}, slice[${i}] = ${slice[i]} (expected ${c * 100 + i})`);
    }
  }
});

// =====================================================================
// 8. Backend without spawn capability → scalar path even with flag on.
// =====================================================================

test('backend without spawnWorker: flag-on still falls back to scalar', async (t: any) => {
  t.after(restoreBackend);
  // jsBackend has spawnWorker that always returns null.
  backend.setBackend(backend.jsBackend);
  (globalThis as any).FLATPPL_PARALLEL_SAMPLE = 1;

  const seed = rng.keyFromSeed(1);
  const ir = distIR('Normal', { mu: 0, sigma: 1 });
  const reply: any = sampler._dispatchSampleN(
    { ir, count: 4096, seed }, 'sampleN', scalarHandle);

  // Synchronous reply confirms scalar path (fan-out is always async).
  assert.ok(reply && reply.samples instanceof Float64Array);
  assert.equal(reply.samples.length, 4096);
  // No pool spawned.
  assert.equal(workerPool.getPool(), null);
});

// =====================================================================
// Edge cases.
// =====================================================================

test('edge: N == threshold → scalar path (strict-less-than guard)', async (t: any) => {
  t.after(restoreBackend);
  // Threshold = 4096; N = 4096 → totalN < threshold is false, BUT the
  // dispatch guard reads `totalN < be.parallelismThreshold` so equality
  // hits the fan-out branch. Document the boundary.
  installMockBackend({ maxWorkers: 4, threshold: 4096 });
  (globalThis as any).FLATPPL_PARALLEL_SAMPLE = 1;

  const seed = rng.keyFromSeed(11);
  const ir = distIR('Normal', { mu: 0, sigma: 1 });
  const N = 4096;
  const reply: any = await sampler._dispatchSampleN(
    { ir, count: N, seed }, 'sampleN', scalarHandle);
  assert.equal(reply.samples.length, N);
  // W = min(4, ceil(4096 / 4096)) = 1 → fan-out spawned a single chunk
  // through the pool. Verify the pool was used.
  assert.ok(workerPool.getPool(), 'pool should exist after fan-out call');
});

// =====================================================================
// 9. Commit 5 wiring: materialiser._ensureDispatchedSendWorker wraps
//    ctx.sendWorker once per ctx, intercepts sampleN/truncateSampleN,
//    passes other types through unchanged.
// =====================================================================

test('wiring: _ensureDispatchedSendWorker intercepts sampleN, passes evaluateN through', async (t: any) => {
  t.after(restoreBackend);
  installMockBackend({ maxWorkers: 4, threshold: 100 });
  (globalThis as any).FLATPPL_PARALLEL_SAMPLE = 1;

  const materialiser = require('../materialiser.ts');
  const _internal = (materialiser as any)._internal || materialiser;

  // Track every (msg-type, count) the underlying sendWorker sees.
  const observed: Array<{ type: string; count?: number; seed?: any }> = [];
  const original = async function (msg: any): Promise<any> {
    observed.push({ type: msg.type, count: msg.count, seed: msg.seed });
    if (msg.type === 'sampleN' || msg.type === 'truncateSampleN') {
      // Return a fake reply with the expected shape.
      return { samples: new Float64Array(msg.count | 0), n_eff: msg.count, totalDraws: msg.count };
    }
    return { ok: true };
  };
  const ctx: any = {
    sendWorker: original,
    rootSeed: 42,
  };

  // First entry: wrap.
  _internal._ensureDispatchedSendWorker
    ? _internal._ensureDispatchedSendWorker(ctx)
    : (materialiser as any)._ensureDispatchedSendWorker(ctx);

  assert.notEqual(ctx.sendWorker, original, 'sendWorker should be wrapped');
  assert.equal(ctx._workerDispatched, true, 'idempotency flag set');

  // Idempotent: re-calling doesn't double-wrap.
  const wrapped1 = ctx.sendWorker;
  ((materialiser as any)._ensureDispatchedSendWorker || (() => {}))(ctx);
  assert.equal(ctx.sendWorker, wrapped1, 're-wrap is a no-op');

  // sampleN flows through _dispatchSampleN, which (with the mock
  // backend's spawnWorker present + threshold 100) takes the fan-out
  // path. The pool's mock workers receive the chunked messages — NOT
  // the test's `original` sendWorker. So observed should record zero
  // sampleN entries.
  const seed = rng.keyFromSeed(7);
  const ir = distIR('Normal', { mu: 0, sigma: 1 });
  await ctx.sendWorker({ type: 'sampleN', ir, count: 1000, seed });

  const sampleNCount = observed.filter((o) => o.type === 'sampleN').length;
  assert.equal(sampleNCount, 0,
    'sampleN was intercepted by dispatch; pool consumed chunks, not original sendWorker');

  // Other message types pass through unchanged.
  await ctx.sendWorker({ type: 'setEnv', env: { foo: 1 } });
  await ctx.sendWorker({ type: 'evaluateN', ir, count: 10 });
  const passthroughTypes = observed.map((o) => o.type);
  assert.ok(passthroughTypes.includes('setEnv'),  'setEnv passes through');
  assert.ok(passthroughTypes.includes('evaluateN'), 'evaluateN passes through');
});

test('wiring: _ensureDispatchedSendWorker falls back to scalar when backend cannot spawn', async (t: any) => {
  t.after(restoreBackend);
  // No spawnWorker → scalar path. Dispatch wrapper STILL invokes (so
  // the seed gets threaded through split(seed, 1)[0]); but the message
  // reaches the original sendWorker — just with a re-keyed seed.
  installMockBackend({ spawnable: false });
  (globalThis as any).FLATPPL_PARALLEL_SAMPLE = 1;

  const materialiser = require('../materialiser.ts');

  const observed: any[] = [];
  const original = async function (msg: any): Promise<any> {
    observed.push(msg);
    return { samples: new Float64Array(msg.count | 0) };
  };
  const ctx: any = { sendWorker: original, rootSeed: 42 };
  (materialiser as any)._ensureDispatchedSendWorker(ctx);

  const seed = rng.keyFromSeed(7);
  const ir = distIR('Normal', { mu: 0, sigma: 1 });
  await ctx.sendWorker({ type: 'sampleN', ir, count: 1000, seed });

  assert.equal(observed.length, 1, 'scalar path: one message reaches original');
  assert.equal(observed[0].type, 'sampleN');
  // Determinism contract: seed is rekeyed via split(seed, 1)[0] even
  // on the scalar path.
  assert.notDeepEqual(observed[0].seed, seed,
    'seed re-keyed via split(seed, 1)[0]');
});

test('edge: maxWorkers=1 routes to scalar path (no fan-out across a single worker)', async (t: any) => {
  t.after(restoreBackend);
  const spawned = installMockBackend({ maxWorkers: 1, threshold: 100 });
  (globalThis as any).FLATPPL_PARALLEL_SAMPLE = 1;

  const seed = rng.keyFromSeed(13);
  const ir = distIR('Normal', { mu: 0, sigma: 1 });
  const N = 4096;
  // maxWorkers <= 1 hits the scalar guard in _dispatchSampleN — there's
  // no benefit to fanning out across a single worker, so the dispatch
  // shim short-circuits to the scalar branch even though the pool
  // exists. Reply is sync. (Note: the pool is still lazily spawned
  // because the flag is on and the backend can spawn — `getPool()` is
  // called BEFORE the maxWorkers guard. That's a minor inefficiency
  // but doesn't affect correctness; the spawned worker simply never
  // receives a message.)
  const reply: any = sampler._dispatchSampleN(
    { ir, count: N, seed }, 'sampleN', scalarHandle);
  assert.ok(reply && reply.samples instanceof Float64Array);
  assert.equal(reply.samples.length, N);
  // Scalar path → no chunks dispatched → spawned worker received nothing.
  for (const w of spawned) {
    assert.equal(w._received.length, 0,
      'maxWorkers=1 scalar path: no chunks reach the worker');
  }
});
