// packages/engine/test/pool-diagnostics.test.ts
// Freeze-then-parallel mh distributes full-length chains across a worker pool;
// each worker ships its chains back and one worker combines them via the
// 'poolDiagnostics' message. This pins the invariant that makes that valid:
// split-R̂ and bulk-ESS are pure functions of the chain SET, so combining the
// chains from several workers gives exactly what a single worker over all the
// chains would report — and, unlike per-worker max-R̂ / sum-ESS, it stays correct
// when a worker holds a single chain (whose bulk-ESS alone is NaN).
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { createWorkerHandler } = require('../worker.ts');
const diagnostics = require('../diagnostics.ts');

function makeChains(): Float64Array[] {
  const chains: Float64Array[] = [];
  for (let c = 0; c < 4; c++) {
    const a = new Float64Array(500);
    for (let i = 0; i < 500; i++) a[i] = Math.sin(0.1 * i + c) + 0.002 * i + 0.3 * c;
    chains.push(a);
  }
  return chains;
}

test('poolDiagnostics combines split chains into the single-run R̂ / bulk-ESS', () => {
  const chains = makeChains();
  const refRHat = diagnostics.splitRHat(chains);
  const refEss = diagnostics.essBulk(chains);
  assert.ok(Number.isFinite(refRHat) && Number.isFinite(refEss), 'reference finite');
  const worker = createWorkerHandler();
  const reply = worker.handle({ type: 'poolDiagnostics', id: 1, chains: { mu: chains } });
  assert.equal(reply.type, 'poolDiagnostics');
  assert.equal(reply.perParam.mu.rHat, refRHat);
  assert.equal(reply.perParam.mu.essBulk, refEss);
});

test('global pooling beats per-worker sum-ESS when workers hold one chain each', () => {
  const chains = makeChains();
  const globalEss = diagnostics.essBulk(chains);
  const perWorkerEss = chains.map((c) => diagnostics.essBulk([c]));
  assert.ok(perWorkerEss.every((e) => Number.isNaN(e)), 'lone-chain bulk-ESS is NaN');
  assert.ok(Number.isFinite(globalEss) && globalEss > 0, 'global bulk-ESS is finite and positive');
});
