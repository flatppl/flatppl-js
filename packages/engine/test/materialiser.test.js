'use strict';

// End-to-end tests for materialiseMeasure(name, ctx).
//
// We exercise the same entry point the viewer uses, supplying our own
// thin getMeasure cache and a synchronous-worker bridge built on top of
// createWorkerHandler. The materialiser is async-Promise-shaped, but
// the worker calls underneath are synchronous, so the Promise chains
// resolve in microtasks — keeping each test small and ordered.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker');

const SAMPLE_COUNT = 1024;
const ROOT_SEED    = 12345;

/**
 * Build a fresh materialisation context for `source` and return a
 * helper that materialises a binding by name. Each call shares one
 * worker handler + one promise cache, mirroring how the viewer's
 * getMeasure works.
 */
function makeCtx(source) {
  const lifted = processSource(source);
  const { derivations } = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: ROOT_SEED });

  const cache = new Map();
  const ctx = {
    derivations,
    bindings:    lifted.bindings,
    fixedValues: lifted.fixedValues || new Map(),
    getMeasure:  (name) => {
      if (cache.has(name)) return cache.get(name);
      const p = materialiser.materialiseMeasure(name, ctx);
      cache.set(name, p);
      return p;
    },
    sendWorker:  (msg) => {
      const reply = worker.handle(msg);
      if (reply && reply.type === 'error') return Promise.reject(new Error(reply.message));
      return Promise.resolve(reply);
    },
    sampleCount: SAMPLE_COUNT,
    rootSeed:    ROOT_SEED,
  };
  return ctx;
}

test('totalmass: weighted Normal broadcasts the scalar mass', async () => {
  const ctx = makeCtx(`
m = weighted(0.5, Normal(mu = 0.0, sigma = 1.0))
z = totalmass(m)
`);
  const z = await ctx.getMeasure('z');
  assert.equal(z.samples.length, SAMPLE_COUNT);
  for (let i = 0; i < z.samples.length; i++) {
    assert.ok(Math.abs(z.samples[i] - 0.5) < 1e-12,
      'totalmass atom ' + i + ' should equal 0.5, got ' + z.samples[i]);
  }
  assert.equal(z.logWeights, null);
  assert.equal(z.logTotalmass, 0);
  assert.equal(z.n_eff, SAMPLE_COUNT);
});

test('totalmass: unit-mass measure broadcasts 1', async () => {
  const ctx = makeCtx(`
m = Normal(mu = 0.0, sigma = 1.0)
z = totalmass(m)
`);
  const z = await ctx.getMeasure('z');
  for (let i = 0; i < z.samples.length; i++) {
    assert.ok(Math.abs(z.samples[i] - 1.0) < 1e-12,
      'unit-mass atom ' + i + ' should equal 1, got ' + z.samples[i]);
  }
});

test('totalmass: composed weights multiply', async () => {
  const ctx = makeCtx(`
m1 = weighted(0.5, Normal(mu = 0.0, sigma = 1.0))
m2 = weighted(0.25, m1)
z  = totalmass(m2)
`);
  const z = await ctx.getMeasure('z');
  for (let i = 0; i < z.samples.length; i++) {
    assert.ok(Math.abs(z.samples[i] - 0.125) < 1e-12,
      'composed totalmass should equal 0.5 * 0.25 = 0.125, got ' + z.samples[i]);
  }
});
