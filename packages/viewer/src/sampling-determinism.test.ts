// @ts-nocheck — browser globals and the worker transport are controlled here.
//
// The pane breakout renders a fresh copy of a panel in each window
// (flatppl-dev/playground-pane-breakout-design.md §2.2, §4.1). That is
// sound only if two viewer mounts with the same model and the same plot
// settings draw the same samples, each with its own sampler worker. These
// tests check it for a forward measure and for an MH posterior, and check
// that a changed seed changes the draws, so equality is not vacuous.

import { registerHooks, createRequire } from 'node:module';
import { test } from 'node:test';
import assert from 'node:assert/strict';

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.endsWith('.js') && context.parentURL?.includes('/packages/viewer/src/')) {
      return nextResolve(specifier.slice(0, -3) + '.ts', context);
    }
    return nextResolve(specifier, context);
  },
});

const require = createRequire(import.meta.url);
const engine = require('../../engine/index.ts');
const { getMeasure } = await import('./engine-facade.ts');
const { cancelAllSampling } = await import('./worker.ts');
const { InProcessWorker } = await import('./test-worker.ts');

const FORWARD = [
  'mu = draw(Normal(mu = 0.0, sigma = 1.0))',
  'x = draw(Normal(mu = mu, sigma = 2.0))',
].join('\n');

const POSTERIOR = [
  'theta ~ Normal(0, 1)',
  'prior = lawof(record(theta = theta))',
  'y ~ Normal(theta, 1)',
  'K = kernelof(y, theta)',
  'L = likelihoodof(K, 0.7)',
  'post = bayesupdate(L, prior)',
].join('\n');

function withWorkers(t) {
  const saved = { Worker: globalThis.Worker, FlatPPLEngine: globalThis.FlatPPLEngine };
  globalThis.Worker = InProcessWorker;
  globalThis.FlatPPLEngine = engine;
  t.after(() => Object.assign(globalThis, saved));
}

/** A viewer mount's sampling state, as mount() and a restored plot snapshot leave it. */
function mountState(source, settings) {
  const result = engine.processSource(source);
  assert.deepEqual(result.diagnostics.filter((d) => d.severity === 'error'), []);
  return {
    SAMPLER_WORKER_URL: 'sampler-worker.min.js', samplerWorker: null, samplerWorkerPromise: null,
    samplerReqId: 0, pendingRequests: new Map(), mcmcPool: [], measureCache: new Map(),
    SAMPLE_COUNT: settings.sampleCount, rootSeed: settings.rootSeed,
    REJECTION_BUDGET: 1000, MARGINALIZATION_COUNT: 100,
    inferenceOpts: { ...settings.inferenceOpts },
    currentSource: source,
    derivationsState: engine.orchestrator.buildDerivations(result.linkedBindings,
      { moduleRegistry: result.linkedModuleRegistry }),
  };
}

function leaf(measure, field) {
  const m = field ? measure.fields[field] : measure;
  return Array.from(m.samples);
}

async function draw(source, name, settings, field) {
  const ctx = mountState(source, settings);
  try {
    const measure = await getMeasure(ctx, name);
    return Object.assign(leaf(measure, field), { diagnostics: measure.diagnostics || null });
  } finally {
    cancelAllSampling(ctx);
  }
}

const IS = { backend: 'is' };
const MH = { backend: 'mh', chains: 2, warmup: 200, draws: 300, seed: null };

test('two mounts draw identical forward samples, and another seed draws others', async (t) => {
  withWorkers(t);
  const settings = { sampleCount: 2000, rootSeed: 1, inferenceOpts: IS };
  const a = await draw(FORWARD, 'x', settings);
  const b = await draw(FORWARD, 'x', settings);
  assert.equal(a.length, 2000);
  assert.ok(a.every(Number.isFinite));
  assert.deepEqual([...b], [...a]);
  const other = await draw(FORWARD, 'x', { ...settings, rootSeed: 2 });
  assert.notDeepEqual([...other], [...a]);
});

test('two mounts draw an identical MH posterior, and another seed draws another', async (t) => {
  withWorkers(t);
  const settings = { sampleCount: 2000, rootSeed: 1, inferenceOpts: MH };
  const a = await draw(POSTERIOR, 'post', settings, 'theta');
  const b = await draw(POSTERIOR, 'post', settings, 'theta');
  // MH diagnostics: the pooled MH path ran, not importance sampling.
  assert.equal(typeof a.diagnostics?.acceptRate, 'number');
  assert.equal(a.length, settings.sampleCount);
  assert.ok(a.every(Number.isFinite));
  assert.deepEqual([...b], [...a]);
  assert.deepEqual(b.diagnostics, a.diagnostics);
  const other = await draw(POSTERIOR, 'post', { ...settings, rootSeed: 2 }, 'theta');
  assert.notDeepEqual([...other], [...a]);
});
