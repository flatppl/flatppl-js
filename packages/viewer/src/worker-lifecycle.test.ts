// @ts-nocheck — browser globals and the worker transport are controlled here.
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
const { sendWorker, cancelAllSampling, runMcmcPool } = await import('./worker.ts');
const { getMeasure } = await import('./engine-facade.ts');
const require = createRequire(import.meta.url);
const engine = require('../../engine/index.ts');
const { createWorkerHandler } = require('../../engine/worker.ts');
const turn = () => new Promise(resolve => setImmediate(resolve));
async function until(predicate) {
  for (let i = 0; i < 20 && !predicate(); i++) await turn();
  assert.ok(predicate(), 'expected worker message was not sent');
}

function transport(t, blocked = false) {
  const saved = { Worker: globalThis.Worker, fetch: globalThis.fetch, FlatPPLEngine: globalThis.FlatPPLEngine };
  const wire = { blocked, workers: [], fetches: [] };
  globalThis.FlatPPLEngine = engine;
  globalThis.fetch = () => new Promise(resolve => wire.fetches.push(resolve));
  globalThis.Worker = class {
    constructor(url) {
      if (wire.blocked && !url.startsWith('blob:')) throw new Error('blob fallback');
      this.listeners = new Map();
      this.messages = [];
      this.terminated = false;
      wire.workers.push(this);
    }
    addEventListener(type, fn) { this.listeners.set(type, fn); }
    postMessage(msg) { this.messages.push(msg); }
    terminate() { this.terminated = true; }
    reply(msg, payload) { this.listeners.get('message')({ data: { id: msg.id, ...payload } }); }
  };
  t.after(() => Object.assign(globalThis, saved));
  return wire;
}
const context = () => ({
  SAMPLER_WORKER_URL: 'worker.js', samplerWorker: null, samplerWorkerPromise: null,
  samplerReqId: 0, pendingRequests: new Map(), mcmcPool: [], SAMPLE_COUNT: 4,
  rootSeed: 1, REJECTION_BUDGET: 100, measureCache: new Map(),
  inferenceOpts: { backend: 'importance' }, currentSource: 'model',
});
const bundle = () => ({ ok: true, text: async () => '' });
const message = (w, type) => w.messages.find(m => m.type === type);
const measure = x => ({ shape: 'record', fields: { x: { samples: new Float64Array([x]) } } });
function setModel(ctx, source) {
  const result = engine.processSource(source);
  assert.deepEqual(result.diagnostics.filter(d => d.severity === 'error'), []);
  ctx.derivationsState = engine.orchestrator.buildDerivations(result.linkedBindings,
    { moduleRegistry: result.linkedModuleRegistry });
  ctx.currentSource = source;
  ctx.measureCache = new Map();
}

test('late materialization cannot overwrite the replacement model cache', async t => {
  const wire = transport(t), ctx = context();
  const handler = createWorkerHandler();
  setModel(ctx, 'x ~ Dirac(1)');
  const old = getMeasure(ctx, 'x');
  await until(() => wire.workers[0] && message(wire.workers[0], 'sampleN'));
  const w = wire.workers[0], oldMsg = message(w, 'sampleN');
  setModel(ctx, 'x ~ Dirac(2)');
  const fresh = getMeasure(ctx, 'x');
  await until(() => w.messages.filter(m => m.type === 'sampleN').length === 2);
  const freshMsg = w.messages.filter(m => m.type === 'sampleN')[1];
  w.reply(freshMsg, handler.handle(freshMsg));
  assert.equal((await fresh).samples[0], 2);
  w.reply(oldMsg, handler.handle(oldMsg));
  await old;
  assert.deepEqual(Array.from((await getMeasure(ctx, 'x')).samples), [2, 2, 2, 2]);
});

test('recursive materialization refuses to cross a model replacement', async t => {
  transport(t);
  const ctx = context();
  setModel(ctx, 'outer = 1\nchild = 2');
  let resume;
  const gate = new Promise(resolve => { resume = resolve; });
  const materialise = engine.materialiser.materialiseMeasure;
  t.mock.method(engine.materialiser, 'materialiseMeasure', async (name, snapshot) => {
    if (name !== 'outer') return materialise(name, snapshot);
    await gate;
    return snapshot.getMeasure('child');
  });
  const old = getMeasure(ctx, 'outer');
  setModel(ctx, 'outer = 3\nchild = 4');
  resume();
  await assert.rejects(old);
});

test('late pooled inference cannot overwrite the replacement cache', async t => {
  const wire = transport(t), ctx = context();
  ctx.inferenceOpts = { backend: 'nested' };
  ctx.derivationsState = { derivations: { post: { kind: 'bayesupdate' } } };
  const old = getMeasure(ctx, 'post');
  await until(() => wire.workers[0] && message(wire.workers[0], 'mcmcRun'));
  ctx.measureCache = new Map();
  const fresh = getMeasure(ctx, 'post');
  await until(() => wire.workers[1] && message(wire.workers[1], 'mcmcRun'));
  wire.workers[1].reply(message(wire.workers[1], 'mcmcRun'), { type: 'mcmcResult', measure: measure(2) });
  const current = await fresh;
  wire.workers[0].reply(message(wire.workers[0], 'mcmcRun'), { type: 'mcmcResult', measure: measure(1) });
  await old;
  assert.equal(await getMeasure(ctx, 'post'), current);
});

test('Stop rejects pending startup promptly and a later request restarts', async t => {
  const wire = transport(t, true), ctx = context();
  const old = sendWorker(ctx, { type: 'sampleN' });
  const rejected = assert.rejects(old);
  cancelAllSampling(ctx);
  await rejected; // bundle fetch is still pending
  const fresh = sendWorker(ctx, { type: 'sampleN' });
  wire.fetches[1](bundle());
  await until(() => wire.workers[0] && message(wire.workers[0], 'sampleN'));
  const w = wire.workers[0];
  w.reply(message(w, 'sampleN'), { type: 'samples', samples: [2] });
  assert.deepEqual((await fresh).samples, [2]);
  wire.fetches[0](bundle());
  await turn();
  assert.equal(wire.workers.length, 1);
  assert.equal(w.terminated, false);
  cancelAllSampling(ctx);
});

test('Stop before direct startup dispatch terminates the worker without sampling', async t => {
  const wire = transport(t), ctx = context();
  const request = sendWorker(ctx, { type: 'sampleN' });
  const rejected = assert.rejects(request);
  cancelAllSampling(ctx);
  await rejected;
  await turn();
  assert.equal(wire.workers[0].terminated, true);
  assert.equal(message(wire.workers[0], 'sampleN'), undefined);
});

test('Stop while reading the worker bundle body prevents construction', async t => {
  const wire = transport(t, true), ctx = context();
  let finishBody;
  const request = sendWorker(ctx, { type: 'sampleN' });
  const rejected = assert.rejects(request);
  wire.fetches[0]({ ok: true, text: () => new Promise(resolve => { finishBody = resolve; }) });
  await until(() => !!finishBody);
  cancelAllSampling(ctx);
  await rejected;
  finishBody('');
  await turn();
  assert.equal(wire.workers.length, 0);
});

test('failed worker startup rejects its requests and permits retry', async t => {
  const wire = transport(t, true), ctx = context();
  t.mock.method(console, 'error', () => {});
  const failed = sendWorker(ctx, { type: 'sampleN' });
  const rejected = assert.rejects(failed);
  wire.fetches[0]({ ok: false, status: 500 });
  await rejected;
  wire.blocked = false;
  const retry = sendWorker(ctx, { type: 'sampleN' });
  await until(() => wire.workers[0] && message(wire.workers[0], 'sampleN'));
  wire.workers[0].reply(message(wire.workers[0], 'sampleN'), { type: 'samples', samples: [3] });
  assert.deepEqual((await retry).samples, [3]);
});

test('cancelled pool startup cannot replace or dispatch through a newer pool', async t => {
  const wire = transport(t, true), ctx = context();
  const old = runMcmcPool(ctx, 'post', { backend: 'emcee', parallel: 2 });
  const rejected = assert.rejects(old);
  wire.fetches[0](bundle());
  await until(() => wire.workers.length === 1);
  cancelAllSampling(ctx);
  wire.blocked = false;
  const fresh = runMcmcPool(ctx, 'post', { backend: 'nested' });
  await until(() => wire.workers[1] && message(wire.workers[1], 'mcmcRun'));
  wire.fetches[1](bundle());
  await rejected;
  assert.equal(wire.workers[0].terminated, true);
  assert.equal(message(wire.workers[0], 'mcmcRun'), undefined);
  assert.equal(wire.workers[1].terminated, false);
  wire.workers[1].reply(message(wire.workers[1], 'mcmcRun'), { type: 'mcmcResult', measure: measure(2) });
  assert.equal((await fresh).fields.x.samples[0], 2);
});

test('pool startup failure terminates late siblings without inference dispatch', async t => {
  const wire = transport(t, true), ctx = context();
  const request = runMcmcPool(ctx, 'post', { backend: 'emcee', parallel: 2 });
  const rejected = assert.rejects(request);
  wire.fetches[0]({ ok: false, status: 500 });
  await rejected;
  wire.fetches[1](bundle());
  await until(() => wire.workers.length === 1);
  await turn();
  assert.equal(wire.workers[0].terminated, true);
  assert.equal(message(wire.workers[0], 'mcmcRun'), undefined);
});
