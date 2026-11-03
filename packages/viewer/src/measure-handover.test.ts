// @ts-nocheck — the cached-sample handover between two windows' mounts.

import { registerHooks } from 'node:module';
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

const { installViewerEnv } = await import('./test-viewer-env.ts');
const { InProcessWorker, settle } = await import('./test-worker.ts');
const { mount } = await import('./main.ts');
const { sampleBytes } = await import('./measure-handover.ts');

const MODEL = [
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

/** A plots window: mount, optional handover and settings, then the model. */
async function plotsWindow({ source = MODEL, target = 'x', handover = null, plot = null, config = {} } = {}) {
  const env = installViewerEnv();
  const { dom } = env;
  Object.assign(dom.window.__FLATPPL_CONFIG__, config);
  dom.document.body.innerHTML = '<div id="surface-root"></div>';
  const view = mount(dom.document.getElementById('surface-root'), { host: { setTarget() {} }, panels: ['plot'] });
  if (handover) view.importMeasures(handover);
  if (plot) view.restorePanel('plot', { plot: { ...view.snapshotPanel('plot').plot, ...plot } });
  const sent = InProcessWorker.posted.length;
  view.update(source, target, {});
  for (let i = 0; i < 100 && env.charts.length === 0; i++) await settle();
  await settle();
  return {
    view, env,
    charts: env.charts.map((c) => c.options),
    workerMessages: InProcessWorker.posted.slice(sent).filter((t) => t !== 'init' && t !== 'setEnv'),
    async close() { view.dispose(); await settle(); env.restore(); },
  };
}

test('samples move to another window and render there without sampling', async () => {
  const a = await plotsWindow();
  assert.ok(a.workerMessages.length > 0, 'the first window sampled');
  const handover = a.view.exportMeasures('x');
  assert.deepEqual(handover.entries.map(([k]) => k), ['x']);
  assert.ok(handover.bytes > 0);
  assert.equal(handover.omitted, undefined);
  assert.equal(a.view.exportMeasures('x').entries.length, 0, 'the sender gave its entries away');
  await a.close();

  const b = await plotsWindow({ handover: structuredClone(handover) });
  assert.equal(b.view.measureHandoverStatus(), 'adopted');
  assert.deepEqual(b.workerMessages, [], 'no sampling in the receiving window');
  assert.deepEqual(b.charts, a.charts, 'the same histogram');
  await b.close();
});

test('an MH posterior moves with its diagnostics and is not run again', async () => {
  const mh = { inferenceOpts: { backend: 'mh', chains: 2, warmup: 200, draws: 300, seed: null } };
  const a = await plotsWindow({ source: POSTERIOR, target: 'post', plot: mh });
  assert.ok(a.workerMessages.includes('mcmcRun'));
  const handover = a.view.exportMeasures('post');
  assert.equal(typeof handover.entries[0][1].diagnostics.acceptRate, 'number');
  await a.close();
  const b = await plotsWindow({ source: POSTERIOR, target: 'post', plot: mh, handover });
  assert.equal(b.view.measureHandoverStatus(), 'adopted');
  assert.equal(b.workerMessages.includes('mcmcRun'), false);
  assert.deepEqual(b.charts, a.charts);
  await b.close();
});

for (const [name, change] of [
  ['a source edit', { source: MODEL + '\nz = 1' }],
  ['another model', { source: POSTERIOR, target: 'post' }],
  ['the sample count', { plot: { sampleCount: 1234 } }],
  ['the seed', { plot: { rootSeed: 2 } }],
  ['the marginalisation count', { plot: { marginalizationCount: 7 } }],
  ['the rejection budget', { plot: { rejectionBudget: 3 } }],
  ['the inference options', { plot: { inferenceOpts: { backend: 'is', chains: 3 } } }],
]) {
  test(`a handover is rejected after a change of ${name}, and the window samples`, async () => {
    const a = await plotsWindow();
    const handover = a.view.exportMeasures('x');
    await a.close();
    const b = await plotsWindow({ ...change, handover });
    assert.match(b.view.measureHandoverStatus(), /^rejected:(model|settings)$/);
    assert.ok(b.workerMessages.length > 0, 'the receiving window sampled');
    await b.close();
  });
}

test('a handover above the cap carries no samples, and the window samples', async () => {
  const a = await plotsWindow();
  const handover = a.view.exportMeasures('x', 16);
  assert.equal(handover.omitted, 'cap');
  assert.deepEqual(handover.entries, []);
  assert.ok(handover.bytes > 16);
  await a.close();
  const b = await plotsWindow({ handover });
  assert.equal(b.view.measureHandoverStatus(), 'rejected:cap');
  assert.ok(b.workerMessages.length > 0);
  await b.close();
});

test('a measure that cannot be cloned stays, warns once, and the receiver samples', async (t) => {
  const a = await plotsWindow();
  const warnings = [];
  t.mock.method(console, 'warn', (...args) => warnings.push(args[0]));
  // Reach the cache through a second export: put a non-cloneable entry first.
  const handoverOk = a.view.exportMeasures('x');
  a.view.importMeasures({ ...handoverOk, entries: [['x', { ...handoverOk.entries[0][1], fn() {} }]] });
  a.view.update(MODEL, 'x', {});
  await settle();
  const handover = a.view.exportMeasures('x');
  assert.equal(handover.omitted, 'clone');
  assert.equal(warnings.length, 1);
  await a.close();
  const b = await plotsWindow({ handover });
  assert.equal(b.view.measureHandoverStatus(), 'rejected:clone');
  assert.ok(b.workerMessages.length > 0);
  await b.close();
});

test('a window that already holds the samples keeps its own', async () => {
  const a = await plotsWindow();
  const handover = a.view.exportMeasures('x');
  await a.close();
  const b = await plotsWindow();
  const own = b.charts;
  const forged = structuredClone(handover);
  forged.entries[0][1].samples = new Float64Array(forged.entries[0][1].samples.length).fill(42);
  b.view.importMeasures(forged);
  b.view.select('x');
  await settle();
  assert.equal(b.view.measureHandoverStatus(), 'adopted');
  assert.deepEqual(b.env.charts.map((c) => c.options).slice(-1), own.slice(-1));
  await b.close();
});

test('sampleBytes counts typed buffers once and plain-array numbers at 8 bytes, nested too', () => {
  const shared = new Float64Array(10);
  assert.equal(sampleBytes({ samples: shared, value: { data: shared.subarray(2) } }), 80);
  assert.equal(sampleBytes({ samples: [1, 2, 3], fields: { a: { samples: [[1, 2], [3, 4]] } } }), 56);
  assert.equal(sampleBytes({ n_eff: 5, logTotalmass: 0, logWeights: null }), 0);
});

test('a large plain-array measure counts toward the cap', async () => {
  const a = await plotsWindow();
  const real = a.view.exportMeasures('x');
  const plain = { ...real.entries[0][1], samples: Array.from(real.entries[0][1].samples), value: undefined };
  plain.samples = plain.samples.concat(plain.samples, plain.samples, plain.samples);
  a.view.importMeasures({ ...real, entries: [['x', plain]] });
  a.view.update(MODEL, 'x', {});
  await settle();
  const handover = a.view.exportMeasures('x', 1000);
  assert.equal(handover.omitted, 'cap');
  assert.ok(handover.bytes >= plain.samples.length * 8, 'plain numbers are counted');
  await a.close();
});

test('the host sets the cap through __FLATPPL_CONFIG__.handoverCapBytes', async () => {
  const a = await plotsWindow({ config: { handoverCapBytes: 16 } });
  const handover = a.view.exportMeasures('x');
  assert.equal(handover.omitted, 'cap');
  await a.close();
  const b = await plotsWindow();
  assert.equal(b.view.exportMeasures('x').omitted, undefined, 'the default cap admits the default draws');
  await b.close();
});
