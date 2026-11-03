// @ts-nocheck — whole-mount determinism tests.
//
// The pane breakout renders a fresh copy of a panel in each window. That is
// sound only if two viewer mounts with the same model and the same plot
// settings draw the same samples, each with its own sampler worker. Each mount
// here takes its settings through restorePanel('plot', …), as a detached
// window will, and the tests compare what the plot pane hands to its charts,
// for a forward measure and for an MH posterior. A changed seed must change
// the draws, so equality is not vacuous.

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

const MH = { backend: 'mh', chains: 2, warmup: 200, draws: 300, seed: null };

/** Mount a plots-only window, apply plot settings through restorePanel, and
 *  return what its charts received once sampling finished. */
async function plotInWindow(source, target, settings) {
  const env = installViewerEnv();
  const { dom } = env;
  let view = null;
  try {
    dom.document.body.innerHTML = '<div id="surface-root"></div>';
    view = mount(dom.document.getElementById('surface-root'), { host: { setTarget() {} }, panels: ['plot'] });
    view.update(source, target, {});
    await settle();
    const before = env.charts.length;
    const sent = InProcessWorker.posted.length;
    const plot = view.snapshotPanel('plot').plot;
    view.restorePanel('plot', { plot: { ...plot, sampleCount: settings.sampleCount, rootSeed: settings.rootSeed,
      inferenceOpts: { ...plot.inferenceOpts, ...settings.inferenceOpts } } });
    for (let i = 0; i < 100 && env.charts.length === before; i++) await settle();
    await settle();
    const charts = env.charts.slice(before).map((c) => c.options);
    assert.ok(charts.length > 0, 'the plot pane drew no chart');
    return { charts, workerMessages: InProcessWorker.posted.slice(sent) };
  } finally {
    view?.dispose();
    await settle();
    env.restore();
  }
}

test('two windows plot identical forward samples, and another seed plots others', async () => {
  const settings = { sampleCount: 2000, rootSeed: 1, inferenceOpts: { backend: 'is' } };
  const a = await plotInWindow(FORWARD, 'x', settings);
  const b = await plotInWindow(FORWARD, 'x', settings);
  assert.ok(a.charts.flat().some((o) => o.series?.length), 'the forward plot has series data');
  assert.deepEqual(b.charts, a.charts);
  const other = await plotInWindow(FORWARD, 'x', { ...settings, rootSeed: 2 });
  assert.notDeepEqual(other.charts, a.charts);
});

test('two windows plot an identical MH posterior, and another seed plots another', async () => {
  const settings = { sampleCount: 2000, rootSeed: 1, inferenceOpts: MH };
  const a = await plotInWindow(POSTERIOR, 'post', settings);
  assert.ok(a.workerMessages.includes('mcmcRun'), 'the pooled MH path ran');
  const b = await plotInWindow(POSTERIOR, 'post', settings);
  assert.ok(a.charts.flat().some((o) => o.series?.length), 'the posterior plot has series data');
  assert.deepEqual(b.charts, a.charts);
  const other = await plotInWindow(POSTERIOR, 'post', { ...settings, rootSeed: 2 });
  assert.notDeepEqual(other.charts, a.charts);
});
