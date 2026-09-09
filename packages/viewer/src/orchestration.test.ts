// @ts-nocheck — test file; compiled separately by node --test (not by tsc)
import { registerHooks } from 'node:module';
import { test } from 'node:test';
import assert from 'node:assert/strict';

// viewer/src uses bundler-style .js extensions in imports (resolved by esbuild
// at build time). Register a resolver hook so Node --test can load .ts source
// directly without a build step.
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.endsWith('.js') && context.parentURL?.includes('/packages/viewer/src/')) {
      return nextResolve(specifier.slice(0, -3) + '.ts', context);
    }
    return nextResolve(specifier, context);
  }
});

const { moduleContextOnUpdate, lowerInputsChanged } = await import('./orchestration.ts');

// moduleContextOnUpdate decides the (path, bundleSources) a sourceUpdate lowers
// against. Both are STICKY to the current model: a model switch carries `path`
// (re-establishing path + bundle); a same-model update (target navigation /
// edit) omits `path`, so the tracked context persists. Regression guard for the
// cross-URL drill-down bug — a same-model re-lower used to reprocess with an
// empty modulePath, so the engine resolved relative load_module deps against an
// empty base (e.g. a remote module's `priors.flatppl` → a bare gallery path).

const PREV = { path: 'https://h/ex/common.flatppl', bundleSources: { 'https://h/ex/priors.flatppl': 'theta = elementof(reals)' } };

test('a model switch adopts the message path + bundle', () => {
  const r = moduleContextOnUpdate(PREV, {
    source: '...', path: 'https://h/ex/other.flatppl',
    bundleSources: { 'https://h/ex/dep.flatppl': 'x = 1' },
  });
  assert.equal(r.path, 'https://h/ex/other.flatppl');
  assert.deepEqual(r.bundleSources, { 'https://h/ex/dep.flatppl': 'x = 1' });
});

test('a model switch to a leaf (path, no bundle) clears the stale bundle', () => {
  const r = moduleContextOnUpdate(PREV, { source: '...', path: 'leaf.flatppl' });
  assert.equal(r.path, 'leaf.flatppl');
  assert.equal(r.bundleSources, null, 'a leaf switch must not inherit the previous bundle');
});

test('a same-model update (no path) PRESERVES the tracked path + bundle', () => {
  // The bug: a target-nav / edit re-lower omits path → used to drop modulePath.
  const r = moduleContextOnUpdate(PREV, { source: '...changed...', pushHistory: true });
  assert.equal(r.path, PREV.path, 'the current module path must survive a same-model re-lower');
  assert.deepEqual(r.bundleSources, PREV.bundleSources);
});

test('an explicit null path (a path-less module, e.g. an embedded block) clears the context', () => {
  const r = moduleContextOnUpdate(PREV, { source: '...', path: null });
  assert.equal(r.path, null);
  assert.equal(r.bundleSources, null);
});

// lowerInputsChanged decides whether an update re-lowers the model. The
// text is not the only input: the VS Code whole-module command used to post
// the source without its load_module bundle, and the bundle arriving with
// the next same-text update was recorded but never lowered, so nothing
// across the module boundary was plottable until the text changed.
const SRC = 'common = load_module("common.flatppl")\nx ~ common.prior';
const BUNDLE = { '/m/common.flatppl': 'prior = Normal(0, 1)' };

test('same text, same path, same bundle: no re-lower', () => {
  assert.equal(lowerInputsChanged(
    { source: SRC, path: '/m/a.flatppl', bundleSources: BUNDLE },
    { source: SRC, path: '/m/a.flatppl', bundleSources: { ...BUNDLE } }), false);
  assert.equal(lowerInputsChanged(
    { source: SRC, path: null, bundleSources: null },
    { source: SRC, path: undefined, bundleSources: undefined }), false);
});

test('a bundle arriving for the same text re-lowers (the module-view-then-visualize-binding case)', () => {
  assert.equal(lowerInputsChanged(
    { source: SRC, path: '/m/a.flatppl', bundleSources: null },
    { source: SRC, path: '/m/a.flatppl', bundleSources: BUNDLE }), true);
  assert.equal(lowerInputsChanged(
    { source: SRC, path: '/m/a.flatppl', bundleSources: BUNDLE },
    { source: SRC, path: '/m/a.flatppl', bundleSources: null }), true);
});

test('a changed dependency text, an extra dependency, or a different path re-lowers', () => {
  const base = { source: SRC, path: '/m/a.flatppl', bundleSources: BUNDLE };
  assert.equal(lowerInputsChanged(base, { ...base, bundleSources: { '/m/common.flatppl': 'prior = Normal(0, 2)' } }), true);
  assert.equal(lowerInputsChanged(base, { ...base, bundleSources: { ...BUNDLE, '/m/extra.flatppl': 'y = 1' } }), true);
  assert.equal(lowerInputsChanged(base, { ...base, path: '/m/b.flatppl' }), true);
  assert.equal(lowerInputsChanged(base, { ...base, source: SRC + '\n' }), true);
});
