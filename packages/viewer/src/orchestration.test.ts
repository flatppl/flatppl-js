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

const { moduleContextOnUpdate } = await import('./orchestration.ts');

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
