// @ts-nocheck — test file; compiled separately by node --test (not by tsc)
import { registerHooks } from 'node:module';
import { test } from 'node:test';
import assert from 'node:assert/strict';

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.endsWith('.js') && context.parentURL?.includes('/packages/viewer/src/')) {
      return nextResolve(specifier.slice(0, -3) + '.ts', context);
    }
    return nextResolve(specifier, context);
  }
});

const {
  clearForwardCaches, setSampleCount, setMarginalizationCount,
  pinForwardSeed, rerollForwardSeed,
} = await import('./forward-draws.ts');

function mkCtx(over = {}) {
  return Object.assign({
    measureCache: new Map([['a', 1]]),
    histogramCache: new Map([['b', 2]]),
    rootSeed: 1,
    SAMPLE_COUNT: 100000,
    MARGINALIZATION_COUNT: 100,
    inferenceOpts: { seed: null },
    plotEnabled: true,
  }, over);
}

test('clearForwardCaches replaces both caches with empty maps', () => {
  const ctx = mkCtx();
  const m0 = ctx.measureCache, h0 = ctx.histogramCache;
  clearForwardCaches(ctx);
  assert.equal(ctx.measureCache.size, 0);
  assert.equal(ctx.histogramCache.size, 0);
  assert.notEqual(ctx.measureCache, m0);
  assert.notEqual(ctx.histogramCache, h0);
});

test('setSampleCount applies a positive integer and clears caches', () => {
  const ctx = mkCtx();
  assert.equal(setSampleCount(ctx, 5000), true);
  assert.equal(ctx.SAMPLE_COUNT, 5000);
  assert.equal(ctx.measureCache.size, 0);
});

test('setSampleCount rejects zero / negative / non-finite without mutating', () => {
  for (const bad of [0, -3, NaN, Infinity]) {
    const ctx = mkCtx();
    assert.equal(setSampleCount(ctx, bad), false);
    assert.equal(ctx.SAMPLE_COUNT, 100000);
    assert.equal(ctx.measureCache.size, 1);
  }
});

test('setSampleCount is a no-op (false) when value is unchanged', () => {
  const ctx = mkCtx();
  assert.equal(setSampleCount(ctx, 100000), false);
  assert.equal(ctx.measureCache.size, 1);
});

test('setMarginalizationCount applies n>=1 and rejects <1', () => {
  const ctx = mkCtx();
  assert.equal(setMarginalizationCount(ctx, 250), true);
  assert.equal(ctx.MARGINALIZATION_COUNT, 250);
  assert.equal(setMarginalizationCount(ctx, 0), false);
});

test('pinForwardSeed sets rootSeed and mirrors into null inferenceOpts.seed', () => {
  const ctx = mkCtx();
  pinForwardSeed(ctx, 42);
  assert.equal(ctx.rootSeed, 42);
  assert.equal(ctx.inferenceOpts.seed, 42);
});

test('pinForwardSeed does not overwrite an already-set inferenceOpts.seed', () => {
  const ctx = mkCtx({ inferenceOpts: { seed: 7 } });
  pinForwardSeed(ctx, 42);
  assert.equal(ctx.rootSeed, 42);
  assert.equal(ctx.inferenceOpts.seed, 7);
});

test('rerollForwardSeed increments rootSeed, clears caches, returns new seed', () => {
  const ctx = mkCtx({ rootSeed: 4 });
  const next = rerollForwardSeed(ctx);
  assert.equal(next, 5);
  assert.equal(ctx.rootSeed, 5);
  assert.equal(ctx.measureCache.size, 0);
});
