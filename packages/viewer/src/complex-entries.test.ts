// A complex array or matrix has no plot that shows both parts, so the viewer
// lists its entries instead of drawing the real part as if it were the value.
// @ts-nocheck — test file; compiled separately by node --test (not by tsc)
import { registerHooks } from 'node:module';
import { test } from 'node:test';
import assert from 'node:assert/strict';

// Resolve viewer/src's bundler-style .js imports to the .ts sources.
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.endsWith('.js') && context.parentURL?.includes('/packages/viewer/src/')) {
      return nextResolve(specifier.slice(0, -3) + '.ts', context);
    }
    return nextResolve(specifier, context);
  }
});

const { complexEntriesText } = await import('./render-samples.ts');

test('a complex matrix lists every entry with both parts, row by row', () => {
  assert.equal(complexEntriesText([1, 0, 3, 1], [2, 1, 0, -1], [2, 2]),
    '[[1 + 2 i, 0 + 1 i], [3 + 0 i, 1 - 1 i]]');
});

test('a complex vector lists its entries', () => {
  assert.equal(complexEntriesText([1, 5], [4, -2]), '[1 + 4 i, 5 - 2 i]');
});

test('too many entries to list says so instead of drawing one part', () => {
  const n = 65;
  assert.equal(complexEntriesText(new Float64Array(n), new Float64Array(n)),
    'complex-valued, 65 entries: no plot shows both parts yet, so none is drawn');
});
