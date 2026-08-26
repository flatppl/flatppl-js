// @ts-nocheck — test file; compiled separately by node --test (not by tsc)
import { registerHooks } from 'node:module';
import { test } from 'node:test';
import assert from 'node:assert/strict';

// Same resolver hook render-field.test.ts uses: viewer/src imports carry
// bundler-style .js extensions that esbuild resolves at build time.
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.endsWith('.js') && context.parentURL?.includes('/packages/viewer/src/')) {
      return nextResolve(specifier.slice(0, -3) + '.ts', context);
    }
    return nextResolve(specifier, context);
  }
});

const { gridEligibleAxes } = await import('./render-profile.ts');

function axis(key, leafKind, path) {
  return { key: key, label: key, kwargName: key.split(/[.[]/)[0],
           path: path || [], leafType: { kind: leafKind } };
}

test('gridEligibleAxes offers every other top-level scalar axis', () => {
  const plan = { axes: [axis('m', 'scalar'), axis('c', 'scalar'), axis('k', 'scalar')] };
  const keys = gridEligibleAxes(plan, 'm').map((a) => a.key);
  assert.deepEqual(keys, ['c', 'k']);
});

test('gridEligibleAxes accepts unrestricted placeholder boundaries', () => {
  const plan = { axes: [axis('arg1', 'any'), axis('arg2', 'any')] };
  assert.deepEqual(gridEligibleAxes(plan, 'arg1').map((a) => a.key), ['arg2']);
});

test('gridEligibleAxes rejects per-slot and non-scalar axes', () => {
  const plan = {
    axes: [
      axis('x', 'scalar'),
      axis('theta[2]', 'scalar', [{ idx: [2] }]),
      axis('obs', 'array'),
    ],
  };
  assert.deepEqual(gridEligibleAxes(plan, 'x'), []);
});

test('gridEligibleAxes is empty for a single-axis plan', () => {
  assert.deepEqual(gridEligibleAxes({ axes: [axis('x', 'scalar')] }, 'x'), []);
  assert.deepEqual(gridEligibleAxes({}, 'x'), []);
});
